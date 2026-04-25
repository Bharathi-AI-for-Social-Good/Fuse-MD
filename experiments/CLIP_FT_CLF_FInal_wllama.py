# import necessary frameworks, libraries
# Input script run parameters 
#
# folder
# language
# batch size
# clip epochs
# clip learning rate
# clip gamma
#
# clf epochs
# clf learning rate
# clf gamma

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import pandas as pd
import matplotlib.pyplot as plt

import sys
import os
os.chdir("/home/rahpon/projects/Misogyny_meme1")
import cv2
from tqdm import tqdm
from datetime import datetime 
# storing the current time in the variable
dt = datetime.now()

# load pre-trained CLIP and MuRIL
from transformers import CLIPModel
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

double_quant_config = BitsAndBytesConfig(
    # load_in_4bit=True,
    # bnb_4bit_use_double_quant=True,
    # bnb_4bit_compute_dtype=torch.float16,
    load_in_8bit=True,
)
tokenizer = AutoTokenizer.from_pretrained("VishnuPJ/MalayaLLM_7B_Base", low_cpu_mem_usage=True, torch_dtype=torch.float16)
text_model = AutoModelForCausalLM.from_pretrained("VishnuPJ/MalayaLLM_7B_Base", low_cpu_mem_usage=True, quantization_config=double_quant_config, torch_dtype=torch.float16)

# device - GPU/CPU accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# extracting encoders from pre-trained model
# Image encoder from CLIP pre-trained
# Text encoder from MuRIL BERT
vision_encoder = model.vision_model
# text_encoder = text_model.bert

# Dataset path, language
folder = os.path.join("training_data")
language = "malayalam"  # set accordingly
dpth = os.path.join(folder, language)

""" 
Use for adding extra token - [PAD], for padding all to same length
Later re-train the model for learning an embedding corresponding to the new vocab element
"""
tokenizer.all_special_ids
special_tokens_dict = {"pad_token" : '[PAD]'}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
tok_id = tokenizer.convert_tokens_to_ids('[PAD]')
text_model.resize_token_embeddings(len(tokenizer))    # resizing embedding matrix of model to accomodate new [PAD]


class LanData(Dataset):
    def __init__(self, mode, dpth):
        """
        Dataset class for loading images and transcriptions of memes
        Add dataset path got above as parameter(dpth)
        mode = train/test, to get training/testing dataset (saved as test and train folders in language folder)
        """

        self.path = os.path.join(dpth, mode)   # mode = train/test
        tscpts = os.path.join(self.path, mode + ".csv")
        self.csv = pd.read_csv(tscpts).dropna(axis = 0)
        toks = tokenizer(self.csv.transcriptions.tolist(), truncation=True, max_length=75, padding = 'max_length', return_tensors = 'pt')
        self.csv["tokens"] = toks.input_ids.tolist()
        self.csv["masks"] = toks.attention_mask.tolist()

        ims = []
        labs = []
        scpts = []
        ms = []
        imageIDs = []
        mu = torch.zeros((1, 3, 1, 1))
        sigma = torch.zeros((1, 3, 1, 1))

        """
        Iterate over all images
        Get mu (mean) and sigma (std-dev) of images
        Normalize for better image pixel value distribution for better training
        """
        tot = 0
        for f in os.listdir(self.path):
            if ".csv" in f:
                continue

            # move-on only if valid datapoint, i.e. has a transcript
            iid = int(f.split('.')[0])
            if iid not in self.csv.image_id.tolist():
                continue
            imageIDs.append(iid); tot += 1

            # image
            ig = cv2.imread(os.path.join(self.path, f))
            ig = cv2.resize(ig, (224, 224), interpolation = cv2.INTER_CUBIC)
            ig = torch.tensor(ig.reshape(1, 3, 224, 224)) / ig.max()
            ims.append(ig)
            mu += ig.mean([2, 3]).reshape((1,3,1,1))
            sigma += ig.std([2, 3]).reshape((1,3,1,1))
        mu /= tot
        sigma /= tot

        tr = 0
        for iid, ig in zip(imageIDs, ims):
            # image
            ig = (ig - mu) / sigma
            ims[tr] = ig
            tr += 1

            #label
            l = self.csv[self.csv.image_id == iid]["labels"].astype(float)
            labs.append(l)

            # transcripts
            s = self.csv[self.csv.image_id == iid]["tokens"].tolist()[0]
            scpts.append(s)

            # masks
            m = self.csv[self.csv.image_id == iid]["masks"].tolist()[0]
            ms.append(m)
            
            if int(l.iloc[0]) == 1 and mode == "train":
                labs.extend([l,] * 3)
                scpts.extend([s,] * 3)
                ms.extend([m,] * 3)
                ims.extend([ig,] * 3)

        self.images = torch.cat(ims, dim = 0)
        self.labels = labs
        self.tokens = scpts
        self.masks = ms
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, ix):
        return self.images[ix].to(torch.bfloat16), torch.tensor(self.labels[ix].item()).to(torch.bfloat16), torch.tensor(self.tokens[ix]), torch.tensor(self.masks[ix]).to(torch.bfloat16)
# hyper-parameters
N = 16 # int(sys.argv[3])    # batch-size
T = 0.07   # temperature factor

train_data = LanData("train", dpth)
test_data = LanData("test", dpth)

train_loader = DataLoader(train_data, batch_size = N, shuffle = True, drop_last = True)
test_loader = DataLoader(test_data, batch_size = N, shuffle = True, drop_last = True)


class TextModel(nn.Module):
    """
    Text classification model using llama
    Keep llama except the final decoding layer
    Add linear layers to transform into a lower dimensional embedding
    """
    def __init__(self, llama):
        """ Pass in the loaded pre-trained llama """
        super(TextModel, self).__init__()
        self.text_base = llama.model
        self.clf_head = nn.Sequential(
            nn.Linear(4096, 512, bias = False),
            nn.Dropout(0.5, inplace = False),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256, bias=False)
        )

    def forward(self, x, mask):
        text_hidden = self.text_base(x, attention_mask = mask).last_hidden_state.mean(1)
        emb = self.clf_head(text_hidden)

        return emb

text_encoder = TextModel(text_model).half().to(device)


class CLIP(nn.Module):
    def __init__(self, v_enc, t_enc, T):
        super(CLIP, self).__init__()

        self.vision_encoder = v_enc
        self.vision_embdg = nn.Linear(1024, 256, bias = True)
        self.text_encoder = t_enc
        # self.text_embdg = nn.Linear(768, 256, bias = True)
        # self.t = T  # temperature paramter, T = .07

    def forward(self, img, txt, mask):
        ve = self.vision_encoder(img).pooler_output
        ve = self.vision_embdg(ve)
        te = self.text_encoder(txt, mask)
        # te = self.text_embdg(te)

        return (ve, te)

clip = CLIP(vision_encoder, text_encoder, T).to(torch.bfloat16).to(device)

class head(nn.Module):
    def __init__(self, embed_size, fusion = "concat"):
        """
        classification MLP on fused embeddings
        inputs:
        embed_size - embedding_size from encoders
        fusion - fusion technique to use for combining embeddings. Can take values:
            "concat" : concatenate for 2 * embed_size input
            "element" : element wise product of each embd
            "sumpool" : learnable fusion with 2 linear layers without bias, and element-wise product, then sum-pooling
                        motivation (https://aclanthology.org/2024.lrec-main.300.pdf)
        """
        super(head, self).__init__()

        if fusion not in ["concat", "element", "sumpool", "gated"]:
            fusion = "concat"
        if fusion == "concat":
            clf_size = 2 * embed_size
        elif fusion == "element":
            clf_size = embed_size
        elif fusion == "gated":
            self.U = nn.Linear(embed_size, embed_size // 2, bias = False)
            self.sig = nn.Sigmoid()
            self.V = nn.Linear(embed_size, embed_size // 2, bias = False)
            self.W = nn.Linear(embed_size // 2, embed_size // 4, bias = False)
            clf_size = embed_size // 4
        else:
            self.w1 = nn.Linear(embed_size, embed_size // 2, bias = False)
            self.w2 = nn.Linear(embed_size, embed_size // 2, bias = False)
            self.sumpool = nn.AvgPool1d(4)
            clf_size = embed_size // 8
        self.fusion = fusion

        self.initial = nn.Linear(clf_size, int(clf_size/2), bias = True)
        self.final = nn.Linear(int(clf_size/2), 1, bias = True)
        self.act = nn.Sigmoid()

    def forward(self, venc, tenc):
        if self.fusion == "concat":
            diag = torch.concat([tenc, venc], dim = -1)
        elif self.fusion == "element":
            diag = tenc * venc
        elif self.fusion == "gated":
            tenc = self.sig(self.U(tenc))
            venc = self.V(venc)
            enc = tenc * venc
            diag = self.W(enc)
        elif self.fusion == "sumpool":
            venc = self.w1(venc)
            tenc = self.w2(tenc)
            diag = self.sumpool(venc * tenc)
        hidden = self.initial(diag)
        res = self.final(hidden)
        out = self.act(res)

        return out


for fusion in ["concat"]: # , "element", "sumpool", "gated"
    clf_head = head(embed_size = 256, fusion = fusion).to(torch.bfloat16).to(device)

    for clf_lr in [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]:
        print("beginning training ", fusion, " with learning rate ", clf_lr)
        clf_epochs = 10
        clf_g = 0.9

        """
        clf_head to train on top of obtained classifiers
        clip encoders fixed in eval mode
        """
        for p in clip.parameters():
            p.requires_grad = False
            
        for p in list(clip.vision_embdg.parameters()) + list(clip.text_encoder.clf_head.parameters()):
            p.requires_grad = True
            
        for p in clf_head.parameters():
            p.requires_grad = True

        trainable_parameters = list(clip.vision_embdg.parameters()) + list(clip.text_encoder.clf_head.parameters()) + list(clf_head.parameters())

        optimizer = torch.optim.Adam(trainable_parameters, lr = clf_lr, betas = (0.9, 0.999), eps = 1e-8)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = clf_g, last_epoch = -1)
        criterion = nn.BCELoss()

        clf_head.train()
        clip.eval()

        for epoch in range(1, clf_epochs + 1):
            its = 0
            epoch_loss = 0
            best_loss = 1_000_000.

            with tqdm(train_loader, unit = "batch") as train_epoch:
                for img, lab, text, mask in train_epoch:
                    img, lab, text, mask = img.to(device), lab.to(device), text.to(device), mask.to(device)
                    train_epoch.set_description(f"Epoch: {epoch}")
                    batch_loss = 0

                    optimizer.zero_grad()
                    (v, t) = clip(img, text, mask)
                    # joint = torch.concat([v, t], dim = 1)
                    out = clf_head(v, t)

                    batch_loss = criterion(out, lab.unsqueeze(dim = 1))
                    batch_loss.backward()
                    optimizer.step()

                    epoch_loss += batch_loss.item()
                    its += 1

            epoch_loss /= its
    #         scheduler.step()

            print("Epoch Loss: ", epoch_loss)
    #         torch.save(clf_head.state_dict(), "classifier_LR" + str(clf_lr) + "_EPOCHS" + str(clf_epochs) + "_GAMMA" + str(clf_g) + "_BS" + str(N))

        # testing
        print("Finding best threshold and evaluating : ")
        clip.eval()
        clf_head.eval()
        ths = [x/10 for x in range(1, 10)]
        accs = []

        for th in ths:
            test_loss = 0
            its = 0
            correct = 0

            with tqdm(test_loader, unit = "batch") as test_epoch:
                for img, lab, text, mask in test_epoch:
                    img, lab, text, mask = img.to(device), lab.to(device), text.to(device), mask.to(device)
                    test_epoch.set_description(f"Testing...")
                    batch_loss = 0

                    (v, t) = clip(img, text, mask)
    #                 joint = torch.concat([v, t], dim = 1)
                    out = clf_head(v, t)
                    pred = (out > th).to(torch.float)
                    correct += (pred == lab.unsqueeze(dim = 1)).sum()

                    batch_loss = criterion(out, lab.unsqueeze(dim = 1))

                    test_loss += batch_loss.item()
                    its += 1

            test_loss /= its
            accuracy = (correct / (its * 16)).item()
            accs.append(accuracy)

            print(f"Accuracy: {accuracy * 100:.2f} %")
        print(f"Average test set loss:  {test_loss:.5f}")

        best_thr = ths[accs.index(max(accs))]
        print(f"Best threshold computed at: {best_thr}")

        print("Now producing results at best threshold:-")

        preds = []
        tru = []

        test_loss = 0
        its = 0
        correct = 0

        with tqdm(test_loader, unit = "batch") as test_epoch:
            for img, lab, text, mask in test_epoch:
                img, lab, text, mask = img.to(device), lab.to(device), text.to(device), mask.to(device)
                test_epoch.set_description(f"Testing...")
                batch_loss = 0

                (v, t) = clip(img, text, mask)
                # joint = torch.concat([v, t], dim = 1)
                out = clf_head(v, t)
                
                pred = (out > best_thr).to(torch.float)
                correct += (pred == lab.unsqueeze(dim = 1)).sum()

                batch_loss = criterion(out, lab.unsqueeze(dim = 1))

                test_loss += batch_loss.item()
                its += 1
                preds.extend(pred.squeeze(1).tolist())
                tru.extend(lab.tolist())

        test_loss /= its
        accuracy = (correct / (its * 16)).item()

        print(f"Accuracy: {accuracy * 100:.2f} %")
        print(f"Average test set loss:  {test_loss:.5f}")

        ConfusionMatrixDisplay(confusion_matrix(tru, preds)).plot()
        plt.savefig("predictions/"+str(language)+"/fusion/multimodal/"+str(language)+"_llama_clip_"+str(fusion)+"fusion_pretrained_"+str(clf_lr)+"_"+str(clf_epochs)+"_"+str(N)+".png")

        print(classification_report(tru, preds, digits = 5))

        cf = confusion_matrix(tru, preds)
        cr = classification_report(tru, preds, digits = 5)
        with open("predictions/"+str(language)+"/fusion/multimodal/"+str(language)+"_llama_clip_"+str(fusion)+"fusion_pretrained"+str(N)+".txt", "a") as text_file:
            text_file.write("Date and time of run = "+str(dt)+"\n")
            text_file.write(str(clf_lr)+"_"+str(clf_epochs)+"_"+str(N)+"\n\n")
            text_file.write(str(cf)+"\n")
            text_file.write(cr)
            text_file.write("\n\n\n")
        
        print("Done training ", fusion, " with learning rate ", clf_lr)
