""" Import modules """
import torch
# import time
# from transformers import T5Tokenizer
# from transformers import T5ForConditionalGeneration, AdamW
import pandas as pd
import os
# os.chdir("/home/rahpon/projects/FUSE_MD")
# import re
import cv2
# from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# import torch.nn.functional as F
# from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from torch import nn
import matplotlib.pyplot as plt
import timm
from datetime import datetime 
# storing the current time in the variable
dt = datetime.now()

# """ set necessary environment variables """
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 

""" 
Load models for vision and text - ViT and llama
Also load tokenizer
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

double_quant_config = BitsAndBytesConfig(
    # load_in_4bit=True,
    # bnb_4bit_use_double_quant=True,
    # bnb_4bit_compute_dtype=torch.float16,
    load_in_8bit=True,
)
# abhinand/tamil-llama-7b-base-v0.1
tokenizer = AutoTokenizer.from_pretrained("VishnuPJ/MalayaLLM_7B_Base", low_cpu_mem_usage=True, dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained("VishnuPJ/MalayaLLM_7B_Base", low_cpu_mem_usage=True, quantization_config=double_quant_config, dtype=torch.float16)

# from transformers import CLIPModel
img_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)

""" set device """
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" 
set dataset location
folder - path containing tamil & malayalam folders
"""
folder = os.path.join("training_data")
language = "malayalam"  # set accordingly
dpth = os.path.join(folder, language)

""" hyper-parameters """
N = 16       # batch-size
T = 0.07     # temperature factor
epochs = 50

""" 
Use for adding extra token - [PAD], for padding all to same length
Later re-train the model for learning an embedding corresponding to the new vocab element
"""
tokenizer.all_special_ids
special_tokens_dict = {"pad_token" : '[PAD]'}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
tok_id = tokenizer.convert_tokens_to_ids('[PAD]')
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)    # resizing embedding matrix of model to accomodate new [PAD]

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
            l = self.csv[self.csv.image_id == iid]["labels"].astype(float)  # Fixed: Column is 'labels', not 'labels '
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
        return self.images[ix].to(torch.float16), torch.tensor(self.labels[ix].item()).to(torch.float16), torch.tensor(self.tokens[ix]), torch.tensor(self.masks[ix]).to(torch.float16)

""" Load datasets using class above """
train_data = LanData("train", dpth)
dev_data = LanData("dev", dpth)
test_data = LanData("test", dpth)

""" Creat pytorch dataloaders from nn.dataset classes above """
train_loader = DataLoader(train_data, batch_size = N, shuffle = True, drop_last = True)
dev_loader = DataLoader(dev_data, batch_size = N, shuffle = False, drop_last = False)
test_loader = DataLoader(test_data, batch_size = N, shuffle = False, drop_last = False)

class ImageModel(nn.Module):
    def __init__(self, vit):
        super(ImageModel, self).__init__()
        vit.head = nn.Linear(768, 128)
        self.embedding = vit

    def forward(self, x):
        return self.embedding(x)

class TextModel(nn.Module):
    """
    Text classification model using llama
    Keep llama except the final decoding layer
    Add linear layers to transform into a lower dimensional embedding
    Add classification head
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
            nn.Linear(512, 1, bias=False)
        )

    def forward(self, x, mask):
        text_hidden = self.text_base(x, attention_mask = mask).last_hidden_state.mean(1)
        clf_out = self.clf_head(text_hidden)

        return clf_out

class TextEmbedding(nn.Module):
    def __init__(self, text_model):
        super(TextEmbedding, self).__init__()
        text_model.clf_head[4] = nn.Linear(512, 128, bias = False)
        self.embedding = text_model

    def forward(self, x, mask):
        return self.embedding(x, mask)

class Ensemble(nn.Module):
    """
    Receives combined logits from both image and text model
    passed to L1 : a linear layer (4 parameters)
    then sigmoid output for final decision logit
    """
    def __init__(self, fusion = "concat"):
        super(Ensemble, self).__init__()
        self.fusion = fusion
        if self.fusion == "concat":
            self.d = 256
        elif self.fusion == "element":
            self.d = 128
        elif self.fusion == "sumpool":
            self.w1 = nn.Linear(128, 64, bias = False)
            self.w2 = nn.Linear(128, 64, bias = False)
            self.sumpool = nn.AvgPool1d(4)
            self.d = 16
        elif self.fusion == "gated":
            self.U = nn.Linear(128, 64, bias = False)
            self.sig = nn.Sigmoid()
            self.V = nn.Linear(128, 64, bias = False)
            self.W = nn.Linear(64, 8, bias = False)
            self.d = 8

        self.initial = nn.Linear(self.d, self.d // 2, bias = True)
        self.final = nn.Linear(self.d // 2, 1, bias = True)
        self.act_out = nn.Sigmoid()

    def forward(self, d1, d2):
        if self.fusion == "concat":
            fused = torch.concat([d1, d2], dim = -1)
        elif self.fusion == "element":
            fused = d1 * d2
        elif self.fusion == "sumpool":
            venc = self.w1(d1)
            tenc = self.w2(d2)
            fused = self.sumpool(venc * tenc)
        elif self.fusion == "gated":
            tenc = self.sig(self.U(d1))
            venc = self.V(d2)
            enc = tenc * venc
            fused = self.W(enc)
        hidden = self.initial(fused)
        out = self.final(hidden)

        if self.training == False:
            out = self.act_out(out)
            
        return out
    
for fusion_method in ["concat","element", "sumpool", "gated"]:  # 
    for elr in [1e-5]: #, 2e-5, 3e-5, 4e-5,5e-5
        print("beginning training ", fusion_method, " with learning rate ", elr)
        print("lr = ", elr)
        # fusion_method = "gated"
        llama_clf = TextModel(model).half().to(device)
        vit_clf = img_model.half().to(device)
        stacker = Ensemble(fusion = fusion_method).half().to(device)

        #load pre-trained weights here
    #     llama_clf.load_state_dict(torch.load("codes/multimodal/models/try/tamilllama/model.bin"))
    # #         # load vit here
    #     vit_clf.load_state_dict(torch.load(".../path/..."))
        #load pre-trained weights here

        llama_clf = TextEmbedding(llama_clf).half().to(device)
        vit_clf = ImageModel(vit_clf).half().to(device)

        # freeze llama pre-trained weights
        for p in llama_clf.embedding.text_base.parameters():
            p.requires_grad = False

        for p in llama_clf.embedding.clf_head.parameters():
            p.requires_grad = True

        # freeze vit pre-trained weights
        for p in vit_clf.embedding.parameters():
            p.requires_grad = False

        for p in vit_clf.embedding.head.parameters():
            p.requires_grad = True

        # set training to True for ensemble module
        for p in stacker.parameters():
            p.requires_grad = True

        # list of trainable parameters to pass to optimizer
        params = list(llama_clf.embedding.clf_head.parameters()) + list(vit_clf.embedding.head.parameters()) + list(stacker.parameters())

        # Adam optimizer
        optimizer = torch.optim.Adam(params, lr = elr, eps = 1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience = 2, threshold = 1e-3)
        criterion = nn.BCEWithLogitsLoss()

        # Early stopping parameters
        early_stopping_patience = 5
        best_dev_loss = float('inf')
        epochs_without_improvement = 0
        best_model_state = None
        best_epoch = 0
        final_lr = elr
        
        llama_clf.train()
        vit_clf.train()
        stacker.train()

        for epoch in range(1, epochs + 1):
            # Training phase
            its = 0
            epoch_loss = 0

            with tqdm(train_loader, unit = "batch") as train_epoch:
                for img, lab, text, mask in train_epoch:
                    img = img.to(device)
                    lab = lab.to(device)
                    text = text.to(device)
                    mask = mask.to(device)
                    train_epoch.set_description(f"Epoch {epoch}")

                    optimizer.zero_grad()

                    d1 = llama_clf(text, mask)
                    d2 = vit_clf(img)

                    out = stacker(d1, d2).squeeze(1)
                    batch_loss = criterion(out, lab)

                    batch_loss.backward()
                    optimizer.step()

                    epoch_loss += batch_loss.item()
                    its += 1
            
            epoch_loss /= its

            # Validation phase
            llama_clf.eval()
            vit_clf.eval()
            stacker.eval()
            
            dev_loss = 0
            dev_its = 0
            
            with torch.no_grad():
                with tqdm(dev_loader, unit = "batch") as dev_epoch:
                    for img, lab, text, mask in dev_epoch:
                        img = img.to(device)
                        lab = lab.to(device)
                        text = text.to(device)
                        mask = mask.to(device)
                        dev_epoch.set_description(f"Validation Epoch {epoch}")

                        d1 = llama_clf(text, mask)
                        d2 = vit_clf(img)

                        out = stacker(d1, d2).squeeze(1)
                        batch_loss = criterion(out, lab)

                        dev_loss += batch_loss.item()
                        dev_its += 1
            
            dev_loss /= dev_its
            
            # Update learning rate scheduler
            scheduler.step(dev_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch}: Train Loss: {epoch_loss:.4f}, Dev Loss: {dev_loss:.4f}, LR: {current_lr:.6f}")

            # Early stopping check
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                epochs_without_improvement = 0
                best_epoch = epoch
                final_lr = current_lr
                # Save best model state - only trainable layers (no quantization issues)
                # Get only the ViT head weights from the full state dict
                vit_full_state = vit_clf.state_dict()
                vit_head_state = {k: v for k, v in vit_full_state.items() if 'head' in k}
                
                best_model_state = {
                    'llama_clf_head': llama_clf.embedding.clf_head.state_dict(),
                    'vit_clf_head': vit_head_state,
                    'stacker': stacker.state_dict()
                }
                print(f"New best model found at epoch {epoch} (dev loss: {dev_loss:.4f})")
            else:
                epochs_without_improvement += 1
                
            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs. Best epoch was {best_epoch}")
                break
                
            # Set back to training mode for next epoch
            llama_clf.train()
            vit_clf.train()
            stacker.train()

        # Load best model state - only trainable layers (no quantization issues)
        if best_model_state is not None:
            try:
                # Load only trainable components - no quantization conflicts
                llama_clf.embedding.clf_head.load_state_dict(best_model_state['llama_clf_head'])
                
                # Load ViT head weights using filtered state dict
                vit_current_state = vit_clf.state_dict()
                vit_current_state.update(best_model_state['vit_clf_head'])
                vit_clf.load_state_dict(vit_current_state, strict=False)
                
                stacker.load_state_dict(best_model_state['stacker'])
                print(f"✅ Loaded best trainable layers from epoch {best_epoch}")
            except Exception as e:
                print(f"Warning: Could not load best model state: {e}")
                print("Continuing with current model state...")
        else:
            print(f"Using current model state from epoch {best_epoch}")

        # Find best threshold using development set
        llama_clf.eval()
        vit_clf.eval()
        stacker.eval()
        
        print("Finding best threshold using development set...")
        ths = [x/10 for x in range(1, 10)]  # 0.1 - 0.9
        dev_f1_scores = []

        for th in ths:
            dev_preds = []
            dev_labels = []

            with torch.no_grad():
                for img, lab, text, mask in dev_loader:
                    img, lab, text, mask = img.to(device), lab.to(device), text.to(device), mask.to(device)

                    d1 = llama_clf(text, mask)
                    d2 = vit_clf(img)

                    out = stacker(d1, d2).squeeze(1)
                    pred = (out > th).to(torch.float)
                    
                    dev_preds.extend(pred.cpu().numpy())
                    dev_labels.extend(lab.cpu().numpy())

            # Calculate macro-F1 score
            dev_f1 = f1_score(dev_labels, dev_preds, average='macro', zero_division=0)
            dev_f1_scores.append(dev_f1)
            print(f"Threshold {th:.1f}: Dev Macro-F1: {dev_f1 * 100:.2f}%")

        best_thr = ths[dev_f1_scores.index(max(dev_f1_scores))]
        best_dev_f1 = max(dev_f1_scores)
        print(f"Best threshold: {best_thr} with dev macro-F1: {best_dev_f1 * 100:.2f}%")

        # Final testing with best threshold
        print("Testing with best threshold...")
        preds = []
        tru = []
        test_loss = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            with tqdm(test_loader, unit = "batch") as test_epoch:
                for img, lab, text, mask in test_epoch:
                    img, lab, text, mask = img.to(device), lab.to(device), text.to(device), mask.to(device)
                    test_epoch.set_description(f"Final Testing...")

                    d1 = llama_clf(text, mask)
                    d2 = vit_clf(img)

                    out = stacker(d1, d2).squeeze(1)
                    pred = (out > best_thr).to(torch.float)
                    
                    test_correct += (pred == lab).sum().item()
                    test_total += lab.size(0)
                    
                    batch_loss = criterion(out, lab)
                    test_loss += batch_loss.item()
                    
                    preds.extend(pred.cpu().tolist())
                    tru.extend(lab.cpu().tolist())

        test_accuracy = test_correct / test_total
        test_loss /= len(test_loader)
        
        # Calculate final test macro-F1 score
        test_macro_f1 = f1_score(tru, preds, average='macro', zero_division=0)

        print(f"Final Test Macro-F1: {test_macro_f1 * 100:.2f}%")
        print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
        print(f"Final Test Loss: {test_loss:.5f}")
        
        # Print training summary
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Fusion Method: {fusion_method}")
        print(f"Initial Learning Rate: {elr}")
        print(f"Final Learning Rate: {final_lr:.6f}")
        print(f"Best Epoch: {best_epoch}")
        print(f"Best Threshold: {best_thr}")
        print(f"Dev Macro-F1 at Best Threshold: {best_dev_f1 * 100:.2f}%")
        print(f"Final Test Macro-F1: {test_macro_f1 * 100:.2f}%")
        print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
        print("="*50)
        
        # plot confusion matrix
        ConfusionMatrixDisplay(confusion_matrix(tru, preds)).plot()
        
        # create directories to save models and metrices
        save_confusion_matrix_path = "predictions/"+str(language)+"/fusion/"
        metrices_path = "predictions/"+str(language)+"/fusion/"
        save_model_path = "trained_model/"+str(language)+"/fusion/"
        
        isExist = os.path.exists(save_confusion_matrix_path)
        if not isExist:
            os.makedirs(save_confusion_matrix_path)
        isExist = os.path.exists(metrices_path)
        if not isExist:
            os.makedirs(metrices_path)
        isExist = os.path.exists(save_model_path)
        if not isExist:
            os.makedirs(save_model_path)
            
        # Save only trainable layers (classification heads and fusion layers)
        model_filename = f"custom_{language}_llamavit_fusion{fusion_method}_{elr}_{best_epoch}_{N}"
        torch.save({
            # Save only trainable components
            'llama_clf_head_state_dict': llama_clf.embedding.clf_head.state_dict(),
            'vit_clf_head_state_dict': {k: v for k, v in vit_clf.state_dict().items() if 'head' in k},
            'stacker_state_dict': stacker.state_dict(),
            'fusion_method': fusion_method,
            'learning_rate': elr,
            'final_learning_rate': final_lr,
            'best_epoch': best_epoch,
            'best_threshold': best_thr,
            'dev_macro_f1': best_dev_f1,
            'test_macro_f1': test_macro_f1,
            'test_accuracy': test_accuracy,
            'model_config': {
                'language': language,
                'batch_size': N,
                'epochs_trained': best_epoch,
                'early_stopping_patience': early_stopping_patience
            },
            # Save tokenizer vocab size for proper model reconstruction
            'tokenizer_vocab_size': len(tokenizer)
        }, save_model_path + model_filename + ".pth")
        
        print(f"Trainable layers saved to: {save_model_path + model_filename}.pth")
        print("Saved components:")
        print("  - LLaMA classification head")
        print("  - ViT classification head") 
        print("  - Fusion/stacker module")
        print("  - Training metadata")
        
        # To load the saved model later, use:
        # IMPORTANT: Only trainable layers are saved, base models remain frozen/quantized
        # 1. Load and setup base models (quantized LLaMA + ViT) exactly as in training
        # 2. Wrap in TextModel/ImageModel then TextEmbedding/ImageModel
        # 3. Load only the trainable layer weights
        # Example:
        # checkpoint = torch.load(save_model_path + model_filename + ".pth")
        # # Setup base models exactly as in training...
        # llama_clf.embedding.clf_head.load_state_dict(checkpoint['llama_clf_head_state_dict'])
        # vit_clf.embedding.head.load_state_dict(checkpoint['vit_clf_head_state_dict'])
        # stacker.load_state_dict(checkpoint['stacker_state_dict'])
        # best_threshold = checkpoint['best_threshold']
            
        # save predictions
        preds_df = pd.DataFrame({'predictions':preds, 'true_labels': tru})
        preds_df.to_csv(save_confusion_matrix_path+"custom_"+str(language)+"_llamavit_fusion"+fusion_method+"_"+str(elr)+"_"+str(best_epoch)+"_"+str(N)+".csv", index = False)
        
        print(classification_report(tru, preds, digits=5, zero_division=0))

        cf = confusion_matrix(tru, preds)
        cr = classification_report(tru, preds, digits=5, zero_division=0)
        
        # saving metrices and confusion matrix
        with open(metrices_path+"custom_"+str(language)+"llama_vit_"+fusion_method+"fusion_"+str(N)+".txt", "a") as text_file:
            text_file.write("Date and time of run = "+str(dt)+"\n")
            text_file.write(f"Fusion: {fusion_method}, Initial LR: {elr}, Final LR: {final_lr:.6f}, Best Epoch: {best_epoch}, Threshold: {best_thr}\n")
            text_file.write(f"Dev Macro-F1: {best_dev_f1 * 100:.2f}%, Test Macro-F1: {test_macro_f1 * 100:.2f}%, Test Accuracy: {test_accuracy * 100:.2f}%\n\n")
            text_file.write(str(cf)+"\n")
            text_file.write(str(cr))
            text_file.write("\n\n\n")