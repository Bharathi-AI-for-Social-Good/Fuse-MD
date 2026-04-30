# %% [code]
import numpy as np 
import pandas as pd 
import torch
import time
import pandas as pd
import os
os.chdir("/home/rahpon/projects/Misogyny_meme1")
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch import nn
import matplotlib.pyplot as plt

# Importing Tamil Llama(2) model from HF
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

double_quant_config = BitsAndBytesConfig(
    # load_in_4bit=True,
    # bnb_4bit_use_double_quant=True,
    load_in_8bit=True,
)

tokenizer = AutoTokenizer.from_pretrained("abhinand/tamil-llama-7b-base-v0.1", low_cpu_mem_usage=True, torch_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained("abhinand/tamil-llama-7b-base-v0.1", low_cpu_mem_usage=True, quantization_config=double_quant_config, torch_dtype=torch.float16)

# device = GPU/CPU as available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Add location of dataset folder containing Tamil and Malayalam folders
Add language as name of folder
"""
folder = os.path.join("training_data")
language = "tamil"  # set accordingly
dpth = os.path.join(folder, language)

# hyper-parameters
N = 16        # batch-size
epochs = 10   # number of epochs
# LR = 4e-5     # learning rate

"""
Llama Tamil doesnt contain [PAD] token
Necessary for processing dataset as batch to have all same size
Add [PAD] as special token
"""
special_tokens_dict = {"pad_token" : '[PAD]'}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
tok_id = tokenizer.convert_tokens_to_ids('[PAD]')
model.resize_token_embeddings(len(tokenizer))                           # resize embedding matrix (+1) as new token added

# Only Text dataloader
class LanData(Dataset):
    """
    Add dataset path got above as parameter(dpth)
    mode = train/test, to get training/testing dataset (saved as test and train folders in language folder)
    """

    def __init__(self, mode, dpth):
        self.path = os.path.join(dpth, mode)
        tscpts = os.path.join(self.path, mode + ".csv")
        self.csv = pd.read_csv(tscpts)
        self.csv = self.csv.dropna(axis = 0)                            # Obtain csv, drop NULLs (text or label not present)

        toks = tokenizer(self.csv.transcriptions.tolist(), truncation=True, max_length=75, padding = 'max_length', return_tensors = 'pt')
        self.csv["tokens"] = toks.input_ids.tolist()
        self.csv["masks"] = toks.attention_mask.tolist()

        labs = []
        scpts = []
        ms = []

        for f in os.listdir(self.path):
            if ".csv" in f:
                continue

            # move on only if valid datapoint, i.e. has a transcript
            iid = int(f.split('.')[0])
            if iid not in self.csv.image_id.tolist():
                continue
            
            #label
            l = self.csv[self.csv.image_id == iid]["labels"].astype(float)
            labs.append(l)

            # transcripts
            s = self.csv[self.csv.image_id == iid]["tokens"].tolist()[0]
            scpts.append(s)

            # masks
            m = self.csv[self.csv.image_id == iid]["masks"].tolist()[0]
            ms.append(m)

        self.labels = labs
        self.tokens = scpts
        self.masks = ms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, ix):
        return torch.tensor(self.labels[ix].item()).to(torch.float16), torch.tensor(self.tokens[ix]), torch.tensor(self.masks[ix]).to(torch.float16)

# Train and test data modules
train_data = LanData("train", dpth)
# test_data = LanData("test", dpth)

# Train and test PyTorch dataloaders
train_loader = DataLoader(train_data, batch_size = N, shuffle = True, drop_last = True)
# test_loader = DataLoader(test_data, batch_size = N, shuffle = True, drop_last = True)

"""
    Text only classification model using Tamil Llama - abhinand/tamil-llama-7b-base-v0.1
    For classification, keep model (base transformer) part of Llama, pre trained on Language Modelling task, and remove the lm_head
    Add classification head as "self.clf_head" shown in Model Class below.
"""
class TextModel(nn.Module):
    def __init__(self, llama):
        super(TextModel, self).__init__()
        self.text_base = llama.model
        self.clf_head = nn.Sequential(
            nn.Linear(4096, 512, bias = False),
            nn.LeakyReLU(0.3, inplace = True),
            nn.Dropout(0.5, inplace = False),
            nn.Linear(512, 1)
        )

    def forward(self, x, mask):
        text_hidden = self.text_base(x, attention_mask = mask).last_hidden_state.mean(1)
        clf_out = self.clf_head(text_hidden)

        return clf_out

llama_clf = TextModel(model).half().to(device)

"""
    First keep base parameters fixed and train only the clf_head with a higher learning rate
    Then make whole model trainable and lower learning rate to fine tune all weights
"""
for p in llama_clf.text_base.parameters():
    p.requires_grad = False

for p in llama_clf.clf_head.parameters():
    p.requires_grad = True

params = list(llama_clf.clf_head.parameters())
optimizer = torch.optim.Adam(params, lr = LR, eps = 1e-4)
criterion = nn.BCEWithLogitsLoss()

llama_clf.train()

for epoch in range(1, epochs+1):
    its = 0
    epoch_loss = 0
    best_loss = 1_000_000.

    with tqdm(train_loader, unit = "batch") as train_epoch:
        for lab, text, mask in train_epoch:
            lab = lab.to(device)
            text = text.to(device)
            mask = mask.to(device)
            train_epoch.set_description(f"Epoch {epoch}")
            batch_loss = 0

            optimizer.zero_grad()

            out = llama_clf(text, mask).squeeze(1)
            batch_loss = criterion(out, lab)

            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item()
            its += 1

    epoch_loss /= its

    print("Average epoch loss: ", epoch_loss)
#     torch.save(llama_clf.state_dict(), "codes/multimodal/models/try/TamilLlamaClassifier.pt")
# llama_clf.save_model("codes/multimodal/models/try/TamilLlamaClassifier/")

tokenizer.save_vocabulary("codes/multimodal/models/try/tamilllama/")
torch.save(llama_clf.state_dict(), "codes/multimodal/models/try/tamilllama/model.bin")

