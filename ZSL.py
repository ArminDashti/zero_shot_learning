import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image
device='cuda'
#%%
# dataset_dir = 'c:/users/armin/desktop/Animals_with_Attributes2/JPEGImages'
dataset_dir = 'c:/arminpc/Animals_with_Attributes2/JPEGImages'
dataset_folders = os.listdir(dataset_dir)
         
# attribute_dir = 'c:/users/armin/desktop/Animals_with_Attributes2/predicate-matrix-binary.txt'
attribute_dir = 'c:/arminpc/Animals_with_Attributes2/predicate-matrix-binary.txt'
df_attribute = pd.read_csv(attribute_dir, sep=" ", header=None)        
df_attribute['classes'] = dataset_folders
#%%
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Resize((256, 256)),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])

    
batch_size = 8
train_dataset = torchvision.datasets.ImageFolder(dataset_dir, transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

v = next(iter(train_dataloader))
#%%
extraction_features_model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True).to(device)
extraction_features_model.classifier = nn.Sequential(*[extraction_features_model.classifier[i] for i in range(4)])
extraction_features_model.eval()

class ZSL(nn.Module):
    def __init__ (self):
        super().__init__()
        self.Linear1 = nn.Linear(85,700)
        self.Linear2 = nn.Linear(700,4096)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # extracted_features = extraction_features_model(x)
        x = self.Linear1(x)
        x = self.relu(x)
        x = self.Linear2(x)
        x = self.relu(x)
        return x
        
        

#%%
def input_binary(input_batch):
    new_input = []
    for i in input_batch:
        i = i.item()
        i = df_attribute.iloc[i].to_numpy()[:-1]
        new_input.append(i)
    return torch.FloatTensor(new_input)
        
        
    
model = ZSL().to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = nn.MSELoss()

for epoch in range(0,2):
    total_loss = 0
    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        input_batch = batch[1]
        input_batch = input_binary(input_batch).to(device)
        output = batch[0].to(device)
        output = extraction_features_model(output).to(device)
        predict = model(input_batch)
        loss = loss_fn(predict, output)
        print(loss.item())
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        if i==1000: sys.exit()
#%%
total_loss