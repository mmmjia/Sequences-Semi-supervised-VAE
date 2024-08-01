#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:27:50 2024

@author: mengqijia
"""

import torch.nn as nn
import os
import time
import torch
import argparse
import pandas as pd
import matplotlib.pyplot as plt
#from torchvision import transforms
from torch.utils.data import DataLoader
from collections import defaultdict
from torch.utils.data import Dataset
import numpy as np
import sys
from vae import Semisupervised_VAE
from train_m2 import *

def loss_fn(recon_x, x, mean, log_var):
    
        BCE = torch.nn.functional.binary_cross_entropy(
            recon_x, x.view(-1), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return (BCE + KLD)



##your data and data preparation
#path = r'C:/Users/user/OneDrive - HKUST Connect/cleanproteindata/monthly-cleanpd-0114/2021-05.xlsx'
path=r'monthly-cleanpd-0114/AllUnique_0114-(Corrected).xlsx'


dataset_oringal_initial=pd.read_excel(path)
dataset_oringal=dataset_oringal_initial[dataset_oringal_initial['MonthIndex']<22]
dataset_oringal0=dataset_oringal['mutation|insertion info']


dataclass=dataset_oringal['class'].tolist()
print(set(dataclass))


dataset=SequenceDataset(dataset_oringal0,num_seq=len(dataset_oringal0),label=dataclass)


# Assuming you have defined your VAE model architecture
vae_model = Semisupervised_VAE(encoder_layer_sizes=encoder_size0,
                          latent_size=latent_size, decoder_layer_sizes=decoder_size0,
                          classify_layer=classification_dim).to(device)

# Load the saved model parameters
vae_model.load_state_dict(torch.load('vae_model_try.pth',map_location=device))
vae_model.eval()


# Set requires_grad to False for each parameter
for param in vae_model.parameters():
    param.requires_grad = False

mutation_try='L18F;T20N;P26S;D138Y;R190S;K417T;E484K;N501Y;D614G;H655Y;L841F;T1027I;V1176F|""'


mutation_try0=mutation_matrix(mutation_try)

label0=0

label0onr_hot=idx2onehot(torch.tensor([label0]), n=classification_dim[-1])

print(label0onr_hot)

mutation_try0=mutation_try0.to(device)
label0onr_hot=label0onr_hot.to(device)

recon_x, mean, log_var, z, label = vae_model(mutation_try0,label0onr_hot)

largest_value, label_index = torch.max(label.view(-1), 0)

print('label:',label_library[label_index-1],'value:',largest_value)


error=torch.nn.functional.binary_cross_entropy(recon_x.view(-1),mutation_try0.view(-1),reduction='sum')
print('construction error',error)
error=loss_fn(recon_x.view(-1), mutation_try0, mean, log_var)


print('error',error.item())
print('lantent',z)

recon_x[recon_x<0.05]=0

nonzero_indices = torch.nonzero(recon_x[0])

print("Indices of non-zero elements:", nonzero_indices/21,'value',recon_x[0][nonzero_indices])
print('------------------------------')
print('number of the seqs',len(dataset_oringal0))




re_error=[]
var0=[]
z1=[]
z2=[]
label_error=0
selected_inex=0
for i in range(len(dataset_oringal0)):
    mutation0=dataset[i][0]
    data_label=dataset[i][1]
    if data_label==0:
        seq_label_onehot=idx2onehot(torch.tensor([0]), n=classification_dim[-1])
    else:
        seq_label_onehot=idx2onehot(torch.tensor([data_label]), n=classification_dim[-1])
    #mutation0=mutation_matrix(data)
    #mutation0=mutation0.view(-1,1274*21)
    mutation0=mutation0.to(device)
    seq_label_onehot=seq_label_onehot.to(device)
    recon_x, mean, log_var, z,label = vae_model(mutation0,seq_label_onehot)
    largest_value, label_index = torch.max(label.view(-1), 0)
    if data_label>0:
         if data_label!=label_index:
              label_error=label_error+1
    error=loss_fn(recon_x.view(-1), mutation0, mean, log_var)
    re_error.append(error.item())
    if error>1000:
        print(dataset_oringal0[i],dataclass[i],'error',error)
        selected_inex=selected_inex+1
    z1.append(z[0][0].item())
    z2.append(z[0][1].item())
    
plt.plot(re_error)
plt.savefig('reconstruct_error.png')
plt.close()

print(f"selected seqs number: {selected_inex}")

print('percentage:' ,selected_inex/len(dataset_oringal0))

print('average error',sum(re_error)/len(re_error))

print('wrong label', label_error)

plt.scatter(z1,z2,s=0.1)
plt.savefig('vaeall.png')
plt.close()

'''
import seaborn as sns
plt.figure()
sns.kdeplot(x=z1,y=z2, cmap='viridis', shade=True)
plt.savefig('kde_plot.png')
plt.close()
'''
import matplotlib.pyplot as plt


values0=['B.1.617.1(Kappa)', '21C Epsilon', 'BA.4&5', 'B.1.351(Beta)',
          'B.1.1.7(Alpha)', 'BA.2.75',  'B.1.526(Iota)', 'P.1(Gamma)', 
          'B.1.617.2(Delta)', 'B.1.621(Mu)', 'C.37(Lambda)', 'BA.2',
            'BA.2.12.1', 'BA.2.86', 'BA.1', 'XBB', 'B.1.525(Eta)']


values=['B.1.1.7(Alpha)', 'B.1.526(Iota)', 'B.1.617.1(Kappa)', '21C Epsilon','B.1.351(Beta)',
          'B.1.621(Mu)', 'C.37(Lambda)', 'BA.2.75',  'BA.2', 'BA.2.12.1', 'BA.2.86', 'BA.1', 'XBB']

fig=plt.figure(figsize=(25,12))

for value in values:
    position = [index for index, item in enumerate(dataclass) if item == value]
    pos_var1 = [z1[position0] for position0 in position]
    pos_var2 = [z2[position0] for position0 in position]
    
    plt.scatter(pos_var1, pos_var2, alpha=0.6, label=value)

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize="15")
plt.savefig('plot.png')

fig=plt.figure(figsize=(25,12))

for value in values0:
    position = [index for index, item in enumerate(dataclass) if item == value]
    pos_var1 = [z1[position0] for position0 in position]
    pos_var2 = [z2[position0] for position0 in position]
    
    plt.scatter(pos_var1, pos_var2, alpha=0.6, label=value)

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize="15")
plt.savefig('plotvar.png')