
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


##here I only use vae loss to check the training result

def loss_fn(recon_x, x, mean, log_var):
    
        BCE = torch.nn.functional.binary_cross_entropy(
            recon_x, x.view(-1), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return (BCE + KLD)
#same as training parameters

encoder_size0=[1274*21,1274,200,15]
latent_size=2
decoder_size0=[15,200,1274,1274*21]
classification_dim=[1274*21,100,25]

##load your data here
path=r'sequence_data.xlsx'


dataset_oringal=pd.read_excel(path)
dataset_oringal0=dataset_oringal['mutation|insertion info']

dataclass=dataset_oringal['class'].tolist()
print(set(dataclass))


aa = 'ACDEFGHIKLMNPQRSTVWY-'
aa2idx = {}
for i in range(len(aa)):   
    aa2idx[aa[i]] = i

def mutation_matrix (data):
    mutation= torch.zeros((1274,21))
    if isinstance(data, str) and not pd.isnull(data):
        items0=data.split('|')
        items=str(items0[0])
        #items=data
        items=items.split(';')
        for site in items:
            pos =int(site[1:-1])
            aap = site[-1]
            mutation[pos][aa2idx[aap]] = 1
    return mutation 


label_library=[ 'B.1.617.1(Kappa)', '21C Epsilon', 
                'B.1.351(Beta)', 'S:677H.Robin1', 
               'B.1.1.7(Alpha)', 'B.1.526(Iota)',  
                'P.1(Gamma)',
               'Original',  '20A.EU2', 
               'B.1.617.2(Delta)', 'B.1.621(Mu)', 'C.37(Lambda)', 
               '21I Delta', 'others', '20E EU1', 'B.1.525(Eta)']

def get_label(label,library=label_library):

    if label in library:
        labelseq=library.index(label)+1
        
    else:
        labelseq=False
        
    return labelseq

class SequenceDataset(Dataset):
    
    def __init__(self, samples,num_seq,label):
        self.samples = samples
        self.num_seq =num_seq
        self.label=label
    
    
    def __len__(self):
        return self.num_seq
    

    def __getitem__(self, index):
        tensor=mutation_matrix(self.samples[index])
        label=get_label(self.label[index])
        
        return tensor,label

dataset=SequenceDataset(dataset_oringal0,num_seq=len(dataset_oringal0),label=dataclass)



def idx2onehot(idx, n):
    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)

    return onehot

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

##for input one seq:
mutation_try='L18F;T20N;P26S;D138Y;R190S;K417T;E484K;N501Y;D614G;H655Y;L841F;T1027I;V1176F|""'


mutation_try0=mutation_matrix(mutation_try)

label0=0

label0onr_hot=idx2onehot(torch.tensor([label0]), n=classification_dim[-1])

print(label0onr_hot)

mutation_try0=mutation_try0.to(device)
label0onr_hot=label0onr_hot.to(device)

recon_x, mean, log_var, z,label = vae_model(mutation_try0,label0onr_hot)

print('label',label)


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

##for amonly detection

re_error=[]
var0=[]
z1=[]
z2=[]
selected_inex=0
for i in range(len(dataset_oringal0)):
    mutation0=dataset[i][0]
    data_label=dataset[i][1]
    if data_label==False:
        seq_label_onehot=idx2onehot(torch.tensor([0]), n=classification_dim[-1])
    else:
        seq_label_onehot=idx2onehot(torch.tensor([data_label]), n=classification_dim[-1])
    #mutation0=mutation_matrix(data)
    #mutation0=mutation0.view(-1,1274*21)
    mutation0=mutation0.to(device)
    seq_label_onehot=seq_label_onehot.to(device)
    recon_x, mean, log_var, z,label = vae_model(mutation0,seq_label_onehot)
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

##plot lantent space
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

values = ['B.1.617.1(Kappa)', '21C Epsilon', 
                'B.1.351(Beta)', 'S:677H.Robin1', 
               'B.1.1.7(Alpha)', 'B.1.526(Iota)',  
                'P.1(Gamma)',
        
               'B.1.617.2(Delta)', 'B.1.621(Mu)', 'C.37(Lambda)', 
               '21I Delta',  'B.1.525(Eta)']

values0=['B.1.617.1(Kappa)', '21C Epsilon', 'BA.4&5', 'B.1.351(Beta)',
          'B.1.1.7(Alpha)', 'BA.2.75',  'B.1.526(Iota)', 'P.1(Gamma)', 
          'B.1.617.2(Delta)', 'B.1.621(Mu)', 'C.37(Lambda)', 'BA.2',
            'BA.2.12.1', 'BA.2.86', 'BA.1', 'XBB', 'B.1.525(Eta)']


values=['B.1.1.7(Alpha)', 'B.1.526(Iota)', 'P.1(Gamma)', 'B.1.617.1(Kappa)', '21C Epsilon','B.1.351(Beta)',
          'B.1.621(Mu)', 'C.37(Lambda)', 'BA.2.75',  'BA.2', 'BA.2.12.1', 'BA.2.86', 'BA.1', 'XBB']

fig=plt.figure(figsize=(25,12))

for value in values:
    position = [index for index, item in enumerate(dataclass) if item == value]
    pos_var1 = [z1[position] for position in position]
    pos_var2 = [z2[position] for position in position]
    
    plt.scatter(pos_var1, pos_var2, alpha=0.6, label=value)

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize="15")
plt.savefig('plot.png')


for value in values0:
    position = [index for index, item in enumerate(dataclass) if item == value]
    pos_var1 = [z1[position] for position in position]
    pos_var2 = [z2[position] for position in position]
    
    plt.scatter(pos_var1, pos_var2, alpha=0.6, label=value)

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize="15")
plt.savefig('plotvar.png')









