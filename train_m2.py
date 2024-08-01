from vae import Semisupervised_VAE
import torch.nn as nn
import random
import os
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
#from torchvision import transforms
from torch.utils.data import DataLoader
from collections import defaultdict
from torch.utils.data import Dataset
import numpy as np
import datetime

##check gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##random seed
#torch.backends.cudnn.deterministic = True
#seed=1314
#torch.manual_seed(seed)
#random.seed(seed)
#np.random.seed(seed)
#if torch.cuda.is_available():
#    torch.cuda.manual_seed_all(seed)

###set parameters###
encoder_size0=[1274*21,1274,500,200,15]
latent_size=2
decoder_size0=[15,200,800,1274,1274*21]
classification_dim=[1274*21,100,25]
batch_size0=80
learning_rate=0.001
epochs=50
alpha=0.05


##amino acid label
aa = 'ACDEFGHIKLMNPQRSTVWY-'
aa2idx = {}
for i in range(len(aa)):   
    aa2idx[aa[i]] = i

#label library
label_library= ['B.1.617.1(Kappa)', '21C Epsilon', 'BA.4&5', 'B.1.351(Beta)', 'B.1.1.7(Alpha)',
                'BA.2.75',  'B.1.526(Iota)', 'P.1(Gamma)', 'Original',  '20A.EU2', 
                'B.1.617.2(Delta)', 'B.1.621(Mu)', 'C.37(Lambda)', 'BA.2', 'BA.2.12.1',
                'BA.2.86', '21I Delta', 'BA.1', '20E EU1',  'XBB', 'B.1.525(Eta)']

##define loss function 


def label_loss_fn(recon_x, x, mean, log_var,y):
    
        BCE = torch.nn.functional.binary_cross_entropy(
            recon_x, x.view(-1,1274*21), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        
        priorlabel=torch.nn.functional.softmax(torch.ones_like(y),dim=0)
        priorlabel.requires_grad = False
        return (BCE + KLD) / x.size(0)+categorical_crossentropy(y, priorlabel)/x.size(0)



def categorical_crossentropy(y,y_target):
    return -torch.sum(y*torch.log(y_target+1e-8))

def cross_entropy_loss(y,y_target):
    
    criterion =  torch.nn.functional.binary_cross_entropy(y , y_target,reduction='sum')

    return -criterion/y.size(0)
         
def unlabel_vae_loss(recon_x,x,mean,log_var,outputlabel):
    
    p=torch.zeros(classification_dim[-1])
    p[0]=1

    prior=torch.nn.functional.softmax(torch.ones_like(p),dim=0)
    prior.requires_grad = False
    cross_entropy0 = -torch.sum(p * torch.log(prior + 1e-8))
    entropy=categorical_crossentropy(outputlabel, outputlabel)
    
    BCE = torch.sum(torch.nn.functional.binary_cross_entropy(
        recon_x, x.view(-1,1274*21),reduction='none'),1)
    KLD = -0.5 * torch.sum((1 + log_var - mean.pow(2) - log_var.exp()),1)
    BCE=BCE.unsqueeze(1)
    KLD=KLD.unsqueeze(1)
    
    labeled_loss=BCE+KLD+cross_entropy0
    #outputlabel=torch.max(outputlabel,1).values


    return torch.sum(outputlabel*labeled_loss)/x.size(0) + entropy/x.size(0)
    

    
##reshape it into a one-hot encoded tensor 
def idx2onehot(idx, n):
    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)

    return onehot


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


    
# def get_label(data, library):
#     labeled_index = []
#     n = len(data)
#     for i in range(n):
#         label = data[i]
#         if label in library:
#             labeled_index.append(library.index(label))
#         else:
#             labeled_index.append(False)
                        
#     return labeled_index

def get_label(label,library=label_library):

    if label in library:
        labelseq=library.index(label)+1
        
    else:
        labelseq=0
        
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

if __name__ == "__main__":
    ##your data and data preparation

    path=r'monthly-cleanpd-0114/AllUnique_0114-(Corrected).xlsx'
    dataset_oringal_initial=pd.read_excel(path)
    dataset_oringal=dataset_oringal_initial[dataset_oringal_initial['MonthIndex']<22]##used for a trial, delete it when running 
    dataset_oringal0=dataset_oringal['mutation|insertion info']
    #dataset_oringal0=dataset_oringal['mutation']
    seq_class=dataset_oringal['class'].copy()


    ##splite into label and unlabel

    ratio=0.1

    n_sample=len(seq_class)

    selected_index = random.sample(range(n_sample), int(ratio * n_sample))

    for i in selected_index:
        seq_class.loc[i]='no_label'




    dataset=SequenceDataset(dataset_oringal0,num_seq=len(dataset_oringal0),label=seq_class)


    # vae = VAE(encoder_layer_sizes=encoder_size0,
    #         latent_size=latent_size,
    #         decoder_layer_sizes=decoder_size0).to(device)

    model=Semisupervised_VAE(encoder_layer_sizes=encoder_size0,
                              latent_size=latent_size, decoder_layer_sizes=decoder_size0,
                              classify_layer=classification_dim).to(device)

    # Load the saved model parameters
    #vae.load_state_dict(torch.load('vae_model0.pth'))
    model.load_state_dict(torch.load('vae_model_try.pth',map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)





    ###split the data into train and validation

    validation_ratio = 0.1

    # Calculate the number of samples to include in the validation set
    val_size = int(validation_ratio * len(dataset))


    # Randomly split the dataset into training and validation sets
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset) - val_size, val_size])


    data_loader_val = DataLoader(dataset=val_dataset, batch_size=batch_size0, shuffle=True)


    #data_loader = DataLoader(dataset=dataset, batch_size=batch_size0, shuffle=True)



    ####start training######

    ts = time.time()
    logs = defaultdict(list)

    data_loader = DataLoader(
                    dataset=train_dataset, batch_size=batch_size0, shuffle=True)

    for epoch in range(epochs):
            model.train()
            
            tracker_epoch = defaultdict(lambda: defaultdict(dict))
            
            
            for iteration, x in enumerate(data_loader):
                
                x,y=x            
                x = x.to(device)
                y=y.to(device)
                
                labeled_seq_index=torch.nonzero(y).view(-1).tolist()
                
                unlabeled_seq_index=torch.nonzero(y==0).view(-1).tolist()
                
                labeled_seq_onehot = idx2onehot(y[labeled_seq_index],n=classification_dim[-1])
                
                seq_onehot = idx2onehot(y,n=classification_dim[-1])
                
                recon_x, mean, log_var, z ,newlabel= model(x,seq_onehot)
                


                label_loss=label_loss_fn(recon_x[labeled_seq_index], x[labeled_seq_index],
                                         mean[labeled_seq_index], log_var[labeled_seq_index],newlabel[labeled_seq_index])
                
                if len(unlabeled_seq_index)==0:
                    unlabel_loss=0
                else:

                    unlabel_loss=unlabel_vae_loss(recon_x[unlabeled_seq_index],x[unlabeled_seq_index],
                                               mean[unlabeled_seq_index],log_var[unlabeled_seq_index],
                                               newlabel[unlabeled_seq_index])
                
                classifify_loss=cross_entropy_loss(newlabel[labeled_seq_index],labeled_seq_onehot)
                #loss = loss_fn(recon_x, x, mean, log_var)
                loss=label_loss+unlabel_loss-alpha*classifify_loss
                

                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #logs['loss'].append(loss.item())
                tracker_epoch['train']['loss'][iteration]=loss.item()
                

                if iteration % 200 == 0 or iteration == len(data_loader)-1:
                    print('train:',epoch, iteration, len(data_loader)-1, loss.item())

            with torch.no_grad():
                model.eval()        
                for val_iter,x_val in enumerate(data_loader_val):
                    x_val,y_val=x_val
                    x_val=x_val.to(device)
                    y_val=y_val.to(device)

                    seq_onehot_val = idx2onehot(y_val,n=classification_dim[-1])
         
                    labeled_seq_index_val=torch.nonzero(y_val).view(-1).tolist()
                    unlabeled_seq_index_val=torch.nonzero(y_val==0).view(-1).tolist()
     
                    labeled_seq_onehot_val = idx2onehot(y_val[labeled_seq_index_val],n=classification_dim[-1])


                    recon_x_val, mean_val, log_var_val, z_val,newlabel_val = model(x_val,seq_onehot_val)
                    
                    label_loss_val=label_loss_fn(recon_x_val[labeled_seq_index_val], x_val[labeled_seq_index_val],
                                             mean_val[labeled_seq_index_val], log_var_val[labeled_seq_index_val],newlabel_val[labeled_seq_index_val])
                    
                    if len(unlabeled_seq_index_val)==0:
                        unlabel_loss_val=0
                    else:
                        unlabel_loss_val=unlabel_vae_loss(recon_x_val[unlabeled_seq_index_val],x_val[unlabeled_seq_index_val],
                                                   mean_val[unlabeled_seq_index_val],log_var_val[unlabeled_seq_index_val],
                                                   newlabel_val[unlabeled_seq_index_val])


                    
                    classifify_loss_val=cross_entropy_loss(newlabel_val[labeled_seq_index_val],labeled_seq_onehot_val)

                    loss_val=label_loss_val+unlabel_loss_val-alpha*classifify_loss_val
                    #loss_val=loss_fn(recon_x_val,x_val,mean_val,log_var_val)
                    
                    if val_iter % 20 == 0 or val_iter == len(data_loader_val)-1:
                        print('validation',epoch, val_iter, len(data_loader_val)-1, loss_val.item())
                    
                    #logs['val_loss'].append(loss_val.item())
                    tracker_epoch['val']['loss'][val_iter]=loss_val.item()
            avg_train_loss=sum(tracker_epoch['train']['loss'].values()) / len(tracker_epoch['train']['loss'])
            avg_val_loss = sum(tracker_epoch['val']['loss'].values()) / len(tracker_epoch['val']['loss'])
            logs['train_loss'].append(avg_train_loss)
            logs['val_loss'].append(avg_val_loss)

    te=time.time()
    print('time',-ts+te)

    torch.save(model.state_dict(), 'vae_model_try.pth')

    # Create a dictionary with column labels and corresponding data arrays
    data = {'Train Loss': np.array(logs['train_loss']), 'Validation Loss': np.array(logs['val_loss'])}

    # Create the DataFrame with column labels
    loss_save = pd.DataFrame(data)

    loss_save.to_excel('lossfunction.xlsx')

    plt.figure()
    plt.plot(np.log(logs['train_loss']))
    plt.plot(np.log(logs['val_loss']))
    plt.title('loss')
    plt.savefig('loss_function')
    now = datetime.datetime.now()
    print('train finished',now)
