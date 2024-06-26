#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 09:26:39 2024

"""
import torch.nn as nn
import torch
#from torchvision import transforms


###VAE model##
class VAE(nn.Module):
    
    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes):

        super().__init__()
        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size)
    
    def forward(self, x):
         x=x.view(-1,1274*21)
         means, log_var = self.encoder(x)
         z = self.reparameterize(means, log_var)
         recon_x = self.decoder(z)

         return recon_x, means, log_var, z
     
    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std
    
    def inference(self, z):
        
        recon_x = self.decoder(z)
        return recon_x
    
 ##encoder   
 
 
class Semisupervised_VAE(nn.Module):
    
    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,classify_layer):

        encoder_layer_sizes[0]=encoder_layer_sizes[0]+classify_layer[-1]
        super().__init__()
        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes,latent_size)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size+classify_layer[-1])
        self.classifier=Classifier(classify_layer)
        self.dropout = nn.Dropout(0.1) 
    
    def forward(self, x,y):

         x,y0 =self.classifier(x)
         
         means, log_var = self.encoder(torch.cat([x,y],dim=1))
         

         z = self.reparameterize(means, log_var)


         recon_x = self.decoder(torch.cat([z,y],dim=1))

         return recon_x, means, log_var, z, y0
     
    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std
    
    def inference(self, z,y):
        
        recon_x = self.decoder(torch.cat([z,y],dim=1))

        return recon_x
    
class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size):

        super().__init__()

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="batchnormalization{:d}".format(i), module=nn.BatchNorm1d(out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            
        self.dropout = nn.Dropout(0.1) 

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)
        #construct nn linear networks, layer_sizes input should like [200,10,5], which means 200->10->5

    def forward(self, x):
        
        
        x = self.MLP(x)
        

        means = self.linear_means(x)
        
        log_vars = self.linear_log_var(x)

        return means, log_vars
#decoder

class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size):

        super().__init__()

        self.MLP = nn.Sequential()
        self.dropout = nn.Dropout(0.1)
        input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="batchnormalization{:d}".format(i), module=nn.BatchNorm1d(out_size))
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z):
        x = self.MLP(z)
        
        return x
    



class Classifier(nn.Module):
    
    def __init__(self, classification_dim):
        """
        Single hidden layer classifier
        with softmax output.
        """
        super(Classifier, self).__init__()
        self.MLP=nn.Sequential()
        for i, (in_size,out_size) in enumerate(zip(classification_dim[:-1],classification_dim[1:])):
            self.MLP.add_module(name="classification{:d}".format(i), module=nn.Linear(in_size,out_size))
            self.MLP.add_module(name="batchnormalization{:d}".format(i), module=nn.BatchNorm1d(out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
              
        self.MLP.add_module(name='softmax', module=nn.Softmax(dim=-1))


    def forward(self, x):
        x=x.view(-1,1274*21)
        y=self.MLP(x)
        return x,y
