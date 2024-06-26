#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 09:26:39 2024

"""
import torch.nn as nn
import torch
#from torchvision import transforms

##reshape it into a one-hot encoded tensor 
def idx2onehot(idx, n):

    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    
    return onehot


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
    
class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size):

        super().__init__()

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="batchnormalization{:d}".format(i), module=nn.BatchNorm1d(out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)
        #construct nn linear networks, layer_sizes input should like [200,10,5], which means 200->10->5

    def forward(self, x):
        
        x=x.view(-1,1274*21)
        
        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars
#decoder

class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size):

        super().__init__()

        self.MLP = nn.Sequential()

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
    
    



