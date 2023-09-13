import pdb
import numpy as np
import pandas as pd
import networkx as nx
import wntr
import torch
import torch.nn as nn

import matplotlib.pyplot as plt


class WassersteinAutoencoder(nn.Module):

    def __init__(
        self,
        encoder_args: dict,
        decoder_args: dict,
        ) -> None:

        super().__init__()

        self.encoder = Encoder(**encoder_args)
        self.decoder = Decoder(**decoder_args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x
    

    def get_leak_location(self, x: torch.Tensor) -> torch.Tensor:

        latent = self.encoder(x)

        leak_location_loss = torch.zeros(x.shape[0], 4)
        for i in range(self.decoder.num_pars):
            pars = i*torch.ones(x.shape[0], 1, dtype=torch.int32)

            out = self.decoder(latent, pars)

            leak_location_loss[:, i] = torch.sum(torch.abs(out - x)**2, dim=1)

        leak_location_values, leak_location_indices = torch.min(leak_location_loss, dim=1)
        
        return leak_location_indices

class Encoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: int,
        latent_dim: int,
        ) -> None:

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        self.activation = nn.LeakyReLU()

        self.input_layer = nn.Linear(self.input_dim, self.hidden_dims[0])
        self.input_batch_norm = nn.BatchNorm1d(self.hidden_dims[0])
        
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(len(self.hidden_dims) - 1):
            self.hidden_layers.append(
                nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1])
            )
            self.batch_norms.append(
                nn.BatchNorm1d(self.hidden_dims[i+1])
            )
            
        self.output_layer = nn.Linear(self.hidden_dims[-1], self.latent_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.input_layer(x)
        x = self.activation(x)
        #x = self.input_batch_norm(x)

        for hidden_layer, batch_norm in zip(self.hidden_layers, self.batch_norms):
            x = hidden_layer(x)
            x = self.activation(x)
            #x = batch_norm(x)


        x = self.output_layer(x)

        return x
    
class Decoder(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: int,
        output_dim: int,
        num_pars: int = None,
        ) -> None:

        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_pars = num_pars

        self.activation = nn.LeakyReLU()

        if self.num_pars is not None:
            self.pars_embedding = nn.Embedding(num_pars, latent_dim)
            self.input_layer = nn.Linear(self.latent_dim + self.latent_dim, self.hidden_dims[0])
        else:
            self.input_layer = nn.Linear(self.latent_dim, self.hidden_dims[0])
        
        self.input_batch_norm = nn.BatchNorm1d(self.hidden_dims[0])

        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(len(self.hidden_dims) - 1):
            self.hidden_layers.append(
                nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1])
            )
            self.batch_norms.append(
                nn.BatchNorm1d(self.hidden_dims[i+1])
            )
            
        self.output_layer = nn.Linear(self.hidden_dims[-1], self.output_dim, bias=False)

    def forward(self, x: torch.Tensor, pars: torch.Tensor = None) -> torch.Tensor:

        if self.num_pars is not None:
            pars = self.pars_embedding(pars)
            pars = torch.squeeze(pars, dim=1)

            x = torch.cat((x, pars), dim=1)

        x = self.input_layer(x)
        x = self.activation(x)
        #x = self.input_batch_norm(x)

        for hidden_layer, batch_norm in zip(self.hidden_layers, self.batch_norms):
            x = hidden_layer(x)
            x = self.activation(x)
            #x = batch_norm(x)

        x = self.output_layer(x)

        return x