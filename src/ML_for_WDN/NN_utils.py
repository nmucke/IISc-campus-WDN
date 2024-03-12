import pdb
import numpy as np
import pandas as pd
import networkx as nx
import tqdm
import wntr
import torch
import torch.nn as nn

from ML_for_WDN.WAE import WassersteinAutoencoder


def train_WAE(
    data: np.ndarray,
    model: WassersteinAutoencoder,
    train_args: dict,
    device: torch.device,
    supervised_pars: np.ndarray = None,
    contrastive_loss: bool = False
) -> None:
    
    epochs = train_args.get('epochs')
    batch_size = train_args.get('batch_size')
    loss_fn = train_args.get('loss_fn')
    lr = train_args.get('lr')
    weight_decay = train_args.get('weight_decay')

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
    )

    model = model.to(device)
    model.train()

    pbar = tqdm.tqdm(
        range(0, epochs),
        total=epochs,
        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
    )

    data = torch.tensor(data, dtype=torch.float32)
    data = data.to(device)
    if supervised_pars is not None:
        supervised_pars = torch.tensor(supervised_pars, dtype=torch.int32).unsqueeze(1)
        supervised_pars = supervised_pars.to(device)
    
    batch_pars = None
    for epoch in pbar:

        # shuffle data
        idx = np.random.permutation(data.shape[0])
        data = data[idx, :]
        if supervised_pars is not None:
            supervised_pars = supervised_pars[idx]
                    
        total_loss = 0
        for batch_idx in range(0, data.shape[0], batch_size):
            
            batch_data = data[batch_idx:batch_idx+batch_size, :]
            
            if supervised_pars is not None:
                batch_pars = supervised_pars[batch_idx:batch_idx+batch_size]
                #batch_pars = batch_pars.to(device)

            #batch_data = batch_data.to(device)

            optimizer.zero_grad()
            
            latent = model.encoder(batch_data)


            z = torch.randn_like(latent)
            latent_loss = MMD(latent, z, kernel='rbf', device=device)

            output = model.decoder(latent, batch_pars)
            loss = loss_fn(output, batch_data) + 1e-3*latent_loss
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        pbar.set_description(f'Loss: {total_loss:.4f} | Latent: {latent_loss.item():.4f}')

    model.eval()
    model.to('cpu')
    data.to('cpu')

    return None



def MMD(
    x: torch.Tensor, 
    y: torch.Tensor,
    kernel: str,
    device: str
    ) -> torch.Tensor:
    """
    Emprical maximum mean discrepancy. The lower the result, 
    the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
        '''
        C = 2*x.shape[-1]*1
        XX += C * (C + dxx)**-1
        YY += C * (C + dyy)**-1
        XY += C * (C + dxy)**-1
        '''
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)

    return torch.mean(XX + YY - 2. * XY)