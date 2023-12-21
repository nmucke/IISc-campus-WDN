import pdb
import numpy as np
import pandas as pd
import networkx as nx
import wntr
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

from ML_for_WDN.campus_dataset import CampusDataset
from ML_for_WDN.data_utils import clean_dataframes, load_data
from ML_for_WDN.WAE import WassersteinAutoencoder

from tqdm import tqdm

from ML_for_WDN.train_utils import MMD

torch.set_default_dtype(torch.float32)

LATENT_DIM = 2
SUPERVISED = False

ENCODER_ARGS = {
    'input_dim': 18,
    'hidden_dims': [16, 12, 8],
    'latent_dim': LATENT_DIM,
}

DECODER_ARGS = {
    'latent_dim': LATENT_DIM,
    'hidden_dims': [8, 12, 16],
    'output_dim': 18,
    'num_pars': 4 if SUPERVISED else None,
}


DATA_FILES_TRAIN = [
    'data/data_no_leak.xlsx',
    'data/data_leak_1.xlsx',
    'data/data_leak_2.xlsx',
    'data/data_leak_3.xlsx',
]

DATA_FILES_TEST = [
    'data/data_no_leak.xlsx',
    'data/data_leak_1.xlsx',
    'data/data_leak_2.xlsx',
    'data/data_leak_3.xlsx',
]

def train_WAE(
    dataloader: torch.utils.data.DataLoader,
    model: WassersteinAutoencoder,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int,
    device: torch.device,
) -> None:
    
    model.train()

    for epoch in range(epochs):
        pbar = tqdm(
                enumerate(dataloader),
                total=int(len(dataloader.dataset)/dataloader.batch_size),
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
            )
        
        total_loss = 0
        for batch_idx, data in pbar:
            
            if model.decoder.num_pars is not None:
                data, pars = data[0].to(device), data[1].to(device)
            else:
                data = data.to(device)
                pars = None
            
            optimizer.zero_grad()
            
            latent = model.encoder(data)

            z = torch.randn_like(latent)
            latent_loss = MMD(latent, z, kernel='rbf', device=device)

            output = model.decoder(latent, pars)
            loss = loss_fn(output, data) + 1e-12*latent_loss
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 50 == 0:
                pbar.set_description(f'Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | Latent: {latent_loss.item():.4f}')
        
        total_loss /= len(dataloader)

    model.eval()

    return None

def get_train_WAE_error_and_latents(
    dataloader: torch.utils.data.DataLoader,
    model: WassersteinAutoencoder,
) -> None:
    
    train_loss_list = []
    latent_list = []
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            
            if model.decoder.num_pars is not None:
                data, pars = data[0], data[1]
            else:
                pars = None
            
            latent = model.encoder(data)
            output = model.decoder(latent, pars)

            train_loss = torch.sqrt((output - data).pow(2).sum(dim=-1)).detach().numpy()

            train_loss_list.append(train_loss)
            latent_list.append(latent.detach().numpy())

    return np.concatenate(train_loss_list, axis=0), np.concatenate(latent_list, axis=0)

    
def test_WAE_anomaly(
    dataset: torch.utils.data.Dataset,
    model: WassersteinAutoencoder,
    prior_latents: dict,
    prior_errors: dict,
) -> None:
    
    model.eval()
    
    test_error = {}
    latent_dict = {}
    true_targets = []
    pred_targets = []
    with torch.no_grad():
        for key in ['no_leak', 'leak_1', 'leak_2', 'leak_3']:

            data_ids = dataset.leak_dataframe_ids[key]
            data = dataset.dataframes.iloc[data_ids]

            data = dataset.preprocessor.transform(data.values)

            data = torch.tensor(data, dtype=torch.float32)
            
            latent = model.encoder(data)
            output = model.decoder(latent)

            test_loss = torch.sqrt((output - data).pow(2).sum(dim=-1)).detach().numpy()

            test_error[key] = test_loss
            latent_dict[key] = latent.detach().numpy()

            if key == 'no_leak':
                true_targets.append(np.zeros_like(test_loss))
            else:
                true_targets.append(np.ones_like(test_loss))

            pred = np.zeros_like(test_loss)
            pred[test_loss > prior_errors['mean'] + 3*prior_errors['std']] = 1

            pred_targets.append(pred)

    true_targets = np.concatenate(true_targets, axis=0)
    pred_targets = np.concatenate(pred_targets, axis=0)

    accuracy = accuracy_score(pred_targets, true_targets)
    cm = confusion_matrix(true_targets, pred_targets)
            
    return test_error, latent_dict, accuracy, cm

def test_WAE_location(
    dataloader: torch.utils.data.DataLoader,
    model: WassersteinAutoencoder,
    loss_fn: nn.Module,                     
    device: torch.device,
) -> None:
    
    model.eval()
    
    test_loss = 0
    with torch.no_grad():
        pbar = tqdm(
                enumerate(dataloader),
                total=int(len(dataloader.dataset)/dataloader.batch_size),
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
            )
        
        one_hot_encoder = OneHotEncoder().fit(np.arange(0, 4).reshape(-1,1))
        
        true_pars = []
        leak_location_pred = []
        for batch_idx, data in pbar:

            if model.decoder.num_pars is not None:
                data, true_pars_batch = data[0], data[1].numpy()
            else:
                pars = None
            
            leak_location_pred_batch = model.get_leak_location(data).detach().numpy().reshape(-1,1)
            
            #true_pars_batch = one_hot_encoder.inverse_transform(true_pars_batch)

            true_pars.append(true_pars_batch)
            leak_location_pred.append(leak_location_pred_batch)

    true_pars = np.concatenate(true_pars)
    leak_location_pred = np.concatenate(leak_location_pred)

    test_loss = accuracy_score(leak_location_pred, true_pars)

    cm = confusion_matrix(true_pars, leak_location_pred)

    return test_loss, cm


def main():

    dataframes = []

    for data_file in DATA_FILES_TRAIN:
        df = load_data(data_file, train=True, fraction=1.0)
        dataframes.append(df)
    
    dataframes = clean_dataframes(dataframes)

    train_dataframes = [df.iloc[:int(0.8*len(df))] for df in dataframes]
    test_dataframes = [df.iloc[int(0.8*len(df)):] for df in dataframes]

    dataset = CampusDataset(
        dataframes=train_dataframes,
        with_leak=True if SUPERVISED else False,
    )

    preprocessor = StandardScaler().fit(dataset.dataframes.values)

    dataset.preprocessor = preprocessor

    # Create a dataloader
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=512,
        shuffle=True,
        num_workers=4,
    )

    # Create the model
    model = WassersteinAutoencoder(
        encoder_args=ENCODER_ARGS,
        decoder_args=DECODER_ARGS,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)

    loss_fn = nn.MSELoss()
    epochs = 400

    device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_WAE(
        dataloader=train_dataloader,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=epochs,
        device=device,
    )

    model.to('cpu')

    test_dataset = CampusDataset(
        dataframes=test_dataframes,
        with_leak=True,
        preprocessor=preprocessor,
    )

    # Create a dataloader
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=0,
    )

    if SUPERVISED:

        location_error, cm = test_WAE_location(
            dataloader=test_dataloader,
            model=model,
            loss_fn=loss_fn,
            device=device,
        )
        print(f'Location accuracy: {location_error}') 
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['No leak', 'Leak 1', 'Leak 2', 'Leak 3'],
        )
        disp.plot()
        plt.show()

    else:
        train_error, train_latents = get_train_WAE_error_and_latents(
            dataloader=train_dataloader,
            model=model,
        )

        prior_errors = {
            'mean': np.mean(train_error),
            'std': np.std(train_error),
        }
        prior_latents = {
            'mean': np.mean(train_latents, axis=0),
            'std': np.std(train_latents, axis=0),
        }

        test_error, latents, accuracy, cm = test_WAE_anomaly(
            dataset=test_dataset,
            model=model,
            prior_errors=prior_errors,
            prior_latents=prior_latents,
        )


        print(f'Accuracy Score: {accuracy}') 

        
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['No leak', 'Leak'],
        )
        disp.plot()
        plt.show()

        plt.figure(figsize=(10, 10))
        for i, key in enumerate(['no_leak', 'leak_1', 'leak_2', 'leak_3']):
            plt.subplot(2, 2, i+1)
            plt.hist(train_error, bins=100, alpha=0.5, label=key, density=True)
            plt.hist(test_error[key], bins=100, alpha=0.5, label=key, density=True)
            plt.axvline(x=prior_errors['mean'], color='black', linestyle='--')
            plt.axvline(x=prior_errors['mean'] + 3*prior_errors['std'], color='black', linestyle='--')
            plt.axvline(x=prior_errors['mean'] - 3*prior_errors['std'], color='black', linestyle='--')
            plt.legend()
        plt.show()

        plt.figure(figsize=(10, 10))
        for i, key in enumerate(['no_leak', 'leak_1', 'leak_2', 'leak_3']):
            plt.subplot(2, 2, i+1)
            plt.hist(train_latents.reshape(-1), bins=100, alpha=0.5, label=key, density=True)
            plt.hist(latents[key].reshape(-1), bins=100, alpha=0.5, label=key, density=True)
            plt.axvline(x=prior_latents['mean'][0], color='black', linestyle='--')
            plt.axvline(x=prior_latents['mean'][0] + 3*prior_latents['std'][0], color='black', linestyle='--')
            plt.axvline(x=prior_latents['mean'][0] - 3*prior_latents['std'][0], color='black', linestyle='--')
            plt.legend()
        plt.show()



if __name__ == '__main__':
    main()