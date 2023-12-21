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
from sklearn.pipeline import Pipeline


from ML_for_WDN.data_utils import clean_dataframes, load_data
from ML_for_WDN.models import UnsupervisedLeakDetector


torch.set_default_dtype(torch.float32)

LATENT_DIM = 8
SUPERVISED = False

ENCODER_ARGS = {
    'hidden_dims': [16, 12, 8],
    'latent_dim': LATENT_DIM,
}

DECODER_ARGS = {
    'latent_dim': LATENT_DIM,
    'hidden_dims': [8, 12, 16],
}


DATA_FILES_TRAIN = [
    'data/data_no_leak.xlsx',
]

DATA_FILES_TEST = [
    'data/data_leak_1.xlsx',
    'data/data_leak_2.xlsx',
    'data/data_leak_3.xlsx',
]


def main():

    columns_to_use = [
        'FM01_flow', 'FM02_head', 'FM03_flow', 'FM05_flow', 'FM06_flow', 'FM08_flow', 'FM09_flow', 'FM11_flow', 'FM13_flow',
        'FM01_head', 'FM02_flow', 'FM03_head', 'FM05_head', 'FM06_head', 'FM08_head', 'FM09_head', 'FM11_head', 'FM13_head',
    ]

    # Train data
    dataframes = []
    for data_file in DATA_FILES_TRAIN:
        df = load_data(data_file)
        dataframes.append(df)
    
    dataframes = clean_dataframes(
        dataframes,
        columns_to_use=columns_to_use,
    )
    train_data = dataframes[0]

    test_data = train_data.iloc[-5000:, :]
    train_data = train_data.iloc[:-5000, :]

    train_data = train_data.values

    # Test data
    dataframes = []
    for data_file in DATA_FILES_TEST:
        df = load_data(data_file)
        dataframes.append(df)

    dataframes = clean_dataframes(
        dataframes,
        columns_to_use=columns_to_use,
    )
    dataframes = pd.concat(dataframes, ignore_index=True)

    test_data = pd.concat([test_data, dataframes], ignore_index=True)
    test_data = test_data.values

    targets = np.zeros((test_data.shape[0]))
    targets[0:5000] = 1
    targets[5000:] = -1


    NN_args = {
        'encoder_args': ENCODER_ARGS,
        'decoder_args': DECODER_ARGS,
    }
    NN_train_args = {
        'epochs': 1000,
        'batch_size': 512,
        'lr': 5e-3,
        'weight_decay': 1e-4,
        'loss_fn': nn.MSELoss(),
    }
    anomaly_detection_args = {
    }
    model = UnsupervisedLeakDetector(
        **NN_args,
        NN_train_args=NN_train_args,
        anomaly_detection_args=anomaly_detection_args,
        device='cpu',
    )

    pipeline = Pipeline([
        ('scaler',  StandardScaler()),
        ('model', model),
    ])

    pipeline.fit(
        X=train_data,
    )

    preds = pipeline.predict(
        X=test_data,
    )
    cm = confusion_matrix(targets, preds)
    print(f'Accuracy: {accuracy_score(targets, preds):0.3f}')
    print(f'Recall: {cm[1,1]/(cm[1,1]+cm[1,0])}')
    print(f'Precision: {cm[1,1]/(cm[1,1]+cm[0,1])}')
        
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['leak', 'No Leak'],
    )
    disp.plot()
    plt.show()





if __name__ == '__main__':
    main()