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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


from ML_for_WDN.data_utils import clean_dataframes, load_data
from ML_for_WDN.models import SupervisedReconstructionLeakDetector


torch.set_default_dtype(torch.float32)

LATENT_DIM = 4

ENCODER_ARGS = {
    'hidden_dims': [12, 10, 8],
    'latent_dim': LATENT_DIM,
}

DECODER_ARGS = {
    'latent_dim': LATENT_DIM,
    'hidden_dims': [8, 10, 12],
    'num_pars': 4
}

DATA_FILES = [
    'data/data_no_leak.xlsx',
    'data/data_leak_1.xlsx',
    'data/data_leak_2.xlsx',
    'data/data_leak_3.xlsx',
]

def main():

    train_fraction = 0.8
    test_fraction = 0.2

    columns_to_use = [
        'FM01_flow', 'FM02_head', 'FM03_flow', 'FM05_flow', 'FM06_flow', 'FM08_flow', 'FM09_flow', 'FM11_flow', 'FM13_flow',
        'FM01_head', 'FM02_flow', 'FM03_head', 'FM05_head', 'FM06_head', 'FM08_head', 'FM09_head', 'FM11_head', 'FM13_head',
    ]

    # Train data
    dataframes = []
    for data_file in DATA_FILES:
        df = load_data(data_file)
        dataframes.append(df)
    
    dataframes = clean_dataframes(
        dataframes,
        columns_to_use=columns_to_use,
    )

    num_train_pr_dataframe = []
    num_test_pr_dataframe = []
    train_dataframes = []
    test_dataframes = []
    for df in dataframes:
        num_train_pr_dataframe.append(int(train_fraction*df.shape[0]))
        train_dataframes.append(df.iloc[:int(train_fraction*df.shape[0]), :])

        num_test_pr_dataframe.append(int(test_fraction*df.shape[0]))
        test_dataframes.append(df.iloc[-int(test_fraction*df.shape[0]):, :])
    
    X_train = pd.concat(train_dataframes, ignore_index=True).values
    X_test = pd.concat(test_dataframes, ignore_index=True).values

    # Create targets
    y_train = []
    y_test = []
    for i in range(len(num_train_pr_dataframe)):
        y_train += [i]*num_train_pr_dataframe[i]
        y_test += [i]*num_test_pr_dataframe[i]
    
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    NN_args = {
        'encoder_args': ENCODER_ARGS,
        'decoder_args': DECODER_ARGS,
    }
    NN_train_args = {
        'epochs': 500,
        'batch_size': 2048,
        'lr': 5e-3,
        'weight_decay': 1e-10,
        'loss_fn': nn.MSELoss(),
    }
    

    #model = SupervisedLeakDetector(
    #    **NN_args,
    #    NN_train_args=NN_train_args,
    #    device='cpu',
    #)

    model = SVC(
        C=1e-2,
        kernel='rbf',
        gamma=1e-2,
    )

    pipeline = Pipeline([
        ('scaler',  MinMaxScaler()),
        ('model', model),
    ])

    pipeline.fit(
        X=X_train,
        y=y_train,
    )

    preds = pipeline.predict(
        X=X_test,
    )
    cm = confusion_matrix(y_test, preds)
    print(f'Accuracy: {accuracy_score(y_test, preds):0.3f}')
        
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['No leak', 'Leak 1', 'Leak 2', 'Leak 3'],
    )
    disp.plot()
    plt.show()





if __name__ == '__main__':
    main()