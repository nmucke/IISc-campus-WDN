import pdb
import numpy as np
import pandas as pd
import torch.nn as nn

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from ML_for_WDN.data_utils import clean_dataframes, load_data

from ML_for_WDN.models import (
    SupervisedReconstructionLeakDetector,
    SupervisedLatentLeakDetector,
    SupervisedLinearRegressionLeakDetector,
    SupervisedPolynomialRegressionLeakDetector
)

def main():


    '''
    leak_train_data_path = 'data/leak_train.xlsx'
    no_leak_train_data_path = 'data/data_base_demand_train_1.xlsx'

    leak_test_data_path = 'data/leak_test.xlsx'
    no_leak_test_data_path = 'data/data_base_demand_test.xlsx'

    df_leak_train = pd.concat(pd.read_excel(leak_train_data_path, sheet_name=None), ignore_index=True)
    df_no_leak_train = pd.concat(pd.read_excel(no_leak_train_data_path, sheet_name=None), ignore_index=True)

    df_leak_test = pd.concat(pd.read_excel(leak_test_data_path, sheet_name=None), ignore_index=True)
    df_no_leak_test = pd.concat(pd.read_excel(no_leak_test_data_path, sheet_name=None), ignore_index=True)
    


    X_leak_train = df_leak_train.iloc[:, 23:]
    X_no_leak_train = df_no_leak_train
    X_train = pd.concat([X_leak_train, X_no_leak_train], axis=0)

    X_leak_test = df_leak_test.iloc[:, 23:]
    X_no_leak_test = df_no_leak_test
    X_test = pd.concat([X_leak_test, X_no_leak_test], axis=0)

    y_leak_train = df_leak_train.iloc()[:, 0]
    y_no_leak_train = np.zeros((df_no_leak_train.shape[0],))
    y_train = np.concatenate((y_leak_train, y_no_leak_train), axis=0)
    y_train = y_train.astype(int)

    y_leak_test = df_leak_test.iloc()[:, 0]
    y_no_leak_test = np.zeros((df_no_leak_test.shape[0],))
    y_test = np.concatenate((y_leak_test, y_no_leak_test), axis=0)
    y_test = y_test.astype(int)

    X_train = X_train.values
    X_test = X_test.values

    # Swap columns 0-10 with 10-20
    X_train[:, 0:10], X_train[:, 10:20] = X_train[:, 10:20], X_train[:, 0:10].copy()
    X_test[:, 0:10], X_test[:, 10:20] = X_test[:, 10:20], X_test[:, 0:10].copy()


    '''

    df_train = pd.read_csv('data/training_data.csv')
    df_test = pd.read_csv('data/testing_data.csv')

    X_train = df_train.iloc[:, :-1].values
    y_train = df_train.iloc[:, -1].values

    X_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values

    # Swap columns 0-10 with 10-20
    X_train[:, 0:10], X_train[:, 10:20] = X_train[:, 10:20], X_train[:, 0:10].copy()
    X_test[:, 0:10], X_test[:, 10:20] = X_test[:, 10:20], X_test[:, 0:10].copy()


    '''
    DATA_FILES = [
        'data/data_no_leak.xlsx',
        'data/data_leak_1.xlsx',
        'data/data_leak_2.xlsx',
        'data/data_leak_3.xlsx',
    ]

    train_fraction = 0.995
    test_fraction = 0.005

    columns_to_use = [
        'FM01_flow', 'FM02_flow', 'FM03_flow', 'FM05_flow', 'FM06_flow', 'FM08_flow', 'FM09_flow', 'FM11_flow', 'FM13_flow',
        'FM01_head', 'FM02_head', 'FM03_head', 'FM05_head', 'FM06_head', 'FM08_head', 'FM09_head', 'FM11_head', 'FM13_head',
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
    '''

    print(f'X_train: {X_train.shape}')
    print(f'X_test: {X_test.shape}')
    print(f'y_train: {y_train.shape}')
    print(f'y_test: {y_test.shape}')

    logistic_regression_args = {
        'penalty': 'l2',
        'C': 1e-2,
        'solver': 'lbfgs',
        'max_iter': 1000,
    }
        
    svc_args = {
        'C': 1e-2,
        'kernel': 'rbf',
        'gamma': 1e-2,
    }

    rf_args = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
    }

    reconstruction_WAE_args = {
        'encoder_args': {
            'hidden_dims': [12, 10, 8],
            'latent_dim': 4,
        },
        'decoder_args': {
            'latent_dim': 4,
            'hidden_dims': [8, 10, 12],
            'num_pars': 4
        },
        'NN_train_args': {
            'epochs': 5,
            'batch_size': 2048,
            'lr': 5e-3,
            'weight_decay': 1e-10,
            'loss_fn': nn.MSELoss(),
        },
        'device': 'cpu'
    }

    latent_WAE_args = {
        'encoder_args': {
            'hidden_dims': [12, 10, 8],
            'latent_dim': 4,
        },
        'decoder_args': {
            'latent_dim': 4,
            'hidden_dims': [8, 10, 12],
        },
        'NN_train_args': {
            'epochs': 5,
            'batch_size': 2048,
            'lr': 5e-3,
            'weight_decay': 1e-10,
            'loss_fn': nn.MSELoss(),
        },
        'device': 'cpu'
    }

    linear_regression_args = {}

    model_list = [
        #LogisticRegression(**logistic_regression_args),
        #SVC(**svc_args),
        #RandomForestClassifier(**rf_args),
        #SupervisedReconstructionLeakDetector(**reconstruction_WAE_args),
        #SupervisedLatentLeakDetector(**latent_WAE_args),
        #SupervisedLinearRegressionLeakDetector(**linear_regression_args),
        SupervisedPolynomialRegressionLeakDetector(**linear_regression_args),
    ]

    for model in model_list:
        pipeline = Pipeline([
            ('scaler',  StandardScaler()),
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
        print(f'{model}: Accuracy: {accuracy_score(y_test, preds):0.3f}')

        #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Leak 1', 'Leak 2', 'Leak 3'])
        #disp.plot()
        #plt.show()
        

if __name__ == '__main__':
    main()