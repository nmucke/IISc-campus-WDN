import pdb
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
import torch.nn as nn

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from ML_for_WDN.data_utils import clean_dataframes, load_data

from ML_for_WDN.prerna_model import (
    PrernaModel
)


from ML_for_WDN.models import (
    SupervisedReconstructionLeakDetector,
    SupervisedLatentLeakDetector,
    SupervisedLinearRegressionLeakDetector,
    SupervisedPolynomialRegressionLeakDetector,
    ClassifierMajorityVote
)

def main():


    #df_train = pd.read_excel('data/leak_train.xlsx')
    #df_test = pd.read_excel('data/leak_test.xlsx')

    df_train = pd.read_csv('data/leak_train_IISc - Filter.csv')
    df_test = pd.read_csv('data/leak_test_IISc - Filter.csv')

    X_train = df_train.loc[:, df_train.columns != 'leak_link']
    y_train = df_train['leak_link']
        
    X_test = df_test.loc[:, df_test.columns != 'leak_link']
    y_test = df_test['leak_link']

    #X_train[:, 0:num_sensors], X_train[:, num_sensors:] = X_train[:, num_sensors:].copy(), X_train[:, 0:num_sensors].copy()
    #X_test[:, 0:num_sensors], X_test[:, num_sensors:] = X_test[:, num_sensors:].copy(), X_test[:, 0:num_sensors].copy()


    print(f'X_train: {X_train.shape}')
    print(f'X_test: {X_test.shape}')
    print(f'y_train: {y_train.shape}')
    print(f'y_test: {y_test.shape}')

    prerna_model_args = {
        'model_args': {},
        'verbose': False,
    }

    logistic_regression_args = {
        'penalty': 'l2',
        'C': 1e-2,
        'solver': 'lbfgs',
        'max_iter': 1000,
        'multi_class': 'multinomial',
        'class_weight': 'balanced',
        'max_iter': 5000
    }
    
    knn_args = {
        'n_neighbors': 5,
        'weights': 'distance',
    }

    rf_args = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
    }

    MLPClassifier_args = {
        'hidden_layer_sizes': (100, 100, 100),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 1e-4,
        'batch_size': 2048,
        'learning_rate': 'adaptive',
        'learning_rate_init': 1e-3,
        'max_iter': 500,
        'shuffle': True,
        'random_state': 1,
        'tol': 1e-4,
        'verbose': False,
        'early_stopping': True,
        'validation_fraction': 0.1,
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
            'epochs': 500,
            'batch_size': 2,
            'lr': 5e-3,
            'weight_decay': 1e-8,
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
            'epochs': 500,
            'batch_size': 1024,
            'lr': 5e-3,
            'weight_decay': 1e-8,
            'loss_fn': nn.MSELoss(),
        },
        'device': 'cpu'
    }

    model_list = [
        #ClassifierMajorityVote(MLPClassifier(**MLPClassifier_args)),
        PrernaModel(**prerna_model_args, classifier='logistic_regression'),
        PrernaModel(**prerna_model_args, classifier='random_forest'),
        ClassifierMajorityVote(LogisticRegression(**logistic_regression_args)),
        ClassifierMajorityVote(KNeighborsClassifier(**knn_args)),
        ClassifierMajorityVote(RandomForestClassifier(**rf_args)),
        #SupervisedReconstructionLeakDetector(**reconstruction_WAE_args),
        #SupervisedLatentLeakDetector(**latent_WAE_args, classifier='random_forest'),
        #SupervisedLatentLeakDetector(**latent_WAE_args, classifier='logistic_regression'),
    ]

    model_names = [
        #'MLP Classifier',
        'Regression + Logistic Regression',
        'Regression + Random Forest Classifier',
        'Logistic Regression',
        'KNN Classifier',
        'Random Forest Classifier',
        #'WAE Reconstruction Model',
        #'WAE + Random Forest Classifier',
        #'WAE + Logistic Regression',
    ]

    model_save_names = [
        #'MLP_Classifier',
        'Regression_Logistic_Regression',
        'Regression_Random_Forest_Classifier',
        'Logistic_Regression',
        'KNN_Classifier',
        'Random_Forest_Classifier',
        #'WAE_Reconstruction_Model',
        #'WAE_Latent_Model_Random_Forest_Classifier',
        #'WAE_Latent_Model_Logistic_Regression',
    ]
    
    '''
    random_ids = np.random.choice(
        X_train.shape[0], 
        size=100, 
        replace=False
    )
    X_train = X_train.iloc[random_ids, :]
    y_train = y_train.iloc[random_ids]
    '''
    
    

    num_test_samples = 3000
    num_samples_pr_test = 300
    
    # Train models
    for model in model_list:

        model.fit(
            X=X_train,
            y=y_train,
        )

    # Test models
    preds = {key: [] for key in model_names}
    true_vals = []

    # Loop over leak location
    for leak in y_test.unique():

        # Get test samples for leak location
        X_test_leak = X_test[y_test==leak]
        y_test_leak = y_test[y_test==leak]

        # Create true labels for leak location
        true_vals_leak = np.ones((num_test_samples,))*leak
        true_vals_leak = true_vals_leak.astype(int)
        true_vals.append(true_vals_leak)

        # Loop over number of test samples
        for i in range(num_test_samples):

            # Get random samples from test set
            random_ids = np.random.choice(
                X_test_leak.shape[0], 
                size=num_samples_pr_test, 
                replace=False
            )

            # Get sensor data for random samples
            X_test_batch = X_test_leak.iloc[random_ids, :]

            # Loop over models
            for (model, model_name, model_save_name) in zip(model_list, model_names, model_save_names):
                
                # Get predictions
                preds_leak = model.predict(
                    X=X_test_batch,
                )

                preds[model_name].append(preds_leak)
    

    true_vals = np.concatenate(true_vals, axis=0)

    # Print accuracy and plot confusion matrix
    for (model_name, model_save_name) in zip(model_names, model_save_names):
        preds[model_name] = np.concatenate(preds[model_name], axis=0)
        
        print(f'{model_name}: Accuracy: {accuracy_score(true_vals, preds[model_name]):0.3f}')

        cm = confusion_matrix(true_vals, preds[model_name])

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No leak', 'Leak 1', 'Leak 2', 'Leak 3'])
        disp.plot()
        plt.title(f'{model_name}, Accuracy: {accuracy_score(true_vals, preds[model_name]):0.3f}')
                    
        plt.savefig(f'figures/confusion_matrix_{model_save_name}.pdf')

        plt.show()

        

        

if __name__ == '__main__':
    main()