import pdb
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
import torch.nn as nn
import ray

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

@ray.remote
def ray_fit_wrapper(
    model,
    X: pd.DataFrame,
    y: pd.DataFrame,
):
    model.fit(
        X=X,
        y=y,
    )

    return model

@ray.remote
def ray_predict_wrapper(
    model_list,
    model_names,
    X_test,
    y_test,
    num_test_samples,
    num_samples_pr_test,
    leak,      
):

    X_test_leak = X_test[y_test==leak]

    preds = {key: [] for key in model_names}

    for i in range(num_test_samples):

        random_ids = np.random.choice(
            X_test_leak.shape[0], 
            size=num_samples_pr_test, 
            replace=False
        )

        X_test_batch = X_test_leak.iloc[random_ids, :]

        for (model, model_name) in zip(model_list, model_names):
            
            preds_leak = model.predict(
                X=X_test_batch,
            )

            preds[model_name].append(preds_leak)
    
    for model_name in model_names:
        preds[model_name] = np.concatenate(preds[model_name], axis=0)

    return preds



def main():

    ray.init(num_cpus=9)
   
    df_train = pd.read_excel('data/leak_train.xlsx')
    df_test = pd.read_excel('data/leak_test.xlsx')

    X_train = df_train.loc[:, df_train.columns != 'leak_link']
    y_train = df_train['leak_link']
        
    X_test = df_test.loc[:, df_test.columns != 'leak_link']
    y_test = df_test['leak_link']

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
            'batch_size': 5096,
            'lr': 1e-2,
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
            'batch_size': 5096,
            'lr': 1e-2,
            'weight_decay': 1e-8,
            'loss_fn': nn.MSELoss(),
        },
        'device': 'cpu'
    }

    model_list = [
        ClassifierMajorityVote(MLPClassifier(**MLPClassifier_args)),
        PrernaModel(**prerna_model_args, classifier='logistic_regression'),
        PrernaModel(**prerna_model_args, classifier='random_forest'),
        ClassifierMajorityVote(LogisticRegression(**logistic_regression_args)),
        ClassifierMajorityVote(KNeighborsClassifier(**knn_args)),
        ClassifierMajorityVote(RandomForestClassifier(**rf_args)),
        SupervisedReconstructionLeakDetector(**reconstruction_WAE_args),
        SupervisedLatentLeakDetector(**latent_WAE_args, classifier='random_forest'),
        SupervisedLatentLeakDetector(**latent_WAE_args, classifier='logistic_regression'),
    ]

    model_names = [
        'MLP Classifier',
        'Regression + Logistic Regression',
        'Regression + Random Forest Classifier',
        'Logistic Regression',
        'KNN Classifier',
        'Random Forest Classifier',
        'WAE Reconstruction Model',
        'WAE + Random Forest Classifier',
        'WAE + Logistic Regression',
    ]

    model_save_names = [
        'MLP_Classifier',
        'Regression_Logistic_Regression',
        'Regression_Random_Forest_Classifier',
        'Logistic_Regression',
        'KNN_Classifier',
        'Random_Forest_Classifier',
        'WAE_Reconstruction_Model',
        'WAE_Latent_Model_Random_Forest_Classifier',
        'WAE_Latent_Model_Logistic_Regression',
    ]
    
    '''
    random_ids = np.random.choice(
        X_train.shape[0], 
        size=50, 
        replace=False
    )
    X_train = X_train.iloc[random_ids, :]
    y_train = y_train.iloc[random_ids]
    '''
    
    

    num_test_samples = 3000
    num_samples_pr_test = 300
    
    
    fitted_models = []
    for model in model_list:
        
        fitted_models.append(ray_fit_wrapper.remote(
            model=model,
            X=X_train,
            y=y_train,
        ))
    
    model_list = ray.get(fitted_models)    

    #preds = {key: [] for key in model_names}
    preds_leak_list = []
    for leak in [0, 1, 2, 3]:
        preds_leak_list.append(ray_predict_wrapper.remote(
            model_list=model_list,
            model_names=model_names,
            X_test=X_test,
            y_test=y_test,
            num_test_samples=num_test_samples,
            num_samples_pr_test=num_samples_pr_test,
            leak=leak,      
        ))

    preds_leak_list = ray.get(preds_leak_list)
   
    '''
    X_test_leak = X_test[y_test==leak]

    for i in range(num_test_samples):

        random_ids = np.random.choice(
            X_test_leak.shape[0], 
            size=num_samples_pr_test, 
            replace=False
        )

        X_test_batch = X_test_leak.iloc[random_ids, :]

        for (model, model_name, model_save_name) in zip(model_list, model_names, model_save_names):
            
            preds_leak = model.predict(
                X=X_test_batch,
            )

            preds[model_name].append(preds_leak)
    '''


    preds = {key: [] for key in model_names}
    for model_name in model_names:
        for preds_leak in preds_leak_list:
            preds[model_name].append(preds_leak[model_name])
        preds[model_name] = np.concatenate(preds[model_name], axis=0)

    true_vals = []
    for leak in [0, 1, 2, 3]:
        true_vals_leak = np.ones((num_test_samples,))*leak
        true_vals_leak = true_vals_leak.astype(int)
        true_vals.append(true_vals_leak)


    true_vals = np.concatenate(true_vals, axis=0)

    for (model_name, model_save_name) in zip(model_names, model_save_names):
        print(f'{model_name}: Accuracy: {accuracy_score(true_vals, preds[model_name]):0.3f}')

        cm = confusion_matrix(true_vals, preds[model_name])

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No leak', 'Leak 1', 'Leak 2', 'Leak 3'])
        disp.plot()
        plt.title(f'{model_name}, Accuracy: {accuracy_score(true_vals, preds[model_name]):0.3f}')
                    
        plt.savefig(f'figures/confusion_matrix_{model_save_name}.pdf')

        #plt.show()

        

if __name__ == '__main__':
    main()