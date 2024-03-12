import pdb
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
import ray
import torch.nn as nn

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM 
from sklearn.neighbors import LocalOutlierFactor

from ML_for_WDN.data_utils import clean_dataframes, load_data

from ML_for_WDN.prerna_model import (
    PrernaModel
)


from ML_for_WDN.models import (
    ClassifierMajorityVote
)


@ray.remote
def ray_fit_wrapper(
    model: nn.Module,
    X: pd.DataFrame,
):
    model.fit(
        X=X,
    )

    return model

def main():

    ray.init(num_cpus=3)

    X_train = pd.read_excel('data/leak_train.xlsx')
    df_test = pd.read_excel('data/leak_test.xlsx')

    #df_train = pd.concat(pd.read_excel('data/leak_train.xlsx', sheet_name=None), ignore_index=True)
    #df_test = pd.concat(pd.read_excel('data/leak_test.xlsx', sheet_name=None), ignore_index=True)

    #df_train = pd.read_csv('data/leak_train_11.csv')
    #df_test = pd.read_csv('data/leak_test_11.csv')

    X_train = X_train[X_train['leak_link'] == 0]
    X_train = X_train.loc[:, X_train.columns != 'leak_link']
        
    X_test = df_test.loc[:, df_test.columns != 'leak_link']
    y_test = df_test['leak_link']
    y_test = y_test.astype(int)

    print(f'X_train: {X_train.shape}')
    print(f'X_test: {X_test.shape}')
    print(f'y_test: {y_test.shape}')

    isolation_forest_args = {
        'n_estimators': 100,
        'contamination': 0.01,
        'random_state': 42,
        'n_jobs': -1,
    }

    one_class_svm_args = {
        'kernel': 'rbf',
        'nu': 0.01,
        'gamma': 0.1,
    }

    local_outlier_factor_args = {
        'n_neighbors': 20,
        'contamination': 0.01,
        'n_jobs': -1,
        'novelty': True,
    }

    model_list = [
        ClassifierMajorityVote(IsolationForest(**isolation_forest_args)),
        ClassifierMajorityVote(OneClassSVM(**one_class_svm_args)),
        ClassifierMajorityVote(LocalOutlierFactor(**local_outlier_factor_args)),
    ]

    model_names = [
        'IsolationForest',
        'OneClassSVM',
        'LocalOutlierFactor',
    ]

    model_save_names = [
        'isolation_forest',
        'one_class_svm',
        'local_outlier_factor',
    ]

    '''
    random_ids = np.random.choice(
        X_train.shape[0], 
        size=250, 
        replace=False
    )
    X_train = X_train.iloc[random_ids, :] 
    '''
    

    num_test_samples = 3000
    num_samples_pr_test = 300
    
    fitted_models = []
    for model in model_list:
        
        fitted_models.append(ray_fit_wrapper.remote(
            model=model,
            X=X_train,
        ))
    
    model_list = ray.get(fitted_models)                   

    preds = {key: [] for key in model_names}
    true_vals = []
    for leak in y_test.unique():

        X_test_leak = X_test[y_test==leak]
        
        if leak == 0:
            true_vals_leak = np.ones((num_test_samples,))
        else:
            true_vals_leak = -np.ones((num_test_samples,))
        true_vals_leak = true_vals_leak.astype(int)
        true_vals.append(true_vals_leak)

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
    

    true_vals = np.concatenate(true_vals, axis=0)

    for (model_name, model_save_name) in zip(model_names, model_save_names):
        preds[model_name] = np.array(preds[model_name])

        cm = confusion_matrix(true_vals, preds[model_name])
        print(f'{model_name}: Accuracy: {accuracy_score(true_vals, preds[model_name]):0.3f}')

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Leak', 'No leak'])
        disp.plot()
        plt.title(f'{model_name}, Accuracy: {accuracy_score(true_vals, preds[model_name]):0.3f}')
                    
        #ax.set_title(f'{model_name}, Accuracy: {accuracy_score(true_vals, preds[model_name]):0.3f}')
        # labels, title and ticks
        plt.savefig(f'figures/confusion_matrix_{model_save_name}.pdf')

        plt.show()

        

if __name__ == '__main__':
    main()