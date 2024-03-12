from itertools import combinations
import pdb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
import pandas as pd
from sklearn import pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import ks_2samp
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import norm



class PrernaModel(BaseEstimator, ClassifierMixin):
    
        def __init__(
            self,
            model_args: dict,
            classifier: str = 'logistic_regression',
            verbose: bool = False,
        ) -> None:
            
            self.model_args = model_args
            self.window_size = model_args.get('window_size')
            if self.window_size is None:
                self.window_size = 50

            self.classifier = classifier
            self.verbose = verbose
    
        def __str__(self) -> str:
            return 'PrernaModel'
        
        def _get_flowrates_for_sensor_pair(
            self, 
            X: pd.DataFrame,
            sensor_pair: tuple,    
        ) -> np.ndarray:
            
            sensor_1 = sensor_pair[0]
            sensor_2 = sensor_pair[1]
            
            flowrate_1 = X[self.link_cols[sensor_1]]
            flowrate_2 = X[self.link_cols[sensor_2]]

            
            return np.array([flowrate_1, flowrate_2]).T
    
        def _get_headloss_for_sensor_pair(
            self, 
            X: pd.DataFrame,
            sensor_pair: tuple,    
        ) -> np.ndarray:
            
            sensor_1 = sensor_pair[0]
            sensor_2 = sensor_pair[1]
            
            headloss = X[self.head_cols[sensor_1]] - X[self.head_cols[sensor_2]]
            
            return np.array([headloss]).T

        def fit(
            self, 
            X: pd.DataFrame,
            y: pd.DataFrame,
        ) -> None:
            
            col_names = X.columns
            self.head_cols = [col for col in col_names if 'Node_head' in col]
            self.link_cols = [col for col in col_names if 'Link_flow' in col]
            
            num_sensors = len(self.head_cols)
            
            ##### Training stage 1 #####
            # Training the regression model for each sensor pair

            # Get all combinations of sensors
            self.combs = list(combinations(range(0,num_sensors),2))

            # Train linear regression on non-leak data
            X_no_leak = X[y==0]
            
            self.lin_models = {}
            for comb in self.combs:
                x_train = self._get_flowrates_for_sensor_pair(X_no_leak, comb)
                y_train = self._get_headloss_for_sensor_pair(X_no_leak, comb)
                lin_model = pipeline.Pipeline([
                    ('poly', PolynomialFeatures(degree=2)),
                    ('linear', LinearRegression())
                ])
                lin_model.fit(x_train, y_train)
                
                lin_model_name = str(comb[0])+str(comb[1])
                self.lin_models[lin_model_name] = lin_model
            
            ##### Training stage 2 #####
            # Training the classifier model
            
            if self.classifier == 'logistic_regression':
                logistic_regression_args = {
                    'penalty': 'l2',
                    'C': 1e-2,
                    'solver': 'lbfgs',
                    'max_iter': 1000,
                }
                self.classifier_model = LogisticRegression(**logistic_regression_args)
            elif self.classifier == 'random_forest':
                random_forest_args = {
                    'n_estimators': 100,
                    'max_depth': 10,
                }
                self.classifier_model = RandomForestClassifier(**random_forest_args)

            # Get error distribution on non-leak data
            no_leak_error = {}
            self.no_leak_error_distributions = {}
            for comb in self.combs:
                x_train = self._get_flowrates_for_sensor_pair(X_no_leak, comb)
                y_train = self._get_headloss_for_sensor_pair(X_no_leak, comb)
                lin_model_name = str(comb[0])+str(comb[1])
                y_pred = self.lin_models[lin_model_name].predict(x_train)
                y_pred = y_pred.reshape(-1)
                y_train = y_train.reshape(-1)
                error = y_train - y_pred
                error = np.convolve(error, np.ones(self.window_size)/self.window_size, 'same')
                no_leak_error[lin_model_name] = error

                self.no_leak_error_distributions[lin_model_name] = norm(
                    np.mean(error),
                    np.std(error)
                )

            classification_training_df = pd.DataFrame(
                columns=self.combs + ['label'],
            )
            for leak in y.unique():
                leak = int(leak)
                leak_classification_training_df = pd.DataFrame(
                    columns=self.combs + ['label']
                )
                for comb in self.combs:
                    x_train = self._get_flowrates_for_sensor_pair(X[y==leak], comb)
                    y_train = self._get_headloss_for_sensor_pair(X[y==leak], comb)
                    
                    lin_model_name = str(comb[0])+str(comb[1])
                    lin_model = self.lin_models[lin_model_name]
                    y_pred = lin_model.predict(x_train)

                    y_pred = y_pred.reshape(-1)
                    y_train = y_train.reshape(-1)
                    error = y_train - y_pred

                    #error = np.convolve(error, np.ones(self.window_size)/self.window_size, 'same')

                    leak_classification_training_df[comb] = error

                leak_classification_training_df['label'] = leak

                classification_training_df = classification_training_df.append(
                    leak_classification_training_df,
                    ignore_index=True
                )
            classification_training_df['label'] = classification_training_df['label'].astype(int)
            
            self.classifier_model.fit(
                classification_training_df[self.combs],
                classification_training_df['label']
            )

            return self

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            
            num_sensors = len(self.head_cols)
            self.combs = list(combinations(range(0,num_sensors),2))
            
            classification_df = pd.DataFrame(
                columns=self.combs
            )
            for comb in self.combs:
                flowrates = self._get_flowrates_for_sensor_pair(X, comb)
                head_loss = self._get_headloss_for_sensor_pair(X, comb)

                lin_model_name = str(comb[0])+str(comb[1])
                head_loss_pred = self.lin_models[lin_model_name].predict(flowrates)
                head_loss_pred = head_loss_pred.reshape(-1)

                head_loss = head_loss.reshape(-1)

                error = head_loss - head_loss_pred

                #error = np.convolve(error, np.ones(self.window_size)/self.window_size, 'same')

                classification_df[comb] = error
                
            predictions = self.classifier_model.predict(classification_df)

            # pick the leak with the majority vote
            predictions = np.array([np.argmax(np.bincount(predictions))])
                                  

            return predictions