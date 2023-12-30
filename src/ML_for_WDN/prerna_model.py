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


def training_data(df,link_names,head_names):
    
    data_flow = np.array(df[link_names])  # convering to litres per sec 
    data_head = np.array(df[head_names])
    

    train_out= data_head[:,0] - data_head[:,1] # deltaH
    train_in = data_flow                       # flow1, flow2
    
    return train_in, train_out

def data_in_out(sensor_pair,df, sensor_list):
    
    h1= sensor_list[sensor_pair[0]-1][0]
    h2= sensor_list[sensor_pair[1]-1][0]
    f1= sensor_list[sensor_pair[0]-1][1]
    f2= sensor_list[sensor_pair[1]-1][1]
    
    link_name = ['Link_flow'+str(f1),'Link_flow'+str(f2)] 
    node_name = ['Node_head'+str(h1),'Node_head'+str(h2)]
    
    data_in, data_out = training_data(df,link_name,node_name)
    
    return data_in, data_out 

def datafortrees(
    lin_models,
    numcases,
    df_recent,
    sample_len,
    casetype, 
    colnames,
    combs
):
    
    df_cases = pd.DataFrame(columns=colnames)
    fracsize_recent=0.3
    
    fracsize_reference = fracsize_recent#sample_len/len(df_train_class_model_ref)
    
    # first loop to randomly select a sample test set
    for i in range(numcases):
        df_recent_sample = df_recent.sample(frac=fracsize_recent)
        df_reference_sample = df_train_class_model_ref.sample(frac=fracsize_reference)
        
        # second loop to cover all possible leak combinations
        comb_data = []
        for comb in combs:
            xtest_rec, ytest_rec = data_in_out(comb,df_recent_sample)
                
            xtest_ref, ytest_ref = data_in_out(comb,df_reference_sample)
            

            # load the linear regression model, make predictions and store results
            model_name = 'linmodel'+str(comb[0])+str(comb[1])+'.pkl'

            lin_model = lin_models[model_name]

            pred_ref = lin_model.predict(xtest_ref).reshape(-1)
            pred_rec = lin_model.predict(xtest_rec).reshape(-1)
            error_ref = (ytest_ref-pred_ref)
            error_rec = (ytest_rec-pred_rec)
            stat, pval = ks_2samp(error_ref,error_rec) #to test the model runs fine
            comb_list = [np.mean(error_ref), np.mean(error_rec),stat,pval]            
            comb_data.extend(comb_list)
            
        comb_series = pd.Series(comb_data,index=df_cases.columns)    
        df_cases = df_cases.append(comb_series,ignore_index=True)
      
    return df_cases

class PrernaModel(BaseEstimator, ClassifierMixin):
    
        def __init__(
            self,
            model_args: dict,
            classifier: str = 'logistic_regression',
            verbose: bool = False,
        ) -> None:
            
            self.model_args = model_args
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
                self.classifier_model = LogisticRegression()
            elif self.classifier == 'random_forest':
                self.classifier_model = RandomForestClassifier()

            # Get error distribution on non-leak data
            no_leak_error_distributions = {}
            for comb in self.combs:
                x_train = self._get_flowrates_for_sensor_pair(X_no_leak, comb)
                y_train = self._get_headloss_for_sensor_pair(X_no_leak, comb)
                lin_model_name = str(comb[0])+str(comb[1])
                y_pred = self.lin_models[lin_model_name].predict(x_train)
                y_pred = y_pred.reshape(-1)
                y_train = y_train.reshape(-1)
                error = y_train - y_pred
                no_leak_error_distributions[lin_model_name] = error

            pbar = tqdm(
                self.combs,
                total=len(self.combs)
            )
            for comb in pbar:
                x_train = self._get_flowrates_for_sensor_pair(X, comb)
                y_train = self._get_headloss_for_sensor_pair(X, comb)
                
                lin_model_name = str(comb[0])+str(comb[1])
                lin_model = self.lin_models[lin_model_name]
                y_pred = lin_model.predict(x_train)

                y_pred = y_pred.reshape(-1)
                y_train = y_train.reshape(-1)
                error = y_train - y_pred

                stat,pval = ks_2samp(
                    no_leak_error_distributions[lin_model_name],
                    error
                ) 
                pdb.set_trace()

            
            

                
             


        def predict(self, X: pd.DataFrame) -> np.ndarray:
            pass