import pdb
import pandas as pd
import sklearn
from sklearn import pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import torch
import matplotlib.pyplot as plt

from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression, LogisticRegression

from ML_for_WDN.WAE import WassersteinAutoencoder
from ML_for_WDN.NN_utils import train_WAE

class ClassifierMajorityVote(BaseEstimator, ClassifierMixin):
    
        def __init__(
            self, 
            classifier
        ):
            
            self.classifier = classifier
            
        
        def __str__(self) -> str:
            return self.classifier.__str__()
    
        def fit(
            self, 
            X: pd.DataFrame,
            y: pd.DataFrame = None,
        ):
            
            #self.scaler = sklearn.preprocessing.StandardScaler()
            self.scaler = sklearn.preprocessing.MinMaxScaler()

            X = X.to_numpy()

            X = self.scaler.fit_transform(X)

            if y is not None:
                self.classifier.fit(X, y)
                self.anomaly_detection_mode = False
            else:
                self.classifier.fit(X)
                self.anomaly_detection_mode = True
    
            return self
        
        def predict(self, X: pd.DataFrame):
            
            X = X.to_numpy()

            X = self.scaler.transform(X)
            
            predictions = self.classifier.predict(X)

            if self.anomaly_detection_mode:
                predictions = predictions.mean()
                if predictions > 0.0:
                    predictions = 1
                else:
                    predictions = -1
            else:
                predictions = np.array([np.argmax(np.bincount(predictions))])

            return predictions
            

class UnsupervisedLeakDetector(BaseEstimator, ClassifierMixin):

    def __init__(
        self, 
        encoder_args: dict,
        decoder_args: dict,
        NN_train_args: dict,
        anomaly_detection_args: dict,
        device: str,
        verbose: bool = False,
    ):
        
        self.encoder_args = encoder_args
        self.decoder_args = decoder_args
        self.NN_train_args = NN_train_args
        self.anomaly_detection_args = anomaly_detection_args
        self.device = device
        self.verbose = verbose
    
    def __str__(self) -> str:
        return 'UnsupervisedLeakDetector'

    def fit(
        self, 
        X: np.ndarray,
        y: np.ndarray = None,
    ):
        
        self.encoder_args['input_dim'] = X.shape[1]
        self.decoder_args['output_dim'] = X.shape[1]

        self.model = WassersteinAutoencoder(
            encoder_args=self.encoder_args,
            decoder_args=self.decoder_args,
        )

        if self.verbose:

            print('########## Training stage 1 ##########')
            print('\n')
            print('Training autoencoder without leak data')
            print('\n')
            print('Autoencoder architecture:')
            print(f'- Latent dimension: {self.encoder_args["latent_dim"]}')
            print(f'- Encoder hidden dimensions: {self.encoder_args["hidden_dims"]}')
            print(f'- Decoder hidden dimensions: {self.decoder_args["hidden_dims"]}')
            print('\n')

        train_WAE(
            data=X,
            model=self.model,
            train_args=self.NN_train_args,
            device=self.device,
        )

        if self.verbose:
            print('\n')
            print('Autoencoder training complete')
            print('\n')

            print('########## Training stage 2 ##########')
            print('\n')

            print('Training anomaly detector using autoencoder')
            print('\n')


        self.latents = self.model.encoder(torch.tensor(X, dtype=torch.float32)).detach().numpy()

        self.latent_anomaly_detectors = []

        # Local outlier factor
        self.latent_local_outlier_factor = LocalOutlierFactor(
            **self.anomaly_detection_args,
            novelty=True,
        )
        self.latent_local_outlier_factor.fit(self.latents)

        # Isolation forest
        self.latent_isolation_forest = IsolationForest(
            **self.anomaly_detection_args
        )
        self.latent_isolation_forest.fit(self.latents)

        self.latent_anomaly_detectors.append(self.latent_local_outlier_factor)
        self.latent_anomaly_detectors.append(self.latent_isolation_forest)

        # Reconstruction error based anomaly detector
        latents = torch.tensor(self.latents, dtype=torch.float32) 
        reconstructions = self.model.decoder(latents).detach().numpy()
        reconstruction_errors =  ((reconstructions - X)**2).sum(axis=-1)
        reconstruction_errors = np.sqrt(reconstruction_errors)

        self.reconstruction_anomaly_detector = LocalOutlierFactor(
            **self.anomaly_detection_args,
            novelty=True,
        )
        self.reconstruction_anomaly_detector.fit(reconstruction_errors.reshape(-1, 1))
        
        if self.verbose:
            print('Anomaly detector training complete')
            print('\n')

        return self
    
    def predict(self, X: np.ndarray):

        latents = self.model.encoder(torch.tensor(X, dtype=torch.float32)).detach().numpy()

        anomaly_scores = []
        for detector in self.latent_anomaly_detectors:
            anomaly_scores.append(detector.predict(latents))

        latents = torch.tensor(latents, dtype=torch.float32) 
        reconstructions = self.model.decoder(latents).detach().numpy()
        reconstruction_errors =  ((reconstructions - X)**2).sum(axis=-1)
        reconstruction_errors = np.sqrt(reconstruction_errors)

        reconstruction_anomaly_scores = self.reconstruction_anomaly_detector.predict(
            reconstruction_errors.reshape(-1, 1)
        )

        anomaly_scores.append(reconstruction_anomaly_scores)
        anomaly_scores = np.array(anomaly_scores).T

        # If majority of the anomaly detectors predict an anomaly, then the sample is an anomaly
        anomaly_scores = np.sum(anomaly_scores, axis=-1)
        anomaly_scores[anomaly_scores < 0] = -1 # outliers are marked as -1
        anomaly_scores[anomaly_scores > 0] = 1 # inliers are marked as 1

        return anomaly_scores
    


class SupervisedReconstructionLeakDetector(BaseEstimator, ClassifierMixin):

    def __init__(
        self, 
        encoder_args: dict,
        decoder_args: dict,
        NN_train_args: dict,
        device: str,
        verbose: bool = False,
    ):
        
        self.encoder_args = encoder_args
        self.decoder_args = decoder_args
        self.NN_train_args = NN_train_args
        self.device = device
        self.verbose = verbose
        
    def __str__(self) -> str:
        return 'SupervisedReconstructionLeakDetector'

    def fit(
        self, 
        X: pd.DataFrame,
        y: pd.DataFrame,
    ):
        
        X = X.to_numpy()
        y = y.to_numpy()


        self.scaler = sklearn.preprocessing.StandardScaler()

        X = self.scaler.fit_transform(X)
        
        self.encoder_args['input_dim'] = X.shape[1]
        self.decoder_args['output_dim'] = X.shape[1]

        self.model = WassersteinAutoencoder(
            encoder_args=self.encoder_args,
            decoder_args=self.decoder_args,
        )

        if self.verbose:

            print('########## Training stage 1 ##########')
            print('\n')
            print('Training autoencoder without leak data')
            print('\n')
            print('Autoencoder architecture:')
            print(f'- Latent dimension: {self.encoder_args["latent_dim"]}')
            print(f'- Encoder hidden dimensions: {self.encoder_args["hidden_dims"]}')
            print(f'- Decoder hidden dimensions: {self.decoder_args["hidden_dims"]}')
            print('\n')

        train_WAE(
            data=X,
            model=self.model,
            train_args=self.NN_train_args,
            device=self.device,
            supervised_pars=y,
        )

        if self.verbose:
            print('\n')
            print('Autoencoder training complete')
            print('\n')

            print('########## Training stage 2 ##########')
            print('\n')

            print('Training anomaly detector using autoencoder')
            print('\n')


    
    def predict(self, X: pd.DataFrame):

        X = X.to_numpy()

        X = self.scaler.transform(X)

        self.model.eval()

        # Get latent representation
        latents = self.model.encoder(torch.tensor(X, dtype=torch.float32)).detach()

        # Reconstruct for all possible leak locations
        leak_location_loss = torch.zeros(X.shape[0], 4)
        latent_leak_location_loss = torch.zeros(X.shape[0], 4)
        for i in range(self.decoder_args['num_pars']):
            pars = i*torch.ones(X.shape[0], 1, dtype=torch.int32)

            out = self.model.decoder(latents, pars)

            leak_location_loss[:, i] = torch.sum(torch.abs(out - torch.tensor(X, dtype=torch.float32))**2, dim=1)

            latent_pred = self.model.encoder(out)

            latent_leak_location_loss[:, i] = torch.sum(torch.abs(latent_pred - latents)**2, dim=1)

        # Normalize reconstruction error
        leak_location_loss = leak_location_loss / leak_location_loss.max()
        latent_leak_location_loss = latent_leak_location_loss / latent_leak_location_loss.max()

        leak_location_loss = leak_location_loss# + latent_leak_location_loss
            
        # Get leak location with lowest reconstruction error
        _, leak_location_indices = torch.min(leak_location_loss, dim=1)

        predictions = np.array([np.argmax(np.bincount(leak_location_indices.detach().numpy()))])

        return predictions
    

class SupervisedLatentLeakDetector(BaseEstimator, ClassifierMixin):

    def __init__(
        self, 
        encoder_args: dict,
        decoder_args: dict,
        NN_train_args: dict,
        device: str,
        classifier: str = 'logistic_regression',
        verbose: bool = False,
    ):
        
        self.encoder_args = encoder_args
        self.decoder_args = decoder_args
        self.NN_train_args = NN_train_args
        self.device = device
        self.verbose = verbose

        self.classifier = classifier
    
    def __str__(self) -> str:
        return 'SupervisedLatentLeakDetector'

    def fit(
        self, 
        X: pd.DataFrame,
        y: pd.DataFrame,
    ):
        
        X = X.to_numpy()
        y = y.to_numpy()

        self.scaler = sklearn.preprocessing.StandardScaler()

        X = self.scaler.fit_transform(X)
                
        self.encoder_args['input_dim'] = X.shape[1]
        self.decoder_args['output_dim'] = X.shape[1]

        self.model = WassersteinAutoencoder(
            encoder_args=self.encoder_args,
            decoder_args=self.decoder_args,
        )

        if self.verbose:
            print('########## Training stage 1 ##########')
            print('\n')
            print('Training autoencoder without leak data')
            print('\n')
            print('Autoencoder architecture:')
            print(f'- Latent dimension: {self.encoder_args["latent_dim"]}')
            print(f'- Encoder hidden dimensions: {self.encoder_args["hidden_dims"]}')
            print(f'- Decoder hidden dimensions: {self.decoder_args["hidden_dims"]}')
            print('\n')

        train_WAE(
            data=X,
            model=self.model,
            train_args=self.NN_train_args,
            device=self.device,
        )


        if self.verbose:
            print('\n')
            print('Autoencoder training complete')
            print('\n')

            print('########## Training stage 2 ##########')
            print('\n')

            print('Training anomaly detector using autoencoder')
            print('\n')


        self.latents = self.model.encoder(torch.tensor(X, dtype=torch.float32)).detach().numpy()
        
        if self.classifier == 'logistic_regression':
            logistic_regression_args = {
                'penalty': 'l2',
                'C': 1e-2,
                'solver': 'lbfgs',
                'max_iter': 1000,
            }
            self.latent_classifier = LogisticRegression(**logistic_regression_args)
        elif self.classifier == 'random_forest':
            random_forest_args = {
                'n_estimators': 100,
                'max_depth': 10,
            }
            self.latent_classifier = sklearn.ensemble.RandomForestClassifier(**random_forest_args)
        
        self.latent_classifier.fit(self.latents, y)

    
    def predict(self, X: pd.DataFrame):

        X = X.to_numpy()

        X = self.scaler.transform(X)

        self.model.eval()

        # Get latent representation
        latents = self.model.encoder(torch.tensor(X, dtype=torch.float32)).detach()
        
        # Predict leak location
        leak_location_indices = self.latent_classifier.predict(latents)


        predictions = np.array([np.argmax(np.bincount(leak_location_indices))])

        return predictions
    


class SupervisedLinearRegressionLeakDetector(BaseEstimator, ClassifierMixin):
    """Leak detector based on linear regression residual
    
    The model is trained in two stages:
    1. Train linear regression models for each pair of sensors
    2. Train a logistic regression classifier based on linear regression residual

    The sensor data, X, is split into two parts: flow rate sensor data and pressure sensor data.
    The dimensions of X is (num_samples, num_sensors*2), where num_sensors is the number of sensors.
    The first num_sensors columns of X are the flow rate sensor data and the last num_sensors columns
    are the pressure sensor data.

    The target data, y, is a vector of integers, where each integer represents a leak location.
    The leak location is encoded as follows:
    - 0: No leak
    - 1: Leak location 1
    - 2: Leak location 2
    - 3: Leak location 3

    Parameters
    ----------
    model_args: dict
        Dictionary of arguments for the model

    verbose: bool
        If True, print training progress

    Attributes
    ----------
    num_sensors: int
        Number of sensors in the dataset

    sensor_pair_list: list
        List of all possible pairs of sensors

    linear_regression_models: dict
        Dictionary of linear regression models with sensor pair, (i, j), as key

    logistic_regression_classifier: LogisticRegression
        Logistic regression classifier trained on linear regression residual

    """

    def __init__(
        self, 
        model_args: dict = None,
        verbose: bool = False,
    ):
        
        self.model_args = model_args
        self.verbose = verbose

    def __str__(self) -> str:
        return 'SupervisedLinearRegressionLeakDetector'

    def fit(
        self, 
        X: np.ndarray,
        y: np.ndarray,
    ):
    
        self.num_sensors = X.shape[1]//2

        if self.verbose:
            print('########## Training stage 1 ##########')
            print('\n')
            print('Training linear regression without leak data')

        # Get all possible pairs of sensors
        self.sensor_pair_list = []
        for i in range(self.num_sensors):
            for j in range(self.num_sensors):
                if i != j:
                    self.sensor_pair_list.append((i, j))

        # Train linear regression models for each pair of sensors        
        # Define deictionary of linear regression models with sensor pair, (i, j), as key 
        self.linear_regression_models = {}
        for sensor_pair in self.sensor_pair_list:

            # Define linear regression model
            self.linear_regression_models[sensor_pair] = LinearRegression()

            # Get pressure sensors ids
            pressure_ids = (sensor_pair[0] + self.num_sensors, sensor_pair[1] + self.num_sensors)

            # Get flow rate and pressure difference data
            flow_rate_data = X[:, sensor_pair]
            pressure_data = X[:, pressure_ids[0]] - X[:, pressure_ids[1]]  

            # Train linear regression model
            self.linear_regression_models[sensor_pair].fit(
                flow_rate_data, 
                pressure_data,
            )
        
        if self.verbose:
            print('\n')
            print('Linear regression training complete')
            print('\n')

            print('########## Training stage 2 ##########')
            print('\n')
            
            print('Training logistic regression classifier based on linear regression residual')

        # Get linear regression residuals
        linear_regression_residuals = np.zeros((X.shape[0], len(self.sensor_pair_list)))
        for i, sensor_pair in enumerate(self.sensor_pair_list):

            # Get pressure sensors ids
            pressure_ids = (sensor_pair[0] + self.num_sensors, sensor_pair[1] + self.num_sensors)

            # Get flow rate and pressure difference data
            flow_rate_data = X[:, sensor_pair]
            pressure_data = X[:, pressure_ids[0]] - X[:, pressure_ids[1]]

            # Get linear regression residual
            pressure_pred = self.linear_regression_models[sensor_pair].predict(flow_rate_data)

            # Store linear regression residual
            linear_regression_residuals[:, i] = pressure_data - pressure_pred

        # Train logistic regression classifier
        logistic_regression_args = {
            'penalty': 'l2',
            'C': 1e-2,
            'solver': 'lbfgs',
            'max_iter': 1000,
        }
        
        self.logistic_regression_classifier = LogisticRegression(**logistic_regression_args)
        self.logistic_regression_classifier.fit(linear_regression_residuals, y)

        if self.verbose:
            print('\n')
            print('Logistic regression classifier training complete')
            print('\n')

        return self

    
    def predict(self, X: np.ndarray):

        # Get linear regression residuals
        linear_regression_residuals = np.zeros((X.shape[0], len(self.sensor_pair_list)))
        for i, sensor_pair in enumerate(self.sensor_pair_list):

            # Get pressure sensors ids
            pressure_ids = (sensor_pair[0] + self.num_sensors, sensor_pair[1] + self.num_sensors)

            # Get flow rate and pressure difference data
            flow_rate_data = X[:, sensor_pair]
            pressure_data = X[:, pressure_ids[0]] - X[:, pressure_ids[1]]  

            # Get linear regression residual
            pressure_pred = self.linear_regression_models[sensor_pair].predict(flow_rate_data)

            # Store linear regression residual
            linear_regression_residuals[:, i] = pressure_data - pressure_pred

        # Predict leak location using logistic regression classifier
        leak_location_preds = self.logistic_regression_classifier.predict(linear_regression_residuals)

        return leak_location_preds
    
class SupervisedPolynomialRegressionLeakDetector(BaseEstimator, ClassifierMixin):
    """Leak detector based on linear regression residual
    
    The model is trained in two stages:
    1. Train polynomial regression models for each pair of sensors
    2. Train a logistic regression classifier based on linear regression residual

    The sensor data, X, is split into two parts: flow rate sensor data and pressure sensor data.
    The dimensions of X is (num_samples, num_sensors*2), where num_sensors is the number of sensors.
    The first num_sensors columns of X are the flow rate sensor data and the last num_sensors columns
    are the pressure sensor data.

    The target data, y, is a vector of integers, where each integer represents a leak location.
    The leak location is encoded as follows:
    - 0: No leak
    - 1: Leak location 1
    - 2: Leak location 2
    - 3: Leak location 3

    Parameters
    ----------
    model_args: dict
        Dictionary of arguments for the model

    verbose: bool
        If True, print training progress

    Attributes
    ----------
    num_sensors: int
        Number of sensors in the dataset

    sensor_pair_list: list
        List of all possible pairs of sensors

    linear_regression_models: dict
        Dictionary of polynomial regression models with sensor pair, (i, j), as key

    logistic_regression_classifier: LogisticRegression
        Logistic regression classifier trained on linear regression residual

    """

    def __init__(
        self, 
        model_args: dict = None,
        verbose: bool = False,
    ):
        
        self.model_args = model_args
        self.verbose = verbose

    def __str__(self) -> str:
        return 'SupervisedPolynomialRegressionLeakDetector'

    def fit(
        self, 
        X: np.ndarray,
        y: np.ndarray,
    ):
    
        self.num_sensors = X.shape[1]//2

        X_no_leak = X[y == 0]


        if self.verbose:
            print('########## Training stage 1 ##########')
            print('\n')
            print('Training polynomial regression without leak data')

        # Get all possible pairs of sensors
        self.sensor_pair_list = []
        for i in range(self.num_sensors):
            for j in range(self.num_sensors):
                if i != j:
                    self.sensor_pair_list.append((i, j))

        # Train polynomial regression models for each pair of sensors        
        # Define deictionary of polynomial regression models with sensor pair, (i, j), as key 
        self.polynomial_regression_models = {}
        for sensor_pair in self.sensor_pair_list:

            # Define polynomial regression model
            self.polynomial_regression_models[sensor_pair] = pipeline.Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('linear', LinearRegression()),
            ])            

            # Get pressure sensors ids
            pressure_ids = (sensor_pair[0] + self.num_sensors, sensor_pair[1] + self.num_sensors)

            # Get flow rate and pressure difference data
            flow_rate_data = X_no_leak[:, sensor_pair]
            pressure_data = X_no_leak[:, pressure_ids[0]] - X_no_leak[:, pressure_ids[1]]  

            # Train linear regression model
            self.polynomial_regression_models[sensor_pair].fit(
                flow_rate_data, 
                pressure_data,
            )
        
        if self.verbose:
            print('\n')
            print('Linear regression training complete')
            print('\n')

            print('########## Training stage 2 ##########')
            print('\n')
            
            print('Training logistic regression classifier based on linear regression residual')

        # Get polynomial regression residuals
        regression_residuals = np.zeros((X.shape[0], len(self.sensor_pair_list)))
        for i, sensor_pair in enumerate(self.sensor_pair_list):

            # Get pressure sensors ids
            pressure_ids = (sensor_pair[0] + self.num_sensors, sensor_pair[1] + self.num_sensors)

            # Get flow rate and pressure difference data
            flow_rate_data = X[:, sensor_pair]
            pressure_data = X[:, pressure_ids[0]] - X[:, pressure_ids[1]]

            # Get linear regression residual
            pressure_pred = self.polynomial_regression_models[sensor_pair].predict(flow_rate_data)

            # Store linear regression residual
            regression_residuals[:, i] = pressure_data - pressure_pred

        # Train logistic regression classifier
        logistic_regression_args = {
            'penalty': 'l2',
            'C': 1e-2,
            'solver': 'lbfgs',
            'max_iter': 1000,
        }
        
        self.logistic_regression_classifier = pipeline.Pipeline([
            ('scaler', sklearn.preprocessing.StandardScaler()),
            ('logistic', LogisticRegression(**logistic_regression_args)),
        ])
        self.logistic_regression_classifier.fit(regression_residuals, y)

        if self.verbose:
            print('\n')
            print('Logistic regression classifier training complete')
            print('\n')

        return self

    
    def predict(self, X: np.ndarray):

        # Get linear regression residuals
        regression_residuals = np.zeros((X.shape[0], len(self.sensor_pair_list)))
        for i, sensor_pair in enumerate(self.sensor_pair_list):

            # Get pressure sensors ids
            pressure_ids = (sensor_pair[0] + self.num_sensors, sensor_pair[1] + self.num_sensors)

            # Get flow rate and pressure difference data
            flow_rate_data = X[:, sensor_pair]
            pressure_data = X[:, pressure_ids[0]] - X[:, pressure_ids[1]]

            # Get linear regression residual
            pressure_pred = self.polynomial_regression_models[sensor_pair].predict(flow_rate_data)

            # Store linear regression residual
            regression_residuals[:, i] = pressure_data - pressure_pred

        # Predict leak location using logistic regression classifier
        leak_location_preds = self.logistic_regression_classifier.predict(regression_residuals)

        return leak_location_preds
    
