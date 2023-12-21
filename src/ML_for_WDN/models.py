import pdb
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import torch

from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

from ML_for_WDN.WAE import WassersteinAutoencoder
from ML_for_WDN.NN_utils import train_WAE

class UnsupervisedLeakDetector(BaseEstimator, ClassifierMixin):

    def __init__(
        self, 
        encoder_args: dict,
        decoder_args: dict,
        NN_train_args: dict,
        anomaly_detection_args: dict,
        device: str,
    ):
        
        self.encoder_args = encoder_args
        self.decoder_args = decoder_args
        self.NN_train_args = NN_train_args
        self.anomaly_detection_args = anomaly_detection_args
        self.device = device
        

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

        '''
        # compute one-sided running average
        anomaly_scores = pd.Series(anomaly_scores)
        anomaly_scores_rolling = anomaly_scores.rolling(window=100).mean()
        anomaly_scores_rolling.iloc[0:100] = anomaly_scores.iloc[0:100]

        anomaly_scores = anomaly_scores_rolling.values
        anomaly_scores[anomaly_scores < 0] = -1 # outliers are marked as -1
        anomaly_scores[anomaly_scores > 0] = 1 # inliers are marked as 1     
        pdb.set_trace()   
        '''
        
        return anomaly_scores
    


class SupervisedReconstructionLeakDetector(BaseEstimator, ClassifierMixin):

    def __init__(
        self, 
        encoder_args: dict,
        decoder_args: dict,
        NN_train_args: dict,
        device: str,
    ):
        
        self.encoder_args = encoder_args
        self.decoder_args = decoder_args
        self.NN_train_args = NN_train_args
        self.device = device
        

    def fit(
        self, 
        X: np.ndarray,
        y: np.ndarray,
    ):
        
        self.encoder_args['input_dim'] = X.shape[1]
        self.decoder_args['output_dim'] = X.shape[1]

        self.model = WassersteinAutoencoder(
            encoder_args=self.encoder_args,
            decoder_args=self.decoder_args,
        )

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

        print('\n')
        print('Autoencoder training complete')
        print('\n')

        print('########## Training stage 2 ##########')
        print('\n')

        print('Training anomaly detector using autoencoder')
        print('\n')


    
    def predict(self, X: np.ndarray):

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

        return leak_location_indices.detach().numpy()
    

class SupervisedLatentLeakDetector(BaseEstimator, ClassifierMixin):

    def __init__(
        self, 
        encoder_args: dict,
        decoder_args: dict,
        NN_train_args: dict,
        device: str,
    ):
        
        self.encoder_args = encoder_args
        self.decoder_args = decoder_args
        self.NN_train_args = NN_train_args
        self.device = device
        

    def fit(
        self, 
        X: np.ndarray,
        y: np.ndarray,
    ):
        
        self.encoder_args['input_dim'] = X.shape[1]
        self.decoder_args['output_dim'] = X.shape[1]

        self.model = WassersteinAutoencoder(
            encoder_args=self.encoder_args,
            decoder_args=self.decoder_args,
        )

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

        print('\n')
        print('Autoencoder training complete')
        print('\n')

        print('########## Training stage 2 ##########')
        print('\n')

        print('Training anomaly detector using autoencoder')
        print('\n')


        self.latents = self.model.encoder(torch.tensor(X, dtype=torch.float32)).detach().numpy()
        logistic_regression_args = {
            'penalty': 'l2',
            'C': 1e-2,
            'solver': 'lbfgs',
            'max_iter': 1000,
        }
        self.latent_classifier = LogisticRegression(**logistic_regression_args)
        self.latent_classifier.fit(self.latents, y)

    
    def predict(self, X: np.ndarray):

        self.model.eval()

        # Get latent representation
        latents = self.model.encoder(torch.tensor(X, dtype=torch.float32)).detach()
        
        # Predict leak location
        leak_location_indices = self.latent_classifier.predict(latents)

        return leak_location_indices#.detach().numpy()