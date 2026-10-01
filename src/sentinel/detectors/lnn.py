#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LNN-based autoencoder anomaly detector module.

This module provides implementation of a Liquid Neural Network (LNN) based autoencoder
for anomaly detection in time series data, offering improved capability for 
handling complex temporal dynamics in the data.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from ncps.wirings import AutoNCP
from ncps.torch import LTC
import copy


class LNNEncoder(nn.Module):
    """
    Liquid Neural Network (LNN) encoder for anomaly detection.
    
    This class implements an encoder using Liquid Neural Networks
    for capturing complex temporal dynamics in time series data.

    Args:
            - n_features (int): Dimension of input features

            - seq_len (int): Length of input sequence

            - latent_dim (int): Dimension of the latent space representation

    """
    def __init__(self, n_features, seq_len, latent_dim=20):

        super(LNNEncoder, self).__init__()
        
        self.seq_len, self.n_features = seq_len, n_features
        self.latent_dim = latent_dim
        
        # Define the wiring for the Liquid Neural Network
        # Ensure units is at least latent_dim + 2 to avoid errors
        units = max(32, latent_dim + 2)
        wiring = AutoNCP(units=units, output_size=latent_dim)

        # Define the LNN-based recurrent layer
        self.rnn = LTC(n_features, wiring, batch_first=True)

    def forward(self, x):
        """
        Forward pass of the LNN encoder model.

        Args:
            x (torch.Tensor): Input features of shape (seq_len * n_features,)

        Returns:
            torch.Tensor: Latent space representation of shape (1, latent_dim)
        """
        x = x.reshape((1, self.seq_len, self.n_features))
        output, _ = self.rnn(x)
        # output[:, -1, :] shape: (1, latent_dim)
        return output[:, -1, :].reshape((1, self.latent_dim))


class Decoder(nn.Module):
    """
    Decoder neural network for anomaly detection.
    
    This class implements a decoder with configurable architecture
    for decoding latent space representation to output features.

    Args:
            - n_features (int): Dimension of output features

            - seq_len (int): Length of output sequence

            - input_dim (int): Dimension of the latent space representation
    """
    def __init__(self, n_features, seq_len, input_dim=20):

        super(Decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features

        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        """
        Forward pass of the decoder model.

        Args:
            x (torch.Tensor): Latent space representation of shape (1, input_dim)

        Returns:
            torch.Tensor: Output features of shape (seq_len * n_features,)
        """
        # x shape: (1, input_dim) → repeat to (1, seq_len, input_dim)
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, (hidden_n, cell_n) = self.rnn(x)
        # x shape: (1, seq_len, hidden_dim) → (seq_len, hidden_dim)
        x = x.reshape((self.seq_len, self.hidden_dim))
        # output shape: (seq_len, n_features) → flatten
        return self.output_layer(x).reshape(-1)


class LNNAutoencoder(nn.Module):
    """
    LNN-based autoencoder neural network for anomaly detection.
    
    This class implements an autoencoder using Liquid Neural Networks
    for the encoder, providing improved capability for handling complex 
    temporal dynamics in time series data.

    Args:
            - n_features (int): Dimension of input features

            - seq_len (int): Length of input sequence

            - latent_dim (int): Dimension of the latent space representation
    """
    def __init__(self, n_features, seq_len, latent_dim=20):
      
        super(LNNAutoencoder, self).__init__()
        
        self.encoder = LNNEncoder(n_features, seq_len, latent_dim)
        self.decoder = Decoder(n_features, seq_len, latent_dim)
    
    def forward(self, x):
        """
        Forward pass of the LNN autoencoder model.
        
        Args:

            x (torch.Tensor): Input features
        
        Returns:

            torch.Tensor: Decoded features
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return decoded


class LNNDetector:
    """
    LNN-based autoencoder anomaly detector.
    
    This class implements an anomaly detector using a Liquid Neural Network-based
    autoencoder for detecting anomalies in time series data, offering improved 
    capability for handling complex temporal dynamics.

    Args:
            - n_features (int): Dimension of input features

            - seq_len (int): Length of input sequence

            - latent_dim (int): Dimension of the latent space representation

            - learning_rate (float): Learning rate for training the model

            - epochs (int): Number of epochs for training the model

            - threshold_multiplier (float): Anomaly threshold multiplier
    """
    def __init__(self, n_features, seq_len, latent_dim=8, 
                 learning_rate=0.001, epochs=20, 
                 threshold_multiplier=3.0):

        self.n_features = n_features
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold_multiplier = threshold_multiplier
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LNNAutoencoder(n_features, seq_len, latent_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.L1Loss(reduction='sum')  # Using L1Loss as in the notebook example
        self.scaler = StandardScaler()
        self.threshold = None
        self.history = {'train': [], 'val': []}
        
    def fit(self, X, X_test=None, verbose=True):
        """
        Fit the LNN-based anomaly detector to the input data.
        
        Args:

            - X (np.ndarray): Input data for training the model

            - X_test (np.ndarray, optional): Test data for validation
            verbose (bool): Whether to print training progress
        
        Returns:

            dict: Training history
        """
        self.scaler = self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        
        best_loss = float('inf')
        best_model_wts = copy.deepcopy(self.model.state_dict())
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_losses = []
            
            for seq_true in X_tensor:
                seq_true = seq_true.to(self.device)
                seq_pred = self.model(seq_true)
                seq_pred = seq_pred.reshape((seq_pred.shape[0],))
                loss = self.criterion(seq_pred, seq_true)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_losses.append(loss.item())
            
            train_loss = np.mean(train_losses)
            self.history['train'].append(train_loss)
            
            # Validation phase
            val_loss = None
            if X_test is not None:
                self.model.eval()
                val_losses = []
                
                with torch.no_grad():
                    for seq_true in X_test_tensor:
                        seq_true = seq_true.to(self.device)
                        seq_pred = self.model(seq_true)
                        seq_pred = seq_pred.reshape((seq_pred.shape[0],))
                        loss = self.criterion(seq_pred, seq_true)
                        val_losses.append(loss.item())
                
                val_loss = np.mean(val_losses)
                self.history['val'].append(val_loss)
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())
            else:
                if train_loss < best_loss:
                    best_loss = train_loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())
            
            if verbose:
                log_message = f'Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}'
                if val_loss is not None:
                    log_message += f', Val Loss: {val_loss:.4f}'
                print(log_message)
        
        # Load the best model
        self.model.load_state_dict(best_model_wts)
        
        # Calculate threshold based on training data
        self._calculate_threshold(X_tensor)
        
        return self.history
        
    def _calculate_threshold(self, X_scaled):
        """
        Calculate the anomaly threshold based on reconstruction errors.
        
        Args:

            X_scaled (torch.Tensor): Scaled input data
            
        Returns:

            float: Anomaly threshold
        """
        self.model.eval()
        reconstruction_errors = []
        
        with torch.no_grad():
            for seq_true in X_scaled:
                seq_true = seq_true.to(self.device)
                seq_pred = self.model(seq_true)
                seq_pred = seq_pred.reshape((seq_pred.shape[0],))
                
                # Calculate L1 distance (reconstruction error)
                error = torch.abs(seq_pred - seq_true).mean().item()
                reconstruction_errors.append(error)
        
        # Set threshold based on distribution of reconstruction errors
        self.threshold = np.mean(reconstruction_errors) + self.threshold_multiplier * np.std(reconstruction_errors)
        
        return self.threshold
        
    def predict(self, X):
        """
        Predict anomalies in the input data.
        
        Args:

            X (np.ndarray): Input data for anomaly detection
            
        Returns:

            np.ndarray: Binary array where 1 indicates anomaly, 0 indicates normal
        """
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        self.model.eval()
        anomalies = []
        
        with torch.no_grad():
            for seq_true in X_tensor:
                seq_true = seq_true.to(self.device)
                seq_pred = self.model(seq_true)
                seq_pred = seq_pred.reshape((seq_pred.shape[0],))
                
                # Calculate L1 distance (reconstruction error)
                error = torch.abs(seq_pred - seq_true).mean().item()
                anomalies.append(1 if error > self.threshold else 0)
        
        return np.array(anomalies)
    
    def anomaly_score(self, X):
        """
        Calculate anomaly scores for the input data.
        
        Args:

            X (np.ndarray): Input data for anomaly detection
        
        Returns:

            np.ndarray: Anomaly scores (reconstruction errors)
        """
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        self.model.eval()
        scores = []
        
        with torch.no_grad():
            for seq_true in X_tensor:
                seq_true = seq_true.to(self.device)
                seq_pred = self.model(seq_true)
                seq_pred = seq_pred.reshape((seq_pred.shape[0],))
                
                # Calculate L1 distance (reconstruction error)
                error = torch.abs(seq_pred - seq_true).mean().item()
                scores.append(error)
        
        return np.array(scores)
    
    def save_model(self, path):
        """
        Save the model to disk.
        
        Args:

            path (str): File path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'threshold': self.threshold,
            'scaler': self.scaler,
        }, path)
        
    def load_model(self, path):
        """
        Load the model from disk.
        
        Args:
        
            path (str): File path to load the model
        """
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.threshold = checkpoint['threshold']
        self.scaler = checkpoint['scaler']