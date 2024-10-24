"""
NYC Taxi example adaptation of
https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-create-a-neural-network-for-regression-with-pytorch.md
"""

import torch
from torch import nn
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class NYCTaxiExampleDataset(torch.utils.data.Dataset):
    """Training data object for our nyc taxi data"""
    def __init__(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """
        This is the setup function for Xtrain and Ytrain
        Input: X_train and Y_train
        Output: Two tensors of X_train and Y_train, and it prints the encoded shape of the processed data
        """
        self.X_train = X_train
        self.y_train = y_train
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        self.X = torch.from_numpy(self._one_hot_X().toarray()) # potentially smarter ways to deal with sparse here
        self.y = torch.from_numpy(self.y_train.values)
        self.X_enc_shape = self.X.shape[-1]
        print(f"encoded shape is {self.X_enc_shape}")
    
    def __len__(self):
        """
        Returns the number of rows in the data set
        Input: None (just needs self to access the data stored in the object)
        Output: Returns the number of rows in the dataset
        """
        return len(self.X)

    def __getitem__(self, i):
        """
        This function allows you to grab a specific ro in your data set
        Input: An integr representing the index of the row you want to retrieve
        Output: Returns a tuple with two tensors, the features and the label for the i-th row
        """
        return self.X[i], self.y[i]
        
    def _one_hot_X(self):
        """
        Helps turn X_train features into a special format (one-hot encoding)
        Input: None (just needs self to access X_train)
        Output: Returns a one-hot encoded version of X_train 
        """
        return self.one_hot_encoder.fit_transform(self.X_train)

class MLP(nn.Module):
    """Multilayer Perceptron for regression. """
    def __init__(self, encoded_shape):
        """
        Initializes the layers and stacks the layers for the netwofk
        Input: An integer representing the number of features in the input column
        Output: No direct output, but it sets up the neural network layers
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(encoded_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1))
    
    def forward(self, x):
        """
        Takes an input tensor and returns the output of a network
        Input: A tensor representing the input data
        Output: Returns the output of the network
        """
        return self.layers(x)