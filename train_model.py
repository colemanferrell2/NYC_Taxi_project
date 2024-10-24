from scripts.data_utils import *
from scripts.model_utils import NYCTaxiExampleDataset 
from scripts.model_utils import MLP 
import torch
from torch import nn
import numpy as np
import random


def main(epochs: int = 5, learning_rate: float = 1e-4):
    """Simple training loop
    Input: Epochs (default 5) and learning_rate (Default 1e-4)
    Output: Return X_train, X_test, y_train, y_test, data, mlp
    """
    # Set fixed random number seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
  
    # load data, clean data, and split data
    raw_df = raw_taxi_df(filename="data/yellow_tripdata_2024-01.parquet")
    clean_df = clean_taxi_df(raw_df=raw_df)
    location_ids = ['PULocationID', 'DOLocationID']
    X_train, X_test, y_train, y_test = split_taxi_data(clean_df=clean_df, 
                                                   x_columns=location_ids, 
                                                   y_column="fare_amount", 
                                                   train_size=500000)

    # Pytorch
    dataset = NYCTaxiExampleDataset(X_train=X_train, y_train=y_train)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)
  
    # Initialize the MLP
    mlp = MLP(encoded_shape=dataset.X_enc_shape)
  
    # Define the loss function and optimizer
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)
  
    # Run the training loop
    for epoch in range(0, epochs): # Epochs
        print(f'Starting epoch {epoch+1}')
        current_loss = 0.0
    
        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
            # Get and prepare inputs
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Perform forward pass
            outputs = mlp(inputs)
            
            # Compute loss
            loss = loss_function(outputs, targets)
            
            # Perform backward pass
            loss.backward()
            
            # Perform optimization
            optimizer.step()
            
            # Print statistics
            current_loss += loss.item()
            if i % 10 == 0:
                print('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss / 500))
            current_loss = 0.0
    # Process is complete.
    print('Training process has finished.')
    return X_train, X_test, y_train, y_test, data, mlp

if __name__ == '__main__':
    main()