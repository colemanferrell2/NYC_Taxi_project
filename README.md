# STOR 674 Homework 4

This assignmet demonstrates practices that enhance the reproducibility of one's code. We are provided code that retrieves trip data from yellow taxi cabs in New York City, including but not entirely encompassing, trip distance, trip time , and fare. An analysis of the data and a deep neural network are executed to help draw conclusions about the taxi data. My task is to reorganize the code to assist in clarity and reproducibility. 

## Steps to run  code

### Step 1: Get data
The data can be retrieved by running the following code:
```
bash data/get_taxi.sh
```
The `.parquet` file corresponding to the yellow taxi cab data will be saved into the `/data` folder.

### Step 2: Installing Dependencies
The below code can install the required dependencies that are nessessary for the remaining codes. This step is essential to be able to run the code in the remaining steps.
```
pip install -r requirements.txt   
```

### Step 3: Clean data and train model
The data can be cleaned and model trained all in one step by executing the code:
```
python train_model.py   
```
If you would like to specify the number of epochs and the learning rate, you can instead run this code:
```
python train_model.py --epochs 5 -- learning_rate 1e-4 
# Replace 5 and 1e-4 by your desired epochs and learning rate
```

## Organization of Files

### `train_model` (main file)
The code implements a training loop for a multi-layer perceptron neural network using PyTorch to predict taxi fares based on the dataset of yellow taxi trips in New York City. It imports utility functions from `scripts.data_utils`, such as `raw_taxi_df`, `clean_taxi_df`, and `split_taxi_data`, to handle data loading, cleaning, and splitting. Additionally, it imports `NYCTaxiExampleDataset` and `MLP` from `scripts.model_utils` to create a dataset object and initialize the MLP model, which is a type of feedforward neural network designed for regression tasks. The training process involves computing the loss and updating model parameters through backpropagation over a specified number of epochs, and finally, the trained model along with the datasets is returned for further analysis.

### Data Folder
This folder contains the script (`get_taxi.sh`) to load the yellow taxi cab data and stores the `.parquet` file

### Scripts
There are two python files that contain classes and functions nessessary for `main()` execution

##### `data_utils.py`
This file is designated to help clean and preprocess the data. First the data is transformed into a data frame, and then the data is cleaned. This includes removing NA observations and removing suspected inaccurate points. (For example, a trip that has a distance of 100 miles, which is uncommon in NYC). Finally, the data is split into training and testing subsets so the neural network can be performed.

##### `model_utils`
There are two classes in this file. The first being `NYCTaxiExampleDataset`, which formats the model and retrieves nessessary information from the data to set up the neural network model. Secondly, `MLP`, is for Multilayer Perceptron for regression.

### `requirement.txt`
Contains list of all dependencies for the analysis




