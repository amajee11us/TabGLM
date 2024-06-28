import os
import copy
import openml
import torch 
import yaml
import numpy as np
import pandas as pd
from typing import Optional
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import pearsonr
from transformers import TapasTokenizer, AutoTokenizer

from .utils import to_dense_array

def pick_dataset_openml(dataset_name, target, random_state): 
    '''
    Fetch OpenML dataset - General wrapper for dataset collections
    '''
    # Get dataset by name
    dataset = openml.datasets.get_dataset(dataset_name)
    print(dataset)
    
    # Get the data itself as a dataframe (or otherwise)
    df, y, _, _ = dataset.get_data(dataset_format="dataframe")

    for column in df.columns:
        if df[column].dtype == 'category':  # Assuming object dtype implies categorical
            df[column] = pd.factorize(df[column])[0]
    
    X, y = df.drop(target, axis=1), df[target]
    X = X.loc[:, X.nunique() > 1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                        random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, 
                                                      random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test

class TGRLDataloader(Dataset):
    '''
    Table-Graph Multi-Modal DataLoader
    Joint dataloader for the TGRL model.
    '''
    def __init__(self, raw_data, labels, multimodal, text_encoder="tapas"):
        
        self.raw_data = raw_data
        self.labels = labels

        self.adjacency_matrix,_,_ = self.compute_adjacency_matrix(
                                                self_loop_weight=20, 
                                                threshold=0.2)
        self.graph_data = self.generate_graph_tensors()

        self.multimodal = multimodal # required switch during inferencing

        # Initialize model and tokenizer
        if text_encoder == "tapas":
            self.tokenizer = TapasTokenizer.from_pretrained('google/tapas-base')
        elif text_encoder == "tapex":
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/tapex-large-finetuned-tabfact")

        self.label_tensor = torch.FloatTensor(self.labels)

    def compute_adjacency_matrix(self, self_loop_weight, threshold):
        index_to_name = {i: n for i, n in enumerate(self.raw_data.columns)}
        name_to_index = {n: i for i, n in enumerate(self.raw_data.columns)}

        adj_matrix = np.zeros((len(self.raw_data.columns), len(self.raw_data.columns)), dtype=float)

        for col_1 in self.raw_data.columns:
            for col_2 in self.raw_data.columns:
                if col_1 == col_2:
                    adj_matrix[name_to_index[col_1], name_to_index[col_2]] = self_loop_weight
                else:
                    corr, _ = pearsonr(self.raw_data[col_1], self.raw_data[col_2])
                    if np.isnan(corr) or abs(corr) < threshold:
                        adj_matrix[name_to_index[col_1], name_to_index[col_2]] = 0
                    else:
                        adj_matrix[name_to_index[col_1], name_to_index[col_2]] = corr

        return adj_matrix, index_to_name, name_to_index

    def generate_graph_tensors(self):
        data = copy.deepcopy(self.raw_data)
        data_np = data.to_numpy()

        # Convert entire data and labels arrays to tensors
        data_tensor = torch.FloatTensor(data_np)

        # Reshape data tensor to match the required shape
        graph_tensor = data_tensor.view(len(data), self.adjacency_matrix.shape[0], 1)

        return graph_tensor

    def extract_sample_text_encoding(self, row):
        # Prepare the table (single row as a table)
        table_df = row.to_frame().T

        # Tokenize the input (no need for a question)
        inputs = self.tokenizer(table=table_df.reset_index(drop=True).astype(str), 
                                padding='max_length', 
                                truncation=True,
                                return_tensors='pt')
        
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}

        return inputs

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        datapoint_idx = self.raw_data.iloc[idx]

        label = self.label_tensor[idx]

        graph_encoding = self.graph_data[idx]

        if not self.multimodal:
            return graph_encoding, label
        
        text_encoding = self.extract_sample_text_encoding(datapoint_idx)

        return graph_encoding, text_encoding, label

class Norm2Scaler:
    """Log normalize and scale data

    Log normalization and scaling procedure as described as norm-2 in the
    DeepInsight paper supplementary information.
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self._min0 = X.min(axis=0)
        self._max = np.log(X + np.abs(self._min0) + 1.01).max()
        return self

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None
                      ) -> np.ndarray:
        self._min0 = X.min(axis=0)
        X_norm = np.log(X + np.abs(self._min0) + 1.01)
        self._max = X_norm.max()
        return X_norm / self._max

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None
                  ) -> np.ndarray:
        X_norm = np.log(X + np.abs(self._min0) + 1.01).clip(0, None)
        return (X_norm / self._max).clip(0, 1)
    
def preprocess_and_load_data(data_config=None):
    '''
    Master script for normalizing and pre-processing openML datasets
    '''
    if data_config is None: 

        # Get the current directory (should be where your notebook is)
        current_notebook_path = os.getcwd()
        # Navigate up to the parent directory (your_project_root)
        parent_directory = os.path.dirname(current_notebook_path)
        # Now, construct the path to the configuration file
        config_path = os.path.join(parent_directory, "configs", "config.yml")
        
        with open(config_path, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        
        config = config['data_config']    
    else: 
        config = data_config
    
    # Load the dataset
    X_train, X_val, X_test, y_train, y_val, y_test = pick_dataset_openml(
        config['dataset'], config['target'], config['random_state']
    )
    
    # Initialize the scalers
    ln = Norm2Scaler()
    scaler_y = StandardScaler() if config['task_type'] == "regression" else LabelEncoder()

    # Normalize the features
    X_train_norm = pd.DataFrame(ln.fit_transform(X_train.to_numpy()), columns=X_train.columns)
    X_val_norm = pd.DataFrame(ln.transform(X_val.to_numpy()), columns=X_val.columns)
    X_test_norm = pd.DataFrame(ln.transform(X_test.to_numpy()), columns=X_test.columns)

    # Calculate the percentage of NAs in each column
    percent_nas = X_train_norm.isna().mean()
    # Create a mask for columns with less than 15% NAs
    columns_to_keep = percent_nas < config['na_threshold']
    
    # Use the mask to select columns with less than 15% NAs in all sets
    X_train_norm = X_train_norm.loc[:, columns_to_keep]
    X_val_norm = X_val_norm.loc[:, columns_to_keep]
    X_test_norm = X_test_norm.loc[:, columns_to_keep]
    
    # Remove rows with any NAs from X and corresponding rows from y
    rows_to_keep_train = ~X_train_norm.isna().any(axis=1)
    X_train_norm = X_train_norm.loc[rows_to_keep_train]
    y_train = y_train[rows_to_keep_train.to_numpy()]  # Apply the same mask to y_train
    
    rows_to_keep_val = ~X_val_norm.isna().any(axis=1)
    X_val_norm = X_val_norm.loc[rows_to_keep_val]
    y_val = y_val[rows_to_keep_val.to_numpy()]  # Apply the same mask to y_val
    
    rows_to_keep_test = ~X_test_norm.isna().any(axis=1)
    X_test_norm = X_test_norm.loc[rows_to_keep_test]
    y_test = y_test[rows_to_keep_test.to_numpy()]  # Apply the same mask to y_test

    # Normalize/Encode the target variable
    if config['task_type'] == "regression": 
        y_train_dense = to_dense_array(y_train)
        y_val_dense = to_dense_array(y_val)
        y_test_dense = to_dense_array(y_test)
    
        y_train_enc = scaler_y.fit_transform(y_train_dense.reshape(-1, 1)).flatten().astype('float32')
        y_val_enc = scaler_y.transform(y_val_dense.reshape(-1, 1)).flatten().astype('float32')
        y_test_enc = scaler_y.transform(y_test_dense.reshape(-1, 1)).flatten().astype('float32')
    else: 
        y_train_enc = scaler_y.fit_transform(to_dense_array(y_train))
        y_val_enc = scaler_y.transform(to_dense_array(y_val))
        y_test_enc = scaler_y.transform(to_dense_array(y_test))
        
    return X_train_norm, X_val_norm, X_test_norm, y_train_enc, y_val_enc, y_test_enc, scaler_y
