import os
import openml
import yaml
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader

def pick_dataset_openml(dataset_name, target, random_state):
    """
    Fetch OpenML dataset - General wrapper for dataset collections
    """
    # Get dataset by name
    dataset = openml.datasets.get_dataset(dataset_name)
    print(dataset)

    # Get the data itself as a dataframe (or otherwise)
    df, y, _, _ = dataset.get_data(dataset_format="dataframe")

    for column in df.columns:
        if df[column].dtype == "category":  # Assuming object dtype implies categorical
            df[column] = pd.factorize(df[column])[0]

    X, y = df.drop(target, axis=1), df[target]
    X = X.loc[:, X.nunique() > 1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def to_dense_array(series):
    # Check if the series is stored as a sparse array
    if isinstance(series.array, pd.arrays.SparseArray):
        # Convert SparseArray to a dense NumPy array
        return series.sparse.to_dense().to_numpy()
    else:
        # If it's not a sparse array, just convert to NumPy array directly
        return series.to_numpy()


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

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        self._min0 = X.min(axis=0)
        X_norm = np.log(X + np.abs(self._min0) + 1.01)
        self._max = X_norm.max()
        return X_norm / self._max

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        X_norm = np.log(X + np.abs(self._min0) + 1.01).clip(0, None)
        return (X_norm / self._max).clip(0, 1)


def preprocess_and_load_data(data_config=None):
    """
    Master script for normalizing and pre-processing openML datasets
    """
    if data_config is None:

        # Get the current directory (should be where your notebook is)
        current_notebook_path = os.getcwd()
        # Navigate up to the parent directory (your_project_root)
        parent_directory = os.path.dirname(current_notebook_path)
        # Now, construct the path to the configuration file
        config_path = os.path.join(parent_directory, "configs", "config.yml")

        with open(config_path, "r") as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        config = config["data_config"]
    else:
        config = data_config

    # Load the dataset
    X_train, X_val, X_test, y_train, y_val, y_test = pick_dataset_openml(
        config["dataset"], config["target"], config["random_state"]
    )

    # Initialize the scalers
    ln = Norm2Scaler()
    scaler_y = (
        StandardScaler() if config["task_type"] == "regression" else LabelEncoder()
    )

    # Normalize the features
    X_train_norm = pd.DataFrame(
        ln.fit_transform(X_train.to_numpy()), columns=X_train.columns
    )
    X_val_norm = pd.DataFrame(ln.transform(X_val.to_numpy()), columns=X_val.columns)
    X_test_norm = pd.DataFrame(ln.transform(X_test.to_numpy()), columns=X_test.columns)

    # Calculate the percentage of NAs in each column
    percent_nas = X_train_norm.isna().mean()
    # Create a mask for columns with less than 15% NAs
    columns_to_keep = percent_nas < config["na_threshold"]

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
    if config["task_type"] == "regression":
        y_train_dense = to_dense_array(y_train)
        y_val_dense = to_dense_array(y_val)
        y_test_dense = to_dense_array(y_test)

        y_train_enc = (
            scaler_y.fit_transform(y_train_dense.reshape(-1, 1))
            .flatten()
            .astype("float32")
        )
        y_val_enc = (
            scaler_y.transform(y_val_dense.reshape(-1, 1)).flatten().astype("float32")
        )
        y_test_enc = (
            scaler_y.transform(y_test_dense.reshape(-1, 1)).flatten().astype("float32")
        )
    else:
        y_train_enc = scaler_y.fit_transform(to_dense_array(y_train))
        y_val_enc = scaler_y.transform(to_dense_array(y_val))
        y_test_enc = scaler_y.transform(to_dense_array(y_test))

    return (
        X_train_norm,
        X_val_norm,
        X_test_norm,
        y_train_enc,
        y_val_enc,
        y_test_enc,
        scaler_y,
    )

def create_dataloader(dataset, phase, batch_size):   
    # Create dataloaders
    shuffle = (phase == 'train')
    setloader = DataLoader(dataset, 
                           batch_size=batch_size, 
                           shuffle=shuffle, 
                           drop_last=False, 
                           num_workers=12, 
                           pin_memory=True)
    return setloader