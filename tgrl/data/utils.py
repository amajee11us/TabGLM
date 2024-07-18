import os
import openml
import yaml
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler, LabelEncoder
from torch.utils.data import DataLoader

from tgrl.utils import set_seed
from tgrl.data.datasets import *

dataset_factory = {
    # TabLLM benchmark datasets
    "bank"         : get_bank_dataset,
    "blood"        : get_blood_dataset,
    "calhousing"   : get_calhousing_dataset,
    "car"          : get_car_dataset,
    "creditg"      : get_creditg_dataset,
    "diabetes"     : get_diabetes_dataset,
    "heart"        : get_heart_dataset,
    "income"       : get_income_dataset,
    "jungle"       : get_jungle_dataset,
    # OpenML benchmark datasets
    "pc3"          : get_openml_dataset,
    "kr_vs_kp"     : get_openml_dataset,
    "mfeat_fourier": get_openml_dataset,
    "coil2000"     : get_openml_dataset,
    "texture"      : get_openml_dataset
}

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
        raise Exception(
            "Invalid data configuration found ! Please provide a valid config file."
        )
    else:
        config = data_config

    # Set dataset seed - currently common
    set_seed(data_config["random_state"])

    # Load the dataset
    assert config["dataset"] in  list(dataset_factory.keys())

    X_train, X_val, X_test, y_train, y_val, y_test = dataset_factory[config["dataset"]](
        config["dataset"], config["target"], config["random_state"]
    )

    # # Initialize the scalers
    ln = Norm2Scaler()
    scaler_y = (
        StandardScaler() if config["task_type"] == "regression" else LabelEncoder()
    )

    # # Normalize the features
    # X_train_norm = pd.DataFrame(
    #     ln.fit_transform(X_train.to_numpy()), columns=X_train.columns
    # )
    # X_val_norm = pd.DataFrame(ln.transform(X_val.to_numpy()), columns=X_val.columns)
    # X_test_norm = pd.DataFrame(ln.transform(X_test.to_numpy()), columns=X_test.columns)

    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_columns = X_train.select_dtypes(include=['number']).columns.tolist()

    print(f"Number of Categorical columns: {len(categorical_columns)}")
    print(f"Number of Numerical columns: {len(numerical_columns)}")


    # Remove any NaN values for numerical data replace with mean 
    X_train[numerical_columns] = X_train[numerical_columns].apply(lambda x: x.fillna(x.mean()), axis=0)
    X_val[numerical_columns] = X_val[numerical_columns].apply(lambda x: x.fillna(x.mean()), axis=0)
    X_test[numerical_columns] = X_test[numerical_columns].apply(lambda x: x.fillna(x.mean()), axis=0)
    
    # Function to drop rows with NaNs in categorical columns in both X and y
    def drop_na_categorical(X, y, categorical_columns):
        na_indices = X[categorical_columns].isna().any(axis=1)
        X_clean = X[~na_indices]
        y_clean = y[~na_indices]
        return X_clean, y_clean

    # Drop rows with NaNs in categorical columns for train, validation, and test sets
    X_train, y_train = drop_na_categorical(X_train, y_train, categorical_columns)
    X_val, y_val = drop_na_categorical(X_val, y_val, categorical_columns)
    X_test, y_test = drop_na_categorical(X_test, y_test, categorical_columns)

    # Extract the numerical columns and normalize them using only X_train
    scaler = MinMaxScaler()
    normalized_numerical_X_train = scaler.fit_transform(X_train[numerical_columns])
    normalized_numerical_X_val = scaler.transform(X_val[numerical_columns])
    normalized_numerical_X_test = scaler.transform(X_test[numerical_columns])

    normalized_numerical_df_train = pd.DataFrame(
        normalized_numerical_X_train, columns=numerical_columns)
    normalized_numerical_df_val = pd.DataFrame(
        normalized_numerical_X_val, columns=numerical_columns)
    normalized_numerical_df_test = pd.DataFrame(
        normalized_numerical_X_test, columns=numerical_columns)

    # Extract the categorical columns and perform one-hot encoding using only X_train
    encoder = OneHotEncoder(sparse_output=False)
    encoded_categorical_X_train = encoder.fit_transform(X_train[categorical_columns])
    encoded_categorical_X_val = encoder.transform(X_val[categorical_columns])
    encoded_categorical_X_test = encoder.transform(X_test[categorical_columns])

    encoded_categorical_df_train = pd.DataFrame(
        encoded_categorical_X_train, columns=encoder.get_feature_names_out(categorical_columns))
    encoded_categorical_df_val = pd.DataFrame(
        encoded_categorical_X_val, columns=encoder.get_feature_names_out(categorical_columns))
    encoded_categorical_df_test = pd.DataFrame(
        encoded_categorical_X_test, columns=encoder.get_feature_names_out(categorical_columns))

    # Concatenate the normalized numerical data and the encoded categorical data for X_train, X_val, and X_test
    X_train_norm = pd.concat([normalized_numerical_df_train, encoded_categorical_df_train], axis=1)
    X_val_norm = pd.concat([normalized_numerical_df_val, encoded_categorical_df_val], axis=1)
    X_test_norm = pd.concat([normalized_numerical_df_test, encoded_categorical_df_test], axis=1)

    # # Calculate the percentage of NAs in each column
    # percent_nas = X_train_norm.isna().mean()
    # # Create a mask for columns with less than 15% NAs
    # columns_to_keep = percent_nas < config["na_threshold"]

    # # Use the mask to select columns with less than 15% NAs in all sets
    # X_train_norm = X_train_norm.loc[:, columns_to_keep]
    # X_val_norm = X_val_norm.loc[:, columns_to_keep]
    # X_test_norm = X_test_norm.loc[:, columns_to_keep]

    # # Remove rows with any NAs from X and corresponding rows from y
    # rows_to_keep_train = ~X_train_norm.isna().any(axis=1)
    # X_train_norm = X_train_norm.loc[rows_to_keep_train]
    # y_train = y_train[rows_to_keep_train.to_numpy()]  # Apply the same mask to y_train

    # rows_to_keep_val = ~X_val_norm.isna().any(axis=1)
    # X_val_norm = X_val_norm.loc[rows_to_keep_val]
    # y_val = y_val[rows_to_keep_val.to_numpy()]  # Apply the same mask to y_val

    # rows_to_keep_test = ~X_test_norm.isna().any(axis=1)
    # X_test_norm = X_test_norm.loc[rows_to_keep_test]
    # y_test = y_test[rows_to_keep_test.to_numpy()]  # Apply the same mask to y_test

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
    shuffle = phase == "train"
    setloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=12,
        pin_memory=True,
    )
    return setloader
