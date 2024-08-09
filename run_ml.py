import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

from tgrl.data import preprocess_and_load_data
from tgrl.utils import get_config, store_metrics_as_csv, set_seed, log_metrics
import argparse
import wandb

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingRegressor, GradientBoostingClassifier, HistGradientBoostingClassifier, HistGradientBoostingClassifier
from catboost import CatBoostClassifier, CatBoostRegressor
import xgboost as xgb

from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error, roc_auc_score


def encode_df(df, numerical_columns, categorical_columns, categorical_encoder):
    raw_data_numerical = df[numerical_columns]
    
    # Encode Categorical columns using the provided LabelEncoder dictionary
    encoded_raw_data_categorical = df[categorical_columns].apply(
        lambda col: categorical_encoder[col.name].transform(col)
    )
    
    # Convert encoded data to DataFrame
    encoded_raw_data_categorical_df = pd.DataFrame(
        encoded_raw_data_categorical,
        columns=categorical_columns,
    )
    
    df = pd.concat([raw_data_numerical, encoded_raw_data_categorical_df], axis=1)
    
    return df
    

def main(config_paths):
    for config_path in config_paths:
        data_config, fit_config = get_config(config_path)
        random_states = data_config["random_states"]  # Assumes both data_config and fit_config have the same random_states
        wandb_project_name = "TGRL_ML" #fit_config.get("project_name", None)

        for seed in random_states:            
            # Set the random state in data_config and fit_config
            data_config["random_state"] = seed
            fit_config["random_state"] = seed
            
            (
                X_train_norm,
                X_val_norm,
                X_test_norm,
                y_train_enc,
                y_val_enc,
                y_test_enc,
                scaler_y,
                cat_encoder,
            ) = preprocess_and_load_data(data_config)

            # Extract Numerical columns and separate
            numerical_columns = X_train_norm.select_dtypes(include=["number"]).columns.tolist()
            
            # Extract Categorical columns and Encode
            categorical_columns = X_train_norm.select_dtypes(include=["object", "category"]).columns.tolist()

            if cat_encoder: 
                X_train_norm = encode_df(X_train_norm, numerical_columns, categorical_columns, cat_encoder)
                X_val_norm = encode_df(X_val_norm, numerical_columns, categorical_columns, cat_encoder)
                X_test_norm = encode_df(X_test_norm, numerical_columns, categorical_columns, cat_encoder)

#            def clean_feature_name(name):
 #               # Replace invalid characters with an underscore or remove them
  #              return str(name).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_')
   #         
    #        def clean_feature_names(feature_names):
     #           # Apply the cleaning function to all feature names
      #          return [clean_feature_name(name) for name in feature_names]
#
 #           # Clean feature names
  #          cleaned_feature_names_train = clean_feature_names(X_train_norm.columns)
   #         cleaned_feature_names_val = clean_feature_names(X_val_norm.columns)
    #        cleaned_feature_names_test = clean_feature_names(X_test_norm.columns)
     #       
      #      # Assign cleaned feature names back to the DataFrames
       #     X_train_norm.columns = cleaned_feature_names_train
        #    X_val_norm.columns = cleaned_feature_names_val
         #   X_test_norm.columns = cleaned_feature_names_test

            models = [
                 {"model": GradientBoostingClassifier, "name": "GradientBoostingClassifier"},
                 #{"model": ExtraTreesClassifier, "name": "ExtraTreesClassifier"},
                {"model": LogisticRegression, "name": "LogisticRegression"},
                {"model": RandomForestClassifier, "name": "RandomForestClassifier"},
                 {"model": CatBoostClassifier, "name": "CatBoostClassifier"},
                 {"model": xgb.XGBClassifier, "name": "XGBClassifier"},
                 #{"model": LGBMClassifier, "name": "LGBMClassifier"},
            ]

            # Loop through each model
            for model_config in models:
                model_class = model_config["model"]
                model_name = model_config["name"]
                model = model_class(random_state=seed)

                wandb.init(project=wandb_project_name)
                wandb.config.random_state = seed

                # Log model name and parameters
                wandb.config.model_name = model_name
                wandb.config.dataset = data_config["dataset"]
    
                # Train the model
                model.fit(X_train_norm, y_train_enc)

                # Predict outputs
                y_train_pred = model.predict(X_train_norm)
                y_val_pred = model.predict(X_val_norm)
                y_test_pred = model.predict(X_test_norm)

                # Predict outputs
                y_train_proba = model.predict_proba(X_train_norm)[:, 1]
                y_val_proba = model.predict_proba(X_val_norm)[:, 1]
                y_test_proba = model.predict_proba(X_test_norm)[:, 1]
                
                print(f"Finished processing {config_path} with random seed {seed}")
    
                print("Aggregating Metrics ...")
                log_metrics(data_config['task_type'], y_train_enc, y_train_pred, y_train_proba, loss=None, phase="train")
                log_metrics(data_config['task_type'], y_val_enc, y_val_pred, y_val_proba, loss=None, phase="val")
                log_metrics(data_config['task_type'], y_test_enc, y_test_pred, y_test_proba, loss=None, phase="test")
                print("Done")

                wandb.finish()


if __name__ == "__main__":
    # Default list of YAML filenames
    default_yaml_names = ["config_bank.yml", "config_blood.yml"]

    # Create the parser
    parser = argparse.ArgumentParser(
        description="Process some YAML configuration files."
    )
    # Add an argument, making it optional with nargs='*'
    parser.add_argument(
        "yaml_names", type=str, nargs="*", help="A list of YAML file names (optional)"
    )
    # Parse the arguments
    args = parser.parse_args()

    # Use command line arguments if provided, otherwise use the default list
    yaml_names_to_process = args.yaml_names if args.yaml_names else default_yaml_names
    # Get the current directory (should be where your notebook is)
    current_path = os.getcwd()
    # Create a list of full paths
    full_paths = [
        os.path.join(current_path, "configs", yaml_name)
        for yaml_name in yaml_names_to_process
    ]

    # Pass the list of YAML filenames to main function
    main(full_paths)
