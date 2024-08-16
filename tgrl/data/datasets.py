'''
This is a placeholder for all tabular data loaders
'''
# from tgrl.data.utils import byte_to_string_columns
import os
import openml
import yaml
import numpy as np
import pandas as pd
from scipy.io.arff import loadarff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATASET_ROOT = "datasets"

binary_mapping = {True: 1, False: 0}

'''
Same OpenML dataset can have multiple IDs.
Thus, we map dataset names to correct IDs.
'''
openml_dataset_name_to_id = {
    "pc3": 1050,
    "kr_vs_kp": 3,
    "mfeat_fourier": 971,
    "coil2000": 298,
    "texture": 40499,
    "balance-scale": 11,
    "mfeat-karhunen": 16,
    "mfeat-morphological": 18,
    "mfeat-zernike": 22,
    "cmc": 23,
    "tic-tac-toe": 50,
    "vehicle": 54,
    "eucalyptus": 188,
    "analcatdata_author": 469,
    "MiceProtein": 40966,
    "steel-plates-fault": 40982,
    "dress-sales": 23381
}

def byte_to_string_columns(data):
    # Convert Column Names in bytes to string
    for col, dtype in data.dtypes.items():
        if dtype == object:  # Only process byte object columns.
            data[col] = data[col].apply(lambda x: x.decode("utf-8"))
    return data

def get_openml_dataset(dataset_name, target, random_state):
    # Get dataset by name
    dataset = openml.datasets.get_dataset(
        openml_dataset_name_to_id[dataset_name]
    )
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

def get_blood_dataset(dataset_name, target, random_state):
    columns = {'V1': 'recency', 'V2': 'frequency', 'V3': 'monetary', 'V4': 'time', 'Class': 'label'}
    dataset = pd.DataFrame(loadarff(os.path.join(
        DATASET_ROOT, dataset_name, 'php0iVrYT.arff'))[0])
    dataset = byte_to_string_columns(dataset)
    dataset.rename(columns=columns, inplace=True)
    dataset['label'] = dataset['label'] == '2'
    dataset['label'] = dataset['label'].replace(binary_mapping)

    X, y = dataset.drop(target, axis=1), dataset[target]
    X = X.loc[:, X.nunique() > 1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def get_bank_dataset(dataset_name, target, random_state):
    columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day',
                   'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
    columns = {'V' + str(i + 1): v for i, v in enumerate(columns)}
    dataset = pd.DataFrame(loadarff(
        os.path.join(DATASET_ROOT, dataset_name, 'phpkIxskf.arff'))[0])
    dataset = byte_to_string_columns(dataset)
    dataset.rename(columns=columns, inplace=True)
    dataset.rename(columns={'Class': 'label'}, inplace=True)
    dataset['label'] = dataset['label'] == '2'
    dataset['label'] = dataset['label'].replace(binary_mapping)

    X, y = dataset.drop(target, axis=1), dataset[target]
    X = X.loc[:, X.nunique() > 1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def get_calhousing_dataset(dataset_name, target, random_state):
    dataset = pd.DataFrame(loadarff(os.path.join(DATASET_ROOT, dataset_name, 'houses.arff'))[0])
    dataset = byte_to_string_columns(dataset)
    dataset.rename(columns={'median_house_value': 'label'}, inplace=True)
    # Make binary task by labelling upper half as true
    median_price = dataset['label'].median()
    dataset['label'] = dataset['label'] > median_price
    dataset['label'] = dataset['label'].replace(binary_mapping) 

    X, y = dataset.drop(target, axis=1), dataset[target]
    X = X.loc[:, X.nunique() > 1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def get_calhousing_reg_dataset(dataset_name, target, random_state):
    dataset_name = "calhousing" # No special name to regression dataset
    dataset = pd.DataFrame(loadarff(os.path.join(DATASET_ROOT, dataset_name, 'houses.arff'))[0])
    dataset = byte_to_string_columns(dataset)
    dataset.rename(columns={'median_house_value': 'label'}, inplace=True)
    # Make binary task by labelling upper half as true
    median_price = dataset['label'].median()
    # dataset['label'] = dataset['label'] > median_price
    # dataset['label'] = dataset['label'].replace(binary_mapping) 

    X, y = dataset.drop(target, axis=1), dataset[target]
    X = X.loc[:, X.nunique() > 1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def get_car_dataset(dataset_name, target, random_state):
    columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety_dict', 'label']
    dataset = pd.read_csv(os.path.join(DATASET_ROOT, dataset_name, 'car.data'), names=columns)
    original_size = len(dataset)
    label_dict = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
    dataset['label'] = dataset['label'].replace(label_dict)

    X, y = dataset.drop(target, axis=1), dataset[target]
    X = X.loc[:, X.nunique() > 1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def get_diabetes_dataset(dataset_name, target, random_state):
    dataset = pd.read_csv(os.path.join(DATASET_ROOT, dataset_name, 'diabetes.csv'))
    original_size = len(dataset)
    dataset = dataset.rename(columns={'Outcome': 'label'})

    X, y = dataset.drop(target, axis=1), dataset[target]
    X = X.loc[:, X.nunique() > 1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def get_creditg_dataset(dataset_name, target, random_state):
    dataset = pd.DataFrame(loadarff(os.path.join(DATASET_ROOT, dataset_name, 'dataset_31_credit-g.arff'))[0])
    dataset = byte_to_string_columns(dataset)
    dataset.rename(columns={'class': 'label'}, inplace=True)
    dataset['label'] = dataset['label'] == 'good'
    dataset['label'] = dataset['label'].replace(binary_mapping)

    X, y = dataset.drop(target, axis=1), dataset[target]
    X = X.loc[:, X.nunique() > 1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test    

def get_heart_dataset(dataset_name, target, random_state):
    dataset = pd.read_csv(os.path.join(DATASET_ROOT, dataset_name, 'heart.csv'))
    original_size = len(dataset)
    dataset = dataset.rename(columns={'HeartDisease': 'label'})

    X, y = dataset.drop(target, axis=1), dataset[target]
    X = X.loc[:, X.nunique() > 1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def get_jungle_dataset(dataset_name, target, random_state):
    dataset = pd.DataFrame(loadarff(os.path.join(DATASET_ROOT, dataset_name, 'jungle_chess_2pcs_raw_endgame_complete.arff'))[0])
    dataset = byte_to_string_columns(dataset)
    dataset.rename(columns={'class': 'label'}, inplace=True)
    dataset['label'] = dataset['label'] == 'w'  # Does white win?
    dataset['label'] = dataset['label'].replace(binary_mapping)

    X, y = dataset.drop(target, axis=1), dataset[target]
    X = X.loc[:, X.nunique() > 1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def get_income_dataset(dataset_name, target, random_state):    
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
                'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
                'native_country', 'label']

    def strip_string_columns(df):
        df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).apply(lambda x: x.str.strip())

    dataset_train = pd.read_csv(os.path.join(DATASET_ROOT, dataset_name, 'adult.data'), names=columns, na_values=['?', ' ?'])
    dataset_train = dataset_train.drop(columns=['fnlwgt', 'education_num'])
    original_size = len(dataset_train)
    strip_string_columns(dataset_train)
    dataset_train['label'] = dataset_train['label'] == '>50K'
    dataset_train['label'] = dataset_train['label'].replace(binary_mapping)

    X, y = dataset_train.drop(target, axis=1), dataset_train[target]
    X = X.loc[:, X.nunique() > 1]
    X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=random_state)
    dataset = dataset_train
    assert len(y_train) + len(y_val) == original_size

    dataset_test = pd.read_csv(os.path.join(DATASET_ROOT, dataset_name, 'adult.test'), names=columns, na_values=['?', ' ?'])
    dataset_test = dataset_test.drop(columns=['fnlwgt', 'education_num'])
    strip_string_columns(dataset_test)
    dataset_test['label'] = dataset_test['label'] == '>50K.'
    dataset_test['label'] = dataset_test['label'].replace(binary_mapping)

    X_test, y_test = dataset_test.drop(target, axis=1), dataset_test[target]
    X_test = X_test.loc[:, X_test.nunique() > 1]

    return X_train, X_val, X_test, y_train, y_val, y_test

def load_carte_dataset(data_name, target, random_state):
    """
    Load a dataset from the CARTE benchmark, preprocess it, and split it into training, validation, and test sets.
    """

    datasets_supported = ["jp_anime", "used_cars_24", "wikiliq_beer", "us_presidential", "us_accidents_counts",
    "journal_sjr", "buy_buy_baby", "anime_planet", "used_cars_pakistan", "us_accidents_severity", "babies_r_us",
    "filmtv_movies", "chocolate_bar_ratings", "wine_enthusiasts_ratings", "museums", "ramen_ratings", "nba_draft",
    "wine_vivino_rating", "employee_salaries", "used_cars_saudi_arabia", "videogame_sales", "michelin", "clear_corpus",
    "yelp", "k_drama", "journal_jcr", "beer_ratings", "wine_dot_com_prices", "used_cars_dot_com", "mydramalist", 
    "bikewale", "movies", "whisky", "wina_pl", "wikiliq_spirit",  "employee_remuneration", "wine_enthusiasts_prices", 
    "used_cars_benz_italy", "wine_vivino_price", "coffee_ratings", "zomato", "rotten_tomatoes", "cardekho", "prescription_drugs", 
    "roger_ebert", "bikedekho", "company_employees", "wine_dot_com_ratings", "spotify", "mlds_salaries", "fifa22_players"]

    if data_name not in datasets_supported:
        raise Exception("Dataset not supported !")
    
    # Load the data from the local directory
    data_dir = os.path.join(
        DATASET_ROOT, data_name, "raw.parquet")
    data = pd.read_parquet(data_dir)
    data.fillna(value=np.nan, inplace=True)
    
    # Load the configuration data
    config_data_dir = os.path.join(
        DATASET_ROOT, data_name, "config_data.json")
    with open(config_data_dir, 'r') as filename:
        config_data = json.load(filename)
    
    target = config_data["target_name"]

    # Separate features and target
    X, y = data.drop(target, axis=1), data[target]
    
    # Remove columns with a single unique value
    X = X.loc[:, X.nunique() > 1]
    
    # Split the data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
