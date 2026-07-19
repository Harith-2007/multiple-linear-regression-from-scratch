import pandas as pd
import numpy as np

def load_data(file_name):
    data = pd.read_csv(file_name)
    return data

def split_randomize_features_targets(data, target_column):
    data = data.copy()
    data["trade_balance_usd"] = (data["exports_usd"] - data["imports_usd"])
    x = data.drop(columns=["country_name", "country_code", target_column])
    feature_names = list(x.columns)
    y = data[target_column] / 1000000000
    x = x.to_numpy()
    y = y.to_numpy()
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(x))
    x = x[indices]
    y = y[indices]
    split_index = int(0.8 * len(x))
    x_train = x[:split_index]
    x_test = x[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    return x_train,x_test,y_train,y_test,feature_names

def z_score_normalization(x):
    mean = np.mean(x,axis=0)
    std = np.std(x,axis=0)
    x_norm = (x-mean)/std
    return x_norm,mean,std

