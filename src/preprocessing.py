import pandas as pd
import numpy as np

def load_data(file_name):
    data = pd.read_csv(file_name)
    return data

def split_features_targets(data, target_column):
    data["trade_balance_usd"]=data["exports_usd"]-data["imports_usd"]
    x = data.drop(columns=["country_name","country_code", target_column])
    feature_names = list(x.columns)
    y = data[target_column]
    y = y / 1000000000

    return x.values , y.values,feature_names

def z_score_normalization(x):
    mean = np.mean(x,axis=0)
    std = np.std(x,axis=0)
    x_norm = (x-mean)/std
    return x_norm,mean,std

