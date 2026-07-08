import pandas as pd
import numpy as np

def load_data(file_name):
    data = pd.read_csv(file_name)
    return data

def split_features_targets(data, target_data):
    x = data.drop(columns=["country_name","country_code",target_data])
    y = data[target_data]
    return x.values , y.values

def z_score_normalization(x):
    mean = np.mean(x,axis=0)
    std = np.std(x,axis=0)
    x_norm = (x-mean)/std
    return x_norm,mean,std

