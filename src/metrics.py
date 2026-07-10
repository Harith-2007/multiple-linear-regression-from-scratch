import numpy as np
import math
from model import predict

def compute_MSE(y,y_pred):
    m=len(y)
    MSE=0
    for i in range(m):
        MSE=MSE+((y_pred[i]-y[i])**2)
    MSE=MSE/m
    return MSE

def compute_RMSE(MSE):
    return math.sqrt(MSE)

def compute_MAE(y, y_pred):
    m = y.shape[0]
    mae = 0
    for i in range(m):
        mae += abs(y_pred[i] - y[i])
    mae = mae / m
    return mae

def compute_R2(y, y_pred):
    m = y.shape[0]
    mean_y = 0
    for i in range(m):
        mean_y += y[i]
    mean_y = mean_y / m
    ssr = 0
    sst = 0
    for i in range(m):
        ssr += (y[i] - y_pred[i]) ** 2
        sst += (y[i] - mean_y) ** 2
    r2 = 1 - (ssr / sst)
    return r2
