import copy

import numpy as np

def predict(x,w,b):
    f_pred=np.dot(x,w)+b
    return f_pred


def compute_cost(x,y,w,b):
    m=x.shape[0]
    cost=0
    for i in range(m):
        cost+=((np.dot(x[i],w)+b)-y[i])**2
    cost=cost/(2*m)
    return cost

def compute_gradient(x,y,w,b):
    m,n=x.shape
    dj_dw=np.zeros(n)
    dj_db=0
    for i in range(m):
        err=(np.dot(x[i],w)+b)-y[i]
        for j in range(n):
            dj_dw[j]+=err*x[i,j]
        dj_db+=err
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    return dj_dw,dj_db

def compute_gradient_descent(x,y,initial_w,initial_b,learning_rate,iterations):
    w=copy.deepcopy(initial_w)
    b=copy.deepcopy(initial_b)
    cost_history = []


    for i in range(iterations):

        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w=w-learning_rate*dj_dw
        b=b-learning_rate*dj_db

        cost_history.append(compute_cost(x,y,w,b))
        if i%(iterations//10)==0:
            print(f"iteration {i:5d}, cost: {cost_history[-1]:6.4e}")
    return w,b,cost_history
