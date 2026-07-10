import copy

import numpy as np

def predict(x,w,b):
    m=x.shape[0]
    f_pred=np.zeros(m)
    for i in range(m):
        f_pred[i]=np.dot(x[i],w)+b
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
        if i > 0 and abs(cost_history[-1] - cost_history[-2]) < 1e-4:
            print(f"Converged at iteration {i}")
            break
    return w,b,cost_history,

def predict_gdp(mean, std, w, b):
    imports = float(input("Enter imports (Billion USD): "))*1000000000
    exports = float(input("Enter exports (Billion USD): "))*1000000000
    population = float(input("Enter population: "))*1000000
    trade_balance = exports - imports
    household_consumption= float(input("Enter household consumption (Billion USD): "))*1000000000
    government_expidenture = float(input("Enter government expidenture (Billion USD): "))*1000000000
    gross_capital_formation = float(input("Enter gross capital formation (Billion USD): "))*1000000000

    x = np.array([imports,exports,population,trade_balance,household_consumption,government_expidenture,gross_capital_formation])
    x = (x - mean) / std
    prediction = predict(x.reshape(1, -1), w, b)
    print(f"\nPredicted GDP: {prediction[0]:.2f} Billion USD")