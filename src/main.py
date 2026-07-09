import numpy as np
from preprocessing import load_data,split_features_targets,z_score_normalization
from model import compute_cost,predict,compute_gradient,compute_gradient_descent
from visualization import plot_cost_comparison,plot_data
data = load_data("../data/world_development_index_2025.csv")

x,y,x_features = split_features_targets(data,"gdp_usd")
x_norm, mean, std= z_score_normalization(x)

print(f"x shape: {x.shape} ")
print(f"y shape: {y.shape} ")

print(f"first 5 rows of data before normalization:")
print(x[:5])
print(f"first 5 rows of data after normalization:")
print(x_norm[:5])
print(f"feature means : ")
print(mean)
print(f"standard deviations: ")
print(std)

temp_w=np.zeros(x.shape[1])
temp_b=0


w_raw,b_raw,cost_his_raw = compute_gradient_descent(x,y,temp_w,temp_b,1e-28,1000)
print(w_raw,b_raw)

w_norm,b_norm,cost_his_norm = compute_gradient_descent(x_norm,y,temp_w,temp_b,1e-2,1000)
print(w_norm,b_norm)

plot_cost_comparison(cost_his_raw,cost_his_norm,save=True)


plot_data(x_norm,y,x_features,save=True)

