import numpy as np
from preprocessing import load_data,split_features_targets,z_score_normalization
from model import predict,compute_gradient_descent,predict_gdp
from visualization import plot_cost_comparison,actual_data,pred_vs_actual,predict_data
from metrics import compute_MSE, compute_RMSE, compute_MAE, compute_R2



data = load_data("../data/world_development_index_2025_data.csv")

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


w_raw,b_raw,cost_his_raw = compute_gradient_descent(x,y,temp_w,temp_b,1e-26,200000)
print(w_raw,b_raw)

w_norm,b_norm,cost_his_norm = compute_gradient_descent(x_norm,y,temp_w,temp_b,0.1,2500)
print(w_norm,b_norm)

plot_cost_comparison(cost_his_raw,cost_his_norm,save=True)




y_prediction=predict(x_norm,w_norm,b_norm)
print(y_prediction)

actual_data(x_norm,y,x_features,save=True)
predict_data(x_norm,y_prediction,x_features,save=True)

print("Weights:", w_norm)
print("Bias:", b_norm)

print("\nFirst 10 predictions:")
print(y_prediction[:10])

print("\nFirst 10 actual values:")
print(y[:10])

mse = compute_MSE(y, y_prediction)
rmse = compute_RMSE(mse)
mae = compute_MAE(y, y_prediction)
r2 = compute_R2(y, y_prediction)

print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")


pred_vs_actual(y,y_prediction)


predict_gdp(mean,std,w_norm,b_norm)


