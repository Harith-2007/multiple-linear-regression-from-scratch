import numpy as np
from preprocessing import load_data,split_randomize_features_targets,z_score_normalization
from model import predict,compute_gradient_descent,predict_gdp
from visualization import plot_cost_comparison,actual_data,pred_vs_actual,predict_data
from metrics import compute_MSE, compute_RMSE, compute_MAE, compute_R2



data = load_data("../data/world_development_index_2025_data.csv")

x_train,x_test,y_train,y_test,x_features = split_randomize_features_targets(data,"gdp_usd")
x_train_norm, mean, std = z_score_normalization(x_train)
x_test_norm = (x_test - mean) / std

print("=" * 60)
print("MULTIPLE LINEAR REGRESSION FROM SCRATCH")
print("=" * 60)
print("\nDataset")
print("-" * 60)
print(f"Training samples: {x_train.shape[0]}")
print(f"Testing samples:  {x_test.shape[0]}")
print(f"Number of features: {x_train.shape[1]}")
print("\nFeatures:")
for i, feature in enumerate(x_features, start=1):
    print(f"{i}. {feature}")
print("\nFirst 5 training rows before normalization:")
print(x_train[:5])
print("\nFirst 5 training rows after normalization:")
print(x_train_norm[:5])
print("\nTraining feature means:")
print(mean)
print("\nTraining feature standard deviations:")
print(std)

temp_w=np.zeros(x_train.shape[1])
temp_b=0

print("\n" + "=" * 60)
print("TRAINING WITH RAW FEATURES")
print("=" * 60)
w_raw,b_raw,cost_his_raw = compute_gradient_descent(x_train,y_train,temp_w,temp_b,1e-26,225700)
print(w_raw,b_raw)

print("\n" + "=" * 60)
print("TRAINING WITH NORMALIZED FEATURES")
print("=" * 60)
w_norm,b_norm,cost_his_norm = compute_gradient_descent(x_train_norm,y_train,temp_w,temp_b,0.1,3200)
print(w_norm,b_norm)

print("\n" + "=" * 60)
print("CONVERGENCE COMPARISON")
print("=" * 60)
print(f"Raw iterations completed:        {len(cost_his_raw):,}")
print(f"Normalized iterations completed: {len(cost_his_norm):,}")
print(f"Raw final cost:                  {cost_his_raw[-1]:,.6f}")
print(f"Normalized final cost:           {cost_his_norm[-1]:,.6f}")
iteration_improvement = len(cost_his_raw) / len(cost_his_norm)
print(f"Normalized training used approximately {iteration_improvement:.1f}x fewer iterations.")

plot_cost_comparison(cost_his_raw,cost_his_norm,save=True)


y_test_prediction=predict(x_test_norm,w_norm,b_norm)
print(y_test_prediction)
mse = compute_MSE(y_test, y_test_prediction)
rmse = compute_RMSE(mse)
mae = compute_MAE(y_test, y_test_prediction)
r2 = compute_R2(y_test, y_test_prediction)

print("\n" + "=" * 60)
print("TEST-SET EVALUATION")
print("=" * 60)

print(f"MSE:  {mse:,.4f}")
print(f"RMSE: {rmse:,.4f} Billion USD")
print(f"MAE:  {mae:,.4f} Billion USD")
print(f"R²:   {r2:.4f}")

print("\nFirst 10 test predictions")
print("-" * 60)
print(f"{'Actual GDP':>15} {'Predicted GDP':>18} {'Error':>15}")

number_to_show = min(10, len(y_test))
for i in range(number_to_show):
    error = y_test_prediction[i] - y_test[i]
    print(f"{y_test[i]:>15.2f} "f"{y_test_prediction[i]:>18.2f} "f"{error:>15.2f}")

actual_data(x_train,y_train,x_features,save=True)
predict_data(x_test,y_test_prediction,x_features,save=True)
pred_vs_actual(y_test,y_test_prediction)

predict_gdp(mean,std,w_norm,b_norm)

