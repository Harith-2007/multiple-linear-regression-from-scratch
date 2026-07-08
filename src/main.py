from preprocessing import load_data,split_features_targets,z_score_normalization

data = load_data("../data/world_development_index_2025.csv")

x,y = split_features_targets(data,"gdp_usd")
x_norm, mean, std= z_score_normalization(x)

print(f"x shape: {x.shape} ")
print(f"y shape: {y.shape} ")

print(f"first rows of data before normalization:")
print(x[:5])
print(f"feature means : ")
print(mean)
print(f"standard deviations: ")
print(std)


