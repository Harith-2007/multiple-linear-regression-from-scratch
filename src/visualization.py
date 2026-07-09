import numpy as np
import matplotlib.pyplot as plt

def plot_cost_comparison(raw_cost_his,norm_cost_his,save=False):
    raw_iterations=np.arange(len(raw_cost_his))
    norm_iterations=np.arange(len(norm_cost_his))
    figure,ax = plt.subplots(figsize=(12,3))
    ax.plot(raw_iterations,raw_cost_his,label="raw_cost")
    ax.plot(norm_iterations,norm_cost_his,label="norm_cost")
    ax.set_title("Cost vs Iterations")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Cost")
    ax.legend()
    ax.grid(True)

    if save:
        plt.savefig("../images/cost_comparison.png", bbox_inches="tight")

    plt.show()

def plot_data(x,y,feature_names,save=False):
    figure,ax = plt.subplots(1,4,figsize=(12,3))
    for i in range(x.shape[1]):
        ax[i].scatter(x[:, i], y)
        ax[i].set_title(f"{feature_names[i]} vs GDP")
        ax[i].set_xlabel(feature_names[i])
        ax[i].grid(True)
        ax[i].set_ylabel("GDP (billions $)")
    plt.tight_layout()

    if save:
        plt.savefig("../images/data_scatter_plots.png", bbox_inches="tight")

    plt.show()

def predict_data(x,y,feature_names,save=False):
    num_features=x.shape[1]
    figure, ax = plt.subplots(1, 3, figsize=(5*num_features, 3))
    for i in range(num_features):
        ax[i].scatter(x[:, i], y)
        ax[i].set_title(f"{feature_names[i]} vs GDP")
        ax[i].set_xlabel(feature_names[i])
        ax[i].grid(True)
        ax[i].set_ylabel("GDP (billions $)")
    plt.tight_layout()

    if save:
        plt.savefig("../images/data_scatter_plots.png", bbox_inches="tight")

    plt.show()
