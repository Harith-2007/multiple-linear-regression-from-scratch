import numpy as np
import matplotlib.pyplot as plt

def plot_cost_comparison(raw_cost_his,norm_cost_his,save=False):
    raw_iterations=np.arange(len(raw_cost_his))
    norm_iterations=np.arange(len(norm_cost_his))
    figure,ax = plt.subplots(figsize=(12,3))
    ax.plot(raw_iterations,raw_cost_his,label=f"raw_cost, iterations: {len(raw_cost_his)-1}")
    ax.plot(norm_iterations,norm_cost_his,label=f"normalized_cost, iterations: {len(norm_cost_his)-1}")
    ax.set_title("Gradient Descent Convergence Comparison")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Cost")
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)

    if save:
        plt.savefig("../images/cost_comparison.png", bbox_inches="tight")

    plt.show()

def actual_data(x,y_actual,feature_names,save=False):
    num_features = x.shape[1]
    num_columns = int(np.ceil(num_features/2))
    figure,ax = plt.subplots(2,4,figsize=(5*num_columns,10),sharey=True)
    ax=ax.flatten()
    for i in range(num_features):
        ax[i].scatter(x[:, i], y_actual)
        title=feature_names[i].replace("_usd","").replace("_"," ")
        ax[i].set_title(f"{title} vs GDP",fontsize=10)
        ax[i].set_xlabel(feature_names[i])
        ax[i].grid(True)
    for j in range(num_features,len(ax)):
        ax[j].set_visible(False)
    ax[0].set_ylabel("GDP (billions $)")
    ax[num_columns].set_ylabel("GDP (billions $)")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15,hspace=0.45,left=0.06)
    if save:
        plt.savefig("../images/actual_scatter_plots.png", bbox_inches="tight")

    plt.show()

def predict_data(x,y_predict,feature_names,save=False):
    num_features = x.shape[1]
    num_columns = int(np.ceil(num_features/2))
    figure,ax = plt.subplots(2,4,figsize=(5*num_columns,10),sharey=True)
    ax=ax.flatten()
    for i in range(num_features):
        ax[i].scatter(x[:, i], y_predict)
        title=feature_names[i].replace("_usd","").replace("_"," ")
        ax[i].set_title(f"{title} vs GDP",fontsize=10)
        ax[i].set_xlabel(feature_names[i])
        ax[i].grid(True)
    for j in range(num_features,len(ax)):
        ax[j].set_visible(False)
    ax[0].set_ylabel("GDP (billions $)")
    ax[num_columns].set_ylabel("GDP (billions $)")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15,hspace=0.45,left=0.06)
    if save:
        plt.savefig("../images/predictions_scatter_plots.png", bbox_inches="tight")

    plt.show()

def pred_vs_actual(y_actual,y_predictions,save=False):
    plt.scatter(y_actual,y_predictions)
    plt.title("Actual vs. Predicted GDP")
    plt.xlabel("Actual GDP billions $")
    plt.ylabel("Predicted GDP billions $")
    plt.tight_layout()
    if save:
        plt.savefig("../images/actual_vs_predicted.png", bbox_inches="tight")
    plt.show()