#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

# The CSV table as a multi-line string
data = """graph,model,acc_mean,acc_std,ami_mean,ami_std,ari_mean,ari_std
eta=0,K-Means,0.648,0.016,0.400,0.015,0.375,0.018
eta=0,Spectral,0.183,0.000,0.017,0.000,0.010,0.000
eta=0,Leiden,0.181,0.006,0.008,0.004,0.005,0.002
eta=0,Node2Vec,0.173,0.003,0.004,0.001,0.002,0.001
eta=0,Attri2Vec,0.171,0.006,0.004,0.004,0.002,0.002
eta=0,DynNode2Vec,0.165,0.001,0.005,0.000,0.002,0.000
eta=0,tNodeEmbed,0.171,0.006,0.004,0.004,0.002,0.002
eta=0,DAEGC,0.390,0.086,0.173,0.054,0.128,0.060
eta=0,DMoN,0.179,0.008,0.004,0.002,0.016,0.002
eta=0,TGC,0.682,0.006,0.433,0.009,0.416,0.009
eta=.25,K-Means,0.648,0.016,0.400,0.015,0.375,0.018
eta=.25,Spectral,0.181,0.000,0.006,0.000,0.004,0.000
eta=.25,Leiden,0.180,0.006,0.010,0.002,0.006,0.002
eta=.25,Node2Vec,0.170,0.004,0.002,0.001,0.001,0.001
eta=.25,Attri2Vec,0.168,0.005,0.002,0.002,0.001,0.001
eta=.25,DynNode2Vec,0.167,0.001,0.006,0.001,0.003,0.000
eta=.25,tNodeEmbed,0.168,0.005,0.002,0.002,0.001,0.001
eta=.25,DAEGC,0.375,0.054,0.151,0.017,0.110,0.025
eta=.25,DMoN,0.182,0.011,0.007,0.004,0.018,0.004
eta=.25,TGC,0.680,0.003,0.431,0.004,0.414,0.004
eta=.5,K-Means,0.648,0.016,0.400,0.015,0.375,0.018
eta=.5,Spectral,0.210,0.000,0.025,0.000,0.018,0.000
eta=.5,Leiden,0.204,0.017,0.019,0.008,0.012,0.005
eta=.5,Node2Vec,0.174,0.006,0.007,0.004,0.004,0.002
eta=.5,Attri2Vec,0.175,0.006,0.005,0.004,0.003,0.002
eta=.5,DynNode2Vec,0.176,0.003,0.005,0.001,0.003,0.000
eta=.5,tNodeEmbed,0.175,0.006,0.005,0.004,0.003,0.002
eta=.5,DAEGC,0.466,0.088,0.218,0.050,0.180,0.058
eta=.5,DMoN,0.196,0.010,0.014,0.004,0.026,0.003
eta=.5,TGC,0.681,0.003,0.432,0.005,0.415,0.005
eta=.75,K-Means,0.648,0.016,0.400,0.015,0.375,0.018
eta=.75,Spectral,0.448,0.000,0.152,0.000,0.135,0.000
eta=.75,Leiden,0.379,0.043,0.132,0.016,0.115,0.017
eta=.75,Node2Vec,0.195,0.001,0.023,0.001,0.014,0.000
eta=.75,Attri2Vec,0.199,0.002,0.026,0.001,0.017,0.000
eta=.75,DynNode2Vec,0.177,0.002,0.012,0.002,0.006,0.001
eta=.75,tNodeEmbed,0.199,0.002,0.026,0.001,0.017,0.000
eta=.75,DAEGC,0.628,0.050,0.356,0.040,0.337,0.055
eta=.75,DMoN,0.251,0.019,0.051,0.007,0.062,0.007
eta=.75,TGC,0.681,0.005,0.434,0.006,0.415,0.007
eta=1,K-Means,0.648,0.016,0.400,0.015,0.375,0.018
eta=1,Spectral,1.000,0.000,1.000,0.000,1.000,0.000
eta=1,Leiden,0.849,0.055,0.945,0.022,0.848,0.048
eta=1,Node2Vec,0.216,0.000,0.066,0.000,0.041,0.000
eta=1,Attri2Vec,0.216,0.000,0.066,0.000,0.041,0.000
eta=1,DynNode2Vec,0.213,0.001,0.060,0.002,0.037,0.001
eta=1,tNodeEmbed,0.216,0.000,0.066,0.000,0.041,0.000
eta=1,DAEGC,1.000,0.000,1.000,0.000,1.000,0.000
eta=1,DMoN,0.918,0.005,0.813,0.011,0.815,0.011
eta=1,TGC,0.687,0.004,0.438,0.005,0.421,0.005
"""
# Read data from the string using pandas
df = pd.read_csv(io.StringIO(data))

plt.rcParams.update({
    # "text.usetex": True,
    # "font.family": "sans-serif",
    # "font.sans-serif": "Helvetica",
})

tab10 = {
    "tab:blue": "#1f77b4",
    "tab:orange": "#ff7f0e",
    "tab:green": "#2ca02c",
    "tab:red": "#d62728",
    "tab:purple": "#9467bd",
    "tab:brown": "#8c564b",
    "tab:pink": "#e377c2",
    "tab:gray": "#7f7f7f",
    "tab:olive": "#bcbd22",
    "tab:cyan": "#17becf",
}

# Extract unique eta values and unique models (for ordering)
etas = sorted(df['graph'].unique(), key=lambda x: float(x.split('=')[1]))
models = df['model'].unique()

# Create a figure with subplots for different metrics
fig, axs = plt.subplots(1, 1, figsize=(5.5, 3.5), sharex=True, constrained_layout=True)
axs = [axs]

# Define metrics and corresponding std error columns and titles.
metrics = [("acc_mean", "acc_std", "Accuracy"),]
        #    ("ami_mean", "ami_std", "AMI"),
        #    ("ari_mean", "ari_std", "ARI")]

# Define colors for different eta values
colors = {model: f"tab:{color}" for model, color in
          zip(models, ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"])}


# X positions for each model
import numpy as np
x = np.arange(len(etas))

for ax, (mean_col, std_col, title) in zip(axs, metrics):
    for model in models:
        # Filter data for the given eta value
        df_model = df[df["model"] == model]

        # Ensure the order of models in the same order as in 'models'
        df_model = df_model.set_index("model").reset_index()

        # Invert the order of the DataFrame
        df_model = df_model[::-1]

        label = "                         "
        color = colors[model]
        linestyle = '--'
        marker = 'o'
        if model in ('K-Means', 'Spectral'):
            linestyle = '-'
        if df_model[mean_col].max() < 0.5:
            linestyle = ':'

        # Plot with error bars
        ax.errorbar(x, df_model[mean_col],
                    yerr=df_model[std_col],
                    label=label,
                    color=color,
                    marker=marker,
                    markeredgecolor=color,
                    linestyle=linestyle,
                    markersize=4,
                    markeredgewidth=1,
                    elinewidth=1,
                    capthick=1,
                    capsize=4,
        )

    # ax.set_title(title)w
    ax.set_ylabel(title)
    ax.grid(True)
    ax.legend(title="", bbox_to_anchor=(.99, 1), loc='upper left', fontsize=12,
              handlelength=1.75, handletextpad=0.5, borderpad=0.2, borderaxespad=0.5,
              frameon=False, title_fontsize=8)
    ax.set_xticks(x)
    xticklabels = ["$\\mathcal{G}_{\\eta=%s}$" % eta.split("=")[-1] for eta in etas][::-1]
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")

axs[-1].set_xlabel("")
axs[-1].set_ylabel("")
axs[-1].set_xticklabels("")
axs[-1].set_yticklabels("")

# plt.tight_layout()
plt.savefig("fig-acc.svg")#, bbox_inches='tight')
# plt.show()
