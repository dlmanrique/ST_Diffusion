import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

input_file = "/home/dvegaa/ST_Diffusion/stDiff_Spared/csv/metodos_pred.csv"

color = ["#1a8899", "#4cb6b8", "#a5e1e3"]

df_pred = pd.read_csv(input_file, sep=',', decimal='.', header=None)
methods = df_pred.iloc[0].tolist()   # First row: Method names
values_1 = df_pred.iloc[1].astype(float).tolist()  # Second row: First metric
values_2 = df_pred.iloc[2].astype(float).tolist()  # Third row: Second metric

# Create figure and primary axis
fig, ax1 = plt.subplots(figsize=(8, 5))

# First y-axis (left)
ax1.set_xlabel("", fontsize=12)
ax1.set_ylabel("Number of datasets improved", color="black", fontsize=12)
ax1.plot(methods, values_1, marker="", linestyle="-", color=color[0], label="")
ax1.tick_params(axis="y", labelcolor="black")
ax1.set_ylim(0,26)


x = np.arange(len(methods))
ax1.fill_between(x, values_1, np.full_like(x, 26), color=color[0], alpha=0.6, label="SpaCKLE")
ax1.fill_between(x, values_1, np.zeros_like(x), color=color[1], alpha=0.6, label="LGDiST")
ax1.set_xlim(0,5)
ax1.legend(title="Training data\ncompleted with:", loc="upper left", fontsize=9)

# Create second y-axis (right)
ax2 = ax1.twinx()
ax2.set_ylabel("Average PCC improvement when using LGDiST", color="black", fontsize=12)
ax2.plot(methods, values_2, marker="o", linestyle="--", color="black", label="")
ax2.tick_params(axis="y", labelcolor="black")
ax2.set_ylim(0,0.12)

# Format right y-axis labels to show percentage
ax2.set_yticklabels([f"{tick*100:.0f}%" for tick in ax2.get_yticks()])

# X-axis labels
ax1.set_xticks(x)
ax1.set_xticklabels(methods)

# Add values on top of points
for i, (v1, v2) in enumerate(zip(values_1, values_2)):

    ha_alignment = "center"
    if i == 0:  # Primer punto (SEPAL)
        ha_alignment = "left"
    elif i == len(x) - 1:  # Último punto (BLEEP)
        ha_alignment = "right"

    ax2.text(x[i], v2 + 0.004, f"{v2*100:.1f}%", ha=ha_alignment, va="center", fontsize=10, color="black")


# Title and grid
plt.title("Gene expression prediction trained on LGDiST vs SpaCKLE", fontsize=14)
ax1.grid(False)

# Show plot
plt.savefig("/home/dvegaa/ST_Diffusion/stDiff_Spared/visualization_results/miccai_plots/pred_plot.svg", bbox_inches='tight')