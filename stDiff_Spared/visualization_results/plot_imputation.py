import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

input_file = "/home/dvegaa/ST_Diffusion/stDiff_Spared/csv/resultados_imputacion_all_new.csv"
#output_file = "/home/dvegaa/ST_Diffusion/stDiff_Spared/resultados_imputacion_all.csv"

#df = pd.read_csv(input_file, sep=';', decimal=',')  # Leer el CSV con separador ';' y decimales con ','
#df.to_csv(output_file, sep=',', decimal='.', index=False)  # Guardar con separador ',' y decimales con '.'

df_new = pd.read_csv(input_file, sep=',', decimal='.')
imputation_dict = df_new.to_dict(orient='list')

violin_df = df_new[["Dataset", 'stDiff*', 'SpaCKLE', 'LGDist']]  
comparison_df = df_new[["Dataset", "SpaCKLE vs LGDist", "stDiff vs LGDist"]] 

# Graficar los datos en un violin plot
min_value = violin_df.select_dtypes(include=['number']).min().min()
max_value = violin_df.select_dtypes(include=['number']).max().max()

import numpy as np
color = ["#1a8899", "#4cb6b8", "#a5e1e3"]

plt.figure(figsize=(9, 8))
sns.violinplot(data=violin_df, palette=color, cut=False)
sns.stripplot(data=violin_df, color='black', alpha=0.7) 
plt.title("MSE for all imputation methods on 26 SpaRED datasets", fontsize=20, pad=45)
plt.xlabel("")
plt.ylabel("MSE per dataset", fontsize=18)
plt.ylim(min_value-0.2, max_value+0.5)
plt.xticks(fontsize=18)
plt.yticks(fontsize=17) 
sns.despine(top=True, right=True)
plt.savefig("/home/dvegaa/ST_Diffusion/stDiff_Spared/visualization_results/miccai_plots/imputation_plot_new.pdf", bbox_inches='tight')
plt.close()

"""

plt.figure(figsize=(12, 6))
sns.catplot(data=violin_df, kind="bar", palette=color)
plt.title("Average MSE for all imputations methods on 26 SpaRED dataset")
plt.xlabel("")
plt.ylabel("Average MSE")
plt.savefig("visualizations/imputation_plot_2.png")


violin_df.columns = violin_df.columns.str.strip()
violin_df = violin_df.melt(id_vars=['Dataset'], var_name='Comparison', value_name='Difference')

order = violin_df["Comparison"].unique()
mean_diff = {comp: violin_df[violin_df["Comparison"] == comp]["Difference"].mean() for comp in order}

plt.figure(figsize=(20, 10))
ax = sns.barplot(data=violin_df, x='Dataset', y='Difference', hue='Comparison', palette=color)
plt.xticks(rotation=90)
plt.axhline(0, color='black', linewidth=1)


# Dibujar líneas para los promedios
for idx, comp in enumerate(order):  # Usa el mismo orden que en la gráfica
    avg_value = mean_diff[comp]
    plt.axhline(avg_value, color=color[idx], linestyle='-.', linewidth=3, label=f"Avg {comp}: {avg_value:.2f}%")


plt.title("MSE difference between methods", fontsize=20)
plt.xlabel("Dataset", fontsize=16)
plt.ylabel("MSE", fontsize=16)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig("visualizations/plot_difference_2.png")


###################################################### METHODS ##################################################################################

comparison_df.columns = comparison_df.columns.str.strip()  # Remove leading/trailing spaces
comparison_df = comparison_df.melt(id_vars=['Dataset'], var_name='Comparison', value_name='Difference')

color = ["#3A506B", "#89B0AE"]
ordered_comparisons = comparison_df["Comparison"].unique()  # Orden original en los datos
mean_diff = {comp: comparison_df[comparison_df["Comparison"] == comp]["Difference"].mean() for comp in ordered_comparisons}

plt.figure(figsize=(20, 10))
ax = sns.barplot(data=comparison_df, x='Dataset', y='Difference', hue='Comparison', palette=color)
plt.xticks(rotation=90)
plt.axhline(0, color='black', linewidth=1)


# Dibujar líneas para los promedios
for idx, comp in enumerate(ordered_comparisons):  # Usa el mismo orden que en la gráfica
    avg_value = mean_diff[comp]
    plt.axhline(avg_value, color=color[idx], linestyle='-.', linewidth=3, label=f"Avg {comp}: {avg_value:.2f}%")


plt.title("Percentage difference between methods", fontsize=20)
plt.xlabel("Dataset", fontsize=16)
plt.ylabel("Difference (%)", fontsize=16)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig("visualizations/plot_difference.png")

"""