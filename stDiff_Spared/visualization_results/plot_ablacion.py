import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

input_file = "/home/dvegaa/ST_Diffusion/stDiff_Spared/ablacion.csv"
#output_file = "/home/dvegaa/ST_Diffusion/stDiff_Spared/ablacion.csv"

#df = pd.read_csv(input_file, sep=';', decimal=',')  # Leer el CSV con separador ';' y decimales con ','
#df.to_csv(output_file, sep=',', decimal='.', index=False)  # Guardar con separador ',' y decimales con '.'

df_new = pd.read_csv(input_file, sep=',', decimal='.')
imputation_dict = df_new.to_dict(orient='list')

violin_df = df_new[["Dataset", "W/out latent space" , "W/out context genes" , "W/ context genes"]]  
df = df_new[["W/out latent space" , "W/out context genes" , "W/ context genes"]].iloc[:-1]

breakpoint()

# Graficar los datos en un violin plot
min_value = violin_df.select_dtypes(include=['number']).min().min()
max_value = violin_df.select_dtypes(include=['number']).max().max()

import numpy as np
color = ["#1a8899", "#a5e1e3", "#a5e1e3"]

# Crear la gráfica
g = sns.catplot(data=violin_df, kind="bar", palette=color, height=8, aspect=1.2)

# Ajustar títulos y etiquetas
g.figure.suptitle("Average MSE for all ablation experiments on 6 SpaRED datasets", fontsize=20, y=0.99)
g.set_axis_labels("", "Average MSE", fontsize=18)

g.set_xticklabels(fontsize=18)
g.set_yticklabels(fontsize=17)

# Ajustar márgenes
plt.tight_layout()

# Guardar la figura
plt.savefig("visualizations/ablation_plot.png")
plt.show()
