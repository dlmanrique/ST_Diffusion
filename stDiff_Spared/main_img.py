from scipy.interpolate import griddata

gene=adata.var["gene_ids"].unique()[0]
gene_index = adata.var_names.get_loc(gene)

slide=adata.obs["slide_id"].unique()[0]
slide_adata = adata[adata.obs['slide_id'] == slide].copy()
spatial_coords = slide_adata.obsm['spatial']
gene_log1p_data = slide_adata.layers['c_t_log1p'][:, gene_index]

mask = slide_adata.layers["mask"][:, gene_index]
gene_log1p_data[mask]=0

plt.figure(figsize=(8, 8))
plt.scatter(spatial_coords[:, 0], spatial_coords[:, 1], c=gene_log1p_data, cmap='viridis', s=10)

# Crear una cuadrícula de puntos para la imagen
x_min, x_max = spatial_coords[:, 0].min(), spatial_coords[:, 0].max()
y_min, y_max = spatial_coords[:, 1].min(), spatial_coords[:, 1].max()

grid_x, grid_y = np.mgrid[x_min:x_max:500j, y_min:y_max:500j]  # 500x500 pixels

# Interpolación de los valores de expresión en la cuadrícula
grid_z = griddata(spatial_coords, gene_log1p_data, (grid_x, grid_y), method='cubic')

# Crear la imagen
plt.figure(figsize=(8, 8))
plt.imshow(grid_z.T, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='viridis')
plt.savefig("plot_img.png")