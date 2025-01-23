import os
import glob
import torch
import umap
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Semilla para reproducibilidad
np.random.seed(42)

def create_UMAP_for_UNI_features(dataset: str):

    pth_features_tensors = glob.glob(os.path.join('UNI', dataset, '*.pt'))

    # Arrays para almacenar los features y etiquetas
    all_features = []
    labels = []

    # Cargar features y asignar etiquetas
    for idx, path in enumerate(pth_features_tensors):
        features = torch.load(path).cpu().numpy()
        all_features.append(features)
        # Etiquetas: 0 para train, 1 para val, 2 para test
        labels.extend([idx] * features.shape[0])

    # Concatenar todos los features en un solo array
    all_features = np.vstack(all_features)
    labels = np.array(labels)

    # Crear el modelo UMAP para reducción a 3D
    # Crear el modelo UMAP para reducción a 2D
    reducer = umap.UMAP(n_components=2, random_state=42)

    # Reducir los datos a 2D
    reduced_features = reducer.fit_transform(all_features)

    # Colores para cada etiqueta
    colors = ['blue', 'green', 'red']  # Train, Val, Test

    # Visualización 2D
    fig, ax = plt.subplots(figsize=(10, 7))

    # Graficar puntos con diferentes colores según la etiqueta
    for label, color in enumerate(colors):
        mask = labels == label
        ax.scatter(
            reduced_features[mask, 0], 
            reduced_features[mask, 1], 
            c=color, label=['Train', 'Val', 'Test'][label], alpha=0.6, s=10)

    # Configuración del gráfico
    ax.set_title("UMAP - Reducción a 2D")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('UNI', 'UMAPs', f'{dataset}.png'), dpi=300)





if __name__ == '__main__':
    datasets = [ '10xgenomic_mouse_brain_sagittal_posterior', 'abalo_human_squamous_cell_carcinoma', 'erickson_human_prostate_cancer_p1',
                'mirzazadeh_human_small_intestine', 'mirzazadeh_mouse_bone', 'mirzazadeh_mouse_brain', 'vicari_human_striatium',
                'vicari_mouse_brain', 'villacampa_lung_organoid', 'villacampa_mouse_brain']
    for dataset in datasets:
        create_UMAP_for_UNI_features(dataset)