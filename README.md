# ST_Diffusion

Para ejecutar el codigo se deben descargar los datos de ejemplo desde este link:

https://drive.google.com/file/d/1wG7OchbLVhFjrhfX49q1tUUwE1n4O3vS/view?usp=drive_link

En el link se encuentra un archivo llamado 'adata.h5ad'. Este archivo se debe colocar en el siguiente folder:  'stDiff_Spared/Example_dataset/'.

Comandos para configurar en environment:
```bash
cd stDiff_Spared
conda create -n st_diff python=3.11.5
pip install -r requirements.txt
```

Comandos para ejecutar el código. En este caso se va a ejecutar el código de la tarea completion extremo. A partir de la información genética de los vecinos, buscamos generar el vector de expresión genética del spot central.

```bash
CUDA_VISIBLE_DEVICES=1 python main_2D.py --debbug_wandb True --vlo True --num_epoch 100 --diffusion_steps 50
```
