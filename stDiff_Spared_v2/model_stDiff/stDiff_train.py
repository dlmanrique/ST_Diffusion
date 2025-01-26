import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from ray.air import session
import os
from .stDiff_scheduler import NoiseScheduler
from utils import *
from metrics import get_metrics



#Seed
seed = 1202
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def train_stDiff(model,
                data,
                args,
                gene_autoencoder,
                image_encoder,
                save_path,
                wandb_logger,
                device=torch.device('cuda'),
                is_tqdm: bool = True,
                is_tune: bool = False,
                exp_name = ""
                ):
    #mask = None 
    """

    Args:
        lr (float): learning rate 
        pred_type (str, optional): noise or x_0. Defaults to 'noise'.
        diffusion_step (int, optional): timestep. Defaults to 1000.
        device (_type_, optional): Defaults to torch.device('cuda:1').
        is_tqdm (bool, optional): tqdm. Defaults to True.
        is_tune (bool, optional):  ray tune. Defaults to False.

    Raises:
        NotImplementedError: _description_
    """
    noise_scheduler = NoiseScheduler(
        num_timesteps=args.train_diffusion_steps,
        beta_schedule='cosine'
    )

    #Define Loss function
    criterion = nn.MSELoss(reduction='mean')
    
    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    
    # TQDM configurations
    if is_tqdm:
        t_epoch = tqdm(range(args.num_epochs), ncols=100)
    else:
        t_epoch = range(args.num_epochs)


    model.to(device)
    model.train()
    min_mse = np.inf
    best_mse = 0
    best_pcc = 0

    #Define keys for spot or neighbors
    if args.num_neighs == -1:
        # Spot type
        original_st_data_key = 'spot_expression'
        encoded_st_data_key = 'encoded_spot_exp'
    else:
        # Neighbors matrix type
        original_st_data_key = 'exp_matrix'
        encoded_st_data_key = 'encoded_exp_matrix'
    
    for epoch in t_epoch:
        epoch_loss = 0.
        for i, (batch_data) in enumerate(data.train_dataloader()): 
            # En este caso x son los vectores de expresion encodeados u originales
            # x_cond es el vector de features de parches

            # Tomamos los datos encodeados si se quiere, sino pues tomamos los datos de st crudos
            if args.gene_autoencoder:
                # Como tengo gene autoencoder, tomo como 'x' los vectores de st encoded
                x, x_cond =  batch_data[encoded_st_data_key], batch_data['patches']
            else:
                # Como no tengo gene autoencoder, la 'x' son los datos st crudos
                x, x_cond =  batch_data[original_st_data_key], batch_data['patches']
            
            x, x_cond = x.float().to(device), x_cond.float().to(device)

            noise = torch.randn(x.shape).to(device)
            
            timesteps = torch.randint(1, args.train_diffusion_steps, (x.shape[0],)).long()

            x_t = noise_scheduler.add_noise(x,
                                            noise,
                                            timesteps=timesteps.cpu())

            # Datos de expresion y les sumo ruido en todas las posiciones
            x_noisy = x_t

            # Como condicion tengo de input el vector de features de patches
            cond = [x_cond]

            pred = model(x_noisy, t=timesteps.to(device), y=cond) 
            
            #Always compute the loss just in the central spot
            if len(x.shape) == 3:
                # Matrix model
                # Create mask to extract just the info of the central spot
                loss_mask = torch.zeros(pred.shape, dtype=torch.bool)
                loss_mask[:,0, :] = True
                loss = criterion(noise[loss_mask], pred[loss_mask])

            else:
                # Spot to spot model
                loss = criterion(noise, pred)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # type: ignore
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        if args.scheduler:
            scheduler.step()  # Update the learning rate
        
        epoch_loss = epoch_loss / (i + 1)  # type: ignore
        wandb_logger.log({"Loss": epoch_loss})
        #loss_visualization.append(epoch_loss)
        if is_tqdm:
            current_lr = scheduler.get_last_lr()[0]# Get the current learning rate
            #current_lr = optimizer.param_groups[0]['lr']
            t_epoch.set_postfix_str(f'noise loss:{epoch_loss:.5f} lr:{current_lr:.6f}')  # type: ignore
        if is_tune:
            session.report({'loss': epoch_loss})
        
        # compare MSE metrics and save best model
        #FIXME: cambiar esto por el 10 original
        if epoch % (args.num_epochs//2) == 0 and epoch != 0:
            model.eval()
            with torch.no_grad():

                metrics_dict_val, imputation_data_val = inference_function(dataloader=valid_dataloader,
                                            gt_data=st_data_val, 
                                            model=model,
                                            max_norm = max_norm[1],
                                            min_norm = min_norm[1],
                                            avg_tensor = avg_tensor,
                                            diffusion_step=args.diffusion_steps_train,
                                            device=device,
                                            args=args
                                            )

                adata_valid.layers["diff_pred"] = imputation_data_val
                #log_pred_image_extreme_completion(adata_valid, args, epoch)
                
            
                
                

            if metrics_dict_val["MSE"] < min_mse:
                min_mse = metrics_dict_val["MSE"]
                best_mse = metrics_dict_val["MSE"]
                best_pcc = metrics_dict_val["PCC-Gene"]
                torch.save(model.state_dict(), os.path.join("Experiments", exp_name, save_path))
            #save_metrics_to_csv(args.metrics_path, args.dataset, "valid", metrics_dict)
            wandb_logger.log({"MSE_val": metrics_dict_val["MSE"], "PCC_val": metrics_dict_val["PCC-Gene"]})

            model.train()

    # Save the best MSE and best PCC on the validation set
    wandb_logger.log({"best_MSE_val":best_mse, "best_PCC_val": best_pcc})

