import torch
import numpy as np
import os
import torch.nn as nn
from tqdm import tqdm
from ray.air import session
import os
from .stDiff_scheduler import NoiseScheduler
from utils import *
import matplotlib.pyplot as plt
import wandb
import datetime 
from spared.metrics import get_metrics
from datetime import datetime
from visualize_imputation import *

# Get parser and parse arguments
parser = get_main_parser()
args = parser.parse_args()
args_dict = vars(args)

#Seed
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def normal_train_stDiff(model,
                 train_dataloader,
                 valid_dataloader,
                 valid_data,
                 valid_masked_data,
                 mask_valid,
                 mask_extreme_completion,
                 max_norm,
                 min_norm, 
                 avg_tensor,
                 wandb_logger,
                 args,
                 adata_valid,
                 lr: float = 1e-4,
                 num_epoch: int = 1400,
                 diffusion_step: int = 1000,
                 device=torch.device('cuda'),
                 is_tqdm: bool = True,
                 is_tune: bool = False,
                 pred_type: str = "noise",
                 save_path = "ckpt/demo_spared.pt",
                 exp_name = ""):
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
        num_timesteps=diffusion_step,
        beta_schedule='cosine'
    )

    #Define Loss function
    criterion = nn.MSELoss()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True, threshold=1e-4, cooldown=10, min_lr=5e-8)
    
    if is_tqdm:
        t_epoch = tqdm(range(num_epoch), ncols=100)
    else:
        t_epoch = range(num_epoch)

    model.train()
    min_mse = np.inf
    best_mse = 0
    best_pcc = 0
    loss_visualization = []
    #exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(os.path.join('Experiments', exp_name), exist_ok=True)

    for epoch in t_epoch:
        epoch_loss = 0.
        for i, (x, x_cond, mask) in enumerate(train_dataloader): 
            #breakpoint()
            #The mask is a binary array, the 1's are the masked data (0 in the values we want to predict)
            #En LDM es alreves entonces si algo tocaría voltearlo
            x, x_cond = x.float().to(device), x_cond.float().to(device)

            noise = torch.randn(x.shape).to(device)
            #torch.rand_like
            # noise.shape: torch.Size([2048, 33])
            
            timesteps = torch.randint(1, diffusion_step, (x.shape[0],)).long()
            # timesteps.shape: torch.Size([2048])
            x_t = noise_scheduler.add_noise(x,
                                            noise,
                                            timesteps=timesteps.cpu())
            # x_t.shape: torch.Size([2048, 33])
            mask = torch.tensor(mask).to(device)
            # mask.shape: torch.Size([33])
            
            #Entrenamos el modelo para que prediga únicamente los valores masqueados
            if args.masked_loss:  
                #Entrenamos el modelo para que prediga solo los datos masqueados
                x_noisy = x_t * (1 - mask) + x * mask
            else:
                #Entrenamos el modelo para que prediga toda la imagen
                x_noisy = x_t
            
            # x_t en los valores masqueados y siempre x original en los valores no masquados
            # x_noisy.shape: torch.Size([2048, 33])
            cond = [x_cond, mask]
            
            pred = model(x_noisy, t=timesteps.to(device), y=cond) 

            mask_boolean = (1-mask).bool() #1 = False y 0 = True
            mask_boolean = mask_boolean[:,:,0]  
            pred = pred[:,:,0] 
            
            if args.loss_type == "noise":
                noise = noise[:,:,0]
                #noise = denormalize_from_minus_one_to_one(noise, min_norm[0], max_norm[0])
                if args.masked_loss:
                    #calculamos loss solo sobre los datos masqueados
                    loss = criterion(noise[mask_boolean], pred[mask_boolean])
                else:
                    #calculasmos loss sobre todos los datos
                    loss = criterion(noise, pred)
            
            elif args.loss_type == "x_start":
                #x = x*max_norm[0] 
                pred = denormalize_from_minus_one_to_one(pred, min_norm[0], max_norm[0])
                x = x[:,:,0]
                x = denormalize_from_minus_one_to_one(x, min_norm[0], max_norm[0])
                if args.masked_loss:
                    loss = criterion(x[mask_boolean], pred[mask_boolean])
                else:
                    loss = criterion(x, pred)
            
            elif args.loss_type == "x_previous":
                pred = denormalize_from_minus_one_to_one(pred, min_norm[0], max_norm[0])
                x_t_1 = noise_scheduler.q_posterior(x, x_t, timesteps.detach().cpu())
                x_t_1 = x_t_1[:,:,0]
                x_t_1 = denormalize_from_minus_one_to_one(x_t_1, min_norm[0], max_norm[0])
                if args.masked_loss:
                    loss = criterion(x_t_1[mask_boolean], pred[mask_boolean])
                else:    
                    loss = criterion(x_t_1, pred)
            
            #breakpoint()
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
            t_epoch.set_postfix_str(f'{pred_type} loss:{epoch_loss:.5f} lr:{current_lr:.6f}')  # type: ignore
        if is_tune:
            session.report({'loss': epoch_loss})
        
        # compare MSE metrics and save best model
        if epoch % (num_epoch//10) == 0 and epoch != 0:
            #breakpoint()
            metrics_dict, imputation_data = inference_function(dataloader=valid_dataloader, 
                                        data=valid_data, 
                                        masked_data=valid_masked_data, 
                                        model=model,
                                        mask=mask_valid,
                                        mask_extreme_completion=mask_extreme_completion,
                                        max_norm = max_norm[1],
                                        min_norm = min_norm[1],
                                        avg_tensor = avg_tensor,
                                        diffusion_step=diffusion_step,
                                        device=device,
                                        args=args
                                        )
            #breakpoint()
            adata_valid.layers["diff_pred"] = imputation_data
            log_pred_image_extreme_completion(adata_valid, args, epoch)

            if metrics_dict["MSE"] < min_mse:
                min_mse = metrics_dict["MSE"]
                best_mse = metrics_dict["MSE"]
                best_pcc = metrics_dict["PCC-Gene"]
                torch.save(model.state_dict(), os.path.join("Experiments", exp_name, save_path))
            #save_metrics_to_csv(args.metrics_path, args.dataset, "valid", metrics_dict)
            wandb_logger.log({"MSE": metrics_dict["MSE"], "PCC": metrics_dict["PCC-Gene"]})
    # Save the best MSE and best PCC on the validation set
    wandb_logger.log({"best_MSE":best_mse, "best_PCC": best_pcc})

