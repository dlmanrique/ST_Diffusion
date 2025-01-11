import torch
import numpy as np
import os
import torch.nn as nn
from tqdm import tqdm
from ray.air import session
import os
from .stDiff_scheduler import NoiseScheduler
from utils import *
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
                 max_norm,
                 min_norm, 
                 avg_tensor,
                 wandb_logger,
                 args,
                 st_data_val,
                 adata_valid,
                 lr: float = 1e-4,
                 num_epoch: int = 1400,
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
        num_timesteps=args.diffusion_steps_train,
        beta_schedule=args.noise_scheduler
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

    #exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(os.path.join('Experiments', exp_name), exist_ok=True)
    
    for epoch in t_epoch:
        epoch_loss = 0.
        for i, (x, x_cond) in enumerate(train_dataloader): 
            # En este caso x son los vectores de expresion normalizados (-1-1)
            # x_cond es el vector de features de UNI con shape de 1024
            x, x_cond = x.float().to(device), x_cond.float().to(device)

            noise = torch.randn(x.shape).to(device)

            
            timesteps = torch.randint(1, args.diffusion_steps_train, (x.shape[0],)).long()

            x_t = noise_scheduler.add_noise(x,
                                            noise,
                                            timesteps=timesteps.cpu())

            # Datos de expresion y les sumo ruido en todas las posiciones
            x_noisy = x_t

            # Como condicion tengo de input el vector de UNI de shape 1024
            cond = [x_cond]

            pred = model(x_noisy, t=timesteps.to(device), y=cond) 

            loss = criterion(noise, pred)

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
            
            metrics_dict, imputation_data = inference_function(dataloader=valid_dataloader,
                                        data=st_data_val, 
                                        model=model,
                                        max_norm = max_norm[1],
                                        min_norm = min_norm[1],
                                        avg_tensor = avg_tensor,
                                        diffusion_step=args.diffusion_steps_train,
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

