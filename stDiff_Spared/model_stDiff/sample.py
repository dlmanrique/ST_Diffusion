import torch
from tqdm import tqdm
import numpy as np
from utils import get_main_parser

# Get parser and parse arguments
parser = get_main_parser()
args = parser.parse_args()
args_dict = vars(args)

#Seed
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def model_sample_stDiff(model, device, dataloader, total_sample, time, is_condi, condi_flag):
    noise = []
    i = 0
    for _, x_cond in dataloader: 
        x_cond = x_cond.float().to(device) 
        t = torch.from_numpy(np.repeat(time, x_cond.shape[0])).long().to(device)
        # celltype = celltype.to(device)
        if not is_condi:
            n = model(total_sample[i:i+len(x_cond)], t, None) 
        else:
            cond = [x_cond]
            n = model(total_sample[i:i+len(x_cond)], t, cond, condi_flag=condi_flag) 
        noise.append(n)
        i = i+len(x_cond)
    noise = torch.cat(noise, dim=0)
    return noise

def sample_stDiff(model,
                dataloader,
                noise_scheduler,
                args,
                device=torch.device('cuda:0'),
                num_step=1000,
                sample_shape=(7060, 2000),
                is_condi=False,
                sample_intermediate=200,
                model_pred_type: str = 'noise',
                omega=0.1):
    #mask = None
    """_summary_

    Args:
        model (_type_): denoising model
        dataloader (_type_): _description_
        noise_scheduler (_type_): _description_
        mask (_type_, optional): _mask_. Defaults to None.
        gt (_type_, optional): _used to get the part of the data that is not missing_. Defaults to None.
        device (_type_, optional): _device_. Defaults to torch.device('cuda:1').
        num_step (int, optional): _timestep_. Defaults to 1000.
        sample_shape (tuple, optional): _sample shape_. Defaults to (7060, 2000).
        is_condi (bool, optional): _whether to use condition_. Defaults to False.
        sample_intermediate (int, optional): _early termination_. Defaults to 200.
        model_pred_type (str, optional): _preditc noise or x0_. Defaults to 'noise'.
        is_classifier_guidance (bool, optional): whether to use cfg. Defaults to False.
        omega (float, optional): classifier guidance hyperparameter. Defaults to 0.1.

    Returns:
        _type_: recon_x
    """
    model.eval()
    x_t = torch.randn(sample_shape).to(device)
    timesteps = list(range(num_step))[::-1]  
    
    if sample_intermediate:
        timesteps = timesteps[:sample_intermediate]

    ts = tqdm(timesteps)
    for t_idx, time in enumerate(ts):
        ts.set_description_str(desc=f'time: {time}')
        with torch.no_grad():
            #model_output -> ruido para este timestep
            model_output = model_sample_stDiff(model,
                                        device=device,
                                        dataloader=dataloader,
                                        total_sample=x_t,  # x_t
                                        time=time,  # t
                                        is_condi=is_condi,
                                        condi_flag=True)
            # x_t -> removemos el ruido predicho por el modelo
            x_t, _ = noise_scheduler.step(model_output,  # noise
                                            torch.from_numpy(np.array(time)).long().to(device),
                                            x_t,
                                            model_pred_type=args.loss_type)


    recon_x = x_t.detach().cpu()
    return recon_x

