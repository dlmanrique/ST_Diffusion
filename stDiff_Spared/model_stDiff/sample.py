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
    for x, x_cond, mask in dataloader: 
        x_cond = x_cond.float().to(device) 
        mask = mask.to(device)
        t = torch.from_numpy(np.repeat(time, x_cond.shape[0])).long().to(device)
        # celltype = celltype.to(device)
        if not is_condi:
            n = model(total_sample[i:i+len(x_cond)], t, None) 
        else:
            cond = [x_cond, mask]
            n = model(total_sample[i:i+len(x_cond)], t, cond, condi_flag=condi_flag) 
        noise.append(n)
        i = i+len(x_cond)
    noise = torch.cat(noise, dim=0)
    return noise

def sample_stDiff(model,
                dataloader,
                noise_scheduler,
                args,
                device=torch.device('cuda:1'),
                mask=None,
                gt = None,
                num_step=1000,
                sample_shape=(7060, 2000),
                is_condi=False,
                sample_intermediate=200,
                model_pred_type: str = 'noise',
                is_classifier_guidance=False,
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
    # BoDiffusion
    #x_t = x_t.unsqueeze(2)
    timesteps = list(range(num_step))[::-1]  
    mask = torch.tensor(mask).to(device)
    #BoDiffusion
    #mask = mask.unsqueeze(2)
    gt = torch.tensor(gt).to(device)
    #BoDiffusion
    #gt = gt.unsqueeze(2)
    x_t =  x_t * (1 - mask) + gt * mask
    cond = gt * mask
    #agregando ruido en lo que no esta maskeado y le sumo el GT (multiplicado por la mascara)
    
    if sample_intermediate:
        timesteps = timesteps[:sample_intermediate]

    ts = tqdm(timesteps)
    for t_idx, time in enumerate(ts):
        ts.set_description_str(desc=f'time: {time}')
        with torch.no_grad():
            #model_output (spots, genes): retorna un tensor con las predicciones de los genes maskeados y cero en el resto (en lo no maskeado)
            model_output = model_sample_stDiff(model,
                                        device=device,
                                        dataloader=dataloader,
                                        total_sample=x_t,  # x_t
                                        time=time,  # t
                                        is_condi=is_condi,
                                        condi_flag=True)
            if is_classifier_guidance:
                model_output_uncondi = model_sample_stDiff(model,
                                                    device=device,
                                                    dataloader=dataloader,
                                                    total_sample=sample,
                                                    time=time,
                                                    is_condi=is_condi,
                                                    condi_flag=False)
                model_output = (1 + omega) * model_output - omega * model_output_uncondi

        # x_{t-1}
        # cambio
        if args.loss_type == "x_previous":
            x_t = model_output
        
        else:
            x_t, _ = noise_scheduler.step(model_output,  # noise
                                            torch.from_numpy(np.array(time)).long().to(device),
                                            x_t,
                                            cond,
                                            model_pred_type=args.loss_type)
        
        if mask is not None:
            x_t = x_t * (1. - mask) + mask * gt  

        #cambio
        if time == 0 and args.loss_type == "x_start":
            sample = model_output
    
    recon_x = x_t.detach().cpu().numpy()
    return recon_x

