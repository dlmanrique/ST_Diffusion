import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F
import numpy as np
import math
from utils import *

# Get parser and parse arguments
parser = get_main_parser()
args = parser.parse_args()
args_dict = vars(args)

#Seed
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """ beta schedule
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class LossVLB():
    def __init__(self,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear"):
        # forward diffusion step
        self.num_timesteps = num_timesteps
        
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == 'cosine':
            self.betas = torch.from_numpy(betas_for_alpha_bar(num_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,).astype(np.float32))
        
        
        self.alphas = 1.0 - self.betas
        
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0) # type: ignore
        
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1)

        # required for q_posterior
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

        #posterior variance
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
            """
            Compute the mean and variance of the diffusion posterior:
                q(x_{t-1} | x_t, x_0)
            """
            assert x_start.shape == x_t.shape
            posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
            )
            posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
            posterior_log_variance_clipped = _extract_into_tensor(
                self.posterior_log_variance_clipped, t, x_t.shape
            )
            assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
            )
            return posterior_mean, posterior_variance, posterior_log_variance_clipped
        
#TODO: posible clamping   
    def p_mean_variance(
        self, model_output, x, t, clip_denoised=False, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        #TODO: revisar este assert
        B, C = x.shape[:2]
        assert t.shape == (B,)
        #TODO: ajustar entrada al modelo
        #model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        model_output, model_var_values = torch.split(model_output, C, dim=1)
        
        #Segun el paper es mejor hacer esto (else)
        min_log = _extract_into_tensor(
        self.posterior_log_variance_clipped, t, x.shape)
        max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
        
        # The model_var_values is [-1, 1] for [min_var, max_var].
        frac = (model_var_values + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log
        model_variance = torch.exp(model_log_variance)
        
        """
        if self.model_var_type == ModelVarType.LEARNED:
            model_log_variance = model_var_values
            model_variance = torch.exp(model_log_variance)
        else:
            min_log = _extract_into_tensor(
                self.posterior_log_variance_clipped, t, x.shape
            )
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            # The model_var_values is [-1, 1] for [min_var, max_var].
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = torch.exp(model_log_variance)
        """
        
        def predict_xstart_from_eps(self, x_t, t, noise):
            """ Get x0 from xt, noise.
            """        
            s1 = self.sqrt_inv_alphas_cumprod[t]
            s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
            
            s1 = s1.reshape(-1, 1, 1).to(x_t.device)
            s2 = s2.reshape(-1, 1, 1).to(x_t.device)
            
            x0 = s1 * x_t - s2 * noise
            return torch.clamp(x0,min=-1,max=1)

        pred_xstart = predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }


    def _vb_terms_bpd(
        self, model_output, x_start, x_t, t, clip_denoised=True, model_kwargs=None):
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                    - 'output': a shape [N] tensor of NLLs or KLs.
                    - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model_output, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)
        
        #Se discretiza la salida en imagenes
        #decoder_nll = -discretized_gaussian_log_likelihood(x_start, means=out["mean"], log_scales=0.5 * out["log_variance"])
        #assert decoder_nll.shape == x_start.shape
        decoder_nll = GaussianLogLikelihood(x_start, out["mean"], out["variance"])
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}
    


class NoiseScheduler():
    def __init__(self,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear"):
        # forward diffusion step
        self.loss_vdl = LossVLB(num_timesteps=1000,
                                beta_start=0.0001,
                                beta_end=0.02,
                                beta_schedule="linear")
        
        self.num_timesteps = num_timesteps
        
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == 'cosine':
            self.betas = torch.from_numpy(betas_for_alpha_bar(num_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,).astype(np.float32))
        
        
        self.alphas = 1.0 - self.betas
        
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0) # type: ignore
        
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            (1 / self.alphas_cumprod) - 1)

        # required for q_posterior
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def reconstruct_x0(self, x_t, t, noise):
        """ Get x0 from xt, noise.
        """        
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        
        s1 = s1.reshape(-1, 1, 1).to(x_t.device)
        s2 = s2.reshape(-1, 1, 1).to(x_t.device)
        
        x0 = s1 * x_t - s2 * noise
        return torch.clamp(x0,min=-1,max=1)
    

    def q_posterior(self, x_0, x_t, t):
        """x_t-1  mean  as part of Reparameteriation
        """        
        
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        
        s1 = s1.reshape(-1, 1, 1).to(x_t.device)
        s2 = s2.reshape(-1, 1, 1).to(x_t.device)
        
        mu = s1 * x_0 + s2 * x_t
        #TODO: agregar clamp
        return mu

    def get_variance(self, t):
        
        try:
            if t == 0:
                return 0
        except:
            pass

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        
        variance = variance.clip(1e-20)
        return variance.to(t.device)
    
    # reverse
    def step(self, 
             model_output, 
             timestep, 
             sample,
             model_pred_type: str='noise'):
        """ reverse diffusioin

        Args:
            model_output (_type_): noise
            timestep (_type_): _t
            sample (_type_): x_t
            model_pred_type (str, optional): _description_. Defaults to 'noise'.

        Returns:
            x_t-1, noise
        """        
        t = timestep
        
        if model_pred_type=='noise':
            pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        elif model_pred_type=='x_start':
            pred_original_sample = model_output
        else:
            raise NotImplementedError()
        
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)  # x_t-1 mean
        
        #variance = 0
        variance = self.loss_vdl._vb_terms_bpd(model_output, pred_original_sample, sample, t)["output"]
        
        try:
            if t > 0:
                noise = torch.randn_like(model_output)
                variance = (self.get_variance(t) ** 0.5) * noise
        except:
            noise = torch.randn_like(model_output)
            variance = self.get_variance(t) ** 0.5
            variance = variance.view(len(timestep), 1, 1).expand(-1, noise.shape[1], noise.shape[2]).to(noise.device)
            variance =  variance * noise
        
        pred_prev_sample = pred_prev_sample + variance  # x_t-1 Reparameteriation
        
        return pred_prev_sample  ,pred_original_sample  

    def add_noise(self, x_start, x_noise, timesteps):  # forward
        # input x_0,noise,t , output x_t
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]
        #s1 = s1.reshape(-1, 1).to(x_start.device)
        s1 = s1.view(len(timesteps), 1, 1).expand(-1, x_start.shape[1], x_start.shape[2]).to(x_start.device)
        #s2 = s2.reshape(-1, 1).to(x_start.device)
        s2 = s2.view(len(timesteps), 1, 1).expand(-1, x_start.shape[1], x_start.shape[2]).to(x_start.device)
        return s1 * x_start + s2 * x_noise

    def undo(self, image_before_step, img_after_model, est_x_0, t, debug=False):
        # add noise
        return self._undo(img_after_model, t)

    def _undo(self, img_out, t):
        beta = self.betas[t]
        img_in_est = torch.sqrt(1 - beta) * img_out + \
                     torch.sqrt(beta) * torch.randn_like(img_out)

        return img_in_est

    def __len__(self):
        return self.num_timesteps
