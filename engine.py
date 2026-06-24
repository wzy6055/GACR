import torch
import numpy as np
import math
import torch.nn.functional as F
from einops import rearrange
from torchvision.utils import make_grid

from utils import tools_scale
from models.sgm.denoiser import ResidualDenoiser
from models.sgm.sigma2st import EDMSigma2St
from models.sgm.sigma_sampling import EDMSampling
from models.sgm.util import append_dims
from models.sgm.loss_weighting import ResidualEDMWeighting
from models.sgm.util import scale_01_from_minus1_1 as scale_01
from models.sgm.sampling import ResidualEulerEDMSampler
from models.sgm.evaluator import img_metrics, avg_img_metrics

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

class OARFlowEngine:
    def __init__(
            self,
            model,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[],
            accelerator=None,
            decoder=None,
            latents_bias=None,
            latents_scale=None,
            loss_stage=None,
            p=3.0
    ):
        self.model = model
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        # self.encoders = encoders
        self.sampler = euler_sampler
        self.p = p
        self.avg_metrics = avg_img_metrics()
        self.scale_01 = scale_01()

    def get_proj_loss(self, zs, zs_tilde):
        proj_loss = 0.
        bsz = zs[0].shape[0]
        for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
            for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1)
                z_j = torch.nn.functional.normalize(z_j, dim=-1)
                proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
        proj_loss /= (len(zs) * bsz)
        return proj_loss

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
            # beta
            beta_t  = self.p * sigma_t
            d_beta_t = self.p * d_sigma_t
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
            # beta
            beta_t  = self.p * sigma_t
            d_beta_t = self.p * d_sigma_t
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t, beta_t, d_beta_t

    def loss_fn(self, input, mu, cond, batch, zs=None):
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((input.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((input.shape[0], 1 ,1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)

        time_input = time_input.to(device=input.device, dtype=input.dtype)
        noises = torch.randn_like(input)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t, beta_t, d_beta_t = self.interpolant(time_input)

        noised_input = alpha_t * input + sigma_t * noises + beta_t * mu

        if self.prediction == 'v':
            model_target = d_alpha_t * input + d_sigma_t * noises + d_beta_t * mu
        else:
            raise NotImplementedError()

        model_output, zs_tilde = self.model(noised_input, time_input.flatten(), cond)
        denoising_loss = mean_flat((model_output - model_target) ** 2)

        if zs is not None:
            # proj_loss = 0
            proj_loss = self.get_proj_loss(zs, zs_tilde)
        else:
            proj_loss = None
        return denoising_loss, proj_loss


    @torch.no_grad()
    def sample(self, cond, mu, uc, batch_size=16, shape=None, return_intermediate=False,return_denoised=False, num_steps=4):
        randn = torch.randn(batch_size, *shape).to(cond.device)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t, beta_t, d_beta_t = self.interpolant(t=1.)
        # xt = (self.p * mu + randn).to(cond.device)
        xt = sigma_t * randn + beta_t * mu
        sample, others = self.sampler(self.model, xt, mu, cond, num_steps=num_steps, return_intermediate=return_intermediate)
        return sample, others

    def __call__(self, batch, zs=None):   # call train
        x, mu, cond = batch['clear'].clone(), batch['cloudy'].clone(), batch['cond'].clone()
        return self.loss_fn(x, mu, cond, batch, zs)

    @torch.no_grad()
    def test_step(self, batch, num_steps=4):  # call test
        target, mu, c = batch['clear'].clone(), batch['cloudy'].clone(), batch['cond'].clone()
        uc = c
        z_mu = mu
        N = z_mu.shape[0]
        samples, _ = self.sample(c, z_mu, shape=z_mu.shape[1:], uc=uc, batch_size=N, num_steps=num_steps)

        for i in range(samples.shape[0]):
            _target = target[i, ...]
            _samples = samples[i, ...]
            _target = self.scale_01(_target)
            _samples = self.scale_01(_samples)
            metrics = img_metrics(target=_target.unsqueeze(0), pred=_samples.unsqueeze(0))
            # self.log_dict(metrics, sync_dist=True, batch_size=1, on_epoch=True)
            # _mu = self.scale_01(mu[i, ...])
            # raw_metrics = self.img_metrics(target=_target.unsqueeze(0), pred=_mu.unsqueeze(0))
            # raw_metrics = {"raw_" + k: v for k, v in raw_metrics.items()}
            # self.log_dict(raw_metrics, sync_dist=True, batch_size=1, on_epoch=True)
            self.avg_metrics.add(metrics)

        return self.avg_metrics

    def _get_denoise_row_from_list(self, samples, desc='', to_rgb_func=None):
        denoise_row = []
        for zd in samples:
            denoise_row.append(zd)
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        if to_rgb_func != None:
            denoise_grid = to_rgb_func(denoise_grid)
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=1, sample=True, return_intermediate=False, return_denoised=False, return_add_mu=False,
                   return_add_noise=False):
        results = dict()
        results["input"] = self.scale_01(batch['clear'].clone().detach())
        results["mean"] = self.scale_01(batch['cloudy'].clone().detach())

        x, mu, c = batch['clear'], batch['cloudy'], batch['cond']
        uc = c
        N = min(x.shape[0], N)

        x = x[:N]
        mu = mu[:N]

        if return_add_mu or return_add_noise:
            sigmas = self.sampler.discretization(
                self.sampler.num_steps, device=c.device
            )
            mus = [tools_scale(x.clone().detach())] if return_add_mu else None
            noises = [tools_scale(x.clone().detach())] if return_add_noise else None

            for i in reversed(self.sampler.get_sigma_gen(self.sampler.num_steps)):
                sigma = sigmas[i]
                st = self.sigma2st(sigma)
                if return_add_mu:
                    _ = x + (1 - st) / st * mu
                    mus.append(tools_scale(_.detach()))
                if return_add_noise:
                    _ = x + (1 - st) / st * mu + torch.randn_like(x) * sigma
                    noises.append(tools_scale(_.detach()))

            if return_add_mu:
                results["mu_shifting"] = self._get_denoise_row_from_list(mus)
            if return_add_noise:
                results["mu_noise_shifting"] = self._get_denoise_row_from_list(noises)

        z_mu = mu
        if sample:
            samples, others = self.sample(
                c, z_mu, shape=z_mu.shape[1:], uc=uc, batch_size=N, return_intermediate=return_intermediate, return_denoised=return_denoised)
            results["samples"] = self.scale_01(samples)
            if return_intermediate:
                # results["intermediate"] = self._get_denoise_row_from_list(others['intermediates'])
                results["intermediate"] = others["intermediate"]
            if return_denoised:
                results["denoised"] = self._get_denoise_row_from_list(others['denoiseds'])

        return results


def euler_sampler(
        model,
        x,
        mu,
        cond,
        num_steps=20,
        heun=False,
        path_type="linear",  # not used, just for compatability
        return_intermediate=False
):
    # setup conditioning
    # _dtype = x.dtype
    t_steps = torch.linspace(1, 0, num_steps + 1, dtype=x.dtype)
    # x_next = x.to(torch.float64)
    x_next = x
    device = x_next.device

    others = {}
    intermediate=[]
    intermediate_input=[]
    intermediate_pred=[]
    intermediate_output=[]

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            if return_intermediate:
                intermediate.append(x_cur)
                intermediate_input.append(x_cur)
            model_input = x_cur
            time_input = torch.ones(model_input.size(0)).to(device=device) * t_cur
            d_cur, zs = model(model_input, time_input, cond)
            if return_intermediate:
                intermediate_pred.append(d_cur)
            x_next = x_cur + (t_next - t_cur) * d_cur
            if return_intermediate:
                intermediate_output.append(x_next)
            if heun and (i < num_steps - 1):
                model_input = x_next
                time_input = torch.ones(model_input.size(0)).to(device=model_input.device) * t_next
                d_prime = model(model_input, time_input)[0]
                x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

    others['zs'] = zs
    if return_intermediate:
        others['intermediate'] = intermediate
        others['intermediate_input'] = intermediate_input
        others['intermediate_pred'] = intermediate_pred
        others['intermediate_output'] = intermediate_output

    return x_next, others
