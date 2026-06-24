from typing import Dict, Union, Optional

import torch
from omegaconf import ListConfig, OmegaConf
from tqdm import tqdm

from .util import append_dims, default, tools_scale
from .guiders import IdentityGuider
from .discretizer import EDMDiscretization

class BaseResidualDiffusionSampler:
    def __init__(
            self,
            discretization_config: Optional[Union[Dict, ListConfig, OmegaConf]]=None,
            num_steps: Union[int, None] = None,
            guider_config: Union[Dict, ListConfig, OmegaConf, None] = None,
            verbose: bool = False,
            device: str = "cuda",
    ):
        self.num_steps = num_steps
        self.discretization = EDMDiscretization(sigma_min=0.001, sigma_max=100.0)
        self.guider = IdentityGuider()
        self.verbose = verbose
        self.device = device

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        uc = default(uc, cond)

        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)

        s_in = x.new_ones([x.shape[0]])

        return x, s_in, sigmas, num_sigmas, cond, uc

    def denoise(self, x, denoiser, sigma, cond, st, uc):
        denoised = denoiser(x, sigma, cond, st=st)
        denoised = self.guider(denoised, sigma)
        return denoised

    def get_sigma_gen(self, num_sigmas):
        sigma_generator = range(num_sigmas - 1)
        if self.verbose:
            print("#" * 30, " Sampling setting ", "#" * 30)
            print(f"Sampler: {self.__class__.__name__}")
            print(f"Discretization: {self.discretization.__class__.__name__}")
            print(f"Guider: {self.guider.__class__.__name__}")
            sigma_generator = tqdm(
                sigma_generator,
                total=num_sigmas,
                desc=f"Sampling with {self.__class__.__name__} for {num_sigmas} steps",
            )
        return sigma_generator


class SingleStepResidualDiffusionSampler(BaseResidualDiffusionSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc, *args, **kwargs):
        raise NotImplementedError

    def euler_step(self, x, d, dt):
        return x + dt * d


# 以下EDM Sampler将sigma_t = t作为前提进行实现
class ResidualEDMSampler(SingleStepResidualDiffusionSampler):
    def __init__(
            self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        self.sigma2st = None

    def set_sigma2st(self, sigma2st):
        self.sigma2st = sigma2st

    def prepare_sampling_loop(self, x, mu, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        uc = default(uc, cond)
        st0 = self.sigma2st(sigmas[0])
        # x = mu + sigmas[0] * st * x
        x = ((1 - st0) / st0) * mu + sigmas[0] * x
        num_sigmas = len(sigmas)

        s_in = x.new_ones([x.shape[0]])

        return x, s_in, sigmas, num_sigmas, cond, uc

    def sampler_step(self, sigma, next_sigma, denoiser, x, mu, cond, uc=None, gamma=0.0):
        st = self.sigma2st(sigma)
        sigma_hat = sigma * (gamma + 1.0)
        st_hat = self.sigma2st(sigma_hat)
        st_hat_derivative = self.sigma2st.get_derivative_st()(sigma_hat)
        st_hat_bc = append_dims(st_hat, x.ndim)
        st_hat_derivative_bc = append_dims(st_hat_derivative, x.ndim)
        sigma_hat_bc = append_dims(sigma_hat, x.ndim)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + ((1 - st_hat) / st_hat - (1 - st) / st) * mu + eps * append_dims(sigma_hat ** 2 - sigma ** 2,
                                                                                     x.ndim) ** 0.5
        # denoised = self.denoise(x, denoiser, sigma_hat, cond, st_hat, uc)
        denoised, zs = self.denoise(x, denoiser, sigma_hat, cond, st_hat, uc)
        # d = - (x - mu) - 2 * st_hat_bc * denoised + 2 * x
        # d = - st_hat_bc * x + mu - (denoised - x) / sigma_hat_bc
        d = (- st_hat_derivative_bc / (st_hat_bc ** 2)) * mu - \
            (denoised + (1 - st_hat_bc) / st_hat_bc * mu - x) / sigma_hat_bc
        # d = - st_hat_bc * (x - mu)  - denoised * st_hat_bc / sigma_hat_bc + x / sigma_hat_bc
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        x = self.possible_correction_step(
            euler_step, x, mu, d, dt, next_sigma, denoiser, cond, uc
        )
        return x, denoised

    def __call__(self, denoiser, x, mu, cond, uc=None, num_steps=None, return_intermediate=False,
                 return_denoised=False):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, mu, cond, uc, num_steps
        )
        intermediates = []
        denoiseds = []
        range_sigmas = self.get_sigma_gen(num_sigmas)
        for i in range_sigmas:
            gamma = (
                # min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                self.s_churn / (num_sigmas - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            if return_intermediate:
                intermediates.append(tools_scale(x.clone().detach()))
            x, denoised = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                mu,
                cond,
                uc,
                gamma,
            )
            if return_denoised:
                denoiseds.append(tools_scale(denoised.clone().detach()))
        others = {}
        if return_intermediate:
            others["intermediates"] = intermediates
        if return_denoised:
            others["denoiseds"] = denoiseds
        return x, others


class ResidualEulerEDMSampler(ResidualEDMSampler):
    def possible_correction_step(
            self, euler_step, x, mu, d, dt, next_sigma, denoiser, cond, uc
    ):
        return euler_step


class ResidualHeunEDMSampler(ResidualEDMSampler):
    def possible_correction_step(
            self, euler_step, x, mu, d, dt, next_sigma, denoiser, cond, uc
    ):
        if torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            return euler_step
        else:
            sigma_bc = append_dims(next_sigma, x.ndim)
            st = self.sigma2st(next_sigma)
            st_derivative = self.sigma2st.get_derivative_st()(next_sigma)

            st_bc = append_dims(st, x.ndim)
            st_derivative_bc = append_dims(st_derivative, x.ndim)

            denoised = self.denoise(euler_step, denoiser, next_sigma, cond, st, uc)
            d_new = (- st_derivative_bc / (st_bc ** 2)) * mu - \
                    (denoised + (1 - st_bc) / st_bc * mu - x) / sigma_bc
            d_prime = (d + d_new) / 2.0

            # apply correction if noise level is not 0
            x = torch.where(
                append_dims(next_sigma, x.ndim) > 0.0, x + d_prime * dt, euler_step
            )
            return x


class TemporalResidualEDMSampler(ResidualEDMSampler):

    def denoise(self, x, denoiser, sigma, cond, st, uc, return_attn=False):
        if return_attn:
            denoised, attn = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc), st=st, return_attn=return_attn)
        else:
            denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc), st=st, return_attn=return_attn)
        denoised = self.guider(denoised, sigma)
        if return_attn:
            return denoised, attn
        else:
            return denoised

    def sampler_step(self, sigma, next_sigma, denoiser, x, mu, cond, uc=None, gamma=0.0, return_attn=False):
        st = self.sigma2st(sigma)
        st_bc = append_dims(st, x.ndim)
        sigma_bc = append_dims(sigma, x.ndim)
        sigma_hat = sigma * (gamma + 1.0)
        st_hat = self.sigma2st(sigma_hat)
        st_hat_derivative = self.sigma2st.get_derivative_st()(sigma_hat)
        st_hat_bc = append_dims(st_hat, x.ndim)
        st_hat_derivative_bc = append_dims(st_hat_derivative, x.ndim)
        sigma_hat_bc = append_dims(sigma_hat, x.ndim)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + ((1 - st_hat_bc) / st_hat_bc - (1 - st_bc) / st_bc) * mu + eps * append_dims(
                sigma_hat ** 2 - sigma ** 2, x.ndim) ** 0.5
        denoised = self.denoise(x, denoiser, sigma_hat, cond, st_hat, uc, return_attn=return_attn)
        if return_attn:
            denoised, attn = denoised
        _denoised = denoised.unsqueeze(dim=1).repeat(1, x.shape[1], 1, 1, 1)
        d = (- st_hat_derivative_bc / (st_hat_bc ** 2)) * mu - \
            (_denoised + (1 - st_hat_bc) / st_hat_bc * mu - x) / sigma_hat_bc
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        x = self.possible_correction_step(
            euler_step, x, mu, d, dt, next_sigma, denoiser, cond, uc
        )
        if return_attn:
            return x, denoised, attn
        else:
            return x, denoised

    def __call__(self, denoiser, x, mu, cond, uc=None, num_steps=None, return_intermediate=False, return_denoised=False,
                 return_attn=False):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, mu, cond, uc, num_steps
        )
        intermediates = []
        denoiseds = []
        attns = []
        range_sigmas = self.get_sigma_gen(num_sigmas)
        for i in range_sigmas:
            gamma = (
                # min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                self.s_churn / (num_sigmas - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            if return_intermediate:
                intermediates.append(tools_scale(x.clone().detach()))
            if return_attn:
                x, denoised, attn = self.sampler_step(
                    s_in * sigmas[i],
                    s_in * sigmas[i + 1],
                    denoiser,
                    x,
                    mu,
                    cond,
                    uc,
                    gamma,
                    return_attn=return_attn
                )
            else:
                x, denoised = self.sampler_step(
                    s_in * sigmas[i],
                    s_in * sigmas[i + 1],
                    denoiser,
                    x,
                    mu,
                    cond,
                    uc,
                    gamma,
                    return_attn=return_attn
                )
            if return_denoised:
                denoiseds.append(tools_scale(denoised.detach()))
            if return_attn:
                # attns.append(tools_scale(attn.detach()))
                attns.append(attn.detach())
        others = {}
        if return_intermediate:
            others["intermediates"] = intermediates
        if return_denoised:
            others["denoiseds"] = denoiseds
        if return_attn:
            others["attns"] = attns
        return x.mean(dim=1), others


class TemporalResidualEulerEDMSampler(TemporalResidualEDMSampler):
    def possible_correction_step(
            self, euler_step, x, mu, d, dt, next_sigma, denoiser, cond, uc
    ):
        return euler_step

