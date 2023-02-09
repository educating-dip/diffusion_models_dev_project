from models import utils as mutils
import torch
import numpy as np
from sampling import NoneCorrector, NonePredictor, shared_corrector_update_fn, shared_predictor_update_fn
import functools
from physics.ct import CT
from utils import show_samples, show_samples_gray, clear, clear_color
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

def PSNR(original, reco):
    mse = torch.mean((original - reco) ** 2)
    if(mse == 0):
        return 100
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr 

def get_pc_radon_MCG(sde, predictor, corrector, inverse_scaler, snr,
                     n_steps=1, probability_flow=False, continuous=False, weight=1.0,
                     denoise=True, eps=1e-5, radon=None, radon_all=None, save_progress=False, save_root=None,
                     lamb_schedule=None, mask=None, measurement_noise=False):
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    def _A(x):
        return radon.A(x)

    def _AT(sinogram):
        return radon.AT(sinogram)

    def _AINV(sinogram):
        return radon.A_dagger(sinogram)

    def _A_all(x):
        return radon_all.A(x)

    def _AINV_all(sinogram):
        return radon_all.A_dagger(sinogram)

    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x, _, _ = update_fn(x, vec_t, model=model)
                return x

        return radon_update_fn

    def get_corrector_update_fn(update_fn):
        def radon_update_fn(model, data, x, t, measurement=None, i=None, norm_const=None):
            vec_t = torch.ones(data.shape[0], device=data.device) * t

            # mn True
            if measurement_noise:
                measurement_mean, std = sde.marginal_prob(measurement, vec_t)
                measurement = measurement_mean + torch.randn_like(measurement) * std[:, None, None, None]

            # input to the score function
            x = x.requires_grad_()
            x_next, x_next_mean, score = update_fn(x, vec_t, model=model)

            lamb = lamb_schedule.get_current_lambda(i)

            # x0 hat estimation
            _, bt = sde.marginal_prob(x, vec_t)
            hatx0 = x + (bt ** 2) * score

            # MCG method
            # norm = torch.linalg.norm(_AINV(measurement - _A(hatx0)))
            norm = torch.norm(_AINV(measurement - _A(hatx0)))
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]
            norm_grad *= weight
            norm_grad = _AINV_all(_A_all(norm_grad) * (1. - mask))

            x_next = x_next + lamb * _AT(measurement - _A(x_next)) / norm_const - norm_grad
            x_next = x_next.detach()
            return x_next

        return radon_update_fn

    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    corrector_radon_update_fn = get_corrector_update_fn(corrector_update_fn)

    def pc_radon(model, data, measurement=None, start_N=0, save_freq=100):
        timesteps = torch.linspace(sde.T, eps, sde.N)

        if start_N == 0:
            x = sde.prior_sampling(data.shape).to(data.device)
        else: 
            x_fbp = _AINV(measurement)
            x_fbp_mean, x_fbp_std = sde.marginal_prob(x_fbp, torch.ones(data.shape[0], device=data.device) * timesteps[start_N])
            x_fbp_noisy = x_fbp_mean + torch.randn_like(x_fbp) * x_fbp_std[:, None, None, None]
            x = x_fbp_noisy

            plt.imsave(save_root / 'recon' / f'noisy_fbp_start_point_for_sde.png', clear(x), cmap='gray')

        psnr_list = [] 
        ones = torch.ones_like(x).to(data.device)
        norm_const = _AT(_A(ones))
        for i in tqdm(range(start_N, sde.N)):
            t = timesteps[i]
            x = predictor_denoise_update_fn(model, data, x, t)
            x = corrector_radon_update_fn(model, data, x, t, measurement=measurement, i=i,
                                          norm_const=norm_const)

            with torch.no_grad():
                psnr = PSNR(inverse_scaler(data), inverse_scaler(x))
                psnr_list.append(psnr.item())
                print(psnr.item())
            if save_progress:
                if (i % save_freq) == 0:
                    plt.imsave(save_root / 'recon' / f'progress{i}.png', clear(x), cmap='gray')

        plt.figure()
        plt.plot(np.arange(start_N+1,sde.N+1), psnr_list)
        plt.xlabel("iteration")
        plt.ylabel("PSNR (max pixel=1)")
        plt.savefig(save_root / 'recon' / f'psnr.png')
        plt.close()
            
        return inverse_scaler(x if denoise else x)

    return pc_radon

def get_pc_radon_song(sde, predictor, corrector, inverse_scaler, snr,
                      n_steps=1, probability_flow=False, continuous=False,
                      denoise=True, eps=1e-5, radon=None, save_progress=False, save_root=None, lamb=1.0,
                      freq=10):
    """ Sparse application of measurement consistency """
    # Define predictor & corrector
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    def _A(x):
        return radon.A(x)

    def _A_dagger(sinogram):
        return radon.A_dagger(sinogram)

    def data_fidelity(mask, x, x_mean, vec_t=None, measurement=None, lamb=lamb, i=None):
        y_mean, std = sde.marginal_prob(measurement, vec_t)
        hat_y = (y_mean + torch.rand_like(y_mean) * std[:, None, None, None]) * mask
        weighted_hat_y = hat_y * lamb

        sino = _A(x)
        sino_meas = sino * mask
        weighted_sino_meas = sino_meas * (1 - lamb)
        sino_unmeas = sino * (1. - mask)

        weighted_sino = weighted_sino_meas + sino_unmeas

        updated_y = weighted_sino + weighted_hat_y
        x = _A_dagger(updated_y)

        sino_mean = _A(x_mean)
        updated_y_mean = sino_mean * mask * (1. - lamb) + sino * (1. - mask) + y_mean * lamb
        x_mean = _A_dagger(updated_y_mean)
        return x, x_mean

    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x, x_mean, _ = update_fn(x, vec_t, model=model)
                return x, x_mean

        return radon_update_fn

    def get_corrector_update_fn(update_fn):
        def radon_update_fn(model, data, mask, x, t, measurement=None, i=None):
            with torch.no_grad():
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x, x_mean, _ = update_fn(x, vec_t, model=model)
                x, x_mean = data_fidelity(mask, x, x_mean, vec_t=vec_t, measurement=measurement, lamb=lamb, i=i)
                return x, x_mean

        return radon_update_fn

    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    corrector_denoise_update_fn = get_update_fn(corrector_update_fn)
    corrector_radon_update_fn = get_corrector_update_fn(corrector_update_fn)

    def pc_radon(model, data, mask, measurement=None):
        with torch.no_grad():
            x = sde.prior_sampling(data.shape).to(data.device)
            timesteps = torch.linspace(sde.T, eps, sde.N)
            for i in tqdm(range(sde.N)):
                t = timesteps[i]
                x, x_mean = predictor_denoise_update_fn(model, data, x, t)
                if (i % freq) == 0:
                    x, x_mean = corrector_radon_update_fn(model, data, mask, x, t, measurement=measurement, i=i)
                else:
                    x, x_mean = corrector_denoise_update_fn(model, data, x, t)
                if save_progress:
                    if (i % 100) == 0:
                        plt.imsave(save_root / 'recon' / f'progress{i}.png', clear(x_mean), cmap='gray')
            return inverse_scaler(x_mean if denoise else x)

    return pc_radon


def get_pc_radon_POCS(sde, predictor, corrector, inverse_scaler, snr,
                      n_steps=1, probability_flow=False, continuous=False,
                      denoise=True, eps=1e-5, radon=None, save_progress=False, save_root=None,
                      lamb_schedule=None, measurement_noise=False, final_consistency=False):
    """ Sparse application of measurement consistency """
    # Define predictor & corrector
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    def _A(x):
        return radon.A(x)

    def _AT(sinogram):
        return radon.AT(sinogram)

    def kaczmarz(x, x_mean, measurement=None, lamb=1.0, i=None,
                 norm_const=None):
        x = x + lamb * _AT(measurement - _A(x)) / norm_const
        x_mean = x
        return x, x_mean

    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x, x_mean, _ = update_fn(x, vec_t, model=model)
                return x, x_mean

        return radon_update_fn

    def get_corrector_update_fn(update_fn):
        def radon_update_fn(model, data, x, t, measurement=None, i=None, norm_const=None):
            with torch.no_grad():
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x, x_mean, _ = update_fn(x, vec_t, model=model)
                lamb = lamb_schedule.get_current_lambda(i)

                if measurement_noise:
                    measurement_mean, std = sde.marginal_prob(measurement, vec_t)
                    measurement = measurement_mean + torch.randn_like(measurement) * std[:, None, None, None]

                x, x_mean = kaczmarz(x, x_mean, measurement=measurement, lamb=lamb, i=i,
                                     norm_const=norm_const)
                return x, x_mean

        return radon_update_fn

    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    corrector_radon_update_fn = get_corrector_update_fn(corrector_update_fn)

    def pc_radon(model, data, measurement=None):
        with torch.no_grad():
            x = sde.prior_sampling(data.shape).to(data.device)

            ones = torch.ones_like(x).to(data.device)
            norm_const = _AT(_A(ones))
            timesteps = torch.linspace(sde.T, eps, sde.N)
            for i in tqdm(range(sde.N)):
                t = timesteps[i]
                x, x_mean = predictor_denoise_update_fn(model, data, x, t)
                x, x_mean = corrector_radon_update_fn(model, data, x, t, measurement=measurement, i=i,
                                                      norm_const=norm_const)
                if save_progress:
                    if (i % 20) == 0:
                        print(f'iter: {i}/{sde.N}')
                        plt.imsave(save_root / 'recon' / f'progress{i}.png', clear(x_mean), cmap='gray')
            # Final step which coerces the data fidelity error term to be zero,
            # and thereby satisfying Ax = y
            if final_consistency:
                x, x_mean = kaczmarz(x, x_mean, measurement, lamb=1.0, norm_const=norm_const)

            return inverse_scaler(x_mean if denoise else x)

    return pc_radon


def get_pc_radon_naive(sde, predictor, corrector, inverse_scaler, snr,
                     n_steps=1, probability_flow=False, continuous=False, weight=1.0,
                     denoise=True, eps=1e-5, save_progress=False, save_root=None,
                     lamb_schedule=None, measurement_noise=False, ray_trafo=None, ray_trafo_adjoint=None, fbp_op=None):

    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x, _, _ = update_fn(x, vec_t, model=model)
                return x

        return radon_update_fn

    def get_corrector_update_fn(update_fn):
        def radon_update_fn(model, data, x, t, measurement=None, i=None, norm_const=None):
            vec_t = torch.ones(data.shape[0], device=data.device) * t

            # mn True
            if measurement_noise:
                measurement_mean, std = sde.marginal_prob(measurement, vec_t)
                measurement = measurement_mean + torch.randn_like(measurement) * std[:, None, None, None]

            # input to the score function
            x_next, x_next_mean, score = update_fn(x, vec_t, model=model)

            lamb = lamb_schedule.get_current_lambda(i)

            x_next = x_next + lamb * ray_trafo_adjoint(measurement - ray_trafo(x_next)) / norm_const

            return x_next

        return radon_update_fn

    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    corrector_radon_update_fn = get_corrector_update_fn(corrector_update_fn)

    def pc_radon(model, data, measurement=None, start_N=0, save_freq=100):
        timesteps = torch.linspace(sde.T, eps, sde.N)

        if start_N == 0:
            x = sde.prior_sampling(data.shape).to(data.device)
        else: 
            x_fbp = fbp_op(measurement)
            x_fbp_mean, x_fbp_std = sde.marginal_prob(x_fbp, torch.ones(data.shape[0], device=data.device) * timesteps[start_N])
            x_fbp_noisy = x_fbp_mean + torch.randn_like(x_fbp) * x_fbp_std[:, None, None, None]
            x = x_fbp_noisy

            plt.imsave(save_root / 'recon' / f'noisy_fbp_start_point_for_sde.png', clear(x), cmap='gray')

        psnr_list = [] 
        ones = torch.ones_like(x).to(data.device)
        norm_const = ray_trafo_adjoint(ray_trafo(ones))*2
        print("AT(A(1)): ", norm_const)
        for i in tqdm(range(start_N, sde.N)):
            t = timesteps[i]
            x = predictor_denoise_update_fn(model, data, x, t)
            x = corrector_radon_update_fn(model, data, x, t, measurement=measurement, i=i,
                                          norm_const=norm_const)

            with torch.no_grad():
                psnr = PSNR(inverse_scaler(data), inverse_scaler(x))
                psnr_list.append(psnr.item())
                print(psnr.item())
            if save_progress:
                if (i % save_freq) == 0:
                    plt.imsave(save_root / 'recon' / f'progress{i}.png', clear(x), cmap='gray')

        plt.figure()
        plt.plot(np.arange(start_N+1,sde.N+1), psnr_list)
        plt.xlabel("iteration")
        plt.ylabel("PSNR (max pixel=1)")
        plt.savefig(save_root / 'recon' / f'psnr.png')
        plt.close()
            
        return inverse_scaler(x if denoise else x)

    return pc_radon
