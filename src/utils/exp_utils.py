import os
import time
import torch
import functools
from math import ceil
from pathlib import Path

from .ema import ExponentialMovingAverage
from ..dataset import LoDoPabDatasetFromDival, MayoDataset, get_disk_dist_ellipses_dataset, get_walnut_data
from ..physics import SimpleTrafo, get_walnut_2d_ray_trafo, simulate
from ..samplers import BaseSampler, Euler_Maruyama_VE_sde_predictor, Langevin_VE_sde_corrector, chain_simple_init, decomposed_diffusion_sampling_VE_sde_predictor, conj_grad_closure
from ..third_party_models import OpenAiUNetModel

def get_standard_score(config, sde, use_ema, load_model=True):

    if config.model.model_name == 'OpenAiUNetModel':
	    score = OpenAiUNetModel(
            image_size=config.data.im_size,
            in_channels=config.model.in_channels,
            model_channels=config.model.model_channels,
            out_channels=config.model.out_channels,
            num_res_blocks=config.model.num_res_blocks,
            attention_resolutions=config.model.attention_resolutions,
            marginal_prob_std=sde.marginal_prob_std,
            channel_mult=config.model.channel_mult,
            conv_resample=config.model.conv_resample,
            dims=config.model.dims,
            num_heads=config.model.num_heads,
            num_head_channels=config.model.num_head_channels,
            num_heads_upsample=config.model.num_heads_upsample,
            use_scale_shift_norm=config.model.use_scale_shift_norm,
            resblock_updown=config.model.resblock_updown,
            use_new_attention_order=config.model.use_new_attention_order
            )
    else:
        raise NotImplementedError

    if config.sampling.load_model_from_path is not None and config.sampling.model_name is not None and load_model: 
        print(f'load score model from path: {config.sampling.load_model_from_path}')
        if use_ema:
            ema = ExponentialMovingAverage(score.parameters(), decay=0.999)
            ema.load_state_dict(torch.load(os.path.join(config.sampling.load_model_from_path,'ema_model.pt')))
            ema.copy_to(score.parameters())
        else:
            score.load_state_dict(torch.load(os.path.join(config.sampling.load_model_from_path, config.sampling.model_name)))

    return score

def get_standard_sampler(args, config, score, sde, ray_trafo, observation=None, filtbackproj=None, device=None):

    if args.method == 'naive':
        predictor = functools.partial(
            Euler_Maruyama_VE_sde_predictor,
            nloglik = lambda x: torch.linalg.norm(observation - ray_trafo(x)))
        sample_kwargs = {
            'num_steps': int(args.num_steps),
            'start_time_step': ceil(float(args.pct_chain_elapsed) * int(args.num_steps)),
            'batch_size': config.sampling.batch_size,
            'im_shape': ray_trafo.im_shape,
            'eps': config.sampling.eps,
            'predictor': {'aTweedy': False, 'penalty': float(args.penalty)},
            'corrector': {}
            }
    elif args.method == 'dps':
        predictor = functools.partial(
            Euler_Maruyama_VE_sde_predictor,
            nloglik = lambda x: torch.linalg.norm(observation - ray_trafo(x)))
        sample_kwargs = {
            'num_steps': int(args.num_steps),
            'batch_size': config.sampling.batch_size,
            'start_time_step': ceil(float(args.pct_chain_elapsed) * int(args.num_steps)),
            'im_shape': ray_trafo.im_shape,
            'eps': config.sampling.eps,
            'predictor': {'aTweedy': True, 'penalty': float(args.penalty)},
            'corrector': {}
            }
    elif args.method == 'dds':
        sample_kwargs = {
            'num_steps': int(args.num_steps),
            'batch_size': config.sampling.batch_size,
            'start_time_step': ceil(float(args.pct_chain_elapsed) * int(args.num_steps)),
            'im_shape': ray_trafo.im_shape,
            'eps': config.sampling.eps,
            'predictor': {'eta': 0.15},
            'corrector': {}
            }
        conj_grad_closure_partial = functools.partial(
            conj_grad_closure,
            ray_trafo=ray_trafo
            )
        predictor = functools.partial(
            decomposed_diffusion_sampling_VE_sde_predictor,
            score=score,
            sde=sde,
            rhs=ray_trafo.trafo_adjoint(observation),
            conj_grad_closure=conj_grad_closure_partial,
            cg_kwargs={'max_iter': 5, 'max_tridiag_iter': 4}
        )
    else:
        raise NotImplementedError

    corrector = None
    if args.add_corrector_step:
        corrector = functools.partial(  Langevin_VE_sde_corrector,
            nloglik = lambda x: torch.linalg.norm(observation - ray_trafo(x))   )
        sample_kwargs['corrector']['corrector_steps'] = 5
        sample_kwargs['corrector']['penalty'] = float(args.penalty)

    init_chain_fn = None
    if sample_kwargs['start_time_step'] > 0:
        init_chain_fn = functools.partial(  
        chain_simple_init,
        sde=sde,
        filtbackproj=filtbackproj,
        start_time_step=sample_kwargs['start_time_step'],
        im_shape=ray_trafo.im_shape,
        batch_size=sample_kwargs['batch_size'],
        device=device
        )

    sampler = BaseSampler(
        score=score,
        sde=sde,
        predictor=predictor,
        init_chain_fn=init_chain_fn,
        corrector=corrector,
        sample_kwargs=sample_kwargs, 
        device=config.device
        )
    
    return sampler

def get_standard_ray_trafo(config):

    if config.forward_op.trafo_name == 'simple_trafo':
        ray_trafo = SimpleTrafo(
            im_shape=(config.data.im_size, config.data.im_size), 
            num_angles=config.forward_op.num_angles)

    elif config.forward_op.trafo_name == 'walnut_trafo':
        ray_trafo = get_walnut_2d_ray_trafo(
            data_path=config.data.data_path,
            matrix_path=config.data.data_path,
            walnut_id=config.data.walnut_id,
            orbit_id=config.forward_op.orbit_id,
            angular_sub_sampling=config.forward_op.angular_sub_sampling,
            proj_col_sub_sampling=config.forward_op.proj_col_sub_sampling
            )
    else: 
        raise NotImplementedError

    return ray_trafo

def get_data_from_ground_truth(ground_truth, ray_trafo, white_noise_rel_stddev):

    ground_truth = ground_truth.unsqueeze(0) if ground_truth.ndim == 3 else ground_truth
    observation = simulate(
        x=ground_truth, 
        ray_trafo=ray_trafo,
        white_noise_rel_stddev=white_noise_rel_stddev,
        return_noise_level=False)
    filtbackproj = ray_trafo.fbp(observation)

    return ground_truth, observation, filtbackproj

def get_standard_dataset(config, ray_trafo=None):

    if config.data.name == 'DiskDistributedEllipsesDataset':
        dataset = get_disk_dist_ellipses_dataset(
        fold='test',
        im_size=config.data.im_size,
        length=config.data.val_length,
        diameter=config.data.diameter,
        max_n_ellipse=config.data.num_n_ellipse,
        device=config.device)
    elif config.data.name == 'Walnut':
        dataset = get_walnut_data(config, ray_trafo)
    elif config.data.name == 'LoDoPabCT':
        dataset = LoDoPabDatasetFromDival(im_size=config.data.im_size)
        dataset = dataset.get_testloader(batch_size=1, num_data_loader_workers=0)
    else:
        raise NotImplementedError

    return dataset

def get_standard_configs(args):

    if args.model_learned_on == 'ellipses': # score-model pre-trainined on dataset configs 
        from configs.disk_ellipses_configs import get_config
    elif args.model_learned_on == 'lodopab':
        from configs.lodopab_configs import get_config
    else:
        raise NotImplementedError
    config = get_config()

    if args.dataset == 'ellipses': 	# validation dataset configs
        from configs.disk_ellipses_configs import get_config
    elif args.dataset == 'lodopab':
        from configs.lodopab_configs import get_config
    elif args.dataset == 'walnut':
        from configs.walnut_configs import get_config
    else:
        raise NotImplementedError
    dataconfig = get_config()

    return config, dataconfig

def get_standard_path(args):

    path = './score_model/outputs/'
    path += args.model_learned_on + '_' + args.dataset

    return Path(os.path.join(path, f'{time.strftime("%d-%m-%Y-%H-%M-%S")}'))
	