import os
import time
import torch
import functools
import yaml
import argparse

from omegaconf import OmegaConf
from math import ceil
from pathlib import Path
from torch.utils.data import TensorDataset

from .sde import VESDE, VPSDE, DDPM, _SCORE_PRED_CLASSES, _EPSILON_PRED_CLASSES
from ..third_party_models import UNetModel
from ..dataset import EllipseDatasetFromDival, get_disk_dist_ellipses_dataset, get_walnut_data, AAPMDataset
from ..physics import SimpleTrafo, get_walnut_2d_ray_trafo, simulate
from ..samplers import (BaseSampler, Euler_Maruyama_sde_predictor, 
    chain_simple_init, decomposed_diffusion_sampling_sde_predictor, 
    adapted_ddim_sde_predictor, tv_loss, _adapt, _score_model_adpt, Ancestral_Sampling)

def get_standard_score(config, sde, use_ema, load_model=True):

    score = create_model(**dict(config.model))
    if load_model:
        score.load_state_dict(torch.load(config.ckpt_path))
        print(f'Model ckpt loaded from {config.ckpt_path}')
    score.convert_to_fp32()
    score.dtype = torch.float32

    return score 

def create_model(
    image_size,
    num_channels,
    in_channels, 
    out_channels,
    num_res_blocks,
    channel_mult='',
    learn_sigma=False,
    use_checkpoint=False,
    attention_resolutions='16',
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    **kwargs
):
    if channel_mult == '':
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256 or image_size == 320:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f'unsupported image size: {image_size}')
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(','))

    attention_ds = []
    for res in attention_resolutions.split(','):
        attention_ds.append(image_size // int(res))
        
    return UNetModel(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=out_channels, #(1 if not learn_sigma else 2),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=None, 
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )


def get_standard_sde(config):

    _sde_classname = config.sde.type.lower()
    if _sde_classname == 'vesde':
        sde = VESDE(
        sigma_min=config.sde.sigma_min, 
        sigma_max=config.sde.sigma_max
        )
    elif _sde_classname == 'vpsde':
        sde = VPSDE(
        beta_min=config.sde.beta_min, 
        beta_max=config.sde.beta_max
        )
    elif _sde_classname== 'ddpm':
        sde = DDPM(
        beta_min=config.sde.beta_min, 
        beta_max=config.sde.beta_max, 
        num_steps=config.sde.num_steps
        )
    else:
        raise NotImplementedError

    return sde

def get_standard_sampler(args, config, score, sde, ray_trafo, observation=None, filtbackproj=None, device=None):

    _sampler_funame = args.method.lower()
    _shape = ray_trafo.im_shape if not hasattr(ray_trafo, 'resize') else ray_trafo.resize.shape
    if any([isinstance(sde, classname) for classname in _SCORE_PRED_CLASSES]):
        if _sampler_funame == 'naive':
            predictor = functools.partial(
                Euler_Maruyama_sde_predictor,
                nloglik = lambda x: torch.linalg.norm(observation - ray_trafo(x)))
            sample_kwargs = {
                'num_steps': int(args.num_steps),
                'start_time_step': ceil(float(args.pct_chain_elapsed) * int(args.num_steps)),
                'batch_size': config.sampling.batch_size,
                'im_shape': [1, _shape],
                'eps': config.sampling.eps,
                'predictor': {'aTweedy': False, 'penalty': float(args.penalty)},
                }
        elif _sampler_funame == 'dps':
            predictor = functools.partial(
                Euler_Maruyama_sde_predictor,
                nloglik = lambda x: torch.linalg.norm(observation - ray_trafo(x)))
            sample_kwargs = {
                'num_steps': int(args.num_steps),
                'batch_size': config.sampling.batch_size,
                'start_time_step': ceil(float(args.pct_chain_elapsed) * int(args.num_steps)),
                'im_shape': [1, _shape],
                'eps': config.sampling.eps,
                'predictor': {'aTweedy': True, 'penalty': float(args.penalty)},
                }
        elif _sampler_funame == 'dds':
            sample_kwargs = {
                'num_steps': int(args.num_steps),
                'batch_size': config.sampling.batch_size,
                'start_time_step': ceil(float(args.pct_chain_elapsed) * int(args.num_steps)),
                'im_shape': [1, _shape],
                'eps': config.sampling.eps,
                'predictor': {'eta': float(args.eta), 'gamma': float(args.gamma), 'use_simplified_eqn': True, 'ray_trafo': ray_trafo},
                }
            predictor = functools.partial(
                decomposed_diffusion_sampling_sde_predictor,
                score=score,
                sde=sde,
                rhs=ray_trafo.trafo_adjoint(observation),
                cg_kwargs={'max_iter': int(args.cg_iter)}
            )
        else:
            raise NotImplementedError(_sampler_funame)

    
    elif any([isinstance(sde, classname) for classname in _EPSILON_PRED_CLASSES]):
        if _sampler_funame == 'naive':
            raise NotImplementedError(_sampler_funame)
        elif _sampler_funame == 'dps':
            predictor = functools.partial(
                Ancestral_Sampling,
                nloglik = lambda x: torch.linalg.norm(observation - ray_trafo(x))) 
            sample_kwargs = {
                'num_steps': int(args.num_steps),
                'batch_size': config.sampling.batch_size,
                'start_time_step': ceil(float(args.pct_chain_elapsed) * int(args.num_steps)),
                'travel_length': config.sampling.travel_length,
                'travel_repeat': config.sampling.travel_repeat,
                'im_shape': [1, *_shape],
                'predictor': {'penalty': float(args.penalty)},
                'early_stopping_pct': float(args.early_stopping_pct)
                }

        elif _sampler_funame == 'dds':
            sample_kwargs = {
                'num_steps': int(args.num_steps),
                'batch_size': config.sampling.batch_size,
                'start_time_step': ceil(float(args.pct_chain_elapsed) * int(args.num_steps)),
                'im_shape': [config.model.in_channels, *_shape],
                'eps': config.sampling.eps,
                'travel_length': config.sampling.travel_length,
                'travel_repeat': config.sampling.travel_repeat, 
                'predictor': {'eta': float(args.eta), 'gamma': float(args.gamma), 'use_simplified_eqn': True, 'ray_trafo': ray_trafo},
                }
            predictor = functools.partial(
                decomposed_diffusion_sampling_sde_predictor,
                score=score,
                sde=sde,
                rhs=ray_trafo.trafo_adjoint(observation),
                cg_kwargs={'max_iter': int(args.cg_iter)}
            )
        else:
            raise NotImplementedError(_sampler_funame)

        assert ceil(float(args.pct_chain_elapsed) * int(args.num_steps)) == 0
        corrector, init_chain_fn = None, None

    sampler = BaseSampler(
        score=score,
        sde=sde,
        predictor=predictor,         
        init_chain_fn=init_chain_fn,
        sample_kwargs=sample_kwargs, 
        device=config.device
        )
    
    return sampler

def get_standard_adapted_sampler(args, config, score, sde, ray_trafo, observation=None, device=None, complex_y=False):

    if args.method.lower() == 'dds':
        try:
            eps = config.sampling.eps 
        except AttributeError:
            eps = 0.
        _shape = ray_trafo.im_shape if not hasattr(ray_trafo, 'resize') else ray_trafo.resize.shape
        sample_kwargs = {
            'num_steps': int(args.num_steps),
            'batch_size': config.sampling.batch_size,
            'start_time_step': 0,
            'im_shape': [config.model.in_channels, *_shape],
            'eps': eps,
            'adapt_freq': int(args.adapt_freq), 
            'predictor': {
                'eta': float(args.eta), 
                'use_simplified_eqn': True, 
                'gamma': float(args.gamma),
                'ray_trafo': ray_trafo 
                },
            'corrector': {},
            'early_stopping_pct': float(args.early_stopping_pct)
            }
        adpt_kwargs = None
        if args.adaptation == 'lora':
            adpt_kwargs = {
            'include_blocks': args.lora_include_blocks, 
            'r': int(args.lora_rank)
            }
        _score_model_adpt(score, impl=args.adaptation, adpt_kwargs=adpt_kwargs)
        lloss_fn = lambda x: torch.mean(
            (ray_trafo(x) - observation).pow(2))  + float(args.tv_penalty) * tv_loss(x)
        adapt_fn = functools.partial(
            _adapt, score=score, sde=sde, loss_fn=lloss_fn, num_steps=int(args.num_optim_step), lr=float(args.lr))
        predictor = functools.partial(
        adapted_ddim_sde_predictor, score=score, 
                sde=sde, 
                adapt_fn=adapt_fn, 
                add_cg=args.add_cg,
                dc_type=args.dc_type,
                rhs=ray_trafo.trafo_adjoint(observation),
                cg_kwargs={'max_iter': int(args.cg_iter)}
            )
    else:
        raise NotImplementedError


    if any([isinstance(sde, classname) for classname in _EPSILON_PRED_CLASSES]):
        try:
            travel_length = config.sampling.travel_length
            travel_repeat = config.sampling.travel_repeat
        except AttributeError:
            travel_length = config.time_travel.travel_length
            travel_repeat = config.time_travel.travel_repeat

        sample_kwargs.update({
            'travel_length': travel_length,
            'travel_repeat': travel_repeat,
            }
        )

    sampler = BaseSampler(
        score=score, 
        sde=sde,
        predictor=predictor,         
        sample_kwargs=sample_kwargs, 
        device=config.device
        )
    
    return sampler

def get_standard_ray_trafo(config):

    if config.forward_op.trafo_name.lower() == 'simple_trafo':
        ray_trafo = SimpleTrafo(
            im_shape=(config.data.im_size, config.data.im_size), 
            num_angles=config.forward_op.num_angles,
            impl=config.forward_op.impl
            )

    elif config.forward_op.trafo_name.lower() == 'walnut_trafo':
        ray_trafo = get_walnut_2d_ray_trafo(
            data_path=config.data.data_path,
            matrix_path=config.data.data_path,
            walnut_id=config.data.walnut_id,
            orbit_id=config.forward_op.orbit_id,
            angular_sub_sampling=config.forward_op.angular_sub_sampling,
            proj_col_sub_sampling=config.forward_op.proj_col_sub_sampling, 
            new_shape=config.data.new_shape
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
    if config.data.name.lower() == 'DiskDistributedEllipsesDataset'.lower():
        if config.data.part == 'val' and config.data.im_size == 256:
            ellipse_path = 'dataset/disk_ellipses_val_256.pt'
            print('Load pre-saved ellipses dataset from ', ellipse_path)
            x_ellipse = torch.load(ellipse_path)
            dataset = TensorDataset(x_ellipse)
        if config.data.part == 'test' and config.data.im_size == 256:
            ellipse_path = 'dataset/disk_ellipses_test_256.pt'
            print('Load pre-saved ellipses dataset from ', ellipse_path)
            x_ellipse = torch.load(ellipse_path)
            dataset = TensorDataset(x_ellipse)
        else:
            dataset = get_disk_dist_ellipses_dataset(
            fold='test',
            im_size=config.data.im_size,
            length=config.data.val_length,
            diameter=config.data.diameter,
            max_n_ellipse=config.data.num_n_ellipse,
            device=config.device)
    elif config.data.name.lower() == 'Walnut'.lower():
        dataset = get_walnut_data(config, ray_trafo)
      
    elif config.data.name.lower() == "aapm":
        dataset = AAPMDataset(part=config.data.part, base_path=config.data.base_path)
    else:
        raise NotImplementedError

    return dataset

def get_standard_train_dataset(config): 

    if config.data.name.lower() == 'EllipseDatasetFromDival'.lower():
        ellipse_dataset = EllipseDatasetFromDival(impl='astra_cuda')
        train_dl = ellipse_dataset.get_trainloader(
            batch_size=config.training.batch_size, 
            num_data_loader_workers=0
        )
    elif config.data.name.lower() == 'DiskDistributedEllipsesDataset'.lower():
        if config.data.num_n_ellipse > 1:
            dataset = get_disk_dist_ellipses_dataset(
                fold='train',
                im_size=config.data.im_size, 
                length=config.data.length,
                diameter=config.data.diameter,
                max_n_ellipse=config.data.num_n_ellipse,
                device=config.device
            )
        else:
            dataset = get_one_ellipses_dataset(
                fold='train',
                im_size=config.data.im_size,
                length=config.data.length,
                diameter=config.data.diameter,
                device=config.device
            )
        train_dl = torch.utils.data.DataLoader(dataset, batch_size=config.data.batch_size, shuffle=False, num_workers=1)
   
    
    return train_dl

def get_standard_configs(args, base_path):
    try:
        _sde_classname = args.sde.lower()
        version = 'version_{:02d}'.format(int(args.version))
    except AttributeError:
        pass 
    if args.model_learned_on.lower() == 'ellipses': 
        path = os.path.realpath(__file__).split('/src')[0]
        with open(os.path.join(path, 'ellipses_configs/ddpm', 'Ellipse256.yml'), 'r') as stream:
            config = yaml.load(stream, Loader=yaml.UnsafeLoader)
            config['ckpt_path'] = args.load_path
            config = OmegaConf.create(config)
    elif args.model_learned_on.lower() == 'aapm':
        path = os.path.realpath(__file__).split('/src')[0]
        with open(os.path.join(path, 'aapm_configs/ddpm', 'AAPM256.yml'), 'r') as stream:
            config = yaml.load(stream, Loader=yaml.UnsafeLoader)
            config['ckpt_path'] = args.load_path
            config = OmegaConf.create(config)
    else:
        raise NotImplementedError

    if args.dataset.lower() == 'ellipses': 	# validation dataset configs
        from configs.disk_ellipses_configs import get_config
    elif args.dataset.lower() == 'walnut':
        from configs.walnut_configs import get_config
    elif args.dataset.lower() == 'aapm':
        from configs.aapm_configs import get_config
    else:
        raise NotImplementedError
    dataconfig = get_config(args)

    return config, dataconfig

def get_standard_dataset_configs(args):
    
    if args.dataset.lower() == 'ellipses': 	# validation dataset configs
        from configs.disk_ellipses_configs import get_config
    elif args.dataset.lower() == 'walnut':
        from configs.walnut_configs import get_config
    elif args.dataset.lower() == 'aapm':
        from configs.aapm_configs import get_config
    else:
        raise NotImplementedError
    dataconfig = get_config(args)

    return dataconfig

def get_standard_path(args, 
                    run_type=None, 
                    path='', 
                    data_part=None):

    path = './outputs/'
    # path = "/localdata/ AlexanderDenker/new_run/outputs"
    path = os.path.join(path,
                    args.model_learned_on + '_' + args.dataset)

    if data_part is not None:
        path = os.path.join(path, data_part)
    
    if run_type == 'adapt':
        path = os.path.join(path,
                    'adapt',
                    'adaptation=' + args.adaptation, 
                    'dc_type=' + str(args.dc_type),
                    'num_steps=' + str(args.num_steps),
                    'num_optim_step=' + str(args.num_optim_step),
                    'tv_penalty' + str(args.tv_penalty))
    elif run_type == 'dds':
        path = os.path.join(path, 
                    run_type,
                    'num_steps=' + str(args.num_steps), 
                    'cg_iter=' + str(args.cg_iter),
                    'gamma=' + str(args.gamma))
    elif run_type == 'dps':
        path = os.path.join(path, 
                    run_type,
                    'num_steps=' + str(args.num_steps), 
                    'penalty=' + str(args.penalty))        
    else:
        pass 

    if not args.dataset == 'walnut':
        if not args.stddev == None:
            path = os.path.join(path, 'noise_level=' + str(args.stddev))

    return Path(os.path.join(path, f'{time.strftime("%d-%m-%Y-%H-%M-%S")}'))

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace