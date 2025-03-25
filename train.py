'''
Modified from https://github.com/eric-mitchell/direct-preference-optimization
'''

import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
from utils import get_local_dir, disable_dropout, init_distributed, get_open_port, log_eta
import os
import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf, DictConfig
import trainers
import wandb
import json
import socket
from typing import Optional, Set
import resource
from typing import Optional, Dict, List, Union, Tuple


def worker_main(rank: int, world_size: int, config: DictConfig, policy: nn.Module, reference_model: Optional[nn.Module] = None, dynamic_params:Dict = None):
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
    if 'FSDP' in config.trainer:
        init_distributed(rank, world_size, port=config.fsdp_port)

    if config.debug:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None
    
    # logging configuration, etas
    if rank == 0 and config.wandb.enabled:
        os.environ['WANDB_CACHE_DIR'] = get_local_dir(config.local_dirs)
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.local_dirs),
            name = config.exp_name + '_group=' + str(dynamic_params['group']) + '_emstep=' + str(dynamic_params['em_iteration']),
        )
        # log eta values
        if dynamic_params and dynamic_params['group'] == 0:
            log_eta(dynamic_params['eta'], dynamic_params['em_iteration'])

    TrainerClass = getattr(trainers, config.trainer)
    print(f'Creating trainer on process {rank} with world size {world_size}')
    trainer = TrainerClass(policy, config, config.seed, config.local_run_dir, reference_model=reference_model, rank=rank, world_size=world_size, dynamic_params=dynamic_params)
    trainer.train()

    if config.num_groups > 1:
        log_numerator_gamma_group = trainer.compute_posterior().to('cpu')
        if self.rank == 0:
            dynamic_params['log_numerator_gamma'][dynamic_params['group'], :] += log_numerator_gamma_group
            print(f'updating gammas for group', dynamic_params['group'])

    # if dynamic_params['em_iteration'] % config.em_iteration_save == 0:
    if dynamic_params['em_iteration'] == dynamic_params['TOTAL_ITERATIONS'] - 1 and config.save_model==True:
        trainer.save()
    if rank == 0 and config.wandb.enabled:
        wandb.finish()

def train_weighted_dpo(config: DictConfig, dynamic_params: Dict = None):
    """Creates/initializes model(s), and kicks off worker process(es)."""
    # build policy and reference model (if any)
    print('building policy')
    model_kwargs = {'device_map': 'balanced'} if config.trainer == 'BasicTrainer' else {}
    policy_dtype = getattr(torch, config.model.policy_dtype)
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True, torch_dtype=policy_dtype, token=os.getenv("HUGGINGFACE_TOKEN"), **model_kwargs)
    disable_dropout(policy)

    if config.loss.name in {'dpo', 'ipo'}:
        print('building reference model')
        reference_model_dtype = getattr(torch, config.model.reference_dtype)
        reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True, torch_dtype=reference_model_dtype, token=os.getenv("HUGGINGFACE_TOKEN"), **model_kwargs)
        disable_dropout(reference_model)
    else:
        reference_model = None
        
    # load reference model from archive for traning dpo (if any)
    if config.model.archive is not None:
        state_dict = torch.load(config.model.archive, map_location='cpu')
        step, metrics = state_dict['step_idx'], state_dict['metrics']
        print(f'loading pre-trained weights at step {step} from {config.model.archive} with metrics {json.dumps(metrics, indent=2)}')
        policy.load_state_dict(state_dict['state'])
        if config.loss.name in {'dpo', 'ipo'}:
            reference_model.load_state_dict(state_dict['state'])
        print('loaded pre-trained weights')

    if 'FSDP' in config.trainer:
        world_size = torch.cuda.device_count()
        print('starting', world_size, 'processes for FSDP training')
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f'setting RLIMIT_NOFILE soft limit to {hard} from {soft}')
        mp.spawn(worker_main, nprocs=world_size, args=(world_size, config, policy, reference_model, dynamic_params), join=True)
    else:
        print('starting single-process worker')
        worker_main(0, 1, config, policy, reference_model, dynamic_params)
