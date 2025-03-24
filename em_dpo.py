'''
Modified from https://github.com/eric-mitchell/direct-preference-optimization
'''

import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import torch.multiprocessing as mp
import transformers
import pandas as pd

from train import train_weighted_dpo
from utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed, get_open_port, update_eta_gamma

import os
import hydra
import trainers
import wandb
import json
import socket
from typing import Optional, Set
import resource
import random
from omegaconf import OmegaConf, DictConfig
from collections import defaultdict


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    '''Main entry point to the program. Validates config. Initiates EM-DPO iterations.'''
    # Resolves config interpolations, validates required keys, adjusts eval frequency, 
    # asserts group size, handles FSDP port, saves config, and sets cache directory.    
    OmegaConf.resolve(config)
    print('run directory')
    print(config.local_run_dir)
    
    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")
    
    if config.eval_every % config.batch_size != 0:
        print('WARNING: eval_every must be divisible by batch_size')
        print('Setting eval_every to', config.eval_every - config.eval_every % config.batch_size)
        config.eval_every = config.eval_every - config.eval_every % config.batch_size
    # if config.num_groups > 1:
        # assert config.num_groups == len(config.dirichlet_alpha)

    print(OmegaConf.to_yaml(config))

    if 'FSDP' in config.trainer and config.fsdp_port is None:
        free_port = get_open_port()
        print('no FSDP port specified; using open port for FSDP:', free_port)
        config.fsdp_port = free_port

    config_path = os.path.join(config.local_run_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)
    os.environ['XDG_CACHE_HOME'] = get_local_dir(config.local_dirs)

    # parameters dynamically updated for the algorithm
    dynamic_params = {}

    if config.num_groups == 1:
        # all weights = 1 if running regular DPO or SFT
        dynamic_params['gamma'] = np.ones((config.num_users, 1)) 
        dynamic_params['eta'] = np.array([1])
    elif config.num_groups > 1:
        # # initialize from a skewed dirichelet distribution to ensure we don't get stuck in a saddle point for EM-DPO
        # np.random.seed(config.seed + 123)
        # dynamic_params['gamma'] = np.random.dirichlet(config.dirichlet_alpha, config.num_users).T
        # dynamic_params['gamma'] = torch.tensor(dynamic_params['gamma'])
        # or just use cluster assignment as the starting point
        dynamic_params['gamma'] = np.zeros((config.num_groups, config.num_users)) 
        if 'imdb' in config.datasets[0]:
            df = pd.read_csv("hf://datasets/keertanavc/imdb_sentiment-grammar_indexed/" + "train.csv")
        elif 'globalopinion' in config.datasets[0]:
            df = pd.read_csv("hf://datasets/keertanavc/globalopinionv4/" + "train.csv")
        for human, cluster in df.groupby('human_label')['cluster'].unique().items():
            dynamic_params['gamma'][cluster, human] = 1
        dynamic_params['gamma'] = torch.tensor(dynamic_params['gamma'])
        dynamic_params['eta'] = dynamic_params['gamma'].mean(axis=1)
        # OR initialize according to cluster DPO

    # variables to keep track of EM step iterations
    dynamic_params['TOTAL_ITERATIONS'] = config.em_steps
    dynamic_params['em_iteration'] = 0

    # Running EM-DPO loop
    while dynamic_params['em_iteration'] < dynamic_params['TOTAL_ITERATIONS']:
        # intermediate calculations for the E step
        dynamic_params['log_numerator_gamma'] = torch.zeros(config.num_groups, config.num_users)
        print('starting a new em iteration!! current gammas and etas are:')
        # print(dynamic_params['gamma'])
        print(dynamic_params['eta'])
        if config.num_groups > 1:
            gamma_filename = 'checkpoints/gamma_iter=' + str(dynamic_params['em_iteration']) + '_seed=' + str(config.seed) + '_exp='+ config.exp_name + '.npy'
            np.save(gamma_filename, np.array(dynamic_params['gamma']))
            eta_filename = 'checkpoints/eta_iter=' + str(dynamic_params['em_iteration']) + '_seed=' + str(config.seed) + '_exp='+ config.exp_name +'.npy'
            np.save(eta_filename, np.array(dynamic_params['eta']))

        for group in range(config.num_groups):
            dynamic_params['group'] = group
            train_weighted_dpo(config, dynamic_params)
            if group == config.num_groups - 1:
                new_gamma, new_eta = \
                    update_eta_gamma(dynamic_params['log_numerator_gamma'], dynamic_params['em_iteration'])

        dynamic_params['gamma'] = new_gamma
        dynamic_params['eta'] = new_eta
        print('new gamma and eta are:')
        print(dynamic_params['gamma'])
        print(dynamic_params['eta'])

        # update iteration count
        dynamic_params['em_iteration'] += 1
        

if __name__ == '__main__':
    main()
