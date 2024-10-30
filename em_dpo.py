import torch
from collections import defaultdict
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
from utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed, get_open_port, update_eta_gamma
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
from train import train_weighted_dpo
import torch

import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # If you use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# def compute_posterior(policies):
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    set_seed = 60
    set_seed(seed)
    dynamic_params = {}
    # initialize from a dirichelet distribution
    # dynamic_params['gamma'] = torch.rand(config.num_groups, config.num_users)
    # dynamic_params['gamma'] /= dynamic_params['gamma'].sum(axis=0)
    dynamic_params['gamma'] = np.random.dirichlet((10, 5), config.num_users).T #make changes for a diff no. of group
    dynamic_params['eta'] = dynamic_params['gamma'].mean(axis=1)

    # variables to keep track of EM step iterations
    dynamic_params['TOTAL_ITERATIONS'] = config.em_steps
    dynamic_params['em_iteration'] = 0

    for iter in range(dynamic_params['TOTAL_ITERATIONS']):
        dynamic_params['em_iteration'] += 1

        # intermediate calculations for the E step
        dynamic_params['log_numerator_gamma'] = torch.zeros(config.num_groups, config.num_users)

        for group in range(config.num_groups):

            print('starting a new em iteration!! current gammas and etas are:')
            print(dynamic_params['gamma'])
            print(dynamic_params['eta'])

            dynamic_params['group'] = group
            train_weighted_dpo(config, dynamic_params)
            if group == config.num_groups - 1:
                dynamic_params['gamma'], dynamic_params['eta'] = \
                    update_eta_gamma(dynamic_params['log_numerator_gamma'], dynamic_params['em_iteration'])

    # if not config.debug:
    #     wandb.finish()

if __name__ == '__main__':
    main()
