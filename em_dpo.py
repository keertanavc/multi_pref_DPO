import torch
from collections import defaultdict
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
from utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed, get_open_port
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



# def compute_posterior(policies):
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    # dict to pass around values by reference
    dynamic_params = {}
    # initially all users are equally likely to be from any subgroup
    dynamic_params['weights_mat'] = torch.ones(config.num_groups, config.num_users) * (1 / config.num_groups)
    # ensemble of policies
    dynamic_params['policies'] = []
    for group in range(config.num_groups):
        print(group)
        dynamic_params['group'] = group
        train_weighted_dpo(config, dynamic_params)
        print(dynamic_params)


    # create function to calculate posterior
    # update weights and loop to step 3
    # return weights and policies

if __name__ == '__main__':
    main()
