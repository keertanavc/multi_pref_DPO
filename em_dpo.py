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


OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))

# def compute_posterior(policies):
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    # initially all users are equally likely to be from any subgroup
    weights_mat = torch.ones(config.num_groups, config.num_users) * (1 / config.num_groups)
    # defaultdict(lambda: [1/config.num_groups] * config.num_groups)
    # weights_init = defaultdict(lambda: [1/n] * n)
    policies = []
    for group in range(config.num_groups):
        # passing weights by referenceß
        config['weights_mat'] = weights_mat
        config['group'] = group
        # policies.append(train_weighted_dpo(config))
        print(train_weighted_dpo(config))
        # weight_init = compute_posterior(policies)

        # posterior_k = log(eta_k)


    # create function to calculate posterior
    # update weights and loop to step 3
    # return weights and policies

if __name__ == '__main__':
    main()
