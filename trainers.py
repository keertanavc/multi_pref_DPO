'''
Modified from https://github.com/eric-mitchell/direct-preference-optimization
'''

import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import LogSoftmax
from torch.nn import Softmax
import transformers
from omegaconf import OmegaConf, DictConfig

import torch.distributed as dist

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import tensor_parallel as tp
import contextlib

from preference_datasets import get_batch_iterator
from utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir,
    get_local_run_dir,
    init_distributed,
    disable_dropout,
    get_open_port
)
# from utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed, get_open_port

import numpy as np
import pandas as pd
import wandb
import tqdm

import random
import os
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple
from copy import deepcopy
import sys

OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))

def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float,
                    label_smoothing: float = 0.0,
                    ipo: bool = False,
                    reference_free: bool = False,
                    weight: torch.FloatTensor = torch.tensor([])) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a bxtch of policy and reference model log probabilities"""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if ipo:
        losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
        if weight.numel() > 0: ###
            losses *= weight ###

    # note that we're not weighing the reward models, only the losses
    chosen_rewards =  beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    return losses, chosen_rewards, rejected_rewards


def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits."""
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor."""
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    return concatenated_batch


class BasicTrainer(object):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1, dynamic_params:Dict = None):
        """A trainer for a language model, supporting either SFT or DPO training."""
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir

        tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
        rank0_print(f'Loading tokenizer {tokenizer_name_or_path}')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs), token=os.getenv("HUGGINGFACE_TOKEN"))
        # self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.policy = policy
        self.reference_model = reference_model

        # weighted DPO parameters
        self.dynamic_params = dynamic_params
        self.weights_dict = {}
        self.weights_dict_allones = {}
        self.num_users = config.num_users
        self.num_groups = config.num_groups
        self.group = self.dynamic_params['group']
        self.em_iteration = self.dynamic_params['em_iteration']
        for i in range(config.num_users):
            if self.num_groups == 1:
                self.weights_dict[i] = self.dynamic_params['gamma'][i]
            else:
                self.weights_dict[i] = self.dynamic_params['gamma'][self.group, i]
            self.weights_dict_allones[i] = 1
        if self.num_groups > 1:
            self.gamma = self.dynamic_params['gamma'].to(self.rank)
            self.eta = self.dynamic_params['eta'].to(self.rank)
        
        # loading training dataset
        data_iterator_kwargs = dict(
            tokenizer=self.tokenizer,
            shuffle=True,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            sft_mode=config.loss.name == 'sft',
            names=config.datasets,
        )
        self.train_iterator = get_batch_iterator(**data_iterator_kwargs, 
                                                weights_dict=self.weights_dict, 
                                                split='train', 
                                                n_epochs=config.n_epochs, 
                                                n_examples=config.n_examples, 
                                                batch_size=config.batch_size, 
                                                silent=rank != 0, 
                                                cache_dir=get_local_dir(config.local_dirs))
        rank0_print(f'Loaded train data iterator')
        
        # loading dataset for posterior compuation
        if self.num_groups > 1:
            self.posterior_batches = get_batch_iterator(**data_iterator_kwargs, 
                                                        weights_dict=self.weights_dict_allones, 
                                                        split='train', 
                                                        n_epochs=1, 
                                                        n_examples=config.n_examples,  
                                                        batch_size=config.batch_size, 
                                                        silent=rank != 0, 
                                                        cache_dir=get_local_dir(config.local_dirs))
            # self.posterior_batches = list(self.posterior_batches)
        rank0_print(f'Loaded train data iterator for posterior computing')

        # loading evaluation dataset
        self.eval_iterator = []
        self.eval_batches = []
        if 'imdb' in config.datasets[0] and self.config.loss.name in {'dpo', 'ipo'}:
            self.eval_data_names = ['imdb_grammar', 'imdb_sentiment']
        elif 'globalopinion' in config.datasets[0] and self.config.loss.name in {'dpo', 'ipo'}:
            self.eval_data_names = ['globalopinion_in', 'globalopinion_mx', 'globalopinion_pk', 'globalopinion_br']
        else:
            self.eval_data_names = [config.datasets[0]]
        for dataset_name in self.eval_data_names:
            data_iterator_kwargs['names'] = [dataset_name]
            self.eval_iterator.append(get_batch_iterator(**data_iterator_kwargs, 
                                                        split='test', 
                                                        n_examples=config.n_eval_examples, 
                                                        batch_size=config.eval_batch_size, 
                                                        silent=rank != 0, 
                                                        cache_dir=get_local_dir(config.local_dirs),
                                                        weights_dict=self.weights_dict_allones))
            
        self.eval_batches = [list(iterator) for iterator in self.eval_iterator]
        rank0_print(f'Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}')

    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""

        # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
        ctx = lambda: (FSDP.summon_full_params(self.policy, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
        with ctx():
            policy_output = self.policy.generate(
                batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        if self.config.loss.name in {'dpo', 'ipo'}:
            ctx = lambda: (FSDP.summon_full_params(self.reference_model, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
            with ctx():
                reference_output = self.reference_model.generate(
                    batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
        policy_output = all_gather_if_needed(policy_output, self.rank, self.world_size)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        if self.config.loss.name in {'dpo', 'ipo'}:
            reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
            reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
            reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)
        else:
            reference_output_decoded = []

        return policy_output_decoded, reference_output_decoded

    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
           We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = concatenated_inputs(batch)
        all_logits = model(concatenated_batch['concatenated_input_ids'], attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
        all_logps = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=False)
        chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]]
        rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]
        return chosen_logps, rejected_logps


    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], loss_config: DictConfig, train=True, eval_data_name=None, weighted_loss=True):
        """Compute the SFT or DPO loss and other metrics for the given batch of inputs."""
        metrics = {}
        # train_test = 'train' if train else 'eval'
        if eval_data_name is None:
            train_test = 'train' if train else 'eval'
        elif 'imdb' in eval_data_name:
            if train:
                train_test = 'train' 
            elif eval_data_name == 'imdb_grammar':
                train_test = 'eval_gramar' 
            elif eval_data_name == 'imdb_sentiment':
                train_test = 'eval_sentiment' 
            else:
                train_test = 'eval' 
        elif 'globalopinion' in eval_data_name:
            if train:
                train_test = 'train'
            elif eval_data_name == 'globalopinion_in':
                train_test = 'eval_in'
            elif eval_data_name == 'globalopinion_mx':
                train_test = 'eval_mx'
            elif eval_data_name == 'globalopinion_pk':
                train_test = 'eval_pk'
            elif eval_data_name == 'globalopinion_br':
                train_test = 'eval_br'
            else:
                train_test = 'eval'
        else:
            raise ValueError(f'unknown evaluation datasetname {eval_data_name}')

        if loss_config.name in {'dpo', 'ipo'}:
            policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy, batch)
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(self.reference_model, batch)

            if loss_config.name == 'dpo':
                loss_kwargs = {'beta': loss_config.beta, 'reference_free': loss_config.reference_free, 'label_smoothing': loss_config.label_smoothing, 'ipo': False}
            elif loss_config.name == 'ipo':
                loss_kwargs = {'beta': loss_config.beta, 'ipo': True}
            else:
                raise ValueError(f'unknown loss {loss_config.name}')
            ###
            if 'weight' in batch and weighted_loss:
                loss_kwargs['weight'] = batch['weight']
            ###

            indices = batch['index']
            indices = all_gather_if_needed(indices, self.rank, self.world_size)
            metrics[f'rewards_{train_test}/index'] = indices.cpu().numpy().toli
            
            losses, chosen_rewards, rejected_rewards = preference_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, **loss_kwargs)

            reward_accuracies = (chosen_rewards > rejected_rewards).float()
            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)
            
            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()

            policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()

        elif loss_config.name == 'sft':
            policy_chosen_logits = self.policy(batch['chosen_input_ids'], attention_mask=batch['chosen_attention_mask']).logits.to(torch.float32)
            policy_chosen_logps = _get_batch_logps(policy_chosen_logits, batch['chosen_labels'], average_log_prob=False)

            losses = -policy_chosen_logps

        policy_chosen_logps = all_gather_if_needed(policy_chosen_logps.detach(), self.rank, self.world_size)
        metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()

        all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
        metrics[f'loss/{train_test}'] = all_devices_losses.cpu().numpy().tolist()

        return losses.mean(), metrics, losses

    def train(self):
        """Begin either SFT or DPO training, with periodic evaluation."""

        rank0_print(f'Using {self.config.optimizer} optimizer')
        self.optimizer = getattr(torch.optim, self.config.optimizer)(self.policy.parameters(), lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.warmup_steps + 1)))

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.config.loss.name in {'dpo', 'ipo'}:
            self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None

        for batch in self.train_iterator:
            #### BEGIN EVALUATION ####
            # sys.stdout.flush()
            # if self.example_counter % self.config.eval_every == 0 and (self.example_counter > 0 or self.config.do_first_eval):
            #     rank0_print(f'Running evaluation after {self.example_counter} train examples')
            #     self.policy.eval()
            #     for i in range(len(self.eval_batches)): 
            #         use_eval_batches = self.eval_batches[i] 
            #         if self.eval_data_names is not None:
            #             eval_data_name = self.eval_data_names[i] 
            #         else:
            #             eval_data_name = None
            #         all_eval_metrics = defaultdict(list)

            #         for eval_batch in (tqdm.tqdm(use_eval_batches, desc='Computing eval metrics') if self.rank == 0 else use_eval_batches):
            #             local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
            #             with torch.no_grad():
            #                 _, eval_metrics, _ = self.get_batch_metrics(local_eval_batch, self.config.loss, train=False, eval_data_name=eval_data_name) ###

            #             for k, v in eval_metrics.items():
            #                 all_eval_metrics[k].extend(v)
                            
            #         if self.config.loss.name in {'dpo', 'ipo'}:
            #             eval_filename = 'eval='+ eval_data_name + '_train='+ str(self.config.datasets[0]) + '_iteration='+ str(self.dynamic_params['em_iteration']) + '_group='+ str(self.dynamic_params['group']) + '.csv'
            #             eval_path = 'eval_csv/' + eval_filename
            #             pd.DataFrame(all_eval_metrics).to_csv(eval_path, index=False)

            #         mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
            #         rank0_print(f'eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}')
            #         if self.config.wandb.enabled and self.rank == 0:
            #             wandb.log(mean_eval_metrics, step=self.example_counter)

                    # if self.example_counter > 0:
                    #     if self.config.debug:
                    #         rank0_print('skipping save in debug mode')
                    #     else:
                    #         output_dir = os.path.join(self.run_dir, f'step-{self.example_counter}')
                    #         rank0_print(f'creating checkpoint to write to {output_dir}...')
                    #         self.save(output_dir, mean_eval_metrics)
            #### END EVALUATION ####

            #### BEGIN TRAINING ####
            self.policy.train()

            start_time = time.time()
            batch_metrics = defaultdict(list)
            for microbatch_idx in range(self.config.gradient_accumulation_steps):
                global_microbatch = slice_and_move_batch_for_device(batch, microbatch_idx, self.config.gradient_accumulation_steps, self.rank)
                local_microbatch = slice_and_move_batch_for_device(global_microbatch, self.rank, self.world_size, self.rank)
                loss, metrics, _ = self.get_batch_metrics(local_microbatch, self.config.loss, train=True)
                (loss / self.config.gradient_accumulation_steps).backward()

                for k, v in metrics.items():
                    batch_metrics[k].extend(v)

            grad_norm = self.clip_gradient()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)
            batch_metrics['grad_norm'].append(grad_norm)

            self.batch_counter += 1
            self.example_counter += self.config.batch_size

            if last_log is None or time.time() - last_log > self.config.minimum_log_interval_secs:
                mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
                mean_train_metrics['counters/examples'] = self.example_counter
                mean_train_metrics['counters/updates'] = self.batch_counter
                rank0_print(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()
            else:
                rank0_print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')
            #### END TRAINING ####
        
        ### BEGIN EVALUATION ###
        self.policy.eval()
        for i in range(len(self.eval_batches)): 
            sys.stdout.flush()
            use_eval_batches = self.eval_batches[i] 
            if self.eval_data_names is not None:
                eval_data_name = self.eval_data_names[i] 
            else:
                eval_data_name = None
            all_eval_metrics = defaultdict(list)

            for eval_batch in (tqdm.tqdm(use_eval_batches, desc='Computing eval metrics') if self.rank == 0 else use_eval_batches):
                local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                with torch.no_grad():
                    _, eval_metrics, _ = self.get_batch_metrics(local_eval_batch, self.config.loss, train=False, eval_data_name=eval_data_name) ###

                for k, v in eval_metrics.items():
                    all_eval_metrics[k].extend(v)
                    
            if self.config.loss.name in {'dpo', 'ipo'} and self.rank == 0:
                eval_filename = 'eval='+ eval_data_name + '_exp='+ str(self.config.exp_name) + '_iteration='+ str(self.dynamic_params['em_iteration']) + '_group='+ str(self.dynamic_params['group']) + '_seed=' + str(self.config.seed) +'.csv'
                eval_path = 'eval_csv/' + eval_filename
                pd.DataFrame(all_eval_metrics).to_csv(eval_path, index=False)

            mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
            rank0_print(f'eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}')
            if self.config.wandb.enabled and self.rank == 0:
                wandb.log(mean_eval_metrics, step=self.example_counter)
        ### END EVALUATION ###

        if self.dynamic_params:
            if self.group == self.num_groups - 1 and self.rank == 0:
                rank0_print('m step completed for iteration', self.dynamic_params['em_iteration'])


    def compute_posterior(self):
        ''' Computing values required from current group's policy for the E-step'''
        self.policy.eval()
        local_value =  torch.zeros(1, self.num_users).to(self.rank)
        for batch in self.posterior_batches:
            local_batch = slice_and_move_batch_for_device(batch, self.rank, self.world_size, self.rank)
            with torch.no_grad():
                _, _, losses = self.get_batch_metrics(local_batch, self.config.loss, train=True, weighted_loss=False)
                labels = local_batch['human_label'].to(self.rank)
                for i in range(len(losses)):
                    label = labels[i]
                    losses = losses.to(self.rank)
                    local_value[0, label] -= losses[i]
        global_value = all_gather_if_needed(local_value, self.rank, self.world_size)
        global_value = torch.sum(global_value, axis = 0)
        ret_val = torch.log(torch.tensor(self.eta[self.group])) + global_value
        return ret_val

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of a non-FSDP policy."""
        return torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm).item()

    def write_state_dict(self, step: int, state: Dict[str, torch.Tensor], metrics: Dict, filename: str, dir_name: Optional[str] = None):
        """Write a checkpoint to disk."""
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, f'LATEST')

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f'writing checkpoint to {output_path}...')
        torch.save({
            'step_idx': step,
            'state': state,
            'metrics': metrics if metrics is not None else {},
        }, output_path)

    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        """Save policy, optimizer, and scheduler state to disk."""

        output_dir = os.path.join(self.run_dir, f'group-{self.group}')

        policy_state_dict = self.policy.state_dict()
        policy_filename = 'group' + str(self.group) + '_emiteration' + str(self.em_iteration) + '_policy.pt'
        self.write_state_dict(self.example_counter, policy_state_dict, metrics, policy_filename, output_dir)
        del policy_state_dict

        optimizer_filename = 'group' + str(self.group) + '_emiteration' + str(self.em_iteration) + '_optimizer.pt'
        optimizer_state_dict = self.optimizer.state_dict()
        self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, optimizer_filename, output_dir)
        del optimizer_state_dict

        scheduler_filename = 'group' + str(self.group) + '_emiteration' + str(self.em_iteration) + '_scheduler.pt'
        scheduler_state_dict = self.scheduler.state_dict()
        self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, scheduler_filename, output_dir)


class FSDPTrainer(BasicTrainer):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1, dynamic_params: Optional[Dict] = None):
        """A trainer subclass that uses PyTorch FSDP to shard the model across multiple GPUs.

           This trainer will shard both the policy and reference model across all available GPUs.
           Models are sharded at the block level, where the block class name is provided in the config.
        """

        super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size, dynamic_params)
        assert config.model.block_name is not None, 'must specify model.block_name (e.g., GPT2Block or GPTNeoXLayer) for FSDP'

        wrap_class = get_block_class_from_model(policy, config.model.block_name)
        model_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={wrap_class},)

        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False
        )

        rank0_print('Sharding policy...')
        mp_dtype = getattr(torch, config.model.fsdp_policy_mp) if config.model.fsdp_policy_mp is not None else None
        policy_mp_policy = MixedPrecision(param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype)
        self.policy = FSDP(policy, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy)

        if config.activation_checkpointing:
            rank0_print('Attempting to enable activation checkpointing...')
            try:
                # use activation checkpointing, according to:
                # https://pytorch.org/blog/scaling-multimodal-foundation-models-in-torchmultimodal-with-pytorch-distributed/
                #
                # first, verify we have FSDP activation support ready by importing:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    checkpoint_wrapper,
                    apply_activation_checkpointing,
                    CheckpointImpl,
                )
                non_reentrant_wrapper = functools.partial(
                    checkpoint_wrapper,
                    offload_to_cpu=False,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
            except Exception as e:
                rank0_print('FSDP activation checkpointing not available:', e)
            else:
                check_fn = lambda submodule: isinstance(submodule, wrap_class)
                rank0_print('Applying activation checkpointing wrapper to policy...')
                apply_activation_checkpointing(self.policy, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)
                rank0_print('FSDP activation checkpointing enabled!')

        if config.loss.name in {'dpo', 'ipo'}:
            rank0_print('Sharding reference model...')
            self.reference_model = FSDP(reference_model, **shared_fsdp_kwargs)

        print('Loaded model on rank', rank)
        dist.barrier()

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of an FSDP policy, gathering the gradients across all GPUs."""
        return self.policy.clip_grad_norm_(self.config.max_grad_norm).item()

    def save(self, output_dir=None, metrics=None):
        """Save policy, optimizer, and scheduler state to disk, gathering from all processes and saving only on the rank 0 process."""
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        output_dir = os.path.join(self.run_dir, f'group-{self.group}')

        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy):
            policy_state_dict = self.policy.state_dict()

        if self.rank == 0:
            policy_filename = 'group' + str(self.group) + '_emiteration' + str(self.em_iteration) + '_policy.pt'
            self.write_state_dict(self.example_counter, policy_state_dict, metrics, policy_filename, output_dir)
        del policy_state_dict
        dist.barrier()

        save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, optim_state_dict_config=save_policy):
            optimizer_state_dict = FSDP.optim_state_dict(self.policy, self.optimizer)
        if save_policy:
            optimizer_state_dict = self.optimizer.state_dict()
        self.optimizer.load_state_dict(optimizer_state_dict)


        if self.rank == 0:
            optimizer_filename = 'group' + str(self.group) + '_emiteration' + str(self.em_iteration) + '_optimizer.pt'
            self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, optimizer_filename, output_dir)
        del optimizer_state_dict
        dist.barrier()

        if self.rank == 0:
            scheduler_filename = 'group' + str(self.group) + '_emiteration' + str(self.em_iteration) + '_scheduler.pt'
            scheduler_state_dict = self.scheduler.state_dict()
            self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, scheduler_filename, output_dir)
        dist.barrier()


class TensorParallelTrainer(BasicTrainer):
    def __init__(self, policy, config, seed, run_dir, reference_model=None, rank=0, world_size=1, dynamic_params=None):
        """A trainer subclass that uses TensorParallel to shard the model across multiple GPUs.

           Based on https://github.com/BlackSamorez/tensor_parallel. Note sampling is extremely slow,
              see https://github.com/BlackSamorez/tensor_parallel/issues/66.
        """
        super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size, dynamic_params)

        rank0_print('Sharding policy...')
        self.policy = tp.tensor_parallel(policy, sharded=True)
        if config.loss.name in {'dpo', 'ipo'}:
            rank0_print('Sharding reference model...')
            self.reference_model = tp.tensor_parallel(reference_model, sharded=False)

    def save(self, output_dir=None, metrics=None):
        """Save (unsharded) policy state to disk."""
        with tp.save_tensor_parallel(self.policy):
            policy_state_dict = self.policy.state_dict()

        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict
