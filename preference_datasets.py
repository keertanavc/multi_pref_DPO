'''
Modified from https://github.com/eric-mitchell/direct-preference-optimization
'''


import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from utils import get_local_dir, TemporarilySeededRandom
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
import random
from bs4 import BeautifulSoup, NavigableString
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple



def get_imdb(split: str, name: str, silent: bool = False, cache_dir: str = None, weights_dict: Dict = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    # assign equal weight to all data points, i.e. perform regular DPO is no weights are passed
    print(f'Loading IMDb dataset ({split} split) from Huggingface...')
    #dataset = datasets.load_dataset("kvseet17/repeatsamplingimdb", split=split, cache_dir=cache_dir)
    dataset = datasets.load_dataset("keertanavc/imdb_sentiment-grammar_indexed", split=split, cache_dir=cache_dir)
    print('done')
    def split_prompt_and_responses(ex):
        row_data = {}
        row_data['prompt'] = ex['prompt']
        row_data['chosen_response'] = ex['chosen']
        row_data['rejected_response'] = ex['rejected']
        row_data['pref_type'] = ex['pref_type']
        row_data['cluster'] = ex['cluster']
        row_data['index'] = ex['index']
        if split == 'train':
            row_data['human_label'] = ex['human_label']
            row_data['weight'] = weights_dict[int(ex['human_label'])]
        elif split == 'test':
            row_data['human_label'] = -10
            row_data['weight'] = 1
        substring_to_remove = '<|endoftext|>'
        row_data['prompt'] = row_data['prompt'].replace(substring_to_remove, "")
        row_data['chosen_response'] = row_data['chosen_response'].replace(substring_to_remove, "")
        row_data['rejected_response'] = row_data['rejected_response'].replace(substring_to_remove, "")
        return row_data

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing IMDb', disable=silent):
        row_data = split_prompt_and_responses(row)
        pref_type = row_data['pref_type']
        cluster = row_data['cluster']
        if name == 'imdb_sentiment':
            if pref_type != 2:
                continue
        elif name == 'imdb_grammar':
            if pref_type != 1:
                continue
        elif name == 'imdb_cluster0':
            if cluster != 0:
                continue
        elif name == 'imdb_cluster1':
            if cluster != 1:
                continue
        prompt = row_data['prompt']
        chosen = row_data['chosen_response']
        rejected = row_data['rejected_response']
        responses = [chosen, rejected]
        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = chosen
        data[prompt]['pref_type'].append(pref_type)
        data[prompt]['human_label'].append(row_data['human_label'])
        data[prompt]['weight'].append(row_data['weight'])
        data[prompt]['index'].append(row_data['index'])
    return data
###

def get_globalopinion(split: str, name: str, silent: bool = False, cache_dir: str = None, weights_dict: Dict = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    # assign equal weight to all data points, i.e. perform regular DPO is no weights are passed
    print(f'Loading Global Opinion dataset ({split} split) from Huggingface...')
    # if 'og' in name:
    print('loading .. keertanavc/globalopinion_og')
    dataset = datasets.load_dataset("keertanavc/globalopinion_og", split=split, cache_dir=cache_dir)
    # else:
    #     print('loading .. keertanavc/globalopinionv5')
    #     dataset = datasets.load_dataset("keertanavc/globalopinionv5", split=split, cache_dir=cache_dir)
    print('done')
    def split_prompt_and_responses(ex):
        row_data = {}
        row_data['prompt'] = ex['prompt']
        row_data['chosen_response'] = ex['chosen']
        row_data['rejected_response'] = ex['rejected']
        row_data['pref_type'] = ex['pref_type']
        row_data['cluster'] = ex['cluster']
        row_data['index'] = ex['index']
        if split == 'train':
            row_data['human_label'] = ex['human_label']
            row_data['weight'] = weights_dict[int(ex['human_label'])]
        elif split == 'test':
            row_data['human_label'] = -100
            row_data['weight'] = 1
        substring_to_remove = '<|endoftext|>'
        row_data['prompt'] = row_data['prompt'].replace(substring_to_remove, "")
        row_data['chosen_response'] = row_data['chosen_response'].replace(substring_to_remove, "")
        row_data['rejected_response'] = row_data['rejected_response'].replace(substring_to_remove, "")
        return row_data

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing Global Opinion LLM Dataset', disable=silent):
        row_data = split_prompt_and_responses(row)
        cluster = row_data['cluster']
        pref_type = row_data['pref_type']
        if 'cluster' in name: # give name as globalopinion_cluster_{i} to train only on cluster i
            if int(cluster) != int(name[-1]):
                continue
        if '_in' in name:
            if pref_type != 'Indonesia':
                continue
        elif '_mx' in name: 
            if pref_type != 'Mexico':
                continue
        elif '_pk' in name: 
            if pref_type != 'Pakistan':
                continue
        elif '_br' in name: 
            if pref_type != 'Britain':
                continue
        prompt = row_data['prompt']
        chosen = row_data['chosen_response']
        rejected = row_data['rejected_response']
        responses = [chosen, rejected]
        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = chosen
        data[prompt]['pref_type'].append(pref_type)
        data[prompt]['human_label'].append(row_data['human_label'])
        data[prompt]['weight'].append(row_data['weight'])
        data[prompt]['index'].append(row_data['index'])
    return data
###

# def get_personal(split: str, name: str, silent: bool = False, cache_dir: str = None, weights_dict: Dict = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
#     # assign equal weight to all data points, i.e. perform regular DPO is no weights are passed
#     print(f'Loading PersonalLLM dataset ({split} split) from Huggingface...')
#     dataset = datasets.load_dataset("keertanavc/personalLLM", split=split, cache_dir=cache_dir)
#     print('done')
#     def split_prompt_and_responses(ex):
#         row_data = {}
#         row_data['prompt'] = ex['prompt']
#         row_data['chosen_response'] = ex['chosen']
#         row_data['rejected_response'] = ex['rejected']
#         row_data['pref_type'] = ex['pref_type']
#         row_data['cluster'] = ex['cluster']
#         row_data['index'] = ex['index']
#         if split == 'train':
#             row_data['human_label'] = ex['human_label']
#             row_data['weight'] = weights_dict[int(ex['human_label'])]
#         elif split == 'test':
#             row_data['human_label'] = -100
#             row_data['weight'] = 1
#         substring_to_remove = '<|endoftext|>'
#         row_data['prompt'] = row_data['prompt'].replace(substring_to_remove, "")
#         row_data['chosen_response'] = row_data['chosen_response'].replace(substring_to_remove, "")
#         row_data['rejected_response'] = row_data['rejected_response'].replace(substring_to_remove, "")
#         return row_data

#     data = defaultdict(lambda: defaultdict(list))
#     for row in tqdm.tqdm(dataset, desc='Processing Personal LLM Dataset', disable=silent):
#         row_data = split_prompt_and_responses(row)
#         cluster = row_data['cluster']
#         if 'cluster' in name: # give name as personal_cluster{i} to train only on cluster i
#             if int(cluster) != int(name[-1]):
#                 continue
#         prompt = row_data['prompt']
#         chosen = row_data['chosen_response']
#         rejected = row_data['rejected_response']
#         responses = [chosen, rejected]
#         n_responses = len(data[prompt]['responses'])
#         data[prompt]['pairs'].append((n_responses, n_responses + 1))
#         data[prompt]['responses'].extend(responses)
#         data[prompt]['sft_target'] = chosen
#         data[prompt]['pref_type'].append(row_data['pref_type'])
#         data[prompt]['human_label'].append(row_data['human_label'])
#         data[prompt]['weight'].append(row_data['weight'])
#         data[prompt]['index'].append(row_data['index'])
#     return data
# ###

def get_dataset(name: str, split: str, silent: bool = False, cache_dir: str = None, weights_dict: Dict = None):
    """Load the given dataset by name. Supported by default are 'shp', 'hh', and 'se'."""
    if 'imdb' in name:
        data = get_imdb(split, name, silent=silent, cache_dir=cache_dir, weights_dict=weights_dict)
    # elif 'personal' in name:
    #     data = get_personal(split, name, silent=silent, cache_dir=cache_dir, weights_dict=weights_dict)
    elif 'globalopinion' in name:
        data = get_globalopinion(split, name, silent=silent, cache_dir=cache_dir, weights_dict=weights_dict)
    else:
        raise ValueError(f"Unknown dataset '{name}'")
    return data


def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Returns a collate function for the given tokenizer.

       The collate function takes a list of examples (dicts, where values are lists of
         ints [tokens] or strings [the original texts]) and returns a batch of examples,
         PyTorch tensors padded to the maximum length. Strings are passed through."""
    def collate_fn(batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                if 'prompt' in k:  # adapted from https://stackoverflow.com/questions/73256206
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith('_input_ids'):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                if 'prompt' in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            elif k == 'weight' or k == 'human_label':
                padded_batch[k] = torch.tensor([ex[k] for ex in batch])
            else:
                padded_batch[k] = [ex[k] for ex in batch]
        return padded_batch
    return collate_fn


def tokenize_batch_element(prompt: str, chosen: str, rejected: str, truncation_mode: str, tokenizer, max_length: int, max_prompt_length: int, weight: int, human_label: int, index: int) -> Dict:
    """Tokenize a single batch element.

       At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
         in case the prompt + chosen or prompt + rejected responses is/are too long. First
         we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

       We also create the labels for the chosen/rejected responses, which are of length equal to
         the sum of the length of the prompt and the chosen/rejected response, with -100 for the
         prompt tokens.
    """
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    assert tokenizer.eos_token_id not in prompt_tokens['input_ids'], f"Prompt contains EOS token: {prompt}"
    assert tokenizer.eos_token_id not in chosen_tokens['input_ids'], f"Chosen response contains EOS token: {chosen}"
    assert tokenizer.eos_token_id not in rejected_tokens['input_ids'], f"Rejected response contains EOS token: {rejected}"

    chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
    chosen_tokens['attention_mask'].append(1)

    rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
    rejected_tokens['attention_mask'].append(1)

    longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

    # if combined sequence is too long, truncate the prompt
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        if truncation_mode == 'keep_start':
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == 'keep_end':
            prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')

    # if that's still too long, truncate the response
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        chosen_tokens = {k: v[:max_length - max_prompt_length] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[:max_length - max_prompt_length] for k, v in rejected_tokens.items()}

    # Create labels
    chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
    chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])

    batch = {}

    batch['prompt'] = prompt
    batch['chosen'] = prompt + chosen
    batch['rejected'] = prompt + rejected
    batch['chosen_response_only'] = chosen
    batch['rejected_response_only'] = rejected
    batch['weight'] = weight
    batch['human_label'] = human_label
    batch['index'] = index

    for k, toks in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens, 'prompt': prompt_tokens}.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens

    return batch


def get_batch_iterator(names: List[str],
                       tokenizer,
                       split: str = 'train',
                       batch_size: int = 1,
                       shuffle: bool = True,
                       max_length: int = 512,
                       max_prompt_length: int = 128,
                       sft_mode: bool = False,
                       n_epochs: Optional[int] = None,
                       n_examples: Optional[int] = None,
                       seed:int = 0,
                       silent: bool = False,
                       cache_dir: Optional[str] = None,
                       weights_dict: Dict = None) -> Iterator[Dict]:
    """Get an iterator over batches of data. Stops after n_epochs or n_examples, whichever comes first.

    Args:
        names: Names of datasets to use.
        tokenizer: Tokenizer to use.
        split: Which split to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data after each epoch.
        max_length: Maximum length of the combined prompt + response.
        max_prompt_length: Maximum length of the prompt.
        sft_mode: Whether to use SFT mode (i.e., return sft_target instead of chosen/rejected). In sft mode, we just return chosen_input_ids, but they contain the sft_target.
        n_epochs: Number of epochs to run for. This or n_examples must be specified.
        n_examples: Number of examples to run for. This or n_epochs must be specified.
        seed: Random seed.
        silent: Whether to silence the progress bar(s).
        cache_dir: Directory to cache the datasets in.
    """
    assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"
    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
        flat_data = []
        for name in names:
            truncation_mode = 'keep_end' if name == 'hh' else 'keep_start'
            for prompt, data in get_dataset(name, split, silent=silent, cache_dir=cache_dir, weights_dict=weights_dict).items():
                assert (len(data['weight']) == len(data['pairs'])) and (len(data['human_label']) == len(data['pairs']))
                flat_data.append((prompt, data['responses'], data['pairs'], data['sft_target'], truncation_mode, data['weight'], data['human_label'], data['index']))
    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    example_idx = 0
    done = False
    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f'Finished generating {n_epochs} epochs on {split} split')
            break
        if shuffle:
            with TemporarilySeededRandom(next(permutation_seeds)):
                random.shuffle(flat_data)

        batch = []
        for row in flat_data:
            prompt, responses, pairs, sft_target, truncation_mode = row[:5]
            weight, human_label, index = row[5:8]
            weight = torch.tensor(weight)
            if done:
                break
            if sft_mode:
                # tokenize_batch_element(prompt: str, chosen: str, rejected: str, truncation_mode: str, tokenizer, max_length: int, max_prompt_length, weight, human_label, index)
                batch_element = tokenize_batch_element(prompt, sft_target, sft_target, truncation_mode, tokenizer, max_length, max_prompt_length, 1, 0, index)
                # batch_element = tokenize_batch_element(prompt, responses[p[0]], responses[p[1]], truncation_mode, tokenizer, max_length, max_prompt_length, weight[indx], human_label[indx])
                batch_element = {k: v for k, v in batch_element.items() if 'rejected' not in k}
                batch.append(batch_element)
                example_idx += 1
                if len(batch) == batch_size:
                    yield collate_fn(batch)
                    if n_examples is not None and example_idx >= n_examples:
                        if not silent:
                            print(f'Finished generating {n_examples} examples on {split} split')
                        done = True
                    batch = []
            else:
                for p in pairs:
                    if done:
                        break
                    indx = int(min(p[0], p[1])/2)
                    batch_element = tokenize_batch_element(prompt, responses[p[0]], responses[p[1]], truncation_mode, tokenizer, max_length, max_prompt_length, weight[indx], human_label[indx], index)
                    batch.append(batch_element)
                    example_idx += 1
                    if len(batch) == batch_size:
                        yield collate_fn(batch)
                        if n_examples is not None and example_idx >= n_examples:
                            if not silent:
                                print(f'FINISHED {n_examples} EXAMPLES on {split} split')
                            done = True
                        batch = []
        if done:
            break

        epoch_idx += 1


def strings_match_up_to_spaces(str_a: str, str_b: str) -> bool:
    """Returns True if str_a and str_b match up to spaces, False otherwise."""
    for idx in range(min(len(str_a), len(str_b)) - 2):
        if str_a[idx] != str_b[idx]:
            if str_a[idx] != ' ' and str_b[idx] != ' ':
                return False
            else:
                if str_a[idx] == ' ':
                    str_a = str_a[:idx] + str_a[idx + 1:]
                else:
                    str_b = str_b[:idx] + str_b[idx + 1:]

    return True
