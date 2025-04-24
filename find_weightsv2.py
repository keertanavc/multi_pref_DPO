import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
import time
import torch.nn.functional as F
import numpy as np
import argparse
import os
import json
import itertools

# CONSTANTS
T = 100000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
# REGRET_WEIGHTS = [(1,1,1,1),(4,1,1,1),(1,4,1,1),(1,1,4,1),(1,1,1,4),\
#                     (2,1,1,1),(1,2,1,1),(1,1,2,1),(1,1,1,2),\
#                     (8,1,1,1),(1,8,1,1),(1,1,8,1),(1,1,1,8)]


def generate_tuples(K, values=[1, 2, 4, 8]):
    result = []    
    for value in values:
        # Create a list with K-1 ones and one non-one value
        temp = [1] * (K - 1) + [value]
        # Generate all possible permutations of this list
        result.extend(set(itertools.permutations(temp)))  # set to avoid duplicates
    return sorted(list(result))


def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

def compute_minmax_regret(df, num_groups, reg_w):
    '''Calculating the regret matrix'''
    L = np.zeros((num_groups, num_groups))
    for i in range(num_groups):
        for j in range(num_groups):
            L[i,j] = df['logp_gen' + str(j) + '_evalpol' + str(i)].mean() - df['logp_gen' + str(j) + '_evalsft'].mean()

    # optimistic EXP
    n, m = num_groups+1,num_groups
    R = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if i == 0:
                R[i,j] = 0
            else:
                R[i,j] = reg_w[i-1] * (L[i-1,i-1] - L[i-1,j])
    print(reg_w)
    print(R)

    etax = 5 * np.sqrt(1/T)
    etay = 5 * np.sqrt(1/T)

    x, y = np.ones((T, n)) / n, np.ones((T, m)) / m
    for t in np.arange(1, T):
        lx = R @ y[t-1]
        lx_pre = R @ y[t-2]
        ly = R.T @ x[t-1]
        ly_pre = R.T @ x[t-2]
        x[t] = x[t-1] * np.exp(2 * etax * lx - etax * lx_pre)
        x[t] /= np.sum(x[t])
        y[t] = y[t-1] * np.exp(-2 * etay * ly + etay * ly_pre)
        y[t] /= np.sum(y[t])
    # weight for each policy is the strategy of the second player
    ybar = np.mean(y, axis=0)
    return ybar

def compute_log_prob_completion(args, policy, tokenizer, df):
    '''Compute log probabilities of completion'''
    def apply_function(row, group):
        ''' Generate function that gives log-prob of completion for given prompt '''
        response_col = 'group'+ str(group) +'_response'
        input_ids = tokenizer(row[response_col], return_tensors="pt").input_ids.to(DEVICE)
        prompt_ids = tokenizer(row['prompt'], return_tensors="pt").input_ids.to(DEVICE)

        with torch.no_grad():
            outputs = policy(input_ids, labels=input_ids)
            logits = outputs.logits

        completion_logits = logits[:, prompt_ids.size(1)-1:-1, :]
        completion_ids = input_ids[:, prompt_ids.size(1):]
        log_probs = F.log_softmax(completion_logits, dim=-1)
        completion_log_probs = log_probs.gather(2, completion_ids.unsqueeze(-1)).squeeze(-1)
        total_log_prob = completion_log_probs.sum().item()

        return total_log_prob
    
    all_states = args['states'] + [args['sft_policy']]
    state_names = ['pol' + str(i) for i in range(len(args['states']))]
    state_names.append('sft')

    for pol_group in range(len(all_states)):
        print('computing for :', state_names[pol_group])
        state_dict = torch.load(all_states[pol_group], map_location=DEVICE)
        policy.load_state_dict(state_dict['state'])#, strict=False)
        policy.eval()

        for eval_group in range(len(args['states'])):
            fun = lambda row: apply_function(row, eval_group)
            df['logp' + '_gen' + str(eval_group) + '_eval' + state_names[pol_group]] = df.apply(fun, axis=1)

def generate_completions(args, policy, tokenizer, df):
    for group in range(len(args['states'])):
        state_dict = torch.load(args['states'][group], map_location=DEVICE)
        policy.load_state_dict(state_dict['state'])#, strict=False)
        policy.eval()
        generator = pipeline('text-generation', model=policy, tokenizer=tokenizer, device=DEVICE)
        # generate text from the given model for given group
        # note: generator outputs the prompt prefixed to the generated response
        def batch_completion(text_prompts):
            responses = generator(text_prompts, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
            return [response[0]['generated_text'] for response in responses]
        
        responses = []
        for i in range(0, len(df), BATCH_SIZE):
            print(f'batch ', i/BATCH_SIZE, ' out of ', args['eval_samples']/BATCH_SIZE, ' batches ')
            start_time = time.time()
            batch_prompts = df['prompt'][i:i + BATCH_SIZE].tolist()
            batch_responses = batch_completion(batch_prompts)
            responses.extend(batch_responses)
            end_time = time.time()
            print('batch time is: ', end_time - start_time)

        df['group'+ str(group) +'_response'] = responses

def main():
    # parse all arguments
    print('being program')
    parser = argparse.ArgumentParser(description="MinMax DPO algorithm")
    parser.add_argument("--sft_policy", type=str, required=True, help='addresses to sft policy')
    parser.add_argument("--dataset", type=str, default='globalopinion', help="Which dataset did you train on?")
    parser.add_argument("--eval_samples", type=int, default=128, help="No. of samples to evaluate on")
    parser.add_argument("--seed", type=int, default=0, help="which seed is this for?")
    parser.add_argument("--num_groups", type=int, default=2, help="how many groups are there?")
    parser.add_argument("--iteration", type=int, default=5, help="for emdpo: how many iterations did the algo run for?")
    parser.add_argument("--exp_name", type=str, default='clusterdpo', help="which experiment is this for?")
    parser.add_argument("--model", type=str, default="mistral7b", help='which LLM are you using?')
    args = parser.parse_args()
    args = vars(args)
    print(args)
    set_seed(args['seed'] + 42)
    print('computing weights for seed = ', args['seed'])
    global REGRET_WEIGHTS
    REGRET_WEIGHTS = generate_tuples(args['num_groups'])
    all_states = []
    for group in range(args['num_groups']):
        if args['exp_name'] == 'clusterdpo':
            # all_states.append("temp/" + args['dataset'] + "_cluster" + str(group) + "_allseeds_seed" + str(args['seed']) + "/group-0/group0_emiteration0_policy.pt")
            all_states.append("temp/" + args['dataset'] +"_emdpo_seed" + str(args['seed']) + "/group-" + str(group) + "/group" + str(group) + "_emiteration" + str(0) +"_policy.pt")
            print(all_states)
        elif args['exp_name'] == 'emdpo':
            all_states.append("temp/" + args['dataset'] +"_emdpo_seed" + str(args['seed']) + "/group-" + str(group) + "/group" + str(group) + "_emiteration" + str(args['iteration']-1) +"_policy.pt")
        elif args['exp_name'] == 'emdpo_hyper':
            all_states.append("temp/globalopinon_hyper_dpo_" + str(args['num_groups']) + "_seed0/group-" + str(group) + "/group" + str(group) + "_emiteration" + str(args['iteration']-1) +"_policy.pt")
            print(all_states[-1])
        elif args['exp_name'] == 'clusterdpo_hyper':
            all_states.append("temp/globalopinon_hyper_dpo_" + str(args['num_groups']) + "_seed0/group-" + str(group) + "/group" + str(group) + "_emiteration0_policy.pt")
            print(all_states[-1])
    args['states'] = all_states
    print('loading model and dataset')
    if args['model'] == 'mistral7b':
        model_name = 'mistralai/Mistral-7B-v0.3'
    tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                            clean_up_tokenization_spaces=True, 
                                            token=os.environ.get("HUGGINGFACE_TOKEN"))
    policy = AutoModelForCausalLM.from_pretrained(model_name, 
                                                torch_dtype='auto', 
                                                low_cpu_mem_usage=True,
                                                token=os.environ.get("HUGGINGFACE_TOKEN")).to(DEVICE)
    if args['dataset'] == 'imdb':
        df = load_dataset("keertanavc/imdb_sentiment-grammar_indexed", split='test')
    elif args['dataset'] == 'globalopinion':
        df = load_dataset("keertanavc/globalopinionv5", split='test')
    df = df.to_pandas()
    df = df[['prompt']][:args['eval_samples']]
    print('generating completions')
    generate_completions(args, policy, tokenizer, df)
    print('computing log probability')
    compute_log_prob_completion(args, policy, tokenizer, df)
    # df.to_csv('generations_cluster_seed0.csv')
    # df = pd.read_csv('compute_weight_sample.csv')
    print('computing weights')
    weights = {}
    for reg_w in REGRET_WEIGHTS:
        weights[str(reg_w)] = list(compute_minmax_regret(df, args['num_groups'], reg_w))
        print('regret weights are: ', reg_w)
        print('combination weights are: ', weights[str(reg_w)])
    filename = "eval_csv/weights/globalopinion_emdpo/exp_" + args['exp_name'] + "_K=" + str(args['num_groups']) + "_seed" + str(args['seed']) + ".json"
    with open(filename, "w") as f:
        json.dump(weights, f, indent=4)

if __name__ == "__main__":
    main()
