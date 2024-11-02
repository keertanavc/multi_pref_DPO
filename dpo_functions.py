import torch
import numpy as np

'''
DPO loss function
'''

def dpo_loss(policy, data, pi_ref, beta):
    loss = 0
    for win, lose, context in data:
        context = torch.tensor(context, dtype=torch.float32)
        pi_theta = policy(context)
        pi_theta = pi_theta / pi_theta.sum()
        log_ratio = torch.log(pi_theta / pi_ref)
        loss = loss + torch.log(torch.sigmoid(beta * (log_ratio[win] - log_ratio[lose])))
    return (-1 * loss) / len(data)
