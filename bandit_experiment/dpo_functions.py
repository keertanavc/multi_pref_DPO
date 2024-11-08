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

'''
Weighted DPO loss function
'''

def weighted_dpo_loss(policy, data, pi_ref, beta, weights, true_types):
    loss = 0
    for idx, (win, lose, context) in enumerate(data):
        context = torch.tensor(context, dtype=torch.float32)
        pi_theta = policy(context)
        pi_theta = pi_theta / pi_theta.sum()
        log_ratio = torch.log(pi_theta / pi_ref)
        loss = loss + weights[true_types[idx]] * torch.log(torch.sigmoid(beta * (log_ratio[win] - log_ratio[lose])))
    return (-1 * loss) / len(data)

'''
M-step DPO loss (replace true_types with types from E-step)
types = etas
training multiple policies
data in user_data format
'''

def m_step_dpo_loss(policies, user_data, pi_ref, beta, gammas, device='cpu'):
    loss_terms = []
    num_users = len(user_data)
    num_users_groups = len(policies)
    for h in range(num_users):
        data = user_data[h]['data']
        for k in range(num_users_groups):
            for win, lose, context in data:
                context = torch.tensor(context, dtype=torch.float32, device=device)
                policy = policies[k]
                pi_theta = policy(context)
                pi_theta = pi_theta / pi_theta.sum()
                log_ratio = torch.log(pi_theta / pi_ref)
                loss_terms.append(gammas[h, k] * torch.log(torch.sigmoid(beta * (log_ratio[win] - log_ratio[lose]))))
                #loss += gammas[h, k] * torch.log(torch.sigmoid(beta * (log_ratio[win] - log_ratio[lose])))
    total_loss = -torch.sum(torch.stack(loss_terms)) / (len(user_data) * len(user_data[0]['data']))
    return total_loss

'''
Batched M-step DPO loss, takes in full user data
'''

def m_step_dpo_loss_batched(policies, user_data, pi_ref, beta, gammas, device='cpu'):
    pi_ref = torch.tensor(pi_ref, dtype=torch.float32, device=device)
    gammas = gammas.to(device)

    loss = 0
    num_users_groups = len(policies)
    num_users = len(user_data)

    for k in range(num_users_groups):
        policy = policies[k]
        group_loss = 0
        for h in range(num_users):
            data = user_data[h]['data']
            wins, loses, contexts = zip(*data)
            contexts = torch.stack(contexts).to(device)
            pi_theta = policy(contexts)
            pi_theta = pi_theta / pi_theta.sum(dim=-1, keepdim=True)
            log_ratio = torch.log(pi_theta / pi_ref)
            group_loss += gammas[h, k] * torch.sum(torch.log(torch.sigmoid(beta * (log_ratio[:, wins] - log_ratio[:, loses]))))
        loss += group_loss
    
    return (-1 * loss) / (num_users * len(user_data[0]['data']))


'''
E-step Update
'''

def e_step_update(policies, data, pi_ref, beta, etas, device='cpu'):
    log_etas = torch.log(etas)
    num_users_groups = len(policies)
    for win, lose, context in data:
        context = torch.tensor(context, dtype=torch.float32)
        for k in range(num_users_groups):
            policy = policies[k]
            pi_theta = policy(context)
            pi_theta = pi_theta / pi_theta.sum()
            log_ratio = torch.log(pi_theta / pi_ref)
            log_etas[k] += torch.log(torch.sigmoid(beta * (log_ratio[win] - log_ratio[lose])))
    log_etas = log_etas - torch.min(log_etas)
    return torch.softmax(log_etas, dim=0)

'''
E-step Update (batched)
Take in all user_data at once
'''

def e_step_update_batched(policies, user_data, pi_ref, beta, etas, device='cpu'):
    pi_ref = torch.tensor(pi_ref, dtype=torch.float32, device=device)
    etas = torch.tensor(etas, dtype=torch.float32, device=device)
    log_etas = torch.log(etas)
    num_users_groups = len(policies)
    num_users = len(user_data)

    log_etas_updates = torch.zeros((num_users, num_users_groups), dtype=torch.float32, device=device)

    for h in range(num_users):
        data = user_data[h]['data']
        wins, loses, contexts = zip(*data)
        #print(contexts)
        contexts = torch.stack(contexts).to(device)
        for k in range(num_users_groups):
            policy = policies[k]
            pi_theta = policy(contexts)
            pi_theta = pi_theta / pi_theta.sum(dim=-1, keepdim=True)
            log_ratio = torch.log(pi_theta / pi_ref)
            log_etas_updates[h, k] = torch.sum(torch.log(torch.sigmoid(beta * (log_ratio[:,     wins] - log_ratio[:, loses]))))

    log_etas += log_etas_updates.sum(dim=0)
    log_etas = log_etas - torch.min(log_etas)
    return torch.softmax(log_etas, dim=0)

def regret_matrix(policies, data, beta, pi_ref):
    num_users_groups = len(policies)
    obj_matrix = torch.zeros_like(torch.empty(num_users_groups, num_users_groups))
    for u in range(num_users_groups):
        for i in range(num_users_groups):
            pi_star_u = policies[u]
            pi_star_i = policies[i]
            count = 0
            for win, lose, context in data:
                count += 1
                context = torch.tensor(context, dtype=torch.float32)
                term1 = torch.dot(pi_star_u(context), torch.log(pi_star_u(context)) / pi_ref)
                term2 = torch.dot(pi_star_i(context), torch.log(pi_star_u(context)) / pi_ref)
                obj_matrix[u, i] += (term1 - term2).detach()
            obj_matrix[u, i] /= count
    return beta * obj_matrix.numpy()