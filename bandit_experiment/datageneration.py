import torch
import numpy as np

CONTEXT_DIM = 3

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_win_lose_pair(reward, pi_ref):
    pair = np.random.choice(len(pi_ref), size=2, replace=False, p=pi_ref.numpy())
    p0_win = sigmoid(reward[pair[0]] - reward[pair[1]])
    p1_win = sigmoid(reward[pair[1]] - reward[pair[0]])
    prob_array = np.array([p0_win, p1_win])/np.sum([p0_win, p1_win])
    win_index = np.random.choice(2, p=prob_array)
    win = pair[win_index]
    lose = pair[1 - win_index]
    return (win, lose)

'''
def generate_data_from_reward(reward, pi_ref, num_pairs):
    win_list = []
    lose_list = []
    for _ in range(num_pairs):
        context = generate_context()
        win, lose = generate_win_lose_pair(reward, pi_ref)
        win_list.append(win)
        lose_list.append(lose)
    return win_list, lose_list


def generate_all_user_data(num_users, num_users_groups, user_dist, data_per_user, rewards, pi_ref):
    user_data = {}
    for i in range(num_users):
        true_type = np.random.choice(num_users_groups, 1, p=user_dist)[0]
        true_reward = rewards[true_type]
        data = np.array(generate_data_from_reward(true_reward, pi_ref, data_per_user))
        user_data[i] = {'data': data}
        user_data[i]['group'] = np.random.choice(num_users_groups, 1)[0]
        user_data[i]['true_type'] = true_type
    return user_data
'''


