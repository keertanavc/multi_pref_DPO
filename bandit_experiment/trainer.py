import torch
import numpy as np
from neuralnets import PolicyNet, RewardNet
from dpo_functions import dpo_loss
from datageneration import generate_win_lose_pair
from torch import optim
import matplotlib.pyplot as plt
import json

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

with open('config.json', 'r') as f:
    config = json.load(f)

ARMS = config["ARMS"]
PI_REF = torch.tensor(np.ones(ARMS) / ARMS).to(device)
NUM_USERS_GROUPS = config["NUM_USERS_GROUPS"]
NUM_USERS = config["NUM_USERS"]
DATA_PER_USER = config["DATA_PER_USER"]
USER_DIST = config["USER_DIST"]

LEARNING_RATE = config["LEARNING_RATE"]
BETA = config["BETA"]
MAX_EPOCHS = config["MAX_EPOCHS"]
EM_STEPS = config["EM_STEPS"]
MINMAX_STEPS = config["MINMAX_STEPS"]
CONTEXT_DIM = config["CONTEXT_DIM"]
HIDDEN_DIM = config["HIDDEN_DIM"]

'''
Given a context and a theta, generate the rewards linearly
'''

def generate_linear_rewards(thetas, context, std=0.01):
    reward = np.dot(thetas, context) + np.random.normal(0, std, ARMS)
    return reward - np.min(reward)

'''
Generate thetas for each user group where each user group is very different
'''

def generate_diverse_user_thetas(num_users_groups=NUM_USERS_GROUPS, arms=ARMS, context_dim=CONTEXT_DIM):
    user_thetas = {}
    for i in range(num_users_groups):
        user_thetas[i] = np.zeros((arms, context_dim))
        user_thetas[i][i, :] = 10
    return user_thetas

'''
Generate thetas for each user group where each user group is random
'''

def generate_random_user_thetas(num_users_groups=NUM_USERS_GROUPS, arms=ARMS, context_dim=CONTEXT_DIM, mean=0, std=1):
    user_thetas = {}
    for i in range(num_users_groups):
        user_thetas[i] = np.random.randn(arms, context_dim) * std + mean
        user_thetas[i] = user_thetas[i] - np.min(user_thetas[i])
    return user_thetas

'''
Generate a context vector
'''

def generate_context(range_min=0, range_max=1, context_dim=CONTEXT_DIM):
    return np.random.uniform(range_min, range_max, (context_dim))

user_thetas = generate_random_user_thetas(std=5)
print(user_thetas)

#Data generation by sampling data per user, generating rewards, and creating a win/lose pair

user_data = {}
for i in range(NUM_USERS):
    true_type = np.random.choice(NUM_USERS_GROUPS, 1, p=USER_DIST)[0]
    true_thetas = user_thetas[true_type]
    data = []
    for j in range(DATA_PER_USER):
        context = generate_context()
        reward = generate_linear_rewards(true_thetas, context)
        win, lose = generate_win_lose_pair(reward, PI_REF.cpu())
        datapoint = (torch.tensor(win).to(device), torch.tensor(lose).to(device), torch.tensor(context).to(device))
        data.append(datapoint)
    user_data[i] = {'data': data, 'group': np.random.choice(NUM_USERS_GROUPS, 1)[0], 'true_type': true_type}

for i in range(10):
    print(user_data[0]['data'][i])

aggregate_data = []
for i in range(NUM_USERS):
    aggregate_data.extend(user_data[i]['data'])

#Initialize policy

dpo_policy = PolicyNet(CONTEXT_DIM, HIDDEN_DIM, ARMS).to(device)
context = generate_context()
print(dpo_policy(torch.tensor(context, dtype=torch.float32).to(device)))
dpo_policy.fc1.requires_grad = False #Freeze first layer
dpo_policy.fc2.requires_grad = True
dpo_optimizer = optim.SGD(dpo_policy.fc2.parameters(), lr=LEARNING_RATE)

last_loss = 0
cur_loss = np.inf
num_epochs = 0

losses = []

while np.abs(cur_loss - last_loss) > 1e-6 and num_epochs < MAX_EPOCHS:
    if num_epochs % 10 == 1:
        print(f"Epoch {num_epochs}, Loss: {cur_loss}")
    dpo_optimizer.zero_grad()
    loss = dpo_loss(dpo_policy, aggregate_data, PI_REF, BETA)
    loss.backward()
    dpo_optimizer.step()
    last_loss = cur_loss
    cur_loss = loss.item()
    losses.append(cur_loss)
    num_epochs += 1

print(f"Training complete after {num_epochs} epochs")
print(cur_loss)

#Plot the loss over epochs

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.savefig(f'dpo_loss_{LEARNING_RATE}_{BETA}.png')

#Check that policy is working, calculate accuracy metrics if necessary

with open(f'dpo_policy_{LEARNING_RATE}_{BETA}.txt', 'w') as f:
    for i in range(NUM_USERS):
        context = generate_context()
        rewards = generate_linear_rewards(user_thetas[user_data[i]['true_type']], context)
        f.write(f"Rewards: {rewards}\n")
        f.write(f"Max Reward: {np.argmax(rewards)}\n")
        f.write(f"Policy: {dpo_policy(torch.tensor(context, dtype=torch.float32).to(device))}\n")
        f.write(f"Policy Argmax: {torch.argmax(dpo_policy(torch.tensor(context, dtype=torch.float32).to(device)))}\n")

torch.save(dpo_policy.state_dict(), f'dpo_policy_{LEARNING_RATE}_{BETA}.pth')
