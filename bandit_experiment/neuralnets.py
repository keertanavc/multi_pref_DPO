import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Neural network definition for policy, 2 fully connected layers with a softmax
'''

class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, output_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x))
    
'''
Neural network definition for reward, 2 fully connected layers
'''
    
class RewardNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, output_dim):
        super(RewardNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)