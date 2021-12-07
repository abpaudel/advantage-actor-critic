import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class Actor(nn.Module):

    def __init__(self, in_size, hidden_size, out_size, device):
        super(Actor, self).__init__()
        self.device = device
        self.hidden1 = nn.Linear(in_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, out_size)
        self.to(device)

    def forward(self, state):
        x = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        out = self.output(x)
        policy = Categorical(logits=out)
        return policy

class Critic(nn.Module):

    def __init__(self, in_size, hidden_size, out_size, device):
        super(Critic, self).__init__()
        self.device = device
        self.hidden1 = nn.Linear(in_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, out_size)
        self.to(device)

    def forward(self, state):
        x = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        out = self.output(x)
        return out
