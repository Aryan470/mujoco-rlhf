import torch
import torch.nn as nn
import torch.nn.functional as F

ACTION_DIM = 3 #3 continuous values Box(-1, 1, (3,), float32)
OBS_DIM = 11 #Box(-Inf, Inf, (11,), float64)

class PolicyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action_dim = ACTION_DIM
        self.observation_dim = OBS_DIM
        self.actor = MLP(OBS_DIM, 128, ACTION_DIM, 2)
        self.critic = MLP(OBS_DIM, 128, 1, 2)
        self.log_std = nn.Parameter(torch.zeros(ACTION_DIM))

    def forward(self, obs):
        # return a tensor of shape (batch_size, action_dim)
        mean = self.actor(obs)
        std = torch.exp(self.log_std)
        eps = torch.randn_like(mean)
        actions = mean + std * eps 
        return actions, self.critic(obs), mean, std


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])   
        for i in range(num_hidden_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, output_size))
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        for layer in range(len(self.layers)-1):
            x = F.relu(self.layers[layer](x))
        x = self.layers[-1](x)
        return x
