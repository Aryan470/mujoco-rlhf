import torch
from policy_model import MLP
import torch.nn as nn

ENSEMBLE_SIZE = 10
OBS_DIM = 11
ACTION_DIM = 3


class RewardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ensemble_size = ENSEMBLE_SIZE
        self.obs_dim = OBS_DIM
        self.act_dim = ACTION_DIM
        self.members = nn.ModuleList([
            SingleRewardModel()
            for _ in range(self.ensemble_size)
        ])
        

    def forward(self, obs, act):
        # return a tensor of shape (batch_size, ensemble_size)
        rewards = []
        for mlp in self.members:
            r = mlp(obs, act)       
            rewards.append(r) 

        return torch.cat(rewards, dim=-1)

class SingleRewardModel(torch.nn.Module):
    def __init__(self, hidden_size=128, num_hidden_layers=2):
        super().__init__()
        self.obs_dim = OBS_DIM
        self.act_dim = ACTION_DIM

        self.model = MLP(self.obs_dim + self.act_dim, hidden_size, 1, num_hidden_layers)

    def forward(self, obs, act):
        # return a tensor of shape (batch_size, 1)
        x = torch.cat([obs, act], dim=-1)
        return self.model(x)


def compute_preference_probs(reward_sum_tensor: torch.Tensor):
    # take a tensor of shape (2, traj_len, ensemble_size) and return a tensor of shape (ensemble_size) storing probability that first is preferred over second for that reward model
    reward_sum_tensor = reward_sum_tensor.permute(2, 0, 1)
    reward_sum_tensor = reward_sum_tensor.sum(dim=-1)
    preference_probs = torch.softmax(reward_sum_tensor, dim=-1)
    return preference_probs

def compute_preference_probs_training(reward_sum_tensor: torch.Tensor):
    # take a tensor of shape (2, batch,ensemble_size) and return a tensor of shape (ensemble_size) storing probability that first is preferred over second for that reward model
    # reward_sum_tensor = reward_sum_tensor.sum(dim=1)
    # reward_sum_tensor = reward_sum_tensor.permute(1, 0)
    # preference_probs = torch.softmax(reward_sum_tensor, dim=-1)
    # probs = (1.0 - 0.1) * preference_probs + 0.1 * 0.5
    reward_sum_tensor = reward_sum_tensor.permute(1, 2, 0)
    preference_probs = torch.softmax(reward_sum_tensor, dim=-1)
    probs = (1.0 - 0.1) * preference_probs + 0.1 * 0.5
    return probs

def compute_preference_probs_loss(
    reward_model,
    seq1_obs, seq1_act,    
    seq2_obs, seq2_act, 
    labels,               
    device,
):
    ensemble_size = reward_model.ensemble_size
    batch_size, traj_len, obs_dim = seq1_obs.shape
    _, _, act_dim = seq1_act.shape

    seq1_obs = seq1_obs.to(device).view(batch_size*traj_len, obs_dim)
    seq1_act = seq1_act.to(device).view(batch_size*traj_len, act_dim)
    seq2_obs = seq2_obs.to(device).view(batch_size*traj_len, obs_dim)
    seq2_act = seq2_act.to(device).view(batch_size*traj_len, act_dim)
    labels = labels.to(device)

    r1 = reward_model(seq1_obs, seq1_act)  
    r2 = reward_model(seq2_obs, seq2_act)

    r1 = r1.view(batch_size, traj_len, ensemble_size).sum(1)
    r2 = r2.view(batch_size, traj_len, ensemble_size).sum(1)

    reward_sum = torch.stack([r1, r2], dim=0)
    prefs = compute_preference_probs_training(
        reward_sum
    ) 

    p_seq1_better = prefs[:, :, 0]
    y = labels.view(batch_size, 1).expand(batch_size, ensemble_size)

    eps = 1e-8
    loss_per = -(
        y * torch.log(p_seq1_better + eps) +
        (1.0 - y) * torch.log(1.0 - p_seq1_better + eps)
    )
    return loss_per.mean()

 