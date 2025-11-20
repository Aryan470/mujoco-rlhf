import torch

ENSEMBLE_SIZE = 10

class RewardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ensemble_size = ENSEMBLE_SIZE

    def forward(self, obs, act):
        # return a tensor of shape (batch_size, ensemble_size)
        return torch.randn(obs.shape[0], self.ensemble_size)

def compute_preference_probs(reward_sum_tensor: torch.Tensor):
    # take a tensor of shape (2, ensemble_size) and return a tensor of shape (ensemble_size) storing probability that first is preferred over second for that reward model
    reward_sum_tensor = torch.permute(reward_sum_tensor, (1, 0))
    preference_probs = torch.softmax(reward_sum_tensor, dim=0)
    return preference_probs