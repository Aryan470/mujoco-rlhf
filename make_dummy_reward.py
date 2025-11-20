import torch
from reward_model import RewardModel

reward_model = RewardModel()
checkpoint = {
    "model_state_dict": reward_model.state_dict(),
}
torch.save(checkpoint, "data/0/models/checkpoints/reward.pt")