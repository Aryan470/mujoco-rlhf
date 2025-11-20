import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np
# import gym
# import gym.spaces
# import gym.envs
# import gym.envs.mujoco
# import gym.envs.mujoco.mujoco_env
from policy_model import PolicyModel
import os

def init_and_save_policy_model(
    checkpoint_path: str = "data/models/checkpoints/untrained_policy.pt",
    seed: int = 0,
    device: str = "cpu",
):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = PolicyModel()
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "seed": seed,
        "action_dim": model.action_dim,
        "observation_dim": model.observation_dim,
        "reward_model": None,
        "optimizer_state_dict": None,
        "epoch": 0,
  
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Initialized PolicyModel and saved checkpoint to: {checkpoint_path}")
    return model



# if __name__ == "__main__":
#     init_and_save_policy_model()