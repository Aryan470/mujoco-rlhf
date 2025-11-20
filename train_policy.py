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
from reward_model import RewardModel
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

def init_and_save_reward_model(
    checkpoint_path: str = "data/0/models/checkpoints/reward.pt",
    seed: int = 0,
    device: str = "cpu",
):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = RewardModel()
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "seed": seed,
        "optimizer_state_dict": None,
        "epoch": 0,
  
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Initialized RewardModel and saved checkpoint to: {checkpoint_path}")
    return model

def load_policy_model(
    checkpoint_path: str = "data/models/checkpoints/untrained_policy.pt",
    reward_checkpoint_path: str = "data/models/checkpoints/TRPO_reward_model.pt",
    output_path: str = "data/models/checkpoints/TRPO_trained_policy.pt",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = PolicyModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model

def load_reward_model(
    checkpoint_path: str = "data/0/models/checkpoints/reward.pt",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = RewardModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model

def train_policy_model(
    policy_checkpoint_path: str = "data/0/models/checkpoints/policy.pt",
    reward_checkpoint_path: str = "data/0/models/checkpoints/reward.pt",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
)
    checkpoint = torch.load(policy_checkpoint_path, map_location=device)
    policy_model = load_policy_model(checkpoint, device)
    reward_model = load_reward_model(reward_checkpoint_path, device)

    reward_model.eval()
    optimizer = optim.Adam(policy_model.parameters(), lr=1e-4)

    return policy_model