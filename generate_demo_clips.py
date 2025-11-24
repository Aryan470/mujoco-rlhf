import os
os.environ["MUJOCO_GL"] = "egl"

import gymnasium as gym
import numpy as np
from gymnasium.utils.save_video import save_video

import argparse
import torch
import pathlib
from policy_model import PolicyModel

def load_models(iteration_base_path: pathlib.Path):
    policy_model = PolicyModel()
    policy_model_checkpoint = torch.load(iteration_base_path / "policy.pt")["model_state_dict"]
    policy_model.load_state_dict(policy_model_checkpoint)
    return policy_model

def generate_clip(clip_idx, policy_model, output_path, max_steps, device):
    env = gym.make("Hopper-v5", render_mode="rgb_array")
    obs, info = env.reset(seed=clip_idx)

    frames = []
    for t in range(max_steps):  # replay until the end of the desired window
        action = policy_model(torch.tensor(obs).float().to(device))[0].detach().cpu().numpy()
        obs, _, _, _, _ = env.step(action)
        frames.append(env.render())

    env.close()

    os.makedirs(output_path, exist_ok=True)
    save_video(
        frames=frames,
        video_length=len(frames),
        video_folder=output_path,
        fps=24,
    )

def main(base_path: pathlib.Path, iteration_idx: int, num_clips: int = 5):
    max_steps = 24 * 50
    # load the policy model and reward model
    print(f"Loading models from {base_path / 'models'}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_model = load_models(base_path / f"{iteration_idx}" / "models" / "checkpoints")
    policy_model.to(device)

    # generate 5 clips
    for clip_idx in range(num_clips):
        generate_clip(clip_idx, policy_model, base_path / f"{iteration_idx}" / "clips" / f"clip_{clip_idx}", max_steps, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=pathlib.Path, required=True)
    parser.add_argument("--iteration_idx", type=int, required=True)
    parser.add_argument("--num_clips", type=int, required=False, default=20)
    args = parser.parse_args()
    main(args.base_path, args.iteration_idx, args.num_clips)