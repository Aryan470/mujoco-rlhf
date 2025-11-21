import gymnasium as gym
import numpy as np
from gymnasium.utils.save_video import save_video
import os
import random
import json
import argparse
import torch
import pathlib
import imageio
from policy_model import PolicyModel
from reward_model import RewardModel
from reward_model import compute_preference_probs as compute_preference_probs
import multiprocessing

def load_models(iteration_base_path: pathlib.Path):
    policy_model = PolicyModel()
    reward_model = RewardModel()
    policy_model_checkpoint = torch.load(iteration_base_path / "policy.pt")["model_state_dict"]
    reward_model_checkpoint = torch.load(iteration_base_path / "reward.pt")["model_state_dict"]
    policy_model.load_state_dict(policy_model_checkpoint)
    reward_model.load_state_dict(reward_model_checkpoint)
    return policy_model, reward_model

def generate_trajectories(policy_model: PolicyModel, reward_model: RewardModel, num_trajectories: int, max_steps: int, device: torch.device):
    # generate trajectories and return a list of sim states and (o, a) sequence
    trajectories = []
    for trajectory_idx in range(num_trajectories):
        env = gym.make("Hopper-v5")
        obs, info = env.reset(seed=trajectory_idx)

        obs_list = []
        act_list = []

        for step in range(max_steps):
            action = policy_model(torch.tensor(obs).float().to(device))[0].detach().cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            obs_list.append(obs.copy())
            act_list.append(action.copy())
        env.close()
        obs_seq_np = np.stack(obs_list, axis=0)
        act_seq_np = np.stack(act_list, axis=0)
        trajectories.append((trajectory_idx, obs_seq_np, act_seq_np))
    return trajectories

def select_pairs(trajectories, reward_model, num_pairs: int, device: torch.device):
    num_candidate_pairs = num_pairs * 10  # currently unused, but fine
    output_pairs = []
    k = 70

    for pair_idx in range(num_pairs):
        trajectory_idx_1 = random.randint(0, len(trajectories) - 1)
        trajectory_idx_2 = random.randint(0, len(trajectories) - 1)

        _, obs_seq_1_np, act_seq_1_np = trajectories[trajectory_idx_1]
        _, obs_seq_2_np, act_seq_2_np = trajectories[trajectory_idx_2]

        start_idx_1 = random.randint(0, obs_seq_1_np.shape[0] - k)
        end_idx_1 = start_idx_1 + k
        start_idx_2 = random.randint(0, obs_seq_2_np.shape[0] - k)
        end_idx_2 = start_idx_2 + k

        # Convert segments to tensors here
        obs_segment_1 = torch.from_numpy(obs_seq_1_np[start_idx_1:end_idx_1]).float().to(device)
        act_segment_1 = torch.from_numpy(act_seq_1_np[start_idx_1:end_idx_1]).float().to(device)
        obs_segment_2 = torch.from_numpy(obs_seq_2_np[start_idx_2:end_idx_2]).float().to(device)
        act_segment_2 = torch.from_numpy(act_seq_2_np[start_idx_2:end_idx_2]).float().to(device)

        with torch.no_grad():
            reward_1 = reward_model(obs_segment_1, act_segment_1).detach().cpu()
            reward_2 = reward_model(obs_segment_2, act_segment_2).detach().cpu()

        preference_probs = compute_preference_probs(torch.stack([reward_1, reward_2]))
        preference_variance = torch.var(preference_probs)
        output_pairs.append(
            (trajectory_idx_1, start_idx_1, end_idx_1,
             trajectory_idx_2, start_idx_2, end_idx_2,
             preference_variance)
        )

    output_pairs.sort(key=lambda x: x[-1], reverse=True)  # or keep ascending if you really want low variance
    return output_pairs[:num_pairs]


def generate_clip(trajectory_idx, obs_seq, act_seq, output_path, start_idx, end_idx):
    env = gym.make("Hopper-v5", render_mode="rgb_array_list")
    env.reset(seed=trajectory_idx)
    for step in range(end_idx):
        action = act_seq[step]
        env.step(action)

    save_video(
        frames=env.render(),
        video_folder=output_path,
        fps=24,
        step_starting_index=start_idx,
        video_length=end_idx - start_idx,
    )
    env.close()

def save_tensors(obs_seq_np, act_seq_np, output_path):
    obs_t = torch.from_numpy(obs_seq_np).float()
    act_t = torch.from_numpy(act_seq_np).float()
    torch.save({"obs": obs_t, "act": act_t}, output_path)

def process_pair(args):
    pair_idx, pair, iteration_idx, output_path, trajectories = args
    (trajectory_idx_1, start_idx_1, end_idx_1,
     trajectory_idx_2, start_idx_2, end_idx_2,
     preference_variance) = pair

    _, obs_seq_1_np, act_seq_1_np = trajectories[trajectory_idx_1]
    _, obs_seq_2_np, act_seq_2_np = trajectories[trajectory_idx_2]

    out_dir_1 = output_path / f"{iteration_idx}" / f"clip_{pair_idx}_1"
    out_dir_2 = output_path / f"{iteration_idx}" / f"clip_{pair_idx}_2"

    generate_clip(trajectory_idx_1, obs_seq_1_np, act_seq_1_np, out_dir_1, start_idx_1, end_idx_1)
    generate_clip(trajectory_idx_2, obs_seq_2_np, act_seq_2_np, out_dir_2, start_idx_2, end_idx_2)

    save_tensors(obs_seq_1_np[start_idx_1:end_idx_1],
                 act_seq_1_np[start_idx_1:end_idx_1],
                 output_path / f"{iteration_idx}" / f"tensor_{pair_idx}_1.pt")
    save_tensors(obs_seq_2_np[start_idx_2:end_idx_2],
                 act_seq_2_np[start_idx_2:end_idx_2],
                 output_path / f"{iteration_idx}" / f"tensor_{pair_idx}_2.pt")

    meta = {
        "pair_id": pair_idx,
        "obs1": {
            "clip_id": f"{pair_idx}_1",
            "tensor_path": str(output_path / f"{iteration_idx}" / f"tensor_{pair_idx}_1.pt"),
            "video_path": str(out_dir_1 / "rl-video-episode-0.mp4"),
        },
        "obs2": {
            "clip_id": f"{pair_idx}_2",
            "tensor_path": str(output_path / f"{iteration_idx}" / f"tensor_{pair_idx}_2.pt"),
            "video_path": str(out_dir_2 / "rl-video-episode-0.mp4"),
        },
    }
    return meta


def render_and_save_pairs(iteration_idx, output_path, trajectories, pairs):
    os.makedirs(output_path / f"{iteration_idx}", exist_ok=True)
    os.makedirs(output_path / "metadata", exist_ok=True)

    arg_list = [
        (pair_idx, pair, iteration_idx, output_path, trajectories)
        for pair_idx, pair in enumerate(pairs)
    ]

    # Optional: cap processes to something smaller to further reduce FD pressure
    num_procs = max(1, multiprocessing.cpu_count() - 4)
    num_procs = min(num_procs, 8)  # hard cap if you like

    with multiprocessing.Pool(processes=num_procs) as pool:
        metadata = pool.map(process_pair, arg_list)

    with open(output_path / "metadata" / f"batch_{iteration_idx}.json", "w") as f:
        wrapped_metadata = {
            "batch_id": iteration_idx,
            "pairs": metadata,
        }
        json.dump(wrapped_metadata, f)


def main(base_path: pathlib.Path, iteration_idx: int):
    num_trajectories = 1000
    num_pairs = 200
    max_steps = 24 * 10
    # load the policy model and reward model
    print(f"Loading models from {base_path / 'models'}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_model, reward_model = load_models(base_path / f"{iteration_idx}" / "models" / "checkpoints")
    policy_model.to(device)
    reward_model.to(device)
    # generate trajectories
    print("Generating trajectories")
    trajectories = generate_trajectories(policy_model, reward_model, num_trajectories, max_steps, device)
    print("Selecting pairs")
    pairs = select_pairs(trajectories, reward_model, num_pairs, device)
    print("Rendering and saving pairs")
    render_and_save_pairs(iteration_idx, base_path, trajectories, pairs)
    print(f"Generated {len(pairs)} pairs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=pathlib.Path, required=True)
    parser.add_argument("--iteration_idx", type=int, required=True)
    args = parser.parse_args()
    main(args.base_path, args.iteration_idx)