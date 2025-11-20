import gymnasium as gym
import os
import random
import json
import argparse
import torch
import pathlib
import imageio
from .policy_model import PolicyModel
from .reward_model import RewardModel

def load_models(iteration_base_path: pathlib.Path):
    policy_model = PolicyModel()
    reward_model = RewardModel()
    policy_model.load_state_dict(torch.load(iteration_base_path / "policy_model.pth"))
    reward_model.load_state_dict(torch.load(iteration_base_path / "reward_model.pth"))
    return policy_model, reward_model

def generate_trajectories(policy_model: PolicyModel, reward_model: RewardModel, num_trajectories: int, max_steps: int):
    # generate trajectories and return a list of sim states and (o, a) sequence
    trajectories = []
    for trajectory_idx in range(num_trajectories):
        env = gym.make("Hopper-v5")
        obs, info = env.reset(seed=trajectory_idx)
        start_env = env.unwrapped.sim.get_state()

        obs_list = []
        act_list = []

        for step in range(max_steps):
            action = policy_model(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            obs_list.append(torch.tensor(obs))
            act_list.append(torch.tensor(action))
        trajectories.append((start_env, torch.stack(obs_list), torch.stack(act_list)))
    return trajectories

def select_pairs(trajectories, reward_model, num_pairs: int):
    num_candidate_pairs = num_pairs * 10
    output_pairs = []
    k = 10
    # select random pairs of segments of size k, such that if rendered they are 1-3s
    for pair_idx in range(num_pairs):
        # select two random trajectory segments
        trajectory_idx_1 = random.randint(0, len(trajectories) - 1)
        trajectory_idx_2 = random.randint(0, len(trajectories) - 1)
        start_idx_1 = random.randint(0, len(trajectories[trajectory_idx_1][1]) - k)
        end_idx_1 = start_idx_1 + k
        start_idx_2 = random.randint(0, len(trajectories[trajectory_idx_2][1]) - k)
        end_idx_2 = start_idx_2 + k

        # score them and save the trajectory index, start and end step of each segments
        obs_segment_1 = trajectories[trajectory_idx_1][1][start_idx_1:end_idx_1]
        act_segment_1 = trajectories[trajectory_idx_1][2][start_idx_1:end_idx_1]
        obs_segment_2 = trajectories[trajectory_idx_2][1][start_idx_2:end_idx_2]
        act_segment_2 = trajectories[trajectory_idx_2][2][start_idx_2:end_idx_2]
        reward_1 = reward_model(obs_segment_1, act_segment_1)
        reward_2 = reward_model(obs_segment_2, act_segment_2)
        preference_probs = reward_model.compute_preference_probs(reward_1, reward_2)
        preference_variance = torch.var(preference_probs)
        output_pairs.append((trajectory_idx_1, start_idx_1, end_idx_1, trajectory_idx_2, start_idx_2, end_idx_2, preference_variance))
    # sort them by variance on the preference probabilities
    output_pairs.sort(key=lambda x: x[-1])
    return output_pairs[:num_pairs]


def generate_clip(sim_state, obs_seq, act_seq, output_path, start_idx, end_idx):
    env = gym.make("Hopper-v5")
    env.unwrapped.sim.set_state(sim_state)
    for step in range(start_idx, end_idx):
        action = act_seq[step]
        env.step(action)

    new_env = gym.make("Hopper-v5", render_mode="rgb_array")
    new_env.unwrapped.sim.set_state(env.unwrapped.sim.get_state())
    frames = []
    for step in range(start_idx, end_idx):
        action = act_seq[step]
        new_env.step(action)
        frames.append(new_env.render())
    imageio.mimsave(output_path, frames, fps=24)
    env.close()
    new_env.close()

def save_tensors(obs_seq, act_seq, output_path):
    torch.save({
        "obs": obs_seq,
        "act": act_seq
    }, output_path)

def render_and_save_pairs(iteration_idx, output_path, trajectories, pairs):
    # take a list of (sim state, (o,a) sequence) and render them to get clips
    # save the clips and metadata to the output path
    metadata = []
    for pair_idx, pair in enumerate(pairs):
        trajectory_idx_1, start_idx_1, end_idx_1, trajectory_idx_2, start_idx_2, end_idx_2, preference_variance = pair
        sim_state_1, obs_seq_1, act_seq_1 = trajectories[trajectory_idx_1]
        sim_state_2, obs_seq_2, act_seq_2 = trajectories[trajectory_idx_2]

        os.makedirs(output_path / f"{iteration_idx}", exist_ok=True)
        generate_clip(sim_state_1, obs_seq_1, act_seq_1, output_path / f"{iteration_idx}" / f"clip_{pair_idx}_1.mp4", start_idx_1, end_idx_1)
        generate_clip(sim_state_2, obs_seq_2, act_seq_2, output_path / f"{iteration_idx}" / f"clip_{pair_idx}_2.mp4", start_idx_2, end_idx_2)

        # save the obs_seq_1, act_seq_1 to a tensor
        save_tensors(obs_seq_1, act_seq_1, output_path / f"{iteration_idx}" / f"tensor_{pair_idx}_1.pt")
        save_tensors(obs_seq_2, act_seq_2, output_path / f"{iteration_idx}" / f"tensor_{pair_idx}_2.pt")
        metadata.append({
            "pair_id": pair_idx,
            "obs1": {
                "clip_id": f"{pair_idx}_1",
                "tensor_path": f"{iteration_idx}" / f"tensor_{pair_idx}_1.pt",
                "video_path": f"{iteration_idx}" / f"clip_{pair_idx}_1.mp4"
            },
            "obs2": {
                "clip_id": f"{pair_idx}_2",
                "tensor_path": f"{iteration_idx}" / f"tensor_{pair_idx}_2.pt",
                "video_path": f"{iteration_idx}" / f"clip_{pair_idx}_2.mp4"
            }
        })

    with open(output_path / f"metadata" / f"batch_{iteration_idx}.json", "w") as f:
        wrapped_metadata = {
            "batch_id": iteration_idx,
            "pairs": metadata
        }
        json.dump(wrapped_metadata, f)



def main(base_path: pathlib.Path, iteration_idx: int):
    # load the policy model and reward model
    print(f"Loading models from {base_path / 'models'}")
    policy_model, reward_model = load_models(base_path / f"{iteration_idx}" / "models")
    # generate trajectories
    print("Generating trajectories")
    trajectories = generate_trajectories(policy_model, reward_model)
    print("Selecting pairs")
    pairs = select_pairs(trajectories, reward_model)
    print("Rendering and saving pairs")
    render_and_save_pairs(iteration_idx, base_path / "clips", trajectories, pairs)
    print(f"Generated {len(pairs)} pairs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=pathlib.Path, required=True)
    parser.add_argument("--iteration_idx", type=int, required=True)
    args = parser.parse_args()
    main(args.base_path, args.iteration_idx)