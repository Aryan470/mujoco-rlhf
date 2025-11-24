import matplotlib.pyplot as plt
import torch
import numpy as np
import gymnasium as gym
from torch.distributions import Normal
from train_policy import load_policy_model
import os
def reward_fn(a, ob):
    """
    from paper website
    """
    backroll = -ob[7]
    height = ob[0]
    vel_act = a[0] * ob[8] + a[1] * ob[9] + a[2] * ob[10]
    backslide = -ob[5]
    return backroll * (1.0 + 0.3 * height + 0.1 * vel_act + 0.05 * backslide)


def run_episode_true_reward(policy_model, env, max_steps=2000,):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_model.eval()
    obs, _ = env.reset(seed=0)

    rewards = []

    with torch.no_grad():
        for t in range(max_steps):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            _, _, mean, std = policy_model(obs_tensor)
            std = std.expand_as(mean)
            dist = Normal(mean, std)
            act = dist.sample().squeeze(0).cpu().numpy()
            next_obs, _, terminated, truncated, _ = env.step(act)
            #true reward
            r = reward_fn(act, obs)
            rewards.append(r)
            obs = next_obs
            if terminated or truncated:
                break

    return np.array(rewards)


def evaluate_and_plot_policies(policy_paths,
                               device="cuda" if torch.cuda.is_available() else "cpu",
                               max_steps=2000,
                               title="hopper backflip (true reward)"):
 
    labels = [f"policy_{(i+1)*200}_examples" for i in range(len(policy_paths))]

    env = gym.make("Hopper-v5", render_mode=None, terminate_when_unhealthy=False)
    dt = getattr(env.unwrapped, "dt", 1.0) 

    plt.figure(figsize=(4, 4))

    for path, label in zip(policy_paths, labels):
        policy = load_policy_model(path, device=device)
        rewards = run_episode_true_reward(policy, env, max_steps=max_steps)

        time = np.arange(len(rewards)) * dt

        if len(rewards) > 5:
            window = 10
            kernel = np.ones(window) / window
            rewards_smoothed = np.convolve(rewards, kernel, mode="valid")
            time_smoothed = time[window - 1:]
        else:
            rewards_smoothed = rewards
            time_smoothed = time

        plt.plot(time, rewards, label=label)

    plt.title(title)
    plt.xlabel("time (s)")
    plt.ylabel("reward_fn per step")
    plt.tight_layout()
    plt.legend()
    # plt.show()

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{title}.png")

    plt.close()
    env.close()


if __name__ == "__main__":
    policy_paths = [
        "data/0/models/checkpoints/policy.pt",
        "data/1/models/checkpoints/policy.pt",
        "data/2/models/checkpoints/policy.pt",
    ]
    evaluate_and_plot_policies(policy_paths,
                               device="cuda" if torch.cuda.is_available() else "cpu",
                               max_steps=2000,
                               title="hopper backflip (true reward)")