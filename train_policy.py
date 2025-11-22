import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from policy_model import PolicyModel
from reward_model import RewardModel
import os
import gymnasium as gym
from torch.distributions import Normal, kl_divergence
from torch.utils.tensorboard import SummaryWriter 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
from reward_model import compute_preference_probs_loss

OBS_DIM = 11
ACTION_DIM = 3

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

def set_flat_params(model, flat):
    idx = 0
    for p in model.actor.parameters():
        n = p.numel()
        p.data.copy_(flat[idx:idx+n].view_as(p))
        idx += n
    return model

def get_flat_params(model):
    flat = []
    for p in model.actor.parameters():
        flat.append(p.data.view(-1))
    return torch.cat(flat)

def get_flat_grad(loss, model):
    grad = torch.autograd.grad(loss, model.actor.parameters())
    return torch.cat([g.view(-1) for g in grad])

def fisher_vector_product(policy_model, states, mean_old, std_old, v, damping=0.1):
    _, _, mean, std = policy_model(states)

    dist_old = Normal(mean_old, std_old)
    dist_new = Normal(mean, std)

    kl = kl_divergence(dist_old, dist_new).sum(-1).mean()

    kl_grad = torch.autograd.grad(kl, policy_model.actor.parameters(), create_graph=True)
    flat_kl_grad = torch.cat([g.view(-1) for g in kl_grad])

    kl_v = (flat_kl_grad * v).sum()
    kl_hvp = torch.autograd.grad(kl_v, policy_model.actor.parameters(), retain_graph=True)
    flat_kl_hvp = torch.cat([g.view(-1) for g in kl_hvp]).detach()

    return flat_kl_hvp + damping * v

def conjugate_gradient(Avp_fn, b, nsteps=10, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)

    for _ in range(nsteps):
        Avp = Avp_fn(p)
        alpha = rdotr / (torch.dot(p, Avp) + 1e-8)
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / (rdotr + 1e-8)
        p = r + beta * p
        rdotr = new_rdotr
    return x

def trpo_step(policy_model, states, actions, advantages, logp_old, mean_old, std_old, max_kl, cg_iters=10, backtrack_coeff=0.8, backtrack_iters=10):
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    _, _, mean_new, std_new = policy_model(states)
    dist_new = Normal(mean_new, std_new)

    logp = dist_new.log_prob(actions).sum(-1)
    ratio = torch.exp(logp - logp_old) 

    surr_loss = -(ratio * advantages).mean()
    grad = get_flat_grad(surr_loss, policy_model).detach()

    def Avp_fn(v):
        return fisher_vector_product(policy_model, states, mean_old, std_old, v)

    step_dir = conjugate_gradient(Avp_fn, -grad, nsteps=cg_iters)

    shs = 0.5 * step_dir.dot(Avp_fn(step_dir))   
    step_size = torch.sqrt(2.0 * max_kl / (shs + 1e-8))
    full_step = step_size * step_dir
    old_params = get_flat_params(policy_model)

    @torch.no_grad()
    def get_loss_and_kl():
        _, _, m_new, s_new = policy_model(states)
        d_new = Normal(m_new, s_new)
        logp_new = d_new.log_prob(actions).sum(-1)
        ratio_new = torch.exp(logp_new - logp_old)
        surr_loss_new = -(ratio_new * advantages).mean()

        d_old = Normal(mean_old, std_old)
        kl_new = kl_divergence(d_old, d_new).sum(-1).mean()
        return surr_loss_new, kl_new
    
    with torch.no_grad():
        old_loss, old_kl = get_loss_and_kl()
    
    for i in range(backtrack_iters):
        coeff = backtrack_coeff ** i
        new_params = old_params + coeff * full_step
        set_flat_params(policy_model, new_params)

        with torch.no_grad():
            new_loss, kl_new = get_loss_and_kl()

        if (kl_new <= max_kl) and (new_loss <= old_loss):
            return
    
    #line search failed
    set_flat_params(policy_model, old_params)
    



def train_policy_model(
    policy_checkpoint_path: str = "data/0/models/checkpoints/policy.pt",
    reward_checkpoint_path: str = "data/0/models/checkpoints/reward.pt",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    steps_per_iter: int = 4096,
    num_iters: int = 300,
    gamma: float = 0.995,
    lam: float = 0.97,
    max_kl: float = 1e-2,
    critic_lr: float = 3e-4,
    critic_updates_iter: int = 80,
    save_base_path: str = "models/checkpoints/policy_trpo.pt",
    log_dir: str = "runs/trpo",
    phase: int = 1,
):
    
    policy_model = load_policy_model(policy_checkpoint_path, device)
    reward_model = load_reward_model(reward_checkpoint_path, device)
    reward_model.eval()
    
    critic_optimizer = optim.Adam(policy_model.critic.parameters(), lr=critic_lr)
    checkpoint = torch.load(policy_checkpoint_path, map_location=device)
    if "optimizer_state_dict" in checkpoint:
        if checkpoint["optimizer_state_dict"] is not None:
            critic_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    log_dir = os.path.join(log_dir, f"phase_{phase}")

    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to: {log_dir}")


    env = gym.make("Hopper-v5", render_mode=None)
    obs, info = env.reset(seed=0)

    for it in range(num_iters):
        policy_model.train()

        obs_buf = []
        act_buf = []
        rew_buf = []
        val_buf = []
        logp_buf = []
        done_buf = []
        mean_buf = []
        std_buf = []

        ep_returns = []
        ep_lengths = []
       
        ep_return, ep_length = 0, 0

        for step in range(steps_per_iter):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            action, value, mean, std = policy_model(obs_tensor)
            std = std.expand_as(mean)
            policy_dist = Normal(mean, std)

            action = policy_dist.sample()                             
            logp = policy_dist.log_prob(action).sum(-1)      

            mean = mean.detach().cpu().numpy()[0]  
            std = std.detach().cpu().numpy()[0]        
                
            action_np = action.squeeze(0).detach().cpu().numpy()

            next_obs, _, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated


            with torch.no_grad():
                obs_as_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action_as_tensor = torch.as_tensor(action_np, dtype=torch.float32, device=device).unsqueeze(0)
                reward_ensemble = reward_model(obs_as_tensor, action_as_tensor)       
                reward = reward_ensemble.mean(dim=-1).item()
            
            ep_return += reward
            ep_length += 1

            obs_buf.append(obs)
            act_buf.append(action_np)
            rew_buf.append(reward)
            val_buf.append(value.item())
            logp_buf.append(logp.item())
            done_buf.append(done)
            mean_buf.append(mean)
            std_buf.append(std)

            obs = next_obs

            if done:
                ep_returns.append(ep_return)
                ep_return = 0.0
                ep_lengths.append(ep_length)
                ep_length = 0
                obs, info = env.reset()

        with torch.no_grad():
            obs_tensor_last = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            _,last_val,_,_ = policy_model(obs_tensor_last)
            last_val = last_val.item() 
        
        obs_tensor = torch.as_tensor(np.array(obs_buf), dtype=torch.float32, device=device)        
        actions_tensor = torch.as_tensor(np.array(act_buf), dtype=torch.float32, device=device)   
        rewards = torch.as_tensor(np.array(rew_buf), dtype=torch.float32, device=device)         
        values = torch.as_tensor(np.array(val_buf + [last_val]), dtype=torch.float32, device=device)  
        logp_old_tensor = torch.as_tensor(np.array(logp_buf), dtype=torch.float32, device=device) 
        mean_old_tensor = torch.as_tensor(np.array(mean_buf), dtype=torch.float32, device=device)  
        std_old_tensor = torch.as_tensor(np.array(std_buf), dtype=torch.float32, device=device)  

        dones_np = np.array(done_buf, dtype=np.bool_)
        dones = torch.as_tensor(dones_np.astype(np.float32), device=device)  

        N = rewards.shape[0]

        advantages = torch.zeros(N, dtype=torch.float32, device=device)
        last_gae_lam = 0.0
        for t in reversed(range(N)):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
            last_gae_lam = delta + gamma * lam * nonterminal * last_gae_lam
            advantages[t] = last_gae_lam
        returns = advantages + values[:-1]

        trpo_step(
            policy_model,
            obs_tensor,
            actions_tensor,
            advantages.detach(),
            logp_old_tensor.detach(),
            mean_old_tensor.detach(),
            std_old_tensor.detach(),
            max_kl=max_kl,
            cg_iters=10,
            backtrack_coeff=0.8,
            backtrack_iters=10,
        )

        policy_model.train()
        for _ in range(critic_updates_iter):
            v_pred = policy_model.critic(obs_tensor).squeeze(-1)
            critic_loss = F.mse_loss(v_pred, returns.detach())
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

        avg_return = float(np.mean(ep_returns)) if ep_returns else 0.0
        avg_ep_len = float(np.mean(ep_lengths)) if ep_lengths else 0.0

        writer.add_scalar("train/avg_return", avg_return, it)
        writer.add_scalar("train/avg_ep_len", avg_ep_len, it)
        if critic_loss is not None:
            writer.add_scalar("train/critic_loss", critic_loss, it)

        save_path = os.path.join(f"data/{phase}",save_base_path)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": policy_model.state_dict(),
                "iteration": it + 1,
                "reward_model": reward_model.state_dict(),
                "optimizer_state_dict": critic_optimizer.state_dict(),
            },
            save_path,
        )
        print(f"Saved policy model to: {save_path}")
        print(f"Iteration {it + 1}, Average Return: {avg_return:.2f}, Average Episode Length: {avg_ep_len:.2f}")

    return policy_model


class PreferenceDatasetFromJSON(Dataset):
    """
    Dataset of human trajectory preferences, reading from the JSON you described.

    Each __getitem__ returns:
        seg1_obs: [T, OBS_DIM]
        seg1_act: [T, ACTION_DIM]
        seg2_obs: [T, OBS_DIM]
        seg2_act: [T, ACTION_DIM]
        label: scalar float in {0.0, 0.5, 1.0}
    """

    GRADE_TO_LABEL = {
    "pair1_better": 1.0,
    "pair2_better": 0.0,
    "similar": 0.5
    }

    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            data = json.load(f)

        pairs = data["pairs"]

        self.pairs = [
            p for p in pairs
            if p.get("grade") in self.GRADE_TO_LABEL
        ]

        if len(self.pairs) == 0:
            raise ValueError("No usable pairs (with valid 'grade') found in JSON.")

    def __len__(self):
        return len(self.pairs)

    def _load_clip(self, clip_info):
        tensor_path = clip_info["tensor_path"]
        x = torch.load(tensor_path, map_location="cpu")

        obs = x["obs"]        
        act = x["act"]          

        return obs.float(), act.float()

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        seg1_obs, seg1_act = self._load_clip(pair["obs1"])
        seg2_obs, seg2_act = self._load_clip(pair["obs2"])

        label = self.GRADE_TO_LABEL[pair["grade"]]
        label = torch.tensor(label, dtype=torch.float32)

        return seg1_obs, seg1_act, seg2_obs, seg2_act, label


def train_reward_model(
    preferences_json_path: str,
    checkpoint_path: str = "data/0/models/checkpoints/reward.pt",
    save_base_path: str = "models/checkpoints/reward_trained.pt",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 32,
    num_epochs: int = 30,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    phase: int = 1,
    log_dir: str = "runs/reward_model",
):
    reward_model = load_reward_model(checkpoint_path, device)
    dataset = PreferenceDatasetFromJSON(preferences_json_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(reward_model.parameters(), lr=lr, weight_decay=weight_decay)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "optimizer_state_dict" in checkpoint:
        if checkpoint["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
 
    log_dir = os.path.join(log_dir, f"phase_{phase}")

    writer = SummaryWriter(log_dir=log_dir)
    global_step = 0

    reward_model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            seq1_obs, seq1_act, seq2_obs, seq2_act, labels = batch

            loss = compute_preference_probs_loss(
                reward_model,
                seq1_obs,
                seq1_act,
                seq2_obs,
                seq2_act,
                labels,
                device=device,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * seq1_obs.size(0)
            writer.add_scalar("reward_model/train_loss", loss.item(), global_step)
            global_step += 1
        
        avg_loss = epoch_loss / len(dataset)
        writer.add_scalar("reward_model/epoch_loss", avg_loss, epoch)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.6f}")

        save_path = os.path.join(f"data/{phase}",save_base_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": reward_model.state_dict(),
                "iteration": epoch + 1,
                "optimizer_state_dict": optimizer.state_dict(),
            },
            save_path,
        )

        print(f"Saved reward model to: {save_path}")
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.6f}")

    return reward_model



def train_proc(phase_num: int):
    train_reward_model(
        preferences_json_path=f"data/metadata/batch_{phase_num - 1}_results.json",
        checkpoint_path=f"data/{phase_num - 1}/models/checkpoints/reward.pt",
        save_base_path="models/checkpoints/reward.pt",
        device="cuda",
        batch_size=32,
        num_epochs=10,
        lr=1e-4,
        weight_decay=1e-4,
        phase=phase_num,
        log_dir="runs/reward_model",
    )
    train_policy_model(
        policy_checkpoint_path=f"data/{phase_num - 1}/models/checkpoints/policy.pt",
        reward_checkpoint_path=f"data/{phase_num - 1}/models/checkpoints/reward.pt",
        device="cuda",
        steps_per_iter=4096,
        num_iters=100,
        gamma=0.99,
        lam=0.97,
        max_kl=1e-2,
        critic_lr=3e-4,
        critic_updates_iter=80,
        save_base_path="models/checkpoints/policy.pt",
        phase=phase_num,
        log_dir="runs/trpo",
    )



if __name__ == "__main__":
    train_reward_model(
        preferences_json_path="data/metadata/batch_0_results.json",
        checkpoint_path="data/0/models/checkpoints/reward.pt",
        save_base_path="models/checkpoints/reward_trained.pt",
        device="cuda",
        batch_size=32,
        num_epochs=30,
        lr=1e-4,
        weight_decay=1e-4,
        phase=1,
        log_dir="runs/reward_model",
    )
    train_policy_model(
        policy_checkpoint_path="data/0/models/checkpoints/policy.pt",
        reward_checkpoint_path="data/0/models/checkpoints/reward.pt",
        device="cuda",
        steps_per_iter=4096,
        num_iters=300,
        gamma=0.995,
        lam=0.97,
        max_kl=1e-2,
        critic_lr=3e-4,
        critic_updates_iter=80,
        save_base_path="models/checkpoints/policy_trpo.pt",
        phase=1,
        log_dir="runs/trpo",
    )

