import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import sys

sys.path.extend([
    os.path.abspath("./assetto_corsa_gym"),
    os.path.abspath("./assetto_corsa_gym/assetto_corsa_gym"),
    os.path.abspath("./algorithm/discor"),
])

from assetto_corsa_gym.algorithm.discor.discor.network import TwinnedStateActionFunction, GaussianPolicy
from assetto_corsa_gym.algorithm.discor.discor.utils import disable_gradients, soft_update, assert_action
# from discor.network import TwinnedStateActionFunction, GaussianPolicy
# from discor.utils import disable_gradients, soft_update, assert_action


class CleanSAC:
    """Production-grade Soft Actor-Critic (SAC) with automatic entropy tuning,

    isolated target network Polyak averaging, gradient clipping, and flat action returns.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        gamma: float = 0.992,
        nstep: int = 3,
        policy_lr: float = 0.0003,
        q_lr: float = 0.0003,
        entropy_lr: float = 0.0003,
        policy_hidden_units: list = [256, 256, 256],
        q_hidden_units: list = [256, 256, 256],
        target_update_coef: float = 0.005,
        max_grad_norm: float = 10.0,
        log_interval: int = 10,
        seed: int = 0,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.nstep = nstep
        self.discount = gamma ** nstep
        self.target_update_coef = target_update_coef
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.learning_steps = 0

        # 1. Actor & Critic Networks
        self.policy_net = GaussianPolicy(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_units=policy_hidden_units
        ).to(self.device)

        self.online_q_net = TwinnedStateActionFunction(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_units=q_hidden_units
        ).to(self.device)

        self.target_q_net = TwinnedStateActionFunction(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_units=q_hidden_units
        ).to(self.device).eval()

        self.target_q_net.load_state_dict(self.online_q_net.state_dict())
        disable_gradients(self.target_q_net)

        # 2. Optimizers
        self.policy_optim = Adam(self.policy_net.parameters(), lr=policy_lr)
        self.q_optim = Adam(self.online_q_net.parameters(), lr=q_lr)

        # 3. Dynamic Entropy Tuning
        self.target_entropy = -float(self.action_dim)
        self.log_alpha = torch.zeros(1, device=self.device, requires_grad=True)
        self.alpha_optim = Adam([self.log_alpha], lr=entropy_lr)
        self.alpha = self.log_alpha.detach().exp()


    @torch.no_grad()
    def explore(self, state: np.ndarray) -> np.ndarray:
        
        """
        Returns action as a flat 1D numpy array.
        """
        
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t, _, _ = self.policy_net(state_t)
        
        action = action_t.squeeze(0).cpu().numpy()
        assert_action(action)
        
        return action


    @torch.no_grad()
    def exploit(self, state: np.ndarray) -> np.ndarray:
        
        """
        Returns deterministic evaluation action as a flat 1D numpy array.
        """
        
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        _, _, action_t = self.policy_net(state_t)
        
        action = action_t.squeeze(0).cpu().numpy()
        assert_action(action)
        
        return action


    def update(self, replay_buffer, batch_size: int = 128) -> dict:
        
        """
        Performs a full learning step: Critic update -> Actor update ->

        Entropy update -> Polyak Target Network soft-update.
        """
        
        self.learning_steps += 1

        # Sample batch from DisCor ReplayBuffer (states, actions, rewards, next_states, dones)
        batch = replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = [torch.as_tensor(x, device=self.device, dtype=torch.float32) for x in batch]

        # ------------------- Critic Update ------------------- #
        curr_q1, curr_q2 = self.online_q_net(states, actions)
        
        with torch.no_grad():
            next_actions, next_entropies, _ = self.policy_net(next_states)
            target_q1, target_q2 = self.target_q_net(next_states, next_actions)
            
            min_target_q = torch.min(target_q1, target_q2) + self.alpha * next_entropies
            target_q = rewards + (1.0 - dones) * self.discount * min_target_q

        q1_loss = torch.mean((curr_q1 - target_q).pow(2))
        q2_loss = torch.mean((curr_q2 - target_q).pow(2))
        q_loss = q1_loss + q2_loss

        self.q_optim.zero_grad()
        q_loss.backward()
        
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.online_q_net.parameters(), self.max_grad_norm)
            
        self.q_optim.step()

        # ------------------- Policy Update ------------------- #
        sampled_actions, entropies, _ = self.policy_net(states)
        q1_pi, q2_pi = self.online_q_net(states, sampled_actions)
        
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = torch.mean(-min_q_pi - self.alpha * entropies)

        self.policy_optim.zero_grad()
        policy_loss.backward()
        
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            
        self.policy_optim.step()

        # ------------------- Entropy Alpha Update ------------ #
        entropy_loss = -torch.mean(self.log_alpha * (self.target_entropy - entropies.detach()))
        self.alpha_optim.zero_grad()
        
        entropy_loss.backward()
        self.alpha_optim.step()
        
        self.alpha = self.log_alpha.detach().exp()

        # ------------------- Target Network Update ----------- #
        soft_update(self.target_q_net, self.online_q_net, self.target_update_coef)

        return {
            "loss/Q": q_loss.item(),
            "loss/policy": policy_loss.item(),
            "loss/alpha": entropy_loss.item(),
            "stats/alpha": self.alpha.item(),
            "stats/mean_Q": curr_q1.mean().item(),
            "stats/entropy": entropies.detach().mean().item(),
        }
        

    def save_models(self, save_dir: str):
        
        os.makedirs(save_dir, exist_ok=True)
        self.policy_net.save(os.path.join(save_dir, "policy_net.pth"))
        
        self.online_q_net.save(os.path.join(save_dir, "online_q_net.pth"))
        self.target_q_net.save(os.path.join(save_dir, "target_q_net.pth"))


    def load_models(self, load_dir: str):
        
        self.policy_net.load(os.path.join(load_dir, "policy_net.pth"))
        self.online_q_net.load(os.path.join(load_dir, "online_q_net.pth"))
        self.target_q_net.load(os.path.join(load_dir, "target_q_net.pth"))