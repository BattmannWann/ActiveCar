import argparse
import logging
import os
import sys
import time
import cv2
import numpy as np
from omegaconf import OmegaConf
import torch

#Ensure local dependencies resolve
sys.path.extend([
    os.path.abspath("./assetto_corsa_gym"),
    os.path.abspath("./assetto_corsa_gym/assetto_corsa_gym"),
    os.path.abspath("./algorithm/discor"),
])

from gym_implementation.ac_active_vis_env import AssettoCorsaActiveVisionEnv
from assetto_corsa_gym.algorithm.discor.discor.network import GaussianPolicy


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_runner")


def parse_args():
    
    parser = argparse.ArgumentParser(description="Evaluate a Trained Active Vision SAC Model in Assetto Corsa")
    parser.add_argument("--config", default="src/config.yaml", type=str, help="Path to config.yaml")
    
    parser.add_argument("--model_path", required=True, type=str, help="Path to policy_net.pth or directory containing it")
    parser.add_argument("--episodes", default=3, type=int, help="Number of evaluation episodes/runs")
    parser.add_argument("--render", action="store_true", default=True, help="Display the 100ms foveal crop window")
    
    return parser.parse_args()


class PolicyEvaluator:
    
    """
    Loads a frozen GaussianPolicy and runs greedy inference (deterministic mode).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_units: list, model_file: str, device: torch.device):
        
        self.device = device
        
        self.policy = GaussianPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_units=hidden_units
        ).to(self.device).eval()

        if not os.path.isfile(model_file):
            raise FileNotFoundError(f"Model file not found at: {model_file}")

        logger.info(f"Loading policy weights from: {model_file}")
        state_dict = torch.load(model_file, map_location=self.device)
        self.policy.load_state_dict(state_dict)

        # Disable gradients for inference
        for param in self.policy.parameters():
            param.requires_grad = False


    @torch.no_grad()
    def get_action(self, state: np.ndarray) -> np.ndarray:
        
        """
        Greedy evaluation action (mean of Gaussian distribution, zero exploration noise).
        """
        
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # GaussianPolicy returns: (sampled_action, entropy, deterministic_mean_action)
        _, _, mean_action = self.policy(state_tensor)
        action = mean_action.squeeze(0).cpu().numpy()
        
        return np.clip(action, -1.0, 1.0)


def resolve_model_path(path_arg: str) -> str:
    
    """
    Finds policy_net.pth whether passed as a file path or directory.
    """
    
    if os.path.isdir(path_arg):
        candidate = os.path.join(path_arg, "policy_net.pth")
        
        if os.path.exists(candidate):
            return candidate
        
        # Look for best_model_step_*.pth
        files = [os.path.join(path_arg, f) for f in os.listdir(path_arg) if f.endswith(".pth")]
        
        if files:
            files.sort(key=os.path.getmtime, reverse=True)
            return files[0]
        
    return path_arg


def main():
    
    args = parse_args()
    config = OmegaConf.load(args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device for inference: {device}")

    #Instantiate the active vision environment
    logger.info("Initializing AssettoCorsaActiveVisionEnv...")
    env = AssettoCorsaActiveVisionEnv(
        config=config.AssettoCorsa,
        crop_size=96,
        send_reset_on_start=True,
        render_crop=args.render,
        visual_refresh_ms=100.0,
    )

    #Setup the policy network
    state_dim = env.observation_space.shape[0]  # 514
    action_dim = env.action_space.shape[0]      # 3
    
    hidden_units = list(config.SAC.policy_hidden_units)
    target_weights = resolve_model_path(args.model_path)

    evaluator = PolicyEvaluator(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_units=hidden_units,
        model_file=target_weights,
        device=device,
    )

    #Evaluation Loop
    logger.info(f"Starting evaluation across {args.episodes} episodes...")

    for ep in range(1, args.episodes + 1):
        
        state = env.reset()
        done = False
        
        ep_reward = 0.0
        step_count = 0
        
        speeds_kmh = []
        start_wall_time = time.time()

        while not done:
            
            # Deterministic, jitter-free policy control
            action = evaluator.get_action(state)

            env.set_actions(action)
            next_state, reward, done, info = env.step()

            state = next_state
            ep_reward += reward
            step_count += 1

            speed_ms = info.get("telemetry", {}).get("speed", 0.0)
            speeds_kmh.append(speed_ms * 3.6)

            if step_count >= env._max_episode_steps:
                logger.info("Episode reached maximum allowed step limit.")
                break

        elapsed = time.time() - start_wall_time
        avg_speed = np.mean(speeds_kmh) if speeds_kmh else 0.0
        max_speed = np.max(speeds_kmh) if speeds_kmh else 0.0

        logger.info(
            f"Episode {ep}/{args.episodes} Finished | "
            f"Reward: {ep_reward:.2f} | "
            f"Steps: {step_count} | "
            f"Avg Speed: {avg_speed:.1f} km/h | "
            f"Max Speed: {max_speed:.1f} km/h | "
            f"Sim Time: {step_count * 0.04:.1f}s | "
            f"Wall Time: {elapsed:.1f}s"
        )

    env.close()
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()