import argparse
import glob
import logging
import os
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np
from omegaconf import OmegaConf
import torch

# Ensure local dependencies resolve
sys.path.extend([
    os.path.abspath("./assetto_corsa_gym"),
    os.path.abspath("./assetto_corsa_gym/assetto_corsa_gym"),
    os.path.abspath("./algorithm/discor"),
])

from gym_implementation.ac_active_vis_env import AssettoCorsaActiveVisionEnv
from assetto_corsa_gym.algorithm.discor.discor.network import GaussianPolicy
from tensorboard_logger import ActiveVisionTBLogger

logging.basicConfig(level = logging.INFO, format = "%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_runner")


def parse_args():
    
    parser = argparse.ArgumentParser(description="Evaluate Trained Active Vision Policy in Assetto Corsa")
    parser.add_argument("--config", default = "src/config.yaml", type = str, help = "Path to config.yaml")
    
    parser.add_argument(
        "--model_path",
        required = True,
        type = str,
        help = "Path to run dir, 'best_models/' folder, step sub-folder, or direct policy_net.pth",
    )
    
    parser.add_argument("--episodes", default = 3, type = int, help = "Number of evaluation episodes")
    parser.add_argument("--render", action = "store_true", default = True, help = "Display OpenCV foveation window")
    parser.add_argument("--log_tb", action = "store_true", default = True, help = "Log eval telemetry and vision to TensorBoard")
    
    return parser.parse_args()


def resolve_policy_checkpoint(target_path: str) -> str:
    
    """
    Intelligently locates 'policy_net.pth' given various logging directory structures:

        1. Direct file path: .../policy_net.pth
        2. Specific checkpoint folder: .../best_models/step_0025000/
        3. 'best_models' directory: resolves 'latest_best/' or the highest step folder
        4. Experiment root: .../outputs/active_vision_TIMESTAMP/
    """
    
    target = os.path.abspath(target_path)

    #Direct file target
    if os.path.isfile(target):
        return target

    if not os.path.isdir(target):
        raise FileNotFoundError(f"Provided path does not exist: {target}")

    #Check if policy_net.pth is directly inside this folder
    direct_policy = os.path.join(target, "policy_net.pth")
    
    if os.path.isfile(direct_policy):
        return direct_policy

    #Check for 'latest_best' pointer directory
    latest_best_ptr = os.path.join(target, "best_models", "latest_best", "policy_net.pth")
    
    if os.path.isfile(latest_best_ptr):
        return latest_best_ptr

    latest_in_folder = os.path.join(target, "latest_best", "policy_net.pth")
    
    if os.path.isfile(latest_in_folder):
        return latest_in_folder

    #Search recursively for step_* directories
    candidates = glob.glob(os.path.join(target, "**", "policy_net.pth"), recursive=True)
    
    if candidates:
        # Sort by step number or last modified timestamp to select the newest/best
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]

    raise FileNotFoundError(f"Could not find any 'policy_net.pth' inside {target}")


class PolicyEvaluator:
    
    """
    Loads GaussianPolicy in deterministic mode (zero exploration noise).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_units: list, model_file: str, device: torch.device):
        
        self.device = device
        self.policy = GaussianPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_units=hidden_units,
        ).to(self.device).eval()

        logger.info(f"Loading weights from: {model_file}")
        state_dict = torch.load(model_file, map_location=self.device)
        self.policy.load_state_dict(state_dict)

        for param in self.policy.parameters():
            param.requires_grad = False


    @torch.no_grad()
    def get_action(self, state: np.ndarray) -> np.ndarray:
        
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # GaussianPolicy returns: (sampled_action, entropy, deterministic_mean)
        _, _, mean_action = self.policy(state_t)
        action = mean_action.squeeze(0).cpu().numpy()
        
        return np.clip(action, -1.0, 1.0)


def main():
    
    args = parse_args()
    config = OmegaConf.load(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device for evaluation inference: {device}")

    #Resolve Checkpoint Path
    model_checkpoint = resolve_policy_checkpoint(args.model_path)
    logger.info(f"Selected Policy Checkpoint: {model_checkpoint}")

    #Determine base experiment folder for TensorBoard logging
    checkpoint_dir = os.path.dirname(model_checkpoint)
    parent_dir = os.path.abspath(os.path.join(checkpoint_dir, "..", ".."))

    #Setup Evaluation TensorBoard Logger
    tb_logger: Optional[ActiveVisionTBLogger] = None
    
    if args.log_tb:
        eval_log_dir = os.path.join(parent_dir, "eval_logs")
        tb_logger = ActiveVisionTBLogger(run_dir = parent_dir, tb_subdir = "eval_logs")
        logger.info(f"Logging evaluation telemetry to: {eval_log_dir}")

    #Instantiate Environment
    logger.info("Initializing AssettoCorsaActiveVisionEnv...")
    
    env = AssettoCorsaActiveVisionEnv(
        config=config.AssettoCorsa,
        crop_size=96,
        send_reset_on_start=True,
        render_crop=args.render,
        visual_refresh_ms=100.0,
    )

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    hidden_units = list(config.SAC.policy_hidden_units)

    evaluator = PolicyEvaluator(
        state_dim = state_dim,
        action_dim = act_dim,
        hidden_units = hidden_units,
        model_file = model_checkpoint,
        device = device,
    )

    #Evaluation Loop
    total_eval_steps = 0
    
    try:
        for ep in range(1, args.episodes + 1):
            
            state = env.reset()
            done = False
            
            ep_reward = 0.0
            ep_steps = 0
            
            speeds_kmh = []
            start_wall_time = time.time()

            logger.info(f"--- Starting Evaluation Episode {ep}/{args.episodes} ---")

            while not done:
                
                total_eval_steps += 1
                ep_steps += 1

                #Deterministic, jitter-free policy control
                action = evaluator.get_action(state)

                env.set_actions(action)
                next_state, reward, done, info = env.step()

                state = next_state
                ep_reward += reward

                telemetry = info.get("telemetry", {})
                speed_kmh = telemetry.get("speed", 0.0) * 3.6
                speeds_kmh.append(speed_kmh)

                # Periodic Visual Logging to TensorBoard during evaluation (every 50 steps)
                if tb_logger and ep_steps % 50 == 0:
                    
                    tb_logger.log_vision_pipeline(
                        step=total_eval_steps,
                        frame_rgb=getattr(env, "latest_frame_rgb", None),
                        heatmap=getattr(env, "latest_heatmap", None),
                        gaze_xy=getattr(env, "latest_gaze", (0, 0)),
                        crop_rgb=getattr(env, "latest_crop", None),
                        tag="Eval_Vision/Fixation_Overview",
                    )

                if ep_steps >= env._max_episode_steps:
                    logger.info(f"Episode {ep} reached maximum step limit ({env._max_episode_steps}).")
                    break

            elapsed = time.time() - start_wall_time
            avg_speed = np.mean(speeds_kmh) if speeds_kmh else 0.0
            max_speed = np.max(speeds_kmh) if speeds_kmh else 0.0

            # Log episode metrics to TensorBoard
            if tb_logger:
                
                tb_logger.log_scalars({
                    "Eval/Episode_Reward": ep_reward,
                    "Eval/Episode_Steps": ep_steps,
                    "Eval/Avg_Speed_kmh": avg_speed,
                    "Eval/Max_Speed_kmh": max_speed,
                    "Eval/Lap_Time_s": ep_steps * 0.04,  # 25 Hz = 0.04s per step
                }, step=ep)

            logger.info(
                f"[RESULT] Ep {ep}/{args.episodes} | "
                f"Reward: {ep_reward:.2f} | "
                f"Avg Speed: {avg_speed:.1f} km/h | "
                f"Peak Speed: {max_speed:.1f} km/h | "
                f"Sim Duration: {ep_steps * 0.04:.2f}s | "
                f"Wall Time: {elapsed:.2f}s"
            )

    finally:
        
        logger.info("Cleaning up sockets and evaluation loggers...")
        
        if tb_logger:
            tb_logger.close()
            
        env.close()


if __name__ == "__main__":
    main()