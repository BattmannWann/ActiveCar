import argparse
from datetime import datetime
import logging
import os
import sys
import time

from omegaconf import OmegaConf
import torch
import numpy as np
from pathlib import Path

# Add paths for local modules and discor algorithm
# sys.path.extend([
#     os.path.abspath("./assetto_corsa_gym"),
#     os.path.abspath("./assetto_corsa_gym/assetto_corsa_gym"),
#     os.path.abspath("./algorithm/discor"),
# ])

SRC_DIR = Path(__file__).resolve().parent
discor_dir = SRC_DIR / "assetto_corsa_gym" / "algorithm" / "discor"

from gym_implementation.ac_active_vis_env import AssettoCorsaActiveVisionEnv
from tensorboard_logger import ActiveVisionTBLogger
import assetto_corsa_gym.common.logging_config as logging_config
from assetto_corsa_gym.common.logger import Logger
import assetto_corsa_gym.common.misc as misc
from clean_SAC import CleanSAC
from assetto_corsa_gym.algorithm.discor.discor.agent import Agent
from visual_processing.fixation_locator import UnbiasedSaliencyEngine, DrivingAssistedSpectralResidualSaliencyEngine

# import common.logging_config as logging_config
# from common.logger import Logger
# import common.misc as misc
# from discor.agent import Agent
# from discor.algorithm import SAC


logger = logging.getLogger(__name__)


def parse_args():
    
    parser = argparse.ArgumentParser(description="Active Vision SAC Training for Assetto Corsa")
    
    parser.add_argument("--config", default="config.yml", type=str, help="Path to configuration file")
    parser.add_argument("--load_path", type=str, default=None, help="Path to load a saved checkpoint")
    parser.add_argument("--test", action="store_true", help="Run in evaluation mode")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Override config key=value")
    
    args = parser.parse_args()
    
    if args.load_path is not None:
        args.load_path = os.path.abspath(args.load_path) + os.sep
        
    return args



class ConsoleTableSummary:
    
    """
    Renders formatted training summaries directly to stdout.
    """

    @staticmethod
    def print_episode_summary(
        episode: int,
        step: int,
        total_steps: int,
        ep_reward: float,
        ep_steps: int,
        speed_kmh: float,
        losses: dict,
        start_time: float,
    ):
        elapsed_sec = max(1.0, time.time() - start_time)
        fps = step / elapsed_sec
        progress_pct = (step / total_steps) * 100.0

        q_loss = losses.get("loss/Q", 0.0)
        pi_loss = losses.get("loss/policy", 0.0)
        alpha = losses.get("stats/alpha", 0.0)
        mean_q = losses.get("stats/mean_Q", 0.0)

        #Estimate remaining wall-clock time
        remaining_sec = (total_steps - step) / max(fps, 1e-3)
        rem_hrs = int(remaining_sec // 3600)
        rem_mins = int((remaining_sec % 3600) // 60)

        print("\n" + "=" * 80)
        print(f"  ACTIVE VISION RACING PIPELINE  |  EPISODE {episode:04d} COMPLETED")
        print("=" * 80)
        print(f" {'METRIC':<25} | {'CURRENT EPISODE':<22} | {'TRAINING HEALTH / STATS':<24}")
        print("-" * 80)
        print(f" {'Total Steps':<25} | {step:,} / {total_steps:,} ({progress_pct:.1f}%) | {'FPS (Sim Hz)':<16}: {fps:.1f}")
        print(f" {'Episode Duration':<25} | {ep_steps * 0.04:.1f}s ({ep_steps} steps)      | {'Time Remaining':<16}: ~{rem_hrs}h {rem_mins:02d}m")
        print(f" {'Episode Reward':<25} | {ep_reward:<22.3f} | {'Critic Loss (Q)':<16}: {q_loss:.4f}")
        print(f" {'Exit Speed':<25} | {speed_kmh:.1f} km/h              | {'Actor Loss (Pi)':<16}: {pi_loss:.4f}")
        print(f" {'Mean Q-Value':<25} | {mean_q:<22.3f} | {'Entropy Alpha':<16}: {alpha:.4f}")
        print("=" * 80 + "\n")



class ActiveVisionAgent(Agent):
    
    """
    Integrates clean SAC updates with periodic visual foveation logging.
    """

    def __init__(self, *args, tb_logger: ActiveVisionTBLogger = None, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.tb_logger = tb_logger
        
        self.latest_losses = {}
        self.train_start_wall_time = time.time()

    def train_episode(self):
        
        self._episodes += 1
        episode_reward = 0.0
        episode_steps = 0
        done = False
        state = self._env.reset()

        while not done:
            self._steps += 1
            episode_steps += 1

            #Action Selection
            if self._steps <= self._start_steps:
                action = self._env.action_space.sample()
                
            else:
                action = self._algo.explore(state)

            self._env.set_actions(action)
            next_state, reward, done, info = self._env.step()

            #Append to Replay Buffer
            self._replay_buffer.append(
                state, action, reward, next_state,
                done and not (episode_steps >= self._env._max_episode_steps)
            )

            state = next_state
            episode_reward += reward

            #Policy & Critic Updates
            if self._steps > self._start_steps and self._steps % self._update_interval == 0:
                
                loss_dict = self._algo.update(self._replay_buffer, self._batch_size)
                self.latest_losses = loss_dict
                
                if self.tb_logger and self._steps % self._log_interval == 0:
                    self.tb_logger.log_scalars(loss_dict, step=self._steps)
                    
                    
            # Fixed-interval checkpointing 
            if getattr(self, "checkpoint_freq", 0) and (self._steps % self.checkpoint_freq == 0):
                
                checkpoint_dir = os.path.join(self._model_dir, "checkpoints", f"step_{self._steps:08d}")
                os.makedirs(checkpoint_dir, exist_ok = True)
                
                logger.info(f"Checkpointing model at {self._steps:,} steps -> {checkpoint_dir}")
                self.save(checkpoint_dir, save_buffer = False)
                

            #Periodic Active Vision Preview Logging (every 100 steps)
            if self.tb_logger and self._steps % 100 == 0:
                
                self.tb_logger.log_vision_pipeline(
                    step=self._steps,
                    frame_rgb=getattr(self._env, "latest_frame_rgb", None),
                    heatmap=getattr(self._env, "latest_heatmap", None),
                    gaze_xy=getattr(self._env, "latest_gaze", (0, 0)),
                    crop_rgb=getattr(self._env, "latest_crop", None),
                )

            if done or (episode_steps >= self._env._max_episode_steps):
                break

        #Episode Summary Telemetry
        if self.tb_logger:
            
            telemetry = info.get("telemetry", {})
            speed_kmh = telemetry.get("speed", 0.0) * 3.6
            
            self.tb_logger.log_scalars({
                "Train/Episode_Reward": episode_reward,
                "Train/Episode_Steps": episode_steps,
                "Train/Speed_kmh": speed_kmh,
            }, step = self._steps)

            self.tb_logger.check_and_save_winner(
                algo = self._algo,
                current_reward = episode_reward,
                step = self._steps,
                telemetry = telemetry,
                
                vision_data = {
                    "frame_rgb": getattr(self._env, "latest_frame_rgb", None),
                    "heatmap": getattr(self._env, "latest_heatmap", None),
                    "gaze_xy": getattr(self._env, "latest_gaze", (0, 0)),
                    "crop_rgb": getattr(self._env, "latest_crop", None),
                }
            )
            
        #Output Console ASCII Summary Table
        ConsoleTableSummary.print_episode_summary(
            episode = self._episodes,
            step = self._steps,
            total_steps = self._steps_total if hasattr(self, "_steps_total") else self._num_steps,
            ep_reward = episode_reward,
            ep_steps = episode_steps,
            speed_kmh = speed_kmh,
            losses = getattr(self, "latest_losses", {}),
            start_time = getattr(self, "train_start_wall_time", time.time()),
        )


def main():
    
    args = parse_args()

    #Load base configuration and merge CLI overrides
    config = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_dotlist(args.overrides)
    config = OmegaConf.merge(config, cli_conf)

    #Setup experiment output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.abspath(f"data/outputs/active_vision_{timestamp}")
    
    os.makedirs(work_dir, exist_ok=True)
    config.work_dir = work_dir
    
    # tb_dir = os.path.join(config.work_dir, "tb_logs")
    tb_logger = ActiveVisionTBLogger(run_dir = config.work_dir)

    logging_config.create_logging(level = logging.DEBUG, file_name = os.path.join(work_dir, "train.log"))
    logging.getLogger().setLevel(logging.INFO)

    misc.get_system_info()

    #Instantiate custom Active Vision Environment
    logger.info("Instantiating AssettoCorsaActiveVisionEnv (Active Vision Foveated Pipeline)...")
    env = AssettoCorsaActiveVisionEnv(config = config.AssettoCorsa, crop_size = 96, send_reset_on_start = True, visual_refresh_ms = 100.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device for SAC Policy & Q-Networks: {device}")

    #Initialise Soft Actor-Critic (SAC)
    obs_dim = env.observation_space.shape[0]  # 514
    act_dim = env.action_space.shape[0]        # 3
    logger.info(f"Setting up SAC with State Dimension: {obs_dim}, Action Dimension: {act_dim}")

    algo = CleanSAC(
        state_dim = obs_dim,
        action_dim = act_dim,
        device = device,
        seed = config.seed,
        **OmegaConf.to_container(config.SAC)
    )

    #Configure logging and telemetry metadata
    config.exp_name = f"active_vision-{config.AssettoCorsa.car}-{config.AssettoCorsa.track}"
    config.action_dim = env.action_dim
    config.steps = config.Agent.num_steps

    wandb_logger = Logger(config.copy()) if not config.disable_wandb else None

    #Instantiate DisCor/SAC Agent
    #For safety on live gym environments, eval_interval=0 avoids spawning secondary threads
    agent = ActiveVisionAgent(
        env = env,
        test_env = env,
        algo = algo,
        log_dir = config.work_dir,
        device = device,
        seed = config.seed,
        #**config.Agent,
        **OmegaConf.to_container(config.Agent),
        wandb_logger = wandb_logger,
        tb_logger = tb_logger
    )

    #Checkpoint loading or training start 
    
    try:
    
        if args.load_path is not None:
            
            logger.info(f"Loading checkpoint from: {args.load_path}")
            agent.load(args.load_path, load_buffer = not args.test)

        if args.test:
            
            logger.info("Running evaluation mode...")
            agent._env.set_eval_mode()
            agent.evaluate()
            
        else:
            
            logger.info("Starting Active Vision Online RL Training Loop...")
            agent.run()
            logger.info("Training completed.")
            
            # tb_logger.close()
            
    finally:
        
        logger.info(f"\nClosing environment and flushing loggers...")
        tb_logger.close()
        env.close()


if __name__ == "__main__":
    main()