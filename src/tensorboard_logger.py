import os
import shutil
import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class ActiveVisionTBLogger:
    
    """
    TensorBoard manager with automated checkpoint preservation and record tracking.
    """

    def __init__(self, run_dir: str, tb_subdir: str = "tb_logs", best_subdir: str = "best_models"):
        
        #Directory Structure
        self.run_dir = run_dir
        self.tb_dir = os.path.join(run_dir, tb_subdir)
        
        self.best_checkpoint_dir = os.path.join(run_dir, best_subdir)
        self.latest_best_dir = os.path.join(self.best_checkpoint_dir, "latest_best")

        os.makedirs(self.tb_dir, exist_ok=True)
        os.makedirs(self.best_checkpoint_dir, exist_ok=True)
        os.makedirs(self.latest_best_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.tb_dir)

        # Tracking High-Water Marks
        self.best_eval_reward = -float("inf")
        self.best_speed_kmh = 0.0
        self.records_broken_count = 0

        # Initialise the Record Ledger Table in TensorBoard Text Tab
        self.writer.add_text(
            "Best_Models/Ledger",
            "| Record # | Step | Episode Reward | Speed (km/h) | Checkpoint Directory |\n"
            "|:---:|:---:|:---:|:---:|:---|\n",
            global_step=0,
        )


    def log_scalars(self, tag_scalar_dict: dict, step: int):
        
        """
        Logs standard RL scalar values (losses, rewards, speeds).
        """
        for tag, value in tag_scalar_dict.items():
            if value is not None and not np.isnan(value):
                self.writer.add_scalar(tag, float(value), step)


    def log_vision_pipeline(
        self,
        step: int,
        frame_rgb: np.ndarray,
        heatmap: np.ndarray,
        gaze_xy: tuple,
        crop_rgb: np.ndarray,
        tag: str = "Vision/Pipeline_Overview",
    ):
        """
        Renders and logs a composite canvas showing:
        [Raw Frame + Gaze Target] | [Saliency Heatmap] | [Foveated Patch]
        """
        
        canvas = self._build_vision_canvas(frame_rgb, heatmap, gaze_xy, crop_rgb)
        
        if canvas is not None:
            tensor_canvas = torch.from_numpy(canvas).permute(2, 0, 1)
            self.writer.add_image(tag, tensor_canvas, step)


    def _build_vision_canvas(self, frame_rgb: np.ndarray, heatmap: np.ndarray, gaze_xy: tuple, crop_rgb: np.ndarray) -> np.ndarray:
        
        if frame_rgb is None or heatmap is None or crop_rgb is None:
            return None

        h, w = frame_rgb.shape[:2]

        # Raw frame with fixation marker
        vis_frame = frame_rgb.copy()
        gx, gy = int(gaze_xy[0]), int(gaze_xy[1])
        cv2.circle(vis_frame, (gx, gy), 6, (255, 0, 0), -1)
        cv2.circle(vis_frame, (gx, gy), 16, (0, 255, 255), 2)

        # Colored Saliency Heatmap
        norm_map = np.uint8(np.clip(heatmap, 0.0, 1.0) * 255)
        color_heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
        color_heatmap_rgb = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)

        # Upscale crop to match frame height for side-by-side alignment
        vis_crop = cv2.resize(crop_rgb, (h, h), interpolation=cv2.INTER_NEAREST)

        return np.hstack([vis_frame, color_heatmap_rgb, vis_crop])


    def check_and_save_winner(self, algo, current_reward: float, step: int, telemetry: dict = None, vision_data: dict = None,) -> bool:
        
        """
        Saves models, updates TensorBoard high-water mark curves,
        and records snapshot data when a new peak reward is achieved.
        """
        
        if current_reward <= self.best_eval_reward:
            return False

        self.best_eval_reward = current_reward
        self.records_broken_count += 1
        
        speed_kmh = (telemetry or {}).get("speed", 0.0) * 3.6
        
        if speed_kmh > self.best_speed_kmh:
            self.best_speed_kmh = speed_kmh

        #Directory-based save for SAC weights
        step_save_dir = os.path.join(self.best_checkpoint_dir, f"step_{step:07d}")
        os.makedirs(step_save_dir, exist_ok=True)
        algo.save_models(step_save_dir)

        #Update the stable pointer 'latest_best'
        algo.save_models(self.latest_best_dir)

        #Log High-Water Mark Curves (Scalars tab)
        self.writer.add_scalar("Best/Max_Episode_Reward", self.best_eval_reward, step)
        self.writer.add_scalar("Best/Max_Speed_kmh", self.best_speed_kmh, step)
        self.writer.add_scalar("Best/Record_Events", self.records_broken_count, step)

        #Log to the Text Tab Ledger
        ledger_entry = (
            f"| **#{self.records_broken_count}** | {step:,} | {current_reward:.3f} | "
            f"{speed_kmh:.1f} | `{step_save_dir}` |\n"
        )
        
        self.writer.add_text("Best_Models/Ledger", ledger_entry, global_step=step)

        #Save Vision Snapshot to TensorBoard Images Tab
        if vision_data:
            
            canvas = self._build_vision_canvas(
                frame_rgb=vision_data.get("frame_rgb"),
                heatmap=vision_data.get("heatmap"),
                gaze_xy=vision_data.get("gaze_xy"),
                crop_rgb=vision_data.get("crop_rgb"),
            )
            
            if canvas is not None:
                tensor_canvas = torch.from_numpy(canvas).permute(2, 0, 1)
                self.writer.add_image("Best_Models/Visual_Snapshot", tensor_canvas, step)

        print(
            f"\n{'='*70}\n"
            f"[NEW RECORD MODEL] Step {step:,} | Reward: {current_reward:.3f} | Speed: {speed_kmh:.1f} km/h\n"
            f"Weights archived to: {step_save_dir}\n"
            f"{'='*70}\n"
        )
        
        return True


    def close(self):
        self.writer.flush()
        self.writer.close()