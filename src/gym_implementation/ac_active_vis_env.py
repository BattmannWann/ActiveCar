import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import gym
from gym.spaces import Box
import mss
import numpy as np
import pygetwindow as gw
import torch

# Add assetto_corsa_gym to path
sys.path.append(os.path.abspath("assetto_corsa_gym/assetto_corsa_gym"))
from AssettoCorsaEnv.ac_client import Client

# Import modular vision components
from visual_processing.vision_feature_encoder import FovealVisionEncoder
from visual_processing.foveal_extractor import FovealExtractor
from visual_processing.fixation_locator import UnbiasedSaliencyEngine


class DirectWindowGrabber:
    """Grabs frames directly from the Assetto Corsa window using mss."""

    def __init__(self, window_title: str = "Assetto Corsa", target_width: int = 320, target_height: int = 240):
        
        self.window_title = window_title
        self.target_width = target_width
        self.target_height = target_height
        
        self.sct = mss.MSS()
        self.win = self._find_window()
        
        self.action_dim = 3
        
        


    def _find_window(self):
        
        windows = gw.getWindowsWithTitle(self.window_title)
        
        for w in windows:
            if ("Assetto Corsa" in w.title) and w.width > 200 and w.height > 200:
                return w
            
        return None


    def grab_bgr(self) -> np.ndarray:
        
        if self.win is not None:
            
            # Crop title bar and window padding
            monitor = {
                "top": self.win.top + 31,
                "left": self.win.left + 8,
                "width": max(1, self.win.width - 16),
                "height": max(1, self.win.height - 39),
            }
            
        else:
            monitor = self.sct.monitors[1]

        sct_img = self.sct.grab(monitor)
        frame = np.array(sct_img)[:, :, :3]
        
        return cv2.resize(frame, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)


class AssettoCorsaActiveVisionEnv(gym.Env):
    
    """Clean custom Gym environment for Assetto Corsa Active Vision.

    Action Space: Box(3,) -> [steer (-1 to 1), acc (-1 to 1), brake (-1 to 1)]
    Observation Space: Box(514,) -> [512-d ResNet-18 foveal features, norm_x, norm_y]
    
    visual_refresh_ms represents the expert biological fixation duration for racing 
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, config: Any, crop_size: int = 96, send_reset_on_start: bool = True, render_crop: bool = True, visual_refresh_ms: float = 100.0):
        
        super().__init__()
        self.config = config
        self.client = Client(config)
        self.send_reset_on_start = send_reset_on_start
        self.render_crop = render_crop
        self.visual_refresh_sec = visual_refresh_ms / 1000.0

        # Screen capture
        self.grabber = DirectWindowGrabber(window_title = "Assetto Corsa", 
            target_width = getattr(config, "final_image_width", 320), target_height = getattr(config, "final_image_height", 240))

        # Vision pipeline modules
        self.saliency_engine = UnbiasedSaliencyEngine()
        self.cropper = FovealExtractor(crop_size = crop_size)
        self.encoder = FovealVisionEncoder(device = None)  # Auto-selects CUDA

        # Action: [steer, throttle, brake]
        self.action_space = Box(low=np.array([-1.0, -1.0, -1.0], dtype=np.float32), high=np.array([1.0, 1.0, 1.0], dtype=np.float32), dtype=np.float32)

        # Observation: 512 visual latent features + 2 normalized gaze coordinates
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.encoder.output_dim,),
            dtype=np.float32,
        )

        # Telemetry & debug caching
        self.latest_frame_rgb = None
        self.latest_heatmap = None
        self.latest_gaze = (0, 0)
        self.latest_crop = None
        self._cached_obs = None
        self._last_vis_time = 0.0
        self._shift_pulse_count = 0 
        self.action_dim = 3
        
        self._max_episode_steps = int(getattr(config, "max_episode_py_time", 120.0) *  getattr(config, "ego_sampling_freq", 25))
        
        if self.render_crop:
            
            self.window_name = f"Active vision - Foveal Crop ({self.visual_refresh_sec * 1000.0}ms)"
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 256, 256)
        

    def _get_active_vision_state(self, force_refresh: bool = False) -> np.ndarray:
        
        now = time.time()
        
        #Refresh the visual encoding after 100ms
        if force_refresh or (now - self._last_vis_time >= self.visual_refresh_sec) or self._cached_obs is None:
            
            self._last_vis_time = now
            
            frame_bgr = self.grabber.grab_bgr()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            heatmap = self.saliency_engine.compute_heatmap(frame_bgr)
            peak_x, peak_y = self.saliency_engine.get_peak_salient_point(heatmap)
            crop_rgb, gaze_coords = self.cropper.extract(frame_rgb, peak_x, peak_y)

            # Update cache for visualization
            self.latest_frame_rgb = frame_rgb
            self.latest_heatmap = heatmap
            self.latest_gaze = (peak_x, peak_y)
            self.latest_crop = crop_rgb
            
            self._cached_obs = self.encoder.encode(crop_rgb, gaze_coords)
            
            if self.render_crop:
                self._render_crop_popup()

        return self._cached_obs
    
    
    
    def _render_crop_popup(self):
        
        """
        Displays the current foveal crop with overlaid gaze coordinates 
        """
        
        if self.latest_crop is None:
            return
        
        #Convert RGB crop back into BGR for the OpenCV display
        display_image = cv2.cvtColor(self.latest_crop, cv2.COLOR_RGB2BGR)
        
        #Upscale the patch to 256x256 for better visibility
        display_image = cv2.resize(display_image, (256, 256), interpolation = cv2.INTER_NEAREST)
        
        #Draw the fixation crosshairs in the centre of the foveal crop
        centre = (128, 128)
        cv2.drawMarker(display_image, centre, color = (0, 255, 0), markerType = cv2.MARKER_CROSS, markerSize = 16, thickness = 1)
        
        #Overlay the normalised gaze coords
        gx, gy = self.latest_gaze
        hud_text = f"Fixation: ({gx}, {gy})"
        
        cv2.putText(display_image, hud_text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(self.window_name, display_image)
        
        #Wait 1ms to ensure that the GUI look can display without blocking the RL
        cv2.waitKey(1)


    def reset(self) -> np.ndarray:
        
        """Resets car position back to the track spawn point and clears controls."""
        
        if self.send_reset_on_start:
            
            # Send TCP reset command to AC plugin server to respawn the car
            self.client.simulation_management.send_reset()
            time.sleep(0.5)

        if not self.client.socket:
            self.client.setup_connection()

        self.client.controls.set_defaults()
        self.client.respond_to_server()
        self.client.step_sim()

        # Reset gear pulse state for launch
        self._shift_pulse_count = 0

        return self._get_active_vision_state()



    def _clean_action(self, action: Any) -> np.ndarray:
        
        if isinstance(action, (tuple, list)):
            action = action[0]
            
        if hasattr(action, "detach"):
            action = action.detach().cpu().numpy()
            
        elif hasattr(action, "cpu"):
            action = action.cpu().numpy()
            
        action = np.squeeze(np.asarray(action, dtype=np.float32))
        
        if action.ndim == 0:
            action = np.zeros(3, dtype=np.float32)
            
        return action


    def set_actions(self, action: Any):
        self.current_action = self._clean_action(action)


    def step(self, action: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, bool, dict]:
        
        if action is None:
            action = getattr(self, "current_action", np.zeros(3, dtype = np.float32))
             
        action = self._clean_action(action)
        
        steer = float(action[0])
        acc = float(action[1])
        brake = float(action[2])

        # Query current gear: 1 = Neutral, 2 = 1st Gear
        current_gear = self.client.state.get("actualGear", 1)

        # Pulse shift_up if car spawned in Neutral
        shift_up = False
        
        if current_gear == 1 and self._shift_pulse_count < 8:
            shift_up = True
            self._shift_pulse_count += 1
            
        elif current_gear > 1:
            self._shift_pulse_count = 0

        self.client.controls.set_controls(
            steer = steer,
            acc = acc,
            brake = brake,
            enable_gear_shift = shift_up,
            shift_up = shift_up,
            shift_down = False,
        )
        
        self.client.respond_to_server()

        # Poll next simulation state
        state_dict = self.client.step_sim()
        obs = self._get_active_vision_state()

        # Reward: speed in m/s (normalised) with penalty for being stationary
        speed = state_dict.get("speed", 0.0)
        reward = float(speed / 30.0)

        terminated = bool(state_dict.get("done", False))
        done = terminated
        info = {"telemetry": state_dict, "terminated": terminated}

        return obs, reward, done, info


    def close(self):
        
        if getattr(self, "render_crop", False):
            cv2.destroyAllWindows()
        
        self.client.close()
        
        
    def seed(self, seed = None):
        pass
    
    
    def render(self, mode = "human"):
        pass
    
    
    def set_eval_mode(self):
        pass