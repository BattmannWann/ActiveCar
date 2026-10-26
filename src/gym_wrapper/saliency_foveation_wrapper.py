import gymnasium as gym
from gymnasium.spaces import Box
import torch
from visual_processing.fixation_locator import UnbiasedSaliencyEngine
from visual_processing.foveal_extractor import FovealExtractor
from visual_processing.vision_feature_encoder import FovealVisionEncoder
import numpy as np
import cv2



class SaliencyFoveatedWrapper(gym.ObservationWrapper):

    """
    Gym wrapper to encapsulate the active vision. The pipeline is as follows:

    Image -> Unbiased Saliency Map -> Peak Argmax
    -> Foveal Crop -> RestNet-18 -> SAC State
    """

    def __init__(self, env: gym.Env, window_grabber, crop_size: int = 96, device: str = None):

        super().__init__(env)
        self.grabber = window_grabber
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.saliency_engine = UnbiasedSaliencyEngine()
        self.cropper = FovealExtractor(crop_size = crop_size)
        self.encoder = FovealVisionEncoder(device = self.device)

        #Gym Observation state should be 514-dimensional (see notebook)
        self.observation_space = Box(
            low = -np.inf,
            high = np.inf,
            shape = (self.encoder.output_dim),
            dtype = np.float32
        )

        self.latest_frame_rgb = None
        self.latest_heatmap = None
        self.latest_gaze = (0, 0)
        self.latest_crop = None


    def observation(self, obs) -> np.ndarray:

        #Get the image
        frame_bgr = self.grabber.grab_bgr()
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        #Get Saliency Map & Peak Fixation Point
        heatmap = self.saliency_engine.compute_heatmap(frame_bgr)
        peak_x, peak_y = self.saliency_engine.get_peak_salient_point(heatmap)

        #Apply Foveation Crop to Most Salient Part
        crop_rgb, gaze_coords = self.cropper.extract(frame_bgr, peak_x, peak_y)

        #Extract info for inspection
        self.latest_frame_rgb = frame_bgr
        self.latest_heatmap = heatmap
        self.latest_gaze = (peak_x, peak_y)
        self.latest_crop = crop_rgb

        rl_state = self.encoder.encode(crop_rgb, gaze_coords)
        return rl_state
