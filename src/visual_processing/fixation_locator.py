import cv2
import numpy as np 
from abc import ABC, abstractmethod
from typing import Tuple




"""
This file contains an abstract class for creating a saliency engine and accompanying implementations.
"""

# === ABSTRACT CLASS ===

class BaseSaliencyEngine(ABC):
    
    @abstractmethod
    def compute_heatmap(self, img_bgr: np.ndarray) -> np.ndarray:
        
        """Calculates normalized 2D heatmap in range [0.0, 1.0]"""
        pass


    def get_peak_salient_point(self, heatmap: np.ndarray) -> tuple[int, int]:
        
        """
        Default peak point finder (argmax). Can be overridden if needed.
        Finds the most salient point, represented in (x, y); where (x, y) = argmax S(x, y)
        """
        _, _, _, max_loc = cv2.minMaxLoc(heatmap)
        return max_loc  # returns (x, y)



# === IMPLEMENTATIONS ===

class UnbiasedSaliencyEngine(BaseSaliencyEngine):

    """
    TLDR; This class will compute the visual saliency (heatmap) of a given image through the Spectral Residual approach.
    
    This following code implements a Visual Saliency Detection Engine using the Spectral Residual (SR) approach.

    In computer vision, **visual saliency** predicts where a human eye would naturally focus when looking at an image 
    (e.g., bright spots, anomalous textures, distinct objects against a background). 
    
    This specific algorithm is based on the 2007 paper by Xiaodi Hou and Liqing Zhang: ["Saliency Detection: A Spectral Residual Approach"](https://cvpr.org).
    """

    def __init__(self, processing_dim: tuple = (128, 128)):
        self.processing_dim = processing_dim


    def compute_heatmap(self, img_bgr: np.ndarray) -> np.ndarray:

        """
        Calculates an unconstrained spectral residual saliency heatmap; i.e. in the range [0.0, 1.0]
        """

        original_h, original_w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        #Downsample the image for global frequency analysis (ENCODING)
        small = cv2.resize(gray, self.processing_dim, interpolation = cv2.INTER_AREA).astype(np.float32)

        #Calculate the (Fast) Fourier Transform to obtain the Log Amplitude and Phase
        fft = np.fft.fft2(small)
        amplitude = np.abs(fft)
        phase = np.angle(fft)

        log_amp = np.log(amplitude + 1e-8)
        avg_log_amp = cv2.blur(log_amp, (3, 3))
        spectral_residual = log_amp - avg_log_amp

        #Inverse FFT to get the Saliency Energy
        combined = np.exp(spectral_residual + 1j * phase)
        saliency = np.abs(np.fft.ifft2(combined)) ** 2

        #Spatial Gaussian Smoothing (to simulate visual acuity spread)
        saliency = cv2.GaussianBlur(saliency.astype(np.float32), (9, 9), sigmaX = 2.5)

        #Upsample (DECODE) back to native frame dimensions and normalise to [0, 1]
        saliency = cv2.resize(saliency, (original_w, original_h), interpolation = cv2.INTER_CUBIC)

        min_v, max_v = np.min(saliency), np.max(saliency)

        if max_v > min_v:
            heatmap = (saliency - min_v) / (max_v - min_v)

        else:
            heatmap = np.zeros_like(saliency)
    

        return heatmap

    
    
class FineGrainedSaliencyEngine(BaseSaliencyEngine):
    
    def __init__(self):
        self.saliency = cv2.saliency.StaticSaliencyFineGrained_create()


    def compute_heatmap(self, img_bgr: np.ndarray) -> np.ndarray:
        
        success, heatmap = self.saliency.computeSaliency(img_bgr)
        
        if not success:
            return np.zeros(img_bgr.shape[:2], dtype=np.float32)
        
        # Normalize to [0, 1]
        min_v, max_v = np.min(heatmap), np.max(heatmap)
        return (heatmap - min_v) / (max_v - min_v + 1e-8)
    

class DrivingAssistedSpectralResidualSaliencyEngine(BaseSaliencyEngine):
    
    """
    Spectral Residual Saliency Engine augmented with a Task-Centric

    View Frustum prior for cockpit/hood driving in Assetto Corsa.

    Suppresses static vehicle cockpit/steering-wheel textures and upper sky
    distractors while leaving the entire road scene, vanishing horizon,
    and peripheral apexes fully unconstrained for bottom-up saliency selection.
    """

    def __init__(self, processing_dim: Tuple[int, int] = (128, 128), cockpit_cutoff_ratio: float = 0.65, mask_softness: float = 0.08, suppress_sky: bool = True, sky_cutoff_ratio: float = 0.15,):
        
        """
        Args:
            processing_dim: Downscaled canvas size for Fourier computation (W, H).
            
            cockpit_cutoff_ratio: Relative vertical screen position (0.0 to 1.0)
                where the road ends and cockpit/dashboard begins. 0.65 means the
                bottom 35% of the frame is attenuated.
                
            mask_softness: Sigmoid temperature controlling transition smoothness.
            
            suppress_sky: Whether to softly down-weight the upper sky band.
            
            sky_cutoff_ratio: Relative vertical position of the sky boundary.
        """
        
        self.processing_dim = processing_dim
        self.cockpit_cutoff_ratio = cockpit_cutoff_ratio
        self.mask_softness = mask_softness
        self.suppress_sky = suppress_sky
        self.sky_cutoff_ratio = sky_cutoff_ratio

        # Precompute the driving frustum prior mask for processing_dim
        self._prior_mask = self._generate_spatial_prior(processing_dim)


    def _generate_spatial_prior(self, shape: Tuple[int, int]) -> np.ndarray:
        
        """
        Pre-computes a smooth 2D vertical transmission mask using logistic sigmoids.
        Avoids sharp step-functions that inject artificial high frequencies into FFT.
        """
        w, h = shape
        
        # Normalised vertical coordinate vector from 0.0 (top) to 1.0 (bottom)
        y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]

        #Lower Cockpit/Steering Wheel Attenuation (Sigmoid transition)
        #Values close to 0 (top) stay near 1.0; values past cockpit_cutoff drop towards 0.0
        cockpit_gate = 1.0 / (1.0 + np.exp((y - self.cockpit_cutoff_ratio) / self.mask_softness))

        #Upper Sky Attenuation (Optional gentle ramp-up)
        if self.suppress_sky:
            sky_gate = 1.0 / (1.0 + np.exp(-(y - self.sky_cutoff_ratio) / 0.05))
            mask_1d = cockpit_gate * sky_gate
            
        else:
            mask_1d = cockpit_gate

        #Broadcast across width (H, W)
        mask_2d = np.repeat(mask_1d, w, axis=1)
        return mask_2d


    def compute_heatmap(self, img_bgr: np.ndarray) -> np.ndarray:
        
        """
        Calculates the Spectral Residual saliency heatmap multiplied by
        the Driving Frustum prior, normalized to [0.0, 1.0].
        """
        
        original_h, original_w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        #Downsample for global Fourier frequency analysis
        small = cv2.resize(gray, self.processing_dim, interpolation=cv2.INTER_AREA).astype(np.float32)

        #Compute 2D FFT to extract Log-Amplitude and Phase
        fft = np.fft.fft2(small)
        amplitude = np.abs(fft)
        phase = np.angle(fft)

        log_amp = np.log(amplitude + 1e-8)
        avg_log_amp = cv2.blur(log_amp, (3, 3))
        spectral_residual = log_amp - avg_log_amp

        #Inverse FFT to synthesize Saliency Energy
        combined = np.exp(spectral_residual + 1j * phase)
        saliency = np.abs(np.fft.ifft2(combined)) ** 2

        #Modulate raw saliency with the Driving Frustum Prior
        # This completely zeroes out steering wheel energy before spatial smoothing
        saliency = saliency * self._prior_mask

        #Spatial Gaussian Smoothing (simulating human visual acuity spread)
        saliency = cv2.GaussianBlur(saliency.astype(np.float32), (9, 9), sigmaX=2.5)

        #Upsample back to native frame resolution and normalize [0, 1]
        saliency = cv2.resize(saliency, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
        saliency = np.clip(saliency, 0.0, None)  # Guard against cubic interpolation negative undershoot

        min_v, max_v = np.min(saliency), np.max(saliency)
        
        if max_v > min_v:
            heatmap = (saliency - min_v) / (max_v - min_v)
            
        else:
            heatmap = np.zeros_like(saliency)

        return heatmap