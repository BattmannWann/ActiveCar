import cv2
import numpy as np 
from abc import ABC, abstractmethod


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
    
    
    
class ColorContrastSaliencyEngine(BaseSaliencyEngine):
    
    def compute_heatmap(self, img_bgr: np.ndarray) -> np.ndarray:
        
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Compute multi-scale spatial gradients (Sobel filter magnitude)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize = 3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize = 3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        
        # Gaussian blur to aggregate energy into regions of interest
        saliency = cv2.GaussianBlur(magnitude, (21, 21), 5.0)
        
        # Normalise to [0, 1]
        min_v, max_v = np.min(saliency), np.max(saliency)
        return (saliency - min_v) / (max_v - min_v + 1e-8)