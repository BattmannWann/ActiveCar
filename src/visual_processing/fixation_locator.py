import cv2
import numpy as np 

class SaliencyEngine:

    """
    This class will compute the visual saliency (heatmap) of a given image
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


    def get_peak_salient_point(self, heatmap: np.ndarray) -> tuple[int, int]:

        """
        Finds the most salient point, represented in (x, y); where (x, y) = argmax S(x, y)
        """

        _, _, _, max_loc = cv2.minMaxLoc(heatmap)
        return max_loc # i.e. (x,y)

    