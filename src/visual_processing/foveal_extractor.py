import cv2 
import numpy as np

class FovealExtractor:

    """
    Extracts a bounded foveal crop around the most salient point of 
    an image and normalises the gaze
    """

    def __init__(self, crop_size: int = 96):
        self.crop_size = crop_size
        self.half_size = crop_size // 2


    def extract(self, img_rgb: np.ndarray, cx: int, cy: int) -> tuple[np.ndarray, np.ndarray]:

        height, width = img_rgb.shape[:2]

        #Clamp the centre coordinate so that the crop remains within image bounds
        clamped_x = int(np.clip(cx, self.half_size, width - self.half_size))
        clamped_y = int(np.clip(cy, self.half_size, height - self.half_size))

        crop = img_rgb[
            clamped_y - self.half_size : clamped_y + self.half_size,
            clamped_x - self.half_size : clamped_x + self.half_size,
        ]

        #Normalise the gaze point to [-1.0, 1.0] such that point 0.0 is the visual centre
        norm_x = (cx / float(width)) * 2.0 - 1.0
        norm_y = (cy / float(height)) * 2.0 - 1.0

        gaze_coords = np.array([norm_x, norm_y], dtype = np.float32)

        return crop, gaze_coords 