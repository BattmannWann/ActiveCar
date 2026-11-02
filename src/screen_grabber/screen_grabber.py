import cv2
import mss
import numpy as np
import pygetwindow as gw
import matplotlib.pyplot as plt


class DirectWindowGrabber:
    """Bypasses AssettoCorsaGym's shared-memory dual_buffer by grabbing

    the 'Assetto Corsa' window directly from Windows desktop memory.
    """

    def __init__(
        self,
        window_title: str = "Assetto Corsa",
        target_width: int = 320,
        target_height: int = 240,
    ):
        self.window_title = window_title
        self.target_width = target_width
        self.target_height = target_height
        self.sct = mss.mss()
        self.win = self._find_window()

    def _find_window(self):
        
        windows = gw.getWindowsWithTitle(self.window_title)
        
        for w in windows:
            
            # Match the actual game simulator window (avoiding Content Manager or folders)
            if w.title.strip() == "Assetto Corsa" or "Assetto Corsa" in w.title:
                if w.width > 100 and w.height > 100:
                    return w
                
        raise RuntimeError(
            f"Could not find running window titled '{self.window_title}'. "
            "Make sure Assetto Corsa is on track in Windowed mode!"
        )

    def grab_bgr(self) -> np.ndarray:
        
        """Captures the AC window and returns a (target_height, target_width, 3) BGR array."""
        
        # Crop out the Windows title bar (top 31px) and window border (8px left/right)
        monitor = {
            "top": self.win.top + 31,
            "left": self.win.left + 8,
            "width": max(1, self.win.width - 16),
            "height": max(1, self.win.height - 39),
        }
        
        sct_img = self.sct.grab(monitor)
        # mss returns BGRA; slice [:, :, :3] for standard 3-channel BGR
        
        img_bgr = np.array(sct_img)[:, :, :3]
        img_bgr = cv2.resize(img_bgr, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA,)
        
        return img_bgr
    


# === Example AC Gym plugin for the grabber, based off of the jupyter notebook script === 
    
# grabber = DirectWindowGrabber(target_width=320, target_height=240)

# env.reset()

# for i in range(100):
#     steer = 0.1 if (i % 2 == 0) else -0.1
#     # Pass 0.5 for throttle so relative actions quickly press the gas pedal
#     env.client.respond_to_server()
#     action = np.array([steer, 0.5, -1.0])
#     next_state, reward, done, info = env.step(action)

#     # Every 25 steps (once per second), grab and display the live game frame!
#     if i % 25 == 0:
#         frame_bgr = grabber.grab_bgr()
#         plt.figure(figsize=(5, 4))
#         plt.imshow(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
#         plt.title(f"Step {i} | Speed: {env.state['speed']*3.6:.1f} km/h")
#         plt.axis("off")
#         plt.show()

#     if done:
#         break

# env.recover_car()