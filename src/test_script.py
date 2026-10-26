import matplotlib.pyplot as plt
from gym_wrapper.saliency_foveation_wrapper import SaliencyFoveatedWrapper
import numpy as np

def test_unbiased_active_vision(env, grabber):
    wrapped_env = SaliencyFoveatedWrapper(env, grabber, crop_size=96)

    # 1. Reset
    state = wrapped_env.reset()
    print(f"Observation space: {wrapped_env.observation_space.shape}")
    print(f"Sampled state shape: {state.shape} (Expected: 514)")

    # 2. Step forward 5 steps
    for step in range(1, 6):
        action = np.array([0.0, 0.5, -1.0])
        state, reward, done, info = wrapped_env.step(action)

        print(f"Step {step} | Peak Fixation: {wrapped_env.latest_gaze}")

        if step == 5:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

            # Full Scene with Fixation Mark
            gx, gy = wrapped_env.latest_gaze
            axes[0].imshow(wrapped_env.latest_frame_rgb)
            axes[0].scatter(
                [gx],
                [gy],
                c="red",
                s=120,
                marker="x",
                linewidths=2.5,
                label=f"Peak Saliency ({gx}, {gy})",
            )
            axes[0].set_title("Input Frame + Saliency Argmax")
            axes[0].legend(loc="lower right")
            axes[0].axis("off")

            # Saliency Heatmap
            axes[1].imshow(wrapped_env.latest_heatmap, cmap="inferno")
            axes[1].set_title("Unbiased Saliency Heatmap")
            axes[1].axis("off")

            # Foveal Crop
            axes[2].imshow(wrapped_env.latest_crop)
            axes[2].set_title("Foveated Crop (Sent to ResNet-18)")
            axes[2].axis("off")

            plt.tight_layout()
            plt.show()

        if done:
            break