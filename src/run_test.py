import numpy as np
from omegaconf import OmegaConf
from gym_implementation.ac_active_vis_env import AssettoCorsaActiveVisionEnv

"""
This file will test that the environment, code, and Assetto Corsa, can all communicate with each other
"""



if __name__ == "__main__":
    
    try:
        config = OmegaConf.load("config.yml")
        
    except Exception:
        
        class DummyConfig:
            simulation_management_server_host_name = "localhost"
            simulation_management_server_port = 2347
            vjoy_executed_by_server = False
            ego_server_host_name = "localhost"
            ego_server_port = 2345
            record_controls_from_client = False
            screen_capture_enable = False
            final_image_width = 320
            final_image_height = 240
            color_mode = "bgr"
            
        config = DummyConfig()

    print("[INFO] Creating AssettoCorsaActiveVisionEnv...")
    env = AssettoCorsaActiveVisionEnv(config, crop_size=96, send_reset_on_start=True)

    try:
        
        # Run 2 consecutive episodes to confirm respawn on reset
        for episode in range(1, 3):
            
            print(f"\n--- Starting Episode {episode} ---")
            obs = env.reset()
            print(f"[RESET] Car respawned. Initial Fixation: {env.latest_gaze}")

            for step in range(1, 41):
                
                # Apply 80% throttle (0.6 in [-1, 1] range)
                action = np.array([0.0, 0.6, -1.0], dtype=np.float32)
                obs, reward, done, info = env.step(action)

                speed_kmh = info["telemetry"].get("speed", 0.0) * 3.6
                gear = info["telemetry"].get("actualGear", 0)

                if step % 10 == 0 or step < 5:
                    print(f"Step {step:02d} | Fixation: {env.latest_gaze} | " f"Gear: {gear} | Speed: {speed_kmh:5.1f} km/h | Reward: {reward:.3f}")

                if done:
                    break
    finally:
        env.close()