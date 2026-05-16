import time
from stable_baselines3 import PPO
from car_env import RacingEnv

print("--- Loading Trained Agent 🏎️ ---")
# 1. Initialize the environment with VISUAL rendering turned ON
env = RacingEnv(render_mode="human")

# 2. Load the binary brain file you saved
model = PPO.load("ppo_racing_agent", env=env)

# 3. Put the agent to the test
state, info = env.reset()
print("Watch your AI tackle the circuit!")

try:
    while True:
        # Use deterministic=True so the AI picks its absolute best choice
        # instead of exploring or making random experimental moves
        action, _states = model.predict(state, deterministic=True)
        
        # Take a step inside the game environment
        state, reward, terminated, truncated, info = env.step(action)
        
        # If it completes the lap or hits a boundary wall, reset cleanly
        if terminated or truncated:
            print("Resetting Episode...")
            state, info = env.reset()
            time.sleep(0.5) # Quick pause to observe the reset point
            
except KeyboardInterrupt:
    print("\nStopping evaluation.")
finally:
    env.close()