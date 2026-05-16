from stable_baselines3 import PPO
from car_env import RacingEnv
import os

env = RacingEnv(render_mode=None)

model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

print(
    "---Starting AI Training Session 🎯---"
)
model.learn(total_timesteps=50000)
print(
    "---Training Completed ✅---"
)

model.save("ppo_racing_agent")
print("Model saved succesfully 🎉")