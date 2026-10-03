import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO

from car_env import RacingEnv

MODEL_DIR = Path(__file__).resolve().parent / "models"


def default_model_path():
    best = MODEL_DIR / "best_model.zip"
    return best if best.exists() else MODEL_DIR / "ppo_racing_final.zip"


def main():
    parser = argparse.ArgumentParser(description="Watch a trained PPO agent drive.")
    parser.add_argument("--model", type=Path, default=None, help="Defaults to models/best_model.zip")
    args = parser.parse_args()

    model_path = args.model or default_model_path()
    if not model_path.exists():
        raise SystemExit(f"No model at {model_path}. Run train.py first.")

    print(f"--- Loading Trained Agent ({model_path.name}) ---")
    env = RacingEnv(render_mode="human")
    model = PPO.load(model_path, env=env)

    state, info = env.reset()
    episode, total_reward = 1, 0.0
    try:
        while not env.window_closed:
            # deterministic=True picks the policy's best action instead of sampling
            action, _ = model.predict(state, deterministic=True)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                outcome = "crashed" if terminated else "time limit"
                print(f"Episode {episode}: {outcome}, laps={info['laps']}, "
                      f"gates={info['gates_passed']}, reward={total_reward:.1f}")
                episode, total_reward = episode + 1, 0.0
                state, info = env.reset()
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping evaluation.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
