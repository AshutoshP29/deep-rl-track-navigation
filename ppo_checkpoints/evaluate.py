import argparse
import os
import time
from pathlib import Path

from stable_baselines3 import PPO

from car_env import RacingEnv
from evaluation import print_summary, run_episode, save_gif, seconds

MODEL_DIR = Path(__file__).resolve().parent / "models"


def default_model_path():
    best = MODEL_DIR / "best_model.zip"
    return best if best.exists() else MODEL_DIR / "ppo_racing_final.zip"


def watch(model):
    env = RacingEnv(render_mode="human")
    state, info = env.reset()
    episode, total_reward = 1, 0.0
    try:
        while not env.window_closed:
            # deterministic=True picks the policy's best action instead of sampling
            action, _ = model.predict(state, deterministic=True)
            laps_before = info["laps"]
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if info["laps"] > laps_before:
                print(f"  Lap {info['laps']}: {seconds(info['lap_times'][-1]):.2f}s")

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


def main():
    parser = argparse.ArgumentParser(description="Watch or measure a trained PPO agent.")
    parser.add_argument("--model", type=Path, default=None, help="Defaults to models/best_model.zip")
    parser.add_argument("--episodes", type=int, default=0,
                        help="Run this many episodes headless and print lap metrics instead of watching")
    parser.add_argument("--gif", type=Path, default=None, help="Record a flying lap to this GIF file")
    args = parser.parse_args()

    model_path = args.model or default_model_path()
    if not model_path.exists():
        raise SystemExit(f"No model at {model_path}. Run train.py first.")
    print(f"--- Loading Trained Agent ({model_path.name}) ---")
    model = PPO.load(model_path, device="cpu")

    if not (args.episodes or args.gif):
        watch(model)
        return

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    if args.episodes:
        env = RacingEnv()
        print_summary([run_episode(env, model) for _ in range(args.episodes)])
    if args.gif:
        result = run_episode(RacingEnv(render_mode="rgb_array"), model, record_lap=1)
        save_gif(result.frames, args.gif)
        print(f"Saved {args.gif} ({len(result.frames)} frames)")


if __name__ == "__main__":
    main()
