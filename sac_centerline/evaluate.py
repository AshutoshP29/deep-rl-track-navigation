import argparse
import os
import time
from pathlib import Path

from stable_baselines3 import SAC

from sac_env import CenterlineRacingEnv

from evaluation import print_summary, run_episode, save_gif, seconds  # From ppo_checkpoints

MODEL_DIR = Path(__file__).resolve().parent / "models"


def default_model_path():
    # best_model.zip is picked on a single start-line episode, so it can be over-fitted to that start.
    # The final model scored almost the same from the start line and never crashed from random starts.
    final = MODEL_DIR / "sac_racing_final.zip"
    return final if final.exists() else MODEL_DIR / "best_model.zip"


def watch(model, random_start):
    env = CenterlineRacingEnv(render_mode="human", random_start=random_start)
    state, info = env.reset()
    episode, total_reward = 1, 0.0
    try:
        while not env.window_closed:
            action, _ = model.predict(state, deterministic=True)
            laps_before = info["laps"]
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if info["laps"] > laps_before:
                print(f"  Lap {info['laps']}: {seconds(info['lap_times'][-1]):.2f}s")

            if terminated or truncated:
                outcome = "crashed" if terminated else "time limit"
                print(f"Episode {episode}: {outcome}, laps={info['laps']}, reward={total_reward:.1f}")
                episode, total_reward = episode + 1, 0.0
                state, info = env.reset()
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping evaluation.")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Watch or measure a trained SAC agent.")
    parser.add_argument("--model", type=Path, default=None, help="Defaults to models/sac_racing_final.zip")
    parser.add_argument("--episodes", type=int, default=0,
                        help="Run this many episodes headless from the start line and print lap metrics")
    parser.add_argument("--random-starts", type=int, default=0,
                        help="Run this many headless episodes from random points on the track")
    parser.add_argument("--gif", type=Path, default=None, help="Record a flying lap to this GIF file")
    parser.add_argument("--watch-random", action="store_true", help="Watch with random start positions")
    args = parser.parse_args()

    model_path = args.model or default_model_path()
    if not model_path.exists():
        raise SystemExit(f"No model at {model_path}. Run train.py first.")
    print(f"--- Loading Trained Agent ({model_path.name}) ---")
    model = SAC.load(model_path)

    if not (args.episodes or args.random_starts or args.gif):
        watch(model, args.watch_random)
        return

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    if args.episodes:
        env = CenterlineRacingEnv()
        print_summary([run_episode(env, model) for _ in range(args.episodes)])
    if args.random_starts:
        print("\nRandom starts:")
        env = CenterlineRacingEnv(random_start=True)
        env.reset(seed=0)
        print_summary([run_episode(env, model) for _ in range(args.random_starts)])
    if args.gif:
        result = run_episode(CenterlineRacingEnv(render_mode="rgb_array"), model, record_lap=1)
        save_gif(result.frames, args.gif)
        print(f"Saved {args.gif} ({len(result.frames)} frames)")


if __name__ == "__main__":
    main()
