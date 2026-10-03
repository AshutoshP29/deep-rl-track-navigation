import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from car_env import RacingEnv

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"
LOG_DIR = HERE / "logs"


def main():
    parser = argparse.ArgumentParser(description="Train the PPO racing agent.")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8, help="Parallel environments")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = make_vec_env(RacingEnv, n_envs=args.n_envs, seed=args.seed, vec_env_cls=SubprocVecEnv)
    eval_env = make_vec_env(RacingEnv, n_envs=1, seed=args.seed + 1000)

    try:
        import tensorboard  # noqa: F401
        tensorboard_log = str(LOG_DIR / "tensorboard")
    except ImportError:
        tensorboard_log = None

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [128, 128]},
        tensorboard_log=tensorboard_log,
        seed=args.seed,
        verbose=1,
    )

    callbacks = [
        CheckpointCallback(
            save_freq=max(100_000 // args.n_envs, 1),
            save_path=str(MODEL_DIR / "checkpoints"),
            name_prefix="ppo_racing",
        ),
        # Keeps the best-scoring policy as models/best_model.zip
        EvalCallback(
            eval_env,
            best_model_save_path=str(MODEL_DIR),
            log_path=str(LOG_DIR),
            eval_freq=max(25_000 // args.n_envs, 1),
            n_eval_episodes=5,
            deterministic=True,
        ),
    ]

    print("--- Starting AI Training Session ---")
    model.learn(total_timesteps=args.timesteps, callback=callbacks)
    model.save(MODEL_DIR / "ppo_racing_final")
    print(f"--- Training Completed. Models saved to {MODEL_DIR} ---")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
