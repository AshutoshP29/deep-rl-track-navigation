import argparse
from functools import partial
from pathlib import Path

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from sac_env import CenterlineRacingEnv

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"
LOG_DIR = HERE / "logs"


class OffsetCurriculum(BaseCallback):
    """Anneals the lateral-offset penalty c1 linearly from `start` to 0.

    Early on, the penalty keeps the car near the centerline, which is an easy way to learn to stay
    on the road. Once it reaches 0 the agent is free to find its own racing line.
    """

    def __init__(self, start, anneal_steps, update_every=1000):
        super().__init__()
        self.start = start
        self.anneal_steps = anneal_steps
        self.update_every = update_every
        self.current = None

    def weight(self):
        return self.start * max(0.0, 1.0 - self.num_timesteps / self.anneal_steps)

    def _on_step(self):
        if self.current is None or self.n_calls % self.update_every == 0:
            weight = self.weight()
            if weight != self.current:
                self.training_env.env_method("set_offset_weight", weight)
                self.logger.record("curriculum/offset_weight", weight)
                self.current = weight
        return True


def main():
    parser = argparse.ArgumentParser(description="Train the SAC racing agent.")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=4, help="Parallel environments")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offset-weight", type=float, default=0.3, help="Starting c1 for the curriculum")
    parser.add_argument("--anneal-frac", type=float, default=0.5,
                        help="Fraction of training over which c1 anneals to 0")
    parser.add_argument("--threads", type=int, default=4, help="Torch CPU threads")
    parser.add_argument("--device", default="auto", help="auto (CUDA if available), cuda or cpu")
    # Each update costs about the same at batch 256 or 512 for this small network (it's overhead
    # bound), so fewer, bigger updates train faster. -1 = one update per transition collected.
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gradient-steps", type=int, default=2, help="Updates per vectorised env step")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    train_env_fn = partial(CenterlineRacingEnv, random_start=True, offset_weight=args.offset_weight)
    env = make_vec_env(train_env_fn, n_envs=args.n_envs, seed=args.seed, vec_env_cls=SubprocVecEnv)
    # Evaluate from the real start line with no offset penalty, so scores are comparable throughout
    eval_env = make_vec_env(CenterlineRacingEnv, n_envs=1, seed=args.seed + 1000)

    try:
        import tensorboard  # noqa: F401
        tensorboard_log = str(LOG_DIR / "tensorboard")
    except ImportError:
        tensorboard_log = None

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=300_000,
        learning_starts=10_000,
        batch_size=args.batch_size,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=args.gradient_steps,
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=tensorboard_log,
        device=args.device,
        seed=args.seed,
        verbose=1,
    )

    callbacks = [
        OffsetCurriculum(args.offset_weight, anneal_steps=int(args.timesteps * args.anneal_frac)),
        CheckpointCallback(
            save_freq=max(100_000 // args.n_envs, 1),
            save_path=str(MODEL_DIR / "checkpoints"),
            name_prefix="sac_racing",
        ),
        # The car and policy are both deterministic here, so one episode is a full evaluation
        EvalCallback(
            eval_env,
            best_model_save_path=str(MODEL_DIR),
            log_path=str(LOG_DIR),
            eval_freq=max(10_000 // args.n_envs, 1),
            n_eval_episodes=1,
            deterministic=True,
        ),
    ]

    print("--- Starting SAC Training Session ---")
    model.learn(total_timesteps=args.timesteps, callback=callbacks, log_interval=20)
    model.save(MODEL_DIR / "sac_racing_final")
    print(f"--- Training Completed. Models saved to {MODEL_DIR} ---")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
