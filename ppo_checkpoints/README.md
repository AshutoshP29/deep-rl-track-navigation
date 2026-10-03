# PPO + Checkpoint Gates

## Files

| File | Purpose |
|---|---|
| `race_circuit.py` | Track loading, road mask, car physics, lidar sensors. Run it to drive manually. |
| `car_env.py` | Gymnasium environment `RacingEnv`. Run it to watch random actions. |
| `train.py` | Trains PPO. Options: `--timesteps`, `--n-envs`, `--seed`. |
| `evaluate.py` | Watches a trained model drive. Option: `--model path/to/model.zip`. |

Training writes to `models/` (`best_model.zip`, `ppo_racing_final.zip`, periodic `checkpoints/`) and `logs/` (eval results, TensorBoard if installed). Both are git-ignored.

## Environment

**Actions (discrete, 5):** steer left, steer right, accelerate, brake/reverse, coast.

**Observation (10 values):**
- 7 lidar distances to the track edge (-90° to +90°), normalised 0-1
- speed / max speed
- sin and cos of the angle between the car's heading and the next gate

**Reward per step:**

| Term | Value |
|---|---|
| Progress toward the next gate | +0.1 per pixel closed (negative when moving away) |
| Passing the next gate in order | +10 |
| Completing a lap | +50 |
| Time penalty | -0.05 |
| Leaving the road | -100, episode ends |

Episodes are cut off after 3000 steps, or after 300 steps without passing a gate.

**Track limits:** a road mask is built once from the asphalt colour in `circuit.png`. The car crashes if any of 5 hitbox points (four corners and centre) leaves it.

## Training

```bash
python train.py --timesteps 1000000 --n-envs 8
tensorboard --logdir logs/tensorboard   # optional
```
