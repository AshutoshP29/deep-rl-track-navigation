# 🏎️ RL Car Agent - Autonomous Racing using Reinforcement Learning

## Overview

RL Car Agent is a custom Reinforcement Learning project where an autonomous vehicle learns to navigate a 2D racing circuit. The environment is built using Gymnasium and Pygame, featuring vehicle dynamics, LiDAR-style sensors, road-mask collision detection, and a checkpoint-based reward system.

The agent learns entirely through interaction with the environment, receiving rewards for making progress toward checkpoints and completing laps, while avoiding leaving the track.

The project contains two approaches, each in its own folder:

| Folder | Status | Approach |
| ------ | ------ | -------- |
| `ppo_checkpoints/` | ✅ Working | PPO, 5 discrete actions, checkpoint-gate reward |
| `sac_centerline/` | 🚧 Planned | SAC, continuous steering/throttle, centerline-progress reward, grip-limited physics |

---

## Features

* Custom Gymnasium-compatible racing environment
* PPO-based reinforcement learning agent
* 2D top-down racing simulation
* 7 LiDAR-style distance sensors (vectorised with NumPy)
* Physics-based vehicle movement
* Checkpoint progression system with lap counting
* Dense, exploit-resistant reward shaping
* Road-mask collision detection with a multi-point hitbox
* Parallel environments for fast training
* Automatic best-model and checkpoint saving
* Real-time visualization using Pygame

---

## Project Structure

```text
deep-rl-track-navigation/
│
├── assets/
│   ├── circuit.png            # Racing track image
│   └── car.png                # Car sprite
│
├── ppo_checkpoints/
│   ├── race_circuit.py        # Track, road mask, vehicle physics, sensors, manual drive
│   ├── car_env.py             # Gymnasium racing environment
│   ├── train.py               # PPO training script
│   ├── evaluate.py            # Watch a trained agent drive
│   ├── README.md
│   ├── models/                # Trained models (generated, git-ignored)
│   └── logs/                  # Eval results / TensorBoard (generated, git-ignored)
│
├── sac_centerline/
│   └── README.md              # Planned SAC + centerline approach
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Environment Description

The racing environment is implemented as a custom Gymnasium environment (`ppo_checkpoints/car_env.py`).

### Observation Space

The agent receives a 10-dimensional observation vector:

| Index | Value | Range |
| ----- | ----- | ----- |
| 0-6   | LiDAR distances to the track edge | 0 to 1 |
| 7     | Speed / max speed | -0.5 to 1 |
| 8     | sin(angle to next checkpoint) | -1 to 1 |
| 9     | cos(angle to next checkpoint) | -1 to 1 |

Sensor angles relative to vehicle heading:

| Sensor | Angle |
| ------ | ----- |
| 1      | -90°  |
| 2      | -60°  |
| 3      | -30°  |
| 4      | 0°    |
| 5      | +30°  |
| 6      | +60°  |
| 7      | +90°  |

Maximum sensor range:

```text
250 pixels
```

Normalized range:

```text
0.0 → Track edge very close
1.0 → No edge detected within range
```

---

## Action Space

The environment uses a discrete action space of size 5.

| Action ID | Action                |
| --------- | --------------------- |
| 0         | Turn Left             |
| 1         | Turn Right            |
| 2         | Accelerate            |
| 3         | Brake / Reverse       |
| 4         | Coast (friction only) |

---

## Vehicle Physics

The car follows simple physics-based motion.

### Parameters

| Parameter      | Value |
| -------------- | ----- |
| Maximum Speed  | 6     |
| Acceleration   | 0.2   |
| Rotation Speed | 5°    |
| Friction       | 0.98  |

### Dynamics

* Forward acceleration
* Reverse movement (up to half of max speed)
* Steering only while moving
* Natural deceleration due to friction
* Continuous heading updates

---

## Sensor System

The vehicle uses ray-casting sensors similar to LiDAR.

Each step:

1. 7 rays are projected from the vehicle center.
2. Rays extend up to 250 pixels, sampled every 3 pixels.
3. A ray stops at the first point that is off the road mask or off screen.
4. Distances are normalized to [0, 1].

All rays are computed at once with NumPy against a precomputed road mask, which is much faster than reading screen pixels one at a time.

This allows the PPO agent to understand its surroundings without directly observing images.

---

## Reward Function

| Term | Reward |
| ---- | ------ |
| Progress toward the next checkpoint | +0.1 per pixel closed (negative when moving away) |
| Passing the next checkpoint in order | +10 |
| Completing a lap | +50 |
| Time penalty (every step) | -0.05 |
| Leaving the track | -100 (episode ends) |

The circuit has 17 sequential checkpoint gates that guide the agent around the track. Gate 0 is the start/finish line, so crossing it after the others completes a lap.

Because the progress reward is the change in distance to the next gate, driving away from it, circling, or oscillating earns nothing. Only real progress around the track pays off.

---

## Episode Termination

### Collision (terminated)

Any hitbox point leaves the road: grass, sand, kerbs, barriers, or the screen edge.

### Truncation

* 3000 simulation steps reached
* 300 steps without passing a checkpoint (stuck or stalling)

---

## Collision Detection

A boolean **road mask** is built once when the track loads:

* Asphalt is the only dark, low-saturation grey on the track image.
* Only the grey region connected to the start line is kept, which removes building outlines and other false positives.

The car is checked at **5 hitbox points** (four corners plus the center), so the car's body can't clip through the track edge.

---

## PPO Training

Training uses Stable-Baselines3 PPO with parallel environments.

### Configuration

```python
model = PPO(
    "MlpPolicy",
    env,                      # 8 parallel environments (SubprocVecEnv)
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=256,
    gamma=0.99,
    ent_coef=0.01,
    policy_kwargs={"net_arch": [128, 128]},
)
```

### Training Command

```bash
cd ppo_checkpoints
python train.py                                   # defaults: 1M steps, 8 envs
python train.py --timesteps 2000000 --n-envs 8 --seed 0
```

### Model Saving

| File | Description |
| ---- | ----------- |
| `models/best_model.zip` | Best policy found by periodic evaluation |
| `models/ppo_racing_final.zip` | Policy at the end of training |
| `models/checkpoints/` | Snapshot every 100k steps |

Optional TensorBoard logging (if `tensorboard` is installed):

```bash
tensorboard --logdir logs/tensorboard
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/AshutoshP29/deep-rl-track-navigation.git

cd deep-rl-track-navigation
```

### Install Dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows (source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

---

## Running the Environment

To drive the car yourself:

```bash
cd ppo_checkpoints
python race_circuit.py
```

Controls:

| Key | Action     |
| --- | ---------- |
| ↑   | Accelerate |
| ↓   | Brake      |
| ←   | Turn Left  |
| →   | Turn Right |

To watch random actions in the environment:

```bash
python car_env.py
```

---

## Training and Evaluating the Agent

```bash
cd ppo_checkpoints
python train.py       # train
python evaluate.py    # watch the best model drive
python evaluate.py --model models/ppo_racing_final.zip
```

The evaluation script prints each episode's outcome, laps, checkpoints passed, and total reward.

---

## Future Improvements

* ✅ ~~Faster training using vectorized environments~~
* ✅ ~~Improved reward shaping~~
* 🚧 SAC with continuous steering and throttle (`sac_centerline/`)
* 🚧 Centerline-progress reward with an optional lateral-offset penalty
* 🚧 Grip-limited physics so the agent can learn real racing lines
* Combined steer + throttle actions for PPO
* Curriculum learning
* Randomized start positions
* Track randomization and multiple racing circuits
* Camera-based observations using CNN policies
* Lap timing and performance metrics

---

## Technologies Used

* Python
* Gymnasium
* Stable-Baselines3
* PPO (Proximal Policy Optimization)
* PyTorch
* NumPy
* Pygame

---

## Results

The trained PPO agent learns to:

* Stay on the road
* Avoid collisions
* Navigate through checkpoints in order
* Maintain forward speed
* Complete laps of the circuit autonomously

Performance can be further improved by increasing training timesteps and tuning PPO hyperparameters.

---

## License

This project is intended for educational and research purposes.

Feel free to modify and extend the environment for your own reinforcement learning experiments.
