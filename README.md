# 🏎️ RL Car Agent - Autonomous Racing using PPO

## Overview

RL Car Agent is a custom Reinforcement Learning project where an autonomous vehicle learns to navigate a 2D racing circuit using Proximal Policy Optimization (PPO). The environment is built using Gymnasium and Pygame, featuring realistic vehicle dynamics, LiDAR-style sensors, collision detection, and a checkpoint-based reward system.

The agent learns entirely through interaction with the environment, receiving rewards for maintaining speed and successfully passing checkpoints while avoiding collisions with barriers and off-track regions.

---

## Features

* Custom Gymnasium-compatible racing environment
* PPO-based reinforcement learning agent
* 2D top-down racing simulation
* 7 LiDAR-style distance sensors
* Physics-based vehicle movement
* Checkpoint progression system
* Reward shaping for efficient learning
* Collision detection using track pixel colors
* Real-time visualization using Pygame
* Model saving and deployment support

---

## Project Structure

```text
RL-Car-Agent/
│
├── car_env.py              # Gymnasium racing environment
├── race_circuit.py         # Vehicle physics and simulator
├── train.py               # PPO training script
│
├── circuit.png            # Racing track image
├── car.png                # Car sprite
│
├── ppo_racing_agent.zip   # Trained PPO model (generated)
│
├── README.md
└── requirements.txt
```

---

## Environment Description

The racing environment is implemented as a custom Gymnasium environment.

### Observation Space

The agent receives a 7-dimensional observation vector containing normalized LiDAR sensor readings.

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
0.0 → Obstacle very close
1.0 → No obstacle detected
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
* Reverse movement
* Steering dependent on vehicle speed
* Natural deceleration due to friction
* Continuous heading updates

---

## Sensor System

The vehicle uses ray-casting sensors similar to LiDAR.

Each frame:

1. Rays are projected from the vehicle center.
2. Rays extend up to 250 pixels.
3. Sensor readings stop when:

   * Grass is detected
   * Barriers are detected
   * Screen boundaries are reached
4. Distances are normalized to [0,1].

This allows the PPO agent to understand its surroundings without directly observing images.

---

## Reward Function

The reward function encourages:

### Positive Rewards

#### Maintaining Speed

```python
reward = speed * 0.2
```

#### Passing Checkpoints

```python
reward += 10.0
```

The racing circuit contains multiple sequential checkpoints that guide the agent around the entire track.

### Negative Rewards

#### Standing Still

```python
reward = -1.0
```

#### Collision

```python
reward = -100.0
```

---

## Episode Termination

Episodes terminate when:

### Collision

The vehicle collides with:

* Grass regions
* Track barriers
* Screen boundaries

### Time Limit

```python
2000 simulation steps
```

---

## Collision Detection

Collision checking is performed using pixel colors from the racing track image.

### Crash Conditions

* High green values → Grass
* Bright white regions → Barriers
* Leaving track boundaries

### Finish Line Exception

Special logic prevents collisions on the checkered finish line by ignoring:

```text
Pure White Pixels
Pure Black Pixels
```

within the designated finish area.

---

## PPO Training

Training uses Stable-Baselines3 PPO.

### Configuration

```python
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003
)
```

### Training Command

```bash
python train.py
```

### Training Duration

```python
total_timesteps = 50000
```

### Model Saving

```python
model.save("ppo_racing_agent")
```

Saved model:

```text
ppo_racing_agent.zip
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/RL-Car-Agent.git

cd RL-Car-Agent
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Environment

To manually test the simulator:

```bash
python race_circuit.py
```

Controls:

| Key | Action     |
| --- | ---------- |
| ↑   | Accelerate |
| ↓   | Brake      |
| ←   | Turn Left  |
| →   | Turn Right |

---

## Training the Agent

```bash
python train.py
```

The PPO agent will begin learning and automatically save the trained model after training is completed.

---

## Future Improvements

* Continuous action space
* Faster training using vectorized environments
* Curriculum learning
* Track randomization
* Multiple racing circuits
* Improved reward shaping
* DQN, SAC and TD3 comparisons
* Camera-based observations using CNN policies
* Lap timing and performance metrics

---

## Technologies Used

* Python
* Gymnasium
* Stable-Baselines3
* PPO (Proximal Policy Optimization)
* NumPy
* Pygame

---

## Results

The trained PPO agent learns to:

* Stay on the road
* Avoid collisions
* Navigate through checkpoints
* Maintain forward speed
* Complete larger portions of the circuit autonomously

Performance can be further improved by increasing training timesteps and tuning PPO hyperparameters.

---

## License

This project is intended for educational and research purposes.

Feel free to modify and extend the environment for your own reinforcement learning experiments.
