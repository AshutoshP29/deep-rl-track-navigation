# Reinforcement Learning for a Race Car

A 2D top-down car learns to drive around a circuit by getting rewards and penalties.

## Approaches

| Folder | Status | Approach |
|---|---|---|
| [ppo_checkpoints/](ppo_checkpoints/) | Working | PPO, 5 discrete actions, reward from checkpoint gates |
| [sac_centerline/](sac_centerline/) | Planned | SAC, continuous steering/throttle, centerline-progress reward, grip-limited physics |

Shared images live in [assets/](assets/).

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows  (source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

## Quick start (PPO + checkpoints)

```bash
cd ppo_checkpoints
python race_circuit.py   # drive the car yourself with the arrow keys
python train.py          # train (1M steps, 8 parallel envs by default)
python evaluate.py       # watch the best saved model drive
```

See each folder's README for details.
