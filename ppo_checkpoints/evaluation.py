"""Episode metrics and GIF recording, shared by both approaches' evaluate.py."""
from dataclasses import dataclass, field

import numpy as np

FPS = 60  # Simulation steps per second, used to turn step counts into lap times


@dataclass
class EpisodeResult:
    crashed: bool
    steps: int
    laps: int
    lap_times: list  # Steps per lap. Lap 1 includes the standing start.
    distance: float
    reward: float
    frames: list = field(default_factory=list, repr=False)

    @property
    def flying_laps(self):
        return self.lap_times[1:]

    @property
    def best_lap(self):
        """Fastest flying lap in steps, or the first lap if that's all there is."""
        laps = self.flying_laps or self.lap_times
        return min(laps) if laps else None

    @property
    def avg_speed(self):
        return self.distance / max(self.steps, 1)


def run_episode(env, model, record_lap=None, frame_every=4, scale=0.4):
    """Run one deterministic episode.

    With record_lap=n, frames are captured during lap n (0 = the standing-start lap) and the
    episode stops when that lap ends. The env must use render_mode="rgb_array".
    """
    obs, info = env.reset()
    frames, trail, total = [], [], 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        if record_lap is not None and info["laps"] == record_lap:
            trail.append((env.car.x * scale, env.car.y * scale))
            if env.steps % frame_every == 0:
                frames.append(_shrink(env.render(), scale, trail))
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if record_lap is not None and info["laps"] > record_lap:
            break
    return EpisodeResult(terminated, info["steps"], info["laps"], info["lap_times"],
                         info["distance"], total, frames)


def _shrink(frame, scale, trail):
    """Downscale a frame and draw the path driven so far this lap (the racing line)."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame)
    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.BILINEAR)
    if len(trail) > 1:
        ImageDraw.Draw(img).line(trail, fill=(255, 230, 0), width=2)
    return img


def save_gif(frames, path, frame_every=4):
    """Save frames as a GIF that plays at real time (FPS / frame_every frames per second)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    first, *rest = [f.convert("P", palette=1, colors=128) for f in frames]  # 1 = adaptive palette
    first.save(path, save_all=True, append_images=rest, duration=round(1000 * frame_every / FPS),
               loop=0, optimize=True)


def seconds(steps):
    return steps / FPS


def print_summary(results):
    print(f"{'ep':>3}  {'outcome':<10} {'laps':>4}  {'best lap':>9}  {'lap times (s)':<32} {'avg speed':>9}  {'reward':>8}")
    for i, r in enumerate(results, 1):
        outcome = "crashed" if r.crashed else "finished"
        best = f"{seconds(r.best_lap):.2f}s" if r.best_lap else "-"
        laps = ", ".join(f"{seconds(t):.2f}" for t in r.lap_times) or "-"
        print(f"{i:>3}  {outcome:<10} {r.laps:>4}  {best:>9}  {laps:<32} {r.avg_speed:>9.2f}  {r.reward:>8.1f}")

    if len(results) > 1:
        bests = [r.best_lap for r in results if r.best_lap]
        print(f"\nCrash rate: {np.mean([r.crashed for r in results]):.0%}   "
              f"Mean laps: {np.mean([r.laps for r in results]):.2f}   "
              f"Best lap: {seconds(min(bests)):.2f}s" if bests else "No laps completed")
