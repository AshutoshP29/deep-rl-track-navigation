import math

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from race_circuit import HEIGHT, WIDTH, PlayerCar, load_track

# Checkpoint gates in driving order. Gate 0 is the start/finish line: the car spawns just
# before it, so the first target is gate 1 and crossing gate 0 again completes a lap.
GATES = (
    ((436, 764), (436, 850)),
    ((216, 750), (143, 800)),
    ((208, 633), (298, 688)),
    ((264, 537), (335, 597)),
    ((460, 507), (460, 590)),
    ((610, 504), (702, 514)),
    ((520, 379), (461, 440)),
    ((448, 300), (377, 256)),
    ((793, 189), (794, 270)),
    ((1163, 287), (1249, 265)),
    ((1107, 467), (1008, 390)),
    ((1108, 496), (1045, 566)),
    ((1246, 627), (1297, 542)),
    ((1249, 656), (1339, 727)),
    ((1117, 669), (1117, 754)),
    ((900, 750), (991, 767)),
    ((662, 764), (662, 850)),
)
FINISH_GATE = 0
FIRST_TARGET_GATE = 1

# Actions -> (steer, throttle)
ACTIONS = (
    (1, 0),   # 0: steer left
    (-1, 0),  # 1: steer right
    (0, 1),   # 2: accelerate
    (0, -1),  # 3: brake / reverse
    (0, 0),   # 4: coast
)

MAX_STEPS = 3000
STALL_STEPS = 300  # Truncate if no gate is passed for this many steps

PROGRESS_SCALE = 0.1  # Per pixel of distance closed toward the next gate
GATE_BONUS = 10.0
LAP_BONUS = 50.0
TIME_PENALTY = 0.05
CRASH_PENALTY = 100.0


def check_intersect(p1, p2, p3, p4):
    """Returns True if line segment p1-p2 intersects with line segment p3-p4"""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


def distance_to_segment(p, a, b):
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)
    t = min(max(t, 0.0), 1.0)
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


class RacingEnv(gym.Env):
    """Observation (10 values):
        0-6  lidar distances to the track edge, 0-1
        7    speed / max_speed, -0.5 to 1
        8-9  sin, cos of the angle between the car's heading and the next gate's midpoint
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(len(ACTIONS))
        low = np.array([0.0] * 7 + [-0.5, -1.0, -1.0], dtype=np.float32)
        high = np.ones(10, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.track, self.road = load_track()
        self.screen = None
        self.window_closed = False
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("2D Top-Down Racer - PPO Agent")
            self.clock = pygame.time.Clock()
            self.track = self.track.convert()

        self.car = PlayerCar()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.car.reset()
        self.steps = 0
        self.steps_since_gate = 0
        self.next_gate = FIRST_TARGET_GATE
        self.gates_passed = 0
        self.laps = 0

        self.sensors = self.car.sense(self.road)
        if self.render_mode == "human":
            self.render()
        return self._observation(), self._info()

    def step(self, action):
        steer, throttle = ACTIONS[int(action)]
        prev_pos = (self.car.x, self.car.y)
        prev_dist = self._distance_to_gate(self.next_gate)

        self.car.step_physics(steer, throttle)
        self.steps += 1
        self.steps_since_gate += 1

        reward = -TIME_PENALTY
        terminated = False

        if not self.car.on_road(self.road):
            reward -= CRASH_PENALTY
            terminated = True
        else:
            # Distance closed toward the target gate: driving away or oscillating earns nothing
            reward += PROGRESS_SCALE * (prev_dist - self._distance_to_gate(self.next_gate))

            if check_intersect(prev_pos, (self.car.x, self.car.y), *GATES[self.next_gate]):
                reward += GATE_BONUS
                self.gates_passed += 1
                self.steps_since_gate = 0
                if self.next_gate == FINISH_GATE:
                    self.laps += 1
                    reward += LAP_BONUS
                self.next_gate = (self.next_gate + 1) % len(GATES)

        truncated = not terminated and (self.steps >= MAX_STEPS or self.steps_since_gate >= STALL_STEPS)

        self.sensors = self.car.sense(self.road)
        if self.render_mode == "human":
            self.render()

        return self._observation(), reward, terminated, truncated, self._info()

    def _distance_to_gate(self, idx):
        return distance_to_segment((self.car.x, self.car.y), *GATES[idx])

    def _observation(self):
        (x1, y1), (x2, y2) = GATES[self.next_gate]
        tx, ty = (x1 + x2) / 2 - self.car.x, (y1 + y2) / 2 - self.car.y
        fx, fy = self.car.forward()
        rel = math.atan2(fx * ty - fy * tx, fx * tx + fy * ty)

        obs = np.empty(10, dtype=np.float32)
        obs[:7] = self.sensors
        obs[7] = self.car.speed / self.car.max_speed
        obs[8] = math.sin(rel)
        obs[9] = math.cos(rel)
        return obs

    def _info(self):
        return {"gates_passed": self.gates_passed, "laps": self.laps, "next_gate": self.next_gate}

    def render(self):
        if self.screen is None:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.window_closed = True

        self.screen.blit(self.track, (0, 0))
        # The next targeted gate draws green, others blue
        for idx, gate in enumerate(GATES):
            color = (0, 255, 0) if idx == self.next_gate else (0, 0, 255)
            pygame.draw.line(self.screen, color, gate[0], gate[1], 3)
        self.car.draw_sensors(self.screen, self.sensors)
        self.car.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


if __name__ == "__main__":
    # Random-action smoke test with rendering
    env = RacingEnv(render_mode="human")
    obs, info = env.reset()
    while not env.window_closed:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
