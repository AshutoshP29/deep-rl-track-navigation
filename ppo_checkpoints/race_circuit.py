"""Track loading, car physics and sensors.

Run this file directly to drive the car yourself with the arrow keys.
"""
import math
import os
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import numpy as np
import pygame

ASSETS = Path(__file__).resolve().parent.parent / "assets"
WIDTH, HEIGHT = 1491, 1053

START_POS = (520, 805)
START_ANGLE = 90  # Degrees. 0 = facing up, 90 = facing left (the start direction)

# Lidar: 7 rays relative to the car's heading, cast up to SENSOR_RANGE pixels
SENSOR_ANGLES = np.array([-90, -60, -30, 0, 30, 60, 90])
SENSOR_RANGE = 250
SENSOR_STEPS = np.arange(3, SENSOR_RANGE + 1, 3)

# Collision points relative to the car centre as (forward, sideways) offsets in pixels.
# Slightly inside the 60x70 sprite so brushing the kerb paint isn't an instant crash.
HITBOX = ((22, 14), (22, -14), (-22, 14), (-22, -14), (0, 0))


@lru_cache(maxsize=1)
def _load_track_cached():
    track = pygame.transform.scale(pygame.image.load(ASSETS / "circuit.png"), (WIDTH, HEIGHT))

    # Asphalt is the only dark, unsaturated grey on the map
    rgb = pygame.surfarray.array3d(track).astype(np.int16)
    hi, lo = rgb.max(axis=2), rgb.min(axis=2)
    grey = (hi - lo < 35) & (hi > 40) & (hi < 160)

    # Keep only the grey region connected to the start line (drops building outlines etc.)
    grey_surface = pygame.surfarray.make_surface(np.repeat(grey[:, :, None], 3, axis=2).astype(np.uint8) * 255)
    grey_mask = pygame.mask.from_threshold(grey_surface, (255, 255, 255), (1, 1, 1, 255))
    road_surface = grey_mask.connected_component(START_POS).to_surface()
    road = pygame.surfarray.array_red(road_surface) > 0  # Indexed [x, y]
    road.flags.writeable = False

    return track, road


def load_track():
    """Return (track surface, boolean road mask indexed [x, y])."""
    track, road = _load_track_cached()
    return track.copy(), road


def is_on_road(road, x, y):
    x, y = int(x), int(y)
    return 0 <= x < WIDTH and 0 <= y < HEIGHT and road[x, y]


class PlayerCar:
    max_speed = 6.0
    acceleration = 0.2
    rotation_speed = 5.0
    friction = 0.98

    def __init__(self):
        img = pygame.image.load(ASSETS / "car.png")
        if pygame.display.get_surface() is not None:
            img = img.convert_alpha()
        self.img = pygame.transform.scale(img, (60, 70))
        self.reset()

    def reset(self):
        self.x, self.y = START_POS
        self.angle = START_ANGLE
        self.speed = 0.0

    def forward(self):
        """Unit vector the car is facing. Pygame's y axis points down."""
        rad = math.radians(self.angle)
        return -math.sin(rad), -math.cos(rad)

    def step_physics(self, steer, throttle):
        """steer: +1 left, -1 right, 0 straight. throttle: +1 accelerate, -1 brake/reverse, 0 coast."""
        if self.speed != 0:
            self.angle += steer * self.rotation_speed

        if throttle > 0:
            self.speed = min(self.speed + self.acceleration, self.max_speed)
        elif throttle < 0:
            self.speed = max(self.speed - self.acceleration, -self.max_speed / 2)
        else:
            self.speed *= self.friction
            if abs(self.speed) < 0.1:
                self.speed = 0.0

        fx, fy = self.forward()
        self.x += fx * self.speed
        self.y += fy * self.speed

    def on_road(self, road):
        fx, fy = self.forward()
        return all(
            is_on_road(road, self.x + fx * fwd - fy * side, self.y + fy * fwd + fx * side)
            for fwd, side in HITBOX
        )

    def sense(self, road):
        """Distance to the track edge along each lidar ray, normalised to 0-1."""
        rads = np.radians(self.angle + SENSOR_ANGLES)[:, None]
        xs = (self.x - np.sin(rads) * SENSOR_STEPS).astype(int)
        ys = (self.y - np.cos(rads) * SENSOR_STEPS).astype(int)

        inside = (xs >= 0) & (xs < WIDTH) & (ys >= 0) & (ys < HEIGHT)
        clear = np.zeros_like(inside)
        clear[inside] = road[xs[inside], ys[inside]]

        blocked = ~clear
        hit = blocked.any(axis=1)
        distances = np.where(hit, SENSOR_STEPS[blocked.argmax(axis=1)], SENSOR_RANGE)
        return distances / SENSOR_RANGE

    def draw_sensors(self, win, distances):
        for rel_angle, dist in zip(SENSOR_ANGLES, distances):
            rad = math.radians(self.angle + rel_angle)
            end = (self.x - math.sin(rad) * dist * SENSOR_RANGE, self.y - math.cos(rad) * dist * SENSOR_RANGE)
            pygame.draw.line(win, (205, 0, 0), (self.x, self.y), end, 1)
            pygame.draw.circle(win, (255, 0, 0), end, 3)

    def draw(self, win):
        rotated_image = pygame.transform.rotate(self.img, self.angle)
        rect = rotated_image.get_rect(center=(self.x, self.y))
        win.blit(rotated_image, rect.topleft)


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("2D Top-Down Racer - Manual Drive")
    clock = pygame.time.Clock()

    track, road = load_track()
    track = track.convert()
    car = PlayerCar()

    running = True
    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        steer = keys[pygame.K_LEFT] - keys[pygame.K_RIGHT]
        throttle = keys[pygame.K_UP] - keys[pygame.K_DOWN]
        car.step_physics(steer, throttle)

        if not car.on_road(road):
            car.reset()

        screen.blit(track, (0, 0))
        car.draw_sensors(screen, car.sense(road))
        car.draw(screen)
        pygame.display.flip()

    pygame.quit()
