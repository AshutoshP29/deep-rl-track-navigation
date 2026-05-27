import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
from race_circuit import PlayerCar, TRACK, WIDTH, HEIGHT

def check_intersect(p1, p2, p3, p4):
    """Returns True if line segment p1-p2 intersects with line segment p3-p4"""
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)

class RacingEnv(gym.Env):

    metadata = {"render_modes": ["human", None]}

    def __init__(self, render_mode=None):
        super(RacingEnv, self).__init__()

        self.render_mode = render_mode

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()
        else: 
            self.screen = pygame.Surface((WIDTH, HEIGHT))

        self.car = PlayerCar()
        self.time_steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.car.reset()
        self.time_steps = 0
        self.prev_x, self.prev_y = self.car.x, self.car.y
   
        self.checkpoints = [           
            [(436, 764), (436, 850)],
        
            [(216, 750), (143, 800)],
            
            [(208, 633), (298, 688)],
            
            [(264, 537), (335, 597)],

            [(460, 507), (460, 590)],

            [(610, 504), (702, 514)],

            [(520, 379), (461, 440)],

            [(448, 300), (377, 256)],

            [(793, 189), (794, 270)],

            [(1163, 287), (1249, 265)],

            [(1107, 467), (1008, 390)],

            [(1108, 496), (1045, 566)],

            [(1246, 627), (1297, 542)],

            [(1249, 656), (1339, 727)],

            [(1117, 669), (1117, 754)],

            [(900, 750), (991, 767)],

            [(662, 764), (662, 850)]
        ]
        
        self.next_checkpoint_idx = 1 # Start by heading toward Gate 1 after leaving start/finish

        # Generate initial screen observations
        self.screen.blit(TRACK, (0, 0))
        raw_sensors = self.car.get_data(self.screen)
        observation = np.array(raw_sensors, dtype=np.float32)

        return observation, {}

    def step(self, action):
        self.time_steps += 1
        self.prev_x, self.prev_y = self.car.x, self.car.y

        # Actions
        if self.car.speed != 0:
            if action == 0: self.car.angle += self.car.rotation_speed
            elif action == 1: self.car.angle -= self.car.rotation_speed
        
        if action == 2:
            self.car.speed = min(self.car.speed + self.car.acceleration, self.car.max_speed)
        elif action == 3:
            self.car.speed = max(self.car.speed - self.car.acceleration, -self.car.max_speed/2)
            
        if action not in [2, 3]:
            self.car.speed *= self.car.friction
            if abs(self.car.speed) < 0.1: self.car.speed = 0
        
        self.car.update(action=action)
        self.screen.blit(TRACK, (0, 0))

        # --- PRECISION COLLISION FILTER FOR THE CHECKERED FLAG ---
        # Instead of bypassing a whole coordinate box, look at the color directly!
        if 0 <= int(self.car.x) < WIDTH and 0 <= int(self.car.y) < HEIGHT:
            pixel_color = self.screen.get_at((int(self.car.x), int(self.car.y)))
            
            # Checkered Flag exception: Pure White (255,255,255) or Pure Black (0,0,0)
            is_checkered_white = (pixel_color[0] > 240 and pixel_color[1] > 240 and pixel_color[2] > 240)
            is_checkered_black = (pixel_color[0] < 15 and pixel_color[1] < 15 and pixel_color[2] < 15)

            if is_checkered_white or is_checkered_black:
                pass # Do not crash on the finish line pattern
            else:
                # Run regular barrier/grass checking everywhere else
                self.car.check_collision(self.screen)
        else:
            self.car.is_alive = False

        reward = 0.0
        terminated = False
        truncated = False

        if not self.car.is_alive:
            reward = -100.0
            terminated = True
        else:
            if self.car.speed <= 0:
                reward = -1.0
            else:
                reward = self.car.speed * 0.2

            # Checkpoint Progress Evaluation
            p1 = (self.prev_x, self.prev_y)
            p2 = (self.car.x, self.car.y)
            p3, p4 = self.checkpoints[self.next_checkpoint_idx]

            if check_intersect(p1, p2, p3, p4):
                # Scale reward down slightly since checkpoints are densely packed now
                reward += 10.0 
                print(f"Passed Gate {self.next_checkpoint_idx}/{len(self.checkpoints)}!")
                self.next_checkpoint_idx = (self.next_checkpoint_idx + 1) % len(self.checkpoints)

        if self.time_steps >= 2000: # Bumped up step count slightly for longer track length
            truncated = True

        raw_sensors = self.car.get_data(self.screen)
        observation = np.array(raw_sensors, dtype=np.float32)

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            # Draw the gates dynamically to monitor agent progress
            for idx, gate in enumerate(self.checkpoints):
                # The next targeted checkpoint draws Green, others draw Blue
                color = (0, 255, 0) if idx == self.next_checkpoint_idx else (0, 0, 255)
                pygame.draw.line(self.screen, color, gate[0], gate[1], 3)

            self.car.draw(self.screen)
            pygame.display.flip()
            self.clock.tick(60)

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    env = RacingEnv(render_mode="human")
    obs, info = env.reset()
    
    running = True
    while running:
        random_action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(random_action)
        
        if terminated or truncated:
            obs, info = env.reset()
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
    env.close()