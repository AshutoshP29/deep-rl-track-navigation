import pygame
import math

# --- Configuration & Setup ---
pygame.init()
WIDTH, HEIGHT = 1500, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Top-Down Racer - RL Environment")
clock = pygame.time.Clock()

ROAD_COLOR = (78, 78, 78) 

# Load Images
TRACK = pygame.image.load("circuit.png").convert()
TRACK = pygame.transform.scale(TRACK, (WIDTH, HEIGHT))

class PlayerCar:
    def __init__(self):
        self.original_img = pygame.image.load("car.png")

        if pygame.display.get_init():
            self.original_img = self.original_img.convert_alpha()
        # Scaled for the larger 1500x800 resolution
        self.img = pygame.transform.scale(self.original_img, (50, 60))
        self.reset()
        
        # Physics constants
        self.max_speed = 6
        self.acceleration = 0.2
        self.rotation_speed = 5
        self.friction = 0.98

    def reset(self):
        # Starting position on the START line
        self.x, self.y = 500, 610 
        self.angle = 90 # Facing Right (adjust to 90 or 270 depending on your car sprite)
        self.speed = 0
        self.is_alive = True
        self.distance_traveled = 0

    def drive(self):
        # Using trigonometry to move in the direction of the angle
        # Note: We subtract from Y because Pygame Y increases downwards
        radians = math.radians(self.angle)
        self.x -= math.sin(radians) * self.speed
        self.y -= math.cos(radians) * self.speed
        
        if self.speed > 0:
            self.distance_traveled += self.speed

    def check_collision(self, screen):
        # Check center pixel of the car
        if 0 <= int(self.x) < WIDTH and 0 <= int(self.y) < HEIGHT:
            pixel_color = screen.get_at((int(self.x), int(self.y)))
            
            # COLLISION CRITERIA: 
            # 1. High Green value = Grass
            # 2. High Red/White values = Barriers
            if pixel_color[1] > 180 or (pixel_color[0] > 220 and pixel_color[2] > 220):
                self.is_alive = False
        else:
            self.is_alive = False # Reset if off-screen

    def update(self, action=None):
        if action is None:
            keys = pygame.key.get_pressed()
            moved = False

            if self.speed != 0:
                if keys[pygame.K_LEFT]:
                    self.angle += self.rotation_speed
                if keys[pygame.K_RIGHT]:
                    self.angle -= self.rotation_speed

            if keys[pygame.K_UP]:
                moved = True
                self.speed = min(self.speed + self.acceleration, self.max_speed)
            if keys[pygame.K_DOWN]:
                moved = True
                self.speed = max(self.speed - self.acceleration, -self.max_speed/2)

            if not moved:
                self.speed *= self.friction
                if abs(self.speed) < 0.1: self.speed = 0
        else:
            pass

        self.drive()

    def get_data(self, screen):
        # The AI's 5 Lidar sensors
        angles = [-90, -45, 0, 45, 90]
        distances = []
        
        # Center of the car
        start_x, start_y = self.x, self.y 

        for rel_angle in angles:
            length = 0
            # Adding car angle to relative ray angle to rotate 'eyes' with car
            total_angle = math.radians(self.angle + rel_angle)
            
            # Cast ray up to 250 pixels
            while length < 250:
                length += 3
                check_x = int(start_x - math.sin(total_angle) * length)
                check_y = int(start_y - math.cos(total_angle) * length)

                if 0 <= check_x < WIDTH and 0 <= check_y < HEIGHT:
                    color = screen.get_at((check_x, check_y))
                    if color[1] > 150 or color[0] > 220: # Barrier or Grass hit
                        break
                else:
                    break
            
            # Normalize to 0.0 - 1.0 for the AI Brain
            distances.append(length / 250.0)
            
            # Visualization
            pygame.draw.line(screen, (255, 0, 0), (start_x, start_y), (check_x, check_y), 1)
            pygame.draw.circle(screen, (255, 0, 0), (check_x, check_y), 3)
        
        return distances

    def draw(self, win):
        rotated_image = pygame.transform.rotate(self.img, self.angle)
        # Center the rotation on current X, Y
        rect = rotated_image.get_rect(center=(self.x, self.y))
        win.blit(rotated_image, rect.topleft)

# --- Main Loop ---
car = PlayerCar()
running = True

while running:
    clock.tick(60)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 1. Logic
    car.update()
    
    # 2. Render Background
    screen.blit(TRACK, (0, 0))
    
    # 3. Collision & Sensor Data
    # IMPORTANT: Always check collision against the screen AFTER blitting the track
    car.check_collision(screen) 
    
    if not car.is_alive:
        car.reset()

    # Get data for the AI (Observation Space)
    sensors = car.get_data(screen) 
    
    # 4. Final Draw
    car.draw(screen)
    pygame.display.flip()

pygame.quit()