import pygame
import math

# --- Configuration & Setup ---
pygame.init()
WIDTH, HEIGHT = 1500, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Top-Down Racer")
clock = pygame.time.Clock()

# Load Images
# .convert_alpha() is essential for PNG transparency
TRACK = pygame.image.load("circuit.png").convert()
TRACK = pygame.transform.scale(TRACK, (WIDTH, HEIGHT))

class PlayerCar:
    def __init__(self):
        # 1. Load the PNG
        self.original_img = pygame.image.load("car.png").convert_alpha()
        
        # 2. Scale the car (adjust 30, 60 based on your preference)
        self.img = pygame.transform.scale(self.original_img, (50, 60))
        
        # 3. Starting position: Aligned with the bottom-left start line
        self.x, self.y = 600, 580 
        self.angle = 90  # Pointing right to start
        
        self.speed = 0
        self.max_speed = 5
        self.acceleration = 0.15
        self.rotation_speed = 5

    def drive(self):
        # Basic movement physics
        radians = math.radians(self.angle)
        # Note: We use sin for x and cos for y because 0 degrees in Pygame is North
        horizontal = math.sin(radians) * self.speed
        vertical = math.cos(radians) * self.speed
        
        self.x -= horizontal
        self.y -= vertical

    def update(self):
        keys = pygame.key.get_pressed()
        moved = False

        # Steering: Only allow steering if the car is moving
        if self.speed != 0:
            if keys[pygame.K_LEFT]:
                self.angle += self.rotation_speed
            if keys[pygame.K_RIGHT]:
                self.angle -= self.rotation_speed

        # Acceleration / Braking
        if keys[pygame.K_UP]:
            moved = True
            self.speed = min(self.speed + self.acceleration, self.max_speed)
        if keys[pygame.K_DOWN]:
            moved = True
            self.speed = max(self.speed - self.acceleration, -self.max_speed/2)

        if not moved:
            # Friction/Coast to a stop
            if self.speed > 0:
                self.speed = max(self.speed - 0.1, 0)
            else:
                self.speed = min(self.speed + 0.1, 0)

        self.drive()

    def draw(self, win):
        # Rotate from the original scaled image to maintain quality
        rotated_image = pygame.transform.rotate(self.img, self.angle)
        
        # IMPORTANT: This centers the rotated image on the car's position
        # Without this, the car will "jump" around when rotating
        rect = rotated_image.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        
        win.blit(rotated_image, rect.topleft)
        

# --- Main Game Loop ---
car = PlayerCar()
running = True

while running:
    clock.tick(60) # Lock to 60 FPS

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 1. Update Logic
    car.update()

    # 2. Render
    screen.blit(TRACK, (0, 0))   # Background first
    car.draw(screen)             # Car on top

    pygame.display.flip()

pygame.quit()