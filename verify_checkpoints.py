import pygame
import numpy as np
from race_circuit import TRACK, WIDTH, HEIGHT

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Checkpoint Verification - Click to set checkpoints")
clock = pygame.time.Clock()

# Draw the track
screen.blit(TRACK, (0, 0))

# Current checkpoints that need verification
checkpoints = [
    [(480, 560), (580, 560)],      # 0: Start/Finish
    [(350, 520), (320, 480)],      # 1: Left entrance
    [(240, 420), (220, 350)],      # 2: Left mid-curve
    [(200, 280), (220, 200)],      # 3: Left curve up
    [(280, 140), (380, 160)],      # 4: Top entry
    [(550, 120), (550, 100)],      # 5: Top mid-left
    [(750, 120), (750, 100)],      # 6: Top center
    [(950, 140), (1000, 160)],     # 7: Top right
    [(1100, 280), (1120, 340)],    # 8: Right upper
    [(1140, 480), (1160, 550)],    # 9: Right lower
    [(1050, 640), (950, 680)],     # 10: Hairpin entry
    [(850, 700), (750, 680)],      # 11: Hairpin exit
]

# Draw checkpoints
for idx, gate in enumerate(checkpoints):
    color = (0, 255, 0) if idx == 0 else (0, 0, 255)
    pygame.draw.line(screen, color, gate[0], gate[1], 3)
    # Draw text label
    font = pygame.font.Font(None, 24)
    text = font.render(str(idx), True, (255, 255, 255))
    screen.blit(text, (gate[0][0], gate[0][1] - 20))

# Draw car starting position
car_pos = (500, 610)
pygame.draw.circle(screen, (255, 0, 0), car_pos, 10)
font = pygame.font.Font(None, 24)
text = font.render("START", True, (255, 255, 0))
screen.blit(text, (car_pos[0] - 20, car_pos[1] + 20))

pygame.display.flip()

# Info
print("GREEN = Start/Finish (checkpoint 0)")
print("BLUE = Other checkpoints (numbered)")
print("RED CIRCLE = Car starting position (500, 610)")
print("\nInspect the checkpoint positions:")
print("1. Do checkpoints align with the racing line?")
print("2. Are they positioned where the car naturally crosses?")
print("3. Is the sequence: Start -> Left side -> Top -> Right -> Hairpin -> Start?")
print("\nCurrent checkpoints:")
for idx, gate in enumerate(checkpoints):
    print(f"  {idx}: {gate[0]} to {gate[1]}")
print("\nPress ESC to close")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

pygame.quit()
