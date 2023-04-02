####################################################################################################
#                                                                                                  #
#    This module is a basic Lunar Lander simulation with a simple thrust and rotation control.     #
#                                                                                                  #
####################################################################################################

import pygame
import numpy as np

# Define constants
WIDTH = 600
HEIGHT = 400
FPS = 60
GRAVITY = 0.05
THRUST = 0.15
FUEL = 100

# Define Lunar Lander class
class LunarLander:
    def __init__(self):
        self.x = WIDTH / 2
        self.y = 50
        self.vx = 0
        self.vy = 0
        self.angle = 0
        self.thrust = 0
        self.fuel = FUEL
        self.landed = False

    def update(self):
        self.vy += GRAVITY
        self.x += self.vx
        self.y += self.vy

        if self.thrust > 0 and self.fuel > 0:
            self.vx += THRUST * np.sin(self.angle)
            self.vy -= THRUST * np.cos(self.angle)
            self.fuel -= 1

    def draw(self, screen):
        lander_img = pygame.image.load('Assets/lander.png')
        lander_img = pygame.transform.scale(lander_img, (90, 130)) 
        lander_img = pygame.transform.rotate(lander_img, self.angle)
        lander_rect = lander_img.get_rect(center=(self.x, self.y))
        screen.blit(lander_img, lander_rect)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Create Lunar Lander object
lander = LunarLander()

# Main game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                lander.thrust = THRUST
            elif event.key == pygame.K_LEFT:
                lander.angle += 5
            elif event.key == pygame.K_RIGHT:
                lander.angle -= 5
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                lander.thrust = 0

    # Update game state
    lander.update()

    # Check for landing
    if lander.y >= HEIGHT - 20:
        lander.landed = True

    # Draw game objects
    screen.fill((0, 0, 0))
    lander.draw(screen)

    # Update display
    pygame.display.flip()

    # Wait for next frame
    clock.tick(FPS)

# Clean up Pygame
pygame.quit()
