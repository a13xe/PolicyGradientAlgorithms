####################################################################################################
#                                                                                                  #
#    This module is a basic Lunar Lander simulation with a simple thrust and rotation control.     #
#                                                                                                  #
####################################################################################################

import pygame
import random
import numpy as np

# Constants
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 900
GRAVITY = 0.2
THRUST = -0.6
FUEL = 1000
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

# Initialize Pygame
pygame.init()
# Create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Lunar Lander")
# Load images
lander_img = pygame.image.load("Assets/lander.png").convert_alpha()
platform_img = pygame.image.load("Assets/platform.png").convert_alpha()
# Resize images
lander_img = pygame.transform.scale(lander_img, (90, 140)) 
platform_img = pygame.transform.scale(platform_img, (250, 45))



###################################
#                                 #
#    Define Lunar Lander class    #
#                                 #
###################################
class LunarLander(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.angle = 0
        self.image = pygame.transform.rotate(lander_img, self.angle)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.vel_x = 0
        self.vel_y = 0
        self.fuel = FUEL
        
    def update(self):
        # Apply gravity
        self.vel_y += GRAVITY        
        self.image = pygame.transform.rotate(lander_img, self.angle)
        
        # Apply thrust
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] and self.fuel > 0:
            self.vel_y += THRUST
            self.fuel -= 1
        
        # Apply turning
        if keys[pygame.K_LEFT]:
            self.vel_x -= 0.1
            self.angle += 1
        elif keys[pygame.K_RIGHT]:
            self.vel_x += 0.1
            self.angle -= 1
        
        # Move lander
        self.rect.x += self.vel_x
        self.rect.y += self.vel_y
        
        print("VelX: {:.2f}".format(self.vel_x), "VelY: {:.2f}".format(self.vel_y), "X: {:.2f}".format(self.rect.x), "Y: {:.2f}".format(self.rect.y), "FUEL: {:.2f}".format(self.fuel))
        # Check for collision with the platform
        if self.rect.colliderect(platform.rect):
            if (self.vel_y < 4) and (self.vel_x < 0.25) and (self.vel_x > -0.25):
                print("Landed!")
                self.kill()
            else:
                print("Crashed!")
                self.kill()
        # Check for out of screen
        if self.rect.x > SCREEN_WIDTH or self.rect.x < 0 or self.rect.y > SCREEN_HEIGHT or self.rect.y < 0:
            print("Crashed!")
            self.kill()



###############################
#                             #
#    Define platform class    #
#                             #
###############################
class Platform(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = platform_img
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y



####################
#                  #
#    Initialize    #
#                  #
####################
# Create the sprites
lander = LunarLander(SCREEN_WIDTH/2, 0)
lander = LunarLander(random.randint(0, SCREEN_WIDTH-150), 0)
platform = Platform(random.randint(0, SCREEN_WIDTH-150), SCREEN_HEIGHT-70)
platform = Platform(random.randint(0, SCREEN_WIDTH-150), SCREEN_HEIGHT-70)
# Create the sprite groups
all_sprites = pygame.sprite.Group()
all_sprites.add(lander, platform)
# Create the clock
clock = pygame.time.Clock()



######################
#                    #
#    Run the game    #
#                    #
######################
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    # Update the game
    all_sprites.update()
    # Draw the screen
    screen.fill(BLACK)
    all_sprites.draw(screen)
    pygame.display.flip()
    # Limit the frame rate
    clock.tick(FPS)



# Quit Pygame
pygame.quit()