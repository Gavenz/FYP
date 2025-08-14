import sys
import pygame

pygame.init()

class Game:
    def __init__(self):
        pygame.init()
        #initialize the game window
        pygame.display.set_caption("My Game")
        self.screen = pygame.display.set_mode((640, 480))

        # create a clock object to manage frame rate
        self.clock = pygame.time.Clock()
        self.img = pygame.image.load("C:/Users/gaven/fyp/FYP-1/pygame_tutorial/pygame_data/images/clouds/cloud_1.png")

# game loop
    def run(self):
        while True:
            self.screen.blit(self.img, (100,200))
            # this gets user input as events with a type attribute
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # this ensures the window updates.
            pygame.display.update()
            self.clock.tick(60)  # limit to 60 frames per second

Game().run()
