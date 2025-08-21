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
        self.img = pygame.image.load("C:/Users/PC/Desktop/GITHUB FYP/FYP/pygame_tutorial/pygame_data/images/clouds/cloud_1.png")
        self.img.set_colorkey((0,0,0))  # set black as transparent color
        
        self.img_pos = [160, 260]
        self.movement = [False,False]

        self.collision_area = pygame.Rect(50,50,500,50) # define a collision area for the image


# game loop
    def run(self):
        while True:
            self.screen.fill ((14,219,248)) #fill the screen with a color, without a trail of clouds
            
            # layering is such that cloud will be drawn on top of background, but below the collision area
            
            self.img_pos[1] += (self.movement[1] - self.movement[0])*5  # Down movement, multiplied by 5 pixels per frame
            self.screen.blit(self.img, self.img_pos)

            img_r = pygame.Rect(self.img_pos[0], self.img_pos[1], self.img.get_width(), self.img.get_height())
            if img_r.colliderect(self.collision_area):
                pygame.draw.rect(self.screen, (0,100,255), self.collision_area)
            else:
                pygame.draw.rect(self.screen, (0,50,155), self.collision_area)

            
            # this gets user input as events with a type attribute
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                # arrowkey events for universal keyboard layouts
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.movement[0] = True
                    if event.key == pygame.K_DOWN:
                        self.movement[1] = True
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_UP:
                        self.movement[0] = False
                    if event.key == pygame.K_DOWN:
                        self.movement[1] = False
            

            # this ensures the window updates.
            pygame.display.update()
            self.clock.tick(60)  # limit to 60 frames per second

Game().run()
