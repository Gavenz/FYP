import pygame

class PhysicsEntity:
    def __init__(self, game, entity_type, pos,size):
        self.game = game
        self.type = entity_type
        self.pos = list(pos)
        