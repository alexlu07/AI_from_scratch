import pygame
import sys
import numpy as np
from pygame.locals import *
import skimage.measure


class Canvas:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((784, 784))

        self.clock = pygame.time.Clock()
        self.brush_size = 20
        self.label = None

    def draw(self):
        self.screen.fill((255, 255, 255))
        pygame.display.flip()
        mouse = False
        while True:
            self.clock.tick(100)
            x, y = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                elif event.type == MOUSEBUTTONDOWN:
                    mouse = True
                elif event.type == MOUSEBUTTONUP:
                    mouse = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_0:
                        self.label = 0
                        return

                    if event.key == pygame.K_1:
                        self.label = 1
                        return

                    if event.key == pygame.K_2:
                        self.label = 2
                        return

                    if event.key == pygame.K_3:
                        self.label = 3
                        return

                    if event.key == pygame.K_4:
                        self.label = 4
                        return

                    if event.key == pygame.K_5:
                        self.label = 5
                        return

                    if event.key == pygame.K_6:
                        self.label = 6
                        return

                    if event.key == pygame.K_7:
                        self.label = 7
                        return

                    if event.key == pygame.K_8:
                        self.label = 8
                        return

                    if event.key == pygame.K_9:
                        self.label = 9
                        return
            if mouse:
                pygame.draw.circle(self.screen, (0, 0, 0), (x, y), self.brush_size)
                pygame.display.flip()

    def get_input(self):
        data = pygame.surfarray.array3d(self.screen)
        before_pooling = 255 - np.average(data, axis=2)
        before_pooling = before_pooling.T
        pooled = skimage.measure.block_reduce(before_pooling, (28, 28), np.mean)
        result = pooled.reshape([1, -1])
        result /= 255

        l = np.zeros([1, 10])
        l[0][self.label] = 1
        return result, l