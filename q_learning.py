import pygame
import numpy as np
import random

SCREEN_WIDTH = 900
SCREEN_HEIGHT = 700
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# learning parameters
ALPHA = 0.05  # learning ratio
GAMMA = 0.99

BLOCKS_X = 50
BLOCKS_Y = 50
BLOCK_WIDTH = SCREEN_WIDTH/BLOCKS_X
BLOCK_HEIGHT = SCREEN_HEIGHT/BLOCKS_Y

GOAL_X = None
GOAL_Y = None

# states = [[0.0 for c in range(BLOCKS_Y)] for r in range(BLOCKS_X)]
states = np.zeros((BLOCKS_X, BLOCKS_Y))
prev_states = np.zeros((BLOCKS_X, BLOCKS_Y))
policy = np.zeros((BLOCKS_X, BLOCKS_Y), dtype=int)
blocks = np.ones((BLOCKS_X, BLOCKS_Y), dtype=int)


# images
arrow = pygame.image.load("arrow.webp")
arrow = pygame.transform.scale(arrow, (SCREEN_WIDTH/BLOCKS_Y*0.75, BLOCK_HEIGHT*0.75))
arrow_left = pygame.transform.rotate(arrow, 90)
arrow_up = pygame.transform.rotate(arrow, 180)
arrow_right = pygame.transform.rotate(arrow, 270)
arrow_down = arrow


trophy = pygame.image.load("trophy.jpg")
trophy = pygame.transform.scale(trophy, (BLOCK_WIDTH, BLOCK_HEIGHT))

X = pygame.image.load("X.webp")
X = pygame.transform.scale(X, (BLOCK_WIDTH, BLOCK_HEIGHT))

circle = pygame.image.load("circle.png")
circle = pygame.transform.scale(circle, (BLOCK_WIDTH, BLOCK_HEIGHT))

images = [circle, arrow_up, arrow_right, arrow_down, arrow_left]


# policy parameters
STAY = 0
UP = 1
RIGHT = 2
DOWN = 3
LEFT = 4


mode = "prep"


def learn():
    global states
    for i in range(1, BLOCKS_X-1):
        for j in range(1, BLOCKS_Y-1):
            if not (i == GOAL_X and j == GOAL_Y):
                states[i, j] = states[i, j] * (1-ALPHA) + \
                              max(states[i+1, j], states[i-1, j], states[i, j+1], states[i, j-1]) * ALPHA * GAMMA
    states = np.minimum(states, blocks)


def set_policy():
    for i in range(1, BLOCKS_X-1):
        for j in range(1, BLOCKS_Y-1):
            policy[i, j] = np.argmax(np.array([states[i, j],
                                               states[i-1, j],
                                               states[i, j+1],
                                               states[i+1, j],
                                               states[i, j-1]]))


def draw():
    for i in range(BLOCKS_X):
        for j in range(BLOCKS_Y):
            pygame.draw.rect(color=(int(states[i, j]*255), int(states[i, j]*255), int(states[i, j]*255)),
                             rect=(i * BLOCK_WIDTH, j * BLOCK_HEIGHT,
                                   BLOCK_WIDTH, BLOCK_HEIGHT), surface=screen)
    draw_blocks()
    draw_trophy()
    pygame.display.update()


def draw_policy():
    for i in range(BLOCKS_X):
        for j in range(BLOCKS_Y):
            screen.blit((images[policy[i, j]]), (i * BLOCK_WIDTH, j * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT))
    pygame.display.update()


def set_boundaries():
    for j in range(0, BLOCKS_X):
        blocks[0, j] = blocks[BLOCKS_Y-1, j] = 0
    for i in range(0, BLOCKS_Y):
        blocks[i, 0] = blocks[i, BLOCKS_X-1] = 0


def set_blocks():
    global GOAL_X
    global GOAL_Y

    keys = pygame.key.get_pressed()
    mx, my = pygame.mouse.get_pos()
    mouse_buttons = pygame.mouse.get_pressed()
    if mouse_buttons[0]:
        blocks[int((mx)/BLOCK_WIDTH), int((my)/BLOCK_HEIGHT)] = 0
    if mouse_buttons[0] and keys[pygame.K_BACKSPACE]:
        blocks[int((mx)/BLOCK_WIDTH), int((my)/BLOCK_HEIGHT)] = 1
    if mouse_buttons[2]:
        GOAL_X = int(mx/BLOCK_WIDTH)
        GOAL_Y = int(my/BLOCK_HEIGHT)
        states[GOAL_X, GOAL_Y] = 1


def draw_blocks():
    for i in range(BLOCKS_X):
        for j in range(BLOCKS_Y):
            if blocks[i][j] == 0:
                screen.blit(X, (i * BLOCK_WIDTH, j * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT))


def draw_trophy():
    screen.blit(trophy, (GOAL_X * BLOCK_WIDTH, GOAL_Y * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT))


set_boundaries()

while mode == "prep":
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
        set_blocks()
        screen.fill((0, 0, 0))
        draw_blocks()
        pygame.display.update()
        if GOAL_X is not None:
            mode = "solve"
while mode == "solve":
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
    prev_states = states.copy()
    learn()
    screen.fill((0, 0, 0))
    draw()
    if np.max(np.abs(states-prev_states)) < 1e-8:
        mode = "policy"
        screen.fill((0, 0, 0))
        set_policy()
        print('now showing the policy')
        draw_policy()
        draw_blocks()
        draw_trophy()
        pygame.display.update()
while mode == "policy":
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

