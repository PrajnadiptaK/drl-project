import pygame
import numpy as np
import torch
import random

from multi_intersection_env import TrafficNetworkEnv
from colight_agent import CoLightAgent

pygame.init()

WIDTH, HEIGHT = 800, 600
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Traffic Signal CoLight Simulation")

WHITE = (245, 245, 245)
GRAY = (170, 170, 170)
BLACK = (0, 0, 0)
GREEN = (50, 200, 50)
RED = (220, 40, 40)

CAR_SPEED = 2
CAR_SIZE = (12, 18)

positions = [(200, 200), (600, 200), (200, 400), (600, 400)]

# Car object
class Car:
    def __init__(self, x, y, stop_x, stop_y, direction):
        self.x = x
        self.y = y
        self.stop_x = stop_x
        self.stop_y = stop_y
        self.direction = direction  # "NS" or "EW"

    def try_move(self, green_light):
        if self.direction == "NS":
            if not green_light and self.y + CAR_SPEED >= self.stop_y - 3:
                return False
            self.y += CAR_SPEED

        elif self.direction == "EW":
            if not green_light and self.x + CAR_SPEED >= self.stop_x - 3:
                return False
            self.x += CAR_SPEED

        return True

# === Load RL agent ===
env = TrafficNetworkEnv()
adj = torch.FloatTensor(env.adj_matrix)

agent = CoLightAgent(5, 2, env.num_nodes)
agent.q_net.load_state_dict(torch.load("trained_colight.pth"))
agent.epsilon = 0.0

clock = pygame.time.Clock()

lanes = {i: {"NS": [], "EW": []} for i in range(env.num_nodes)}

state = env.reset()
state_matrix = np.array(list(state.values()))

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    win.fill(WHITE)

    # Draw intersections
    for pos in positions:
        pygame.draw.rect(win, GRAY, (*pos, 60, 60))

    # RL step
    qs = agent.q_net(torch.FloatTensor(state_matrix), adj)
    actions = agent.select_actions(qs)

    next_state, _, _ = env.step(actions)
    next_matrix = np.array(list(next_state.values()))

    for idx, pos in enumerate(positions):

        # Extract queue values
        q_ns = int(state_matrix[idx][0])
        q_ew = int(state_matrix[idx][1])

        # Determine phase
        phase = 0 if state_matrix[idx][2] > state_matrix[idx][3] else 1

        # Draw signals
        pygame.draw.circle(win, GREEN if phase == 0 else RED, (pos[0] + 30, pos[1] - 10), 8)
        pygame.draw.circle(win, GREEN if phase == 1 else RED, (pos[0] - 10, pos[1] + 30), 8)

        # Random arrivals instead of full reset-spawn
        if random.random() < 0.05:  # 5% chance new NS car arrives
            lanes[idx]["NS"].append(Car(pos[0] + 22, pos[1] - 150, pos[0] + 22, pos[1] + 60, "NS"))

        if random.random() < 0.05:  # 5% chance new EW car arrives
            lanes[idx]["EW"].append(Car(pos[0] - 150, pos[1] + 22, pos[0] + 60, pos[1] + 22, "EW"))

        # Update and draw cars
        for direction in ["NS", "EW"]:
            remove = []
            green = (phase == 0 if direction == "NS" else phase == 1)

            for car in lanes[idx][direction]:
                car.try_move(green)

                # draw car
                pygame.draw.rect(win, BLACK, (car.x, car.y, *CAR_SIZE))

                # remove if passed
                if direction == "NS" and car.y > pos[1] + 120:
                    remove.append(car)
                elif direction == "EW" and car.x > pos[0] + 120:
                    remove.append(car)

            for car in remove:
                lanes[idx][direction].remove(car)

    state_matrix = next_matrix

    pygame.display.update()
    clock.tick(30)
