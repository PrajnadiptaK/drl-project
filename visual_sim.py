import pygame
import numpy as np
import torch

from multi_intersection_env import TrafficNetworkEnv
from colight_agent import CoLightAgent

pygame.init()

WIDTH, HEIGHT = 800, 600
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Traffic Signal CoLight Simulation")

WHITE = (245, 245, 245)
GRAY = (120, 120, 120)
GREEN = (60, 200, 60)
RED = (220, 50, 50)
BLACK = (0, 0, 0)

CAR_SPEED = 2
CAR_SIZE = (12, 18)

# 4 intersections in arranged grid
positions = [(200, 200), (600, 200), (200, 400), (600, 400)]


class Car:
    def __init__(self, x, y, lane, stop_x, stop_y, dir):
        self.x = x
        self.y = y
        self.lane = lane
        self.stop_x = stop_x
        self.stop_y = stop_y
        self.dir = dir  # "NS" or "EW"

    def update(self, green):
        # NS lane -> move downward
        if self.dir == "NS":
            # stop if red and approaching stop line
            if (not green) and self.y + CAR_SPEED >= self.stop_y:
                return False
            self.y += CAR_SPEED

        # EW lane -> move right
        else:
            if (not green) and self.x + CAR_SPEED >= self.stop_x:
                return False
            self.x += CAR_SPEED

        return True  # moved


# ========== LOAD ENV + MODEL ==========
env = TrafficNetworkEnv()
adj = torch.FloatTensor(env.adj_matrix)

agent = CoLightAgent(5, 2, env.num_nodes)
agent.q_net.load_state_dict(torch.load("trained_colight.pth"))
agent.epsilon = 0.0

clock = pygame.time.Clock()

# cars stored persistently per intersection
lanes = {i: {"NS": [], "EW": []} for i in range(env.num_nodes)}

state = env.reset()
state_matrix = np.array(list(state.values()))

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    win.fill(WHITE)

    # draw intersection blocks + visible traffic lights
    for idx, pos in enumerate(positions):
        pygame.draw.rect(win, GRAY, (*pos, 60, 60))

        # extract phase from state representation
        phase = np.argmax(state_matrix[idx][2:4])

        # NS indicator (top-middle of intersection)
        ns_color = GREEN if phase == 0 else RED
        pygame.draw.circle(win, ns_color, (pos[0] + 30, pos[1] + 10), 8)

        # EW indicator (right-middle of intersection)
        ew_color = GREEN if phase == 1 else RED
        pygame.draw.circle(win, ew_color, (pos[0] + 50, pos[1] + 30), 8)

    # RL action
    qs = agent.q_net(torch.FloatTensor(state_matrix), adj)
    actions = agent.select_actions(qs)

    next_state, _, _ = env.step(actions)
    next_matrix = np.array(list(next_state.values()))

    # update cars visually
    for idx, pos in enumerate(positions):
        q_ns = int(state_matrix[idx][0])
        q_ew = int(state_matrix[idx][1])

        # stop line positions
        stop_x = pos[0] + 60
        stop_y = pos[1] + 60

        # phase: 0=NS green, 1=EW green
        phase = np.argmax(state_matrix[idx][2:4])

        # spawn cars only to match queue count
        while len(lanes[idx]["NS"]) < q_ns:
            lanes[idx]["NS"].append(Car(pos[0] + 25, pos[1] - 100, "NS", pos[0] + 25, pos[1] + 60, "NS"))
        while len(lanes[idx]["EW"]) < q_ew:
            lanes[idx]["EW"].append(Car(pos[0] - 100, pos[1] + 25, "EW", pos[0] + 60, pos[1] + 25, "EW"))

        # update movement
        for lane in ["NS", "EW"]:
            remove = []
            for c in lanes[idx][lane]:
                green = (phase == 0 if lane == "NS" else phase == 1)
                moved = c.update(green)
                # remove if crossed intersection
                if (c.dir == "NS" and c.y > pos[1] + 100) or (c.dir == "EW" and c.x > pos[0] + 100):
                    remove.append(c)

                # draw
                pygame.draw.rect(win, BLACK, (c.x, c.y, *CAR_SIZE))

            for r in remove:
                lanes[idx][lane].remove(r)

    state_matrix = next_matrix

    pygame.display.update()
    clock.tick(30)
