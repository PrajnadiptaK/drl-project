import pygame
import numpy as np
import torch

from multi_intersection_env import TrafficNetworkEnv
from colight_agent import CoLightAgent

# ========== Pygame Setup ==========
pygame.init()
WIDTH, HEIGHT = 800, 600
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("CoLight – Queue & Signal Visualization")

WHITE = (245, 245, 245)
GRAY = (180, 180, 180)
BLACK = (0, 0, 0)
GREEN = (60, 200, 60)
RED = (220, 50, 50)
BLUE = (50, 100, 220)

FONT = pygame.font.SysFont("Arial", 16)

# 4 intersections arranged in grid
positions = [(200, 200), (600, 200), (200, 400), (600, 400)]

# How often to let RL/environment step (in frames)
FPS = 30
STEP_EVERY = 5   # RL step every 5 frames so humans can see changes

# ========== Load Environment + Trained CoLight Agent ==========

env = TrafficNetworkEnv()
adj = torch.FloatTensor(env.adj_matrix)

state_dim = 5
action_dim = 2
num_nodes = env.num_nodes

agent = CoLightAgent(state_dim, action_dim, num_nodes)
agent.q_net.load_state_dict(torch.load("trained_colight.pth"))
agent.epsilon = 0.0  # pure exploitation

clock = pygame.time.Clock()

state_dict = env.reset()
state_matrix = np.array(list(state_dict.values()))
done = False
frame_counter = 0

running = True
while running:
    # ---------- Event handling ----------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    win.fill(WHITE)

    # ---------- RL + Env Step (every STEP_EVERY frames) ----------
    if not done and frame_counter % STEP_EVERY == 0:
        state_tensor = torch.FloatTensor(state_matrix)
        q_values = agent.q_net(state_tensor, adj)
        actions = agent.select_actions(q_values)

        next_state_dict, rewards, done = env.step(actions)
        state_matrix = np.array(list(next_state_dict.values()))

    # ---------- Drawing Intersections, Lights, and Queues ----------
    for idx, pos in enumerate(positions):
        # Draw intersection box
        pygame.draw.rect(win, GRAY, (*pos, 80, 80), border_radius=6)

        # Extract state for this intersection
        q_ns = float(state_matrix[idx][0])
        q_ew = float(state_matrix[idx][1])
        phase0 = state_matrix[idx][2]
        phase1 = state_matrix[idx][3]
        phase = 0 if phase0 > phase1 else 1  # 0 = NS green, 1 = EW green

        # ---- Traffic lights (small circles) ----
        # NS light (top)
        ns_color = GREEN if phase == 0 else RED
        pygame.draw.circle(win, ns_color, (pos[0] + 40, pos[1] - 10), 8)

        # EW light (left)
        ew_color = GREEN if phase == 1 else RED
        pygame.draw.circle(win, ew_color, (pos[0] - 10, pos[1] + 40), 8)

        # ---- Queue bars ----
        # Scale queues to max visual height/width
        max_bar = 60.0
        ns_height = min(max_bar, q_ns * 2.0)  # scale factor 2
        ew_width = min(max_bar, q_ew * 2.0)

        # NS queue bar (vertical) – below intersection
        ns_rect = pygame.Rect(pos[0] + 25, pos[1] + 85 - ns_height, 10, ns_height)
        pygame.draw.rect(win, BLUE, ns_rect)

        # EW queue bar (horizontal) – to the right of intersection
        ew_rect = pygame.Rect(pos[0] + 85, pos[1] + 25, ew_width, 10)
        pygame.draw.rect(win, BLUE, ew_rect)

        # ---- Labels ----
        text = FONT.render(f"Node {idx}", True, BLACK)
        win.blit(text, (pos[0] + 20, pos[1] - 30))

        qtext = FONT.render(f"NS:{q_ns:.1f}  EW:{q_ew:.1f}", True, BLACK)
        win.blit(qtext, (pos[0] - 20, pos[1] + 90))

    # ---------- Info / Status text ----------
    status = "Running" if not done else "Episode done (env time limit)"
    info_text = FONT.render(f"{status}", True, BLACK)
    win.blit(info_text, (20, 20))

    pygame.display.update()
    clock.tick(FPS)
    frame_counter += 1

pygame.quit()
