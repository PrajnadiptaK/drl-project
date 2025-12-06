import numpy as np
from multi_intersection_env import TrafficNetworkEnv
from colight_agent import CoLightAgent
import torch
from logger import Logger

env = TrafficNetworkEnv()

# Load adjacency matrix ONCE from environment and convert to tensor
adj = torch.FloatTensor(env.adj_matrix)

logger = Logger("train_colight")

state_dim = 5
action_dim = 2
num_nodes = env.num_nodes

agent = CoLightAgent(state_dim, action_dim, num_nodes)

episodes = 100
batch_size = 16
target_update = 10

for ep in range(episodes):
    state_dict = env.reset()

    # Convert state dict to matrix for batch input
    state_matrix = np.array(list(state_dict.values()))

    total_reward = 0
    done = False

    while not done:
        # Convert to tensor once per step
        state_tensor = torch.FloatTensor(state_matrix)

        # Use adjacency loaded from env (no hardcoding)
        q_values = agent.q_net(state_tensor, adj)

        # Agent chooses actions per node
        actions = agent.select_actions(q_values)

        next_state_dict, rewards, done = env.step(actions)

        reward_vector = np.array(list(rewards.values()))
        next_state_matrix = np.array(list(next_state_dict.values()))

        # Replay buffer push
        agent.store(state_matrix, actions, reward_vector, next_state_matrix, done)

        # Training step (CoLight attention uses adj)
        agent.train_step(batch_size, adj)

        # Transition to next state
        state_matrix = next_state_matrix

        total_reward += reward_vector.sum()

    # Exploration decay
    agent.decay_epsilon()

    # Target network sync
    if ep % target_update == 0:
        agent.update_target()

    msg = f"Episode {ep + 1}/{episodes}, Total reward: {total_reward:.2f}, epsilon:{agent.epsilon:.3f}"
    print(msg)
    logger.write(msg)

# ========= SAVE TRAINED MODEL =========
torch.save(agent.q_net.state_dict(), "trained_colight.pth")
logger.write("✔ Model saved")
logger.close()
print("✔ Log saved using logger.py")