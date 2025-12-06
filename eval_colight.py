from multi_intersection_env import TrafficNetworkEnv
from colight_agent import CoLightAgent
import numpy as np
import torch

env = TrafficNetworkEnv()

# Get adjacency from environment instead of hardcoding
adj = torch.FloatTensor(env.adj_matrix)

# Load your trained agent
agent = CoLightAgent(5, 2, 4)

agent.epsilon = 0.0  # evaluation mode

episodes = 20
results = []

for ep in range(episodes):
    states = env.reset()
    state_matrix = np.array(list(states.values()))

    done = False
    total_reward = 0

    while not done:

        # USE env-provided adjacency instead of hardcoding
        q_vals = agent.q_net(
            torch.FloatTensor(state_matrix),
            adj
        )

        actions = {i: q_vals[i].argmax().item() for i in range(4)}

        next_states, rewards, done = env.step(actions)

        state_matrix = np.array(list(next_states.values()))
        total_reward += sum(rewards.values())

    results.append(total_reward)

print("Average Eval Reward:", sum(results) / len(results))
print("Eval Rewards:", results)
