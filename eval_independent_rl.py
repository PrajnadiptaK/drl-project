import numpy as np
import torch
from multi_intersection_env import TrafficNetworkEnv
from train_independent_rl import LocalAgent
from logger import Logger

env = TrafficNetworkEnv()
logger = Logger("eval_independent_rl")

num_nodes = env.num_nodes

agents = {}
for i in range(num_nodes):
    agent = LocalAgent(5, 2)
    agent.q_net.load_state_dict(torch.load(f"agent_{i}.pth"))
    agent.epsilon = 0.0
    agents[i] = agent

episodes = 20
results = []

for ep in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        actions = {i: agents[i].act(state[i]) for i in range(num_nodes)}
        next_state, rewards, done = env.step(actions)
        total_reward += sum(rewards.values())
        state = next_state

    results.append(total_reward)
    msg = f"Episode {ep+1}/{episodes} — Total reward: {total_reward:.2f}"
    print(msg)
    logger.write(msg)

summary = f"=== Independent RL Eval Complete ===\nAverage Reward: {sum(results)/len(results):.2f}"
print(summary)
logger.write(summary)
logger.close()
print("Eval log saved")
