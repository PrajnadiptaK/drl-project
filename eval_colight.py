from multi_intersection_env import TrafficNetworkEnv
from colight_agent import CoLightAgent
import numpy as np
import torch
from logger import Logger

env = TrafficNetworkEnv()
logger = Logger("eval_colight")

adj = torch.FloatTensor(env.adj_matrix)

agent = CoLightAgent(5, 2, env.num_nodes)
agent.q_net.load_state_dict(torch.load("trained_colight.pth"))
agent.epsilon = 0.0

episodes = 20
results = []

for ep in range(episodes):
    states = env.reset()
    state_matrix = np.array(list(states.values()))
    done = False
    total_reward = 0

    while not done:
        q_vals = agent.q_net(torch.FloatTensor(state_matrix), adj)
        actions = {i: q_vals[i].argmax().item() for i in range(env.num_nodes)}

        next_states, rewards, done = env.step(actions)
        state_matrix = np.array(list(next_states.values()))
        total_reward += sum(rewards.values())

    results.append(total_reward)
    msg = f"Episode {ep+1}/{episodes} — Total reward: {total_reward:.2f}"
    print(msg)
    logger.write(msg)

summary = f"=== CoLight Eval Complete ===\nAverage Reward: {sum(results)/len(results):.2f}"
print(summary)
logger.write(summary)
logger.close()
print("Eval log saved")
