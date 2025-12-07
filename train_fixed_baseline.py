import numpy as np
from multi_intersection_env import TrafficNetworkEnv
from logger import Logger

env = TrafficNetworkEnv()
logger = Logger("fixed_baseline")

episodes = 20
results = []

for ep in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # fixed policy: extend every intersection always
        actions = {i: 0 for i in range(env.num_nodes)}

        next_state, rewards, done = env.step(actions)

        total_reward += sum(rewards.values())
        state = next_state

    results.append(total_reward)
    msg = f"[Fixed-time] Episode {ep + 1}/{episodes} -> Total reward {total_reward:.2f}"
    print(msg)
    logger.write(msg)

summary = f"=== Fixed-time Baseline Completed ===\nAverage reward = {sum(results)/len(results):.2f}"
print(summary)
logger.write(summary)

logger.close()
print("log saved for fixed baseline")
