import numpy as np
from multi_intersection_env import TrafficNetworkEnv
from logger import Logger

SWITCH_PERIOD = 6

def fixed_time_policy(env, state_dict, step_counters):
    actions = {}
    for i in range(env.num_nodes):
        if step_counters[i] >= SWITCH_PERIOD:
            actions[i] = 1
            step_counters[i] = 0
        else:
            actions[i] = 0
            step_counters[i] += 1
    return actions

env = TrafficNetworkEnv()
logger = Logger("eval_fixed_time")

episodes = 20
results = []

for ep in range(episodes):
    state_dict = env.reset()
    step_counters = {i: 0 for i in range(env.num_nodes)}
    done = False
    total_reward = 0

    while not done:
        actions = fixed_time_policy(env, state_dict, step_counters)
        next_state, rewards, done = env.step(actions)
        total_reward += sum(rewards.values())
        state_dict = next_state

    results.append(total_reward)
    msg = f"Episode {ep+1}/{episodes} — Total reward: {total_reward:.2f}"
    print(msg)
    logger.write(msg)

summary = f"=== Fixed-Time Eval Complete ===\nAverage Reward: {sum(results)/len(results):.2f}"
print(summary)
logger.write(summary)
logger.close()
print("Eval log saved")
