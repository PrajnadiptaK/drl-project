import numpy as np
from multi_intersection_env import TrafficNetworkEnv

# ========================================================
# FIXED TIME CONTROLLER PARAMETERS
# ========================================================

SWITCH_PERIOD = 6   # switch every 6 decisions (adjustable)

def fixed_time_policy(env, state_dict, step_counters):
    """
    Simple fixed-time policy:
    - If current phase time < SWITCH_PERIOD: extend
    - else: switch and reset its timer
    """
    actions = {}

    for i in range(env.num_nodes):
        phase_elapsed_norm = state_dict[i][4]  # last element of state: normalized phase time

        # phase_elapsed_norm is scaled 0-1, convert back to steps
        elapsed_steps = int(phase_elapsed_norm * env.nodes[i].max_green_time / env.nodes[i].dt)

        if step_counters[i] >= SWITCH_PERIOD:
            actions[i] = 1   # SWITCH
            step_counters[i] = 0
        else:
            actions[i] = 0   # EXTEND
            step_counters[i] += 1

    return actions


# ========================================================
# RUN BASELINE
# ========================================================

env = TrafficNetworkEnv()
episodes = 20
results = []

for ep in range(episodes):
    state_dict = env.reset()
    done = False
    total_reward = 0

    # per-node cycle timers
    step_counters = {i: 0 for i in range(env.num_nodes)}

    while not done:
        # choose actions using fixed controller
        actions = fixed_time_policy(env, state_dict, step_counters)

        next_state, rewards, done = env.step(actions)

        total_reward += sum(rewards.values())

        state_dict = next_state

    results.append(total_reward)
    print(f"Episode {ep+1}/{episodes} — Total reward: {total_reward:.2f}")

print("\n=== Fixed-Time Baseline Results ===")
print("Average Reward:", sum(results)/len(results))
print("All Episode Rewards:", results)
