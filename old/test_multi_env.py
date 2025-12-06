from multi_intersection_env import TrafficNetworkEnv
import numpy as np

env = TrafficNetworkEnv()

state = env.reset()
print("Initial state:", state)

done = False
step_count = 0

while not done and step_count < 5:
    # random actions: 0 or 1 for each node
    actions = {i: np.random.randint(0, 2) for i in range(4)}
    next_state, rewards, done = env.step(actions)

    print(f"Step {step_count+1}: actions={actions}, rewards={rewards}")

    state = next_state
    step_count += 1
