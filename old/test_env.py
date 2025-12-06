from single_intersection_env import SingleIntersectionEnv
import numpy as np

env = SingleIntersectionEnv()

state = env.reset()
print("Initial state:", state)

done = False
step_count = 0

while not done and step_count < 10:  # only 10 steps for now
    action = np.random.randint(0, 2)  # random: 0 or 1
    next_state, reward, done, info = env.step(action)

    print(f"Step {step_count+1}: action={action}, "
          f"queues=(NS:{info['queue_ns']}, EW:{info['queue_ew']}), reward={reward}")

    step_count += 1
