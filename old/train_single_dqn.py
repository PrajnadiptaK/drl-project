from single_intersection_env import SingleIntersectionEnv
from dqn_agent import DQNAgent
import numpy as np

env = SingleIntersectionEnv()

state_dim = 5   # because our state vector is [q_ns, q_ew, ph0, ph1, phase_elapsed]
action_dim = 2  # extend or switch

agent = DQNAgent(state_dim, action_dim)

episodes = 200
target_update_freq = 20
batch_size = 32

for ep in range(episodes):
    state = env.reset()
    total_reward = 0

    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        agent.buffer.push(state, action, reward, next_state, done)
        agent.train_step(batch_size)

        state = next_state
        total_reward += reward

    agent.decay_epsilon()

    if ep % target_update_freq == 0:
        agent.update_target()

    print(f"Episode {ep+1}/{episodes}, Total reward: {total_reward:.2f}, epsilon:{agent.epsilon:.3f}")
