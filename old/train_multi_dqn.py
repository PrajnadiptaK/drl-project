from multi_intersection_env import TrafficNetworkEnv
from shared_dqn_agent import SharedDQNAgent
import numpy as np

env = TrafficNetworkEnv()

state_dim = 5   # queue_ns, queue_ew, phase_0, phase_1, phase_elapsed
action_dim = 2  # extend or switch

agent = SharedDQNAgent(state_dim, action_dim)

episodes = 150
target_update_freq = 20
batch_size = 32

for ep in range(episodes):
    state = env.reset()
    total_reward = 0

    done = False
    while not done:
        actions = {}
        for i in range(env.num_nodes):
            actions[i] = agent.select_action(state[i])

        next_state, rewards, done = env.step(actions)

        for i in range(env.num_nodes):
            agent.store(state[i], actions[i], rewards[i], next_state[i], done)

        agent.train_step(batch_size)

        state = next_state
        total_reward += sum(rewards.values())

    agent.decay_epsilon()

    if ep % target_update_freq == 0:
        agent.update_target()

    print(f"Episode {ep+1}/{episodes}, Total reward: {total_reward:.2f}, epsilon:{agent.epsilon:.3f}")
