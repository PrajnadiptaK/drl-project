import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from multi_intersection_env import TrafficNetworkEnv
from logger import Logger

logger = Logger("train_independent_rl")
# ===========================================
# Small DQN for local RL agent
# ===========================================

class LocalQNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(LocalQNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# ===========================================
# Local Agent wrapper
# ===========================================

class LocalAgent:
    def __init__(self, state_dim, action_dim, lr=0.001):
        self.q_net = LocalQNet(state_dim, action_dim)
        self.target_net = LocalQNet(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = 0.95

        self.epsilon = 1.0
        self.eps_min = 0.1
        self.eps_decay = 0.995

        self.memory = []

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 2)  # 2 actions
        q_values = self.q_net(torch.FloatTensor(state))
        return q_values.argmax().item()

    def store(self, s, a, r, ns, done):
        self.memory.append((s, a, r, ns, done))
        if len(self.memory) > 5000:
            self.memory.pop(0)

    def train_step(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = np.random.choice(len(self.memory), batch_size, replace=False)

        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in batch:
            s, a, r, ns, d = self.memory[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_vals = self.q_net(states).gather(1, actions.unsqueeze(-1)).squeeze()

        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = (q_vals - target).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)


# ===========================================
# Training Script
# ===========================================

env = TrafficNetworkEnv()

num_nodes = env.num_nodes
state_dim = 5
action_dim = 2

agents = {i: LocalAgent(state_dim, action_dim) for i in range(num_nodes)}

episodes = 100
results = []

for ep in range(episodes):
    state_dict = env.reset()
    total_reward = 0
    done = False

    while not done:
        actions = {}
        for i in range(num_nodes):
            actions[i] = agents[i].act(state_dict[i])

        next_states, rewards, done = env.step(actions)

        for i in range(num_nodes):
            agents[i].store(
                state_dict[i],
                actions[i],
                rewards[i],        # local reward
                next_states[i],
                done
            )
            agents[i].train_step()

        state_dict = next_states
        total_reward += sum(rewards.values())

    for i in range(num_nodes):
        agents[i].decay_epsilon()
        if ep % 10 == 0:
            agents[i].update_target()

    results.append(total_reward)
    msg = f"Episode {ep + 1}/{episodes}, Total reward: {total_reward:.2f}"
    print(msg)
    logger.write(msg)

# ============= SAVE MODEL ONCE AFTER TRAINING =============
for i in range(num_nodes):
    torch.save(agents[i].q_net.state_dict(), f"agent_{i}.pth")

logger.write("✔ Independent RL models saved successfully")

summary = f"=== Independent RL Training Complete ===\nAverage Episodic Reward: {sum(results)/len(results):.2f}"
print(summary)
logger.write(summary)

logger.close()
print("✔ log saved using logger.py")
