import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from colight_qnet import CoLightQNetwork


class CoLightReplayBuffer:
    def __init__(self, capacity=20000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state_matrix, actions, reward_vector, next_state_matrix, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (state_matrix, actions, reward_vector, next_state_matrix, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        sampled = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*sampled)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)


class CoLightAgent:
    def __init__(self, state_dim, action_dim, num_nodes, lr=1e-3, gamma=0.99, embed_dim=32):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_nodes = num_nodes
        self.gamma = gamma

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.q_net = CoLightQNetwork(state_dim, embed_dim, action_dim, num_nodes)
        self.target_net = CoLightQNetwork(state_dim, embed_dim, action_dim, num_nodes)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.buffer = CoLightReplayBuffer()

    def select_actions(self, q_values):
        actions = {}
        for i in range(self.num_nodes):
            if random.random() < self.epsilon:
                actions[i] = random.randrange(self.action_dim)
            else:
                actions[i] = q_values[i].argmax().item()
        return actions

    def store(self, state_mat, actions, rewards, next_state_mat, done):
        self.buffer.push(state_mat, actions, rewards, next_state_mat, done)

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def train_step(self, batch_size, adj):
        if len(self.buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        states = torch.FloatTensor(states)         # [B, N, D]
        rewards = torch.FloatTensor(rewards)       # [B, N]
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(-1)

        # Repeat adjacency across batch
        adj_tensor = torch.FloatTensor(adj).unsqueeze(0).repeat(batch_size, 1, 1)  # [B, N, N]

        q_values = []
        next_q_values = []

        for b in range(batch_size):
            q_values.append(self.q_net(states[b], adj_tensor[b]))
            next_q_values.append(self.target_net(next_states[b], adj_tensor[b]))

        q_values = torch.stack(q_values)
        next_q_values = torch.stack(next_q_values)

        B = batch_size
        td_target = torch.zeros_like(q_values)

        for b in range(B):
            for i in range(self.num_nodes):
                td_target[b, i] = rewards[b, i] + self.gamma * next_q_values[b, i].max() * (1 - dones[b])

        loss = ((q_values - td_target.detach()) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
