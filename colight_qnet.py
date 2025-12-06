import torch
import torch.nn as nn
from gat_layer import GATLayer


class CoLightQNetwork(nn.Module):
    def __init__(self, state_dim, embed_dim, action_dim, num_nodes):
        super().__init__()

        self.num_nodes = num_nodes

        # Local state encoder
        self.embed = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )

        # Graph attention layer
        self.gat = GATLayer(embed_dim, embed_dim)

        # Q-head (per node)
        self.q_head = nn.Linear(embed_dim, action_dim)

    def forward(self, x, adj):
        """
        x: [N, state_dim]
        adj: [N, N] adjacency matrix
        Returns: Q-values for each node [N, action_dim]
        """
        h = self.embed(x)                     # [N, embed_dim]
        h_gat = self.gat(h, adj)              # neighbor-aggregated embeddings
        q_values = self.q_head(h_gat)         # per-node action values
        return q_values
