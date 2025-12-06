import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """
    Simple single-head Graph Attention layer
    Used to compute neighbor-aware embeddings.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Linear(2 * out_dim, 1, bias=False)

    def forward(self, x, adj):
        """
        x: tensor [N, in_dim]         node features
        adj: adjacency matrix [N, N]  1 = neighbors
        """
        h = self.W(x)  # [N, out_dim]
        N = h.size(0)

        # Prepare attention scores
        h_repeat_i = h.unsqueeze(1).repeat(1, N, 1)
        h_repeat_j = h.unsqueeze(0).repeat(N, 1, 1)

        pair_features = torch.cat([h_repeat_i, h_repeat_j], dim=-1)

        e = self.attn(pair_features).squeeze(-1)  # [N, N]

        # Mask non-neighbors by setting attention to very negative
        e = e.masked_fill(adj == 0, float('-inf'))

        # Softmax attention
        attn_weights = torch.softmax(e, dim=1)

        # Weighted sum of neighbors
        h_new = torch.matmul(attn_weights, h)

        return h_new
