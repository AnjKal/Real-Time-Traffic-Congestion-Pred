import torch
import torch.nn as nn
import torch.nn.functional as F


class TrafficGCN(nn.Module):

    def __init__(self, input_features=1, hidden_features=16, output_features=1):
        super().__init__()

        self.gcn1 = nn.Linear(input_features, hidden_features)
        self.gcn2 = nn.Linear(hidden_features, hidden_features)
        self.output_layer = nn.Linear(hidden_features, output_features)

    def forward(self, x, adjacency):

        identity = torch.eye(
            adjacency.size(0),
            device=adjacency.device
        )

        adjacency = adjacency + identity

        degree = torch.sum(adjacency, dim=1)
        degree_inv = torch.pow(degree.clamp(min=1e-8), -0.5)

        degree_matrix = torch.diag(degree_inv)

        normalized_adjacency = (
            degree_matrix @ adjacency @ degree_matrix
        )

        x = normalized_adjacency @ x
        x = F.relu(self.gcn1(x))

        x = normalized_adjacency @ x
        x = F.relu(self.gcn2(x))

        return self.output_layer(x)