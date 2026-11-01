import torch
from torch import nn

FORECAST_MODE = "persistence_residual_v1"


class DiffusionGraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, supports, diffusion_steps=2):
        super().__init__()
        self.diffusion_steps = diffusion_steps
        self.register_buffer("supports", supports)
        self.projection = nn.Linear(input_dim * (1 + supports.shape[0] * diffusion_steps), output_dim)

    def forward(self, x):
        features = [x]
        for support in self.supports:
            propagated = x
            for _ in range(self.diffusion_steps):
                propagated = torch.einsum("nm,bmf->bnf", support, propagated)
                features.append(propagated)
        return self.projection(torch.cat(features, dim=-1))


class GraphGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, supports, diffusion_steps=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gates = DiffusionGraphConv(input_dim + hidden_dim, 2 * hidden_dim, supports, diffusion_steps)
        self.candidate = DiffusionGraphConv(input_dim + hidden_dim, hidden_dim, supports, diffusion_steps)

    def forward(self, x, h):
        combined = torch.cat([x, h], dim=-1)
        reset, update = torch.sigmoid(self.gates(combined)).chunk(2, dim=-1)
        candidate_input = torch.cat([x, reset * h], dim=-1)
        candidate = torch.tanh(self.candidate(candidate_input))
        return update * h + (1.0 - update) * candidate


class BangaloreDCRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, horizon, supports, diffusion_steps=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.cell = GraphGRUCell(input_dim, hidden_dim, supports, diffusion_steps)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon)
        )

    def forward(self, x):
        batch, steps, nodes, _ = x.shape
        h = x.new_zeros(batch, nodes, self.hidden_dim)
        for t in range(steps):
            h = self.cell(x[:, t], h)
        # Forecast changes from the most recent speed. This skip connection
        # gives the model a persistence starting point and lets the graph
        # recurrent state learn the horizon-specific correction.
        residual = self.output(h).permute(0, 2, 1).unsqueeze(-1)
        last_speed = x[:, -1, :, 0].unsqueeze(1).unsqueeze(-1)
        return last_speed + residual
