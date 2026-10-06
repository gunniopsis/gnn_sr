from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig
from .graph import CROSS_RELATION, HEC_RELATION, TUFLOW_RELATION


class CompactMLP(nn.Module):
    """Two affine transforms with one activation and optional output LayerNorm."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        negative_slope: float,
        layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class EdgeConditionedMPNNLayer(nn.Module):
    """MeshGraphNet-style transient edge messages with sum aggregation."""

    def __init__(self, hidden_dim: int, edge_dim: int, negative_slope: float, dropout: float):
        super().__init__()
        self.edge_mlp = CompactMLP(
            2 * hidden_dim + edge_dim,
            hidden_dim,
            hidden_dim,
            negative_slope,
            layer_norm=True,
            dropout=dropout,
        )
        self.node_mlp = CompactMLP(
            2 * hidden_dim,
            hidden_dim,
            hidden_dim,
            negative_slope,
            layer_norm=True,
            dropout=dropout,
        )

    def forward(
        self, node_state: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        source, destination = edge_index
        message_input = torch.cat((node_state[source], node_state[destination], edge_attr), dim=-1)
        messages = self.edge_mlp(message_input)
        aggregated = torch.zeros_like(node_state)
        aggregated.index_add_(0, destination, messages)
        update = self.node_mlp(torch.cat((node_state, aggregated), dim=-1))
        return node_state + update


class CrossScaleFusion(nn.Module):
    """One directed HEC-to-TUFLOW message followed by residual concatenation fusion."""

    def __init__(self, hidden_dim: int, edge_dim: int, negative_slope: float, dropout: float):
        super().__init__()
        self.message_mlp = CompactMLP(
            2 * hidden_dim + edge_dim,
            hidden_dim,
            hidden_dim,
            negative_slope,
            layer_norm=True,
            dropout=dropout,
        )
        self.fusion_mlp = CompactMLP(
            2 * hidden_dim,
            hidden_dim,
            hidden_dim,
            negative_slope,
            layer_norm=True,
            dropout=dropout,
        )

    def forward(
        self,
        coarse_state: torch.Tensor,
        fine_state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        source, destination = edge_index
        messages = self.message_mlp(
            torch.cat((coarse_state[source], fine_state[destination], edge_attr), dim=-1)
        )
        aggregated = torch.zeros_like(fine_state)
        aggregated.index_add_(0, destination, messages)
        return fine_state + self.fusion_mlp(torch.cat((fine_state, aggregated), dim=-1))


class TwoScaleFloodGNN(nn.Module):
    """Residual WSE super-resolution model with separate coarse and fine processors."""

    def __init__(self, config: ModelConfig, node_input_dim: int = 4, edge_dim: int = 5):
        super().__init__()
        hidden = config.hidden_dim
        slope = config.leaky_relu_slope
        dropout = config.dropout
        self.hec_encoder = CompactMLP(
            node_input_dim, hidden, hidden, slope, layer_norm=True, dropout=dropout
        )
        self.fine_encoder = CompactMLP(
            node_input_dim, hidden, hidden, slope, layer_norm=True, dropout=dropout
        )
        self.hec_processor = nn.ModuleList(
            EdgeConditionedMPNNLayer(hidden, edge_dim, slope, dropout)
            for _ in range(config.coarse_layers)
        )
        self.cross_fusion = CrossScaleFusion(hidden, edge_dim, slope, dropout)
        self.fine_processor = nn.ModuleList(
            EdgeConditionedMPNNLayer(hidden, edge_dim, slope, dropout)
            for _ in range(config.fine_layers)
        )
        self.decoder = CompactMLP(hidden, 1, hidden, slope, layer_norm=False, dropout=dropout)

    def forward(self, graph, hec_features: torch.Tensor, fine_features: torch.Tensor):
        hec_state = self.hec_encoder(hec_features)
        fine_state = self.fine_encoder(fine_features)
        coarse_relation = graph[HEC_RELATION]
        for layer in self.hec_processor:
            hec_state = layer(hec_state, coarse_relation.edge_index, coarse_relation.edge_attr)
        cross_relation = graph[CROSS_RELATION]
        fine_state = self.cross_fusion(
            hec_state,
            fine_state,
            cross_relation.edge_index,
            cross_relation.edge_attr,
        )
        fine_relation = graph[TUFLOW_RELATION]
        for layer in self.fine_processor:
            fine_state = layer(fine_state, fine_relation.edge_index, fine_relation.edge_attr)
        return self.decoder(fine_state).squeeze(-1)


def physical_outputs(
    raw_wse: torch.Tensor, terrain: torch.Tensor, dry_threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_depth = raw_wse - terrain
    depth = torch.where(raw_depth >= dry_threshold, raw_depth, torch.zeros_like(raw_depth))
    return terrain + depth, depth
