import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import numpy as np

from gnnsr.config import ModelConfig
from gnnsr.graph import GraphArrays, to_heterodata
from gnnsr.model import TwoScaleFloodGNN


@pytest.mark.parametrize("width", [16, 64, 128, 256])
def test_model_produces_one_residual_per_fine_node(width):
    graph = GraphArrays(
        coarse_global_ids=np.array([0, 1]),
        fine_global_ids=np.array([0, 1, 2]),
        target_mask=np.array([True, True, False]),
        coarse_static=np.zeros((2, 2), dtype=np.float32),
        fine_static=np.zeros((3, 2), dtype=np.float32),
        coarse_edge_index=np.array([[0, 1], [1, 0]]),
        coarse_edge_attr=np.zeros((2, 5), dtype=np.float32),
        cross_edge_index=np.array([[0, 0, 1], [0, 1, 2]]),
        cross_edge_attr=np.zeros((3, 5), dtype=np.float32),
        fine_edge_index=np.array([[0, 1, 1, 2], [1, 0, 2, 1]]),
        fine_edge_attr=np.zeros((4, 5), dtype=np.float32),
        fine_parent_local=np.array([0, 0, 1]),
        fallback_parent_mask=np.zeros(3, dtype=bool),
    )
    data = to_heterodata(graph)
    model = TwoScaleFloodGNN(ModelConfig(hidden_dim=width))
    output = model(data, torch.zeros((2, 4)), torch.zeros((3, 4)))
    assert output.shape == (3,)
    output.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
