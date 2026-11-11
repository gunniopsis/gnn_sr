import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from gnnsr.train import _restore_random_state, resolve_device


def test_explicit_cpu_device_is_accepted():
    assert resolve_device("cpu") == torch.device("cpu")


def test_explicit_cuda_device_fails_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_device("cuda")


def test_restore_cuda_random_state_moves_checkpoint_tensors_to_cpu(monkeypatch):
    expected_state = torch.tensor([1, 2, 3], dtype=torch.uint8)

    class SimulatedCudaState:
        def cpu(self):
            return expected_state

    restored = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: restored.extend(states))

    _restore_random_state({"cuda_random_state_all": [SimulatedCudaState()]})

    assert restored == [expected_state]
