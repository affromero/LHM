"""Real restricted hub loading with a small CPU model at the GPU boundary."""

import importlib
import pickle
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


class ExecutableState:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return eval, (f"__import__('pathlib').Path({str(self.marker)!r}).write_text('executed')",)


@pytest.fixture
def backbone(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2] / "LHM/models/encoders/dinov2"
    name = "security_dinov2"
    package = ModuleType(name)
    package.__path__ = [str(source)]
    models = ModuleType(name + ".models")
    models.__path__ = [str(source / "models")]
    vision = ModuleType(name + ".models.vision_transformer")
    vision.vit_base = lambda **kwargs: torch.nn.Linear(2, 2, bias=False)
    monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, name + ".models", models)
    monkeypatch.setitem(sys.modules, vision.__name__, vision)
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))
    cache = Path(torch.hub.get_dir()) / "checkpoints"
    cache.mkdir(parents=True)
    checkpoint = cache / "dinov2_vitb14_pretrain.pth"
    module = importlib.import_module(name + ".hub.backbones")
    yield module, checkpoint
    for imported in tuple(sys.modules):
        if imported.startswith(name + "."):
            sys.modules.pop(imported)


def test_cached_checkpoint_cannot_execute_pickle_payload(backbone, tmp_path):
    module, checkpoint = backbone
    marker = tmp_path / "executed.txt"
    torch.save({"weight": ExecutableState(marker)}, checkpoint)
    torch.hub.load_state_dict_from_url(
        "https://example.invalid/dinov2_vitb14_pretrain.pth", map_location="cpu"
    )
    assert marker.read_text() == "executed"
    marker.unlink()
    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        module.dinov2_vitb14()
    assert not marker.exists()


def test_tensor_checkpoint_preserves_model_outputs(backbone):
    module, checkpoint = backbone
    torch.save(
        {"weight": torch.tensor([[1., 2.], [3., 4.]]), "mask_token": torch.zeros(1)},
        checkpoint,
    )
    model = module.dinov2_vitb14()
    torch.testing.assert_close(model(torch.ones(1, 2)), torch.tensor([[3., 7.]]))
