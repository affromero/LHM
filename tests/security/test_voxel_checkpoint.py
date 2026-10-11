"""Exercise canonical voxel cache processing with CPU geometry boundaries."""

import ast
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


class ExecutableVoxel:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return eval, (
            f"__import__('pathlib').Path({str(self.marker)!r}).write_text('executed')",
        )


@pytest.fixture
def voxel_cache(tmp_path, monkeypatch):
    source = (
        Path(__file__).resolve().parents[2]
        / "LHM/models/rendering/smpl_x_voxel_dense_sampling.py"
    )
    module = ast.parse(source.read_text(), filename=str(source))
    model = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "SMPLXVoxelMeshModel"
    )
    method = next(
        node
        for node in model.body
        if isinstance(node, ast.FunctionDef) and node.name == "voxel_skinning_init"
    )
    namespace = {"torch": torch, "np": np, "os": os, "avaliable_device": lambda: "cpu"}
    exec(  # noqa: S102 - Compile the tracked canonical method without GPU imports.
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    monkeypatch.delenv("TORCH_FORCE_WEIGHTS_ONLY_LOAD", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    cache = tmp_path / "pretrained_models/voxel_grid/voxel_4.pth"
    cache.parent.mkdir(parents=True)
    vertices = torch.tensor([[[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]]])
    flat = torch.arange(64, dtype=torch.float32).reshape(32, 2)
    instance = SimpleNamespace(
        smplx_layer=SimpleNamespace(lbs_weights=torch.ones(2, 2)),
        smpl_x=SimpleNamespace(shape_param_dim=2),
        get_neutral_pose_human=lambda **kwargs: (None, vertices, None),
        voxel_smooth_register=lambda *args, **kwargs: flat.clone(),
    )
    return namespace["voxel_skinning_init"], instance, cache, flat


def test_generated_voxel_cache_reloads_with_identical_values_and_bounds(voxel_cache):
    initialize, instance, cache, flat = voxel_cache
    generated, bounds = initialize(instance, voxel_size=4)
    assert cache.is_file()
    expected = flat.view(4, 4, 2, 2).permute(3, 2, 1, 0)
    torch.testing.assert_close(generated, expected)
    torch.testing.assert_close(
        bounds, torch.tensor([[-1.05, 1.05], [-2.1, 2.1], [-3.15, 3.15]])
    )
    reloaded, reloaded_bounds = initialize(instance, voxel_size=4)
    torch.testing.assert_close(reloaded, generated)
    torch.testing.assert_close(reloaded_bounds, bounds)


def test_voxel_cache_rejects_executable_pickle(voxel_cache, tmp_path):
    initialize, instance, cache, _ = voxel_cache
    marker = tmp_path / "executed.txt"
    torch.save(ExecutableVoxel(marker), cache)
    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        initialize(instance, voxel_size=4)
    assert not marker.exists()
