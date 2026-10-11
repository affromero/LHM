"""CPU configuration tests using the shipped training YAML and real directories."""

from pathlib import Path

import pytest

from LHM.utils.inference_config import parse_inference_configs


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    monkeypatch.delenv("APP_MODEL_NAME", raising=False)
    monkeypatch.delenv("APP_INFER", raising=False)
    monkeypatch.chdir(Path(__file__).resolve().parents[2])


def test_explicit_snapshot_preserves_500m_configuration_without_downloads(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    cfg, train = parse_inference_configs(
        [
            "model_name=LHM-500M",
            f"model_path={snapshot}",
            "export_mesh=True",
        ]
    )
    assert cfg.model_name == str(snapshot.resolve())
    assert cfg.source_size == train.dataset.source_image_res == 1024
    assert cfg.render_size == train.dataset.render_image.high == 512
    assert cfg.src_head_size == 112
    assert cfg.export_mesh is True
    assert cfg.export_video is None
    assert Path(cfg.mesh_dump).parts[:2] == ("exps", "meshs")


@pytest.mark.parametrize("kind", ["missing", "regular-file"])
def test_explicit_snapshot_requires_a_directory(tmp_path, kind):
    snapshot = tmp_path / "snapshot"
    if kind == "regular-file":
        snapshot.write_text("not a model directory")
    with pytest.raises(FileNotFoundError, match="model_path"):
        parse_inference_configs(["model_name=LHM-500M", f"model_path={snapshot}"])


def test_environment_cannot_pair_1b_configuration_with_500m_snapshot(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("APP_MODEL_NAME", "LHM-1B")
    with pytest.raises(ValueError, match="conflicts"):
        parse_inference_configs(["model_name=LHM-500M", f"model_path={tmp_path}"])


def test_matching_environment_keeps_explicit_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_MODEL_NAME", "LHM-500M")
    cfg, train = parse_inference_configs(
        ["model_name=LHM-500M", f"model_path={tmp_path}"]
    )
    assert cfg.model_name == str(tmp_path.resolve())
    assert cfg.source_size == train.dataset.source_image_res


@pytest.mark.parametrize(
    "selector",
    [None, "unknown", "OTHER-500M", "LHM-unknown", "LHM-500M-typo", "LHM-MINI-HF"],
)
def test_unsupported_model_identifier_fails_before_resolution(tmp_path, selector):
    argv = [f"model_path={tmp_path}"]
    if selector is not None:
        argv.append(f"model_name={selector}")
    with pytest.raises(ValueError, match="model_name"):
        parse_inference_configs(argv)


def test_cli_overrides_inference_yaml_and_preserves_training_configuration(tmp_path):
    inference = tmp_path / "inference.yaml"
    inference.write_text(
        "source_size: 640\nrender_size: 256\nexport_mesh: false\nlogger: DEBUG\n"
    )
    cfg, train = parse_inference_configs(
        [
            "--infer",
            str(inference),
            "model_name=LHM-500M",
            f"model_path={tmp_path}",
            "export_mesh=True",
            "render_size=384",
        ]
    )
    assert cfg.source_size == 640
    assert cfg.render_size == 384
    assert cfg.export_mesh is True
    assert cfg.logger == "DEBUG"
    assert train.dataset.source_image_res == 1024


def test_environment_inference_yaml_retains_existing_precedence(tmp_path, monkeypatch):
    inference = tmp_path / "inference.yaml"
    inference.write_text("source_size: 640\n")
    monkeypatch.setenv("APP_INFER", str(inference))
    cfg, _ = parse_inference_configs(
        [
            "--infer",
            str(tmp_path / "missing.yaml"),
            "model_name=LHM-500M",
            f"model_path={tmp_path}",
        ]
    )
    assert cfg.source_size == 640
