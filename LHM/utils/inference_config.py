"""Canonical inference configuration without importing GPU model modules."""

import argparse
import os
from pathlib import Path

from omegaconf import OmegaConf

from LHM.utils.model_card import (
    MODEL_CONFIG,
    HuggingFace_MODEL_CARD,
    ModelScope_MODEL_CARD,
)


def parse_inference_configs(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", type=str)
    args, unknown = parser.parse_known_args(argv)
    cli_cfg = OmegaConf.from_cli(unknown)
    cli_cfg.setdefault("export_mesh", None)
    cli_cfg.setdefault("export_video", None)

    cli_name = cli_cfg.get("model_name")
    env_name = os.environ.get("APP_MODEL_NAME")
    explicit_path = cli_cfg.pop("model_path", None)
    if explicit_path is not None and env_name and cli_name and env_name != cli_name:
        raise ValueError(
            "APP_MODEL_NAME conflicts with the explicit model_path model_name"
        )
    model_name = env_name or cli_name
    if not isinstance(model_name, str):
        raise ValueError("model_name is required")
    if (
        model_name not in HuggingFace_MODEL_CARD
        and model_name not in ModelScope_MODEL_CARD
    ):
        raise ValueError(f"Unsupported model_name: {model_name}")
    parts = model_name.split("-")
    cfg_train = OmegaConf.load(MODEL_CONFIG[parts[1]])
    if explicit_path is not None:
        if not isinstance(explicit_path, str) or not Path(explicit_path).is_dir():
            raise FileNotFoundError(
                f"Local model_path directory is missing: {explicit_path}"
            )
        cli_cfg.model_name = str(Path(explicit_path).resolve())
    else:
        from LHM.utils.model_download_utils import AutoModelQuery

        cli_cfg.model_name = AutoModelQuery().query(model_name)

    cfg = OmegaConf.create()
    cfg.source_size = cfg_train.dataset.source_image_res
    cfg.src_head_size = cfg_train.dataset.get("src_head_size", 112)
    cfg.render_size = cfg_train.dataset.render_image.high
    relative_path = os.path.join(
        cfg_train.experiment.parent,
        cfg_train.experiment.child,
        os.path.basename(cli_cfg.model_name).split("_")[-1],
    )
    cfg.save_tmp_dump = os.path.join("exps", "save_tmp", relative_path)
    cfg.image_dump = os.path.join("exps", "images", relative_path)
    cfg.video_dump = os.path.join("exps", "videos", relative_path)
    cfg.mesh_dump = os.path.join("exps", "meshs", relative_path)

    infer_path = os.environ.get("APP_INFER") or args.infer
    if infer_path is not None:
        cfg.merge_with(OmegaConf.load(infer_path))
    cfg.motion_video_read_fps = 6
    cfg.merge_with(cli_cfg)
    cfg.setdefault("logger", "INFO")
    return cfg, cfg_train
