"""Safety and override tests for the GAN ablation config helper."""

import argparse
import importlib.util
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/create_gan_ablation_config.py"
SPEC = importlib.util.spec_from_file_location("create_gan_ablation_config", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def arguments(tmp_path, **updates):
    values = {
        "name": "gan_test_spectral",
        "parent": Path(__file__).resolve().parents[1] / "configs/gan_sr_v2.json",
        "note": "unit-test ablation",
        "content": None,
        "gradient": None,
        "spectral": 0.4,
        "feature_matching": None,
        "adversarial": None,
        "adversarial_start_step": None,
        "max_steps": 10,
        "destination": tmp_path / "config.json",
    }
    values.update(updates)
    return argparse.Namespace(**values)


def test_build_config_changes_only_requested_loss_and_isolates_outputs(tmp_path):
    config, destination = MODULE.build_config(arguments(tmp_path))
    assert destination == (tmp_path / "config.json").resolve()
    assert config["lambda_spectral"] == 0.4
    assert config["lambda_content"] == 5.0
    assert config["max_steps"] == 10
    assert config["output_dir"] == "runs/gan_test_spectral"
    assert config["parent_config"] == "configs/gan_sr_v2.json"


def test_invalid_name_and_negative_weight_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="name"):
        MODULE.build_config(arguments(tmp_path, name="Bad Name"))
    with pytest.raises(ValueError, match="lambda_spectral"):
        MODULE.build_config(arguments(tmp_path, spectral=-1.0))
