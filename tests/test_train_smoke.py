import json
from pathlib import Path

import pytest
import torch

from conftest import make_config


@pytest.fixture(scope="module")
def smoke_runs(synthetic_root):
    from train_flow import train as train_flow
    from train_flow_ar import train as train_ar
    from train_gan import train as train_gan

    common = dict(base_channels=4, levels=2, attention=False, attention_heads=1,
                  preview_samples=1, validation_samples=1,
                  preview_sampler_steps=1, validation_sampler_steps=1)
    flow = make_config(synthetic_root, "smoke_flow_test", **common)
    ar = make_config(synthetic_root, "smoke_ar_test", model_kind="autoregressive",
                     lag_base_channels=2, lag_dropout=0.0, lag_path_dropout=0.0,
                     lag_guidance_scale=0.25, enforce_coarse_consistency=True,
                     rollout_days=2, rollout_sampler_steps=1,
                     rollout_netcdf_every=1, horizon=1, **common)
    gan = make_config(synthetic_root, "smoke_gan_test", model_kind="gan",
                      noise_channels=2, generator_residual=True,
                      discriminator_base_channels=4, discriminator_levels=2,
                      discriminator_learning_rate=1e-4, critic_steps=1,
                      lambda_content=10.0, lambda_adversarial=1.0,
                      adversarial_start_step=0, **common)
    wgan = make_config(
        synthetic_root,
        "smoke_wgan_gp_test",
        model_kind="gan",
        noise_channels=2,
        generator_residual=True,
        discriminator_base_channels=4,
        discriminator_levels=2,
        discriminator_learning_rate=1e-4,
        critic_steps=3,
        adversarial_objective="wasserstein_gp",
        gradient_penalty_weight=10.0,
        gradient_penalty_target=1.0,
        lambda_content=10.0,
        lambda_adversarial=1.0,
        adversarial_start_step=0,
        **common,
    )
    statuses = {
        "flow": train_flow(flow, smoke_steps=3, device_name="cpu"),
        "ar": train_ar(ar, smoke_steps=3, device_name="cpu"),
        "gan": train_gan(gan, smoke_steps=3, device_name="cpu"),
        "wgan": train_gan(wgan, smoke_steps=3, device_name="cpu"),
    }
    return {
        "statuses": statuses,
        "configs": {"flow": flow, "ar": ar, "gan": gan, "wgan": wgan},
    }


@pytest.mark.parametrize("name", ["flow", "ar", "gan", "wgan"])
def test_smoke_end_to_end(smoke_runs, name):
    status = smoke_runs["statuses"][name]; config = smoke_runs["configs"][name]
    output = Path(config["smoke_output_dir"])
    assert status["status"] == "passed" and status["step"] == 3
    assert (output / "status.json").is_file() and (output / "checkpoint.pt").is_file()
    assert list((output / "predictions").glob("*.png"))
    assert list((output / "netcdf").glob("*.nc"))
    history = json.loads((output / "training_history.json").read_text())["history"]
    assert history and all(all(not isinstance(value, float) or value == value for value in record.values()) for record in history)
    if name in ("gan", "wgan"):
        assert status["adversarial_exercised"] is True
        assert all("critic_gradient_norm" in record for record in history)
        assert all("discriminator_learning_rate" in record for record in history)
        checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=False)
        assert "scheduler_generator" in checkpoint
        assert "scheduler_discriminator" in checkpoint
    if name == "wgan":
        assert status["adversarial_objective"] == "wasserstein_gp"
        assert status["critic_steps"] == 3
        assert all(record["critic_steps"] == 3 for record in history)
        assert all("gradient_penalty" in record for record in history)
        assert all("gradient_penalty_norm" in record for record in history)
    if name == "ar":
        assert list((output / "netcdf").glob("rollout*.nc"))
        assert list((output / "predictions").glob("rollout_skill*.png"))


def test_resume_continues(smoke_runs):
    from train_flow import train
    config = smoke_runs["configs"]["flow"]
    status = train(config, smoke_steps=4, device_name="cpu")
    assert status["status"] == "passed" and status["step"] == 4
    history = json.loads((Path(config["smoke_output_dir"]) / "training_history.json").read_text())["history"]
    assert [record["step"] for record in history] == [1, 2, 3, 4]


def test_gan_external_fork_continues_with_configured_learning_rate(smoke_runs):
    from train_gan import train

    parent = smoke_runs["configs"]["gan"]
    config = dict(parent)
    config["name"] = "smoke_gan_external_fork"
    config["smoke_output_dir"] = str(
        Path(parent["smoke_output_dir"]).parent / "smoke_gan_external_fork"
    )
    config["resume_from"] = str(Path(parent["smoke_output_dir"]) / "checkpoint.pt")
    config["continuation_learning_rate"] = 5.0e-5
    config["scheduler_kind"] = "constant"
    config["warmup_steps"] = 0
    config["min_learning_rate_factor"] = 1.0

    status = train(config, smoke_steps=4, device_name="cpu")
    history = json.loads(
        (Path(config["smoke_output_dir"]) / "training_history.json").read_text()
    )["history"]
    assert status["status"] == "passed" and status["step"] == 4
    assert [record["step"] for record in history] == [1, 2, 3, 4]
    assert history[-1]["learning_rate"] == pytest.approx(5.0e-5)
    assert history[-1]["discriminator_learning_rate"] == pytest.approx(5.0e-5)
