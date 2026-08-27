import json
from pathlib import Path

import pytest

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
    statuses = {
        "flow": train_flow(flow, smoke_steps=3, device_name="cpu"),
        "ar": train_ar(ar, smoke_steps=3, device_name="cpu"),
        "gan": train_gan(gan, smoke_steps=3, device_name="cpu"),
    }
    return {"statuses": statuses, "configs": {"flow": flow, "ar": ar, "gan": gan}}


@pytest.mark.parametrize("name", ["flow", "ar", "gan"])
def test_smoke_end_to_end(smoke_runs, name):
    status = smoke_runs["statuses"][name]; config = smoke_runs["configs"][name]
    output = Path(config["smoke_output_dir"])
    assert status["status"] == "passed" and status["step"] == 3
    assert (output / "status.json").is_file() and (output / "checkpoint.pt").is_file()
    assert list((output / "predictions").glob("*.png"))
    assert list((output / "netcdf").glob("*.nc"))
    history = json.loads((output / "training_history.json").read_text())["history"]
    assert history and all(all(not isinstance(value, float) or value == value for value in record.values()) for record in history)
    if name == "gan": assert status["adversarial_exercised"] is True
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
