import torch

from check_conditioning import compare_predictions


def test_compare_predictions_detects_useful_conditioning():
    target = torch.ones(2, 1, 4, 4)
    conditioned = target + 0.1
    ablated = target + 1.0
    mask = torch.ones_like(target)
    metrics = compare_predictions(conditioned, ablated, target, mask)
    assert metrics["condition_response_rmse_normalized"] > 0
    assert metrics["conditioned_mse_normalized"] < metrics["ablated_mse_normalized"]
    assert metrics["conditioned_skill_vs_ablation"] > 0
