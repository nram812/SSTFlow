#!/usr/bin/env python3
"""Append the validated AR/NOAA analysis to both comparison notebooks.

The legacy exploratory cells are retained for provenance.  Cells carrying the
``extended_evaluation_v1`` tag are replaced atomically on each invocation, so
the update is idempotent and the reported numbers always come from the latest
validated products.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/extended_evaluation/extended_model_comparison.json"
CLIMATE_REPORT = ROOT / "reports/climate_change/requested_models_climate_change_evaluation.json"
NOTEBOOKS = (ROOT / "testset_evaluation.ipynb", ROOT / "model_intercomparison.ipynb")
TAG = "extended_evaluation_v1"
CLIMATE_TAG = "climate_change_evaluation_v2"


def markdown_cell(text: str, tag: str = TAG) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {"tags": [tag]},
        "source": [line + "\n" for line in text.rstrip().splitlines()],
    }


def code_cell(source: str, text_output: str = "", tag: str = TAG) -> dict:
    outputs = []
    if text_output:
        outputs = [{"name": "stdout", "output_type": "stream", "text": [text_output]}]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": [tag]},
        "outputs": outputs,
        "source": [line + "\n" for line in source.rstrip().splitlines()],
    }


def _table(ar_entries: dict | None, gan_entries: dict | None = None) -> str:
    entries = {**(ar_entries or {}), **(gan_entries or {})}
    if not entries:
        return "_Production products are still pending._"
    lines = [
        "| Model | RMSE (°C) | Bias (°C) | Evolution ratio | EAC r |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in entries.items():
        lines.append(
            f"| {name} | {metrics['rmse_c']:.3f} | {metrics['bias_c']:.3f} | "
            f"{metrics['evolution_ratio']:.2f} | {metrics['point_correlations']['EAC']:.3f} |"
        )
    return "\n".join(lines)


def cells(report: dict) -> list[dict]:
    noaa = report.get("noaa")
    noaa_summary = "_Production products are still pending._"
    if noaa:
        noaa_summary = (
            f"The final 0.05° model has daily test RMSE **{noaa['rmse_c']:.3f} °C**, "
            f"1-pixel coastal RMSE **{noaa['coast_1px_rmse_c']:.3f} °C**, and "
            f">8-pixel interior RMSE **{noaa['interior_rmse_c']:.3f} °C**. "
            f"The downscaled ACCESS-CM2 2080s−1980s domain-mean signal is "
            f"**{noaa['access_warming_mean_c']:.3f} °C**."
        )
    narrative = f"""
---

# Validated extension: temporal memory and 0.05° satellite transfer

This section was generated from independently validated final NetCDF products.
It intentionally does **not** concatenate the NOAA-transfer output with the
0.1° OFAM model registry: the target product, grid, ocean mask, and test years
are different. The autoregressive comparison is also time-aligned to the common
364 generated days in 2011 and has no truth resets.

## 2011 autoregressive comparison

{_table(report.get('ar'), report.get('gan'))}

Interpretation: the useful autoregressive model must improve temporal coherence
without allowing yesterday's state to override today's coarse SST. The residual
memory experiment freezes the successful current-SST backbone, conditions only
on yesterday's within-block anomaly, caps its FiLM contribution, and projects
every generated day to the current coarse ocean-block means. Common latent noise
couples stochastic texture through time without changing daily marginals.

The GAN rows use the same 364 dates and truth, but they are direct conditional
samples rather than free-running rollouts. Their evolution ratio therefore
measures frame-to-frame variability (including stochastic texture), not memory
stability. This is why RMSE, evolution ratio and regional correlation must be
read together rather than collapsed into one ranking.

The accompanying coarse-balanced Flow-AR animation uses the validated,
truth-reset-free 2011 rollout and a fixed SST colour scale. The Australian domain and the
Perth/southwest, Ningaloo/northwest shelf, and East Australian Current insets
show whether memory follows both today's coarse boundary and OFAM truth.

## NOAA 0.05° transfer and climate deployment

{noaa_summary}

Coastline metrics are reported separately because the failed 2×2 projection
experiment mixed incompatible NOAA and OFAM masks. The replacement predicts the
1024×1024 NOAA field directly and evaluates only on the fixed NOAA ocean mask.
"""
    figure_code = """from pathlib import Path
from IPython.display import Image, Video, display

extended_figure_dir = Path(BASE_DIR if 'BASE_DIR' in globals() else '.') / 'figures/extended_evaluation'
for filename in (
    'autoregressive_2011_skill_and_eac_timeseries.png',
    'autoregressive_2011_mean_bias_maps.png',
    'noaa_5km_test_climatology_and_bias.png',
    'noaa_5km_access_cm2_warming_signal.png',
):
    path = extended_figure_dir / filename
    if path.is_file():
        display(Image(filename=str(path)))

animation_dir = Path(BASE_DIR if 'BASE_DIR' in globals() else '.') / 'figures'
preview = animation_dir / 'flow_ar_legacy_coarse_balanced_sst_preview.gif'
video = animation_dir / 'flow_ar_legacy_coarse_balanced_sst_comparison_300frames.mp4'
if preview.is_file():
    display(Image(filename=str(preview)))
if video.is_file():
    display(Video(filename=str(video), embed=False, html_attributes='controls loop'))
"""
    provenance_code = """import json
from pathlib import Path

report_path = Path(BASE_DIR if 'BASE_DIR' in globals() else '.') / 'reports/extended_evaluation/extended_model_comparison.json'
extended_report = json.loads(report_path.read_text())
extended_report
"""
    output = json.dumps(report, indent=2) + "\n"
    return [markdown_cell(narrative), code_cell(figure_code), code_cell(provenance_code, output)]


def climate_cells(report: dict) -> list[dict]:
    perfect = report["perfect_framework"]
    perfect_models = report.get("perfect_framework_models", {"flow_sr_combined": perfect})
    perfect_signal = perfect["climate_signal"]
    access = report["access_cm2"]
    combined = access["flow_sr_combined"]["annual_signal_preservation"]
    noaa = access["flow_sr_noaa_5km"]["annual_signal_preservation"]
    perfect_labels = {
        "flow_sr_combined": "Flow-SR, historical + future",
        "gan_v2_combined": "GAN-v2, historical + future",
        "gan_v2b_combined": "GAN-v2b, historical + future",
        "gan_v3_combined": "GAN-v3, historical + future",
    }
    access_labels = {
        "flow_sr_historical": "Flow-SR, historical only",
        "flow_sr_combined": "Flow-SR, historical + future",
        "gan_v2_historical": "GAN-v2, historical only",
        "gan_v2_combined": "GAN-v2, historical + future",
        "gan_v2b_historical": "GAN-v2b, historical only",
        "gan_v2b_combined": "GAN-v2b, historical + future",
        "gan_v3_historical": "GAN-v3, historical only",
        "gan_v3_combined": "GAN-v3, historical + future",
        "flow_sr_noaa_5km": "NOAA 0.05° Flow-SR transfer",
    }
    rows = [
        ("Perfect OFAM", perfect_labels[key], "2011–2014", "2098–2101", value["climate_signal"])
        for key, value in perfect_models.items()
    ]
    rows.extend(
        (
            "Imperfect ACCESS-CM2",
            access_labels.get(key, key),
            "1980–1989",
            "2080–2089",
            value["annual_signal_preservation"],
        )
        for key, value in access.items()
    )
    table = [
        "| Framework | Model | Historical | Future | Reference ΔT | Predicted ΔT | Ratio | Signal RMSE | Spatial r |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for framework, model, historical, future, metrics in rows:
        table.append(
            f"| {framework} | {model} | {historical} | {future} | "
            f"{metrics['target_mean_c']:.3f} °C | {metrics['prediction_mean_c']:.3f} °C | "
            f"{metrics['mean_signal_ratio']:.3f} | {metrics['rmse_c']:.3f} °C | "
            f"{metrics['spatial_correlation']:.3f} |"
        )
    historical = perfect["historical_skill"]
    future = perfect["future_skill"]
    narrative = f"""
---

# Validated climate-change evaluation

Two questions are deliberately separated. In the **perfect OFAM framework**,
the generated fine grid can be compared with paired fine-resolution truth. In
the **imperfect ACCESS-CM2 deployment**, no fine-resolution truth exists, so
the test is whether the generated climate-change field preserves the supplied
32×32 ACCESS-CM2 signal after exact mask-aware re-coarsening.

{chr(10).join(table)}

The combined model's daily RMSE is **{historical['rmse_c']:.3f} °C** on the
2011–2014 OFAM test and **{future['rmse_c']:.3f} °C** on the 2098–2101 test.
Its predicted OFAM mean warming differs from truth by only
**{perfect_signal['mean_bias_c']:+.3f} °C**. It also retains
**{combined['mean_signal_ratio'] * 100:.1f}%** of the ACCESS-CM2 mean driving
signal. By contrast, the NOAA decoder fine-tune retains
**{noaa['mean_signal_ratio'] * 100:.1f}%** annually and has pronounced seasonal
damping. This is evidence of climate-response forgetting during
observation-only decoder fine-tuning, not evidence that the ACCESS deployment
is inaccurate against an unavailable fine-resolution truth.

All scalar map statistics are ocean-only and cosine-latitude weighted.
Perkins skill score and SST extreme diagnostics are proposed additions and are
not silently substituted by the existing marine-heatwave proxy.
"""
    figure_code = """from pathlib import Path
from IPython.display import Image, display

climate_figure_dir = Path(BASE_DIR if 'BASE_DIR' in globals() else '.') / 'figures/climate_change'
for filename in (
    'flow_sr_combined_ofam_climate_change_signal.png',
    'flow_sr_combined_ofam_signal_error_and_seasons.png',
    'access_cm2_signal_preservation_requested_models.png',
    'access_cm2_signal_preservation_seasonal.png',
    'ofam_combined_flow_gan_climate_signal_comparison.png',
    'access_cm2_flow_gan_training_period_comparison.png',
    'access_cm2_flow_gan_signal_ratio_comparison.png',
):
    path = climate_figure_dir / filename
    if path.is_file():
        display(Image(filename=str(path)))
"""
    provenance_code = """import json
from pathlib import Path

climate_report_path = Path(BASE_DIR if 'BASE_DIR' in globals() else '.') / 'reports/climate_change/requested_models_climate_change_evaluation.json'
climate_change_report = json.loads(climate_report_path.read_text())
climate_change_report
"""
    return [
        markdown_cell(narrative, CLIMATE_TAG),
        code_cell(figure_code, tag=CLIMATE_TAG),
        code_cell(provenance_code, json.dumps(report, indent=2) + "\n", CLIMATE_TAG),
    ]


def update(path: Path, additions: list[dict]) -> None:
    notebook = json.loads(path.read_text())
    retained = [
        cell for cell in notebook["cells"]
        if not {TAG, CLIMATE_TAG}.intersection(cell.get("metadata", {}).get("tags", []))
    ]
    if path.name == "model_intercomparison.ipynb" and not any(
        "Exploratory provenance" in "".join(cell.get("source", []))
        for cell in retained[:2]
    ):
        retained.insert(0, markdown_cell(
            "# Model intercomparison\n\n"
            "## Exploratory provenance\n\n"
            "The original exploratory cells are retained below for provenance. "
            "The authoritative, reproducible extension is the final section."
        ))
        # The provenance header should survive future tag-based replacement.
        retained[0]["metadata"] = {}
    notebook["cells"] = retained + additions
    temporary = path.with_suffix(".ipynb.partial")
    temporary.write_text(json.dumps(notebook, indent=1) + "\n")
    os.replace(temporary, path)


def main() -> None:
    if not REPORT.is_file():
        raise FileNotFoundError(REPORT)
    report = json.loads(REPORT.read_text())
    climate_report = json.loads(CLIMATE_REPORT.read_text()) if CLIMATE_REPORT.is_file() else None
    additions = cells(report)
    if climate_report is not None:
        additions.extend(climate_cells(climate_report))
    for notebook in NOTEBOOKS:
        update(notebook, additions)
        print(f"updated {notebook}")


if __name__ == "__main__":
    main()
