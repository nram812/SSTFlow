#!/usr/bin/env python3
"""Plot and animate the flow-SR SST evaluation at full and regional scales.

The layout is deliberately reviewed as a static PNG before animation is
rendered.  It compares the authoritative coarse input, the stochastic
super-resolution prediction, and the native OFAM target using one fixed SST
colour scale.  The mask is taken directly from each stored field; no coastline
data or map reprojection is introduced.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import xarray as xr


DEFAULT_INPUT = Path(
    "/esi/project/niwa03712/rampaln/PUBLICATIONS/2026/SSTDownscaling/"
    "runs/flow_sr/evaluation/full_test_samples.nc"
)
DEFAULT_GAN_INPUT = Path(
    "/esi/project/niwa03712/rampaln/PUBLICATIONS/2026/SSTDownscaling/"
    "runs/gan_sr_v3_hard_consistency/evaluation/full_test_samples.nc"
)
DEFAULT_OUTPUT_DIR = Path(
    "/esi/project/niwa03712/rampaln/PUBLICATIONS/2026/SSTDownscaling/figures"
)
FULL_EXTENT = (107.35, 158.45, -45.5, -8.0)
SST_LIMITS = (5.0, 34.0)
SST_CMAP = LinearSegmentedColormap.from_list(
    "sst",
    (
        "#233b70",
        "#276b9b",
        "#2ba7b0",
        "#55c667",
        "#b8de4f",
        "#fde725",
        "#f5a142",
        "#df513a",
        "#8b1d3d",
    ),
    N=256,
)


@dataclass(frozen=True)
class Region:
    name: str
    short_name: str
    extent: tuple[float, float, float, float]
    colour: str
    marker: tuple[float, float] | None = None
    marker_label: str | None = None


REGIONS = (
    Region(
        "Perth and southwest coast",
        "Perth / southwest",
        (108.5, 122.5, -40.5, -26.5),
        "#f28e2b",
        (115.86, -31.95),
        "Perth",
    ),
    Region(
        "Ningaloo and northwest shelf",
        "Ningaloo / northwest",
        (107.5, 121.5, -30.0, -16.0),
        "#8f63b8",
        (114.13, -21.93),
        "Ningaloo",
    ),
    Region(
        "East Australian Current",
        "East Australian Current",
        (144.5, 158.5, -41.5, -27.5),
        "#00a6a6",
        (153.0, -30.0),
        "EAC",
    ),
)

SOURCES = (
    ("sst_coarse", "Coarse input", "1.6° block mean"),
    ("sst_generated", "Flow prediction", "0.1° generated SST"),
    ("sst_target", "OFAM ground truth", "0.1° target SST"),
)


def coordinate_edges(values: np.ndarray) -> np.ndarray:
    """Convert monotonic cell centres to edges for exact pcolormesh geometry."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.diff(values) > 0):
        raise ValueError("Coordinates must be one-dimensional and increasing")
    middle = 0.5 * (values[:-1] + values[1:])
    return np.concatenate(
        ([values[0] - 0.5 * (values[1] - values[0])], middle,
         [values[-1] + 0.5 * (values[-1] - values[-2])])
    )


def coordinate_names(variable: str) -> tuple[str, str]:
    return ("lat_lr", "lon_lr") if variable == "sst_coarse" else ("lat", "lon")


def subset_for_extent(
    values: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    extent: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    west, east, south, north = extent
    iy = np.flatnonzero((lat >= south) & (lat <= north))
    ix = np.flatnonzero((lon >= west) & (lon <= east))
    if not iy.size or not ix.size:
        raise ValueError(f"Extent {extent} does not intersect the data grid")
    return values[np.ix_(iy, ix)], lat[iy], lon[ix]


def validate_dataset(dataset: xr.Dataset) -> None:
    required = {"sst_coarse", "sst_generated", "sst_target"}
    missing = required.difference(dataset.data_vars)
    if missing:
        raise ValueError(f"Missing variables: {sorted(missing)}")
    if dataset.sizes.get("time", 0) < 1:
        raise ValueError("Dataset has no time steps")
    for coordinate in ("lat", "lon", "lat_lr", "lon_lr"):
        values = dataset[coordinate].values
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError(f"Invalid coordinate {coordinate}")
        if not np.all(np.diff(values) > 0):
            raise ValueError(f"Coordinate {coordinate} is not increasing")
    for variable in required:
        units = dataset[variable].attrs.get("units", dataset.attrs.get("units"))
        if units not in ("degrees C", "degree_Celsius", "degrees_C"):
            raise ValueError(f"Unexpected units for {variable}: {units!r}")
        sample = dataset[variable].isel(time=0).values
        finite = sample[np.isfinite(sample)]
        if not finite.size or finite.min() < -3 or finite.max() > 45:
            raise ValueError(f"Implausible SST range for {variable}")


def decorate_axis(
    axis: plt.Axes,
    extent: tuple[float, float, float, float],
    *,
    labels: bool,
) -> None:
    west, east, south, north = extent
    axis.set_xlim(west, east)
    axis.set_ylim(south, north)
    axis.set_facecolor("#edf0f2")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(color="#5b6770", linewidth=0.35, alpha=0.24)
    for spine in axis.spines.values():
        spine.set_color("#68737d")
        spine.set_linewidth(0.65)
    axis.tick_params(labelsize=7, length=2.5, colors="#35414b")
    if labels:
        axis.set_xlabel("Longitude (°E)", fontsize=8)
        axis.set_ylabel("Latitude (°N)", fontsize=8)
    else:
        axis.set_xticklabels([])
        axis.set_yticklabels([])


def draw_field(
    axis: plt.Axes,
    values: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    extent: tuple[float, float, float, float],
    *,
    labels: bool,
):
    mesh = axis.pcolormesh(
        coordinate_edges(lon),
        coordinate_edges(lat),
        np.ma.masked_invalid(values),
        cmap=SST_CMAP,
        vmin=SST_LIMITS[0],
        vmax=SST_LIMITS[1],
        shading="flat",
        rasterized=True,
    )
    decorate_axis(axis, extent, labels=labels)
    return mesh


class SSTComparisonFigure:
    """Reusable artists for the static proof and the eventual animation."""

    def __init__(self, dataset: xr.Dataset, frame: int):
        self.dataset = dataset
        self.frame = int(frame)
        self.meshes: dict[tuple[str, str], object] = {}
        self.slices: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        self.figure = plt.figure(figsize=(15.8, 16.2), facecolor="#f7f9fb")
        grid = self.figure.add_gridspec(
            4,
            3,
            height_ratios=(1.08, 1.0, 1.0, 1.0),
            hspace=0.16,
            wspace=0.065,
            left=0.065,
            right=0.985,
            bottom=0.075,
            top=0.925,
        )

        for column, (variable, heading, subtitle) in enumerate(SOURCES):
            axis = self.figure.add_subplot(grid[0, column])
            values, lat, lon = self.read_field(variable, self.frame, FULL_EXTENT)
            mesh = draw_field(
                axis, values, lat, lon, FULL_EXTENT, labels=(column == 0)
            )
            if column == 0:
                axis.set_xlabel("")
            self.meshes[(variable, "Australia")] = mesh
            axis.set_title(
                f"{heading}\n{subtitle}",
                fontsize=12,
                weight="bold",
                color="#172a3a",
                pad=8,
            )
            for region in REGIONS:
                west, east, south, north = region.extent
                axis.add_patch(
                    Rectangle(
                        (west, south),
                        east - west,
                        north - south,
                        fill=False,
                        linewidth=1.65,
                        edgecolor=region.colour,
                        zorder=8,
                    )
                )
                axis.text(
                    west + 0.25,
                    north - 0.55,
                    region.short_name,
                    color=region.colour,
                    fontsize=6.7,
                    weight="bold",
                    ha="left",
                    va="top",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78,
                          "pad": 1.2},
                    zorder=9,
                )

        for row, region in enumerate(REGIONS, start=1):
            for column, (variable, _, _) in enumerate(SOURCES):
                axis = self.figure.add_subplot(grid[row, column])
                values, lat, lon = self.read_field(variable, self.frame, region.extent)
                mesh = draw_field(
                    axis,
                    values,
                    lat,
                    lon,
                    region.extent,
                    labels=(column == 0),
                )
                if column == 0 and row < len(REGIONS):
                    axis.set_xlabel("")
                self.meshes[(variable, region.name)] = mesh
                if region.marker is not None:
                    axis.scatter(
                        *region.marker,
                        s=15,
                        marker="o",
                        facecolor="white",
                        edgecolor=region.colour,
                        linewidth=1.1,
                        zorder=10,
                    )
                    axis.annotate(
                        region.marker_label,
                        region.marker,
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=6.5,
                        weight="bold",
                        color="#172a3a",
                        zorder=10,
                    )
                if column == 0:
                    axis.text(
                        0.0,
                        1.035,
                        region.name,
                        transform=axis.transAxes,
                        ha="left",
                        va="bottom",
                        fontsize=9,
                        weight="bold",
                        color=region.colour,
                    )

        self.date_text = self.figure.text(
            0.5,
            0.968,
            "",
            ha="center",
            va="center",
            fontsize=13,
            weight="bold",
            color="#172a3a",
        )
        self.figure.suptitle(
            "Sea-surface temperature super-resolution",
            x=0.5,
            y=0.993,
            fontsize=17,
            weight="bold",
            color="#102a43",
        )
        colour_axis = self.figure.add_axes((0.25, 0.025, 0.50, 0.018))
        colourbar = self.figure.colorbar(
            self.meshes[("sst_target", "Australia")],
            cax=colour_axis,
            orientation="horizontal",
        )
        colourbar.set_label("SST (°C) — fixed scale for every panel and frame", fontsize=9)
        colourbar.ax.tick_params(labelsize=8)
        self.update(self.frame)

    def read_field(
        self,
        variable: str,
        frame: int,
        extent: tuple[float, float, float, float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lat_name, lon_name = coordinate_names(variable)
        lat = self.dataset[lat_name].values
        lon = self.dataset[lon_name].values
        values = self.dataset[variable].isel(time=frame).values
        return subset_for_extent(values, lat, lon, extent)

    def update(self, frame: int):
        for variable, _, _ in SOURCES:
            values, _, _ = self.read_field(variable, frame, FULL_EXTENT)
            self.meshes[(variable, "Australia")].set_array(
                np.ma.masked_invalid(values).ravel()
            )
            for region in REGIONS:
                values, _, _ = self.read_field(variable, frame, region.extent)
                self.meshes[(variable, region.name)].set_array(
                    np.ma.masked_invalid(values).ravel()
                )
        timestamp = np.datetime_as_string(self.dataset.time.values[frame], unit="D")
        self.date_text.set_text(
            f"{timestamp}  |  AB3/PC, 75 sampling steps"
        )
        return tuple(self.meshes.values()) + (self.date_text,)


def write_manifest(
    path: Path,
    input_path: Path,
    dataset: xr.Dataset,
    frame: int,
) -> None:
    manifest = {
        "input": str(input_path.resolve()),
        "time_start": np.datetime_as_string(dataset.time.values[0], unit="D"),
        "time_end": np.datetime_as_string(dataset.time.values[-1], unit="D"),
        "time_steps": int(dataset.sizes["time"]),
        "layout_frame": int(frame),
        "layout_date": np.datetime_as_string(dataset.time.values[frame], unit="D"),
        "full_extent": list(FULL_EXTENT),
        "sst_limits_celsius": list(SST_LIMITS),
        "regions": [
            {"name": region.name, "extent": list(region.extent)} for region in REGIONS
        ],
        "columns": [variable for variable, _, _ in SOURCES],
        "animation_rendered": False,
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--layout-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--dpi", type=int, default=110)
    arguments = parser.parse_args()

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(arguments.input) as dataset:
        validate_dataset(dataset)
        if not 0 <= arguments.frame < dataset.sizes["time"]:
            raise ValueError(f"Frame {arguments.frame} is outside the dataset")
        comparison = SSTComparisonFigure(dataset, arguments.frame)
        layout_path = arguments.output_dir / "flow_sr_sst_animation_layout.png"
        comparison.figure.savefig(layout_path, dpi=arguments.dpi, facecolor="#f7f9fb")
        write_manifest(
            arguments.output_dir / "flow_sr_sst_animation_layout.json",
            arguments.input,
            dataset,
            arguments.frame,
        )
        print(f"Wrote layout proof: {layout_path}", flush=True)

        if arguments.layout_only:
            plt.close(comparison.figure)
            return

        frames = np.arange(0, dataset.sizes["time"], arguments.stride)
        if arguments.max_frames is not None:
            frames = frames[: arguments.max_frames]
        output = arguments.output or arguments.output_dir / "flow_sr_sst_comparison.gif"
        animation = FuncAnimation(
            comparison.figure,
            comparison.update,
            frames=frames,
            interval=1000 / arguments.fps,
            blit=False,
            cache_frame_data=False,
        )
        if output.suffix.lower() == ".mp4":
            try:
                import imageio_ffmpeg
            except ImportError as error:
                raise RuntimeError(
                    "MP4 output requires the imageio-ffmpeg dependency"
                ) from error
            matplotlib.rcParams["animation.ffmpeg_path"] = (
                imageio_ffmpeg.get_ffmpeg_exe()
            )
            writer = FFMpegWriter(
                fps=arguments.fps,
                codec="libx264",
                bitrate=5000,
                extra_args=[
                    "-vf",
                    "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                ],
            )
        elif output.suffix.lower() == ".gif":
            writer = PillowWriter(fps=arguments.fps)
        else:
            raise ValueError("Animation output must end in .gif or .mp4")
        animation.save(output, writer=writer, dpi=arguments.dpi)
        plt.close(comparison.figure)
        manifest_path = arguments.output_dir / "flow_sr_sst_animation_layout.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.update({
            "animation_rendered": True,
            "animation": {
                "output": str(output.resolve()),
                "format": output.suffix.lower().lstrip("."),
                "frame_count": int(frames.size),
                "frame_start": int(frames[0]),
                "frame_end": int(frames[-1]),
                "date_start": np.datetime_as_string(
                    dataset.time.values[frames[0]], unit="D"
                ),
                "date_end": np.datetime_as_string(
                    dataset.time.values[frames[-1]], unit="D"
                ),
                "stride_days": int(arguments.stride),
                "fps": int(arguments.fps),
                "duration_seconds": float(frames.size / arguments.fps),
                "dpi": int(arguments.dpi),
            },
        })
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Wrote animation: {output}", flush=True)


if __name__ == "__main__":
    main()
