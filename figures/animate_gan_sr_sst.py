#!/usr/bin/env python3
"""Plot and animate the GAN-SR SST evaluation at full and regional scales.

Compares:
1. Authoritative coarse input (1.6° block mean)
2. GAN super-resolution prediction (0.1° generated SST)
3. Native OFAM ground truth target (0.1° target SST)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import xarray as xr


DEFAULT_INPUT = Path(
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
    ("sst_generated", "GAN prediction", "0.1° generated SST"),
    ("sst_target", "OFAM ground truth", "0.1° target SST"),
)


def coordinate_edges(values: np.ndarray) -> np.ndarray:
    """Convert monotonic cell centres to edges for exact pcolormesh geometry."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.diff(values) > 0):
        raise ValueError("Coordinates must be one-dimensional and increasing")
    middle = 0.5 * (values[:-1] + values[1:])
    return np.concatenate(
        (
            [values[0] - 0.5 * (values[1] - values[0])],
            middle,
            [values[-1] + 0.5 * (values[-1] - values[-2])],
        )
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


class GANSSTFigure:
    """Reusable artists for static proof and GAN animation."""

    def __init__(self, dataset: xr.Dataset, frame: int):
        self.dataset = dataset
        self.frame = int(frame)
        self.meshes: dict[tuple[str, str], object] = {}
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
            mesh = draw_field(axis, values, lat, lon, FULL_EXTENT, labels=(column == 0))
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
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
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
            "GAN Sea-Surface Temperature Super-Resolution",
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
            f"{timestamp}  |  Holdout Test Day {frame + 1} / {self.dataset.sizes['time']}  |  GAN-SR"
        )
        return tuple(self.meshes.values()) + (self.date_text,)


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

    if not arguments.input.exists():
        alt = Path("/esi/project/niwa03712/rampaln/PUBLICATIONS/2026/SSTDownscaling/runs/gan_sr_v2/evaluation/full_test_samples.nc")
        if alt.exists():
            arguments.input = alt

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(arguments.input) as dataset:
        if not 0 <= arguments.frame < dataset.sizes["time"]:
            raise ValueError(f"Frame {arguments.frame} is outside the dataset")

        comparison = GANSSTFigure(dataset, arguments.frame)
        layout_path = arguments.output_dir / "gan_sr_sst_animation_layout.png"
        comparison.figure.savefig(layout_path, dpi=arguments.dpi, facecolor="#f7f9fb")
        print(f"Wrote layout proof: {layout_path}", flush=True)

        if arguments.layout_only:
            plt.close(comparison.figure)
            return

        frames = np.arange(0, dataset.sizes["time"], arguments.stride)
        if arguments.max_frames is not None:
            frames = frames[: arguments.max_frames]
        output = arguments.output or arguments.output_dir / "gan_sr_sst_comparison.gif"
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
                matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                pass
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
        print(f"Wrote animation: {output}", flush=True)


if __name__ == "__main__":
    main()
