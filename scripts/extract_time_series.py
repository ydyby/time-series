"""Extract Sentinel-1 backscatter time series using random rectangular samples.

The script expects a directory containing multi-band GeoTIFF files named like::

    S1_2020_M01_ASCENDING_20200107_g17d.tif

Every raster must share the same spatial footprint and contain the nine bands
listed in :data:`BAND_NAMES`.  A categorical ground-truth raster is required in
``zhenzhi.dat`` (or a user supplied path).  For every land-cover class, the
script samples several random rectangles that are completely contained within
the class mask.  Mean values inside each rectangle are computed for every time
step and band, and the resulting time series are saved as CSV files alongside
diagnostic figures after optional Savitzky–Golay smoothing.

Example usage
-------------

.. code-block:: bash

    python scripts/extract_time_series.py \
        --rectangles-per-class 5 \
        --min-rect-size 15 15 \
        --max-rect-size 40 40 \
        --sg-window 7 --sg-order 2

"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUT = BASE / "outputs"
FIGS = OUT / "figs"
S1_DIR = DATA / "s1"


CLASS_NAMES = {
    0: "建筑",
    1: "水体",
    2: "盐田",
    3: "海水",
    4: "滩涂",
    5: "芦苇",
    6: "水稻",
    7: "草地",
    8: "其他",
}


BAND_NAMES = (
    "VV_lin",
    "VH_lin",
    "VV_dB",
    "VH_dB",
    "DB_mean",
    "RVI",
    "H",
    "A",
    "alpha_deg",
)


FNAME_RE = re.compile(
    r"S1_(?:(?P<year>\d{4})_M(?P<month>\d{2})_)?(?P<date>\d{8})_"
    r"(?P<pass>ASC|DESC|ASCENDING|DESCENDING)_(?P<gap>g\d+)"
)


@dataclass(frozen=True)
class RectSample:
    """Description of a sampled rectangle."""

    class_id: int
    index: int
    row: int
    col: int
    height: int
    width: int

    @property
    def class_name(self) -> str:
        return CLASS_NAMES.get(self.class_id, str(self.class_id))

    @property
    def label(self) -> str:
        return f"{self.class_name} #{self.index}"

    def as_dict(self) -> Dict[str, int]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "index": self.index,
            "row": self.row,
            "col": self.col,
            "height": self.height,
            "width": self.width,
        }


@dataclass
class TifInfo:
    path: Path
    date: datetime
    pass_direction: str
    gap: str


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def find_tifs(tif_dir: Path) -> List[TifInfo]:
    tifs: List[TifInfo] = []
    for tif_path in sorted(tif_dir.glob("*.tif")):
        match = FNAME_RE.search(tif_path.name)
        if not match:
            logging.warning("Skipping file with unexpected name: %s", tif_path.name)
            continue
        date_str = match.group("date")
        date = datetime.strptime(date_str, "%Y%m%d")
        pass_dir = match.group("pass")
        gap = match.group("gap")
        tifs.append(TifInfo(tif_path, date, pass_dir, gap))
    tifs.sort(key=lambda item: item.date)
    logging.info("Found %d GeoTIFF scenes", len(tifs))
    if not tifs:
        raise FileNotFoundError(f"No GeoTIFF files matching {FNAME_RE.pattern} found in {tif_dir}")
    return tifs


def read_first_profile(tif_path: Path) -> Tuple[int, int]:
    with rasterio.open(tif_path) as src:
        if src.count != len(BAND_NAMES):
            raise ValueError(
                f"Expected {len(BAND_NAMES)} bands, found {src.count} in {tif_path.name}"
            )
        height, width = src.height, src.width
    logging.info("Scene dimensions: %d (rows) × %d (cols)", height, width)
    return height, width


def load_truth(
    truth_path: Path,
    expected_shape: Tuple[int, int],
    dtype: np.dtype,
    delimiter: Optional[str] = None,
    truth_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    if not truth_path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {truth_path}")

    logging.info("Loading ground-truth map from %s", truth_path)
    shape_hint = truth_shape or expected_shape
    data: np.ndarray
    suffix = truth_path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        data = np.load(truth_path)
    elif suffix in {".tif", ".tiff"}:
        with rasterio.open(truth_path) as src:
            data = src.read(1)
    else:
        try:
            data = np.loadtxt(truth_path, dtype=dtype, delimiter=delimiter)
        except ValueError:
            logging.debug("np.loadtxt failed; falling back to np.fromfile for %s", truth_path)
            data = np.fromfile(truth_path, dtype=dtype)
        if data.ndim == 1:
            if math.prod(shape_hint) != data.size:
                raise ValueError(
                    "Unable to reshape ground-truth array. Provide --truth-shape to disambiguate."
                )
            data = data.reshape(shape_hint)
    data = np.asarray(data, dtype=dtype)
    if data.shape != expected_shape:
        try:
            data = data.reshape(expected_shape)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"Ground-truth map shape {data.shape} does not match expected shape {expected_shape}."
            ) from exc
    return data


def sample_rectangles(
    mask: np.ndarray,
    n_rectangles: int,
    min_size: Tuple[int, int],
    max_size: Tuple[int, int],
    rng: np.random.Generator,
    max_attempts: int,
) -> List[Tuple[int, int, int, int]]:
    height, width = mask.shape
    if not np.any(mask):
        return []

    rectangles: List[Tuple[int, int, int, int]] = []
    attempts = 0
    min_h, min_w = min_size
    max_h = min(max_size[0], height)
    max_w = min(max_size[1], width)
    if min_h > max_h or min_w > max_w:
        logging.debug(
            "Skipping sampling because min rectangle size (%s) exceeds image size (%s)",
            min_size,
            mask.shape,
        )
        return []

    while len(rectangles) < n_rectangles and attempts < max_attempts:
        rect_h = int(rng.integers(min_h, max_h + 1))
        rect_w = int(rng.integers(min_w, max_w + 1))
        max_row = height - rect_h
        max_col = width - rect_w
        if max_row < 0 or max_col < 0:
            attempts += 1
            continue
        row = int(rng.integers(0, max_row + 1))
        col = int(rng.integers(0, max_col + 1))
        patch = mask[row : row + rect_h, col : col + rect_w]
        if np.all(patch):
            rectangles.append((row, col, rect_h, rect_w))
        attempts += 1

    if len(rectangles) < n_rectangles:
        logging.warning(
            "Only found %d/%d rectangles for the class; try increasing max-attempts or relaxing the size range.",
            len(rectangles),
            n_rectangles,
        )
    return rectangles


def build_samples(
    truth: np.ndarray,
    class_ids: Sequence[int],
    rectangles_per_class: int,
    min_size: Tuple[int, int],
    max_size: Tuple[int, int],
    rng: np.random.Generator,
    max_attempts: int,
    ignore_values: Sequence[int],
) -> List[RectSample]:
    samples: List[RectSample] = []
    ignore_mask = np.isin(truth, list(ignore_values)) if ignore_values else np.zeros_like(truth, dtype=bool)
    for class_id in class_ids:
        mask = (truth == class_id) & (~ignore_mask)
        rects = sample_rectangles(mask, rectangles_per_class, min_size, max_size, rng, max_attempts)
        if not rects:
            logging.warning("No valid rectangles found for class %s (%s)", class_id, CLASS_NAMES.get(class_id, class_id))
            continue
        for idx, (row, col, h, w) in enumerate(rects, start=1):
            samples.append(RectSample(class_id, idx, row, col, h, w))
            logging.debug(
                "Class %s rectangle #%d -> row:%d col:%d height:%d width:%d",
                class_id,
                idx,
                row,
                col,
                h,
                w,
            )
    if not samples:
        raise RuntimeError("No rectangles were sampled; unable to continue.")
    logging.info("Sampled %d rectangles across %d classes", len(samples), len(set(s.class_id for s in samples)))
    return samples


def compute_time_series(
    tifs: Sequence[TifInfo],
    samples: Sequence[RectSample],
    bands_to_use: Sequence[str],
    sg_window: Optional[int],
    sg_order: Optional[int],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, float]]]:
    band_indices = {name: idx for idx, name in enumerate(BAND_NAMES)}
    invalid_bands = [band for band in bands_to_use if band not in band_indices]
    if invalid_bands:
        raise ValueError(f"Unknown band names requested: {invalid_bands}")

    band_series: Dict[str, Dict[str, List[float]]] = {
        band: {sample.label: [] for sample in samples} for band in bands_to_use
    }
    stats: Dict[str, Dict[str, float]] = {band: {} for band in bands_to_use}

    for tif in tifs:
        with rasterio.open(tif.path) as src:
            stack = src.read(out_dtype="float32")
        for sample in samples:
            window = stack[
                :, sample.row : sample.row + sample.height, sample.col : sample.col + sample.width
            ]
            means = np.nanmean(window.reshape(window.shape[0], -1), axis=1)
            for band in bands_to_use:
                value = float(means[band_indices[band]])
                band_series[band][sample.label].append(value)

    index = pd.to_datetime([tif.date for tif in tifs])
    smoothed_frames: Dict[str, pd.DataFrame] = {}
    for band in bands_to_use:
        data = {label: np.array(values, dtype=float) for label, values in band_series[band].items()}
        df = pd.DataFrame(data, index=index).sort_index()
        if sg_window and sg_order is not None and sg_window > 2:
            if sg_window % 2 == 0:
                sg_window += 1  # enforce odd window for Savitzky-Golay
            for column in df.columns:
                series = df[column].to_numpy()
                if np.count_nonzero(~np.isnan(series)) < sg_order + 2:
                    logging.warning(
                        "Skipping smoothing for %s (%s): not enough valid samples.",
                        band,
                        column,
                    )
                    continue
                filtered = savgol_filter(series, sg_window, sg_order, mode="interp")
                df[column] = filtered
        smoothed_frames[band] = df
        # summary statistics for diagnostics
        stats[band]["min"] = float(np.nanmin(df.to_numpy()))
        stats[band]["max"] = float(np.nanmax(df.to_numpy()))
        stats[band]["mean"] = float(np.nanmean(df.to_numpy()))
    return smoothed_frames, stats


def plot_band(
    band: str,
    df: pd.DataFrame,
    samples: Sequence[RectSample],
    fig_path: Path,
) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    class_to_color = {class_id: plt.get_cmap("tab10")(class_id % 10) for class_id in CLASS_NAMES}

    plt.figure(figsize=(12, 6))
    for sample in samples:
        label = sample.label
        if label not in df.columns:
            continue
        color = class_to_color.get(sample.class_id)
        plt.plot(df.index, df[label], label=label, color=color, linewidth=2)
    plt.title(f"{band} 时序曲线")
    plt.xlabel("观测日期")
    plt.ylabel(f"{band} 平均值")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()


def save_band_csv(band: str, df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_index().to_csv(out_path, index_label="date")


def save_metadata(
    out_dir: Path,
    tifs: Sequence[TifInfo],
    samples: Sequence[RectSample],
    per_band_files: Dict[str, Dict[str, str]],
) -> None:
    metadata = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "tifs": [
            {
                "path": str(tif.path.relative_to(BASE)),
                "date": tif.date.isoformat(),
                "pass": tif.pass_direction,
                "gap": tif.gap,
            }
            for tif in tifs
        ],
        "samples": [sample.as_dict() for sample in samples],
        "bands": per_band_files,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def parse_size_argument(values: Sequence[int], name: str) -> Tuple[int, int]:
    if len(values) == 1:
        return int(values[0]), int(values[0])
    if len(values) != 2:
        raise argparse.ArgumentTypeError(f"{name} expects one or two integers")
    return int(values[0]), int(values[1])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tif-dir", type=Path, default=S1_DIR, help="Directory containing Sentinel-1 GeoTIFF stacks")
    parser.add_argument("--truth-path", type=Path, default=DATA / "zhenzhi.dat", help="Ground-truth file path")
    parser.add_argument("--rectangles-per-class", type=int, default=3, help="Number of rectangles to sample for each class")
    parser.add_argument(
        "--min-rect-size",
        type=int,
        nargs="+",
        default=(12, 12),
        help="Minimum rectangle size as one integer (square) or two integers (rows cols)",
    )
    parser.add_argument(
        "--max-rect-size",
        type=int,
        nargs="+",
        default=(32, 32),
        help="Maximum rectangle size as one integer (square) or two integers (rows cols)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    parser.add_argument("--max-attempts", type=int, default=5000, help="Maximum attempts per class when sampling rectangles")
    parser.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=sorted(CLASS_NAMES.keys()),
        help="Subset of class IDs to include",
    )
    parser.add_argument(
        "--ignore-values",
        type=int,
        nargs="*",
        default=[-1],
        help="Label values that should be ignored (treated as unlabeled)",
    )
    parser.add_argument(
        "--bands",
        nargs="*",
        default=list(BAND_NAMES),
        help="Band names to include in the analysis",
    )
    parser.add_argument("--sg-window", type=int, default=0, help="Savitzky–Golay window length (must be odd)")
    parser.add_argument("--sg-order", type=int, default=2, help="Savitzky–Golay polynomial order")
    parser.add_argument("--truth-dtype", default="int16", help="Data type of ground-truth values (e.g., int16)")
    parser.add_argument("--truth-delimiter", default=None, help="Delimiter passed to numpy.loadtxt for the truth map")
    parser.add_argument(
        "--truth-shape",
        type=int,
        nargs=2,
        metavar=("ROWS", "COLS"),
        help="Optional shape hint when reading binary .dat ground-truth files",
    )
    parser.add_argument("--fig-format", default="png", help="File format for saved figures (png, pdf, ...)")
    parser.add_argument("--out-dir", type=Path, default=OUT / "timeseries", help="Directory for CSV/metadata outputs")
    parser.add_argument("--fig-dir", type=Path, default=FIGS, help="Directory for time-series figures")
    parser.add_argument("--show", action="store_true", help="Display plots interactively in addition to saving them")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    setup_logging(args.verbose)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    tifs = find_tifs(args.tif_dir)
    height, width = read_first_profile(tifs[0].path)
    truth = load_truth(
        args.truth_path,
        expected_shape=(height, width),
        dtype=np.dtype(args.truth_dtype),
        delimiter=args.truth_delimiter,
        truth_shape=tuple(args.truth_shape) if args.truth_shape else None,
    )

    min_size = parse_size_argument(args.min_rect_size, "--min-rect-size")
    max_size = parse_size_argument(args.max_rect_size, "--max-rect-size")
    if min_size[0] > max_size[0] or min_size[1] > max_size[1]:
        raise ValueError("Minimum rectangle size cannot exceed maximum size")

    rng = np.random.default_rng(args.seed)
    samples = build_samples(
        truth=truth,
        class_ids=args.classes,
        rectangles_per_class=args.rectangles_per_class,
        min_size=min_size,
        max_size=max_size,
        rng=rng,
        max_attempts=args.max_attempts,
        ignore_values=args.ignore_values,
    )

    sg_window = args.sg_window if args.sg_window and args.sg_window > 0 else None
    sg_order = args.sg_order if sg_window else None
    frames, stats = compute_time_series(tifs, samples, args.bands, sg_window, sg_order)

    per_band_files: Dict[str, Dict[str, str]] = {}
    for band, df in frames.items():
        csv_path = args.out_dir / f"{band}.csv"
        fig_path = args.fig_dir / f"{band}.{args.fig_format}"
        save_band_csv(band, df, csv_path)
        plot_band(band, df, samples, fig_path)
        per_band_files[band] = {
            "csv": str(csv_path.relative_to(BASE)),
            "figure": str(fig_path.relative_to(BASE)),
            "min": stats[band]["min"],
            "max": stats[band]["max"],
            "mean": stats[band]["mean"],
        }
        if args.show:
            plt.show()

    save_metadata(args.out_dir, tifs, samples, per_band_files)
    logging.info("Time-series extraction complete. Outputs saved to %s", args.out_dir)


if __name__ == "__main__":
    main()

