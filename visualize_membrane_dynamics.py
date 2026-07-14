"""Reproduce the archived SHD output-dynamics visualizations.

The expensive checkpoint evaluation is intentionally not repeated here.  This
script renders the report figures from the deterministic archives written by
the evaluation: ``*_per_run.json``, ``*_sample_manifest.csv``, and
``*_selected_traces.npz``.  The notebook calls this same file, so its figures
cannot drift from the command-line versions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


DEFAULT_PREFIX = "shd_threshold_1p0_membrane_dynamics"


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def display_condition(condition: str) -> str:
    lower = condition.lower()
    if "no_im" in lower or "noim" in lower or "no-im" in lower:
        return "matched no IM"
    if "threshold" in lower and "im" in lower:
        return "output threshold IM"
    return condition.replace("_", " ")


def legend_condition(condition: str) -> str:
    return "no IM" if "no_im" in condition.lower() else "threshold IM"


def require_files(prefix: Path) -> dict[str, Path]:
    paths = {
        "per_run": Path(f"{prefix}_per_run.json"),
        "sample_manifest": Path(f"{prefix}_sample_manifest.csv"),
        "selected_traces": Path(f"{prefix}_selected_traces.npz"),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        formatted = "\n  ".join(missing)
        raise FileNotFoundError(
            "The archived visualization inputs are incomplete. Missing:\n  " + formatted
        )
    return paths


def load_archives(prefix: Path):
    paths = require_files(prefix)
    with paths["per_run"].open(encoding="utf-8") as handle:
        per_run = json.load(handle)
    with paths["sample_manifest"].open(newline="", encoding="utf-8-sig") as handle:
        sample_rows = list(csv.DictReader(handle))

    aggregate = per_run["condition_aggregate"]
    conditions = tuple(aggregate)
    if len(conditions) != 2:
        raise ValueError(f"expected exactly two archived conditions, got {conditions}")
    reference = next(
        (condition for condition in conditions if "no_im" in condition.lower()),
        conditions[0],
    )
    comparison = next(condition for condition in conditions if condition != reference)

    result0 = per_run["results"][0]
    dt = float(result0["dt"])
    time_steps = int(result0["time_steps"])
    target = float(result0["membrane_target"])
    threshold_by_condition = {
        str(row["condition"]): float(row["physical_threshold"])
        for row in per_run["results"]
    }

    rows_by_label = {int(row["label"]): row for row in sample_rows}
    if sorted(rows_by_label) != list(range(len(rows_by_label))):
        raise ValueError("sample manifest must contain consecutive class labels from zero")
    class_names = [rows_by_label[index]["label_name"] for index in range(len(rows_by_label))]
    main_labels = [int(row["label"]) for row in sample_rows if row.get("role") == "main"]

    traces: dict[str, dict[str, object]] = {}
    array_keys = (
        "predictions",
        "competitor_indices",
        "counts",
        "output_pre_reset",
        "output_post_reset",
        "output_spikes",
        "input_event_count_by_time",
    )
    with np.load(paths["selected_traces"]) as archive:
        sample_indices = np.asarray(archive["sample_indices"])
        labels = np.asarray(archive["labels"])
        for condition in (reference, comparison):
            identity = safe_name(condition)
            traces[condition] = {
                "sample_indices": sample_indices,
                "labels": labels,
                "threshold": threshold_by_condition[condition],
                **{
                    key: np.asarray(archive[f"{identity}_{key}"])
                    for key in array_keys
                },
            }

    for condition in (reference, comparison):
        spikes = np.asarray(traces[condition]["output_spikes"])
        if spikes.shape[1] != time_steps:
            raise AssertionError(
                f"{condition}: trace has {spikes.shape[1]} steps, expected {time_steps}"
            )
    return (
        aggregate,
        traces,
        sample_rows,
        class_names,
        main_labels,
        reference,
        comparison,
        dt,
        time_steps,
        target,
    )


def assert_trace_conservation(
    trace: Mapping[str, object], position: int, true_class: int
) -> np.ndarray:
    spikes = np.asarray(trace["output_spikes"])[position]
    counts = np.asarray(trace["counts"])[position]
    reconstructed = spikes.sum(axis=0)
    if not np.array_equal(reconstructed, counts):
        raise AssertionError("full raster sum does not equal the archived class counts")
    prediction = int(np.asarray(trace["predictions"])[position])
    if prediction != int(np.argmax(reconstructed)):
        raise AssertionError("prediction is not argmax of the archived spike counts")
    pre = np.asarray(trace["output_pre_reset"])[position]
    post = np.asarray(trace["output_post_reset"])[position]
    threshold = float(trace["threshold"])
    if not np.array_equal((pre > threshold).astype(spikes.dtype), spikes):
        raise AssertionError("raster disagrees with pre-reset threshold crossings")
    if not np.allclose(post, pre - spikes * threshold, rtol=0.0, atol=1e-6):
        raise AssertionError("pre-reset, spike, and post-reset archives are inconsistent")
    alternatives = reconstructed.copy()
    alternatives[true_class] = -1.0
    competitor = int(np.asarray(trace["competitor_indices"])[position])
    if competitor != int(np.argmax(alternatives)):
        raise AssertionError("competitor is not the strongest non-true class")
    return reconstructed


def selected_input_end(
    traces: Mapping[str, Mapping[str, object]], reference: str, position: int, dt: float
) -> tuple[int, float]:
    activity = np.asarray(traces[reference]["input_event_count_by_time"])[position]
    active_steps = np.flatnonzero(activity > 0)
    if not active_steps.size:
        return -1, 0.0
    last_step = int(active_steps[-1])
    return last_step, (last_step + 1) * dt


def plot_time_course(
    path: Path,
    aggregate: Mapping[str, object],
    reference: str,
    comparison: str,
    dt: float,
    target: float,
    near_eps: Sequence[float] = (0.1, 0.25),
) -> None:
    layer_name = "output"
    conditions = (reference, comparison)
    colors = {reference: "#4C78A8", comparison: "#F58518"}
    time_steps = len(
        aggregate[reference]["layers"][layer_name]["time_series"]["membrane_mean"]["mean"]
    )
    time = np.arange(time_steps) * dt
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 11.0), sharex=True)
    specs = [
        ("membrane_mean", "Mean pre-reset membrane", "U pre-reset"),
        (
            "running_proxy",
            "Prefix pooled-mean proxy (training-form diagnostic)",
            r"$(\bar U_{\leq t}-\theta)^2$",
        ),
        ("firing_rate", "Actual threshold-crossing / spike rate", "probability / step"),
        (
            "binary_entropy_bits",
            "Time-local H2 (solid); prefix-pooled H2 (dashed)",
            "bits",
        ),
        (
            "fraction_within_eps_large",
            f"Mass within ±{near_eps[1]:g} of threshold (diagnostic)",
            "fraction",
        ),
        (
            "correct_minus_competitor",
            "Correct minus final-strongest-competitor spike rate",
            "rate difference",
        ),
    ]
    for ax, (metric, title, ylabel) in zip(axes.flat, specs):
        for condition in conditions:
            layer = aggregate[condition]["layers"][layer_name]
            if metric == "correct_minus_competitor":
                mean = np.asarray(layer["correct_minus_competitor_firing_rate"]["mean"])
                sd = np.asarray(layer["correct_minus_competitor_firing_rate"]["sd"])
            else:
                mean = np.asarray(layer["time_series"][metric]["mean"])
                sd = np.asarray(layer["time_series"][metric]["sd"])
            ax.plot(time, mean, color=colors[condition], lw=2.2, label=display_condition(condition))
            if np.any(sd > 0):
                ax.fill_between(
                    time, mean - sd, mean + sd, color=colors[condition], alpha=0.16
                )
            if metric == "binary_entropy_bits":
                running_mean = np.asarray(
                    layer["time_series"]["running_binary_entropy_bits"]["mean"]
                )
                ax.plot(time, running_mean, color=colors[condition], lw=1.35, ls="--")
        if metric == "membrane_mean":
            ax.axhline(
                target,
                color="black",
                ls="--",
                lw=1.1,
                label="physical threshold = output IM target",
            )
        if metric == "running_proxy":
            ax.set_yscale("log")
        if metric == "binary_entropy_bits":
            ax.axhline(1.0, color="0.35", ls=":", lw=1.0, label="1-bit maximum")
        if metric == "correct_minus_competitor":
            ax.axhline(0.0, color="black", lw=0.8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "SHD output membrane → threshold crossing → entropy "
        "(mean ± seed SD; output layer is directly IM-regularized)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_final_count_bars(
    ax: plt.Axes,
    traces: Mapping[str, Mapping[str, object]],
    reference: str,
    comparison: str,
    position: int,
    true_class: int,
    class_names: Sequence[str],
    dt: float,
    show_legend: bool,
) -> None:
    last_step, input_end = selected_input_end(traces, reference, position, dt)
    comparison_last_step, comparison_input_end = selected_input_end(
        traces, comparison, position, dt
    )
    if comparison_last_step != last_step or not math.isclose(
        comparison_input_end, input_end, abs_tol=1e-12
    ):
        raise AssertionError("paired traces disagree on encoded input end")
    y = np.arange(len(class_names), dtype=np.float64)
    styles = (
        (reference, -0.20, "#4C78A8"),
        (comparison, 0.20, "#F28E2B"),
    )
    for condition, offset, color in styles:
        counts = assert_trace_conservation(traces[condition], position, true_class)
        spikes = np.asarray(traces[condition]["output_spikes"])[position]
        if last_step >= 0:
            input_counts = spikes[: last_step + 1].sum(axis=0)
            post_input_counts = spikes[last_step + 1 :].sum(axis=0)
        else:
            input_counts = np.zeros(spikes.shape[1], dtype=np.float64)
            post_input_counts = spikes.sum(axis=0)
        if not np.array_equal(input_counts + post_input_counts, counts):
            raise AssertionError("bar segments do not conserve full class counts")
        ax.barh(
            y + offset,
            input_counts,
            height=0.36,
            color=color,
            edgecolor="white",
            linewidth=0.25,
        )
        ax.barh(
            y + offset,
            post_input_counts,
            left=input_counts,
            height=0.36,
            color=color,
            alpha=0.30,
            hatch="////",
            edgecolor=color,
            linewidth=0.35,
        )
    ax.axhspan(true_class - 0.48, true_class + 0.48, color="#00A087", alpha=0.10)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=6.2)
    ax.invert_yaxis()
    for tick_index, tick_label in enumerate(ax.get_yticklabels()):
        if tick_index == true_class:
            tick_label.set_color("#007A65")
            tick_label.set_fontweight("bold")
    ref_pred = int(np.asarray(traces[reference]["predictions"])[position])
    cmp_pred = int(np.asarray(traces[comparison]["predictions"])[position])
    ax.set_title(
        f"Final count, all 20 classes (input end {input_end:.3f} s)\n"
        f"no IM pred={class_names[ref_pred]}; IM pred={class_names[cmp_pred]}",
        fontsize=9.2,
    )
    duration = np.asarray(traces[reference]["output_spikes"]).shape[1] * dt
    ax.set_xlabel(f"spikes over full {duration:g} s")
    ax.grid(axis="x", alpha=0.18)
    if show_legend:
        ax.legend(
            handles=[
                Patch(facecolor="#4C78A8", label=f"color: {legend_condition(reference)}"),
                Patch(facecolor="#F28E2B", label=f"color: {legend_condition(comparison)}"),
                Patch(facecolor="0.62", label="opaque: through input end"),
                Patch(
                    facecolor="0.75",
                    alpha=0.45,
                    hatch="////",
                    edgecolor="0.4",
                    label="hatched: after encoded input ended",
                ),
            ],
            frameon=False,
            fontsize=6.4,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
        )


def plot_selected_rasters(
    path: Path,
    traces: Mapping[str, Mapping[str, object]],
    reference: str,
    comparison: str,
    sample_rows: Sequence[Mapping[str, object]],
    class_names: Sequence[str],
    main_labels: Sequence[int],
    dt: float,
) -> None:
    selected = [row for row in sample_rows if int(row["label"]) in main_labels]
    conditions = (reference, comparison)
    fig, axes = plt.subplots(
        len(selected),
        3,
        figsize=(19.5, 3.25 * len(selected)),
        gridspec_kw={"width_ratios": (1.35, 1.35, 1.0)},
    )
    if len(selected) == 1:
        axes = np.asarray([axes])
    for row_index, sample in enumerate(selected):
        position = next(
            index
            for index, value in enumerate(traces[reference]["sample_indices"])
            if int(value) == int(sample["sample_index"])
        )
        _, input_end = selected_input_end(traces, reference, position, dt)
        _, comparison_input_end = selected_input_end(traces, comparison, position, dt)
        if not math.isclose(comparison_input_end, input_end, abs_tol=1e-12):
            raise AssertionError("paired traces disagree on encoded input end")
        for column_index, condition in enumerate(conditions):
            trace = traces[condition]
            assert_trace_conservation(trace, position, int(sample["label"]))
            spikes = np.asarray(trace["output_spikes"])[position].T
            duration = spikes.shape[1] * dt
            ax = axes[row_index, column_index]
            ax.imshow(
                spikes,
                aspect="auto",
                interpolation="nearest",
                cmap="Greys",
                vmin=0,
                vmax=1,
                extent=(0.0, duration, len(class_names) - 0.5, -0.5),
            )
            ax.axvspan(input_end, duration, color="#BDBDBD", alpha=0.20, zorder=2)
            ax.axvline(
                input_end,
                color="#6F6F6F",
                ls="--",
                lw=0.9,
                zorder=3,
                label="input end",
            )
            ax.axhline(int(sample["label"]), color="#00A087", lw=1.2)
            prediction = int(np.asarray(trace["predictions"])[position])
            ax.set_title(
                f"{display_condition(condition)} | final pred={class_names[prediction]}"
            )
            if column_index == 0:
                ax.set_ylabel(f"index {sample['sample_index']}\n{sample['label_name']}")
            ax.set_yticks(range(len(class_names)))
            ax.set_yticklabels(class_names, fontsize=6)
            ax.set_xlabel("time (s)")
            ax.set_xlim(0.0, duration)
        plot_final_count_bars(
            axes[row_index, 2],
            traces,
            reference,
            comparison,
            position,
            int(sample["label"]),
            class_names,
            dt,
            show_legend=row_index == 0,
        )
    fig.suptitle(
        "All 20 output-class spike rasters and final counts\n"
        "green=true class; dashed line=input end; gray/hatching=no further encoded input events"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def render(input_prefix: Path, output_dir: Path, output_prefix: str | None = None) -> list[Path]:
    (
        aggregate,
        traces,
        sample_rows,
        class_names,
        main_labels,
        reference,
        comparison,
        dt,
        _time_steps,
        target,
    ) = load_archives(input_prefix)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_prefix or input_prefix.name
    raster_path = output_dir / f"{stem}_selected_sample_spike_rasters.png"
    time_course_path = output_dir / f"{stem}_output_time_course.png"
    plot_selected_rasters(
        raster_path,
        traces,
        reference,
        comparison,
        sample_rows,
        class_names,
        main_labels,
        dt,
    )
    plot_time_course(
        time_course_path,
        aggregate,
        reference,
        comparison,
        dt,
        target,
    )
    return [raster_path, time_course_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the archived SHD membrane-dynamics report figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-prefix",
        type=Path,
        default=Path("im_snn_runs/summary") / DEFAULT_PREFIX,
        help="archive prefix, without the _per_run.json suffix",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("im_snn_runs/summary"),
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="output filename prefix; defaults to the input prefix basename",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in render(args.input_prefix, args.output_dir, args.output_prefix):
        print("wrote", path)


if __name__ == "__main__":
    main()
