import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

import matplotlib.pyplot as plt


SEED_ORDER = [2020, 42, 123, 309, 969]

SETTINGS = [
    {
        "name": "lif_mlp_noim",
        "label": "Non-spiking mean membrane",
        "summary_csv": Path("runs/summary/shd_multiseed_best_test_eval.csv"),
    },
    {
        "name": "spiking_count",
        "label": "Spiking output count",
        "root": Path("runs/ff_output_3way/spiking_count"),
    },
    {
        "name": "spiking_count_im",
        "label": "Spiking output count + IM",
        "root": Path("runs/ff_output_3way/spiking_count_im"),
    },
]


def load_eval_json(path: Path) -> Dict[str, object]:
    with open(path) as handle:
        blob = json.load(handle)
    if isinstance(blob, list):
        if len(blob) != 1:
            raise ValueError(f"Expected one eval row in {path}, found {len(blob)}")
        return dict(blob[0])
    return dict(blob)


def find_eval_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    missing = []

    for setting in SETTINGS:
        if "summary_csv" in setting:
            rows_by_seed: Dict[int, Dict[str, object]] = {}
            with open(setting["summary_csv"]) as handle:
                for row in csv.DictReader(handle):
                    if row["setting"] != setting["name"]:
                        continue
                    seed = int(row["seed"])
                    rows_by_seed[seed] = {
                        "setting": setting["name"],
                        "setting_label": setting["label"],
                        "seed": seed,
                        "output_mode": "nonspiking",
                        "readout": "mean_membrane",
                        "use_im_loss_train": row["use_im_loss_train"],
                        "im_include_output_train": False,
                        "im_loss_weight_train": 0.001,
                        "im_hidden_weight_train": 1.0,
                        "im_output_weight_train": 1.0,
                        "epoch": int(row["best_epoch"]),
                        "best_val_acc": float(row["best_val_acc_percent"]) / 100.0,
                        "test_acc_percent": float(row["test_acc_percent"]),
                        "test_ce_loss": float(row["test_ce_loss"]),
                        "checkpoint": row["checkpoint"],
                        "eval_json": str(setting["summary_csv"]),
                    }

            for seed in SEED_ORDER:
                row = rows_by_seed.get(seed)
                if row is None:
                    missing.append((setting["name"], seed))
                    continue
                rows.append(row)
            continue

        eval_by_seed: Dict[int, Path] = {}
        root = setting["root"]
        if root.exists():
            for eval_path in sorted(root.glob("*/test_eval.json")):
                row = load_eval_json(eval_path)
                seed = int(row["seed"])
                previous = eval_by_seed.get(seed)
                if previous is None or eval_path.stat().st_mtime > previous.stat().st_mtime:
                    eval_by_seed[seed] = eval_path

        for seed in SEED_ORDER:
            eval_path = eval_by_seed.get(seed)
            if eval_path is None:
                missing.append((setting["name"], seed))
                continue
            row = load_eval_json(eval_path)
            row["setting"] = setting["name"]
            row["setting_label"] = setting["label"]
            row["eval_json"] = str(eval_path)
            rows.append(row)

    if missing:
        raise RuntimeError(f"Missing eval outputs: {missing}")

    setting_index = {setting["name"]: idx for idx, setting in enumerate(SETTINGS)}
    seed_index = {seed: idx for idx, seed in enumerate(SEED_ORDER)}
    rows.sort(key=lambda row: (setting_index[row["setting"]], seed_index[int(row["seed"])]))
    return rows


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    summary_rows: List[Dict[str, object]] = []
    for setting in SETTINGS:
        values = [
            float(row["test_acc_percent"])
            for row in rows
            if row["setting"] == setting["name"]
        ]
        if not values:
            continue
        summary_rows.append(
            {
                "setting": setting["name"],
                "setting_label": setting["label"],
                "n": len(values),
                "mean_test_acc_percent": mean(values),
                "std_test_acc_percent": stdev(values) if len(values) > 1 else 0.0,
                "min_test_acc_percent": min(values),
                "max_test_acc_percent": max(values),
                "seed_test_acc_percent": ";".join(f"{value:.4f}" for value in values),
            }
        )
    return summary_rows


def write_detail_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "setting",
        "setting_label",
        "seed",
        "output_mode",
        "readout",
        "use_im_loss_train",
        "im_include_output_train",
        "im_loss_weight_train",
        "im_hidden_weight_train",
        "im_output_weight_train",
        "best_epoch",
        "best_val_acc_percent",
        "test_acc_percent",
        "test_ce_loss",
        "test_im_loss",
        "test_output_im_loss",
        "test_firing_rate_layer_0",
        "test_firing_rate_output",
        "checkpoint",
        "eval_json",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "setting": row["setting"],
                    "setting_label": row["setting_label"],
                    "seed": row["seed"],
                    "output_mode": row["output_mode"],
                    "readout": row["readout"],
                    "use_im_loss_train": row["use_im_loss_train"],
                    "im_include_output_train": row["im_include_output_train"],
                    "im_loss_weight_train": row["im_loss_weight_train"],
                    "im_hidden_weight_train": row["im_hidden_weight_train"],
                    "im_output_weight_train": row["im_output_weight_train"],
                    "best_epoch": row["epoch"],
                    "best_val_acc_percent": 100.0 * float(row["best_val_acc"]),
                    "test_acc_percent": row["test_acc_percent"],
                    "test_ce_loss": row["test_ce_loss"],
                    "test_im_loss": row.get("test_im_loss", ""),
                    "test_output_im_loss": row.get("test_output_im_loss", ""),
                    "test_firing_rate_layer_0": row.get("test_firing_rate_layer_0", ""),
                    "test_firing_rate_output": row.get("test_firing_rate_output", ""),
                    "checkpoint": row["checkpoint"],
                    "eval_json": row["eval_json"],
                }
            )


def write_summary_csv(path: Path, summary_rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "setting",
        "setting_label",
        "n",
        "mean_test_acc_percent",
        "std_test_acc_percent",
        "min_test_acc_percent",
        "max_test_acc_percent",
        "seed_test_acc_percent",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def plot_hist(path: Path, summary_rows: List[Dict[str, object]]) -> None:
    labels = [row["setting_label"] for row in summary_rows]
    means = [float(row["mean_test_acc_percent"]) for row in summary_rows]
    stds = [float(row["std_test_acc_percent"]) for row in summary_rows]
    mins = [float(row["min_test_acc_percent"]) for row in summary_rows]
    maxs = [float(row["max_test_acc_percent"]) for row in summary_rows]
    colors = ["#4c78a8", "#59a14f", "#f28e2b"]

    x_positions = list(range(len(summary_rows)))
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    bars = ax.bar(
        x_positions,
        means,
        width=0.58,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.92,
    )

    lower = [mean_value - min_value for mean_value, min_value in zip(means, mins)]
    upper = [max_value - mean_value for max_value, mean_value in zip(maxs, means)]
    ax.errorbar(
        x_positions,
        means,
        yerr=[lower, upper],
        fmt="none",
        ecolor="black",
        elinewidth=1.4,
        capsize=6,
        capthick=1.4,
        label="5-seed min-max",
    )

    for bar, mean_value, std_value, min_value, max_value in zip(bars, means, stds, mins, maxs):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            max_value + 1.2,
            f"{mean_value:.2f} +/- {std_value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("SHD test accuracy (%)")
    ax.set_title("FF-LIF output settings on SHD, 5 seeds")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize FF-LIF SHD output-setting runs")
    parser.add_argument("--output_dir", default="runs/summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    rows = find_eval_rows()
    summary_rows = summarize(rows)

    detail_csv = output_dir / "shd_ff_output_3way_test_eval.csv"
    summary_csv = output_dir / "shd_ff_output_3way_summary.csv"
    json_path = output_dir / "shd_ff_output_3way_summary.json"
    hist_path = output_dir / "shd_ff_output_3way_hist.png"

    write_detail_csv(detail_csv, rows)
    write_summary_csv(summary_csv, summary_rows)
    with open(json_path, "w") as handle:
        json.dump({"results": rows, "summary": summary_rows}, handle, indent=2)
    plot_hist(hist_path, summary_rows)

    for row in summary_rows:
        print(
            "{setting}: n={n} mean={mean:.2f} std={std:.2f} min={minv:.2f} max={maxv:.2f}".format(
                setting=row["setting"],
                n=row["n"],
                mean=row["mean_test_acc_percent"],
                std=row["std_test_acc_percent"],
                minv=row["min_test_acc_percent"],
                maxv=row["max_test_acc_percent"],
            )
        )
    print(f"wrote {detail_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {json_path}")
    print(f"wrote {hist_path}")


if __name__ == "__main__":
    main()
