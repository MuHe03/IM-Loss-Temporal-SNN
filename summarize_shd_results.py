import argparse
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib.pyplot as plt
import torch

from eval_temporal import eval_checkpoint


RUN_ROOTS = {
    "lif_mlp_noim": [
        "runs/temporal_baseline",
        "runs/temporal_multiseed/lif_mlp_baseline",
        "runs/shd/temporal_multiseed/lif_mlp_baseline",
    ],
    "lif_mlp_im": [
        "runs/temporal_imloss",
        "runs/temporal_multiseed/lif_mlp_imloss",
        "runs/shd/temporal_multiseed/lif_mlp_imloss",
    ],
    "rsnn_lif_noim": [
        "runs/temporal_rsnn_baseline",
        "runs/temporal_multiseed/rsnn_lif_baseline",
        "runs/shd/temporal_multiseed/rsnn_lif_baseline",
    ],
    "rsnn_lif_im": [
        "runs/temporal_rsnn_imloss",
        "runs/temporal_multiseed/rsnn_lif_imloss",
        "runs/shd/temporal_multiseed/rsnn_lif_imloss",
    ],
}

SEED_ORDER = [2020, 42, 123, 309, 969]
SETTING_ORDER = ["lif_mlp_noim", "lif_mlp_im", "rsnn_lif_noim", "rsnn_lif_im"]
SETTING_LABELS = {
    "lif_mlp_noim": "FF-LIF",
    "lif_mlp_im": "FF-LIF + IM",
    "rsnn_lif_noim": "RSNN",
    "rsnn_lif_im": "RSNN + IM",
}


def find_checkpoints(run_root: str = "runs") -> Dict[str, Dict[int, str]]:
    checkpoints: Dict[str, Dict[int, str]] = {setting: {} for setting in SETTING_ORDER}
    run_root_path = Path(run_root)
    run_root_overrides = {
        "lif_mlp_noim": [run_root_path / "shd" / "temporal_multiseed" / "lif_mlp_baseline"],
        "lif_mlp_im": [run_root_path / "shd" / "temporal_multiseed" / "lif_mlp_imloss"],
        "rsnn_lif_noim": [run_root_path / "shd" / "temporal_multiseed" / "rsnn_lif_baseline"],
        "rsnn_lif_im": [run_root_path / "shd" / "temporal_multiseed" / "rsnn_lif_imloss"],
    }
    for setting, roots in RUN_ROOTS.items():
        candidate_roots = [Path(root) for root in roots] + run_root_overrides[setting]
        for root_path in candidate_roots:
            if not root_path.exists():
                continue
            for ckpt_path in sorted(root_path.glob("*/best.pth")):
                config_path = ckpt_path.parent / "config.json"
                if not config_path.exists():
                    continue
                with open(config_path) as handle:
                    config_blob = json.load(handle)
                seed = int(config_blob["config"].get("seed", 2020))
                checkpoints[setting][seed] = str(ckpt_path)
    return checkpoints


def summarize(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    summary = {}
    for setting in SETTING_ORDER:
        values = [
            float(row["test_acc_percent"])
            for row in rows
            if row["setting"] == setting
        ]
        if not values:
            continue
        mean = sum(values) / len(values)
        if len(values) > 1:
            var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            std = var ** 0.5
        else:
            std = 0.0
        summary[setting] = {
            "n": len(values),
            "mean_test_acc_percent": mean,
            "std_test_acc_percent": std,
            "min_test_acc_percent": min(values),
            "max_test_acc_percent": max(values),
        }
    return summary


def write_csv(path: str, rows: List[Dict[str, object]]):
    fieldnames = [
        "setting",
        "arch",
        "use_im_loss_train",
        "seed",
        "hidden_size",
        "best_epoch",
        "best_val_acc_percent",
        "test_acc_percent",
        "test_ce_loss",
        "checkpoint",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "setting": row["setting"],
                    "arch": row["arch"],
                    "use_im_loss_train": row["use_im_loss_train"],
                    "seed": row["seed"],
                    "hidden_size": row["hidden_size"],
                    "best_epoch": row["epoch"],
                    "best_val_acc_percent": 100.0 * float(row["best_val_acc"]),
                    "test_acc_percent": row["test_acc_percent"],
                    "test_ce_loss": row["test_ce_loss"],
                    "checkpoint": row["checkpoint"],
                }
            )


def plot_bars(path: str, rows: List[Dict[str, object]], summary: Dict[str, Dict[str, float]]):
    x_positions = list(range(len(SETTING_ORDER)))
    means = [summary[setting]["mean_test_acc_percent"] for setting in SETTING_ORDER]
    stds = [summary[setting]["std_test_acc_percent"] for setting in SETTING_ORDER]
    labels = [SETTING_LABELS[setting] for setting in SETTING_ORDER]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(
        x_positions,
        means,
        yerr=stds,
        capsize=6,
        width=0.62,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        error_kw={"elinewidth": 1.4, "capthick": 1.4},
    )
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            mean + std + 0.55,
            "{:.2f} +/- {:.2f}".format(mean, std),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("SHD test accuracy (%)")
    ax.set_title("SHD temporal SNN best-checkpoint test accuracy, mean +/- std over 5 seeds")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate and summarize SHD temporal SNN multiseed results"
    )
    parser.add_argument("--run_root", default="runs")
    parser.add_argument("--output_dir", default="runs/summary")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoints = find_checkpoints(args.run_root)
    missing = []
    rows = []
    eval_args = SimpleNamespace(
        data_root=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        dt=None,
        t_stop=None,
        compute_activity=False,
        im_target_rate=0.02,
    )

    for setting in SETTING_ORDER:
        for seed in SEED_ORDER:
            checkpoint = checkpoints.get(setting, {}).get(seed)
            if checkpoint is None:
                missing.append((setting, seed))
                continue
            metrics = eval_checkpoint(checkpoint, eval_args)
            metrics["setting"] = setting
            rows.append(metrics)
            print(
                "{} seed {}: test {:.2f}% val {:.2f}%".format(
                    SETTING_LABELS[setting],
                    seed,
                    metrics["test_acc_percent"],
                    100.0 * float(metrics["best_val_acc"]),
                )
            )

    if missing:
        raise RuntimeError("Missing checkpoints: {}".format(missing))

    rows.sort(key=lambda row: (SETTING_ORDER.index(row["setting"]), SEED_ORDER.index(int(row["seed"]))))
    summary = summarize(rows)

    json_path = os.path.join(args.output_dir, "shd_multiseed_best_test_eval.json")
    csv_path = os.path.join(args.output_dir, "shd_multiseed_best_test_eval.csv")
    summary_path = os.path.join(args.output_dir, "shd_multiseed_summary.csv")
    plot_path = os.path.join(args.output_dir, "shd_multiseed_test_acc_bars.png")

    with open(json_path, "w") as handle:
        json.dump({"results": rows, "summary": summary}, handle, indent=2)
    write_csv(csv_path, rows)

    with open(summary_path, "w", newline="") as handle:
        fieldnames = [
            "setting",
            "n",
            "mean_test_acc_percent",
            "std_test_acc_percent",
            "min_test_acc_percent",
            "max_test_acc_percent",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for setting in SETTING_ORDER:
            row = {"setting": setting}
            row.update(summary[setting])
            writer.writerow(row)

    plot_bars(plot_path, rows, summary)

    print("wrote", csv_path)
    print("wrote", summary_path)
    print("wrote", plot_path)


if __name__ == "__main__":
    main()
