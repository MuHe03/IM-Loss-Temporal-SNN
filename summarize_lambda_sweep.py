import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib.pyplot as plt


def load_json(path: Path) -> Dict[str, object]:
    with open(path) as handle:
        return json.load(handle)


def latest_rows(dataset: str, loss_type: str, run_root: Path) -> List[Dict[str, object]]:
    root = run_root / dataset.lower() / "lambda_sweep" / loss_type
    rows_by_key: Dict[Tuple[float, int], Tuple[float, Dict[str, object]]] = {}

    for metrics_path in sorted(root.glob("lambda_*/*/metrics.json")):
        config_path = metrics_path.parent / "config.json"
        if not config_path.exists():
            continue

        config_blob = load_json(config_path)
        config = dict(config_blob.get("config", {}))
        meta = dict(config_blob.get("meta", {}))
        metrics_blob = load_json(metrics_path)
        test = dict(metrics_blob.get("test", {}))

        im_lambda = float(config.get("im_loss_weight", 0.0))
        seed = int(config.get("seed", meta.get("seed", 0)))
        key = (im_lambda, seed)
        row = {
            "dataset": dataset,
            "condition": "spiking_count_{}_im".format(loss_type),
            "loss_type": loss_type,
            "lambda": im_lambda,
            "seed": seed,
            "best_epoch": metrics_blob.get("best_epoch", ""),
            "best_val_acc_percent": 100.0 * float(metrics_blob.get("best_val_acc", 0.0)),
            "test_acc_percent": 100.0 * float(test.get("acc", 0.0)),
            "test_ce_loss": test.get("ce_loss", ""),
            "test_im_loss": test.get("im_loss", ""),
            "test_output_im_loss": test.get("output_im_loss", ""),
            "test_firing_rate_output": test.get("firing_rate_output", ""),
            "test_entropy_output": test.get("entropy_output", ""),
            "test_no_output_spike_fraction": test.get("no_output_spike_fraction", ""),
            "test_output_spike_count_mean": test.get("output_spike_count_mean", ""),
            "test_correct_class_output_spike_count_mean": test.get(
                "correct_class_output_spike_count_mean", ""
            ),
            "test_max_competing_output_spike_count_mean": test.get(
                "max_competing_output_spike_count_mean", ""
            ),
            "test_correct_output_gt_competing_fraction": test.get(
                "correct_output_gt_competing_fraction", ""
            ),
            "run_dir": str(metrics_path.parent),
            "checkpoint": metrics_blob.get(
                "test_checkpoint", str(metrics_path.parent / "best.pth")
            ),
        }
        mtime = metrics_path.stat().st_mtime
        if key not in rows_by_key or mtime > rows_by_key[key][0]:
            rows_by_key[key] = (mtime, row)

    rows = [row for _, row in rows_by_key.values()]
    rows.sort(key=lambda row: (float(row["lambda"]), int(row["seed"])))
    return rows


def numeric_values(rows: List[Dict[str, object]], key: str) -> List[float]:
    values = []
    for row in rows:
        value = row.get(key, "")
        if value == "":
            continue
        values.append(float(value))
    return values


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    summary_rows = []
    lambdas = sorted({float(row["lambda"]) for row in rows})
    for im_lambda in lambdas:
        lambda_rows = [row for row in rows if float(row["lambda"]) == im_lambda]
        test_acc = numeric_values(lambda_rows, "test_acc_percent")
        val_acc = numeric_values(lambda_rows, "best_val_acc_percent")
        output_rate = numeric_values(lambda_rows, "test_firing_rate_output")
        entropy = numeric_values(lambda_rows, "test_entropy_output")
        no_output = numeric_values(lambda_rows, "test_no_output_spike_fraction")

        summary_rows.append(
            {
                "lambda": im_lambda,
                "n": len(lambda_rows),
                "mean_best_val_acc_percent": mean(val_acc) if val_acc else 0.0,
                "mean_test_acc_percent": mean(test_acc) if test_acc else 0.0,
                "std_test_acc_percent": stdev(test_acc) if len(test_acc) > 1 else 0.0,
                "mean_test_firing_rate_output": mean(output_rate) if output_rate else "",
                "mean_test_entropy_output": mean(entropy) if entropy else "",
                "mean_test_no_output_spike_fraction": mean(no_output) if no_output else "",
            }
        )
    return summary_rows


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_accuracy(path: Path, summary_rows: List[Dict[str, object]], dataset: str) -> None:
    labels = ["{:.4g}".format(float(row["lambda"])) for row in summary_rows]
    x_positions = list(range(len(summary_rows)))
    val = [float(row["mean_best_val_acc_percent"]) for row in summary_rows]
    test = [float(row["mean_test_acc_percent"]) for row in summary_rows]
    test_std = [float(row["std_test_acc_percent"]) for row in summary_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_positions, val, marker="o", label="validation")
    ax.errorbar(
        x_positions,
        test,
        yerr=test_std,
        marker="s",
        capsize=5,
        label="test",
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("IM loss weight")
    ax.set_ylabel("{} accuracy (%)".format(dataset))
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_diagnostics(path: Path, summary_rows: List[Dict[str, object]]) -> None:
    labels = ["{:.4g}".format(float(row["lambda"])) for row in summary_rows]
    x_positions = list(range(len(summary_rows)))

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label in (
        ("mean_test_firing_rate_output", "output firing rate"),
        ("mean_test_entropy_output", "output entropy"),
        ("mean_test_no_output_spike_fraction", "no-output fraction"),
    ):
        values = [row.get(key, "") for row in summary_rows]
        if any(value != "" for value in values):
            ax.plot(
                x_positions,
                [float(value or 0.0) for value in values],
                marker="o",
                label=label,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("IM loss weight")
    ax.set_ylabel("diagnostic value")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize output-layer IM lambda sweeps")
    parser.add_argument("--dataset", default="Randman", choices=["Randman", "SHD"])
    parser.add_argument("--loss_type", default="threshold", choices=["rate", "threshold"])
    parser.add_argument("--run_root", default="runs")
    parser.add_argument("--output_dir", default="runs/summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = Path(args.output_dir)

    rows = latest_rows(args.dataset, args.loss_type, Path(args.run_root))
    if not rows:
        raise RuntimeError(
            "No lambda-sweep metrics found under {}/{}/lambda_sweep/{}".format(
                args.run_root, args.dataset.lower(), args.loss_type
            )
        )

    summary_rows = summarize(rows)
    selected = max(summary_rows, key=lambda row: float(row["mean_best_val_acc_percent"]))

    prefix = "{}_{}_lambda_sweep".format(args.dataset.lower(), args.loss_type)
    detail_csv = output_dir / "{}_detail.csv".format(prefix)
    summary_csv = output_dir / "{}_summary.csv".format(prefix)
    json_path = output_dir / "{}_summary.json".format(prefix)
    accuracy_plot = output_dir / "{}_accuracy.png".format(prefix)
    diagnostics_plot = output_dir / "{}_diagnostics.png".format(prefix)

    detail_fields = [
        "dataset",
        "condition",
        "loss_type",
        "lambda",
        "seed",
        "best_epoch",
        "best_val_acc_percent",
        "test_acc_percent",
        "test_ce_loss",
        "test_im_loss",
        "test_output_im_loss",
        "test_firing_rate_output",
        "test_entropy_output",
        "test_no_output_spike_fraction",
        "test_output_spike_count_mean",
        "test_correct_class_output_spike_count_mean",
        "test_max_competing_output_spike_count_mean",
        "test_correct_output_gt_competing_fraction",
        "run_dir",
        "checkpoint",
    ]
    summary_fields = [
        "lambda",
        "n",
        "mean_best_val_acc_percent",
        "mean_test_acc_percent",
        "std_test_acc_percent",
        "mean_test_firing_rate_output",
        "mean_test_entropy_output",
        "mean_test_no_output_spike_fraction",
    ]

    write_csv(detail_csv, rows, detail_fields)
    write_csv(summary_csv, summary_rows, summary_fields)
    with open(json_path, "w") as handle:
        json.dump({"results": rows, "summary": summary_rows, "selected": selected}, handle, indent=2)
    plot_accuracy(accuracy_plot, summary_rows, args.dataset)
    plot_diagnostics(diagnostics_plot, summary_rows)

    print(
        "selected lambda {:.4g} by mean validation accuracy {:.2f}%".format(
            float(selected["lambda"]),
            float(selected["mean_best_val_acc_percent"]),
        )
    )
    print("wrote", detail_csv)
    print("wrote", summary_csv)
    print("wrote", json_path)
    print("wrote", accuracy_plot)
    print("wrote", diagnostics_plot)


if __name__ == "__main__":
    main()
