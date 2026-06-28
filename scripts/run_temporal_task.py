"""Cross-platform launcher for local temporal experiment tasks.

This mirrors the bash wrapper task grids but works from PowerShell, cmd, Colab,
Linux, and macOS because it only depends on the Python standard library.
"""

import argparse
import os
import shlex
import subprocess
import sys
from typing import List, Tuple


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def python_bin() -> str:
    return os.environ.get("PYTHON", sys.executable)


def parse_csv_floats(name: str, default: str) -> List[float]:
    values = []
    for raw_value in env(name, default).split(","):
        raw_value = raw_value.strip()
        if raw_value:
            values.append(float(raw_value))
    if not values:
        raise ValueError(f"{name} must contain at least one float value")
    return values


def parse_csv_ints(name: str, default: str) -> List[int]:
    values = []
    for raw_value in env(name, default).split(","):
        raw_value = raw_value.strip()
        if raw_value:
            values.append(int(raw_value))
    if not values:
        raise ValueError(f"{name} must contain at least one integer value")
    return values


def lambda_tag(value: float) -> str:
    text = "{:.8g}".format(value)
    return text.replace("-", "m").replace(".", "p")


def dataset_run_root() -> str:
    return f"{env('RUNS_ROOT', 'runs')}/{env('DATASET', 'Randman').lower()}"


def common_train_args(output_dir: str, seed: int) -> List[str]:
    dataset = env("DATASET", "Randman")
    cmd = [
        python_bin(),
        "train_temporal.py",
        "--dataset",
        dataset,
        "--data_root",
        env("DATA_ROOT", "datasets/SHD"),
        "--output_dir",
        output_dir,
        "--seed",
        str(seed),
        "--epochs",
        env("EPOCHS", "100"),
        "--batch_size",
        env("BATCH_SIZE", "64"),
        "--num_workers",
        env("NUM_WORKERS", "4"),
        "--dt",
        env("DT", "0.005"),
        "--t_stop",
        env("T_STOP", "1.4"),
        "--hidden_size",
        env("HIDDEN_SIZE", "256"),
        "--num_layers",
        env("NUM_LAYERS", "1"),
        "--learning_rate",
        env("LEARNING_RATE", "0.001"),
        "--randman_train_samples",
        env("RANDMAN_TRAIN_SAMPLES", "1024"),
        "--randman_val_samples",
        env("RANDMAN_VAL_SAMPLES", "256"),
        "--randman_test_samples",
        env("RANDMAN_TEST_SAMPLES", "256"),
        "--randman_input_size",
        env("RANDMAN_INPUT_SIZE", "128"),
        "--randman_num_classes",
        env("RANDMAN_NUM_CLASSES", "10"),
        "--randman_num_steps",
        env("RANDMAN_NUM_STEPS", "100"),
        "--randman_manifold_dim",
        env("RANDMAN_MANIFOLD_DIM", "2"),
        "--randman_spike_prob",
        env("RANDMAN_SPIKE_PROB", "0.35"),
        "--randman_noise_rate",
        env("RANDMAN_NOISE_RATE", "0.001"),
        "--randman_jitter_std",
        env("RANDMAN_JITTER_STD", "1.0"),
    ]
    if env("NO_CUDA", "0") == "1":
        cmd.append("--no_cuda")
    if os.environ.get("MAX_TRAIN_BATCHES"):
        cmd.extend(["--max_train_batches", os.environ["MAX_TRAIN_BATCHES"]])
    if os.environ.get("MAX_EVAL_BATCHES"):
        cmd.extend(["--max_eval_batches", os.environ["MAX_EVAL_BATCHES"]])
    return cmd


def temporal_multiseed(task_id: int) -> Tuple[str, List[str]]:
    if task_id < 0 or task_id > 15:
        raise ValueError("temporal_multiseed task_id must be in [0, 15]")
    seeds = [42, 123, 309, 969]
    group, seed_idx = divmod(task_id, 4)
    seed = seeds[seed_idx]

    dataset_root = dataset_run_root()
    settings = [
        ("lif_mlp_baseline", "lif_mlp", f"{dataset_root}/temporal_multiseed/lif_mlp_baseline", []),
        (
            "lif_mlp_imloss",
            "lif_mlp",
            f"{dataset_root}/temporal_multiseed/lif_mlp_imloss",
            [
                "--use_im_loss",
                "--im_loss_type",
                "rate",
                "--im_loss_weight",
                env("IM_LOSS_WEIGHT", "0.001"),
                "--im_target_rate",
                env("IM_TARGET_RATE", "0.02"),
            ],
        ),
        ("rsnn_lif_baseline", "rsnn_lif", f"{dataset_root}/temporal_multiseed/rsnn_lif_baseline", []),
        (
            "rsnn_lif_imloss",
            "rsnn_lif",
            f"{dataset_root}/temporal_multiseed/rsnn_lif_imloss",
            [
                "--use_im_loss",
                "--im_loss_type",
                "rate",
                "--im_loss_weight",
                env("IM_LOSS_WEIGHT", "0.001"),
                "--im_target_rate",
                env("IM_TARGET_RATE", "0.02"),
            ],
        ),
    ]
    name, arch, output_dir, extra = settings[group]
    cmd = common_train_args(output_dir, seed)
    cmd.extend(["--arch", arch, "--readout", env("READOUT", "max_membrane")])
    cmd.extend(extra)
    return f"{name} seed={seed}", cmd


def ff_output_3way(task_id: int) -> Tuple[str, List[str]]:
    if task_id < 0 or task_id > 14:
        raise ValueError("ff_output_3way task_id must be in [0, 14]")
    seeds = [2020, 42, 123, 309, 969]
    group, seed_idx = divmod(task_id, 5)
    seed = seeds[seed_idx]
    readout = env("READOUT", "max_membrane")
    dataset_root = dataset_run_root()

    settings = [
        (
            "nonspiking",
            f"{dataset_root}/ff_output_3way/nonspiking",
            ["--arch", "lif_mlp", "--output_mode", "nonspiking", "--readout", readout],
        ),
        (
            "spiking_count",
            f"{dataset_root}/ff_output_3way/spiking_count",
            ["--arch", "lif_mlp", "--output_mode", "spiking_count", "--readout", readout],
        ),
        (
            "spiking_count_im",
            f"{dataset_root}/ff_output_3way/spiking_count_im",
            [
                "--arch",
                "lif_mlp",
                "--output_mode",
                "spiking_count",
                "--readout",
                readout,
                "--use_im_loss",
                "--im_loss_type",
                "rate",
                "--im_include_output",
                "--im_hidden_weight",
                "0.0",
                "--im_output_weight",
                "1.0",
                "--im_loss_weight",
                env("IM_LOSS_WEIGHT", "0.001"),
                "--im_target_rate",
                env("IM_TARGET_RATE", "0.02"),
            ],
        ),
    ]
    name, output_dir, extra = settings[group]
    cmd = common_train_args(output_dir, seed)
    cmd.extend(extra)
    return f"{name} seed={seed}", cmd


def ff_output_threshold_im(task_id: int) -> Tuple[str, List[str]]:
    if task_id < 0 or task_id > 4:
        raise ValueError("ff_output_threshold_im task_id must be in [0, 4]")
    seeds = [2020, 42, 123, 309, 969]
    seed = seeds[task_id]
    dataset_root = dataset_run_root()
    output_dir = f"{dataset_root}/ff_output_3way/spiking_count_threshold_im"
    cmd = common_train_args(output_dir, seed)
    cmd.extend(
        [
            "--arch",
            "lif_mlp",
            "--output_mode",
            "spiking_count",
            "--readout",
            env("READOUT", "max_membrane"),
            "--use_im_loss",
            "--im_loss_type",
            "threshold",
            "--im_include_output",
            "--im_hidden_weight",
            "0.0",
            "--im_output_weight",
            "1.0",
            "--im_loss_weight",
            env("IM_LOSS_WEIGHT", "0.001"),
        ]
    )
    return f"spiking_count_threshold_im seed={seed}", cmd


def ff_output_lambda_sweep(task_id: int) -> Tuple[str, List[str]]:
    lambdas = parse_csv_floats(
        "IM_LAMBDAS",
        "0.0,0.0001,0.0003,0.001,0.003,0.01",
    )
    seeds = parse_csv_ints("SEEDS", "2020,42,123")
    task_count = len(lambdas) * len(seeds)
    if task_id < 0 or task_id >= task_count:
        raise ValueError(
            "ff_output_lambda_sweep task_id must be in [0, {}]".format(
                task_count - 1
            )
        )

    lambda_idx, seed_idx = divmod(task_id, len(seeds))
    im_lambda = lambdas[lambda_idx]
    seed = seeds[seed_idx]
    loss_type = env("LAMBDA_SWEEP_IM_LOSS_TYPE", "threshold")
    if loss_type not in ("rate", "threshold"):
        raise ValueError("LAMBDA_SWEEP_IM_LOSS_TYPE must be 'rate' or 'threshold'")

    dataset_root = dataset_run_root()
    output_dir = (
        f"{dataset_root}/lambda_sweep/{loss_type}/lambda_{lambda_tag(im_lambda)}"
    )
    cmd = common_train_args(output_dir, seed)
    cmd.extend(
        [
            "--arch",
            "lif_mlp",
            "--output_mode",
            "spiking_count",
            "--readout",
            env("READOUT", "max_membrane"),
            "--use_im_loss",
            "--im_loss_type",
            loss_type,
            "--im_include_output",
            "--im_hidden_weight",
            "0.0",
            "--im_output_weight",
            "1.0",
            "--im_loss_weight",
            str(im_lambda),
        ]
    )
    if loss_type == "rate":
        cmd.extend(["--im_target_rate", env("IM_TARGET_RATE", "0.02")])
    return (
        "spiking_count_{}_im lambda={} seed={}".format(loss_type, im_lambda, seed),
        cmd,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one local temporal experiment task",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "suite",
        choices=[
            "temporal_multiseed",
            "ff_output_3way",
            "ff_output_threshold_im",
            "ff_output_lambda_sweep",
        ],
    )
    parser.add_argument("task_id", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builders = {
        "temporal_multiseed": temporal_multiseed,
        "ff_output_3way": ff_output_3way,
        "ff_output_threshold_im": ff_output_threshold_im,
        "ff_output_lambda_sweep": ff_output_lambda_sweep,
    }
    label, cmd = builders[args.suite](args.task_id)
    print(f"Running {args.suite} task {args.task_id}: {label}")
    print("Command:", shlex.join(cmd))
    if args.dry_run:
        return
    raise SystemExit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
