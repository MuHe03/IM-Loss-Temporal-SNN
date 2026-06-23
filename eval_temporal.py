import argparse
import json
import os
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.shd import SHDCollate, SHDEventDataset
from models.temporal_snn import (
    FeedForwardLIFClassifier,
    RecurrentLIFClassifier,
    firing_rate_stats,
    temporal_im_loss,
    temporal_threshold_im_loss,
)


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int):
        self.total += float(value) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return predictions.eq(labels).float().mean().item()


def output_firing_stats(output_spikes: torch.Tensor) -> Dict[str, float]:
    if not torch.is_tensor(output_spikes) or output_spikes.numel() == 0:
        return {}
    class_rates = output_spikes.detach().mean(dim=(0, 1))
    return {
        "firing_rate_output": class_rates.mean().item(),
        "silent_neuron_ratio_output": (class_rates <= 0).float().mean().item(),
    }


def build_model(config: Dict[str, object], meta: Dict[str, object]) -> nn.Module:
    arch = config.get("arch", "lif_mlp")

    hidden_size = int(config.get("hidden_size", 256))
    num_layers = int(config.get("num_layers", 1))
    hidden_sizes = [hidden_size] * num_layers

    common_kwargs = {
        "input_size": int(meta.get("num_inputs", 700)),
        "hidden_sizes": hidden_sizes,
        "num_classes": int(meta.get("num_classes", 20)),
        "beta": float(config.get("beta", 0.95)),
        "readout_beta": float(config.get("readout_beta", 0.95)),
        "threshold": float(config.get("threshold", 1.0)),
        "surrogate_slope": float(config.get("surrogate_slope", 25.0)),
        "readout": str(config.get("readout", "mean_membrane")),
    }
    if arch == "lif_mlp":
        return FeedForwardLIFClassifier(
            **common_kwargs,
            output_mode=str(config.get("output_mode", "nonspiking")),
        )
    if arch == "rsnn_lif":
        return RecurrentLIFClassifier(
            **common_kwargs,
            recurrent_scale=float(config.get("recurrent_scale", 0.5)),
        )
    raise ValueError("unsupported temporal arch: {}".format(arch))


def evaluate(
    model,
    loader,
    criterion,
    device,
    compute_activity,
    im_loss_type,
    target_rate,
    threshold,
    include_output=False,
    output_weight=1.0,
):
    model.eval()
    ce_meter = AverageMeter()
    acc_meter = AverageMeter()
    im_meter = AverageMeter()
    output_im_meter = AverageMeter()
    firing_meters: Dict[str, AverageMeter] = {}

    with torch.no_grad():
        for batch in loader:
            spikes = batch["spikes"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)

            logits, aux = model(
                spikes, lengths=lengths, return_spikes=compute_activity
            )
            ce_loss = criterion(logits, labels)
            batch_size = labels.size(0)

            ce_meter.update(ce_loss.item(), batch_size)
            acc_meter.update(accuracy(logits, labels), batch_size)

            if compute_activity:
                if im_loss_type == "threshold":
                    hidden_im_loss = temporal_threshold_im_loss(
                        aux["hidden_membranes"],
                        lengths=lengths,
                        threshold=threshold,
                    )
                else:
                    hidden_im_loss = temporal_im_loss(
                        aux["hidden_spikes"], lengths=lengths, target_rate=target_rate
                    )
                output_im_loss = ce_loss.detach().new_tensor(0.0)
                if include_output:
                    if im_loss_type == "threshold" and torch.is_tensor(
                        aux.get("output_membranes")
                    ):
                        output_im_loss = temporal_threshold_im_loss(
                            [aux["output_membranes"]],
                            lengths=lengths,
                            threshold=threshold,
                        )
                    elif torch.is_tensor(aux.get("output_spikes")):
                        output_im_loss = temporal_im_loss(
                            [aux["output_spikes"]],
                            lengths=lengths,
                            target_rate=target_rate,
                        )
                im_loss = hidden_im_loss + output_weight * output_im_loss
                im_meter.update(im_loss.item(), batch_size)
                output_im_meter.update(output_im_loss.item(), batch_size)
                for name, value in firing_rate_stats(aux["hidden_spikes"]).items():
                    firing_meters.setdefault(name, AverageMeter()).update(
                        value, batch_size
                    )
                for name, value in output_firing_stats(aux.get("output_spikes", [])).items():
                    firing_meters.setdefault(name, AverageMeter()).update(
                        value, batch_size
                    )

    metrics = {
        "test_acc": acc_meter.avg,
        "test_acc_percent": acc_meter.avg * 100.0,
        "test_ce_loss": ce_meter.avg,
    }
    if compute_activity:
        metrics["test_im_loss"] = im_meter.avg
        metrics["test_output_im_loss"] = output_im_meter.avg
        for name, meter in firing_meters.items():
            metrics["test_{}".format(name)] = meter.avg
    return metrics


def eval_checkpoint(path: str, args) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    meta = checkpoint.get("meta", {})

    data_root = args.data_root or str(config.get("data_root", "datasets/SHD"))
    dt = args.dt if args.dt is not None else float(meta.get("dt", config.get("dt", 0.005)))
    t_stop = (
        args.t_stop
        if args.t_stop is not None
        else float(meta.get("t_stop", config.get("t_stop", 1.4)))
    )
    input_mode = str(meta.get("input_mode", config.get("input_mode", "binary")))

    dataset = SHDEventDataset(root=data_root, split="test")
    collate_fn = SHDCollate(
        dt=dt,
        t_stop=t_stop,
        num_units=int(meta.get("num_inputs", 700)),
        input_mode=input_mode,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=collate_fn,
    )

    model = build_model(config, meta).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    criterion = nn.CrossEntropyLoss()
    compute_activity = args.compute_activity or bool(config.get("use_im_loss", False))
    im_loss_type = str(config.get("im_loss_type", "rate"))
    target_rate = float(config.get("im_target_rate", args.im_target_rate))
    threshold = float(config.get("threshold", 1.0))
    include_output = bool(config.get("im_include_output", False))
    output_weight = float(config.get("im_output_weight", 1.0))

    metrics = evaluate(
        model,
        loader,
        criterion,
        torch.device(args.device),
        compute_activity=compute_activity,
        im_loss_type=im_loss_type,
        target_rate=target_rate,
        threshold=threshold,
        include_output=include_output,
        output_weight=output_weight,
    )
    metrics.update(
        {
            "checkpoint": path,
            "arch": config.get("arch", "lif_mlp"),
            "seed": config.get("seed"),
            "hidden_size": config.get("hidden_size"),
            "num_layers": config.get("num_layers"),
            "output_mode": config.get("output_mode", "nonspiking"),
            "readout": config.get("readout", "mean_membrane"),
            "epoch": checkpoint.get("epoch"),
            "best_val_acc": checkpoint.get("best_val_acc"),
            "dt": dt,
            "t_stop": t_stop,
            "num_steps": collate_fn.num_steps,
            "input_mode": input_mode,
            "use_im_loss_train": bool(config.get("use_im_loss", False)),
            "im_loss_weight_train": float(config.get("im_loss_weight", 0.0)),
            "im_loss_type_train": im_loss_type,
            "im_hidden_weight_train": float(config.get("im_hidden_weight", 1.0)),
            "im_include_output_train": bool(config.get("im_include_output", False)),
            "im_output_weight_train": float(config.get("im_output_weight", 1.0)),
            "im_target_rate_train": target_rate,
        }
    )
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate temporal-native SHD checkpoints on the test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--t_stop", type=float, default=None)
    parser.add_argument("--compute_activity", action="store_true")
    parser.add_argument("--im_target_rate", type=float, default=0.02)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    results: List[Dict[str, object]] = []
    for checkpoint in args.checkpoints:
        metrics = eval_checkpoint(checkpoint, args)
        results.append(metrics)
        print(
            "{} | test acc {:.2f}% ce {:.4f} epoch {} best_val {:.2f}%".format(
                checkpoint,
                metrics["test_acc_percent"],
                metrics["test_ce_loss"],
                metrics["epoch"],
                100.0 * float(metrics["best_val_acc"] or 0.0),
            )
        )

    if args.output is not None:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
