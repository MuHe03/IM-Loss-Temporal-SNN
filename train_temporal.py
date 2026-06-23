import argparse
import datetime
import json
import os
import random
from typing import Dict, Optional

import torch
import torch.nn as nn

from data.shd import build_shd_loaders
from models.temporal_snn import (
    FeedForwardLIFClassifier,
    RecurrentLIFClassifier,
    firing_rate_stats,
    temporal_im_loss,
    temporal_threshold_im_loss,
)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def make_optimizer(args, model):
    if args.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    raise ValueError("unsupported optimizer {}".format(args.optimizer))


def run_epoch(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    args,
    optimizer: Optional[torch.optim.Optimizer] = None,
    max_batches: int = 0,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    ce_meter = AverageMeter()
    im_meter = AverageMeter()
    total_meter = AverageMeter()
    acc_meter = AverageMeter()
    firing_meters: Dict[str, AverageMeter] = {}
    output_im_meter = AverageMeter()

    for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break

        spikes = batch["spikes"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits, aux = model(spikes, lengths=lengths, return_spikes=args.use_im_loss)
            ce_loss = criterion(logits, labels)
            if args.use_im_loss:
                if args.im_loss_type == "threshold":
                    hidden_im_loss = temporal_threshold_im_loss(
                        aux["hidden_membranes"],
                        lengths=lengths,
                        threshold=args.threshold,
                    )
                else:
                    hidden_im_loss = temporal_im_loss(
                        aux["hidden_spikes"],
                        lengths=lengths,
                        target_rate=args.im_target_rate,
                    )
                output_im_loss = ce_loss.detach().new_tensor(0.0)
                if args.im_include_output:
                    if args.im_loss_type == "threshold" and torch.is_tensor(
                        aux.get("output_membranes")
                    ):
                        output_im_loss = temporal_threshold_im_loss(
                            [aux["output_membranes"]],
                            lengths=lengths,
                            threshold=args.threshold,
                        )
                    elif torch.is_tensor(aux.get("output_spikes")):
                        output_im_loss = temporal_im_loss(
                            [aux["output_spikes"]],
                            lengths=lengths,
                            target_rate=args.im_target_rate,
                        )
                im_loss = (
                    args.im_hidden_weight * hidden_im_loss
                    + args.im_output_weight * output_im_loss
                )
                loss = ce_loss + args.im_loss_weight * im_loss
                batch_firing_stats = firing_rate_stats(aux["hidden_spikes"])
                batch_firing_stats.update(output_firing_stats(aux.get("output_spikes", [])))
            else:
                im_loss = ce_loss.detach().new_tensor(0.0)
                output_im_loss = ce_loss.detach().new_tensor(0.0)
                loss = ce_loss
                batch_firing_stats = {}

            if is_train:
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

        batch_size = labels.size(0)
        ce_meter.update(ce_loss.item(), batch_size)
        im_meter.update(im_loss.item(), batch_size)
        output_im_meter.update(output_im_loss.item(), batch_size)
        total_meter.update(loss.item(), batch_size)
        acc_meter.update(accuracy(logits.detach(), labels), batch_size)
        for name, value in batch_firing_stats.items():
            firing_meters.setdefault(name, AverageMeter()).update(value, batch_size)

        if is_train and args.log_interval > 0 and (batch_idx + 1) % args.log_interval == 0:
            print(
                "batch {}/{} ce {:.4f} im {:.6f} loss {:.4f} acc {:.2f}".format(
                    batch_idx + 1,
                    len(loader),
                    ce_meter.avg,
                    im_meter.avg,
                    total_meter.avg,
                    acc_meter.avg * 100.0,
                )
            )

    metrics = {
        "ce_loss": ce_meter.avg,
        "im_loss": im_meter.avg,
        "output_im_loss": output_im_meter.avg,
        "total_loss": total_meter.avg,
        "acc": acc_meter.avg,
    }
    for name, meter in firing_meters.items():
        metrics[name] = meter.avg
    return metrics


def save_checkpoint(path: str, model, optimizer, scheduler, args, meta, epoch, best_val_acc):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "config": vars(args),
        "meta": meta,
        "epoch": epoch,
        "best_val_acc": best_val_acc,
    }
    torch.save(checkpoint, path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Temporal-native SHD LIF/RSNN baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="SHD", choices=["SHD"])
    parser.add_argument("--data_root", default="datasets/SHD")
    parser.add_argument("--output_dir", default="runs/temporal")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2020)

    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--t_stop", type=float, default=1.4)
    parser.add_argument("--input_mode", choices=["binary", "count"], default="binary")
    parser.add_argument("--valid_split", type=float, default=0.1)

    parser.add_argument("--arch", choices=["lif_mlp", "rsnn_lif"], default="lif_mlp")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--readout_beta", type=float, default=0.95)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--surrogate_slope", type=float, default=25.0)
    parser.add_argument("--recurrent_scale", type=float, default=0.5)
    parser.add_argument(
        "--readout", choices=["mean_membrane", "last_membrane"], default="mean_membrane"
    )
    parser.add_argument(
        "--output_mode",
        choices=["nonspiking", "spiking_count"],
        default="nonspiking",
    )

    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adamw")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--cosine_lr", action="store_true")

    parser.add_argument("--use_im_loss", action="store_true")
    parser.add_argument(
        "--im_loss_type",
        choices=["rate", "threshold"],
        default="rate",
        help="rate matches the previous spike-rate regularizer; threshold uses the paper-style membrane-threshold loss",
    )
    parser.add_argument("--im_loss_weight", type=float, default=1e-3)
    parser.add_argument("--im_hidden_weight", type=float, default=1.0)
    parser.add_argument("--im_include_output", action="store_true")
    parser.add_argument("--im_output_weight", type=float, default=1.0)
    parser.add_argument("--im_target_rate", type=float, default=0.02)

    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda:0" if use_cuda else "cpu")

    train_loader, val_loader, test_loader, meta = build_shd_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dt=args.dt,
        t_stop=args.t_stop,
        input_mode=args.input_mode,
        valid_split=args.valid_split,
        seed=args.seed,
        pin_memory=use_cuda,
    )

    hidden_sizes = [args.hidden_size] * args.num_layers
    if args.arch == "lif_mlp":
        model = FeedForwardLIFClassifier(
            input_size=meta["num_inputs"],
            hidden_sizes=hidden_sizes,
            num_classes=meta["num_classes"],
            beta=args.beta,
            readout_beta=args.readout_beta,
            threshold=args.threshold,
            surrogate_slope=args.surrogate_slope,
            readout=args.readout,
            output_mode=args.output_mode,
        )
    elif args.arch == "rsnn_lif":
        model = RecurrentLIFClassifier(
            input_size=meta["num_inputs"],
            hidden_sizes=hidden_sizes,
            num_classes=meta["num_classes"],
            beta=args.beta,
            readout_beta=args.readout_beta,
            threshold=args.threshold,
            surrogate_slope=args.surrogate_slope,
            readout=args.readout,
            recurrent_scale=args.recurrent_scale,
        )
    else:
        raise ValueError("unsupported arch {}".format(args.arch))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(args, model)
    scheduler = None
    if args.cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, eta_min=0.0, T_max=args.epochs
        )

    im_tag = "im" if args.use_im_loss else "noim"
    output_tag = "spkout" if args.output_mode == "spiking_count" else args.readout
    run_name = "{}_{}_{}_{}_seed{}_dt{}_{}".format(
        args.dataset.lower(),
        args.arch,
        output_tag,
        im_tag,
        args.seed,
        args.dt,
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as handle:
        json.dump({"config": vars(args), "meta": meta}, handle, indent=2)

    print("device:", device)
    print("run_dir:", run_dir)
    print("meta:", meta)
    print(model)

    eval_loader = val_loader if val_loader is not None else test_loader
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            args,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
        )
        val_metrics = run_epoch(
            model,
            eval_loader,
            criterion,
            device,
            args,
            optimizer=None,
            max_batches=args.max_eval_batches,
        )
        if scheduler is not None:
            scheduler.step()

        print(
            "epoch {}/{} train acc {:.2f} ce {:.4f} im {:.6f} | val acc {:.2f} ce {:.4f}".format(
                epoch,
                args.epochs,
                train_metrics["acc"] * 100.0,
                train_metrics["ce_loss"],
                train_metrics["im_loss"],
                val_metrics["acc"] * 100.0,
                val_metrics["ce_loss"],
            )
        )

        save_checkpoint(
            os.path.join(run_dir, "last.pth"),
            model,
            optimizer,
            scheduler,
            args,
            meta,
            epoch,
            best_val_acc,
        )
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            save_checkpoint(
                os.path.join(run_dir, "best.pth"),
                model,
                optimizer,
                scheduler,
                args,
                meta,
                epoch,
                best_val_acc,
            )

    test_metrics = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        args,
        optimizer=None,
        max_batches=args.max_eval_batches,
    )
    with open(os.path.join(run_dir, "metrics.json"), "w") as handle:
        json.dump({"test": test_metrics, "best_val_acc": best_val_acc}, handle, indent=2)
    print(
        "test acc {:.2f} ce {:.4f}".format(
            test_metrics["acc"] * 100.0, test_metrics["ce_loss"]
        )
    )


if __name__ == "__main__":
    main()
