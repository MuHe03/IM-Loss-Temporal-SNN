"""Data-free smoke test for the temporal SNN stack.

This verifies that the temporal model, spiking output readout, IM losses, and a
single backward pass work before downloading SHD or running long experiments.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as exc:
    missing = exc.name or "a required package"
    raise SystemExit(
        "Missing dependency '{}'. Install project dependencies with "
        "'python -m pip install -r requirements.txt' first.".format(missing)
    ) from exc

from models.temporal_snn import (
    FeedForwardLIFClassifier,
    firing_rate_stats,
    temporal_im_loss,
    temporal_threshold_im_loss,
)


def run_case(output_mode: str) -> None:
    torch.manual_seed(0)

    time_steps = 12
    batch_size = 4
    input_size = 16
    num_classes = 3

    spikes = (torch.rand(time_steps, batch_size, input_size) < 0.12).float()
    lengths = torch.full((batch_size,), time_steps, dtype=torch.long)
    labels = torch.tensor([0, 1, 2, 1], dtype=torch.long)

    model = FeedForwardLIFClassifier(
        input_size=input_size,
        hidden_sizes=(8,),
        num_classes=num_classes,
        beta=0.9,
        readout_beta=0.9,
        threshold=1.0,
        surrogate_slope=25.0,
        readout="max_membrane",
        output_mode=output_mode,
    )
    criterion = nn.CrossEntropyLoss()

    logits, aux = model(spikes, lengths=lengths, return_spikes=True)
    ce_loss = criterion(logits, labels)
    rate_loss = temporal_im_loss(aux["hidden_spikes"], lengths=lengths)
    threshold_loss = temporal_threshold_im_loss(
        aux["hidden_membranes"], lengths=lengths, threshold=1.0
    )
    loss = ce_loss + 1e-3 * (rate_loss + threshold_loss)

    if output_mode == "spiking_count":
        output_rate_loss = temporal_im_loss([aux["output_spikes"]], lengths=lengths)
        loss = loss + 1e-3 * output_rate_loss

    loss.backward()
    stats = firing_rate_stats(aux["hidden_spikes"])
    print(
        "{}: logits={} ce={:.4f} rate_im={:.6f} threshold_im={:.6f} {}".format(
            output_mode,
            tuple(logits.shape),
            ce_loss.item(),
            rate_loss.item(),
            threshold_loss.item(),
            stats,
        )
    )


def main() -> None:
    print("torch", torch.__version__)
    run_case("nonspiking")
    run_case("spiking_count")
    print("temporal smoke test passed")


if __name__ == "__main__":
    main()
