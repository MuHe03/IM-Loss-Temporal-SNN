import math
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class RandmanSpikeDataset(Dataset):
    """Deterministic random-manifold temporal spike classification dataset.

    Each class owns a fixed random projection from a low-dimensional latent
    variable to input-neuron spike times. Each sample draws a latent point on
    that class manifold, converts projected coordinates into spike times, and
    adds optional sparse background spikes.
    """

    _SPLIT_OFFSETS = {"train": 0, "val": 1, "test": 2}

    def __init__(
        self,
        split: str = "train",
        num_samples: int = 1024,
        num_units: int = 128,
        num_classes: int = 10,
        num_steps: int = 100,
        manifold_dim: int = 2,
        spike_prob: float = 0.35,
        noise_rate: float = 0.001,
        jitter_std: float = 1.0,
        seed: int = 2020,
    ):
        if split not in self._SPLIT_OFFSETS:
            raise ValueError("split must be 'train', 'val', or 'test'")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if num_units <= 0:
            raise ValueError("num_units must be positive")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1")
        if num_steps <= 1:
            raise ValueError("num_steps must be greater than 1")
        if manifold_dim <= 0:
            raise ValueError("manifold_dim must be positive")
        if not 0.0 < spike_prob <= 1.0:
            raise ValueError("spike_prob must be in (0, 1]")
        if noise_rate < 0.0:
            raise ValueError("noise_rate must be non-negative")
        if jitter_std < 0.0:
            raise ValueError("jitter_std must be non-negative")

        self.split = split
        self.num_samples = int(num_samples)
        self.num_units = int(num_units)
        self.num_classes = int(num_classes)
        self.num_steps = int(num_steps)
        self.manifold_dim = int(manifold_dim)
        self.spike_prob = float(spike_prob)
        self.noise_rate = float(noise_rate)
        self.jitter_std = float(jitter_std)
        self.seed = int(seed)
        self.split_seed = self.seed + 100_003 * self._SPLIT_OFFSETS[split]

        prototype_gen = torch.Generator().manual_seed(self.seed)
        self.class_weights = torch.randn(
            self.num_classes,
            self.num_units,
            self.manifold_dim,
            generator=prototype_gen,
        )
        self.class_phase = 2.0 * math.pi * torch.rand(
            self.num_classes, self.num_units, generator=prototype_gen
        )
        self.class_time_shift = torch.rand(
            self.num_classes, self.num_units, generator=prototype_gen
        )
        self.unit_gain = 0.5 + torch.rand(self.num_units, generator=prototype_gen)

        label_gen = torch.Generator().manual_seed(self.split_seed)
        labels = torch.arange(self.num_samples, dtype=torch.long) % self.num_classes
        self.labels = labels[torch.randperm(self.num_samples, generator=label_gen)]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if index < 0 or index >= self.num_samples:
            raise IndexError(index)

        generator = torch.Generator().manual_seed(self.split_seed + int(index))
        label = self.labels[index]
        label_idx = int(label.item())
        latent = torch.rand(self.manifold_dim, generator=generator)

        projection = (self.class_weights[label_idx] * latent).sum(dim=1)
        phase = projection * self.unit_gain + self.class_phase[label_idx]
        warped = 0.5 + 0.5 * torch.sin(phase)
        shifted = (warped + self.class_time_shift[label_idx]) % 1.0
        spike_times = shifted * float(self.num_steps - 1)

        if self.jitter_std > 0:
            spike_times = spike_times + torch.randn(
                self.num_units, generator=generator
            ) * self.jitter_std
        spike_indices = spike_times.round().long().clamp(0, self.num_steps - 1)

        active = torch.rand(self.num_units, generator=generator) < self.spike_prob
        unit_indices = torch.arange(self.num_units)[active]
        spikes = torch.zeros(self.num_steps, self.num_units, dtype=torch.float32)
        if unit_indices.numel() > 0:
            spikes[spike_indices[active], unit_indices] = 1.0

        if self.noise_rate > 0:
            noise = torch.rand(
                self.num_steps, self.num_units, generator=generator
            ) < self.noise_rate
            spikes = torch.maximum(spikes, noise.to(spikes.dtype))

        return {
            "spikes": spikes,
            "label": label,
            "length": torch.tensor(self.num_steps, dtype=torch.long),
        }


class RandmanCollate:
    def __call__(self, samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        spikes = torch.stack([sample["spikes"] for sample in samples], dim=1)
        labels = torch.stack([sample["label"] for sample in samples]).long()
        lengths = torch.stack([sample["length"] for sample in samples]).long()
        return {
            "spikes": spikes,
            "labels": labels,
            "lengths": lengths,
            "event_lengths": spikes.sum(dim=(0, 2)).long(),
        }


def build_randman_loaders(
    batch_size: int = 64,
    num_workers: int = 0,
    train_samples: int = 1024,
    val_samples: int = 256,
    test_samples: int = 256,
    num_units: int = 128,
    num_classes: int = 10,
    num_steps: int = 100,
    manifold_dim: int = 2,
    spike_prob: float = 0.35,
    noise_rate: float = 0.001,
    jitter_std: float = 1.0,
    seed: int = 2020,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, object]]:
    train_dataset = RandmanSpikeDataset(
        split="train",
        num_samples=train_samples,
        num_units=num_units,
        num_classes=num_classes,
        num_steps=num_steps,
        manifold_dim=manifold_dim,
        spike_prob=spike_prob,
        noise_rate=noise_rate,
        jitter_std=jitter_std,
        seed=seed,
    )
    val_dataset = RandmanSpikeDataset(
        split="val",
        num_samples=val_samples,
        num_units=num_units,
        num_classes=num_classes,
        num_steps=num_steps,
        manifold_dim=manifold_dim,
        spike_prob=spike_prob,
        noise_rate=noise_rate,
        jitter_std=jitter_std,
        seed=seed,
    )
    test_dataset = RandmanSpikeDataset(
        split="test",
        num_samples=test_samples,
        num_units=num_units,
        num_classes=num_classes,
        num_steps=num_steps,
        manifold_dim=manifold_dim,
        spike_prob=spike_prob,
        noise_rate=noise_rate,
        jitter_std=jitter_std,
        seed=seed,
    )
    collate_fn = RandmanCollate()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    meta = {
        "dataset": "Randman",
        "num_inputs": num_units,
        "num_classes": num_classes,
        "num_steps": num_steps,
        "layout": "TBN",
        "train_size": train_samples,
        "val_size": val_samples,
        "test_size": test_samples,
        "manifold_dim": manifold_dim,
        "spike_prob": spike_prob,
        "noise_rate": noise_rate,
        "jitter_std": jitter_std,
        "seed": seed,
        "input_is_native_spike_train": True,
    }
    return train_loader, val_loader, test_loader, meta
