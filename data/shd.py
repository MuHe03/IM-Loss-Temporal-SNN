import math
import os
from typing import Dict, List, Optional, Tuple

import h5py
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class SHDEventDataset(Dataset):
    """SHD HDF5 dataset that returns native spike events.

    The HDF5 handle is opened lazily per worker. Each sample keeps the original
    event semantics: spike times in seconds and cochlea unit ids.
    """

    def __init__(self, root: str = "datasets/SHD", split: str = "train"):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        self.root = root
        self.split = split
        self.path = os.path.join(root, "shd_{}.h5".format(split))
        if not os.path.exists(self.path):
            gz_path = self.path + ".gz"
            if os.path.exists(gz_path):
                raise FileNotFoundError(
                    "{} is missing. Found {}, please decompress it first.".format(
                        self.path, gz_path
                    )
                )
            raise FileNotFoundError(self.path)

        with h5py.File(self.path, "r") as handle:
            self._length = len(handle["labels"])
            self.keys = [
                key.decode("utf-8") if hasattr(key, "decode") else str(key)
                for key in handle["extra"]["keys"][:]
            ]

        self._handle = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def _ensure_open(self):
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        handle = self._ensure_open()
        times = torch.as_tensor(
            handle["spikes"]["times"][index], dtype=torch.float32
        )
        units = torch.as_tensor(
            handle["spikes"]["units"][index], dtype=torch.long
        )
        label = torch.tensor(int(handle["labels"][index]), dtype=torch.long)
        return {"times": times, "units": units, "label": label}


class SHDCollate:
    """Convert event lists into a fixed-window dense temporal spike batch."""

    def __init__(
        self,
        dt: float = 0.005,
        t_stop: float = 1.4,
        num_units: int = 700,
        input_mode: str = "binary",
    ):
        if dt <= 0:
            raise ValueError("dt must be positive")
        if t_stop <= 0:
            raise ValueError("t_stop must be positive")
        if input_mode not in ("binary", "count"):
            raise ValueError("input_mode must be 'binary' or 'count'")
        self.dt = float(dt)
        self.t_stop = float(t_stop)
        self.num_units = int(num_units)
        self.input_mode = input_mode
        self.num_steps = int(math.ceil(self.t_stop / self.dt))

    def __call__(self, samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch_size = len(samples)
        spikes = torch.zeros(
            self.num_steps, batch_size, self.num_units, dtype=torch.float32
        )
        labels = torch.empty(batch_size, dtype=torch.long)
        event_lengths = torch.empty(batch_size, dtype=torch.long)
        lengths = torch.full((batch_size,), self.num_steps, dtype=torch.long)

        for batch_idx, sample in enumerate(samples):
            labels[batch_idx] = sample["label"]
            times = sample["times"]
            units = sample["units"]
            event_lengths[batch_idx] = times.numel()

            if times.numel() == 0:
                continue

            time_idx = torch.floor(times / self.dt).long()
            valid = (
                (time_idx >= 0)
                & (time_idx < self.num_steps)
                & (units >= 0)
                & (units < self.num_units)
            )
            if not torch.any(valid):
                continue

            time_idx = time_idx[valid]
            units = units[valid]
            batch_indices = torch.full_like(time_idx, batch_idx)
            if self.input_mode == "binary":
                spikes[time_idx, batch_indices, units] = 1.0
            else:
                values = torch.ones_like(time_idx, dtype=spikes.dtype)
                spikes.index_put_(
                    (time_idx, batch_indices, units), values, accumulate=True
                )

        return {
            "spikes": spikes,
            "labels": labels,
            "lengths": lengths,
            "event_lengths": event_lengths,
        }


def build_shd_loaders(
    root: str = "datasets/SHD",
    batch_size: int = 64,
    num_workers: int = 4,
    dt: float = 0.005,
    t_stop: float = 1.4,
    num_units: int = 700,
    input_mode: str = "binary",
    valid_split: float = 0.1,
    seed: int = 2020,
    pin_memory: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader, Dict[str, object]]:
    train_dataset = SHDEventDataset(root=root, split="train")
    test_dataset = SHDEventDataset(root=root, split="test")

    if valid_split < 0 or valid_split >= 1:
        raise ValueError("valid_split must be in [0, 1)")

    val_dataset = None
    if valid_split > 0:
        val_len = int(round(len(train_dataset) * valid_split))
        train_len = len(train_dataset) - val_len
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = random_split(
            train_dataset, [train_len, val_len], generator=generator
        )

    collate_fn = SHDCollate(
        dt=dt, t_stop=t_stop, num_units=num_units, input_mode=input_mode
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = None
    if val_dataset is not None:
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
        "dataset": "SHD",
        "num_inputs": num_units,
        "num_classes": 20,
        "dt": dt,
        "t_stop": t_stop,
        "num_steps": collate_fn.num_steps,
        "layout": "TBN",
        "input_mode": input_mode,
        "train_size": len(train_dataset),
        "val_size": 0 if val_dataset is None else len(val_dataset),
        "test_size": len(test_dataset),
        "keys": getattr(
            train_dataset.dataset if hasattr(train_dataset, "dataset") else train_dataset,
            "keys",
            [],
        ),
        "input_is_native_spike_train": True,
    }
    return train_loader, val_loader, test_loader, meta
