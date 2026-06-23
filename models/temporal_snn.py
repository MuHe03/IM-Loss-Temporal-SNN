from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class FastSigmoidSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, slope: float):
        ctx.save_for_backward(input_tensor)
        ctx.slope = slope
        return (input_tensor > 0).to(input_tensor.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (input_tensor,) = ctx.saved_tensors
        slope = ctx.slope
        grad = grad_output / (slope * input_tensor.abs() + 1.0).pow(2)
        return grad, None


def spike_fn(input_tensor: torch.Tensor, slope: float = 25.0) -> torch.Tensor:
    return FastSigmoidSpike.apply(input_tensor, slope)


class FeedForwardLIFClassifier(nn.Module):
    """Feed-forward LIF SNN for native temporal spike batches.

    Input layout is time-first: [T, B, input_size]. Hidden layers are spiking
    LIF layers; the readout is a non-spiking leaky membrane.
    """

    def __init__(
        self,
        input_size: int = 700,
        hidden_sizes: Sequence[int] = (256,),
        num_classes: int = 20,
        beta: float = 0.95,
        readout_beta: float = 0.95,
        threshold: float = 1.0,
        surrogate_slope: float = 25.0,
        readout: str = "mean_membrane",
        output_mode: str = "nonspiking",
    ):
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer")
        if readout not in ("mean_membrane", "last_membrane"):
            raise ValueError("readout must be 'mean_membrane' or 'last_membrane'")
        if output_mode not in ("nonspiking", "spiking_count"):
            raise ValueError("output_mode must be 'nonspiking' or 'spiking_count'")

        self.input_size = int(input_size)
        self.hidden_sizes = [int(size) for size in hidden_sizes]
        self.num_classes = int(num_classes)
        self.beta = float(beta)
        self.readout_beta = float(readout_beta)
        self.threshold = float(threshold)
        self.surrogate_slope = float(surrogate_slope)
        self.readout = readout
        self.output_mode = output_mode

        layer_sizes = [self.input_size] + self.hidden_sizes
        self.layers = nn.ModuleList(
            nn.Linear(layer_sizes[idx], layer_sizes[idx + 1])
            for idx in range(len(self.hidden_sizes))
        )
        self.classifier = nn.Linear(self.hidden_sizes[-1], self.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def _readout_logits(
        self, readout_trace: torch.Tensor, lengths: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.readout == "last_membrane":
            if lengths is None:
                return readout_trace[-1]
            last_idx = lengths.clamp(1, readout_trace.size(0)) - 1
            batch_idx = torch.arange(readout_trace.size(1), device=readout_trace.device)
            return readout_trace[last_idx, batch_idx]

        if lengths is None:
            return readout_trace.mean(dim=0)

        steps = torch.arange(readout_trace.size(0), device=readout_trace.device)
        valid = steps[:, None] < lengths.clamp(1, readout_trace.size(0))[None, :]
        valid = valid.to(readout_trace.dtype).unsqueeze(-1)
        summed = (readout_trace * valid).sum(dim=0)
        denom = lengths.clamp(1, readout_trace.size(0)).to(readout_trace.dtype)
        return summed / denom[:, None]

    def _spike_count_logits(
        self, output_spikes: torch.Tensor, lengths: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if lengths is None:
            return output_spikes.sum(dim=0)

        steps = torch.arange(output_spikes.size(0), device=output_spikes.device)
        valid = steps[:, None] < lengths.clamp(1, output_spikes.size(0))[None, :]
        valid = valid.to(output_spikes.dtype).unsqueeze(-1)
        return (output_spikes * valid).sum(dim=0)

    def forward(
        self,
        spikes: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_spikes: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        if spikes.dim() != 3:
            raise ValueError("expected spikes with shape [T, B, N]")
        if spikes.size(-1) != self.input_size:
            raise ValueError(
                "expected input_size {}, got {}".format(self.input_size, spikes.size(-1))
            )

        time_steps, batch_size, _ = spikes.shape
        device = spikes.device
        dtype = spikes.dtype
        hidden_mem = [
            torch.zeros(batch_size, size, device=device, dtype=dtype)
            for size in self.hidden_sizes
        ]
        readout_mem = torch.zeros(
            batch_size, self.num_classes, device=device, dtype=dtype
        )

        readout_trace = []
        output_trace = []
        output_membrane_trace = []
        hidden_traces: List[List[torch.Tensor]] = [[] for _ in self.hidden_sizes]
        hidden_membrane_traces: List[List[torch.Tensor]] = [
            [] for _ in self.hidden_sizes
        ]

        for step in range(time_steps):
            x = spikes[step]
            for layer_idx, layer in enumerate(self.layers):
                hidden_mem[layer_idx] = self.beta * hidden_mem[layer_idx] + layer(x)
                pre_reset_mem = hidden_mem[layer_idx]
                z = spike_fn(hidden_mem[layer_idx] - self.threshold, self.surrogate_slope)
                hidden_mem[layer_idx] = hidden_mem[layer_idx] - z.detach() * self.threshold
                x = z
                if return_spikes:
                    hidden_traces[layer_idx].append(z)
                    hidden_membrane_traces[layer_idx].append(pre_reset_mem)

            readout_mem = self.readout_beta * readout_mem + self.classifier(x)
            if self.output_mode == "spiking_count":
                output_pre_reset_mem = readout_mem
                out_z = spike_fn(
                    readout_mem - self.threshold,
                    self.surrogate_slope,
                )
                readout_mem = readout_mem - out_z.detach() * self.threshold
                output_trace.append(out_z)
                if return_spikes:
                    output_membrane_trace.append(output_pre_reset_mem)
            readout_trace.append(readout_mem)

        readout_trace_tensor = torch.stack(readout_trace, dim=0)
        output_trace_tensor = None
        if self.output_mode == "spiking_count":
            output_trace_tensor = torch.stack(output_trace, dim=0)
            logits = self._spike_count_logits(output_trace_tensor, lengths)
        else:
            logits = self._readout_logits(readout_trace_tensor, lengths)

        aux: Dict[str, object] = {"readout_trace": readout_trace_tensor}
        if return_spikes:
            aux["hidden_spikes"] = [
                torch.stack(layer_trace, dim=0) for layer_trace in hidden_traces
            ]
            aux["hidden_membranes"] = [
                torch.stack(layer_trace, dim=0)
                for layer_trace in hidden_membrane_traces
            ]
            if output_trace_tensor is not None:
                aux["output_spikes"] = output_trace_tensor
                aux["output_membranes"] = torch.stack(output_membrane_trace, dim=0)
        else:
            aux["hidden_spikes"] = []
            aux["hidden_membranes"] = []
            if output_trace_tensor is not None:
                aux["output_spikes"] = []
                aux["output_membranes"] = []
        return logits, aux


class RecurrentLIFClassifier(nn.Module):
    """Recurrent LIF SNN for native temporal spike batches.

    Each hidden layer has a recurrent connection from its previous timestep
    spikes: v_t = beta * v_{t-1} + W_in x_t + W_rec z_{t-1}.
    """

    def __init__(
        self,
        input_size: int = 700,
        hidden_sizes: Sequence[int] = (128,),
        num_classes: int = 20,
        beta: float = 0.95,
        readout_beta: float = 0.95,
        threshold: float = 1.0,
        surrogate_slope: float = 25.0,
        readout: str = "mean_membrane",
        recurrent_scale: float = 0.5,
    ):
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer")
        if readout not in ("mean_membrane", "last_membrane"):
            raise ValueError("readout must be 'mean_membrane' or 'last_membrane'")

        self.input_size = int(input_size)
        self.hidden_sizes = [int(size) for size in hidden_sizes]
        self.num_classes = int(num_classes)
        self.beta = float(beta)
        self.readout_beta = float(readout_beta)
        self.threshold = float(threshold)
        self.surrogate_slope = float(surrogate_slope)
        self.readout = readout
        self.recurrent_scale = float(recurrent_scale)

        layer_sizes = [self.input_size] + self.hidden_sizes
        self.input_layers = nn.ModuleList(
            nn.Linear(layer_sizes[idx], layer_sizes[idx + 1])
            for idx in range(len(self.hidden_sizes))
        )
        self.recurrent_layers = nn.ModuleList(
            nn.Linear(size, size, bias=False) for size in self.hidden_sizes
        )
        self.classifier = nn.Linear(self.hidden_sizes[-1], self.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.input_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in self.recurrent_layers:
            nn.init.orthogonal_(layer.weight)
            layer.weight.data.mul_(self.recurrent_scale)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def _readout_logits(
        self, readout_trace: torch.Tensor, lengths: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.readout == "last_membrane":
            if lengths is None:
                return readout_trace[-1]
            last_idx = lengths.clamp(1, readout_trace.size(0)) - 1
            batch_idx = torch.arange(readout_trace.size(1), device=readout_trace.device)
            return readout_trace[last_idx, batch_idx]

        if lengths is None:
            return readout_trace.mean(dim=0)

        steps = torch.arange(readout_trace.size(0), device=readout_trace.device)
        valid = steps[:, None] < lengths.clamp(1, readout_trace.size(0))[None, :]
        valid = valid.to(readout_trace.dtype).unsqueeze(-1)
        summed = (readout_trace * valid).sum(dim=0)
        denom = lengths.clamp(1, readout_trace.size(0)).to(readout_trace.dtype)
        return summed / denom[:, None]

    def forward(
        self,
        spikes: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_spikes: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        if spikes.dim() != 3:
            raise ValueError("expected spikes with shape [T, B, N]")
        if spikes.size(-1) != self.input_size:
            raise ValueError(
                "expected input_size {}, got {}".format(self.input_size, spikes.size(-1))
            )

        time_steps, batch_size, _ = spikes.shape
        device = spikes.device
        dtype = spikes.dtype
        hidden_mem = [
            torch.zeros(batch_size, size, device=device, dtype=dtype)
            for size in self.hidden_sizes
        ]
        prev_spikes = [
            torch.zeros(batch_size, size, device=device, dtype=dtype)
            for size in self.hidden_sizes
        ]
        readout_mem = torch.zeros(
            batch_size, self.num_classes, device=device, dtype=dtype
        )

        readout_trace = []
        hidden_traces: List[List[torch.Tensor]] = [[] for _ in self.hidden_sizes]
        hidden_membrane_traces: List[List[torch.Tensor]] = [
            [] for _ in self.hidden_sizes
        ]

        for step in range(time_steps):
            x = spikes[step]
            new_spikes = []
            for layer_idx, (input_layer, recurrent_layer) in enumerate(
                zip(self.input_layers, self.recurrent_layers)
            ):
                recurrent_input = recurrent_layer(prev_spikes[layer_idx])
                hidden_mem[layer_idx] = (
                    self.beta * hidden_mem[layer_idx]
                    + input_layer(x)
                    + recurrent_input
                )
                pre_reset_mem = hidden_mem[layer_idx]
                z = spike_fn(hidden_mem[layer_idx] - self.threshold, self.surrogate_slope)
                hidden_mem[layer_idx] = hidden_mem[layer_idx] - z.detach() * self.threshold
                x = z
                new_spikes.append(z)
                if return_spikes:
                    hidden_traces[layer_idx].append(z)
                    hidden_membrane_traces[layer_idx].append(pre_reset_mem)
            prev_spikes = new_spikes

            readout_mem = self.readout_beta * readout_mem + self.classifier(x)
            readout_trace.append(readout_mem)

        readout_trace_tensor = torch.stack(readout_trace, dim=0)
        logits = self._readout_logits(readout_trace_tensor, lengths)

        aux: Dict[str, object] = {"readout_trace": readout_trace_tensor}
        if return_spikes:
            aux["hidden_spikes"] = [
                torch.stack(layer_trace, dim=0) for layer_trace in hidden_traces
            ]
            aux["hidden_membranes"] = [
                torch.stack(layer_trace, dim=0)
                for layer_trace in hidden_membrane_traces
            ]
        else:
            aux["hidden_spikes"] = []
            aux["hidden_membranes"] = []
        return logits, aux


def temporal_im_loss(
    hidden_spikes: Sequence[torch.Tensor],
    lengths: Optional[torch.Tensor] = None,
    target_rate: float = 0.02,
) -> torch.Tensor:
    if not hidden_spikes:
        return torch.tensor(0.0)

    losses = []
    for trace in hidden_spikes:
        if lengths is None:
            firing_rate = trace.mean()
        else:
            steps = torch.arange(trace.size(0), device=trace.device)
            valid = steps[:, None] < lengths.clamp(1, trace.size(0))[None, :]
            valid = valid.to(trace.dtype).unsqueeze(-1)
            firing_rate = (trace * valid).sum() / (valid.sum() * trace.size(-1))
        losses.append((firing_rate - target_rate) ** 2)
    return torch.stack(losses).mean()


def temporal_threshold_im_loss(
    membrane_traces: Sequence[torch.Tensor],
    lengths: Optional[torch.Tensor] = None,
    threshold: float = 1.0,
) -> torch.Tensor:
    if not membrane_traces:
        return torch.tensor(0.0)

    losses = []
    for trace in membrane_traces:
        if lengths is None:
            membrane_mean = trace.mean()
        else:
            steps = torch.arange(trace.size(0), device=trace.device)
            valid = steps[:, None] < lengths.clamp(1, trace.size(0))[None, :]
            valid = valid.to(trace.dtype).unsqueeze(-1)
            membrane_mean = (trace * valid).sum() / (valid.sum() * trace.size(-1))
        losses.append((membrane_mean - threshold) ** 2)
    return torch.stack(losses).mean()


def firing_rate_stats(hidden_spikes: Sequence[torch.Tensor]) -> Dict[str, float]:
    stats = {}
    for idx, trace in enumerate(hidden_spikes):
        layer_rates = trace.detach().mean(dim=(0, 1))
        stats["firing_rate_layer_{}".format(idx)] = layer_rates.mean().item()
        stats["silent_neuron_ratio_layer_{}".format(idx)] = (
            layer_rates <= 0
        ).float().mean().item()
    return stats
