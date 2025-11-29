"""Neural network modules (actor/critic) used by the FAADP controller."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn import init

__all__ = ["Actor", "Critic"]


def _init_linear_layers(layers: Iterable[nn.Module], mode: str = "normal") -> None:
    """Apply Xavier initialization to all ``nn.Linear`` layers."""
    for module in layers:
        if isinstance(module, nn.Linear):
            if mode == "normal":
                init.xavier_normal_(module.weight)
            else:
                init.xavier_uniform_(module.weight)
            init.constant_(module.bias, 0.0)


class Actor(nn.Module):
    """Feed-forward policy network used to compute control commands."""

    def __init__(self, input_size: int, output_size: int, lr: float = 1e-3) -> None:
        super().__init__()
        self._out_gain = torch.tensor([2.0, 0.3])
        self._norm_matrix = torch.ones(input_size, dtype=torch.float32)
        hidden = 256
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, output_size),
            nn.Tanh(),
        )
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt, step_size=1000, gamma=0.95, last_epoch=-1
        )
        _init_linear_layers(self.layers)
        # Stabilize the final layer to avoid overly aggressive initial actions.
        last_linear = list(self.layers.children())[-2]
        if isinstance(last_linear, nn.Linear):
            last_linear.weight.data.mul_(1e-4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        scaled = torch.mul(inputs, self._norm_matrix)
        return torch.mul(self._out_gain, self.layers(scaled))

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the detached policy output for convenience."""
        return self.forward(inputs).detach()

    def save_parameters(self, directory: str | Path) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(directory) / "actor.pth")

    def load_parameters(self, directory: str | Path) -> None:
        state = torch.load(Path(directory) / "actor.pth", map_location="cpu")
        self.load_state_dict(state)

    # Legacy aliases
    def saveParameters(self, directory: str | Path) -> None:  # noqa: N802
        self.save_parameters(directory)

    def loadParameters(self, directory: str | Path) -> None:  # noqa: N802
        self.load_parameters(directory)


class Critic(nn.Module):
    """Value network trained with policy evaluation."""

    def __init__(self, input_size: int, output_size: int, lr: float = 1e-3) -> None:
        super().__init__()
        hidden = 256
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, output_size),
        )
        self._norm_matrix = torch.ones(input_size, dtype=torch.float32)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt, step_size=1000, gamma=0.95, last_epoch=-1
        )
        _init_linear_layers(self.layers, mode="uniform")
        last_linear = list(self.layers.children())[-1]
        if isinstance(last_linear, nn.Linear):
            last_linear.weight.data.mul_(1e-4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        scaled = torch.mul(inputs.view(inputs.shape[0], -1), self._norm_matrix)
        return self.layers(scaled)

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward(inputs).detach()

    def save_parameters(self, directory: str | Path) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(directory) / "critic.pth")

    def load_parameters(self, directory: str | Path) -> None:
        state = torch.load(Path(directory) / "critic.pth", map_location="cpu")
        self.load_state_dict(state)

    def saveParameters(self, directory: str | Path) -> None:  # noqa: N802
        self.save_parameters(directory)

    def loadParameters(self, directory: str | Path) -> None:  # noqa: N802
        self.load_parameters(directory)
