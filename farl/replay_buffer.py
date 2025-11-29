"""Simple replay-buffer implementation used by training and simulation."""

from __future__ import annotations

import random
from typing import Generic, Iterable, List, Sequence, TypeVar

__all__ = ["ReplayBuffer"]

T = TypeVar("T")


class ReplayBuffer(Generic[T]):
    """Circular replay buffer with uniform sampling."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._buffer: List[T | None] = []
        self._position = 0

    def push(self, item: T) -> None:
        """Insert ``item`` into the buffer, overwriting the oldest entry."""
        if len(self._buffer) < self.capacity:
            self._buffer.append(item)
        else:
            self._buffer[self._position] = item
        self._position = (self._position + 1) % self.capacity

    def extend(self, items: Iterable[T]) -> None:
        """Push multiple items into the buffer."""
        for item in items:
            self.push(item)

    def sample(self, batch_size: int) -> Sequence[T]:
        """Return ``batch_size`` uniformly sampled items."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self):
            raise ValueError(
                f"cannot sample {batch_size} items from buffer of size {len(self)}"
            )
        return random.sample(self._buffer, batch_size)  # type: ignore[arg-type]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._buffer)
