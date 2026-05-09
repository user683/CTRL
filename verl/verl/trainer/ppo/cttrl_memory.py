from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class MemoryItem:
    priority: float
    prompt: str = field(compare=False)
    response: str = field(compare=False)
    reliability: float = field(compare=False)
    consistency: float = field(compare=False)
    variance: float = field(compare=False)
    answer: str | None = field(compare=False, default=None)
    metadata: dict[str, Any] = field(compare=False, default_factory=dict)


class CognitiveReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = max(1, int(capacity))
        self._heap: list[MemoryItem] = []

    def __len__(self) -> int:
        return len(self._heap)

    def add(self, item: MemoryItem) -> bool:
        if len(self._heap) < self.capacity:
            heapq.heappush(self._heap, item)
            return True

        if self._heap[0].priority >= item.priority:
            return False

        heapq.heapreplace(self._heap, item)
        return True

    def sample(self, batch_size: int) -> list[MemoryItem]:
        if not self._heap:
            return []

        batch_size = min(int(batch_size), len(self._heap))
        return random.sample(self._heap, batch_size)

    def peek_lowest_priority(self) -> float | None:
        if not self._heap:
            return None
        return self._heap[0].priority
