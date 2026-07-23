"""Small deterministic bounded collections for non-authoritative process memory."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, MutableSet


class BoundedSet[Item: object](MutableSet[Item]):
    """Insertion-ordered set that evicts the oldest item at a strict capacity."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("bounded-set capacity must be positive")
        self.capacity = capacity
        self.evictions = 0
        self._items: OrderedDict[Item, None] = OrderedDict()

    def __contains__(self, item: object) -> bool:
        return item in self._items

    def __iter__(self) -> Iterator[Item]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, item: Item) -> None:
        if item in self._items:
            self._items.move_to_end(item)
            return
        self._items[item] = None
        if len(self._items) > self.capacity:
            self._items.popitem(last=False)
            self.evictions += 1

    def discard(self, item: Item) -> None:
        self._items.pop(item, None)
