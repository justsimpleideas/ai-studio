"""In-process publish/subscribe used to stream progress to the UI (Server-Sent Events)."""

from __future__ import annotations

import queue
import threading


class EventHub:
    def __init__(self, backlog: int = 256) -> None:
        self._subscribers: set[queue.Queue] = set()
        self._lock = threading.Lock()
        self._backlog = backlog

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=self._backlog)
        with self._lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._subscribers.discard(q)

    def publish(self, kind: str, data: dict) -> None:
        event = {"type": kind, **data}
        with self._lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(event)
            except queue.Full:
                # A stalled client must never block generation. Drop its oldest event instead.
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except (queue.Empty, queue.Full):
                    pass

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)
