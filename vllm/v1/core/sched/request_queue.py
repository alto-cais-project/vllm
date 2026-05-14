# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from vllm.v1.request import Request


class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""
    FCFS = "fcfs"
    PRIORITY = "priority"
    ALTO = "alto"


class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass

    @abstractmethod
    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse order."""
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        self.extendleft(reversed(requests))

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [
            req for req in self if req not in requests_to_remove
        ]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse order."""
        return super().__reversed__()


class PriorityRequestQueue(RequestQueue):
    """
    A priority queue that supports heap operations.

    Requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[int, float, Request]] = []

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap,
                       (request.priority, request.arrival_time, request))

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        _, _, request = heapq.heappop(self._heap)
        return request

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        _, _, request = self._heap[0]
        return request

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.
        
        Note: In a priority queue, there is no concept of prepending to the 
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to priority policy.
        
        Note: In a priority queue, there is no concept of prepending to the 
        front. Requests are ordered by (priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._heap = [(p, t, r) for p, t, r in self._heap if r != request]
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        self._heap = [(p, t, r) for p, t, r in self._heap
                      if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            _, _, request = heapq.heappop(heap_copy)
            yield request

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse priority order."""
        return reversed(list(self))

    
class AltoRequestQueue(RequestQueue):
    """
    Alto ancestry-aware root-level queue.

    Required trace headers:
    - alto-root-id: root request/tree id
    - alto-flow-id: parent ancestry path within the root
    - alto-local-id: local child id under the parent flow

    Scheduling:
    - Across roots: token-aware round-robin, accounted by the scheduler.
    - Within each root: heap order by ancestry path, local id, arrival time.
    """
    ROOT_FLOW = "__root__"

    def __init__(self) -> None:
        self._roots: dict[
            int, list[tuple[tuple[int, ...], int, float, Request]]
        ] = {}
        self._root_order: deque[tuple[int, int]] = deque()
        self._active_roots: set[int] = set()
        self._root_epochs: dict[int, int] = {}
        self._root_tokens_used: dict[int, int] = {}
        self._size = 0

        self._cached_root: int | None = None
        self._cached_request: Request | None = None

    @staticmethod
    def _parse_int(value: str | None) -> int:
        """Parse a required integer trace header value."""
        if value is None:
            raise ValueError("Missing required ALTO integer trace header")
        return int(value)
    
    def request_keys(self, request: Request) -> tuple[int, tuple[int, ...], int]:
        """Extract Alto root id, flow id, and local id for hierarchical DRR."""
        headers = request.trace_headers or {}

        root_id = self._parse_int(headers.get("alto-root-id"))
        flow_id = headers.get("alto-flow-id") or self.ROOT_FLOW
        local_id = self._parse_int(headers.get("alto-local-id"))

        return root_id, self._parse_flow_rank(flow_id), local_id
    
    @staticmethod
    def request_root_id(request: Request) -> int:
        headers = request.trace_headers or {}
        root_id = headers.get("alto-root-id")
        if root_id is None:
            raise ValueError("Missing required ALTO root trace header")
        return int(root_id)
    @classmethod
    def _parse_flow_rank(cls, flow_id: str) -> tuple[int, ...]:
        if flow_id in ("", "0", cls.ROOT_FLOW):
            return ()
        return tuple(int(part) for part in flow_id.split("/") if part)  
    
    def clear_cache(self) -> None:
        """Clear the cached peek candidate after any queue mutation."""
        self._cached_root = None
        self._cached_request = None

    def num_active_roots(self) -> int:
        return len(self._active_roots)

    def active_root_ids(self) -> set[int]:
        """Return root ids that currently have waiting requests."""
        return set(self._active_roots)
    
    def has_other_active_root(self, root_id: int) -> bool:
        if len(self._active_roots) > 1:
            return True
        return bool(self._active_roots) and root_id not in self._active_roots
    
    def activate_root(self, root_id: int) -> None:
        """Create and activate a root so it participates in round-robin."""
        if root_id not in self._roots:
            self._roots[root_id] = []
        if root_id not in self._active_roots:
            epoch = self._root_epochs.get(root_id, 0) + 1
            self._root_epochs[root_id] = epoch

            self._active_roots.add(root_id)
            self._root_order.append((root_id, epoch))
            self._root_tokens_used.setdefault(root_id, 0)
    
    def remove_root_if_empty(self, root_id: int) -> None:
        """Deactivate an empty root while keeping root-order cleanup lazy."""
        if self._roots.get(root_id):
            return

        self._roots.pop(root_id, None)
        self._active_roots.discard(root_id)
        self._root_tokens_used.pop(root_id, None)

    def _is_live_root_entry(self, root_id: int, epoch: int) -> bool:
        """Return whether a root-order entry is still valid."""
        return (root_id in self._active_roots
                and self._root_epochs.get(root_id) == epoch
                and bool(self._roots.get(root_id)))
    
    def fetch_next_live_root(self) -> int | None:
        """Return the next active non-empty root, lazily dropping stale entries."""
        while self._root_order:
            root_id, epoch = self._root_order[0]
            if self._is_live_root_entry(root_id, epoch):
                return root_id
            self._root_order.popleft()
        return None
    
    def select_request(self) -> tuple[int, Request]:
        """Select and cache the next request without removing it."""
        if self._cached_root is not None and self._cached_request is not None:
            return self._cached_root, self._cached_request

        root_id = self.fetch_next_live_root()
        if root_id is None:
            raise IndexError("peek from an empty AltoRequestQueue")

        _, _, _, request = self._roots[root_id][0]

        self._cached_root = root_id
        self._cached_request = request
        return root_id, request

    def peek_root_id(self) -> int:
        """Peek at the selected request's root id without removing it."""
        root_id, _ = self.select_request()
        return root_id
    
    def current_root_token_budget(self, root_token_quantum: int) -> int:
        """Return the remaining token budget for the selected root."""
        root_id, _ = self.select_request()
        return self.root_token_budget(root_id, root_token_quantum)

    def root_token_budget(self, root_id: int,
                          root_token_quantum: int) -> int:
        """Return the remaining token budget for a root."""
        used = self._root_tokens_used.get(root_id, 0)
        return max(0, max(1, root_token_quantum) - used)

    def reset_root_budget(self, root_id: int) -> None:
        """Reset a root's accumulated token usage."""
        self._root_tokens_used[root_id] = 0

    def defer_current_root(self) -> bool:
        """Move the current waiting root behind other active waiting roots."""
        root_id = self.fetch_next_live_root()
        if root_id is None or len(self._active_roots) <= 1:
            return False

        self._root_tokens_used[root_id] = 0
        self._root_order.rotate(-1)
        self.clear_cache()
        return True

    def account_root_tokens(self, root_id: int, num_scheduled_tokens: int,
                            root_token_quantum: int) -> None:
        """Charge scheduled prefill tokens to a root's monopoly cap."""
        quantum = max(1, root_token_quantum)
        used = self._root_tokens_used.get(root_id, 0)
        used += max(0, num_scheduled_tokens)

        if used < quantum:
            self._root_tokens_used[root_id] = used
            return

        self._root_tokens_used[root_id] = quantum

        if root_id in self._active_roots:
            current_root = self.fetch_next_live_root()
            if current_root == root_id and len(self._active_roots) > 1:
                self._root_tokens_used[root_id] = 0
                self._root_order.rotate(-1)
                self.clear_cache()
    
    def add_request(self, request: Request) -> None:
        """Add a request into its root heap using ancestry order."""
        root_id, flow_rank, local_id = self.request_keys(request)
        self.activate_root(root_id)

        heapq.heappush(self._roots[root_id],
                       (flow_rank, local_id, request.arrival_time, request))
        self._size += 1
        self.clear_cache()

    def _pop_selected_request(self) -> tuple[int, Request, bool]:
        root_id, cached_request = self.select_request()
        _, _, _, request = heapq.heappop(self._roots[root_id])
        assert request == cached_request

        self._size -= 1
        self.clear_cache()

        has_remaining_requests = bool(self._roots[root_id])
        if not has_remaining_requests:
            self.remove_root_if_empty(root_id)
        return root_id, request, has_remaining_requests
    
    def pop_request_without_accounting(self) -> Request:
        """Pop a skipped request without consuming root fairness budget."""
        _, request, _ = self._pop_selected_request()
        return request

    def pop_request_and_account(self, num_scheduled_tokens: int,
                                root_token_quantum: int) -> Request:
        """Pop a scheduled request and account its token cost to the root."""
        root_id, request, _ = self._pop_selected_request()
        self.account_root_tokens(root_id, num_scheduled_tokens,
                                 root_token_quantum)
        return request

    def pop_request(self) -> Request:
        """Pop a request and advance the root rotation."""
        return self.pop_request_and_account(1, 1)


    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        _, request = self.select_request()
        return request
    
    def prepend_request(self, request: Request) -> None:
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        self.remove_requests([request])

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests across all active Alto roots."""
        self.clear_cache()

        requests_to_remove = set(requests)

        for root_id, heap in list(self._roots.items()):
            new_heap = [(flow_rank, local_id, t, r) 
                        for (flow_rank, local_id, t, r) in heap 
                        if r not in requests_to_remove]
            removed = len(heap) - len(new_heap)
            if removed == 0:
                continue
        
            heapq.heapify(new_heap)
            self._roots[root_id] = new_heap
            self._size -= removed
            self.remove_root_if_empty(root_id)

    def prune_root_budgets(self, live_root_ids: set[int]) -> None:
        """Drop token budget state for roots with no active prefill work."""
        for root_id in list(self._root_tokens_used):
            if root_id not in live_root_ids:
                self._root_tokens_used.pop(root_id, None)

    def __bool__(self) -> bool:
        """Return whether any root currently has waiting requests."""
        return self._size > 0

    def __len__(self) -> int:
        """Return the total number of requests across all roots."""
        return self._size

    def __iter__(self) -> Iterator[Request]:
        """Iterate over requests in DRR order without mutating the queue."""
        root_order = deque(
            root_id for root_id, epoch in self._root_order
            if self._is_live_root_entry(root_id, epoch)
        )
        roots = {
            root_id: heap[:]
            for root_id, heap in self._roots.items()
            if root_id in self._active_roots and heap
        }
        while root_order:
            root_id = root_order.popleft()
            heap = roots.get(root_id)
            if not heap:
                continue

            _, _, _, request = heapq.heappop(heap)
            yield request

            if heap:
                root_order.append(root_id)

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the current DRR order in reverse."""
        return reversed(list(self))

def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    elif policy == SchedulingPolicy.ALTO:
        return AltoRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
