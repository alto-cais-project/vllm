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
    ALTO_DRR = "alto_drr"


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

class AltoDRRRequestQueue(RequestQueue):
    """
    Alto ancestry-aware DRR queue V0.

    - Flow key comes from x-alto-flow-id.
    - Local ordering inside a flow uses alto-local-id.
    - Cross-flow scheduling is one request per active flow per round.
    """
    DEFAULT_FLOW = "__default__"
    QUANTUM = 1
    COST = 1

    def __init__(self) -> None:
        self._flows: dict[str, list[tuple[int, float, Request]]] = {}
        self._flow_order: deque[str] = deque()
        self._active_flows: set[str] = set()
        self._deficit: dict[str, int] = {}
        self._size = 0

        self._cached_flow: str | None = None
        self._cached_request: Request | None = None

    @staticmethod
    def _parse_int(value: str | None, default: int = 0) -> int:
        """Parse an integer trace header value with a default fallback."""
        if value is None:
            return default
        return int(value)
        
    def clear_cache(self) -> None:
        """Clear the cached peek candidate after any queue mutation."""
        self._cached_flow = None
        self._cached_request = None

    def request_keys(self, request: Request) -> tuple[str, int]:
        """Extract the Alto flow id and local id used for DRR ordering."""
        headers = request.trace_headers or {}

        flow_id = headers.get("alto-flow-id") or self.DEFAULT_FLOW
        local_id = self._parse_int(headers.get("alto-local-id"))

        return flow_id, local_id
    
    def activate_flow(self, flow_id: str) -> None:
        """Create and activate a flow so it participates in round-robin."""
        if flow_id not in self._flows:
            self._flows[flow_id] = []
        if flow_id not in self._active_flows:
            self._active_flows.add(flow_id)
            self._flow_order.append(flow_id)
            self._deficit.setdefault(flow_id, 0)
    
    def remove_flow_if_empty(self, flow_id: str) -> None:
        """Remove an empty flow while avoiding O(n) deque removal."""
        if self._flows.get(flow_id):
            return

        self._flows.pop(flow_id, None)
        self._active_flows.discard(flow_id)
        self._deficit.pop(flow_id, None)

        if self._flow_order and self._flow_order[0] == flow_id:
            self._flow_order.popleft()
    
    def fetch_next_live_flow(self) -> str | None:
        """Return the next active non-empty flow, lazily dropping stale ones."""
        while self._flow_order:
            flow_id = self._flow_order[0]
            if flow_id in self._active_flows and self._flows.get(flow_id):
                return flow_id
            self._flow_order.popleft()
        return None
    
    def select_request(self) -> tuple[str, Request]:
        """Select and cache the next request without removing it."""
        if self._cached_flow is not None and self._cached_request is not None:
            return self._cached_flow, self._cached_request

        flow_id = self.fetch_next_live_flow()
        if flow_id is None:
            raise IndexError("peek from an empty AltoDRRRequestQueue")

        self._deficit[flow_id] = self._deficit.get(flow_id, 0) + self.QUANTUM

        heap = self._flows[flow_id]
        _,_,request = heap[0]

        self._cached_flow = flow_id
        self._cached_request = request
        return flow_id, request
    
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to DRR policy."""
        flow_id, local_id = self.request_keys(request)
        self.activate_flow(flow_id)

        heapq.heappush(self._flows[flow_id],
                       (local_id, request.arrival_time, request))
        self._size += 1
        self.clear_cache()

    def pop_request(self) -> Request:
        """Pop a request and advance the flow rotation."""
        flow_id, cached_request = self.select_request()
        heap = self._flows[flow_id]
        _, _, request = heapq.heappop(heap)
        assert request == cached_request
        self._size -= 1
        self._deficit[flow_id] = max(
            0, self._deficit.get(flow_id, 0) - self.COST
        )
        self.clear_cache()

        if heap:
            self._flow_order.rotate(-1)
        else:
            self.remove_flow_if_empty(flow_id)

        return request


    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        _, request = self.select_request()
        return request
    
    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to DRR policy.
        
        DRR has no literal front of queue because requests are grouped by flow."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to DRR policy.
        Used when skipped or preempted requests return to the waiting queue."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.clear_cache()
        for flow_id, heap in list(self._flows.items()):
            new_heap = [(o, t, r) for (o, t, r) in heap if r != request]
            removed = len(heap) - len(new_heap)
            if removed == 0:
                continue
        
            heapq.heapify(new_heap)
            self._flows[flow_id] = new_heap
            self._size -= removed
            self.remove_flow_if_empty(flow_id)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests across all active Alto flows."""
        self.clear_cache()

        requests_to_remove = set(requests)
        for flow_id, heap in list(self._flows.items()):
            new_heap = [(o, t, r) for (o, t, r) in heap if r not in requests_to_remove]
            removed = len(heap) - len(new_heap)
            if removed == 0:
                continue
        
            heapq.heapify(new_heap)
            self._flows[flow_id] = new_heap
            self._size -= removed
            self.remove_flow_if_empty(flow_id)


    def __bool__(self) -> bool:
        """Return whether any flow currently has waiting requests."""
        return self._size > 0

    def __len__(self) -> int:
        """Return the total number of requests across all flows."""
        return self._size

    def __iter__(self) -> Iterator[Request]:
        """Iterate over requests in DRR order without mutating the queue."""
        flow_order = deque(
            flow_id for flow_id in self._flow_order
            if flow_id in self._active_flows and self._flows.get(flow_id)
        )
        flows = {
            flow_id: heap[:]
            for flow_id, heap in self._flows.items()
            if flow_id in self._active_flows and heap
        }
        while flow_order:
            flow_id = flow_order.popleft()
            heap = flows.get(flow_id)
            if not heap:
                continue

            _, _, request = heapq.heappop(heap)
            yield request

            if heap:
                flow_order.append(flow_id)

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the current DRR order in reverse."""
        return reversed(list(self))

def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    elif policy == SchedulingPolicy.ALTO_DRR:
        return AltoDRRRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
