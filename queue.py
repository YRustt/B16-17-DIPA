import heapq


class QueueIsEmpty(Exception):
    pass


class PriorityQueue:
    def __init__(self):
        self._queue = []

    def insert(self, key, value):
        heapq.heappush(self._queue, (key, value))

    def extract_minimum(self):
        if len(self._queue) == 0:
            raise QueueIsEmpty()
        return heapq.heappop(self._queue)


class HierarchicalQueue:
    def __init__(self):
        self._queue = []
        self._max_priority = None

    def insert(self, key, value):
        heapq.heappush(self._queue, (min(key, self._max_priority)
                                     if self._max_priority is not None else key, value))

    def extract_minimum(self):
        if len(self._queue) == 0:
            raise QueueIsEmpty()
        if self._max_priority is None or self._queue[0][0] != self._max_priority:
            self._max_priority = self._queue[0][0]
        return heapq.heappop(self._queue)
