import collections


class Counter(object):
    """A counter returns items in the specific range."""

    def __init__(self, items=[], min_count=1, max_count=None):
        self.counter = collections.Counter()
        self.min_count = min_count
        self.max_count = max_count
        self.update(items)

    def __len__(self):
        return len(self.counter)

    def update(self, items=[]):
        self.counter.update(items)

    def reset(self):
        self.counter.clear()

    def get(self, min_count=None, max_count=None):
        """Return the items if its count in [min_count, max_count]."""
        min_count = self.min_count if min_count is None else min_count
        max_count = self.max_count if max_count is None else max_count
        s = set()
        for key, count in self.counter.items():
            if min_count is not None and count < min_count:
                continue
            if max_count is not None and count > max_count:
                continue
            s.add(key)
        return s

    def size(self, min_count=None, max_count=None):
        """Return the size of counter in [min_count, max_count]."""
        min_count = self.min_count if min_count is None else min_count
        max_count = self.max_count if max_count is None else max_count
        s = 0
        for key, count in self.counter.items():
            if min_count is not None and count < min_count:
                continue
            if max_count is not None and count > max_count:
                continue
            s += 1
        return s