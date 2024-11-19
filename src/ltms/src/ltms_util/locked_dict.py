from collections.abc import MutableMapping
from threading import RLock

class LockedDict(MutableMapping):

    def __init__(self, *args, **kwds):
        self._dict = dict(*args, **kwds)
        self._lock = RLock()

    def __len__(self):
        return len(self._dict)
    
    def __iter__(self):
        with self._lock:
            yield from self._dict.__iter__()
    
    def __getitem__(self, key):
        with self._lock:
            return self._dict.__getitem__(key)
    
    def __setitem__(self, key, value):
        with self._lock:
            return self._dict.__setitem__(key, value)
    
    def __delitem__(self, key):
        with self._lock:
            return self._dict.__delitem__(key)

    def add(self, key, dict1=None, **kwds):
        assert bool(dict1 is not None) ^ bool(kwds), 'Use either dict1 or keywords'
        with self._lock:
            self._dict[key] = kwds if dict1 is None else dict1

    def select(self, key):
        with self._lock:
            yield self._dict[key]
    
    def snapshot(self):
        return list(self)
