import threading


class ThreadSafeDict:
    def __init__(self, *args, **kwargs):
        self._lock = threading.Lock()
        self._dict = dict(*args, **kwargs)

    def __getitem__(self, key):
        with self._lock:
            return self._dict[key]

    def __setitem__(self, key, value):
        with self._lock:
            self._dict[key] = value

    def __delitem__(self, key):
        with self._lock:
            del self._dict[key]

    def get(self, key, default=None):
        with self._lock:
            return self._dict.get(key, default)

    def setdefault(self, key, default=None):
        with self._lock:
            return self._dict.setdefault(key, default)

    def pop(self, key, default=None):
        with self._lock:
            return self._dict.pop(key, default)

    def update(self, *args, **kwargs):
        with self._lock:
            self._dict.update(*args, **kwargs)

    def keys(self):
        with self._lock:
            return list(self._dict.keys())

    def values(self):
        with self._lock:
            return list(self._dict.values())

    def items(self):
        with self._lock:
            return list(self._dict.items())

    def clear(self):
        with self._lock:
            self._dict.clear()

    def copy(self):
        with self._lock:
            return ThreadSafeDict(self._dict.copy())

    def __contains__(self, key):
        with self._lock:
            return key in self._dict

    def __len__(self):
        with self._lock:
            return len(self._dict)

    def __str__(self):
        with self._lock:
            return str(self._dict)

    def __repr__(self):
        with self._lock:
            return f"{self.__class__.__name__}({self._dict})"

    def incOrDefault(self, key, value):
        with self._lock:
            if key in self._dict:
                self._dict[key] += value
            else:
                self._dict[key] = value
