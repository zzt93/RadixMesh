from abc import ABC, abstractmethod


class BaseSyncAlgo(ABC):
    @abstractmethod
    def next(self):
        pass


class RingSyncAlgo(BaseSyncAlgo):
    def __init__(self):
        pass

    def next(self):
        pass


def get_sync_algo():
    return RingSyncAlgo()
