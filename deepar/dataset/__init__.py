from abc import ABC


class Dataset(ABC):
    def __init__(self):
        super().__init__()

    def next_batch(self, **kwargs):
        pass
