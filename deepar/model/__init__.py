from abc import ABC


class NNModel(ABC):
    def __init__(self):
        super().__init__()

    def net_structure(self, **kwargs):
        pass

    def instantiate_and_fit(self, **kwargs):
        pass

