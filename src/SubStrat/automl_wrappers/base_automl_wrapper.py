from abc import ABC, abstractmethod

class BaseAutomlWarpper(ABC):
    def __init__(self):
        pass

    def fit(self, *args, **kwargs):
        self._fit_sub_dataset()
        self._fit_full_dataset()
    
    @abstractmethod
    def _fit_sub_dataset(self):
        pass
    
    @abstractmethod
    def _fit_full_dataset(self):
        pass