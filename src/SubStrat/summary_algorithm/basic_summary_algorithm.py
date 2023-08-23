
"""
Basic Summary algoritm.
"""
import pandas as pd
from abc import ABC, abstractclassmethod


class BasicSummaryAlgorithm(ABC):
    """
    Basic summary algorithm, every algorithm need to inherit this class.
    """

    def __init__(self, *args, **kwargs):
        pass

    @abstractclassmethod
    def run(self) -> pd.DataFrame:
        """
        Run this function for reducing the size of the dataset.
        """
        raise NotImplementedError
