import pandas as pd
from scipy.stats import entropy
import math
import numpy as np
from collections import Counter

def l1_dist(value1: float,
            value2: float):
        return math.fabs(value1 - value2)

def mean_entropy(df: pd.DataFrame):
    try:
        return np.nanmean([entropy(list(Counter(df[col]).values()), base=2) for col in list(df)])
    except:
        return np.nanmean([entropy(df[col], base=2) for col in list(df)])


class BaseFitness:
    def __init__(self, *args, **kwargs):
        pass
    def score(self, *args, **kwargs) -> float: 
        raise NotImplementedError()


class MeanEntroyFitness(BaseFitness):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def score(self, full_ds: pd.DataFrame,
                     sub_ds: pd.DataFrame) -> float:
        """
        The L1 (Manhattan) distance between the mean entropy of both the dataset and subset dataset
        :param full_ds: the dataset (pandas' dataframe)
        :param sub_ds: the subset of the dataset (pandas' dataframe)
        :return: the score between them ranging (0, inf)
        """
        dist = l1_dist(mean_entropy(full_ds), mean_entropy(sub_ds))
        return 1.0 / dist if dist != 0 else float('inf')