import pandas as pd
from typing import List

class Gene(object):

    def __init__(self, sub_rows: List[int] = None, sub_cols: List[int] = None) -> None:
        if isinstance(sub_rows, list):
            sub_rows.sort()
        if isinstance(sub_cols, list):
            sub_cols.sort()
        self.gene_rows = sub_rows
        self.gene_cols = sub_cols
        self.score = None


    def __repr__(self) -> str:
        return f"score: {self.score}, rows:{self.gene_rows}, cols:{self.gene_cols}"