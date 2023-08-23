from typing import List
import pandas as pd
import random
from SubStrat.genetic_sub_algorithm.gene import Gene
from SubStrat.genetic_sub_algorithm.population import Population
from SubStrat.sub_algo_utils.fitness import BaseFitness
from SubStrat.genetic_sub_algorithm.selection import BaseSelection
from abc import ABC, abstractmethod

class BaseInitPopulationAlgorithm(ABC):
    def __init__(self, dataset: pd.DataFrame, sub_col_size: int, sub_row_size: int, target_column_name: str, population_size: int,
                 fitness: BaseFitness=None, selection: BaseSelection=None, mutation_rate: float=0) -> None:
        self._dataset: pd.DataFrame = dataset
        self._sub_col_size: int = sub_col_size
        self._sub_row_size: int = sub_row_size
        self._target_column_name: set = target_column_name
        self._population_size: int = population_size
        self._fitness = fitness
        self._selection = selection
        self._mutation_rate = mutation_rate
    
    @abstractmethod
    def init_population(self) -> Population: 
        raise NotImplementedError
    


class InitRandomPopulation(BaseInitPopulationAlgorithm):

    def __init__(self, dataset: pd.DataFrame, sub_col_size: int, sub_row_size: int, target_column_name: str, population_size: int,
                 fitness: BaseFitness=None, selection: BaseSelection=None, mutation_rate: float=0.001) -> None:
        super().__init__(dataset, sub_col_size, sub_row_size, target_column_name, population_size, fitness, selection, mutation_rate)

    def init_population(self) -> Population:
        genes: List[Gene] = []
        for _ in range(self._population_size):
            origin_rows_number, oringin_cols_number = self._dataset.shape
            # sample rows
            sub_rows = random.sample(range(origin_rows_number), self._sub_row_size)
            # sample columns
            copy_columns  = list(self._dataset.columns) 

            target_column_index = copy_columns.index(self._target_column_name)
            columns_indexes = list(range(len(copy_columns)))
            columns_indexes.pop(target_column_index)
            sub_cols = random.sample(columns_indexes, self._sub_col_size-1)
            sub_cols.append(target_column_index)
            
            genes.append(Gene(sub_rows, sub_cols))

        return Population(dataset=self._dataset, population_size=self._population_size, 
                          target_column=self._target_column_name, init_genes=genes, fitness=self._fitness, 
                          selection = self._selection, mutation_rage=self._mutation_rate)

