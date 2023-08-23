import pandas as pd
from tqdm import tqdm

from SubStrat.summary_algorithm.basic_summary_algorithm import BasicSummaryAlgorithm
from SubStrat.genetic_sub_algorithm.selection import BaseSelection
from SubStrat.sub_algo_utils.fitness import BaseFitness
from SubStrat.sub_algo_utils.sub_dataset_size import get_sub_dataset_size
from SubStrat.genetic_sub_algorithm.population_initialization import InitRandomPopulation, BaseInitPopulationAlgorithm
from SubStrat.genetic_sub_algorithm.population import Population
from SubStrat.genetic_sub_algorithm.gene import Gene


import time
from tqdm import tqdm

class GeneticSubAlgorithmn(BasicSummaryAlgorithm): 
    """
    A class that implements a genetic algorithm to summarize datasets.

    Attributes
    ----------
    dataset : pd.DataFrame
        The original dataset that needs to be summarized.
    target_column_name : str
        The name of the target column in the dataset.
    sub_row_size : int
        The number of rows for the subset of the dataset (summary). 
        If not provided, it will be calculated based on a predefined rule.
    sub_col_size : int
        The number of columns for the subset of the dataset (summary). 
        If not provided, it will be calculated based on a predefined rule.
    population_size : int
        The number of individuals in the population for the genetic algorithm.
    fitness : BaseFitness
        The fitness function used in the genetic algorithm. 
        If not provided, a default fitness function will be used.
    selection : BaseSelection
        The selection operator used in the genetic algorithm. 
        If not provided, a default selection operator will be used.
    mutation_rate : float
        The mutation rate used in the genetic algorithm.
    num_generation : int
        The number of generations the genetic algorithm will run for.
    init_pop : BaseInitPopulationAlgorithm
        The algorithm used to initialize the population for the genetic algorithm. 
        If not provided, a default algorithm will be used.
    stagnation_limit : int
        Number of generations without improvement in best gene score before stopping.
    time_limit : float
        Maximum time in seconds the `run` function can execute.

    Methods
    -------
    run() -> pd.DataFrame
        Runs the genetic algorithm and returns the best found subset of the dataset.
    """

    def __init__(self, dataset: pd.DataFrame, target_column_name: str, sub_col_size: int=None, 
                 sub_row_size: int=None, population_size: int=100, fitness: BaseFitness=None, 
                 selection: BaseSelection=None, mutation_rate: float=0.01, num_generation=25, 
                 population_init: BaseInitPopulationAlgorithm=None, stagnation_limit: int=25, time_limit: float=float('inf')):
        
        self.dataset = dataset
        self.target_column_name = target_column_name

        # Calculate default sizes if not provided
        default_sub_row_size, default_sub_col_size = get_sub_dataset_size(dataset) if sub_row_size is None or sub_col_size is None else (None, None)

        # Use provided sizes or defaults
        self.sub_row_size = sub_row_size or default_sub_row_size
        self.sub_col_size = sub_col_size or default_sub_col_size
        self.population_size = population_size
        self.fitness = fitness
        self.selection = selection
        self.mutation_rate = mutation_rate
        self.num_generation = num_generation
        self.stagnation_limit = stagnation_limit
        self.time_limit = time_limit

        init_pop_class = population_init or InitRandomPopulation
        self.init_pop = init_pop_class(dataset=self.dataset, sub_col_size=self.sub_col_size,
                                       sub_row_size=self.sub_row_size, target_column_name=self.target_column_name,
                                       fitness=self.fitness, selection=self.selection, 
                                       mutation_rate=self.mutation_rate, population_size=self.population_size)

    def run(self) -> pd.DataFrame:
        # print(f"Running genetic summary algorithm with #generations: {self.num_generation}")

        pop: Population = self.init_pop.init_population()
        pop.calc_gene_score()

        best_score = float('-inf')
        stagnation_counter = -1
        start_time = time.time()

        for _ in tqdm(range(self.num_generation)):
            if time.time() - start_time > self.time_limit:
                # print("Time limit reached. Stopping.")
                break
            current_best_score = pop.get_best_pop_gene().score
            if current_best_score <= best_score:
                stagnation_counter += 1
                if stagnation_counter >= self.stagnation_limit:
                    # print(f"No improvement for several generations. Stopping. score:{current_best_score}")
                    break
            else:
                best_score = current_best_score
                stagnation_counter = 0

            pop.next_generation()

        best_gene: Gene = pop.get_best_pop_gene()
        # print(f"Finished Genetic Algo, best gene with scoer of: {best_gene.score}, with time: {time.time() - start_time}")
        sub_data = self.dataset.iloc[best_gene.gene_rows, best_gene.gene_cols]
        return sub_data


        
        


