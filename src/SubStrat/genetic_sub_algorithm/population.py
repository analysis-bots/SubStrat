from typing import List
from SubStrat.genetic_sub_algorithm.gene import Gene
from SubStrat.sub_algo_utils.fitness import BaseFitness, MeanEntroyFitness
from SubStrat.genetic_sub_algorithm.selection import BaseSelection, RoyaltySelection, RoyaltyRouletteSelection
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import random



class Population:
    def __init__(self, dataset: pd.DataFrame, population_size: int, target_column: str, init_genes: List[Gene]=None, 
                 fitness: BaseFitness=None, selection: BaseSelection=None, mutation_rage: float=0.001) -> None:
        self.dataset = dataset
        self.population_size: int = population_size
        self.genes: List[Gene] = init_genes if init_genes else []
        self.generation = -1 if not init_genes else 0
        self.fitness = fitness if fitness else MeanEntroyFitness()
        self.selection = selection if selection else RoyaltyRouletteSelection(0.1)
        self.target_column = target_column
        self.mutation_rage = mutation_rage

    def _calc_gene_score(self, gene: Gene):
        score = self.fitness.score(self.dataset, self.dataset.iloc[gene.gene_rows, gene.gene_cols]) 
        gene.score = score

    def calc_gene_score(self):
        with ThreadPoolExecutor() as executor:
            executor.map(self._calc_gene_score, self.genes)

    @staticmethod
    def _mutate_gene(gene: Gene, row_count: int, col_count: int, mutation_rage: float, target_col: int):
        # Calculate prc
        
        # Check if need to to mutation
        if random.random() <= mutation_rage:
            prc = row_count / (row_count + col_count)

            # Decide whether to mutate a row or a column
            if random.random() < prc:
                # Mutate a row
                # Choose a row index to replace
                row_to_replace = random.choice(gene.gene_rows)
                # Choose a new row index that is not already in gene_rows
                new_row = random.choice([r for r in range(row_count) if r not in gene.gene_rows])
                # Replace the chosen row index
                gene.gene_rows[gene.gene_rows.index(row_to_replace)] = new_row
            else:
                # Mutate a column
                # Choose a column index to replace, making sure it's not the target column
                col_to_replace = random.choice([c for c in gene.gene_cols if c != target_col])
                # Choose a new column index that is not already in gene_cols and is not the target column
                new_col = random.choice([c for c in range(col_count) if c not in gene.gene_cols and c != target_col])
                # Replace the chosen column index
                gene.gene_cols[gene.gene_cols.index(col_to_replace)] = new_col
        return gene
    

    def mutate_all(self):
        # Get the number of rows and columns in the DataFrame
        row_count, col_count = self.dataset.shape
        target_col = self.dataset.columns.get_loc(self.target_column)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._mutate_gene, gene, row_count, col_count, self.mutation_rage, target_col) for gene in self.genes]

            mutated_genes = [future.result() for future in futures]

        return mutated_genes

    def keep_generaion_size(self, best_genes: List[Gene] = []):
        self.calc_gene_score()
        sorted_genes = sorted(self.genes, key=lambda individual: individual.score if individual.score is not None else 0, reverse=True)        
        sorted_genes.extend(best_genes)
        if len(sorted_genes) < self.population_size:
            t = self.population_size - len(sorted_genes)
            n = random.choices(best_genes, k=t)
            sorted_genes.extend(n)
        elif len(sorted_genes) > self.population_size:
            sorted_genes = sorted_genes[:self.population_size]  
        self.genes = sorted_genes

    def next_generation(self, articale: bool=False):
        """
        Create next generation from last generation, can be done just if genes already set
        """
        if not self.genes:
            raise RuntimeError("No init genes to create next generation")
        self.calc_gene_score()
        if not articale:
            self.genes = self.selection.select(self.genes)
            self.genes = self.crossover()      
            self.mutate_all()
            self.calc_gene_score()
            self.keep_generaion_size()
            self.generation +=1
        else:
            self.mutate_all()
            self.calc_gene_score()
            self.genes = self.crossover(k=0)    
            self.calc_gene_score()
            selecton =  RoyaltyRouletteSelection(0.1)
            self.genes = selecton.select(self.genes)
        self.calc_gene_score()

    @staticmethod
    def remove_duplicates_rows(gene_rows,  other_gene_rows: List[int]):
        seen = set()
        output = []
        for index in gene_rows:
            if index not in seen:
                output.append(index)
                seen.add(index)
        
        # In case there were duplicates, replace them with new unique elements
        while len(output) < len(gene_rows):
            new_index = random.choice([i for i in other_gene_rows if i not in seen])
            output.append(new_index)
            seen.add(new_index)
        
        return output

    @staticmethod
    def remove_duplicates_cols(gene_cols, target_col_index, other_gene_cols: List[int]):
        seen = set()
        output = []
        for index in gene_cols:
            if index not in seen:
                output.append(index)
                seen.add(index)
        
        # In case there were duplicates, replace them with new unique elements
        while len(output) < len(gene_cols):
            new_index = random.choice([i for i in other_gene_cols if i not in seen and i != target_col_index])
            output.append(new_index)
            seen.add(new_index)
        
        # In case target column is not in output, replace a random column with it
        if target_col_index not in output:
            replace_index = random.choice([i for i in range(len(output)) if i != target_col_index])
            output[replace_index] = target_col_index
        return output

        
    def crossover_2_genes(self, parent1: Gene, parent2: Gene, target_col_index: int) -> tuple[Gene, Gene]:
        crossover_point_rows = random.randint(1, len(parent1.gene_rows) - 1)
       
        # choose the crossover spot that is not the target column
        if len(parent1.gene_cols) == 2:
            crossover_point_cols = 1
        else:
            while True:
                crossover_point_cols = random.randint(1, len(parent1.gene_cols) - 1)
                if crossover_point_cols != parent1.gene_cols.index(target_col_index):
                    break


        child1_gene_rows = parent1.gene_rows[:crossover_point_rows] + parent2.gene_rows[crossover_point_rows:]
        child1_gene_cols = parent1.gene_cols[:crossover_point_cols] + parent2.gene_cols[crossover_point_cols:]

        child2_gene_rows = parent2.gene_rows[:crossover_point_rows] + parent1.gene_rows[crossover_point_rows:]
        child2_gene_cols = parent2.gene_cols[:crossover_point_cols] + parent1.gene_cols[crossover_point_cols:]

        # Check if target column is in children, if not, add it
        child1_gene_cols = Population.remove_duplicates_cols(child1_gene_cols, target_col_index, parent2.gene_cols)
        child1_gene_rows = Population.remove_duplicates_rows(child1_gene_rows, parent2.gene_rows)
        # remove duplicates and ensure target column in child2
        child2_gene_cols = Population.remove_duplicates_cols(child2_gene_cols, target_col_index, parent1.gene_cols)
        child2_gene_rows = Population.remove_duplicates_rows(child2_gene_rows, parent1.gene_rows)

        # Create new Gene instances
        child1 = Gene(sub_rows=child1_gene_rows, sub_cols=child1_gene_cols)
        child2 = Gene(sub_rows=child2_gene_rows, sub_cols=child2_gene_cols)

        return child1, child2


    def crossover(self, k=0) -> List[Gene]:
        target_col_index = self.dataset.columns.get_loc(self.target_column)

        tmp_gengs = list(self.genes)
        parents_pairs = []

        for _ in range(0, len(self.genes) -k, 2):
            if len(tmp_gengs) >= 2:
                parent_gene_1 = tmp_gengs.pop(random.randrange(len(tmp_gengs)))
                parent_gene_2 = tmp_gengs.pop(random.randrange(len(tmp_gengs)))
                parents_pairs.append((parent_gene_1, parent_gene_2))

        def generate_children(pair):
            parent_gene_1, parent_gene_2 = pair
            return self.crossover_2_genes(parent_gene_1, parent_gene_2, target_col_index)

        with ThreadPoolExecutor() as executor:
            children = list(executor.map(generate_children, parents_pairs))

        new_genes = []
        for pair in children:
            child1, child2 = pair
            new_genes.append(child1)
            new_genes.append(child2)
        return new_genes

    def set_init_genes (self, genes: List[Gene]) -> None:
        if genes:
            self.genes = genes
            self.generation = 0

    @staticmethod
    def get_best_gene(genes: List[Gene]) -> Gene:
        sorted_genes = sorted(genes, key=lambda individual: individual.score if individual.score is not None else float('inf'), reverse=True)
        temp_best = sorted_genes[0]
        return temp_best

    def get_best_pop_gene(self) -> Gene:
        return self.get_best_gene(self.genes)