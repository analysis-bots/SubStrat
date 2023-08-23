from typing import List
from SubStrat.genetic_sub_algorithm.gene import Gene
import random
from abc import ABC, abstractmethod
import numpy as np

class BaseSelection(ABC):
    def __init__(self) -> None:
        pass

    def select(self, genes: List[Gene], *args, **kwargs) -> List[Gene]:
        if not all(gene.score is not None for gene in genes):
            raise RuntimeError("Genes has no score, please make sure to calculate scores first.")
        return self._select(genes, *args, **kwargs)
    
    @abstractmethod
    def _select( genes: List[Gene]) -> List[Gene]:
        pass
    

class RoyaltySelection(BaseSelection):
    def __init__(self, royalty_rate: float) -> None:
        self.royalty_rate = royalty_rate
    
    def _select(self, genes: List[Gene]) -> List[Gene]:

        s = len(genes)
        k = int(s * self.royalty_rate)
        genes.sort(key=lambda individual: individual.score if individual.score is not None else 0, reverse=True)

        # Automatically select the top k individuals
        selected = genes[:k]
        
        return selected


class RoyaltyRouletteSelection(BaseSelection):
    def __init__(self, royalty_rate: float) -> None:
        self.royalty_rate = royalty_rate
    
    def _select(self, genes: List[Gene]) -> List[Gene]:

        s = len(genes)
        k = int(s * self.royalty_rate)
        genes.sort(key=lambda individual: individual.score if individual.score is not None else 0, reverse=True)

        # Automatically select the top k individuals
        selected = genes[:k]

        rest_of_genes = genes[k:]
        total_fitness = sum([gene.score for gene in rest_of_genes])
        
        # Calculate selection probabilities
        selection_probs = [gene.score / total_fitness for gene in rest_of_genes]
        
        # Sample (with replacement) from rest_of_genes using the selection probabilities
        # to fill the rest of the next generation population
        selected.extend(np.random.choice(rest_of_genes, size=len(genes) - k, p=selection_probs))        
        return selected


        
    
    
