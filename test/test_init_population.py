import unittest
import pandas as pd
from SubStrat.genetic_sub_algorithm.population_initialization import InitRandomPopulation
from SubStrat.genetic_sub_algorithm.population import Population

class TestInitPopulation(unittest.TestCase):

    def test_sanity_size_init_pop(self):
        data = {'Name':['Jai', 'Princi', 'Gaurav', 'Anuj', 'Geeku'],
        'Age':[27, 24, 22, 32, 15],
        'Address':['Delhi', 'Kanpur', 'Allahabad', 'Kannauj', 'Noida'],
        'Qualification':['Msc', 'MA', 'MCA', 'Phd', '10th']}
        
        
        df = pd.DataFrame(data)
        init = InitRandomPopulation(df, 2,2, "Age", 20)
        pop: Population = init.init_population()
        self.assertEqual(len(pop.genes), 20)

    
    def test_sanity_init_has_traget_col(self):
        data = {'Name':['Jai', 'Princi', 'Gaurav', 'Anuj', 'Geeku'],
        'Age':[27, 24, 22, 32, 15],
        'Address':['Delhi', 'Kanpur', 'Allahabad', 'Kannauj', 'Noida'],
        'Qualification':['Msc', 'MA', 'MCA', 'Phd', '10th']}
        df = pd.DataFrame(data)
        init = InitRandomPopulation(df, 2,2, "Age", 20)
        pop: Population = init.init_population()
        for gene in pop.genes:
            self.assertIn(1, gene.gene_cols)

    
if __name__ == '__main__':
    unittest.main()