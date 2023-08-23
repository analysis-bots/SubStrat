import unittest
import pandas as pd
import unittest.mock as mock
from sklearn.datasets import load_iris
from SubStrat.genetic_sub_algorithm.population_initialization import InitRandomPopulation
from SubStrat.genetic_sub_algorithm.population import Population
from SubStrat.genetic_sub_algorithm.selection import RoyaltySelection
from SubStrat.sub_algo_utils.sub_dataset_size import get_reuglar_subdataset_size, get_sub_dataset_size

class TestPopulation(unittest.TestCase):

    def test_sanity_mutate_has_target_col(self):
        data = {'Name':['Jai', 'Princi', 'Gaurav', 'Anuj', 'Geeku'],
        'Age':[27, 24, 22, 32, 15],
        'Address':['Delhi', 'Kanpur', 'Allahabad', 'Kannauj', 'Noida'],
        'Qualification':['Msc', 'MA', 'MCA', 'Phd', '10th']}
        df = pd.DataFrame(data)
        init = InitRandomPopulation(df, 2,2, "Age", 1)
        pop: Population = init.init_population()
        gene = pop.genes[0]
        row_count, col_count = df.shape

        def random1():
            return 1

        with mock.patch("random.random", random1):
            new_gene = pop._mutate_gene(gene, row_count, col_count, 1, 1)
        self.assertIn(1, new_gene.gene_cols)


    def test_crossover_2_genes_has_target_col(self):
        data = {'Name':['Jai', 'Princi', 'Gaurav', 'Anuj', 'Geeku'],
        'Age':[27, 24, 22, 32, 15],
        'Address':['Delhi', 'Kanpur', 'Allahabad', 'Kannauj', 'Noida'],
        'Qualification':['Msc', 'MA', 'MCA', 'Phd', '10th']}
        df = pd.DataFrame(data)
        init = InitRandomPopulation(df, 2,2, "Age", 2)
        pop: Population = init.init_population()
        gene1, gene2 = pop.genes[0], pop.genes[1]
        child1, child2 = pop.crossover_2_genes(gene1, gene2, 1)
        self.assertIn(1, child1.gene_cols)
        self.assertIn(1, child2.gene_cols)

    def test_score_calc(self):
        data = {'Name':['Jai', 'Princi', 'Gaurav', 'Anuj', 'Geeku'],
        'Age':[27, 24, 22, 32, 15],
        'Address':['Delhi', 'Kanpur', 'Allahabad', 'Kannauj', 'Noida'],
        'Qualification':['Msc', 'MA', 'MCA', 'Phd', '10th']}
        df = pd.DataFrame(data)
        init = InitRandomPopulation(df, 2,2, "Age",1)
        pop: Population = init.init_population()
        gene =  pop.genes[0]
        pop._calc_gene_score(gene)
        self.assertIsNotNone(gene.score)

    def test_scores(self):
        data = load_iris(as_frame=True).frame
        init = InitRandomPopulation(data, 4, 50 , "target",500)
        pop: Population = init.init_population()
        pop.calc_gene_score()
        genes_sorted = sorted(pop.genes, key=lambda gene: gene.score, reverse=True)
        best = genes_sorted[0]
        worst = genes_sorted[499]
        self.assertNotEqual(best.score, worst.score)

    def test_selection_no_scores(self):
        data = load_iris(as_frame=True).frame
        init = InitRandomPopulation(data, 4, 50 , "target",500)
        pop: Population = init.init_population()
        with self.assertRaises(RuntimeError):
            pop.selection.select(pop.genes)

    def test_selecion_royaly(self):
        data = load_iris(as_frame=True).frame
        init = InitRandomPopulation(data, 4, 50 , "target",500)
        pop: Population = init.init_population()
        pop.calc_gene_score()
        genes = pop.genes
        royal_rate = 0.1
        sorted_genes = sorted(genes, key=lambda individual: individual.score if individual.score is not None else float('inf'), reverse=True)
        top_k = sorted_genes[:int(len(genes)*royal_rate)]

        r = RoyaltySelection(royal_rate)
        pop.selection = r
        new_genes = pop.selection.select(genes)
        for g in top_k:
            self.assertIn(g, new_genes)
    
    def test_sanity_next_geration(self):
        data = load_iris(as_frame=True).frame
        init = InitRandomPopulation(data, 4, 50 , "target",20)
        pop: Population = init.init_population()
        self.assertEqual(0, pop.generation)
        pop.next_generation()
        self.assertEqual(1, pop.generation)
    
    def test_sanity_next_generaion_same_size(self):
        size = 20
        data = load_iris(as_frame=True).frame
        init = InitRandomPopulation(data, 4, 50 , "target", size)
        pop: Population = init.init_population()
        pop.next_generation()
        self.assertEqual(size, len(pop.genes))


    # def test_end_to_end_small_db(self):
    #     data = load_iris(as_frame=True).frame
    #     num_rows, num_cols = get_sub_dataset_size(data)
    #     init = InitRandomPopulation(data, num_cols, num_rows , "target", 50)
    #     pop: Population = init.init_population()
    #     pop.calc_gene_score()
    #     temp_best = pop.get_best_pop_gene()
    #     best_socre = temp_best.score
    #     print(f"Before running best score: {best_socre}")
    #     for i in range(5):
    #         pop.next_generation()
    #         temp_best = pop.get_best_pop_gene()
    #         print(f"Gen #{i}, Best score: {temp_best.score}")
    #         self.assertGreaterEqual(temp_best.score, best_socre)
    #         best_socre = temp_best.score
    
    # def test_like_article_end_to_end_small_db(self):
    #     data = load_iris(as_frame=True).frame
    #     num_rows, num_cols = get_sub_dataset_size(data)
    #     init = InitRandomPopulation(data, num_cols, num_rows , "target", 50)
    #     pop: Population = init.init_population()
    #     pop.calc_gene_score()
    #     sorted_genes = sorted(pop.genes, key=lambda individual: individual.score if individual.score is not None else float('inf'), reverse=True)
    #     temp_best = sorted_genes[0]
    #     best_socre = temp_best.score
    #     print(f"Before running best score: {best_socre}")
    #     for i in range(5):
    #         pop.next_generation(articale=True)
    #         sorted_genes = sorted(pop.genes, key=lambda individual: individual.score if individual.score is not None else float('inf'), reverse=True)
    #         temp_best = sorted_genes[0]
    #         print(f"Gen #{i}, Best score: {temp_best.score}")
    #         self.assertGreaterEqual(temp_best.score, best_socre)
    #         best_socre = temp_best.score

    # def test_end_to_end_mid_db(self):
    #     data = pd.read_csv("../data/dataset_2_headech_prodrom.csv")
    #     num_rows, num_cols = get_sub_dataset_size(data)
    #     init = InitRandomPopulation(data, num_cols, num_rows , "target", 50)
    #     pop: Population = init.init_population()
    #     pop.calc_gene_score()
    #     temp_best = pop.get_best_pop_gene()
    #     best_socre = temp_best.score
    #     print(f"Before running best score: {best_socre}")
    #     for i in range(5):
    #         pop.next_generation()
    #         temp_best = pop.get_best_pop_gene()
    #         print(f"Gen #{i}, Best score: {temp_best.score}")
    #         self.assertGreaterEqual(temp_best.score, best_socre)
    #         best_socre = temp_best.score

    # def test_like_article_end_to_end_mid_db(self):
    #     data = pd.read_csv("../data/dataset_2_headech_prodrom.csv")
    #     num_rows, num_cols = get_sub_dataset_size(data)
    #     init = InitRandomPopulation(data, num_cols, num_rows , "target", 50)
    #     pop: Population = init.init_population()
    #     pop.calc_gene_score()
    #     sorted_genes = sorted(pop.genes, key=lambda individual: individual.score if individual.score is not None else float('inf'), reverse=True)
    #     temp_best = sorted_genes[0]
    #     best_socre = temp_best.score
    #     print(f"Before running best score: {best_socre}")
    #     for i in range(5):
    #         pop.next_generation(articale=True)
    #         sorted_genes = sorted(pop.genes, key=lambda individual: individual.score if individual.score is not None else float('inf'), reverse=True)
    #         temp_best = sorted_genes[0]
    #         print(f"Gen #{i}, Best score: {temp_best.score}")
    #         self.assertGreaterEqual(temp_best.score, best_socre)
    #         best_socre = temp_best.score

        
if __name__ == '__main__':
    unittest.main()