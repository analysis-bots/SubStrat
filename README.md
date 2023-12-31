# SubStrat Package: A Subset-Based Optimization Strategy for Faster AutoML

`SubStrat` is a Python package designed to optimize AutoML running times on large datasets.
It wraps existing AutoML tools such as AutoSklearn, TPOT and H2O, and instead of execute them on directly on the entire dataset, SubStrat uses a genetic-based algorithm to find a small yet representative data subset which preserves a characteristic of the original one. It then employs the AutoML tool on the small subset, and finally, it refines the resulted pipeline by executing a restricted, much shorter, AutoML process on the large dataset.

 *** `SubStrat` is still under development, currently supporting `AutoSklearn`. 
Follow us here for updates. ***


`SubStrat`is based on the VLDB 2023 Paper:
- Teddy Lazebnik, Amit Somech, and Abraham Itzhak Weinberg. [SubStrat: A
Subset-Based Optimization Strategy for Faster AutoML](https://www.vldb.org/pvldb/vol16/p772-somech.pdf). PVLDB, 16(4): 772 -
780, 2022. doi:10.14778/3574245.3574261

## Features

- **Automated Machine Learning (AutoML)**: Using the power of tools such as `AutoSklearn` library, users can seamlessly train and fine-tune machine learning models on their dataset.
  
- **Genetic Dataset Summarization**: `SubStrat` uses a genetic algorithm for summarizing datasets, providing concise data representations while retaining vital information.

#### SubStrat Flow
- Run the genDST algorithm for finding a data subset that preserves dataset entropy.
- Run the AutoML tool on the data subset (fast) and obtain intermediate ML pipeline.
- Improve the intermediate pipeline by employing a restricted AutoML run on the full dataset.
- Return the optimized ML pipeline.

#### Installation and Setupt
Very recomended to use `venv`.
```bash
python3 -m venv subsrat_vev
cd subsrat_vev
sourch bin/activate
```
Installing the SubStrat package.

```bash
pip install substart-automl
```
## Usage

```python
from SubStrat import SubStrat

# Initialize SubStrat with a dataset and target column
s = SubStrat(dataset=my_dataset, target_col_name='target')
# Excute SubStrat flow
cls = s.run()
```
[See basic example here](Examples/Notebooks/basic_example.ipynb)  
[Example in google Colab](https://colab.research.google.com/drive/1AnsUZuiPGvn1bVfV0YuRMkjOKSoUlUQx?usp=sharing)

## Classes

### SubStrat

Provides the primary interface for the AutoML functionalities.

#### Attributes:

- `dataset`: Input dataset (pandas DataFrame).
- `target_col_name`: Name of the target column in the dataset.
- `input_classifier`: Classifier instance (optional). Defaults to an instance from `AutoSklearn`.
- `summary_algorithm`: Algorithm to summarize the dataset. Defaults to `GeneticSubAlgorithmn`.
- `desired_accuracy`:Desired accuracy for the output classifier.

#### Methods:
 - `run()`: Executes the SubStrat flow, and returns `AutoSklearnClassifier`.

### GeneticSubAlgorithmn

Implements the genetic algorithm for dataset summarization.

#### Attributes:

- `dataset`: The original dataset that needs to be summarized.
- `target_column_name`: The name of the target column in the dataset.
- `sub_row_size`: The number of rows for the subset of the dataset (summary). If not provided, it will be calculated based on a predefined rule.
- `sub_col_size`: The number of columns for the subset of the dataset (summary). If not provided, it will be calculated based on a predefined rule.
- `population_size`: The number of individuals in the population for the genetic algorithm.
- `fitness`: The fitness function used in the genetic algorithm. If not provided, a default fitness function will be used.
- `selection`: The selection operator used in the genetic algorithm. If not provided, a default selection operator will be used.
- `mutation_rate`: The mutation rate used in the genetic algorithm.
- `num_generation`: The number of generations the genetic algorithm will run for.
- `init_pop`: The algorithm used to initialize the population for the genetic algorithm. If not provided, a default algorithm will be used.
- `stagnation_limit`: Number of generations without improvement in best gene score before stopping.
- `time_limit`: Maximum time in seconds the `run` function can execute.
#### Methods:

- `run()`: Executes the genetic algorithm and returns the best subset of the dataset.


## futute features
 - Add verbose mode.
 - Add the option to use more AutoML frameworks, like TPOT.
 - Make SubStrat more configable by the user.
 - Make the UX more friendly.


# Citing information
Teddy Lazebnik, Amit Somech, and Abraham Itzhak Weinberg. 2022. SubStrat: A Subset-Based Optimization Strategy for Faster AutoML. Proc. VLDB Endow. 16, 4 (December 2022), 772–780. https://doi.org/10.14778/3574245.3574261 


