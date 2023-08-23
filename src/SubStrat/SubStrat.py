import pandas as pd
import sklearn
from datetime import datetime
import sklearn.metrics
import sklearn.model_selection
from autosklearn.classification import AutoSklearnClassifier
from SubStrat.summary_algorithm.basic_summary_algorithm import BasicSummaryAlgorithm
from SubStrat.summary_algorithm.genetic_sub_algorithm import GeneticSubAlgorithmn
from SubStrat.automl_wrappers.sklean_wrapper import (clone_classifier, 
                                                     create_finetune_classifier, 
                                                     make_callback)

class SubStrat:
    def __init__(self, 
                 dataset: pd.DataFrame, 
                 target_col_name: str, 
                 cls: AutoSklearnClassifier = None,
                 summary_algorithm: BasicSummaryAlgorithm = None, 
                 summary_algo_time_limit: float = float("inf"),
                 desired_accuracy: float = 0.95, 
                 random_state: int = 1):
        """
        Initialize the SubStrat class.

        Parameters:
        - dataset: Full dataset as a DataFrame.
        - target_col_name: Name of the target column.
        - cls: An instance of AutoSklearnClassifier.
        - summary_algorithm: The algorithm to summarize the dataset.
        - summary_algo_time_limit: Time limit for the summary algorithm.
        - desired_accuracy: Desired accuracy for the output classifier.
        - random_state: Random seed for reproducibility. (default is 1)
        """
        
        self.dataset = dataset
        self.target_col_name = target_col_name 
        self.input_classifier = cls
        self.desired_accuracy = desired_accuracy
        
        if not summary_algorithm:
            self.summary_algorithm = GeneticSubAlgorithmn(dataset, target_col_name, time_limit=summary_algo_time_limit)
        else:
            self.summary_algorithm = summary_algorithm
        self.random_state = random_state
        
    def _split_dataset(self, dataset: pd.DataFrame):
        """
        Helper function to split the dataset into train and test sets.
        """
        X = dataset.drop(self.target_col_name, axis=1).values
        Y = dataset[self.target_col_name].values
        return sklearn.model_selection.train_test_split(X, Y, random_state=self.random_state)

    def _train_and_evaluate(self, classifier: AutoSklearnClassifier, X_train, y_train, X_test, y_test, message=""):
        """
        Helper function to train a classifier and evaluate its performance.
        """
        
        print(f"Start fit {message} at {datetime.now()}")
        classifier.fit(X_train, y_train, X_test, y_test)
        y_hat = classifier.predict(X_test)
        accuracy = sklearn.metrics.accuracy_score(y_test, y_hat)
        print(f"{message} Accuracy score: {accuracy:.4f}")

    def run(self) -> AutoSklearnClassifier:
        print("Starting summary algorithm")
        subset_dataset:pd.DataFrame = self.summary_algorithm.run()

        # Use helper function to split datasets
        X_train, X_test, y_train, y_test = self._split_dataset(self.dataset)
        SUB_X_train, SUB_X_test, SUB_y_train, SUB_y_test = self._split_dataset(subset_dataset)
        
        # Use helper function to train and evaluate classifier on subset data
        sub_callback = make_callback(0.85)
        sub_cls = clone_classifier(self.input_classifier, sub_callback)
        self._train_and_evaluate(sub_cls, SUB_X_train, SUB_y_train, SUB_X_test, SUB_y_test, message="Sub data")
        
        # Fine-tune the classifier
        full_callback = make_callback(self.desired_accuracy)
        fine_tune_classifier = create_finetune_classifier(self.input_classifier, sub_cls, callback=full_callback)
        
        # Use helper function to train and evaluate the fine-tuned classifier on full data
        self._train_and_evaluate(fine_tune_classifier, X_train, y_train, X_test, y_test, message="Fine-tuned")
        return fine_tune_classifier
        
