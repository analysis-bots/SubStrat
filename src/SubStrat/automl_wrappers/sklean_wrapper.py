from autosklearn.classification import AutoSklearnClassifier
from autosklearn.constants import *


from autosklearn.classification import AutoSklearnClassifier
from smac.optimizer.smbo import SMBO
from smac.runhistory.runhistory import RunInfo, RunValue

def clone_classifier(input_classifier: AutoSklearnClassifier = None, 
                     callback=None) -> AutoSklearnClassifier:
    """
    Clone an existing classifier or create a new one.

    Parameters:
    - input_classifier: An existing AutoSklearnClassifier instance.
    - callback: Optional callback function to handle early stopping.

    Returns:
    - A cloned instance of the AutoSklearnClassifier.
    """
    
    new_cls = AutoSklearnClassifier()
    if input_classifier:
        new_cls.set_params(**input_classifier.get_params())
    new_cls.n_jobs=-1
    new_cls.memory_limit = 32768*2
    new_cls.time_left_for_this_task = 60*25
    
    # If callback is provided, it can be used during the fit process
    if callback:
        new_cls.get_trials_callback = callback
    
    return new_cls

def fetch_best_model_name_from_cls(cls: AutoSklearnClassifier) -> str:
    """
    Fetch the best model's name from the classifier's leaderboard.

    Parameters:
    - cls: An AutoSklearnClassifier instance.

    Returns:
    - Best model's name.
    """
    # best_model_loc = list(cls.show_models())[0]
    # best_model = cls.show_models()[best_model_loc]
    # model_name
    leaderboard = cls.leaderboard()
    best_model_data = leaderboard.loc[leaderboard['rank'] == 1].to_dict()
    model_id = leaderboard.loc[leaderboard['rank'] == 1].index[0]
    model_name = best_model_data['type'][model_id]
    # print(f"Model ID: {model_id}")
    # print(f"Model Name: {model_name}")
    
    return model_name

def create_finetune_classifier(input_classifier: AutoSklearnClassifier, 
                               trained_classifier: AutoSklearnClassifier, 
                               callback=None) -> AutoSklearnClassifier:
    """
    Create a fine-tuned classifier based on a trained classifier.

    Parameters:
    - input_classifier: Original classifier instance to base the fine-tuned classifier upon.
    - trained_classifier: Classifier instance that has been trained.
    - callback: Optional callback function to handle early stopping.

    Returns:
    - A fine-tuned instance of the AutoSklearnClassifier.
    """
    
    fine_tuned_cls = AutoSklearnClassifier()
    model_name = fetch_best_model_name_from_cls(trained_classifier)
    
    # Updating parameters for the fine-tuned classifier
    if input_classifier:
        params = input_classifier.get_params()
    else:
        params = dict()
    params.update({
        "include": {'classifier': [model_name]},
        "initial_configurations_via_metalearning": 0,
        # "ensemble_kwargs": {'ensemble_size': 1},
        "memory_limit": 32768,
        "n_jobs": -1,
        "time_left_for_this_task":60*15
    })

    
    fine_tuned_cls.set_params(**params)
    
    # If callback is provided, it can be used during the fit process
    # Note: you may need to adjust this based on how you want the callback to be used.
    if callback:
        fine_tuned_cls.get_trials_callback = callback
    
    return fine_tuned_cls

def make_callback(desired_accuracy=0.95):
    """
    Create a callback function to handle early stopping based on desired accuracy.

    Parameters:
    - desired_accuracy: Desired accuracy threshold to trigger the callback.

    Returns:
    - A callback function.
    """
    
    corresponding_cost = 1 - desired_accuracy
    
    def callback(smbo: SMBO, run_info: RunInfo, result: RunValue, time_left: float) -> bool:
        """Stop early if we achieve desired accuracy"""
        # If the achieved cost is less than or equal to the corresponding cost of the desired accuracy
        if result.cost <= corresponding_cost:
            print(f"Stopping early! Achieved accuracy of {1 - result.cost:.4f} which is above or equal to the threshold of {desired_accuracy:.4f}.")
            return False
        return True
    
    return callback
