from typing import Any, Dict, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from src.utils.logging import log

class ModelFactory:
    """
    A factory class to instantiate machine learning models based on type and task.
    """
    
    _MODELS = {
        "classification": {
            "random_forest": RandomForestClassifier,
            "gradient_boosting": GradientBoostingClassifier,
            "svm": SVC,
            "logistic_regression": LogisticRegression,
        },
        "regression": {
            "random_forest": RandomForestRegressor,
            "svm": SVR,
            "linear_regression": LinearRegression,
        }
    }

    @staticmethod
    def get_model(
        model_name: str, 
        task: str = "classification", 
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Returns an instance of the requested model.
        """
        if task not in ModelFactory._MODELS:
            log.error(f"Unsupported task: {task}")
            raise ValueError(f"Task '{task}' is not supported.")
            
        if model_name not in ModelFactory._MODELS[task]:
            log.error(f"Model '{model_name}' not found for task '{task}'")
            available = list(ModelFactory._MODELS[task].keys())
            raise ValueError(f"Model '{model_name}' not found. Available: {available}")

        params = params or {}
        model_class = ModelFactory._MODELS[task][model_name]
        log.info(f"Instantiating {model_name} for {task} with params: {params}")
        return model_class(**params)
