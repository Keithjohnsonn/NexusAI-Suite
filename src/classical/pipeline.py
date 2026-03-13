import os
import joblib
import pandas as pd
from typing import Any, Optional, Dict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.classical.factory import ModelFactory
from src.utils.logging import log

class MLPipeline:
    """
    A unified interface for training, evaluating, and persisting 
    Classical Machine Learning models.
    """
    
    def __init__(
        self, 
        model_name: str, 
        task: str = "classification", 
        params: Optional[Dict[str, Any]] = None,
        numeric_features: Optional[list] = None,
        categorical_features: Optional[list] = None
    ):
        self.model_name = model_name
        self.task = task
        self.params = params or {}
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        
        # Build preprocessor
        transformers = []
        if self.numeric_features:
            transformers.append(("num", StandardScaler(), self.numeric_features))
        if self.categorical_features:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_features))
            
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="passthrough"
        )
        
        # Build core model
        self.model = ModelFactory.get_model(model_name, task, self.params)
        
        # Create full pipeline
        self.pipeline = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("model", self.model)
        ])
        
    def fit(self, X: pd.DataFrame, y: Any) -> "MLPipeline":
        """
        Trains the full pipeline.
        """
        log.info(f"Starting training for {self.model_name}...")
        self.pipeline.fit(X, y)
        log.info("Training complete.")
        return self
        
    def predict(self, X: pd.DataFrame) -> Any:
        """
        Predicts labels or values for input data.
        """
        log.info(f"Predicting with {self.model_name}...")
        return self.pipeline.predict(X)
        
    def save(self, path: str):
        """
        Persists the pipeline to disk.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)
        log.info(f"Model saved to: {path}")
        
    @staticmethod
    def load(path: str) -> Pipeline:
        """
        Loads a persisted pipeline from disk.
        """
        log.info(f"Loading model from: {path}")
        return joblib.load(path)
