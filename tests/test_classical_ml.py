import pytest
import pandas as pd
import numpy as np
from src.classical.pipeline import MLPipeline
from src.classical.factory import ModelFactory

def test_model_factory():
    """
    Test if ModelFactory correctly instantiates models.
    """
    model = ModelFactory.get_model("random_forest", task="classification")
    assert model is not None
    assert hasattr(model, "fit")

def test_ml_pipeline_fit_predict():
    """
    Test the end-to-end Classical ML pipeline with dummy data.
    """
    # Dummy data
    X = pd.DataFrame({
        "age": [25, 30, 35, 40],
        "income": [50000, 60000, 70000, 80000],
        "category": ["A", "B", "A", "B"]
    })
    y = np.array([0, 1, 0, 1])
    
    pipeline = MLPipeline(
        model_name="random_forest",
        numeric_features=["age", "income"],
        categorical_features=["category"]
    )
    
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    
    assert len(preds) == 4
    assert set(preds).issubset({0, 1})
