import pandas as pd
import pytest

from src.application.use_cases.evaluate_classifiers import (
    EvaluateClassifiersUseCase,
    EvaluationResult,
)
from src.domain.entities.hierarchy_graph import FunctionNode, HierarchyGraph
from src.domain.interfaces.classifier import HierarchicalClassifier
from src.domain.interfaces.evaluator import HierarchicalEvaluator


class FakeClassifier(HierarchicalClassifier):
    def __init__(self, predictions: list[str]):
        self._predictions = predictions

    def train(self, X, y, hierarchy):
        pass

    def predict(self, X: pd.DataFrame) -> list[str]:
        return self._predictions[: len(X)]


class FakeEvaluator(HierarchicalEvaluator):
    def evaluate(self, y_true: list[str], y_pred: list[str]) -> dict:
        return {"hP": 0.8, "hR": 0.7, "hF": 0.75}


@pytest.fixture
def X_test():
    return pd.DataFrame({
        "seq_length": [6.0, 12.0],
        "molecular_weight": [600.0, 1200.0],
        "aa_A": [0.2, 0.1],
    })


@pytest.fixture
def y_test():
    return pd.Series(["GO:0005524", "GO:0003824"])


class TestEvaluateClassifiersUseCase:
    def test_returns_evaluation_result(self, X_test, y_test):
        clf = FakeClassifier(["GO:0005524", "GO:0003824"])
        evaluator = FakeEvaluator()
        use_case = EvaluateClassifiersUseCase(evaluator)

        result = use_case.execute(clf, "FakeRF", X_test, y_test)
        assert isinstance(result, EvaluationResult)
        assert result.classifier_name == "FakeRF"

    def test_contains_metrics(self, X_test, y_test):
        clf = FakeClassifier(["GO:0005524", "GO:0003824"])
        evaluator = FakeEvaluator()
        use_case = EvaluateClassifiersUseCase(evaluator)

        result = use_case.execute(clf, "FakeRF", X_test, y_test)
        assert "hP" in result.metrics
        assert "hR" in result.metrics
        assert "hF" in result.metrics

    def test_contains_predictions(self, X_test, y_test):
        clf = FakeClassifier(["GO:0005524", "GO:0003824"])
        evaluator = FakeEvaluator()
        use_case = EvaluateClassifiersUseCase(evaluator)

        result = use_case.execute(clf, "FakeRF", X_test, y_test)
        assert result.y_pred == ["GO:0005524", "GO:0003824"]
