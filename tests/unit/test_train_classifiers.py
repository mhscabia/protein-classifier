import pandas as pd
import pytest

from src.application.use_cases.train_classifiers import (
    TrainClassifiersUseCase,
    TrainResult,
)
from src.domain.entities.hierarchy_graph import FunctionNode, HierarchyGraph
from src.domain.interfaces.classifier import HierarchicalClassifier


class FakeClassifier(HierarchicalClassifier):
    def __init__(self):
        self.trained = False
        self.train_count = 0

    def train(self, X: pd.DataFrame, y: pd.Series, hierarchy: HierarchyGraph) -> None:
        self.trained = True
        self.train_count = len(X)

    def predict(self, X: pd.DataFrame) -> list[str]:
        return ["GO:0001"] * len(X)


@pytest.fixture
def config():
    return {
        "data": {
            "raw_path": "data/raw/",
            "processed_path": "data/processed/",
            "uniprot_limit": 500,
            "go_namespace": "molecular_function",
        },
        "model": {"random_seed": 42, "test_size": 0.2, "validation_size": 0.1},
        "logging": {"level": "INFO"},
    }


@pytest.fixture
def hierarchy():
    graph = HierarchyGraph()
    graph.add_node(FunctionNode(term_id="GO:0001", name="root"))
    return graph


@pytest.fixture
def processed_data():
    n = 20
    return pd.DataFrame({
        "protein_id": [f"P{i:04d}" for i in range(n)],
        "sequence": ["ACDEFG"] * n,
        "go_terms": ["GO:0001"] * n,
        "seq_length": [6.0] * n,
        "molecular_weight": [600.0] * n,
        "aa_A": [0.17] * n,
    })


class TestTrainClassifiersUseCase:
    def test_returns_train_result(self, config, hierarchy, processed_data):
        clf = FakeClassifier()
        use_case = TrainClassifiersUseCase(clf, config=config)

        result = use_case.execute(processed_data, hierarchy)
        assert isinstance(result, TrainResult)

    def test_classifier_is_trained(self, config, hierarchy, processed_data):
        clf = FakeClassifier()
        use_case = TrainClassifiersUseCase(clf, config=config)

        result = use_case.execute(processed_data, hierarchy)
        assert result.classifier is clf
        assert clf.trained is True

    def test_splits_data(self, config, hierarchy, processed_data):
        clf = FakeClassifier()
        use_case = TrainClassifiersUseCase(clf, config=config)

        result = use_case.execute(processed_data, hierarchy)
        expected_test = int(len(processed_data) * config["model"]["test_size"])
        assert len(result.X_test) == expected_test
        assert len(result.y_test) == expected_test
        assert clf.train_count == len(processed_data) - expected_test
