import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.domain.entities.hierarchy_graph import FunctionNode, HierarchyGraph
from src.domain.interfaces.classifier import HierarchicalClassifier
from src.infrastructure.prediction.inference_pipeline import InferencePipeline


class FakeClassifier(HierarchicalClassifier):
    """Classificador falso que retorna predicoes fixas."""

    def __init__(self, predictions: list[str]):
        self._predictions = predictions

    def train(self, X, y, hierarchy):
        pass

    def predict(self, X: pd.DataFrame) -> list[str]:
        self.last_X = X
        return self._predictions


@pytest.fixture
def hierarchy():
    h = HierarchyGraph()
    h.add_node(FunctionNode("GO:0003674", "molecular_function"))
    h.add_node(FunctionNode("GO:0005488", "binding", parent_ids=["GO:0003674"]))
    return h


class TestInferencePipeline:
    def test_returns_go_terms(self, hierarchy):
        clf = FakeClassifier(["GO:0005488;GO:0003674"])
        pipeline = InferencePipeline(classifier=clf)

        result = pipeline.predict("ACDEFG")

        assert isinstance(result, set)
        assert "GO:0005488" in result
        assert "GO:0003674" in result

    def test_extracts_correct_features(self, hierarchy):
        clf = FakeClassifier(["GO:0005488"])
        pipeline = InferencePipeline(classifier=clf)

        pipeline.predict("AAA")

        df = clf.last_X
        assert len(df) == 1
        assert "seq_length" in df.columns
        assert "molecular_weight" in df.columns
        assert "aa_A" in df.columns
        assert df["seq_length"].iloc[0] == 3.0
        assert df["aa_A"].iloc[0] == pytest.approx(1.0)

    def test_applies_scaler_when_provided(self, hierarchy):
        clf = FakeClassifier(["GO:0005488"])

        scaler = StandardScaler()
        dummy_data = pd.DataFrame({
            "seq_length": [100.0, 200.0, 300.0],
            "molecular_weight": [10000.0, 20000.0, 30000.0],
            **{f"aa_{aa}": np.random.rand(3) for aa in "ACDEFGHIKLMNPQRSTVWY"},
        })
        scaler.fit(dummy_data)

        pipeline = InferencePipeline(classifier=clf, scaler=scaler)
        pipeline.predict("ACDEF")

        df = clf.last_X
        # Com scaler aplicado, valores devem ser transformados (nao os raw)
        assert df["seq_length"].iloc[0] != 5.0

    def test_no_scaler_uses_raw_values(self, hierarchy):
        clf = FakeClassifier(["GO:0005488"])
        pipeline = InferencePipeline(classifier=clf, scaler=None)

        pipeline.predict("ACDEF")

        df = clf.last_X
        assert df["seq_length"].iloc[0] == 5.0

    def test_empty_prediction_returns_empty_set(self, hierarchy):
        clf = FakeClassifier([""])
        pipeline = InferencePipeline(classifier=clf)

        result = pipeline.predict("AAA")

        assert result == set()
