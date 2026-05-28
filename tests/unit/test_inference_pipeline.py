import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.domain.entities.hierarchy_graph import FunctionNode, HierarchyGraph
from src.domain.interfaces.classifier import HierarchicalClassifier
from src.infrastructure.prediction.inference_pipeline import InferencePipeline

ESM_DIM = 4


class FakeClassifier(HierarchicalClassifier):
    def __init__(self, predictions: list[str]):
        self._predictions = predictions

    def train(self, X, y, hierarchy):
        pass

    def predict(self, X: pd.DataFrame) -> list[str]:
        self.last_X = X
        return self._predictions


class FakeEmbedder:
    def embed_single(self, sequence: str) -> np.ndarray:
        return np.ones(ESM_DIM) * len(sequence)


@pytest.fixture
def hierarchy():
    h = HierarchyGraph()
    h.add_node(FunctionNode("GO:0003674", "molecular_function"))
    h.add_node(FunctionNode("GO:0005488", "binding", parent_ids=["GO:0003674"]))
    return h


class TestInferencePipeline:
    def test_returns_go_terms(self, hierarchy):
        clf = FakeClassifier(["GO:0005488;GO:0003674"])
        pipeline = InferencePipeline(classifier=clf, embedder=FakeEmbedder())

        result = pipeline.predict("ACDEFG")

        assert isinstance(result, set)
        assert "GO:0005488" in result
        assert "GO:0003674" in result

    def test_extracts_correct_features(self, hierarchy):
        clf = FakeClassifier(["GO:0005488"])
        pipeline = InferencePipeline(classifier=clf, embedder=FakeEmbedder())

        pipeline.predict("AAA")

        df = clf.last_X
        assert len(df) == 1
        assert all(c.startswith("esm_") for c in df.columns)
        assert len(df.columns) == ESM_DIM
        assert df["esm_0"].iloc[0] == pytest.approx(3.0)

    def test_applies_scaler_when_provided(self, hierarchy):
        clf = FakeClassifier(["GO:0005488"])
        embedder = FakeEmbedder()

        scaler = StandardScaler()
        dummy = np.random.rand(10, ESM_DIM)
        scaler.fit(dummy)

        pipeline = InferencePipeline(classifier=clf, scaler=scaler, embedder=embedder)
        pipeline.predict("ACDEF")

        df = clf.last_X
        assert df["esm_0"].iloc[0] != pytest.approx(5.0)

    def test_no_scaler_uses_raw_values(self, hierarchy):
        clf = FakeClassifier(["GO:0005488"])
        pipeline = InferencePipeline(classifier=clf, scaler=None, embedder=FakeEmbedder())

        pipeline.predict("ACDEF")

        df = clf.last_X
        assert df["esm_0"].iloc[0] == pytest.approx(5.0)

    def test_empty_prediction_returns_empty_set(self, hierarchy):
        clf = FakeClassifier([""])
        pipeline = InferencePipeline(classifier=clf, embedder=FakeEmbedder())

        result = pipeline.predict("AAA")

        assert result == set()
