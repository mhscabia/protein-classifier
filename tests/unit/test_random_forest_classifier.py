import pandas as pd
import pytest

from src.domain.entities.hierarchy_graph import FunctionNode, HierarchyGraph
from src.infrastructure.models.random_forest_classifier import (
    RandomForestHierarchicalClassifier,
    _get_feature_columns,
    _parse_go_terms,
)


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
    """molecular_function -> binding -> ATP binding
                          -> catalytic activity
    """
    graph = HierarchyGraph()
    graph.add_node(FunctionNode(
        term_id="GO:0003674", name="molecular_function",
        parent_ids=[], children_ids=["GO:0005488", "GO:0003824"],
    ))
    graph.add_node(FunctionNode(
        term_id="GO:0005488", name="binding",
        parent_ids=["GO:0003674"], children_ids=["GO:0005524"],
    ))
    graph.add_node(FunctionNode(
        term_id="GO:0005524", name="ATP binding",
        parent_ids=["GO:0005488"], children_ids=[],
    ))
    graph.add_node(FunctionNode(
        term_id="GO:0003824", name="catalytic activity",
        parent_ids=["GO:0003674"], children_ids=[],
    ))
    return graph


@pytest.fixture
def train_data():
    """Dataset sintetico com features numericas e GO terms."""
    n = 30
    data = {
        "protein_id": [f"P{i:04d}" for i in range(n)],
        "sequence": ["ACDEFG"] * n,
        "go_terms": (
            ["GO:0005524"] * 10
            + ["GO:0003824"] * 10
            + ["GO:0005524;GO:0003824"] * 10
        ),
        "seq_length": [6.0] * 10 + [12.0] * 10 + [9.0] * 10,
        "molecular_weight": [600.0] * 10 + [1200.0] * 10 + [900.0] * 10,
        "aa_A": [0.2] * 10 + [0.1] * 10 + [0.15] * 10,
        "aa_C": [0.1] * 10 + [0.2] * 10 + [0.15] * 10,
    }
    return pd.DataFrame(data)


class TestHelpers:
    def test_get_feature_columns(self):
        df = pd.DataFrame({
            "protein_id": ["P1"],
            "seq_length": [1.0],
            "molecular_weight": [100.0],
            "aa_A": [0.5],
            "go_terms": ["GO:0001"],
        })
        cols = _get_feature_columns(df)
        assert "seq_length" in cols
        assert "molecular_weight" in cols
        assert "aa_A" in cols
        assert "protein_id" not in cols
        assert "go_terms" not in cols

    def test_parse_go_terms_single(self):
        series = pd.Series(["GO:0001"])
        result = _parse_go_terms(series)
        assert result == [["GO:0001"]]

    def test_parse_go_terms_multiple(self):
        series = pd.Series(["GO:0001;GO:0002"])
        result = _parse_go_terms(series)
        assert result == [["GO:0001", "GO:0002"]]

    def test_parse_go_terms_empty(self):
        series = pd.Series([""])
        result = _parse_go_terms(series)
        assert result == [[]]


class TestRandomForestClassifier:
    def test_train_and_predict(self, config, hierarchy, train_data):
        clf = RandomForestHierarchicalClassifier(config=config)
        clf.train(train_data, train_data["go_terms"], hierarchy)

        predictions = clf.predict(train_data.head(5))
        assert len(predictions) == 5
        assert all(isinstance(p, str) for p in predictions)

    def test_predictions_contain_ancestors(self, config, hierarchy, train_data):
        clf = RandomForestHierarchicalClassifier(config=config)
        clf.train(train_data, train_data["go_terms"], hierarchy)

        predictions = clf.predict(train_data.head(5))
        for pred in predictions:
            if not pred:
                continue
            terms = pred.split(";")
            for term in terms:
                if term == "GO:0003674":
                    continue
                ancestors = hierarchy.get_ancestors(term)
                for anc in ancestors:
                    assert anc in terms, (
                        f"Ancestral {anc} de {term} ausente na predicao"
                    )

    def test_raises_without_features(self, config, hierarchy):
        df = pd.DataFrame({"protein_id": ["P1"], "go_terms": ["GO:0001"]})
        clf = RandomForestHierarchicalClassifier(config=config)

        with pytest.raises(ValueError, match="feature"):
            clf.train(df, df["go_terms"], hierarchy)
