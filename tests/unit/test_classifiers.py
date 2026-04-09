import numpy as np
import pandas as pd
import pytest

from src.domain.entities.hierarchy_graph import FunctionNode, HierarchyGraph
from src.infrastructure.models.random_forest_classifier import (
    RandomForestHierarchicalClassifier,
)
from src.infrastructure.models.svm_classifier import SVMHierarchicalClassifier


@pytest.fixture
def config():
    return {
        "data": {
            "raw_path": "data/raw/",
            "processed_path": "data/processed/",
            "uniprot_limit": 500,
            "go_namespace": "molecular_function",
        },
        "model": {
            "random_seed": 42,
            "test_size": 0.2,
            "validation_size": 0.1,
            "rf_n_estimators": 10,
        },
        "logging": {"level": "INFO"},
    }


@pytest.fixture
def hierarchy():
    """Hierarquia: root -> binding -> ATP_binding
                   root -> catalytic"""
    graph = HierarchyGraph()
    graph.add_node(
        FunctionNode(
            term_id="GO:0003674",
            name="molecular_function",
            parent_ids=[],
            children_ids=["GO:0005488", "GO:0003824"],
        )
    )
    graph.add_node(
        FunctionNode(
            term_id="GO:0005488",
            name="binding",
            parent_ids=["GO:0003674"],
            children_ids=["GO:0005524"],
        )
    )
    graph.add_node(
        FunctionNode(
            term_id="GO:0005524",
            name="ATP binding",
            parent_ids=["GO:0005488"],
            children_ids=[],
        )
    )
    graph.add_node(
        FunctionNode(
            term_id="GO:0003824",
            name="catalytic activity",
            parent_ids=["GO:0003674"],
            children_ids=[],
        )
    )
    return graph


@pytest.fixture
def training_data():
    """Dados sinteticos com 2 clusters separaveis."""
    rng = np.random.RandomState(42)

    n_binding = 30
    n_catalytic = 30

    X_binding = rng.randn(n_binding, 5) + np.array([2, 0, 0, 0, 0])
    X_catalytic = rng.randn(n_catalytic, 5) + np.array([-2, 0, 0, 0, 0])

    X = pd.DataFrame(
        np.vstack([X_binding, X_catalytic]),
        columns=["f1", "f2", "f3", "f4", "f5"],
    )

    y_labels = ["GO:0005524"] * n_binding + ["GO:0003824"] * n_catalytic
    y = pd.Series(y_labels, name="go_terms")

    return X, y


@pytest.fixture
def rf_classifier(config):
    return RandomForestHierarchicalClassifier(config=config)


@pytest.fixture
def svm_classifier(config):
    return SVMHierarchicalClassifier(config=config)


# ---------------------------------------------------------------------------
# Testes compartilhados (parametrizados para RF e SVM)
# ---------------------------------------------------------------------------


class TestTraining:
    @pytest.mark.parametrize("clf_fixture", ["rf_classifier", "svm_classifier"])
    def test_train_creates_node_classifiers(
        self, clf_fixture, hierarchy, training_data, request
    ):
        clf = request.getfixturevalue(clf_fixture)
        X, y = training_data
        clf.train(X, y, hierarchy)

        # root (GO:0003674) deve ser all_positive (todos possuem)
        assert "GO:0003674" in clf._all_positive_nodes

        # binding e catalytic devem ter classificadores binarios
        assert len(clf._node_classifiers) > 0

    @pytest.mark.parametrize("clf_fixture", ["rf_classifier", "svm_classifier"])
    def test_labels_expanded_with_ancestors(
        self, clf_fixture, hierarchy, training_data, request
    ):
        clf = request.getfixturevalue(clf_fixture)
        X, y = training_data
        clf._hierarchy = hierarchy

        labels = clf._expand_labels(y)

        # GO:0005524 (ATP binding) deve expandir para incluir binding e root
        first = labels[0]
        assert "GO:0005524" in first
        assert "GO:0005488" in first
        assert "GO:0003674" in first


class TestPrediction:
    @pytest.mark.parametrize("clf_fixture", ["rf_classifier", "svm_classifier"])
    def test_predict_returns_list_of_strings(
        self, clf_fixture, hierarchy, training_data, request
    ):
        clf = request.getfixturevalue(clf_fixture)
        X, y = training_data
        clf.train(X, y, hierarchy)

        preds = clf.predict(X)
        assert isinstance(preds, list)
        assert len(preds) == len(X)
        assert all(isinstance(p, str) for p in preds)

    @pytest.mark.parametrize("clf_fixture", ["rf_classifier", "svm_classifier"])
    def test_predictions_contain_valid_go_terms(
        self, clf_fixture, hierarchy, training_data, request
    ):
        clf = request.getfixturevalue(clf_fixture)
        X, y = training_data
        clf.train(X, y, hierarchy)

        valid_ids = set(hierarchy.get_all_node_ids())
        preds = clf.predict(X)
        for pred_str in preds:
            if not pred_str:
                continue
            terms = pred_str.split(";")
            for term in terms:
                assert term in valid_ids, f"{term} nao esta na hierarquia"


class TestHierarchicalConsistency:
    @pytest.mark.parametrize("clf_fixture", ["rf_classifier", "svm_classifier"])
    def test_child_implies_parent(
        self, clf_fixture, hierarchy, training_data, request
    ):
        """Se um filho e predito, todos os seus ancestrais devem estar presentes."""
        clf = request.getfixturevalue(clf_fixture)
        X, y = training_data
        clf.train(X, y, hierarchy)

        preds = clf.predict(X)
        for pred_str in preds:
            if not pred_str:
                continue
            terms = set(pred_str.split(";"))
            for term in terms:
                ancestors = hierarchy.get_ancestors(term)
                for anc in ancestors:
                    if anc in set(hierarchy.get_all_node_ids()):
                        assert anc in terms, (
                            f"Ancestral {anc} ausente na predicao que contem {term}"
                        )

    @pytest.mark.parametrize("clf_fixture", ["rf_classifier", "svm_classifier"])
    def test_root_always_predicted(
        self, clf_fixture, hierarchy, training_data, request
    ):
        """A raiz (molecular_function) deve estar em toda predicao nao-vazia."""
        clf = request.getfixturevalue(clf_fixture)
        X, y = training_data
        clf.train(X, y, hierarchy)

        preds = clf.predict(X)
        for pred_str in preds:
            if not pred_str:
                continue
            terms = pred_str.split(";")
            assert "GO:0003674" in terms


class TestSeparability:
    @pytest.mark.parametrize("clf_fixture", ["rf_classifier", "svm_classifier"])
    def test_reasonable_accuracy_on_separable_data(
        self, clf_fixture, hierarchy, training_data, request
    ):
        """Com dados sinteticos bem separados, a acuracia deve ser razoavel."""
        clf = request.getfixturevalue(clf_fixture)
        X, y = training_data
        clf.train(X, y, hierarchy)

        preds = clf.predict(X)
        correct = 0
        for pred_str, true_term in zip(preds, y):
            if true_term in pred_str.split(";"):
                correct += 1

        accuracy = correct / len(y)
        assert accuracy > 0.8, f"Acuracia muito baixa: {accuracy:.2f}"


class TestEdgeCases:
    def test_single_class_all_same_label(self, config, hierarchy):
        """Todas as amostras com o mesmo label."""
        clf = RandomForestHierarchicalClassifier(config=config)
        X = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
        y = pd.Series(["GO:0005524", "GO:0005524", "GO:0005524"])

        clf.train(X, y, hierarchy)
        preds = clf.predict(X)

        assert len(preds) == 3
        for pred_str in preds:
            assert "GO:0005524" in pred_str.split(";")

    def test_predict_single_sample(self, config, hierarchy, training_data):
        clf = RandomForestHierarchicalClassifier(config=config)
        X, y = training_data
        clf.train(X, y, hierarchy)

        single = X.iloc[[0]]
        preds = clf.predict(single)
        assert len(preds) == 1
        assert isinstance(preds[0], str)
