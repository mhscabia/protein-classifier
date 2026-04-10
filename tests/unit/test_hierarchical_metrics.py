import pytest

from src.domain.entities.hierarchy_graph import FunctionNode, HierarchyGraph
from src.infrastructure.evaluation.hierarchical_metrics import (
    HierarchicalMetricsEvaluator,
    _expand_with_ancestors,
    _parse_terms,
)


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestParseTerms:
    def test_single_term(self):
        assert _parse_terms("GO:0001") == {"GO:0001"}

    def test_multiple_terms(self):
        assert _parse_terms("GO:0001;GO:0002") == {"GO:0001", "GO:0002"}

    def test_empty_string(self):
        assert _parse_terms("") == set()

    def test_trims_whitespace(self):
        assert _parse_terms(" GO:0001 ; GO:0002 ") == {"GO:0001", "GO:0002"}


class TestExpandWithAncestors:
    def test_leaf_expands_to_root(self, hierarchy):
        expanded = _expand_with_ancestors({"GO:0005524"}, hierarchy)
        assert expanded == {"GO:0005524", "GO:0005488", "GO:0003674"}

    def test_root_stays_same(self, hierarchy):
        expanded = _expand_with_ancestors({"GO:0003674"}, hierarchy)
        assert expanded == {"GO:0003674"}

    def test_mid_node(self, hierarchy):
        expanded = _expand_with_ancestors({"GO:0005488"}, hierarchy)
        assert expanded == {"GO:0005488", "GO:0003674"}


# ---------------------------------------------------------------------------
# HierarchicalMetricsEvaluator
# ---------------------------------------------------------------------------


class TestHierarchicalMetrics:
    def test_perfect_prediction(self, hierarchy):
        evaluator = HierarchicalMetricsEvaluator(hierarchy)
        y_true = ["GO:0005524", "GO:0003824"]
        y_pred = ["GO:0005524", "GO:0003824"]

        metrics = evaluator.evaluate(y_true, y_pred)
        assert metrics["hP"] == pytest.approx(1.0)
        assert metrics["hR"] == pytest.approx(1.0)
        assert metrics["hF"] == pytest.approx(1.0)

    def test_completely_wrong(self, hierarchy):
        evaluator = HierarchicalMetricsEvaluator(hierarchy)
        # True: ATP binding (leaf), Pred: catalytic activity (different branch)
        # True expanded: {GO:0005524, GO:0005488, GO:0003674}
        # Pred expanded: {GO:0003824, GO:0003674}
        # Intersection: {GO:0003674} -> 1
        y_true = ["GO:0005524"]
        y_pred = ["GO:0003824"]

        metrics = evaluator.evaluate(y_true, y_pred)
        # hP = 1/2 = 0.5, hR = 1/3 ≈ 0.3333
        assert metrics["hP"] == pytest.approx(0.5)
        assert metrics["hR"] == pytest.approx(1 / 3, abs=0.001)
        assert metrics["hF"] > 0

    def test_partial_match(self, hierarchy):
        evaluator = HierarchicalMetricsEvaluator(hierarchy)
        # True: ATP binding, Pred: binding (parent only)
        # True expanded: {GO:0005524, GO:0005488, GO:0003674}
        # Pred expanded: {GO:0005488, GO:0003674}
        # Intersection: {GO:0005488, GO:0003674} -> 2
        y_true = ["GO:0005524"]
        y_pred = ["GO:0005488"]

        metrics = evaluator.evaluate(y_true, y_pred)
        assert metrics["hP"] == pytest.approx(1.0)  # 2/2
        assert metrics["hR"] == pytest.approx(2 / 3, abs=0.001)  # 2/3
        assert metrics["hF"] == pytest.approx(0.8, abs=0.001)

    def test_empty_predictions(self, hierarchy):
        evaluator = HierarchicalMetricsEvaluator(hierarchy)
        y_true = ["GO:0005524"]
        y_pred = [""]

        metrics = evaluator.evaluate(y_true, y_pred)
        assert metrics["hP"] == pytest.approx(0.0)
        assert metrics["hR"] == pytest.approx(0.0)
        assert metrics["hF"] == pytest.approx(0.0)

    def test_mismatched_lengths(self, hierarchy):
        evaluator = HierarchicalMetricsEvaluator(hierarchy)
        with pytest.raises(ValueError, match="mesmo tamanho"):
            evaluator.evaluate(["GO:0001"], ["GO:0001", "GO:0002"])

    def test_multi_label_true_and_pred(self, hierarchy):
        evaluator = HierarchicalMetricsEvaluator(hierarchy)
        y_true = ["GO:0005524;GO:0003824"]
        y_pred = ["GO:0005524;GO:0003824"]

        metrics = evaluator.evaluate(y_true, y_pred)
        assert metrics["hP"] == pytest.approx(1.0)
        assert metrics["hR"] == pytest.approx(1.0)
        assert metrics["hF"] == pytest.approx(1.0)

    def test_multiple_samples(self, hierarchy):
        evaluator = HierarchicalMetricsEvaluator(hierarchy)
        y_true = ["GO:0005524", "GO:0003824", "GO:0005524"]
        y_pred = ["GO:0005524", "GO:0003824", "GO:0003824"]

        metrics = evaluator.evaluate(y_true, y_pred)
        assert 0 < metrics["hP"] <= 1.0
        assert 0 < metrics["hR"] <= 1.0
        assert 0 < metrics["hF"] <= 1.0
