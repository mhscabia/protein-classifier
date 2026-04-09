import pytest

from src.domain.entities.hierarchy_graph import FunctionNode, HierarchyGraph
from src.infrastructure.evaluation.hierarchical_metrics import (
    HierarchicalMetricsEvaluator,
)


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
def evaluator(hierarchy):
    return HierarchicalMetricsEvaluator(hierarchy=hierarchy)


# ---------------------------------------------------------------------------
# _parse_terms
# ---------------------------------------------------------------------------


class TestParseTerms:
    def test_parses_semicolon_separated(self):
        result = HierarchicalMetricsEvaluator._parse_terms(
            "GO:0005524;GO:0003824"
        )
        assert result == {"GO:0005524", "GO:0003824"}

    def test_handles_whitespace(self):
        result = HierarchicalMetricsEvaluator._parse_terms(
            " GO:0005524 ; GO:0003824 "
        )
        assert result == {"GO:0005524", "GO:0003824"}

    def test_empty_string(self):
        assert HierarchicalMetricsEvaluator._parse_terms("") == set()

    def test_single_term(self):
        result = HierarchicalMetricsEvaluator._parse_terms("GO:0005524")
        assert result == {"GO:0005524"}


# ---------------------------------------------------------------------------
# _expand_with_ancestors
# ---------------------------------------------------------------------------


class TestExpandWithAncestors:
    def test_leaf_expands_to_full_path(self, evaluator):
        result = evaluator._expand_with_ancestors({"GO:0005524"})
        assert result == {"GO:0005524", "GO:0005488", "GO:0003674"}

    def test_intermediate_expands_to_root(self, evaluator):
        result = evaluator._expand_with_ancestors({"GO:0005488"})
        assert result == {"GO:0005488", "GO:0003674"}

    def test_root_stays_as_is(self, evaluator):
        result = evaluator._expand_with_ancestors({"GO:0003674"})
        assert result == {"GO:0003674"}

    def test_unknown_term_kept_without_expansion(self, evaluator):
        result = evaluator._expand_with_ancestors({"GO:UNKNOWN"})
        assert result == {"GO:UNKNOWN"}

    def test_multiple_terms_merged(self, evaluator):
        result = evaluator._expand_with_ancestors(
            {"GO:0005524", "GO:0003824"}
        )
        expected = {
            "GO:0005524",
            "GO:0005488",
            "GO:0003674",
            "GO:0003824",
        }
        assert result == expected


# ---------------------------------------------------------------------------
# Predicao perfeita: hP = hR = hF = 1.0
# ---------------------------------------------------------------------------


class TestPerfectPrediction:
    def test_perfect_match(self, evaluator):
        y_true = ["GO:0005524", "GO:0003824"]
        y_pred = ["GO:0005524", "GO:0003824"]

        result = evaluator.evaluate(y_true, y_pred)

        assert result["hP"] == pytest.approx(1.0)
        assert result["hR"] == pytest.approx(1.0)
        assert result["hF"] == pytest.approx(1.0)

    def test_perfect_match_with_ancestors_in_pred(self, evaluator):
        """predict() retorna conjuntos expandidos; y_true sao termos folha."""
        y_true = ["GO:0005524"]
        y_pred = ["GO:0003674;GO:0005488;GO:0005524"]

        result = evaluator.evaluate(y_true, y_pred)

        # y_true expandido = {root, binding, ATP_binding}
        # y_pred ja contem  = {root, binding, ATP_binding}
        assert result["hP"] == pytest.approx(1.0)
        assert result["hR"] == pytest.approx(1.0)
        assert result["hF"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Predicao parcialmente correta
# ---------------------------------------------------------------------------


class TestPartialPrediction:
    def test_predict_ancestor_only(self, evaluator):
        """Prediz apenas 'binding' quando o verdadeiro e 'ATP binding'.
        Hierarquico deve dar credito parcial."""
        y_true = ["GO:0005524"]
        y_pred = ["GO:0003674;GO:0005488"]

        result = evaluator.evaluate(y_true, y_pred)

        # true_expanded  = {root, binding, ATP_binding} (3 termos)
        # pred_expanded  = {root, binding}               (2 termos)
        # intersecao     = {root, binding}               (2 termos)
        # hP = 2/2 = 1.0
        # hR = 2/3 ≈ 0.6667
        assert result["hP"] == pytest.approx(1.0)
        assert result["hR"] == pytest.approx(2 / 3, abs=1e-4)
        assert result["hF"] == pytest.approx(0.8, abs=1e-4)

    def test_predict_wrong_branch(self, evaluator):
        """Prediz 'catalytic' quando o verdadeiro e 'ATP binding'."""
        y_true = ["GO:0005524"]
        y_pred = ["GO:0003674;GO:0003824"]

        result = evaluator.evaluate(y_true, y_pred)

        # true_expanded = {root, binding, ATP_binding}  (3)
        # pred_expanded = {root, catalytic}             (2)
        # intersecao    = {root}                        (1)
        # hP = 1/2 = 0.5
        # hR = 1/3 ≈ 0.3333
        assert result["hP"] == pytest.approx(0.5)
        assert result["hR"] == pytest.approx(1 / 3, abs=1e-4)

    def test_over_prediction(self, evaluator):
        """Prediz termos extras alem do correto — reduz precision."""
        y_true = ["GO:0003824"]
        y_pred = ["GO:0003674;GO:0003824;GO:0005488;GO:0005524"]

        result = evaluator.evaluate(y_true, y_pred)

        # true_expanded = {root, catalytic}                      (2)
        # pred_expanded = {root, catalytic, binding, ATP_binding} (4)
        # intersecao    = {root, catalytic}                      (2)
        # hP = 2/4 = 0.5
        # hR = 2/2 = 1.0
        assert result["hP"] == pytest.approx(0.5)
        assert result["hR"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Multiplas amostras — formula agregada (micro-average)
# ---------------------------------------------------------------------------


class TestMultipleSamples:
    def test_aggregation_across_samples(self, evaluator):
        """A formula soma numeradores e denominadores (micro-average)."""
        y_true = ["GO:0005524", "GO:0003824"]
        y_pred = [
            "GO:0003674;GO:0005488;GO:0005524",
            "GO:0003674;GO:0003824",
        ]

        result = evaluator.evaluate(y_true, y_pred)

        # Amostra 1: T={root,binding,ATP} P={root,binding,ATP} |T∩P|=3 |P|=3 |T|=3
        # Amostra 2: T={root,catalytic}   P={root,catalytic}   |T∩P|=2 |P|=2 |T|=2
        # hP = (3+2)/(3+2) = 1.0
        # hR = (3+2)/(3+2) = 1.0
        assert result["hP"] == pytest.approx(1.0)
        assert result["hR"] == pytest.approx(1.0)
        assert result["hF"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Comparacao flat vs hierarquico
# ---------------------------------------------------------------------------


class TestFlatComparison:
    def test_flat_metrics_present(self, evaluator):
        y_true = ["GO:0005524"]
        y_pred = ["GO:0005524"]
        result = evaluator.evaluate(y_true, y_pred)

        assert "flat_P" in result
        assert "flat_R" in result
        assert "flat_F" in result

    def test_hierarchical_gives_more_credit_than_flat(self, evaluator):
        """Prediz ancestral correto: hierarquico da credito, flat nao."""
        y_true = ["GO:0005524"]
        y_pred = ["GO:0003674;GO:0005488"]

        result = evaluator.evaluate(y_true, y_pred)

        # Hierarquico: prediz 2 dos 3 termos expandidos -> hR = 2/3
        # Flat: y_true_raw={ATP_binding}, y_pred_raw={root,binding}
        #       intersecao = {} -> flat_R = 0
        assert result["hR"] > result["flat_R"]
        assert result["hF"] > result["flat_F"]

    def test_flat_exact_match(self, evaluator):
        """Quando predicao bate exatamente no raw, flat_P = flat_R = 1."""
        y_true = ["GO:0005524;GO:0003824"]
        y_pred = ["GO:0005524;GO:0003824"]

        result = evaluator.evaluate(y_true, y_pred)

        assert result["flat_P"] == pytest.approx(1.0)
        assert result["flat_R"] == pytest.approx(1.0)
        assert result["flat_F"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Casos extremos
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_prediction(self, evaluator):
        y_true = ["GO:0005524"]
        y_pred = [""]

        result = evaluator.evaluate(y_true, y_pred)

        assert result["hP"] == pytest.approx(0.0)
        assert result["hR"] == pytest.approx(0.0)
        assert result["hF"] == pytest.approx(0.0)

    def test_empty_true_labels(self, evaluator):
        y_true = [""]
        y_pred = ["GO:0005524"]

        result = evaluator.evaluate(y_true, y_pred)

        assert result["hR"] == pytest.approx(0.0)

    def test_both_empty(self, evaluator):
        y_true = [""]
        y_pred = [""]

        result = evaluator.evaluate(y_true, y_pred)

        assert result["hP"] == pytest.approx(0.0)
        assert result["hR"] == pytest.approx(0.0)
        assert result["hF"] == pytest.approx(0.0)

    def test_single_root_prediction(self, evaluator):
        """Prediz apenas a raiz."""
        y_true = ["GO:0005524"]
        y_pred = ["GO:0003674"]

        result = evaluator.evaluate(y_true, y_pred)

        # true_expanded = {root, binding, ATP_binding} (3)
        # pred_expanded = {root}                       (1)
        # intersecao    = {root}                       (1)
        # hP = 1/1 = 1.0
        # hR = 1/3 ≈ 0.333
        assert result["hP"] == pytest.approx(1.0)
        assert result["hR"] == pytest.approx(1 / 3, abs=1e-4)
