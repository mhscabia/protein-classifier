"""Metricas de avaliacao hierarquica (hP, hR, hF) e flat para comparacao.

Formulas baseadas em Vens et al. (2008) e Silla & Freitas (2010):

    hP = sum_i |T_i ∩ P_i| / sum_i |P_i|
    hR = sum_i |T_i ∩ P_i| / sum_i |T_i|
    hF = 2 * hP * hR / (hP + hR)

Onde T_i e P_i sao os conjuntos de termos verdadeiros e preditos
para a amostra i, estendidos com todos os ancestrais no DAG.
"""

from src.domain.entities.hierarchy_graph import HierarchyGraph
from src.domain.interfaces.evaluator import HierarchicalEvaluator
from src.shared.logger import get_logger

logger = get_logger(__name__)


class HierarchicalMetricsEvaluator(HierarchicalEvaluator):
    """Calcula metricas hierarquicas e flat para classificacao GO."""

    def __init__(self, hierarchy: HierarchyGraph):
        self._hierarchy = hierarchy

    def evaluate(self, y_true: list[str], y_pred: list[str]) -> dict:
        true_sets = [self._parse_terms(t) for t in y_true]
        pred_sets = [self._parse_terms(p) for p in y_pred]

        # --- Metricas hierarquicas: expande ambos com ancestrais ---
        true_expanded = [self._expand_with_ancestors(s) for s in true_sets]
        pred_expanded = [self._expand_with_ancestors(s) for s in pred_sets]

        hP = self._precision(true_expanded, pred_expanded)
        hR = self._recall(true_expanded, pred_expanded)
        hF = self._f_measure(hP, hR)

        # --- Metricas flat: sem expansao de ancestrais ---
        flat_P = self._precision(true_sets, pred_sets)
        flat_R = self._recall(true_sets, pred_sets)
        flat_F = self._f_measure(flat_P, flat_R)

        logger.info(
            "Hierarquico — hP: %.4f  hR: %.4f  hF: %.4f", hP, hR, hF
        )
        logger.info(
            "Flat       — fP: %.4f  fR: %.4f  fF: %.4f",
            flat_P,
            flat_R,
            flat_F,
        )

        return {
            "hP": round(hP, 6),
            "hR": round(hR, 6),
            "hF": round(hF, 6),
            "flat_P": round(flat_P, 6),
            "flat_R": round(flat_R, 6),
            "flat_F": round(flat_F, 6),
        }

    @staticmethod
    def _parse_terms(terms_str: str) -> set[str]:
        """Converte string semicolon-separated em conjunto de termos."""
        if not terms_str:
            return set()
        return {t.strip() for t in str(terms_str).split(";") if t.strip()}

    def _expand_with_ancestors(self, terms: set[str]) -> set[str]:
        """Expande um conjunto de termos adicionando todos os ancestrais."""
        expanded = set(terms)
        for term_id in terms:
            if self._hierarchy.get_node(term_id) is not None:
                expanded.update(self._hierarchy.get_ancestors(term_id))
        return expanded

    @staticmethod
    def _precision(
        true_sets: list[set[str]], pred_sets: list[set[str]]
    ) -> float:
        """P = sum |T_i ∩ P_i| / sum |P_i|"""
        numerator = 0
        denominator = 0
        for t, p in zip(true_sets, pred_sets):
            numerator += len(t & p)
            denominator += len(p)
        if denominator == 0:
            return 0.0
        return numerator / denominator

    @staticmethod
    def _recall(
        true_sets: list[set[str]], pred_sets: list[set[str]]
    ) -> float:
        """R = sum |T_i ∩ P_i| / sum |T_i|"""
        numerator = 0
        denominator = 0
        for t, p in zip(true_sets, pred_sets):
            numerator += len(t & p)
            denominator += len(t)
        if denominator == 0:
            return 0.0
        return numerator / denominator

    @staticmethod
    def _f_measure(precision: float, recall: float) -> float:
        """F = 2PR / (P + R)"""
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
