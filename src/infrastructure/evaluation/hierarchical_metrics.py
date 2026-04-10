import numpy as np

from src.domain.entities.hierarchy_graph import HierarchyGraph
from src.domain.interfaces.evaluator import HierarchicalEvaluator
from src.shared.logger import get_logger

logger = get_logger(__name__)


def _expand_with_ancestors(
    terms: set[str], hierarchy: HierarchyGraph
) -> set[str]:
    """Expande um conjunto de termos GO com todos os seus ancestrais."""
    expanded = set(terms)
    for term in terms:
        expanded.update(hierarchy.get_ancestors(term))
    return expanded


def _parse_terms(term_str: str) -> set[str]:
    """Converte string de termos GO separados por ; em um set."""
    return {t.strip() for t in term_str.split(";") if t.strip()}


class HierarchicalMetricsEvaluator(HierarchicalEvaluator):
    """Calcula metricas hierarquicas: hP, hR, hF.

    Baseado em Verspoor et al. (2006) e Kiritchenko et al. (2005):
    - hP = sum(|pred_i & true_i|) / sum(|pred_i|)
    - hR = sum(|pred_i & true_i|) / sum(|true_i|)
    - hF = 2 * hP * hR / (hP + hR)

    Onde pred_i e true_i sao os conjuntos de termos expandidos
    com ancestrais para a amostra i.
    """

    def __init__(self, hierarchy: HierarchyGraph):
        self._hierarchy = hierarchy

    def evaluate(self, y_true: list[str], y_pred: list[str]) -> dict:
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"y_true e y_pred devem ter o mesmo tamanho: "
                f"{len(y_true)} != {len(y_pred)}"
            )

        sum_intersection = 0
        sum_pred = 0
        sum_true = 0

        for true_str, pred_str in zip(y_true, y_pred):
            true_terms = _parse_terms(true_str)
            pred_terms = _parse_terms(pred_str)

            true_expanded = _expand_with_ancestors(true_terms, self._hierarchy)
            pred_expanded = _expand_with_ancestors(pred_terms, self._hierarchy)

            sum_intersection += len(pred_expanded & true_expanded)
            sum_pred += len(pred_expanded)
            sum_true += len(true_expanded)

        hp = sum_intersection / sum_pred if sum_pred > 0 else 0.0
        hr = sum_intersection / sum_true if sum_true > 0 else 0.0
        hf = (2 * hp * hr / (hp + hr)) if (hp + hr) > 0 else 0.0

        logger.info("hP=%.4f  hR=%.4f  hF=%.4f", hp, hr, hf)

        return {"hP": round(hp, 4), "hR": round(hr, 4), "hF": round(hf, 4)}
