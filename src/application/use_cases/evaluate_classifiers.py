from dataclasses import dataclass

import pandas as pd

from src.domain.interfaces.classifier import HierarchicalClassifier
from src.domain.interfaces.evaluator import HierarchicalEvaluator
from src.shared.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Resultado da avaliacao de um classificador."""

    classifier_name: str
    metrics: dict
    y_pred: list[str]


class EvaluateClassifiersUseCase:
    """Orquestra a avaliacao de classificadores hierarquicos."""

    def __init__(
        self,
        evaluator: HierarchicalEvaluator,
    ):
        self._evaluator = evaluator

    def execute(
        self,
        classifier: HierarchicalClassifier,
        classifier_name: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> EvaluationResult:
        logger.info("Avaliando classificador: %s", classifier_name)

        y_pred = classifier.predict(X_test)
        y_true = y_test.tolist()

        metrics = self._evaluator.evaluate(y_true, y_pred)

        logger.info(
            "%s — hP=%.4f  hR=%.4f  hF=%.4f",
            classifier_name,
            metrics["hP"],
            metrics["hR"],
            metrics["hF"],
        )

        return EvaluationResult(
            classifier_name=classifier_name,
            metrics=metrics,
            y_pred=y_pred,
        )
