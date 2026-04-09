from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.domain.entities.hierarchy_graph import HierarchyGraph
from src.domain.interfaces.classifier import HierarchicalClassifier
from src.shared.config_loader import load_config
from src.shared.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainResult:
    """Resultado do treinamento de um classificador."""

    classifier: HierarchicalClassifier
    X_test: pd.DataFrame
    y_test: pd.Series


class TrainClassifiersUseCase:
    """Orquestra o treinamento de classificadores hierarquicos."""

    def __init__(
        self,
        classifier: HierarchicalClassifier,
        config: dict | None = None,
    ):
        self._classifier = classifier
        self._config = config or load_config()

    def execute(
        self,
        data: pd.DataFrame,
        hierarchy: HierarchyGraph,
    ) -> TrainResult:
        seed = self._config["model"]["random_seed"]
        test_size = self._config["model"]["test_size"]

        feature_cols = [
            c for c in data.columns
            if c.startswith(("seq_length", "molecular_weight", "aa_"))
        ]

        X = data[feature_cols]
        y = data["go_terms"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed,
        )

        logger.info(
            "Split: %d treino, %d teste (test_size=%.2f)",
            len(X_train),
            len(X_test),
            test_size,
        )

        self._classifier.train(X_train, y_train, hierarchy)

        return TrainResult(
            classifier=self._classifier,
            X_test=X_test,
            y_test=y_test,
        )
