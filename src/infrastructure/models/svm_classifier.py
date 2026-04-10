import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC

from src.domain.entities.hierarchy_graph import HierarchyGraph
from src.domain.interfaces.classifier import HierarchicalClassifier
from src.shared.config_loader import load_config
from src.shared.logger import get_logger

logger = get_logger(__name__)

FEATURE_PREFIX = ("seq_length", "molecular_weight", "aa_")


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c.startswith(FEATURE_PREFIX)
    ]


def _parse_go_terms(series: pd.Series) -> list[list[str]]:
    """Converte coluna go_terms (str separada por ;) em listas."""
    return [
        [t.strip() for t in str(val).split(";") if t.strip()]
        for val in series
    ]


class SVMHierarchicalClassifier(HierarchicalClassifier):
    """Classificador hierarquico baseado em SVM (OneVsRest).

    Estrategia: flat multi-label com propagacao hierarquica.
    Usa OneVsRestClassifier com SVC para multi-label,
    garantindo consistencia hierarquica via propagacao de ancestrais.
    """

    def __init__(self, config: dict | None = None):
        self._config = config or load_config()
        seed = self._config["model"]["random_seed"]
        self._clf = OneVsRestClassifier(
            SVC(kernel="rbf", random_state=seed, probability=False),
        )
        self._mlb = MultiLabelBinarizer()
        self._hierarchy: HierarchyGraph | None = None
        self._feature_cols: list[str] = []

    def train(self, X: pd.DataFrame, y: pd.Series, hierarchy: HierarchyGraph) -> None:
        self._hierarchy = hierarchy
        self._feature_cols = _get_feature_columns(X)

        if not self._feature_cols:
            raise ValueError("Nenhuma coluna de feature encontrada no DataFrame")

        X_features = X[self._feature_cols].values
        y_labels = _parse_go_terms(y)

        # Propagar ancestrais nos labels de treino
        y_augmented = self._augment_with_ancestors(y_labels)

        y_binary = self._mlb.fit_transform(y_augmented)
        logger.info(
            "Treinando SVM: %d amostras, %d features, %d classes",
            X_features.shape[0],
            X_features.shape[1],
            y_binary.shape[1],
        )

        self._clf.fit(X_features, y_binary)
        logger.info("Treinamento SVM concluido")

    def predict(self, X: pd.DataFrame) -> list[str]:
        X_features = X[self._feature_cols].values
        y_binary = self._clf.predict(X_features)
        raw_labels = self._mlb.inverse_transform(y_binary)

        predictions = []
        for labels in raw_labels:
            if labels:
                augmented = set(labels)
                for term in labels:
                    if self._hierarchy:
                        augmented.update(self._hierarchy.get_ancestors(term))
                predictions.append(";".join(sorted(augmented)))
            else:
                predictions.append("")

        return predictions

    def _augment_with_ancestors(self, labels_list: list[list[str]]) -> list[list[str]]:
        """Adiciona ancestrais a cada conjunto de labels."""
        augmented = []
        for labels in labels_list:
            expanded = set(labels)
            for term in labels:
                if self._hierarchy:
                    expanded.update(self._hierarchy.get_ancestors(term))
            augmented.append(sorted(expanded))
        return augmented
