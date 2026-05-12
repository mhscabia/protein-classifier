import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.domain.entities.hierarchy_graph import HierarchyGraph
from src.domain.interfaces.classifier import HierarchicalClassifier
from src.shared.config_loader import load_config
from src.shared.logger import get_logger

logger = get_logger(__name__)

FEATURE_PREFIX = ("seq_length", "molecular_weight", "aa_", "esm_")


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(FEATURE_PREFIX)]


def _parse_go_terms(series: pd.Series) -> list[list[str]]:
    return [
        [t.strip() for t in str(val).split(";") if t.strip()]
        for val in series
    ]


class LCNClassifier(HierarchicalClassifier):
    """Classificador hierarquico Local Classifier per Node (LCN).

    Treina um RandomForest binario por no do DAG. Predicao top-down:
    comeca na raiz, desce pela hierarquia enquanto o classificador do no
    retorna positivo.
    """

    def __init__(self, config: dict | None = None):
        self._config = config or load_config()
        self._seed = self._config["model"]["random_seed"]
        self._node_classifiers: dict[str, RandomForestClassifier] = {}
        self._hierarchy: HierarchyGraph | None = None
        self._feature_cols: list[str] = []

    def train(self, X: pd.DataFrame, y: pd.Series, hierarchy: HierarchyGraph) -> None:
        self._hierarchy = hierarchy
        self._feature_cols = _get_feature_columns(X)

        if not self._feature_cols:
            raise ValueError("Nenhuma coluna de feature encontrada no DataFrame")

        X_features = X[self._feature_cols].values
        y_labels = _parse_go_terms(y)
        y_expanded = self._augment_with_ancestors(y_labels, hierarchy)

        positive_sets = [set(labels) for labels in y_expanded]

        trained = 0
        skipped = 0
        for term_id in hierarchy.get_all_node_ids():
            node = hierarchy.get_node(term_id)
            y_binary = [1 if term_id in pos else 0 for pos in positive_sets]
            n_positive = sum(y_binary)

            if n_positive < 2:
                skipped += 1
                continue

            clf = RandomForestClassifier(
                n_estimators=100,
                random_state=self._seed,
                class_weight="balanced",
                n_jobs=-1,
            )
            clf.fit(X_features, y_binary)
            self._node_classifiers[term_id] = clf
            trained += 1

        logger.info(
            "LCN treino concluido: %d classificadores treinados, %d nos ignorados (< 2 positivos)",
            trained,
            skipped,
        )

    def predict(self, X: pd.DataFrame) -> list[str]:
        X_features = X[self._feature_cols].values
        roots = [
            self._hierarchy.get_node(tid)
            for tid in self._hierarchy.get_all_node_ids()
            if not self._hierarchy.get_node(tid).parent_ids
        ]

        predictions = []
        for i in range(len(X_features)):
            row = X_features[i].reshape(1, -1)
            predicted = set()
            queue = list(roots)

            while queue:
                node = queue.pop(0)
                clf = self._node_classifiers.get(node.term_id)
                if clf is None:
                    continue
                if clf.predict(row)[0] == 1:
                    predicted.add(node.term_id)
                    for child_id in node.children_ids:
                        child = self._hierarchy.get_node(child_id)
                        if child:
                            queue.append(child)

            predictions.append(";".join(sorted(predicted)) if predicted else "")

        return predictions

    def _augment_with_ancestors(
        self, labels_list: list[list[str]], hierarchy: HierarchyGraph
    ) -> list[list[str]]:
        augmented = []
        for labels in labels_list:
            expanded = set(labels)
            for term in labels:
                expanded.update(hierarchy.get_ancestors(term))
            augmented.append(sorted(expanded))
        return augmented
