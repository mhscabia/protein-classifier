"""Base para classificadores hierarquicos usando Local Classifier per Node (LCN).

Estrategia:
1. Treina um classificador binario para cada no do DAG
2. Na predicao, percorre top-down: so avalia filhos se o pai foi positivo
3. Retorna todos os nos positivos como termos GO preditos
"""

from abc import abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from src.domain.entities.hierarchy_graph import HierarchyGraph
from src.domain.interfaces.classifier import HierarchicalClassifier
from src.shared.logger import get_logger

logger = get_logger(__name__)


class BaseHierarchicalLCN(HierarchicalClassifier):
    """Classificador hierarquico generico usando LCN."""

    def __init__(self, seed: int):
        self._seed = seed
        self._hierarchy: HierarchyGraph | None = None
        self._node_classifiers: dict[str, BaseEstimator] = {}
        self._all_positive_nodes: set[str] = set()
        self._root_ids: list[str] = []

    @abstractmethod
    def _create_estimator(self) -> BaseEstimator:
        """Retorna o estimador scikit-learn para cada no."""
        ...

    def train(
        self, X: pd.DataFrame, y: pd.Series, hierarchy: HierarchyGraph
    ) -> None:
        self._hierarchy = hierarchy
        self._root_ids = self._find_roots()

        sample_labels = self._expand_labels(y)
        all_terms = set()
        for labels in sample_labels:
            all_terms.update(labels)

        X_array = X.values

        trained = 0
        for term_id in sorted(all_terms):
            if hierarchy.get_node(term_id) is None:
                continue

            binary_y = np.array(
                [1 if term_id in labels else 0 for labels in sample_labels]
            )

            n_positive = int(binary_y.sum())
            if n_positive == 0:
                continue
            if n_positive == len(binary_y):
                self._all_positive_nodes.add(term_id)
                continue

            clf = self._create_estimator()
            clf.fit(X_array, binary_y)
            self._node_classifiers[term_id] = clf
            trained += 1

        logger.info(
            "Treinados %d classificadores binarios (%d nos sempre positivos)",
            trained,
            len(self._all_positive_nodes),
        )

    def predict(self, X: pd.DataFrame) -> list[str]:
        X_array = X.values
        results = []

        for i in range(len(X_array)):
            sample = X_array[i : i + 1]
            predicted = self._predict_top_down(sample)
            results.append(";".join(sorted(predicted)) if predicted else "")

        return results

    def _expand_labels(self, y: pd.Series) -> list[set[str]]:
        """Expande labels adicionando todos os ancestrais de cada termo."""
        sample_labels = []
        for terms_str in y:
            raw = str(terms_str).split(";") if terms_str else []
            terms = {t.strip() for t in raw if t.strip()}

            expanded = set(terms)
            for term_id in terms:
                if self._hierarchy.get_node(term_id) is not None:
                    expanded.update(self._hierarchy.get_ancestors(term_id))
            sample_labels.append(expanded)

        return sample_labels

    def _find_roots(self) -> list[str]:
        """Encontra nos raiz (sem pais) no DAG."""
        roots = []
        for node_id in self._hierarchy.get_all_node_ids():
            node = self._hierarchy.get_node(node_id)
            if not node.parent_ids:
                roots.append(node_id)
        return sorted(roots)

    def _predict_top_down(self, sample: np.ndarray) -> set[str]:
        """Percorre o DAG top-down, incluindo nos positivos e seus filhos."""
        predicted: set[str] = set()
        queue = list(self._root_ids)

        while queue:
            term_id = queue.pop(0)
            if term_id in predicted:
                continue

            is_positive = self._classify_node(term_id, sample)
            if is_positive:
                predicted.add(term_id)
                node = self._hierarchy.get_node(term_id)
                if node:
                    queue.extend(node.children_ids)

        return predicted

    def _classify_node(self, term_id: str, sample: np.ndarray) -> bool:
        """Classifica uma amostra para um no especifico."""
        if term_id in self._all_positive_nodes:
            return True
        if term_id in self._node_classifiers:
            return self._node_classifiers[term_id].predict(sample)[0] == 1
        return False
