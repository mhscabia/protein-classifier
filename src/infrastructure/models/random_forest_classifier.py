import pandas as pd

from src.domain.entities.hierarchy_graph import HierarchyGraph
from src.domain.interfaces.classifier import HierarchicalClassifier


class RandomForestHierarchicalClassifier(HierarchicalClassifier):
    def train(self, X: pd.DataFrame, y: pd.Series, hierarchy: HierarchyGraph) -> None:
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> list[str]:
        raise NotImplementedError
