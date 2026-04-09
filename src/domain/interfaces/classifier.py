from abc import ABC, abstractmethod

import pandas as pd

from src.domain.entities.hierarchy_graph import HierarchyGraph


class HierarchicalClassifier(ABC):
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, hierarchy: HierarchyGraph) -> None: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> list[str]: ...
