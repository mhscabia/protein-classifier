from abc import ABC, abstractmethod

from src.domain.entities.hierarchy_graph import HierarchyGraph


class HierarchyBuilder(ABC):
    @abstractmethod
    def build(self, go_terms: list[str]) -> HierarchyGraph: ...
