from src.domain.entities.hierarchy_graph import HierarchyGraph
from src.domain.interfaces.hierarchy_builder import HierarchyBuilder


class GODagBuilder(HierarchyBuilder):
    def build(self, go_terms: list[str]) -> HierarchyGraph:
        raise NotImplementedError
