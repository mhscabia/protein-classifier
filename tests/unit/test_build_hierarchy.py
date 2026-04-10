import pytest

from src.application.use_cases.build_hierarchy import BuildHierarchyUseCase
from src.domain.entities.hierarchy_graph import FunctionNode, HierarchyGraph
from src.domain.interfaces.hierarchy_builder import HierarchyBuilder


class FakeHierarchyBuilder(HierarchyBuilder):
    def build(self, go_terms: list[str]) -> HierarchyGraph:
        graph = HierarchyGraph()
        for term_id in go_terms:
            graph.add_node(FunctionNode(term_id=term_id, name=f"name_{term_id}"))
        return graph


class TestBuildHierarchyUseCase:
    def test_returns_hierarchy_graph(self):
        builder = FakeHierarchyBuilder()
        use_case = BuildHierarchyUseCase(builder)

        result = use_case.execute(["GO:0001", "GO:0002"])
        assert isinstance(result, HierarchyGraph)
        assert len(result) == 2

    def test_empty_terms(self):
        builder = FakeHierarchyBuilder()
        use_case = BuildHierarchyUseCase(builder)

        result = use_case.execute([])
        assert len(result) == 0

    def test_delegates_to_builder(self):
        builder = FakeHierarchyBuilder()
        use_case = BuildHierarchyUseCase(builder)

        result = use_case.execute(["GO:0005524"])
        assert result.get_node("GO:0005524") is not None
        assert result.get_node("GO:0005524").name == "name_GO:0005524"
