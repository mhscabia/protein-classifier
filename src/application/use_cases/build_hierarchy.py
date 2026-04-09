from src.domain.entities.hierarchy_graph import HierarchyGraph
from src.domain.interfaces.hierarchy_builder import HierarchyBuilder
from src.shared.logger import get_logger

logger = get_logger(__name__)


class BuildHierarchyUseCase:
    """Orquestra a construcao do DAG de termos GO."""

    def __init__(self, builder: HierarchyBuilder):
        self._builder = builder

    def execute(self, go_terms: list[str]) -> HierarchyGraph:
        logger.info("Construindo hierarquia para %d termos GO...", len(go_terms))
        hierarchy = self._builder.build(go_terms)
        logger.info("Hierarquia construida: %d nos", len(hierarchy))
        return hierarchy
