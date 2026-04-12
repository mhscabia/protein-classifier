from collections import deque
from dataclasses import dataclass, field


@dataclass
class FunctionNode:
    term_id: str
    name: str
    parent_ids: list[str] = field(default_factory=list)
    children_ids: list[str] = field(default_factory=list)


class HierarchyGraph:
    """DAG de termos GO — estrutura central do sistema."""

    def __init__(self):
        self._nodes: dict[str, FunctionNode] = {}

    def add_node(self, node: FunctionNode) -> None:
        self._nodes[node.term_id] = node

    def get_node(self, term_id: str) -> FunctionNode | None:
        return self._nodes.get(term_id)

    def get_ancestors(self, term_id: str) -> list[str]:
        """Retorna todos os ancestrais — essencial para métricas hierárquicas."""
        visited: set[str] = set()
        queue: deque[str] = deque(
            self._nodes[term_id].parent_ids if term_id in self._nodes else []
        )
        while queue:
            parent_id = queue.popleft()
            if parent_id not in visited:
                visited.add(parent_id)
                if parent_id in self._nodes:
                    queue.extend(self._nodes[parent_id].parent_ids)
        return list(visited)

    def get_all_node_ids(self) -> list[str]:
        """Retorna todos os IDs de nos no grafo."""
        return list(self._nodes.keys())

    def __len__(self) -> int:
        return len(self._nodes)
