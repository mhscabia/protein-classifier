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
        self._full_graph: "HierarchyGraph | None" = None

    def add_node(self, node: FunctionNode) -> None:
        self._nodes[node.term_id] = node

    def get_node(self, term_id: str) -> FunctionNode | None:
        return self._nodes.get(term_id)

    def set_full_graph(self, full: "HierarchyGraph") -> None:
        """Anexa o DAG completo (pre-filtro) para resolver ancestrais de termos
        que foram filtrados. Usado pelo LCN para expandir labels mesmo quando
        o termo-folha foi removido pelo filtro de suporte minimo."""
        self._full_graph = full

    def get_ancestors(self, term_id: str) -> list[str]:
        """Retorna todos os ancestrais — essencial para métricas hierárquicas.

        Se o termo nao esta neste grafo mas existe no grafo completo anexado,
        retorna ancestrais (do grafo completo) que estao neste grafo filtrado.
        """
        if term_id in self._nodes:
            visited: set[str] = set()
            queue: deque[str] = deque(self._nodes[term_id].parent_ids)
            while queue:
                parent_id = queue.popleft()
                if parent_id not in visited:
                    visited.add(parent_id)
                    if parent_id in self._nodes:
                        queue.extend(self._nodes[parent_id].parent_ids)
            return list(visited)

        if self._full_graph is not None and self._full_graph is not self:
            full_ancestors = self._full_graph.get_ancestors(term_id)
            return [a for a in full_ancestors if a in self._nodes]

        return []

    def get_all_node_ids(self) -> list[str]:
        """Retorna todos os IDs de nos no grafo."""
        return list(self._nodes.keys())

    def __len__(self) -> int:
        return len(self._nodes)
