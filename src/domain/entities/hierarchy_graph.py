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
        ancestors = []
        queue = list(self._nodes[term_id].parent_ids) if term_id in self._nodes else []
        while queue:
            parent_id = queue.pop(0)
            if parent_id not in ancestors:
                ancestors.append(parent_id)
                if parent_id in self._nodes:
                    queue.extend(self._nodes[parent_id].parent_ids)
        return ancestors

    def __len__(self) -> int:
        return len(self._nodes)
