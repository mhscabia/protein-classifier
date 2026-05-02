from src.shared.config_loader import load_config
from src.infrastructure.hierarchy.go_dag_builder import GODagBuilder

TERMO = "GO:0030165"

config = load_config()
builder = GODagBuilder(config)
graph = builder.build([TERMO])

ancestrais = graph.get_ancestors(TERMO)
todos = ancestrais + [TERMO]
todos_ordenados = sorted(todos, key=lambda t: len(graph.get_ancestors(t)))

for term_id in todos_ordenados:
    node = graph.get_node(term_id)
    print(term_id, "—", node.name if node else "")
