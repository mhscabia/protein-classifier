from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from src.domain.entities.hierarchy_graph import HierarchyGraph
from src.shared.logger import get_logger

logger = get_logger(__name__)


def plot_dag_predictions(
    hierarchy: HierarchyGraph,
    predicted_terms: set[str],
    output_path: Path,
) -> None:
    """Gera subgrafo do DAG com nos preditos destacados em cor diferente."""
    G = nx.DiGraph()

    # Coletar nos relevantes: preditos + ancestrais
    relevant_nodes: set[str] = set(predicted_terms)
    for term in predicted_terms:
        relevant_nodes.update(hierarchy.get_ancestors(term))

    # Construir grafo com nos relevantes
    for node_id in relevant_nodes:
        node = hierarchy.get_node(node_id)
        if node is None:
            continue
        label = f"{node.term_id}\n{node.name[:25]}" if node.name else node.term_id
        G.add_node(node_id, label=label)
        for parent_id in node.parent_ids:
            if parent_id in relevant_nodes:
                G.add_edge(parent_id, node_id)

    if len(G) == 0:
        logger.warning("Nenhum no relevante para plotar no DAG")
        return

    # Cores: preditos em verde, ancestrais em azul claro
    node_colors = [
        "#4CAF50" if n in predicted_terms else "#90CAF9"
        for n in G.nodes()
    ]

    fig, ax = plt.subplots(figsize=(14, 10))

    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except (ImportError, Exception):
        pos = nx.spring_layout(G, seed=42, k=2.0)

    labels = nx.get_node_attributes(G, "label")

    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=1800, alpha=0.9, ax=ax,
    )
    nx.draw_networkx_edges(
        G, pos, arrows=True, arrowsize=20, edge_color="#616161", alpha=0.7, ax=ax,
    )
    nx.draw_networkx_labels(
        G, pos, labels=labels, font_size=7, font_weight="bold", ax=ax,
    )

    # Legenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4CAF50", label="Termos preditos"),
        Patch(facecolor="#90CAF9", label="Ancestrais"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)
    ax.set_title("Hierarquia GO — Termos Preditos", fontsize=14, fontweight="bold")
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Grafico DAG salvo em %s", output_path)


def plot_metrics_comparison(
    hierarchical_metrics: dict,
    flat_metrics: dict,
    classifier_name: str,
    output_path: Path,
) -> None:
    """Gera grafico de barras comparando metricas hierarquicas vs flat."""
    labels = ["Precision", "Recall", "F-measure"]
    h_values = [
        hierarchical_metrics["hP"],
        hierarchical_metrics["hR"],
        hierarchical_metrics["hF"],
    ]
    f_values = [
        flat_metrics["flat_P"],
        flat_metrics["flat_R"],
        flat_metrics["flat_F"],
    ]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))

    bars_h = ax.bar(
        [i - width / 2 for i in x], h_values, width,
        label="Hierarquica", color="#1976D2", alpha=0.85,
    )
    bars_f = ax.bar(
        [i + width / 2 for i in x], f_values, width,
        label="Flat", color="#FF7043", alpha=0.85,
    )

    # Valores nas barras
    for bar in bars_h:
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9,
        )
    for bar in bars_f:
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9,
        )

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        f"Metricas Hierarquicas vs Flat — {classifier_name}",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Grafico de comparacao salvo em %s", output_path)
