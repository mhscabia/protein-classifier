import json
from pathlib import Path

from src.domain.entities.hierarchy_graph import FunctionNode, HierarchyGraph
from src.domain.interfaces.hierarchy_builder import HierarchyBuilder
from src.shared.config_loader import load_config
from src.shared.logger import get_logger

logger = get_logger(__name__)


class GODagBuilder(HierarchyBuilder):
    """Constroi o DAG de termos GO a partir do arquivo go_terms.json."""

    def __init__(self, config: dict | None = None):
        self._config = config or load_config()
        self._raw_path = Path(self._config["data"]["raw_path"])
        self._namespace = self._config["data"]["go_namespace"]
        self._min_term_support = int(
            self._config.get("hierarchy", {}).get("min_term_support", 0)
        )

    def build(
        self,
        go_terms: list[str],
        term_counts: dict[str, int] | None = None,
        min_support: int | None = None,
    ) -> HierarchyGraph:
        terms_data = self._load_terms()
        graph = self._build_graph(terms_data)
        graph = self._filter_relevant(graph, go_terms)

        threshold = self._min_term_support if min_support is None else int(min_support)
        if threshold > 0 and term_counts is not None:
            graph = self._filter_by_support(graph, term_counts, threshold)

        logger.info(
            "DAG construido: %d nos para %d termos de entrada (min_support=%d)",
            len(graph),
            len(go_terms),
            threshold if term_counts is not None else 0,
        )
        return graph

    def _filter_by_support(
        self,
        graph: HierarchyGraph,
        term_counts: dict[str, int],
        min_support: int,
    ) -> HierarchyGraph:
        """Mantem termos com >= min_support proteinas + ancestrais."""
        keep_ids: set[str] = set()
        for term_id in graph.get_all_node_ids():
            if term_counts.get(term_id, 0) >= min_support:
                keep_ids.add(term_id)
                keep_ids.update(graph.get_ancestors(term_id))

        if not keep_ids:
            logger.warning(
                "Filtro min_support=%d removeu todos os termos — DAG vazio",
                min_support,
            )
            return HierarchyGraph()

        filtered = HierarchyGraph()
        for term_id in keep_ids:
            original = graph.get_node(term_id)
            if original is None:
                continue
            filtered.add_node(
                FunctionNode(
                    term_id=original.term_id,
                    name=original.name,
                    parent_ids=[p for p in original.parent_ids if p in keep_ids],
                    children_ids=[c for c in original.children_ids if c in keep_ids],
                )
            )
        logger.info(
            "Filtro por suporte: %d -> %d nos (min_support=%d)",
            len(graph),
            len(filtered),
            min_support,
        )
        return filtered

    def _load_terms(self) -> list[dict]:
        """Carrega go_terms.json gerado pelo GOClient (Modulo 1)."""
        path = self._raw_path / "go_terms.json"
        if not path.exists():
            logger.error("Arquivo nao encontrado: %s", path)
            raise FileNotFoundError(
                f"go_terms.json nao encontrado em {path}. "
                "Execute o Modulo 1 (aquisicao de dados) primeiro."
            )

        with open(path, "r", encoding="utf-8") as f:
            terms = json.load(f)

        logger.info("Carregados %d termos de %s", len(terms), path)
        return terms

    def _build_graph(self, terms_data: list[dict]) -> HierarchyGraph:
        """Monta o HierarchyGraph a partir da lista de dicts do JSON."""
        graph = HierarchyGraph()

        # Primeiro passo: criar todos os nos com parent_ids
        for term in terms_data:
            if term.get("is_obsolete", False):
                continue

            node = FunctionNode(
                term_id=term["term_id"],
                name=term.get("name", ""),
                parent_ids=list(term.get("parent_ids", [])),
            )
            graph.add_node(node)

        # Segundo passo: preencher children_ids a partir dos parent_ids
        for term in terms_data:
            if term.get("is_obsolete", False):
                continue

            term_id = term["term_id"]
            node = graph.get_node(term_id)
            if node is None:
                continue

            for parent_id in node.parent_ids:
                parent_node = graph.get_node(parent_id)
                if parent_node and term_id not in parent_node.children_ids:
                    parent_node.children_ids.append(term_id)

        return graph

    def _filter_relevant(
        self, graph: HierarchyGraph, go_terms: list[str]
    ) -> HierarchyGraph:
        """Filtra o grafo mantendo apenas termos relevantes e seus ancestrais."""
        relevant_ids: set[str] = set()

        for term_id in go_terms:
            if graph.get_node(term_id) is None:
                continue
            relevant_ids.add(term_id)
            relevant_ids.update(graph.get_ancestors(term_id))

        if not relevant_ids:
            logger.warning(
                "Nenhum dos %d termos de entrada encontrado no DAG",
                len(go_terms),
            )
            return HierarchyGraph()

        filtered = HierarchyGraph()
        for term_id in relevant_ids:
            original = graph.get_node(term_id)
            if original is None:
                continue

            node = FunctionNode(
                term_id=original.term_id,
                name=original.name,
                parent_ids=[
                    pid for pid in original.parent_ids if pid in relevant_ids
                ],
                children_ids=[
                    cid for cid in original.children_ids if cid in relevant_ids
                ],
            )
            filtered.add_node(node)

        return filtered
