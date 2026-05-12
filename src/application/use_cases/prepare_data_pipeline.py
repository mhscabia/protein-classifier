from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from src.domain.entities.hierarchy_graph import HierarchyGraph
from src.domain.interfaces.data_source import ProteinDataSource
from src.domain.interfaces.hierarchy_builder import HierarchyBuilder
from src.domain.interfaces.preprocessor import ProteinPreprocessor
from src.shared.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """Resultado do pipeline de preparacao de dados (Modulos 1-3)."""

    proteins: pd.DataFrame
    hierarchy: HierarchyGraph


class PrepareDataPipeline:
    """Orquestra Modulos 1-3: aquisicao -> pre-processamento -> hierarquia."""

    def __init__(
        self,
        data_source: ProteinDataSource,
        preprocessor: ProteinPreprocessor,
        hierarchy_builder: HierarchyBuilder,
        go_client=None,
    ):
        self._data_source = data_source
        self._preprocessor = preprocessor
        self._hierarchy_builder = hierarchy_builder
        self._go_client = go_client

    def execute(self, limit: int) -> PipelineResult:
        # Modulo 1: aquisicao
        logger.info("=== Modulo 1: Aquisicao de dados ===")
        raw_data = self._data_source.fetch_proteins(limit)

        if not self._data_source.verify_conformity(raw_data):
            raise ValueError(
                "Dados adquiridos nao passaram na verificacao de conformidade"
            )
        logger.info("Aquisicao OK: %d proteinas", len(raw_data))

        # Modulo 2: pre-processamento
        logger.info("=== Modulo 2: Pre-processamento ===")
        cleaned = self._preprocessor.clean(raw_data)
        processed = self._preprocessor.normalize(cleaned)
        logger.info("Pre-processamento OK: %d proteinas", len(processed))

        # Modulo 3: construcao da hierarquia
        logger.info("=== Modulo 3: Construcao da hierarquia ===")
        all_go_terms = self._extract_go_terms(processed)
        term_counts = self._count_term_support(processed)
        self._ensure_go_terms_fetched(all_go_terms)
        hierarchy = self._hierarchy_builder.build(all_go_terms, term_counts=term_counts)
        logger.info("Hierarquia OK: %d nos", len(hierarchy))

        return PipelineResult(proteins=processed, hierarchy=hierarchy)

    def _ensure_go_terms_fetched(self, all_go_terms: list[str]) -> None:
        """Chama go_client.fetch_go_terms() apenas se go_terms.json ainda nao existe."""
        if self._go_client is None:
            return
        raw_path = Path(self._go_client._raw_path)
        go_terms_file = raw_path / "go_terms.json"
        if go_terms_file.exists():
            logger.info("go_terms.json ja existe — pulando busca no QuickGO")
            return
        logger.info("go_terms.json nao encontrado — buscando %d termos GO...", len(all_go_terms))
        self._go_client.fetch_go_terms(all_go_terms)

    @staticmethod
    def _extract_go_terms(data: pd.DataFrame) -> list[str]:
        """Extrai todos os termos GO unicos do DataFrame processado."""
        all_terms: set[str] = set()
        for terms_str in data["go_terms"]:
            for term in str(terms_str).split(";"):
                term = term.strip()
                if term:
                    all_terms.add(term)
        return sorted(all_terms)

    @staticmethod
    def _count_term_support(data: pd.DataFrame) -> dict[str, int]:
        """Conta quantas proteinas tem cada termo GO (sem expansao de ancestrais)."""
        counts: dict[str, int] = {}
        for terms_str in data["go_terms"]:
            seen: set[str] = set()
            for term in str(terms_str).split(";"):
                term = term.strip()
                if term and term not in seen:
                    seen.add(term)
                    counts[term] = counts.get(term, 0) + 1
        return counts
