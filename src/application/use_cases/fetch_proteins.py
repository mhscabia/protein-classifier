import pandas as pd

from src.domain.interfaces.data_source import ProteinDataSource
from src.shared.logger import get_logger

logger = get_logger(__name__)


class FetchProteinsUseCase:
    """Orquestra a aquisicao de proteinas e termos GO."""

    def __init__(self, data_source: ProteinDataSource):
        self._data_source = data_source

    def execute(self, limit: int) -> pd.DataFrame:
        logger.info("Buscando ate %d proteinas...", limit)
        data = self._data_source.fetch_proteins(limit)

        if not self._data_source.verify_conformity(data):
            raise ValueError(
                "Dados adquiridos nao passaram na verificacao de conformidade"
            )

        logger.info("Aquisicao concluida: %d proteinas validas", len(data))
        return data
