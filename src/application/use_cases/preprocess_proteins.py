import pandas as pd

from src.domain.interfaces.preprocessor import ProteinPreprocessor
from src.shared.logger import get_logger

logger = get_logger(__name__)


class PreprocessProteinsUseCase:
    """Orquestra limpeza e normalização de dados de proteínas."""

    def __init__(self, preprocessor: ProteinPreprocessor):
        self._preprocessor = preprocessor

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Iniciando pré-processamento de %d registros", len(data))
        cleaned = self._preprocessor.clean(data)
        normalized = self._preprocessor.normalize(cleaned)
        logger.info("Pré-processamento concluído: %d registros finais", len(normalized))
        return normalized
