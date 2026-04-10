from src.infrastructure.prediction.inference_pipeline import InferencePipeline
from src.shared.logger import get_logger

logger = get_logger(__name__)


class ClassifyProteinUseCase:
    """Classifica uma sequencia de proteina em termos GO."""

    def __init__(self, inference_pipeline: InferencePipeline):
        self._pipeline = inference_pipeline

    def execute(self, sequence: str) -> set[str]:
        """Recebe uma sequencia de aminoacidos e retorna termos GO preditos."""
        logger.info("Classificando sequencia de %d aminoacidos", len(sequence))

        predicted_terms = self._pipeline.predict(sequence)

        logger.info("Termos GO preditos: %s", sorted(predicted_terms))
        return predicted_terms
