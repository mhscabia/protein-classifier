from pathlib import Path

from huggingface_hub import hf_hub_download

from src.shared.logger import get_logger

logger = get_logger(__name__)

_MODEL_FILES = ("model.joblib", "hierarchy.joblib")


def download_models(repo_id: str, dest_path: str) -> None:
    Path(dest_path).mkdir(parents=True, exist_ok=True)
    for filename in _MODEL_FILES:
        logger.info("Baixando %s de %s...", filename, repo_id)
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=dest_path,
            repo_type="model",
        )
        logger.info("%s salvo em %s", filename, dest_path)
