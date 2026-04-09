import logging

from src.shared.config_loader import load_config
from src.shared.logger import get_logger


def main() -> None:
    config = load_config("config.yaml")

    log_level = config.get("logging", {}).get("level", "INFO")
    logging.basicConfig(level=getattr(logging, log_level))
    logger = get_logger(__name__)

    logger.info("Protein Classifier — iniciando pipeline")
    logger.info("Configuração carregada: uniprot_limit=%d", config["data"]["uniprot_limit"])


if __name__ == "__main__":
    main()
