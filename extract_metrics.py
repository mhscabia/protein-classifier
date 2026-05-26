"""Extrai métricas do modelo persistido em N amostras do test set.

Uso: python extract_metrics.py

Configurar sample_size via config.yaml em evaluation.sample_size.
Não recarrega o modelo ESM — usa as features já no proteins_clean.csv.
"""
import logging

from src.shared.config_loader import load_config
from src.shared.logger import get_logger
from src.application.use_cases.extract_metrics import ExtractMetricsUseCase
from src.infrastructure.reporting.markdown_report_writer import MarkdownReportWriter


def main() -> None:
    config = load_config("config.yaml")
    log_level = config.get("logging", {}).get("level", "INFO")
    logging.basicConfig(level=getattr(logging, log_level))
    logger = get_logger(__name__)

    logger.info(
        "Extraindo métricas em %d amostras do test set...",
        config["evaluation"]["sample_size"],
    )
    result = ExtractMetricsUseCase(config).execute()
    path = MarkdownReportWriter(config).write(result)

    logger.info("Relatório salvo em: %s", path)
    print(f"\n  ✓ Relatório gerado: {path}")
    print(f"  ✓ N={result.sample_size}  hP={result.hierarchical['hP']:.4f}  "
          f"hR={result.hierarchical['hR']:.4f}  hF={result.hierarchical['hF']:.4f}")
    print(f"  ✓ Latência: {result.latency_ms_per_sample:.1f} ms/amostra "
          f"({result.total_latency_s:.2f} s total)")


if __name__ == "__main__":
    main()
