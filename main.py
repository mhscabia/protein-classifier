import logging

from src.shared.config_loader import load_config
from src.shared.logger import get_logger

from src.infrastructure.data_sources.uniprot_client import UniProtClient
from src.infrastructure.preprocessing.pandas_preprocessor import PandasPreprocessor
from src.infrastructure.hierarchy.go_dag_builder import GODagBuilder
from src.infrastructure.models.random_forest_classifier import (
    RandomForestHierarchicalClassifier,
)
from src.infrastructure.models.svm_classifier import SVMHierarchicalClassifier
from src.infrastructure.evaluation.hierarchical_metrics import (
    HierarchicalMetricsEvaluator,
)

from src.application.use_cases.prepare_data_pipeline import PrepareDataPipeline
from src.application.use_cases.train_classifiers import TrainClassifiersUseCase
from src.application.use_cases.evaluate_classifiers import EvaluateClassifiersUseCase


def main() -> None:
    config = load_config("config.yaml")

    log_level = config.get("logging", {}).get("level", "INFO")
    logging.basicConfig(level=getattr(logging, log_level))
    logger = get_logger(__name__)

    logger.info("Protein Classifier — iniciando pipeline")
    logger.info(
        "Configuracao carregada: uniprot_limit=%d", config["data"]["uniprot_limit"]
    )

    # === Modulos 1-3: Aquisicao, Pre-processamento, Hierarquia ===
    data_pipeline = PrepareDataPipeline(
        data_source=UniProtClient(config),
        preprocessor=PandasPreprocessor(config),
        hierarchy_builder=GODagBuilder(config),
    )
    pipeline_result = data_pipeline.execute(limit=config["data"]["uniprot_limit"])

    proteins = pipeline_result.proteins
    hierarchy = pipeline_result.hierarchy

    # === Modulos 4-5: Treinamento e Avaliacao ===
    classifiers = {
        "RandomForest": RandomForestHierarchicalClassifier(config),
        "SVM": SVMHierarchicalClassifier(config),
    }

    evaluator = HierarchicalMetricsEvaluator(hierarchy)
    eval_use_case = EvaluateClassifiersUseCase(evaluator)

    results = []
    for name, clf in classifiers.items():
        logger.info("=== Treinando %s ===", name)
        train_uc = TrainClassifiersUseCase(classifier=clf, config=config)
        train_result = train_uc.execute(proteins, hierarchy)

        logger.info("=== Avaliando %s ===", name)
        eval_result = eval_use_case.execute(
            classifier=train_result.classifier,
            classifier_name=name,
            X_test=train_result.X_test,
            y_test=train_result.y_test,
        )
        results.append(eval_result)

    # === Comparacao de resultados ===
    logger.info("=== Comparacao de resultados ===")
    for r in results:
        logger.info(
            "%s: hP=%.4f  hR=%.4f  hF=%.4f",
            r.classifier_name,
            r.metrics["hP"],
            r.metrics["hR"],
            r.metrics["hF"],
        )

    best = max(results, key=lambda r: r.metrics["hF"])
    logger.info(
        "Melhor classificador: %s (hF=%.4f)", best.classifier_name, best.metrics["hF"]
    )


if __name__ == "__main__":
    main()
