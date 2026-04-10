import logging
from pathlib import Path

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
from src.infrastructure.prediction.inference_pipeline import InferencePipeline
from src.infrastructure.visualization.result_visualizer import (
    plot_dag_predictions,
    plot_metrics_comparison,
)

from src.application.use_cases.prepare_data_pipeline import PrepareDataPipeline
from src.application.use_cases.train_classifiers import TrainClassifiersUseCase
from src.application.use_cases.evaluate_classifiers import EvaluateClassifiersUseCase
from src.application.use_cases.classify_protein import ClassifyProteinUseCase


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
    preprocessor = PandasPreprocessor(config)
    data_pipeline = PrepareDataPipeline(
        data_source=UniProtClient(config),
        preprocessor=preprocessor,
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
    flat_results = {}
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

        # Metricas flat para comparacao
        y_true = train_result.y_test.tolist()
        flat_metrics = evaluator.evaluate_flat(y_true, eval_result.y_pred)
        flat_results[name] = flat_metrics

    # === Comparacao de resultados ===
    logger.info("=== Comparacao de resultados ===")
    for r in results:
        fm = flat_results[r.classifier_name]
        logger.info(
            "%s: hP=%.4f  hR=%.4f  hF=%.4f | flat_P=%.4f  flat_R=%.4f  flat_F=%.4f",
            r.classifier_name,
            r.metrics["hP"], r.metrics["hR"], r.metrics["hF"],
            fm["flat_P"], fm["flat_R"], fm["flat_F"],
        )

    best = max(results, key=lambda r: r.metrics["hF"])
    logger.info(
        "Melhor classificador: %s (hF=%.4f)", best.classifier_name, best.metrics["hF"]
    )

    # === Modulo 6: Previsao e Visualizacao ===
    logger.info("=== Modulo 6: Previsao e Visualizacao ===")

    # Sequencia de exemplo: primeira proteina do dataset
    example_sequence = proteins["sequence"].iloc[0]
    logger.info(
        "Sequencia de exemplo: %s... (%d aa)",
        example_sequence[:30], len(example_sequence),
    )

    # Inferencia com o melhor classificador
    best_clf = next(
        r for r in results if r.classifier_name == best.classifier_name
    )
    best_classifier_obj = classifiers[best.classifier_name]
    pipeline = InferencePipeline(
        classifier=best_classifier_obj, scaler=preprocessor.scaler,
    )
    classify_uc = ClassifyProteinUseCase(pipeline)
    predicted_terms = classify_uc.execute(example_sequence)

    # Visualizacoes
    output_dir = Path(config.get("output", {}).get("path", "data/output/"))
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_dag_predictions(
        hierarchy, predicted_terms, output_dir / "dag_predictions.png",
    )
    plot_metrics_comparison(
        best.metrics,
        flat_results[best.classifier_name],
        best.classifier_name,
        output_dir / "metrics_comparison.png",
    )

    logger.info("Pipeline concluido com sucesso")


if __name__ == "__main__":
    main()
