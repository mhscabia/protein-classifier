import logging
from pathlib import Path

from src.shared.config_loader import load_config
from src.shared.logger import get_logger
from src.shared import presenter

from src.infrastructure.data_sources.uniprot_client import UniProtClient
from src.infrastructure.data_sources.go_client import GOClient
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

    mode = config.get("pipeline", {}).get("mode", "auto")

    logger.info("Protein Classifier — iniciando pipeline  [modo: %s]", mode)

    # === Módulos 1-3: Aquisição, Pré-processamento, Hierarquia ===
    preprocessor = PandasPreprocessor(config)
    data_pipeline = PrepareDataPipeline(
        data_source=UniProtClient(config),
        preprocessor=preprocessor,
        hierarchy_builder=GODagBuilder(config),
        go_client=GOClient(config),
    )
    pipeline_result = data_pipeline.execute(limit=config["data"]["uniprot_limit"])

    proteins = pipeline_result.proteins
    hierarchy = pipeline_result.hierarchy

    fetched_count = config["data"]["uniprot_limit"]
    clean_count = len(proteins)
    presenter.print_data_pipeline_result(fetched_count, clean_count, len(hierarchy))
    presenter.pause_if_interactive(mode, "Módulo 4-5")

    # === Módulos 4-5: Treinamento e Avaliação ===
    classifiers = {
        "RandomForest": RandomForestHierarchicalClassifier(config),
        "SVM": SVMHierarchicalClassifier(config),
    }

    evaluator = HierarchicalMetricsEvaluator(hierarchy)
    eval_use_case = EvaluateClassifiersUseCase(evaluator)

    results = []
    flat_results = {}
    for name, clf in classifiers.items():
        logger.info("Treinando %s...", name)
        train_uc = TrainClassifiersUseCase(classifier=clf, config=config)
        train_result = train_uc.execute(proteins, hierarchy)

        logger.info("Avaliando %s...", name)
        eval_result = eval_use_case.execute(
            classifier=train_result.classifier,
            classifier_name=name,
            X_test=train_result.X_test,
            y_test=train_result.y_test,
        )
        results.append(eval_result)

        y_true = train_result.y_test.tolist()
        flat_results[name] = evaluator.evaluate_flat(y_true, eval_result.y_pred)

    best = max(results, key=lambda r: r.metrics["hF"])

    presenter.print_metrics_table(results, flat_results)
    presenter.print_best_classifier(best.classifier_name, best.metrics["hF"])
    presenter.pause_if_interactive(mode, "Módulo 6")

    # === Módulo 6: Previsão e Visualização ===
    example_sequence = proteins["sequence"].iloc[0]

    best_classifier_obj = classifiers[best.classifier_name]
    pipeline = InferencePipeline(
        classifier=best_classifier_obj, scaler=preprocessor.scaler,
    )
    classify_uc = ClassifyProteinUseCase(pipeline)
    predicted_terms = classify_uc.execute(example_sequence)

    output_dir = Path(config.get("output", {}).get("path", "data/output/"))
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_dag_predictions(hierarchy, predicted_terms, output_dir / "dag_predictions.png")
    plot_metrics_comparison(
        best.metrics,
        flat_results[best.classifier_name],
        best.classifier_name,
        output_dir / "metrics_comparison.png",
    )

    presenter.print_prediction_result(
        predicted_terms,
        hierarchy,
        example_sequence,
        best.classifier_name,
        str(output_dir),
    )

    logger.info("Pipeline concluido com sucesso")


if __name__ == "__main__":
    main()
