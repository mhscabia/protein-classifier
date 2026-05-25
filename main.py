import datetime
import logging
from pathlib import Path

import pandas as pd

from src.shared.config_loader import load_config
from src.shared.logger import get_logger
from src.shared import presenter

from src.infrastructure.data_sources.uniprot_client import UniProtClient
from src.infrastructure.data_sources.go_client import GOClient
from src.infrastructure.preprocessing.pandas_preprocessor import PandasPreprocessor
from src.infrastructure.preprocessing.esm_embedder import ESMEmbedder
from src.infrastructure.hierarchy.go_dag_builder import GODagBuilder
from src.infrastructure.models.lcn_classifier import LCNClassifier
from src.infrastructure.evaluation.hierarchical_metrics import (
    HierarchicalMetricsEvaluator,
)
from src.infrastructure.prediction.inference_pipeline import InferencePipeline
from src.infrastructure.visualization.result_visualizer import (
    plot_dag_predictions,
    plot_metrics_comparison,
)
from src.infrastructure.persistence.model_persistence import (
    hierarchy_exists,
    is_compatible_meta,
    load_hierarchy,
    save_hierarchy,
    save_model,
    try_load_model,
)

from src.application.use_cases.prepare_data_pipeline import PrepareDataPipeline
from src.application.use_cases.train_classifiers import TrainClassifiersUseCase
from src.application.use_cases.evaluate_classifiers import EvaluateClassifiersUseCase
from src.application.use_cases.classify_protein import ClassifyProteinUseCase


def _build_embedder(config: dict):
    features_cfg = config.get("features", {}) or {}
    if not features_cfg.get("use_esm", False):
        return None
    return ESMEmbedder(
        model_name=features_cfg["esm_model"],
        cache_path=features_cfg.get("esm_cache_path"),
        max_length=features_cfg.get("esm_max_length", 1022),
        batch_size=features_cfg.get("esm_batch_size", 32),
    )


def _get_sequence(config: dict, persist_path: str) -> str:
    predict_cfg = config.get("predict", {}) or {}
    source = predict_cfg.get("sequence_source", "index")
    if source == "input":
        return input("Cole a sequência de aminoácidos: ").strip()
    idx = int(predict_cfg.get("sequence_index", 7))
    proteins_csv = Path(config["data"]["processed_path"]) / "proteins_clean.csv"
    proteins = pd.read_csv(proteins_csv)
    return proteins["sequence"].iloc[idx]


def main() -> None:
    config = load_config("config.yaml")

    log_level = config.get("logging", {}).get("level", "INFO")
    logging.basicConfig(level=getattr(logging, log_level))
    logger = get_logger(__name__)

    mode = config.get("pipeline", {}).get("mode", "auto")
    use_esm = bool(config.get("features", {}).get("use_esm", False))

    logger.info(
        "Protein Classifier — iniciando pipeline  [modo: %s, use_esm=%s]",
        mode,
        use_esm,
    )

    persist_path = config.get("model", {}).get("persist_path", "data/models/")

    # Carrega o modelo uma única vez — reutilizado em todos os caminhos
    _loaded = try_load_model(persist_path)
    if _loaded is not None:
        _lcn_disk, _scaler_disk, _meta_disk = _loaded
        cache_compatible = is_compatible_meta(_meta_disk, use_esm=use_esm)
        if not cache_compatible:
            logger.warning(
                "Modelo persistido incompatível com use_esm=%s — será retreinado", use_esm
            )
    else:
        _lcn_disk = _scaler_disk = _meta_disk = None
        cache_compatible = False

    embedder = _build_embedder(config)
    lcn_metrics = None
    lcn_flat = None

    # Estado 1: tudo pronto — pula direto para previsão
    if cache_compatible and hierarchy_exists(persist_path):
        logger.info("Modelo e DAG encontrados em disco — pulando para previsão")
        lcn, scaler = _lcn_disk, _scaler_disk
        hierarchy = load_hierarchy(persist_path)

    else:
        # Estado 2 ou 3: precisa do pipeline de dados (M1-M3)
        preprocessor = PandasPreprocessor(config, embedder=embedder)
        data_pipeline = PrepareDataPipeline(
            data_source=UniProtClient(config),
            preprocessor=preprocessor,
            hierarchy_builder=GODagBuilder(config),
            go_client=GOClient(config),
        )
        pipeline_result = data_pipeline.execute(limit=config["data"]["uniprot_limit"])
        proteins = pipeline_result.proteins
        hierarchy = pipeline_result.hierarchy

        presenter.print_data_pipeline_result(
            config["data"]["uniprot_limit"], len(proteins), len(hierarchy)
        )

        # Salva o DAG se ainda não existe
        if not hierarchy_exists(persist_path):
            save_hierarchy(hierarchy, persist_path)
            logger.info("DAG salvo em %s", persist_path)

        presenter.pause_if_interactive(mode, "Módulo 4-5")

        # Estado 2: modelo já existe, só faltava o DAG
        if cache_compatible:
            logger.info("Modelo encontrado em disco — pulando treinamento")
            lcn, scaler = _lcn_disk, _scaler_disk
            presenter.pause_if_interactive(mode, "Módulo 6")

        # Estado 3: treina do zero
        else:
            lcn = LCNClassifier(config)
            evaluator = HierarchicalMetricsEvaluator(hierarchy)
            eval_use_case = EvaluateClassifiersUseCase(evaluator)

            logger.info("Treinando LCN...")
            train_uc = TrainClassifiersUseCase(classifier=lcn, config=config)
            train_result = train_uc.execute(proteins, hierarchy)

            scaler = preprocessor.scaler
            feature_dim = scaler.mean_.shape[0] if scaler is not None else None
            save_model(
                lcn,
                scaler,
                {
                    "classifier_name": "LCN",
                    "uniprot_limit": config["data"]["uniprot_limit"],
                    "date": datetime.date.today().isoformat(),
                    "use_esm": use_esm,
                    "feature_dim": feature_dim,
                },
                persist_path,
            )
            logger.info("Modelo salvo em %s", persist_path)

            logger.info("Avaliando LCN...")
            eval_result = eval_use_case.execute(
                classifier=train_result.classifier,
                classifier_name="LCN",
                X_test=train_result.X_test,
                y_test=train_result.y_test,
            )
            lcn_metrics = eval_result.metrics
            y_true = train_result.y_test.tolist()
            lcn_flat = evaluator.evaluate_flat(y_true, eval_result.y_pred)

            presenter.print_metrics_table([eval_result], {"LCN": lcn_flat})
            presenter.pause_if_interactive(mode, "Módulo 6")

    # === Módulo 6: Previsão e Visualização ===
    example_sequence = _get_sequence(config, persist_path)

    pipeline = InferencePipeline(
        classifier=lcn,
        scaler=scaler,
        embedder=embedder,
    )
    classify_uc = ClassifyProteinUseCase(pipeline)
    predicted_terms = classify_uc.execute(example_sequence)

    output_dir = Path(config.get("output", {}).get("path", "data/output/"))
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_dag_predictions(hierarchy, predicted_terms, output_dir / "dag_predictions.png")
    if lcn_metrics and lcn_flat:
        plot_metrics_comparison(
            lcn_metrics,
            lcn_flat,
            "LCN",
            output_dir / "metrics_comparison.png",
        )

    presenter.print_prediction_result(
        predicted_terms,
        hierarchy,
        example_sequence,
        "LCN",
        str(output_dir),
    )

    # Libera o modelo PyTorch antes do garbage collector para evitar lentidão na saída
    if embedder is not None and embedder._model is not None:
        del embedder._model
        del embedder._tokenizer
        embedder._model = None
        embedder._tokenizer = None

    logger.info("Pipeline concluído com sucesso")


if __name__ == "__main__":
    main()
