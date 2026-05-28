import argparse
import datetime
import logging
import sys
from pathlib import Path

import pandas as pd
import questionary
from rich.console import Console
from rich.panel import Panel

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
from src.infrastructure.persistence.hf_downloader import download_models

from src.application.use_cases.prepare_data_pipeline import PrepareDataPipeline
from src.application.use_cases.train_classifiers import TrainClassifiersUseCase
from src.application.use_cases.evaluate_classifiers import EvaluateClassifiersUseCase
from src.application.use_cases.classify_protein import ClassifyProteinUseCase

console = Console()


def _build_embedder(config: dict) -> ESMEmbedder:
    features_cfg = config.get("features", {}) or {}
    return ESMEmbedder(
        model_name=features_cfg["esm_model"],
        cache_path=features_cfg.get("esm_cache_path"),
        max_length=features_cfg.get("esm_max_length", 1022),
        batch_size=features_cfg.get("esm_batch_size", 32),
    )


def _get_sequence(config: dict) -> str:
    predict_cfg = config.get("predict", {}) or {}
    source = predict_cfg.get("sequence_source", "index")
    if source == "input":
        return questionary.text("Cole a sequência de aminoácidos:").ask().strip()
    idx = int(predict_cfg.get("sequence_index", 7))
    proteins_csv = Path(config["data"]["processed_path"]) / "proteins_clean.csv"
    proteins = pd.read_csv(proteins_csv)
    return proteins["sequence"].iloc[idx]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Protein Function Classifier — classifica funções biológicas de proteínas usando hierarquia GO.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemplos:\n"
            "  python main.py                  # usa modelo pré-treinado (baixa do HF se necessário)\n"
            "  python main.py --train          # treina com número de proteínas do config.yaml\n"
            "  python main.py --train 5000     # treina com 5000 proteínas\n"
        ),
    )
    parser.add_argument(
        "--train",
        type=int,
        nargs="?",
        const=-1,
        metavar="N",
        help=(
            "Treina o modelo do zero. "
            "N = número de proteínas a buscar (padrão: valor de data.uniprot_limit no config.yaml)."
        ),
    )
    return parser.parse_args()


def _ask_train_samples(default: int) -> int:
    """Mostra menu para o usuário escolher quantas proteínas treinar."""
    choices = [
        questionary.Choice("5.000  (rápido, ~30 min)", value=5000),
        questionary.Choice("10.000", value=10000),
        questionary.Choice(f"15.000 (padrão, melhor qualidade)", value=15000),
        questionary.Choice("Customizado", value=0),
    ]
    result = questionary.select(
        "Quantas proteínas para o treinamento?",
        choices=choices,
    ).ask()

    if result == 0:
        raw = questionary.text(
            "Digite o número de proteínas:",
            validate=lambda v: v.isdigit() and int(v) > 0 or "Digite um número inteiro positivo",
        ).ask()
        return int(raw)

    return result


def _resolve_model(config: dict, persist_path: str, logger) -> tuple:
    """Carrega modelo local ou guia o usuário pelo menu de opções.

    Retorna (lcn, scaler, meta, compatible, force_train).
    force_train=True indica que o usuário escolheu treinar pelo menu.
    """
    with console.status("[cyan]Verificando modelo em cache...[/cyan]", spinner="dots"):
        _loaded = try_load_model(persist_path)
    if _loaded is not None:
        lcn_disk, scaler_disk, meta_disk = _loaded
        compatible = is_compatible_meta(meta_disk)
        if not compatible:
            logger.warning("Modelo persistido incompatível com a configuração atual — será retreinado")
        return lcn_disk, scaler_disk, meta_disk, compatible, False

    console.print("\n[yellow]Nenhum modelo encontrado localmente.[/yellow]")
    choice = questionary.select(
        "O que deseja fazer?",
        choices=[
            questionary.Choice("Baixar modelo pré-treinado do Hugging Face Hub  (recomendado)", value="download"),
            questionary.Choice("Treinar novo modelo", value="train"),
            questionary.Choice("Sair", value="exit"),
        ],
    ).ask()

    if choice == "exit" or choice is None:
        console.print("Saindo.")
        sys.exit(0)

    if choice == "train":
        default_limit = config["data"]["uniprot_limit"]
        n_proteins = _ask_train_samples(default_limit)
        config["data"]["uniprot_limit"] = n_proteins
        return None, None, None, False, True

    hf_repo = config.get("model", {}).get("hf_repo", "")
    logger.info("Iniciando download do modelo pré-treinado (~2 GB)...")
    download_models(hf_repo, persist_path)
    logger.info("Modelos salvos em %s", persist_path)

    _loaded = try_load_model(persist_path)
    if _loaded is not None:
        lcn_disk, scaler_disk, meta_disk = _loaded
        compatible = is_compatible_meta(meta_disk)
        return lcn_disk, scaler_disk, meta_disk, compatible, False

    return None, None, None, False, False


def main() -> None:
    args = _parse_args()
    force_train = args.train is not None

    config = load_config("config.yaml")

    if force_train and args.train > 0:
        config["data"]["uniprot_limit"] = args.train

    log_level = config.get("logging", {}).get("level", "INFO")
    logging.getLogger().setLevel(getattr(logging, log_level))
    logger = get_logger(__name__)

    mode = config.get("pipeline", {}).get("mode", "auto")

    console.print(
        Panel.fit(
            "[bold cyan]Protein Function Classifier[/bold cyan]\n"
            "[dim]Classificação hierárquica de funções biológicas via Gene Ontology[/dim]",
            border_style="cyan",
        )
    )

    if force_train:
        n_proteins = config["data"]["uniprot_limit"]
        logger.info("Modo treinamento: %d proteínas", n_proteins)
    else:
        logger.info("Modo: %s", mode)

    persist_path = config.get("model", {}).get("persist_path", "data/models/")

    if force_train:
        _lcn_disk = _scaler_disk = _meta_disk = None
        cache_compatible = False
    else:
        _lcn_disk, _scaler_disk, _meta_disk, cache_compatible, train_from_menu = _resolve_model(
            config, persist_path, logger
        )
        if train_from_menu:
            force_train = True

    embedder = _build_embedder(config)
    lcn_metrics = None
    lcn_flat = None

    # Estado 1: tudo pronto — pula direto para previsão
    if not force_train and cache_compatible and hierarchy_exists(persist_path):
        logger.info("Modelo e DAG encontrados em disco — pulando para previsão")
        lcn, scaler = _lcn_disk, _scaler_disk
        with console.status("[cyan]Carregando hierarquia GO do cache...[/cyan]", spinner="dots"):
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

        if not hierarchy_exists(persist_path):
            save_hierarchy(hierarchy, persist_path)
            logger.info("DAG salvo em %s", persist_path)

        presenter.pause_if_interactive(mode, "Módulo 4-5")

        # Estado 2: modelo já existe, só faltava o DAG
        if not force_train and cache_compatible:
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
    example_sequence = _get_sequence(config)

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

    if embedder is not None and embedder._model is not None:
        del embedder._model
        del embedder._tokenizer
        embedder._model = None
        embedder._tokenizer = None

    logger.info("Pipeline concluído com sucesso")


if __name__ == "__main__":
    main()
