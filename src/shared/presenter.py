"""Saída formatada para o pipeline — cabeçalhos, tabelas e pausa interativa."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.entities.hierarchy_graph import HierarchyGraph

_BOX_WIDTH = 62


def _box(title: str) -> None:
    top    = "╔" + "═" * (_BOX_WIDTH - 2) + "╗"
    middle = "║  " + title.ljust(_BOX_WIDTH - 4) + "║"
    bottom = "╚" + "═" * (_BOX_WIDTH - 2) + "╝"
    print(f"\n{top}\n{middle}\n{bottom}")


def _row(label: str, value: str, label_width: int = 38) -> None:
    print(f"  {label:<{label_width}}:  {value}")


def _divider() -> None:
    print("  " + "─" * (_BOX_WIDTH - 4))


def print_data_pipeline_result(
    fetched: int, after_clean: int, hierarchy_nodes: int
) -> None:
    _box("MÓDULOS 1-3 — Aquisição, Pré-processamento e Hierarquia")
    _row("Proteínas buscadas (UniProt)", str(fetched))
    _row("Proteínas após limpeza", str(after_clean))
    _row("Nós no grafo GO (hierarquia)", f"{hierarchy_nodes:,}".replace(",", "."))


def print_metrics_table(results: list, flat_results: dict) -> None:
    _box("MÓDULOS 4-5 — Treinamento e Avaliação")
    header = (
        f"  {'Classificador':<14}  "
        f"{'hP':>7}  {'hR':>7}  {'hF':>7}  "
        f"| {'flat_P':>7}  {'flat_R':>7}  {'flat_F':>7}"
    )
    print(header)
    _divider()
    for r in results:
        fm = flat_results[r.classifier_name]
        print(
            f"  {r.classifier_name:<14}  "
            f"{r.metrics['hP']:>7.4f}  {r.metrics['hR']:>7.4f}  {r.metrics['hF']:>7.4f}  "
            f"| {fm['flat_P']:>7.4f}  {fm['flat_R']:>7.4f}  {fm['flat_F']:>7.4f}"
        )


def print_best_classifier(name: str, hf: float) -> None:
    print(f"\n  ★  Melhor classificador: {name}  (hF = {hf:.4f})")


def print_prediction_result(
    predicted_terms: set[str],
    hierarchy: HierarchyGraph,
    example_sequence: str,
    classifier_name: str,
    output_dir: str,
) -> None:
    _box("MÓDULO 6 — Previsão de Função Biológica")
    preview = example_sequence[:40] + ("..." if len(example_sequence) > 40 else "")
    _row("Proteína de exemplo", f"{preview} ({len(example_sequence)} aa)")
    _row("Classificador usado", classifier_name)
    print()
    print("  Funções biológicas previstas:")
    for term_id in sorted(predicted_terms):
        node = hierarchy.get_node(term_id)
        name = node.name if node else "—"
        print(f"    {term_id}  —  {name}")
    print()
    _row("Visualizações salvas em", output_dir)


def pause_if_interactive(mode: str, next_step: str = "") -> None:
    if mode != "interactive":
        return
    prompt = f"\n  [ Pressione ENTER para continuar{' → ' + next_step if next_step else ''} ]  "
    input(prompt)
    print()
