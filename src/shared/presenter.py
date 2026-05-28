from __future__ import annotations

from typing import TYPE_CHECKING

import questionary
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from src.domain.entities.hierarchy_graph import HierarchyGraph

console = Console()


def print_data_pipeline_result(
    fetched: int, after_clean: int, hierarchy_nodes: int
) -> None:
    console.rule("[bold cyan]Módulos 1–3 — Aquisição, Pré-processamento e Hierarquia[/bold cyan]")
    console.print(f"  Proteínas buscadas (UniProt)   [bold]{fetched:,}[/bold]")
    console.print(f"  Proteínas após limpeza         [bold]{after_clean:,}[/bold]")
    console.print(f"  Nós no grafo GO (hierarquia)   [bold]{hierarchy_nodes:,}[/bold]")
    console.print()


def print_metrics_table(results: list, flat_results: dict) -> None:
    console.rule("[bold cyan]Módulos 4–5 — Treinamento e Avaliação[/bold cyan]")

    table = Table(show_header=True, header_style="bold magenta", padding=(0, 2))
    table.add_column("Classificador", style="cyan")
    table.add_column("hP", justify="right")
    table.add_column("hR", justify="right")
    table.add_column("hF", justify="right", style="bold green")
    table.add_column("flat_P", justify="right")
    table.add_column("flat_R", justify="right")
    table.add_column("flat_F", justify="right")

    for r in results:
        fm = flat_results[r.classifier_name]
        table.add_row(
            r.classifier_name,
            f"{r.metrics['hP']:.4f}",
            f"{r.metrics['hR']:.4f}",
            f"{r.metrics['hF']:.4f}",
            f"{fm['flat_P']:.4f}",
            f"{fm['flat_R']:.4f}",
            f"{fm['flat_F']:.4f}",
        )

    console.print(table)
    console.print()


def print_best_classifier(name: str, hf: float) -> None:
    console.print(f"\n  [bold yellow]★[/bold yellow]  Melhor classificador: [cyan]{name}[/cyan]  (hF = {hf:.4f})")


def print_prediction_result(
    predicted_terms: set[str],
    hierarchy: HierarchyGraph,
    example_sequence: str,
    classifier_name: str,
    output_dir: str,
) -> None:
    console.rule("[bold cyan]Módulo 6 — Previsão de Função Biológica[/bold cyan]")
    preview = example_sequence[:40] + ("..." if len(example_sequence) > 40 else "")
    console.print(f"  Proteína         [dim]{preview}[/dim] ({len(example_sequence)} aa)")
    console.print(f"  Classificador    [cyan]{classifier_name}[/cyan]")

    leaf_terms = hierarchy.get_leaf_predicted(predicted_terms)
    console.print(
        f"\n  Funções previstas: [bold]{len(predicted_terms)}[/bold] total, "
        f"[bold green]{len(leaf_terms)}[/bold green] mais específicos\n"
    )

    for term_id in sorted(leaf_terms):
        node = hierarchy.get_node(term_id)
        name = node.name if node else "—"
        console.print(f"    [yellow]{term_id}[/yellow]  [dim]—[/dim]  {name}")

    console.print(f"\n  Visualizações salvas em [dim]{output_dir}[/dim]")
    console.print()


def pause_if_interactive(mode: str, next_step: str = "") -> None:
    if mode != "interactive":
        return
    label = f"Continuar para {next_step}?" if next_step else "Continuar?"
    questionary.confirm(label, default=True).ask()
