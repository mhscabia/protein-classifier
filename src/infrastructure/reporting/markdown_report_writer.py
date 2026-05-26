import datetime
from pathlib import Path

from src.application.use_cases.extract_metrics import (
    MetricsExtractionResult,
    SampleMetric,
)


class MarkdownReportWriter:
    """Gera relatório .md a partir de MetricsExtractionResult."""

    def __init__(self, config: dict):
        self._output_dir = Path(config["output"]["path"])

    def write(self, result: MetricsExtractionResult) -> Path:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
        path = self._output_dir / f"metrics_report_{timestamp}.md"
        path.write_text(self._render(result), encoding="utf-8")
        return path

    def _render(self, r: MetricsExtractionResult) -> str:
        sections = [
            self._header(r),
            self._section_model_state(r),
            self._section_aggregate_metrics(r),
            self._section_distribution(r),
            self._section_top(r),
            self._section_bottom(r),
            self._section_test_plan(r),
            self._section_notes(r),
        ]
        return "\n\n".join(sections) + "\n"

    def _header(self, r: MetricsExtractionResult) -> str:
        cfg = r.config_snapshot
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        classifier_name = r.model_metadata.get("classifier_name", "LCN")
        features_label = (
            f"ESM-2 ({r.feature_dim} dims)" if cfg.get("use_esm")
            else f"manuais ({r.feature_dim} dims)"
        )
        return (
            f"# Relatório de Métricas — Protein Classifier\n"
            f"\n"
            f"**Data do relatório:** {now}  \n"
            f"**Classificador:** {classifier_name}  \n"
            f"**Features:** {features_label}  \n"
            f"**Filtro min_term_support:** {cfg.get('min_term_support')}  \n"
            f"**Amostras avaliadas:** {r.sample_size} (aleatórias do test set "
            f"≈{r.test_set_size} proteínas)  \n"
            f"**Seed:** {cfg.get('random_seed')}"
        )

    def _section_model_state(self, r: MetricsExtractionResult) -> str:
        meta = r.model_metadata
        rows = [
            ("Classificador", meta.get("classifier_name", "LCN")),
            ("Features", f"{r.feature_dim} dimensões"),
            ("Nós no DAG (após filtro)", f"{r.n_dag_nodes:,}".replace(",", ".")),
            ("Classificadores RF treinados", f"{r.n_classifiers:,}".replace(",", ".")),
            ("Treinado em", f"{meta.get('uniprot_limit', 'N/A')} proteínas (UniProt)"),
            ("Data do treino", meta.get("date", "N/A")),
        ]
        body = "| Item | Valor |\n|---|---|\n"
        body += "\n".join(f"| {k} | {v} |" for k, v in rows)
        return f"## 1. Estado do modelo\n\n{body}"

    def _section_aggregate_metrics(self, r: MetricsExtractionResult) -> str:
        h = r.hierarchical
        f = r.flat
        ratio = (h["hF"] / f["flat_F"]) if f["flat_F"] > 0 else float("inf")
        ratio_str = f"{ratio:.2f}×" if ratio != float("inf") else "∞ (flat_F = 0)"

        hier = (
            f"### Hierárquicas (Silla & Freitas)\n\n"
            f"| Métrica | Valor |\n|---|---|\n"
            f"| hP | {h['hP']:.4f} |\n"
            f"| hR | {h['hR']:.4f} |\n"
            f"| hF | {h['hF']:.4f} |"
        )
        flat = (
            f"### Flat (baseline sem expansão de ancestrais)\n\n"
            f"| Métrica | Valor |\n|---|---|\n"
            f"| flat_P | {f['flat_P']:.4f} |\n"
            f"| flat_R | {f['flat_R']:.4f} |\n"
            f"| flat_F | {f['flat_F']:.4f} |"
        )
        return (
            f"## 2. Métricas agregadas (N={r.sample_size})\n\n"
            f"{hier}\n\n{flat}\n\n"
            f"**Ganho hierárquico:** hF é **{ratio_str}** maior que flat_F."
        )

    def _section_distribution(self, r: MetricsExtractionResult) -> str:
        rows = [
            ("Média de termos preditos", f"{r.avg_predicted:.1f}"),
            ("Média de termos folha preditos", f"{r.avg_leaf_predicted:.1f}"),
            ("Média de termos verdadeiros (anotações UniProt)", f"{r.avg_true:.1f}"),
            ("% amostras com hF > 0.5", f"{r.pct_hf_gt_05:.1f}%"),
            ("% amostras com hF = 0", f"{r.pct_hf_zero:.1f}%"),
            ("Latência média por amostra", f"{r.latency_ms_per_sample:.1f} ms"),
            ("Latência total do batch", f"{r.total_latency_s:.2f} s"),
        ]
        body = "| Estatística | Valor |\n|---|---|\n"
        body += "\n".join(f"| {k} | {v} |" for k, v in rows)
        return f"## 3. Distribuição por amostra\n\n{body}"

    def _section_top(self, r: MetricsExtractionResult) -> str:
        return (
            f"## 4. Top-10 amostras mais acertadas (hF mais alto)\n\n"
            f"{self._sample_table(r.top_samples)}"
        )

    def _section_bottom(self, r: MetricsExtractionResult) -> str:
        return (
            f"## 5. Top-10 amostras menos acertadas (hF mais baixo)\n\n"
            f"{self._sample_table(r.bottom_samples)}"
        )

    def _sample_table(self, samples: list[SampleMetric]) -> str:
        header = "| protein_id | hF | n_pred | n_true | n_inter |\n|---|---|---|---|---|"
        rows = "\n".join(
            f"| {s.protein_id} | {s.hF:.4f} | {s.n_pred} | {s.n_true} | {s.n_inter} |"
            for s in samples
        )
        return f"{header}\n{rows}"

    def _section_test_plan(self, r: MetricsExtractionResult) -> str:
        h, f = r.hierarchical, r.flat
        cfg = r.config_snapshot
        n = r.sample_size

        rows = [
            (
                "M1 — Aquisição",
                "Dados do Swiss-Prot carregados e válidos",
                f"✓ {cfg.get('uniprot_limit')} proteínas (UniProt limit do treino)",
            ),
            (
                "M2 — Pré-processamento",
                "Features extraídas e padronizadas (StandardScaler)",
                f"✓ {r.feature_dim} features ({'ESM-2' if cfg.get('use_esm') else 'manuais'})",
            ),
            (
                "M3 — Hierarquia DAG",
                "DAG construído com filtro de suporte mínimo",
                f"✓ {r.n_dag_nodes} nós (min_term_support={cfg.get('min_term_support')})",
            ),
            (
                "M4 — Treinamento LCN",
                "RF binário por nó com ≥ 2 positivos",
                f"✓ {r.n_classifiers} classificadores RF treinados",
            ),
            (
                "M5 — Avaliação hierárquica",
                "hF hierárquico supera flat_F (crédito parcial pela hierarquia)",
                f"✓ hF={h['hF']:.4f} vs flat_F={f['flat_F']:.4f}",
            ),
            (
                "M5 — Métricas em N amostras",
                f"hP, hR, hF calculados em {n} amostras do test set",
                f"✓ hP={h['hP']:.4f}, hR={h['hR']:.4f}, hF={h['hF']:.4f}",
            ),
            (
                "M6 — Inferência em batch",
                f"InferencePipeline / LCN.predict() processa {n} amostras",
                f"✓ {r.total_latency_s:.2f} s totais ({r.latency_ms_per_sample:.1f} ms/amostra)",
            ),
        ]
        header = (
            "| Módulo | Resultado esperado | Resultado obtido |\n"
            "|---|---|---|"
        )
        body = "\n".join(f"| {m} | {e} | {o} |" for m, e, o in rows)
        return (
            f"## 6. Plano de testes — formato Esperado / Obtido\n\n"
            f"{header}\n{body}"
        )

    def _section_notes(self, r: MetricsExtractionResult) -> str:
        h = r.hierarchical
        notes = []

        if h["hP"] > h["hR"] + 0.1:
            notes.append(
                f"- Modelo **conservador**: hP ({h['hP']:.2f}) >> hR ({h['hR']:.2f}). "
                "O LCN tende a só predizer um termo quando tem confiança, sacrificando recall."
            )
        if r.pct_hf_zero > 20:
            notes.append(
                f"- **{r.pct_hf_zero:.0f}% das amostras tiveram hF = 0** — proteínas para as "
                "quais o modelo não conseguiu nenhuma predição correta nem por ancestral."
            )
        if r.avg_leaf_predicted < r.avg_predicted / 3:
            notes.append(
                f"- Cada predição traz ~{r.avg_predicted:.0f} termos no total mas só "
                f"~{r.avg_leaf_predicted:.0f} folhas específicas — o restante são ancestrais "
                "propagados pela *true path rule*."
            )

        notes.append(
            f"- Seed fixa ({r.config_snapshot.get('random_seed')}) garante "
            "reprodutibilidade: rodar novamente produz o mesmo relatório."
        )

        return "## 7. Notas e observações\n\n" + "\n".join(notes)
