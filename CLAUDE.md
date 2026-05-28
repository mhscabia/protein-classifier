# CLAUDE.md — Protein Classifier
## Previsão de Função de Proteínas com Classificação Hierárquica

> Este arquivo é lido automaticamente pelo Claude Code a cada sessão.
> **Nunca remova ou edite este arquivo sem instrução explícita.**

---

## OBJETIVO DO SISTEMA

Pipeline de aprendizado de máquina que recebe dados de proteínas e classifica
suas funções biológicas usando classificação hierárquica, respeitando a estrutura
DAG da Gene Ontology (GO).

---

## ARQUITETURA — CLEAN ARCHITECTURE ESTRITA

```
protein_classifier/
│
├── main.py                      ← Ponto de entrada
├── extract_metrics.py           ← Utilitário de extração de métricas
├── config.yaml                  ← Parâmetros centralizados
│
├── src/
│   ├── domain/
│   │   ├── entities/            ← Protein, FunctionNode, HierarchyGraph
│   │   └── interfaces/          ← ABCs: HierarchicalClassifier, ProteinDataSource, etc.
│   │
│   ├── application/
│   │   └── use_cases/           ← PrepareDataPipeline, TrainClassifiers,
│   │                               EvaluateClassifiers, ClassifyProtein,
│   │                               ExtractMetrics
│   │
│   ├── infrastructure/
│   │   ├── data_sources/        ← M1: UniProtClient, GOClient
│   │   ├── preprocessing/       ← M2: PandasPreprocessor + ESMEmbedder
│   │   ├── hierarchy/           ← M3: GODagBuilder (com filtro por suporte)
│   │   ├── models/              ← M4: LCNClassifier
│   │   ├── evaluation/          ← M5: HierarchicalMetricsEvaluator
│   │   ├── prediction/          ← M6: InferencePipeline
│   │   ├── persistence/         ← ModelPersistence, HFDownloader
│   │   ├── reporting/           ← MarkdownReportWriter
│   │   └── visualization/       ← ResultVisualizer
│   │
│   └── shared/                  ← logger, config_loader, presenter
│
├── tests/
│   ├── unit/
│   └── integration/
│
└── data/
    ├── raw/                     ← proteins.csv, go_terms.json
    ├── processed/               ← proteins_clean.csv, esm_embeddings.npz
    ├── models/                  ← modelos persistidos (gitignored)
    └── output/                  ← Gráficos gerados (gitignored)
```

### Regra de dependência (NUNCA violar)
```
infrastructure → application → domain
```

---

## FLUXO PRINCIPAL (M1–M6)

### M1 — Aquisição de dados ✅
- `uniprot_client.py` — busca proteínas da Swiss-Prot via API REST com paginação
- `go_client.py` — coleta termos GO e ancestrais via QuickGO em lotes de 25
- Saída: `data/raw/proteins.csv` e `data/raw/go_terms.json`

### M2 — Pré-processamento ✅
- `pandas_preprocessor.py` — limpeza + normalização
- Features: embeddings ESM-2 (320 dims) via `esm_embedder.py`
- `StandardScaler` aplicado e persistido junto ao modelo

### M3 — Hierarquia DAG ✅
- `go_dag_builder.py`
- Lê `go_terms.json`, monta `HierarchyGraph` com `FunctionNode`
- Filtra termos relevantes (presentes nas proteínas) + ancestrais
- Filtro por `min_term_support` — remove termos com poucas proteínas anotadas;
  ancestrais dos termos mantidos são sempre preservados

### M4 — Treinamento ✅ (LCN-only)
- `lcn_classifier.py` — RF binário por nó, travessia top-down BFS
- Reconhece colunas com prefixo `esm_`
- Propaga ancestrais nos labels antes de treinar e nas predições

### M5 — Avaliação ✅
- `hierarchical_metrics.py`
- `evaluate()`: hP, hR, hF com expansão de ancestrais antes de comparar
- `evaluate_flat()`: mesmas métricas sem expansão — baseline
- hP = Σ|pred∩true|/Σ|pred| ; hR = Σ|pred∩true|/Σ|true| ; hF = média harmônica

### M6 — Inferência e Visualização ✅
- `inference_pipeline.py` — extrai embeddings ESM-2, aplica scaler, classifica
- `result_visualizer.py` — gera `dag_predictions.png` e `metrics_comparison.png`

### Persistência ✅
- `model_persistence.py` — salva/carrega classificador + scaler + metadata
- Metadata inclui `feature_dim` para validar consistência no load
- `hf_downloader.py` — baixa modelos do HuggingFace Hub

### ESM-2 — Embeddings via Protein Language Model ✅
- Modelo: `facebook/esm2_t6_8M_UR50D` (8M parâmetros, 320 dims, CPU-friendly)
- O embedding **não prediz função** — é extrator de features; LCN é o classificador
- Sequências truncadas em `esm_max_length=1022` tokens
- Cache em `data/processed/esm_embeddings.npz` com `protein_ids` alinhados;
  recomputa automaticamente quando o conjunto de IDs muda
- Primeira execução baixa ~30MB do HuggingFace — requer conexão

---

## CONVENÇÕES — OBRIGATÓRIAS

- `snake_case` funções/variáveis, `PascalCase` classes
- Responsabilidade única por função
- Sem comentários óbvios
- DRY
- Seed `42` em todo código que usa aleatoriedade
- Todos os parâmetros no `config.yaml`, nunca hardcoded
- Todo módulo novo em `infrastructure/` deve ter teste em `tests/unit/`
- Dependências sempre apontam para dentro: `infrastructure → application → domain`

---

## TECNOLOGIAS PERMITIDAS

| Biblioteca | Uso |
|---|---|
| pandas | M2, M3 |
| numpy | M2, M5 |
| scikit-learn | M4, M5 |
| matplotlib | M5, M6 |
| requests | M1 |
| pyyaml | shared |
| pytest | tests |
| networkx | M6 visualização |
| joblib | persistence |
| transformers | M2 — ESM embeddings |
| torch | M2 — ESM embeddings (CPU) |

**⚠️ Não adicionar bibliotecas fora desta lista sem aprovação explícita.**

---

## RISCOS

| Risco | Sinal | Ação |
|---|---|---|
| ESM lento em CPU | > 5 min para 500 proteínas | Reduzir `esm_batch_size` |
| Cache ESM desatualizado | Dataset mudou, cache não | `protein_ids` invalidam automaticamente |
| Filtro remove classes importantes | DAG com < 10 nós | Reduzir `min_term_support` para 10 |
| Recall baixo | hR < 0.25 | Combinar ESM com filtro; tentar modelo maior |
| Sequências > 1022 aa | ESM trunca silenciosamente | Esperado; configurável |
| Download ESM falha (offline) | Erro do HuggingFace | Pré-baixar modelo ou rodar online primeiro |

---

## STATUS

| Módulo | Status |
|---|---|
| M1 — Aquisição | ✅ |
| M2 — Pré-processamento | ✅ |
| M3 — Hierarquia DAG | ✅ |
| M4 — Treinamento LCN | ✅ |
| M5 — Avaliação hierárquica | ✅ |
| M6 — Inferência e visualização | ✅ |
| Persistência do modelo | ✅ |
| ESM-2 embeddings | ✅ |
| Filtro GO min_support | ✅ |

---

## HISTÓRICO

- **Classificadores:** anteriormente havia 3 classificadores (LCN + SVM + RandomForest);
  apenas o LCN permanece no fluxo principal.
  Branch `archive/all-classifiers` preserva a versão completa.

- **Features:** anteriormente havia regime alternativo de 22 features manuais
  (`seq_length`, `molecular_weight`, `aa_*`) controlado por flag `use_esm`.
  Removido — ESM-2 é o único regime.

- **Use cases:** anteriormente havia use cases avulsos por módulo
  (`FetchProteinsUseCase`, `PreprocessProteinsUseCase`, `BuildHierarchyUseCase`);
  consolidados em `PrepareDataPipeline`.
