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
├── config.yaml                  ← Parâmetros centralizados
│
├── src/
│   ├── domain/
│   │   ├── entities/            ← Protein, FunctionNode, HierarchyGraph
│   │   └── interfaces/          ← ABCs: HierarchicalClassifier, ProteinDataSource, etc.
│   │
│   ├── application/
│   │   └── use_cases/           ← Orquestração dos módulos
│   │
│   ├── infrastructure/
│   │   ├── data_sources/        ← M1: UniProtClient, GOClient
│   │   ├── preprocessing/       ← M2: PandasPreprocessor + ESMEmbedder
│   │   ├── hierarchy/           ← M3: GODagBuilder (com filtro por suporte)
│   │   ├── models/              ← M4: LCN
│   │   ├── evaluation/          ← M5: HierarchicalMetricsEvaluator
│   │   ├── prediction/          ← M6: InferencePipeline
│   │   ├── persistence/         ← ModelPersistence
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

## O QUE ESTÁ IMPLEMENTADO

### M1 — Aquisição de dados ✅
- `uniprot_client.py` — busca proteínas da Swiss-Prot via API REST com paginação
- `go_client.py` — coleta termos GO e ancestrais via QuickGO em lotes de 25
- Saída: `data/raw/proteins.csv` e `data/raw/go_terms.json`

### M2 — Pré-processamento ✅
- `pandas_preprocessor.py` — limpeza + normalização
- Dois regimes de features (controlado por `features.use_esm` na config):
  - **Manual (22 dims):** `seq_length`, `molecular_weight`, `aa_A`...`aa_Y`
  - **ESM-2 (320 dims):** embedding contextual via `esm_embedder.py`
- `StandardScaler` aplicado em ambos e persistido junto ao modelo

### M3 — Hierarquia DAG ✅
- `go_dag_builder.py`
- Lê `go_terms.json`, monta `HierarchyGraph` com `FunctionNode`
- Filtra termos relevantes (presentes nas proteínas) + ancestrais
- Filtro adicional opcional por `min_term_support` — remove termos com poucas
  proteínas anotadas; ancestrais dos termos mantidos são sempre preservados

### M4 — Treinamento ✅ (LCN-only)
- `lcn_classifier.py` — RF binário por nó, travessia top-down BFS
- `_get_feature_columns()` reconhece prefixos manuais e `esm_`
- Propaga ancestrais nos labels antes de treinar e nas predições

### M5 — Avaliação ✅
- `hierarchical_metrics.py`
- `evaluate()`: hP, hR, hF com expansão de ancestrais antes de comparar
- `evaluate_flat()`: mesmas métricas sem expansão — baseline
- hP = Σ|pred∩true|/Σ|pred| ; hR = Σ|pred∩true|/Σ|true| ; hF = média harmônica

### M6 — Inferência e Visualização ✅
- `inference_pipeline.py` — extrai features (manual ou ESM), aplica scaler, classifica
- `result_visualizer.py` — gera `dag_predictions.png` e `metrics_comparison.png`

### Persistência ✅
- `model_persistence.py` — salva classificador + scaler + metadata
- Metadata inclui `use_esm` e `feature_dim` para validar consistência no load
- `main.py` invalida cache automaticamente se o regime de features mudou

### Resultados (LCN, baseline 22 features, 10000 proteínas, seed=42)

| Configuração | hP | hR | hF | flat_F | Nós DAG |
|---|---|---|---|---|---|
| Baseline (22 features, sem filtro) | 0.8384 | 0.2196 | 0.3480 | 0.0377 | ~600 |
| + filtro min_support=20 | — | — | — | — | — |
| + ESM-2 (sem filtro) | — | — | — | — | — |
| + ESM-2 + filtro min_support=20 | — | — | — | — | — |

**Diagnóstico do recall baixo (baseline):**
- `hP` alta (~0.84): predições tendem a estar corretas.
- `hR` baixo (~0.22): modelo conservador.
- **Causa raiz:** 22 features de composição simples não carregam informação
  contextual/estrutural; com ~600 nós, a maioria dos RFs binários treina com
  poucos positivos e aprende a sempre prever negativo.
- **Solução (Bloco E):** representação melhor (ESM-2) + redução do espaço de classes.

---

## BLOCO E — ESM-2 + Simplificação GO (em execução)

Duas alavancas independentes — efeitos somam.

### Filtro do DAG por suporte mínimo (`min_term_support`)
- Remove termos GO com menos que N proteínas anotadas no dataset.
- Ancestrais dos termos mantidos são sempre preservados.
- Configurado em `hierarchy.min_term_support`.

### ESM-2 — Embeddings via Protein Language Model
- Modelo: `facebook/esm2_t6_8M_UR50D` (8M parâmetros, 320 dims, CPU-friendly).
- O embedding **não prediz função** — é extrator de features. O LCN continua
  sendo o classificador hierárquico.
- Sequências truncadas em `esm_max_length=1022` tokens.
- Cache em `data/processed/esm_embeddings.npz` com `protein_ids` alinhados;
  recomputa quando o conjunto de IDs muda.
- Flag `features.use_esm` controla o regime; `false` mantém retrocompatibilidade.
- Primeira execução baixa ~30MB do HuggingFace — requer conexão.

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
- `use_esm: false` deve sempre funcionar — retrocompatibilidade obrigatória
- Trocar `use_esm` invalida o modelo persistido — `main.py` cuida disso

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
| Recall não melhora com ESM | hR < 0.25 | Combinar com filtro; tentar modelo maior |
| Modelo com features incompatíveis | Erro de shape no load | Metadata `use_esm`/`feature_dim` invalida cache |
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
| Bloco E — Filtro GO min_support | 🔄 |
| Bloco E — ESM-2 embeddings | 🔄 |

> Branch `archive/all-classifiers` preserva a versão com RF + SVM + LCN para referência histórica.

---

## CHECKLIST DE IMPLEMENTAÇÃO

### Blocos A–D — concluídos
- [x] TASK-01..25 — LCN, persistência, escala, refatoração LCN-only

### Bloco E — PLM + Simplificação GO
- [ ] TASK-26 — `hierarchy.min_term_support: 20` em `config.yaml`
- [ ] TASK-27 — `_filter_by_support()` em `go_dag_builder.py`
- [ ] TASK-28 — Integrar filtro no `build()`
- [ ] TASK-29 — `tests/unit/test_go_filter.py` (2 testes)
- [ ] TASK-30 — Pipeline com filtro: registrar nº de nós e métricas
- [ ] TASK-31 — Atualizar tabela
- [ ] TASK-32 — `transformers`/`torch` em `requirements.txt`
- [ ] TASK-33 — Bloco `features:` em `config.yaml`
- [ ] TASK-34 — `esm_embedder.py`
- [ ] TASK-35 — Adaptar `pandas_preprocessor.py` (embedder no construtor)
- [ ] TASK-36 — Adaptar `inference_pipeline.py` (embedder opcional)
- [ ] TASK-37 — Estender `FEATURE_PREFIX` no LCN com `esm_`
- [ ] TASK-38 — `model_persistence.py` registra `use_esm`/`feature_dim`
- [ ] TASK-39 — `main.py` instancia ESM e invalida cache se regime mudou
- [ ] TASK-40 — `tests/unit/test_esm_embedder.py` (4 testes)
- [ ] TASK-41 — `pytest tests/unit/` verde
- [ ] TASK-42 — Pipeline com `use_esm=true` + filtro
- [ ] TASK-43 — Confirmar retrocompatibilidade `use_esm=false`
- [ ] TASK-44 — Atualizar tabela e STATUS
