# Plan vivo — LEXICON RESIDUAL RESEARCH (KAGGLE_AGE_2026_PUJ)

> Documento de seguimiento de la fase. **Se actualiza al cerrar cada fase.**
> Creado: 2026-09-25 · Rama: `kaggle-age-2026-puj-lexicons` · CPU pura (D-007, sin GPU)

## Pregunta de la fase

> ¿Puede un pequeño número de lexicones lingüísticos generales aportar una señal de registro, etapa
> vital o dominio temático que **complemente** a V12?

No es "las palabras del lexicón son mejores que TF-IDF" (§3): es si la **pertenencia a categorías
lingüísticas de alto nivel** captura dimensiones que no se expresan de forma robusta con tokens
individuales.

## Decisiones de diseño (pre-registradas)

| # | Decisión | Motivo |
|---|---|---|
| DD-1 | **Lexicones autoelaborados en el proyecto** (`lexicons/**/*.txt`), sin descargas | Satisface §1 completo (cond. 1-11): no son embeddings, no son modelos, no llevan etiquetas, no hubo corpus externo, licencia propia, procedencia declarada |
| DD-2 | Se guardan **en español correcto con tildes** y se transliteran en carga | El archivo es auditable por un humano; el matching es determinista (§13) |
| DD-3 | 24 subcategorías: 6 registro + 6 etapas de vida + 12 dominios temáticos | Cada conjunto léxico pertenece a **una sola** familia; resuelve el solape B/C del brief (trabajo, familia, educación) |
| DD-4 | **3 features por subcategoría** (count, ratio, presence) + 1 agregado de familia = **75** | Con 140 chars, `density_per_100_words` es afín a `ratio_over_words` y `count_unique` colapsa en `presence` -> las 5 del §11 son redundantes por construcción. Total dentro del objetivo 10-100 (§23) |
| DD-5 | Token en >1 subcategoría -> **se elimina** (ambigüedad) | Evita doble conteo; regla determinista |
| DD-6 | **Control de permutación**: `V12 + 75 columnas permutadas` | El bloque real debe superar al bloque del mismo tamaño con la señal destruida; si no, la señal es nula aunque ambos "mejoren" |
| DD-7 | Complementariedad (§21) es **diagnóstico, no veto** | Un modelo lexicon-only es débil por construcción y siempre parecería neto-negativo; usarlo como puerta cerraría la fase sin evaluarla |
| DD-8 | Comparación **pareada**: mismos folds, misma semilla, dos cabezas (LogReg primaria, MLP secundaria) | Atribuye el delta al bloque de features y comprueba que el efecto no dependa del tipo de cabeza |

## Criterios pre-registrados

- **Gate de adopción**: Δ ≥ +0,003 **y** ≥4/5 folds positivos (§27), más: sin deterioro fuerte de
  clases centrales, reproducible, complementario, sin leakage.
- **Redundancia (§12)**: se descarta y documenta toda subcategoría altamente correlacionada con
  longitud, nº de palabras, `stopword_ratio` o `content_ratio`.
- **Cierre (§29)**: A (nadie llega al gate) · B (señal pero redundante) · C (marginal inestable) ·
  D (complementariedad insuficiente) · E (una familia supera el gate -> candidato).
- **Prohibido**: holdout (agotado), submissions automáticas, optimizar contra el leaderboard, tocar
  `age_submission_v12*.csv`, force push, merge automático, modificar resultados retroactivamente.
- **Expectativa previa declarada: NULA.** Con el techo agregado del 67% del error y señales léxicas
  previas de +0,0003, el resultado nulo es el más probable; §29 B/C/D lo hacen cierre legítimo.

## Estado de fases

| Fase | Contenido | Entregable | Estado |
|---|---|---|---|
| **Fase 00** | Perfil del entorno (portable) | `config/hardware.json`, `plan/00_maquina.md` | OK 16 núcleos/32 hilos, 31,2 GB, RX 9070 XT |
| **P0** | Auditoría histórica y cobertura de hipótesis | `reports/age_lexicon_hypothesis_coverage.md` | OK 0 experimentos previos mencionan un canal léxico (L2/L3/L4) |
| **P1** | Catálogo de lexicones compatibles | `reports/age_lexicon_catalog.csv`, `lexicons/**` | OK 24 subcat, 1.588->1.369 entradas, 1.091 tokens; cobertura registro 5,32% / vida 0,58% / temas 9,35% |
| **P2** | Registro / informalidad | `features/lexicon_features_registro.npy` | en curso en curso (P2-P5 en `phase38`) |
| **P3** | Etapas de vida / contexto social | `features/lexicon_features_vida.npy` | en curso en curso (P2-P5 en `phase38`) |
| **P4** | Dominios temáticos | `features/lexicon_features_temas.npy` | en curso en curso (P2-P5 en `phase38`) |
| **P5** | Ablación y redundancia | `reports/age_lexicon_ablation.csv` | en curso en curso (P2-P5 en `phase38`) |
| **P6** | Fusión con V12 (familias prometedoras + control permutado) | `reports/age_lexicon_results.csv` | - pendiente |
| **P7** | Análisis de error y complementariedad | `age_lexicon_complementarity.md`, `age_lexicon_error_analysis.md` | - pendiente |
| **P8** | Cierre y memoria | `reports/age_lexicon_audit.md` + memoria del proyecto | - pendiente |

## Hallazgos de la auditoría de cobertura (P0)

- **67 experimentos previos y ninguno de lexicon** -> la línea es genuinamente nueva.
- Precedente de recurso léxico cerrado ya aceptado en el proyecto: `SPANISH_STOPWORDS`
  (~140 entradas autoelaboradas, `src/textometry.py`) -> alimenta `stopword_ratio` y `content_ratio`.
- Único canal léxico **supervisado**: `P(clase|entidad)` para mention/hashtag/rt_source
  (`src/advanced_features.py:326`) -> **no existe canal por palabra libre**, ni general.
- Corpus español **ASCII sin tildes**, 140 chars, **sin emojis** (D-009) -> los recursos basados en
  emoji son inertes por construcción; todo lexicón exige transliteración (DD-2).
- Señal temática ya observada **solo de forma observacional** (Gate A: fandoms/MTV en 13-17 vs
  política/noticias en 65-xx), nunca como canal de features externo y no supervisado -> **ABIERTA**.
- **Auditoría §32**: `transformers`, `sentence-transformers` y `torch` están instalados en el venv y
  listados en `requirements.txt`; uso en el camino de modelado = 0; único import del proyecto es
  `src/gpu.py:130` (detección de GPU) -> se documenta en D-038 sin maquillar el entorno.

## Restricciones que NO cambian

Restricciones absolutas de AGE (sin embeddings, transformers, modelos externos, vectorizadores
aprendidos nuevos, datasets/labels externos ni etiquetas de test) y V12 congelado.

---

_Skill de referencia: `state-of-the-art-research` no aplica: no se buscan nuevas arquitecturas sino
recursos léxicos simbólicos generalistas, cuya admisibilidad se audita en P1, no se cita SOTA de modelos._
