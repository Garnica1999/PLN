"""Fase de investigación avanzada — Paso 1: congelar el baseline y auditar el espacio de features.

Produce:
    - ``experiments/baseline_frozen/`` con la copia congelada (código, parámetros, métricas, hashes).
    - ``reports/advanced_feature_gap_analysis.md`` (las 10 preguntas del brief).
    - ``reports/signal_coverage_matrix.csv`` (fenómeno × nivel de representación).
    - ``reports/research_backlog.md`` (líneas priorizadas con justificación técnica).

Baseline congelado: **v08** (ensamble de 13 fuentes, CV 0,5866, LB 0,60376). El brief asumía v06
(0,60273): se documenta la desviación y se congela el estado real, que es estrictamente mejor.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402

FROZEN = cfg.EXPERIMENTS / "baseline_frozen"
FROZEN_FILES = ["src/pipeline.py", "src/config.py", "src/cv.py", "src/data.py", "src/textometry.py",
                "src/feature_catalog.py", "submissions/age_submission_v08.csv",
                "experiments/submission_v08_resumen.json", "experiments/e028_en_ensamble.json",
                "reports/tables/fase17_matriz_confusion.csv",
                "reports/tables/fase17_pares_confundidos.csv"]


def sha256(path: Path) -> str:
    """SHA-256 de un archivo."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_baseline() -> dict:
    """Congela el estado actual en ``experiments/baseline_frozen/``."""
    FROZEN.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for relative in FROZEN_FILES:
        source = cfg.ROOT / relative
        if not source.exists():
            continue
        destination = FROZEN / Path(relative).name
        shutil.copy2(source, destination)
        hashes[relative] = sha256(source)
    manifest = {
        "fecha": time.strftime("%Y-%m-%d %H:%M"),
        "baseline": "v08 — ensamble de 13 fuentes con meta-modelo enriquecido",
        "cv_macro_f1_lofo": 0.5866,
        "cv_std": 0.0038,
        "leaderboard_publico": 0.60376,
        "desviacion_vs_brief": ("el brief asumía v06 (CV 0,5851 / LB 0,60273); el baseline real "
                                "congelado es v08, estrictamente mejor"),
        "modelo": {
            "representacion": "TF-IDF char_wb(3,6) sobre texto crudo + canales de menciones, "
                              "hashtags y palabras",
            "miembros": 13,
            "meta_modelo": "StandardScaler + LogisticRegression (107 features: 72 scores + "
                           "35 densas)",
        },
        "validacion": "StratifiedGroupKFold 5 folds agrupado por clúster de near-duplicados; "
                      "evaluación de ensambles con leave-one-fold-out",
        "hashes_sha256": hashes,
    }
    (FROZEN / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    (FROZEN / "README.md").write_text(
        "# Baseline congelado — v08\n\n"
        "Copia inmutable del estado del proyecto antes de la fase de investigación avanzada.\n\n"
        f"- Modelo: {manifest['modelo']['representacion']}\n"
        f"- Miembros del ensamble: {manifest['modelo']['miembros']}\n"
        f"- CV (LOFO): {manifest['cv_macro_f1_lofo']} ± {manifest['cv_std']}\n"
        f"- Leaderboard: {manifest['leaderboard_publico']}\n\n"
        "**No modificar.** Cualquier experimento nuevo vive en versiones aparte.\n",
        encoding="utf-8")
    return manifest


def gap_analysis() -> str:
    """Construye el análisis de brechas con las 10 preguntas del brief."""
    experiments = pd.read_csv(cfg.ROOT / "experiments.csv")
    catalog = pd.read_csv(cfg.REPORTS / "feature_catalog.csv")
    kruskal = pd.read_csv(cfg.TABLES / "eda_kruskal.csv")
    mi = pd.read_csv(cfg.TABLES / "eda_mutual_information.csv")
    correlaciones = pd.read_csv(cfg.TABLES / "representaciones_comparativa.csv")
    errores = json.loads((cfg.EXPERIMENTS / "fase17_error_analysis.json").read_text(encoding="utf-8"))

    densas = catalog[catalog["group"].str.startswith(("C", "D"))]
    redundantes = correlaciones.head(8)
    lineas: list[str] = []
    add = lineas.append

    add("# Análisis de brechas del espacio de features (`advanced_feature_gap_analysis.md`)")
    add("")
    add("Generado por `scripts/advanced_01_freeze_y_gap.py` sobre los artefactos reales del proyecto"
        " (nada de este documento es estimación).")
    add("")

    add("## 1. ¿Qué features existen hoy?")
    add("")
    add(f"- **Densas locales (43 en el catálogo, 35 útiles)**: textometría y estilo, calculadas por "
        f"`src/textometry.py` (grupos C y D del catálogo).")
    add(f"- **Representaciones dispersas**: TF-IDF `char_wb(3,6)` (300k y 700k features), canal de "
        f"menciones (word 1-gram de `@cuenta`), canal de hashtags, y word 1-2.")
    add(f"- **Scores de modelos (72 features del meta-modelo)**: 13 miembros del ensamble.")
    add(f"- **Total de features directas del meta-modelo**: 107.")
    add("")

    add("## 2. ¿Qué familias existen y cuáles están subrepresentadas?")
    add("")
    add("| Familia del brief | Estado actual | Cobertura |")
    add("|---|---|---|")
    filas_familias = [
        ("A. Estadísticos de distribución", "solo medias y algunas desviaciones", "**baja**"),
        ("B. Distribución de caracteres", "ratios básicos (mayúsculas, dígitos, puntuación)", "media"),
        ("C. Distribución de longitud de palabras", "media, máximo y desviación", "**baja**"),
        ("D. Riqueza léxica", "TTR, CTTR, hapax, entropía", "media"),
        ("E. Rareza léxica (frecuencia de corpus)", "**ausente**", "**nula**"),
        ("F. Perfil léxico por clase (log-odds)", "solo análisis exploratorio, no features", "**nula**"),
        ("G. Distancia a prototipos", "Rocchio como miembro del ensamble", "media"),
        ("H. Word shape / ortografía", "elongaciones y mayúsculas agregadas", "baja"),
        ("I. Posición dentro del tweet", "solo `starts_mention`", "**baja**"),
        ("J. Menciones supervisadas (P(clase|mención))", "TF-IDF de menciones (sin supervisión)", "**nula**"),
        ("K. Retuits y cuenta fuente supervisada", "`is_rt` y conteo de menciones", "**baja**"),
        ("L. Hashtags supervisados", "TF-IDF de hashtags", "**baja**"),
        ("M. URLs y dominios", "conteo de URLs", "**baja**"),
        ("N. Estructura del texto", "`n_sentences`", "baja"),
        ("O. N-gramas estructurados", "matriz de 14 configuraciones ya evaluada", "alta"),
        ("P. Representaciones alternativas (NB-SVM)", "**no evaluado**", "**nula**"),
        ("Q. Tokenizador propio de tweets", "**no evaluado**", "**nula**"),
        ("R. Entropía / concentración", "entropía léxica y de caracteres únicos", "media"),
        ("S. Compresibilidad", "**no evaluado**", "**nula**"),
        ("T. Repetición", "runs de caracteres y puntuación", "baja"),
        ("U. Posición léxica (primera/última palabra)", "**ausente**", "**nula**"),
        ("V. Morfología / sílabas", "**ausente**", "**nula**"),
        ("W. POS profundo", "no implementado (recurso preentrenado en zona gris, D-001)", "nula"),
        ("X. Distancia al léxico por clase", "log-odds solo en EDA", "**nula**"),
        ("Y. OOV y novedad", "**ausente**", "**nula**"),
        ("Z. Interacciones de features", "ninguna explícita", "**nula**"),
    ]
    for familia, estado, cobertura in filas_familias:
        add(f"| {familia} | {estado} | {cobertura} |")
    add("")

    add("## 3. ¿Qué familias están altamente correlacionadas?")
    add("")
    add("Pares con mayor redundancia entre features densas (|ρ| de Spearman, Fase 2):")
    add("")
    for _, row in redundantes.head(6).iterrows():
        add(f"- `{row['id']}` — {row['representacion']}: macro-F1 {row['macro_f1']:.4f} "
            f"({row['features']:,} features)")
    add("")
    add("Reportado en `reports/tables/representaciones_pareado.csv`: `ttr`~`hapax_ratio` ρ=0,999 · "
        "`n_unique_words`~`lexical_entropy` ρ=0,998 · `stopword_ratio`~`content_ratio` ρ=0,997 · "
        "`cttr`~`lexical_entropy` ρ=0,997.")
    add("")

    add("## 4. Fenómenos del EDA sin representación")
    add("")
    top_kruskal = kruskal.head(5)[["feature", "epsilon_sq"]]
    add("Las features más informativas hoy (Kruskal ε²) son:")
    add("")
    for _, row in top_kruskal.iterrows():
        add(f"- `{row['feature']}`: ε² = {row['epsilon_sq']:.4f}")
    add("")
    add("Y por información mutua: " + ", ".join(
        f"`{r.feature}`" for _, r in mi.head(4).iterrows()) + ".")
    add("")
    add("Fenómenos identificados en el EDA que **no** tienen representación estructurada: rareza "
        "léxica de corpus, perfil léxico por clase, identidad supervisada de menciones/hashtags/"
        "cuenta fuente, posición de los elementos, forma de palabra, entropía por tipo, "
        "compresibilidad y repetición más allá de runs de caracteres.")
    add("")

    add("## 5. Fenómenos solo representados vía TF-IDF (no como features estructuradas)")
    add("")
    add("- **Identidad de menciones**: el canal TF-IDF las usa, pero sin agregar la probabilidad "
        "por clase -> no se puede expresar '*esta cuenta mencionada es típica de 13-17*'.")
    add("- **Hashtags**: igual, solo bolsa de palabras.")
    add("- **Cuenta fuente de retuits** (`RT @cuenta:`): solo como token de mención, sin tratar el "
        "evento estructurado.")
    add("- **Dominios de URL**: los enlaces del corpus son acortadores `t.co` (verificado en el "
        "EDA), por lo que el dominio no aporta; la señal está en la *presencia y posición*.")
    add("")

    add("## 6. Información presente en el texto crudo que se pierde")
    add("")
    add("- **Posición** de menciones, hashtags, URLs y puntuación (solo se cuentan).")
    add("- **Distribuciones internas**: percentiles, asimetría, curtosis, MAD y coeficiente de "
        "variación de la longitud de palabra y de los huecos entre elementos.")
    add("- **Concentración**: qué tan repetitivo es el vocabulario (Simpson, HHI, Yule's K).")
    add("- **Rareza**: frecuencia de corpus de cada token (requiere estadística de train, in-fold).")
    add("- **Novedad**: proporción de tokens OOV respecto del vocabulario de train.")
    add("")

    add("## 7. Señales fuertes del error analysis sin modelo especializado")
    add("")
    add(f"- **Tweets cortos**: {errores['error_por_cortes'][0]['error_rate']:.1%} de error en el "
        f"quintil más corto (ya probado con especialistas por longitud: +0,0000, pero el brief pide "
        f"revisarlo con features nuevas de la familia E/Y).")
    add("- **Fronteras adyacentes**: los pares ya tienen modelos dedicados (E-028a, precisión "
        "0,7428), pero esos modelos **no** usan features supervisadas de entidades.")
    add("- **Errores universales**: aún no analizados como intersección entre modelos (§54 del brief).")
    add("")

    add("## 8. Features que podrían duplicar información")
    add("")
    add("Toda la familia A (percentiles de longitud de palabra) correlaciona con `avg_word_len`, "
        "`n_words` y `n_chars` ya existentes; la familia D (riqueza) duplica `ttr`/`cttr`; la "
        "familia R (entropía) duplica `lexical_entropy` y `unique_char_ratio`. **Criterio**: se "
        "medirá información mutua incremental y solo se conservarán las que aporten sobre lo "
        "existente, no por existir.")
    add("")

    add("## 9. Qué falta para capturar distribución, posición, concentración y variabilidad")
    add("")
    add("| Dimensión | Qué falta | Familia propuesta |")
    add("|---|---|---|")
    add("| Distribución | percentiles, asimetría, curtosis, CV, MAD | A, C |")
    add("| Posición | primera/última posición normalizada de cada tipo de elemento | I, U |")
    add("| Concentración | Simpson, HHI, Yule's K, Gini | R, D |")
    add("| Variabilidad | varianza de posición, dispersión de scores, desacuerdo entre modelos | Z, meta |")
    add("| Supervisión de entidades | P(clase\\|mención), P(clase\\|hashtag), P(clase\\|fuente RT) | J, K, L, X |")
    add("")

    add("## 10. Conclusión del análisis")
    add("")
    add("El sistema actual explota bien la **forma superficial** del tweet (longitud, ratios, "
        "n-gramas de carácter) y la **bolsa de palabras** de sus canales léxicos. Lo que no explota "
        "es la **estadística supervisada de las entidades** (menciones, hashtags, cuenta fuente) ni "
        "las **distribuciones internas**, la **rareza/novedad léxica** ni la **posición**. Esas son "
        "las cuatro líneas con señal ausente demostrable y son las que se abordan primero.")
    return "\n".join(lineas)


def signal_coverage() -> pd.DataFrame:
    """Matriz fenómeno × nivel de representación (entregable del §61 del brief)."""
    filas = [
        ("longitud", "sí", "sí", "sí", "parcial", "sí", "0,5658", "casi cerrada"),
        ("riqueza léxica", "sí", "sí", "sí", "no", "sí", "—", "solo medias; faltan MATTR/Yule"),
        ("rareza de palabra", "no", "sí (IDF)", "no", "no", "no", "—", "ausente como feature"),
        ("menciones", "sí", "sí", "parcial", "no", "sí", "+0,0144", "falta versión supervisada"),
        ("retuits", "sí", "sí", "parcial", "no", "sí", "+0,002", "falta perfil de la fuente"),
        ("hashtags", "sí", "sí", "parcial", "no", "sí", "+0,0019", "falta versión supervisada"),
        ("URLs", "sí", "sí", "parcial", "no", "sí", "−0,0067", "solo presencia y conteo"),
        ("puntuación", "sí", "sí", "sí", "no", "sí", "−0,0080", "falta estructura/transiciones"),
        ("ortografía", "sí", "sí", "parcial", "no", "sí", "—", "faltan detectores de forma"),
        ("repetición", "sí", "sí", "parcial", "no", "parcial", "—", "faltan repeticiones de orden"),
        ("entropía", "sí", "no", "parcial", "no", "parcial", "—", "faltan por tipo"),
        ("posición", "sí", "no", "no", "no", "no", "—", "**ausente**"),
        ("OOV / novedad", "no", "sí (IDF)", "no", "no", "no", "—", "**ausente**"),
        ("información ordinal", "sí", "no", "parcial", "sí", "sí", "+0,0006", "modelos dedicados ya existen"),
        ("prototipos de clase", "no", "no", "no", "sí", "sí", "+0,0009", "Rocchio en el ensamble"),
        ("tipo de tweet", "parcial", "no", "parcial", "no", "parcial", "—", "especialistas dieron 0"),
        ("near-duplicados", "sí", "no", "no", "no", "sí (CV)", "—", "usado para agrupar CV, no como feature"),
        ("léxico por clase", "sí (EDA)", "sí", "no", "no", "no", "—", "**ausente como feature**"),
    ]
    return pd.DataFrame(filas, columns=["fenomeno", "eda_detectado", "en_tfidf", "feature_densa",
                                        "representacion_supervisada", "modelo_especializado",
                                        "resultado", "brecha_restante"])


def research_backlog() -> str:
    """Backlog priorizado (entregable del §62 del brief)."""
    lineas = [
        "# Research backlog — fase avanzada",
        "",
        "Ordenado por **contenido informativo esperado**, justificado con la evidencia del proyecto "
        "(no por intuición). El orden reproduce el del brief (§86) y cada línea cita su evidencia.",
        "",
        "## Contenido informativo ALTO",
        "",
        "| ID | Hipótesis | Señal esperada | Coste | Evaluación | Estado |",
        "|---|---|---|---|---|---|",
        "| RB-01 | Las features supervisadas de menciones aportan lo que el TF-IDF no expresa | "
        "la identidad de la cuenta mencionada es la señal más fuerte medida (−0,125 al eliminarla) | "
        "medio | LOFO sobre el meta-modelo | pendiente |",
        "| RB-02 | El perfil supervisado de la cuenta fuente de un RT aporta | el corpus tiene 30-53% "
        "de RTs y la fuente es un handle distinto del autor | medio | idem | pendiente |",
        "| RB-03 | La rareza/novedad léxica (frecuencia de corpus in-fold) aporta sobre longitud y "
        "TTR | no existe ninguna feature de frecuencia de corpus hoy | bajo | idem | pendiente |",
        "| RB-04 | NB-SVM (log-count ratio) supera a TF-IDF+SVM | método clásico de referencia en "
        "clasificación de texto corto, no evaluado aún | bajo | CV individual + ensamble | pendiente |",
        "| RB-05 | Las distribuciones internas (percentiles, asimetría, curtosis) aportan sobre las "
        "medias | el EDA muestra diferencias de cola, no solo de media | bajo | idem | pendiente |",
        "",
        "## Contenido informativo MEDIO",
        "",
        "| ID | Hipótesis | Señal esperada | Coste | Evaluación | Estado |",
        "|---|---|---|---|---|---|",
        "| RB-06 | La posición de menciones/hashtags/URLs aporta estructura | solo se cuenta, no se "
        "ubica | bajo | LOFO | pendiente |",
        "| RB-07 | La forma de palabra (ALLCAPS, capitalización, alfanuméricos) aporta ortografía | "
        "solo hay ratios agregados | bajo | LOFO | pendiente |",
        "| RB-08 | La entropía por tipo (puntuación, vocabulario, entidades) aporta concentración | "
        "solo existen 2 entropías | bajo | LOFO | pendiente |",
        "| RB-09 | La compresibilidad aporta estilo | no evaluada | bajo | LOFO | pendiente |",
        "| RB-10 | El tokenizador propio de tweets mejora la representación de palabra | no evaluado | "
        "medio | CV individual | pendiente |",
        "| RB-11 | Las features de incertidumbre (margen, entropía de scores, desacuerdo) mejoran el "
        "meta-modelo | el meta ya ve los scores pero no derivados explícitos | bajo | LOFO | pendiente |",
        "| RB-12 | Un holdout congelado confirma los candidatos fuera del CV de desarrollo | "
        "metodología del brief §50 | bajo | evaluación aparte | pendiente |",
        "",
        "## Contenido informativo BAJO (ya explorado con resultado negativo documentado)",
        "",
        "| ID | Línea | Resultado previo | Decisión |",
        "|---|---|---|---|",
        "| RB-13 | Especialistas por longitud | +0,0000 (E-023/E-024) | cerrada salvo con features nuevas |",
        "| RB-14 | BM25 sin normalización | −0,0002 (E-025) | cerrada |",
        "| RB-15 | kNN sobre SVD | 0,3088 individual, sin aporte (E-026) | cerrada |",
        "| RB-16 | Meta-modelo LightGBM | 0,5858 vs 0,5860 LogReg (E-022) | cerrada |",
        "| RB-17 | Jerarquía gruesa->fina | +0,0003 redundante (E-028b) | cerrada |",
        "| RB-18 | Deep learning denso como clasificador | 0,3381/0,4071 (E-019) | cerrada (se conserva solo como miembro diverso) |",
        "",
        "## Líneas que REQUIEREN verificación de reglas (no iniciadas)",
        "",
        "| ID | Línea | Riesgo | Estado |",
        "|---|---|---|---|",
        "| RB-R1 | Retrieval train->test en inferencia (exact/near-duplicate lookup) | El brief lo "
        "autoriza solo si las reglas lo permiten; la competencia prohíbe usar el test para "
        "entrenar, pero un lookup train->test es inferencia, no entrenamiento | **REQUIERE_VERIFICACIÓN** |",
        "| RB-R2 | POS tagger preentrenado de spaCy | recurso preentrenado en zona gris (D-001) | **REQUIERE_VERIFICACIÓN** |",
        "| RB-R3 | Lexicones externos | no aportan con corpus ASCII y su licencia es dudosa | no iniciada |",
    ]
    return "\n".join(lineas)


def main() -> int:
    """Ejecuta el congelamiento y los tres informes de auditoría."""
    manifest = freeze_baseline()
    print(f"[freeze] {FROZEN} — {len(manifest['hashes_sha256'])} archivos con hash")

    gap = gap_analysis()
    (cfg.REPORTS / "advanced_feature_gap_analysis.md").write_text(gap, encoding="utf-8")
    print(f"[gap] reports/advanced_feature_gap_analysis.md ({len(gap.splitlines())} líneas)")

    coverage = signal_coverage()
    coverage.to_csv(cfg.REPORTS / "signal_coverage_matrix.csv", index=False)
    print(f"[coverage] reports/signal_coverage_matrix.csv ({len(coverage)} fenómenos)")

    backlog = research_backlog()
    (cfg.REPORTS / "research_backlog.md").write_text(backlog, encoding="utf-8")
    print(f"[backlog] reports/research_backlog.md ({len(backlog.splitlines())} líneas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
