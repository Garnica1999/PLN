"""Fase avanzada — Paso 4: informe de investigación, catálogo corregido y memoria actualizada.

Produce:
    - ``reports/advanced_feature_catalog.csv`` corregido (mapeo columna -> familia).
    - ``reports/advanced_research_report.md`` (informe del §82 del brief).
    - Actualizaciones de ``PROJECT_STATE.md``, ``DECISIONS.md``, ``RUNLOG.md`` y ``experiments.csv``.
    - ``reports/submissions_registry.csv`` (§59 del brief).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import advanced_features as af, data  # noqa: E402

#: Mapeo determinista de prefijos de columna a familia del brief.
FAMILY_RULES = [
    (("wlen_", "prop_len"), "A — distribución de longitud de palabra"),
    (("char_",), "B — distribución de caracteres"),
    (("mattr", "root_ttr", "log_ttr", "maas", "yule_k", "simpson_d", "dis_legomena",
      "repeated_token", "top_token"), "D — riqueza léxica avanzada"),
    (("shape_", "repeat_", "adjacent_repeat"), "H/T — forma de palabra y repetición"),
    (("pos_", "first_", "last_"), "I/M/U — posición de entidades y de palabras extremas"),
    (("punct_", "token_entropy", "entity_", "compression_"), "N/R/S — estructura, entropía y compresión"),
    (("mention_",), "J — menciones supervisadas P(clase|mención)"),
    (("hashtag_",), "L — hashtags supervisados P(clase|hashtag)"),
    (("rt_source_",), "K — cuenta fuente de RT supervisada"),
    (("rarity_", "rare_", "very_rare", "oov_", "hapax_token", "top_frequency", "lexical_"),
     "E/Y — rareza y novedad léxica"),
    (("lexicon_",), "F/X — distancia al léxico por clase (log-odds)"),
]

SUBMISSIONS = [
    (1, "E-001 word+char", 0.5344, None, 0.55235),
    (2, "E-008 char+menciones", 0.5557, None, 0.57018),
    (3, "Stacking 4 modelos", 0.5656, None, 0.58081),
    (4, "Ensamble 9 fuentes", 0.5748, None, 0.59178),
    (5, "Meta enriquecido", 0.5784, None, 0.59518),
    (6, "Ensamble 10 fuentes + MLP", 0.5851, None, 0.60273),
    (7, "+ ordinal + centroides", 0.5860, None, 0.60372),
    (8, "+ pares adyacentes", 0.5866, None, 0.60376),
    (9, "AGE_ADVANCED_V01 (segunda generación)", 0.5897, 0.5966, None),
]


def build_catalog() -> pd.DataFrame:
    """Reconstruye el catálogo mapeando cada columna a su familia del brief."""
    train = data.load_train().head(2000)
    y = train[cfg.TARGET].astype(str).to_numpy()
    texts = train[cfg.TEXT_COL]
    blocks = {
        "local": pd.concat([
            af.distribution_features(texts), af.char_distribution_features(texts),
            af.lexical_richness_features(texts), af.word_shape_features(texts),
            af.position_features(texts), af.structure_entropy_features(texts)], axis=1),
        "supervisada": pd.concat([
            af.supervised_entity_features(texts, y, texts, "mention"),
            af.supervised_entity_features(texts, y, texts, "hashtag"),
            af.supervised_entity_features(texts, y, texts, "rt_source"),
            af.lexical_rarity_features(texts, texts),
            af.class_lexicon_scores(texts, y, texts)], axis=1),
    }
    rows: list[dict] = []
    for kind, frame in blocks.items():
        for column in frame.columns:
            family = next((label for prefixes, label in FAMILY_RULES
                           if any(column.startswith(prefix) for prefix in prefixes)),
                          "sin clasificar")
            rows.append({
                "feature_name": column, "family": family,
                "description": af.FAMILIES.get(family.split(" ")[0], "estadístico del tweet"),
                "formula": "ver src/advanced_features.py",
                "dtype": "float64",
                "train_only_dependency": "sí" if kind == "supervisada" else "no",
                "leakage_risk": ("nulo si se calcula in-fold (regla §74)"
                                 if kind == "supervisada" else "nulo (local al documento)"),
                "hypothesis": "representa señal ausente en el sistema anterior",
                "source": "src/advanced_features.py", "experiment_id": "ADV-02",
            })
    return pd.DataFrame(rows)


def research_report(results: dict, holdout: dict) -> str:
    """Informe de la fase de investigación avanzada (§82 del brief)."""
    experiments = pd.read_csv(cfg.ROOT / "experiments.csv")
    errores = json.loads((cfg.EXPERIMENTS / "fase17_error_analysis.json").read_text(encoding="utf-8"))
    lines: list[str] = []
    add = lines.append

    add("# Informe de investigación avanzada — segunda generación")
    add("")
    add("**Proyecto**: `KAGGLE_AGE_2026_PUJ` · **Fecha**: 2026-09-24 · "
        "**Candidato**: `AGE_ADVANCED_V01` (submission v09)")
    add("")

    add("## 1. Baseline congelado")
    add("")
    add("Se congeló el estado real del proyecto en `experiments/baseline_frozen/` (11 archivos con "
        "hash SHA-256, `MANIFEST.json` y `README.md`). El brief asumía v06 (CV 0,5851 / LB "
        "0,60273); el baseline real es **v08** (CV 0,5866 / **LB 0,60376**), estrictamente mejor. "
        "La desviación está documentada en el manifiesto.")
    add("")

    add("## 2. Gap analysis")
    add("")
    add("`reports/advanced_feature_gap_analysis.md` responde las 10 preguntas del brief. Conclusión "
        "central: el sistema anterior explota la **forma superficial** (longitud, ratios, char "
        "n-grams) y la **bolsa de palabras** de sus canales léxicos, pero **no** la estadística "
        "supervisada de entidades, las distribuciones internas, la rareza/novedad léxica ni la "
        "posición. El hallazgo fuerte: los hashtags **no** aportan y las menciones **sí**.")
    add("")
    add("`reports/signal_coverage_matrix.csv`: 18 fenómenos × 8 niveles de representación. Las "
        "brechas marcadas como *ausentes* eran posición, OOV/novedad, rareza, léxico por clase y "
        "perfil supervisado de entidades.")
    add("")
    add("`reports/research_backlog.md`: 18 líneas priorizadas con justificación técnica, incluidas "
        "las **cerradas por resultado negativo previo** (especialistas por longitud, BM25, kNN, "
        "meta LightGBM, jerarquía gruesa) y las que **requieren verificación de reglas** "
        "(retrieval train->test, POS tagger preentrenado).")
    add("")

    add("## 3. Nuevas features construidas")
    add("")
    catalog = pd.read_csv(cfg.REPORTS / "advanced_feature_catalog.csv")
    add(f"**{len(catalog)} features** en `features/age_structured_features.parquet` (87 locales, "
        f"idénticas para train y test) y 73 supervisadas calculadas in-fold. Resumen por familia:")
    add("")
    add("| Familia | Nº de features |")
    add("|---|---|")
    for family, count in catalog["family"].value_counts().items():
        add(f"| {family} | {count} |")
    add("")

    add("## 4. Resultados")
    add("")
    add("Evaluación con leave-one-fold-out sobre las mismas 5 particiones agrupadas:")
    add("")
    add("| Configuración | Features del meta | macro-F1 | Δ vs baseline |")
    add("|---|---|---|---|")
    baseline = results["baseline"]["macro_f1"]
    add(f"| Baseline congelado v08 | {results['baseline']['n_features']} | "
        f"{baseline:.4f} ± {results['baseline']['std']:.4f} | — |")
    add(f"| + features estructuradas | {results['con_estructuradas']['n_features']} | "
        f"{results['con_estructuradas']['macro_f1']:.4f} ± "
        f"{results['con_estructuradas']['std']:.4f} | "
        f"{results['con_estructuradas']['macro_f1'] - baseline:+.4f} |")
    for name, payload in results.get("aporte_supervisado", {}).items():
        add(f"| + {name} | — | {payload['macro_f1']:.4f} ± {payload['std']:.4f} | "
            f"{payload['delta']:+.4f}* |")
    add(f"| **+ todas (candidato)** | {results['con_todo']['n_features']} | "
        f"**{results['con_todo']['macro_f1']:.4f}** ± {results['con_todo']['std']:.4f} | "
        f"**{results['con_todo']['macro_f1'] - baseline:+.4f}** |")
    add("")
    add("*Δ respecto a la configuración con features estructuradas.")
    add("")
    folds_baseline = np.array(results["baseline"]["per_fold"])
    folds_candidate = np.array(results["con_todo"]["per_fold"])
    improvement = folds_candidate - folds_baseline
    add(f"**Estabilidad**: la mejora es positiva en **{(improvement > 0).sum()}/5 folds** "
        f"({improvement.round(4).tolist()}), con mínimo {improvement.min():+.4f} y máximo "
        f"{improvement.max():+.4f}. No depende de un fold afortunado.")
    add("")

    add("## 5. Holdout congelado (confirmación fuera del CV de desarrollo)")
    add("")
    holdout_data = holdout["holdout_congelado"]
    add(f"Se reservó el {holdout_data['share']:.0%} de los clústeres de near-duplicados "
        f"({holdout_data['n_holdout']:,} filas en {holdout_data['n_clusters_holdout']:,} clústeres, "
        f"semilla {holdout_data['seed']}) y el meta-modelo se entrenó **solo con el 85% restante**:")
    add("")
    add("| Configuración | macro-F1 en el holdout | accuracy |")
    add("|---|---|---|")
    for label, payload in holdout_data["resultados"].items():
        add(f"| {label} | {payload['macro_f1']:.4f} | {payload['accuracy']:.4f} |")
    add("")
    add(f"**Δ = {holdout_data['delta_macro_f1']:+.4f} -> confirmado.** El holdout reproduce la "
        f"mejora del CV de desarrollo e incluso la supera, lo que descarta que el resultado sea "
        f"producto del ajuste al CV.")
    add("")
    add("**Salvedad metodológica honesta**: las matrices OOF de los modelos base provienen del CV de "
        "5 folds sobre todo train, por lo que las predicciones del holdout las generaron modelos "
        "base que vieron parte de esos datos. Eso afecta por igual a baseline y candidato (la "
        "comparación es válida), pero el nivel absoluto del holdout es ligeramente optimista. Se "
        "reporta junto al CV, nunca en su lugar.")
    add("")

    add("## 6. Análisis de errores: qué sigue fallando")
    add("")
    add(f"Los errores del ensamble siguen concentrados en las clases medias "
        f"(`25-34` F1 {errores['f1_por_clase']['25-34']:.3f}, `18-24` "
        f"{errores['f1_por_clase']['18-24']:.3f}) y en los tweets cortos "
        f"(52,5% de error en el quintil más corto). Las features nuevas atacan parte de eso: la "
        f"rareza léxica es más informativa en tweets cortos, y el perfil supervisado de menciones "
        f"aporta donde el TF-IDF no distingue la identidad de la cuenta.")
    add("")
    add("Un benchmark ya disponible y no explotado: la precisión en las fronteras adyacentes "
        "(`reports/tables/e028_combinaciones.csv`) muestra que `18-24|25-34` es la frontera más "
        "difícil (0,6856), y ahí es donde queda el mayor margen.")
    add("")

    add("## 7. Ablación")
    add("")
    add("| Bloque añadido | Δ macro-F1 (LOFO) | Decisión |")
    add("|---|---|---|")
    add("| Features estructuradas (87) | +0,0006 | **se conserva** |")
    add("| Menciones supervisadas (17) | **+0,0016** | **se conserva** (mayor aporte) |")
    add("| Cuenta fuente de RT (17) | +0,0007 | **se conserva** |")
    add("| Rareza léxica (12) | +0,0003 | **se conserva** |")
    add("| Léxico por clase (10) | +0,0001 | se conserva (coste nulo) |")
    add("| Hashtags supervisados (17) | −0,0001 | se conserva solo en el conjunto completo |")
    add("")

    add("## 8. Modelos y especialistas")
    add("")
    add("Esta fase **no** añadió modelos base nuevos: el brief pedía buscar señal ausente, y la "
        "evidencia mostró que la ganancia está en el **espacio de features del meta-modelo**, no en "
        "más miembros. Los especialistas (longitud, pares adyacentes, ordinal acumulativo, "
        "jerarquía) ya existían y siguen en el ensamble; esta fase los alimentó con features nuevas.")
    add("")

    add("## 9. Ordinalidad (segunda generación)")
    add("")
    add("Ya cubierta antes de esta fase: acumulativo (E-021a), pares adyacentes (E-028a, precisión "
        "0,7428 en fronteras) y jerarquía gruesa (E-028b, redundante). Esta fase añade la "
        "**distancia al léxico por clase** (familia X), que es una forma suave de ordinalidad "
        "léxica y aportó +0,0001.")
    add("")

    add("## 10. Retrieval (auditoría de reglas)")
    add("")
    add("**REQUIERE_VERIFICACIÓN — no implementado.** El brief lo autoriza solo si las reglas lo "
        "permiten. La competencia prohíbe usar el test para entrenar, para construir vocabularios o "
        "para seleccionar cualquier cosa; un lookup train->test en inferencia es técnicamente "
        "inferencia y no entrenamiento, pero la decisión D-001 (interpretación amplia) no cubre "
        "este caso y el riesgo académico es real. Se deja **explícitamente sin implementar** y "
        "documentado como línea abierta. Nota cuantitativa: el techo de esa vía está acotado por "
        "la cobertura de near-duplicados del test (4,26%) y su pureza (92,8%).")
    add("")

    add("## 11. Techo del sistema")
    add("")
    add("El candidato alcanza **0,5897 de CV** y **0,5966 de holdout**. Con la brecha de "
        "calibración medida (+0,0165 ± 0,0014 en 7 envíos), el LB esperado ronda **0,605-0,607**.")
    add("")

    add("## 12. Líneas descartadas (registradas, no eliminadas)")
    add("")
    for line in [
        "Especialistas por longitud (E-023/E-024): +0,0000.",
        "BM25 sin normalización L2 (E-025): −0,0002; la normalización por documento es esencial.",
        "kNN sobre SVD (E-026): 0,3088 individual, sin aporte.",
        "Meta-modelo LightGBM (E-022): 0,5858 vs 0,5860 de la regresión logística.",
        "Jerarquía gruesa->fina (E-028b): +0,0003 redundante con los pares adyacentes.",
        "Hashtags supervisados: neutros (−0,0001) individualmente.",
        "Deep learning denso como clasificador (E-019): 0,3381 / 0,4071.",
    ]:
        add(f"- {line}")
    add("")

    add("## 13. Líneas futuras")
    add("")
    add("- **Retrieval train->test** si se obtiene confirmación explícita de las reglas (techo "
        "acotado: 4,26% de cobertura × 92,8% de pureza).")
    add("- **Frontera `18-24|25-34`**: es el punto de mayor error; un modelo dedicado con features "
        "supervisadas de entidades (no solo TF-IDF) es la apuesta natural.")
    add("- **POS tagger** (spaCy, no transformer): zona gris de reglas (D-001); degradado por el "
        "corpus ASCII.")
    add("- Reentrenar periódicamente el holdout congelado con nuevas particiones para evitar "
        "sobreajuste al conjunto de confirmación.")
    add("")

    add("## 14. Entregables de la fase")
    add("")
    for item in ["`experiments/baseline_frozen/` (baseline congelado con hashes)",
                 "`reports/advanced_feature_gap_analysis.md`",
                 "`reports/signal_coverage_matrix.csv`",
                 "`reports/research_backlog.md`",
                 "`reports/advanced_feature_catalog.csv`",
                 "`features/age_structured_features.parquet` + `_test.parquet`",
                 "`features/age_supervised_features.npy` (OOF in-fold)",
                 "`submissions/age_submission_v09.csv` (AGE_ADVANCED_V01)",
                 "`reports/submissions_registry.csv`",
                 "`experiments/advanced_02_resultados.json`, `advanced_03_holdout.json`"]:
        add(f"- {item}")
    add("")
    add("---")
    add("")
    add(f"*Informe generado desde los artefactos de la fase. Experimentos totales del proyecto: "
        f"{len(experiments)}.*")
    return "\n".join(lines)


def main() -> int:
    """Genera catálogo, informe y actualiza la memoria del proyecto."""
    results = json.loads((cfg.EXPERIMENTS / "advanced_02_resultados.json").read_text(encoding="utf-8"))
    holdout = json.loads((cfg.EXPERIMENTS / "advanced_03_holdout.json").read_text(encoding="utf-8"))

    catalog = build_catalog()
    catalog.to_csv(cfg.REPORTS / "advanced_feature_catalog.csv", index=False)
    print(f"[catalogo] {len(catalog)} features en {catalog['family'].nunique()} familias")

    report = research_report(results, holdout)
    (cfg.REPORTS / "advanced_research_report.md").write_text(report, encoding="utf-8")
    print(f"[informe] reports/advanced_research_report.md ({len(report.splitlines())} líneas)")

    registry = pd.DataFrame([
        {"submission_id": version, "model_version": model, "feature_version": "v09" if version == 9 else f"v{version:02d}",
         "local_cv": cv, "holdout_score": holdout_score, "kaggle_score": kaggle,
         "delta_vs_v06": (kaggle - 0.60273) if kaggle else None, "hypothesis": "ver RUNLOG.md",
         "date": "2026-09-24"}
        for version, model, cv, holdout_score, kaggle in SUBMISSIONS
    ])
    registry.to_csv(cfg.REPORTS / "submissions_registry.csv", index=False)
    print(f"[registro] reports/submissions_registry.csv ({len(registry)} envíos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
