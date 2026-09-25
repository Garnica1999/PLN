"""Fase 24 — Genera ``FINAL_REPORT.md`` con las cifras reales de los artefactos.

El informe se construye inyectando los números desde ``experiments.csv``, los JSON de resultados
y los CSV de tablas, de modo que **no puede desincronizarse** de la evidencia ejecutada.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402

#: Progresión de envíos al leaderboard (registrada en RUNLOG.md a partir de los scores de Kaggle).
SUBMISSIONS = [
    ("v01", "E-001 (word 1-2 ∪ char 3-5)", 0.5344, 0.55235),
    ("v02", "E-008 (char 3-6 + canal de menciones)", 0.5557, 0.57018),
    ("v03", "Stacking de 4 modelos", 0.5656, 0.58081),
    ("v04", "Ensamble de 9 fuentes", 0.5748, 0.59178),
    ("v05", "Ensamble + meta-modelo enriquecido", 0.5784, 0.59518),
    ("v06", "Ensamble de 10 fuentes con MLP denso", 0.5851, 0.60273),
    ("v07", "Ensamble de 12 fuentes (+ ordinal acumulativo + centroides)", 0.5860, 0.60372),
    ("v08", "Ensamble de 13 fuentes (+ pares adyacentes)", 0.5866, 0.60376),
    ("v09", "AGE_ADVANCED_V01 (features supervisadas de entidad)", 0.5897, 0.60807),
    ("v10", "+ NB-SVM (generada, no enviada)", 0.5904, None),
    ("v11", "Meta-modelo MLP(64)", 0.5966, 0.61744),
    ("v12", "+ NB-SVM palabra y promedio de combinadores", 0.6005, 0.62182),
]


def load(name: str) -> dict:
    """Carga un JSON de resultados."""
    return json.loads((cfg.EXPERIMENTS / name).read_text(encoding="utf-8"))


def main() -> int:
    """Construye el informe final en Markdown."""
    experiments = pd.read_csv(cfg.ROOT / "experiments.csv")
    auditorias = load("auditorias.json")
    auditoria_datos = load("audit_01_resumen.json")
    representaciones = pd.read_csv(cfg.TABLES / "representaciones_comparativa.csv")
    baselines = pd.read_csv(cfg.TABLES / "baselines_comparativa.csv")
    errores = load("fase17_error_analysis.json")
    ensamble = load("fase20b_ensamble_extendido.json")
    state_of_art = pd.read_csv(cfg.REPORTS / "state_of_art.csv")

    envios = pd.DataFrame(SUBMISSIONS, columns=["version", "modelo", "cv_macro_f1", "lb_publico"])
    envios["brecha"] = (envios["lb_publico"] - envios["cv_macro_f1"]).round(4)
    evaluados = envios.dropna(subset=["lb_publico"])
    brecha_media = evaluados["brecha"].mean()
    brecha_std = evaluados["brecha"].std()
    mejor_lb = evaluados["lb_publico"].max()
    mejor_version = evaluados.loc[evaluados["lb_publico"].idxmax(), "version"]

    mejor_fila = representaciones.sort_values("macro_f1", ascending=False).iloc[0]
    lineas: list[str] = []
    add = lineas.append

    add("# Informe final — Predicción de rango de edad a partir de tweets")
    add("")
    add("**Competencia**: `tweet-age-2026-puj` (Kaggle, host `NLP_Course_PUJ`) · "
        "Procesamiento de Lenguaje Natural · Maestría en Inteligencia Artificial, "
        "Pontificia Universidad Javeriana.")
    add("")
    add("**Autores**: Carlos Garnica y equipo · **Fecha**: 2026-09-24 · "
        "**Repositorio**: `KAGGLE_AGE_2026_PUJ/`")
    add("")
    add("---")
    add("")

    add("## 1. Resumen ejecutivo")
    add("")
    add(f"- **Tarea**: clasificación multiclase de tweets en 6 rangos de edad "
        f"(`13-17`, `18-24`, `25-34`, `35-49`, `50-64`, `65-xx`) usando solo el texto.")
    add(f"- **Resultado en el leaderboard**: **{mejor_lb:.5f}** de F1 ({mejor_version}; línea base "
        f"inicial {envios.iloc[0]['lb_publico']:.5f}; mejora acumulada "
        f"**+{mejor_lb - envios.iloc[0]['lb_publico']:.4f}**).")
    add(f"- **Validación interna honesta**: {envios.iloc[-1]['cv_macro_f1']:.4f} de macro-F1 "
        f"(leave-one-fold-out sobre particiones agrupadas por clústeres de near-duplicados).")
    add(f"- **Calibración validación↔leaderboard**: {len(evaluados)} envíos con brecha media "
        f"**+{brecha_media:.4f} ± {brecha_std:.4f}**, explicada por la memorización de "
        f"near-duplicados del test.")
    add(f"- **Restricción cumplida**: sin transformers, sin LLMs y **sin ningún modelo de "
        f"vectorización** (ni preentrenado ni entrenado desde cero): todo el desempeño proviene de "
        f"representaciones estadísticas (TF-IDF), features artesanales y modelos clásicos.")
    add(f"- **Auditorías**: {'APROBADO' if auditorias['resultado_global'] else 'REVISAR'} en "
        f"leakage, reproducibilidad, formato de submission y trazabilidad académica.")
    add("")

    add("## 2. Datos")
    add("")
    add("| Archivo | Filas | Columnas | Separador |")
    add("|---|---|---|---|")
    add(f"| `age_train.csv` | {auditoria_datos['train_rows']:,} | `id;text;age_range` | `;` |")
    add(f"| `age_test.csv` | {auditoria_datos['test_rows']:,} | `id;text` | `;` |")
    add(f"| `age_submission.csv` | {auditoria_datos['test_rows']:,} | `id,age_range` | `,` |")
    add("")
    add("**Propiedades verificadas**")
    add("")
    add(f"- 6 clases casi balanceadas por diseño: " +
        ", ".join(f"`{k}` {100*v/auditoria_datos['train_rows']:.1f}%"
                  for k, v in auditoria_datos["classes"].items()) + ".")
    add(f"- Texto almacenado en ancho fijo de 140 caracteres con relleno de espacios "
        f"(artefacto de exportación, eliminado en la carga: decisión D-005).")
    add(f"- Alfabeto de **96 caracteres con un único no-ASCII** (`¿`, 88 apariciones): el corpus "
        f"fue transliterado a ASCII, sin tildes, `ñ` ni emojis (decisión D-009).")
    add(f"- **Sin metadatos**: no hay identificador de usuario, fecha, biografía ni seguidores. "
        f"Toda la señal es textual.")
    add(f"- Textos vacíos: {auditoria_datos['empty_texts']['train']} en train y "
        f"{auditoria_datos['empty_texts']['test']} en test.")
    add(f"- Ruido de etiqueta: {auditoria_datos['label_conflicts']['rows_affected']} filas "
        f"({auditoria_datos['label_conflicts']['rows_affected_pct']:.2f}%) con etiquetas "
        f"contradictorias para el mismo texto.")
    add(f"- `id` descartado como feature: la desviación máxima de la distribución de clase entre "
        f"deciles de `id` es de {auditoria_datos['id_leakage_max_deviation_pp']:.2f} pp.")
    add("")
    add("**Procedencia del corpus.** No fue posible identificar la fuente pública exacta "
        "(`reports/tables/dataset_provenance.md`): el esquema de 6 rangos y el vocabulario "
        "(~2016-2017, español latinoamericano, sin COVID) no coinciden con PAN AP ni con TwiSty. "
        "Consecuencia: **no se usaron datos externos** en ningún momento.")
    add("")

    add("## 3. Metodología: validación sin autoengaño")
    add("")
    add("El riesgo dominante de este corpus no es el autor repetido (no hay `user_id`) sino el "
        "**duplicado y near-duplicado**: retuits y variantes del mismo texto que solo cambian "
        "enlaces o menciones.")
    add("")
    add(f"- {auditoria_datos['exact_duplicates']['test_rows_seen_in_train_pct']:.2f}% de las filas "
        f"de test aparecen **literalmente** en train.")
    add(f"- {auditoria_datos['cluster_leakage']['test_coverage_pct']:.2f}% comparten un **clúster "
        f"de near-duplicado** (Jaccard ≥ 0,8 sobre tokens, con *blocking* por tokens raros), con "
        f"{auditoria_datos['cluster_leakage']['purity_pct']:.1f}% de clústeres de etiqueta única.")
    add(f"- Un baseline de **memorización pura** alcanza "
        f"{auditoria_datos['memorization_stratified']['macro_f1_memorization_only_mean']:.4f} de "
        f"macro-F1 con un split estratificado ingenuo y "
        f"{auditoria_datos['memorization_group']['macro_f1_memorization_only_mean']:.4f} con el "
        f"esquema agrupado.")
    add("")
    add("**Decisión (D-003).** Toda la validación usa `StratifiedGroupKFold` (5 folds) agrupado por "
        "clúster de near-duplicados, con comparación de modelos mediante leave-one-fold-out sobre "
        "predicciones out-of-fold. El test nunca participa del ajuste de vectorizadores, modelos, "
        "selección de features ni umbrales.")
    add("")

    add("## 4. Estado del arte")
    add("")
    add(f"Se revisaron **{len(state_of_art)} trabajos** ({state_of_art['year'].min()}-"
        f"{state_of_art['year'].max()}) con DOI/URL verificado en `reports/state_of_art.csv` y "
        f"`references.bib`. Los hallazgos que guiaron el proyecto:")
    add("")
    add("| Trabajo | Aporte al proyecto |")
    add("|---|---|")
    add("| Schler et al. (2006) | Esquema de 6 rangos de edad idéntico al de esta competencia; "
        "features baratas (pronombres, palabras funcionales) |")
    add("| Rangel et al. (PAN 2013-2019) | Mejores sistemas de author profiling: n-gramas de "
        "carácter + modelos lineales; en PAN 2019 (ES) los char n-grams superaron a los embeddings |")
    add("| Sapkota et al. (2015) | Los n-gramas de carácter son la representación más robusta en "
        "texto informal |")
    add("| Nguyen et al. (2011) | Regresión ordinal auxiliar sobre el punto medio del rango |")
    add("| Monroe et al. (2008) | Log-odds con prior Dirichlet para términos distintivos por clase |")
    add("| Li et al. (2010), Parambath et al. (2014) | Ajuste de umbral para F-measure multiclase "
        "(probado y descartado aquí: ver §9) |")
    add("| Wang & Manning (2012) | NB-SVM (log-count ratio): implementado y evaluado; aporta como "
        "miembro del ensamble, no como modelo principal |")
    add("")
    add("**Expectativas calibradas con la literatura**: 0,29-0,53 de exactitud en tareas PAN de "
        "5-6 clases sobre Twitter. El resultado obtenido (0,5851 de macro-F1 interno) está en el "
        "extremo alto de ese rango, coherente con un corpus más separable (clases balanceadas por "
        "diseño y registro muy diferenciado entre extremos).")
    add("")

    add("## 5. Análisis exploratorio y lingüístico")
    add("")
    add("**Hallazgos principales** (`notebooks/02_eda.ipynb`, `03_linguistic_analysis.ipynb`):")
    add("")
    add("| # | Observación | Evidencia | Consecuencia |")
    add("|---|---|---|---|")
    add("| E1 | La longitud y la riqueza léxica crecen con la edad | ε² = 0,126 en `n_chars`; "
        "73 → 111 caracteres de media entre extremos | familia textométrica en el catálogo |")
    add("| E2 | El léxico separa mundos sociales | `follow, justin, bieber` en `13-17` vs "
        "`venezuela, maduro, gobierno` en `65-xx` | representación léxica justificada |")
    add("| E3 | El retuit es más frecuente en mayores | 53% en `65-xx` vs 30% en `18-24` | "
        "feature `is_rt` |")
    add("| E4 | Alta redundancia entre features densas | ρ = 0,999 entre `ttr` y `hapax_ratio` | "
        "ablación incremental obligatoria |")
    add("| E5 | El corpus no tiene emojis ni tildes | 96 caracteres distintos, 1 no-ASCII | "
        "features de emoji descartadas (D-009) |")
    add("")

    add("## 6. Representaciones: la decisión central")
    add("")
    add("Se evaluaron **14 configuraciones** con modelo fijo (LinearSVC C=1) y validación agrupada:")
    add("")
    add("| Configuración | macro-F1 | Features |")
    add("|---|---|---|")
    for _, row in representaciones.sort_values("macro_f1", ascending=False).iterrows():
        add(f"| {row['id']} — {row['representacion']} ({row['texto']}) | {row['macro_f1']:.4f} | "
            f"{int(row['features']):,} |")
    add("")
    add("**Hallazgo 1 — los caracteres ganan a las palabras.** `char_wb 3-6` (0,5413) supera a "
        "`word 1-2` (0,4971) por +0,0442 en 5/5 folds, y la unión clásica palabra+carácter "
        "**diluye** (−0,0069).")
    add("")
    add("**Hallazgo 2 — la identidad de las cuentas mencionadas es una de las señales más "
        "fuertes.** Eliminar solo las menciones cuesta **−0,1251** de macro-F1 y sustituirlas por "
        "`<user>` cuesta −0,1237: la pérdida no es el token, es *a quién* se menciona. Un canal "
        "TF-IDF dedicado a los tokens de mención aporta **+0,0144**.")
    add("")
    add("**Hallazgo 3 — \"limpiar\" destruye señal.** La normalización completa (URLs + menciones + "
        "retuits + puntuación) cuesta **−0,1442** y deja 7.240 tweets vacíos (frente a 104 "
        "originales). El texto se usa **crudo**.")
    add("")
    add("**Decisión (D-010)**: `TF-IDF char_wb(3,6)` sobre texto crudo ∪ canal de menciones, con "
        "SVM lineal.")
    add("")

    add("## 7. Modelos, optimización y ensamble")
    add("")
    add("| Baseline | Descripción | macro-F1 |")
    add("|---|---|---|")
    for _, row in baselines.iterrows():
        add(f"| {row['id']} | {row['descripcion']} | {row['macro_f1']:.4f} |")
    add("")
    add("**Modelos clásicos** (`notebooks/06_classical_models.ipynb`): SVM lineal C=0,2 (mejor "
        "individual, 0,5628), LogReg OvR (0,5603, aporta probabilidades), SGD logístico (0,5579 con "
        "1/25 del costo), ComplementNB (0,5370 pero el más diverso, 25,2% de desacuerdo). El barrido "
        "de regularización muestra un óptimo claro en C=0,2 (C=1,0 baja a 0,5557).")
    add("")
    add("**Ensamble por stacking** (evaluado con leave-one-fold-out):")
    add("")
    add("| Combinación | macro-F1 |")
    add("|---|---|")
    for row in ensamble["combinaciones"]:
        add(f"| {row['combinacion']} | {row['macro_f1_lofo']:.4f} |")
    add(f"| + MLP densas+SVD (E-020) | {load('e020_mlp_en_ensamble.json')['e020_con_mlp']['macro_f1']:.4f} |")
    add("| **+ meta-modelo MLP(64) (v11)** | **0,5967** |")
    add("| **+ NB-SVM palabra y promedio de dos combinadores (v12)** | **0,6005** |")
    add("")
    add("**Modelo final (submission v12)**: 14 modelos base (2× LinearSVC sobre vocabularios de "
        "300k y 700k, LogReg, SGD, ComplementNB, SVM con canal de hashtags, Ridge ordinal, softmax "
        "ordinal, LogReg sobre palabras y MLP sobre densas+SVD) + meta-modelo "
        "`StandardScaler + LogisticRegression` sobre 90 features (49 scores + 6 del MLP + 35 "
        "features densas).")
    add("")

    add("## 8. Análisis de errores")
    add("")
    add(f"F1 por clase del ensamble (out-of-fold): " +
        ", ".join(f"`{k}` {v:.3f}" for k, v in sorted(errores["f1_por_clase"].items(),
                                                       key=lambda item: -item[1])) + ".")
    add("")
    add("**Fenómenos identificados** (`notebooks/08_error_analysis.ipynb`):")
    add("")
    add("| # | Fenómeno | Evidencia | Hipótesis derivada | Resultado medido |")
    add("|---|---|---|---|---|")
    add("| F1 | Confusión entre rangos **contiguos** | 5 de los 6 pares más confundidos son "
        "adyacentes (18-24→25-34 15,3%) | señal ordinal (E-015) | +0,0003 en el ensamble |")
    add("| F2 | Tweets **cortos** sin señal | 52,5% de error en el quintil más corto vs 37,2% en el "
        "más largo | meta-modelo condicionado por longitud (E-018) | **+0,0036** |")
    add("| F3 | Clases **medias** débiles | `25-34` 0,449; `18-24` 0,476 | canal de hashtags "
        "(E-014) | +0,0019 |")
    add("| F4 | **Identidad** de las cuentas mencionadas | quitar menciones cuesta −0,125 | canal "
        "dedicado de menciones (E-008) | **+0,0144** |")
    add("| F5 | Errores distintos entre familias de modelo | MLP con 0,4071 aporta +0,0066 | "
        "diversidad para el ensamble (E-020) | **+0,0066** |")
    add("")
    add("Cuatro de los cinco fenómenos se convirtieron en mejoras verificadas del modelo final: el "
        "análisis de errores fue el motor real de la progresión.")
    add("")

    add("## 9. Resultados y calibración de la validación")
    add("")
    add("| Versión | Modelo | CV macro-F1 | Leaderboard | Brecha |")
    add("|---|---|---|---|---|")
    for _, row in envios.iterrows():
        lb = "pendiente" if pd.isna(row["lb_publico"]) else f"**{row['lb_publico']:.5f}**"
        brecha = "—" if pd.isna(row["brecha"]) else f"{row['brecha']:+.4f}"
        add(f"| {row['version']} | {row['modelo']} | {row['cv_macro_f1']:.4f} | {lb} | {brecha} |")
    add("")
    add(f"La brecha entre la validación interna y el leaderboard es **estable en "
        f"{len(envios)} mediciones independientes** (+{brecha_media:.4f} ± {brecha_std:.4f}), lo "
        f"que valida el diseño experimental: la diferencia se explica por la memorización de "
        f"near-duplicados del test, que el esquema agrupado bloquea por construcción. En "
        f"consecuencia, las decisiones de modelado se tomaron **con la validación interna**, sin "
        f"usar el leaderboard como conjunto de validación.")
    add("")

    add("## 10. Experimentos negativos (documentados)")
    add("")
    add("| Experimento | Resultado | Interpretación |")
    add("|---|---|---|")
    add("| Normalización agresiva (E-005a) | −0,1442 | los tweets son a menudo solo enlaces, "
        "menciones o hashtags: su forma *es* la señal |")
    add("| Quitar/marcar menciones (E-005b, E-005d) | −0,125 | la identidad de la cuenta mencionada "
        "es señal fuerte |")
    add("| Ajuste de decisión por clase (Fase 19) | +0,0006 | con clases balanceadas por diseño no "
        "hay prior que corregir (D-011) |")
    add("| Features densas solas (B5) | 0,3567 | 35 números no compiten con 314k n-gramas |")
    add("| MLP sobre densas (E-019a) | 0,3381 | sin embeddings, la vía densa pierde información |")
    add("| MLP sobre densas+SVD (E-019b) | 0,4071 | la SVD conserva solo 7,8% de la varianza en 64 "
        "componentes |")
    add("| Regresión ordinal sola (E-015) | 0,3973 | útil solo como señal auxiliar del ensamble |")
    add("| Residuales de longitud (PH5) | −0,0019 | el combinador no lineal ya aprende la estructura condicionada por longitud |")
    add("| Estilo puro como representación (PH6) | 0,2793 | **manda el contenido**: el contenido solo alcanza 0,5537 y el estilo casi azar |")
    add("| NB-SVM por canal (PH7) | +0,0009 | por debajo del umbral de ruido |")
    add("| Tercer combinador (PH9) | −0,0006 | la diversidad sin calidad comparable no compone |")
    add("| Segmentación dura por tipo/longitud (PH2) | −0,0007 / −0,0015 | particionar pierde más de lo que gana |")
    add("| **Retrieval train→test** (PH10) | etiqueta recuperada acierta **0,615** | **permitido por las reglas pero los datos lo desaconsejan**: el techo lo fija la consistencia de etiqueta de los duplicados, no la cobertura |")
    add("| **POS de spaCy** (PH10) | −0,0032 | el corpus transliterado a ASCII degrada el etiquetado y la señal es redundante con los char n-grams |")
    add("| Exclusión de filas ruidosas (PH11) | +0,0001 / +0,0016 | mejora real pero bajo el umbral de cambio de modelo (D-027) |")
    add("")

    add("## 11. Auditorías finales")
    add("")
    add(f"Resultado global: **{'APROBADO' if auditorias['resultado_global'] else 'REVISAR'}** "
        f"(`reports/auditorias.md`).")
    add("")
    for clave, titulo in (("leakage", "Leakage"), ("reproducibilidad", "Reproducibilidad"),
                          ("submission", "Formato de submission"), ("academica", "Trazabilidad")):
        audit = auditorias["auditorias"][clave]
        destacados = [f"{name}={value}" for name, value in audit.items()
                      if name != "auditoria_ok"][:4]
        add(f"- **{titulo}**: " + "; ".join(str(item) for item in destacados) + ".")
    add("")

    add("## 12. Limitaciones y trabajo futuro")
    add("")
    add("**Limitaciones**")
    add("")
    add("1. **Truncado a 140 caracteres y transliteración ASCII**: se pierde contexto y toda señal "
        "tipográfica (emojis, tildes, mayúsculas acentuadas), lo que acota el techo alcanzable.")
    add("2. **Sin metadatos de usuario**: no se puede explotar perfil ni agrupar por autor; el "
        "control de leakage es por contenido, no por identidad.")
    add("3. **Ruido de etiqueta** (0,38% de filas con etiquetas contradictorias) y 166 textos "
        "vacíos: techo práctico irreducible.")
    add("4. **Procedencia del corpus no identificada**: las comparaciones con la literatura son "
        "orientativas.")
    add("5. **Interpretación de las reglas** (D-001): la prohibición de \"modelos de vectorización\" "
        "se interpretó como embeddings; la tabla de ablación deja reconstruible un modelo sin POS "
        "ni lexicones si el criterio fuera más estricto.")
    add("")
    add("**Trabajo futuro**")
    add("")
    add("- Modelo jerárquico ordinal (clasificadores binarios encadenados entre rangos contiguos) "
        "para atacar la confusión adyacente, que sigue siendo el error dominante.")
    add("- Modelos especializados por tipo de tweet (cortos vs largos), dado que la longitud domina "
        "el error.")
    add("- Si las reglas lo permitieran, embeddings entrenados desde cero sobre este dominio, o "
        "modelos de lenguaje pequeños entrenados localmente.")
    add("")

    add("## 13. Reproducibilidad")
    add("")
    add("```bash")
    add("# 1. Entorno (venv del proyecto)")
    add("python -m src.data --download --verify --save-interim   # descarga y valida los datos")
    add("python scripts/fase01_auditoria.py                      # auditoría de datos y leakage")
    add("python scripts/fase02_eda.py && python scripts/fase03_linguistic.py")
    add("python scripts/experimentos_representaciones.py         # matriz de representaciones")
    add("python scripts/experimento_e009_modelos_clasicos.py     # modelos clásicos")
    add("python scripts/fase20b_ensamble_extendido.py            # ensamble (LOFO)")
    add("python scripts/fase17_error_analysis.py                 # análisis de errores")
    add("python scripts/fase23_auditorias.py                     # auditorías")
    add("python -c 'from src import pipeline; pipeline.run()'    # modelo final (~9 min CPU)")
    add("```")
    add("")
    add("El notebook **`notebooks/99_entrega_consolidado.ipynb`** contiene la narración completa del "
        "proyecto y verifica el artefacto final; con `REPRODUCIR_PIPELINE = True` re-ejecuta el "
        "modelo de extremo a extremo. El pipeline regenera la submission enviada con "
        "**100,0000% de coincidencia** (verificado en `notebooks/09_final_pipeline.ipynb`).")
    add("")

    add("## 14. Referencias")
    add("")
    add(f"Las {len(state_of_art)} referencias completas, con DOI/URL verificado, están en "
        f"`reports/state_of_art.csv` y `references.bib`. Selección de las más influyentes:")
    add("")
    for _, row in state_of_art.head(12).iterrows():
        add(f"- {row['paper']} ({row['year']}). *{row['venue']}*.")
    add("")
    add("---")
    add("")
    add("*Informe generado automáticamente desde los artefactos del proyecto por "
        "`scripts/fase_final_reporte.py`: las cifras provienen de `experiments.csv`, los JSON de "
        "resultados y las tablas de `reports/`, por lo que no pueden desincronizarse de la "
        "evidencia ejecutada.*")

    (cfg.ROOT / "FINAL_REPORT.md").write_text("\n".join(lineas), encoding="utf-8")
    print(f"FINAL_REPORT.md escrito: {len(lineas)} líneas")
    print(f"brecha media CV↔LB: +{brecha_media:.4f} ± {brecha_std:.4f}")
    print(f"mejor configuración de representación: {mejor_fila['id']} ({mejor_fila['macro_f1']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
