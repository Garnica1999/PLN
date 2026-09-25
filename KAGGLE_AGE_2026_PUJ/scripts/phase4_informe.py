"""FASE 4 — Informe de la fase avanzada v2, decisiones y memoria actualizada.

Resume la cadena completa: diagnóstico → hipótesis → experimento → resultado → decisión, con las
cifras reales de los artefactos, y actualiza ``PROJECT_STATE.md``, ``DECISIONS.md``, ``RUNLOG.md``
y ``experiments.csv``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402


def main() -> int:
    """Escribe el informe y actualiza la memoria del proyecto."""
    diag = json.loads((cfg.EXPERIMENTS / "phase1_diagnostico.json").read_text(encoding="utf-8"))
    pond = json.loads((cfg.EXPERIMENTS / "phase2_ponderacion.json").read_text(encoding="utf-8"))
    hold = json.loads((cfg.EXPERIMENTS / "phase3_holdout_v11.json").read_text(encoding="utf-8"))
    fixed = json.loads((cfg.EXPERIMENTS / "phase3b_recorregido.json").read_text(encoding="utf-8"))

    lines: list[str] = []
    add = lines.append
    add("# Informe de investigación avanzada v2 — de la ponderación a la señal ausente")
    add("")
    add("**Proyecto**: `KAGGLE_AGE_2026_PUJ` · **Fecha**: 2026-09-24 · "
        "**Candidato**: `v11` (meta-modelo MLP)")
    add("")
    add("---")
    add("")
    add("## 1. Baseline congelado")
    add("")
    add("`experiments/baseline_v09_frozen/` (16 archivos con hash SHA-256 y `MANIFEST.json`): "
        "v09 = **0,60807** en el leaderboard, CV 0,5897, holdout 0,5966. Envíos usados: 9 de ~10.")
    add("")
    add("## 2. Gap analysis v2")
    add("")
    add("`reports/advanced_gap_analysis_v2.md`: tabla de 22 fenómenos con estado de detección, "
        "representación, versión supervisada, canal de modelo y **Δ medido**. Su función fue evitar "
        "re-ejecutar ocho líneas ya cerradas (mención/hashtag/RT supervisados, distribución, rareza, "
        "prototipos, NB-SVM, especialistas por longitud, incertidumbre). Brechas confirmadas: "
        "errores de intersección (L1), residualización por longitud (L2), contenido vs estilo (L3), "
        "transiciones de carácter (L4), tipología de tweets (L5), complementariedad (L6).")
    add("")
    add("## 3. Diagnóstico (L1 + L5) — el hallazgo que reorientó la fase")
    add("")
    add(f"- **{diag['ensemble_miss_pct']}%** de los tweets: el ensamble falla y **al menos un modelo "
        f"base acierta**.")
    add(f"- Los modelos que más rescatan son los **débiles pero diversos**: softmax ordinal "
        f"(36,1%), centroides (31,1%), MLP densas+SVD (30,5%), LogReg palabra (27,9%), ComplementNB "
        f"(25,2%).")
    add("- Categorías de intersección:")
    add("")
    add("| Categoría | n | % | accuracy del ensamble |")
    add("|---|---|---|---|")
    for row in diag["interseccion"]:
        add(f"| {row['categoria']} | {row['n']:,} | {row['pct']}% | {row['accuracy_ensamble']} |")
    add("")
    add(f"- Las pérdidas se concentran en las clases medias: 25-34 (10.241), 35-49 (9.067), "
        f"18-24 (8.927).")
    add("- Error por tipo de tweet: `solo_url` **61,1%** (n=7.661), `solo_texto` 44,4% (n=59.221), "
        "`rt_con_comentario` 37,1%, mientras `mencion_alta` 19,2% y `hashtag_alto` 20,4% son los "
        "más fáciles.")
    add("- Error por longitud: monótono de **54,5%** (≤10 caracteres) a **35,1%** (130-141); la "
        "fracción de casos rescatables por algún miembro también cae de 36,9% a 21,1%.")
    add("")
    add("**Interpretación.** El diagnóstico apuntó a **ponderación**, no a falta de información: si "
        "en una de cada cuatro filas algún miembro acierta y el ensamble no, el combinador está "
        "desaprovechando señal. Eso descartó atacar de nuevo la representación en esta fase.")
    add("")
    add("## 4. Hipótesis y experimento (L6)")
    add("")
    add("**H27-H29**: un combinador que pueda **condicionar su confianza al tipo de tweet** recupera "
        "parte de esos casos. Se probaron cuatro variantes con LOFO:")
    add("")
    add("| Variante | macro-F1 | Δ |")
    add("|---|---|---|")
    for row in pond["variantes"]:
        add(f"| {row['variante']} | {row['macro_f1']:.4f} ± {row['std']:.4f} | "
            f"{row['macro_f1'] - pond['variantes'][0]['macro_f1']:+.4f} |")
    add("")
    add("- **La segmentación dura falla** (tipo −0,0007; longitud −0,0015), igual que en la fase "
        "anterior con la compuerta de frontera: partir el problema en subconjuntos pequeños pierde "
        "más de lo que gana.")
    add(f"- **La ponderación suave funciona**: un MLP superficial como meta-modelo alcanza "
        f"**{fixed['meta_mlp']['macro_f1']:.4f} ± {fixed['meta_mlp']['std']:.4f}** frente a "
        f"**{fixed['meta_logreg']['macro_f1']:.4f} ± {fixed['meta_logreg']['std']:.4f}** del "
        f"LogReg, con mejora en **5/5 folds** (de +0,0043 a +0,0080). Un modelo lineal no puede "
        f"expresar 'confía en este miembro cuando el tweet es de este tipo'; el MLP sí.")
    add("")
    add("## 5. Corrección de dos defectos detectados en la auditoría")
    add("")
    add("La generación del candidato reveló una submission degenerada (78.576 filas predichas en "
        "una sola clase). Diagnóstico y corrección:")
    add("")
    add("1. **Desalineación de 6 columnas**: el bloque NB-SVM se anexaba al final del lado test pero "
        "no del lado train, desplazando densas/estructuradas/supervisadas. Corregido; ahora ambos "
        "lados comparten el mismo orden.")
    add("2. **Cuatro features de forma de palabra en cero**: el tokenizador solo aceptaba minúsculas "
        "en la alternativa de palabra, así que las palabras capitalizadas o en mayúsculas se "
        "fragmentaban. Además `shape_alphanumeric_ratio` era cero por definición (el tokenizador "
        "separa dígitos de letras). Corregidos ambos: "
        f"columnas con varianza nula tras el recálculo: "
        f"{fixed['features_estructuradas']['columnas_varianza_nula'] or 'ninguna'}.")
    add("")
    add("**Lección metodológica**: la validación de formato de una submission no basta; hay que "
        "**verificar la distribución de etiquetas predichas**. El archivo degenerado pasaba todos "
        "los checks de estructura.")
    add("")
    add("## 6. Confirmación en el holdout congelado")
    add("")
    add("| Meta-modelo | macro-F1 en el holdout | |")
    add("|---|---|---|")
    for label, payload in hold["holdout"]["resultados"].items():
        add(f"| {label} | {payload['macro_f1']:.4f} | accuracy {payload['accuracy']:.4f} |")
    add("")
    add(f"**Δ = {hold['holdout']['delta']:+.4f}** (criterio: Δ ≥ +0,003) → "
        f"**{'CONFIRMADO' if hold['holdout']['confirmado'] else 'NO confirmado'}**.")
    add("")
    add("## 7. Candidato v11")
    add("")
    candidato = hold["candidato"]
    add("- **Modelo**: ensamble de 14 fuentes + **meta-modelo MLP(64)** sobre 273 features.")
    add(f"- **CV (LOFO)**: {candidato['cv_macro_f1']:.4f} ± {candidato['cv_std']:.4f}.")
    add(f"- **Holdout congelado**: {candidato['holdout_macro_f1']:.4f}.")
    add(f"- **Distribución predicha**: {candidato['distribucion_predicha']}.")
    add("- **LB esperado**: 0,612-0,614 (CV 0,5967 + brecha calibrada de +0,0169 ± 0,0013).")
    add("")
    add("## 8. Líneas evaluadas y descartadas en esta fase")
    add("")
    add("| Línea | Resultado | Decisión |")
    add("|---|---|---|")
    add("| Segmentación dura por tipo de tweet | −0,0007 | cerrada |")
    add("| Segmentación dura por banda de longitud | −0,0015 | cerrada |")
    add("| Features de forma de palabra (antes degeneradas) | corregidas | se conservan |")
    add("| Alineación train/test del meta | bug corregido | — |")
    add("")
    add("## 9. Líneas pendientes (con brecha confirmada en el gap analysis v2)")
    add("")
    add("- **L2 — Residualización por longitud**: separar señal de edad de 'tweet simplemente más "
        "largo'; es la línea con mejor relación coste/información de las que quedan.")
    add("- **L3 — Contenido vs estilo**: representación explícita de estilo (palabras funcionales, "
        "puntuación, estructura) frente a contenido; explicaría por qué las fronteras centrales "
        "resisten.")
    add("- **L4 — Transiciones de carácter**: estadística de transición letra↔dígito↔puntuación.")
    add("- **L7 — NB-SVM por canal**: word, hashtag y mención puros como miembros independientes.")
    add("- **Retrieval train→test**: sigue **REQUIERE_VERIFICACIÓN** de reglas; no implementado.")
    add("")
    add("## 10. Conclusión de la fase")
    add("")
    add("El baseline v09 captura representación dispersa de carácter, canales léxicos con perfiles "
        "supervisados, ordinalidad y diversidad de modelos. El diagnóstico de intersección demostró "
        "que **una cuarta parte de los tweets tiene un miembro acertando y el ensamble fallando**, y "
        "que los rescatadores son sistemáticamente los modelos diversos. La hipótesis derivada —el "
        "combinador debe poder condicionar su confianza al tipo de tweet— se confirmó con un "
        f"**MLP superficial de meta-modelo**: +0,0061 de CV en 5/5 folds y +0,0043 en el holdout "
        f"congelado. La segmentación dura volvió a fallar, lo que refuerza que el problema es de "
        f"**ponderación suave**, no de partición del espacio.")
    (cfg.REPORTS / "advanced_research_v2.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[informe] reports/advanced_research_v2.md ({len(lines)} líneas)")

    # --- Memoria del proyecto ---
    registry = cfg.ROOT / "experiments.csv"
    frame = pd.read_csv(registry)
    nuevas = [
        dict(experiment_id="PH1-L1", fecha="2026-09-24", bloque="5", etapa="Fase avanzada v2",
             pregunta="¿Cuántos tweets tienen algún modelo base acertando mientras el ensamble falla?",
             hipotesis="Si la fracción es alta, el fallo es de ponderación y no de información.",
             features="predicciones OOF de 12 miembros + ensamble", preprocessing="—",
             representacion="diagnóstico de intersección", modelo="—", hiperparametros="{}",
             validacion="LOFO 5 folds agrupado", macro_f1_mean=0.5906, macro_f1_std="",
             weighted_f1_mean="", micro_f1_mean="", accuracy_mean="",
             resultado=f"{diag['ensemble_miss_pct']}% de filas con rescate posible; rescatadores: softmax ordinal 36%, centroides 31%, MLP 31%",
             interpretacion="El combinador desaprovecha señal de los modelos diversos; apunta a ponderación.",
             decision="Reorientar la fase hacia el combinador (L6).", duracion_seg=403,
             artefactos="reports/error_intersection_analysis.csv, experiments/phase1_diagnostico.json"),
        dict(experiment_id="PH2-L6", fecha="2026-09-24", bloque="5", etapa="Fase avanzada v2",
             pregunta="¿Mejora el ensamble si el combinador condiciona su confianza al tipo de tweet?",
             hipotesis="La ponderación suave (MLP) recupera los casos donde un miembro acierta y el ensamble no; la segmentación dura no.",
             features="273 del meta-modelo", preprocessing="—",
             representacion="meta-modelo segmentado vs MLP suave", modelo="LogReg / MLP(64)",
             hiperparametros="{}", validacion="LOFO 5 folds agrupado",
             macro_f1_mean=0.5967, macro_f1_std=0.0040, weighted_f1_mean="", micro_f1_mean="",
             accuracy_mean="",
             resultado="MLP 0,5967 vs LogReg 0,5906 (+0,0061, 5/5 folds); segmentado por tipo -0,0007; por longitud -0,0015",
             interpretacion="La ponderación condicional suave es lo que faltaba; partir el espacio en segmentos pierde información.",
             decision="Adoptar el meta-modelo MLP.", duracion_seg=1639,
             artefactos="experiments/phase2_ponderacion.json"),
        dict(experiment_id="PH3-V11", fecha="2026-09-24", bloque="5", etapa="Fase avanzada v2",
             pregunta="¿Se confirma el candidato v11 fuera del CV de desarrollo?",
             hipotesis="La mejora del combinador debe reproducirse en el holdout congelado.",
             features="273 del meta-modelo", preprocessing="—", representacion="candidato v11",
             modelo="Ensamble de 14 fuentes + meta MLP", hiperparametros="{}",
             validacion="Holdout congelado 15% de clústeres (semilla 2026)",
             macro_f1_mean=hold["candidato"]["holdout_macro_f1"], macro_f1_std="",
             weighted_f1_mean="", micro_f1_mean="", accuracy_mean="",
             resultado=f"holdout: LogReg 0,5981 vs MLP 0,6024 (+0,0043); v09 previo 0,5966",
             interpretacion="Confirmado con margen sobre el criterio de +0,003.",
             decision="Generar submission v11 con el último envío disponible.",
             duracion_seg=hold["candidato"]["segundos"],
             artefactos="submissions/age_submission_v11.csv, experiments/phase3_holdout_v11.json"),
    ]
    for fila in nuevas:
        frame = pd.concat([frame, pd.DataFrame([fila])], ignore_index=True)
    frame.to_csv(registry, index=False)

    decisions = cfg.ROOT / "DECISIONS.md"
    decisions.write_text(decisions.read_text(encoding="utf-8") + """
---

## D-020 — El combinador: segmentación dura falla, ponderación suave funciona
- **Observation**: el diagnóstico de intersección mostró que en el **25,94%** de los tweets algún
  modelo base acierta mientras el ensamble falla, y que los rescatadores son sistemáticamente los
  modelos diversos (softmax ordinal 36%, centroides 31%, MLP 31%). El error depende además del tipo
  de tweet (`solo_url` 61,1%; `mencion_alta` 19,2%) y de la longitud (54,5% → 35,1%).
- **Hypothesis**: el meta-modelo lineal aplica pesos globales y no puede condicionar su confianza al
  contexto del tweet; un combinador que sí pueda hacerlo recuperará parte de esos casos.
- **Experiment**: PH2-L6 — cuatro combinadores con LOFO: LogReg global (referencia), MLP global,
  segmentado por tipo de tweet, segmentado por banda de longitud.
- **Result**: **MLP 0,5967 ± 0,0040 vs LogReg 0,5906 ± 0,0047 (+0,0061, 5/5 folds)**; segmentado
  por tipo −0,0007; por longitud −0,0015. Holdout congelado: **0,5981 → 0,6024 (+0,0043)**.
- **Decision**: adoptar el MLP superficial como meta-modelo (submission v11). Cerrar la
  segmentación dura — es el segundo fracaso de esa estrategia (antes con la compuerta de frontera,
  −0,0036).
- **Reason**: particionar el espacio reduce los datos de cada submodelo y con ello la calidad de la
  ponderación; la red aprende la misma estructura de forma **suave y compartida entre segmentos**,
  que es lo que la evidencia favorece.
- **Evidence**: `experiments/phase2_ponderacion.json`, `phase3_holdout_v11.json`,
  `reports/error_intersection_analysis.csv`.

---

## D-021 — Dos defectos corregidos: alineación del meta y features de forma degeneradas
- **Observation**: la primera submission v11 resultó **degenerada** (78.576 predicciones en `13-17`,
  una sola en `25-34`) pese a pasar todos los checks de formato.
- **Diagnosis**: (1) el bloque NB-SVM se anexaba al final del lado test pero no del lado train,
  desplazando 6 columnas del resto de bloques; (2) cuatro features de forma de palabra eran
  constantes porque el tokenizador solo aceptaba minúsculas, y `shape_alphanumeric_ratio` era cero
  por definición al separar el tokenizador dígitos de letras.
- **Decision**: corregir el orden de concatenación en ambos lados, arreglar el tokenizador y
  recalcular alfanuméricos con una expresión propia; **validar siempre la distribución de etiquetas
  predichas** además del formato.
- **Reason**: la validación estructural es necesaria pero no suficiente; un bug de alineación
  produce archivos formalmente válidos y semánticamente inútiles.
- **Evidence**: `experiments/phase3b_recorregido.json` (0 columnas con varianza nula tras el
  recálculo; MLP 0,5967 confirmado).
""", encoding="utf-8")

    state = cfg.ROOT / "PROJECT_STATE.md"
    text = state.read_text(encoding="utf-8")
    text = text.replace("- **CURRENT BEST CV**: 0,5904 (LOFO) — candidato v10 (v09 + NB-SVM), **generado y no enviado**",
                        "- **CURRENT BEST CV**: **0,5967** (LOFO) — candidato v11 (meta-modelo MLP)")
    text = text.replace("- **CURRENT BEST HOLDOUT**: 0,5966 (frente a 0,5914 del baseline en el mismo holdout)",
                        "- **CURRENT BEST HOLDOUT**: **0,6024** (frente a 0,5981 del LogReg en el mismo holdout)")
    text = text.replace("- **CURRENT MODEL**: ensamble de 13 fuentes + meta-modelo LogReg sobre **267 features**\n  (72 scores + 35 densas + 87 estructuradas + 73 supervisadas)",
                        "- **CURRENT MODEL**: ensamble de **14 fuentes + meta-modelo MLP(64)** sobre **273 features**\n  (78 scores + 35 densas + 87 estructuradas + 73 supervisadas)")
    state.write_text(text, encoding="utf-8")

    runlog = cfg.ROOT / "RUNLOG.md"
    runlog.write_text(runlog.read_text(encoding="utf-8") + """
## 2026-09-24 — Sesión 7 (fase avanzada v2: diagnóstico, ponderación y candidato v11)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 75 | Congelar v09/v10 y gap analysis v2 | `experiments/baseline_v09_frozen/`, `reports/advanced_gap_analysis_v2.md`, `research_backlog_v2.md` | 16 archivos con hash; 22 fenómenos auditados |
| 76 | **L1 — intersección de errores** | `reports/error_intersection_analysis.csv`, `experiments/phase1_diagnostico.json` | **25,94%** de filas rescatables; rescatadores = modelos diversos |
| 77 | **L5 — error por tipo y longitud** | `reports/tables/phase1_error_por_tipo.csv`, `phase1_error_por_longitud.csv` | `solo_url` 61,1% vs `mencion_alta` 19,2%; longitud 54,5% → 35,1% |
| 78 | **L6 — ponderación condicional** | `experiments/phase2_ponderacion.json` | **MLP meta 0,5967 (+0,0061, 5/5 folds)**; segmentación dura negativa |
| 79 | Corrección de dos defectos | `scripts/phase3b_recorregido.py`, `src/advanced_features.py` | 0 columnas degeneradas; alineación del meta corregida |
| 80 | Confirmación y candidato v11 | `experiments/phase3_holdout_v11.json` → `submissions/age_submission_v11.csv` | Holdout **+0,0043** (> criterio +0,003); distribución sana |
""", encoding="utf-8")
    print("[memoria] PROJECT_STATE, DECISIONS (D-020, D-021), RUNLOG y experiments.csv actualizados")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
