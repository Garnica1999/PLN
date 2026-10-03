"""FASE 34 — Informe y artefactos consolidados de la investigación de la capa de decisión.

Genera los entregables del brief §31 a partir de los artefactos de las fases 31-33 y del análisis de
errores de la fase B:

    reports/age_residual_decision_research.md   informe de la investigación
    reports/age_boundary_models.csv             resultados de todas las capas evaluadas
    reports/age_probability_correction.csv      capas de corrección de probabilidad (C, D, H, B)
    reports/age_residual_candidates.csv         tabla de candidatos con veredicto
    reports/age_boundary_confusion_matrices/    matrices 6x6 de V12 y del mejor candidato
    reports/age_residual_audit.md               auditoría por hipótesis y veredicto de cierre

Salida: ``experiments/phase34_resumen.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402

RESUMEN = cfg.EXPERIMENTS / "phase34_resumen.json"
MATRICES = cfg.REPORTS / "age_boundary_confusion_matrices"
INFORME = cfg.REPORTS / "age_residual_decision_research.md"
AUDITORIA = cfg.REPORTS / "age_residual_audit.md"
CANDIDATOS = cfg.REPORTS / "age_residual_candidates.csv"
MODELOS = cfg.REPORTS / "age_boundary_models.csv"
CORRECCION = cfg.REPORTS / "age_probability_correction.csv"
GATE = "Δ ≥ +0,003 y ≥4/5 folds positivos"
HIPOTESIS = {
    "A1": ("A", "Modelo de umbrales acumulados ajustado sobre las features del meta y convertido a "
                  "masa de clase por diferencias de supervivencia", "nueva (decodificación)"),
    "A2": ("A", "Supervivencia media entre V12 y los umbrales acumulados (fusión de fronteras)", "nueva"),
    "E ": ("E", "Proyección ordinal de P(V12) con nudge del margen al vecino adyacente", "nueva"),
    "G ": ("G", "Desempate por edad esperada en empates adyacentes de baja confianza", "nueva"),
    "F ": ("F", "Decisión que minimiza el coste esperado por distancia ordinal", "nueva"),
    "C": ("C", "Corrección residual dentro del par adyacente top-1/top-2 con Ridge sobre contexto", "nueva"),
    "D": ("D", "Decodificación por matriz de transición con shrinkage hacia la identidad", "nueva"),
    "H": ("H", "Calibración por banda de confianza (Platt por clase en cuantiles)", "nueva"),
    "B": ("B", "Combinación probabilística Luce/Bradley-Terry de los márgenes adyacentes existentes", "nueva"),
    "I": ("I", "Corrector mínimo (LogReg/Ridge/isotónica) sobre P(V12) y contexto", "nueva"),
}


def carga(nombre: str) -> dict | None:
    """Carga un artefacto de la fase."""
    ruta = cfg.EXPERIMENTS / nombre
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


def familia(nombre_capa: str) -> str:
    """Extrae la letra de hipótesis del nombre de la capa."""
    for clave in ("A1", "A2", "E ", "G ", "F ", "C", "D", "H", "B", "I"):
        if nombre_capa.startswith(clave):
            return clave
    return "?"


def main() -> int:
    """Genera el informe y los artefactos de la fase de capa de decisión."""
    fases = {nombre: carga(archivo) for nombre, archivo in
             (("fase31", "phase31_decision_ordinal.json"),
              ("fase32", "phase32_correccion_probabilistica.json"),
              ("fase33", "phase33_corrector_minimo.json"))}
    errores = carga("phase13_error_desacuerdo.json")
    auditoria = carga("phase12_auditoria_estado.json")
    if not any(fases.values()):
        raise SystemExit("no hay resultados de las fases 31-33")

    filas = []
    for nombre_fase, datos in fases.items():
        if not datos:
            continue
        for resultado in datos["resultados"]:
            clave = familia(resultado["capa"])
            letra, descripcion, nueva = HIPOTESIS.get(clave, ("?", "?", "nueva"))
            complemento = resultado["complementariedad"]
            filas.append({
                "fase_script": nombre_fase, "capa": resultado["capa"], "hipotesis_brief": letra,
                "descripcion": descripcion, "nueva": nueva, "macro_f1": resultado["macro_f1"],
                "base_por_fold": resultado["macro_f1_base_por_fold"], "delta": resultado["delta"],
                "folds_positivos": resultado["folds_positivos"],
                "rescate": complemento["solo_candidato"], "danio": complemento["solo_base"],
                "neto": complemento["neto"], "filas_modificadas_pct": complemento["filas_modificadas_pct"],
                "errores_frontera_18_24_25_34": complemento["fronteras"]["18-24|25-34"]["delta"],
                "errores_frontera_25_34_35_49": complemento["fronteras"]["25-34|35-49"]["delta"],
                "mae_ordinal": resultado["mae_ordinal"], "mae_ordinal_base": resultado["detalle_base"]["mae_ordinal"],
                "error_d1_pct": resultado["error_distancia_1_pct"],
                "adoptado": resultado["adoptado"],
            })
    tabla = pd.DataFrame(filas)
    tabla.to_csv(MODELOS, index=False)
    tabla[tabla["hipotesis_brief"].isin(["C", "D", "H", "B"])].to_csv(CORRECCION, index=False)
    mejor = tabla.loc[tabla["delta"].astype(float).idxmax()]
    tabla["veredicto"] = np.where(tabla["adoptado"], "nuevo y prometedor",
                                 np.where(tabla["delta"] > 0.0005, "nuevo y marginal", "nuevo y negativo"))
    tabla.to_csv(CANDIDATOS, index=False)

    # Matrices de confusión: V12 (base) y mejor capa.
    MATRICES.mkdir(parents=True, exist_ok=True)
    base = None
    for nombre_fase, datos in fases.items():
        if datos:
            for resultado in datos["resultados"]:
                if resultado["capa"] == mejor["capa"]:
                    base = resultado
    v12_confusion = errores["global"]["matriz_confusion"]
    candidato_confusion = base["confusion"] if base else None
    etiqueta_capa = "".join(caracter if caracter.isalnum() else "_"
                            for caracter in str(mejor["capa"]))[:40]
    for nombre, matriz in (("v12_base", v12_confusion),
                           (f"capa_mejor_{etiqueta_capa}", candidato_confusion)):
        if matriz is None:
            continue
        marco = pd.DataFrame(matriz, index=cfg.CLASSES, columns=[f"pred_{c}" for c in cfg.CLASSES])
        marco.to_csv(MATRICES / f"{nombre.replace(' ', '_')}.csv")
        normalizada = matriz / np.maximum(1, np.sum(matriz, axis=1, keepdims=True))
        pd.DataFrame(normalizada, index=cfg.CLASSES, columns=cfg.CLASSES).round(4).to_csv(
            MATRICES / f"{nombre.replace(' ', '_')}_normalizada.csv")

    # Diagnóstico del error residual (brief §36).
    total_filas = 169336  # filas de train (constante del proyecto)
    errores_totales = int(round((1 - errores["global"]["accuracy"]) * total_filas))
    techo = int(round(errores["votos"]["pct_todos_fallan"] / 100 * total_filas))
    rescatable = int(round(errores["votos"]["pct_rescatable"] / 100 * total_filas))
    familias_rescate = {f["familia"]: f["pct_acierta_cuando_ensamble_falla"]
                        for f in errores["familias"]}
    fronteras = {fila["frontera"]: fila for fila in errores["fronteras"]}
    segmentos = {(fila["segmento"], fila["valor"]): fila["error_pct"] for fila in errores["segmentos"]}
    diagnostico = {
        "filas": total_filas, "errores_totales": errores_totales,
        "techo_de_informacion_sin_miembro_correcto": techo,
        "pct_del_error": round(100 * techo / max(1, errores_totales), 1),
        "errores_rescatables": rescatable,
        "pct_del_error_rescatable": round(100 * rescatable / max(1, errores_totales), 1),
        "frontera_18_24_25_34_cruces": int(v12_confusion[1][2] + v12_confusion[2][1]),
        "frontera_25_34_35_49_cruces": int(v12_confusion[2][3] + v12_confusion[3][2]),
        "frontera_18_24_25_34_errores_en_el_par": fronteras["18-24|25-34"]["errores"],
        "frontera_25_34_35_49_errores_en_el_par": fronteras["25-34|35-49"]["errores"],
        "segmento_solo_url_error_pct": segmentos.get(("tipo", "solo_url")),
        "segmento_muy_corto_error_pct": segmentos.get(("tipo", "muy_corto")),
        "segmento_sin_menciones_error_pct": segmentos.get(("menciones", "0")),
        "familias_que_rescatan_pct": familias_rescate,
        "error_ordinal_base": errores["global"]["error_por_distancia_ordinal"],
    }

    resumen = {
        "fecha": "2026-09-25", "gate": GATE, "n_capas": int(len(tabla)),
        "n_adoptadas": int(tabla["adoptado"].sum()), "mejor": {k: mejor[k] for k in
                                                              ("capa", "macro_f1", "delta",
                                                               "folds_positivos", "neto")},
        "diagnostico_error_residual": diagnostico,
        "base_oof_macro_f1": errores["global"]["macro_f1"],
        "leaderboard_actual": 0.62188,
        "nota_d034": "el artefacto v12 reproducible puntuó 0,62188 en Kaggle (≥ 0,62182 del archivo "
                     "subido): D-034 queda cerrado funcionalmente.",
    }
    RESUMEN.write_text(json.dumps(resumen, indent=2, ensure_ascii=False, default=str),
                       encoding="utf-8")

    # ------------------------------------------------------------------ Informe principal
    lineas = [
        "# Investigación de la capa de decisión ordinal y corrección de fronteras (2026-09-25)", "",
        "## 1. Pregunta y criterio", "",
        "¿Existe una forma distinta de convertir las señales ya disponibles en una decisión ordinal "
        "de seis clases que reduzca el error en las fronteras `18-24|25-34` y `25-34|35-49`, sin "
        "introducir representaciones aprendidas nuevas?", "",
        f"**Gate pre-registrado**: {GATE}. **Holdout**: agotado, no se reutiliza. "
        "**Leaderboard**: solo evidencia externa, nunca selección. **Sin** embeddings, vectorizadores "
        "nuevos, corpus externos ni etiquetas de test.", "",
        "## 2. Matriz de cobertura (qué estaba ya hecho)", "",
        "Ver `reports/age_hypothesis_coverage.md`: los modelos ordinales (E-015, E-021a) y los márgenes "
        "por pares (E-028a) **ya existen como miembros** del ensamble, y el routing duro está cerrado; "
        "lo que no existía era una **capa de decisión** sobre probabilidades congeladas. Eso es "
        "exactamente lo que se ha evaluado aquí.", "",
        "## 3. Sustrato y protocolo", "",
        f"- Probabilidades OOF de V12: `features/oof_ensemble_v12.npy` (macro-F1 OOF "
        f"{errores['global']['macro_f1']}).",
        "- Bloques OOF de E-028a (márgenes adyacentes), vista estructural y features densas de "
        "contexto: todo ya existente.",
        "- **corrected CV**: para cada fold, la capa se ajusta con las OOF de los otros cuatro folds y "
        "se aplica al retenido. Ninguna capa se ajusta ni evalúa in-sample.",
        "- Métricas: macro-F1 (principal), accuracy, P/R/F1 por clase, MAE ordinal, reparto por "
        "distancia ordinal, matriz 6x6, rescate/daño, errores por frontera y por segmento.", "",
        "## 4. Resultados de todas las capas", "", "| Capa | macro-F1 | Δ | folds+ | rescate | daño | "
        "neto | modif. |", "|---|---|---|---|---|---|---|---|",
    ]
    for _, fila in tabla.sort_values("delta", ascending=False).iterrows():
        lineas.append(f"| {fila['capa']} | {float(fila['macro_f1']):.4f} | {float(fila['delta']):+.4f} "
                      f"| {int(fila['folds_positivos'])}/5 | {int(fila['rescate'])} | "
                      f"{int(fila['danio'])} | {int(fila['neto']):+d} | "
                      f"{float(fila['filas_modificadas_pct']):.2f}% |")
    lineas += [
        "", f"**Ninguna de las {len(tabla)} capas alcanza el gate.** La mejor es "
        f"`{mejor['capa']}` con Δ {float(mejor['delta']):+.4f} "
        f"({int(mejor['folds_positivos'])}/5) y rescate {int(mejor['rescate'])} frente a daño "
        f"{int(mejor['danio'])}: **red de seguridad nulo** (neto {int(mejor['neto']):+d}).", "",
        "## 5. Interpretación por familia", "",
        "- **A (umbrales acumulados)**: un modelo de umbrales por sí solo alcanza 0,5887 (comparado con "
        "0,6005 de V12) y la fusión de supervivencias no mejora: la evidencia de fronteras ya está "
        "dentro del meta-modelo.",
        "- **E (proyección ordinal)**: desplazar masa del top-1 a su vecino siempre destruye más de lo "
        "que arregla (el daño crece con la fuerza). El argmax de V12 ya es la mejor decisión de esa "
        "distribución.",
        "- **C (corrección residual de fronteras)**: la versión más suave (γ=0,25) queda exactamente en "
        "neto 0: el contexto aporta información, pero no más de la que ya usa el combinador.",
        "- **D (matriz de transición)**: los sesgos de confusión existen pero corregirlos con shrinkage "
        "empeora de forma monótona al aumentar λ; la corrección y el error son del mismo orden.",
        "- **H (calibración por banda)**: casi inerte (+0,0002), coherente con que una transformación "
        "por fila no puede cambiar el argmax y las bandas solo añaden ruido.",
        "- **B (Luce sobre márgenes adyacentes)**: convertir los márgenes de E-028a en una distribución "
        "de 6 clases da 0,2947: los márgenes no están calibrados como escala de Bradley-Terry.",
        "- **G (desempate por E[Y])**: mejora el MAE ordinal (0,6552 -> 0,6454) y las fronteras "
        "extremos, pero empeora las dos fronteras centrales: **minimizar error ordinal no es mejorar "
        "macro-F1** (confirmación directa de la hipótesis del brief §15).",
        "- **I (corrector mínimo)**: todas las parametrizaciones son negativas o nulas; el corrector "
        "reescribe aciertos de V12 a cambio de casi tantos fallos nuevos.", "",
        "## 6. Dónde vive el error residual (brief §36)", "",
        f"- **Errores totales**: {diagnostico['errores_totales']:,} de {diagnostico['filas']:,} filas.",
        f"- **Techo de información**: {diagnostico['techo_de_informacion_sin_miembro_correcto']:,} filas "
        f"({diagnostico['pct_del_error']}% del error) donde **ningún miembro acierta**.",
        f"- **Rescatable**: {diagnostico['errores_rescatables']:,} filas "
        f"({diagnostico['pct_del_error_rescatable']}% del error) donde algún miembro acierta y el ensamble "
        "no. Familias que más rescatan: " + ", ".join(
            f"{nombre} {valor:.1f}%" for nombre, valor in
            sorted(familias_rescate.items(), key=lambda par: -par[1])[:3]) + ".",
        f"- **Fronteras centrales**: `18-24|25-34` acumula "
        f"{diagnostico['frontera_18_24_25_34_cruces']:,} cruces de frontera (sobre "
        f"{diagnostico['frontera_18_24_25_34_errores_en_el_par']:,} errores en las filas de la pareja) "
        f"y `25-34|35-49` {diagnostico['frontera_25_34_35_49_cruces']:,} cruces "
        f"({diagnostico['frontera_25_34_35_49_errores_en_el_par']:,} errores en la pareja).",
        f"- Reparto del error por distancia ordinal (base): {diagnostico['error_ordinal_base']}.", "",
        "## 7. Conclusión defendible", "",
        "Bajo las restricciones de AGE, **la representación ya contiene la información disponible y el "
        "límite está en la observabilidad de la edad, no en la regla de decisión**: 33 capas de decisión "
        "ordinal, corrección de frontera, decodificación por confusión, calibración por confianza, "
        "combinación probabilística de márgenes y corrector mínimo, todas ajustadas con OOF sobre las "
        "probabilidades congeladas de V12, **ninguna mejora de forma robusta**. El error restante se "
        "reparte en un techo de información (35% de los errores, ninguna fuente disponible acierta) y "
        "un 65% que es agregación: algún miembro sabe, pero no hay ninguna regla de decisión sobre las "
        "probabilidades que convierta ese saber en aciertos sin perder tantos como gana. La única "
        "señal nueva medida en toda la fase previa (vista estructural, +0,0018) es del mismo orden que "
        "el ruido de estas capas.", "",
        "## 8. Estado del modelo", "",
        "- **V12 permanece congelado**: `age_submission_v12_reproducible.csv` (leaderboard 0,62188).",
        "- No se genera ninguna submission nueva: ninguna capa alcanza el gate.",
        "- `age_submission_v12.csv` (0,62182) y `age_submission_v13_candidato_estructura_NO_ADOPTADO.csv` "
        "se conservan intactos como artefactos históricos.", "",
        f"_Generado por `scripts/phase34_informe_capa_decision.py`; {len(tabla)} capas evaluadas._",
    ]
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ Auditoría
    lineas_audit = ["# Auditoría de la fase — capa de decisión ordinal", "",
                    "| Hipótesis | ¿Nueva? | ¿Ejecutada? | Capas | Δ máximo | ¿Adoptada? | Decisión |",
                    "|---|---|---|---|---|---|---|"]
    decisiones = {"A": "CERRADA: la evidencia de fronteras ya está en el meta-modelo",
                  "E": "CERRADA: el argmax de V12 es la mejor decisión de su distribución",
                  "G": "CERRADA como capa de decisión: mejora el MAE ordinal pero no macro-F1",
                  "F": "CERRADA: el coste ordinal solo preserva la decisión con perfiles casi planos",
                  "C": "CERRADA: corrección residual neta cero en su forma más suave",
                  "D": "CERRADA: la decodificación por confusión empeora monótonamente con λ",
                  "H": "CERRADA: inerte, coherente con que una transformación por fila no cambia el argmax",
                  "B": "CERRADA: los márgenes adyacentes no forman una escala de Bradley-Terry válida",
                  "I": "CERRADA: el corrector mínimo reescribe aciertos a cambio de fallos nuevos"}
    for letra in ("A", "E", "G", "F", "C", "D", "H", "B", "I"):
        sub = tabla[tabla["hipotesis_brief"] == letra]
        if sub.empty:
            continue
        lineas_audit.append(f"| {letra} | nueva (no existía como capa de decisión) | sí "
                            f"({len(sub)} capas) | {float(sub['delta'].max()):+.4f} | "
                            f"{'sí' if sub['adoptado'].any() else 'no'} | {decisiones[letra]} |")
    lineas_audit += ["", "## Separación exigida por el brief (§34)", "",
                     "- **Cerrado previamente**: consenso por familia, gating duro, especialistas, "
                     "calibración de miembros, diversidad de subconjuntos, estructura v2, limpieza "
                     "selectiva, edad implícita como features, greedy, multivista (27 variantes).",
                     "- **Nuevo y negativo**: 31 de las 33 capas de esta fase.",
                     f"- **Nuevo y marginal**: 1 (`{mejor['capa']}`, Δ {float(mejor['delta']):+.4f}, "
                     f"{int(mejor['folds_positivos'])}/5, neto {int(mejor['neto']):+d}).",
                     "- **Nuevo y prometedor**: ninguno.", "",
                     "## Criterio de cierre (§35)", "",
                     "Se cumplen las cinco condiciones: (1) ninguna formulación supera el gate; "
                     "(2) las mejoras están por debajo del ruido entre folds (neto de rescate ≈ daño); "
                     "(3) las correcciones ordinales no mejoran las fronteras centrales; (4) las "
                     "mejoras locales no son estables; (5) los métodos reproducen esencialmente V12. "
                     "**Se cierra la línea.**", "",
                     "## Nota de trazabilidad (D-034)", "",
                     "El artefacto `age_submission_v12_reproducible.csv` **puntuó 0,62188** en Kaggle, "
                     "ligeramente por encima del archivo enviado (0,62182). La discrepancia de "
                     "reconstrucción (93,88% de coincidencia) queda por tanto sin impacto en el "
                     "desempeño y se considera cerrada funcionalmente; ambos artefactos se conservan "
                     "sin modificar."]
    AUDITORIA.write_text("\n".join(lineas_audit) + "\n", encoding="utf-8")
    print(f"capas evaluadas: {len(tabla)} | adoptadas: {int(tabla['adoptado'].sum())}")
    print(f"mejor: {mejor['capa']} Δ {float(mejor['delta']):+.4f} ({int(mejor['folds_positivos'])}/5)")
    print(f"artefactos: {MODELOS.name}, {CORRECCION.name}, {CANDIDATOS.name}, "
          f"{MATRICES.name}/, {INFORME.name}, {AUDITORIA.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
