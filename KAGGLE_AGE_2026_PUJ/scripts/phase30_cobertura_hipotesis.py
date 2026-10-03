"""FASE 30 — Matriz de cobertura de hipótesis (capa de decisión ordinal y corrección de fronteras).

Objetivo
    Antes de ejecutar nada, dejar constancia verificable de **qué formulaciones ordinales y de
    decisión ya fueron evaluadas** (y con qué resultado) y **cuáles quedan genuinamente abiertas**,
    para no repetir líneas cerradas.

Salida: ``reports/age_hypothesis_coverage.csv`` y ``reports/age_hypothesis_coverage.md``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402

CSV_PATH = cfg.REPORTS / "age_hypothesis_coverage.csv"
MD_PATH = cfg.REPORTS / "age_hypothesis_coverage.md"

FILAS = [
    # id, hipótesis del brief, qué existe ya, evidencia, estado, por qué la formulación nueva difiere
    ("A", "Modelo ordinal acumulativo (cumulative/threshold)",
     "Miembros E-015 (softmax ordinal) y E-021a (Frank & Hall, 5 binarios) en el ensamble; "
     "E-028b (jerarquía gruesa) cerrada",
     "e021_ordinal_centroides.json (E-021a macro-F1 0.5527); experiments.csv E-028b -0.0003",
     "MODELO CERRADO / DECODIFICACIÓN ABIERTA",
     "La decodificación por CDF (fusionar o promediar las funciones de distribución acumuladas y "
     "diferenciar para obtener p_k) nunca se evaluó; tampoco un modelo de umbrales acumulados "
     "ajustado sobre las features del meta."),
    ("B", "Fronteras adyacentes por pares",
     "Miembro E-028a con 5 márgenes por par (accuracy por frontera 0.688-0.818, aporte +0.0006); "
     "routing duro pair-SVM cerrado",
     "e028_ordinal_fino.json; advanced_07_frontera.json (enrutamiento 0.0000, compuerta -0.0036)",
     "MODELO EXISTE / COMBINACIÓN PROBABILÍSTICA ABIERTA",
     "Nunca se convirtieron los 5 márgenes en una distribución de 6 clases por un mecanismo "
     "coherente (Bradley-Terry/LUCE) ni se combinaron con P(V12)."),
    ("C", "Corrección suave de fronteras",
     "Gating duro, segmentación por tipo/longitud y enrutamiento cerrados; calibración de miembros "
     "y blending en logits cerrada; consenso/gating como features cerrado",
     "phase2_ponderacion.json; advanced_07_frontera.json; phase14/phase15/phase29.json",
     "ABIERTA",
     "Las líneas cerradas reentrenan el meta o añaden features; esta redistribuye masa entre clases "
     "adyacentes sobre P(V12) congelada, con coeficientes pequeños y regularización."),
    ("D", "Decodificación por matriz de confusión",
     "La matriz de confusión se analizó (fase 17, fase B) pero nunca se usó como capa de decisión; "
     "no existe decodificación por matriz de transición en el proyecto",
     "grep de 'matriz de transici|confusion.*decod|isotonic' sin resultados en scripts/ y src/",
     "ABIERTA",
     "Mecanismo nuevo: corregir los sesgos sistemáticos de confusión con shrinkage hacia la identidad, "
     "estimada con OOF."),
    ("E", "Regularización ordinal de probabilidades",
     "No existe proyección ni suavizado ordinal de probabilidades",
     "grep 'proyecc|redistribu|isotonic' sin resultados",
     "ABIERTA",
     "Nuevo:(kernel ordinal Toeplitz o proyección unimodal) sobre P(V12) congelada."),
    ("F", "Decisión con coste ordinal",
     "Ajuste de pesos por clase para macro-F1 cerrado (+0.0006, D-011); E[Y] como feature cerrado "
     "(-0.0003, fase G1)",
     "fase19_pesos_por_clase.json; phase16_ordinal_greedy_g1g2.json",
     "PARCIALMENTE ABIERTA",
     "Nunca se minimizó un costo por distancia ordinal como regla de decisión; se mide contra "
     "macro-F1 sin asumir que mejore."),
    ("G", "Edad esperada (E[Y]) y decodificación discreta",
     "E[Y] como feature del meta cerrado; no como regla de desempate local",
     "phase16_ordinal_greedy_g1g2.json (g1_ordinal -0.0003)",
     "ABIERTA (versión desempate)",
     "Aquí E[Y] no alimenta al modelo: decide únicamente empates adyacentes de baja confianza."),
    ("H", "Calibración por baja confianza",
     "Calibración global por miembro/combinador cerrada (D-029)",
     "phase15_calibracion_scores.json",
     "ABIERTA",
     "La cerrada transforma todo el vector con una temperatura global; esta aplica un escalado por "
     "banda de confianza, que sí puede cambiar el argmax."),
    ("I", "Corrector de error muy pequeño",
     "No existe un corrector post-hoc sobre probabilidades OOF congeladas; las variantes de la fase "
     "residual añadían features y reentrenaban el meta",
     "phase14_consenso_familias.json; phase15b_diversidad_subsets.json",
     "ABIERTA",
     "Modelo pequeño (LogReg/Ridge/isotónica) sobre P(V12)+contexto, entrenado solo con OOF."),
    ("Z", "Líneas de la fase residual ya cerradas (no repetir)",
     "Consenso por familia, gating, calibración, diversidad de subconjuntos, estructura v2, limpieza "
     "selectiva, edad implícita, greedy, multivista",
     "phase20_resumen.json: 27 variantes, mejor +0.0018 (vista estructural, no adoptada)",
     "CERRADAS",
     "Ninguna alcanza el gate; la vista estructural queda como evidencia de señal complementaria "
     "pequeña, no como justificación para más features."),
    ("HOLD", "Holdout congelado",
     "25.204 filas, semilla 2026, usado en las fases 3, 8, 19 y 23",
     "phase19_holdout_VistaEstructura.json",
     "AGOTADO",
     "La nueva fase no lo reutiliza: corrected CV es la evidencia primaria."),
]


def main() -> int:
    """Genera la matriz de cobertura de hipótesis en CSV y Markdown."""
    import csv

    with CSV_PATH.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["id", "hipotesis", "que_existe_ya", "evidencia", "estado",
                           "por_que_la_formulacion_nueva_difiere"])
        for fila in FILAS:
            escritor.writerow(fila)
    lineas = ["# Matriz de cobertura de hipótesis — capa de decisión ordinal", "",
              "Generada por `scripts/phase30_cobertura_hipotesis.py` antes de ejecutar la fase.", "",
              "| # | Hipótesis | Qué existe ya | Evidencia | Estado | Por qué la nueva difiere |",
              "|---|---|---|---|---|---|"]
    for identificador, hipotesis, existe, evidencia, estado, motivo in FILAS:
        lineas.append(f"| {identificador} | {hipotesis} | {existe} | `{evidencia}` | **{estado}** | "
                      f"{motivo} |")
    abiertas = [fila for fila in FILAS if "ABIERTA" in fila[4]]
    lineas += ["", f"**Líneas genuinamente abiertas: {len(abiertas)}** "
              f"({', '.join(fila[0] for fila in abiertas)}).", "",
              "Criterio de la nueva fase: cada experimento debe responder a una hipótesis **nueva** "
              "diferente de las marcadas como cerradas, ajustarse solo con OOF y superar "
              "Δ ≥ +0,003 con ≥4/5 folds."]
    MD_PATH.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"cobertura: {CSV_PATH.name} y {MD_PATH.name} | líneas abiertas: {len(abiertas)}")
    for fila in abiertas:
        print(f"  {fila[0]}: {fila[4]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
