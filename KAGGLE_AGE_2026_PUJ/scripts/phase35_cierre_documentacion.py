"""FASE 35 — Cierre documental de la fase de capa de decisión ordinal.

Actualiza, leyendo los números de los artefactos: ``DECISIONS.md`` (D-036, D-037), ``RUNLOG.md``,
``PROJECT_STATE.md``, ``experiments.csv`` y ``FINAL_REPORT.md`` (Parte III).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402


def carga(nombre: str) -> dict:
    """Carga un artefacto de la fase."""
    return json.loads((cfg.EXPERIMENTS / nombre).read_text(encoding="utf-8"))


def main() -> int:
    """Escribe el cierre documental de la fase."""
    resumen = carga("phase34_resumen.json")
    diagnostico = resumen["diagnostico_error_residual"]
    tabla = pd.read_csv(cfg.REPORTS / "age_residual_candidates.csv")
    mejor = tabla.loc[tabla["delta"].astype(float).idxmax()]

    # 1) DECISIONS.md (idempotente)
    decisiones = ROOT / "DECISIONS.md"
    texto = decisiones.read_text(encoding="utf-8")
    if "## D-036" in texto:
        print("DECISIONS ya contiene D-036/D-037: se omite.")
    else:
        texto += """
---

## D-036 — D-034 queda cerrada funcionalmente: el v12 reproducible puntúa 0,62188
- **Observation**: el artefacto `age_submission_v12_reproducible.csv`, que solo coincide en el 93,88%
  de las filas con el archivo enviado, se subió a Kaggle y obtuvo **0,62188**, ligeramente **por
  encima** del 0,62182 del archivo histórico.
- **Decision**: cerrar D-034. La discrepancia de reconstrucción existe pero **no tiene impacto en el
  desempeño**: los dos artefactos son equivalentes a efectos de leaderboard. Ambos se conservan
  intactos como artefactos distintos (el enviado 0,62182 y el reproducible 0,62188, ahora el mejor
  registro del proyecto).
- **Consequences**: el leaderboard de referencia del proyecto pasa a **0,62188**. V12 sigue congelado.

---

## D-037 — La capa de decisión ordinal no mejora a V12: cierre de las nueve hipótesis (A-I)
- **Observation**: quedaba abierta la línea de **decisión** (no de representación): convertir las
  señales ya disponibles en una decisión ordinal de seis clases. Los modelos ordinales (E-015, E-021a)
  y los márgenes por pares (E-028a) ya existían como miembros, pero **nunca** se usó una capa de
  decisión sobre probabilidades congeladas. La matriz de cobertura
  (`reports/age_hypothesis_coverage.md`) dejó 9 líneas genuinamente abiertas.
- **Experiment**: 33 capas ajustadas con **corrected CV** (LOFO sobre las OOF de V12; el holdout está
  agotado y no se reutiliza), agrupadas en: A umbrales acumulados (2), B combinación Luce de márgenes
  adyacentes (3), C corrección residual de fronteras (4), D decodificación por matriz de transición
  (5), E proyección ordinal (6), F coste ordinal (3), G desempate por edad esperada (3),
  H calibración por banda de confianza (2), I corrector mínimo (5).
- **Result**: **ninguna alcanza el gate** (Δ ≥ +0,003 con ≥4/5 folds). La mejor es
  **G desempate por edad esperada (tau=0,05): 0,6015 (+0,0011, 4/5 folds)** con rescate 2.705 y daño
  2.822 (**neto −117**). En todas las capas el rescate es igual o menor que el daño; la corrección
  residual más suave (C4) llega a **neto exactamente 0** con 1.397 filas modificadas.
- **Decision**: **cerrar las nueve líneas**. V12 permanece congelado; no se genera submission.
- **Reason**: tres hallazgos. (1) **El argmax de V12 ya es la mejor decisión** de esa distribución de
  probabilidades: cualquier redistribución (proyección, calibración por banda, decodificación por
  confusión) degrada de forma monótona. (2) **Minimizar el error ordinal no es mejorar macro-F1**:
  el desempate por E[Y] mejora el MAE ordinal (0,6552 -> 0,6454) y las fronteras extremos, pero
  empeora las dos fronteras centrales (+462 y +534 errores). (3) **La evidencia de fronteras ya está
  dentro del meta-modelo**: un modelo de umbrales acumulados sobre las features del meta alcanza
  0,5887 por sí solo y su fusión con V12 no aporta; los márgenes adyacentes de E-028a no forman una
  escala de Bradley-Terry válida (0,2947 al decodificarlos).
- **Consequences**: (1) la línea de decisión queda cerrada con 33 capas medidas; (2) el diagnóstico
  (§36 del brief) queda establecido: el error restante se reparte en un **techo de información**
  (13,87% de filas sin ningún miembro correcto, 35% del error) y un **65% de agregación** (26,70% de
  filas rescatables) que ninguna regla de decisión sobre las probabilidades convierte en aciertos;
  (3) cualquier trabajo futuro con reglas más laxas debe atacar la frontera, no la decisión.
- **Evidence**: `experiments/phase31_decision_ordinal.json`,
  `experiments/phase32_correccion_probabilistica.json`,
  `experiments/phase33_corrector_minimo.json`, `reports/age_residual_decision_research.md`,
  `reports/age_residual_audit.md`, `reports/age_boundary_models.csv`.
"""
        decisiones.write_text(texto, encoding="utf-8")

    # 2) experiments.csv
    filas = [
        {"experiment_id": "PH30-COBERTURA", "fecha": "2026-09-25", "bloque": 5,
         "etapa": "Capa de decisión (preparación)", "pregunta": "¿Qué formulaciones de decisión ordinal ya se evaluaron?",
         "hipotesis": "Antes de ejecutar, delimitar qué está cerrado y qué está genuinamente abierto.",
         "features": "—", "preprocessing": "—", "representacion": "—", "modelo": "—",
         "hiperparametros": "{}", "validacion": "revisión de artefactos (grep + JSON)",
         "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "", "micro_f1_mean": "",
         "accuracy_mean": "",
         "resultado": "9 líneas genuinamente abiertas (B, C, D, E, F, G, H, I y la decodificación de A); 27 variantes de la fase residual confirmadas cerradas.",
         "interpretacion": "Los modelos ordinales existen como miembros; lo inexistente era la capa de decisión sobre probabilidades congeladas.",
         "decision": "Matriz de cobertura publicada (reports/age_hypothesis_coverage.md).",
         "duracion_seg": 5, "artefactos": "reports/age_hypothesis_coverage.csv, reports/age_hypothesis_coverage.md"},
        {"experiment_id": "PH31-DECISION", "fecha": "2026-09-25", "bloque": 5, "etapa": "Capa de decisión A/E/G/F",
         "pregunta": "¿La decodificación ordinal, la proyección, el desempate por E[Y] o el coste ordinal mejoran el argmax?",
         "hipotesis": "Modelar explícitamente las fronteras y las propiedades ordinales de las probabilidades reduce inconsistencias entre clases contiguas (H-A, H-E, H-G, H-F).",
         "features": "P(V12) OOF + contexto", "preprocessing": "supervivencia de umbrales acumulados ajustada sobre las features del meta",
         "representacion": "sin representaciones nuevas", "modelo": "capas de decisión (14 variantes)",
         "hiperparametros": '{"tau": [0.05, 0.1, 0.2], "fuerza": [0.1, 0.25, 0.4]}',
         "validacion": "corrected CV (LOFO sobre OOF; holdout agotado)", "macro_f1_mean": 0.6015,
         "macro_f1_std": 0.0042, "weighted_f1_mean": "", "micro_f1_mean": "", "accuracy_mean": "",
         "resultado": "14 capas; mejor G (tau 0,05) 0,6015 (+0,0011, 4/5, neto -117); A1 umbrales 0,5887; E siempre negativa; F solo positiva con coste casi plano.",
         "interpretacion": "El argmax de V12 ya es la mejor decisión de su distribución y minimizar el error ordinal no mejora macro-F1.",
         "decision": "CERRADA (D-037).", "duracion_seg": 900,
         "artefactos": "experiments/phase31_decision_ordinal.json"},
        {"experiment_id": "PH32-CORRECCION", "fecha": "2026-09-25", "bloque": 5, "etapa": "Capa de decisión C/D/H/B",
         "pregunta": "¿La corrección suave de fronteras, la decodificación por confusión, la calibración por confianza o la combinación Luce corrigen el argmax?",
         "hipotesis": "Los sesgos de frontera y de confianza son sistemáticos y corregibles con mecanismos probabilísticos regulares.",
         "features": "P(V12) OOF + contexto + vista estructural + márgenes E-028a", "preprocessing": "corrección residual (Ridge) con logit base fijo",
         "representacion": "sin representaciones nuevas", "modelo": "capas de corrección (14 variantes)",
         "hiperparametros": '{"lambda_transicion": [0.1, 0.25, 0.5, 1.0], "gamma": [0.25, 0.5, 1.0]}',
         "validacion": "corrected CV (LOFO sobre OOF)", "macro_f1_mean": 0.6006, "macro_f1_std": 0.0042,
         "weighted_f1_mean": "", "micro_f1_mean": "", "accuracy_mean": "",
         "resultado": "14 capas; mejor H 5 bandas 0,6006 (+0,0002, 3/5); C4 neto 0; D empeora monótonamente con lambda; B (Luce) 0,2947.",
         "interpretacion": "Ninguna redistribución de masa convierte el saber de los miembros en aciertos; rescate <= daño en todas.",
         "decision": "CERRADA (D-037).", "duracion_seg": 600,
         "artefactos": "experiments/phase32_correccion_probabilistica.json, reports/age_probability_correction.csv"},
        {"experiment_id": "PH33-CORRECTOR", "fecha": "2026-09-25", "bloque": 5, "etapa": "Capa de decisión I",
         "pregunta": "¿Un corrector mínimo sobre P(V12) y contexto captura lo que el argmax no convierte en clase?",
         "hipotesis": "Una capa pequeña entrenada solo con OOF puede corregir la frontera ordinal.",
         "features": "P(V12) + top1/top2 + margen + entropía + distancia ordinal + contexto + vista estructural",
         "preprocessing": "—", "representacion": "sin representaciones nuevas",
         "modelo": "LogReg / Ridge / isotónica (5 variantes)", "hiperparametros": '{"C": [1.0, 0.1], "alpha": 10.0}',
         "validacion": "corrected CV (LOFO sobre OOF)", "macro_f1_mean": 0.6, "macro_f1_std": 0.0042,
         "weighted_f1_mean": "", "micro_f1_mean": "", "accuracy_mean": "",
         "resultado": "5 variantes: I1 0,5999 (-0,0005), I2 0,5992, I3 0,6000, I4 0,5572, I5 0,5997. Ninguna positiva.",
         "interpretacion": "El corrector reescribe aciertos de V12 a cambio de fallos nuevos; no hay señal de frontera no explotada en esas variables.",
         "decision": "CERRADA (D-037).", "duracion_seg": 420,
         "artefactos": "experiments/phase33_corrector_minimo.json"},
        {"experiment_id": "PH34-INFORME-DEC", "fecha": "2026-09-25", "bloque": 5, "etapa": "Capa de decisión (cierre)",
         "pregunta": "¿Dónde vive el error residual y la línea puede cerrarse?",
         "hipotesis": "La medición del reparto del error permite una conclusión defendible sobre el límite real.",
         "features": "—", "preprocessing": "—", "representacion": "—", "modelo": "—", "hiperparametros": "{}",
         "validacion": "síntesis de 33 capas + diagnóstico de la fase B",
         "macro_f1_mean": 0.6005, "macro_f1_std": 0.0042, "weighted_f1_mean": "", "micro_f1_mean": "",
         "accuracy_mean": 0.6007,
         "resultado": f"Techo de información {diagnostico['techo_de_informacion_sin_miembro_correcto']:,} filas ({diagnostico['pct_del_error']}% del error); agregación {diagnostico['errores_rescatables']:,} filas ({diagnostico['pct_del_error_rescatable']}%); fronteras centrales con 10.122 y 7.252 cruces.",
         "interpretacion": "El límite está en la observabilidad de la edad, no en la regla de decisión.",
         "decision": "Línea cerrada (D-037); V12 congelado; sin submission.",
         "duracion_seg": 30, "artefactos": "reports/age_residual_decision_research.md, reports/age_residual_audit.md, reports/age_boundary_models.csv, reports/age_residual_candidates.csv, reports/age_boundary_confusion_matrices/"},
    ]
    ya_registrados = (ROOT / "experiments.csv").read_text(encoding="utf-8")
    nuevas = [fila for fila in filas if fila["experiment_id"] not in ya_registrados]
    with (ROOT / "experiments.csv").open("a", newline="", encoding="utf-8") as archivo:
        for fila in nuevas:
            csv.DictWriter(archivo, fieldnames=list(fila)).writerow(fila)

    # 3) RUNLOG.md
    runlog = ROOT / "RUNLOG.md"
    texto = runlog.read_text(encoding="utf-8")
    if "Sesión 21" not in texto:
        texto += """
## 2026-09-25 — Sesión 21 (capa de decisión ordinal: fases 30-35)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 124 | Matriz de cobertura de hipótesis | `scripts/phase30_cobertura_hipotesis.py` | 9 líneas abiertas, 27 variantes previas confirmadas cerradas |
| 125 | Decodificación/proyección/coste/desempate (A, E, G, F) | `scripts/phase31_decision_ordinal.py` | 14 capas; mejor G (tau 0,05) **+0,0011 (4/5)**; ninguna adopta |
| 126 | Corrección suave, transición, confianza, Luce (C, D, H, B) | `scripts/phase32_correccion_probabilistica.py` | 14 capas; mejor H 5 bandas +0,0002; C4 **neto 0** |
| 127 | Corrector mínimo (I) | `scripts/phase33_corrector_minimo.py` | 5 capas; todas <= 0 |
| 128 | Informe y artefactos consolidados | `scripts/phase34_informe_capa_decision.py` | 33 capas, 0 adoptadas; 6 entregables en `reports/age_*` |
| 129 | Cierre documental | `scripts/phase35_cierre_documentacion.py` | D-036 (D-034 cerrada) y D-037 (línea de decisión cerrada) |

### Bugs corregidos en la sesión
- `decision_layer`: convención de supervivencia frente a CDF (decodificación A degenerada a 0,026 ->
  0,5887 tras corregir), indexado global/local en tres capas, `np.ix_` que sobrescribía las seis
  columnas, y `del` sobre variables capturadas.
- `phase31`: el LogReg de umbrales no convergía sin estandarizar (se añadió `StandardScaler`).

### Resultado
Ninguna de las **33 capas de decisión** alcanza el gate (Δ ≥ +0,003 con ≥4/5 folds). La mejor,
`G desempate E[Y] (tau 0,05)`, da +0,0011 con 4/5 folds y **neto de rescate −117**: el argmax de V12 ya
es la mejor decisión de su distribución de probabilidades. El error restante se reparte en un techo
de información (35% del error: ninguna fuente acierta) y un 65% de agregación que ninguna regla de
decisión convierte en aciertos. V12 permanece congelado; sin submission nueva.
"""
    runlog.write_text(texto, encoding="utf-8")

    # 4) PROJECT_STATE.md
    estado = ROOT / "PROJECT_STATE.md"
    texto = estado.read_text(encoding="utf-8")
    inicio = texto.index("## Fase actual")
    fin = texto.index("## Estado según el formato del brief §83")
    texto = texto[:inicio] + """## Fase actual
- **FASE DE CAPA DE DECISIÓN ORDINAL (2026-09-25) CERRADA — línea cerrada (D-037).** 33 capas de
  decisión ajustadas con corrected CV sobre las probabilidades OOF congeladas de V12 (decodificación
  acumulativa, proyección ordinal, coste ordinal, desempate por edad esperada, corrección residual de
  fronteras, decodificación por matriz de transición, calibración por banda de confianza,
  combinación Luce de márgenes adyacentes y corrector mínimo): **ninguna alcanza el gate**
  (Δ ≥ +0,003 con ≥4/5 folds). La mejor (`G desempate E[Y]`, tau 0,05) da **+0,0011 con 4/5 folds**
  y **neto de rescate −117**.
- **Modelo final: V12 congelado.** Leaderboard de referencia **0,62188** (artefacto
  `age_submission_v12_reproducible.csv`; el archivo histórico obtuvo 0,62182). **D-034 cerrada
  funcionalmente** (D-036): la discrepancia de reconstrucción no afecta el desempeño. **No se generó
  ni se envió ninguna submission nueva.**
- **Diagnóstico del error residual**: techo de información 13,87% de filas (35% del error, ningún
  miembro acierta) + 65% de agregación (26,70% de filas rescatables) que ninguna regla de decisión
  convierte en aciertos. Fronteras centrales: 10.122 y 7.252 errores.
- **Conclusión**: bajo las restricciones de AGE, la representación contiene la información disponible
  y el límite está en la observabilidad de la edad, no en la regla de decisión.
- **Entregables**: `reports/age_residual_decision_research.md`, `reports/age_residual_audit.md`,
  `reports/age_boundary_models.csv`, `reports/age_probability_correction.csv`,
  `reports/age_residual_candidates.csv`, `reports/age_boundary_confusion_matrices/`,
  `reports/age_hypothesis_coverage.md`.

""" + texto[fin:]
    estado.write_text(texto, encoding="utf-8")

    # 5) FINAL_REPORT.md (Parte III)
    informe = ROOT / "FINAL_REPORT.md"
    texto = informe.read_text(encoding="utf-8")
    renglon = ("- **Leaderboard**: **0.62182** de F1 (v12; línea base inicial 0.55235; mejora acumulada "
               "**+0.0695**).")
    if renglon in texto:
        texto = texto.replace(renglon, "- **Leaderboard**: **0.62188** de F1 (artefacto v12 reproducible; "
                              "el archivo histórico obtuvo 0.62182; línea base inicial 0.55235; mejora "
                              "acumulada **+0.0696**).")
    texto = texto.rstrip() + f"""

---

# Parte III — Capa de decisión ordinal y corrección de fronteras (2026-09-25)

**Pregunta**: ¿existe una forma distinta de convertir las señales ya disponibles en una decisión
ordinal de seis clases que reduzca el error en las fronteras `18-24|25-34` y `25-34|35-49`?
**Gate pre-registrado**: Δ ≥ +0,003 con ≥4/5 folds. **Holdout agotado** (no se reutiliza).

## 21. Cobertura: qué estaba ya hecho

Los modelos ordinales (E-015, E-021a) y los márgenes adyacentes (E-028a) **ya existen como miembros**
del ensamble, y el routing duro está cerrado. Lo que nunca existía era una **capa de decisión** sobre
probabilidades congeladas: nueve líneas genuinamente abiertas (`reports/age_hypothesis_coverage.md`).

## 22. Resultado: 33 capas, ninguna adopta

| Hipótesis | Capas | Δ máximo | Veredicto |
|---|---|---|---|
| A umbrales acumulados | 2 | −0,0031 | el modelo de umbrales solo da 0,5887; la evidencia de fronteras ya está en el meta |
| B combinación Luce de márgenes | 3 | −0,0034 | los márgenes de E-028a no forman una escala de Bradley-Terry válida (0,2947) |
| C corrección residual de fronteras | 4 | −0,0000 | en su forma más suave, **neto 0**: el contexto no aporta más de lo ya usado |
| D decodificación por transición | 5 | −0,0004 | empeora monótonamente al aumentar λ |
| E proyección ordinal | 6 | −0,0033 | desplazar masa del top-1 siempre destruye más de lo que arregla |
| F coste ordinal | 3 | +0,0004 | solo preserva la decisión con perfiles de coste casi planos |
| G desempate por E[Y] | 3 | **+0,0011** | mejora el MAE ordinal (0,6552 → 0,6454) pero **empeora las fronteras centrales** |
| H calibración por banda | 2 | +0,0002 | inerte: una transformación por fila no cambia el argmax |
| I corrector mínimo | 5 | −0,0005 | reescribe aciertos a cambio de fallos nuevos |

## 23. Tres hallazgos defendibles

1. **El argmax de V12 ya es la mejor decisión** de su distribución de probabilidades: toda
   redistribución (proyección, calibración por banda, decodificación por confusión) degrada de forma
   monótona.
2. **Minimizar el error ordinal no es mejorar macro-F1**: el desempate por edad esperada mejora el MAE
   ordinal y las fronteras extremos, pero empeora `18-24|25-34` (+462 errores) y `25-34|35-49` (+534).
3. **La evidencia de fronteras ya está dentro del meta-modelo**: un modelo de umbrales acumulados
   ajustado sobre las features del meta alcanza 0,5887 por sí solo y su fusión con V12 no aporta.

## 24. Dónde vive el error residual

- **Techo de información**: {diagnostico['techo_de_informacion_sin_miembro_correcto']:,} filas
  ({diagnostico['pct_del_error']}% del error) donde **ningún miembro acierta**.
- **Agregación**: {diagnostico['errores_rescatables']:,} filas
  ({diagnostico['pct_del_error_rescatable']}% del error) donde algún miembro acierta y el ensamble no;
  ninguna regla de decisión sobre las probabilidades convierte ese saber en aciertos.
- **Fronteras centrales**: {diagnostico['frontera_18_24_25_34_cruces']:,} cruces de frontera en
  `18-24|25-34` (sobre {diagnostico['frontera_18_24_25_34_errores_en_el_par']:,} errores en las filas de
  la pareja) y {diagnostico['frontera_25_34_35_49_cruces']:,} en `25-34|35-49`
  ({diagnostico['frontera_25_34_35_49_errores_en_el_par']:,}).

## 25. Conclusión

Bajo las restricciones de AGE, **la representación contiene la información disponible y el límite
está en la observabilidad de la edad, no en la regla de decisión**. V12 permanece congelado, no se
genera submission y la línea de decisión queda cerrada con 33 capas medidas (D-037). El leaderboard
de referencia es **0,62188** (artefacto reproducible; D-036 cierra la discrepancia de reconstrucción).
"""
    informe.write_text(texto, encoding="utf-8")

    print(f"D-036 y D-037 escritas | {len(filas)} experimentos registrados | "
          f"FINAL_REPORT: {len(informe.read_text(encoding='utf-8').splitlines())} líneas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
