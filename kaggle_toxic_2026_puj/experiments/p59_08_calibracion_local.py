"""P59 step 08: track T, local-window calibration. CLOSED by duplication, no training.

Why this track is closed without consuming an experiment
--------------------------------------------------------
Track T proposes to take the raw local-window max score produced by B1 and
recalibrate it with Platt scaling fitted train-fold-only, then compare at V03's
frozen thresholds. It is closed on two independent legs, and the first one is
algebraic rather than historical.

**Leg 1, algebraic.** Platt scaling is a monotone affine map in the logit:

    p_cal = sigmoid(a * logit(p_raw) + b)

A fixed decision threshold `t` on the calibrated score means

    p_cal >= t   <=>   a*logit(p_raw) + b >= logit(t)
                <=>   logit(p_raw) >= (logit(t) - b) / a
                <=>   p_raw   >= sigmoid((logit(t) - b) / a)

so applying Platt and keeping `t` fixed is **exactly** a re-selection of the
per-label decision threshold on the raw local score. It introduces no
information; it only moves where the cut sits. The brief itself rules that out
in its own words when it says T "no es threshold sweep".

**Leg 2, precedent, and it is a root closure.** `V8-09` measured, across fifteen
independent probes, that F1 as a function of the predicted-positive rate is
unimodal and that **V03 sits exactly at its maximum**. Its own words: moving the
rate in either direction costs leaderboard, "y eso cierra de raiz la familia
entera de calibraciones, umbrales y pesos por etiqueta". On top of that:

- `V10-04` ran Platt, clipped isotonic and a three-parameter beta on V03's own
  scores with the calibrator nested in the fold, over five partition seeds. Platt
  genuinely works: mean ECE falls from 0,058293 to 0,024801. But the nested mean
  delta is **+0,000148** with IC95 [-0,005221, +0,005932], and the decision
  closes with an explicit, pre-registered clause: "No se reintenta con mas
  variantes de calibracion porque el criterio de cierre estaba escrito antes de
  ejecutar y ya se cumplio con la isotonica".
- `E2-14` calibrated M2 and META: +0,0018 with 3 of 5 folds, and -0,0017, both
  "CERRADA sin mejora".
- `V4-H01` applied Platt to the 36-feature meta scores: **-0,0171**, worse.
- `V9-02` combined Platt with Saerens EM: not adopted.

So calibration is not unexplored here, it is explored four times and closed at
the root by a measurement that does not depend on fitting anything.

What distinguishes U, which is not closed
-----------------------------------------
Track U is *not* a threshold change. Fusing a second score into the first
changes the **ranking**, which a monotone recalibration cannot do, and that is
the exact reason it escapes the `V8-09` closure. U is run separately for that
reason and is not affected by this closure.

Usage
-----
    /home/carlos/venv/bin/python experiments/p59_08_calibracion_local.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    V03_CCV,
    V03_SHA,
    V03_SUB,
)

OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
CELDA_B1 = PROJECT_ROOT / "features" / "p59_local_window_cells.npz"
JSON_T = PROJECT_ROOT / "reports" / "p59_T_local_calibration.json"
MD_T = PROJECT_ROOT / "reports" / "p59_T_local_calibration.md"

V03_UMBRALES = (0.44, 0.51, 0.46)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P59-08] {mensaje}", flush=True)


SUSTITUCIONES = {
    chr(0x2014): " - ", chr(0x2013): "-", chr(0x2212): "-", chr(0x00B1): "+/-",
    chr(0x00AB): '"', chr(0x00BB): '"', chr(0x201C): '"', chr(0x201D): '"',
    chr(0x2018): "'", chr(0x2019): "'", chr(0x2264): "<=", chr(0x2265): ">=",
    chr(0x00D7): "x", chr(0x2192): "->", chr(0x00B7): ".", chr(0x2026): "...",
    chr(0x00BF): "?", chr(0x00A1): "!", chr(0x00A7): "parrafo ", chr(0x00A0): " ",
}


def a_ascii(texto: str) -> str:
    """Translitera a ASCII puro conservando el significado.

    Args:
        texto: cadena de entrada.

    Returns:
        La cadena equivalente en ASCII.
    """
    if not isinstance(texto, str):
        texto = str(texto)
    for k, v in SUSTITUCIONES.items():
        texto = texto.replace(k, v)
    salida = []
    import unicodedata
    for ch in unicodedata.normalize("NFKD", texto):
        if ord(ch) < 128:
            salida.append(ch)
            continue
        if unicodedata.category(ch).startswith("M"):
            continue
        partes = unicodedata.decomposition(ch).split()
        base = chr(int(partes[0], 16)) if partes else ""
        salida.append(base if base and ord(base) < 128 else "?")
    return "".join(salida)


def main() -> None:
    """Registra el cierre de T por duplicacion funcional, sin entrenar.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P59-T Local-window calibration. Cierre por duplicacion funcional, SIN entrenar")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    log(f"   V03 sha256 verificado e intacto: {hash_real[:16]}...")

    v = np.load(OOF_V03, allow_pickle=True)
    ccv_ref = float(f1_score(v["y_true"], v["pred"], average="macro"))
    if abs(ccv_ref - V03_CCV) > 1e-12:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")
    if not CELDA_B1.exists():
        raise SystemExit("PARADA: falta features/p59_local_window_cells.npz; ejecuta antes P59-07")
    b1 = np.load(CELDA_B1, allow_pickle=True)
    ccv_b1 = float(f1_score(b1["y_true"], b1["pred_b1"], average="macro"))
    log(f"   T0 control: V03 {ccv_ref!r} exacto | B1 (insumo de T) {ccv_b1!r}")

    informe = {
        "familia": "P59_T_LOCAL_CALIBRATION",
        "naturaleza": ("CIERRE POR DUPLICACION FUNCIONAL. 0 experimentos, 0 modelos, 0 scores, "
                       "0 submissions, 0 consultas a Kaggle, 0 holdout, 0 test. V03 no se modifica"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "track": "P59-T",
        "estado": "CLOSED_BY_DUPLICATION",
        "experimentos_consumidos": 0,
        "dependencia": "B1, de la que hereda el score local max. No ejecutada porque B1 cerro y la"
                       " prueba de duplicacion no necesita el score",
        "T0_control": {
            "corrected_cv_V03": repr(ccv_ref),
            "reproduce_exacto": bool(abs(ccv_ref - V03_CCV) <= 1e-12),
            "corrected_cv_B1_insumo": round(ccv_b1, 10),
        },
        "T1_gap_check": {
            "veredicto": "EXISTE EQUIVALENCIA FUNCIONAL",
            "pierna_1_algebraica": {
                "platt": "p_cal = sigmoid(a*logit(p_raw) + b)",
                "consecuencia": ("con umbral fijo t, p_cal >= t equivale EXACTAMENTE a "
                                 "p_raw >= sigmoid((logit(t) - b)/a), es decir un "
                                 "reseleccion del umbral por etiqueta sobre el score crudo"),
                "por_que_eso_mata_T": ("una reparametrizacion monotona no aporta informacion, solo "
                                       "mueve el corte. El encargo mismo excluye el threshold sweep "
                                       "al definir T"),
            },
            "pierna_2_precedente": [
                {"id": "V8-09",
                 "hallazgo": ("F1 como funcion de la tasa de positivos predicha es UNIMODAL y V03 "
                              "esta exactamente en su maximo, medido sobre quince sondas "
                              "independientes"),
                 "cierre": ("cierra de raiz la familia entera de calibraciones, umbrales y pesos por "
                            "etiqueta, y no depende de ajustar nada")},
                {"id": "V10-04",
                 "que_hizo": ("Platt, isotonica recortada y beta de tres parametros sobre los scores "
                              "de V03, con calibrador anidado en el pliegue y cinco semillas"),
                 "resultado": ("la calibracion funciona de verdad: ECE medio de 0,058293 a 0,024801 "
                               "con Platt. Pero el delta medio anidado de Platt es +0,000148 con "
                               "IC95 [-0,005221, +0,005932]"),
                 "cierre": ("NO ACCIONABLE, con clausula pre-registrada explicita: no se reintenta "
                            "con mas variantes de calibracion porque el criterio de cierre ya se "
                            "cumplio")},
                {"id": "E2-14-MEJOR-M2", "resultado": "+0,0018 con 3 de 5 pliegues, CERRADA sin mejora"},
                {"id": "E2-14-MEJOR-META", "resultado": "-0,0017, CERRADA sin mejora"},
                {"id": "V4-H01", "resultado": "Platt sobre los scores del meta de 36 rasgos: -0,0171, empeora"},
                {"id": "V9-02", "resultado": "Platt mas EM de Saerens, sonda no adoptada ni enviada"},
            ],
        },
        "por_que_U_no_esta_cerrado": (
            "U fusiona un segundo score en el primero, y eso cambia el RANKING, que ninguna "
            "reparametrizacion monotona puede hacer. Esa es exactamente la linea algebraica que "
            "separa U de T, y por eso la fusion continua no cae en el cierre de V8-09 ni en la "
            "clausula de V10-04. U se ejecuta por separado y este cierre no la afecta"),
        "decision": (
            "CLOSED_BY_DUPLICATION. 0 experimentos consumidos. No es que la calibracion del score "
            "local haya fallado: es que no es un experimento, porque Platt a umbral fijo ES un "
            "cambio de umbral, y esa familia esta cerrada de raiz por V8-09 y de forma "
            "especifica por V10-04. Sin sonda, sin entrenamiento, sin submissions. "
            "V03 remains the best verified leaderboard result."),
        "formulacion": "V03 remains the best verified leaderboard result.",
        "gastos": {"submissions": 0, "consultas_kaggle": 0, "uso_holdout": 0, "uso_test": 0,
                   "entrenamientos": 0, "segundos": 0.1},
    }
    JSON_T.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    L: list[str] = []
    A = L.append
    A("# P59-T - Calibracion del score local: CLOSED_BY_DUPLICATION")
    A("")
    A(f"Generado por `experiments/p59_08_calibracion_local.py`. Fecha {inicio.isoformat(timespec='seconds')}. "
      "Sin entrenar.")
    A("")
    A("## Que proponia T")
    A("")
    A("Tomar el score local crudo de B1, `max_j p(ventana_j)`, y calibrarlo con Platt ajustado solo "
      "con el training fold, y compararlo con V03 y con B1 crudo a los umbrales congelados.")
    A("")
    A("## T0 control")
    A("")
    A("```text")
    A(f"V03 {ccv_ref!r}  reproduce exacto: {abs(ccv_ref - V03_CCV) <= 1e-12}")
    A(f"B1 (insumo de T) {ccv_b1!r}")
    A("```")
    A("")
    A("## T1 gap check: EXISTE EQUIVALENCIA FUNCIONAL")
    A("")
    A("### Pierna 1, algebrica")
    A("")
    A("Platt es una transformacion afin en el logit, `p_cal = sigmoid(a*logit(p_raw) + b)`. Con un "
      "umbral fijo `t`:")
    A("")
    A("```text")
    A("p_cal >= t  <=>  a*logit(p_raw) + b >= logit(t)")
    A("           <=>  p_raw >= sigmoid((logit(t) - b)/a)")
    A("```")
    A("")
    A("Aplicar Platt y mantener `t` **es exactamente** reseleccionar el umbral por etiqueta sobre el "
      "score crudo. No aporta informacion, solo mueve el corte. Y el encargo excluye el threshold "
      "sweep al definir T.")
    A("")
    A("### Pierna 2, precedente, y es un cierre de raiz")
    A("")
    A("| id | que midio | resultado |")
    A("|---|---|---|")
    A("| V8-09 | F1 frente a la tasa de positivos predicha, sobre quince sondas independientes | "
      "es **unimodal y V03 esta exactamente en el maximo**; cierra de raiz calibraciones, umbrales "
      "y pesos por etiqueta |")
    A("| V10-04 | Platt, isotonica recortada y beta de tres parametros sobre los scores de V03, "
      "calibrador anidado, cinco semillas | la calibracion funciona: ECE 0,058293 -> 0,024801. Pero "
      "el delta medio anidado de Platt es **+0,000148** con IC95 [-0,005221, +0,005932]. Cerrada con "
      "clausula pre-registrada: no se reintenta con mas variantes |")
    A("| E2-14-MEJOR-M2 | calibracion de M2 | +0,0018 con 3 de 5 pliegues, CERRADA |")
    A("| E2-14-MEJOR-META | calibracion de META | -0,0017, CERRADA |")
    A("| V4-H01 | Platt sobre los scores del meta de 36 rasgos | **-0,0171**, empeora |")
    A("| V9-02 | Platt mas EM de Saerens | sonda no adoptada ni enviada |")
    A("")
    A("La calibracion no esta sin explorar aqui: esta explorada cuatro veces y cerrada de raiz por "
      "una medicion que no depende de ajustar nada.")
    A("")
    A("## Por que U no esta cerrada")
    A("")
    A("U fusiona un segundo score dentro del primero, y eso cambia el **ranking**, que ninguna "
      "reparametrizacion monotona puede hacer. Esa es la linea algebraica exacta que separa U de T, y "
      "por eso la fusion continua no cae en el cierre de V8-09 ni en la clausula de V10-04.")
    A("")
    A("## Decision")
    A("")
    A("```text")
    A(f"decision  {informe['decision']}")
    A("```")
    A("")
    A("> V03 remains the best verified leaderboard result.")
    A("")
    A("*Seccion generada por `experiments/p59_08_calibracion_local.py`. Sin push.*")
    MD_T.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_T, JSON_T):
        if destino.exists() and any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("   escrito reports/p59_T_local_calibration.md y .json")
    log("P59-T VEREDICTO: CLOSED_BY_DUPLICATION | 0 experimentos consumidos")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
