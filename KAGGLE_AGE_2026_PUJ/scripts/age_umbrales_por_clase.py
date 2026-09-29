"""Umbrales por clase para macro-F1: la regla de decision que AGE nunca probo.

## Por que esto no es una familia cerrada

Conviene ser preciso, porque la palabra "umbral" aparece en el historial y podria parecer un
duplicado. **No lo es:**

| Experimento historico | Que era |
|---|---|
| `V20-F3-UMBRALES` | una **cadena de cinco discriminadores binarios anidados** con monotonia: una arquitectura ordinal, no una regla de decision |
| `P33`, `P34`, `G12` | correcciones que **inyectan un residuo** de otra fuente sobre el score |
| `PH32/PH33` | calibracion y capas de correccion sobre las probabilidades |
| **esto** | **la regla de decision**: con que multiplicador por clase se convierte P en argmax |

La diferencia importa: las tres familias cerradas intentan **anadir informacion**. Esto no anade
nada. Cambia **como se decide** con la informacion que ya existe.

## La teoria dice que aqui hay margen, y la medicion lo confirma

Para macro-F1 la regla optima **no** es `argmax`. Es la regla plug-in (Lipton et al., ICML 2014;
Dembczynski et al.): cada clase lleva un multiplicador, y se predice `argmax_c (w_c * p_c)`. La
razon es que macro-F1 promedia F1 **por clase**, asi que una clase con recall bajo pesa lo mismo
que una con recall alto, y `argmax` favorece sistematicamente a las clases faciles.

Y la medicion de V30 lo muestra sin ambiguedad:

```text
las seis clases de train son PERFECTAMENTE balanceadas (0,160 a 0,170)
las predicciones de V30 NO lo estan (0,146 a 0,187)
el F1 por clase va de 0,4992 (25-34) a 0,7351 (65-xx)
```

25-34 esta **infra-predicha** (0,152 predicho frente a 0,167 real) y es la peor clase. 18-24 y
50-64 estan **sobre-predichas**. Un desplazamiento del reparto hacia las clases infra-predichas
sube el recall de las dificiles y, con el, el macro-F1.

## La estructura anidada, y el control

Los multiplicadores se optimizan **dentro** de cada fold externo, con los otros cuatro folds como
validacion interna. `outer_val` solo recibe la regla ya fijada.

Los controles que deciden si el efecto es real:

- **`permutado`**: los mismos multiplicadores aplicados a las clases en orden barajado. Si rinde
  igual, lo que ayuda no es el ajuste por clase.
- **`uniforme`**: multiplicadores todos a 1, que es `argmax` puro. La referencia.
- **`aleatorio`**: multiplicadores al azar con la misma dispersion. Mide si cualquier perturbacion
  del reparto ayuda.

El gate es el doble del proyecto: delta >= +0,0015 con 4/5 folds **y** delta por encima de la
variabilidad de entrenamiento medida en P00 (8,9e-04) con signo consistente.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "age_umbrales_por_clase.json"
TABLA_CSV = cfg.REPORTS / "age_class_thresholds.csv"
CONTROLES_CSV = cfg.REPORTS / "age_class_thresholds_controls.csv"
FECHA = "2026-10-10"
CV_V30 = 0.6134471
N_CLASES = 6
SEMILLA = 20261010
VARIABILIDAD = 0.0008876
CANDIDATO = "ortografia+composicion+discurso"


def _macro(clase, prob) -> float:
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro",
                          labels=list(range(N_CLASES))))


def optimiza_pesos(clase, prob, idx) -> np.ndarray:
    """Coordinacion sobre los multiplicadores por clase, maximizando macro-F1 en `idx`.

    Se optimiza en **log-espacio** y con ascenso por coordenadas: para cada clase se prueba una
    rejilla de multiplicadores dejando las demas fijas, y se itera hasta que no mejora. Es barato,
    determinista y no necesita gradientes. Se ancla la clase 0 en 1.0 porque multiplicar **todos**
    los pesos por una constante no cambia ningun argmax: solo hay N-1 grados de libertad reales.
    """
    pesos = np.ones(N_CLASES)
    rejilla = np.exp(np.linspace(-0.7, 0.7, 29))
    mejor = _macro(clase[idx], prob[idx] * pesos)
    for _ in range(8):
        mejoro = False
        for c in range(1, N_CLASES):
            base = pesos[c]
            for f in rejilla:
                pesos[c] = base * f
                valor = _macro(clase[idx], prob[idx] * pesos)
                if valor > mejor + 1e-9:
                    mejor, base, mejoro = valor, pesos[c], True
            pesos[c] = base
        if not mejoro:
            break
    return pesos


def main() -> int:
    """Evalua la regla plug-in por clase con estructura anidada y tres controles."""
    autotest_numerico()
    inicio = time.time()
    car = ROOT / "artifacts" / "p33_scores" / CANDIDATO.replace("+", "_")
    clase = np.load(car / "y_true.npy")
    fold_idx = np.load(car / "fold_idx.npy")
    p = np.load(car / "v30_prob.npy")
    # Normalizar para que los multiplicadores operen sobre una escala comparable.
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    rng = np.random.default_rng(SEMILLA)
    print(f"=== V30 {_macro(clase, p):.7f} | canonico {CV_V30} ===", flush=True)

    base_pred = np.argmax(p, axis=1)
    f1_por_clase = f1_score(clase, base_pred, average=None, labels=list(range(N_CLASES)))
    dist_real = np.bincount(clase, minlength=N_CLASES) / len(clase)
    dist_pred = np.bincount(base_pred, minlength=N_CLASES) / len(base_pred)
    print("\n=== el desajuste que motiva la regla ===", flush=True)
    print(f"  {'clase':>6} {'real':>7} {'predicho':>9} {'F1':>7} {'desvio':>8}", flush=True)
    for i, c in enumerate(cfg.CLASSES):
        print(f"  {c:>6} {dist_real[i]:>7.3f} {dist_pred[i]:>9.3f} {f1_por_clase[i]:>7.4f} "
              f"{dist_pred[i] - dist_real[i]:>+8.3f}", flush=True)

    print("\n=== regla plug-in, multiplicadores ajustados SOLO en los otros folds ===", flush=True)
    brazos = {"uniforme": np.zeros((len(clase), N_CLASES))}
    for b in ("plug_in", "permutado", "aleatorio"):
        brazos[b] = np.zeros((len(clase), N_CLASES))
    pesos_por_fold = []
    for k in range(5):
        tr = np.flatnonzero(fold_idx != k)
        va = np.flatnonzero(fold_idx == k)
        w = optimiza_pesos(clase, p, tr)
        pesos_por_fold.append([round(float(x), 4) for x in w])
        brazos["uniforme"][va] = p[va]
        brazos["plug_in"][va] = p[va] * w
        # Control 1: los MISMOS pesos, asignados a las clases en orden barajado.
        brazos["permutado"][va] = p[va] * w[rng.permutation(N_CLASES)]
        # Control 2: pesos aleatorios con la misma dispersion que los ajustados.
        brazos["aleatorio"][va] = p[va] * np.exp(rng.normal(0, np.log(w).std() + 1e-9,
                                                            size=N_CLASES))
        print(f"  fold {k}: pesos {[round(float(x), 3) for x in w]}", flush=True)

    print("\n=== resultados ===", flush=True)
    filas = []
    for nombre, prob in brazos.items():
        cv = _macro(clase, prob)
        folds = [_macro(clase[fold_idx == k], prob[fold_idx == k]) for k in range(5)]
        bf = [_macro(clase[fold_idx == k], p[fold_idx == k]) for k in range(5)]
        pred = np.argmax(prob, axis=1)
        ok_c, ok_v = pred == clase, base_pred == clase
        filas.append({
            "brazo": nombre, "corrected_cv": round(cv, 7), "delta_vs_v30": round(cv - CV_V30, 7),
            "folds": [round(v, 6) for v in folds],
            "folds_positivos": int(sum(1 for a, b in zip(folds, bf) if a > b)),
            "rescate": int((~ok_v & ok_c).sum()), "danio": int((ok_v & ~ok_c).sum()),
            "neto": int((~ok_v & ok_c).sum()) - int((ok_v & ~ok_c).sum()),
            "filas_que_cambian": int((pred != base_pred).sum()),
            "ratio_sobre_variabilidad": round(abs(cv - CV_V30) / VARIABILIDAD, 3),
        })
    tabla = pd.DataFrame(filas)
    tabla.to_csv(TABLA_CSV, index=False)
    for _, f in tabla.iterrows():
        print(f"  {f['brazo']:10s} {f['corrected_cv']:.7f}  {f['delta_vs_v30']:+.7f}  "
              f"folds+ {f['folds_positivos']}/5  cambia {f['filas_que_cambian']:>6,d}  "
              f"neto {f['neto']:>+6,d}  ratio {f['ratio_sobre_variabilidad']:.2f}", flush=True)

    d = {f["brazo"]: f["delta_vs_v30"] for f in filas}
    print("\n=== atribucion ===", flush=True)
    print(f"  plug_in - permutado = {d['plug_in'] - d['permutado']:+.7f}", flush=True)
    print(f"  plug_in - aleatorio = {d['plug_in'] - d['aleatorio']:+.7f}", flush=True)
    print(f"  plug_in - uniforme  = {d['plug_in'] - d['uniforme']:+.7f}", flush=True)
    print(f"\n  pesos por fold: {pesos_por_fold}", flush=True)

    if d["plug_in"] <= d["uniforme"]:
        veredicto = "SIN APORTE: la regla por clase no mejora al argmax"
    elif d["plug_in"] <= d["permutado"] or d["plug_in"] <= d["aleatorio"]:
        veredicto = ("INVALIDADO: mejora al argmax pero no a los controles, asi que lo que ayuda "
                     "no es el ajuste por clase")
    elif d["plug_in"] >= 0.0015 and filas[1]["folds_positivos"] >= 4:
        veredicto = "PROMISING: cruza el gate doble"
    elif d["plug_in"] > 0:
        veredicto = (f"MARGINAL: {d['plug_in']:+.7f}, por debajo del gate de +0,0015 y a "
                     f"{abs(d['plug_in']) / VARIABILIDAD:.2f} de la variabilidad de entrenamiento")
    else:
        veredicto = "SIN APORTE"
    print(f"\n=== {veredicto} ===", flush=True)

    pd.DataFrame([{"comparacion": "plug_in - permutado", "delta": round(d["plug_in"] - d["permutado"], 7)},
                  {"comparacion": "plug_in - aleatorio", "delta": round(d["plug_in"] - d["aleatorio"], 7)},
                  {"comparacion": "plug_in - uniforme", "delta": round(d["plug_in"] - d["uniforme"], 7)},
                  {"comparacion": "permutado - uniforme", "delta": round(d["permutado"] - d["uniforme"], 7)}]
                 ).to_csv(CONTROLES_CSV, index=False)

    RESULTADO.write_text(json.dumps({
        "fase": "umbrales por clase para macro-F1", "fecha": FECHA, "experimenta": True,
        "baseline_v30": CV_V30,
        "por_que_no_es_duplicado": (
            "V20-F3-UMBRALES era una cadena de cinco discriminadores binarios anidados, o sea una "
            "ARQUITECTURA ordinal. P33/P34/G12 inyectaban un residuo de otra fuente. PH32/PH33 "
            "calibraban las probabilidades. Esto no anade informacion: cambia la REGLA DE DECISION "
            "con la informacion existente, que es otra cosa"),
        "evidencia_que_lo_motiva": {
            "clases_balanceadas_en_train": [round(float(x), 4) for x in dist_real],
            "prediccion_de_v30_no_balanceada": [round(float(x), 4) for x in dist_pred],
            "f1_por_clase": [round(float(x), 4) for x in f1_por_clase],
            "lectura": "las clases son perfectamente balanceadas y argmax favorece a las faciles, "
                       "que es exactamente el caso en el que la regla plug-in supera a argmax para "
                       "macro-F1",
        },
        "resultados": tabla.to_dict("records"),
        "pesos_por_fold": pesos_por_fold,
        "veredicto": veredicto,
        "protocolo": {"pesos": "ascenso por coordenadas en log-espacio, ajustados SOLO en los "
                               "otros 4 folds (anidado)", "semilla": SEMILLA,
                      "grados_de_libertad": "5 (la clase 0 se ancla en 1.0; multiplicar todos los "
                                            "pesos por una constante no cambia ningun argmax)"},
        "restricciones": {"usa_test": False, "usa_holdout": False, "entrena_modelos": False,
                          "usa_leaderboard": False},
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
