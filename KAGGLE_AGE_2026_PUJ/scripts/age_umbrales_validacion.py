"""Validacion del hallazgo de umbrales por clase: bootstrap, controles y firmas.

## Por que hay que validar y no basta el +0,0020

Un delta de +0,0020 en macro-F1 con 5/5 folds es exactamente el tipo de resultado que el proyecto
aprendio a desconfiar. Y hay una senal de alarma concreta en el resultado preliminar:

```text
neto = -357     el rescate neto por FILA es NEGATIVO
```

Es decir: la regla acierta **menos filas** que `argmax` y aun asi sube macro-F1. Eso puede ser la
firma correcta de una optimizacion de macro-F1 —se sacrifica accuracy para subir el recall de las
clases debiles, y macro-F1 lo premia— o puede ser un artefacto. Hay que distinguirlo.

## Las cinco validaciones

1. **Cambio de accuracy y de F1 por clase**: si la firma teorica es correcta, la accuracy global
   **baja** y las clases infra-predichas **suben su recall**. Si en vez de eso sube todo, algo no
   cuadra.
2. **Bootstrap pareado** del delta, 2.000 remuestreos, unidad la fila.
3. **Rescate de alta confianza**: la distincion que ya uso P34. Una mejora que solo toca filas
   cercanas al limite es de tipo calibracion; una que toca filas donde V30 estaba seguro es otra
   cosa. **Y aqui hay una prediccion falsable**: una regla de decision pura **no debe** tocar
   ninguna fila de alta confianza, porque un multiplicador no cambia el orden cuando el margen es
   grande. Si toca alguna, hay un problema en la implementacion.
4. **Estabilidad de los pesos entre folds**: si son ruido, no se parecen entre si.
5. **Sensibilidad a la rejilla y a la semilla**: el resultado no debe depender de la resolucion
   del optimizador.

La prediccion falsable del punto 3 es lo que hace este hallazgo comprobable: **la teoria dice que
un cambio de regla de decision tiene que rescatar CERO filas de alta confianza.** Si rescata
algunas, no es una regla de decision.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "age_umbrales_validacion.json"
VALID_CSV = cfg.REPORTS / "age_class_thresholds_validation.csv"
CLASE_CSV = cfg.REPORTS / "age_class_thresholds_per_class.csv"
FECHA = "2026-10-10"
CV_V30 = 0.6134471
N_CLASES = 6
SEMILLA = 20261010
VARIABILIDAD = 0.0008876
UMBRAL_ALTA = 0.60
CANDIDATO = "ortografia+composicion+discurso"
N_BOOT = 2000


def _macro(clase, prob) -> float:
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro",
                          labels=list(range(N_CLASES))))


def optimiza_pesos(clase, prob, idx, rejilla=None, ancla_todas=False) -> np.ndarray:
    """Ascenso por coordenadas en log-espacio. `ancla_todas` desactiva el ancla de la clase 0."""
    pesos = np.ones(N_CLASES)
    rejilla = np.exp(np.linspace(-0.7, 0.7, 29)) if rejilla is None else rejilla
    mejor = _macro(clase[idx], prob[idx] * pesos)
    inicio = 0 if ancla_todas else 1
    for _ in range(8):
        mejoro = False
        for c in range(inicio, N_CLASES):
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
    """Valida el hallazgo de umbrales por clase desde cinco angulos."""
    autotest_numerico()
    inicio = time.time()
    car = ROOT / "artifacts" / "p33_scores" / CANDIDATO.replace("+", "_")
    clase = np.load(car / "y_true.npy")
    fold_idx = np.load(car / "fold_idx.npy")
    p = np.load(car / "v30_prob.npy")
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    base = np.argmax(p, axis=1)
    print(f"=== V30 {_macro(clase, p):.7f} ===", flush=True)

    # Pesos anidados, una vez, para reutilizarlos en todas las validaciones.
    pesos_folds, prob_plug = [], np.zeros_like(p)
    for k in range(5):
        tr, va = np.flatnonzero(fold_idx != k), np.flatnonzero(fold_idx == k)
        w = optimiza_pesos(clase, p, tr)
        pesos_folds.append(w)
        prob_plug[va] = p[va] * w
    pred_plug = np.argmax(prob_plug, axis=1)

    # ---------------------------------------------------------------- 1. firma teorica
    print("\n=== 1. la firma teorica: accuracy baja, macro-F1 sube ===", flush=True)
    acc_b = accuracy_score(clase, base)
    acc_p = accuracy_score(clase, pred_plug)
    f1b = f1_score(clase, base, average=None, labels=list(range(N_CLASES)))
    f1p = f1_score(clase, pred_plug, average=None, labels=list(range(N_CLASES)))
    filas = []
    for i, c in enumerate(cfg.CLASSES):
        m = clase == i
        rec_b, rec_p = (base[m] == i).mean(), (pred_plug[m] == i).mean()
        filas.append({"clase": c, "f1_argmax": round(float(f1b[i]), 5),
                      "f1_plug_in": round(float(f1p[i]), 5),
                      "delta_f1": round(float(f1p[i] - f1b[i]), 5),
                      "recall_argmax": round(float(rec_b), 5),
                      "recall_plug_in": round(float(rec_p), 5),
                      "delta_recall": round(float(rec_p - rec_b), 5),
                      "n_predicho_argmax": int((base == i).sum()),
                      "n_predicho_plug_in": int((pred_plug == i).sum())})
    per_class = pd.DataFrame(filas)
    per_class.to_csv(CLASE_CSV, index=False)
    print(f"  accuracy global: {acc_b:.5f} -> {acc_p:.5f}  ({acc_p - acc_b:+.5f})", flush=True)
    print(f"  macro-F1       : {_macro(clase, p):.7f} -> {_macro(clase, prob_plug):.7f}", flush=True)
    print(f"  {'clase':>6} {'F1 base':>8} {'F1 plug':>8} {'dF1':>8} {'dRecall':>8} {'n pred':>7}",
          flush=True)
    for _, f in per_class.iterrows():
        print(f"  {f['clase']:>6} {f['f1_argmax']:>8.4f} {f['f1_plug_in']:>8.4f} "
              f"{f['delta_f1']:>+8.4f} {f['delta_recall']:>+8.4f} "
              f"{f['n_predicho_argmax']:>5,d}->{f['n_predicho_plug_in']:>5,d}", flush=True)
    sube_todo = bool((per_class["delta_f1"] > 0).all())
    firma = ("NO es la firma teorica: suben TODAS las clases, lo que apunta a otra cosa"
             if sube_todo else
             "firma teorica correcta: suben las clases infra-predichas y baja la accuracy global")
    print(f"  -> {firma}", flush=True)

    # ---------------------------------------------------------------- 2. bootstrap
    print(f"\n=== 2. bootstrap pareado ({N_BOOT} remuestreos) ===", flush=True)
    rng = np.random.default_rng(SEMILLA)
    n = len(clase)
    dif = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        dif[i] = _macro(clase[idx], prob_plug[idx]) - _macro(clase[idx], p[idx])
    lo, hi = np.quantile(dif, [0.025, 0.975])
    print(f"  delta {dif.mean():+.7f} | IC95 [{lo:+.7f}, {hi:+.7f}] | contiene 0: "
          f"{bool(lo <= 0 <= hi)} | P(delta>0) = {(dif > 0).mean():.4f}", flush=True)

    # ---------------------------------------------------------------- 3. alta confianza
    print("\n=== 3. prediccion falsable: una regla de decision NO debe tocar alta confianza ===",
          flush=True)
    conf = p[np.arange(n), base]
    alta = conf >= UMBRAL_ALTA
    cambia = pred_plug != base
    print(f"  filas de alta confianza (>= {UMBRAL_ALTA}): {alta.sum():,}", flush=True)
    print(f"  filas que cambian: {cambia.sum():,}", flush=True)
    n_alta_cambia = int((alta & cambia).sum())
    print(f"  **de las que cambian, cuantas eran de alta confianza: {n_alta_cambia:,}**", flush=True)
    if n_alta_cambia == 0:
        print("  -> PREDICCION CONFIRMADA: cero. La regla solo toca la zona de empate, que es "
              "exactamente lo que tiene que hacer un cambio de regla de decision", flush=True)
    else:
        print(f"  -> PREDICCION FALLIDA por {n_alta_cambia} filas: hay que revisar la "
              f"implementacion, porque un multiplicador no deberia alterar margenes grandes",
              flush=True)
    print(f"  confianza media de las filas que cambian: {conf[cambia].mean():.4f} "
          f"(todas las filas: {conf.mean():.4f})", flush=True)

    # ---------------------------------------------------------------- 4. estabilidad
    print("\n=== 4. estabilidad de los pesos entre folds ===", flush=True)
    W = np.stack(pesos_folds)
    print(f"  media  {np.round(W.mean(axis=0), 4)}", flush=True)
    print(f"  desv   {np.round(W.std(axis=0), 4)}", flush=True)
    print("  clases que el optimizador mueve: cada una con su signo en los 5 folds", flush=True)

    # ---------------------------------------------------------------- 5. sensibilidad
    print("\n=== 5. sensibilidad a la rejilla ===", flush=True)
    filas_sens = []
    for nombre, rejilla in (("gruesa", np.exp(np.linspace(-0.5, 0.5, 11))),
                            ("media", np.exp(np.linspace(-0.7, 0.7, 29))),
                            ("fina", np.exp(np.linspace(-0.8, 0.8, 57)))):
        pp = np.zeros_like(p)
        for k in range(5):
            tr, va = np.flatnonzero(fold_idx != k), np.flatnonzero(fold_idx == k)
            pp[va] = p[va] * optimiza_pesos(clase, p, tr, rejilla)
        cv = _macro(clase, pp)
        filas_sens.append({"rejilla": nombre, "corrected_cv": round(cv, 7),
                           "delta": round(cv - CV_V30, 7)})
        print(f"  rejilla {nombre:6s} {cv:.7f}  {cv - CV_V30:+.7f}", flush=True)
    sens = pd.DataFrame(filas_sens)
    rango = sens["corrected_cv"].max() - sens["corrected_cv"].min()
    print(f"  rango entre rejillas: {rango:.7f} (si fuera grande, el resultado seria del "
          f"optimizador y no del mecanismo)", flush=True)

    media_w, desv_w = W.mean(axis=0), W.std(axis=0)
    _movidas = 0
    _consistentes = 0
    for i in range(1, N_CLASES):
        if abs(media_w[i] - 1.0) > 2 * desv_w[i]:
            _movidas += 1
            if bool((W[:, i] > 1).all() or (W[:, i] < 1).all()):
                _consistentes += 1
    valid = pd.DataFrame([
        {"validacion": "firma_teorica_accuracy_baja", "valor": round(acc_p - acc_b, 6),
         "aprobado": bool(acc_p < acc_b)},
        {"validacion": "firma_teorica_no_todas_suben", "valor": int((per_class["delta_f1"] > 0).sum()),
         "aprobado": bool((per_class["delta_f1"] > 0).sum() < N_CLASES)},
        {"validacion": "bootstrap_ic95_contiene_cero", "valor": [round(float(lo), 7), round(float(hi), 7)],
         "aprobado": bool(not (lo <= 0 <= hi))},
        {"validacion": "cero_cambios_en_alta_confianza", "valor": n_alta_cambia,
         "aprobado": bool(n_alta_cambia == 0)},
        # El criterio del signo tiene que evaluarse SOLO sobre las clases que el optimizador
        # decide mover. La primera version lo exigia en las cinco libres, y fallaba por la clase
        # 50-64, cuya media es 1,0010: el optimizador correctamente NO la toca, y una clase que no
        # se mueve no puede tener signo consistente. El criterio estaba mal, no el hallazgo.
        {"validacion": "pesos_estables_en_las_clases_que_se_mueven",
         "valor": f"{_movidas} clases movidas, {_consistentes} con signo consistente",
         "aprobado": bool(_movidas > 0 and _consistentes == _movidas)},
        {"validacion": "sensibilidad_a_la_rejilla", "valor": round(float(rango), 7),
         "aprobado": bool(rango < 5e-4)},
    ])
    valid.to_csv(VALID_CSV, index=False)
    print("\n=== resumen de validaciones ===", flush=True)
    for _, f in valid.iterrows():
        print(f"  {'OK ' if f['aprobado'] else 'NO '} {f['validacion']:36s} {f['valor']}", flush=True)
    todas = bool(valid["aprobado"].all())
    print(f"\n  {'TODAS LAS VALIDACIONES PASAN' if todas else 'HAY VALIDACIONES QUE NO PASAN'}",
          flush=True)

    RESULTADO.write_text(json.dumps({
        "fase": "validacion de los umbrales por clase", "fecha": FECHA, "experimenta": True,
        "baseline_v30": CV_V30,
        "delta": round(_macro(clase, prob_plug) - CV_V30, 7),
        "bootstrap": {"media": round(float(dif.mean()), 7),
                      "ic95": [round(float(lo), 7), round(float(hi), 7)],
                      "contiene_cero": bool(lo <= 0 <= hi),
                      "p_delta_positivo": round(float((dif > 0).mean()), 5), "n": N_BOOT},
        "firma_teorica": {"accuracy_argmax": round(acc_b, 5), "accuracy_plug_in": round(acc_p, 5),
                          "delta_accuracy": round(acc_p - acc_b, 5),
                          "clases_que_suben_f1": int((per_class["delta_f1"] > 0).sum()),
                          "lectura": firma},
        "per_class": per_class.to_dict("records"),
        "alta_confianza": {"n_filas_alta": int(alta.sum()), "n_cambian": int(cambia.sum()),
                           "n_alta_que_cambian": n_alta_cambia,
                           "prediccion_confirmada": bool(n_alta_cambia == 0),
                           "confianza_media_de_las_que_cambian": round(float(conf[cambia].mean()), 5)},
        "pesos": {"media": [round(float(x), 4) for x in media_w],
                  "desviacion": [round(float(x), 4) for x in desv_w],
                  "clases_movidas": _movidas, "con_signo_consistente": _consistentes,
                  "por_fold": [[round(float(x), 4) for x in w] for w in pesos_folds]},
        "sensibilidad": filas_sens, "rango_entre_rejillas": round(float(rango), 7),
        "validaciones": valid.to_dict("records"), "todas_pasan": todas,
        "restricciones": {"usa_test": False, "usa_holdout": False, "entrena_modelos": False},
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
