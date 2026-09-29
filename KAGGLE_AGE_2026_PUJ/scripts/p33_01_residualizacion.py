"""P33+ - 01: P04 a P09. Residualizacion multiclase, control barajado y correccion minima.

## La pregunta que este script responde

P03 hallo que la confianza del candidato ordena las filas con AUC ~0,69 en el sentido util. Pero
la tabla de control de P03 mostro tambien algo decisivo:

```text
AUC del MARGEN DE V30 para predecir sus propios errores = 0,8005
AUC del margen del CANDIDATO para lo mismo              = 0,677
```

Es decir, **V30 ya mide la dificultad de la fila mucho mejor que el candidato**. La hipotesis de
P33 es que el candidato tiene algo mas alla de la dificultad, y este script hace exactamente la
operacion que la pone a prueba: **quitarle la parte que V30 ya explica y mirar que queda**.

## La estructura anidada, sin excepciones

Por cada fold externo:

```text
outer_train
    |
    +-- inner_train   -> se ajustan a_c y b_c de la regresion z_C = a_c + b_c * z_V30
    |
    +-- inner_val     -> se elige beta, el unico parametro de la correccion
    |
outer_val            -> se aplica a_c, b_c y beta ya fijados. Nunca se ve durante el ajuste
```

`outer_val` no participa en ninguna decision. Si participara, el resultado no seria un OOF y todo
el protocolo del proyecto estaria vacio.

## Los cinco brazos, y por que estan los cinco

| Brazo | Que responde |
|---|---|
| V30 | referencia |
| +residuo **real** | ¿aporta la correspondencia correcta? |
| +residuo **barajado** | ¿aporta la correspondencia, o solo que se pueda sumar un residuo? |
| +**ruido aleatorio** de igual media y desviacion | ¿aporta por permitir una correccion extra? |
| +residuo con **intercepto** | ¿es señal de edad o solo un efecto de calibración? |

El brazo de intercepto es el que separa dos cosas que se confunden mucho: si la mejora aparece
**solo** con `alpha`, entonces lo que se ha encontrado es que el score de V30 estaba mal
calibrado, no que el candidato diga la edad. El brief lo llama `CALIBRATION_ONLY` y es una
clasificacion distinta de `REPRESENTATION_SIGNAL`, asi que hay que poder distinguirla.

## La prueba que decide, y va primero

Antes de corregir nada, P05 pregunta: **¿el residuo conserva informacion?** Si el AUC del residuo
contra la etiqueta one-vs-rest es ~0,50 en las seis clases, no hay nada que corregir y las
formulas de correccion serian ruido con forma matematica.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

SCORES = ROOT / "artifacts" / "p33_scores"
RESULTADO = cfg.EXPERIMENTS / "p33_01_residualizacion.json"
RESIDUAL_CSV = cfg.REPORTS / "p33_residual_analysis.csv"
CORRECCIONES_CSV = cfg.REPORTS / "p33_score_corrections.csv"
CONTROLES_CSV = cfg.REPORTS / "p33_attribution_controls.csv"
FECHA = "2026-10-02"
CV_V30 = 0.6134471
N_CLASES = 6
EPS = 1e-6
SEMILLA = 20261002
CANDIDATO = "ortografia+composicion+discurso"
#: El brief pide empezar con un unico beta compartido, no seis. Se exploran estos valores y se
#: elige con macro-F1 en inner_val, nunca en outer_val.
BETAS = (-0.60, -0.40, -0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.60)
VARIABILIDAD = 0.0008876


def _macro(clase: np.ndarray, prob: np.ndarray) -> float:
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro",
                          labels=list(range(N_CLASES))))


def logit(p: np.ndarray) -> np.ndarray:
    """log(p / (1 - p)) con clipping. Es la transformacion que pide el brief para P04."""
    q = np.clip(p, EPS, 1.0 - EPS)
    return np.log(q / (1.0 - q))


def a_prob(logit_z: np.ndarray) -> np.ndarray:
    """Inversa de `logit`, con renormalizacion: sirve para volver a argmax sin perder el orden."""
    z = logit_z - logit_z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cargar(nombre: str) -> dict:
    carpeta = SCORES / nombre.replace("+", "_")
    d = {k: np.load(carpeta / f"{k}.npy") for k in
         ("y_true", "fold_idx", "v30_prob", "cand_prob")}
    d["manifest"] = json.loads((carpeta / "manifest.json").read_text(encoding="utf-8"))
    return d


def ajusta_residuo(z_c: np.ndarray, z_v30: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ajusta z_C[c] = a_c + b_c * z_V30[c] **por clase** y solo sobre `indices`.

    Una recta por clase y no una sola: los seis scores de un mismo modelo tienen escalas y
    relaciones con V30 distintas, y forzarles la misma pendiente es una restriccion que el brief
    no pide. Con una recta por clase el residuo es, por construccion, ortogonal a z_V30 dentro de
    esa clase, que es justo lo que se quiere eliminar.
    """
    a = np.zeros(N_CLASES)
    b = np.zeros(N_CLASES)
    for c in range(N_CLASES):
        x = z_v30[indices, c]
        y = z_c[indices, c]
        var = float(x.var())
        b[c] = float(((x - x.mean()) * (y - y.mean())).mean() / var) if var > 1e-12 else 0.0
        a[c] = float(y.mean() - b[c] * x.mean())
    return a, b


def indice_interno(fold_idx: np.ndarray, outer: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Parte outer_train en inner_train e inner_val, de forma estratificada y determinista."""
    train = np.flatnonzero(fold_idx != outer)
    y = np.load(SCORES / CANDIDATO.replace("+", "_") / "y_true.npy")
    partes_train, partes_val = [], []
    for c in range(N_CLASES):
        idx = train[y[train] == c]
        idx = idx[rng.permutation(len(idx))]
        corte = int(0.75 * len(idx))
        partes_train.append(idx[:corte])
        partes_val.append(idx[corte:])
    return np.concatenate(partes_train), np.concatenate(partes_val)


def corrige(z_v30: np.ndarray, residual: np.ndarray, beta: float) -> np.ndarray:
    """z_final[c] = z_V30[c] + beta * residual[c]. Devuelve probabilidades para el argmax."""
    return a_prob(z_v30 + beta * residual)


def main() -> int:
    """Ejecuta P04 a P09 sobre el candidato elegido."""
    autotest_numerico()
    inicio = time.time()
    d = cargar(CANDIDATO)
    clase, fold_idx = d["y_true"], d["fold_idx"]
    oof_v30, p_c = d["v30_prob"], d["cand_prob"]
    print(f"=== P33 sobre {CANDIDATO} ===", flush=True)
    print(f"  V30 {_macro(clase, oof_v30):.7f} | canonico {CV_V30}", flush=True)

    z_v30 = logit(oof_v30)
    z_c = logit(p_c)
    rng = np.random.default_rng(SEMILLA)

    # ---------------------------------------------------------------- P04 + P05
    print("\n=== P04 residualizacion + P05 ¿queda informacion? ===", flush=True)
    residual = np.zeros_like(z_c)
    oof = {nombre: np.zeros((len(clase), N_CLASES)) for nombre in
           ("real", "barajado", "ruido", "con_intercepto")}
    betas_por_fold, auc_por_fold = [], []
    filas_residuo = []
    for outer in range(5):
        inner_tr, inner_va = indice_interno(fold_idx, outer, rng)
        a, b = ajusta_residuo(z_c, z_v30, inner_tr)
        res = z_c - (a[None, :] + b[None, :] * z_v30)

        # P05: ¿el residuo dice algo? Medido sobre outer_val, que no interviene en el ajuste.
        outer_va = np.flatnonzero(fold_idx == outer)
        for c in range(N_CLASES):
            objetivo = (clase[outer_va] == c).astype(int)
            if objetivo.sum() in (0, len(objetivo)):
                continue
            filas_residuo.append({
                "fold": outer, "clase": cfg.CLASSES[c],
                "auc_residuo_vs_ovr": round(float(roc_auc_score(objetivo, res[outer_va, c])), 5),
                "auc_zc_vs_ovr": round(float(roc_auc_score(objetivo, z_c[outer_va, c])), 5),
                "auc_zv30_vs_ovr": round(float(roc_auc_score(objetivo, z_v30[outer_va, c])), 5),
            })
        auc_por_fold.append(float(np.mean([f["auc_residuo_vs_ovr"] for f in filas_residuo
                                           if f["fold"] == outer])))

        # P07: beta se elige SOLO en inner_val.
        interno = np.zeros((len(clase), N_CLASES))
        interno[inner_va] = corrige(z_v30[inner_va], res[inner_va], 0.0)
        mejor_beta, mejor_val = 0.0, _macro(clase[inner_va], interno[inner_va])
        for beta in BETAS:
            if beta == 0.0:
                continue
            prueba = corrige(z_v30[inner_va], res[inner_va], beta)
            valor = _macro(clase[inner_va], prueba)
            if valor > mejor_val:
                mejor_val, mejor_beta = valor, beta
        betas_por_fold.append(mejor_beta)

        # Los cuatro brazos sobre outer_val, con los parametros ya fijados.
        residual[outer_va] = res[outer_va]
        oof["real"][outer_va] = corrige(z_v30[outer_va], res[outer_va], mejor_beta)
        # P06: barajar SOLO la correspondencia fila-residuo. V30, labels y folds intactos.
        oof["barajado"][outer_va] = corrige(
            z_v30[outer_va], res[outer_va][rng.permutation(len(outer_va))], mejor_beta)
        # P15: ruido con la misma media y desviacion que el residuo real.
        ruido = rng.normal(res[inner_va].mean(), res[inner_va].std(), size=res[outer_va].shape)
        oof["ruido"][outer_va] = corrige(z_v30[outer_va], ruido, mejor_beta)
        # P08: con intercepto. alpha tambien se elige en inner_val, y no se usa para decidir.
        alfa, mejor_iv = 0.0, mejor_val
        for alfa_prueba in (-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05):
            prueba = a_prob(z_v30[inner_va] + alfa_prueba + mejor_beta * res[inner_va])
            valor = _macro(clase[inner_va], prueba)
            if valor > mejor_iv:
                mejor_iv, alfa = valor, alfa_prueba
            oof["con_intercepto"][outer_va] = a_prob(
            z_v30[outer_va] + alfa + mejor_beta * res[outer_va])
        print(f"  fold {outer}: beta {mejor_beta:+.2f} | AUC residuo vs OVR "
              f"{auc_por_fold[-1]:.4f}", flush=True)

    tabla_res = pd.DataFrame(filas_residuo)
    tabla_res.to_csv(RESIDUAL_CSV, index=False)
    por_clase = tabla_res.groupby("clase")[["auc_residuo_vs_ovr", "auc_zc_vs_ovr",
                                            "auc_zv30_vs_ovr"]].mean().round(5)
    print(f"\n  {'clase':8s} {'residuo':>9s} {'z_C':>9s} {'z_V30':>9s}", flush=True)
    for c, f in por_clase.iterrows():
        print(f"  {c:8s} {f['auc_residuo_vs_ovr']:>9.4f} {f['auc_zc_vs_ovr']:>9.4f} "
              f"{f['auc_zv30_vs_ovr']:>9.4f}", flush=True)
    media_residuo = float(por_clase["auc_residuo_vs_ovr"].mean())
    print(f"\n  AUC medio del RESIDUO: {media_residuo:.4f}", flush=True)
    print("  (0,50 = sin informacion; por encima de 0,50 el residuo dice algo)", flush=True)

    # ---------------------------------------------------------------- P07-P09
    print("\n=== P07-P09 correccion minima y controles ===", flush=True)
    filas_corr, filas_ctrl = [], []
    v30_pred = np.argmax(oof_v30, axis=1)
    for nombre, o in oof.items():
        cv = _macro(clase, o)
        folds = [_macro(clase[fold_idx == k], o[fold_idx == k]) for k in range(5)]
        base = [_macro(clase[fold_idx == k], oof_v30[fold_idx == k]) for k in range(5)]
        pred = np.argmax(o, axis=1)
        ok_c, ok_v = pred == clase, v30_pred == clase
        rescate = int((~ok_v & ok_c).sum())
        danio = int((ok_v & ~ok_c).sum())
        filas_corr.append({
            "brazo": nombre, "corrected_cv": round(cv, 7), "delta_vs_v30": round(cv - CV_V30, 7),
            "folds": [round(v, 6) for v in folds], "delta_por_fold": [round(a - b, 6)
                                                                     for a, b in zip(folds, base)],
            "folds_positivos": int(sum(1 for a, b in zip(folds, base) if a > b)),
            "rescate": rescate, "danio": danio, "neto": rescate - danio,
        })
        filas_ctrl.append({"brazo": nombre, "delta_vs_v30": round(cv - CV_V30, 7),
                           "delta_sobre_ruido": None, "shuffle_control": nombre == "barajado"})
    tabla_corr = pd.DataFrame(filas_corr)
    real = float(tabla_corr.loc[tabla_corr["brazo"] == "real", "delta_vs_v30"].iloc[0])
    for nombre in ("barajado", "ruido", "con_intercepto"):
        delta = float(tabla_corr.loc[tabla_corr["brazo"] == nombre, "delta_vs_v30"].iloc[0])
        filas_ctrl.append({"brazo": nombre, "delta_vs_v30": round(delta, 7),
                           "delta_sobre_ruido": None,
                           "shuffle_control": nombre == "barajado",
                           "atribuible_al_residuo": round(real - delta, 7)})
    for f in filas_corr:
        print(f"  {f['brazo']:14s} {f['corrected_cv']:.7f}  {f['delta_vs_v30']:+.7f}  "
              f"folds+ {f['folds_positivos']}/5  neto {f['neto']:+,}", flush=True)
    tabla_corr.to_csv(CORRECCIONES_CSV, index=False)
    pd.DataFrame([f for f in filas_ctrl if f.get("atribuible_al_residuo") is not None]
                 ).to_csv(CONTROLES_CSV, index=False)

    deltas = {f["brazo"]: f["delta_vs_v30"] for f in filas_corr}
    atrib = deltas["real"] - deltas["barajado"]
    print(f"\n  real - barajado = {atrib:+.7f}  "
          f"(atribuible a la correspondencia del residuo)", flush=True)
    print(f"  real - ruido   = {deltas['real'] - deltas['ruido']:+.7f}  "
          f"(atribuible a la senal y no a la correccion extra)", flush=True)
    print(f"  con intercepto vs sin = {deltas['con_intercepto'] - deltas['real']:+.7f}  "
          f"(si domina, es efecto de calibracion)", flush=True)
    print(f"\n  betas elegidos por fold (inner_val): {betas_por_fold}", flush=True)
    print(f"  variabilidad de entrenamiento de P00: {VARIABILIDAD}", flush=True)
    for f in filas_corr:
        print(f"  ratio |delta| de {f['brazo']:14s} / variabilidad = "
              f"{abs(f['delta_vs_v30']) / VARIABILIDAD:>7.2f}", flush=True)

    RESULTADO.write_text(json.dumps({
        "fase": "P33+ 01 residualizacion, control barajado y correccion minima",
        "fecha": FECHA, "experimenta": True, "baseline_v30": CV_V30,
        "candidato": CANDIDATO,
        "p05_informacion_del_residuo": {
            "auc_medio_residuo_vs_ovr": round(media_residuo, 5),
            "por_clase": por_clase.reset_index().to_dict("records"),
            "lectura": (
                "por encima de 0,50 el residuo conserva informacion de edad; en torno a 0,50 no "
                "conserva nada y las correcciones de abajo serian ruido con forma matematica"),
        },
        "p07_correcciones": tabla_corr.to_dict("records"),
        "p06_atribucion": {
            "real_menos_barajado": round(atrib, 7),
            "real_menos_ruido": round(deltas["real"] - deltas["ruido"], 7),
            "con_intercepto_menos_sin": round(deltas["con_intercepto"] - deltas["real"], 7),
            "betas_por_fold": betas_por_fold,
        },
        "p19_variabilidad": {
            "variabilidad_de_entrenamiento": VARIABILIDAD,
            "ratios": {f["brazo"]: round(abs(f["delta_vs_v30"]) / VARIABILIDAD, 3)
                       for f in filas_corr},
        },
        "protocolo": {
            "estructura": "outer_train se parte en inner_train (75%) e inner_val (25%). a_c y b_c "
                          "se ajustan en inner_train, beta y alpha se eligen en inner_val, y "
                          "outer_val solo recibe la correccion ya fijada",
            "betas_explorados": list(BETAS),
            "eps": EPS,
            "semilla": SEMILLA,
        },
        "restricciones": {"usa_test": False, "usa_red": False, "outer_val_en_el_ajuste": False},
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
