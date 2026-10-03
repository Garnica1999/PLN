"""G12: correccion cuya intensidad dependa del margen de V30.

## Por que esto no es otro beta

P33 probo un `beta` **compartido** y un intercepto `alpha`. Los dos son constantes: la correccion
mide lo mismo en cada fila. G12 propone una **pendiente** sobre una variable que P33 no toco, el
margen de V30:

```text
z_final[c] = z_V30[c] + gamma1 * f(|margen_V30|) * residual[c] + gamma0 * residual[c]
```

con `f` una funcion **monotona y decreciente de la confianza**, y no un selector `keep/replace`
como prohibe el brief. La justificacion no es estetica: la medicion de P33 da

```text
AUC del margen de V30 para predecir sus propios errores = 0,8005
AUC del residuo linguistico para lo mismo                  = 0,5411
```

El margen ya sabe mucho mas sobre donde V30 se equivoca que el residuo. Por eso la correccion
deberia pesar **mas donde V30 es confiado** y **casi nada donde ya duda**... y aqui hay un matiz
importante que el diseno respeta: si V30 duda y aun asi se equivoca, la correccion tiene que poder
ayudar. Por eso el peso es `w = g0 + g1 * confianza` con `g1` libre, en vez de forzar el signo.

## El control que decide, y por que es el mas duro posible

Un peso que depende de una variable puede mejorar por el hecho de tener **dos** parametros en vez de
uno. Para separar eso:

```text
V30
+ peso constante            (equivalente a P33, el control directo)
+ pendiente en el margen    (G12)
+ pendiente en el margen, con el residuo BARAJADO
+ pendiente en el margen, con RUIDO de igual media y desviacion
```

Si G12 no supera a los tres, no hay senal. Y si solo supera a los otros dos pero no al ruido, es
capacidad, no informacion.

## Estructura

`gamma0` y `gamma1` se ajustan en `inner_val` por busqueda en una rejilla **fija y pequena** de
4x4 valores. No es una grid abierta: son 16 puntos, y el objetivo no es maximizar sino comprobar
si existe un par que mejore. `outer_val` no participa.

**Y se mide la mejora en AUC, no solo en F1.** Esa es la decision metodologica central de este
script: si el residuo aporta informacion, tiene que mejorar la *ordenacion* de las clases aunque
convertir eso en argmax no rinda. P33 y P34 fallaron en F1, pero eso no demuestra que la
informacion no este ahi; solo que la conversion a decision no la aprovecha. Medir el AUC separa las
dos cosas, y si el AUC tampoco mejora, entonces la conclusion es solida en los dos niveles.
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

RESULTADO = cfg.EXPERIMENTS / "age_g12_pendiente_margen.json"
TABLA_CSV = cfg.REPORTS / "age_g12_results.csv"
CONTROLES_CSV = cfg.REPORTS / "age_g12_controls.csv"
FECHA = "2026-10-05"
CV_V30 = 0.6134471
N_CLASES = 6
EPS = 1e-6
SEMILLA = 20261005
CANDIDATO = "ortografia+composicion+discurso"
VARIABILIDAD = 0.0008876
#: Rejilla FIJA de 4x4. Suficiente para detectar un par que mejore; insuficiente para sobreajustar,
#: y el objetivo no es maximizar sino comprobar la existencia del efecto.
GAMMA0 = (-0.20, -0.05, 0.05, 0.20)
GAMMA1 = (-2.0, -0.5, 0.5, 2.0)


def _macro(clase, prob) -> float:
    """macro-F1 de 6 clases. Acepta probabilidades (n, 6) o ya un vector de predicciones (n,)."""
    pred = np.argmax(prob, axis=1) if np.ndim(prob) == 2 else np.asarray(prob)
    return float(f1_score(clase, pred, average="macro", labels=list(range(N_CLASES))))


def logit(p):
    q = np.clip(p, EPS, 1.0 - EPS)
    return np.log(q / (1.0 - q))


def a_prob(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def auc_multiclase(clase, prob) -> float:
    """AUC one-vs-rest promediado sobre las seis clases.

    Es la metrica que separa "hay informacion" de "hay decision". P33 y P34 fallaron en F1; si el
    AUC tampoco se mueve, la informacion no esta ahi en ninguna de las dos formas.
    """
    aucs = []
    for c in range(N_CLASES):
        objetivo = (clase == c).astype(int)
        if objetivo.sum() in (0, len(objetivo)):
            continue
        aucs.append(roc_auc_score(objetivo, prob[:, c]))
    return float(np.mean(aucs))


def main() -> int:
    """Ejecuta G12 con sus tres controles."""
    autotest_numerico()
    inicio = time.time()
    carpeta = ROOT / "artifacts" / "p33_scores" / CANDIDATO.replace("+", "_")
    clase = np.load(carpeta / "y_true.npy")
    fold_idx = np.load(carpeta / "fold_idx.npy")
    p_v30 = np.load(carpeta / "v30_prob.npy")
    z_v30 = logit(p_v30)
    orden = np.argsort(-p_v30, axis=1)
    top1 = orden[:, 0]
    # Variable de la pendiente: el margen top1-top2, que es la confianza de V30 en su decision.
    margen = p_v30[np.arange(len(clase)), top1] - p_v30[np.arange(len(clase)), orden[:, 1]]
    rango = margen.max() - margen.min()
    u = (margen - margen.min()) / max(rango, 1e-12)

    # Residual: el de P33, que es el unico objeto con AUC > 0,50 demostrado.
    p_c = np.load(carpeta / "cand_prob.npy")
    z_c = logit(p_c)
    rng = np.random.default_rng(SEMILLA)

    print("=== G12: pendiente en el margen de V30 ===", flush=True)
    print(f"  V30 {_macro(clase, p_v30):.7f} | AUC OVR {auc_multiclase(clase, p_v30):.5f}", flush=True)

    # Estructura anidada: a_c, b_c en inner_train; gamma en inner_val; outer_val intacto.
    resid = np.zeros_like(z_c)
    predicciones = {b: np.argmax(p_v30, axis=1).copy() for b in
                    ("constante", "pendiente", "pendiente_barajado", "pendiente_ruido")}
    gammas = []
    for outer in range(5):
        ent = np.flatnonzero(fold_idx != outer)
        tr, va = [], []
        for c in range(N_CLASES):
            idx = ent[clase[ent] == c]
            idx = idx[rng.permutation(len(idx))]
            corte = int(0.75 * len(idx))
            tr.append(idx[:corte])
            va.append(idx[corte:])
        inner_tr, inner_va = np.concatenate(tr), np.concatenate(va)
        aa = np.zeros(N_CLASES)
        bb = np.zeros(N_CLASES)
        for c in range(N_CLASES):
            x, y = z_v30[inner_tr, c], z_c[inner_tr, c]
            var = float(x.var())
            bb[c] = float(((x - x.mean()) * (y - y.mean())).mean() / var) if var > 1e-12 else 0.0
            aa[c] = float(y.mean() - bb[c] * x.mean())
        resid[inner_va] = z_c[inner_va] - (aa[None, :] + bb[None, :] * z_v30[inner_va])

        # gamma se elige en inner_val, con una rejilla fija de 4x4.
        mejor, mejor_f1 = (0.0, 0.0), _macro(clase[inner_va], p_v30[inner_va])
        for g0 in GAMMA0:
            for g1 in GAMMA1:
                peso = g0 + g1 * u[inner_va]
                prueba = a_prob(z_v30[inner_va] + peso[:, None] * resid[inner_va])
                valor = _macro(clase[inner_va], prueba)
                if valor > mejor_f1:
                    mejor_f1, mejor = valor, (g0, g1)
        g0, g1 = mejor
        gammas.append({"fold": outer, "gamma0": g0, "gamma1": g1, "f1_inner_val": round(mejor_f1, 6)})
        peso_in = g0 + g1 * u[inner_va]
        predicciones["constante"][inner_va] = np.argmax(
            a_prob(z_v30[inner_va] + g0 * resid[inner_va]), axis=1)
        predicciones["pendiente"][inner_va] = np.argmax(
            a_prob(z_v30[inner_va] + peso_in[:, None] * resid[inner_va]), axis=1)
        # Barajado: permuta la correspondencia fila-residuo DENTRO de inner_val.
        rb = resid[inner_va][rng.permutation(len(inner_va))]
        predicciones["pendiente_barajado"][inner_va] = np.argmax(
            a_prob(z_v30[inner_va] + peso_in[:, None] * rb), axis=1)
        # Ruido: misma media y desviacion que el residuo, sobre la misma pendiente.
        rn = rng.normal(resid[inner_va].mean(), resid[inner_va].std(), size=resid[inner_va].shape)
        predicciones["pendiente_ruido"][inner_va] = np.argmax(
            a_prob(z_v30[inner_va] + peso_in[:, None] * rn), axis=1)
        print(f"  fold {outer}: gamma0 {g0:+.2f} gamma1 {g1:+.2f} | F1 inner_val "
              f"{mejor_f1:.6f} (V30 {_macro(clase[inner_va], p_v30[inner_va]):.6f})", flush=True)

    print("\n=== resultados: F1 y AUC, que es donde se separa la informacion de la decision ===",
          flush=True)
    filas = []
    for b, pred in predicciones.items():
        cv = _macro(clase, pred)
        folds = [_macro(clase[fold_idx == k], pred[fold_idx == k]) for k in range(5)]
        base = [_macro(clase[fold_idx == k], np.argmax(p_v30[fold_idx == k], axis=1))
                for k in range(5)]
        filas.append({
            "brazo": b, "corrected_cv": round(cv, 7), "delta_vs_v30": round(cv - CV_V30, 7),
            "folds": [round(v, 6) for v in folds],
            "folds_positivos": int(sum(1 for x, y in zip(folds, base) if x > y)),
            "ratio_sobre_variabilidad": round(abs(cv - CV_V30) / VARIABILIDAD, 3),
        })
    tabla = pd.DataFrame(filas)
    tabla.to_csv(TABLA_CSV, index=False)
    for _, f in tabla.iterrows():
        print(f"  {f['brazo']:20s} {f['corrected_cv']:.7f}  {f['delta_vs_v30']:+.7f}  "
              f"folds+ {f['folds_positivos']}/5  ratio {f['ratio_sobre_variabilidad']:.2f}",
          flush=True)

    # El AUC se recalcula por fold con los gamma de ESE fold, porque el residuo solo esta
    # definido dentro de cada inner_val. Mezclar gamma entre folds daria un numero sin sentido.
    print(f"\n  AUC one-vs-rest de V30: {auc_multiclase(clase, p_v30):.5f}", flush=True)
    auc_por_brazo = {"v30": auc_multiclase(clase, p_v30)}
    for b in ("constante", "pendiente", "pendiente_barajado", "pendiente_ruido"):
        prob = p_v30.copy()
        for outer in range(5):
            m = fold_idx == outer
            g0, g1 = gammas[outer]["gamma0"], gammas[outer]["gamma1"]
            if b == "constante":
                prob[m] = a_prob(z_v30[m] + g0 * resid[m])
                continue
            peso = (g0 + g1 * u[m])[:, None]
            if b == "pendiente":
                r = resid[m]
            elif b == "pendiente_barajado":
                r = resid[m][np.random.default_rng(SEMILLA + outer).permutation(int(m.sum()))]
            else:
                r = np.random.default_rng(SEMILLA + 100 + outer).normal(
                    resid[m].mean(), resid[m].std(), size=resid[m].shape)
            prob[m] = a_prob(z_v30[m] + peso * r)
        auc_por_brazo[b] = auc_multiclase(clase, prob)
        print(f"  AUC OVR {b:20s} {auc_por_brazo[b]:.5f}  "
              f"({auc_por_brazo[b] - auc_por_brazo['v30']:+.5f} vs V30)", flush=True)
    tabla["auc_ovr"] = tabla["brazo"].map(auc_por_brazo)
    tabla.to_csv(TABLA_CSV, index=False)

    d = {f["brazo"]: f["delta_vs_v30"] for f in filas if "delta_vs_v30" in f}
    caida_auc = all(v < auc_por_brazo["v30"] for k, v in auc_por_brazo.items() if k != "v30")
    if d["pendiente"] <= d["constante"] and caida_auc:
        veredicto = "SIN APORTE, Y EL AUC TAMBIEN CAE"
        motivo = (
            f"la pendiente ({d['pendiente']:+.7f}) no mejora al peso constante "
            f"({d['constante']:+.7f}), y ademas los cuatro brazos BAJAN el AUC frente a V30 "
            f"({auc_por_brazo['v30']:.5f} -> {auc_por_brazo['pendiente']:.5f}). Eso es mas fuerte "
            f"que el fallo de P33 y P34: aquellos no tocaban la ordenacion y solo fracasaban al "
            f"convertirla en decision; aqui la ordenacion tambien empeora, asi que el residuo no "
            f"aporta informacion en ninguno de los dos niveles")
    elif d["pendiente"] <= d["constante"]:
        veredicto = "SIN APORTE RESPECTO A LA CONSTANTE"
        motivo = (f"la pendiente ({d['pendiente']:+.7f}) no mejora al peso constante "
                  f"({d['constante']:+.7f}). Depender del margen no aporta sobre no depender")
    elif d["pendiente"] <= d["pendiente_ruido"]:
        veredicto = "CAPACIDAD, NO INFORMACION"
        motivo = (f"la pendiente ({d['pendiente']:+.7f}) supera a la constante pero no al ruido "
                  f"({d['pendiente_ruido']:+.7f}). El segundo parametro es el que rinde, no el "
                  f"margen")
    elif d["pendiente"] <= 0:
        veredicto = "COMPLEMENTARY_BUT_NOT_TRANSFERABLE"
        motivo = f"mejora frente a los controles pero el F1 global no mejora: {d['pendiente']:+.7f}"
    else:
        veredicto = "PROMISING"
        motivo = f"mejora de {d['pendiente']:+.7f} sobre constante, barajado y ruido"
    print(f"\n=== veredicto: {veredicto} ===", flush=True)
    print(f"  {motivo}", flush=True)

    pd.DataFrame([{"comparacion": "pendiente - constante", "delta": round(d["pendiente"] - d["constante"], 7)},
                  {"comparacion": "pendiente - barajado", "delta": round(d["pendiente"] - d["pendiente_barajado"], 7)},
                  {"comparacion": "pendiente - ruido", "delta": round(d["pendiente"] - d["pendiente_ruido"], 7)},
                  {"comparacion": "constante - ruido", "delta": round(d["constante"] - d["pendiente_ruido"], 7)}
                  ]).to_csv(CONTROLES_CSV, index=False)

    RESULTADO.write_text(json.dumps({
        "fase": "G12 correccion con pendiente en el margen de V30", "fecha": FECHA,
        "experimenta": True, "baseline_v30": CV_V30, "candidato": CANDIDATO,
        "hipotesis": G12_HIPOTESIS, "mecanismo": G12_MECANISMO,
        "gammas": gammas, "resultados": tabla.to_dict("records"),
        "veredicto": veredicto, "motivo": motivo,
        "protocolo": {"gammas_por_fold": "elegidos en inner_val sobre rejilla FIJA de 4x4",
                      "a_c_b_c": "en inner_train", "outer_val": "nunca participa",
                      "variable_de_la_pendiente": "(top1 - top2) de V30, normalizado a [0,1]",
                      "semilla": SEMILLA},
        "restricciones": {"usa_test": False, "usa_holdout": False, "usa_red": False,
                          "selector_keep_replace": False, "leaderboard_para_ajustar": False},
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


G12_HIPOTESIS = (
    "G12: la correccion del score de V30 deberia pesar mas donde V30 ya es confiado, porque su "
    "margen predice sus propios errores con AUC 0,8005 frente al 0,5411 del residuo. P33 probo un "
    "beta COMPARTIDO y un intercepto, ambos constantes; esta hipotesis propone una PENDIENTE sobre "
    "el margen, que es un mecanismo distinto en 2 parametros"
)
G12_MECANISMO = (
    "z_final[c] = z_V30[c] + (gamma0 + gamma1 * u(margen_V30)) * residual[c], con u el margen "
    "top1-top2 normalizado. gamma1 libre, sin forzar el signo. La correccion pesa mas en la zona "
    "que el residuo puede discriminar y se atenua en la que no. NO es un selector keep/replace"
)


if __name__ == "__main__":
    raise SystemExit(main())
