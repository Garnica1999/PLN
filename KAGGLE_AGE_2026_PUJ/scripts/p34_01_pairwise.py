"""P34+ - 01: P03 a P14. Scores pairwise, residualizacion local, correccion y controles.

## La correccion que abre la fase

P34 apoya su hipotesis en que "el 90 % de los confusiones de V30 son entre bandas adyacentes".
**Ese 90 % era falso.** P02 de esta fase lo midio sobre **todos** los 65.403 errores:

```text
adyacentes          39.202   59,94 %
saltos no adyacentes 26.201  40,06 %
```

El numero de V31 salio de mirar las **diez confusiones mas frecuentes**, que por construccion son
adyacentes, y de ahi salio "90 % no adyacente = 10 %". Contado sobre el total, es 59,94 %.

**Esto no mata la hipotesis**, pero le quita bastante margen: el techo de lo que se puede ganar
resolviendo solo pares adyacentes es mas bajo de lo que se suponia. Y deja una correccion
pendiente en el registro de V31, que se escribe en el artefacto.

## Los dos scores, y por que no se mezclan

```text
d_V30(i,j) = log((P_V30[i] + 1e-6) / (P_V30[j] + 1e-6))     sobre PROBABILIDADES
d_C(i,j)   = score_C[i] - score_C[j]                          sobre decision_function CRUDO
```

La diferencia en el operador es la correcta, no un descuido: en un ratio de probabilidades el
denominador es una probabilidad, y en una diferencia de decision_function cada termino es una
puntuacion de margen. Forzar el mismo operador sobre scores de naturaleza distinta es el error
analogo al de P32.

## La correccion local y su puerta

P05 a P07 definen una correccion estrictamente local, y la limitacion es el punto:

- solo actua cuando `top1` y `top2` de V30 son **adyacentes**;
- solo puede **intercambiar esas dos clases**;
- **no puede** llevar la fila a una tercera clase ni a una no adyacente.

Eso la hace muy distinta de V28-03, que **sustituia** el multiclase por cinco binarios y perdio
0,067553. Aqui V30 decide, y la correccion solo reordena las dos clases que V30 ya puso en
disputa.

## Los controles que deciden

Un residuo pequeno puede mejorar el F1 por el motivo equivocado, asi que los tres brazos van
juntos y el brief es explicito sobre cual manda:

```text
real > barajado   ->  la correspondencia fila-residuo aporta
real > aleatorio  ->  la senal aporta, y no solo que se pueda sumar un termino mas
```

Si `real <= aleatorio`, el resultado se clasifica INVALIDATED aunque `real > barajado`.
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

RESULTADO = cfg.EXPERIMENTS / "p34_01_pairwise.json"
SCORES_CSV = cfg.REPORTS / "p34_pairwise_scores.csv"
RESIDUALS_CSV = cfg.REPORTS / "p34_pairwise_residuals.csv"
RESULTADOS_CSV = cfg.REPORTS / "p34_pairwise_results.csv"
CONTROLES_CSV = cfg.REPORTS / "p34_pairwise_controls.csv"
FECHA = "2026-10-03"
CV_V30 = 0.6134471
N_CLASES = 6
EPS = 1e-6
SEMILLA = 20261003
CANDIDATO = "ortografia+composicion+discurso"
VARIABILIDAD = 0.0008876
UMBRAL_ALTA_CONFIANZA = 0.60
PAREJAS = {"A1": (0, 1), "A2": (1, 2), "A3": (2, 3), "A4": (3, 4), "A5": (4, 5)}
BETAS = (-1.0, -0.6, -0.4, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0)


def _macro(clase: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(clase, pred, average="macro", labels=list(range(N_CLASES))))


def main() -> int:
    """Ejecuta P03 a P14 sobre las cinco parejas adyacentes."""
    autotest_numerico()
    inicio = time.time()
    carpeta = ROOT / "artifacts" / "p33_scores" / CANDIDATO.replace("+", "_")
    clase = np.load(carpeta / "y_true.npy")
    fold_idx = np.load(carpeta / "fold_idx.npy")
    p_v30 = np.load(carpeta / "v30_prob.npy")
    # **decision_function CRUDO**, no el softmax persistido. Es lo que manda el brief.
    score_c = np.load(cfg.FEATURES / "v31_canales_oof.npz")[CANDIDATO].astype(np.float64)
    n = len(clase)
    print(f"=== P34 sobre {CANDIDATO} ===", flush=True)
    print(f"  V30 {_macro(clase, np.argmax(p_v30, axis=1)):.7f} | canonico {CV_V30}", flush=True)
    print(f"  score_C: decision_function crudo, rango {score_c.min():.3f} a "
          f"{score_c.max():.3f}, suma de fila NEGATIVA ({score_c.sum(axis=1).max():.3f})", flush=True)

    # ---------------------------------------------------------------- P02/P05: puerta
    orden_v30 = np.argsort(-p_v30, axis=1)
    top1 = orden_v30[:, 0]
    top2 = orden_v30[:, 1]
    adyacente_top = np.abs(top1 - top2) == 1
    print("\n=== P05 filas donde top1 y top2 de V30 son adyacentes ===", flush=True)
    print(f"  {int(adyacente_top.sum()):,} de {n:,} ({adyacente_top.mean():.2%})", flush=True)
    v30_pred = top1.copy()
    v30_conf = p_v30[np.arange(n), top1]

    # ---------------------------------------------------------------- P03: scores pairwise
    d_v30 = {k: np.log((p_v30[:, i] + EPS) / (p_v30[:, j] + EPS)) for k, (i, j) in PAREJAS.items()}
    d_c = {k: score_c[:, i] - score_c[:, j] for k, (i, j) in PAREJAS.items()}
    filas_scores = []
    for k, (i, j) in PAREJAS.items():
        filas_scores.append({
            "par": k, "clase_i": cfg.CLASSES[i], "clase_j": cfg.CLASSES[j],
            "d_v30_media": round(float(d_v30[k].mean()), 5),
            "d_v30_std": round(float(d_v30[k].std()), 5),
            "d_c_media": round(float(d_c[k].mean()), 5),
            "d_c_std": round(float(d_c[k].std()), 5),
            "corr_d_c_d_v30": round(float(np.corrcoef(d_c[k], d_v30[k])[0, 1]), 5),
            "acuerdo_signo": round(float((np.sign(d_c[k]) == np.sign(d_v30[k])).mean()), 5),
            "operador_v30": "log((P_i+eps)/(P_j+eps)) sobre probabilidad",
            "operador_c": "score_i - score_j sobre decision_function crudo",
        })
    tabla_scores = pd.DataFrame(filas_scores)
    tabla_scores.to_csv(SCORES_CSV, index=False)
    print("\n=== P03 scores pairwise ===", flush=True)
    for _, f in tabla_scores.iterrows():
        print(f"  {f['par']} {f['clase_i']:>6s}/{f['clase_j']:<6s} corr {f['corr_d_c_d_v30']:>7.4f} "
              f"| acuerdo de signo {f['acuerdo_signo']:.4f}", flush=True)

    # ---------------------------------------------------------------- P03-P04: residual
    rng = np.random.default_rng(SEMILLA)
    resid = {k: np.zeros(n) for k in PAREJAS}
    betas = {}
    filas_res = []
    for outer in range(5):
        ent = np.flatnonzero(fold_idx != outer)
        y_ent = clase[ent]
        partes_tr, partes_va = [], []
        for c in range(N_CLASES):
            idx = ent[y_ent == c]
            idx = idx[rng.permutation(len(idx))]
            corte = int(0.75 * len(idx))
            partes_tr.append(idx[:corte])
            partes_va.append(idx[corte:])
        inner_tr, inner_va = np.concatenate(partes_tr), np.concatenate(partes_va)

        for k, (i, j) in PAREJAS.items():
            # P03: a y b se ajustan SOLO en inner_train.
            x, y = d_v30[k][inner_tr], d_c[k][inner_tr]
            var = float(x.var())
            b = float(((x - x.mean()) * (y - y.mean())).mean() / var) if var > 1e-12 else 0.0
            a = float(y.mean() - b * x.mean())
            resid[k] = d_c[k] - (a + b * d_v30[k])
            filas_res.append({
                "par": k, "fold": outer, "a": round(a, 5), "b": round(b, 5),
                "residuo_media": round(float(resid[k][inner_va].mean()), 5),
                "residuo_std": round(float(resid[k][inner_va].std()), 5),
                "corr_residuo_d_v30": round(float(np.corrcoef(resid[k][inner_va],
                                                              d_v30[k][inner_va])[0, 1]), 5),
            })
        # P06: beta se elige en inner_val, con la correccion local y la puerta de adyacencia.
        mejor, mejor_cv = 0.0, _macro(clase[inner_va], v30_pred[inner_va])
        for beta in BETAS:
            prueba = aplica_correccion(inner_va, d_v30, resid, top1, top2, adyacente_top, beta)
            valor = _macro(clase[inner_va], prueba[inner_va])
            if valor > mejor_cv:
                mejor_cv, mejor = valor, beta
        betas[outer] = mejor
        print(f"  fold {outer}: beta {mejor:+.2f} | F1 inner_val {mejor_cv:.6f} "
              f"(V30 {_macro(clase[inner_va], v30_pred[inner_va]):.6f})", flush=True)

    tabla_res = pd.DataFrame(filas_res)
    tabla_res.to_csv(RESIDUALS_CSV, index=False)

    # ---------------------------------------------------------------- P13: AUC por pareja
    print("\n=== P13 AUC del residuo por pareja (¿dice cual de las dos es?) ===", flush=True)
    filas_auc = []
    for k, (i, j) in PAREJAS.items():
        # Subconjunto donde la verdad cae en la pareja, y la decision V30 se reduce a i vs j.
        en_pareja = np.flatnonzero((clase == i) | (clase == j))
        # En este subconjunto, "acertar" significa que V30 no se equivoca dentro de la pareja.
        v30_entro = (((top1[en_pareja] == i) & (clase[en_pareja] == i)) |
                     ((top1[en_pareja] == j) & (clase[en_pareja] == j)))
        # El candidato acierta en la pareja cuando el signo de d_C coincide con la clase real:
        # d_C > 0 favorece a i, d_C < 0 favorece a j.
        objetivo = (clase[en_pareja] == i).astype(int)
        cand_entro = (np.sign(d_c[k][en_pareja]) == np.sign(2 * objetivo - 1))
        filas_auc.append({
            "par": k, "n_filas_en_la_pareja": int(len(en_pareja)),
            "v30_acierta_en_la_pareja": round(float(v30_entro.mean()), 5),
            "candidato_acierta_en_la_pareja": round(float(cand_entro.mean()), 5),
            "auc_residuo": round(float(roc_auc_score(v30_entro, resid[k][en_pareja])), 5),
            "auc_d_v30": round(float(roc_auc_score(v30_entro, d_v30[k][en_pareja])), 5),
        })
    tabla_auc = pd.DataFrame(filas_auc)
    for _, f in tabla_auc.iterrows():
        print(f"  {f['par']} n={f['n_filas_en_la_pareja']:>6,d} | V30 {f['v30_acierta_en_la_pareja']:.4f} "
              f"| cand {f['candidato_acierta_en_la_pareja']:.4f} | AUC residuo "
              f"{f['auc_residuo']:.4f} | AUC d_V30 {f['auc_d_v30']:.4f}", flush=True)

    # ---------------------------------------------------------------- P08-P10: correccion
    print("\n=== P08-P10 correccion local + controles ===", flush=True)
    oof = {b: v30_pred.copy() for b in ("real", "barajado", "aleatorio", "v30")}
    for outer in range(5):
        va = np.flatnonzero(fold_idx == outer)
        beta = betas[outer]
        oof["real"][va] = aplica_correccion(va, d_v30, resid, top1, top2, adyacente_top, beta)[va]
        # Los controles tienen que ser vectores de la MISMA longitud que el residuo real, porque
        # la correccion se aplica con una mascara de filas de todo el dataset. Se baraja el
        # residuo dentro de `va` y se deja intacto fuera, que es lo que significa "permutar la
        # correspondencia fila-residuo sin tocar los folds".
        r_bar = {k: resid[k].copy() for k in PAREJAS}
        for k in PAREJAS:
            r_bar[k][va] = resid[k][va][rng.permutation(len(va))]
        oof["barajado"][va] = aplica_correccion(va, d_v30, r_bar, top1, top2, adyacente_top,
                                                beta)[va]
        r_ale = {k: resid[k].copy() for k in PAREJAS}
        for k in PAREJAS:
            r_ale[k][va] = rng.normal(resid[k][va].mean(), resid[k][va].std(), size=len(va))
        oof["aleatorio"][va] = aplica_correccion(va, d_v30, r_ale, top1, top2, adyacente_top,
                                                 beta)[va]

    filas = []
    for b, pred in oof.items():
        cv = _macro(clase, pred)
        folds = [_macro(clase[fold_idx == k], pred[fold_idx == k]) for k in range(5)]
        base = [_macro(clase[fold_idx == k], v30_pred[fold_idx == k]) for k in range(5)]
        ok_c, ok_v = pred == clase, v30_pred == clase
        filas.append({
            "brazo": b, "corrected_cv": round(cv, 7), "delta_vs_v30": round(cv - CV_V30, 7),
            "fold_scores": json.dumps([round(v, 6) for v in folds]),
            "folds_positivos": int(sum(1 for x, y in zip(folds, base) if x > y)),
            "rescate": int((~ok_v & ok_c).sum()), "danio": int((ok_v & ~ok_c).sum()),
            "neto": int((~ok_v & ok_c).sum()) - int((ok_v & ~ok_c).sum()),
        })
    tabla_fin = pd.DataFrame(filas)
    tabla_fin.to_csv(RESULTADOS_CSV, index=False)
    for _, f in tabla_fin.iterrows():
        print(f"  {f['brazo']:10s} {f['corrected_cv']:.7f}  {f['delta_vs_v30']:+.7f}  "
              f"folds+ {f['folds_positivos']}/5  rescate {f['rescate']:,} danio {f['danio']:,} "
              f"neto {f['neto']:+,}", flush=True)

    deltas = {f["brazo"]: f["delta_vs_v30"] for _, f in tabla_fin.iterrows()}
    # ---------------------------------------------------------------- P09: por pareja
    print("\n=== P09 rescate y danio por pareja (correccion real) ===", flush=True)
    filas_par = []
    for k, (i, j) in PAREJAS.items():
        en = (clase == i) | (clase == j)
        actua = en & adyacente_top
        cambio = actua & (oof["real"] != v30_pred)
        antes = int((v30_pred[actua] == clase[actua]).sum())
        despues = int((oof["real"][actua] == clase[actua]).sum())
        filas_par.append({
            "par": k, "filas_adyacentes": int(actua.sum()),
            "filas_corregidas": int(cambio.sum()),
            "aciertos_antes": antes, "aciertos_despues": despues,
            "rescate": int((cambio & (v30_pred != clase) & (oof["real"] == clase)).sum()),
            "danio": int((cambio & (v30_pred == clase) & (oof["real"] != clase)).sum()),
        })
        filas_par[-1]["neto"] = filas_par[-1]["rescate"] - filas_par[-1]["danio"]
        print(f"  {k} {cfg.CLASSES[i]:>6s}/{cfg.CLASSES[j]:<6s} adyacentes "
              f"{actua.sum():>6,d} | rescata {filas_par[-1]['rescate']:>5,d} | dania "
              f"{filas_par[-1]['danio']:>5,d} | neto {filas_par[-1]['neto']:>+6,d}", flush=True)

    # ---------------------------------------------------------------- P11-P12
    error = v30_pred != clase
    s1 = np.ones(n, dtype=bool)
    s2 = adyacente_top
    s3 = adyacente_top & error
    filas_loc = []
    for nombre, m in (("S1 todas las filas", s1), ("S2 top1/top2 adyacentes", s2),
                      ("S3 error adyacente de V30", s3)):
        filas_loc.append({
            "escenario": nombre, "n": int(np.sum(m)),
            "f1_v30": round(_macro(clase[m], v30_pred[m]), 5),
            "f1_corregido": round(_macro(clase[m], oof["real"][m]), 5),
            "delta": round(_macro(clase[m], oof["real"][m]) - _macro(clase[m], v30_pred[m]), 5),
        })
    print("\n=== P11 localidad ===", flush=True)
    for f in filas_loc:
        print(f"  {f['escenario']:26s} n={f['n']:>7,d}  V30 {f['f1_v30']:.5f} -> "
              f"corregido {f['f1_corregido']:.5f}  {f['delta']:+.5f}", flush=True)

    alta = adyacente_top & error & (v30_conf >= UMBRAL_ALTA_CONFIANZA)
    baja = adyacente_top & error & (v30_conf < UMBRAL_ALTA_CONFIANZA)
    resc_alto = int((alta & (oof["real"] == clase)).sum())
    dano_alto = int((~alta & adyacente_top & (v30_pred == clase) & (oof["real"] != clase)).sum())
    print("\n=== P12 rescate de alta confianza ===", flush=True)
    print(f"  errores adyacentes de alta confianza: {int(alta.sum()):,}", flush=True)
    print(f"  errores adyacentes de baja confianza : {int(baja.sum()):,}", flush=True)
    print(f"  rescatados de alta confianza: {resc_alto:,} | danados de alta confianza: "
          f"{dano_alto:,} | neto {resc_alto - dano_alto:+,}", flush=True)

    # ---------------------------------------------------------------- P14
    filas_por_fold_beta = [{"fold": o, "beta_shared": betas[o]} for o in range(5)]

    # ---------------------------------------------------------------- P22: clasificacion
    d_real = deltas["real"]
    d_bar = deltas["barajado"]
    d_ale = deltas["aleatorio"]
    filas_pos = int(tabla_fin.loc[tabla_fin["brazo"] == "real", "folds_positivos"].iloc[0])
    auc_alguno = bool((tabla_auc["auc_residuo"] > 0.5).any())
    if d_real <= d_ale:
        mecanismo = "INVALIDATED"
        motivo = (f"la correccion con senal real ({d_real:+.7f}) no supera a la correccion con "
                  f"ruido aleatorio ({d_ale:+.7f}). No hay evidencia de que la correspondencia "
                  f"del residuo sea lo que aporta")
    elif d_real <= 0:
        mecanismo = ("COMPLEMENTARY_BUT_NOT_TRANSFERABLE" if auc_alguno else "CLOSED")
        motivo = (f"el residuo dice algo (AUC > 0,50 en al menos una pareja) pero la correccion "
                  f"no mejora el F1: {d_real:+.7f}")
    elif filas_pos < 4:
        mecanismo = "MARGINAL" if abs(d_real) >= 0.0005 else "COMPLEMENTARY_BUT_NOT_TRANSFERABLE"
        motivo = f"mejora de {d_real:+.7f} pero solo en {filas_pos}/5 folds"
    else:
        mecanismo = "PROMISING"
        motivo = f"mejora de {d_real:+.7f} con {filas_pos}/5 folds y controles favorables"
    print(f"\n=== P22 mecanismo: {mecanismo} ===", flush=True)
    print(f"  {motivo}", flush=True)
    print(f"  real {d_real:+.7f} | barajado {d_bar:+.7f} | aleatorio {d_ale:+.7f}", flush=True)
    print(f"  real - aleatorio = {d_real - d_ale:+.7f}  "
          f"| real - barajado = {d_real - d_bar:+.7f}", flush=True)
    print(f"  ratio |delta| / variabilidad de entrenamiento = "
          f"{abs(d_real) / VARIABILIDAD:.2f}", flush=True)
    print(f"  AUC del residuo por encima de 0,50 en alguna pareja: {auc_alguno}", flush=True)

    pd.DataFrame([{"brazo": "real", "delta": d_real, "delta_menos_barajado": round(d_real - d_bar, 7),
                   "delta_menos_aleatorio": round(d_real - d_ale, 7),
                   "shuffle_control_pasa": bool(d_real > d_bar),
                   "random_control_pasa": bool(d_real > d_ale)}]
                 ).to_csv(CONTROLES_CSV, index=False)

    RESULTADO.write_text(json.dumps({
        "fase": "P34+ 01 scores pairwise, residualizacion local y correccion", "fecha": FECHA,
        "experimenta": True, "baseline_v30": CV_V30, "candidato": CANDIDATO,
        "correccion_del_registro": {
            "afirmacion_incorrecta": "V31 reporto 90% de confusiones adyacentes",
            "valor_real": 0.5994,
            "por_que_se_equivoco": (
                "V31 conto la fraccion no adyacente entre las DIEZ confusiones mas frecuentes, que "
                "por construccion son casi todas adyacentes. Contado sobre los 65.403 errores "
                "completos es 59,94% adyacente y 40,06% salto"),
            "efecto_sobre_la_hipotesis": (
                "la hipotesis de P34 sigue siendo valida en su forma (las adyacentes son la "
                "mayoria) pero su techo es mucho mas bajo de lo que se suponia: solo el 59,94% de "
                "los errores son elegibles para una correccion pairwise"),
        },
        "p03_scores": tabla_scores.to_dict("records"),
        "p04_residuales": tabla_res.to_dict("records"),
        "p13_auc_por_pareja": tabla_auc.to_dict("records"),
        "p10_resultados": tabla_fin.to_dict("records"),
        "p09_por_pareja": filas_par,
        "p11_localidad": filas_loc,
        "p12_rescate_alta_confianza": {"errores_adyacentes_alta": int(alta.sum()),
                                       "errores_adyacentes_baja": int(baja.sum()),
                                       "rescatados_alta": resc_alto, "danados_alta": dano_alto,
                                       "neto_alta": resc_alto - dano_alto},
        "p14_betas": filas_por_fold_beta,
        "p22_mecanismo": mecanismo, "p22_motivo": motivo,
        "p20_variabilidad": {"variabilidad": VARIABILIDAD,
                             "ratio_delta": round(abs(d_real) / VARIABILIDAD, 3)},
        "protocolo": {
            "correccion": "solo intercambia top1 y top2 de V30, y solo si son adyacentes. No puede "
                          "llevar la fila a una tercera clase ni a una no adyacente",
            "a_b_beta": "a y b de d_C = a + b * d_V30 se ajustan en inner_train; beta se elige en "
                        "inner_val; outer_val solo recibe la correccion ya fijada",
            "betas": BETAS, "eps": EPS, "semilla": SEMILLA,
        },
        "restricciones": {"usa_test": False, "usa_holdout": False, "usa_red": False,
                          "divide_score_por_suma": False, "score_c_es_decision_function": True,
                          "outer_val_en_el_ajuste": False, "leaderboard_para_ajustar": False},
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


def aplica_correccion(idx, d_v30, resid, top1, top2, adyacente, beta):
    """Correccion local: reordena top1 y top2 **solo si** son adyacentes.

    No es un clasificador ordinal. V30 decide cuales dos clases estan en disputa, y esta funcion
    solo decide el orden entre esas dos. Las otras cuatro clases no se tocan.
    """
    pred = top1.copy()
    mascara = adyacente & np.isin(np.arange(len(top1)), idx)
    # El margen se orienta hacia la clase que V30 puso primera: si top1 < top2 se usa d(i,j) con
    # su signo, y si top1 > top2 se usa el mismo margen con el signo cambiado. Asi una sola
    # comparacion "mayor que cero gana" basta para decidir el orden, sin ramas por direccion.
    for k, (i, j) in PAREJAS.items():
        sel = mascara & (((top1 == i) & (top2 == j)) | ((top1 == j) & (top2 == i)))
        if not sel.any():
            continue
        base = np.where(top1[sel] == i, d_v30[k][sel], -d_v30[k][sel])
        corregido = base + beta * resid[k][sel]
        se_invierte = corregido < 0
        pred[sel] = np.where(se_invierte, top2[sel], top1[sel])
    return pred


if __name__ == "__main__":
    raise SystemExit(main())
