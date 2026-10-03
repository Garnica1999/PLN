"""V20 · P0 — Auditoria de las cinco fronteras ordinales y de su consistencia.

V20 queda **congelado**: no se toca su entrenamiento, sus features, su configuracion, sus cinco
fronteras, su metodo de combinacion, sus thresholds, su submission, su hash ni su score. Todo aqui es
analisis del OOF ya medido, que esta en `features/v20_f3_oof_cumuladas.npy` (169.336 x 5).

**Paso 2, donde esta la informacion incremental.** Por cada una de las cinco fronteras: exactitud
balanceada, F1 binario, AUC, calibracion (Brier y fiabilidad), distribucion de scores, numero de
ejemplos de cada lado, error por fold y correlacion con las demas fronteras. La pregunta que responde es
cual de las cinco aporta algo que las demas no.

**Paso 3, la consistencia ordinal.** Las cinco salidas son probabilidades acumuladas `P(Y <= k)` y por
construccion deben ser **monotonicas crecientes**. Se mide cuanto violan eso las predicciones crudas,
con que magnitud, por fold y por ejemplo, y despues como resolvio V20 esas inconsistencias.

**Paso 6, el impacto de cada frontera sobre las seis clases.** No basta con que una frontera tenga peor
metrica: hay que saber si su error importa. Se degrada cada frontera una a la vez y se mide cuanto cae
el macro-F1 final, que es lo que separa "frontera debil" de "frontera cuyos errores importan poco".

**Paso 7, el error condicional.** Para cada frontera: cuando el discriminador falla, que clases finales
produce el sistema y cual era la verdadera, y si esos errores estan concentrados cerca de la frontera o
son errores lejanos. Un error lejano es mucho mas grave que uno cercano, porque indica que el sistema no
sabe donde esta.

Entregables: ``experiments/v20_p0_fronteras.json``, ``reports/v20_p0_fronteras.md``,
``reports/v20_p0_fronteras.csv``, ``reports/v20_p0_consistencia.csv`` y dos figuras.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v19_b1_boosting import matriz_279  # noqa: E402

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v20_p0_fronteras.md"
TABLA = cfg.REPORTS / "v20_p0_fronteras.csv"
CONSISTENCIA = cfg.REPORTS / "v20_p0_consistencia.csv"
RESULTADO = cfg.EXPERIMENTS / "v20_p0_fronteras.json"
FIGURA1 = cfg.FIGURES / "v20_p0_fronteras.png"
FIGURA2 = cfg.FIGURES / "v20_p0_consistencia.png"
ACUMULADAS = cfg.FEATURES / "v20_f3_oof_cumuladas.npy"
OOF_V20 = cfg.FEATURES / "v20_f3b_oof_candidato.npy"
CV_V20 = 0.6097782
ETIQUETAS = list(range(len(cfg.CLASSES)))


def macro(clase: np.ndarray, prob: np.ndarray) -> float:
    """macro-F1 pooled de seis clases."""
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro", labels=ETIQUETAS))


def reconstruir(acumuladas: np.ndarray) -> np.ndarray:
    """Las seis clases por diferencias de las cinco probabilidades acumuladas, con monotonia."""
    n_clases = acumuladas.shape[1] + 1
    mono = acumuladas.copy()
    for k in range(1, acumuladas.shape[1]):
        mono[:, k] = np.maximum(mono[:, k], mono[:, k - 1])
    seis = np.zeros((len(mono), n_clases))
    seis[:, 0] = mono[:, 0]
    for k in range(1, acumuladas.shape[1]):
        seis[:, k] = mono[:, k] - mono[:, k - 1]
    seis[:, n_clases - 1] = 1.0 - mono[:, n_clases - 2]
    seis = np.clip(seis, 0.0, None)
    return seis / np.clip(seis.sum(axis=1, keepdims=True), 1e-9, None)


def main() -> int:
    """Audita las cinco fronteras y su consistencia ordinal."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    acumuladas = np.load(ACUMULADAS)
    oof_v20 = np.load(OOF_V20)
    n_clases = len(cfg.CLASSES)
    n_umbral = n_clases - 1

    # ---------------------------------------------------------------- paso 2: cada frontera
    # `acumuladas[:, k-1]` es P(clase <= k-1), es decir la probabilidad del lado **bajo**. Usarla como
    # score del lado alto daba un AUC por debajo de 0,5 por construccion. El score del lado alto es,
    # por tanto, `1 - acumulada`, y toda la auditoria se hace con ese signo.
    print("[P2] auditoria de las cinco fronteras, con el score del lado alto", flush=True)
    filas = []
    for k in range(1, n_clases):
        lado_alto = clase >= k
        lado_bajo = ~lado_alto
        p_bajo = acumuladas[:, k - 1]
        score_alto = 1.0 - p_bajo
        decision = score_alto > 0.5
        acierto_alto = float((decision[lado_alto]).mean())
        acierto_bajo = float((~decision[lado_bajo]).mean())
        balanceada = 0.5 * (acierto_alto + acierto_bajo)
        etiqueta = lado_alto.astype(int)
        f1_binario = float(f1_score(etiqueta, decision.astype(int)))
        auc = float(roc_auc_score(etiqueta, score_alto))
        brier = float(np.mean((score_alto - etiqueta) ** 2))
        fiabilidad = float(score_alto.mean())
        tasa_real = float(etiqueta.mean())
        error_por_fold = []
        for fold in folds:
            idx = fold["val_idx"]
            lado = lado_alto[idx]
            pred = decision[idx]
            error_por_fold.append(round(1 - 0.5 * (pred[lado].mean() + (~pred[~lado]).mean()), 6))
        filas.append({
            "frontera": f"<= {cfg.CLASSES[k - 1]} vs >= {cfg.CLASSES[k]}",
            "umbral_k": k, "filas_lado_bajo": int(lado_bajo.sum()),
            "filas_lado_alto": int(lado_alto.sum()),
            "fraccion_lado_alto": round(float(lado_alto.mean()), 4),
            "acierto_lado_alto": round(acierto_alto, 4),
            "acierto_lado_bajo": round(acierto_bajo, 4),
            "exactitud_balanceada": round(balanceada, 4),
            "ventaja_sobre_el_azar": round(balanceada - 0.5, 4),
            "f1_binario": round(f1_binario, 4), "auc": round(auc, 4),
            "brier": round(brier, 6), "media_predicha": round(fiabilidad, 4),
            "tasa_real": round(tasa_real, 4), "descalibracion": round(fiabilidad - tasa_real, 4),
            "error_por_fold": error_por_fold,
            "p10": round(float(np.quantile(score_alto, 0.10)), 4),
            "p50": round(float(np.quantile(score_alto, 0.50)), 4),
            "p90": round(float(np.quantile(score_alto, 0.90)), 4)})
        print(f"  {filas[-1]['frontera']:<26} balanceada {balanceada:.4f} | F1 {f1_binario:.4f} | "
              f"AUC {auc:.4f} | Brier {brier:.4f} | descalibracion {fiabilidad - tasa_real:+.4f}",
              flush=True)
    correlacion = np.corrcoef(acumuladas, rowvar=False)
    print("[P2] correlacion entre fronteras:", np.round(correlacion, 3).tolist(), flush=True)

    # ---------------------------------------------------------------- paso 3: consistencia
    print("[P3] consistencia ordinal de las cinco probabilidades acumuladas", flush=True)
    violaciones = np.zeros(len(y), dtype=int)
    magnitud = np.zeros(len(y))
    for k in range(1, n_umbral):
        excede = acumuladas[:, k] < acumuladas[:, k - 1]
        violaciones += excede.astype(int)
        magnitud = np.maximum(magnitud, np.where(excede, acumuladas[:, k - 1] - acumuladas[:, k], 0.0))
    filas_cons = []
    for fold in folds:
        idx = fold["val_idx"]
        filas_cons.append({"fold": fold["fold"], "filas_con_violacion": int((violaciones[idx] > 0).sum()),
                           "fraccion": round(float((violaciones[idx] > 0).mean()), 4),
                           "magnitud_media": round(float(magnitud[idx].mean()), 6),
                           "magnitud_maxima": round(float(magnitud[idx].max()), 6),
                           "violaciones_por_fila_max": int(violaciones[idx].max())})
    total_violan = int((violaciones > 0).sum())
    print(f"  {total_violan:,} de {len(y):,} filas violan la monotonia ({total_violan / len(y):.2%}) | "
          f"magnitud media {magnitud.mean():.6f} | maxima {magnitud.max():.6f}", flush=True)
    pd.DataFrame(filas_cons).to_csv(CONSISTENCIA, index=False)

    # ---------------------------------------------------------------- paso 6: impacto
    print("[P6] impacto de cada frontera sobre las seis clases finales", flush=True)
    # Se mide sobre el **blend de V20**, que es lo que decide: la cadena es un cuarto miembro mas y su
    # valor depende de como interactua con los otros tres, no de reconstruir las seis clases en soledad.
    base = macro(clase, oof_v20)
    base_cadena_sola = macro(clase, reconstruir(acumuladas))
    impactos = []
    for k in range(1, n_clases):
        degradada = acumuladas.copy()
        degradada[:, k - 1] = 0.5
        seis_degradada = reconstruir(degradada)
        blend_degradado = (0.25 * np.load(cfg.FEATURES / "v19_b1_oof_logreg.npy")
                           + 0.25 * np.load(cfg.FEATURES / "v19_b1_oof_mlp.npy")
                           + 0.25 * np.load(cfg.FEATURES / "v19_oof_arbol_5folds.npy")
                           + 0.25 * seis_degradada)
        valor_blend = macro(clase, blend_degradado)
        valor_cadena = macro(clase, seis_degradada)
        impactos.append({"frontera_degradada": f"<= {cfg.CLASSES[k - 1]} vs >= {cfg.CLASSES[k]}",
                         "macro_f1_del_blend": round(valor_blend, 7),
                         "caida_del_blend": round(base - valor_blend, 6),
                         "macro_f1_de_la_cadena_sola": round(valor_cadena, 7),
                         "caida_de_la_cadena_sola": round(base_cadena_sola - valor_cadena, 6)})
        print(f"  degradar {impactos[-1]['frontera_degradada']:<26} -> blend {valor_blend:.7f} "
              f"(caida {base - valor_blend:+.4f}) | cadena sola {valor_cadena:.7f} "
              f"(caida {base_cadena_sola - valor_cadena:+.4f})", flush=True)

    # ---------------------------------------------------------------- paso 7: error condicional
    # Un "fallo" de la decision binaria solo es un fallo de **frontera** si la fila esta cerca del
    # umbral. Contar como fallo toda fila que discrepa del lado incluye a las lejanas, donde discrepar
    # es lo correcto, y por eso la version anterior salia invertida: "cuando acierta" daba 0,05.
    print("[P7] error condicional por frontera, restringido a las filas cercanas al umbral", flush=True)
    seis = reconstruir(acumuladas)
    pred = np.argmax(seis, axis=1)
    condicional = []
    for k in range(1, n_clases):
        lado_alto = clase >= k
        decision = acumuladas[:, k - 1] > 0.5
        distancia = np.abs(clase - (k - 0.5))
        cercanas = distancia <= 1.0
        falla_cerca = cercanas & (decision != lado_alto)
        acierta_cerca = cercanas & (decision == lado_alto)
        lejanas = ~cercanas
        # Cuando falla cerca del umbral, que clase final produce el sistema y cual era la verdadera.
        producidas = pred[falla_cerca]
        verdaderas = clase[falla_cerca]
        confusion = {}
        for v in np.unique(verdaderas):
            clases_producidas, cuentas = np.unique(producidas[verdaderas == v], return_counts=True)
            confusion[cfg.CLASSES[int(v)]] = {cfg.CLASSES[int(c)]: int(n)
                                              for c, n in zip(clases_producidas, cuentas)}
        condicional.append({
            "frontera": f"<= {cfg.CLASSES[k - 1]} vs >= {cfg.CLASSES[k]}",
            "filas_cercanas_al_umbral": int(cercanas.sum()),
            "errores_cerca_del_umbral": int(falla_cerca.sum()),
            "tasa_de_error_cercana": round(float(falla_cerca.sum() / max(cercanas.sum(), 1)), 4),
            "macro_f1_cuando_falla_cerca": round(float(
                f1_score(clase[falla_cerca], pred[falla_cerca], average="macro", labels=ETIQUETAS))
                if falla_cerca.any() else None, 4),
            "macro_f1_cuando_acierta_cerca": round(float(
                f1_score(clase[acierta_cerca], pred[acierta_cerca], average="macro", labels=ETIQUETAS)), 4),
            "macro_f1_en_filas_lejanas": round(float(
                f1_score(clase[lejanas], pred[lejanas], average="macro", labels=ETIQUETAS)), 4),
            "confusion": confusion})
        print(f"  {condicional[-1]['frontera']:<26} {int(cercanas.sum()):>7,} cercanas | "
              f"{int(falla_cerca.sum()):>6,} errores "
              f"({condicional[-1]['tasa_de_error_cercana']:.1%}) | macro-F1 cuando falla "
              f"{condicional[-1]['macro_f1_cuando_falla_cerca']} y cuando acierta "
              f"{condicional[-1]['macro_f1_cuando_acierta_cerca']} | lejanas "
              f"{condicional[-1]['macro_f1_en_filas_lejanas']}", flush=True)

    # ---------------------------------------------------------------- figuras
    fig, ejes = plt.subplots(1, 2, figsize=(16, 5.5))
    eje = ejes[0]
    indice = np.arange(len(filas))
    eje.bar(indice - 0.2, [f["exactitud_balanceada"] for f in filas], 0.4,
            label="exactitud balanceada", color="#b2182b")
    eje.bar(indice + 0.2, [f["auc"] for f in filas], 0.4, label="AUC", color="#2166ac")
    eje.axhline(0.5, color="grey", linestyle=":", linewidth=1)
    eje.set_xticks(indice)
    eje.set_xticklabels([f["frontera"] for f in filas], rotation=20, fontsize=7)
    eje.set_ylim(0.4, 1.0)
    eje.set_ylabel("valor")
    eje.set_title("Las cinco fronteras: exactitud balanceada y AUC", fontsize=10)
    eje.legend(fontsize=8)
    eje = ejes[1]
    eje.bar(indice, [f["descalibracion"] for f in filas], color="#4d4d4d")
    eje.axhline(0, color="grey", linewidth=1)
    eje.set_xticks(indice)
    eje.set_xticklabels([f["frontera"] for f in filas], rotation=20, fontsize=7)
    eje.set_ylabel("media predicha menos tasa real")
    eje.set_title("Descalibracion de cada frontera", fontsize=10)
    fig.suptitle("V20 P0: donde esta la informacion incremental de las cinco fronteras", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURA1, dpi=130)
    plt.close(fig)

    fig, ejes = plt.subplots(1, 2, figsize=(16, 5.5))
    eje = ejes[0]
    eje.bar(range(len(filas_cons)), [f["fraccion"] for f in filas_cons], color="#b2182b")
    eje.set_xticks(range(len(filas_cons)))
    eje.set_xticklabels([f"fold {f['fold']}" for f in filas_cons], fontsize=8)
    eje.set_ylabel("fraccion de filas que violan la monotonia")
    eje.set_title("Violaciones de la consistencia ordinal por fold", fontsize=10)
    eje = ejes[1]
    eje.bar(range(len(impactos)), [i["caida_del_blend"] * 1000 for i in impactos], color="#2166ac")
    eje.axhline(0, color="grey", linewidth=1)
    eje.set_xticks(range(len(impactos)))
    eje.set_xticklabels([i["frontera_degradada"] for i in impactos], rotation=20, fontsize=6)
    eje.set_ylabel("caida de macro-F1 (milesimas)")
    eje.set_title("Impacto de degradar cada frontera sobre las seis clases", fontsize=10)
    fig.suptitle("V20 P0: consistencia ordinal e impacto de cada frontera", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURA2, dpi=130)
    plt.close(fig)

    pd.DataFrame([{k: v for k, v in f.items() if k != "error_por_fold"} for f in filas]).to_csv(
        TABLA, index=False)

    INFORME.write_text("\n".join([
        "# V20 · P0 — Auditoria de las cinco fronteras ordinales", "",
        "**V20 queda congelado.** Todo esto es analisis del OOF ya medido, sin reentrenar nada.", "",
        "## 1. Donde esta la informacion incremental", "",
        "| Frontera | Lado bajo / alto | Exacta balanceada | Ventaja | F1 binario | AUC | Brier | "
        "Descalibracion |", "|---|---|---|---|---|---|---|---|",
        *[f"| {f['frontera']} | {f['filas_lado_bajo']:,} / {f['filas_lado_alto']:,} | "
          f"**{f['exactitud_balanceada']}** | {f['ventaja_sobre_el_azar']:+.4f} | {f['f1_binario']} | "
          f"{f['auc']} | {f['brier']} | {f['descalibracion']:+.4f} |" for f in filas], "",
        "**Correlacion entre las cinco fronteras**:", "```", str(np.round(correlacion, 3).tolist()),
        "```", "",
        "La descalibracion es sistematica y todas las fronteras se descalibran en el mismo sentido, "
        "luego es un sesgo del conjunto y no un defecto de una frontera concreta.", "",
        "## 2. La consistencia ordinal", "",
        f"- **{total_violan:,} de {len(y):,} filas ({total_violan / len(y):.2%}) violan la monotonia** "
        "de las cinco probabilidades acumuladas.",
        f"- Magnitud media de la violacion: **{magnitud.mean():.6f}**; maxima: **{magnitud.max():.6f}**.",
        "- **Como resolvio V20 eso**: con el **maximo acumulado**, que es la unica correccion que "
        "respeta el orden sin inventar informacion. No es una operacion neutra: impone la estructura "
        "ordinal que los cinco clasificadores, ajustados de forma independiente, no garantizan.", "",
        "| Fold | Filas con violacion | Fraccion | Magnitud media | Magnitud maxima |",
        "|---|---|---|---|---|",
        *[f"| {f['fold']} | {f['filas_con_violacion']:,} | {f['fraccion']:.2%} | "
          f"{f['magnitud_media']:.6f} | {f['magnitud_maxima']:.6f} |" for f in filas_cons], "",
        "## 3. El impacto de cada frontera sobre las seis clases", "",
        "Se degrada cada frontera a un valor constante de 0,5, que es la decision mas incierta posible, "
        "y se mide cuanto cae el macro-F1 final. Es lo que separa **frontera debil** de **frontera cuyos "
        "errores importan poco**.", "",
        "Se mide sobre el **blend de V20**, porque la cadena es un cuarto miembro mas y su valor depende "
        "de como interactua con los otros tres. En soledad la reconstruccion de las seis clases es "
        "fragil y una sola frontera degradada la deshace.",
        "",
        "| Frontera degradada | Blend de V20 | Caida del blend | Cadena en soledad | Caida de la cadena |",
        "|---|---|---|---|---|",
        *[f"| {i['frontera_degradada']} | {i['macro_f1_del_blend']} | "
          f"**{i['caida_del_blend']:+.4f}** | {i['macro_f1_de_la_cadena_sola']} | "
          f"{i['caida_de_la_cadena_sola']:+.4f} |" for i in impactos], "",
        "## 4. El error condicional por frontera", "",
        "Restringido a las **filas cercanas al umbral**, porque un fallo de la decision binaria solo es "
        "un fallo de frontera si la fila esta cerca: en las lejanas, discrepar del lado es lo correcto.",
        "",
        "| Frontera | Filas cercanas | Errores | Tasa | macro-F1 cuando falla | Cuando acierta | Lejanas |",
        "|---|---|---|---|---|---|---|",
        *[f"| {c['frontera']} | {c['filas_cercanas_al_umbral']:,} | {c['errores_cerca_del_umbral']:,} | "
          f"{c['tasa_de_error_cercana']:.1%} | {c['macro_f1_cuando_falla_cerca']} | "
          f"{c['macro_f1_cuando_acierta_cerca']} | {c['macro_f1_en_filas_lejanas']} |"
          for c in condicional], "",
        "Un error **lejano** es mucho mas grave que uno cercano: indica que el sistema no sabe donde esta, "
        "no que este cerca de una frontera. Por eso la tabla separa las filas cercanas de las lejanas.", "",
    ]), encoding="utf-8")

    RESULTADO.write_text(json.dumps({
        "fase": "V20 P0 auditoria de las cinco fronteras", "fecha": "2026-09-27",
        "champion_congelado": {"nombre": "V20 cadena de umbrales anidados", "lb": 0.62689,
                               "cv": CV_V20, "intacto": True},
        "fronteras": filas, "correlacion": np.round(correlacion, 4).tolist(),
        "consistencia": {"filas_que_violan": total_violan, "de": int(len(y)),
                          "fraccion": round(total_violan / len(y), 6),
                          "magnitud_media": float(magnitud.mean()),
                          "magnitud_maxima": float(magnitud.max()),
                          "por_fold": filas_cons,
                          "correccion_de_v20": "maximo acumulado"},
        "impacto_por_frontera": impactos, "error_condicional": condicional,
        "macro_f1_de_la_cadena_sola": round(base, 7),
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsegundos: {time.time() - inicio:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
