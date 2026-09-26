"""V15 · B — Poda de variables redundantes: ¿V12 mejora con 197 variables en vez de 279?

V15a midió la estructura de redundancia de las 312 variables (279 de V12 más 33 lingüísticas de
V14) y encontró que es **perfectamente reproducible**: el conjunto de pares con |rho| >= 0,90 es
idéntico en los cinco folds (Jaccard 1,0000). Eso autoriza podar sin miedo a que la decisión
dependa del fold.

Criterio de poda, aplicado **in-fold** en V15a: se agrupan las variables con |rho| >= 0,90 y **se
conserva una por cluster, la de mayor eta cuadrado con la salida dentro de ese fold**; las variables
que no están en ningún cluster se conservan todas; y las columnas constantes se eliminan por
definición, porque una variable que nunca varía no puede informar de nada. El resultado son **197
variables** (de 312), y el índice exacto por fold está en `features/v15_conservadas_fold{0..4}.npy`.

Este script responde a la pregunta operativa que hay detrás de la petición: **si se eliminan las
variables de entrada redundantes, ¿el modelo mejora, empeora o no cambia?** Tres respuestas posibles y
las tres son informativas:

* Si **mejora**, la redundancia estaba costando precisión y V12 se puede simplificar sin perder (y
  quizá ganando), lo que además la haría más interpretable y más barata.
* Si **no cambia** de forma significativa, la redundancia es inocua para la predicción —lo que ya
  insinúa P0b— y la poda se documenta como una mejora de higiene, no de rendimiento.
* Si **empeora**, significa que la redundancia no es superflua: hay variáveis casi idénticas que
  aporta Tiny pesos distintos y el ensemble se apoya en ella. Es un resultado que también hay que
  poder leer sin adornos.

El gate de adopción es el de D-042, con su segunda partición. Además se mide qué se pierde: la
accuracy en las dos fronteras centrales y la complementariedad, porque una poda que mejora la global
empeorando una frontera central no se adopta.

Entregables: ``experiments/v15b_poda.json``, ``reports/v15_poda.md``,
``reports/v15_poda_comparativa.csv``, ``features/v15b_oof.npy``.
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

from v13_p1b_fusion_tokens import (PARES, bootstrap_diferencia, complementariedad,  # noqa: E402
                                   metricas_frontera, metricas_globales)
from v14b_fusion_linguistico import evaluar  # noqa: E402

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v15_poda.md"
COMPARATIVA = cfg.REPORTS / "v15_poda_comparativa.csv"
OOF = cfg.FEATURES / "v15b_oof.npy"
RESULTADO = cfg.EXPERIMENTS / "v15b_poda.json"
DELTA_MINIMO = 0.001
FOLDS_POSITIVOS_MINIMOS = 3
FOLDS_SEGUNDA_PARTICION = 4
FOLDS_SEGUNDA_PARTICION_MINIMOS = 3


def main() -> int:
    """Evalúa la matriz podada contra V12 con el gate revisado de D-042."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])
    clusters_ids = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters_ids, n_splits=5, n_repeats=1, scheme="group")
    folds4 = cvmod.make_folds(y, groups=clusters_ids, n_splits=FOLDS_SEGUNDA_PARTICION,
                              n_repeats=1, scheme="group")

    from v15a_correlacion import matriz_completa
    matriz, nombres = matriz_completa()
    indices = {fold["fold"]: np.load(cfg.FEATURES / f"v15_conservadas_fold{fold['fold']}.npy")
               for fold in folds}
    print(f"[poda] {matriz.shape[1]} variables de entrada -> "
          f"{len(indices[0])} conservadas", flush=True)

    referencia = np.load(cfg.FEATURES / "oof_ensemble_v12.npy")
    metricas_referencia = metricas_globales(referencia, clase)
    fronteras_referencia = metricas_frontera(referencia, clase, PARES)
    por_fold_referencia = [float(f1_score(
        y[fold["val_idx"]], np.array(cfg.CLASSES)[np.argmax(referencia[fold["val_idx"]], axis=1)],
        average="macro", labels=cfg.CLASSES)) for fold in folds]
    print(f"[referencia] V12 {metricas_referencia['macro_f1']} | folds "
          f"{[round(v, 4) for v in por_fold_referencia]}", flush=True)

    # La poda depende del fold: cada fold usa su indice, calculado sin ver su validacion.
    probabilidades = np.zeros((len(y), len(cfg.CLASSES)))
    por_fold_poda = []
    for fold in folds:
        columnas = indices[fold["fold"]]
        sub = matriz[:, columnas]
        validacion = fold["val_idx"]
        entrenamiento = np.concatenate([otro["val_idx"] for otro in folds
                                        if otro["fold"] != fold["fold"]])
        from v13_p1b_fusion_tokens import logreg
        from phase2_ponderacion import EncodedMLP
        lineal = logreg().fit(sub[entrenamiento], y[entrenamiento])
        red = EncodedMLP().fit(sub[entrenamiento], y[entrenamiento])
        probabilidades[validacion] = 0.5 * lineal.predict_proba(sub[validacion]) + \
            0.5 * red.model.predict_proba(red.scaler.transform(sub[validacion]))
        etiquetas = np.array(cfg.CLASSES)[np.argmax(probabilidades[validacion], axis=1)]
        por_fold_poda.append(float(f1_score(y[validacion], etiquetas, average="macro",
                                            labels=cfg.CLASSES)))
        print(f"    fold {fold['fold']} ({len(columnas)} variables): {por_fold_poda[-1]:.4f}",
              flush=True)
    metricas_poda = metricas_globales(probabilidades, clase)
    fronteras_poda = metricas_frontera(probabilidades, clase, PARES)
    delta = metricas_poda["macro_f1"] - metricas_referencia["macro_f1"]
    folds_positivos = sum(1 for a, b in zip(por_fold_poda, por_fold_referencia) if a > b)
    np.save(OOF, probabilidades)
    print(f"[poda] {metricas_poda['macro_f1']} | delta {delta:+.4f} | folds positivos "
          f"{folds_positivos}/5 | folds {[round(v, 4) for v in por_fold_poda]}", flush=True)

    # ---------------------------------------------------------------- segunda particion
    _, por_fold4_poda = evaluar(matriz[:, indices[0]], y, folds4)
    base4, _, _, _, _ = __import__("phase2_ponderacion", fromlist=["build_features"]).build_features()
    base4 = np.concatenate([base4, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    _, por_fold4_referencia = evaluar(base4, y, folds4)
    diferencias4 = [a - b for a, b in zip(por_fold4_poda, por_fold4_referencia)]
    positivos4 = sum(1 for d in diferencias4 if d > 0)
    print(f"[segunda particion] diferencias {[round(d, 4) for d in diferencias4]} | positivos "
          f"{positivos4}/4 | media {float(np.mean(diferencias4)):+.4f}", flush=True)

    intervalo = bootstrap_diferencia(referencia, probabilidades, clase, len(cfg.CLASSES))
    fronteras_mejoras = sum(1 for etiqueta in PARES.values()
                            if fronteras_poda[etiqueta]["accuracy"]
                            > fronteras_referencia[etiqueta]["accuracy"])
    criterios = {
        "efecto_minimo": bool(delta >= DELTA_MINIMO),
        "folds_positivos": bool(folds_positivos >= FOLDS_POSITIVOS_MINIMOS),
        "significacion": bool(intervalo[0] > 0),
        "segunda_particion": bool(positivos4 >= FOLDS_SEGUNDA_PARTICION_MINIMOS),
        "fronteras": bool(fronteras_mejoras >= len(PARES)),
    }
    adoptable = all(criterios.values())
    complemento = complementariedad(referencia, probabilidades, clase)

    pd.DataFrame([
        {"modelo": "V12 (279 variables, sin podar)", "macro_f1": metricas_referencia["macro_f1"],
         "accuracy": metricas_referencia["accuracy"], "delta": 0.0,
         "folds_positivos": "-", "18-24|25-34": fronteras_referencia["18-24|25-34"]["accuracy"],
         "25-34|35-49": fronteras_referencia["25-34|35-49"]["accuracy"]},
        {"modelo": f"V12 podado ({len(indices[0])} variables)", "macro_f1": metricas_poda["macro_f1"],
         "accuracy": metricas_poda["accuracy"], "delta": round(delta, 4),
         "folds_positivos": f"{folds_positivos}/5",
         "18-24|25-34": fronteras_poda["18-24|25-34"]["accuracy"],
         "25-34|35-49": fronteras_poda["25-34|35-49"]["accuracy"]},
    ]).to_csv(COMPARATIVA, index=False)

    lineas = [
        "# V15 · B — Poda de variables redundantes", "",
        "## 1. Qué se podó y con qué criterio", "",
        f"De **{matriz.shape[1]} variables** de entrada quedan **{len(indices[0])}**. El criterio se "
        "aplicó in-fold en V15a: un representante por cluster de |rho| >= 0,90, elegido como el de "
        "mayor eta cuadrado **dentro del fold**; se conservan todas las variables que no están en "
        "ningún cluster; y se eliminan por definición las columnas constantes. La estructura de "
        "pares es **idéntica en los cinco folds** (Jaccard 1,0000), así que la poda no es una "
        "decisión que dependa del fold.", "",
        "## 2. Resultado", "",
        "| Modelo | macro-F1 | accuracy | Delta | Folds positivos |", "|---|---|---|---|---|",
        f"| V12 ({matriz.shape[1] - 33} variables, sin podar) | {metricas_referencia['macro_f1']} | "
        f"{metricas_referencia['accuracy']} | - | - |",
        f"| V12 podado ({len(indices[0])} variables) | {metricas_poda['macro_f1']} | "
        f"{metricas_poda['accuracy']} | {delta:+.4f} | {folds_positivos}/5 |", "",
        f"Folds podado: {[round(v, 4) for v in por_fold_poda]} · folds V12: "
        f"{[round(v, 4) for v in por_fold_referencia]}", "",
        "## 3. Segunda partición (4 folds)", "",
        "| Fold | V12 | Podado | Diferencia |", "|---|---|---|---|",
    ]
    for indice, (a, b) in enumerate(zip(por_fold4_poda, por_fold4_referencia), start=1):
        lineas.append(f"| {indice} | {b:.4f} | {a:.4f} | {a - b:+.4f} |")
    lineas += ["", f"**Positivos: {positivos4}/4** · IC95 del bootstrap pareado "
                  f"[{intervalo[0]:+.4f}, {intervalo[1]:+.4f}]", "",
              "## 4. Doble métrica (§36)", "",
              "| Frontera | V12 | Podado |", "|---|---|---|"]
    for etiqueta in PARES.values():
        lineas.append(f"| {etiqueta} | {fronteras_referencia[etiqueta]['accuracy']} | "
                      f"{fronteras_poda[etiqueta]['accuracy']} |")
    lineas += ["", "Complementariedad: rescate "
                  f"{complemento['global']['rescate']:,} · daño {complemento['global']['dano']:,} · "
                  f"**neto {complemento['global']['neto']:+,}**", "",
              "## 5. Gate D-042", "", "| Criterio | Valor | Cumple |", "|---|---|---|"]
    valores = {
        "efecto_minimo": f"delta {delta:+.4f} >= {DELTA_MINIMO}",
        "folds_positivos": f"{folds_positivos}/5 >= {FOLDS_POSITIVOS_MINIMOS}",
        "significacion": f"IC95 [{intervalo[0]:+.4f}, {intervalo[1]:+.4f}]",
        "segunda_particion": f"{positivos4}/4 >= {FOLDS_SEGUNDA_PARTICION_MINIMOS}",
        "fronteras": f"{fronteras_mejoras}/{len(PARES)} mejoran",
    }
    for nombre, cumple in criterios.items():
        lineas.append(f"| {nombre.replace('_', ' ')} | {valores[nombre]} | "
                      f"{'si' if cumple else '**no**'} |")
    lineas += ["", "## 6. Lectura", ""]
    if delta > DELTA_MINIMO and folds_positivos >= FOLDS_POSITIVOS_MINIMOS and positivos4 >= FOLDS_SEGUNDA_PARTICION_MINIMOS:
        lineas += ["**La poda mejora a V12 de forma reproducible.** Es el primer resultado positivo "
                   "del proyecto: no viene de variables nuevas, sino de quitar las que sobran. El "
                   "candidato se somete a la segunda partición que exige D-042."]
    elif abs(delta) < 0.001 and positivos4 >= 2:
        lineas += [
            "**La poda es estadísticamente neutra**: quitar 115 de 312 variables, entre ellas 15 "
            "duplicados exactos, no cambia la macro-F1 de forma medible. Es la respuesta esperable "
            "como ya habia medido P0b: la colinealidad afecta a la interpretación de "
            "coeficientes, no a la predicción. El valor de la poda es de higiene e interpretabilidad, "
            "no de rendimiento, y eso se documenta sin disfrazarlo de mejora."]
    else:
        lineas += [
            f"**La poda empeora a V12** (delta {delta:+.4f}). Es un resultado con contenido: las "
            "variables casi idénticas no eran superfluas, y el ensemble se apoyaba en esa "
            "redundancia. Tiene una lectura posible —el ruido de muchas variables weakly "
            "correlacionadas actúa como un conjunto de-appointed weak learners— y otra más simple: "
            "las diferencias están dentro del ruido de la estimación, y por eso el IC95 hay que "
            "leerlo antes que el delta puntual."]
    lineas.append("")
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    RESULTADO.write_text(json.dumps({
        "fase": "V15 · B poda de variables redundantes", "fecha": "2026-09-26",
        "n_entradas": int(matriz.shape[1]), "n_conservadas": int(len(indices[0])),
        "referencia": {"macro_f1": metricas_referencia["macro_f1"],
                       "per_fold": [round(v, 4) for v in por_fold_referencia]},
        "poda": {"macro_f1": metricas_poda["macro_f1"], "accuracy": metricas_poda["accuracy"],
                 "per_fold": [round(v, 4) for v in por_fold_poda]},
        "delta": round(delta, 4), "folds_positivos": folds_positivos,
        "segunda_particion": {"por_fold_poda": [round(v, 4) for v in por_fold4_poda],
                              "por_fold_referencia": [round(v, 4) for v in por_fold4_referencia],
                              "positivos": positivos4},
        "bootstrap_ic95": [round(v, 4) for v in intervalo],
        "criterios": criterios, "adopta": adoptable, "complementariedad": complemento,
        "fronteras": {"referencia": fronteras_referencia, "poda": fronteras_poda},
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\ndelta {delta:+.4f} | folds {folds_positivos}/5 | 2a particion {positivos4}/4 | "
          f"IC95 [{intervalo[0]:+.4f}, {intervalo[1]:+.4f}] | podar: "
          f"{'ADOPTABLE' if adoptable else 'no adoptable como candidato'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
