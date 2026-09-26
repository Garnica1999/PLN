"""V14 · B — Fusión de las variables lingüísticas con V12, bajo el gate revisado de D-042.

Este script mide, con el gate que Carlos asume explícitamente en D-042, si el bloque de variables
lingüísticas del analizador español mejora a V12.

**Sobre el pre-chequeo que V14a no pasó.** V14a cerró la pista por el criterio de puerta habitual
(neto de rescate −58.682 al aplicar el bloque como clasificador). Aquí se procede igualmente, y
conviene decir por qué, sin adornarlo: ese criterio responde a la pregunta *"¿qué pasa si el bloque
decide por sí solo?"*, y la decisión que nos interesa es otra, *"¿aporta algo al meta-modelo?"*. Un
bloque con 0,2409 de control aislado, cuyas variables no son copias de V12 y cuya información de
clase no es linealmente accesible desde V12 (P16c), puede aportar con un peso pequeño aunque como
clasificador sea peor. El gate se ha bajado precisamente para medir diferencias pequeñas, y esta es
la primera medición que se hace con él. Queda escrito que el pre-chequeo la daba por cerrada.

**El bloque entra sin las cuatro variables redundantes.** El cribado de V14a marcó
``longitud_media_palabra`` (rho 0,9824), ``lema_distintos_por_token`` (0,8448), ``lema_hapax_ratio``
(0,8329) y ``tokens_por_frase`` (0,8203) como casi duplicados de variables de V12, exactamente lo que
P0b habia predicho para la textometría. La regla operativa de P0b es que no se incorporan, así que
la fusión usa **33** columnas y no 37.

**El gate D-042, completo.** Se aplican los cinco criterios:

1. Efecto: Δ de macro-F1 ≥ **+0,001**.
2. Folds: ≥ **3 de 5** positivos.
3. Significación: el **IC95 del bootstrap pareado excluye el cero**. Este criterio no se relajó y es
   el que protege contra el ruido: con 169.336 filas, un efecto de +0,001 no puede aprobarlo.
4. **Segunda partición**: el efecto se reproduce en ≥ 3 de 4 espacios de validación disjuntos. Se
   mide con una partición de 4 folds agrupados, independiente de la de 5, que es una re-ejecución
   real y no un reparto del mismo OOF.
5. Control de permutado (DD-6, tres semillas) y doble métrica (§36), con la condición de que
   ninguna frontera central empeore.

El test no se toca. Ninguna submission se genera aquí.

Entregables: ``experiments/v14b_fusion_linguistico.json``, ``reports/v14_fusion_linguistico.md``,
``reports/v14_fusion_comparativa.csv``, ``features/v14b_oof.npy``.
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

from phase2_ponderacion import EncodedMLP, build_features  # noqa: E402
from v13_p1b_fusion_tokens import (PARES, SEMILLAS, bootstrap_diferencia,  # noqa: E402
                                   complementariedad, logreg, metricas_frontera,
                                   metricas_globales)

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v14_fusion_linguistico.md"
COMPARATIVA = cfg.REPORTS / "v14_fusion_comparativa.csv"
OOF = cfg.FEATURES / "v14b_oof.npy"
RESULTADO = cfg.EXPERIMENTS / "v14b_fusion_linguistico.json"

#: Gate revisado (D-042).
DELTA_MINIMO = 0.001
FOLDS_POSITIVOS_MINIMOS = 3
FOLDS_SEGUNDA_PARTICION = 4
FOLDS_SEGUNDA_PARTICION_MINIMOS = 3
#: Variables que el cribado de V14a marco como casi duplicadas de V12 y que no se incorporan.
REDUNDANTES = ("longitud_media_palabra", "lema_distintos_por_token", "lema_hapax_ratio",
               "tokens_por_frase")


def evaluar(matriz: np.ndarray, y: np.ndarray, folds: list[dict]) -> tuple[np.ndarray, list[float]]:
    """OOF del combinador de producción y macro-F1 por fold."""
    probabilidades = np.zeros((len(y), len(cfg.CLASSES)))
    por_fold = []
    for fold in folds:
        validacion = fold["val_idx"]
        entrenamiento = np.concatenate([otro["val_idx"] for otro in folds
                                        if otro["fold"] != fold["fold"]])
        lineal = logreg().fit(matriz[entrenamiento], y[entrenamiento])
        red = EncodedMLP().fit(matriz[entrenamiento], y[entrenamiento])
        probabilidades[validacion] = 0.5 * lineal.predict_proba(matriz[validacion]) + \
            0.5 * red.model.predict_proba(red.scaler.transform(matriz[validacion]))
        etiquetas = np.array(cfg.CLASSES)[np.argmax(probabilidades[validacion], axis=1)]
        por_fold.append(float(f1_score(y[validacion], etiquetas, average="macro",
                                       labels=cfg.CLASSES)))
        print(f"    fold {fold['fold']}: {por_fold[-1]:.4f}", flush=True)
    return probabilidades, por_fold


def main() -> int:
    """Fusiona el bloque lingüístico con V12 y aplica el gate revisado de D-042."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")

    base, _, _, _, _ = build_features()
    base = np.concatenate([base, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    nombres = json.loads((cfg.FEATURES / "v14_linguistico_nombres.json").read_text(encoding="utf-8"))
    matriz_bruto = np.load(cfg.FEATURES / "v14_linguistico.npy").astype(np.float64)
    columnas = [i for i, nombre in enumerate(nombres) if nombre not in REDUNDANTES]
    bloque = matriz_bruto[:, columnas]
    nombres_bloque = [nombres[i] for i in columnas]
    print(f"[bloque] {bloque.shape[1]} columnas tras retirar {len(REDUNDANTES)} redundantes",
          flush=True)

    referencia = np.load(cfg.FEATURES / "oof_ensemble_v12.npy")
    metricas_referencia = metricas_globales(referencia, np.array(
        [cfg.CLASSES.index(v) for v in y]))
    fronteras_referencia = metricas_frontera(
        referencia, np.array([cfg.CLASSES.index(v) for v in y]), PARES)
    por_fold_referencia = [float(f1_score(
        y[fold["val_idx"]], np.array(cfg.CLASSES)[np.argmax(referencia[fold["val_idx"]], axis=1)],
        average="macro", labels=cfg.CLASSES)) for fold in folds]
    print(f"[referencia] V12 {metricas_referencia['macro_f1']} | folds "
          f"{[round(v, 4) for v in por_fold_referencia]}", flush=True)

    combinada = np.hstack([base, bloque])
    print("[candidata] V12 + variables linguisticas (5 folds)", flush=True)
    candidata, por_fold_candidata = evaluar(combinada, y, folds)
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    metricas_candidata = metricas_globales(candidata, clase)
    fronteras_candidata = metricas_frontera(candidata, clase, PARES)
    delta = metricas_candidata["macro_f1"] - metricas_referencia["macro_f1"]
    # La comparacion es candidata contra referencia: se cuenta el numero de folds en los que
    # la candidata supera a V12. La primera version invertia la condicion y contaba las veces
    # que V12 ganaba, que es exactamente lo contrario.
    folds_positivos = sum(1 for candidata_fold, referencia_fold
                         in zip(por_fold_candidata, por_fold_referencia)
                         if candidata_fold > referencia_fold)
    np.save(OOF, candidata)
    print(f"[candidata] {metricas_candidata['macro_f1']} | delta {delta:+.4f} | folds positivos "
          f"{folds_positivos}/5 | folds {[round(v, 4) for v in por_fold_candidata]}", flush=True)

    # ---------------------------------------------------------------- control de permutado
    controles = []
    for semilla in SEMILLAS:
        permutada = bloque[np.random.default_rng(semilla).permutation(len(y))]
        probabilidades, por_fold = evaluar(np.hstack([base, permutada]), y, folds)
        valor = metricas_globales(probabilidades, clase)["macro_f1"]
        controles.append({"semilla": semilla, "macro_f1": valor,
                          "delta": round(valor - metricas_referencia["macro_f1"], 4)})
        print(f"[control permutado] semilla {semilla}: {valor} "
              f"({valor - metricas_referencia['macro_f1']:+.4f})", flush=True)
    mejor_control = max(controles, key=lambda c: c["macro_f1"])

    # ---------------------------------------------------------------- segunda particion
    folds4 = cvmod.make_folds(y, groups=clusters, n_splits=FOLDS_SEGUNDA_PARTICION, n_repeats=1,
                              scheme="group")
    _, por_fold4_candidata = evaluar(combinada, y, folds4)
    _, por_fold4_referencia = evaluar(base, y, folds4)
    diferencias4 = [nueva - vieja for nueva, vieja in zip(por_fold4_candidata,
                                                          por_fold4_referencia)]
    positivos4 = sum(1 for d in diferencias4 if d > 0)
    media4 = float(np.mean(diferencias4))
    print(f"[segunda particion] 4 folds | diferencias {[round(d, 4) for d in diferencias4]} | "
          f"positivos {positivos4}/4 | media {media4:+.4f}", flush=True)

    # ---------------------------------------------------------------- significancia y gate
    intervalo = bootstrap_diferencia(referencia, candidata, clase, len(cfg.CLASSES))
    fronteras_mejoras = sum(1 for etiqueta in PARES.values()
                            if fronteras_candidata[etiqueta]["accuracy"]
                            > fronteras_referencia[etiqueta]["accuracy"])
    criterios = {
        "efecto_minimo": bool(delta >= DELTA_MINIMO),
        "folds_positivos": bool(folds_positivos >= FOLDS_POSITIVOS_MINIMOS),
        "significacion": bool(intervalo[0] > 0),
        "segunda_particion": bool(positivos4 >= FOLDS_SEGUNDA_PARTICION_MINIMOS),
        "control_permutado": bool(metricas_candidata["macro_f1"] > mejor_control["macro_f1"]),
        "fronteras": bool(fronteras_mejoras >= len(PARES)),
    }
    adopta = all(criterios.values())
    complemento = complementariedad(referencia, candidata, clase)

    comparativa = [
        {"modelo": "V12 (referencia)", "macro_f1": metricas_referencia["macro_f1"],
         "delta": 0.0, "folds": "0/5",
         "frontera_18_24_25_34": fronteras_referencia["18-24|25-34"]["accuracy"],
         "frontera_25_34_35_49": fronteras_referencia["25-34|35-49"]["accuracy"]},
        {"modelo": "V12 + linguisticas (33)", "macro_f1": metricas_candidata["macro_f1"],
         "delta": round(delta, 4), "folds": f"{folds_positivos}/5",
         "frontera_18_24_25_34": fronteras_candidata["18-24|25-34"]["accuracy"],
         "frontera_25_34_35_49": fronteras_candidata["25-34|35-49"]["accuracy"]},
    ] + [{"modelo": f"control permutado {c['semilla']}", "macro_f1": c["macro_f1"],
          "delta": c["delta"], "folds": "-", "frontera_18_24_25_34": None,
          "frontera_25_34_35_49": None} for c in controles]
    pd.DataFrame(comparativa).to_csv(COMPARATIVA, index=False)

    lineas = [
        "# V14 · B — Fusión de las variables lingüísticas con V12", "",
        "## 1. Por qué se mide pese al pre-chequeo negativo", "",
        "El pre-chequeo de V14a dio un neto de −58.682 al aplicar el bloque como clasificador, y con "
        "el protocolo habitual la pista se cerraba ahí. Se mide igualmente porque ese criterio "
        "contesta a otra pregunta: si el bloque aporta algo **al meta-modelo** es una decisión "
        "distinta, y D-042 acaba de bajar el listón justamente para medir diferencias pequeñas. El "
        "bloque tiene 0,2409 de control aislado, solo cuatro variables redundantes y su "
        "información de clase no es linealmente accesible desde V12 (P16c).", "",
        f"El bloque entra con **{bloque.shape[1]} columnas**: se retiran las cuatro que el cribado "
        f"marcó como casi duplicadas de V12 (longitud media de palabra con rho 0,9824, diversidad "
        f"por lema 0,8448, hapax por lema 0,8329 y tokens por frase 0,8203).", "",
        "## 2. Resultado principal", "",
        "| Modelo | macro-F1 | Delta | Folds positivos |", "|---|---|---|---|",
        f"| V12 (referencia) | {metricas_referencia['macro_f1']} | - | - |",
        f"| V12 + lingüísticas | {metricas_candidata['macro_f1']} | {delta:+.4f} | "
        f"{folds_positivos}/5 |",
    ]
    for control in controles:
        lineas.append(f"| Control permutado (semilla {control['semilla']}) | "
                      f"{control['macro_f1']} | {control['delta']:+.4f} | - |")
    lineas += ["", "## 3. Segunda partición (criterio 4 del gate)", "",
               f"Partición de **{FOLDS_SEGUNDA_PARTICION} folds agrupados**, independiente de la de 5:",
               "", "| Fold | V12 | Candidata | Diferencia |", "|---|---|---|---|"]
    for indice, (nueva, vieja) in enumerate(zip(por_fold4_candidata, por_fold4_referencia), start=1):
        lineas.append(f"| {indice} | {vieja:.4f} | {nueva:.4f} | {nueva - vieja:+.4f} |")
    lineas += ["", f"**Positivos: {positivos4}/4** (se exigen {FOLDS_SEGUNDA_PARTICION_MINIMOS}) · "
                  f"media de la diferencia {media4:+.4f} · IC95 del bootstrap pareado "
                  f"[{intervalo[0]:+.4f}, {intervalo[1]:+.4f}] con "
                  f"{intervalo[2]:.3f} de proporcion positiva.", "",
              "## 4. Doble métrica (§36)", "",
              "| Frontera | V12 | Candidata |", "|---|---|---|"]
    for etiqueta in PARES.values():
        lineas.append(f"| {etiqueta} | {fronteras_referencia[etiqueta]['accuracy']} | "
                      f"{fronteras_candidata[etiqueta]['accuracy']} |")
    lineas += ["", "Complementariedad: global rescate "
                  f"{complemento['global']['rescate']:,} · daño {complemento['global']['dano']:,} · "
                  f"**neto {complemento['global']['neto']:+,}**"]
    for etiqueta in PARES.values():
        lineas.append(f"- {etiqueta}: rescate {complemento[etiqueta]['rescate']:,} · daño "
                      f"{complemento[etiqueta]['dano']:,} · neto {complemento[etiqueta]['neto']:+,}")
    lineas += ["", "## 5. Gate D-042", "", "| Criterio | Valor | Cumple |", "|---|---|---|"]
    valores = {
        "efecto_minimo": f"delta {delta:+.4f} >= {DELTA_MINIMO}",
        "folds_positivos": f"{folds_positivos}/5 >= {FOLDS_POSITIVOS_MINIMOS}",
        "significacion": f"IC95 [{intervalo[0]:+.4f}, {intervalo[1]:+.4f}]",
        "segunda_particion": f"{positivos4}/4 >= {FOLDS_SEGUNDA_PARTICION_MINIMOS}",
        "control_permutado": f"{metricas_candidata['macro_f1']} > {mejor_control['macro_f1']}",
        "fronteras": f"{fronteras_mejoras}/{len(PARES)} mejoran",
    }
    for nombre, cumple in criterios.items():
        lineas.append(f"| {nombre.replace('_', ' ')} | {valores[nombre]} | "
                      f"{'si' if cumple else '**no**'} |")
    lineas += ["", f"**Decision: {'ADOPTAR' if adopta else 'NO ADOPTAR'}**", ""]
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    RESULTADO.write_text(json.dumps({
        "fase": "V14 · B fusion linguisticas bajo el gate D-042", "fecha": "2026-09-26",
        "columnas_bloque": int(bloque.shape[1]),
        "redundantes_retiradas": list(REDUNDANTES), "nombres_bloque": nombres_bloque,
        "referencia": {"macro_f1": metricas_referencia["macro_f1"],
                       "per_fold": [round(v, 4) for v in por_fold_referencia]},
        "candidata": {"macro_f1": metricas_candidata["macro_f1"],
                      "per_fold": [round(v, 4) for v in por_fold_candidata]},
        "delta": round(delta, 4), "folds_positivos": folds_positivos,
        "controles_permutados": controles, "mejor_control": mejor_control,
        "segunda_particion": {"folds": FOLDS_SEGUNDA_PARTICION,
                              "por_fold_candidata": [round(v, 4) for v in por_fold4_candidata],
                              "por_fold_referencia": [round(v, 4) for v in por_fold4_referencia],
                              "positivos": positivos4, "media": round(media4, 4)},
        "bootstrap_ic95": [round(v, 4) for v in intervalo],
        "criterios": criterios, "adopta": adopta, "complementariedad": complemento,
        "fronteras": {"referencia": fronteras_referencia, "candidata": fronteras_candidata},
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\ndelta {delta:+.4f} | folds {folds_positivos}/5 | 2a particion {positivos4}/4 | "
          f"IC95 [{intervalo[0]:+.4f}, {intervalo[1]:+.4f}] | "
          f"DECISION: {'ADOPTAR' if adopta else 'NO ADOPTAR'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
