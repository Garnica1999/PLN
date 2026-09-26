"""V16 — Arquitectura del meta-modelo: dejar de promediar las dos cabezas a peso fijo.

Todos los diagnósticos de V13, V14 y V15 apuntan a lo mismo: **el cuello de botella no son los
datos, es la arquitectura**. El resumen es este. V12 combina las dos cabezas con un promedio fijo
0,5·LogReg + 0,5·MLP. Eso es una decisión de diseño de la primera versión del proyecto, no una
conclusión medida: nadie ha comprobado si 0,5 es el peso correcto, ni si un promedio es la forma
correcta de combinar dos clasificadores. V13 lo dejó apuntado como hipótesis y V14 lo corroboró
(V14 añade las mejores variables disponibles y el resultado se reproduce en dos particiones), así que
esta fase mide la Palanca que queda.

**Tres arquitecturas, mismo protocolo, misma matriz de variables.**

* **A0, la de producción**: promedio fijo 0,5 + 0,5. Es la referencia y su OOF ya está calculado
  (`features/oof_ensemble_v12.npy`), así que no se reentrena.
* **A1, peso aprendido**: sigue siendo un promedio, pero el peso se elige por fold. Para no mirar la
  validación, el peso se escoge sobre una partición **interior** del train exterior (20 %, agrupada
  por duplicados para que ningún casi-duplicado cruce), entrenando ahí las dos cabezas. El peso es
  un escalar y se elige sobre una rejilla gruesa; se aplica a la validación exterior con las cabezas
  ya entrenadas en todo el train exterior.
* **A2, combinador logístico**: las doce probabilidades de las dos cabezas alimentan una
  LogReg, es decir **apilamiento en dos niveles** en lugar de un promedio. Es la única de las tres
  que puede dar peso distinto a cada clase y a cada cabeza, y por eso es la que de verdad cambia la
  arquitectura y no solo un parámetro.

Ninguna de las tres introduce información nueva: todas usan exactamente las 279 variables de V12.
Si alguna mejora, el hallazgo es que **el mismo modelo llevaba desde el principio una decisión de
combinación que nunca se optimizó**, y eso es distinto de "no hay más señal".

Criterios, idénticos a los de siempre: el gate de D-042 (efecto ≥ +0,001, ≥3/5 folds, IC95 pareado
que excluya el cero, segunda partición ≥3/4) y la doble métrica (§36) con la condición de que
ninguna frontera central empeore. La selección de qué arquitectura se confirma se hace con la
partición de 5 folds, y la **segunda partición de 4 folds** es la confirmación independiente: es un
control de selección, no una medida más.

El test no se toca. Ninguna submission se genera aquí.

Entregables: ``experiments/v16_arquitectura.json``, ``reports/v16_arquitectura.md``,
``reports/v16_arquitectura_comparativa.csv``, ``features/v16_oof_*.npy``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from phase2_ponderacion import EncodedMLP, build_features  # noqa: E402
from v13_p1b_fusion_tokens import (PARES, bootstrap_diferencia, complementariedad,  # noqa: E402
                                   logreg, metricas_frontera, metricas_globales)

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v16_arquitectura.md"
COMPARATIVA = cfg.REPORTS / "v16_arquitectura_comparativa.csv"
RESULTADO = cfg.EXPERIMENTS / "v16_arquitectura.json"

DELTA_MINIMO = 0.001
FOLDS_POSITIVOS_MINIMOS = 3
FOLDS_SEGUNDA_PARTICION = 4
FOLDS_SEGUNDA_PARTICION_MINIMOS = 3
#: Rejilla del peso en A1. Gruesa a proposito: es un unico parametro y no se quiere que el
#: grid sea otra fuente de sobreajuste.
PESOS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
#: Fraccion del train exterior que se aparta para elegir el combinador.
FRACCION_INTERIOR = 0.2


def partido_agrupado(indices: np.ndarray, clusters: np.ndarray,
                     generador: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Reparte filas en dos bloques sin que un grupo de duplicados cruce entre ellos."""
    grupos_unicos = np.unique(clusters[indices])
    generador.shuffle(grupos_unicos)
    corte = int(len(grupos_unicos) * FRACCION_INTERIOR)
    en_validacion = set(grupos_unicos[:corte].tolist())
    mascara = np.array([grupo in en_validacion for grupo in clusters[indices]])
    return indices[~mascara], indices[mascara]


def combinar(lineal: np.ndarray, red: np.ndarray, arquitectura: str,
             ajustado) -> np.ndarray:
    """Aplica el combinador elegido a las probabilidades de las dos cabezas."""
    if arquitectura == "promedio_fijo":
        return 0.5 * lineal + 0.5 * red
    if arquitectura == "peso_aprendido":
        peso, _ = ajustado
        return peso * lineal + (1.0 - peso) * red
    if arquitectura == "combinador_logistico":
        _, modelo = ajustado
        juntos = np.hstack([lineal, red])
        return modelo.predict_proba(juntos)
    raise ValueError(arquitectura)


def main() -> int:
    """Compara tres arquitecturas de combinacion sobre las mismas 279 variables de V12."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clusters_ids = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters_ids, n_splits=5, n_repeats=1, scheme="group")
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])

    base, _, _, _, _ = build_features()
    base = np.concatenate([base, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    print(f"[matriz] {base.shape[1]} variables de V12, sin variables nuevas", flush=True)

    referencia = np.load(cfg.FEATURES / "oof_ensemble_v12.npy")
    metricas_referencia = metricas_globales(referencia, clase)
    fronteras_referencia = metricas_frontera(referencia, clase, PARES)
    por_fold_referencia = [float(f1_score(
        y[fold["val_idx"]], np.array(cfg.CLASSES)[np.argmax(referencia[fold["val_idx"]], axis=1)],
        average="macro", labels=cfg.CLASSES)) for fold in folds]
    print(f"[A0 referencia] {metricas_referencia['macro_f1']} | folds "
          f"{[round(v, 4) for v in por_fold_referencia]}", flush=True)

    resultados: dict[str, dict] = {}
    for arquitectura in ("peso_aprendido", "combinador_logistico"):
        generador = np.random.default_rng(cfg.SEED)
        probabilidades = np.zeros((len(y), len(cfg.CLASSES)))
        por_fold = []
        elecciones = []
        for fold in folds:
            validacion = fold["val_idx"]
            entrenamiento = np.concatenate([otro["val_idx"] for otro in folds
                                            if otro["fold"] != fold["fold"]])
            interior_train, interior_valid = partido_agrupado(entrenamiento, clusters_ids,
                                                               generador)
            # 1) las dos cabezas entrenadas en el interior, para ajustar el combinador
            lineal_in = logreg().fit(base[interior_train], y[interior_train])
            red_in = EncodedMLP().fit(base[interior_train], y[interior_train])
            p_lineal_in = lineal_in.predict_proba(base[interior_valid])
            p_red_in = red_in.model.predict_proba(red_in.scaler.transform(base[interior_valid]))
            if arquitectura == "peso_aprendido":
                mejor_peso, mejor_valor = 0.5, -np.inf
                for peso in PESOS:
                    combinacion = peso * p_lineal_in + (1 - peso) * p_red_in
                    nombres_in = np.array(cfg.CLASSES)[np.argmax(combinacion, axis=1)]
                    valor = float(f1_score(y[interior_valid], nombres_in,
                                           average="macro", labels=cfg.CLASSES))
                    if valor > mejor_valor:
                        mejor_peso, mejor_valor = peso, valor
                ajustado = (mejor_peso, None)
            else:
                conjunto = np.hstack([p_lineal_in, p_red_in])
                modelo = LogisticRegression(C=1.0, max_iter=3000, random_state=cfg.SEED)
                modelo.fit(conjunto, y[interior_valid])
                ajustado = (None, modelo)
            # 2) las dos cabezas entrenadas en todo el train exterior
            lineal = logreg().fit(base[entrenamiento], y[entrenamiento])
            red = EncodedMLP().fit(base[entrenamiento], y[entrenamiento])
            p_lineal = lineal.predict_proba(base[validacion])
            p_red = red.model.predict_proba(red.scaler.transform(base[validacion]))
            probabilidades[validacion] = combinar(p_lineal, p_red, arquitectura, ajustado)
            etiquetas = np.array(cfg.CLASSES)[np.argmax(probabilidades[validacion], axis=1)]
            por_fold.append(float(f1_score(y[validacion], etiquetas, average="macro",
                                            labels=cfg.CLASSES)))
            elecciones.append({"fold": fold["fold"],
                               "peso": mejor_peso if arquitectura == "peso_aprendido" else None})
            print(f"    fold {fold['fold']}: {por_fold[-1]:.4f}"
                  + (f" (peso {mejor_peso})" if arquitectura == "peso_aprendido" else ""), flush=True)
        metricas = metricas_globales(probabilidades, clase)
        delta = metricas["macro_f1"] - metricas_referencia["macro_f1"]
        folds_positivos = sum(1 for a, b in zip(por_fold, por_fold_referencia) if a > b)
        resultados[arquitectura] = {
            "macro_f1": metricas["macro_f1"], "accuracy": metricas["accuracy"],
            "delta": round(delta, 4), "folds_positivos": folds_positivos,
            "por_fold": [round(v, 4) for v in por_fold], "elecciones": elecciones,
            "fronteras": metricas_frontera(probabilidades, clase, PARES),
            "complementariedad": complementariedad(referencia, probabilidades, clase),
            "oof": probabilidades,
        }
        np.save(cfg.FEATURES / f"v16_oof_{arquitectura}.npy", probabilidades)
        print(f"[{arquitectura}] {metricas['macro_f1']} | delta {delta:+.4f} | folds positivos "
              f"{folds_positivos}/5", flush=True)

    # ---------------------------------------------------------------- segunda particion
    mejor_arquitectura = max(resultados, key=lambda k: resultados[k]["macro_f1"])
    print(f"[seleccion] mejor en la particion de 5: {mejor_arquitectura}", flush=True)
    folds4 = cvmod.make_folds(y, groups=clusters_ids, n_splits=FOLDS_SEGUNDA_PARTICION,
                              n_repeats=1, scheme="group")
    generador = np.random.default_rng(cfg.SEED + 7)
    probabilidades4 = np.zeros((len(y), len(cfg.CLASSES)))
    por_fold4 = []
    for fold in folds4:
        validacion = fold["val_idx"]
        entrenamiento = np.concatenate([otro["val_idx"] for otro in folds4
                                        if otro["fold"] != fold["fold"]])
        interior_train, interior_valid = partido_agrupado(entrenamiento, clusters_ids, generador)
        lineal_in = logreg().fit(base[interior_train], y[interior_train])
        red_in = EncodedMLP().fit(base[interior_train], y[interior_train])
        p_lineal_in = lineal_in.predict_proba(base[interior_valid])
        p_red_in = red_in.model.predict_proba(red_in.scaler.transform(base[interior_valid]))
        if mejor_arquitectura == "peso_aprendido":
            mejor_peso, mejor_valor = 0.5, -np.inf
            for peso in PESOS:
                combinacion = peso * p_lineal_in + (1 - peso) * p_red_in
                nombres_in = np.array(cfg.CLASSES)[np.argmax(combinacion, axis=1)]
                valor = float(f1_score(y[interior_valid], nombres_in, average="macro",
                                       labels=cfg.CLASSES))
                if valor > mejor_valor:
                    mejor_peso, mejor_valor = peso, valor
            ajustado = (mejor_peso, None)
        else:
            modelo = LogisticRegression(C=1.0, max_iter=3000, random_state=cfg.SEED)
            modelo.fit(np.hstack([p_lineal_in, p_red_in]), y[interior_valid])
            ajustado = (None, modelo)
        lineal = logreg().fit(base[entrenamiento], y[entrenamiento])
        red = EncodedMLP().fit(base[entrenamiento], y[entrenamiento])
        probabilidades4[validacion] = combinar(lineal.predict_proba(base[validacion]),
                                               red.model.predict_proba(
                                                   red.scaler.transform(base[validacion])),
                                               mejor_arquitectura, ajustado)
        por_fold4.append(float(f1_score(
            y[validacion], np.array(cfg.CLASSES)[np.argmax(probabilidades4[validacion], axis=1)],
            average="macro", labels=cfg.CLASSES)))
        print(f"    2a particion, fold {fold['fold']}: {por_fold4[-1]:.4f}", flush=True)
    _, por_fold4_referencia = _oof(base, y, folds4)
    diferencias4 = [a - b for a, b in zip(por_fold4, por_fold4_referencia)]
    positivos4 = sum(1 for d in diferencias4 if d > 0)
    print(f"[2a particion] diferencias {[round(d, 4) for d in diferencias4]} | positivos "
          f"{positivos4}/4", flush=True)

    mejor = resultados[mejor_arquitectura]
    intervalo = bootstrap_diferencia(referencia, mejor["oof"], clase, len(cfg.CLASSES))
    fronteras_mejoras = sum(1 for etiqueta in PARES.values()
                            if mejor["fronteras"][etiqueta]["accuracy"]
                            > fronteras_referencia[etiqueta]["accuracy"])
    criterios = {
        "efecto_minimo": bool(mejor["delta"] >= DELTA_MINIMO),
        "folds_positivos": bool(mejor["folds_positivos"] >= FOLDS_POSITIVOS_MINIMOS),
        "significacion": bool(intervalo[0] > 0),
        "segunda_particion": bool(positivos4 >= FOLDS_SEGUNDA_PARTICION_MINIMOS),
        "fronteras": bool(fronteras_mejoras >= len(PARES)),
    }
    adopta = all(criterios.values())

    comparativa = [{"arquitectura": "A0 promedio fijo 0,5+0,5 (produccion)",
                    "macro_f1": metricas_referencia["macro_f1"], "delta": 0.0,
                    "folds_positivos": "-"}]
    for arquitectura, medidas in resultados.items():
        comparativa.append({"arquitectura": arquitectura, "macro_f1": medidas["macro_f1"],
                            "delta": medidas["delta"],
                            "folds_positivos": f"{medidas['folds_positivos']}/5"})
    pd.DataFrame(comparativa).to_csv(COMPARATIVA, index=False)

    lineas = [
        "# V16 — Arquitectura del meta-modelo", "",
        "## 1. Qué se cambia", "",
        "V12 combina las dos cabezas con un promedio fijo 0,5 + 0,5. Es una decisión de diseño de la "
        "primera versión y **nunca se optimizó**. V13, V14 y V15 apuntan a que el cuello de botella "
        "es arquitectónico, así que aquí se mide la palanca: mismo protocolo, mismas 279 variables, "
        "tres formas de combinar.", "",
        "- **A0**: promedio fijo 0,5 + 0,5 (producción, y su OOF ya estaba calculado).",
        "- **A1, peso aprendido**: sigue siendo un promedio, pero el peso se elige por fold sobre una "
        "partición **interior** del train exterior (20 %, agrupada por duplicados), de modo que no se "
        "mira la validación.",
        "- **A2, combinador logístico**: las doce probabilidades de las dos cabezas alimentan una "
        "LogReg, es decir apilamiento en dos niveles. Es la que puede dar peso distinto a cada "
        "clase y a cada cabeza, y la que de verdad cambia la arquitectura.", "",
        "Ninguna introduce información nueva. Si alguna mejora, el hallazgo es que el modelo llevaba "
        "desde el principio una decisión de combinación no medida, que es distinto de \"no hay más "
        "señal\".", "",
        "## 2. Resultado en la partición de 5 folds", "",
        "| Arquitectura | macro-F1 | Delta | Folds positivos |", "|---|---|---|---|",
        f"| A0 promedio fijo (producción) | {metricas_referencia['macro_f1']} | - | - |",
    ]
    for arquitectura, medidas in resultados.items():
        lineas.append(f"| {arquitectura} | {medidas['macro_f1']} | {medidas['delta']:+.4f} | "
                      f"{medidas['folds_positivos']}/5 |")
    lineas += ["", "Pesos aprendidos por fold en A1: "
                  + ", ".join(f"fold {e['fold']} = {e['peso']}" for e in
                              resultados["peso_aprendido"]["elecciones"]), "",
              "## 3. Segunda partición (4 folds, arquitectura elegida en la de 5)", "",
              f"La arquitectura que se confirma es **{mejor_arquitectura}**.", "",
              "| Fold | A0 | Elegida | Diferencia |", "|---|---|---|---|"]
    for indice, (a, b) in enumerate(zip(por_fold4, por_fold4_referencia), start=1):
        lineas.append(f"| {indice} | {b:.4f} | {a:.4f} | {a - b:+.4f} |")
    lineas += ["", f"**Positivos: {positivos4}/4**", "",
              "## 4. Doble métrica (§36) y complementariedad", "",
              "| Frontera | A0 | Elegida |", "|---|---|---|"]
    for etiqueta in PARES.values():
        lineas.append(f"| {etiqueta} | {fronteras_referencia[etiqueta]['accuracy']} | "
                      f"{mejor['fronteras'][etiqueta]['accuracy']} |")
    complemento = mejor["complementariedad"]
    lineas += ["", f"Complementariedad global: rescate {complemento['global']['rescate']:,} · daño "
                  f"{complemento['global']['dano']:,} · **neto {complemento['global']['neto']:+,}**", "",
              "## 5. Gate D-042", "", "| Criterio | Valor | Cumple |", "|---|---|---|"]
    valores = {
        "efecto_minimo": f"delta {mejor['delta']:+.4f} >= {DELTA_MINIMO}",
        "folds_positivos": f"{mejor['folds_positivos']}/5 >= {FOLDS_POSITIVOS_MINIMOS}",
        "significacion": f"IC95 [{intervalo[0]:+.4f}, {intervalo[1]:+.4f}]",
        "segunda_particion": f"{positivos4}/4 >= {FOLDS_SEGUNDA_PARTICION_MINIMOS}",
        "fronteras": f"{fronteras_mejoras}/{len(PARES)} mejoran",
    }
    for nombre, cumple in criterios.items():
        lineas.append(f"| {nombre.replace('_', ' ')} | {valores[nombre]} | "
                      f"{'si' if cumple else '**no**'} |")
    lineas += ["", f"**Decision: {'ADOPTAR' if adopta else 'NO ADOPTAR'}**", ""]
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    RESULTADO.write_text(json.dumps({
        "fase": "V16 arquitectura del meta-modelo", "fecha": "2026-09-26",
        "n_variables": int(base.shape[1]), "variables_nuevas": 0,
        "arquitecturas": {a: {k: v for k, v in m.items() if k not in ("oof", "fronteras",
                                                                     "complementariedad")}
                          for a, m in resultados.items()},
        "referencia": {"macro_f1": metricas_referencia["macro_f1"],
                       "per_fold": [round(v, 4) for v in por_fold_referencia]},
        "elegida": mejor_arquitectura,
        "segunda_particion": {"por_fold": [round(v, 4) for v in por_fold4],
                              "por_fold_referencia": [round(v, 4) for v in por_fold4_referencia],
                              "positivos": positivos4},
        "bootstrap_ic95": [round(v, 4) for v in intervalo],
        "criterios": criterios, "adopta": adopta,
        "fronteras_referencia": fronteras_referencia, "fronteras_mejor": mejor["fronteras"],
        "complementariedad": complemento, "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nelegida {mejor_arquitectura} | delta {mejor['delta']:+.4f} | 2a particion "
          f"{positivos4}/4 | IC95 [{intervalo[0]:+.4f}, {intervalo[1]:+.4f}] | "
          f"DECISION: {'ADOPTAR' if adopta else 'NO ADOPTAR'}")
    return 0


def _oof(matriz: np.ndarray, y: np.ndarray, folds: list[dict]):
    """OOF del promedio fijo, para tener la referencia de la segunda particion."""
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
        por_fold.append(float(f1_score(
            y[validacion], np.array(cfg.CLASSES)[np.argmax(probabilidades[validacion], axis=1)],
            average="macro", labels=cfg.CLASSES)))
    return probabilidades, por_fold


if __name__ == "__main__":
    raise SystemExit(main())
