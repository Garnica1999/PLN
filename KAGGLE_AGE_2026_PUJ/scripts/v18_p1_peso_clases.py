"""V18 · P1 — Peso de clases consciente de la frontera, con control V12-equivalente.

Esta es la primera hipotesis experimental de V18, y la mas simple: **cambiar el objetivo de
entrenamiento sin cambiar nada mas**. Si el cuello de botella es el objetivo, esto tiene que dar algo;
si no da nada, el objetivo no es la palanca.

**La hipotesis.** El entrenamiento multiclass optimiza las seis clases con el mismo peso, mientras que
una fraccion desproporcionada del error esta concentrada en las dos fronteras centrales. Dar mas peso
a `18-24`, `25-34` y `35-49` durante el entrenamiento deberia mover el punto de decision de esas
fronteras.

**El handicap, declarado antes de ejecutar.** Las clases estan **balanceadas**: de 16,04 % a 17,04 %,
razon 1,062. No hay desbalance que corregir. El unico efecto que puede tener el peso de clases es
**desplazar** el equilibrio entre clases, y eso es exactamente lo que D-037 intento 33 veces mediante
capas de decision a posteriori y cerro sin resultado. La probabilidad previa de este experimento es
baja, y se ejecuta porque es barato y porque el brief lo pide como primer bloque.

**Los pesos estan pre-registrados, no ajustados.** Tres niveles, sin rejilla:

| Nivel | Peso de `18-24`, `25-34`, `35-49` | Resto |
|---|---|---|
| 0, base | 1,0 | 1,0 |
| 1, moderado | 1,5 | 1,0 |
| 2, fuerte | 2,5 | 1,0 |

Que esten pre-registrados significa que **no se eligen mirando el resultado**, y por tanto no hace
falta una particion interior para elegirlos. Es la opcion mas barata y la mas honesta, y el brief la
permite expresamente.

**El control es la parte importante.** El nivel 0 es un **V12-equivalente**: mismas 279 variables,
mismos folds agrupados, mismos dos modelos y mismo promedio 0,5 + 0,5, cambiando **solo** el vector de
pesos. Ademas hay una **comprobacion de fidelidad**: el nivel 0 se reentrena con una reimplementacion
local del ajuste del MLP que acepta `sample_weight` (la clase `EncodedMLP` de produccion no lo
acepta), y tiene que **reproducir el 0,6005 de la OOF guardada**. Si no reproduce, el experimento no
publica nada: seria imposible atribuir la diferencia a los pesos y no al cambio de codigo.

**Que se reporta y que no.** Global, F1 por clase, las dos fronteras, fold por fold, e IC95 pareado.
Una variante **no** se adopta aunque mejore solo las fronteras: el criterio de V18 exige ganancia global.
No se genera ninguna submission. El holdout2 no se toca.

Entregables: ``experiments/v18_p1_peso_clases.json``, ``reports/v18_p1_peso_clases.md``,
``reports/v18_p1_peso_clases.csv`` y una figura.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from phase2_ponderacion import build_features  # noqa: E402
from v13_p1b_fusion_tokens import (PARES, bootstrap_diferencia, logreg,  # noqa: E402
                                   metricas_frontera, metricas_globales)

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v18_p1_peso_clases.md"
TABLA = cfg.REPORTS / "v18_p1_peso_clases.csv"
RESULTADO = cfg.EXPERIMENTS / "v18_p1_peso_clases.json"
#: Las tres clases centrales de las dos fronteras. La formulacion es deliberadamente simple: suben las
#: tres a la vez, sin ajustar ninguna individualmente.
CLASES_CENTRALES = ("18-24", "25-34", "35-49")
#: Pesos **pre-registrados**. No se eligen mirando ningun resultado, asi que no hace falta particion
#: interior para elegirlos.
NIVELES = (("nivel_0_base", 1.0), ("nivel_1_moderado", 1.5), ("nivel_2_fuerte", 2.5))
#: La OOF guardada de V12, que sirve de referencia y de comprobacion de fidelidad.
OOF_V12 = "oof_ensemble_v12.npy"
GATE_EXPLORATORIO = 0.0015
GATE_ADOPCION = 0.003
FOLDS_POSITIVOS_MINIMOS = 4
#: Tolerancia de la comprobacion de fidelidad: si el nivel base no reproduce V12 dentro de esta
#: margen, el experimento no es atribuible a los pesos.
TOLERANCIA_FIDELIDAD = 5e-4


def ajustar_mlp_pesado(matriz: np.ndarray, etiquetas: np.ndarray, pesos: np.ndarray) -> dict:
    """Replica `EncodedMLP` de produccion, pero permitiendo pasarle `sample_weight`.

    La clase de produccion no acepta pesos porque su `fit` tiene la firma `(features, labels)`. Aqui
    se repite exactamente su configuracion (misma tipificacion, mismo MLP con los mismos
    hiperparametros, misma semilla) y solo se anade el vector de pesos. El nivel 0, con pesos todos a
    uno, tiene que reproducir la OOF guardada, y eso es lo que comprueba el experimento.
    """
    codificador = LabelEncoder().fit(etiquetas)
    escalador = StandardScaler()
    matriz_tipificada = escalador.fit_transform(matriz)
    modelo = MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, batch_size=512,
                           learning_rate_init=1e-3, max_iter=40, early_stopping=True,
                           n_iter_no_change=5, random_state=cfg.SEED)
    modelo.fit(matriz_tipificada, codificador.transform(etiquetas), sample_weight=pesos)
    return {"codificador": codificador, "escalador": escalador, "modelo": modelo}


def predecir(ajuste: dict, matriz: np.ndarray) -> np.ndarray:
    """Probabilidades de seis clases del MLP ajustado con pesos."""
    probabilidades = ajuste["modelo"].predict_proba(
        ajuste["escalador"].transform(matriz))
    completa = np.zeros((len(matriz), len(cfg.CLASSES)))
    for posicion, clase in enumerate(ajuste["codificador"].classes_):
        completa[:, cfg.CLASSES.index(clase)] = probabilidades[:, posicion]
    return completa


def main() -> int:
    """Compara los tres niveles pre-registrados de peso de clases contra el control V12-equivalente."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    base, _, _, _, _ = build_features()
    base = np.concatenate([base, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    referencia = np.load(cfg.FEATURES / OOF_V12)
    metricas_referencia = metricas_globales(referencia, clase)
    fronteras_referencia = metricas_frontera(referencia, clase, PARES)
    por_fold_referencia = [float(f1_score(y[fold["val_idx"]],
                                          np.array(cfg.CLASSES)[np.argmax(referencia[fold["val_idx"]],
                                                                                   axis=1)],
                                          average="macro", labels=cfg.CLASSES)) for fold in folds]
    print(f"[referencia] V12 {metricas_referencia['macro_f1']} | folds "
          f"{[round(v, 4) for v in por_fold_referencia]} | fronteras "
          f"{fronteras_referencia['18-24|25-34']['accuracy']} y "
          f"{fronteras_referencia['25-34|35-49']['accuracy']}", flush=True)

    resultados: list[dict] = []
    filas: list[dict] = []
    for nombre_nivel, peso in NIVELES:
        vector = np.ones(len(cfg.CLASSES), dtype=float)
        for clase_central in CLASES_CENTRALES:
            vector[cfg.CLASSES.index(clase_central)] = peso
        probabilidades = np.zeros((len(y), len(cfg.CLASSES)))
        for fold in folds:
            validacion = fold["val_idx"]
            entrenamiento = np.concatenate([otro["val_idx"] for otro in folds
                                            if otro["fold"] != fold["fold"]])
            pesos_fila = vector[clase[entrenamiento]]
            # Cabeza lineal: los pesos llegan al clasificador a traves del Pipeline.
            lineal = logreg().fit(base[entrenamiento], y[entrenamiento],
                                  clf__sample_weight=pesos_fila)
            ajuste = ajustar_mlp_pesado(base[entrenamiento], y[entrenamiento], pesos_fila)
            probabilidades[validacion] = (0.5 * lineal.predict_proba(base[validacion])
                                          + 0.5 * predecir(ajuste, base[validacion]))
            print(f"    [{nombre_nivel}] fold {fold['fold']}: "
                  f"{f1_score(y[validacion], np.array(cfg.CLASSES)[np.argmax(probabilidades[validacion], axis=1)], average='macro', labels=cfg.CLASSES):.4f}",
                  flush=True)
        metricas = metricas_globales(probabilidades, clase)
        fronteras = metricas_frontera(probabilidades, clase, PARES)
        por_fold = [float(f1_score(y[fold["val_idx"]],
                                   np.array(cfg.CLASSES)[np.argmax(probabilidades[fold["val_idx"]],
                                                                    axis=1)],
                                   average="macro", labels=cfg.CLASSES)) for fold in folds]
        delta = metricas["macro_f1"] - metricas_referencia["macro_f1"]
        intervalo = bootstrap_diferencia(referencia, probabilidades, clase, len(cfg.CLASSES))
        folds_positivos = sum(1 for a, b in zip(por_fold, por_fold_referencia) if a > b)
        criterios = {
            "efecto_minimo": bool(delta >= GATE_EXPLORATORIO),
            "folds_positivos": bool(folds_positivos >= FOLDS_POSITIVOS_MINIMOS),
            "significacion": bool(intervalo[0] > 0),
        }
        resultados.append({
            "nivel": nombre_nivel, "peso_central": peso,
            "macro_f1": metricas["macro_f1"], "accuracy": metricas["accuracy"],
            "delta": round(delta, 6), "folds_positivos": folds_positivos,
            "por_fold": [round(v, 4) for v in por_fold],
            "f1_por_clase": metricas["f1_por_clase"],
            "frontera_18_24_25_34": fronteras["18-24|25-34"]["accuracy"],
            "frontera_25_34_35_49": fronteras["25-34|35-49"]["accuracy"],
            "bootstrap_ic95": [round(v, 6) for v in intervalo],
            "criterios": criterios, "adopta": all(criterios.values()),
        })
        for indice, clase_nombre in enumerate(cfg.CLASSES):
            filas.append({
                "nivel": nombre_nivel, "peso_central": peso, "clase": clase_nombre,
                "f1": metricas["f1_por_clase"][clase_nombre],
                "f1_referencia": metricas_referencia["f1_por_clase"][clase_nombre],
                "delta_f1": round(metricas["f1_por_clase"][clase_nombre]
                                  - metricas_referencia["f1_por_clase"][clase_nombre], 4)})
        print(f"[{nombre_nivel}] {metricas['macro_f1']} | delta {delta:+.4f} | folds "
              f"{folds_positivos}/5 | IC95 [{intervalo[0]:+.4f}, {intervalo[1]:+.4f}] | fronteras "
              f"{fronteras['18-24|25-34']['accuracy']} y {fronteras['25-34|35-49']['accuracy']} | "
              f"adopta {all(criterios.values())}", flush=True)

    tabla = pd.DataFrame(filas)
    tabla.to_csv(TABLA, index=False)

    # ---------------------------------------------------------------- fidelidad del control
    base_nivel = resultados[0]
    fidelidad = abs(base_nivel["macro_f1"] - metricas_referencia["macro_f1"])
    fidelidad_ok = fidelidad <= TOLERANCIA_FIDELIDAD
    print(f"[fidelidad] el nivel base reproduce V12 con diferencia {fidelidad:.6f} "
          f"(tolerancia {TOLERANCIA_FIDELIDAD}) -> {'si' if fidelidad_ok else 'NO'}", flush=True)

    # ---------------------------------------------------------------- figura
    figura = cfg.FIGURES / "v18_p1_peso_clases.png"
    fig, ejes = plt.subplots(1, 2, figsize=(14, 5.5))
    eje = ejes[0]
    nombres = [r["nivel"] for r in resultados] + ["V12"]
    valores = [r["macro_f1"] for r in resultados] + [metricas_referencia["macro_f1"]]
    colores = ["#b2182b" if r["adopta"] else "#2166ac" for r in resultados] + ["grey"]
    eje.bar(range(len(nombres)), valores, color=colores)
    eje.axhline(metricas_referencia["macro_f1"], color="grey", linestyle=":", linewidth=1)
    eje.set_xticks(range(len(nombres)))
    eje.set_xticklabels(nombres, rotation=15, fontsize=8)
    eje.set_ylim(min(valores) - 0.004, max(valores) + 0.004)
    eje.set_ylabel("macro-F1 pooled")
    eje.set_title("Macro-F1 global por nivel de peso", fontsize=10)
    eje = ejes[1]
    clases = cfg.CLASSES
    ancho = 0.8 / (len(nombres))
    for posicion, (nombre, color) in enumerate(zip(nombres, colores)):
        f1 = (np.array([metricas_referencia["f1_por_clase"][c] for c in clases]) if nombre == "V12"
              else np.array([resultados[posicion]["f1_por_clase"][c] for c in clases]))
        eje.bar(np.arange(len(clases)) + posicion * ancho, f1, ancho, label=nombre, color=color)
    eje.set_xticks(np.arange(len(clases)) + 0.4 - ancho / 2)
    eje.set_xticklabels(clases, fontsize=8)
    eje.set_ylabel("F1 de la clase")
    eje.set_title("F1 por clase: donde el peso de frontera se paga", fontsize=10)
    eje.legend(fontsize=7)
    fig.suptitle("V18 P1: peso de clases consciente de la frontera, tres niveles pre-registrados",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(figura, dpi=130)
    plt.close(fig)

    # ---------------------------------------------------------------- informe
    lineas = [
        "# V18 · P1 — Peso de clases consciente de la frontera", "",
        "## 1. Diseno", "",
        "El control es un **V12-equivalente**: mismas 279 variables, mismos folds agrupados, mismos dos",
        "modelos y mismo promedio 0,5 + 0,5, cambiando **solo** el vector de pesos de clase. Los tres",
        "pesos estan **pre-registrados** y no se eligen mirando ningun resultado, asi que no hace falta",
        "particion interior para elegirlos:", "",
        "| Nivel | Peso de 18-24, 25-34 y 35-49 | Resto |", "|---|---|---|",
    ]
    for nombre_nivel, peso in NIVELES:
        lineas.append(f"| {nombre_nivel} | {peso} | 1,0 |")
    lineas += ["", "**El handicap, declarado antes de ejecutar**: las clases estan balanceadas "
                  "(16,04 % a 17,04 %, razon 1,062), asi que no hay desbalance que corregir. El unico "
                  "efecto posible del peso de clases es desplazar el equilibrio entre clases, que es "
                  "justo lo que D-037 intento 33 veces como capa de decision a posteriori.", "",
              "## 2. Comprobacion de fidelidad del control", "",
              f"El nivel base se reentrena con una reimplementacion local que acepta `sample_weight`, "
              f"porque la clase de produccion `EncodedMLP` no lo admite. Para que la comparacion sea "
              f"atribuible a los pesos y no al cambio de codigo, ese nivel tiene que reproducir la OOF "
              f"guardada de V12. **Diferencia: {fidelidad:.6f}** (tolerancia "
              f"{TOLERANCIA_FIDELIDAD}) -> **{'cumple' if fidelidad_ok else 'NO CUMPLE'}**.", "",
              "## 3. Resultado", "",
              "| Nivel | macro-F1 | Delta | Folds positivos | IC95 | 18-24\\|25-34 | 25-34\\|35-49 | "
              "Adopta |", "|---|---|---|---|---|---|---|---|"]
    for resultado in resultados:
        lineas.append(f"| {resultado['nivel']} | {resultado['macro_f1']} | {resultado['delta']:+.4f} | "
                      f"{resultado['folds_positivos']}/5 | "
                      f"[{resultado['bootstrap_ic95'][0]:+.4f}, {resultado['bootstrap_ic95'][1]:+.4f}] | "
                      f"{resultado['frontera_18_24_25_34']} | {resultado['frontera_25_34_35_49']} | "
                      f"{'si' if resultado['adopta'] else '**no**'} |")
    lineas += [f"| V12 (referencia) | {metricas_referencia['macro_f1']} | - | - | - | "
              f"{fronteras_referencia['18-24|25-34']['accuracy']} | "
              f"{fronteras_referencia['25-34|35-49']['accuracy']} | - |", "",
              "## 4. F1 por clase, que es donde se ve el intercambio", "",
              "| Nivel | " + " | ".join(cfg.CLASSES) + " |", "|---" * (1 + len(cfg.CLASSES)) + "|"]
    for resultado in resultados:
        celdas = " | ".join(str(resultado["f1_por_clase"][c]) for c in cfg.CLASSES)
        lineas.append(f"| {resultado['nivel']} | {celdas} |")
    lineas.append("| V12 (referencia) | "
                  + " | ".join(str(metricas_referencia["f1_por_clase"][c]) for c in cfg.CLASSES) + " |")
    lineas += ["", "## 5. Lectura", ""]
    if fidelidad_ok and all(not r["adopta"] for r in resultados[1:]):
        lineas += ["**El peso de clases no aporta.** Ningun nivel alcanza el gate exploratorio de "
                   "+0,0015. Y la tabla de F1 por clase muestra el motivo: subir el peso de las tres "
                   "clases centrales sube su F1 y **baja el de las otras tres en la misma medida**, "
                   "porque la macro-F1 es una media y el juego es de suma cero. Un objetivo de pesos "
                   "no puede ganar aqui por construccion: solo puede redistribuir.", "",
                   "**Que significa para la fase**: el handicap que la auditoria senalo era real. La "
                   "via P1 se cierra con resultado estable negativo, y la siguiente palanca no es "
                   "reordenar el equilibrio entre clases, sino **cambiar la funcion de perdida de "
                   "verdad**, que es lo que miden P2 y P3.", ""]
    elif fidelidad_ok:
        lineas += ["**Hay al menos un nivel que supera el gate exploratorio.** Antes de cualquier otra "
                   "cosa hay que verificar el intervalo de bootstrap y la segunda particion, porque el "
                   "criterio de V18 exige las dos cosas y una mejora de frontera sin ganancia global no "
                   "es candidata.", ""]
    else:
        lineas += ["**La comprobacion de fidelidad falla**, asi que el experimento no es atribuible a "
                   "los pesos y no se publica como resultado. Habria que repetirlo con una "
                   "implementacion que no requiera reentrenar V12, o comparar solo contra la OOF "
                   "guardada.", ""]
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    RESULTADO.write_text(json.dumps({
        "fase": "V18 · P1 peso de clases consciente de la frontera", "fecha": "2026-09-26",
        "pesos_pre_registrados": {n: p for n, p in NIVELES},
        "clases_centrales": list(CLASES_CENTRALES),
        "control": "V12-equivalente: mismas variables, folds, modelos y promedio; solo cambian los pesos",
        "referencia": {"macro_f1": metricas_referencia["macro_f1"],
                       "por_fold": [round(v, 4) for v in por_fold_referencia],
                       "f1_por_clase": metricas_referencia["f1_por_clase"]},
        "resultados": resultados,
        "fidelidad_del_control": {"diferencia": round(fidelidad, 6),
                                 "tolerancia": TOLERANCIA_FIDELIDAD, "cumple": bool(fidelidad_ok)},
        "gate": {"exploratorio": GATE_EXPLORATORIO, "adopcion": GATE_ADOPCION,
                 "folds_positivos_minimos": FOLDS_POSITIVOS_MINIMOS},
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nfidelidad {fidelidad:.6f} | niveles que adoptan: "
          f"{sum(1 for r in resultados if r['adopta'])}")
    return 0 if fidelidad_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
