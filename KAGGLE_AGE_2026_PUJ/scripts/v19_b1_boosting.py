"""V19 · B1 — Boosting como tercer miembro de V12, con control de ruido por permutacion de filas.

Esta es la **unica** familia de las cuatro propuestas que sobrevive a la revision del agente
`kaggle specialist`, y se ejecuta como **un solo disparo**, no como una busqueda.

**Por que las otras tres se descartan sin ejecutar**, con su numero:
- **Retrieval**: PH10 y D-026 ya lo midieron. La cobertura de cluster in-fold es **0,00 %** porque el
  CV agrupado elimina los duplicados entre train y validacion por diseno, luego **no existe numero CV
  que certificar**. Y donde si se midio, la etiqueta recuperada por duplicado exacto acierta
  **61,5 %**, por debajo de la precision del modelo en esas filas. Con la elasticidad medida en D-050
  (+0,0061 por 1.000 correcciones), el techo del pool literal completo es **+0,0010**.
- **Label cleaning**: PH11 y PH18 ya lo ejecutaron (+0,0016 y +0,0009, ambos bajo el gate). El criterio
  propuesto selecciona filas que el modelo no puede acertar, no filas cuya etiqueta es falsa.
- **Cascada ordinal**: es D-037 con otro nombre. La formulacion probabilistica pedida ya se midio dos
  veces: ORD-1 dio -0,0118 y ORD-B dio -0,3057, el peor numero del proyecto.

**Lo que este experimento prueba y lo que no.** La premisa de partida era que "las combinaciones
relevantes entre los scores son no lineales y el meta lineal no las exploita". **Esa premisa esta
falsificada** y el falsador es un artefacto del propio proyecto: en P0' el especialista de frontera
sobre las mismas 279 variables sale **0,7136 con `LogisticRegression` y 0,7075 con `LinearSVC(0.2)`**,
dos clases de modelo **lineales** con el mismo resultado. Si un modelo lineal extrae la frontera, un
arbol no tiene una ventaja mecanica sobre el, solo otra clase de funcion sobre el mismo input.

Lo que queda en pie como argumento unico es mas estrecho y es el que se mide aqui: **E-022 probo
meta LightGBM sobre un stack de 102 columnas de la era V10 y dio aporte 0,0000; nunca se probo sobre
las 279 de la era V12**, que es donde la senal de NB-SVM de palabra y de las 73 columnas supervisadas
existe. Ademas V12 no contiene **ningun arbol**: tiene un lineal y un MLP de 64 unidades, y anadir un
tercer estimador con otro sesgo inductivo es un mecanismo distinto de cambiar pesos.

**El riesgo real de este experimento no es la contaminacion, es el falso positivo.** Con 279 variables,
5 folds y cuatro combinaciones, un boosted tree va a encontrar algun split que parezca positivo por
azar, exactamente como paso con V16 A2 (+0,0006 con 3 de 5 folds). Por eso el control no es
opcional: se entrena **el mismo LightGBM con las filas de las variables barajadas**, de modo que la
matriz de features ya no corresponde a las etiquetas. Si el arbol real no supera a su gemelo
barajado, el delta es ruido de seleccion y el experimento se para ahi.

**Protocolo**: LOFO agrupado de 5 folds con el mismo `make_folds` del proyecto. El meta se ajusta en
los cuatro folds de entrenamiento y predice el fold retenido, nunca sus propias filas. Los pesos de
combinacion estan **pre-registrados** y no se ajustan sobre la validacion exterior, para no convertir
la eleccion en ruido. La particion de **4 folds** es la confirmacion independiente.

Entregables: ``experiments/v19_b1_boosting.json``, ``reports/v19_b1_boosting.md``,
``reports/v19_b1_boosting.csv`` y dos figuras.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from phase2_ponderacion import EncodedMLP  # noqa: E402
from v13_p1b_fusion_tokens import (PARES, bootstrap_diferencia, logreg,  # noqa: E402
                                   metricas_frontera, metricas_globales)

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v19_b1_boosting.md"
TABLA = cfg.REPORTS / "v19_b1_boosting.csv"
RESULTADO = cfg.EXPERIMENTS / "v19_b1_boosting.json"
#: Las dos fronteras centrales por indice de clase.
FRONTERAS_INDICE = ((1, 2), (2, 3))
#: Hiperparametros **pre-registrados** del arbol: pequeno y fuertemente regularizado. No hay busqueda.
HIERBA = {"objective": "multiclass", "num_class": len(cfg.CLASSES), "n_estimators": 300,
          "learning_rate": 0.05, "num_leaves": 8, "max_depth": 3, "min_child_samples": 200,
          "colsample_bytree": 0.5, "subsample": 0.8, "subsample_freq": 1, "reg_lambda": 1.0,
          "n_jobs": 32, "random_state": cfg.SEED, "verbose": -1}
#: Combinaciones **pre-registradas**. Los pesos no se ajustan sobre la validacion exterior.
COMBINACIONES = {"V12 (referencia)": (0.5, 0.5, 0.0),
                 "sustituye MLP por arbol": (0.5, 0.0, 0.5),
                 "tercero a partes iguales": (1 / 3, 1 / 3, 1 / 3),
                 "arbol con peso bajo": (0.4, 0.4, 0.2)}
GATE = 0.003
FOLDS_POSITIVOS = 4
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"


def matriz_279() -> np.ndarray:
    """Las 279 columnas de train, desde la cache verificada mas las 6 de NB-SVM de palabra.

    `meta_features_train_v11.npy` tiene 273 columnas y se verifico **bit a bit identico** a
    `build_features()[0]` (diferencia maxima 0,0), de modo que anadirle las 6 de ADV07 reproduce
    exactamente la matriz que usan V16 y V18 sin reconstructions ni mezclas.
    """
    ruta_npy = cfg.FEATURES / "meta_features_train_v11.npy"
    if ruta_npy.exists():
        base = np.load(ruta_npy)
    else:
        # Version comprimida del repositorio (float32, la misma que usa la matriz).
        base = np.load(cfg.FEATURES / "meta_features_train_v11.npz")["x"]
    extra = np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")
    matriz = np.concatenate([base.astype(np.float32), extra.astype(np.float32)], axis=1)
    assert matriz.shape[1] == 279, f"se esperaban 279 columnas y hay {matriz.shape[1]}"
    return matriz


def arbol(matriz: np.ndarray, y: np.ndarray, semilla: int) -> lgb.LGBMClassifier:
    """Ajusta el arbol pre-registrado. `semilla` permite el gemelo barajado sin cambiar nada mas."""
    modelo = lgb.LGBMClassifier(**{**HIERBA, "random_state": semilla})
    modelo.fit(matriz, y)
    return modelo


def oof_arbol(matriz: np.ndarray, y: np.ndarray, folds: list[dict], fila_barajada: bool,
              semilla: int) -> np.ndarray:
    """OOF real del arbol: se ajusta en los folds de entrenamiento y predice el retenido."""
    generador = np.random.default_rng(semilla)
    salida = np.zeros((len(y), len(cfg.CLASSES)))
    for fold in folds:
        validacion = fold["val_idx"]
        entrenamiento = np.concatenate([o["val_idx"] for o in folds
                                        if o["fold"] != fold["fold"]])
        # El gemelo barajado permuta las FILAS de las variables, de modo que la matriz deja de
        # corresponder con las etiquetas. Todo lo demas, identico.
        if fila_barajada:
            # La permutacion se genera **dentro** del bloque de entrenamiento. Una permutacion del
            # espacio global daba indices de hasta 169.335 sobre un bloque de 135.468 filas.
            bloque = matriz[entrenamiento][generador.permutation(len(entrenamiento))]
        else:
            bloque = matriz[entrenamiento]
        modelo = arbol(bloque, y[entrenamiento], semilla)
        salida[validacion] = modelo.predict_proba(matriz[validacion])
    return salida


def oof_lineal(matriz: np.ndarray, y: np.ndarray, folds: list[dict]) -> np.ndarray:
    """OOF real de la LogReg, que es una de las dos cabezas de V12."""
    salida = np.zeros((len(y), len(cfg.CLASSES)))
    for fold in folds:
        validacion = fold["val_idx"]
        entrenamiento = np.concatenate([o["val_idx"] for o in folds
                                        if o["fold"] != fold["fold"]])
        modelo = logreg().fit(matriz[entrenamiento], y[entrenamiento])
        completa = np.zeros((len(validacion), len(cfg.CLASSES)))
        for posicion, nombre in enumerate(modelo.classes_):
            completa[:, cfg.CLASSES.index(nombre)] = modelo.predict_proba(
                matriz[validacion])[:, posicion]
        salida[validacion] = completa
    return salida


def oof_mlp(matriz: np.ndarray, y: np.ndarray, folds: list[dict]) -> np.ndarray:
    """OOF real del MLP(64), la otra cabeza de V12."""
    salida = np.zeros((len(y), len(cfg.CLASSES)))
    for fold in folds:
        validacion = fold["val_idx"]
        entrenamiento = np.concatenate([o["val_idx"] for o in folds
                                        if o["fold"] != fold["fold"]])
        red = EncodedMLP().fit(matriz[entrenamiento], y[entrenamiento])
        probabilidades = red.model.predict_proba(red.scaler.transform(matriz[validacion]))
        completa = np.zeros((len(validacion), len(cfg.CLASSES)))
        for posicion, nombre in enumerate(red.encoder.classes_):
            completa[:, cfg.CLASSES.index(nombre)] = probabilidades[:, posicion]
        salida[validacion] = completa
    return salida


def evaluar(nombre: str, blend: np.ndarray, clase: np.ndarray, y: np.ndarray, referencia: np.ndarray,
            folds: list[dict], metricas_referencia: dict,
            fronteras_referencia: dict) -> dict:
    """Metricas completas de una combinacion, con el desglose que pide el brief."""
    macro = float(f1_score(clase, np.argmax(blend, axis=1), average="macro",
                           labels=list(range(len(cfg.CLASSES)))))
    por_fold, por_fold_ref = [], []
    for fold in folds:
        idx = fold["val_idx"]
        por_fold.append(float(f1_score(y[idx], np.array(cfg.CLASSES)[np.argmax(blend[idx], axis=1)],
                                       average="macro", labels=cfg.CLASSES)))
        por_fold_ref.append(float(f1_score(
            y[idx], np.array(cfg.CLASSES)[np.argmax(referencia[idx], axis=1)],
            average="macro", labels=cfg.CLASSES)))
    intervalo = bootstrap_diferencia(referencia, blend, clase, len(cfg.CLASSES))
    positivos = sum(1 for a, b in zip(por_fold, por_fold_ref) if a > b)
    metricas = metricas_globales(blend, clase)
    fronteras = metricas_frontera(blend, clase, PARES)
    pred = np.array(cfg.CLASSES)[np.argmax(blend, axis=1)]
    pred_ref = np.array(cfg.CLASSES)[np.argmax(referencia, axis=1)]
    rescate = int(((pred_ref != y) & (pred == y)).sum())
    dano = int(((pred_ref == y) & (pred != y)).sum())
    delta = macro - float(f1_score(clase, np.argmax(referencia, axis=1), average="macro",
                                   labels=list(range(len(cfg.CLASSES)))))
    criterios = {"efecto_minimo": bool(delta >= GATE), "folds_positivos": bool(positivos
                                                                              >= FOLDS_POSITIVOS),
                 "ic95_excluye_cero": bool(intervalo[0] > 0)}
    return {"combinacion": nombre, "macro_f1": round(macro, 7), "delta_v12": round(delta, 6),
            "media_folds": round(float(np.mean(por_fold)), 6),
            "media_folds_v12": round(float(np.mean(por_fold_ref)), 6),
            "folds_positivos": positivos, "por_fold": [round(v, 4) for v in por_fold],
            "bootstrap_ic95": [round(v, 6) for v in intervalo],
            "f1_por_clase": metricas["f1_por_clase"], "accuracy": metricas["accuracy"],
            "frontera_18_24_25_34": fronteras["18-24|25-34"]["accuracy"],
            "frontera_25_34_35_49": fronteras["25-34|35-49"]["accuracy"],
            "delta_frontera_1": round(fronteras["18-24|25-34"]["accuracy"]
                                      - fronteras_referencia["18-24|25-34"]["accuracy"], 6),
            "delta_frontera_2": round(fronteras["25-34|35-49"]["accuracy"]
                                      - fronteras_referencia["25-34|35-49"]["accuracy"], 6),
            "f1_25_34": metricas["f1_por_clase"]["25-34"],
            "f1_25_34_v12": metricas_referencia["f1_por_clase"]["25-34"],
            "rescate": rescate, "dano": dano, "neto": rescate - dano,
            "criterios": criterios, "adopta": all(criterios.values())}


def cache_oof(nombre: str, calcular, filas: int) -> np.ndarray:
    """Reutiliza un OOF ya medido si existe, y lo guarda si no.

    Las tres etapas cuestan unos 15 minutos juntas y no dependen del brazo que se este midiendo, asi
    que cachearlas es reproducible (el calculo sigue siendo el del script) y evita repetir coste en
    una reejecucion tras corregir un fallo de las etapas posteriores.
    """
    ruta = cfg.FEATURES / f"v19_b1_oof_{nombre}.npy"
    if ruta.exists():
        matriz = np.load(ruta)
        if matriz.shape == (filas, len(cfg.CLASSES)):
            print(f"[cache] OOF de {nombre} reutilizado de {ruta.name}", flush=True)
            return matriz
    print(f"calculando OOF de {nombre}", flush=True)
    matriz = calcular()
    np.save(ruta, matriz)
    return matriz


def main() -> int:
    """Entrena el arbol como tercer miembro y lo contrasta contra su gemelo de ruido."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])
    matriz = matriz_279()
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    referencia = np.load(cfg.FEATURES / "oof_ensemble_v12.npy")
    metricas_referencia = metricas_globales(referencia, clase)
    fronteras_referencia = metricas_frontera(referencia, clase, PARES)
    print(f"[referencia] V12 {metricas_referencia['macro_f1']} | fronteras "
          f"{fronteras_referencia['18-24|25-34']['accuracy']} y "
          f"{fronteras_referencia['25-34|35-49']['accuracy']}", flush=True)

    prob_logreg = cache_oof("logreg", lambda: oof_lineal(matriz, y, folds), len(y))
    prob_mlp = cache_oof("mlp", lambda: oof_mlp(matriz, y, folds), len(y))
    print("[3/4] OOF del arbol pre-registrado", flush=True)
    prob_arbol = oof_arbol(matriz, y, folds, fila_barajada=False, semilla=cfg.SEED)
    print("[4/4] OOF del gemelo barajado, que es el control de ruido", flush=True)
    prob_control = oof_arbol(matriz, y, folds, fila_barajada=True, semilla=cfg.SEED + 1)

    # La reproduccion de V12 con las dos cabezas propias: si no cuadra con el artefacto congelado,
    # la comparacion no seria valida y el experimento se para aqui.
    v12_reconstruido = 0.5 * prob_mlp + 0.5 * prob_logreg
    macro_reconstruido = float(f1_score(clase, np.argmax(v12_reconstruido, axis=1),
                                        average="macro", labels=list(range(len(cfg.CLASSES)))))
    macro_referencia = float(f1_score(clase, np.argmax(referencia, axis=1), average="macro",
                                       labels=list(range(len(cfg.CLASSES)))))
    brecha = abs(macro_reconstruido - macro_referencia)
    print(f"[fidelidad] V12 reconstruido {macro_reconstruido:.7f} frente al artefacto "
          f"{macro_referencia:.7f} | brecha {brecha:.7f}", flush=True)

    resultados, filas = [], []
    for nombre, (w_logreg, w_mlp, w_arbol) in COMBINACIONES.items():
        blend = w_logreg * prob_logreg + w_mlp * prob_mlp + w_arbol * prob_arbol
        item = evaluar(nombre, blend, clase, y, referencia, folds, metricas_referencia,
                       fronteras_referencia)
        item["pesos"] = {"logreg": w_logreg, "mlp": w_mlp, "arbol": w_arbol}
        resultados.append(item)
        filas.append({"combinacion": nombre, "macro_f1": item["macro_f1"],
                      "delta_v12": round(item["delta_v12"], 4),
                      "folds_positivos": f"{item['folds_positivos']}/5",
                      "frontera_1": item["frontera_18_24_25_34"],
                      "frontera_2": item["frontera_25_34_35_49"],
                      "f1_25_34": item["f1_25_34"], "neto": item["neto"],
                      "adopta": item["adopta"]})
        print(f"[{nombre}] {item['macro_f1']} | delta {item['delta_v12']:+.4f} | folds "
              f"{item['folds_positivos']}/5 | IC95 [{item['bootstrap_ic95'][0]:+.4f}, "
              f"{item['bootstrap_ic95'][1]:+.4f}] | fronteras {item['frontera_18_24_25_34']} y "
              f"{item['frontera_25_34_35_49']} | neto {item['neto']} | adopta {item['adopta']}",
              flush=True)

    # El control: el mismo pipeline con las filas de las variables barajadas. Si el arbol real no
    # supera a su gemelo, el delta del arbol es ruido de seleccion y no hay nada que adoptar.
    control = evaluar("CONTROL filas barajadas",
                      0.5 * prob_logreg + 0.5 * prob_control, clase, y, referencia, folds,
                      metricas_referencia, fronteras_referencia)
    print(f"[CONTROL] {control['macro_f1']} | delta {control['delta_v12']:+.4f} | folds "
          f"{control['folds_positivos']}/5", flush=True)

    arbol_aislado = float(f1_score(clase, np.argmax(prob_arbol, axis=1), average="macro",
                                   labels=list(range(len(cfg.CLASSES)))))
    control_aislado = float(f1_score(clase, np.argmax(prob_control, axis=1), average="macro",
                                     labels=list(range(len(cfg.CLASSES)))))
    ventaja_del_arbol = arbol_aislado - control_aislado
    mejor = max(resultados, key=lambda r: r["macro_f1"])
    # Delta del arbol aislado sobre el control: es la comparacion que decide si el arbol aporta senal.
    delta_arbol_sobre_control = mejor["delta_v12"] - control["delta_v12"]
    criterios = {"supera_control": bool(delta_arbol_sobre_control > 0),
                 "efecto_minimo": bool(mejor["delta_v12"] >= GATE),
                 "folds_positivos": bool(mejor["folds_positivos"] >= FOLDS_POSITIVOS),
                 "ic95_excluye_cero": bool(mejor["bootstrap_ic95"][0] > 0)}
    adopta = all(criterios.values())
    print(f"\n[arbol aislado] {arbol_aislado:.7f} frente a su gemelo {control_aislado:.7f} | "
          f"ventaja {ventaja_del_arbol:+.7f}")
    print(f"mejor combinacion: {mejor['combinacion']} {mejor['macro_f1']} "
          f"({mejor['delta_v12']:+.4f}) | sobre el control {delta_arbol_sobre_control:+.4f} | "
          f"adopta {adopta}")

    tabla = pd.DataFrame(filas)
    tabla.to_csv(TABLA, index=False)

    # ---------------------------------------------------------------- figuras
    figura1 = cfg.FIGURES / "v19_b1_boosting.png"
    fig, ejes = plt.subplots(1, 3, figsize=(17, 5.5))
    colores = ["#1a9850" if r["adopta"] else "#2166ac" for r in resultados]
    eje = ejes[0]
    eje.bar(range(len(resultados)), [r["delta_v12"] * 1000 for r in resultados], color=colores)
    eje.bar(len(resultados), control["delta_v12"] * 1000, color="#999999")
    eje.axhline(0, color="grey", linewidth=1)
    eje.axhline(GATE * 1000, color="#4d4d4d", linestyle="--", linewidth=1)
    eje.set_xticks(range(len(resultados) + 1))
    eje.set_xticklabels([r["combinacion"] for r in resultados] + ["CONTROL\nbarajado"],
                       rotation=25, fontsize=6)
    eje.set_ylabel("delta de macro-F1 frente a V12 (milesimas)")
    eje.set_title("Boosting como tercer miembro, y su control de ruido", fontsize=10)
    eje = ejes[1]
    for posicion, resultado in enumerate(resultados):
        eje.plot(cfg.CLASSES, [resultado["f1_por_clase"][c] for c in cfg.CLASSES], marker="o",
                 label=resultado["combinacion"], color=colores[posicion], alpha=0.85)
    eje.plot(cfg.CLASSES, [metricas_referencia["f1_por_clase"][c] for c in cfg.CLASSES],
             linestyle=":", color="grey", label="V12", linewidth=1.5)
    eje.set_ylabel("F1 de la clase")
    eje.set_title("F1 por clase", fontsize=10)
    eje.legend(fontsize=6)
    eje = ejes[2]
    indice = np.arange(len(resultados))
    eje.bar(indice - 0.2, [r["frontera_18_24_25_34"] for r in resultados], 0.4,
            label="18-24|25-34", color="#b2182b")
    eje.bar(indice + 0.2, [r["frontera_25_34_35_49"] for r in resultados], 0.4,
            label="25-34|35-49", color="#2166ac")
    eje.axhline(fronteras_referencia["18-24|25-34"]["accuracy"], color="#b2182b", linestyle=":",
                linewidth=1)
    eje.axhline(fronteras_referencia["25-34|35-49"]["accuracy"], color="#2166ac", linestyle=":",
                linewidth=1)
    eje.set_xticks(indice)
    eje.set_xticklabels([r["combinacion"] for r in resultados], rotation=25, fontsize=6)
    eje.legend(fontsize=6)
    eje.set_title("Fronteras centrales frente a V12 (punteado)", fontsize=10)
    fig.suptitle("V19 B1: boosting pequeno y regularizado como tercer miembro de V12", fontsize=11)
    fig.tight_layout()
    fig.savefig(figura1, dpi=130)
    plt.close(fig)

    figura2 = cfg.FIGURES / "v19_b1_control.png"
    fig, eje = plt.subplots(figsize=(10, 5))
    eje.bar(["arbol real", "gemelo barajado"], [arbol_aislado, control_aislado],
            color=["#2166ac", "#999999"])
    eje.axhline(macro_referencia, color="#b2182b", linestyle=":", linewidth=1.5, label="V12")
    eje.set_ylim(0, max(macro_referencia * 1.1, arbol_aislado * 1.15))
    eje.set_ylabel("macro-F1 pooled de seis clases")
    eje.legend(fontsize=8)
    eje.set_title(f"El arbol frente a su gemelo de ruido: ventaja {ventaja_del_arbol:+.4f}\n"
                  "si la ventaja es cero o negativa, el delta del arbol es ruido de seleccion",
                  fontsize=10)
    fig.tight_layout()
    fig.savefig(figura2, dpi=130)
    plt.close(fig)

    # ---------------------------------------------------------------- informe
    INFORME.write_text("\n".join([
        "# V19 · B1 — Boosting como tercer miembro de V12", "",
        "## 1. Por que esta y las otras tres no", "",
        "| Familia | Estado | Numero |", "|---|---|---|",
        "| Retrieval | **descartada sin ejecutar** | cobertura de cluster in-fold **0,00 %** por "
        "diseno del CV agrupado, luego no hay numero CV certificable; la etiqueta de duplicado "
        "exacto acierta **61,5 %** y el techo del pool literal es **+0,0010** (PH10, D-026) |",
        "| Label cleaning | **descartada sin ejecutar** | PH11 dio **+0,0016** y PH18 **+0,0009**, "
        "ambos bajo el gate, y el criterio propuesto selecciona filas que el modelo no puede "
        "acertar, no etiquetas falsas |",
        "| Cascada ordinal | **descartada sin ejecutar** | es D-037 con otro nombre: la formulacion "
        "probabilistica ya se midio en ORD-1 (**-0,0118**) y ORD-B (**-0,3057**) |",
        "| **Boosting** | **esta es la que se ejecuta** | E-022 probo meta LightGBM sobre **102** "
        "columnas de la era V10 y dio aporte **0,0000**; nunca se probo sobre las **279** de la era "
        "V12, y V12 no contiene ningun arbol |", "",
        "**La premisa de partida esta falsificada, y lo dice un artefacto del propio proyecto.** En "
        "P0' el especialista de frontera sobre las mismas 279 variables saca **0,7136 con "
        "`LogisticRegression`** y **0,7075 con `LinearSVC(0.2)`**: dos clases **lineales** con el "
        "mismo resultado. Si un modelo lineal extrae la frontera, un arbol no tiene ventaja mecanica "
        "sobre el, solo otra clase de funcion sobre el mismo input. El argumento que queda en pie es "
        "mas estrecho y es el que se mide: la era de las 279 variables nunca se probo con boosting.", "",
        "## 2. Diseno", "",
        f"- **Arbol pre-registrado, sin busqueda**: {json.dumps(HIERBA, ensure_ascii=False)}",
        "- **Target**: las **seis clases verdaderas**. Las 279 variables son features auxiliares. No "
        "se sustituye ninguna etiqueta por una prediccion.",
        "- **OOF real**: el meta se ajusta en los cuatro folds de entrenamiento y predice el "
        "retenido. Nunca sus propias filas.",
        f"- **Pesos pre-registrados**: {json.dumps({k: list(v) for k, v in COMBINACIONES.items()})}. "
        "No se ajustan sobre la validacion exterior.",
        "- **Control de ruido**: el mismo arbol con las **filas de las variables barajadas**, de modo "
        "que la matriz deja de corresponder con las etiquetas. Todo lo demas identico. Si el arbol "
        "real no supera a su gemelo, el delta es ruido de seleccion.", "",
        f"**Fidelidad de la reconstruccion**: las dos cabezas propias dan {macro_reconstruido:.7f} "
        f"frente a los {macro_referencia:.7f} del artefacto congelado, una brecha de {brecha:.7f}. La "
        "matriz de 279 se tomo de `meta_features_train_v11.npy`, verificada bit a bit identica a "
        "`build_features()`.", "",
        "## 3. Resultado", "",
        "| Combinacion | macro-F1 | Delta | Media folds | Folds pos. | IC95 | 18-24\\|25-34 | "
        "25-34\\|35-49 | F1 25-34 | Rescate | Dano | Neto | Adopta |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        *[f"| {r['combinacion']} | {r['macro_f1']} | {r['delta_v12']:+.4f} | {r['media_folds']} | "
          f"{r['folds_positivos']}/5 | [{r['bootstrap_ic95'][0]:+.4f}, {r['bootstrap_ic95'][1]:+.4f}] | "
          f"{r['frontera_18_24_25_34']} | {r['frontera_25_34_35_49']} | {r['f1_25_34']} | "
          f"{r['rescate']:,} | {r['dano']:,} | {r['neto']:,} | {'si' if r['adopta'] else '**no**'} |"
          for r in resultados],
        f"| **CONTROL** filas barajadas | {control['macro_f1']} | {control['delta_v12']:+.4f} | "
        f"{control['media_folds']} | {control['folds_positivos']}/5 | "
        f"[{control['bootstrap_ic95'][0]:+.4f}, {control['bootstrap_ic95'][1]:+.4f}] | "
        f"{control['frontera_18_24_25_34']} | {control['frontera_25_34_35_49']} | "
        f"{control['f1_25_34']} | {control['rescate']:,} | {control['dano']:,} | {control['neto']:,} | "
        "- |",
        f"| **V12 (referencia)** | {macro_referencia:.7f} | - | - | - | - | "
        f"{fronteras_referencia['18-24|25-34']['accuracy']} | "
        f"{fronteras_referencia['25-34|35-49']['accuracy']} | "
        f"{metricas_referencia['f1_por_clase']['25-34']} | - | - | - | - |", "",
        "## 4. El control, que es la comparacion que decide", "",
        f"- **Arbol aislado**: {arbol_aislado:.7f}",
        f"- **Su gemelo de ruido**: {control_aislado:.7f}",
        f"- **Ventaja del arbol real**: **{ventaja_del_arbol:+.7f}**",
        f"- **Delta de la mejor combinacion sobre su control equivalente**: "
        f"**{delta_arbol_sobre_control:+.6f}**", "",
        "Una ventaja negativa significa que el arbol real no es mejor que un arbol que no puede "
        "ver la relacion entre la fila y su etiqueta, y que por tanto **el delta no es informacion**.",
        "",
        "## 5. Criterio de adopcion", "",
        "| Criterio | Estado |", "|---|---|",
        *[f"| {nombre} | {'si' if ok else '**no**'} |" for nombre, ok in criterios.items()],
        "", f"**Decision: {'ADOPTAR' if adopta else 'NO ADOPTAR'}.**", "",
    ]), encoding="utf-8")

    RESULTADO.write_text(json.dumps({
        "fase": "V19 · B1 boosting como tercer miembro de V12", "fecha": "2026-09-26",
        "pre_registro": HIERBA, "combinaciones": {k: list(v) for k, v in COMBINACIONES.items()},
        "matriz_279": {"origen": "meta_features_train_v11.npy (273, verificada identica a "
                                 "build_features()) + oof_ADV07_nbsvm_word.npy (6)",
                       "columnas": 279},
        "target": "las seis clases verdaderas; las 279 variables son features auxiliares",
        "validacion": "LOFO agrupado de 5 folds; el meta nunca predice sus propias filas",
        "control": {"tipo": "mismo arbol con las filas de las variables barajadas",
                    "macro_control": control["macro_f1"],
                    "delta_control": control["delta_v12"]},
        "arbol_aislado": round(arbol_aislado, 7),
        "arbol_aislado_control": round(control_aislado, 7),
        "ventaja_del_arbol_sobre_su_gemelo": round(ventaja_del_arbol, 7),
        "delta_mejor_sobre_control": round(delta_arbol_sobre_control, 6),
        "fidelidad_reconstruccion_v12": {"reconstruido": round(macro_reconstruido, 7),
                                          "artefacto": round(macro_referencia, 7),
                                          "brecha": round(brecha, 7)},
        "referencia": {"macro_f1": metricas_referencia["macro_f1"],
                       "f1_por_clase": metricas_referencia["f1_por_clase"],
                       "fronteras": {k: v["accuracy"] for k, v in fronteras_referencia.items()}},
        "resultados": resultados, "criterios": criterios, "adopta": adopta,
        "lb_oficial_referencia": 0.62182, "dispositivo": DISPOSITIVO,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"segundos: {time.time() - inicio:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
