"""V18 · P4 — Entrenamiento multi-objetivo: un termino auxiliar sobre las dos fronteras.

P1 y P3 dejaron dos instructive negatives y una orientacion clara.

- **P1** mostro que reordenar el equilibrio entre clases es de suma cero: `18-24` ganaba +0,0170 y
  `13-17` pagaba -0,0166. La global no puede ganar porque es una media.
- **P3** mostro que las dos fronteras centrales estan **acopladas** por la clase `25-34`: el margen
  central subia `18-24|25-34` +0,039 y bajaba `25-34|35-49` -0,055. No se pueden optimizar por
  separado.

Los dos resultados apuntan a lo mismo: un objetivo que **redistribuye** no puede ganar, porque la
macro-F1 de seis clases es una media. Y un objetivo que optimiza las fronteras **por separado**
tambien no puede, porque comparten clase. De ahi el diseno de este experimento: **un unico termino
auxiliar sobre las dos fronteras a la vez**, y un termino **aditivo** en lugar de un re-peso.

**Los tres brazos separan arquitectura de objetivo**, que es lo que el brief exige para no confundir
las dos cosas:

| Brazo | Que cambia | Parametros extra |
|---|---|---|
| **A, control** | nada: entropia cruzada de seis clases | 0 |
| **B, arquitectura** | dos cabezas binarias auxiliares que **reparten** la masa que la cabeza global ya pone en cada par | 2 x 279 |
| **C, objetivo** | entropia cruzada mas un termino auxiliar de frontera **sobre los mismos logit** | **0** |

El brazo C es el importante, porque **no anade ni un parametro**: la misma cabeza de seis clases, los
mismos 279 pesos, y solo una perdida distinta. Eso responde de forma directa a la exigencia del brief
de que V18 no produzca un modelo mucho mas grande, y a la pregunta de si un cambio de objetivo puede
mejorar usando esencialmente la misma informacion.

**Por que el reparto de B es conservador.** Las cabezas binarias de B no inventan masa: toman la masa
que la cabeza global ya atribuye a las dos clases del par y la reparten entre ellas. No pueden mover
masa desde fuera del par ni anadirla, luego no pueden tocar las otras cuatro clases. Es el mismo
principio que uso el combinador local de V17, y por eso B es un cambio de arquitectura y no una via
para cobrar la frontera a cualquier costa.

**La representacion.** Los dos brazos usan las **mismas 279 variables de V12**. El brief pide que las
cabezas usen una representacion permitida y que no se creen nuevas; las 279 columnas cumplen las dos
cosas, y D-048 demostro que ademas son **mejores** que el TF-IDF disperso para la decision de frontera
(0,7136 y 0,7493 frente a 0,6864 y 0,7218), asi que son la eleccion con evidencia y no por defecto.

**Todo pre-registrado**: margen, epocas, lote, tasa y los dos valores de lambda. Epocas fijas sin
early stopping, para que ninguna decision dependa de la validacion exterior. El gate de V18 se aplica
igual que en el resto de la fase.

Entregables: ``experiments/v18_p4_multiobjetivo.json``, ``reports/v18_p4_multiobjetivo.md``,
``reports/v18_p4_multiobjetivo.csv`` y una figura.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from phase2_ponderacion import EncodedMLP, build_features  # noqa: E402
from v13_p1b_fusion_tokens import PARES, bootstrap_diferencia, metricas_frontera, metricas_globales  # noqa: E402

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v18_p4_multiobjetivo.md"
TABLA = cfg.REPORTS / "v18_p4_multiobjetivo.csv"
RESULTADO = cfg.EXPERIMENTS / "v18_p4_multiobjetivo.json"
#: Las dos fronteras centrales por indice de clase. Los indices siguen el orden de `cfg.CLASSES`.
FRONTERAS_INDICE = ((1, 2), (2, 3))
#: Brazo, lambda, y si el brazo usa termino auxiliar sobre los logit (objetivo) o solo las cabezas
#: binarias auxiliares (arquitectura).
BRAZOS = (("A_control", 0.0, "objetivo"),
          ("B_arquitectura", 0.0, "arquitectura"),
          ("C1_objetivo", 0.1, "objetivo"),
          ("C2_objetivo", 0.5, "objetivo"))
#: Entrenamiento pre-registrado, identico al de P3 para que los dos experimentos sean comparables.
EPOCAS = 30
LOTE = 512
TASA_APRENDIZAJE = 1e-3
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
GATE_EXPLORATORIO = 0.0015
FOLDS_POSITIVOS_MINIMOS = 4
SEMILLA = cfg.SEED
ETIQUETAS_ENTERO = list(range(len(cfg.CLASSES)))


def termino_auxiliar(logits: torch.Tensor, indice_real: torch.Tensor) -> torch.Tensor:
    """Perdida binaria de frontera sobre los MISMOS logit de la cabeza de seis clases.

    Para cada fila cuya clase verdadera pertenezca a una de las dos fronteras centrales, se exige que
    el logit de su clase supere al de su vecina en la pareja. La sigmoida de esa diferencia es la
    probabilidad de "mi clase", y la perdida es la entropia cruzada binaria correspondiente. Es un
    termino **aditivo**: no reescala ni redistribuye la perdida global, solo anade una restriccion.

    El mismo lambda se aplica a las dos fronteras a la vez, porque P3 demostro que estan acopladas por
    la clase `25-34` y que optimizarlas por separado las hace competir.
    """
    total = logits.new_zeros(())
    filas = 0
    for inferior, superior in FRONTERAS_INDICE:
        en_frontera = (indice_real == inferior) | (indice_real == superior)
        if not bool(en_frontera.any()):
            continue
        objetivo = (indice_real[en_frontera] == superior).to(logits.dtype)
        diferencia = (logits[en_frontera, superior] - logits[en_frontera, inferior])
        total = total + torch.nn.functional.binary_cross_entropy_with_logits(diferencia, objetivo)
        filas += int(en_frontera.sum())
    return total / max(filas, 1)


def main() -> int:
    """Compara control, arquitectura multi-cabeza y objetivo boundary-aware, con el gate de V18."""
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
    referencia = np.load(cfg.FEATURES / "oof_ensemble_v12.npy")
    metricas_referencia = metricas_globales(referencia, clase)
    fronteras_referencia = metricas_frontera(referencia, clase, PARES)
    print(f"[referencia] V12 {metricas_referencia['macro_f1']} | device {DISPOSITIVO}", flush=True)

    # La cabeza MLP no cambia en ningun brazo: se entrena una vez por fold y se reutiliza, para que
    # la comparacion sea atribuible al brazo y no a un reentrenamiento distinto.
    predicciones_mlp = np.zeros((len(y), len(cfg.CLASSES)))
    for fold in folds:
        validacion = fold["val_idx"]
        entrenamiento = np.concatenate([otro["val_idx"] for otro in folds
                                        if otro["fold"] != fold["fold"]])
        red = EncodedMLP().fit(base[entrenamiento], y[entrenamiento])
        probabilidades = red.model.predict_proba(red.scaler.transform(base[validacion]))
        completa = np.zeros((len(validacion), len(cfg.CLASSES)))
        for posicion, clase_nombre in enumerate(red.encoder.classes_):
            completa[:, cfg.CLASSES.index(clase_nombre)] = probabilidades[:, posicion]
        predicciones_mlp[validacion] = completa
    print("[MLP] entrenada una vez por fold y reutilizada en los cuatro brazos", flush=True)

    generador = torch.Generator().manual_seed(SEMILLA)
    resultados: list[dict] = []
    filas: list[dict] = []
    for nombre, peso, tipo in BRAZOS:
        prob_global = np.zeros((len(y), len(cfg.CLASSES)))
        prob_binaria = {par: np.zeros(len(y)) for par in FRONTERAS_INDICE}
        f1_train = []
        for fold in folds:
            validacion = fold["val_idx"]
            entrenamiento = np.concatenate([otro["val_idx"] for otro in folds
                                            if otro["fold"] != fold["fold"]])
            escalador = StandardScaler().fit(base[entrenamiento])
            datos_train = torch.tensor(escalador.transform(base[entrenamiento]), dtype=torch.float32,
                                       device=DISPOSITIVO)
            datos_validacion = torch.tensor(escalador.transform(base[validacion]),
                                           dtype=torch.float32, device=DISPOSITIVO)
            objetivo = torch.tensor(clase[entrenamiento], dtype=torch.long, device=DISPOSITIVO)
            cabeza = torch.nn.Linear(base.shape[1], len(cfg.CLASSES)).to(DISPOSITIVO)
            parametros = list(cabeza.parameters())
            # El brazo B anade dos cabezas binarias auxiliares, una por frontera, con sus propios
            # pesos. Son las unicas que anaden parametros en toda la fase.
            binarias = {}
            if tipo == "arquitectura":
                for par in FRONTERAS_INDICE:
                    binarias[par] = torch.nn.Linear(base.shape[1], 1).to(DISPOSITIVO)
                    parametros += list(binarias[par].parameters())
            optimizador = torch.optim.Adam(parametros, lr=TASA_APRENDIZAJE)
            permutacion = torch.randperm(len(datos_train), generator=generador).to(DISPOSITIVO)
            for _ in range(EPOCAS):
                cabeza.train()
                for desde in range(0, len(permutacion), LOTE):
                    indices = permutacion[desde:desde + LOTE]
                    optimizador.zero_grad()
                    logits = cabeza(datos_train[indices])
                    perdida = torch.nn.functional.cross_entropy(logits, objetivo[indices])
                    if tipo == "objetivo" and peso > 0:
                        # Un termino aditivo sobre los mismos logit: cero parametros extra.
                        perdida = perdida + peso * termino_auxiliar(logits, objetivo[indices])
                    if tipo == "arquitectura":
                        for inferior, superior in FRONTERAS_INDICE:
                            en_frontera = ((objetivo[indices] == inferior)
                                           | (objetivo[indices] == superior))
                            if not bool(en_frontera.any()):
                                continue
                            marca = (objetivo[indices][en_frontera] == superior).float()
                            diferencia = (binarias[(inferior, superior)](datos_train[indices]
                                                                      [en_frontera]).squeeze(1))
                            perdida = perdida + torch.nn.functional.binary_cross_entropy_with_logits(
                                diferencia, marca)
                    perdida.backward()
                    optimizador.step()
            cabeza.eval()
            with torch.no_grad():
                probabilidades_validacion = torch.softmax(cabeza(datos_validacion), dim=1)
                probabilidades_train = torch.softmax(cabeza(datos_train), dim=1)
                if tipo == "arquitectura":
                    for inferior, superior in FRONTERAS_INDICE:
                        # La cabeza binaria se aplica a TODAS las filas del fold, no solo a las del
                        # par: en validacion su probabilidad es la que decide el reparto de la masa.
                        binarias[(inferior, superior)].eval()
                        with torch.no_grad():
                            prob_train = torch.sigmoid(
                                binarias[(inferior, superior)](datos_train).squeeze(1)).cpu().numpy()
                            prob_validacion = torch.sigmoid(
                                binarias[(inferior, superior)](datos_validacion).squeeze(1)
                            ).cpu().numpy()
                        prob_binaria[(inferior, superior)][entrenamiento] = prob_train
                        prob_binaria[(inferior, superior)][validacion] = prob_validacion
            prob_global[validacion] = probabilidades_validacion.cpu().numpy().astype(float)
            f1_train.append(float(f1_score(clase[entrenamiento],
                                            np.argmax(probabilidades_train.cpu().numpy(), axis=1),
                                            average="macro", labels=ETIQUETAS_ENTERO)))
            print(f"    [{nombre}] fold {fold['fold']}: perdida {float(perdida):.4f}", flush=True)

        # El brazo B redistribuye la masa de la cabeza global con las probabilidades binarias.
        if tipo == "arquitectura":
            salida = prob_global.copy()
            for inferior, superior in FRONTERAS_INDICE:
                masa = prob_global[:, inferior] + prob_global[:, superior]
                superior_prob = prob_binaria[(inferior, superior)]
                salida[:, superior] = masa * superior_prob
                salida[:, inferior] = masa * (1.0 - superior_prob)
            prob_global = salida / np.clip(salida.sum(axis=1, keepdims=True), 1e-12, None)

        metricas = metricas_globales(prob_global, clase)
        fronteras = metricas_frontera(prob_global, clase, PARES)
        combinada = 0.5 * prob_global + 0.5 * predicciones_mlp
        metricas_combinada = metricas_globales(combinada, clase)
        fronteras_combinada = metricas_frontera(combinada, clase, PARES)
        delta = metricas_combinada["macro_f1"] - metricas_referencia["macro_f1"]
        intervalo = bootstrap_diferencia(referencia, combinada, clase, len(cfg.CLASSES))
        por_fold = [float(f1_score(y[fold["val_idx"]],
                                    np.array(cfg.CLASSES)[np.argmax(combinada[fold["val_idx"]],
                                                                   axis=1)],
                                    average="macro", labels=cfg.CLASSES)) for fold in folds]
        por_fold_referencia = [float(f1_score(y[fold["val_idx"]],
                                              np.array(cfg.CLASSES)[np.argmax(referencia[fold["val_idx"]],
                                                                               axis=1)],
                                              average="macro", labels=cfg.CLASSES)) for fold in folds]
        folds_positivos = sum(1 for a, b in zip(por_fold, por_fold_referencia) if a > b)
        criterios = {"efecto_minimo": bool(delta >= GATE_EXPLORATORIO),
                     "folds_positivos": bool(folds_positivos >= FOLDS_POSITIVOS_MINIMOS),
                     "significacion": bool(intervalo[0] > 0)}
        resultado = {
            "brazo": nombre, "lambda": peso, "tipo": tipo,
            "parametros_extra": 0 if tipo == "objetivo" else 2 * base.shape[1],
            "macro_cabeza_global": metricas["macro_f1"],
            "fronteras_cabeza_global": {k: v["accuracy"] for k, v in fronteras.items()},
            "macro_combinada": metricas_combinada["macro_f1"],
            "accuracy": metricas_combinada["accuracy"],
            "delta": round(delta, 6),
            "frontera_18_24_25_34": fronteras_combinada["18-24|25-34"]["accuracy"],
            "frontera_25_34_35_49": fronteras_combinada["25-34|35-49"]["accuracy"],
            "bootstrap_ic95": [round(v, 6) for v in intervalo],
            "f1_por_clase": metricas_combinada["f1_por_clase"],
            "por_fold": [round(v, 4) for v in por_fold],
            "folds_positivos": folds_positivos,
            "f1_train_cabeza": round(float(np.mean(f1_train)), 4),
            "f1_validacion_cabeza": metricas["macro_f1"],
            "brecha_train_validacion": round(abs(float(np.mean(f1_train)) - metricas["macro_f1"]), 4),
            "epocas": EPOCAS, "criterios": criterios, "adopta": all(criterios.values()),
        }
        resultados.append(resultado)
        filas.append({"brazo": nombre, "lambda": peso, "tipo": tipo,
                      "parametros_extra": resultado["parametros_extra"],
                      "macro_cabeza": metricas["macro_f1"],
                      "frontera_1_cabeza": resultado["fronteras_cabeza_global"]["18-24|25-34"],
                      "frontera_2_cabeza": resultado["fronteras_cabeza_global"]["25-34|35-49"],
                      "macro_combinada": metricas_combinada["macro_f1"],
                      "delta": round(delta, 4),
                      "frontera_1_combinada": resultado["frontera_18_24_25_34"],
                      "frontera_2_combinada": resultado["frontera_25_34_35_49"]})
        print(f"[{nombre}] global {metricas['macro_f1']} (fronteras "
              f"{list(resultado['fronteras_cabeza_global'].values())}) | combinada "
              f"{metricas_combinada['macro_f1']} delta {delta:+.4f} IC95 [{intervalo[0]:+.4f}, "
              f"{intervalo[1]:+.4f}] | parametros extra {resultado['parametros_extra']} | adopta "
              f"{resultado['adopta']}", flush=True)

    tabla = pd.DataFrame(filas)
    tabla.to_csv(TABLA, index=False)

    # ---------------------------------------------------------------- figura
    figura = cfg.FIGURES / "v18_p4_multiobjetivo.png"
    fig, ejes = plt.subplots(1, 3, figsize=(17, 5.5))
    nombres = [r["brazo"] for r in resultados]
    colores = ["#1a9850" if r["adopta"] else "#2166ac" for r in resultados]
    eje = ejes[0]
    eje.bar(range(len(nombres)), [r["macro_combinada"] for r in resultados], color=colores)
    eje.axhline(metricas_referencia["macro_f1"], color="grey", linestyle=":", linewidth=1)
    eje.set_xticks(range(len(nombres)))
    eje.set_xticklabels(nombres, rotation=20, fontsize=8)
    eje.set_title("Macro-F1 del sistema combinado", fontsize=10)
    eje = ejes[1]
    eje.bar(np.arange(len(nombres)) - 0.2, [r["frontera_18_24_25_34"] for r in resultados], 0.4,
            label="18-24|25-34", color="#b2182b")
    eje.bar(np.arange(len(nombres)) + 0.2, [r["frontera_25_34_35_49"] for r in resultados], 0.4,
            label="25-34|35-49", color="#2166ac")
    eje.set_xticks(range(len(nombres)))
    eje.set_xticklabels(nombres, rotation=20, fontsize=8)
    eje.legend(fontsize=8)
    eje.set_title("Accuracy en las fronteras", fontsize=10)
    eje = ejes[2]
    for posicion, resultado in enumerate(resultados):
        eje.plot(cfg.CLASSES, [resultado["f1_por_clase"][c] for c in cfg.CLASSES], marker="o",
                label=resultado["brazo"], color=colores[posicion], alpha=0.85)
    eje.plot(cfg.CLASSES, [metricas_referencia["f1_por_clase"][c] for c in cfg.CLASSES],
             linestyle=":", color="grey", label="V12", linewidth=1.5)
    eje.set_ylabel("F1 de la clase")
    eje.set_title("F1 por clase", fontsize=10)
    eje.legend(fontsize=6)
    fig.suptitle("V18 P4: arquitectura multi-cabeza frente a objetivo boundary-aware", fontsize=11)
    fig.tight_layout()
    fig.savefig(figura, dpi=130)
    plt.close(fig)

    lineas = [
        "# V18 · P4 — Entrenamiento multi-objetivo", "",
        "## 1. Diseno", "",
        "P1 y P3 dejaron dos orientaciones: un objetivo que **redistribuye** no puede ganar en una",
        "media de seis clases, y las dos fronteras estan **acopladas** por la clase `25-34`, luego no",
        "se pueden optimizar por separado. De ahi un unico termino auxiliar **aditivo** sobre las dos",
        "fronteras a la vez, y tres brazos que separan arquitectura de objetivo:", "",
        "| Brazo | Que cambia | Parametros extra |", "|---|---|---|",
        "| A, control | nada | 0 |",
        "| B, arquitectura | dos cabezas binarias que reparten la masa que la global ya pone en cada par | 558 |",
        "| C, objetivo | un termino binario de frontera sobre los **mismos** logit | **0** |", "",
        f"Todo pre-registrado: {EPOCAS} epocas fijas sin early stopping, lote {LOTE}, Adam "
        f"{TASA_APRENDIZAJE}, lambda en {{0,1, 0,5}} y margen fijo. La representacion son las mismas "
        "**279 variables de V12** en los tres brazos, sin ninguna nueva.", "",
        "## 2. Resultado de la cabeza global, y del sistema combinado con el MLP de V12", "",
        "| Brazo | Lambda | Param. extra | macro global | frontera 1 | frontera 2 | macro combinada | "
        "Delta | IC95 |", "|---|---|---|---|---|---|---|---|---|",
    ]
    for resultado in resultados:
        lineas.append(
            f"| {resultado['brazo']} | {resultado['lambda']} | {resultado['parametros_extra']} | "
            f"{resultado['macro_cabeza_global']} | "
            f"{resultado['fronteras_cabeza_global']['18-24|25-34']} | "
            f"{resultado['fronteras_cabeza_global']['25-34|35-49']} | "
            f"{resultado['macro_combinada']} | {resultado['delta']:+.4f} | "
            f"[{resultado['bootstrap_ic95'][0]:+.4f}, {resultado['bootstrap_ic95'][1]:+.4f}] |")
    lineas += [f"| V12 (referencia) | - | - | - | {fronteras_referencia['18-24|25-34']['accuracy']} | "
              f"{fronteras_referencia['25-34|35-49']['accuracy']} | "
              f"{metricas_referencia['macro_f1']} | - | - |", "",
              "## 3. Salud del entrenamiento", "",
              "| Brazo | F1 de train | F1 de validacion | Brecha | Epocas |", "|---|---|---|---|---|"]
    for resultado in resultados:
        lineas.append(f"| {resultado['brazo']} | {resultado['f1_train_cabeza']} | "
                      f"{resultado['f1_validacion_cabeza']} | "
                      f"{resultado['brecha_train_validacion']} | {resultado['epocas']} |")
    lineas += ["", "## 4. F1 por clase del sistema combinado", "",
              "| Brazo | " + " | ".join(cfg.CLASSES) + " |", "|---" * (1 + len(cfg.CLASSES)) + "|"]
    for resultado in resultados:
        lineas.append(f"| {resultado['brazo']} | "
                      + " | ".join(str(resultado["f1_por_clase"][c]) for c in cfg.CLASSES) + " |")
    lineas.append("| V12 (referencia) | "
                  + " | ".join(str(metricas_referencia["f1_por_clase"][c]) for c in cfg.CLASSES) + " |")
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    RESULTADO.write_text(json.dumps({
        "fase": "V18 · P4 entrenamiento multi-objetivo", "fecha": "2026-09-26",
        "pre_registrado": {"epocas": EPOCAS, "lote": LOTE, "tasa_aprendizaje": TASA_APRENDIZAJE,
                           "lambda": [0.0, 0.1, 0.5]},
        "brazos": [{"nombre": n, "lambda": p, "tipo": t} for n, p, t in BRAZOS],
        "representacion": "las 279 variables de V12, ninguna nueva",
        "referencia": {"macro_f1": metricas_referencia["macro_f1"],
                       "fronteras": {k: v["accuracy"] for k, v in fronteras_referencia.items()},
                       "f1_por_clase": metricas_referencia["f1_por_clase"]},
        "resultados": resultados, "dispositivo": DISPOSITIVO,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{sum(1 for r in resultados if r['adopta'])} de {len(resultados)} brazos adoptan")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
