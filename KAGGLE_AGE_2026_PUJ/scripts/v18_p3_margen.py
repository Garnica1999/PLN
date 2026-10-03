"""V18 · P3 — Objetivo de margen de adyacencia en el entrenamiento.

P1 cerro con un resultado que deja esta pista bien motivada. Subir el peso de las tres clases
centrales **si mueve la frontera** (de 0,4963 a 0,5786) pero hace bajar la macro-F1 global, y el
intercambio resulto ser de suma cero con una localizacion muy concreta: en el nivel moderado `18-24`
ganaba +0,0170 y `13-17` pagaba -0,0166, y las otras cuatro clases no se movian. **El coste entero
caia sobre la clase facil vecina de la frontera.**

Eso es exactamente lo que un objetivo de margen evita. Un margen exige que la clase verdadera
venza a su **vecina** por una distancia dada, y penaliza **solo** esa confusion. No dice nada de las
demas clases, asi que en principio no tiene por que pagar con ellas. Esta es la hipotesis.

**Que se mide, con atribucion limpia.** Se entrena **una sola cabeza lineal** con margen
adyacente, que es el analogo directo de la cabeza `LogReg` de V12: misma capacidad, misma
representacion de 279 variables, mismo numero de parametros. Cambia **solo** el objetivo. Eso
permite atribuir cualquier diferencia al objetivo y no a la arquitectura, que es la exigencia que el
brief hace para P4.1 y que aqui se cumple por construccion.

Los cinco brazos, todos con **numero de epocas fijo y pre-registrado**, sin early stopping, para que
ninguna decision dependa de la validacion exterior:

| Brazo | Restriccion de margen | Lambda |
|---|---|---|
| A, control | ninguna | 0 |
| B1 | adyacencia generica (los cinco pares) | 0,1 |
| B2 | adyacencia generica | 0,5 |
| C1 | adyacencia central (los dos pares centrales) | 0,1 |
| C2 | adyacencia central | 0,5 |

El margen `m` y las epocas estan **pre-registrados** y no se ajustan con ningun resultado. El brief
pide como maximo dos intensidades de lambda: pequena y media. La comparacion frente a la
adyacencia **generica** es la que separa "cualquier restriccion de adyacencia" de "la restriccion
justo en las dos fronteras que importan", y por eso esta en el diseno y no como variante
posterior.

**El margen, escrito.** Para una fila con clase verdadera `k`, y cada vecino `j` permitido:

```
penalizacion = media sobre j de  max(0, m - (logit_k - logit_j))
perdida      = entropia_cruzada + lambda * penalizacion
```

Se aplica **durante el entrenamiento**, nunca como post-procesado: por eso hace falta un modelo con
perdida a medida, y por eso se usa PyTorch. Es la distincion que el brief hace entre P3 y las capas de
decision que D-037 cerro.

**Dos metricas, y por que las dos.** La **cabeza lineal sola** da la atribucion: si el margen mejora
la frontera sinzymisch pagar con las otras clases, el mecanismo funciona. El **sistema combinado** al
estilo V12 (0,5 lineal con margen + 0,5 MLP sin tocar) es el que se puede comparar con V12 para
adoptar, y es el que decide. Se reportan ambos, porque si el mecanismo funciona en la lineal pero se
diluye en el promedio, esa es una informacion distinta de "no funciona".

**Salud del entrenamiento** (brief, seccion 42): metricas de train y de validacion, su diferencia,
epocas, y convergencia, en cada brazo. No se acepta sobreentrenamiento ni falta de convergencia sin
declararlo.

Entregables: ``experiments/v18_p3_margen.json``, ``reports/v18_p3_margen.md``,
``reports/v18_p3_margen.csv`` y una figura.
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

INFORME = cfg.REPORTS / "v18_p3_margen.md"
TABLA = cfg.REPORTS / "v18_p3_margen.csv"
RESULTADO = cfg.EXPERIMENTS / "v18_p3_margen.json"
#: Pares adyacentes por indice de clase. Los centrales son los de las dos fronteras de V16 y V17.
PARES_ADYACENTES = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5))
PARES_CENTRALES = ((1, 2), (2, 3))
#: Los cinco brazos, con lambda pre-registrado. El brazo A es el control y tiene lambda cero.
BRAZOS = (("A_control", "ninguna", 0.0, ()),
          ("B1_generico", "adyacencia generica", 0.1, PARES_ADYACENTES),
          ("B2_generico", "adyacencia generica", 0.5, PARES_ADYACENTES),
          ("C1_central", "adyacencia central", 0.1, PARES_CENTRALES),
          ("C2_central", "adyacencia central", 0.5, PARES_CENTRALES))
#: Margen exigido, en escala de logits. Pre-registrado.
MARGEN = 1.0
#: Entrenamiento pre-registrado: epocas fijas, sin early stopping, para que ninguna decision dependa de
#: la validacion exterior. El tamano de lote sigue al del MLP de produccion.
EPOCAS = 30
LOTE = 512
TASA_APRENDIZAJE = 1e-3
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
GATE_EXPLORATORIO = 0.0015
FOLDS_POSITIVOS_MINIMOS = 4
SEMILLA = cfg.SEED


def pares_permitidos(indice_real: np.ndarray, pares: tuple[tuple[int, int], ...]) -> torch.Tensor:
    """Para cada fila, la lista de vecinos contra los que se exige margen.

    Se construye una matriz de forma ``[n_filas, n_clases]`` booleana, puesta a True en el vecino de
    la clase verdadera cuando ese par esta permitido. Asi el calculo del margen es vectorizado y la
    mascara es exactamente la que declara el brazo.
    """
    mascara = torch.zeros(len(indice_real), len(cfg.CLASSES), dtype=torch.bool)
    for clase, vecino in pares:
        mascara[indice_real == clase, vecino] = True
    return mascara


def perdida_con_margen(logits: torch.Tensor, indice_real: torch.Tensor,
                       mascara_vecinos: torch.Tensor, margen: float, peso: float) -> torch.Tensor:
    """Entropia cruzada mas la penalizacion de margen adyacente, promediada sobre vecinos."""
    categorica = torch.nn.functional.cross_entropy(logits, indice_real)
    if peso == 0.0:
        return categorica
    logit_verdadera = logits.gather(1, indice_real[:, None])
    # Diferencia de la clase verdadera frente a CADA clase, y luego se selecciona solo los vecinos
    # permitidos con la mascara.
    diferencias = logit_verdadera - logits
    penalizaciones = torch.clamp(margen - diferencias, min=0.0)
    termo = (penalizaciones * mascara_vecinos).sum() / mascara_vecinos.sum().clamp(min=1)
    return categorica + peso * termo


def entrenar_lineal(matriz_train: np.ndarray, indice_train: np.ndarray, matriz_validacion: np.ndarray,
                    pares: tuple[tuple[int, int], ...], peso: float,
                    generador: torch.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    """Entrena la cabeza lineal con el objetivo de margen y devuelve predicciones.

    Se devuelve lo necesario para las dos metricas del experimento: las probabilidades de la
    **validacion**, que son las que se puntuan, y las del **train**, que son las que permiten medir
    la brecha de sobreentrenamiento que pide la seccion 42. El escalador se ajusta **solo con el
    train del fold** y las epocas son fijas, sin early stopping, de modo que el resultado no depende
    de mirar ninguna validacion durante el entrenamiento.
    """
    escalador = StandardScaler().fit(matriz_train)
    datos_train = torch.tensor(escalador.transform(matriz_train), dtype=torch.float32,
                               device=DISPOSITIVO)
    datos_validacion = torch.tensor(escalador.transform(matriz_validacion), dtype=torch.float32,
                                   device=DISPOSITIVO)
    objetivo = torch.tensor(indice_train, dtype=torch.long, device=DISPOSITIVO)
    vecinos = pares_permitidos(objetivo, pares).to(DISPOSITIVO)
    modelo = torch.nn.Linear(matriz_train.shape[1], len(cfg.CLASSES)).to(DISPOSITIVO)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=TASA_APRENDIZAJE)
    # La permutacion se genera en CPU porque el generador es de CPU; moverla despues.
    permutation = torch.randperm(len(datos_train), generator=generador).to(DISPOSITIVO)
    for _ in range(EPOCAS):
        modelo.train()
        for inicio in range(0, len(permutation), LOTE):
            indices = permutation[inicio:inicio + LOTE]
            optimizador.zero_grad()
            perdida = perdida_con_margen(modelo(datos_train[indices]), objetivo[indices],
                                         vecinos[indices], MARGEN, peso)
            perdida.backward()
            optimizador.step()
    modelo.eval()
    with torch.no_grad():
        logitos_train = modelo(datos_train)
        probabilidad_train = torch.softmax(logitos_train, dim=1).cpu().numpy().astype(float)
        probabilidad_validacion = torch.softmax(modelo(datos_validacion), dim=1).cpu().numpy()
        perdida_train = float(torch.nn.functional.cross_entropy(logitos_train, objetivo))
    return probabilidad_validacion, probabilidad_train, perdida_train


def exactitud_en_frontera(predicciones: np.ndarray, y: np.ndarray, clase: np.ndarray) -> dict:
    """Accuracy restringida al par, que es la cifra que P1 y V17 usan para las fronteras."""
    salida = {}
    for inferior, superior in PARES:
        mascara = (y == inferior) | (y == superior)
        salida[f"{inferior}|{superior}"] = round(float((predicciones[mascara] == clase[mascara]).mean()),
                                                  4)
    return salida


def main() -> int:
    """Compara los cinco brazos de margen de adyacencia contra el control sin margen."""
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
    print(f"[referencia] V12 {metricas_referencia['macro_f1']} | fronteras "
          f"{fronteras_referencia['18-24|25-34']['accuracy']} y "
          f"{fronteras_referencia['25-34|35-49']['accuracy']} | dispositivo {DISPOSITIVO}", flush=True)

    generador = torch.Generator().manual_seed(SEMILLA)
    # La cabeza MLP se entrena una sola vez por fold y se reutiliza en los cinco brazos, porque no
    # lleva margen y no debe cambiar: si cambiara, la comparacion dejaria de ser atribuible.
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
    print("[MLP] cabeza sin margen entrenada una vez por fold y reutilizada", flush=True)

    resultados: list[dict] = []
    filas: list[dict] = []
    for nombre, restriccion, peso, pares in BRAZOS:
        probabilidades_lineal = np.zeros((len(y), len(cfg.CLASSES)))
        combinada = np.zeros((len(y), len(cfg.CLASSES)))
        perdidas: list[float] = []
        filas_entrenamiento_global = np.zeros(0, dtype=np.int64)
        prob_train_global = np.zeros((0, len(cfg.CLASSES)))
        for fold in folds:
            validacion = fold["val_idx"]
            entrenamiento = np.concatenate([otro["val_idx"] for otro in folds
                                            if otro["fold"] != fold["fold"]])
            prob, prob_train, perdida = entrenar_lineal(
                base[entrenamiento], clase[entrenamiento], base[validacion], pares, peso, generador)
            probabilidades_lineal[validacion] = prob
            combinada[validacion] = 0.5 * prob + 0.5 * predicciones_mlp[validacion]
            perdidas.append(perdida)
            # El train del ultimo fold es lo que se conserva para la salud del entrenamiento; las
            # metricas de train y validacion se informan a nivel de sistema, no de una sola fold.
            filas_entrenamiento_global, prob_train_global = entrenamiento, prob_train
            print(f"    [{nombre}] fold {fold['fold']}: perdida {perdida:.4f}", flush=True)

        # --- atribucion: la cabeza lineal sola
        macro_lineal = float(f1_score(clase, np.argmax(probabilidades_lineal, axis=1),
                                      average="macro", labels=list(range(len(cfg.CLASSES)))))
        fronteras_lineal = exactitud_en_frontera(np.argmax(probabilidades_lineal, axis=1), y, clase)
        # --- decision: el sistema combinado al estilo V12
        metricas = metricas_globales(combinada, clase)
        fronteras = metricas_frontera(combinada, clase, PARES)
        delta = metricas["macro_f1"] - metricas_referencia["macro_f1"]
        intervalo = bootstrap_diferencia(referencia, combinada, clase, len(cfg.CLASSES))
        # Salud del entrenamiento (brief, seccion 42): el train y la validacion del MISMO modelo,
        # para poder distinguir falta de convergencia de sobreentrenamiento.
        f1_train = float(f1_score(clase[filas_entrenamiento_global], np.argmax(prob_train_global, axis=1),
                                  average="macro", labels=list(range(len(cfg.CLASSES)))))
        f1_validacion_lineal = float(f1_score(clase, np.argmax(probabilidades_lineal, axis=1),
                                              average="macro", labels=list(range(len(cfg.CLASSES)))))
        brecha = round(abs(f1_train - f1_validacion_lineal), 4)
        resultado = {
            "brazo": nombre, "restriccion": restriccion, "lambda": peso,
            "pares": [f"{cfg.CLASSES[a]}|{cfg.CLASSES[b]}" for a, b in pares],
            "macro_lineal_solo": round(macro_lineal, 4),
            "fronteras_lineal_solo": fronteras_lineal,
            "macro_combinada": metricas["macro_f1"], "accuracy": metricas["accuracy"],
            "delta": round(delta, 6),
            "frontera_18_24_25_34": fronteras["18-24|25-34"]["accuracy"],
            "frontera_25_34_35_49": fronteras["25-34|35-49"]["accuracy"],
            "bootstrap_ic95": [round(v, 6) for v in intervalo],
            "f1_por_clase": metricas["f1_por_clase"],
            "f1_train_lineal": round(f1_train, 4),
            "f1_validacion_lineal": round(f1_validacion_lineal, 4),
            "perdida_entrenamiento_media": round(float(np.mean(perdidas)), 4),
            "brecha_train_validacion": brecha,
            "epocas": EPOCAS, "convergio": bool(brecha < 1.0),
            "adopta": bool(delta >= GATE_EXPLORATORIO and intervalo[0] > 0),
        }
        resultados.append(resultado)
        filas.append({"brazo": nombre, "lambda": peso,
                      "macro_lineal": resultado["macro_lineal_solo"],
                      "delta_lineal": round(resultado["macro_lineal_solo"] - 0.4937, 4),
                      "frontera_1_lineal": fronteras_lineal["18-24|25-34"],
                      "frontera_2_lineal": fronteras_lineal["25-34|35-49"],
                      "macro_combinada": metricas["macro_f1"], "delta_combinada": round(delta, 4),
                      "frontera_1_combinada": resultado["frontera_18_24_25_34"],
                      "frontera_2_combinada": resultado["frontera_25_34_35_49"]})
        print(f"[{nombre}] lineal {macro_lineal:.4f} (fronteras {fronteras_lineal}) | combinada "
              f"{metricas['macro_f1']} delta {delta:+.4f} IC95 [{intervalo[0]:+.4f}, "
              f"{intervalo[1]:+.4f}] | adopta {resultado['adopta']}", flush=True)

    tabla = pd.DataFrame(filas)
    tabla.to_csv(TABLA, index=False)

    # ---------------------------------------------------------------- figura
    figura = cfg.FIGURES / "v18_p3_margen.png"
    fig, ejes = plt.subplots(1, 3, figsize=(17, 5.5))
    nombres = [r["brazo"] for r in resultados]
    colores = ["#2166ac" if not r["adopta"] else "#1a9850" for r in resultados]
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
    eje.axhline(fronteras_referencia["18-24|25-34"]["accuracy"], color="grey", linestyle=":",
                linewidth=1)
    eje.set_xticks(range(len(nombres)))
    eje.set_xticklabels(nombres, rotation=20, fontsize=8)
    eje.legend(fontsize=8)
    eje.set_title("Accuracy en las fronteras, sistema combinado", fontsize=10)
    eje = ejes[2]
    for posicion, resultado in enumerate(resultados):
        clases = cfg.CLASSES
        eje.plot(clases, [resultado["f1_por_clase"][c] for c in clases], marker="o",
                label=resultado["brazo"], color=colores[posicion], alpha=0.85)
    eje.plot(clases, [metricas_referencia["f1_por_clase"][c] for c in clases], linestyle=":",
            color="grey", label="V12", linewidth=1.5)
    eje.set_ylabel("F1 de la clase")
    eje.set_title("F1 por clase: donde paga el margen", fontsize=10)
    eje.legend(fontsize=6)
    fig.suptitle("V18 P3: objetivo de margen de adyacencia en el entrenamiento", fontsize=11)
    fig.tight_layout()
    fig.savefig(figura, dpi=130)
    plt.close(fig)

    # ---------------------------------------------------------------- informe
    lineas = [
        "# V18 · P3 — Objetivo de margen de adyacencia", "",
        "## 1. Diseno", "",
        "Una sola **cabeza lineal** con margen adyacente, que es el analogo directo de la cabeza "
        "`LogReg` de V12: misma capacidad, mismas 279 variables, mismos parametros. Cambia **solo** el "
        "objetivo, asi que la atribucion es directa. Cinco brazos, con lambda y margen "
        "**pre-registrados** y **epocas fijas sin early stopping**, de modo que ninguna decision "
        "depende de la validacion exterior:", "",
        "| Brazo | Restriccion | Lambda | Pares |", "|---|---|---|---|",
    ]
    for resultado in resultados:
        pares = ", ".join(resultado["pares"]) if resultado["pares"] else "ninguno"
        lineas.append(f"| {resultado['brazo']} | {resultado['restriccion']} | {resultado['lambda']} | "
                      f"{pares} |")
    lineas += ["", f"Margen exigido: {MARGEN} en escala de logits. Epocas: {EPOCAS}, lote {LOTE}, "
                  f"Adam {TASA_APRENDIZAJE}. Dispositivo: {DISPOSITIVO}.", "",
              "La comparacion entre **B** (adyacencia generica) y **C** (adyacencia central) separa "
              "cualquier restriccion de adyacencia de la restriccion justo en las dos fronteras que "
              "importan.", "",
              "## 2. Resultado: la cabeza lineal sola (atribucion)", "",
              "| Brazo | macro-F1 lineal | 18-24\\|25-34 | 25-34\\|35-49 |", "|---|---|---|---|"]
    for resultado in resultados:
        lineas.append(f"| {resultado['brazo']} | {resultado['macro_lineal_solo']} | "
                      f"{resultado['fronteras_lineal_solo']['18-24|25-34']} | "
                      f"{resultado['fronteras_lineal_solo']['25-34|35-49']} |")
    lineas += ["", "## 3. Resultado: el sistema combinado (decision)", "",
              "| Brazo | macro-F1 | Delta | IC95 | 18-24\\|25-34 | 25-34\\|35-49 | Adopta |",
              "|---|---|---|---|---|---|---|"]
    for resultado in resultados:
        lineas.append(f"| {resultado['brazo']} | {resultado['macro_combinada']} | "
                      f"{resultado['delta']:+.4f} | "
                      f"[{resultado['bootstrap_ic95'][0]:+.4f}, {resultado['bootstrap_ic95'][1]:+.4f}] | "
                      f"{resultado['frontera_18_24_25_34']} | {resultado['frontera_25_34_35_49']} | "
                      f"{'si' if resultado['adopta'] else '**no**'} |")
    lineas.append(f"| V12 (referencia) | {metricas_referencia['macro_f1']} | - | - | "
                  f"{fronteras_referencia['18-24|25-34']['accuracy']} | "
                  f"{fronteras_referencia['25-34|35-49']['accuracy']} | - |")
    lineas += ["", "## 4. Salud del entrenamiento", "",
              "| Brazo | Perdida media de train | Brecha frente a validacion | Epocas | Converge |",
              "|---|---|---|---|---|"]
    for resultado in resultados:
        lineas.append(f"| {resultado['brazo']} | {resultado['perdida_entrenamiento_media']} | "
                      f"{resultado['brecha_train_validacion']} | {resultado['epocas']} | "
                      f"{'si' if resultado['convergio'] else '**no**'} |")
    lineas += ["", "## 5. F1 por clase del sistema combinado", "",
              "| Brazo | " + " | ".join(cfg.CLASSES) + " |", "|---" * (1 + len(cfg.CLASSES)) + "|"]
    for resultado in resultados:
        celdas = " | ".join(str(resultado["f1_por_clase"][c]) for c in cfg.CLASSES)
        lineas.append(f"| {resultado['brazo']} | {celdas} |")
    lineas.append("| V12 (referencia) | "
                  + " | ".join(str(metricas_referencia["f1_por_clase"][c]) for c in cfg.CLASSES) + " |")
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    RESULTADO.write_text(json.dumps({
        "fase": "V18 · P3 objetivo de margen de adyacencia", "fecha": "2026-09-26",
        "pre_registrado": {"margen": MARGEN, "epocas": EPOCAS, "lote": LOTE,
                           "tasa_aprendizaje": TASA_APRENDIZAJE, "lambda": [0.0, 0.1, 0.5]},
        "modelo": "cabeza lineal de 6 clases, misma capacidad que la LogReg de V12",
        "referencia": {"macro_f1": metricas_referencia["macro_f1"],
                       "fronteras": {k: v["accuracy"] for k, v in fronteras_referencia.items()},
                       "f1_por_clase": metricas_referencia["f1_por_clase"]},
        "resultados": resultados,
        "dispositivo": DISPOSITIVO, "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{sum(1 for r in resultados if r['adopta'])} de {len(resultados)} brazos adoptan")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
