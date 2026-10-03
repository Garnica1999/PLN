"""V13 · P3a — Especialistas binarios de frontera, construidos sobre las variables de V12.

Por que esta pista y no otra. V13 ha cerrado cuatro pistas de variables nuevas (P1 evidencia por
token, P17 pronombres, P20 modalidad y los lexicones de la fase anterior) y P16/P16c han medido por
que: la redundancia lineal no explica el fracaso, pero **anadir variables al meta-modelo no mueve la
macro-F1** en ningun caso, ni con controles de permutado que dicen que la mejora es ruido. Todo eso
es un cambio de **representacion**. La pista P3 es el unico cambio de **estructura de decision** que
queda sin probar, y es ademas donde el problema es mayor: segun phase13 y P16b, la exactitud de V12 en
las dos fronteras centrales es 0,4963 y 0,4974 frente a 0,6004 global, y solo en ahi se concentra el
error.

Ojo con lo que D-037 ya cerro: se cerraron 33 capas de decision **a posteriori** sobre las
probabilidades de V12 (reglas de desempate, umbrales por clase, correccion de probabilidades). Esto
no es eso. Aqui se entrena un modelo **nuevo** sobre las filas del par, con las mismas 279 variables de
V12 y su propio fold, y solo se usa para **redecidir entre las dos clases del par** cuando V12 ya ha
dicho que la fila es de una de ellas. Esa es la forma ``gate(especialista, V12)`` que §37 exige: el
especialista nunca reemplaza a V12 globalmente, y por construccion no puede tocar las filas en las que
V12 no está en el par.

**Dos etapas con puerta**, igual que P17b y por el mismo motivo (P15 ayudo a convertir esa leccion en
protocolo). La etapa 1 es barata: cinco folds por par, dos modelos, y mide cuanto recupera el
especialista sobre el error de V12 *dentro* del par. Si el neto no alcanza el umbral, la pista se
cierra sin pagar la evaluacion completa con control de permutado y gate.

Criterios de la puerta (justificados en el propio tamaño del problema):

* El especialista tiene que **superar la exactitud de V12 dentro del par en al menos +0,010**. Por
  debajo, el cambio no compensa ni el riesgo.
* El **neto de rescate** dentro del par tiene que ser de al menos **+1.000 filas** en total entre las
  dos fronteras (el 1,7 % de los 57.448 errores que V12 comete dentro de los pares).
* El specialists tiene que mejorar **las dos** fronteras, no una a costa de la otra.

El test no se toca. Ninguna submission se genera aqui.

Entregables: ``reports/v3_especialistas.md``, ``reports/v3_especialistas_franjas.csv``,
``experiments/v3_p3a_especialistas.json``.
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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from phase2_ponderacion import build_features  # noqa: E402

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v3_especialistas.md"
FRANJAS = cfg.REPORTS / "v3_especialistas_franjas.csv"
RESULTADO = cfg.EXPERIMENTS / "v3_p3a_especialistas.json"

PARES = {("18-24", "25-34"): "18-24|25-34", ("25-34", "35-49"): "25-34|35-49"}
#: Umbrales de la puerta, justificados en el docstring.
MEJORA_MINIMA = 0.010
RESCATE_MINIMO = 1000


def especialista_logreg() -> Pipeline:
    """Especialista binario lineal, con el mismo tipificador que la cabeza de V12."""
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, max_iter=3000, random_state=cfg.SEED))])


def especialista_mlp() -> Pipeline:
    """Especialista binario no lineal, con la misma forma que la otra cabeza de V12."""
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, batch_size=512,
                                           learning_rate_init=1e-3, max_iter=40,
                                           early_stopping=True, n_iter_no_change=5,
                                           random_state=cfg.SEED))])


def main() -> int:
    """Entrena los especialistas de frontera y decide con la puerta de rescate."""
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
    pred_v12 = np.argmax(referencia, axis=1)
    confianza_v12 = referencia.max(axis=1)
    print(f"[V12] {base.shape[1]} variables | exactitud global "
          f"{float((pred_v12 == clase).mean()):.4f}", flush=True)

    filas: list[dict] = []
    specialists = {etiqueta: {nombre: np.zeros(len(y), dtype=np.int8) for nombre in ("logreg", "mlp")}
                   for etiqueta in PARES.values()}
    for (clase_a, clase_b), etiqueta in PARES.items():
        indice_a, indice_b = cfg.CLASSES.index(clase_a), cfg.CLASSES.index(clase_b)
        en_par = np.isin(clase, (indice_a, indice_b))
        # Filas donde la puerta puede actuar: V12 ya dice que la fila es de una de las dos clases.
        actuable = np.isin(pred_v12, (indice_a, indice_b))
        for nombre, constructor in (("logreg", especialista_logreg),
                                    ("mlp", especialista_mlp)):
            for fold in folds:
                validacion = fold["val_idx"]
                entrenamiento = np.concatenate([otro["val_idx"] for otro in folds
                                                if otro["fold"] != fold["fold"]])
                filas_entrenamiento = entrenamiento[en_par[entrenamiento]]
                if len(np.unique(clase[filas_entrenamiento])) < 2:
                    continue
                modelo = constructor().fit(base[filas_entrenamiento], clase[filas_entrenamiento])
                objetivo = validacion[actuable[validacion]]
                specialists[etiqueta][nombre][objetivo] = modelo.predict(base[objetivo])
        for nombre in ("logreg", "mlp"):
            pred = specialists[etiqueta][nombre]
            # Fila no actuable: se conserva la prediccion de V12.
            completa = np.where(actuable & (pred > 0), pred, pred_v12)
            acierto_v12_par = pred_v12[en_par] == clase[en_par]
            acierto_esp_par = completa[en_par] == clase[en_par]
            acierto_v12_act = pred_v12[actuable] == clase[actuable]
            acierto_esp_act = completa[actuable] == clase[actuable]
            rescate = int((~acierto_v12_act & acierto_esp_act).sum())
            dano = int((acierto_v12_act & ~acierto_esp_act).sum())
            filas.append({
                "frontera": etiqueta, "modelo": nombre,
                "filas_par": int(en_par.sum()),
                "accuracy_v12_par": round(float(acierto_v12_par.mean()), 4),
                "accuracy_especialista_par": round(float(acierto_esp_par.mean()), 4),
                "mejora_par": round(float(acierto_esp_par.mean() - acierto_v12_par.mean()), 4),
                "filas_actuables": int(actuable.sum()),
                "accuracy_v12_actuables": round(float(acierto_v12_act.mean()), 4),
                "accuracy_especialista_actuables": round(float(acierto_esp_act.mean()), 4),
                "rescate": rescate, "dano": dano, "neto": rescate - dano,
                "macro_f1_global": round(float(f1_score(clase, completa, average="macro",
                                                          labels=list(range(len(cfg.CLASSES))))), 4),
                "filas_modificadas": int((completa != pred_v12).sum()),
            })
            print(f"[{etiqueta} · {nombre}] par: {acierto_v12_par.mean():.4f} -> "
                  f"{acierto_esp_par.mean():.4f} ({acierto_esp_par.mean() - acierto_v12_par.mean():+.4f})"
                  f" | actuables: {acierto_v12_act.mean():.4f} -> {acierto_esp_act.mean():.4f}"
                  f" | rescate {rescate:,} dano {dano:,} neto {rescate - dano:+,}", flush=True)

    tabla = pd.DataFrame(filas)
    tabla.to_csv(FRANJAS, index=False)

    # ---------------------------------------------------------------- puerta
    mejor_por_frontera = {etiqueta: tabla[tabla["frontera"] == etiqueta].sort_values(
        "mejora_par", ascending=False).iloc[0] for etiqueta in PARES.values()}
    netos = {etiqueta: int(fila["neto"]) for etiqueta, fila in mejor_por_frontera.items()}
    total = sum(netos.values())
    mejoras = {etiqueta: float(fila["mejora_par"]) for etiqueta, fila in mejor_por_frontera.items()}
    pasa = (all(mejora >= MEJORA_MINIMA for mejora in mejoras.values())
            and total >= RESCATE_MINIMO
            and all(neto > 0 for neto in netos.values()))
    print(f"[puerta] mejoras {mejoras} | netos {netos} | total {total} | "
          f"umbral mejora {MEJORA_MINIMA} y rescate {RESCATE_MINIMO} | pasa {pasa}", flush=True)

    # Macro-F1 global de la combinacion gate(especialista, V12) con el mejor modelo por frontera.
    combinada = pred_v12.copy()
    for etiqueta, fila in mejor_por_frontera.items():
        pred = specialists[etiqueta][fila["modelo"]]
        actuable = pred > 0
        combinada = np.where(actuable, pred, combinada)
    macro_global = float(f1_score(clase, combinada, average="macro",
                                  labels=list(range(len(cfg.CLASSES)))))
    macro_referencia = float(f1_score(clase, pred_v12, average="macro",
                                      labels=list(range(len(cfg.CLASSES)))))
    print(f"[global] macro-F1 de V12 {macro_referencia:.4f} | con la puerta "
          f"{macro_global:.4f} ({macro_global - macro_referencia:+.4f}) | filas modificadas "
          f"{int((combinada != pred_v12).sum()):,}", flush=True)

    lineas = [
        "# V13 · P3a — Especialistas binarios de frontera sobre las variables de V12", "",
        "## 1. Qué se entrena", "",
        "Para cada par de frontera, un modelo **nuevo** entrenado sobre las filas del train cuyo "
        "valor verdadero es una de las dos clases, con las mismas 279 variables de V12 y su propio "
        "fold. En inferencia solo redecide **entre las dos clases del par**, y solo cuando V12 ya ha "
        "dicho que la fila es de una de ellas: es la forma `gate(especialista, V12)` de §37, y por "
        "construcción no puede tocar las filas en las que V12 no está en el par.", "",
        "Esto **no** es la línea que D-037 cerró: allí se aplicaron 33 reglas a posteriori sobre las "
        "probabilidades de V12; aquí se entrena un modelo con sus propios datos de entrenamiento.", "",
        "## 2. Resultado por frontera y por modelo", "",
        "| Frontera | Modelo | Accuracy V12 en el par | Accuracy del especialista | Mejora | "
        "Rescate | Daño | Neto |", "|---|---|---|---|---|---|---|---|",
    ]
    for _, fila in tabla.iterrows():
        lineas.append(f"| {fila['frontera']} | {fila['modelo']} | {fila['accuracy_v12_par']} | "
                      f"**{fila['accuracy_especialista_par']}** | {fila['mejora_par']:+.4f} | "
                      f"{fila['rescate']:,} | {fila['dano']:,} | {fila['neto']:+,} |")
    lineas += ["", "## 3. Efecto global de la puerta", "",
               f"- macro-F1 de V12: **{macro_referencia:.4f}**",
               f"- macro-F1 con la puerta (mejor modelo por frontera): **{macro_global:.4f}** "
               f"({macro_global - macro_referencia:+.4f})",
               f"- Filas cuya prediccion cambia: {int((combinada != pred_v12).sum()):,}", "",
               "## 4. Puerta", "",
               "| Criterio | Valor | Cumple |", "|---|---|---|"]
    for etiqueta in PARES.values():
        lineas.append(f"| mejora >= {MEJORA_MINIMA} en {etiqueta} | {mejoras[etiqueta]:+.4f} | "
                      f"{'si' if mejoras[etiqueta] >= MEJORA_MINIMA else '**no**'} |")
    lineas += [f"| rescate neto total >= {RESCATE_MINIMO} | {total:+,} | "
               f"{'si' if total >= RESCATE_MINIMO else '**no**'} |",
               f"| las dos fronteras con neto positivo | {netos} | "
               f"{'si' if all(n > 0 for n in netos.values()) else '**no**'} |", ""]
    lineas += ["## 5. Decisión", ""]
    if pasa:
        lineas += [
            "**La puerta se supera**: procede la evaluación completa (P3b) con control de permutado, "
            "gate §24 completo, doble métrica §36 y la predicción falsable por cabeza (DD-8).",
            "Antes de eso conviene una comprobación barata y valiosa: si el especialista gana por "
            "separar mejor las dos clases o simplemente por tener menos clases que decidir, porque "
            "eso cambia lo que se puede esperar de un modelo de frontera en general.", ""]
    else:
        lineas += [
            "**La puerta no se supera**: la pista P3 se cierra sin pagar la evaluación completa. Un "
            "especialista que solo repita lo que ya hace V12 no aporta, y un especialista que "
            "redecida peor destruye filas que V12 tenía bien.",
            "Con esto queda agotado también el segundo tipo de cambio posible. P15 había dicho que "
            "los miembros no tienen señal de confianza y P16d que el bloque de P1 tampoco; ahora se "
            "añade que **ni siquiera un modelo entrenado específicamente para la frontera mejora la "
            "decisión dentro de ella**. Las tres vías que quedaban (más variables, más modelos, más "
            "estructura de decisión) fallan con evidencia medida, no con suposiciones.", ""]
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    RESULTADO.write_text(json.dumps({
        "fase": "V13 · P3a especialistas binarios de frontera", "fecha": "2026-09-26",
        "n_variables": int(base.shape[1]),
        "referencia_macro_f1": round(macro_referencia, 4),
        "macro_f1_con_puerta": round(macro_global, 4),
        "delta_global": round(macro_global - macro_referencia, 4),
        "filas_modificadas": int((combinada != pred_v12).sum()),
        "por_frontera": tabla.to_dict(orient="records"),
        "criterios": {"mejora_minima": MEJORA_MINIMA, "rescate_minimo": RESCATE_MINIMO,
                      "mejoras": mejoras, "netos": netos, "rescate_total": total},
        "pasa": pasa, "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nartefactos: {INFORME.name}, {FRANJAS.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
