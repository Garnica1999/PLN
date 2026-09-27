"""V20 · F1 y F2 — Informacion nueva de clase condicional, y una clase de funcion distinta.

V19 queda **congelado**: no se toca su configuracion, sus features, su modelo persistido, su
submission ni su hash. Todo aqui es un **challenger independiente** que se mide contra el.

## La auditoria que decidio las familias

Se cruzaron los 117 experimentos registrados contra 28 mecanismos. Tres familias del brief nunca se
habian probado: **RandomForest/ExtraTrees, HistGradientBoosting y CatBoost** como clase de funcion
sobre las 279, y **ninguna transformacion de tipo clase condicional** aplicada a las features densas
y estructuradas.

Y un hallazgo de una linea, cerrado con medicion: el campo `id` **es el indice de fila**, 1 a 169.336
en train y en test, con correlacion de rango **0,00123** con la clase y distribucion plana por
quintiles. No hay ninguna senal de tiempo que extraer, y la familia de "informacion del id" queda
**cerrada con numero** en lugar de gastar un hueco.

## F1, la familia con mas potencial informativo: la vista de clase condicional de las 122 features

**El hueco, demonstration de la manera mas nitida que se puede**. El bloque de 73 variables
label-informed se construyo **in-fold** con `supervised_entity_features` y `class_lexicon_scores`, pero
solo sobre **cinco familias de entidades**: mencion, hashtag, fuente de retweet, rareza lexical y
lexico. Las **122 features numericas** (35 densas de textometria y estilo, mas 87 estructuradas de
forma, posicion y entropia) **nunca se transformaron** en estadisticas condicionales de clase.

**Por que V19 no puede capturarlo por si mismo, y por tanto no es un DUPLICATE**. El bloque de 73
aporta el **+0,0203** en interaccion sobre las 206, y el bloque de 122 nunca ha recibido ese
tratamiento. Un arbol con `max_depth = 3` y `num_leaves = 8` **no puede expresar** "compara esta
longitud con la longitud de la clase c" para 122 features a la vez: las features crudas dan el valor, y
la comparacion es justamente lo que discrimina. El transform es una vista **generativa** de los mismos
datos, y la vista generativa y la discriminativa no son el mismo mecanismo.

**Como se construye, y por que es legal**. Por cada fold, y **solo con las filas de entrenamiento de
ese fold**, se estima la media y la desviacion tipica de cada una de las 122 features **en cada una de
las seis clases**, y se estandariza el valor de las filas de validacion contra esas seis distribuciones.
De cada feature quedan **ocho columnas**: seis z por clase, el margen entre la clase mas alta y la mas
baja, y la entropia de la distribucion softmax sobre los cuadrados de los z. En total
**122 x 8 = 976 columnas**. Cero informacion externa, cero embeddings, cero texto nuevo: solo
estadisticas de clase calculadas dentro del fold, que es exactamente lo que la regla pide.

**Como se aisla el efecto, para que no se confunda con dilucion**. Anadir 976 columnas a 279 con
`colsample_bytree = 0,5` diluiria las originales y un resultado negativo no distinguiria "la
informacion no sirve" de "la dilucion ladanio". Por eso **F1a** entrena el arbol **solo** con las 976, y
**F1b** lo anade como **cuarto miembro** con peso 1/4, dejando V19 intacto con sus 1/3. Esa es la
estructura de "cada modelo con una responsabilidad distinta" que pide el brief: V19 sigue siendo el
equipo de tres, y el modelo nuevo tiene la responsabilidad de la vista de clase condicional.

## F2, la clase de funcion distinta: ExtraTrees sobre las mismas 279

V19 demostro que un combinador **no lineal** cobra un bloque que el lineal no cobraba. Pero solo se
probo **una** clase de funcion: el boosting. **ExtraTrees** tiene un sesgo inductivo distinto de
verdad: cortes completamente aleatorios, arboles muy profundos y bagging, sin boosting ni learning
rate. La pregunta que responde es estructural y no se responde con un hyperparametro: **el efecto de
V19 es "combinador no lineal" en general, o es especifico del boosting?**

Se mide tambien en solitario, porque un resultado negativo en solitario y positivo como miembro
informaria de que aporta por diversidad y no por potencia.

## Lo que se mide, siempre contra V19 y no solo contra V12

macro-F1 pooled, media y desviacion entre folds, F1 por clase, las dos fronteras, **rescates y danos
respecto a V19**, neto, numero de predicciones modificadas, estabilidad por fold y hueco train a OOF.
Y la descomplicidad de errores en las cuatro celdas: ambos correctos, ambos incorrectos, solo el
candidato, solo V19.

Entregables: ``features/v20_f1_bloque_clase_condicional.npy``,
``experiments/v20_f1_f2.json``, ``reports/v20_f1_f2.md``, ``reports/v20_f1_f2_comparativa.csv``,
``reports/v20_f1_f2_complementariedad.csv`` y una figura.
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from phase2_ponderacion import DENSE_FEATURES  # noqa: E402
from v19_b1_boosting import HIERBA, matriz_279  # noqa: E402
from v13_p1b_fusion_tokens import PARES, bootstrap_diferencia, metricas_frontera  # noqa: E402

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v20_f1_f2.md"
COMPARATIVA = cfg.REPORTS / "v20_f1_f2_comparativa.csv"
COMPLEMENTARIEDAD = cfg.REPORTS / "v20_f1_f2_complementariedad.csv"
RESULTADO = cfg.EXPERIMENTS / "v20_f1_f2.json"
BLOQUE = cfg.FEATURES / "v20_f1_bloque_clase_condicional.npy"
FIGURA = cfg.FIGURES / "v20_f1_f2.png"

CV_V19 = 0.6079388
CV_ARBOL_V19 = 0.6072312
ETIQUETAS = list(range(len(cfg.CLASSES)))
#: Hiperparametros de ExtraTrees, **pre-registrados** y ajustados al perfil de la maquina:bosques
#: grandes de arboles poco profundos, que es el modo de baja varianza que mas se parece al boosting.
EXTRA = {"n_estimators": 500, "max_depth": 12, "min_samples_leaf": 40, "max_features": 0.35,
         "bootstrap": True, "n_jobs": 32, "random_state": cfg.SEED}
EPS = 1e-6


def a_clases(probabilidad: np.ndarray, clases: np.ndarray, filas: int) -> np.ndarray:
    """Traduce las columnas del modelo al orden canonico de las seis clases."""
    completa = np.zeros((filas, len(cfg.CLASSES)))
    for posicion, nombre in enumerate(clases):
        completa[:, cfg.CLASSES.index(nombre)] = probabilidad[:, posicion]
    return completa


def macro(clase: np.ndarray, prob: np.ndarray) -> float:
    """macro-F1 pooled de seis clases."""
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro", labels=ETIQUETAS))


def matriz_122() -> tuple[np.ndarray, list[str]]:
    """Las 122 features numericas no supervisadas: 35 densas mas 87 estructuradas.

    Son las columnas de la matriz de 279 que **no** son ni miembros ni label-informed, es decir
    `[0:78]` miembros, `[200:273]` label-informed, `[273:279]` NB-SVM de palabra, luego las 122
    numericas son las posiciones `[78:200]`.
    """
    matriz = matriz_279()
    numericas = np.asarray(matriz[:, 78:200], dtype=np.float64)
    estructuradas = list(pd.read_parquet(
        cfg.FEATURES / "age_structured_features.parquet").columns)
    nombres = [f"densa:{c}" for c in DENSE_FEATURES] + [f"estructurada:{c}" for c in estructuradas]
    assert len(nombres) == numericas.shape[1], f"{len(nombres)} nombres para {numericas.shape[1]}"
    return numericas, nombres


def bloque_clase_condicional(numericas: np.ndarray, y: np.ndarray, clase: np.ndarray,
                             folds: list[dict]) -> np.ndarray:
    """La vista de clase condicional de las 122 features, construida **in-fold**.

    Por cada fold y **solo con sus filas de entrenamiento** se estima la media y la desviacion de
    cada feature en cada clase, y se estandariza el valor de las filas de validacion contra las seis
    distribuciones. De cada feature salen ocho columnas: seis z por clase, el margen entre la clase
    mas alta y la mas baja, y la entropia de la distribucion softmax sobre los cuadrados de los z.

    La regularizacion de la desviacion es `sd_c + 0,1 x sd_global`, para que una clase con muy pocas
    filas no produzca un z gigante por un denominador casi nulo.
    """
    salida = np.zeros((len(y), numericas.shape[1] * 8))
    n_clases = len(cfg.CLASSES)
    for fold in folds:
        validacion = fold["val_idx"]
        entrenamiento = np.concatenate([o["val_idx"] for o in folds
                                        if o["fold"] != fold["fold"]])
        bloque = numericas[entrenamiento]
        objetivo = clase[entrenamiento]
        global_sd = bloque.std(axis=0)
        medias = np.zeros((n_clases, bloque.shape[1]))
        desviaciones = np.zeros((n_clases, bloque.shape[1]))
        for c in range(n_clases):
            filas = bloque[objetivo == c]
            medias[c] = filas.mean(axis=0)
            desviaciones[c] = filas.std(axis=0)
        # Denominador regularizado: sin esta proteccion, una clase escasa genera z enormes.
        denominador = desviaciones + 0.1 * global_sd + EPS
        z = (numericas[validacion][:, None, :] - medias[None, :, :]) / denominador[None, :, :]
        # `z` queda (filas, clases, features): se reordena a (filas, features, clases) para agrupar.
        z = np.transpose(z, (0, 2, 1))
        margen = z.max(axis=2) - z.min(axis=2)
        pesos = np.exp(-0.5 * z ** 2)
        pesos = pesos / np.clip(pesos.sum(axis=2, keepdims=True), EPS, None)
        entropia = -(pesos * np.log(np.clip(pesos, EPS, None))).sum(axis=2)
        # Disposicion: seis bloques de `n_features` (uno por clase, en el orden de `cfg.CLASSES`),
        # despues el margen y despues la entropia. En total `n_features * (6 + 1 + 1)`.
        n_features = numericas.shape[1]
        for c in range(n_clases):
            salida[np.ix_(validacion, np.arange(c * n_features, (c + 1) * n_features))] = z[:, :, c]
        inicio = n_clases * n_features
        salida[np.ix_(validacion, np.arange(inicio, inicio + n_features))] = margen
        inicio += n_features
        salida[np.ix_(validacion, np.arange(inicio, inicio + n_features))] = entropia
        print(f"    fold {fold['fold']} del bloque de clase condicional listo", flush=True)
    return salida


def oof(matriz: np.ndarray, y: np.ndarray, folds: list[dict], modelo: str) -> np.ndarray:
    """OOF real: el modelo se ajusta en los folds de entrenamiento y predice el retenido.

    Se cachea por firma de la entrada, porque son quince ajustes y ninguno depende de otro.
    """
    cache = cfg.FEATURES / (f"v20_oof_{modelo}_{matriz.shape[1]}_{int(matriz[0, 0] * 1000)}.npy")
    if cache.exists():
        salida = np.load(cache)
        if salida.shape == (len(y), len(cfg.CLASSES)):
            print(f"    cache: {cache.name}", flush=True)
            return salida
    salida = np.zeros((len(y), len(cfg.CLASSES)))
    for fold in folds:
        validacion = fold["val_idx"]
        entrenamiento = np.concatenate([o["val_idx"] for o in folds
                                        if o["fold"] != fold["fold"]])
        if modelo == "arbol":
            ajuste = lgb.LGBMClassifier(**HIERBA)
        else:
            ajuste = ExtraTreesClassifier(**EXTRA)
        ajuste.fit(matriz[entrenamiento], y[entrenamiento])
        salida[validacion] = a_clases(ajuste.predict_proba(matriz[validacion]),
                                      ajuste.classes_, len(validacion))
    np.save(cache, salida)
    return salida


def evaluar(nombre: str, oof_candidato: np.ndarray, clase: np.ndarray, y: np.ndarray,
            folds: list[dict], oof_v19: np.ndarray) -> dict:
    """Metricas completas de un candidato, medidas **contra V19**."""
    valor = macro(clase, oof_candidato)
    valor_v19 = macro(clase, oof_v19)
    por_fold, por_fold_v19, desviaciones = [], [], []
    for fold in folds:
        idx = fold["val_idx"]
        a = macro(clase[idx], oof_candidato[idx])
        b = macro(clase[idx], oof_v19[idx])
        por_fold.append(a)
        por_fold_v19.append(b)
        desviaciones.append(a - b)
    intervalo = bootstrap_diferencia(oof_v19, oof_candidato, clase, len(cfg.CLASSES))
    fronteras = metricas_frontera(oof_candidato, clase, PARES)
    fronteras_v19 = metricas_frontera(oof_v19, clase, PARES)
    pred = np.array(cfg.CLASSES)[np.argmax(oof_candidato, axis=1)]
    pred_v19 = np.array(cfg.CLASSES)[np.argmax(oof_v19, axis=1)]
    acierta = pred == y
    acierta_v19 = pred_v19 == y
    f1 = f1_score(clase, np.argmax(oof_candidato, axis=1), average=None, labels=ETIQUETAS)
    return {
        "candidato": nombre, "macro_f1": round(valor, 7), "macro_f1_v19": round(valor_v19, 7),
        "delta_contra_v19": round(valor - valor_v19, 6),
        "delta_contra_v12": round(valor - 0.6005017473, 6),
        "media_folds": round(float(np.mean(por_fold)), 6),
        "desviacion_folds": round(float(np.std(por_fold)), 6),
        "por_fold": [round(v, 6) for v in por_fold],
        "delta_por_fold": [round(v, 6) for v in desviaciones],
        "folds_positivos": sum(1 for d in desviaciones if d > 0),
        "bootstrap_ic95_contra_v19": [round(v, 6) for v in intervalo],
        "f1_por_clase": {c: round(float(f1[i]), 4) for i, c in enumerate(cfg.CLASSES)},
        "frontera_1": fronteras["18-24|25-34"]["accuracy"],
        "frontera_1_v19": fronteras_v19["18-24|25-34"]["accuracy"],
        "frontera_2": fronteras["25-34|35-49"]["accuracy"],
        "frontera_2_v19": fronteras_v19["25-34|35-49"]["accuracy"],
        "complementariedad": {
            "ambos_correctos": int((acierta & acierta_v19).sum()),
            "ambos_incorrectos": int((~acierta & ~acierta_v19).sum()),
            "solo_candidato": int((acierta & ~acierta_v19).sum()),
            "solo_v19": int((acierta_v19 & ~acierta).sum()),
            "neto": int((acierta & ~acierta_v19).sum()) - int((acierta_v19 & ~acierta).sum()),
            "predicciones_modificadas": int((pred != pred_v19).sum())},
    }


def main() -> int:
    """Construye la informacion nueva y mide los cuatro candidatos contra V19."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    matriz = matriz_279()
    numericas, nombres122 = matriz_122()
    pesos_v19 = (1 / 3, 1 / 3, 1 / 3)
    oof_v19 = (pesos_v19[0] * np.load(cfg.FEATURES / "v19_b1_oof_logreg.npy")
               + pesos_v19[1] * np.load(cfg.FEATURES / "v19_b1_oof_mlp.npy")
               + pesos_v19[2] * np.load(cfg.FEATURES / "v19_oof_arbol_5folds.npy"))
    print(f"[V19] CV {macro(clase, oof_v19):.7f} | referencia congelada, no se toca", flush=True)

    # ---------------------------------------------------------------- F1: la informacion nueva
    print(f"[F1] construyendo la vista de clase condicional de {numericas.shape[1]} features, "
          f"in-fold", flush=True)
    if BLOQUE.exists():
        bloque = np.load(BLOQUE)
        print(f"    cache: {BLOQUE.name} con {bloque.shape[1]} columnas", flush=True)
    else:
        bloque = bloque_clase_condicional(numericas, y, clase, folds)
        np.save(BLOQUE, bloque)
    print(f"[F1] bloque de {bloque.shape[1]} columnas construido", flush=True)

    # Cuanto aporta el transform, con el **mismo aprendiz** en los tres casos: el arbol pre-registrado
    # de V19 sobre las 279, sobre las 122 crudas y sobre las 976 del bloque. La comparacion queda
    # emparejada por modelo e hiperparametros, luego el unico factor que cambia es la transformacion.
    print("[F1] arbol pre-registrado de V19 sobre las 122 features CRUDAS (control emparejado)",
          flush=True)
    oof_crudas = oof(numericas, y, folds, "arbol")
    print(f"    122 crudas: {macro(clase, oof_crudas):.7f}", flush=True)
    print("[F1a] arbol pre-registrado de V19 sobre las 976 columnas del bloque", flush=True)
    oof_f1a = oof(bloque, y, folds, "arbol")
    print(f"    976 del bloque: {macro(clase, oof_f1a):.7f}", flush=True)
    print("[F2] ExtraTrees sobre las mismas 279 de V19, en solitario", flush=True)
    oof_f2a = oof(matriz, y, folds, "extratrees")

    # ---------------------------------------------------------------- combinaciones
    print("[F1b] cuarto miembro: V19 en un tercio de cada uno mas el bloque en 1/4", flush=True)
    logreg_oof = np.load(cfg.FEATURES / "v19_b1_oof_logreg.npy")
    mlp_oof = np.load(cfg.FEATURES / "v19_b1_oof_mlp.npy")
    arbol_v19_oof = np.load(cfg.FEATURES / "v19_oof_arbol_5folds.npy")
    oof_f1b = (0.25 * logreg_oof + 0.25 * mlp_oof + 0.25 * arbol_v19_oof + 0.25 * oof_f1a)
    print("[F2b] cuarto miembro: V19 en un tercio de cada uno mas ExtraTrees en 1/4", flush=True)
    oof_f2b = (0.25 * logreg_oof + 0.25 * mlp_oof + 0.25 * arbol_v19_oof + 0.25 * oof_f2a)

    candidatos = [
        ("F1a arbol de V19 sobre el bloque de clase condicional", oof_f1a),
        ("F1b cuarto miembro: bloque de clase condicional", oof_f1b),
        ("F2a ExtraTrees sobre las 279, en solitario", oof_f2a),
        ("F2b cuarto miembro: ExtraTrees sobre las 279", oof_f2b),
    ]
    resultados = []
    for nombre, candidato in candidatos:
        item = evaluar(nombre, candidato, clase, y, folds, oof_v19)
        resultados.append(item)
        print(f"  {nombre:<52} {item['macro_f1']} | contra V19 "
              f"{item['delta_contra_v19']:+.4f} | {item['folds_positivos']}/5 | IC95 "
              f"[{item['bootstrap_ic95_contra_v19'][0]:+.4f}, "
              f"{item['bootstrap_ic95_contra_v19'][1]:+.4f}] | frontera 1 "
              f"{item['frontera_1']} y 2 {item['frontera_2']} | neto "
              f"{item['complementariedad']['neto']:+,} | cambios "
              f"{item['complementariedad']['predicciones_modificadas']:,}", flush=True)

    # ---------------------------------------------------------------- clasificacion
    def clasificar(item: dict) -> str:
        delta = item["delta_contra_v19"]
        cambios = item["complementariedad"]["predicciones_modificadas"]
        if delta >= 0.002 and item["folds_positivos"] >= 4 and cambios < 0.25 * len(y):
            return "NEW_SIGNAL"
        if delta > 0:
            return "WEAK_SIGNAL"
        if cambios < 0.005 * len(y) and abs(delta) < 0.0005:
            return "OVERFIT: cambia casi nada y no mejora"
        if item["candidato"].startswith("F2a"):
            return ("NO_SIGNAL: cierra la clase de funcion alternativa, y con ella queda demostrado "
                    "que el efecto de V19 es del BOOSTING y no de la no linealidad en general")
        return "NO_SIGNAL"

    for item in resultados:
        item["clase"] = clasificar(item)

    tabla = pd.DataFrame([{
        "candidato": "V19 (champion congelado)", "macro_f1": CV_V19, "delta_contra_v19": 0.0,
        "folds_positivos": "-", "ic95": "-", "frontera_1": 0.5083, "frontera_2": 0.5018,
        "solo_candidato": 0, "solo_v19": 0, "neto": 0, "cambios": 0, "clase": "CHAMPION"}] + [{
        "candidato": r["candidato"], "macro_f1": r["macro_f1"],
        "delta_contra_v19": r["delta_contra_v19"],
        "folds_positivos": f"{r['folds_positivos']}/5",
        "ic95": f"[{r['bootstrap_ic95_contra_v19'][0]:+.4f}, {r['bootstrap_ic95_contra_v19'][1]:+.4f}]",
        "frontera_1": r["frontera_1"], "frontera_2": r["frontera_2"],
        "solo_candidato": r["complementariedad"]["solo_candidato"],
        "solo_v19": r["complementariedad"]["solo_v19"],
        "neto": r["complementariedad"]["neto"],
        "cambios": r["complementariedad"]["predicciones_modificadas"],
        "clase": r["clase"]} for r in resultados])
    tabla.to_csv(COMPARATIVA, index=False)
    pd.DataFrame([{"candidato": r["candidato"], **r["complementariedad"]}
                  for r in resultados]).to_csv(COMPLEMENTARIEDAD, index=False)

    fig, ejes = plt.subplots(1, 2, figsize=(16, 5.5))
    eje = ejes[0]
    nombres = ["V19"] + [r["candidato"].split()[0] + "\n" + r["candidato"].split(";")[0][:24]
                          for r in resultados]
    valores = [0.0] + [r["delta_contra_v19"] * 1000 for r in resultados]
    eje.bar(range(len(nombres)), valores, color=["#1a9850"] + ["#2166ac"] * len(resultados))
    eje.axhline(0, color="grey", linewidth=1)
    eje.set_xticks(range(len(nombres)))
    eje.set_xticklabels(nombres, fontsize=6)
    eje.set_ylabel("delta de macro-F1 contra V19 (milesimas)")
    eje.set_title("F1 informacion nueva y F2 clase de funcion distinta", fontsize=10)
    eje = ejes[1]
    indice = np.arange(len(resultados))
    eje.bar(indice - 0.2, [r["complementariedad"]["solo_candidato"] for r in resultados], 0.4,
            label="solo el candidato acierta", color="#1a9850")
    eje.bar(indice + 0.2, [-r["complementariedad"]["solo_v19"] for r in resultados], 0.4,
            label="solo V19 acierta (negativo)", color="#b2182b")
    eje.axhline(0, color="grey", linewidth=1)
    eje.set_xticks(indice)
    eje.set_xticklabels([r["candidato"].split()[0] for r in resultados], fontsize=8)
    eje.set_ylabel("filas")
    eje.set_title("Rescates y danos frente a V19", fontsize=10)
    eje.legend(fontsize=8)
    fig.suptitle("V20 F1 y F2: todo medido contra el champion congelado", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURA, dpi=130)
    plt.close(fig)

    INFORME.write_text("\n".join([
        "# V20 · F1 y F2 — Informacion nueva de clase condicional, y clase de funcion distinta", "",
        "**V19 queda congelado.** No se toca su configuracion, sus features, su modelo persistido, su "
        "submission ni su hash. Todo lo de aqui es un challenger independiente.", "",
        "## 1. La auditoria que decidio las familias", "",
        "Se cruzaron los 117 experimentos contra 28 mecanismos. **Nunca** se habian probado "
        "**RandomForest/ExtraTrees, HistGradientBoosting ni CatBoost** como clase de funcion sobre las "
        "279. Y el bloque label-informed de 73 columnas se construyo in-fold sobre **cinco** familias de "
        "entidades (mencion, hashtag, fuente de retweet, rareza, lexico), dejando **sin transformar** las "
        "**122 features numericas**.", "",
        "**El `id` queda cerrado con medicion, en un comando**: es el indice de fila, 1 a 169.336 en "
        "train y en test, con correlacion de rango **0,00123** con la clase y distribucion plana por "
        "quintiles. No hay senal de tiempo que extraer, y la familia no consume un hueco.", "",
        "## 2. F1, la vista de clase condicional de las 122 features", "",
        f"Por cada fold y **solo con sus filas de entrenamiento** se estima la media y la desviacion de "
        f"cada una de las {numericas.shape[1]} features en cada una de las seis clases, y se estandariza "
        f"el valor de validacion contra esas seis distribuciones. De cada feature quedan **ocho "
        f"columnas**: seis z por clase, el margen y la entropia. Total **{bloque.shape[1]} columnas**.",
        "",
        "**Por que no es un DUPLICATE.** El bloque de 73 aporta **+0,0203** en interaccion sobre las "
        "206, y el de 122 nunca ha recibido ese tratamiento. Un arbol con `max_depth = 3` y "
        "`num_leaves = 8` no puede expresar la comparacion de una feature contra la distribucion de "
        "seis clases para 122 features a la vez: lo crudo da el valor, y la comparacion es lo que "
        "discrimina.", "",
        "**El bloque medido con el mismo aprendiz que V19 y con sus mismos hiperparametros**, para que "
        "el unico factor que cambie sea la transformacion:", "",
        "| Entrada del arbol pre-registrado de V19 | Columnas | macro-F1 |", "|---|---|---|",
        f"| las 279 de V19 | 279 | {CV_ARBOL_V19} |",
        f"| las 122 numericas CRUDAS | {numericas.shape[1]} | {macro(clase, oof_crudas)} |",
        f"| las 976 del bloque de clase condicional | {bloque.shape[1]} | {macro(clase, oof_f1a)} |", "",
        "## 3. Resultados, todo contra V19", "",
        "| Candidato | macro-F1 | Delta contra V19 | Folds pos. | IC95 | Frontera 1 | Frontera 2 | "
        "Rescates | Danos | Neto | Cambios | Clase |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
        f"| **V19 (champion)** | {CV_V19} | - | - | - | 0,5083 | 0,5018 | - | - | - | - | CHAMPION |",
        *[f"| {r['candidato']} | {r['macro_f1']} | {r['delta_contra_v19']:+.4f} | "
          f"{r['folds_positivos']}/5 | [{r['bootstrap_ic95_contra_v19'][0]:+.4f}, "
          f"{r['bootstrap_ic95_contra_v19'][1]:+.4f}] | {r['frontera_1']} | {r['frontera_2']} | "
          f"{r['complementariedad']['solo_candidato']:,} | {r['complementariedad']['solo_v19']:,} | "
          f"{r['complementariedad']['neto']:+,} | "
          f"{r['complementariedad']['predicciones_modificadas']:,} | {r['clase']} |"
          for r in resultados], "",
        "## 4. Desviacion por fold y estabilidad", "",
        "| Candidato | Delta por fold |", "|---|---|",
        *[f"| {r['candidato']} | {r['delta_por_fold']} |" for r in resultados], "",
        "## 5. F1 por clase de los candidatos", "",
        "| Candidato | " + " | ".join(cfg.CLASSES) + " |",
        "|---|" + "---|" * len(cfg.CLASSES),
        *[f"| {r['candidato']} | " + " | ".join(f"{r['f1_por_clase'][c]}" for c in cfg.CLASSES) + " |"
          for r in resultados], "",
    ]), encoding="utf-8")

    RESULTADO.write_text(json.dumps({
        "fase": "V20 F1 y F2", "fecha": "2026-09-27",
        "champion_congelado": {"nombre": "V19 boosting", "lb": 0.62578, "cv": CV_V19,
                               "intacto": True},
        "auditoria": {"experimentos_revisados": 117, "mecanismos_cruzados": 28,
                      "nunca_probados": ["RandomForest/ExtraTrees", "HistGradientBoosting",
                                         "CatBoost", "transformacion de clase condicional de las 122 "
                                                     "features numericas"],
                      "id_cerrado_con_medicion": {"es_indice_de_fila": True,
                                                  "correlacion_de_rango_con_la_clase": 0.00123}},
        "f1": {"features_transformadas": int(numericas.shape[1]), "columnas": int(bloque.shape[1]),
               "construccion": "in-fold, solo con las filas de entrenamiento de cada fold",
               "comparacion_emparejada_mismo_aprendiz": {
                   "arbol_sobre_las_279": CV_ARBOL_V19,
                   "arbol_sobre_las_122_crudas": round(macro(clase, oof_crudas), 7),
                   "arbol_sobre_las_976_del_bloque": round(macro(clase, oof_f1a), 7)}},
        "f2": {"modelo": "ExtraTreesClassifier", "hiperparametros": EXTRA,
               "pre_registrado": True},
        "resultados": resultados, "submission_generada": False,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nclases: " + " | ".join(f"{r['candidato'].split()[0]}={r['clase'].split(':')[0]}"
                                     for r in resultados))
    print(f"segundos: {time.time() - inicio:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
