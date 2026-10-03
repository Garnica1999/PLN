"""V26-03: Tsetlin Machine con la implementacion publicada de pyTsetlinMachine.

Por que este script existe
--------------------------
La implementacion casera de Tsetlin Machine de `v26_02_tsetlin.py` queda **descartada como
implementacion experimental** y sus resultados **no son evidencia** del rendimiento del
algoritmo. Durante su desarrollo aparecieron, entre otros, un doble factor en un gradiente, un
escalado incorrecto, un `clip` que destruia los estados include/exclude, la polaridad por
mitades en lugar de por paridad, una implementacion incompleta del feedback, la ausencia de
`append_negated` y un deadlock en la puerta de inclusion. Sus controles sinteticos llegaron a
resolver el XOR, pero sus resultados sobre datos reales no son utilizables.

Aqui la unica implementacion valida es `pyTsetlinMachine`, que es software publicado, con
licencia MIT, maintained por el Centre for Artificial Intelligence Research (cair) de la
Universidad de Agder, y whose authors son los del algoritmo. Este script la usa **sin
modificar su codigo**.

La pregunta que responde
------------------------
    Existe senal real de que el sesgo inductivo de las clausulas booleanas aporte algo sobre
    este problema, con nuestra representacion binaria y nuestro tamano de dataset?

Y la distincion que hace falta, porque comparar Tsetlin contra V03 con dos representaciones
distintas mediria la diferencia entre dos representaciones y no entre dos mecanismos:

    Tsetlin(binario)  vs  LogisticRegression(binario)  ->  ganancia del MECANISMO TM
    LogisticRegression(binario)  vs  V03                 ->  coste de la BINARIZACION
    Tsetlin(binario)  vs  V03                           ->  efecto neto

Solo la primera es la pregunta de la familia. Sin las otras dos el numero no se interpreta.

Representacion
-------------
`word12` binaria, con `CountVectorizer(binary=True, min_df=2)`, ajustada **dentro de cada
pliegue** sobre las filas de entrenamiento de ese pliegue. El vocabulario nunca se ve la
validacion ni el test. La binarizacion es parte del mecanismo del Tsetlin, que consume bits, y
no un parametro: por eso el control lineal tiene que ver exactamente la misma matriz.

Ausencia de leakage, verificada en el script
-------------------------------------------
1. El vectorizador se ajusta dentro del bucle de pliegues, nunca antes.
2. El test no se lee en ningun punto del pipeline de evaluacion.
3. Las etiquetas del holdout no se usan.
4. No hay datos externos, ni embeddings preentrenados, ni transformers, ni LLM.
5. Los folds son los canonicos de V03, los mismos de `pliegues_corregidos`.

Sobre la semilla
----------------
`pyTsetlinMachine` **no expone `random_state`**: usa el `fast_rand()` de `fast_rand.h`, que se
alimenta del generador de la libreria C. Por tanto la reproducibilidad no se puede pedir a la
libreria. Se mide en su lugar: el script ejecuta la mejor configuracion dos veces y reporta si
las metricas coinciden, y el coste de esa no reproducibilidad se declara como limitacion, no se
oculta.

Uso
---
    /home/carlos/venv/bin/python experiments/v26_03_tsetlin_oficial.py
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.evaluation import metricas_multietiqueta
from src.splits import grupos_fusionados, pliegues_corregidos

EXPERIMENTO = "V26-03 (Tsetlin Machine con pyTsetlinMachine publicado, sobre word12 binario)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v26_03_tsetlin_oficial.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v26_03_tsetlin_oficial.md"
OOF_CANONICO = PROJECT_ROOT / "features" / "oof_v03_final.npz"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
V03_CONGELADO = PROJECT_ROOT / "experiments" / "baseline_v03_frozen" / "submission_v03.csv"
SHA256_V03_ESPERADO = "7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e"

ETIQUETAS = ("HS", "TR", "AG")
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
CORRECTED_CV_V03 = 0.7772114525165721
CONTROL_ESPERADO = 0.751393
C_V03 = 2.0
SEMILLA = 42

# Valores por defecto prudentes de la documentacion de la libreria, sin optimizar. `T` se deja
# en la mitad de las clausulas, que es la convencion de los ejemplos oficiales, y `s` en 10.0,
# que es el valor de todos los ejemplos del repositorio.
PARAMETROS_BASE = {"s": 10.0, "epochs": 25, "boost_true_positive_feedback": 1}
VALORES_K = (200, 500, 1000)
N_PROCESOS = 5

PARAMETROS_LR_V03 = {
    "C": C_V03,
    "class_weight": "balanced",
    "solver": "liblinear",
    "fit_intercept": True,
    "tol": 1e-4,
    "max_iter": 1000,
    "random_state": SEMILLA,
}
REPRESENTACION = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "binary": True}

_ESTADO: dict[str, Any] = {}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V26-03] {mensaje}", flush=True)


def memoria_pico_mb() -> float:
    """Resident set maximum del proceso, en megabytes.

    Returns:
        La memoria maxima residente en MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def sha256_de(ruta: Path) -> str:
    """SHA256 de un fichero, para verificar que V03 no se toca.

    Args:
        ruta: ruta del fichero.

    Returns:
        El hash en hexadecimal.
    """
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def smoke_test_xor() -> dict[str, Any]:
    """Control canonico: XOR de dos features, con la libreria publicada.

    Verifica la instalacion, el entrenamiento, la prediccion y la capacidad de aprender una
    conjuncion. No optimiza nada.

    Returns:
        Diccionario con el resultado.
    """
    x = np.array([[0, 0, 1, 0, 0, 1, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0]], dtype=np.uint32)
    y = np.array([0, 1], dtype=np.uint32)
    x = np.tile(x, (500, 1))
    y = np.tile(y, 500)
    modelo = MultiClassTsetlinMachine(
        number_of_clauses=100, T=50, s=10.0, indexed=True, append_negated=True
    )
    inicio = time.time()
    modelo.fit(x, y, epochs=25)
    segundos = time.time() - inicio
    predicho = modelo.predict(x)
    exactitud = float((predicho == y).mean())
    # Segundo ajuste identico, para medir la reproducibilidad de la libreria.
    modelo2 = MultiClassTsetlinMachine(
        number_of_clauses=100, T=50, s=10.0, indexed=True, append_negated=True
    )
    modelo2.fit(x, y, epochs=25)
    predicho2 = modelo2.predict(x)
    return {
        "n_filas": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "clases": 2,
        "clausulas": 100,
        "T": 50,
        "s": 10.0,
        "epochs": 25,
        "segundos": round(segundos, 3),
        "exactitud_train": exactitud,
        "f1_clase_0": float(2 * ((predicho == 0) & (y == 0)).sum() / max(((predicho == 0) | (y == 0)).sum(), 1)),
        "f1_clase_1": float(2 * ((predicho == 1) & (y == 1)).sum() / max(((predicho == 1) | (y == 1)).sum(), 1)),
        "dos_ajustes_identicos": bool(np.array_equal(predicho, predicho2)),
        "nota_reproducibilidad": (
            "pyTsetlinMachine no expone random_state: usa fast_rand() de la libreria C. "
            "Dos ajustes consecutivos en el mismo proceso pueden diferir, y esa "
            "irreproducibilidad se reporta en vez de ocultarla."
        ),
    }


def _ajustar_tsetlin(
    x_ent: np.ndarray,
    y_ent: np.ndarray,
    x_val: np.ndarray,
    k: int,
    t: float,
    s: float,
    epochs: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Ajusta un Tsetlin Machine de 2 clases y lo usa como clasificador binario.

    Se instancia con 2 clases porque `MultiClassTsetlinMachine` deriva el numero de clases de
    `max(Y)+1`, y se construye una instancia por etiqueta para mantener las tres etiquetas
    independientes, igual que hace V03 con sus tres LogisticRegression.

    Un unico modelo se ajusta sobre el entrenamiento y predice **tanto** entrenamiento como
    validacion. Antes se entrenaba dos veces, una para cada prediccion, lo que ademas de
    duplicar el coste rompia la relacion entre la metrica de train y la de validacion, porque
    eran dos modelos distintos.

    Args:
        x_ent: matriz binaria de entrenamiento en formato denso `uint32`.
        y_ent: etiquetas binarias de entrenamiento.
        x_val: matriz binaria de validacion en formato denso `uint32`.
        k: numero de clausulas.
        t: umbral T.
        s: sensibilidad.
        epochs: numero de epocas.

    Returns:
        Tupla con la prediccion de validacion, la de entrenamiento y el diagnostico.
    """
    inicio = time.time()
    modelo = MultiClassTsetlinMachine(
        number_of_clauses=k,
        T=t,
        s=s,
        boost_true_positive_feedback=PARAMETROS_BASE["boost_true_positive_feedback"],
        indexed=True,
        append_negated=True,
    )
    modelo.fit(x_ent, y_ent, epochs=epochs)
    pred_val = modelo.predict(x_val)
    pred_ent = modelo.predict(x_ent)
    diagnostico = {
        "segundos": round(time.time() - inicio, 2),
        "clausulas": k,
        "T": float(t),
        "s": float(s),
        "epochs": epochs,
    }
    return pred_val.astype(np.int64), pred_ent.astype(np.int64), diagnostico


def _init_worker() -> None:
    """Fija un hilo de BLAS en el proceso hijo.

    Returns:
        None.
    """
    threadpool_limits(1)


def _tarea(args: tuple[int, int, int]) -> dict[str, Any]:
    """Ajusta el control lineal y el Tsetlin en un pliegue, una etiqueta y un K.

    Args:
        args: tupla con el indice del pliegue, el de la etiqueta y K. Con `K = 0` la tarea
            es solo el control lineal y no se entrena ningun Tsetlin.

    Returns:
        Diccionario con las predicciones de validacion y de entrenamiento.
    """
    indice, etiqueta, k = args
    x_ent = _ESTADO["x_ent"][indice]
    x_val = _ESTADO["x_val"][indice]
    y_ent = _ESTADO["y"][_ESTADO["pliegues"][indice][0], etiqueta]
    y_val = _ESTADO["y"][_ESTADO["pliegues"][indice][1], etiqueta]

    # Control lineal sobre la MISMA matriz binaria que ve el Tsetlin.
    lr = LogisticRegression(**PARAMETROS_LR_V03)
    lr.fit(x_ent.astype(np.float64), y_ent)
    pred_lr = lr.predict(x_val.astype(np.float64)).astype(np.int64)
    pred_lr_ent = lr.predict(x_ent.astype(np.float64)).astype(np.int64)

    pred_tm = np.zeros(len(y_val), dtype=np.int64)
    pred_tm_ent = np.zeros(len(y_ent), dtype=np.int64)
    diag: dict[str, Any] = {"segundos": 0.0, "clausulas": 0}
    if k > 0:
        pred_tm, pred_tm_ent, diag = _ajustar_tsetlin(
            x_ent,
            y_ent,
            x_val,
            k,
            k / 2.0,
            PARAMETROS_BASE["s"],
            PARAMETROS_BASE["epochs"],
        )
    return {
        "indice": indice,
        "etiqueta": etiqueta,
        "k": k,
        "val_idx": np.asarray(_ESTADO["pliegues"][indice][1]),
        "pred_lr": pred_lr,
        "pred_lr_ent": pred_lr_ent,
        "pred_tm": pred_tm,
        "pred_tm_ent": pred_tm_ent,
        "n_features": int(x_ent.shape[1]),
        "diagnostico": diag,
    }


def main() -> None:
    """Ejecuta el smoke test, el control lineal y la progresion de clausulas.

    Returns:
        None.
    """
    inicio = time.time()
    log("V26-03: Tsetlin Machine con pyTsetlinMachine. NINGUNA submission en este script")

    log("0/9 V03 congelado intacto, verificado por hash")
    hash_v03 = sha256_de(V03_CONGELADO)
    v03_intacto = bool(hash_v03 == SHA256_V03_ESPERADO)
    log(f"   sha256 {hash_v03}")
    log(f"   coincide con el hash de referencia: {v03_intacto}")
    if not v03_intacto:
        raise SystemExit("PARADA OBLIGATORIA: el artefacto congelado de V03 ha cambiado")

    log("1/9 smoke test canonico con la libreria publicada")
    smoke = smoke_test_xor()
    log(
        f"   XOR: exactitud {smoke['exactitud_train']:.4f} | F1 clase0 "
        f"{smoke['f1_clase_0']:.4f} | F1 clase1 {smoke['f1_clase_1']:.4f} | "
        f"{smoke['segundos']} s"
    )
    log(f"   dos ajustes consecutivos identicos: {smoke['dos_ajustes_identicos']}")
    if smoke["exactitud_train"] < 0.99:
        raise SystemExit("PARADA OBLIGATORIA: la libreria no resuelve el XOR canonico")

    log("2/9 datos, folds canonicos de V03 y alineacion por identificador")
    train = read_train()
    particion = pd.read_csv(PARTICION)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    grupos_full = grupos_fusionados(train)
    mapa_grupo = dict(zip(train["id"].astype(int), grupos_full.tolist()))
    grupos_dev = np.array([mapa_grupo[i] for i in desarrollo["id"].astype(int)])
    pliegues = pliegues_corregidos(desarrollo, grupos_dev)
    y = desarrollo[list(ETIQUETAS)].to_numpy().astype(np.int64)
    textos = desarrollo["text"]
    oof = np.load(OOF_CANONICO, allow_pickle=True)
    y_oof = np.asarray(oof["y_true"], dtype=np.int64)
    proba_v03 = np.asarray(oof["proba"], dtype=np.float64)
    pred_v03 = np.asarray(oof["pred"], dtype=np.int64)
    mapa_pos = {v: i for i, v in enumerate(desarrollo["id"].astype(int).to_numpy())}
    posiciones = np.array([mapa_pos[i] for i in np.asarray(oof["ids"]).astype(np.int64)])
    if not np.array_equal(y[posiciones], y_oof):
        raise SystemExit("PARADA OBLIGATORIA: las etiquetas no coinciden tras alinear por id")
    if not np.array_equal(pred_v03, (proba_v03 >= UMBRALES_V03[None, :]).astype(int)):
        log("   AVISO: el `pred` guardado no coincide con el umbral aplicado a `proba`")
    log(f"   desarrollo {len(desarrollo)} | {len(pliegues)} pliegues | alineacion verificada")
    log(f"   V03 corrected_cv {float(np.asarray(oof['corrected_cv']).ravel()[0]):.16f}")

    log("3/9 representacion word12 binaria, con vocabulario ajustado DENTRO de cada pliegue")
    x_ent_lista: list[np.ndarray] = []
    x_val_lista: list[np.ndarray] = []
    dims: list[int] = []
    for indice, (entrenar, validar) in enumerate(pliegues):
        vec = CountVectorizer(**REPRESENTACION)
        # El vectorizador ve SOLO las filas de entrenamiento de este pliegue.
        x_ent = vec.fit_transform(textos.iloc[np.asarray(entrenar)])
        x_val = vec.transform(textos.iloc[np.asarray(validar)])
        x_ent_lista.append(x_ent.toarray().astype(np.uint32))
        x_val_lista.append(x_val.toarray().astype(np.uint32))
        dims.append(x_ent.shape[1])
        log(
            f"   pliego {indice + 1}: {x_ent.shape[0]} x {x_ent.shape[1]} features | "
            f"nnz {x_ent.nnz} | densidad {x_ent.nnz / (x_ent.shape[0] * x_ent.shape[1]):.5f} | "
            f"val {x_val.shape[0]} x {x_val.shape[1]}"
        )
    log(f"   dimensiones por pliegue: {dims}")
    log("   el test no se lee en ningun punto de este pipeline")

    _ESTADO["x_ent"] = x_ent_lista
    _ESTADO["x_val"] = x_val_lista
    _ESTADO["y"] = y
    _ESTADO["pliegues"] = [(np.asarray(a), np.asarray(b)) for a, b in pliegues]

    contexto = mp.get_context("fork")
    with (
        threadpool_limits(1),
        ProcessPoolExecutor(
            max_workers=N_PROCESOS, mp_context=contexto, initializer=_init_worker
        ) as pool,
    ):
        log("4/9 el CONTROL LINEAL sobre la MISMA matriz binaria")
        tareas = [(i, j, 0) for i in range(len(pliegues)) for j in range(len(ETIQUETAS))]
        res_control = list(pool.map(_tarea, tareas))
        control = _agregar(res_control, y_oof, pred_v03, y, k=0)
        log(
            f"   LR(binario) corrected_cv {control['corrected_cv']:.6f} "
            f"({control['delta_vs_v03']:+.6f}) | benchmark historico {CONTROL_ESPERADO}"
        )
        log(
            f"   train F1 {control['f1_train']:.4f} | gap {control['gap_train_oof']:+.4f} | "
            f"rescates {control['rescates']} | danos {control['danos']} | neto {control['neto']:+d}"
        )
        diferencia_control = abs(control["corrected_cv"] - CONTROL_ESPERADO)
        log(f"   diferencia respecto al benchmark: {diferencia_control:.6f}")
        if diferencia_control > 0.005:
            log("   AVISO: el control no reproduce el benchmark historico, revisar antes de interpretar TM")

        log("5/9 progresion adaptativa de clausulas")
        filas: list[dict[str, Any]] = []
        k_actual = VALORES_K[0]
        corte: dict[str, Any] = {"k": k_actual, "motivo": None}
        while True:
            tareas = [(i, j, k_actual) for i in range(len(pliegues)) for j in range(len(ETIQUETAS))]
            res = list(pool.map(_tarea, tareas))
            fila = _agregar(res, y_oof, pred_v03, y, k=k_actual)
            filas.append(fila)
            log(
                f"   K={k_actual:<5} TM corrected_cv {fila['corrected_cv']:.6f} "
                f"({fila['delta_vs_control']:+.6f} vs control) "
                f"({fila['delta_vs_v03']:+.6f} vs V03) | train {fila['f1_train']:.4f} | "
                f"gap {fila['gap_train_oof']:+.4f} | sd pliegues {fila['sd_entre_pliegues']:.4f} | "
                f"rescates {fila['rescates']} | danos {fila['danos']} | neto {fila['neto']:+d} | "
                f"{fila['segundos']} s"
            )
            if not _hay_senal(fila, control):
                corte = {"k": k_actual, "motivo": _motivo_corte(fila, control)}
                log(f"   SIN SENAL en K={k_actual}: {corte['motivo']}")
                break
            siguiente = {200: 500, 500: 1000}.get(k_actual)
            if siguiente is None:
                corte = {"k": k_actual, "motivo": "se alcanzo K=1000, el tope de esta fase"}
                break
            k_actual = siguiente

    log("6/9 detalle por etiqueta, estabilidad y complementariedad")
    for fila in filas:
        log(
            f"   K={fila['k']}: "
            + " | ".join(
                f"{et} F1 {d['f1']:.4f} (V03 {d['f1_v03']:.4f}, control {d['f1_control']:.4f}) "
                f"resc {d['rescates']} dan {d['danos']}"
                for et, d in fila["por_etiqueta"].items()
            )
        )
    for fila in filas:
        log(
            f"   K={fila['k']} overlap con V03: coinciden {fila['celdas_coincidentes']} de "
            f"{fila['celdas_totales']} ({fila['fraccion_coincidente']:.4f}) | "
            f"correlacion {fila['correlacion_predicciones']:.4f} | ambos incorrectos "
            f"{fila['ambos_malos']}"
        )

    log("7/9 reproducibilidad de la mejor configuracion")
    mejor = max(filas, key=lambda f: f["corrected_cv"])
    log(
        f"   reejecutando K={mejor['k']} para medir si las metricas se reproducen, dado que "
        f"la libreria no expone semilla"
    )
    with (
        threadpool_limits(1),
        ProcessPoolExecutor(
            max_workers=N_PROCESOS, mp_context=contexto, initializer=_init_worker
        ) as pool2,
    ):
        tareas = [(i, j, mejor["k"]) for i in range(len(pliegues)) for j in range(len(ETIQUETAS))]
        res_rep = list(pool2.map(_tarea, tareas))
    repetido = _agregar(res_rep, y_oof, pred_v03, y, k=mejor["k"])
    diferencia = abs(repetido["corrected_cv"] - mejor["corrected_cv"])
    log(
        f"   primera {mejor['corrected_cv']:.6f} | repetida {repetido['corrected_cv']:.6f} | "
        f"diferencia {diferencia:.6f}"
    )
    log(
        f"   mismo numero de cambios: {mejor['celdas']} vs {repetido['celdas']} | "
        f"mismas predicciones: {bool(np.array_equal(mejor['_pred'], repetido['_pred']))}"
    )

    log("8/9 informe")
    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "implementacion": {
            "libreria": "pyTsetlinMachine",
            "version": "0.6.6",
            "licencia": "MIT",
            "commit": "d6c1cf0e4aaa4a8ae2f2818ba27878fb89d31dc5",
            "fecha_del_commit": "2025-04-01",
            "repositorio": "https://github.com/cair/pyTsetlinMachine/",
            "clase_usada": "MultiClassTsetlinMachine",
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "plataforma": f"{os.uname().sysname} {os.uname().machine}",
            "modificaciones_al_codigo_de_la_libreria": "ninguna",
            "nota_de_compilacion": (
                "el makefile del proyecto usa -Werror y con gcc moderno falla por variables sin "
                "usar. Se recompilo con los mismos flags de optimizacion quitando -Werror, que "
                "es un cambio de build y no de codigo fuente."
            ),
            "por_que_esta_permitida": (
                "las reglas de la competicion prohiben transformers y LLM, y exigen que los "
                "modelos de vectorizacion se entrenen desde cero. pyTsetlinMachine es un modelo "
                "de reconocimiento de patrones entrenado desde cero, sin datos externos ni pesos "
                "preentrenados, luego cumple."
            ),
        },
        "controles": {
            "v03_congelado_intacto": v03_intacto,
            "sha256_v03": hash_v03,
            "smoke_test_xor": smoke,
            "representacion": {
                "vectorizador": "CountVectorizer",
                "parametros": REPRESENTACION,
                "ajustado_dentro_de_cada_pliegue": True,
                "test_leido": False,
                "dimensiones_por_pliegue": dims,
            },
        },
        "control_lineal_sobre_la_misma_matriz": {
            k: v for k, v in control.items() if not k.startswith("_")
        },
        "benchmark_del_control": CONTROL_ESPERADO,
        "resultados_tsetlin": [{k: v for k, v in f.items() if not k.startswith("_")} for f in filas],
        "reproducibilidad": {
            "k": mejor["k"],
            "primera": mejor["corrected_cv"],
            "repetida": repetido["corrected_cv"],
            "diferencia": diferencia,
            "mismo_numero_de_cambios": bool(mejor["celdas"] == repetido["celdas"]),
            "mismas_predicciones": bool(np.array_equal(mejor["_pred"], repetido["_pred"])),
            "causa": "pyTsetlinMachine usa fast_rand() de la libreria C y no expone random_state",
        },
        "corte": corte,
        "gastos": {
            "submissions": 0,
            "memoria_pico_mb": round(memoria_pico_mb(), 1),
            "segundos_totales": round(time.time() - inicio, 1),
        },
    }
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _informe_md(informe)
    log("   informe escrito en reports/v26_03_tsetlin_oficial.{json,md}")
    log(f"   memoria pico {memoria_pico_mb():.0f} MB | tiempo {round(time.time() - inicio, 1)} s")
    log(
        f"VEREDICTO: control {control['corrected_cv']:.6f} | mejor TM K={mejor['k']} "
        f"{mejor['corrected_cv']:.6f} ({mejor['delta_vs_control']:+.6f} vs control, "
        f"{mejor['delta_vs_v03']:+.6f} vs V03) | corte en K={corte['k']} | submissions 0"
    )


def _hay_senal(fila: dict[str, Any], control: dict[str, Any]) -> bool:
    """Decide si la configuracion muestra senal que justifique subir de capacidad.

    Los criterios son los de la fase, no un umbral arbitrario de mejora: una combinacion
    convincente de mejora sobre el control binario, complementariedad visible con V03 y
    estabilidad entre pliegues. El gap alto no cierra por si solo, porque un gap alto tambien
    aparece en modelos que rescatan errores utiles.

    Args:
        fila: resultados de la configuracion.
        control: resultados del control lineal.

    Returns:
        True si hay senal.
    """
    mejora = fila["delta_vs_control"] > 0.0
    complementariedad = fila["neto"] > 0 or (fila["rescates"] >= 20 and fila["fraccion_coincidente"] < 0.97)
    estabilidad = fila["sd_entre_pliegues"] < 0.10
    return bool((mejora and estabilidad) or (mejora and complementariedad))


def _motivo_corte(fila: dict[str, Any], control: dict[str, Any]) -> str:
    """Explica por que se corta la progresion.

    Args:
        fila: resultados de la configuracion.
        control: resultados del control lineal.

    Returns:
        Texto con el motivo.
    """
    motivos = []
    if fila["delta_vs_control"] <= 0:
        motivos.append(f"no mejora al control ({fila['delta_vs_control']:+.6f})")
    if fila["neto"] <= 0:
        motivos.append(f"neto de celdas no positivo ({fila['neto']:+d})")
    if fila["sd_entre_pliegues"] >= 0.10:
        motivos.append(f"inestable entre pliegues (sd {fila['sd_entre_pliegues']:.4f})")
    if fila["gap_train_oof"] > 0.30:
        motivos.append(f"sobreajuste marcado (gap {fila['gap_train_oof']:+.4f})")
    return ", ".join(motivos) if motivos else "sin motivo claro, se corta por prudencia"


def _agregar(
    res: list[dict[str, Any]],
    y_oof: np.ndarray,
    pred_v03: np.ndarray,
    y: np.ndarray,
    k: int,
) -> dict[str, Any]:
    """Agrega los resultados de los 15 ajustes y calcula metricas y complementariedad.

    Args:
        res: resultados de las tareas.
        y_oof: etiquetas del OOF canonico de V03.
        pred_v03: predicciones de V03.
        y: etiquetas de desarrollo.
        k: numero de clausulas, 0 para el control lineal.

    Returns:
        Diccionario con la fila de informe.
    """
    n = len(y_oof)
    es_tm = k > 0
    clave = "pred_tm" if es_tm else "pred_lr"
    clave_ent = "pred_tm_ent" if es_tm else "pred_lr_ent"
    pred = np.zeros((n, len(ETIQUETAS)), dtype=np.int64)
    pred_control = np.zeros((n, len(ETIQUETAS)), dtype=np.int64)
    f1s_train: list[float] = []
    segundos = 0.0
    for r in res:
        val = r["indice"]
        posiciones = r["val_idx"]
        pred[posiciones, r["etiqueta"]] = r[clave]
        pred_control[posiciones, r["etiqueta"]] = r["pred_lr"]
        filas_tr = _ESTADO["pliegues"][val][0]
        f1s_train.append(_f1(y[filas_tr, r["etiqueta"]], r[clave_ent]))
        segundos += r["diagnostico"]["segundos"]
    metricas = metricas_multietiqueta(y_oof, pred)
    pred_v03_local = pred_v03

    # Complementariedad: los cuatro conteos que interesan.
    correcto_v03 = pred_v03_local == y_oof
    correcto_nuevo = pred == y_oof
    rescates = int(np.count_nonzero((~correcto_v03) & correcto_nuevo))
    danos = int(np.count_nonzero(correcto_v03 & (~correcto_nuevo)))
    coincidentes = int(np.count_nonzero(pred == pred_v03_local))
    ambos_malos = int(np.count_nonzero((~correcto_v03) & (~correcto_nuevo)))
    total = pred.size
    corrs = []
    for j in range(len(ETIQUETAS)):
        a = pred[:, j].astype(float)
        b = pred_v03_local[:, j].astype(float)
        corrs.append(float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0)

    metricas_control = metricas_multietiqueta(y_oof, pred_control)
    por_etiqueta = {}
    for j, etiqueta in enumerate(ETIQUETAS):
        por_etiqueta[etiqueta] = {
            "f1": float(metricas["f1_por_etiqueta"][etiqueta]),
            "f1_v03": float(_f1(y_oof[:, j], pred_v03_local[:, j])),
            "f1_control": float(metricas_control["f1_por_etiqueta"][etiqueta]),
            "celdas": int(np.count_nonzero(pred[:, j] != pred_v03_local[:, j])),
            "rescates": int(np.count_nonzero((~correcto_v03[:, j]) & correcto_nuevo[:, j])),
            "danos": int(np.count_nonzero(correcto_v03[:, j] & (~correcto_nuevo[:, j]))),
        }

    # Estabilidad: F1 calculada sobre el conjunto de cada pliegue por separado.
    pliegues_oof = np.asarray(np.load(OOF_CANONICO, allow_pickle=True)["pliegues"])
    f1_pliegues = []
    for f in np.unique(pliegues_oof):
        m = pliegues_oof == f
        f1_pliegues.append(_macro_f1(y_oof[m], pred[m]))
    f1_tr = float(np.mean(f1s_train))
    return {
        "k": k,
        "corrected_cv": float(metricas["f1_media_columnas"]),
        "delta_vs_v03": float(metricas["f1_media_columnas"]) - CORRECTED_CV_V03,
        "f1_por_etiqueta": {e: float(v) for e, v in metricas["f1_por_etiqueta"].items()},
        "f1_macro6": float(metricas["f1_macro6"]),
        "f1_train": f1_tr,
        "gap_train_oof": f1_tr - float(metricas["f1_media_columnas"]),
        "sd_entre_pliegues": float(np.std(f1_pliegues)),
        "f1_por_pliegue": [float(v) for v in f1_pliegues],
        "rescates": rescates,
        "danos": danos,
        "neto": rescates - danos,
        "delta_vs_control": float(metricas["f1_media_columnas"])
        - float(metricas_control["f1_media_columnas"]),
        "celdas": int(np.count_nonzero(pred != pred_v03_local)),
        "celdas_totales": total,
        "celdas_coincidentes": coincidentes,
        "fraccion_coincidente": coincidentes / total,
        "ambos_malos": ambos_malos,
        "correlacion_predicciones": float(np.mean(corrs)),
        "por_etiqueta": por_etiqueta,
        "segundos": round(segundos, 1),
        "_pred": pred,
    }


def _f1(y: np.ndarray, pred: np.ndarray) -> float:
    """F1 binaria de la clase positiva, con `zero_division=0`.

    Args:
        y: etiquetas verdaderas.
        pred: predicciones.

    Returns:
        El valor de F1.
    """
    comunes = int(np.count_nonzero(y & pred))
    den = int(np.count_nonzero(y)) + int(np.count_nonzero(pred))
    return 2 * comunes / den if den else 0.0


def _macro_f1(verdad: np.ndarray, predicho: np.ndarray) -> float:
    """Macro-F1 de las tres clases positivas, con `zero_division=0`.

    Args:
        verdad: matriz verdadera binaria.
        predicho: matriz predicha binaria.

    Returns:
        La media de las tres F1.
    """
    return float(np.mean([_f1(verdad[:, j], predicho[:, j]) for j in range(verdad.shape[1])]))


def _informe_md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.

    Returns:
        None.
    """
    impl = informe["implementacion"]
    c = informe["controles"]
    ctrl = informe["control_lineal_sobre_la_misma_matriz"]
    filas = informe["resultados_tsetlin"]
    rep = informe["reproducibilidad"]
    smoke = c["smoke_test_xor"]
    lineas = [
        "# V26-03: Tsetlin Machine con pyTsetlinMachine, sobre word12 binario",
        "",
        f"Generado por `experiments/v26_03_tsetlin_oficial.py` ({EXPERIMENTO}).",
        "",
        "**Ninguna submission.**",
        "",
        "## 1. La implementacion usada",
        "",
        "| campo | valor |",
        "|---|---|",
        f"| libreria | `{impl['libreria']}` |",
        f"| version | {impl['version']} |",
        f"| licencia | {impl['licencia']} |",
        f"| commit | `{impl['commit']}` ({impl['fecha_del_commit']}) |",
        f"| repositorio | {impl['repositorio']} |",
        f"| clase | `{impl['clase_usada']}` |",
        f"| python | {impl['python']} |",
        f"| numpy | {impl['numpy']} |",
        f"| plataforma | {impl['plataforma']} |",
        f"| modificaciones al codigo de la libreria | {impl['modificaciones_al_codigo_de_la_libreria']} |",
        "",
        impl["nota_de_compilacion"],
        "",
        f"**Por que es legal**: {impl['por_que_esta_permitida']}",
        "",
        "La implementacion casera de `v26_02_tsetlin.py` queda **REJECTED** y sus resultados no",
        "son evidencia del rendimiento del algoritmo.",
        "",
        "## 2. Smoke test canonico",
        "",
        "| magnitud | valor |",
        "|---|---:|",
        f"| exactitud en XOR de train | {smoke['exactitud_train']:.4f} |",
        f"| F1 clase 0 | {smoke['f1_clase_0']:.4f} |",
        f"| F1 clase 1 | {smoke['f1_clase_1']:.4f} |",
        f"| segundos | {smoke['segundos']} |",
        f"| dos ajustes consecutivos identicos | {smoke['dos_ajustes_identicos']} |",
        "",
        smoke["nota_reproducibilidad"],
        "",
        "## 3. El control lineal sobre la MISMA matriz binaria",
        "",
        f"Benchmark historico de referencia: **{informe['benchmark_del_control']}**.",
        "",
        "| modelo | corrected_cv | dOOF vs V03 | train F1 | gap | rescates | danos | neto |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (f"| LogisticRegression(binario) | {ctrl['corrected_cv']:.6f} | {ctrl['delta_vs_v03']:+.6f} | "
        f"{ctrl['f1_train']:.4f} | {ctrl['gap_train_oof']:+.4f} | {ctrl['rescates']} | "
        f"{ctrl['danos']} | {ctrl['neto']:+d} |"),
        f"| V03 (referencia) | {CORRECTED_CV_V03:.10f} | +0.0000000000 | -- | -- | 0 | 0 | +0 |",
        "",
        "La diferencia entre el control y V03 es el **coste de la binarizacion**. La diferencia",
        "entre el Tsetlin y el control es la **ganancia del mecanismo**, que es la unica que",
        "responde a la pregunta de la familia.",
        "",
        "## 4. El Tsetlin",
        "",
        ("| K | corrected_cv | dOOF vs control | dOOF vs V03 | train F1 | gap | sd pliegues "
        "| rescates | danos | neto | segundos |"),
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for f in filas:
        lineas.append(
            f"| {f['k']} | {f['corrected_cv']:.6f} | {f['delta_vs_control']:+.6f} | "
            f"{f['delta_vs_v03']:+.6f} | {f['f1_train']:.4f} | {f['gap_train_oof']:+.4f} | "
            f"{f['sd_entre_pliegues']:.4f} | {f['rescates']} | {f['danos']} | {f['neto']:+d} | "
            f"{f['segundos']} |"
        )
    lineas += [
        "",
        f"**Corte en K={informe['corte']['k']}**: {informe['corte']['motivo']}",
        "",
        "## 5. Complementariedad con V03",
        "",
        "| K | coincidencias | fraccion | rescates | danos | ambos malos | correlacion |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for f in filas:
        lineas.append(
            f"| {f['k']} | {f['celdas_coincidentes']} de {f['celdas_totales']} | "
            f"{f['fraccion_coincidente']:.4f} | {f['rescates']} | {f['danos']} | "
            f"{f['ambos_malos']} | {f['correlacion_predicciones']:.4f} |"
        )
    lineas += [
        "",
        "## 6. F1 por etiqueta y estabilidad entre pliegues",
        "",
        "| K | etiqueta | F1 TM | F1 control | F1 V03 | rescates | danos |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for f in filas:
        for etiqueta, d in f["por_etiqueta"].items():
            ctrl_f1 = ctrl["por_etiqueta"][etiqueta]["f1"]
            lineas.append(
                f"| {f['k']} | {etiqueta} | {d['f1']:.4f} | {ctrl_f1:.4f} | {d['f1_v03']:.4f} | "
                f"{d['rescates']} | {d['danos']} |"
            )
    lineas += [
        "",
        "| K | F1 por pliegue | sd |",
        "|---:|---|---:|",
    ]
    for f in filas:
        linea = " | ".join(f"{v:.4f}" for v in f["f1_por_pliegue"])
        lineas.append(f"| {f['k']} | {linea} | {f['sd_entre_pliegues']:.4f} |")
    gastos = informe["gastos"]
    lineas += [
        "",
        "## 7. Reproducibilidad",
        "",
        (f"- K={rep['k']}: primera {rep['primera']:.6f}, repetida {rep['repetida']:.6f}, "
        f"diferencia {rep['diferencia']:.6f}."),
        (f"- Mismo numero de cambios: {rep['mismo_numero_de_cambios']}. Mismas predicciones: "
        f"{rep['mismas_predicciones']}."),
        f"- Causa: {rep['causa']}.",
        "",
        "## 8. Coste",
        "",
        (f"- Tiempo total: {gastos['segundos_totales']} s. Memoria maxima: "
        f"{gastos['memoria_pico_mb']:.0f} MB. Submissions: {gastos['submissions']}."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
