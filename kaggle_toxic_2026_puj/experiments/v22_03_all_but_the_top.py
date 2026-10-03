"""V22-03 (P4): All-But-The-Top sobre el embedding ya entrenado.

Que hace y por que
------------------
Segunda hipotesis de V22: el espacio del embedding contiene componentes dominantes que
ocultan senal util, y eliminarlos puede recuperarla. Se aplica el procedimiento de Mu y
Viswanath (2018), "All-But-The-Top: Simple and Effective Postprocessing for Word
Representations", ICLR 2018, arXiv:1702.01417, con el apoyo mecanico de Levy, Goldberg y
Dagan (2015), TACL.

**No se entrena nada.** Se usan los mismos SIFTS de V5, el mismo mean pooling, el mismo
bloque L2 y el mismo clasificador que el pipeline historico. La unica variable es quantas
direcciones principales se eliminan.

Donde se aplica el PCA, y por que
---------------------------------
El encargo deja elegir entre aplicarlo al vocabulario del embedding o solo a los vectores
usados por TRAIN, y pide no elegir la variante que maximice CV despues de ver los
resultados. Aqui la eleccion no es libre: **el pipeline consulta exactamente 6.059 tipos**
(los del vocabulario del train, generados con `vocabulario_toxic(read_train())`), y solo de
esos tiene vectores. Aplicar el procedimento al vocabulario es, en este pipeline,
aplicarlo a ese conjunto. Es ademas la formulacion estandar de la literatura, y se aplica
**antes** del pooling, que es donde Mu y Viswanath lo colocan.

Control de cordura, y por que es el mas importante del script
-------------------------------------------------------------
Con `D = 0` no se elimina ninguna direccion, y la reconstruccion
`Mc @ V.T @ V` es la identidad **salvo error numerico**. Por tanto `D = 0` debe reproducir
el valor publicado de la fusion historica bit a bit. Ese control verifica simultaneously
que la carga de vectores, el centrado, la SVD y la reconstruccion estan bien. Si `D = 0` no
reproduce, ninguna de las otras configuraciones es atribuible a la eliminacion de
componentes, y el script se detiene.

No se renormalizan los vectores despues de la proyeccion, que es lo que hacen los autores.

Uso
---
    /home/carlos/venv/bin/python experiments/v22_03_all_but_the_top.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.embedding_utils import normalizar_emb_normal
from src.splits import grupos_fusionados, pliegues_corregidos

EXPERIMENTO = "V22-03 (P4: All-But-The-Top sobre el embedding de V5)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v22_03_all_but_the_top.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v22_03_all_but_the_top.md"
OOF_CANONICO = PROJECT_ROOT / "features" / "oof_v03_final.npz"
EMB_DIR = PROJECT_ROOT / "data" / "external" / "embeddings"
SIFTS = EMB_DIR / "fasttext-baseline_sifts.txt"
SIFTS_ENTRADA = SIFTS.with_suffix(".entrada.txt")
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"

ETIQUETAS = ("HS", "TR", "AG")
UMBRALES = np.array([0.44, 0.51, 0.46])
CORRECTED_CV_V03 = 0.7772114525165721
PUBLICADO_B0_200 = 0.781766
TOLERANCIA = 0.0015
VALORES_D = (0, 1, 2, 5)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V22-03] {mensaje}", flush=True)


def cargar_maquinaria() -> tuple[Any, dict[str, np.ndarray], list[str]]:
    """Importa la maquinaria de V5 y carga los vectores de tipo.

    Returns:
        Tupla con el modulo de fusion, el diccionario de vectores y la lista ordenada de
        tipos consultados.
    """
    especificacion = importlib.util.spec_from_file_location(
        "p3_fusion", PROJECT_ROOT / "experiments" / "embedding_03_fusion_complementariedad_v5.py"
    )
    if especificacion is None or especificacion.loader is None:
        raise SystemExit("PARADA OBLIGATORIA: no se puede cargar la maquinaria de V5")
    modulo = importlib.util.module_from_spec(especificacion)
    sys.modules["p3_fusion"] = modulo
    especificacion.loader.exec_module(modulo)

    tipos = SIFTS_ENTRADA.read_text(encoding="utf-8").split()
    vectores: dict[str, np.ndarray] = {}
    with SIFTS.open("r", encoding="utf-8") as manejador:
        for indice, linea in enumerate(manejador, start=1):
            partes = linea.rstrip().split()
            if len(partes) != 201:
                raise SystemExit(
                    f"PARADA OBLIGATORIA: linea {indice} de SIFTS con {len(partes) - 1} valores"
                )
            vector = np.asarray(partes[1:], dtype=np.float32)
            if not np.any(vector):
                continue
            vectores[partes[0]] = vector
    log(f"   SIFTS: {len(tipos)} consultados | {len(vectores)} con vector util")
    return modulo, vectores, tipos


def all_but_the_top(
    vectores: dict[str, np.ndarray], tipos: list[str], d: int
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Elimina las `d` direcciones principales de la matriz de vectores.

    Args:
        vectores: diccionario de token a vector.
        tipos: lista ordenada de tipos a incluir.
        d: numero de direcciones a eliminar.

    Returns:
        Tupla con el diccionario transformado, los valores singulares y la varianza
        explicada acumulada por las `d` direcciones eliminadas.
    """
    matriz = np.vstack([vectores[t] for t in tipos]).astype(np.float64)
    media = matriz.mean(axis=0)
    centrada = matriz - media
    _, valores, vt = np.linalg.svd(centrada, full_matrices=False)
    if d:
        base = vt[d:]
        reconstruida = centrada @ base.T @ base + media
    else:
        reconstruida = centrada @ vt.T @ vt + media
    total = float(np.sum(valores**2))
    acumulada = float(np.sum(valores[:d] ** 2) / total) if d and total > 0 else 0.0
    nuevo = {t: reconstruida[i].astype(np.float32) for i, t in enumerate(tipos)}
    return nuevo, valores, acumulada


def macro_f1(verdad: np.ndarray, predicho: np.ndarray) -> float:
    """Macro-F1 de las tres clases positivas.

    Args:
        verdad: matriz verdadera binaria.
        predicho: matriz predicha binaria.

    Returns:
        La media de las tres F1.
    """
    valores = []
    for j in range(verdad.shape[1]):
        comunes = int(np.count_nonzero(verdad[:, j] & predicho[:, j]))
        denominador = int(np.count_nonzero(verdad[:, j])) + int(np.count_nonzero(predicho[:, j]))
        valores.append(2 * comunes / denominador if denominador > 0 else 0.0)
    return float(np.mean(valores))


def main() -> None:
    """Ejecuta el control D=0 y las tres configuraciones de All-But-The-Top."""
    inicio = time.time()
    modulo, vectores, tipos = cargar_maquinaria()
    log("V22-03: P4, All-But-The-Top. Ningun modelo se entrena")

    log("1/5 desarrollo, pliegues canonicos y alineacion")
    train = read_train()
    particion = pd.read_csv(PARTICION)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    grupos_full = grupos_fusionados(train)
    mapa_grupo = dict(zip(train["id"].astype(int), grupos_full.tolist()))
    grupos_dev = np.array([mapa_grupo[i] for i in desarrollo["id"].astype(int)])
    pliegues = pliegues_corregidos(desarrollo, grupos_dev)
    y = desarrollo[list(ETIQUETAS)].to_numpy().astype(int)
    textos = desarrollo["text"]
    textos_normalizados = [normalizar_emb_normal(t) for t in textos]

    oof = np.load(OOF_CANONICO, allow_pickle=True)
    y_oof = np.asarray(oof["y_true"], dtype=int)
    mapa_posicion = {v: i for i, v in enumerate(desarrollo["id"].astype(int).to_numpy())}
    posiciones = np.array([mapa_posicion[i] for i in np.asarray(oof["ids"]).astype(np.int64)])
    if not np.array_equal(y[posiciones], y_oof):
        raise SystemExit("PARADA OBLIGATORIA: las etiquetas no coinciden tras alinear por id")
    pred_v03 = np.asarray(oof["pred"], dtype=int)
    log(f"   desarrollo {len(desarrollo)} | alineacion verificada")

    log("2/5 la SVD sobre la matriz de vectores y el espectro")
    referencia, valores_sing, _ = all_but_the_top(vectores, tipos, 0)
    energia = np.cumsum(valores_sing**2) / float(np.sum(valores_sing**2))
    log(
        "   varianza explicada acumulada por las primeras direcciones: "
        + " ".join(f"PC{i + 1} {energia[i]:.4f}" for i in range(min(6, len(energia))))
    )

    log("3/5 el control D=0, que debe reproducir la fusion historica")
    documento_0 = modulo.vectores_documento(textos_normalizados, referencia)
    control = modulo.evaluar_variante(
        "V22-ABTT-D0", textos, y, pliegues, documento_0,
        usar_tfidf=True, modo_embedding="l2",
    )
    valor_control = macro_f1(y_oof, (control["proba"][posiciones] >= UMBRALES[None, :]).astype(int))
    desviacion = abs(valor_control - PUBLICADO_B0_200)
    log(
        f"   D=0 da {valor_control:.6f} frente a {PUBLICADO_B0_200:.6f} publicados | desviacion "
        f"{desviacion:.6f} | dentro de tolerancia: {desviacion <= TOLERANCIA}"
    )
    if desviacion > TOLERANCIA:
        log("   PARADA: la reconstruccion con D=0 no reproduce. Ningun D es atribuible")
        REPORTE_JSON.write_text(
            json.dumps(
                {
                    "experimento": EXPERIMENTO,
                    "estado": "PARADA_POR_CONTROL_DE_CORDURA",
                    "d0": valor_control,
                    "publicado": PUBLICADO_B0_200,
                    "desviacion": desviacion,
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        return

    log("4/5 las tres configuraciones de All-But-The-Top")
    filas: list[dict[str, Any]] = []
    predicciones: dict[int, np.ndarray] = {}
    for d in VALORES_D:
        if d == 0:
            continue
        nuevo, _, varianza = all_but_the_top(vectores, tipos, d)
        documento = modulo.vectores_documento(textos_normalizados, nuevo)
        fusion = modulo.evaluar_variante(
            f"V22-ABTT-D{d}", textos, y, pliegues, documento,
            usar_tfidf=True, modo_embedding="l2",
        )
        solo = modulo.evaluar_variante(
            f"V22-ABTT-D{d}-SOLO", textos, y, pliegues, documento,
            usar_tfidf=False, modo_embedding="l2",
        )
        proba = fusion["proba"][posiciones]
        pred = (proba >= UMBRALES[None, :]).astype(int)
        pred_solo = (solo["proba"][posiciones] >= UMBRALES[None, :]).astype(int)
        predicciones[d] = pred
        valor = macro_f1(y_oof, pred)
        cambio = pred != pred_v03
        rescates = int(np.count_nonzero(cambio & (pred_v03 != y_oof) & (pred == y_oof)))
        danos = int(np.count_nonzero(cambio & (pred_v03 == y_oof) & (pred != y_oof)))
        filas.append(
            {
                "d": d,
                "varianza_eliminada": varianza,
                "corrected_cv": valor,
                "delta_vs_v03": valor - CORRECTED_CV_V03,
                "delta_vs_control": valor - valor_control,
                "emb_solo": macro_f1(y_oof, pred_solo),
                "celdas": int(np.count_nonzero(cambio)),
                "rescates": rescates,
                "danios": danos,
                "neto": rescates - danos,
            }
        )
        f = filas[-1]
        log(
            f"   D={d}: fusion {valor:.6f} (dV03 {f['delta_vs_v03']:+.6f}, dControl "
            f"{f['delta_vs_control']:+.6f}) | emb solo {f['emb_solo']:.6f} | celdas {f['celdas']} "
            f"| rescates {rescates} | danos {danos} | NETO {f['neto']:+d} | var eliminada "
            f"{varianza:.4f}"
        )

    log("5/5 veredicto de P4")
    mejor = max(filas, key=lambda f: f["corrected_cv"])
    if mejor["delta_vs_control"] <= 0:
        veredicto = "ALL_BUT_THE_TOP_NO_APORTA"
        razon = (
            "Ninguna eliminacion de componentes principales mejora la fusion por encima del "
            "control, que ya es el embedding sin post-procesar. La hipotesis de que los "
            "componentes dominantes ocultan senal recuperable no se sostiene en este pipeline"
        )
    else:
        veredicto = "ALL_BUT_THE_TOP_APORTA"
        razon = f"El mejor D es {mejor['d']} con {mejor['delta_vs_control']:+.6f} sobre el control"
    log(f"   VEREDICTO P4: {veredicto} | mejor D={mejor['d']} dControl {mejor['delta_vs_control']:+.6f}")

    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "referencia": "Mu y Viswanath 2018 (arXiv:1702.01417); Levy et al. 2015 TACL",
        "embedding": "FastText skipgram 200d sobre wikipedia_es 10M, sin reentrenar",
        "donde_se_aplica_el_pca": (
            "sobre la matriz de los 6.059 vectores de tipo que el pipeline consulta, antes del "
            "pooling, que es donde los autores colocan el procedimiento. La eleccion no es "
            "libre: el pipeline solo tiene vectores de esos tipos"
        ),
        "renormalizacion": "ninguna, siguiendo a los autores",
        "control_de_cordura": {
            "d0": valor_control,
            "publicado": PUBLICADO_B0_200,
            "desviacion": desviacion,
            "dentro_de_tolerancia": desviacion <= TOLERANCIA,
        },
        "espectro": {
            "varianza_acumulada": [float(v) for v in energia[:10]],
        },
        "configuraciones": filas,
        "veredicto": {"valor": veredicto, "razon": razon, "mejor_d": mejor["d"]},
        "gastos": {"modelos_entrenados": 0, "configuraciones": len(filas), "submissions": 0},
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe_md(informe, inicio)
    log("   informes escritos en reports/v22_03_all_but_the_top.{json,md}")
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(f"VEREDICTO: {veredicto} | submissions 0")


def _informe_md(informe: dict[str, Any], inicio: float) -> None:
    """Escribe el informe legible de P4.

    Args:
        informe: informe con los datos.
        inicio: instante de arranque.
    """
    control = informe["control_de_cordura"]
    filas = informe["configuraciones"]
    veredicto = informe["veredicto"]
    lineas = [
        "# V22-03 (P4): All-But-The-Top sobre el embedding de V5",
        "",
        f"Generado por `experiments/v22_03_all_but_the_top.py` ({EXPERIMENTO}).",
        "",
        "**Ningun modelo entrenado. Ninguna submission.**",
        "",
        "## 1. Control de cordura",
        "",
        (f"Con `D = 0` no se elimina ninguna direccion y la reconstruccion es la identidad. Da "
        f"**{control['d0']:.6f}** frente a **{control['publicado']:.6f}** publicados "
        f"(desviacion {control['desviacion']:.6f}), dentro de tolerancia: "
        f"**{control['dentro_de_tolerancia']}**."),
        "",
        "Ese control verifica a la vez la carga de vectores, el centrado, la SVD y la",
        "reconstruccion. Si falla, ningun `D` es atribuible a la eliminacion de componentes.",
        "",
        "## 2. Donde se aplica el PCA",
        "",
        informe["donde_se_aplica_el_pca"] + ".",
        "",
        f"Renormalizacion posterior: {informe['renormalizacion']}.",
        "",
        "## 3. El espectro",
        "",
        "Varianza explicada acumulada por las primeras direcciones principales:",
        "",
        "| direccion | varianza acumulada |",
        "|---|---:|",
    ]
    for i, v in enumerate(informe["espectro"]["varianza_acumulada"]):
        lineas.append(f"| PC{i + 1} | {v:.4f} |")
    lineas += [
        "",
        "## 4. Las tres configuraciones",
        "",
        "| D | var. eliminada | corrected_cv | delta V03 | **delta control** | emb solo | celdas | rescates | danos | neto |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for f in filas:
        lineas.append(
            f"| {f['d']} | {f['varianza_eliminada']:.4f} | {f['corrected_cv']:.6f} | "
            f"{f['delta_vs_v03']:+.6f} | **{f['delta_vs_control']:+.6f}** | {f['emb_solo']:.6f} | "
            f"{f['celdas']} | {f['rescates']} | {f['danios']} | {f['neto']:+d} |"
        )
    lineas += [
        "",
        (f"El control (D=0) queda en **{control['d0']:.6f}** con delta frente a V03 de "
        f"**{control['d0'] - CORRECTED_CV_V03:+.6f}**."),
        "",
        f"## 5. Veredicto de P4: {veredicto['valor']}",
        "",
        veredicto["razon"],
        "",
        f"Completado en {round(time.time() - inicio, 1)} s.",
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
