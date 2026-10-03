"""V21-01: triaje competitivo de las configuraciones de embedding ya entrenadas.

Que hace y por que
------------------
V21 no debe convertirse en otra ronda de experimentacion. Este script **no entrena nada**.
Toma los `.vec` que V5 y V5.1 ya dejaron en disco, reevalua su fusion con V03 reutilizando la
maquinaria original del proyecto, y mide el **perfil de cambios de celdas**: cuantas celdas
cambian, cuantas se rescatan, cuantas se danan, y por etiqueta.

Por que el perfil de celdas y no solo el delta de metrica
---------------------------------------------------------
En este proyecto el saldo de celdas ha predicho la transferencia al leaderboard mejor que el
delta de F1. V16 envio cinco sondas con deltaOOF positivo y las cinco bajaron. V20 tiene
deltaOOF +0.002219 con un neto de -6 y LB -0.00643. Un delta de metrica sin el recuento de
celdas que lo acompana no decide nada sobre transferencia.

Controles, y por que cada uno esta
----------------------------------
**El control de cordura es una fusion, no el champion.** Antes de medir WS-3 y VS-100 se
reevalua `FUSION-B0-200` con el `.vec` baseline de V5, que tiene un valor conocido
(0.7818). Si la maquinaria no reproduce ese numero, las cifras de WS-3 y VS-100 no son
atribuibles a esas configuraciones y el script se detiene. Es el mismo criterio que se uso
en V20, donde la reproduccion exacta de la submission congelada fue el control mas fuerte
disponible.

**La alineacion de filas se hace por identificador, nunca por posicion.** El OOF canonico de
V03 esta en un orden de filas distinto del dataframe de desarrollo, y en V20 una
alineacion por posicion ya habria producido un recuento de celdas sin sentido. Aqui el
mapa es por id, se verifica que sea unico y que las etiquetas coincidan.

**El pooling y la normalizacion son los del proyecto.** Se reutilizan `vectores_documento`
(pooling A, media) y `bloque_embedding` (modo l2) importando el modulo de V5, no
reimplementando la logica. El bloque denso estandarizado quedo demostrado como destructivo
(0.6901 frente a 0.7818) y no se toca.

Uso
---
    /home/carlos/venv/bin/python experiments/v21_01_triaje_competitivo.py
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
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.embedding_utils import normalizar_emb_normal
from src.splits import grupos_fusionados, pliegues_corregidos

EXPERIMENTO = "V21-01 (triaje competitivo de las configuraciones de embedding ya entrenadas)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v21_01_triaje_competitivo.json"
OOF_CANONICO = PROJECT_ROOT / "features" / "oof_v03_final.npz"
EMB_DIR = PROJECT_ROOT / "data" / "external" / "embeddings"

ETIQUETAS = ("HS", "TR", "AG")
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
CORRECTED_CV_V03 = 0.7772114525165721
# (nombre, prefijo de los SIFTS, fichero .vec del que se lee la dimension, valor publicado)
REFERENCIAS = (
    ("FUSION-B0-200", "fasttext-baseline", "fasttext_baseline.vec", 0.7818),
    ("FUSION-WS-3", "fasttext-wiki-10m-ws-3", "fasttext_wiki_10m_ws_3.vec", 0.7830),
    ("FUSION-VS-100", "fasttext-wiki-10m-vs-100", "fasttext_wiki_10m_vs_100.vec", 0.7823),
)
TOLERANCIA = 0.0015
CANDIDATOS = ("FUSION-WS-3", "FUSION-VS-100", "L1 C=4.0")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V21-01] {mensaje}", flush=True)


def _cargar_modulo(nombre: str, fichero: str) -> Any:
    """Importa un modulo de V5 por ruta de fichero.

    Args:
        nombre: clave con la que se registra en ``sys.modules``.
        fichero: nombre del script dentro de ``experiments``.

    Returns:
        El modulo cargado.
    """
    ruta = PROJECT_ROOT / "experiments" / fichero
    especificacion = importlib.util.spec_from_file_location(nombre, ruta)
    if especificacion is None or especificacion.loader is None:
        raise SystemExit(f"PARADA OBLIGATORIA: no se puede cargar {fichero}")
    modulo = importlib.util.module_from_spec(especificacion)
    sys.modules[nombre] = modulo
    especificacion.loader.exec_module(modulo)
    return modulo


def cargar_maquinaria() -> Any:
    """Importa el modulo de V5 con la maquinaria de fusion.

    Returns:
        El modulo cargado.
    """
    return _cargar_modulo(
        "p3_fusion", "embedding_03_fusion_complementariedad_v5.py"
    )


def cargar_sifts(
    prefijo: str, tipos_consulta: list[str], dimension_esperada: int
) -> dict[str, np.ndarray]:
    """Carga los vectores de tipo desde los ficheros SIFTS, como hace V5.1.

    **Por que los SIFTS y no el `.vec`.** V5.1 no lee el `.vec` para obtener vectores: usa
    `vectores_de_tipos`, que invoca al binario de fastText en modo `print-word-vectors` sobre
    el `.bin` y cachea la salida. Eso importa mas de lo que parece. El `.vec` solo exporta
    los 100.098 tipos del vocabulario del modelo, mientras que el `.bin` **compone vector con
    subpalabras para cualquier palabra**, incluidas las que no estan en el vocabulario. De
    los 6.059 tipos que V5.1 consulta, solo 3.777 aparecen en el `.vec`; los 2.282 restantes
    si tienen vector en el modelo. Al construir los documentos con el `.vec` se obtiene un
    pooling distinto, y FUSION-B0-200 daba 0.7728 en vez de los 0.7818 publicados, con lo que
    el control de cordura detuvo la ejecucion. Con los SIFTS se reproduce la construccion de
    V5.1 exactamente.

    Se verifican dos cosas antes de usar los vectores: que la entrada de los SIFTS coincide
    exactamente con los tipos consultados, y que cada linea aporta `dimension` valores. Los
    vectores nulos se descartan, que es lo que hace `vectores_de_tipos`.

    Args:
        prefijo: nombre base de los ficheros, por ejemplo ``fasttext-baseline``.
        tipos_consulta: lista ordenada de tipos que V5.1 consulta.
        dimension_esperada: dimension que declara la cabecera del `.vec`.

    Returns:
        Diccionario de token a vector, sin vectores nulos.
    """
    entrada = EMB_DIR / f"{prefijo}_sifts.entrada.txt"
    salida = EMB_DIR / f"{prefijo}_sifts.txt"
    for ruta in (entrada, salida):
        if not ruta.exists():
            raise SystemExit(
                f"PARADA OBLIGATORIA: falta el artefacto {ruta.name}, que es el que V5.1 uso "
                f"para construir los vectores de tipo. Con el `.vec` el resultado no reproduce"
            )
    tipos_entrada = entrada.read_text(encoding="utf-8").split()
    if tipos_entrada != tipos_consulta:
        raise SystemExit(
            f"PARADA OBLIGATORIA: la entrada de {prefijo} no coincide con los tipos consultados "
            f"({len(tipos_entrada)} frente a {len(tipos_consulta)}). Si el conjunto de tipos es "
            f"otro, el pooling no es el de V5"
        )
    vectores: dict[str, np.ndarray] = {}
    nulos = 0
    with salida.open("r", encoding="utf-8") as manejador:
        for indice, linea in enumerate(manejador, start=1):
            partes = linea.rstrip().split()
            if len(partes) != dimension_esperada + 1:
                raise SystemExit(
                    f"PARADA OBLIGATORIA: la linea {indice} de {salida.name} tiene "
                    f"{len(partes) - 1} valores y se esperaban {dimension_esperada}"
                )
            vector = np.asarray(partes[1:], dtype=np.float32)
            if not np.any(vector):
                nulos += 1
                continue
            vectores[partes[0]] = vector
    log(
        f"   {prefijo}_sifts.txt: {len(vectores)} vectores utiles de {len(tipos_consulta)} "
        f"consultados | {nulos} nulos descartados | dimension {dimension_esperada}"
    )
    if not vectores:
        raise SystemExit(f"PARADA OBLIGATORIA: {prefijo} no devolvio ningun vector utilizable")
    return vectores


def dimension_de_vec(fichero: str) -> int:
    """Lee la dimension declarada en la cabecera de un `.vec` de fastText.

    Args:
        fichero: nombre del `.vec`.

    Returns:
        La dimension.
    """
    with (EMB_DIR / fichero).open("r", encoding="utf-8") as archivo:
        _, dimension = map(int, archivo.readline().split())
    return dimension


def fusion_oof(
    modulo: Any,
    textos: pd.Series,
    y: np.ndarray,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    documento: np.ndarray,
    nombre: str,
) -> np.ndarray:
    """Probabilidades OOF de la fusion dispersa + densa.

    No reimplementa la evaluacion: **delega en `evaluar_variante` de V5**, que es la
    funcion que produjo los valores publicados. Eso importa por dos motivos. Uno, V5 une el
    bloque denso con `sparse.hstack` sobre un `csr_matrix`; una reimplementacion que use
    `np.hstack(...).toarray()` entrega a liblinear una matriz densa de 27.431 columnas, y
    con el vectorizador de L2 eso cambia el camino de optimizacion y el resultado. Con la
    fusion B0-200 esa diferencia valio 0.009 de corrected_cv, suficiente para que el control
    de cordura detuviese la ejecucion. Dos, delegar garantiza que pooling, normalizacion y
    metricas son las del proyecto y no una copia que puede divergir sin avisar.

    Args:
        modulo: modulo de V5 con la maquinaria.
        textos: columna de textos del desarrollo.
        y: matriz verdadera.
        pliegues: pliegues corregidos.
        documento: matriz de vectores de documento alineada con ``textos``.
        nombre: identificador de la variante, solo para el registro interno del modulo.

    Returns:
        Probabilidades OOF con la forma de ``y``.
    """
    return modulo.evaluar_variante(
        nombre,
        textos,
        y,
        pliegues,
        documento,
        usar_tfidf=True,
        modo_embedding="l2",
    )["proba"]


def l1_oof(
    textos: np.ndarray,
    y: np.ndarray,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    modulo: Any,
) -> np.ndarray:
    """Probabilidades OOF de LogisticRegression con L1 y C=4.0.

    Args:
        textos: columna de textos del desarrollo.
        y: matriz verdadera.
        pliegues: pares de indices.
        modulo: modulo de V5 con la maquinaria de matrices.

    Returns:
        Probabilidades OOF.
    """
    proba = np.zeros_like(y, dtype=float)
    for ajuste, validacion in pliegues:
        disperso_ajuste, disperso_validacion = modulo.matrices_tfidf(
            textos.iloc[ajuste], textos.iloc[validacion]
        )
        for j in range(y.shape[1]):
            modelo = LogisticRegression(
                C=4.0,
                penalty="l1",
                class_weight="balanced",
                solver="liblinear",
                max_iter=2000,
                random_state=42,
            )
            modelo.fit(disperso_ajuste, y[ajuste, j])
            proba[validacion, j] = modelo.predict_proba(disperso_validacion)[:, 1]
    return proba


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


def perfil(
    nombre: str, pred: np.ndarray, pred_v03: np.ndarray, y: np.ndarray
) -> dict[str, Any]:
    """Mide el perfil de cambios de una candidata respecto a V03.

    Args:
        nombre: nombre de la candidata.
        pred: prediccion de la candidata.
        pred_v03: prediccion del champion.
        y: matriz verdadera.

    Returns:
        Diccionario con el delta de metrica y el recuento de celdas.
    """
    cambio = pred != pred_v03
    rescates = int(np.count_nonzero(cambio & (pred_v03 != y) & (pred == y)))
    danos = int(np.count_nonzero(cambio & (pred_v03 == y) & (pred != y)))
    por_etiqueta = {}
    for j, etiqueta in enumerate(ETIQUETAS):
        c = cambio[:, j]
        r = int(np.count_nonzero(c & (pred_v03[:, j] != y[:, j]) & (pred[:, j] == y[:, j])))
        d = int(np.count_nonzero(c & (pred_v03[:, j] == y[:, j]) & (pred[:, j] != y[:, j])))
        por_etiqueta[etiqueta] = {
            "cambios": int(np.count_nonzero(c)),
            "rescates": r,
            "danios": d,
            "neto": r - d,
            "tasa_positiva_v03": float(pred_v03[:, j].mean()),
            "tasa_positiva_candidata": float(pred[:, j].mean()),
        }
    valor = macro_f1(y, pred)
    filas = int(np.count_nonzero(cambio.any(axis=1)))
    return {
        "nombre": nombre,
        "corrected_cv": valor,
        "delta_f1": valor - CORRECTED_CV_V03,
        "celdas": int(np.count_nonzero(cambio)),
        "rescates": rescates,
        "danios": danos,
        "neto": rescates - danos,
        "filas_afectadas": filas,
        "por_etiqueta": por_etiqueta,
        "mascara_cambio": cambio,
    }


def main() -> None:
    """Reevalua las tres configuraciones y mide su perfil de cambios."""
    inicio = time.time()
    modulo = cargar_maquinaria()
    modulo_p1 = _cargar_modulo("p1_corpus", "embedding_01_corpus_wikipedia_v5.py")
    log("V21-01: triaje competitivo. Ningun modelo se entrena en este script")

    log("1/7 desarrollo y mapa de alineacion por identificador")
    train = read_train()
    particion = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv")
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    grupos_full = grupos_fusionados(train)
    mapa_grupo = dict(zip(train["id"].astype(int), grupos_full.tolist()))
    grupos_dev = np.array([mapa_grupo[i] for i in desarrollo["id"].astype(int)])
    pliegues = pliegues_corregidos(desarrollo, grupos_dev)
    y = desarrollo[list(ETIQUETAS)].to_numpy().astype(int)
    textos_crudos = desarrollo["text"]
    textos = desarrollo["text"].astype(str).to_numpy()

    oof = np.load(OOF_CANONICO, allow_pickle=True)
    ids_oof = np.asarray(oof["ids"]).astype(np.int64)
    proba_v03 = np.asarray(oof["proba"], dtype=float)
    y_oof = np.asarray(oof["y_true"], dtype=int)
    mapa_posicion = {v: i for i, v in enumerate(desarrollo["id"].astype(int).to_numpy())}
    posiciones = np.array([mapa_posicion[i] for i in ids_oof])
    if len(set(posiciones.tolist())) != len(posiciones):
        raise SystemExit("PARADA OBLIGATORIA: el mapa de alineacion por id no es unico")
    if not np.array_equal(y[posiciones], y_oof):
        raise SystemExit("PARADA OBLIGATORIA: las etiquetas no coinciden tras alinear por id")
    log(f"   desarrollo {len(desarrollo)} filas | alineacion 1 a 1 verificada | {len(pliegues)} pliegues")
    pred_v03 = (proba_v03 >= UMBRALES_V03[None, :]).astype(int)

    log("2/7 tipos consultados y normalizacion, identicos a los de V5.1")
    # V5.1 construye `tipos` como `sorted(vocabulario_toxic(read_train()))`, es decir los
    # 6.059 tipos del vocabulario del TRAIN, y consulta al modelo solo esos. Es un detalle
    # que cambia el pooling y con el cambia el resultado.
    tipos_consulta = sorted(modulo_p1.vocabulario_toxic(read_train()))
    log(f"   tipos consultados (vocabulario del train): {len(tipos_consulta)}")
    textos_normalizados = [normalizar_emb_normal(texto) for texto in textos]

    log("3/7 reevaluacion de las tres configuraciones de embedding")
    probabilidades: dict[str, np.ndarray] = {}
    controles: list[dict[str, Any]] = []
    for nombre, prefijo, fichero, esperado in REFERENCIAS:
        vectores = cargar_sifts(prefijo, tipos_consulta, dimension_de_vec(fichero))
        documento = modulo.vectores_documento(textos_normalizados, vectores)
        nulas = int(np.count_nonzero(np.linalg.norm(documento, axis=1) == 0))
        log(f"   {nombre}: documento {documento.shape} | filas con norma 0: {nulas}")
        proba = fusion_oof(modulo, textos_crudos, y, pliegues, documento, nombre)[posiciones]
        pred = (proba >= UMBRALES_V03[None, :]).astype(int)
        valor = macro_f1(y_oof, pred)
        desviacion = abs(valor - esperado)
        coincide = desviacion <= TOLERANCIA
        controles.append(
            {
                "configuracion": nombre,
                "corrected_cv": valor,
                "publicado": esperado,
                "desviacion": desviacion,
                "dentro_de_tolerancia": coincide,
            }
        )
        log(
            f"   {nombre}: corrected_cv {valor:.6f} | publicado {esperado:.4f} | "
            f"desviacion {desviacion:.6f} | dentro de tolerancia: {coincide}"
        )
        if not coincide:
            log("   PARADA: la maquinaria no reproduce el valor publicado")
            REPORTE_JSON.write_text(
                json.dumps(
                    {
                        "experimento": EXPERIMENTO,
                        "estado": "PARADA_POR_CONTROL_DE_CORDURA",
                        "controles": controles,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            return
        probabilidades[nombre] = proba

    log("4/7 L1 C=4.0 recalculado en el mismo marco, para comparar y solapar")
    proba_l1 = l1_oof(textos_crudos, y, pliegues, modulo)
    pred_l1 = (proba_l1[posiciones] >= UMBRALES_V03[None, :]).astype(int)
    log(f"   L1 C=4.0: corrected_cv {macro_f1(y_oof, pred_l1):.6f}")

    log("5/7 perfil de cambios de cada candidata respecto a V03")
    predicciones = {
        "FUSION-WS-3": (probabilidades["FUSION-WS-3"] >= UMBRALES_V03[None, :]).astype(int),
        "FUSION-VS-100": (probabilidades["FUSION-VS-100"] >= UMBRALES_V03[None, :]).astype(int),
        "L1 C=4.0": pred_l1,
    }
    perfiles: dict[str, dict[str, Any]] = {}
    for nombre in CANDIDATOS:
        p = perfil(nombre, predicciones[nombre], pred_v03, y_oof)
        perfiles[nombre] = p
        log(
            f"   {nombre:<14} dF1 {p['delta_f1']:+.6f} | celdas {p['celdas']:>4} | rescates "
            f"{p['rescates']:>4} | danos {p['danios']:>4} | NETO {p['neto']:>+4} | filas "
            f"{p['filas_afectadas']:>4}"
        )

    log("6/7 solapamiento de los cambios entre candidatas")
    solapes: dict[str, Any] = {}
    for a, b in (("FUSION-WS-3", "FUSION-VS-100"), ("FUSION-WS-3", "L1 C=4.0"),
                 ("FUSION-VS-100", "L1 C=4.0")):
        ma = perfiles[a]["mascara_cambio"]
        mb = perfiles[b]["mascara_cambio"]
        comunes = int(np.count_nonzero(ma & mb))
        unicas_a = int(np.count_nonzero(ma & ~mb))
        unicas_b = int(np.count_nonzero(mb & ~ma))
        proporcion = comunes / min(int(np.count_nonzero(ma)), int(np.count_nonzero(mb)))
        solapes[f"{a} vs {b}"] = {
            "celdas_comunes": comunes,
            "solo_primera": unicas_a,
            "solo_segunda": unicas_b,
            "proporcion_de_solape": proporcion,
        }
        log(
            f"   {a} vs {b}: {comunes} comunes ({proporcion:.1%} de la menor) | "
            f"solo {unicas_a} | solo {unicas_b}"
        )

    log("7/7 informe")
    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "champion": {"version": "V03", "corrected_cv": CORRECTED_CV_V03},
        "protocolo": {
            "pliegues": "src.splits.pliegues_corregidos, StratifiedGroupKFold(5, shuffle, 42) "
            "sobre grupos de near-duplicado",
            "filas": len(desarrollo),
            "alineacion": "por identificador contra features/oof_v03_final.npz",
            "pooling": "A (media), reutilizado de V5",
            "bloque_denso": "L2 por documento",
            "clasificador": "PARAMETROS_LR de V5 (C=2.0, balanced, liblinear)",
            "umbrales": "congelados de V03",
        },
        "controles_de_cordura": controles,
        "perfiles": {
            nombre: {k: v for k, v in p.items() if k != "mascara_cambio"}
            for nombre, p in perfiles.items()
        },
        "solapes": solapes,
        "gastos": {"modelos_entrenados": 0, "submissions": 0},
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log(f"   informe en {REPORTE_JSON.relative_to(PROJECT_ROOT)}")
    log(f"completado en {round(time.time() - inicio, 1)} s")
    for nombre in CANDIDATOS:
        p = perfiles[nombre]
        log(
            f"VEREDICTO {nombre}: dF1 {p['delta_f1']:+.6f} | neto {p['neto']:+d} de "
            f"{p['celdas']} celdas"
        )


if __name__ == "__main__":
    main()
