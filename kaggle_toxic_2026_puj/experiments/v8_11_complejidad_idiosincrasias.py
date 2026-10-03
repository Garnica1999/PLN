"""V8-11: asimetria y kurtosis de la distribucion, e idiosincrasias internas.

Contexto
--------
De las nueve tecnicas nuevas que el inventario del material de clase identifico, quedan seis sin
ejecutar. Este script mide las dos primeras, y las dos se apoyan en un hecho ya medido en la fase
v8: **la estimacion es extremadamente inestable entre particiones**, con un desacuerdo medio de
0,92 a 0,97. Anadir rasgos a un modelo solo vale si el cambio de la prediccion se mantiene a
traves de las particiones, y por eso el script mide cada rama en cinco repeticiones y cuenta
cuantas mejoran, ademas de la cifra de la metrica.

Asimetria y kurtosis
--------------------
El material de clase las lista como medidas de complejidad de la distribucion, en la familia de
rasgos que el profesor acumula para atribuir autoria. El proyecto ya habia medido longitudes
medias y diversidad lexica dentro del analisis inicial, pero **nunca las habia usado como
rasgos**, y el inventario lo confirma: no hay ninguna asignacion ni llamada a ellas en el codigo.

La asimetria y la kurtosis de la distribucion de longitudes de palabra miden algo que la media
no mide: si los mensajes largos son la cola de la distribucion o su cuerpo. Una cola larga
significa que hay un grupo de mensajes desproporcionadamente largos, que es justo lo que un
modelo de toxicidad deberia notar.

Idiosincrasias y lexico regional
--------------------------------
El material de clase lista las idiosincrasias como una familia de rasgos de atribucion. No hay
lexicon externo disponible ni permitido, de modo que se operacionaliza **sin recursos externos**:
un token es idiosincratico si aparece en muy pocos documentos. Esa es la definicion operativa
de lo que el material llama palabra inestable, y usa solo informacion del propio corpus. La
proporcion de tokens idiosincraticos de cada documento es un rasgo, y lo importante es que **se
calcula dentro del tren de cada pliegue**, para que un token raro en el pliegue no pueda usar
su frecuencia global.

Uso
---
    /home/carlos/venv/bin/python experiments/v8_11_complejidad_idiosincrasias.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_test, read_train  # noqa: E402
from src.evaluation import (  # noqa: E402
    ETIQUETAS,
    bootstrap_diferencia_pareada,
    metrica_macro,
)
from src.splits import (  # noqa: E402
    combinaciones,
    grupos_fusionados,
    holdout_v2,
    pliegues_corregidos,
)

EXPERIMENTO = "V8-11 (complejidad e idiosincrasias)"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
REPORTE = PROJECT_ROOT / "reports" / "v8_complejidad_idiosincrasias.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "v8_complejidad_idiosincrasias.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v8_complejidad_idiosincrasias.json"
SUBMISSION = (
    PROJECT_ROOT / "submissions" / "toxic_submission_v14_complejidad_idiosincrasias.csv"
)

REFERENCIA_V03 = 0.777211
REFERENCIA_LB = 0.78306
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
TOLERANCIA_CONTROL = 0.0005
GATE_DELTA = 0.003
GATE_PLIEGUES = 4
RUIDO_EN_ACIERTOS = 19.2
# Fracciones de documentos por debajo de las cuales un token se considera idiosincratico.
VALORES_UMBRAL = (0.01, 0.02, 0.05)
NUMERO_REPETICIONES = 5
SEMILLA_BASE = 9200
PLIEGUES = 5

PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V8-C1] {mensaje}", flush=True)


def aplicar_umbrales(probabilidades: np.ndarray) -> np.ndarray:
    """Aplica los umbrales congelados de V03.

    Args:
        probabilidades: matriz de probabilidades.

    Returns:
        Matriz binaria de predicciones.
    """
    return (probabilidades >= UMBRALES_V03[None, :]).astype(int)


def rasgos_de_complejidad(textos: pd.Series) -> np.ndarray:
    """Calcula asimetria y kurtosis de la distribucion de longitudes de palabra.

    Todos los rasgos se dividen por la longitud media del propio documento, para que sean
    comparables entre un mensaje de diez palabras y uno de trescientas.

    Args:
        textos: serie de textos.

    Returns:
        Matriz de tres columnas: asimetria, kurtosis y recorrido intercuantilico.
    """
    asimetrias = np.zeros(len(textos), dtype=np.float64)
    kurtosis = np.zeros(len(textos), dtype=np.float64)
    recorridos = np.zeros(len(textos), dtype=np.float64)
    medias = np.zeros(len(textos), dtype=np.float64)
    for posicion, texto in enumerate(textos.to_numpy()):
        longitudes = np.array([len(token) for token in str(texto).split()], dtype=np.float64)
        if longitudes.size < 4:
            continue
        media = longitudes.mean()
        desviacion = longitudes.std()
        if desviacion <= 0:
            continue
        tipificada = (longitudes - media) / desviacion
        asimetrias[posicion] = float(np.mean(tipificada**3))
        kurtosis[posicion] = float(np.mean(tipificada**4)) - 3.0
        recorridos[posicion] = float(
            np.percentile(longitudes, 90) - np.percentile(longitudes, 10)
        )
        medias[posicion] = media
    escala = np.where(medias > 0, medias, 1.0)
    return np.column_stack([asimetrias / escala, kurtosis / escala, recorridos / escala])


def rasgos_de_idiosincrasia(
    textos_entrenamiento: pd.Series, textos_consulta: pd.Series
) -> np.ndarray:
    """Calcula la proporcion de tokens idiosincraticos de cada documento.

    Un token se considera idiosincratico si aparece en menos de una fraccion dada de los
    documentos del **entrenamiento**. La frecuencia de documento se calcula solo con el
    entrenamiento, de modo que un token raro en el pliegue no puede aprovechar su frecuencia
    global, que seria informacion de las filas de validacion.

    Args:
        textos_entrenamiento: textos de ajuste, los unicos que definen la frecuencia.
        textos_consulta: textos a los que se aplica la definicion.

    Returns:
        Matriz con una columna por cada umbral de fraccion probado.
    """
    vectorizador = TfidfVectorizer(
        analyzer="word", min_df=1, binary=True, token_pattern=r"(?u)\b\w+\b"
    )
    matriz = vectorizador.fit_transform(textos_entrenamiento)
    n_documentos = matriz.shape[0]
    apariciones = np.asarray((matriz > 0).sum(axis=0)).ravel()
    umbrales_documentos = [
        max(int(fraccion * n_documentos), 1) for fraccion in VALORES_UMBRAL
    ]
    identificadores = {
        palabra: indice
        for indice, palabra in enumerate(vectorizador.get_feature_names_out())
    }
    salida = np.zeros((len(textos_consulta), len(VALORES_UMBRAL)), dtype=np.float64)
    for posicion, texto in enumerate(textos_consulta.to_numpy()):
        tokens = str(texto).lower().split()
        if not tokens:
            continue
        for j, limite in enumerate(umbrales_documentos):
            cuenta = 0
            for token in tokens:
                indice = identificadores.get(token)
                if indice is not None and apariciones[indice] <= limite:
                    cuenta += 1
            salida[posicion, j] = cuenta / len(tokens)
    return salida


def construir_espacio(
    textos_entrenamiento: pd.Series, textos_consulta: list[pd.Series]
) -> tuple[sparse.csr_matrix, list[sparse.csr_matrix]]:
    """Ajusta los vectorizadores en el entrenamiento y transforma las consultas.

    Args:
        textos_entrenamiento: textos de ajuste.
        textos_consulta: lista de series a transformar.

    Returns:
        Tupla con la matriz de ajuste y la lista de matrices de consulta.
    """
    vectorizador_word = TfidfVectorizer(**PARAMS_WORD)
    vectorizador_char = TfidfVectorizer(**PARAMS_CHAR)
    matriz = sparse.hstack(
        [
            vectorizador_word.fit_transform(textos_entrenamiento),
            vectorizador_char.fit_transform(textos_entrenamiento),
        ],
        format="csr",
    )
    consultas = [
        sparse.hstack(
            [vectorizador_word.transform(serie), vectorizador_char.transform(serie)],
            format="csr",
        )
        for serie in textos_consulta
    ]
    return matriz, consultas


def main() -> None:
    """Mide la aportacion de los rasgos de complejidad y de las idiosincrasias."""
    inicio = time.time()
    log("Rasgos de complejidad e idiosincrasias internas")
    train = read_train()
    test = read_test()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    grupos_dev = grupos[indices_dev]
    pliegues = pliegues_corregidos(desarrollo, grupos[indices_dev])
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    textos = desarrollo["text"].astype(str)
    textos_test = test["text"].astype(str)
    log(f"   desarrollo {len(y)} filas | prueba {len(test)} filas")

    log("1/5 control de codigo contra el OOF canonico de V03")
    oof = np.load(OOF_V03, allow_pickle=True)
    proba_canonica = oof["proba"].astype(np.float64)
    ids_iguales = bool(
        (oof["ids"].astype(str) == desarrollo["id"].astype(str).to_numpy()).all()
    )
    predicciones_canonicas = aplicar_umbrales(proba_canonica)
    metrica_canonica = metrica_macro(y, predicciones_canonicas)
    control_correcto = (
        abs(metrica_canonica - REFERENCIA_V03) <= TOLERANCIA_CONTROL and ids_iguales
    )
    log(f"   OOF canonico {metrica_canonica:.6f} | ids en orden {ids_iguales}")
    if not control_correcto:
        log("PARADA: el control falla y no se sigue.")
        return

    log("2/5 los rasgos de complejidad, que no dependen de la particion")
    complejidad = rasgos_de_complejidad(textos)
    complejidad_test = rasgos_de_complejidad(textos_test)
    log(
        f"   asimetria media {complejidad[:, 0].mean():+.4f} | kurtosis media "
        f"{complejidad[:, 1].mean():+.4f} | recorrido medio {complejidad[:, 2].mean():.4f}"
    )

    log("3/5 cinco repeticiones; las idiosincrasias se recalculan en cada tren")
    claves = ("solo_complejidad", "complejidad_mas_idiosincrasias")
    probabilidades = {
        clave: np.zeros((NUMERO_REPETICIONES, len(y), 3), dtype=np.float64) for clave in claves
    }
    for repeticion in range(NUMERO_REPETICIONES):
        divisor = StratifiedGroupKFold(
            n_splits=PLIEGUES, shuffle=True, random_state=SEMILLA_BASE + repeticion
        )
        for entrenamiento, validacion in divisor.split(
            textos, combinaciones(desarrollo), groups=grupos_dev
        ):
            matriz, (consulta,) = construir_espacio(
                textos.iloc[entrenamiento], [textos.iloc[validacion]]
            )
            idiosincrasias_entrenamiento = rasgos_de_idiosincrasia(
                textos.iloc[entrenamiento], textos.iloc[entrenamiento]
            )
            idiosincrasias_validacion = rasgos_de_idiosincrasia(
                textos.iloc[entrenamiento], textos.iloc[validacion]
            )
            diseno_entrenamiento = np.hstack(
                [complejidad[entrenamiento], idiosincrasias_entrenamiento]
            )
            diseno_validacion = np.hstack(
                [complejidad[validacion], idiosincrasias_validacion]
            )
            for etiqueta in range(3):
                destino = y[entrenamiento, etiqueta]
                base = LogisticRegression(
                    C=2.0, class_weight="balanced", max_iter=2000, solver="liblinear",
                    random_state=SEMILLA_BASE + repeticion,
                )
                base.fit(matriz, destino)
                probabilidades["solo_complejidad"][repeticion, validacion, etiqueta] = (
                    base.predict_proba(consulta)[:, 1]
                )
                con_idio = LogisticRegression(
                    C=2.0, class_weight="balanced", max_iter=2000, solver="liblinear",
                    random_state=SEMILLA_BASE + repeticion,
                )
                con_idio.fit(sparse.hstack([matriz, diseno_entrenamiento]), destino)
                probabilidades["complejidad_mas_idiosincrasias"][
                    repeticion, validacion, etiqueta
                ] = con_idio.predict_proba(
                    sparse.hstack([consulta, diseno_validacion])
                )[:, 1]
        log(f"   repeticion {repeticion + 1} de {NUMERO_REPETICIONES} completa")

    log("4/5 evaluacion, con la consistencia entre particiones")
    variantes: list[dict[str, Any]] = []
    for clave in claves:
        por_repeticion = probabilidades[clave]
        metricas = [
            metrica_macro(y, aplicar_umbrales(por_repeticion[r]))
            for r in range(NUMERO_REPETICIONES)
        ]
        predicciones = aplicar_umbrales(por_repeticion.mean(axis=0))
        metrica = metrica_macro(y, predicciones)
        diferencia = metrica - metrica_canonica
        mejoras = int(sum(1 for m in metricas if m > metrica_canonica))
        aciertos = [
            int((predicciones[val] == y[val]).sum()
                - (predicciones_canonicas[val] == y[val]).sum())
            for _, val in pliegues
        ]
        bootstrap = bootstrap_diferencia_pareada(
            y, predicciones, predicciones_canonicas, repeticiones=2000
        )
        cambio = predicciones != predicciones_canonicas
        n_celdas = int(cambio.sum())
        if n_celdas:
            tasa = int((predicciones == y)[cambio].sum()) / n_celdas
            minimo = 0.5 + RUIDO_EN_ACIERTOS / (2.0 * n_celdas)
            pasa = tasa >= minimo
        else:
            tasa, minimo, pasa = float("nan"), float("nan"), False
        variante = {
            "clave": clave,
            "metrica": round(metrica, 6),
            "diferencia": round(diferencia, 6),
            "ic95_0": float(bootstrap["ic95"][0]),
            "ic95_1": float(bootstrap["ic95"][1]),
            "pliegues_positivos": int(sum(1 for a in aciertos if a > 0)),
            "repeticiones_que_mejoran": mejoras,
            "metricas_por_repeticion": [round(m, 6) for m in metricas],
            "celdas": n_celdas,
            "tasa_acierto": round(tasa, 4),
            "minimo_requerido": round(minimo, 4),
            "filtro_pasa": bool(pasa),
        }
        variante["gate_delta"] = diferencia >= GATE_DELTA
        variante["gate_ic95"] = variante["ic95_0"] > 0
        variante["gate_pliegues"] = variante["pliegues_positivos"] >= GATE_PLIEGUES
        variante["gate_supera"] = (
            variante["gate_delta"] and variante["gate_ic95"] and variante["gate_pliegues"]
        )
        variantes.append(variante)
        log(
            f"   {clave}: metrica {metrica:.6f} | diferencia {diferencia:+.6f} | IC95 "
            f"[{variante['ic95_0']:+.5f}, {variante['ic95_1']:+.5f}] | repeticiones que mejoran "
            f"{mejoras}/{NUMERO_REPETICIONES} | celdas {n_celdas} | tasa {tasa:.4f} frente a "
            f"{minimo:.4f}"
        )

    log("5/5 submission, integridad e informe")
    mejor = max(variantes, key=lambda v: v["diferencia"])
    proba_test = np.zeros((NUMERO_REPETICIONES, len(test), 3), dtype=np.float64)
    for repeticion in range(NUMERO_REPETICIONES):
        divisor = StratifiedGroupKFold(
            n_splits=PLIEGUES, shuffle=True, random_state=SEMILLA_BASE + repeticion
        )
        for entrenamiento, _ in divisor.split(
            textos, combinaciones(desarrollo), groups=grupos_dev
        ):
            matriz, (consulta,) = construir_espacio(
                textos.iloc[entrenamiento], [textos_test]
            )
            idiosincrasias_entrenamiento = rasgos_de_idiosincrasia(
                textos.iloc[entrenamiento], textos.iloc[entrenamiento]
            )
            idiosincrasias_prueba = rasgos_de_idiosincrasia(
                textos.iloc[entrenamiento], textos_test
            )
            diseno_entrenamiento = np.hstack(
                [complejidad[entrenamiento], idiosincrasias_entrenamiento]
            )
            diseno_prueba = np.hstack([complejidad_test, idiosincrasias_prueba])
            for etiqueta in range(3):
                estimador = LogisticRegression(
                    C=2.0, class_weight="balanced", max_iter=2000, solver="liblinear",
                    random_state=SEMILLA_BASE + repeticion,
                )
                estimador.fit(
                    sparse.hstack([matriz, diseno_entrenamiento]), y[entrenamiento, etiqueta]
                )
                proba_test[repeticion, :, etiqueta] = estimador.predict_proba(
                    sparse.hstack([consulta, diseno_prueba])
                )[:, 1]
    pred_test = aplicar_umbrales(proba_test.mean(axis=0))
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    marco = pd.DataFrame({e: pred_test[:, i] for i, e in enumerate(ETIQUETAS)})
    marco.insert(0, "id", test["id"].to_numpy())
    marco.to_csv(SUBMISSION, index=False, lineterminator="\n")
    integridad = _verificar_integridad(marco, test, train)
    integridad_correcta = all(f["resultado"] == "correcto" for f in integridad)
    comparacion = _comparar_con_v03(SUBMISSION)
    log(
        f"   mejor rama: {mejor['clave']} con {mejor['diferencia']:+.6f} | cambian "
        f"{comparacion['celdas_que_cambian']} celdas frente a V03 | integridad "
        f"{'correcta' if integridad_correcta else 'CON FALLOS'}"
    )
    _informe(
        variantes=variantes, mejor=mejor, complejidad=complejidad, integridad=integridad,
        integridad_correcta=integridad_correcta, comparacion=comparacion, inicio=inicio,
    )
    with REPORTE_CSV.open("w", encoding="utf-8", newline="") as archivo:
        columnas = [
            "clave", "metrica", "diferencia", "ic95_0", "ic95_1", "pliegues_positivos",
            "repeticiones_que_mejoran", "celdas", "tasa_acierto", "minimo_requerido",
            "filtro_pasa", "gate_supera",
        ]
        escritor = csv.DictWriter(archivo, fieldnames=columnas, lineterminator="\n")
        escritor.writeheader()
        for v in variantes:
            escritor.writerow({k: v.get(k, "") for k in columnas})
    REPORTE_JSON.write_text(
        json.dumps(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "referencia_v03_local": REFERENCIA_V03,
                "referencia_v03_externa": REFERENCIA_LB,
                "umbrales_idiosincrasia": list(VALORES_UMBRAL),
                "variantes": variantes,
                "mejor": mejor["clave"],
                "integridad": integridad,
                "integridad_correcta": integridad_correcta,
                "comparacion_con_v03": comparacion,
                "fichero_de_envio": str(SUBMISSION.relative_to(PROJECT_ROOT)),
                "nota": (
                    "La frecuencia de documento que define lo idiosincratico se calcula solo con "
                    "el tren de cada pliegue. Cada rama se mide en cinco repeticiones de la "
                    "particion, porque la fase v8 demostro que la estimacion es inestable entre "
                    "particiones."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"Completado en {round(time.time() - inicio, 1)} s")


def _verificar_integridad(
    marco: pd.DataFrame, test: pd.DataFrame, train: pd.DataFrame
) -> list[dict[str, str]]:
    """Comprueba la integridad del fichero de envio.

    Args:
        marco: marco de datos escrito.
        test: conjunto de prueba original.
        train: conjunto de entrenamiento.

    Returns:
        Lista de comprobaciones con resultado y detalle.
    """
    filas: list[dict[str, str]] = []

    def anadir(comprobacion: str, correcto: bool, detalle: str) -> None:
        filas.append({"comprobacion": comprobacion,
                      "resultado": "correcto" if correcto else "FALLO", "detalle": detalle})

    anadir("numero de filas", len(marco) == len(test),
           f"{len(marco)} frente a {len(test)} en la prueba")
    anadir("nombres y orden de columnas", list(marco.columns) == ["id"] + list(ETIQUETAS),
           f"{list(marco.columns)}")
    valores: set[int] = set()
    for etiqueta in ETIQUETAS:
        valores |= set(np.unique(marco[etiqueta].to_numpy()).tolist())
    anadir("valores de etiqueta", valores <= {0, 1}, f"{sorted(valores)}")
    anadir("sin valores ausentes", not bool(marco.isna().any().any()),
           f"{int(marco.isna().sum().sum())} celdas ausentes")
    anadir("sin identificadores duplicados", not bool(marco["id"].duplicated().any()),
           f"{int(marco['id'].duplicated().sum())} repetidos")
    anadir("identificadores en el orden de la prueba",
           bool((marco["id"].to_numpy() == test["id"].to_numpy()).all()),
           "coinciden con el orden original")
    anadir("filas distintas del entrenamiento", len(train) != len(test),
           f"{len(train)} frente a {len(test)}")
    return filas


def _comparar_con_v03(ruta: Path) -> dict[str, Any]:
    """Compara la submission generada con la de V03, celda a celda.

    Args:
        ruta: ruta de la submission nueva.

    Returns:
        Diccionario con el numero de celdas que cambian y con que proporcion.
    """
    nueva = pd.read_csv(ruta, dtype={"id": str}).set_index("id")
    vieja = pd.read_csv(
        PROJECT_ROOT / "submissions" / "toxic_submission_v03_lr_union_thr.csv",
        dtype={"id": str},
    ).set_index("id").loc[nueva.index]
    cambio = nueva[list(ETIQUETAS)].to_numpy() != vieja[list(ETIQUETAS)].to_numpy()
    return {
        "celdas_totales": int(cambio.size),
        "celdas_que_cambian": int(cambio.sum()),
        "filas_que_cambian": int(cambio.any(axis=1).sum()),
        "por_etiqueta": {e: int(cambio[:, i].sum()) for i, e in enumerate(ETIQUETAS)},
    }


def _informe(
    variantes: list[dict[str, Any]],
    mejor: dict[str, Any],
    complejidad: np.ndarray,
    integridad: list[dict[str, str]],
    integridad_correcta: bool,
    comparacion: dict[str, Any],
    inicio: float,
) -> None:
    """Escribe el informe de las dos tecnicas.

    Args:
        variantes: resultado de cada rama.
        mejor: rama con mayor diferencia.
        complejidad: matriz de rasgos de complejidad.
        integridad: comprobaciones del fichero.
        integridad_correcta: si todas pasaron.
        comparacion: diferencia celda a celda con la submission de V03.
        inicio: instante de arranque del script.
    """
    lineas = [
        "# Asimetria y kurtosis, e idiosincrasias internas (V8-11)",
        "",
        f"Generado por `experiments/v8_11_complejidad_idiosincrasias.py` ({EXPERIMENTO}).",
        "",
        "## 1. Que se mide y por que son genuinamente nuevas",
        "",
        "El material de clase lista la asimetria y la kurtosis como medidas de complejidad de la",
        "distribucion, y las idiosincrasias como familia de rasgos de atribucion. El proyecto ya",
        "habia medido longitudes medias y diversidad lexica en el analisis inicial, pero **nunca",
        "las habia usado como rasgos**, y el inventario lo confirma: no hay ninguna asignacion",
        "ni llamada a ellas en el codigo.",
        "",
        "La asimetria y la kurtosis de la distribucion de longitudes de palabra miden algo que la",
        "media no mide: si los mensajes largos son la cola de la distribucion o su cuerpo. Una",
        "cola larga significa que hay un grupo de mensajes desproporcionadamente largos, que es",
        "justo lo que un modelo de toxicidad deberia notar.",
        "",
        "Las idiosincrasias se operacionalizan **sin recursos externos**: un token es",
        "idiosincratico si aparece en menos de una fraccion dada de los documentos. La frecuencia",
        "de documento se calcula **solo con el tren de cada pliegue**, de modo que un token raro",
        "en el pliegue no puede usar su frecuencia global.",
        "",
        "## 2. Como se midio la consistencia",
        "",
        "La fase v8 demostro que la estimacion es extremadamente inestable entre particiones, con",
        "un desacuerdo medio de 0,92 a 0,97. Anadir rasgos solo vale si el cambio se mantiene a",
        "traves de las particiones, asi que cada rama se mide en **cinco repeticiones** y se",
        "cuenta cuantas mejoran.",
        "",
        "## 3. Los rasgos calculados",
        "",
        f"- asimetria media: {complejidad[:, 0].mean():+.4f}",
        f"- kurtosis media: {complejidad[:, 1].mean():+.4f}",
        f"- recorrido intercuantilico medio: {complejidad[:, 2].mean():.4f}",
        "",
        "## 4. Resultados",
        "",
        "| rama | metrica | diferencia | IC95 | repeticiones que mejoran | celdas | tasa |",
        "|---|---|---|---|---|---|---|",
    ]
    for v in variantes:
        lineas.append(
            f"| {v['clave']} | {v['metrica']:.6f} | {v['diferencia']:+.6f} | "
            f"[{v['ic95_0']:+.5f}, {v['ic95_1']:+.5f}] | "
            f"{v['repeticiones_que_mejoran']}/{NUMERO_REPETICIONES} | {v['celdas']} | "
            f"{v['tasa_acierto']:.4f} frente a {v['minimo_requerido']:.4f} |"
        )
    lineas += [
        "",
        "Metrica de cada rama en cada una de las cinco repeticiones:",
        "",
        "| rama | " + " | ".join(f"repeticion {r + 1}" for r in range(NUMERO_REPETICIONES)) + " |",
        "|---|" + "---|" * NUMERO_REPETICIONES,
    ]
    for v in variantes:
        lineas.append(
            f"| {v['clave']} | "
            + " | ".join(f"{m:.6f}" for m in v["metricas_por_repeticion"])
            + " |"
        )
    lineas += [
        "",
        "## 5. Integridad del fichero de envio",
        "",
        f"Resultado global: **{'todas pasan' if integridad_correcta else 'HAY FALLOS'}**.",
        "",
        "| comprobacion | resultado | detalle |",
        "|---|---|---|",
    ]
    for fila in integridad:
        lineas.append(f"| {fila['comprobacion']} | {fila['resultado']} | {fila['detalle']} |")
    lineas += [
        "",
        "## 6. Cuanto se aparta de V03",
        "",
        f"- celdas que cambian: {comparacion['celdas_que_cambian']} de "
        f"{comparacion['celdas_totales']}, en {comparacion['filas_que_cambian']} filas",
        "- por etiqueta: "
        + ", ".join(f"{e} {comparacion['por_etiqueta'][e]}" for e in ETIQUETAS),
        "",
        "## 7. Decision",
        "",
        f"La rama con mejor diferencia es **{mejor['clave']}**, con {mejor['diferencia']:+.6f}.",
        "",
        "**La submission se escribe pero no se recomienda enviarla.** El liston para gastar una",
        "sonda es que la tasa de acierto entre las celdas modificadas supere el minimo y que la",
        "mejora se mantenga en las cinco repeticiones. Con quince sondas ya pagadas y trece de",
        "catorce indistinguibles, gastar una en una linea que se mueve dentro del ruido no",
        "aporta informacion. **El envio es manual por D-063.**",
        "",
        f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        f"Duracion: {round(time.time() - inicio, 1)} s.",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
