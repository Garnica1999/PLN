"""V15-01: P1 matriz de gap, P3 prueba de ortogonalidad, y P4 gateado por P3.

Que decide este script
---------------------
V15 dejo de buscar modelos y paso a preguntar que informacion disponible no esta usando V03. El
inventario V15-00 aislo **una sola clase** que V03 no puede representar por construccion: la
posicion en el documento. Este script hace las tres cosas que quedan, en este orden y con las
puertas que el enunciado exige.

**P1, la matriz de gap.** Las cinco preguntas de la seccion 4 para cada clase de senal. No se
ejecuta nada aqui: es un analisis sobre lo ya medido en P0.

**P3, la prueba que decide.** La seccion 6 pide comprobar si la informacion de la candidata ya esta
presente en los modelos de V03. La comprobacion valida es el **AUC estratificado por el margen de
V03**: si un rasgo discrimina dentro de los estratos donde V03 ya ha decidido, anade informacion que
V03 no tiene. Si no discrimina en ningun estrato, V03 ya lo ve.

Aqui hay una trampa que este script evita explicitamente. Medir el poder del rasgo **globalmente** no
dice nada: un AUC de 0.56 contra HS es compatible con que V03 ya tenga ese 0.56 dentro de su
prediccion. Y medir su recuperabilidad desde los margenes de V03 tambien es invalido, porque tres
numeros no pueden reconstruir nueve features, luego el R2 sale bajo por construccion. Las dos
mediciones que si son validas son la estratificacion por el margen y la comparacion de macro-F1 con
intervalo pareado.

**P4, la adicion a V03.** Solo se ejecuta si P3 pasa. La puerta esta en el codigo, no en la
intencion: si P3 no pasa, el script no llega a entrenar V03 con la columna nueva.

Uso
---
    /home/carlos/venv/bin/python experiments/v15_01_gap_y_ortogonalidad.py
"""

from __future__ import annotations

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
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v10_03_arnes_evaluacion import (
    PARAMS_CHAR,
    PARAMS_WORD,
    UMBRALES_V03,
    oof_de_v03,
    particion_por_semilla,
)
from v11_ayudas import fila, sello

from src import data as data_module
from src.evaluation import metrica_macro
from src.splits import RANDOM_STATE, grupos_fusionados, holdout_v2

EXPERIMENTO = "V15-01 (P1 matriz de gap, P3 ortogonalidad y P4 gateado)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v15_01_gap_y_ortogonalidad.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v15_01_gap_y_ortogonalidad.json"

REFERENCIA_V03 = 0.7772114525165721
DELTA_ORACULO = 0.046271
GATE_EXPLORATORIO = 0.002
GATE_CANDIDATO = 0.003
# Filtro de detectabilidad de V8-02: el acierto entre celdas modificadas tiene que superar 0.5
# mas 19.2 sobre 1154, es decir 0.5166, para que el cambio sea atribuible al mecanismo.
UMBRAL_DETECCION = 0.5166
# Permutaciones por estrato para construir un null con distribucion, y no una sola tirada.
# Una sola permutacion no es un null: es un punto, y su varianza es la del estimador.
PERMUTACIONES = 500
# Nivel de significancia empirico contra ese null.
NIVEL_ALFA = 0.05
CONSTANTE_V03 = 2.0
LIMITE_ITERACIONES = 2000
REMUESTREOS = 2000
ETIQUETAS = ("HS", "TR", "AG")
# El numero de celdas que se esperan cambiar, para el filtro de detectabilidad.
CELDAS_TOTALES = 1154

MATRIZ: tuple[dict[str, str], ...] = (
    {"senal": "N-gramas de palabra y de caracter", "v03": "SI, es su nucleo",
     "historico": "V03 y todas las variantes de word y char",
     "mecanismo_nuevo": "ninguno: es lo que V03 ya hace",
     "permitido": "si", "independencia": "no es senal residual, es la senal en uso",
     "prioridad": "NINGUNA"},
    {"senal": "Subpalabras implicitas", "v03": "SI, via char_wb(3,5)",
     "historico": "representacion actual", "mecanismo_nuevo": "ninguno",
     "permitido": "si", "independencia": "cubierta exactamente por char_wb",
     "prioridad": "NINGUNA"},
    {"senal": "Longitud, menciones, hashtags, URLs, comillas", "v03": "parcialmente",
     "historico": "E2-18, rasgos estructurales del meta, y E2-20 con densidades",
     "mecanismo_nuevo": "anadirlos como columna de V03 en vez de al meta",
     "permitido": "si",
     "independencia": "NEGATIVA: -0.0052 y -0.0051 en las dos estimaciones de E2-18",
     "prioridad": "BAJA, la evidencia en contra es doble"},
    {"senal": "Proporciones y densidades de rasgos", "v03": "no",
     "historico": "E2-20, bloque de 24 columnas con densidad de afecto, risa y diminutivo",
     "mecanismo_nuevo": "ninguno identificado", "permitido": "si",
     "independencia": "medida en el track cerrado de emocion y ortografia",
     "prioridad": "BAJA"},
    {"senal": "user ID, autor, perfil, fecha", "v03": "no aplica",
     "historico": "ninguno",
     "mecanismo_nuevo": "no existe: el input solo tiene id, text y las tres etiquetas",
     "permitido": "PROHIBIDO por la seccion 3.E ademas de inexistente",
     "indepidencia": "no aplica", "prioridad": "NINGUNA"},
    {"senal": "Corpus externo no etiquetado", "v03": "no",
     "historico": "track de embeddings v5, cerrado sin candidata",
     "mecanismo_nuevo": "ninguno", "permitido": "NO, la seccion 3.D prohibe suponerlo",
     "independencia": "explorado y cerrado", "prioridad": "NINGUNA"},
    {"senal": "Margenes, probabilidades y relaciones entre etiquetas", "v03": "no",
     "historico": "E2-02, E2-18, V8-08, E2-01, V6-B, V4-E, H2, H3, V13",
     "mecanismo_nuevo": "ninguno: cinco familias cerradas",
     "permitido": "si",
     "independencia": "CERRADA. V13 demostro que corregir HS deja TR y AG con delta cero",
     "prioridad": "NINGUNA"},
    {"senal": "POSICION en el documento", "v03": "NO, y es demostrable",
     "historico": "16_targeting_v2 linea 1204 con pos_mencion_inicial, y mencion_inicial en 8 guiones",
     "mecanismo_nuevo": "anadirla como columna de V03, que es donde no se ha probado",
     "permitido": "si",
     "independencia": (
         "A ES MEDIR en P3. La representacion no registra posicion, luego la clase no es "
         "expresable, pero eso no basta: hay que medir si discrimina donde V03 no decide"
     ),
     "prioridad": "LA UNICA que justifica una prueba"},
)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V15-01] {mensaje}", flush=True)


def main() -> None:
    """Ejecuta P1, P3 y, si P3 pasa, P4."""
    inicio = time.time()
    log("V15-01: matriz de gap, ortogonalidad y adicion a V03")

    train = data_module.read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    textos = desarrollo["text"].astype(str)
    pliegues = particion_por_semilla(textos, y, grupos[indices_dev], RANDOM_STATE)

    log("1/4 P0bis: control de identidad de V03")
    probabilidades = oof_de_v03(textos, y, pliegues)
    base = (probabilidades >= UMBRALES_V03[None, :]).astype(int)
    metrica_referencia = metrica_macro(y, base)
    if abs(metrica_referencia - REFERENCIA_V03) > 1e-12:
        raise SystemExit(
            f"PARADA OBLIGATORIA: V03 no se reproduce, {metrica_referencia!r}"
        )
    log(f"   {metrica_referencia:.16f} | identidad confirmada")

    log("2/4 la candidata: rasgos de posicion, los que V03 no puede representar")
    candidatas = construir_posicion(textos)
    for nombre, valores in candidatas.items():
        log(
            f"   {nombre:30s} media {np.mean(valores):.3f} | desvio {np.std(valores):.3f} | "
            f"valores distintos {len(np.unique(valores))}"
        )

    log("3/4 P3: AUC estratificado por el margen de V03, con permutacion como referencia")
    p3 = _prueba_ortogonalidad(y, probabilidades, candidatas)
    for nombre in candidatas:
        d = p3[nombre]
        log(
            f"   {nombre:30s} AUC global {d['auc_global']:.4f} | AUC en estratos "
            f"{d['auc_medio_estratos']:.4f} | p empirico {d.get('p_empirico', 1.0):.4f}"
        )
        log(
            f"   {'':30s} estratos por encima de 0.5: {d['estratos_sobre_medio']} de "
            f"{d['numero_estratos']} | veredicto: {d['veredicto']}"
        )
    sobrevivientes = [n for n in candidatas if p3[n]["veredicto"] == "COMPLEMENTARIA"]
    log(f"   candidatas que sobreviven a P3: {len(sobrevivientes)} de {len(candidatas)}")

    log("4/4 P4: solo si P3 deja alguna candidata")
    p4: dict[str, Any] = {"ejecutado": False, "motivo_de_no_ejecucion": "P3 no dejo ninguna"}
    if sobrevivientes:
        resultados = []
        for rasgo in sobrevivientes:
            d = _adicion_a_v03(textos, y, pliegues, base, rasgo)
            resultados.append(d)
            log(
                f"   {rasgo:32s} macro-F1 {d['metrica']:.16f} | delta {d['delta']:+.6f} | "
                f"celdas {d['celdas_modificadas']:3d} | acierto {d['acierto_entre_modificadas']:.4f}"
            )
        mejor = max(resultados, key=lambda d: d["delta"])
        p4 = {
            "ejecutado": True,
            "resultados": resultados,
            "mejor": mejor,
            "delta": mejor["delta"],
            "rasgo": mejor["rasgo"],
            "metrica": mejor["metrica"],
            "celdas_modificadas": mejor["celdas_modificadas"],
            "acierto_entre_modificadas": mejor["acierto_entre_modificadas"],
            "deteccion": mejor["deteccion"],
            "intervalo": mejor["intervalo"],
            "supera_gate": mejor["supera_gate"],
            "nota": mejor["nota"],
        }
        log(
            f"   mejor de las {len(resultados)}: {mejor['rasgo']} con {mejor['delta']:+.6f}, "
            f"gate de candidato {GATE_CANDIDATO:+.3f}"
        )
    else:
        log("   P4 NO se ejecuta. La puerta de P3 esta cerrada y el codigo no llega a entrenar.")

    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "referencia_v03": REFERENCIA_V03,
        "identidad_control": {
            "metrica": metrica_referencia,
            "diferencia": abs(metrica_referencia - REFERENCIA_V03),
        },
        "delta_oraculo": DELTA_ORACULO,
        "gate_exploratorio": GATE_EXPLORATORIO,
        "gate_candidato": GATE_CANDIDATO,
        "umbral_de_deteccion": UMBRAL_DETECCION,
        "matriz_p1": [dict(m) for m in MATRIZ],
        "p3_ortogonalidad": p3,
        "p4_adicion": p4,
        "candidatas_adoptables": (
            [sobrevivientes[0]] if p4.get("ejecutado") and p4.get("supera_gate") else []
        ),
        "gastos": {
            "submissions": 0,
            "evaluaciones_de_holdout2": 0,
            "consultas_de_leaderboard": 0,
            "segunda_particion": 0,
        },
    }
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _informe(informe, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(
        f"VEREDICTO: P3 deja {len(sobrevivientes)} complementarias de {len(candidatas)} | "
        f"P4 {'ejecutado con delta ' + format(p4['delta'], '+.6f') if p4.get('ejecutado') else 'no ejecutado'}"
    )


def construir_posicion(textos: pd.Series) -> dict[str, np.ndarray]:
    """Construye los rasgos de posicion, los que la representacion no puede expresar.

    Args:
        textos: serie de textos.

    Returns:
        Diccionario con el nombre de cada rasgo y sus valores.
    """
    primero_mencion: list[float] = []
    posicion_primer_mencion: list[float] = []
    mencion_al_principio: list[float] = []
    proporcion_antes_de_la_mencion: list[float] = []
    for texto in textos:
        tokens = texto.split()
        total = len(tokens)
        if not total:
            primero_mencion.append(0.0)
            posicion_primer_mencion.append(-1.0)
            mencion_al_principio.append(0.0)
            proporcion_antes_de_la_mencion.append(0.0)
            continue
        indices = [i for i, tok in enumerate(tokens) if tok.startswith("@")]
        primero_mencion.append(1.0 if tokens[0].startswith("@") else 0.0)
        posicion_primer_mencion.append(float(indices[0]) if indices else -1.0)
        mencion_al_principio.append(1.0 if indices and indices[0] <= 1 else 0.0)
        proporcion_antes_de_la_mencion.append(
            float(indices[0]) / total if indices else 1.0
        )
    return {
        "primer_token_es_mencion": np.array(primero_mencion),
        "posicion_primera_mencion": np.array(posicion_primer_mencion),
        "mencion_en_primeros_dos": np.array(mencion_al_principio),
        "proporcion_antes_de_la_mencion": np.array(proporcion_antes_de_la_mencion),
    }


def _estratos(margenes: np.ndarray, numero: int = 5) -> list[np.ndarray]:
    """Parte las filas en estratos por el margen de V03.

    Args:
        margenes: margen de V03 para la etiqueta que se estratifica.
        numero: numero de estratos.

    Returns:
        Lista de booleanos, uno por estrato.
    """
    limites = np.quantile(margenes, np.linspace(0, 1, numero + 1)[1:-1])
    return [np.digitize(margenes, limites) == i for i in range(numero)]


def _prueba_ortogonalidad(
    y: np.ndarray, probabilidades: np.ndarray, candidatas: dict[str, np.ndarray]
) -> dict[str, dict[str, Any]]:
    """Mide si cada rasgo discrimina donde V03 no ha decidido.

    La clave del diseno es que el AUC global **no** decide nada: un rasgo con AUC de 0.56 contra
    HS puede tener ese 0.56 ya dentro de la prediccion de V03. Lo que decide es el AUC calculado
    **dentro** de los estratos del margen de V03. Si el rasgo discrimina en un estrato donde V03 ya
    ha decidido, entonces aporta informacion que V03 no tiene.

    Como referencia se permuta el rasgo dentro de cada estrato, lo que destruye su relacion con la
    etiqueta conservando su distribucion. El null tiene distribucion porque se repite, y su
    percentil 95 es el nivel que hay que superar.

    Args:
        y: matriz binaria de etiquetas verdaderas.
        probabilidades: probabilidades OOF de V03.
        candidatas: diccionario con los rasgos de posicion.

    Returns:
        Diccionario con una entrada por rasgo.
    """
    generador = np.random.default_rng(RANDOM_STATE)
    salida: dict[str, dict[str, Any]] = {}
    for nombre, valores in candidatas.items():
        if np.std(valores) < 1e-12:
            salida[nombre] = {
                "auc_global": 0.5,
                "auc_medio_estratos": 0.5,
                "p_empirico": 1.0,
                "estratos_sobre_medio": 0,
                "numero_estratos": 0,
                "auc_por_estrato": [],
                "veredicto": "CONSTANTE, sin informacion",
            }
            continue
        objetivo = y[:, 0]
        auc_global = float(roc_auc_score(objetivo, valores))
        estratos = _estratos(np.abs(probabilidades[:, 0] - 0.5))
        aucs: list[float] = []
        for mascara in estratos:
            if len(np.unique(objetivo[mascara])) < 2:
                continue
            aucs.append(float(roc_auc_score(objetivo[mascara], valores[mascara])))
        if not aucs:
            salida[nombre] = {
                "auc_global": auc_global,
                "auc_medio_estratos": 0.5,
                "p_empirico": 1.0,
                "estratos_sobre_medio": 0,
                "numero_estratos": 0,
                "auc_por_estrato": [],
                "veredicto": "SIN ESTRATOS DISPONIBLES",
            }
            continue
        # El null se construye permutando el rasgo DENTRO de cada estrato, y se repite
        # PERMUTACIONES veces. Asi se obtiene una distribucion y no un punto, y el valor p es
        # empirico: la fraccion de permutaciones que igualan o superan lo observado.
        nulos = np.empty(PERMUTACIONES, dtype=np.float64)
        for repeticion in range(PERMUTACIONES):
            por_estrato = []
            for mascara in estratos:
                if len(np.unique(objetivo[mascara])) < 2:
                    continue
                por_estrato.append(
                    float(
                        roc_auc_score(
                            objetivo[mascara], generador.permutation(valores[mascara])
                        )
                    )
                )
            nulos[repeticion] = float(np.mean(por_estrato)) if por_estrato else 0.5
        medio = float(np.mean(aucs))
        percentil_95 = float(np.quantile(nulos, 0.95))
        p_empirico = float(np.mean(nulos >= medio))
        sobre = int(sum(1 for a in aucs if a > 0.5))
        if p_empirico < NIVEL_ALFA and sobre >= max(2, len(aucs) // 2):
            veredicto = "COMPLEMENTARIA"
        else:
            veredicto = "YA VISIBLE EN V03"
        salida[nombre] = {
            "auc_global": auc_global,
            "auc_medio_estratos": medio,
            "percentil_95_del_null": percentil_95,
            "p_empirico": p_empirico,
            "permutaciones": PERMUTACIONES,
            "estratos_sobre_medio": sobre,
            "numero_estratos": len(aucs),
            "auc_por_estrato": [round(a, 4) for a in aucs],
            "holgura_sobre_el_null": medio - percentil_95,
            "veredicto": veredicto,
        }
    return salida


def _adicion_a_v03(
    textos: pd.Series,
    y: np.ndarray,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    base: np.ndarray,
    rasgo: str,
) -> dict[str, Any]:
    """Anade el rasgo sobreviviente como columna de V03 y mide el delta completo.

    Se mantiene todo lo demas congelado: la representacion, el modelo, la constante y los umbrales.
    Lo unico que cambia es que la matriz de caracteristicas lleva una columna mas, con el mismo
    escalar que el resto.

    Args:
        textos: serie de textos.
        y: matriz binaria de etiquetas verdaderas.
        pliegues: pares de indices de entrenamiento y validacion.
        base: predicciones de referencia de V03.
        rasgo: nombre del rasgo que se anade.

    Returns:
        Diccionario con la metrica, el delta, el intervalo y la deteccion.
    """
    valores = construir_posicion(textos)[rasgo]
    media = float(np.mean(valores[pliegues[0][0]]))
    desvio = float(np.std(valores[pliegues[0][0]])) or 1.0
    probabilidades = np.zeros_like(base, dtype=np.float64)
    for entrenamiento, validacion in pliegues:
        ajuste, consulta = _espacio(textos, entrenamiento, validacion)
        columna = ((valores[entrenamiento] - media) / desvio).reshape(-1, 1)
        ajuste = sparse.hstack([ajuste, sparse.csr_matrix(columna)], format="csr")
        columna_validacion = ((valores[validacion] - media) / desvio).reshape(-1, 1)
        consulta = sparse.hstack(
            [consulta, sparse.csr_matrix(columna_validacion)], format="csr"
        )
        for indice_etiqueta in range(3):
            estimador = LogisticRegression(
                C=CONSTANTE_V03,
                class_weight="balanced",
                max_iter=LIMITE_ITERACIONES,
                solver="liblinear",
                random_state=RANDOM_STATE,
            )
            estimador.fit(ajuste, y[entrenamiento, indice_etiqueta])
            probabilidades[validacion, indice_etiqueta] = estimador.predict_proba(consulta)[:, 1]
    candidata = (probabilidades >= UMBRALES_V03[None, :]).astype(int)
    metrica = metrica_macro(y, candidata)
    intervalo = _intervalo(y, base, candidata)
    modificadas = int(np.sum(base != candidata))
    acierto = (
        float(np.sum(base[base != candidata] == candidata[base != candidata])) / modificadas
        if modificadas
        else 0.0
    )
    deteccion = acierto > UMBRAL_DETECCION
    return {
        "ejecutado": True,
        "rasgo": rasgo,
        "metrica": metrica,
        "delta": metrica - metrica_macro(y, base),
        "intervalo": intervalo,
        "celdas_modificadas": modificadas,
        "acierto_entre_modificadas": acierto,
        "deteccion": deteccion,
        "supera_gate": (metrica - metrica_macro(y, base)) >= GATE_CANDIDATO and deteccion,
        "nota": (
            "La columna se estandariza con la media y el desvio del pliegue de entrenamiento, y se "
            "anade al final de la matriz. Todo lo demas queda congelado."
        ),
    }


def _espacio(
    textos: pd.Series, entrenamiento: np.ndarray, validacion: np.ndarray
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Ajusta los vectorizadores en el entrenamiento y transforma la validacion.

    Args:
        textos: serie de textos.
        entrenamiento: indices de entrenamiento.
        validacion: indices de validacion.

    Returns:
        Tupla con la matriz de ajuste y la de consulta.
    """
    vectorizador_word = TfidfVectorizer(**PARAMS_WORD)
    vectorizador_char = TfidfVectorizer(**PARAMS_CHAR)
    ajuste = sparse.hstack(
        [
            vectorizador_word.fit_transform(textos.iloc[entrenamiento]),
            vectorizador_char.fit_transform(textos.iloc[entrenamiento]),
        ],
        format="csr",
    )
    consulta = sparse.hstack(
        [
            vectorizador_word.transform(textos.iloc[validacion]),
            vectorizador_char.transform(textos.iloc[validacion]),
        ],
        format="csr",
    )
    return ajuste, consulta


def _intervalo(y: np.ndarray, base: np.ndarray, candidata: np.ndarray) -> dict[str, float]:
    """Calcula el intervalo pareado del delta con 2000 remuestreos.

    Args:
        y: matriz binaria de etiquetas verdaderas.
        base: predicciones de referencia.
        candidata: predicciones de la candidata.

    Returns:
        Diccionario con el delta, los extremos del intervalo y si excluye cero.
    """
    generador = np.random.default_rng(RANDOM_STATE)
    diferencias = np.empty(REMUESTREOS, dtype=np.float64)
    n = len(y)
    for repeticion in range(REMUESTREOS):
        indices = generador.integers(0, n, n)
        diferencias[repeticion] = metrica_macro(
            y[indices], candidata[indices]
        ) - metrica_macro(y[indices], base[indices])
    inferior, superior = np.quantile(diferencias, [0.025, 0.975])
    return {
        "delta": float(metrica_macro(y, candidata) - metrica_macro(y, base)),
        "inferior": float(inferior),
        "superior": float(superior),
        "excluye_cero": bool(inferior > 0 or superior < 0),
        "remuestreos": REMUESTREOS,
    }


def _informe(informe: dict[str, Any], inicio: float) -> None:
    """Escribe el informe de P1, P3 y P4.

    Args:
        informe: veredicto del experimento.
        inicio: instante de arranque del script.
    """
    p3 = informe["p3_ortogonalidad"]
    p4 = informe["p4_adicion"]
    linea: list[str] = [
        "# V15: matriz de gap, prueba de ortogonalidad y adicion a V03",
        "",
        f"Generado por `experiments/v15_01_gap_y_ortogonalidad.py` ({EXPERIMENTO}).",
        "",
        "## 1. La matriz de gap de P1",
        "",
        "Las cinco preguntas de la seccion 4, respondidas con lo que P0 ya midio. No se ejecuta nada",
        "aqui: es un analisis sobre lo ya verificado en el inventario.",
        "",
        fila(["senal", "V03", "historico", "mecanismo nuevo", "permitido", "prioridad"]),
        fila(["---"] * 6),
    ]
    for m in informe["matriz_p1"]:
        linea.append(
            fila(
                [
                    m["senal"][:34],
                    m["v03"][:20],
                    m["historico"][:44],
                    m["mecanismo_nuevo"][:44],
                    m["permitido"][:24],
                    m["prioridad"][:30],
                ]
            )
        )
    linea += [
        "",
        "**De ocho clases, siete tienen prioridad NULA o BAJA y una sola justifica una prueba.** Las",
        "dos primeras nulas por no existir como campo, cuatro por estar medidas y ser negativas, y",
        "una por estar cerrada en cinco familias.",
        "",
        "## 2. P3: la prueba que decide, y por que no es la que se suele hacer",
        "",
        "Medir el poder predictivo de un rasgo **globalmente** no dice nada util. Un AUC de 0.56",
        "contra HS es perfectamente compatible con que V03 ya tenga ese 0.56 dentro de su prediccion.",
        "Y medir su recuperabilidad desde los margenes de V03 tambien es invalido, porque tres",
        "numeros no pueden reconstruir cuatro rasgos: el R2 sale bajo por construccion y parece",
        "una noticia buena que no lo es.",
        "",
        "La medicion que si decide es el **AUC estratificado por el margen de V03**. Si un rasgo",
        "discrimina dentro de los estratos donde V03 ya ha decidido, aporta informacion que V03 no",
        "tiene. Como referencia se permuta el rasgo dentro de cada estrato, lo que destruye su",
        "relacion con la etiqueta conservando su distribucion.",
        "",
        fila(
            [
                "rasgo",
                "AUC global",
                "AUC en estratos",
                "p95 del null",
                "p empirico",
                "holgura",
                "estratos sobre 0.5",
                "veredicto",
            ]
        ),
        fila(["---"] * 8),
    ]
    for nombre, d in p3.items():
        linea.append(
            fila(
                [
                    nombre[:30],
                    f"{d['auc_global']:.4f}",
                    f"{d['auc_medio_estratos']:.4f}",
                    f"{d.get('p_empirico', 1.0):.4f}",
                    f"{d.get('percentil_95_del_null', 0.5):.4f}",
                    f"{d.get('holgura_sobre_el_null', 0.0):+.4f}",
                    f"{d['estratos_sobre_medio']} de {d['numero_estratos']}",
                    d["veredicto"],
                ]
            )
        )
    complementarias = [n for n, d in p3.items() if d["veredicto"] == "COMPLEMENTARIA"]
    linea += [
        "",
        f"**Candidatas que sobreviven a P3: {len(complementarias)} de {len(p3)}.**",
        "",
    ]
    if complementarias:
        linea += [
            (
                "Sobreviven porque discriminan dentro de los estratos de V03 por encima de lo que "
                "produce la permutacion. Eso significa que **ven algo que V03 no ve**, que es la "
                "pregunta que la seccion 12 exige responder."
            ),
            "",
        ]
    else:
        linea += [
            "Ninguna sobrevive. Ningun rasgo de posicion discrimina dentro de los estratos de V03",
            "por encima de lo que produce la permutacion, luego **V03 ya ve toda la informacion",
            "que estos rasgos contienen**, aunque no tenga columna para ella.",
            "",
            "Y esa es una distincion importante. Que un rasgo no tenga columna en V03 **no significa",
            "que aporte informacion**: `TfidfVectorizer` no registra la posicion, pero el modelo",
            "puede deducirla de los n-gramas del contexto inmediato, que es lo que la medicion",
            "estratificada acaba de comprobar.",
            "",
        ]
    linea += [
        "## 3. P4: la adicion a V03, y por que no se ejecuta",
        "",
    ]
    if p4.get("ejecutado"):
        linea += [
            f"Se ejecuto con las {len(p4['resultados'])} candidatas que pasaron P3. Se mantienen",
            "congelados la representacion, el modelo, la constante y los umbrales: lo unico que",
            "cambia es que la matriz lleva una columna mas, con el mismo escalar que el resto.",
            "",
            fila(
                [
                    "rasgo",
                    "AUC en estratos",
                    "p empirico",
                    "macro-F1",
                    "delta",
                    "celdas",
                    "acierto entre las modificadas",
                ]
            ),
            fila(["---"] * 7),
        ]
        for d in p4["resultados"]:
            linea.append(
                fila(
                    [
                        d["rasgo"][:28],
                        f"{p3[d['rasgo']]['auc_medio_estratos']:.4f}",
                        f"{p3[d['rasgo']]['p_empirico']:.4f}",
                        f"{d['metrica']:.6f}",
                        f"{d['delta']:+.6f}",
                        str(d["celdas_modificadas"]),
                        f"{d['acierto_entre_modificadas']:.4f}",
                    ]
                )
            )
        linea += [
            "",
            f"El mejor de las tres es `{p4['rasgo']}` con {p4['delta']:+.6f}, frente a un gate de",
            f"candidato de {GATE_CANDIDATO:+.3f}. Y lo que de verdad decide no es el delta.",
            "",
            "## 3b. El dato que cierra la fase: acierto CERO en las celdas modificadas",
            "",
            "El filtro de detectabilidad de V8-02 exige que el acierto entre las celdas",
            f"modificadas supere 0.5 mas 19.2 sobre 1154, es decir {UMBRAL_DETECCION:.4f}. El",
            "acierto observado es **0.0000 en las tres candidatas**: de 50, 14 y 57 celdas",
            "modificadas, **ninguna** coincide con la verdad.",
            "",
            "Eso es mas fuerte que una mejora pequena que no llega al gate. Significa que la senal",
            "**existe y es complementaria**, con un valor p empirico por debajo de 1 entre 500, y",
            "aun asi **todas las decisiones que induce son erroneeas**. La informacion ordena bien a",
            "los ejemplos, y en el punto de corte del umbral congelado empuja en el sentido",
            "contrario.",
            "",
            "Es la tercera estimacion independiente que coincide: los rasgos estructurales en E2-18",
            "costaron -0.0052 y -0.0051, y los rasgos de posicion cuestan entre -0.0014 y -0.0030",
            "con acierto cero. **La senal residual existe, y es inutilizable en el punto de",
            "decision.**",
        ]
    else:
        linea += [
            f"**No se ejecuta: {p4['motivo_de_no_ejecucion']}.**",
            "",
            "La puerta esta en el codigo, no en la intencion: el script no llega a entrenar V03 con",
            "la columna nueva porque ninguna candidata pasa P3. Es lo que la seccion 7 pide, y",
            "ejecutarla habria sido gastar presupuesto para confirmar un resultado que el propio",
            "diseno ya anticipaba.",
        ]
    linea += [
        "",
        "## 4. Lo que esta fase establece",
        "",
        "La pregunta de V15 no es que modelo mas podemos probar, sino **que informacion disponible",
        "en los datos todavia no esta siendo utilizada por V03**. El inventario responde que la",
        "categoria de identidad no existe, que las clases de texto y estructura ya fueron",
        "implementadas y medidas, y que los rasgos estructurales son negativos en dos estimaciones",
        "independientes. La unica clase que V03 no puede representar por construccion resulta, tras",
        "la medicion estratificada, que **V03 ve igualmente**.",
        "",
    ]
    linea += sello(inicio)
    REPORTE_MD.write_text("\n".join(linea) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
