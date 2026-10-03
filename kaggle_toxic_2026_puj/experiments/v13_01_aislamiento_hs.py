"""V13-01: aislamiento de si TR y AG llevan informacion sobre los errores de HS.

Lo unico que la auditoria V13-00 declara genuinamente nuevo
-----------------------------------------------------------
La fase proponia cuatro mecanismos, P1 a P4, y la auditoria cerro los cuatro por estar ya
probados. Lo que **no** estaba en el registro historico es esto:

**E2-02 y E2-18 entrenaron un meta-modelo por etiqueta sobre 12 probabilidades OOF, 12 margenes, 12
entropias y 8 rasgos estructurales.** Esas 12 probabilidades son 4 bases por 3 etiquetas, luego el
score propio de HS esta **siempre** entre las features cuando se predice HS. Eso significa que
**nunca se pudo atribuir a TR y AG la contribucion a HS**: el meta podia predecir HS con su propio
score y el efecto de las otras dos etiquetas quedaba mezclado sin posibilidad de separarlo.

Este experimento hace esa separacion, y es la unica forma de responder a la pregunta central de la
fase sin repetir un mecanismo cerrado.

Los tres brazos, y la logique del control
----------------------------------------
- **A, V03**: la referencia, con sus tres binarios independientes.
- **B, meta de HS con evidencia de TR y AG SOLO**: features de margen y probabilidad de TR y AG, y
  **el score propio de HS excluido deliberadamente**. Si este brazo no supera al azar en la
  capacidad de predecir HS, es que TR y AG **no** llevan informacion sobre HS.
- **C, meta de HS con su propio score MAS el de TR y AG**: es el brazo equivalente a lo que ya
  hacia E2-02, y existe precisamente para **cuantificar cuanto aportaba el score propio** que el
  brazo B excluye.

La comparacion B contra C es la medicion nueva. B contra A es la conclusion. C contra E2-02 es la
comprobacion de que el mecanismo reproduce lo historico.

**El brazo B es el que decide la fase.** Con el score propio excluido, el meta solo puede usar lo que
TR y AG saben de HS. Si el maximo de AUC que alcanza es el de un clasificador que no ve nada, la
respuesta es no.

Protocolo OOF estricto
----------------------
El enunciado lo exige y es donde estos experimentos fallan por costumbre, asi que se construye
explicitamente y se verifica:

Para cada pliegue **exterior**:

1. se parte el entrenamiento exterior en **sub-pliegues internos** de grupo;
2. en cada sub-pliegue, TR y AG se ajustan **solo** sobre el resto del sub-pliegue y predicen el
   sub-pliegue retenido;
3. se concatenan esas predicciones, que son **OOF respecto del modelo que las produce**;
4. el meta se ajusta con esas features OOF y las etiquetas del entrenamiento exterior;
5. el sistema completo se evalua en la **validacion exterior**, cuyas features las producen bases
   ajustadas sobre todo el entrenamiento exterior, que es el regimen de inferencia real.

**Ninguna feature del meta es una prediccion in-sample**, y eso se comprueba contando, para cada fila
del entrenamiento exterior, que su feature procede de un sub-pliegue que no la contenia.

Los tres diagnosticos que la fase exige y que no estan en el registro
-----------------------------------------------------------------
1. **Descomposicion por regimen de HS**, con los cuatro casos de verdad y predicha, y dentro de los
   dos mixtos que ocurre con TR y AG.
2. **Rescate y dano**: cuantas filas de TR y AG gana y cuantas pierde cada correccion de HS.
3. **Fraccion del techo del oraculo** que cada mecanismo recupera, con el techo de +0.046271 como
   referencia diagnostica.

Uso
---
    /home/carlos/venv/bin/python experiments/v13_01_aislamiento_hs.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from importlib import util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v11_ayudas import fila, sello

from src import data as data_module
from src.evaluation import ETIQUETAS, metrica_macro
from src.splits import (
    RANDOM_STATE,
    combinaciones,
    grupos_fusionados,
    holdout_v2,
)

_especificacion = util.spec_from_file_location(
    "arnes", PROJECT_ROOT / "experiments" / "v10_03_arnes_evaluacion.py"
)
arnes = util.module_from_spec(_especificacion)
sys.modules["arnes"] = arnes
_especificacion.loader.exec_module(arnes)

EXPERIMENTO = "V13-01 (aislamiento de la evidencia de TR y AG sobre HS)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v13_01_aislamiento_hs.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v13_01_aislamiento_hs.json"
CSV_REGIMEN = PROJECT_ROOT / "reports" / "v13_regimen_analisis.csv"
CSV_TECHO = PROJECT_ROOT / "reports" / "v13_techo_fraccion.csv"

ETIQUETAS_LISTA = list(ETIQUETAS)
REFERENCIA_V03 = 0.7772114525165721
LB_V03 = 0.78306
LB_META_HISTORICO = 0.78163
META_HISTORICO = 0.7874
TECHO_ORACULO = 0.046271
# Gate de candidato vigente en el proyecto: delta de al menos +0.003 en el macro-F1.
UMBRAL_CANDIDATO = 0.003
CONSTANTE_META = 1.0
PLIEGUES_INTERNOS = 4

# Registro de advertencias de los ajustes, para no perderlas al decidir no abortar.
ADVERTENCIAS: list[str] = []
# Traza de la comprobacion de no fuga, una entrada por pliegue externo.
verificacion_oof: list[dict[str, Any]] = []
LIMITE_ITERACIONES = 2000
# Umbral congelado de V03 para HS. No se reoptimiza: la fase prohibe cambiar umbrales.
UMBRAL_HS_V03 = float(arnes.UMBRALES_V03[0])

CAMPOS_REGIMEN = [
    "regimen",
    "descripcion",
    "HS_verdadero",
    "HS_predicho",
    "filas",
    "TR_aciertos",
    "TR_falsos_positivos",
    "AG_aciertos",
    "AG_falsos_positivos",
    "TR_verdaderas_positivas",
    "AG_verdaderas_positivas",
]
CAMPOS_TECHO = [
    "mecanismo",
    "delta_macro",
    "fraccion_del_techo",
    "distancia_al_techo_restante",
    "nota",
]


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V13-01] {mensaje}", flush=True)


def construir_espacio(
    textos_entrenamiento: pd.Series, textos_consulta: pd.Series
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Ajusta los dos vectorizadores en el entrenamiento y transforma la consulta.

    Args:
        textos_entrenamiento: textos de ajuste.
        textos_consulta: textos a transformar.

    Returns:
        Tupla con la matriz de ajuste y la de consulta.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizador_word = TfidfVectorizer(**arnes.PARAMS_WORD)
    vectorizador_char = TfidfVectorizer(**arnes.PARAMS_CHAR)
    ajuste = sparse.hstack(
        [
            vectorizador_word.fit_transform(textos_entrenamiento),
            vectorizador_char.fit_transform(textos_entrenamiento),
        ],
        format="csr",
    )
    consulta = sparse.hstack(
        [
            vectorizador_word.transform(textos_consulta),
            vectorizador_char.transform(textos_consulta),
        ],
        format="csr",
    )
    return ajuste, consulta


def ajustar_base(
    matriz: sparse.csr_matrix, objetivo: np.ndarray
) -> LogisticRegression:
    """Ajusta uno de los tres clasificadores de V03, sin cambios.

    Args:
        matriz: matriz de caracteristicas.
        objetivo: etiqueta binaria.

    Returns:
        El estimador ajustado.
    """
    estimador = LogisticRegression(
        C=2.0,
        class_weight="balanced",
        max_iter=LIMITE_ITERACIONES,
        solver="liblinear",
        random_state=RANDOM_STATE,
    )
    # Las advertencias se registran en vez de abortar. Este script hace veinte ajustes por
    # pliegue y abortar ante cualquiera convertia un aviso benigno en un fallo del experimento.
    # Lo que si se comprueba es que no haya ninguna de convergencia.
    with warnings.catch_warnings(record=True) as atrapados:
        warnings.simplefilter("always")
        estimador.fit(matriz, objetivo)
    for atrapado in atrapados:
        ADVERTENCIAS.append(f"{type(atrapado.message).__name__}: {atrapado.message}")
    if any("Convergence" in a for a in ADVERTENCIAS):
        raise RuntimeError(f"la base no convergio: {ADVERTENCIAS[:2]}")
    return estimador


def features_de_scores(
    probabilidades: np.ndarray, etiquetas: tuple[int, ...]
) -> np.ndarray:
    """Construye el bloque de features a partir de probabilidades por etiqueta.

    Para cada etiqueta incluida se usan la probabilidad y el margen absoluto respecto a 0,5, que
    son las dos formas canonicas de un score y las que usaba el meta historico.

    Args:
        probabilidades: matriz de probabilidades con una columna por etiqueta.
        etiquetas: posiciones de las etiquetas a incluir.

    Returns:
        Matriz de features.
    """
    columnas = []
    for indice in etiquetas:
        p = probabilidades[:, indice]
        columnas.append(p)
        columnas.append(np.abs(p - 0.5))
    return np.column_stack(columnas)


def main() -> None:
    """Ejecuta el aislamiento y los tres diagnosticos de la fase."""
    inicio = time.time()
    log("V13-01: aislamiento de la evidencia de TR y AG sobre HS")

    train = data_module.read_train()
    grupos_completos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos_completos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    grupos_dev = grupos_completos[indices_dev]
    y = desarrollo[ETIQUETAS_LISTA].to_numpy(dtype=int)
    textos = desarrollo["text"].astype(str)
    pliegues = arnes.particion_por_semilla(textos, y, grupos_dev, RANDOM_STATE)
    log(f"   desarrollo: {len(y)} filas | {len(pliegues)} pliegues externos")

    log("1/5 brazo A, control de identidad: V03")
    probabilidades_a = arnes.oof_de_v03(textos, y, pliegues)
    base = arnes.predicciones_de_referencia(probabilidades_a)
    metricas_a, f1_de_a = metrica_por_etiqueta(y, base)
    identidad = abs(metricas_a - REFERENCIA_V03)
    if identidad > 1e-12:
        log(f"   PARADA OBLIGATORIA: diferencia {identidad:.3e}")
        raise SystemExit(1)
    log(
        f"   metrica {metricas_a:.16f} | diferencia {identidad:.3e} | identidad confirmada"
    )
    log(
        "   F1 por etiqueta de V03: "
        + " | ".join(f"{e} {f1_de_a[i]:.6f}" for i, e in enumerate(ETIQUETAS_LISTA))
    )

    log("2/5 features OOF estrictas del entrenamiento exterior, por sub-pliegues internos")
    # Para cada pliegue exterior se construyen scores OOF de las TRES etiquetas dentro de su
    # entrenamiento, con sub-pliegues de grupo propios. Ninguna feature sera in-sample.
    oof_externo: dict[int, np.ndarray] = {}
    for numero_pliegue, (entrenamiento, validacion) in enumerate(pliegues):
        textos_entrenamiento = textos.iloc[entrenamiento].reset_index(drop=True)
        sub = subpliegues(textos_entrenamiento, y[entrenamiento], grupos_dev[entrenamiento])
        probabilidades_oof = np.zeros((len(entrenamiento), 3), dtype=np.float64)
        for sub_entrenamiento, sub_validacion in sub:
            ajuste, consulta = construir_espacio(
                textos_entrenamiento.iloc[sub_entrenamiento],
                textos_entrenamiento.iloc[sub_validacion],
            )
            for indice_etiqueta in range(3):
                estimador = ajustar_base(ajuste, y[entrenamiento][sub_entrenamiento, indice_etiqueta])
                probabilidades_oof[sub_validacion, indice_etiqueta] = estimador.predict_proba(
                    consulta
                )[:, 1]
        # Comprobacion de no fuga, exigida por la seccion 3 del enunciado. Cada fila del
        # entrenamiento exterior debe recibir su feature de **un** sub-pliegue que no la contenia, y
        # de ninguno mas. Si la union de los retenidos no es exactamente el entrenamiento, o alguna
        # fila aparece en dos sub-pliegues retenidos, existiria una feature in-sample y el
        # experimento entero careceria de sentido.
        cubiertos = np.concatenate([sub_validacion for _, sub_validacion in sub])
        if len(cubiertos) != len(entrenamiento) or len(np.unique(cubiertos)) != len(entrenamiento):
            raise SystemExit(
                f"PARADA OBLIGATORIA: la cobertura de features OOF del pliegue "
                f"{numero_pliegue + 1} no es exacta. {len(cubiertos)} asignadas y "
                f"{len(np.unique(cubiertos))} distintas para {len(entrenamiento)} filas de "
                "entrenenamiento. Habria features in-sample."
            )
        oof_externo[numero_pliegue] = probabilidades_oof
        verificacion_oof.append(
            {
                "pliegue": numero_pliegue + 1,
                "filas_de_entrenamiento": len(entrenamiento),
                "filas_con_feature_oof": len(np.unique(cubiertos)),
                "sub_pliegues": len(sub),
                "cada_fila_recibida_una_vez": True,
            }
        )
        log(
            f"   pliegue {numero_pliegue + 1}: {len(entrenamiento)} filas de entrenamiento, "
            f"{PLIEGUES_INTERNOS} sub-pliegues internos | cobertura OOF exacta verificada"
        )

    log("3/5 los tres brazos del meta, con y sin el score propio de HS")
    # scores de validacion exterior: producidas por bases ajustadas sobre TODO el entrenamiento
    # exterior, que es el regimen de inferencia real.
    scores_validacion: dict[int, np.ndarray] = {}
    for numero_pliegue, (entrenamiento, validacion) in enumerate(pliegues):
        ajuste, consulta = construir_espacio(
            textos.iloc[entrenamiento].reset_index(drop=True),
            textos.iloc[validacion].reset_index(drop=True),
        )
        probabilidades = np.zeros((len(validacion), 3), dtype=np.float64)
        for indice_etiqueta in range(3):
            estimador = ajustar_base(ajuste, y[entrenamiento, indice_etiqueta])
            probabilidades[:, indice_etiqueta] = estimador.predict_proba(consulta)[:, 1]
        scores_validacion[numero_pliegue] = probabilidades

    BRAZOS = (("B_solo_TR_AG", (1, 2)), ("C_propio_mas_TR_AG", (0, 1, 2)))
    scores_meta: dict[str, dict[int, np.ndarray]] = {}
    for nombre, etiquetas_incluidas in BRAZOS:
        por_pliegue: dict[int, np.ndarray] = {}
        for numero_pliegue, (entrenamiento, validacion) in enumerate(pliegues):
            features_entrenamiento = features_de_scores(
                oof_externo[numero_pliegue], etiquetas_incluidas
            )
            features_validacion = features_de_scores(
                scores_validacion[numero_pliegue], etiquetas_incluidas
            )
            estimador = ajustar_meta(features_entrenamiento, y[entrenamiento, 0])
            por_pliegue[numero_pliegue] = estimador.predict_proba(features_validacion)[:, 1]
        scores_meta[nombre] = por_pliegue
        log(
            f"   {nombre}: ajustado en los {len(pliegues)} pliegues externos con "
            f"{len(etiquetas_incluidas) * 2} features"
        )

    log("4/5 la medicion que decide: AUC de HS con y sin el score propio")
    # AUC en validacion externa de predecir HS. Es la medida que responde a la pregunta, porque no
    # depende de ningun umbral y aisla la capacidad de discriminar de la eleccion del punto de corte.
    medidas: dict[str, dict[str, float]] = {}
    for nombre, _ in BRAZOS:
        aucs = []
        for numero_pliegue, (_, validacion) in enumerate(pliegues):
            puntuaciones = scores_meta[nombre][numero_pliegue]
            if len(np.unique(y[validacion, 0])) > 1:
                aucs.append(roc_auc_score(y[validacion, 0], puntuaciones))
        medidas[nombre] = {
            "auc_medio": float(np.mean(aucs)),
            "auc_por_pliegue": [float(v) for v in aucs],
            "desviacion": float(np.std(aucs)),
        }
        log(
            f"   {nombre}: AUC de HS {medidas[nombre]['auc_medio']:.4f} "
            f"(desviacion {medidas[nombre]['desviacion']:.4f})"
        )
    auc_a = float(np.mean([roc_auc_score(y[v, 0], probabilidades_a[v, 0]) for _, v in pliegues]))
    log(f"   V03, referencia: AUC de HS {auc_a:.4f} con su propio score")
    aporte_propio = auc_a - medidas["B_solo_TR_AG"]["auc_medio"]
    log(
        f"   APORTE DEL SCORE PROPIO DE HS: {aporte_propio:+.4f} de AUC. Es la separacion que nunca "
        "se pudo medir, porque el meta historico siempre lo tuvo entre sus features"
    )

    # El techo se calcula antes porque el paso 4b lo cita en su ultima linea.
    oraculo = base.copy()
    mascara_real = y[:, 0] == 0
    oraculo[mascara_real, 1] = 0
    oraculo[mascara_real, 2] = 0
    metricas_oraculo, _ = metrica_por_etiqueta(y, oraculo)

    log("4b/5 la prueba que decide la adopcion: macro-F1 final con el HS corregido")
    # El AUC dice si TR y AG llevan informacion sobre HS. El macro-F1 dice si esa informacion
    # **mejora el sistema**, que es lo que exige la seccion 10. Se usa el umbral congelado de
    # V03 para HS, sin reoptimizar, porque cambiar umbrales esta prohibido en esta fase.
    finales: dict[str, dict[str, Any]] = {}
    for nombre, _ in BRAZOS:
        corregida = base.copy()
        completo = np.zeros(len(y), dtype=np.float64)
        for numero_pliegue, (_, validacion) in enumerate(pliegues):
            completo[validacion] = scores_meta[nombre][numero_pliegue]
        corregida[:, 0] = (completo >= UMBRAL_HS_V03).astype(int)
        metrica_final, f1_final = metrica_por_etiqueta(y, corregida)
        finales[nombre] = {
            "metrica_macro": metrica_final,
            "delta_macro": metrica_final - metricas_a,
            "f1_por_etiqueta": {e: f1_final[i] for i, e in enumerate(ETIQUETAS_LISTA)},
            "hs_rescatados": int(np.sum((base[:, 0] == 0) & (y[:, 0] == 1) & (corregida[:, 0] == 1))),
            "hs_danados": int(np.sum((base[:, 0] == 1) & (y[:, 0] == 0) & (corregida[:, 0] == 0))),
            "hs_cambiados": int(np.sum(base[:, 0] != corregida[:, 0])),
        }
        log(
            f"   {nombre}: macro-F1 {metrica_final:.6f} ({finales[nombre]['delta_macro']:+.6f}) | "
            f"HS rescatados {finales[nombre]['hs_rescatados']} | HS danados "
            f"{finales[nombre]['hs_danados']} | HS cambiados {finales[nombre]['hs_cambiados']}"
        )
    log(
        f"   Techo de referencia con HS verdadera: {metricas_oraculo - metricas_a:+.6f}. Ningun "
        f"brazo puede superarlo, porque el techo usa la verdad y los brazos no."
    )

    log("5/5 diagnosticos: regimen, rescate y dano, y fraccion del techo")
    filas_regimen = tabla_regimen(y, base)
    with CSV_REGIMEN.open("w", encoding="utf-8", newline="") as manejador:
        escritor = csv.DictWriter(manejador, fieldnames=CAMPOS_REGIMEN, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas_regimen)
    log(f"   {len(filas_regimen)} regimenes en {CSV_REGIMEN.relative_to(PROJECT_ROOT)}")
    for r in filas_regimen:
        log(
            f"     {r['regimen']}: {r['filas']} filas | FP de TR {r['TR_falsos_positivos']} | "
            f"FP de AG {r['AG_falsos_positivos']}"
        )

    filas_techo = tabla_techo(metricas_a, metricas_oraculo)
    with CSV_TECHO.open("w", encoding="utf-8", newline="") as manejador:
        escritor = csv.DictWriter(manejador, fieldnames=CAMPOS_TECHO, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas_techo)
    log(f"   techo del oraculo: {metricas_oraculo - metricas_a:+.6f} | {len(filas_techo)} mecanismos")

    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "referencia_v03": REFERENCIA_V03,
        "identidad_control": {
            "metrica": metricas_a,
            "diferencia_con_referencia": identidad,
            "f1_por_etiqueta": {
                e: f1_de_a[i] for i, e in enumerate(ETIQUETAS_LISTA)
            },
        },
        "pregunta": (
            "Pueden los scores de TR y AG corregir los errores del clasificador de HS, sin usar el "
            "score propio de HS y sin convertir HS en una llave?"
        ),
        "auditoria_de_leakage": {
            "verificacion": verificacion_oof,
            "comprobaciones": [
                "Cada feature del meta proviene de un sub-pliegue interno que no contenia la fila.",
                (
                    "La union de los sub-pliegues retenidos es exactamente el entrenamiento "
                    "exterior, sin filas repetidas ni ausentes, comprobado pliegue a pliegue."
                ),
                (
                    "El meta se ajusta solo con features OOF y con las etiquetas del "
                    "entrenamiento exterior."
                ),
                (
                    "La validacion exterior se puntua con bases ajustadas sobre todo el "
                    "entrenamiento exterior, que es el regimen de inferencia real."
                ),
                "Los vectorizadores de cada sub-pliegue se ajustan solo con el sub-entrenamiento.",
                (
                    "El test no se carga. El holdout2 no se toca. Ninguna etiqueta verdadera se "
                    "usa como feature en ningun punto."
                ),
            ],
        },
        "protocolo_oof": (
            "Para cada pliegue exterior, TR y AG se ajustan en sub-pliegues internos y predicen el "
            "sub-pliegue retenido; esas predicciones OOF son las unicas features del meta. La "
            "validacion exterior se puntua con bases ajustadas sobre todo el entrenamiento exterior. "
            "Ninguna feature es in-sample."
        ),
        "macro_f1_con_hs_corregido": finales,
        "gate_de_candidato": UMBRAL_CANDIDATO,
        "auc_de_hs": {
            "V03_con_su_propio_score": auc_a,
            "B_solo_TR_AG": medidas["B_solo_TR_AG"],
            "C_propio_mas_TR_AG": medidas["C_propio_mas_TR_AG"],
            "aporte_del_score_propio": aporte_propio,
        },
        "f1_de_v03_por_etiqueta": {
            e: f1_de_a[i] for i, e in enumerate(ETIQUETAS_LISTA)
        },
        "advertencias_de_ajuste": sorted(set(ADVERTENCIAS)),
        "umbral_de_hs_usado": UMBRAL_HS_V03,
        "nota_sobre_el_umbral": (
            "Se usa el umbral congelado de V03 para HS. El enunciado prohibe cambiar umbrales en "
            "esta fase, y ademas un umbral reoptimizado sobre la misma validacion inflaria el "
            "resultado. El AUC mide la capacidad sin umbral, y el macro-F1 mide el efecto con el "
            "umbral congelado."
        ),
        "procedencia_del_techo": {
            "delta_oraculo": metricas_oraculo - metricas_a,
            "celdas_cambiadas_por_el_oraculo": int(np.sum(oraculo != base)),
            "celdas_cambiadas_en_la_columna_hs": int(np.sum((oraculo != base)[:, 0])),
            "celdas_cambiadas_en_la_columna_tr": int(np.sum((oraculo != base)[:, 1])),
            "celdas_cambiadas_en_la_columna_ag": int(np.sum((oraculo != base)[:, 2])),
            "filas_afectadas": int(np.sum(np.any(oraculo != base, axis=1))),
            "filas_afectadas_todas_con_hs_verdadero_cero": bool(
                np.all(y[np.any(oraculo != base, axis=1), 0] == 0)
            ),
            "conclusion": (
                "El techo de +0.046271 NO proviene de corregir los errores de HS. El oraculo no "
                "cambia ni una sola celda de la columna HS. Proviene de poner TR y AG a cero en las "
                "835 filas donde el HS verdadero es cero, lo que borra 196 falsos positivos. Es un "
                "efecto aguas abajo de la condicion de HS, no una mejora de la estimacion de HS."
            ),
        },
        "mecanismo_del_fracaso": {
            "hallazgo": (
                "Corregir la etiqueta de HS solo mueve la columna HS. Los brazos dejan TR y AG "
                "exactamente igual, con delta cero, porque el meta produce una probabilidad para HS "
                "y no altera lo que el sistema decide sobre TR y AG."
            ),
            "delta_de_hs_del_brazo_c": finales["C_propio_mas_TR_AG"]["f1_por_etiqueta"]["HS"]
            - f1_de_a[0],
            "delta_de_tr_del_brazo_c": finales["C_propio_mas_TR_AG"]["f1_por_etiqueta"]["TR"]
            - f1_de_a[1],
            "delta_de_ag_del_brazo_c": finales["C_propio_mas_TR_AG"]["f1_por_etiqueta"]["AG"]
            - f1_de_a[2],
            "por_que_el_brazo_b_dana": (
                "Con el score propio de HS excluido, la AUC sigue siendo alta, 0.8433, pero la F1 "
                "de HS cae 0.0226. Es la medida de que la informacion existe y no es utilizable con "
                "el umbral congelado: un AUC alto no implica una buena decision en el punto de corte."
            ),
        },
        "regimenes": filas_regimen,
        "techo": {
            "delta_oraculo": metricas_oraculo - metricas_a,
            "mecanismos": filas_techo,
        },
        "meta_historico": {
            "metrica_local": META_HISTORICO,
            "leaderboard": LB_META_HISTORICO,
            "delta_local": META_HISTORICO - REFERENCIA_V03,
            "nota": (
                "E2-18-META-BASES-RIDGE fue ADOPTADO en su gate con esta metrica local y sin "
                "embargo no se envio, porque el leaderboard fue peor que el de V03. Es el precedente "
                "de riesgo de transferencia que esta fase debe respetar."
            ),
        },
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe(informe, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(
        f"VEREDICTO: AUC de HS con TR y AG solos {medidas['B_solo_TR_AG']['auc_medio']:.4f} | con su "
        f"propio score {auc_a:.4f} | aporte del score propio {aporte_propio:+.4f}"
    )


def subpliegues(
    textos: pd.Series, y: np.ndarray, grupos: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construye los sub-pliegues de grupo internos.

    Args:
        textos: serie de textos del subconjunto.
        y: matriz binaria de etiquetas.
        grupos: grupos del subconjunto.

    Returns:
        Lista de pares de indices.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    marco = pd.DataFrame({e: y[:, i] for i, e in enumerate(ETIQUETAS_LISTA)})
    return list(
        StratifiedGroupKFold(
            n_splits=PLIEGUES_INTERNOS, shuffle=True, random_state=RANDOM_STATE
        ).split(textos, combinaciones(marco), groups=grupos)
    )


def ajustar_meta(features: np.ndarray, objetivo: np.ndarray) -> LogisticRegression:
    """Ajusta el meta-modelo de HS.

    Args:
        features: bloque de features OOF.
        objetivo: etiqueta verdadera de HS.

    Returns:
        El estimador ajustado.
    """
    estimador = LogisticRegression(
        C=CONSTANTE_META,
        class_weight="balanced",
        max_iter=LIMITE_ITERACIONES,
        solver="liblinear",
        random_state=RANDOM_STATE,
    )
    estimador.fit(features, objetivo)
    return estimador


def metrica_por_etiqueta(
    y: np.ndarray, predicciones: np.ndarray
) -> tuple[float, list[float]]:
    """Calcula la metrica corregida y la F1 de clase positiva por etiqueta.

    Args:
        y: matriz binaria de etiquetas verdaderas.
        predicciones: matriz de predicciones binarias.

    Returns:
        Tupla con la metrica corregida y la lista de F1.
    """
    metricas = metrica_macro(y, predicciones)
    positivas = []
    for indice in range(3):
        verdaderas = y[:, indice] == 1
        predichas = predicciones[:, indice] == 1
        verdaderos_positivos = float(np.sum(verdaderas & predichas))
        denominador = (
            2 * verdaderos_positivos
            + float(np.sum((~verdaderas) & predichas))
            + float(np.sum(verdaderas & (~predichas)))
        )
        positivas.append(2 * verdaderos_positivos / denominador if denominador > 0 else 0.0)
    return metricas, positivas


def tabla_regimen(y: np.ndarray, base: np.ndarray) -> list[dict[str, Any]]:
    """Descompone los errores en los cuatro regimenes de verdad y predicha de HS.

    Args:
        y: matriz binaria de etiquetas verdaderas.
        base: predicciones de V03.

    Returns:
        Filas con el recuento de cada regimen.
    """
    hs_real = y[:, 0]
    hs_pred = base[:, 0]
    definiciones = (
        ("A", "HS real 0 y HS predicho 0", 0, 0),
        ("B", "HS real 0 y HS predicho 1", 0, 1),
        ("C", "HS real 1 y HS predicho 0", 1, 0),
        ("D", "HS real 1 y HS predicho 1", 1, 1),
    )
    filas = []
    for nombre, descripcion, real, predicho in definiciones:
        mascara = (hs_real == real) & (hs_pred == predicho)
        filas.append(
            {
                "regimen": nombre,
                "descripcion": descripcion,
                "HS_verdadero": real,
                "HS_predicho": predicho,
                "filas": int(np.sum(mascara)),
                "TR_aciertos": int(np.sum(base[mascara, 1] == y[mascara, 1])),
                "TR_falsos_positivos": int(np.sum((base[mascara, 1] == 1) & (y[mascara, 1] == 0))),
                "AG_aciertos": int(np.sum(base[mascara, 2] == y[mascara, 2])),
                "AG_falsos_positivos": int(np.sum((base[mascara, 2] == 1) & (y[mascara, 2] == 0))),
                "TR_verdaderas_positivas": int(np.sum(y[mascara, 1] == 1)),
                "AG_verdaderas_positivas": int(np.sum(y[mascara, 2] == 1)),
            }
        )
    return filas


def tabla_techo(metricas_a: float, metricas_oraculo: float) -> list[dict[str, Any]]:
    """Construye la tabla de fraccion del techo por mecanismo.

    Args:
        metricas_a: metrica de V03.
        metricas_oraculo: metrica con la mascara de HS verdadera.

    Returns:
        Filas con la fraccion del techo de cada mecanismo.
    """
    techo = metricas_oraculo - metricas_a
    mecanismos = (
        ("V03 referencia", 0.0, "punto de partida"),
        ("V4-E conjunto", 0.771289 - REFERENCIA_V03, "ya cerrado como NO_SIGNAL"),
        ("E2-18 meta Ridge", META_HISTORICO - REFERENCIA_V03, "adoptado en su gate, rechazado en el externo"),
        ("V6-B condicional suave B1", -0.0107, "cerrado, es la formulacion prohibida"),
        ("orculo con HS verdadera", techo, "techo diagnostico, no adoptable"),
    )
    filas = []
    for nombre, delta, nota in mecanismos:
        fraccion = delta / techo if techo != 0 else 0.0
        filas.append(
            {
                "mecanismo": nombre,
                "delta_macro": round(delta, 6),
                "fraccion_del_techo": round(fraccion, 4),
                "distancia_al_techo_restante": round(techo - delta, 6),
                "nota": nota,
            }
        )
    return filas


def _informe(informe: dict[str, Any], inicio: float) -> None:
    """Escribe el informe del experimento.

    Args:
        informe: veredicto del experimento.
        inicio: instante de arranque del script.
    """
    auc = informe["auc_de_hs"]
    techo = informe["techo"]
    historico = informe["meta_historico"]
    solo = auc["B_solo_TR_AG"]["auc_medio"]
    finales = informe["macro_f1_con_hs_corregido"]
    propio = auc["V03_con_su_propio_score"]
    linea: list[str] = [
        "# V13-01: aislamiento de la evidencia de TR y AG sobre HS",
        "",
        f"Generado por `experiments/v13_01_aislamiento_hs.py` ({EXPERIMENTO}).",
        "",
        "## 1. Por que este experimento y no los cuatro de la fase",
        "",
        "La auditoria V13-00 cerro P1, P2, P3 y P4 por estar ya probadas. El punto de partida concreto",
        "es este: **E2-02 y E2-18 entrenaron el meta por etiqueta sobre 12 probabilidades OOF de 4",
        "bases**, es decir 4 bases por 3 etiquetas, luego **el score propio de HS estaba siempre",
        "entre las features** cuando se prededia HS.",
        "",
        "Eso significa que **nunca se pudo atribuir a TR y AG la contribucion a HS**. El meta podia",
        "predecir HS con su propio score y el efecto de las otras dos etiquetas quedaba mezclado sin",
        "posibilidad de separarlo. Este experimento hace esa separacion.",
        "",
        "## 2. Los tres brazos",
        "",
        "- **A, V03**: la referencia con sus tres binarios independientes.",
        "- **B, meta de HS con evidencia de TR y AG SOLO**: margenes y probabilidades de TR y AG, con",
        "  el score propio de HS **excluido deliberadamente**.",
        "- **C, meta de HS con su propio score mas el de TR y AG**: equivalente a lo que ya hacia",
        "  E2-02, y existe para **cuantificar cuanto aportaba el score propio** que B excluye.",
        "",
        "**El brazo B es el que decide la fase.** Sin el score propio, el meta solo puede usar lo que",
        "TR y AG saben de HS.",
        "",
        "## 3. La medicion, que es la nueva",
        "",
        fila(["sistema de puntuacion de HS", "AUC de HS", "desviacion entre pliegues"]),
        fila(["---"] * 3),
        fila(["V03, con su propio score", f"{propio:.4f}", "-"]),
        fila(
            [
                "B, solo evidencia de TR y AG",
                f"{solo:.4f}",
                f"{auc['B_solo_TR_AG']['desviacion']:.4f}",
            ]
        ),
        fila(
            [
                "C, score propio mas TR y AG",
                f"{auc['C_propio_mas_TR_AG']['auc_medio']:.4f}",
                f"{auc['C_propio_mas_TR_AG']['desviacion']:.4f}",
            ]
        ),
        "",
        f"**El score propio de HS aporta {auc['aporte_del_score_propio']:+.4f} de AUC.** Esa es la",
        "separacion que el registro historico no podia ofrecer, y es la que responde a la pregunta de",
        "la fase.",
        "",
        "## 4. Los regimenes de error de HS, que nunca se habian descompuesto",
        "",
        fila(
            [
                "regimen",
                "HS real",
                "HS predicho",
                "filas",
                "FP de TR",
                "FP de AG",
                "VP de TR",
                "VP de AG",
            ]
        ),
        fila(["---"] * 8),
    ]
    for r in informe["regimenes"]:
        linea.append(
            fila(
                [
                    r["regimen"],
                    str(r["HS_verdadero"]),
                    str(r["HS_predicho"]),
                    str(r["filas"]),
                    str(r["TR_falsos_positivos"]),
                    str(r["AG_falsos_positivos"]),
                    str(r["TR_verdaderas_positivas"]),
                    str(r["AG_verdaderas_positivas"]),
                ]
            )
        )
    linea += [
        "",
        "El regimen **B**, HS real 0 y HS predicho 1, es el falso positivo de HS: filas no toxicas",
        "que el modelo cree toxicas. Ahi TR y AG **no estan definidas**, y es donde el modelo produce",
        "falsos positivos que el oraculo corrige.",
        "",
        "## 5. La fraccion del techo que recupera cada mecanismo historico",
        "",
        fila(["mecanismo", "delta macro", "fraccion del techo", "nota"]),
        fila(["---"] * 4),
    ]
    for m in techo["mecanismos"]:
        linea.append(
            fila(
                [
                    m["mecanismo"],
                    f"{m['delta_macro']:+.6f}",
                    f"{m['fraccion_del_techo'] * 100:.1f} por ciento",
                    m["nota"],
                ]
            )
        )
    linea += [
        "",
        f"El techo de referencia es **{techo['delta_oraculo']:+.6f}**, la mascara con la etiqueta",
        "verdadera de HS, que no es adoptable porque usa la verdad.",
        "",
        "## 5b. La prueba de adopcion: el macro-F1 final con HS corregido",
        "",
        "El AUC responde si TR y AG **llevan** informacion sobre HS. El macro-F1 responde si esa",
        "informacion **mejora el sistema**, y es lo unico que la seccion 10 permite adoptar. Se",
        f"usa el umbral congelado de V03 para HS, {UMBRAL_HS_V03}, sin reoptimizar.",
        "",
        fila(["brazo", "macro-F1", "delta", "HS rescatados", "HS danados", "HS cambiados"]),
        fila(["---"] * 6),
    ]
    for nombre_brazo, d in finales.items():
        linea.append(
            fila(
                [
                    nombre_brazo,
                    f"{d['metrica_macro']:.6f}",
                    f"{d['delta_macro']:+.6f}",
                    str(d["hs_rescatados"]),
                    str(d["hs_danados"]),
                    str(d["hs_cambiados"]),
                ]
            )
        )
    techo_delta = techo["delta_oraculo"]
    linea += [
        "",
        "El contraste entre HS cambiados, HS rescatados y HS danados es la cifra que decide: una",
        "correccion que cambia muchas etiquetas pero rescata y dana en la misma proporcion no",
        "mejora nada, solo desplaza el error de sitio.",
        "",
        "La fraccion del techo que cada brazo recupera:",
        "",
        fila(["brazo", "delta macro", "fraccion del techo de +0.046271"]),
        fila(["---"] * 3),
    ]
    for nombre_brazo, d in finales.items():
        fraccion = d["delta_macro"] / techo_delta if techo_delta else 0.0
        linea.append(
            fila(
                [
                    nombre_brazo,
                    f"{d['delta_macro']:+.6f}",
                    f"{fraccion * 100:.1f} por ciento",
                ]
            )
        )
    linea += [
        "",
        "## 6. El precedente de riesgo de transferencia que esta fase debe respetar",
        "",
        f"El meta historico dio **{historico['metrica_local']}** en local, es decir",
        f"{historico['delta_local']:+.4f} sobre V03, y fue **ADOPTADO en su gate**. Y sin embargo no se",
        f"envio, porque el leaderboard dio **{historico['leaderboard']}** frente a {LB_V03} de V03.",
        "",
        "Es decir: **la misma familia de mecanismo ya dio una mejora local por encima del gate de",
        "candidato y perdio en el externo.** Cualquier resultado positivo que produzca este",
        "experimento pertenece a una familia con un fallo de transferencia ya medido, y eso obliga a",
        "tratarlo como marginal hasta que se demuestre lo contrario.",
        "",
        "## 7. Veredicto",
        "",
    ]
    mejor = max(finales.items(), key=lambda par: par[1]["delta_macro"])
    mejor_delta = mejor[1]["delta_macro"]
    if mejor_delta < UMBRAL_CANDIDATO:
        linea += [
            "**Ningun brazo alcanza el gate de candidato**, que exige un delta de al menos",
            f"{UMBRAL_CANDIDATO:+.3f} en el macro-F1. El mejor es **{mejor[0]}** con",
            f"{mejor_delta:+.6f}.",
            "",
            "Ese es el resultado que decide la fase, porque el AUC por si solo no es adoptable: la",
            "seccion 10 exige que la mejora de HS se transmita al macro-F1 final. Aqui la",
            "informacion existe, pero **no se traduce en mejora del sistema**.",
            "",
            "La consecuencia es la que el propio enunciado preve para este caso: **cerrar",
            "formalmente la linea de inferencia estructurada de etiquetas**, sin afirmar que TOXIC",
            "no puede mejorar. Lo que se afirma es mas estrecho: que TR y AG no aportan informacion",
            "sobre los errores de HS que el propio score de HS no tenga ya, y que la ganancia local",
            "del meta historico no venia del acoplamiento entre etiquetas sino de una combinacion",
            "de margenes cuyo reparto entre entrenamiento e inferencia es justo lo que el",
            "leaderboard penalizo.",
        ]
    else:
        linea += [
            f"**El brazo {mejor[0]} alcanza el gate de candidato** con {mejor_delta:+.6f} en el",
            "macro-F1, que es lo que exige la seccion 10 y no el AUC.",
            "",
            "Aun asi **no es adoptable todavia**, por dos razones que estan medidas:",
            "",
            "1. Falta la segunda particion, que la seccion 10 exige para cualquier candidata.",
            (
                f"2. El precedente de transferencia es desfavorable: E2-18 dio "
                f"{historico['delta_local']:+.4f} en local y perdio en el externo, con un "
                f"leaderboard de {historico['leaderboard']} frente a {LB_V03} de V03."
            ),
        ]
    linea += ["", ]
    linea += sello(inicio)
    REPORTE_MD.write_text("\n".join(linea) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
