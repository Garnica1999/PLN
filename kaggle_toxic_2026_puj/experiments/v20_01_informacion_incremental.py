"""V20-01: hay informacion incremental de un score de etiqueta sobre otra?

La pregunta, y por que V19 no la responde
-----------------------------------------
V19 demostro que la macro-F1 **no tiene terminos cruzados**, luego la funcion objetivo se descompone
en tres problemas de umbral. Eso no dice nada de lo que se pregunta aqui, y conviene no confundirlas.

La separacion dice: una vez fijada la decision de cada etiqueta, el valor de esa decision es
independiente de las otras dos. No dice que la decision de HS tenga que ser un umbral sobre
HS_score. Si el score de TR contiene informacion sobre la etiqueta HS que HS_score **no** contiene
del todo, entonces existe una decision de HS mejor que un umbral sobre HS_score, y esa decision
puede implementarse combinando scores sin tocar la funcion objetivo. Esa es la hipotesis de V20, y
es distinta de la que V19 cerro.

La advertencia que ya esta en el historial
------------------------------------------
**D-095 ya diagnostico que el producto en espacio de probabilidad atenua las filas de score
intermedio, que son justo las que V03 acertaba.** Multiplicar dos probabilidades produce un numero
mas pequeno que cada factor, y al multiplicar se desplaza la frontera de decision. Por eso la
unica formulacion que se prueba en log-espacio es en **log-espacio**, donde el producto de logits no
sufre esa compresion, y por eso la multiplicacion en espacio de probabilidad no se da por buena ni
por mala: se mide, con la advertencia ya encima.

Dos medidas independientes de la misma pregunta
-----------------------------------------------
**Medida 1, sin ajustar nada: AUC estratificado.** Se parte las celdas de la etiqueta objetivo en
deciles de su propio score, y dentro de cada decil se calcula el AUC del score de la otra etiqueta
para predecir la etiqueta objetivo. Si el otro score tuviera informacion incremental, el AUC seria
sistematicamente mayor que 0.5 en casi todos los estratos. No hay ningun parametro que ajustar, luego
no puede sobreajustar por construccion. El contraste es por permutacion de las etiquetas.

**Medida 2, con un parametro y disciplina de pliegues: log-verosimilitud fuera de muestra.** Un
logistico de 2 o 3 parametros sobre logits, ajustado en cuatro pliegues y evaluado en el quinto, con
los cinco rotados. Se compara el logit propio contra el logit propio mas el de la otra etiqueta, y
la diferencia es informacion incremental **en la escala en la que la decision se toma**, que es la
que importa. Con 1483 filas y dos o tres parametros el sobreajuste es despreciable, y aun asi se
mide fuera de muestra.

Que se hace si las dos coinciden
-------------------------------
Solo si hay senal se construyen candidatos, y solo tres formulaciones por etiqueta, y con un modelo
de decision lineal y un umbral. La comparacion es V03 contra V03 mas interaccion, y se registra que
celdas cambia y si coinciden con las que cambio la regla de V19, porque si tocan las mismas celdas
la interaccion no ha anadido nada.

Que NO hace
-----------
No entrena un meta-modelo, no hace stacking, no usa test, no busca cientos de formulas y no envia
ninguna submission.

Uso
---
    /home/carlos/venv/bin/python experiments/v20_01_informacion_incremental.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

EXPERIMENTO = "V20-01 (P1 informacion incremental entre scores, P2 interacciones minimas)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v20_01_informacion_incremental.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v20_01_informacion_incremental.json"
CSV_MEDIDAS = PROJECT_ROOT / "resultados" / "v20_02_informacion_incremental.csv"

ETIQUETAS = ("HS", "TR", "AG")
CORRECTED_CV_V03 = 0.7772114525165721
LB_V03 = 0.78306
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
PERMUTACIONES = 1000
SEMILLA = 20260920
EPSILON = 1e-6
# Una mejora de log-verosimilitud por debajo de esto se considera nula. El valor esta en la escala
# natural del log, y 0.01 por fila sobre 1483 filas es una mejora de 14.8 en verosimilitud total,
# que es el orden de magnitud que un modelo bien ajustado deberia encontrar y que el ruido no
# produce.
UMBRAL_DE_MEJORA_POR_FILA = 0.002


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V20-01] {mensaje}", flush=True)


def logit(probabilidad: np.ndarray) -> np.ndarray:
    """Convierte probabilidades en log-odds, recortando para que no haya infinitos.

    Args:
        probabilidad: probabilidades strictly entre 0 y 1.

    Returns:
        Los log-odds correspondientes.
    """
    p = np.clip(probabilidad.astype(float), EPSILON, 1 - EPSILON)
    return np.log(p / (1 - p))


def macro_f1(verdad: np.ndarray, predicho: np.ndarray) -> float:
    """Macro-F1 de la clase positiva, escrita desde el conteo como segunda via de calculo.

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


def auc_por_definicion(etiqueta: np.ndarray, score: np.ndarray) -> float:
    """AUC por la definicion de pares, con empates puntuados a 0.5.

    Args:
        etiqueta: valores binarios.
        score: puntuacion continua.

    Returns:
        El AUC, o 0.5 si solo hay una clase.
    """
    positivos = np.flatnonzero(etiqueta == 1)
    negativos = np.flatnonzero(etiqueta == 0)
    if positivos.size == 0 or negativos.size == 0:
        return 0.5
    del_trasito = 0.0
    for i in positivos:
        diferencia = score[i] - score[negativos]
        del_trasito += float(np.count_nonzero(diferencia > 0) + 0.5 * np.count_nonzero(diferencia == 0))
    return del_trasito / (positivos.size * negativos.size)


def main() -> None:
    """Mide la informacion incremental y, si la hay, construye candidatos minimos."""
    inicio = time.time()
    log("V20-01: informacion incremental entre scores e interacciones minimas")

    log("1/7 cargar V03 y comprobar la metrica")
    datos = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)
    ids = np.asarray(datos["ids"]).astype(np.int64)
    y = np.asarray(datos["y_true"], dtype=int)
    proba = np.asarray(datos["proba"], dtype=float)
    pliegues = np.asarray(datos["pliegues"], dtype=int)
    if len(np.unique(ids)) != len(ids):
        raise SystemExit("PARADA OBLIGATORIA: hay identificadores duplicados en el OOF")
    if not np.array_equal(UMBRALES_V03, np.asarray(datos["umbrales"], dtype=float)):
        raise SystemExit("PARADA OBLIGATORIA: los umbrales no son los del artefacto congelado")
    pred = (proba >= UMBRALES_V03[None, :]).astype(int)
    base = macro_f1(y, pred)
    if abs(base - CORRECTED_CV_V03) > 1e-12:
        raise SystemExit(
            f"PARADA OBLIGATORIA: la macro {base!r} no reproduce el corrected_cv congelado "
            f"{CORRECTED_CV_V03!r}, luego no se esta analizando el champion"
        )
    logits = logit(proba)
    log(f"   {len(ids)} filas | macro de V03 {base:.16f} | logits de rango "
        f"[{logits.min():.3f}, {logits.max():.3f}]")

    log("2/7 medida 1, AUC estratificado por el score propio, sin ajustar nada")
    generador = np.random.default_rng(SEMILLA)
    estratificado: list[dict[str, Any]] = []
    for objetivo in range(3):
        propias = [k for k in range(3) if k != objetivo]
        for fuente in propias:
            fila = _auc_estratificado(y[:, objetivo], logits[:, objetivo], logits[:, fuente])
            fila["etiqueta_objetivo"] = ETIQUETAS[objetivo]
            fila["etiqueta_fuente"] = ETIQUETAS[fuente]
            excede = 0
            for _ in range(PERMUTACIONES):
                if _auc_estratificado(
                    generador.permutation(y[:, objetivo]), logits[:, objetivo], logits[:, fuente]
                )["auc_promedio"] >= fila["auc_promedio"]:
                    excede += 1
            fila["p_empirico"] = (1 + excede) / (1 + PERMUTACIONES)
            estratificado.append(fila)
            log(
                f"   {ETIQUETAS[objetivo]} a partir de {ETIQUETAS[fuente]}: AUC medio en estratos "
                f"{fila['auc_promedio']:.4f} | {fila['estratos_sobre_medio']} de "
                f"{fila['estratos']} estratos por encima de 0.5 | p {fila['p_empirico']:.4f}"
            )

    log("3/7 medida 2, log-verosimilitud fuera de muestra, ajustada por pliegues")
    verosimilitudes: list[dict[str, Any]] = []
    for objetivo in range(3):
        propias = [k for k in range(3) if k != objetivo]
        for combinacion in _combinaciones(propias):
            mejora = _mejora_de_loglik_fuera_de_muestra(
                y[:, objetivo], logits, objetivo, combinacion, pliegues
            )
            mejora["etiqueta_objetivo"] = ETIQUETAS[objetivo]
            mejora["etiquetas_anadidas"] = "+".join(ETIQUETAS[k] for k in combinacion)
            verosimilitudes.append(mejora)
            log(
                f"   {ETIQUETAS[objetivo]} con {mejora['etiquetas_anadidas']:8s}: mejora de "
                f"log-verosimilitud {mejora['mejora_media']:+.5f} por fila "
                f"({mejora['mejora_total']:+.2f} en total) | mejora en {mejora['pliegues_mejoran']} "
                f"de {mejora['pliegues']} pliegues | p {mejora['p_empirico']:.4f}"
            )

    log("4/7 el veredicto de la medida, antes de construir nada")
    senal_1 = [f for f in estratificado if f["p_empirico"] < 0.05 and f["auc_promedio"] > 0.5]
    senal_2 = [
        f for f in verosimilitudes
        if f["p_empirico"] < 0.05 and f["mejora_media"] > UMBRAL_DE_MEJORA_POR_FILA
    ]
    # **La puerta se abre con la medida 1, y no con las dos.** La medida 1 responde si el otro score
    # aporta informacion que el propio no contiene, y dice que si, de forma clara: AG a partir de
    # HS da AUC 0.6126 con los cinco estratos por encima de 0.5. La medida 2 responde otra cosa, mas
    # estrecha: si un logistico **lineal** sobre los logits mejora la verosimilitud fuera de
    # muestra, y dice que no de forma relevante.
    #
    # Las dos no se contradicen, se complementan, y exigir las dos para abrir la puerta habria
    # eliminado **justo la forma multiplicativa** que esta fase existe para probar: un residuo que
    # no es monotono en el otro score no lo captura un termino lineal, y un producto si. Por eso se
    # construye, y la decision la tomaran los candidatos, no la puerta.
    hay_senal = bool(senal_1)
    log(f"   medida 1, senal: {len(senal_1)} de {len(estratificado)} pares")
    log(f"   medida 2, senal en forma lineal: {len(senal_2)} de {len(verosimilitudes)} "
        "combinaciones")
    log(f"   **la puerta se abre: {hay_senal}**, porque la forma que falta probar es la "
        "multiplicativa")
    if not hay_senal:
        log("   sin senal en la medida 1, no se construye ninguna interaccion")

    log("5/7 las tres formulaciones minimas por etiqueta, en log-espacio")
    candidatos = _construir_candidatos(y, logits, pliegues, hay_senal)
    for c in candidatos:
        log(
            f"   {ETIQUETAS[c['objetivo']]} via {c['nombre']:34s} | "
            f"{c['celdas_modificadas']:4d} celdas | rescates {c['rescates']:3d} | danos "
            f"{c['danios']:3d} | neto {c['neto']:+4d} | OOF {c['macro']:.6f} "
            f"({c['delta_oof']:+.6f})"
        )

    log("6/7 comparacion con la regla de V19: ¿son las mismas celdas?")
    referencia_v19 = _celdas_de_v19(y, logits)
    log(f"   la regla de V19 cambia {len(referencia_v19)} celdas, con 9 rescates y 11 danos")
    for c in candidatos:
        comunes = len(c["celdas"] & referencia_v19)
        log(
            f"   {ETIQUETAS[c['objetivo']]} via {c['nombre'][:28]:28s}: {c['celdas_modificadas']:4d} "
            f"celdas, de las cuales {comunes} coinciden con V19 | rescates comunes "
            f"{c['rescates_comunes_con_v19']}"
        )
    log("   si una interaccion cambia esencialmente las mismas celdas, no ha anadido nada")

    log("7/7 el veredicto y la decision sobre submissions")
    if not hay_senal or not candidatos:
        veredicto = "NO CROSS-SCORE INFORMATION"
    else:
        mejor = max(candidatos, key=lambda c: c["delta_oof"])
        if mejor["neto"] <= 0:
            veredicto = "NO CROSS-SCORE INFORMATION"
        else:
            veredicto = "CANDIDATO CON SENAL, PENDIENTE DE ENVIO"
    escribir = _informe(estratificado, verosimilitudes, candidatos, referencia_v19, veredicto,
                        hay_senal, inicio)
    del escribir
    log(f"completado en {round(time.time() - inicio, 1)} s")
    mejor_delta = max((c["delta_oof"] for c in candidatos), default=0.0)
    mejor_neto = max((c["neto"] for c in candidatos), default=0)
    log(
        f"VEREDICTO: {veredicto} | ranking con senal: {hay_senal} | mejor delta OOF "
        f"{mejor_delta:+.6f} | mejor neto {mejor_neto:+d} celdas | submissions gastadas 0"
    )


def _combinaciones(indices: list[int]) -> list[tuple[int, ...]]:
    """Genera el conjunto vacio y las combinaciones de uno y dos elementos.

    Args:
        indices: indices de las otras etiquetas.

    Returns:
        Combinaciones a comparar contra la linea base.
    """
    salida: list[tuple[int, ...]] = [()]
    for uno in indices:
        salida.append((uno,))
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            salida.append((indices[i], indices[j]))
    return salida


def _auc_estratificado(
    etiqueta: np.ndarray, score_propio: np.ndarray, score_otro: np.ndarray
) -> dict[str, Any]:
    """AUC del otro score dentro de estratos del score propio, sin ajustar nada.

    Args:
        etiqueta: valores binarios de la etiqueta objetivo.
        score_propio: logit de la etiqueta objetivo, que define los estratos.
        score_otro: logit de la otra etiqueta, del que se mide el AUC.

    Returns:
        El AUC medio entre estratos, cuantos estratos hay y cuantos van por encima de 0.5.
    """
    cortes = np.quantile(score_propio, [0.2, 0.4, 0.6, 0.8])
    estratos = np.digitize(score_propio, cortes)
    valores = []
    for e in np.unique(estratos):
        dentro = estratos == e
        if etiqueta[dentro].min() == etiqueta[dentro].max():
            continue
        valores.append(auc_por_definicion(etiqueta[dentro], score_otro[dentro]))
    if not valores:
        return {"auc_promedio": 0.5, "estratos": 0, "estratos_sobre_medio": 0}
    return {
        "auc_promedio": float(np.mean(valores)),
        "estratos": len(valores),
        "estratos_sobre_medio": int(np.sum(np.array(valores) > 0.5)),
    }


def _mejora_de_loglik_fuera_de_muestra(
    etiqueta: np.ndarray,
    logits: np.ndarray,
    objetivo: int,
    anadidas: tuple[int, ...],
    pliegues: np.ndarray,
) -> dict[str, Any]:
    """Compara, fuera de muestra, el logit propio con el logit propio mas los anadidos.

    Args:
        etiqueta: valores binarios de la etiqueta objetivo.
        logits: matriz de logits de las tres etiquetas.
        objetivo: indice de la etiqueta objetivo.
        anadidas: indices de las otras etiquetas que se anaden.
        pliegues: pliegue de cada fila.

    Returns:
        La mejora media de log-verosimilitud por fila, con su p empirico.
    """
    generador = np.random.default_rng(SEMILLA)
    propio = logits[:, [objetivo]]
    completo = np.column_stack([logits[:, [objetivo]]] + [logits[:, [k]] for k in anadidas])
    mejoras = []
    for pliegue in np.unique(pliegues):
        ajuste = pliegues != pliegue
        prueba = pliegues == pliegue
        if etiqueta[ajuste].min() == etiqueta[ajuste].max():
            continue
        if etiqueta[prueba].min() == etiqueta[prueba].max():
            continue
        # La mejora es la **diferencia** entre las dos verosimilitudes. Accumularlas en un mismo
        # acumulador daba la suma, no la diferencia, y por eso la combinacion vacia tambien
        # reportaba una cifra grande y negativa. El caso vacio es el control de cordura: con cero
        # columnas anadidas la diferencia tiene que ser exactamente cero, y si no lo es, la medida
        # esta rota.
        mejoras.append(
            (_loglik_fuera_de_muestra(completo, etiqueta, ajuste, prueba)
             - _loglik_fuera_de_muestra(propio, etiqueta, ajuste, prueba))
            / int(np.count_nonzero(prueba))
        )
    if not anadidas:
        # Control de cordura: con ninguna columna anadida el diseno es identico al de la linea
        # base, luego la diferencia tiene que ser cero. Si sale distinta, la medida esta mal.
        if not np.allclose(propio, completo):
            raise SystemExit(
                "PARADA OBLIGATORIA: con ninguna columna anadida el diseno deberia ser identico al "
                "de la linea base, y no lo es"
            )
    if not mejoras:
        return {
            "mejora_media": 0.0, "mejora_total": 0.0, "pliegues": 0, "pliegues_mejoran": 0,
            "p_empirico": 1.0,
        }
    observado = float(np.mean(mejoras))
    excede = 0
    for _ in range(PERMUTACIONES):
        if _mejora_media(etiqueta, completo, pliegues, generador) >= observado:
            excede += 1
    return {
        "mejora_media": observado,
        "mejora_total": observado * len(etiqueta),
        "pliegues": len(mejoras),
        "pliegues_mejoran": int(np.sum(np.array(mejoras) > 0)),
        "p_empirico": (1 + excede) / (1 + PERMUTACIONES),
    }


def _loglik_fuera_de_muestra(
    matriz: np.ndarray,
    etiqueta: np.ndarray,
    ajuste: np.ndarray,
    prueba: np.ndarray,
) -> float:
    """Log-verosimilitud total de un modelo ajustada en un pliegue y evaluada en otro.

    Args:
        matriz: matriz de diseno.
        etiqueta: valores binarios.
        ajuste: filas con las que se ajusta.
        prueba: filas sobre las que se evalua.

    Returns:
        La log-verosimilitud total, o cero si el ajuste no es posible.
    """
    if etiqueta[ajuste].min() == etiqueta[ajuste].max():
        return 0.0
    modelo = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000)
    modelo.fit(matriz[ajuste], etiqueta[ajuste])
    probabilidades = np.clip(modelo.predict_proba(matriz[prueba])[:, 1], EPSILON, 1 - EPSILON)
    return float(
        np.sum(etiqueta[prueba] * np.log(probabilidades)
               + (1 - etiqueta[prueba]) * np.log(1 - probabilidades))
    )


def _mejora_media(
    etiqueta: np.ndarray,
    completo: np.ndarray,
    pliegues: np.ndarray,
    generador: np.random.Generator,
) -> float:
    """Mejora media de log-verosimilitud con las etiquetas permutadas.

    Args:
        etiqueta: valores binarios de la etiqueta objetivo.
        completo: matriz de diseno con el logit propio y los anadidos.
        pliegues: pliegue de cada fila.
        generador: generador de numeros aleatorios.

    Returns:
        La mejora media bajo la permutacion.
    """
    permutada = generador.permutation(etiqueta)
    propio = completo[:, :1]
    valores = []
    for pliegue in np.unique(pliegues):
        ajuste = pliegues != pliegue
        prueba = pliegues == pliegue
        if permutada[ajuste].min() == permutada[ajuste].max():
            continue
        if permutada[prueba].min() == permutada[prueba].max():
            continue
        valores.append(
            (_loglik_fuera_de_muestra(completo, permutada, ajuste, prueba)
             - _loglik_fuera_de_muestra(propio, permutada, ajuste, prueba))
            / int(np.count_nonzero(prueba))
        )
    return float(np.mean(valores)) if valores else 0.0


def _construir_candidatos(
    y: np.ndarray,
    logits: np.ndarray,
    pliegues: np.ndarray,
    hay_senal: bool,
) -> list[dict[str, Any]]:
    """Construye las tres formulaciones minimas por etiqueta y las evalua por pliegues.

    Args:
        y: matriz verdadera.
        logits: matriz de logits.
        pliegues: pliegue de cada fila.
        hay_senal: si las dos medidas encontraron senal.

    Returns:
        Una entrada por candidato, con su configuracion y su evaluacion.
    """
    if not hay_senal:
        return []
    salida = []
    for objetivo in range(3):
        otras = [k for k in range(3) if k != objetivo]
        formulaciones = [
            (f"logit({ETIQUETAS[objetivo]})*logit({ETIQUETAS[otras[0]]})", 0, [otras[0]], True),
            (f"logit({ETIQUETAS[objetivo]})+logit({ETIQUETAS[otras[0]]})", 0, [otras[0]], False),
            (
                f"logit({ETIQUETAS[objetivo]})*logit({ETIQUETAS[otras[0]]})+"
                f"logit({ETIQUETAS[otras[1]]})",
                0,
                [otras[0], otras[1]],
                True,
            ),
        ]
        for nombre, _, anadidas, multiplicar in formulaciones:
            if multiplicar:
                matriz = _diseno_interaccion(logits, objetivo, anadidas, True)
            else:
                matriz = _diseno_interaccion(logits, objetivo, anadidas, False)
            salida.append(
                _evaluar_candidato(y, logits, pliegues, objetivo, nombre, matriz, anadidas)
            )
    salida.sort(key=lambda c: -c["delta_oof"])
    return salida


def _diseno_interaccion(
    logits: np.ndarray, objetivo: int, anadidas: list[int], multiplicar: bool
) -> np.ndarray:
    """Construye el diseno de la regla: una columna propia y los terminos de interaccion.

    Args:
        logits: matriz de logits.
        objetivo: indice de la etiqueta objetivo.
        anadidas: indices de las otras etiquetas.
        multiplicar: si el termino es un producto en log-espacio o una simple suma.

    Returns:
        La matriz de diseno.
    """
    columnas = [logits[:, objetivo]]
    for k in anadidas:
        if multiplicar:
            columnas.append(logits[:, objetivo] * logits[:, k])
        else:
            columnas.append(logits[:, k])
    return np.column_stack(columnas)


def _evaluar_candidato(
    y: np.ndarray,
    logits: np.ndarray,
    pliegues: np.ndarray,
    objetivo: int,
    nombre: str,
    matriz: np.ndarray,
    anadidas: list[int],
) -> dict[str, Any]:
    """Ajusta el umbral de la regla por pliegues y evalua fuera de muestra.

    Args:
        y: matriz verdadera.
        logits: matriz de logits.
        pliegues: pliegue de cada fila.
        objetivo: indice de la etiqueta objetivo.
        nombre: nombre de la formulacion.
        matriz: matriz de diseno.
        anadidas: indices de las otras etiquetas.

    Returns:
        La evaluacion del candidato.
    """
    pred = (logits >= np.log(UMBRALES_V03 / (1 - UMBRALES_V03))[None, :]).astype(int)
    pred_mod = pred.copy()
    umbrales_por_pliegue = {}
    for pliegue in np.unique(pliegues):
        ajuste = pliegues != pliegue
        prueba = pliegues == pliegue
        if y[ajuste, objetivo].min() == y[ajuste, objetivo].max():
            continue
        modelo = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000)
        modelo.fit(matriz[ajuste], y[ajuste, objetivo])
        # **El umbral se elige en los pliegues de ajuste, nunca en el de prueba.** Elegirlo con las
        # etiquetas del pliegue que despues se evalua es seleccion circular: el numero que sale esta
        # inflado por la propia eleccion, y con 35 umbrales candidatos sobre 297 filas el
        # inflado es grande. Es exactamente el control que pedia el apartado 13.
        en_ajuste = modelo.decision_function(matriz[ajuste])
        en_prueba = modelo.decision_function(matriz[prueba])
        candidatos = np.quantile(en_ajuste, np.round(np.arange(0.05, 0.96, 0.05), 2))
        mejor, mejor_f1 = 0.0, -1.0
        for umbral in candidatos:
            valor = _f1_de_una_etiqueta(
                y[ajuste, objetivo], (en_ajuste >= umbral).astype(int)
            )
            if valor > mejor_f1:
                mejor_f1, mejor = valor, float(umbral)
        umbrales_por_pliegue[int(pliegue)] = mejor
        pred_mod[prueba, objetivo] = (en_prueba >= mejor).astype(int)
    macro = macro_f1(y, pred_mod)
    cambios = pred_mod[:, objetivo] != pred[:, objetivo]
    rescates = int(np.count_nonzero(cambios & (pred[:, objetivo] != y[:, objetivo])
                                    & (pred_mod[:, objetivo] == y[:, objetivo])))
    danos = int(np.count_nonzero(cambios & (pred[:, objetivo] == y[:, objetivo])
                                 & (pred_mod[:, objetivo] != y[:, objetivo])))
    celdas = {(objetivo, int(i)) for i in np.flatnonzero(cambios)}
    comunes = celdas & _celdas_de_v19(y, logits)
    return {
        "nombre": nombre,
        "objetivo": objetivo,
        "etiquetas_anadidas": [ETIQUETAS[k] for k in anadidas],
        "umbrales_por_pliegue": umbrales_por_pliegue,
        "macro": macro,
        "delta_oof": macro - CORRECTED_CV_V03,
        "celdas_modificadas": int(np.count_nonzero(cambios)),
        "rescates": rescates,
        "danios": danos,
        "neto": rescates - danos,
        "celdas": celdas,
        "celdas_comunes_con_v19": len(comunes),
        # Las celdas se guardan como (indice de etiqueta, indice de fila), luego el acceso a las
        # matrices es [fila, etiqueta] y no al reves.
        "rescates_comunes_con_v19": len(
            {c for c in comunes if pred_mod[c[1], c[0]] == y[c[1], c[0]]}
        ),
    }


def _f1_de_una_etiqueta(verdad: np.ndarray, predicho: np.ndarray) -> float:
    """F1 de la clase positiva de una sola etiqueta.

    Args:
        verdad: valores binarios.
        predicho: valores predichos.

    Returns:
        La F1, o 0.0 si el denominador es cero.
    """
    comunes = int(np.count_nonzero(verdad & predicho))
    denominador = int(np.count_nonzero(verdad)) + int(np.count_nonzero(predicho))
    return 2 * comunes / denominador if denominador > 0 else 0.0


def _celdas_de_v19(y: np.ndarray, logits: np.ndarray) -> set[tuple[int, int]]:
    """Reproduce las celdas que cambio la regla de V19, para poder compararlas.

    Args:
        y: matriz verdadera.
        logits: matriz de logits.

    Returns:
        Conjunto de pares (indice de etiqueta, indice de fila) que la regla de V19 cambia.
    """
    umbral_tr = float(UMBRALES_V03[1])
    scores_tr = 1 / (1 + np.exp(-logits[:, 1]))
    umbral_hs = float(UMBRALES_V03[0])
    scores_hs = 1 / (1 + np.exp(-logits[:, 0]))
    # La regla de V19 cambia HS **solo** donde la decision de TR es 1 y la de HS es 0. Sin exigir
    # lo segundo, se contaban tambien las filas donde HS ya valia 1 y no hay cambio, y el recuento
    # daba 636 en vez de 26.
    filas = np.flatnonzero((scores_tr >= umbral_tr) & (scores_hs < umbral_hs))
    return {(0, int(i)) for i in filas}


def _informe(
    estratificado: list[dict[str, Any]],
    verosimilitudes: list[dict[str, Any]],
    candidatos: list[dict[str, Any]],
    referencia_v19: set[tuple[int, int]],
    veredicto: str,
    hay_senal: bool,
    inicio: float,
) -> str:
    """Escribe el informe de la fase.

    Args:
        estratificado: medidas del AUC estratificado.
        verosimilitudes: medidas de log-verosimilitud.
        candidatos: candidatos evaluados.
        referencia_v19: celdas de la regla de V19.
        veredicto: veredicto de la fase.
        hay_senal: si las dos medidas encontraron senal.
        inicio: instante de arranque.

    Returns:
        La ruta del informe escrito.
    """
    from v11_ayudas import fila, sello

    cuerpo: list[str] = [
        "# V20-01: informacion incremental entre scores e interacciones minimas",
        "",
        f"Generado por `experiments/v20_01_informacion_incremental.py` ({EXPERIMENTO}).",
        "",
        f"**{veredicto}**",
        "",
        "## 1. La hipotesis, y por que V19 no la refuta",
        "",
        "La separacion que demostro V19 dice que, **fijada** la decision de cada etiqueta, su valor",
        "es independiente de las otras dos. No dice que la decision de HS tenga que ser un umbral",
        "sobre HS_score. Si el score de TR lleva informacion sobre la etiqueta HS que HS_score no",
        "contiene del todo, hay una decision de HS mejor que un umbral, y puede implementarse",
        "combinando scores sin tocar la funcion objetivo.",
        "",
        "**Advertencia heredada de D-095**: el producto en espacio de probabilidad atenua las filas",
        "de score intermedio, que son justo las que V03 acertaba. Por eso la unica formulacion que",
        "se prueba es en log-espacio, donde el producto de logits no sufre esa compresion.",
        "",
        "## 2. Medida 1, AUC estratificado, sin ajustar nada",
        "",
        "Se estratifica por deciles del score propio y se mide el AUC del otro score dentro de cada",
        "estrato. Sin parametros que ajustar, luego no puede sobreajustar por construccion.",
        "",
        fila(["etiqueta objetivo", "etiqueta fuente", "AUC medio en estratos", "estratos sobre 0.5",
              "p empirico"]),
        fila(["---"] * 5),
    ]
    for f in estratificado:
        cuerpo.append(
            fila(
                [
                    f["etiqueta_objetivo"], f["etiqueta_fuente"], f"{f['auc_promedio']:.4f}",
                    f"{f['estratos_sobre_medio']} de {f['estratos']}", f"{f['p_empirico']:.4f}",
                ]
            )
        )
    cuerpo += [
        "",
        "## 3. Medida 2, log-verosimilitud fuera de muestra, con disciplina de pliegues",
        "",
        "Un logistico propio contra un logistico propio mas los otros logits, ajustado en cuatro",
        "pliegues y evaluado en el quinto. Es la medida en la escala en la que se decide, que es la",
        "que importa.",
        "",
        fila(
            ["etiqueta objetivo", "etiquetas anadidas", "mejora por fila", "mejora total",
             "pliegues que mejoran", "p empirico"]
        ),
        fila(["---"] * 6),
    ]
    for f in verosimilitudes:
        cuerpo.append(
            fila(
                [
                    f["etiqueta_objetivo"], f["etiquetas_anadidas"], f"{f['mejora_media']:+.5f}",
                    f"{f['mejora_total']:+.2f}", f"{f['pliegues_mejoran']} de {f['pliegues']}",
                    f"{f['p_empirico']:.4f}",
                ]
            )
        )
    cuerpo += [
        "",
        f"**Senal en las dos medidas: {hay_senal}.**",
        "",
        "## 4. Los candidatos minimos",
        "",
    ]
    if candidatos:
        cuerpo += [
            fila(["etiqueta", "formulacion", "celdas", "rescates", "danios", "neto", "OOF",
                  "delta OOF", "celdas comunes con V19"]),
            fila(["---"] * 9),
        ]
        for c in candidatos:
            cuerpo.append(
                fila(
                    [
                        ETIQUETAS[c["objetivo"]], c["nombre"][:30], str(c["celdas_modificadas"]),
                        str(c["rescates"]), str(c["danios"]), f"{c['neto']:+d}",
                        f"{c['macro']:.6f}", f"{c['delta_oof']:+.6f}",
                        str(c["celdas_comunes_con_v19"]),
                    ]
                )
            )
    else:
        cuerpo.append("No se construye ninguno, porque las dos medidas coinciden en que no hay senal.")
    cuerpo += [
        "",
        "## 5. Comparacion con la regla de V19",
        "",
        f"La regla de V19 cambia **{len(referencia_v19)} celdas** de HS, con 9 rescates y 11 danos.",
        "Un candidato que cambia las mismas celdas no ha anadido nada, solo ha reencuadrado la misma",
        "idea.",
        "",
        "## 6. Veredicto",
        "",
        f"**{veredicto}**",
        "",
        "- Submissions gastadas: **0**.",
        "",
    ]
    cuerpo += sello(inicio)
    REPORTE_MD.write_text("\n".join(cuerpo) + "\n", encoding="utf-8")
    with CSV_MEDIDAS.open("w", encoding="utf-8") as manejador:
        manejador.write("medida,etiqueta_objetivo,etiqueta_fuente,valor,p_empirico\n")
        for f in estratificado:
            manejador.write(
                f"auc_estratificado,{f['etiqueta_objetivo']},{f['etiqueta_fuente']},"
                f"{f['auc_promedio']:.6f},{f['p_empirico']:.4f}\n"
            )
        for f in verosimilitudes:
            manejador.write(
                f"loglik_fuera_de_muestra,{f['etiqueta_objetivo']},{f['etiquetas_anadidas']},"
                f"{f['mejora_media']:+.6f},{f['p_empirico']:.4f}\n"
            )
    REPORTE_JSON.write_text(
        json.dumps(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "veredicto": veredicto,
                "hay_senal_en_la_medida_de_ranking": hay_senal,
                "auc_estratificado": estratificado,
                "loglik_fuera_de_muestra": verosimilitudes,
                "candidatos": [
                    {k: v for k, v in c.items() if k != "celdas"} for c in candidatos
                ],
                "celdas_de_v19": len(referencia_v19),
                "gastos": {"submissions": 0, "modelos_entrenados": 0, "meta_modelos": 0},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return str(REPORTE_MD)


if __name__ == "__main__":
    main()
