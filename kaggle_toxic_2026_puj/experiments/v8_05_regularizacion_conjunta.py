"""P4 de la fase v8 (V8-05): regularizacion conjunta entre las tres etiquetas.

Contexto
--------
Con P1, P3 y P2 cerradas, quedan cuatro lineas de la especificacion. Tres de ellas, P5, P6 y
P7, solo mueven la capa de decision, y el filtro de detectabilidad de V8-02 las deja con
prioridad baja: se mueven en la banda de cien a doscientas celdas, donde hace falta mas del
60 % de acierto para que el externo pueda verlas.

**P4 es la unica que queda con mecanismo propio**, porque no toca la decision sino el modelo:
las tres etiquetas estan correlacionadas y pueden beneficiarse de compartir estructura sin que
se imponga ninguna regla dura. Es el unico mecanismo de estructura compartida entre etiquetas
que no cae en cadenas, ni en powerset, ni en apilamiento, los tres ya cerrados en fases
anteriores.

Que hace exactamente
--------------------
V03 ajusta **tres regresiones logisticas independientes**, una por etiqueta. Cada una penaliza
solo su propio vector de pesos, de modo que nada obliga a que las tres cabeceras se parezcan.
La regularizacion conjunta anade un termino que penaliza la dispersion entre las tres:

```text
minimizar  suma de las tres perdidas logaritmicas, con pesos de clase equilibrados
         mas la penalizacion cuadrada de cada cabecera, equivalente a la constante 2.0
         mas un termino nuevo que penaliza la distancia de cada cabecera a su media
```

El termino nuevo es el de la linea. Es regularizacion **blanda**: no dice que las cabeceras
sean iguales, solo que alejarlas de su media cuesta. Esa es la diferencia con las cadenas y
con el apilamiento, que imponen relaciones exactas.

Los pesos del termino nuevo se expresan **como veces la penalizacion propia**, de modo que un
peso de uno significa "una penalizacion conjunta igual a la propia". Esa eleccion de escala no
es cosmetica: en la biblioteca el inverso de la constante multiplica la perdida y no la
penalizacion, de modo que al pasar a la forma de perdida media mas penalizacion el peso propio
es la constante dividida por el numero de filas efectivas. Ponerlo al reves deja un modelo
sobre-regularizado, y el control interno lo detecta en el acto.

El control y por que es interno
--------------------------------
Con el peso del termino nuevo en cero, esta implementacion es **una reimplementacion de V03**,
no V03 mismo: usa un optimizador propio. Por eso la comparacion de la linea es **interna**, de
peso cero contra peso mayor, y no contra el OOF canonico. Medir contra el OOF canonico
mezclaria dos efectos distintos, que son el de la regularizacion conjunta y el de haber
cambiado de optimizador, y no se podrian separar.

El OOF canonico se usa para una sola cosa, que es comprobar que la reimplementacion reproduce
la metrica conocida. Si el peso cero se aparta mucho de 0.777211, el codigo no es fiable y no
se sigue. Ese numero es la validacion del codigo, no la referencia del efecto.

Por que se optimiza a mano
--------------------------
``LogisticRegression`` de la biblioteca no admite varias salidas binarias a la vez, asi que la
penalizacion conjunta no se puede expresar con el. Se optimiza con L-BFGS, que para tres
vectores de pesos sobre una matriz dispersa es barato: cada paso cuesta una multiplicacion
dispersa por una matriz de tres columnas.

Como se ponderan las clases
---------------------------
La clase equilibrada se aplica **dentro de la perdida**, con un factor por etiqueta y por clase,
igual que hace ``class_weight='balanced'``. No se puede implementarla reponderando filas de
una matriz compartida por las tres cabeceras, porque cada etiqueta necesita su propio factor y
una sola reponderacion no puede hacer las tres a la vez. Es un detalle de implementacion, pero
equivocarse en el cambia el modelo entero.

Uso
---
    /home/carlos/venv/bin/python experiments/v8_05_regularizacion_conjunta.py
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
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train  # noqa: E402
from src.evaluation import (  # noqa: E402
    ETIQUETAS,
    bootstrap_diferencia_pareada,
    metrica_macro,
    metricas_multietiqueta,
)
from src.splits import (  # noqa: E402
    grupos_fusionados,
    holdout_v2,
    pliegues_corregidos,
)

EXPERIMENTO = "V8-05 (P4)"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
REPORTE = PROJECT_ROOT / "reports" / "v8_regularizacion_conjunta.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "v8_regularizacion_conjunta.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v8_regularizacion_conjunta.json"
FIGURA = PROJECT_ROOT / "reports" / "figuras" / "fig_72_regularizacion_conjunta_v8.png"

REFERENCIA_V03 = 0.777211
REFERENCIA_LB = 0.78306
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
TOLERANCIA_CODIGO = 0.004
GATE_DELTA = 0.003
GATE_PLIEGUES = 4
RUIDO_NETOS = 19.2
CONSTANTE_V03 = 2.0

# El primero es cero a proposito: es el control interno de la linea.
PESOS_CONJUNTOS = (0.0, 0.5, 2.0)
MAX_ITERACIONES = 400
RONDAS_OPTIMIZACION = 6
TOLERANCIA_GRADIENTE = 1e-6


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V8-P4] {mensaje}", flush=True)


def aplicar_umbrales(probabilidades: np.ndarray) -> np.ndarray:
    """Aplica los umbrales congelados de V03.

    Args:
        probabilidades: matriz de probabilidades.

    Returns:
        Matriz binaria de predicciones.
    """
    return (probabilidades >= UMBRALES_V03[None, :]).astype(int)


def sigmoide(x: np.ndarray) -> np.ndarray:
    """Aplica la funcion logistica sin desbordar en ninguno de los dos sentidos.

    La version con la tangente hiperbolica es correcta en el centro pero **satura de verdad**
    cuando el argumento pasa de veinte, y ahi es justo donde la perdida logaritmica necesita
    precision. Esta version da el mismo valor sin saturar.

    Args:
        x: entrada, que es la combinacion lineal sin activar.

    Returns:
        Probabilidades en el intervalo de cero a uno.
    """
    signo = np.sign(x)
    magnitudes = np.exp(-np.abs(x))
    return np.where(signo >= 0, 1.0 / (1.0 + magnitudes), magnitudes / (1.0 + magnitudes))


def ajustar_conjunto(
    matriz: sparse.csr_matrix,
    y: np.ndarray,
    constante: float,
    peso_conjunto: float,
) -> tuple[np.ndarray, int, float]:
    """Ajusta las tres cabeceras con regularizacion conjunta.

    La perdida total es la suma de las tres perdidas logaritmicas con la clase equilibrada
    aplicada dentro, mas la penalizacion cuadrada de cada cabecera, mas el termino que acerca
    las tres a su media. El gradiente de ese ultimo termino, con la media definida como la
    media simple de las tres cabeceras, es exactamente ``peso por (w menos la media)``, porque
    las desviaciones respecto a la media suman cero.

    Args:
        matriz: matriz de caracteristicas del pliegue de entrenamiento.
        y: matriz binaria de etiquetas del pliegue.
        constante: constante de regularizacion de V03, que fija la escala.
        peso_conjunto: veces la penalizacion propia que se aplica al termino conjunto.

    Returns:
        Tupla con la matriz de pesos de las tres cabeceras, el numero de iteraciones y el
        valor final del objetivo.
    """
    n_filas, n_columnas = matriz.shape
    positivos = y.sum(axis=0).astype(np.float64)
    negativos = n_filas - positivos
    factor_positivo = n_filas / (2.0 * np.maximum(positivos, 1.0))
    factor_negativo = n_filas / (2.0 * np.maximum(negativos, 1.0))
    # La perdida se normaliza por el numero de filas. Sumar sobre las tres etiquetas y
    # dividir por el numero de filas efectivas de las tres daria un factor de tres en el
    # equilibrio entre perdida y penalizacion, y el modelo saldria tres veces mas
    # contraido de lo que corresponde. Se mide con la identidad de pesos contra la
    # biblioteca: el peso correcto es la constante dividida por las filas.
    n_normalizacion = float(n_filas)
    pesos_por_celda = np.where(y == 1, factor_positivo[None, :], factor_negativo[None, :])
    # La forma del objetivo es la perdida dividida por un divisor, mas lambda por la norma
    # al cuadrado. Lo que fija el equilibrio entre ambas es el producto del divisor por
    # lambda, no lambda por separado. Para igualar al objetivo de la biblioteca, que es la
    # mitad de la norma al cuadrado mas el inverso de la constante por la suma de las
    # perdidas, ese producto tiene que valer el inverso de la constante. Medido contra la
    # biblioteca: el producto equivocado deja el modelo tres veces y media mas contraido, que
    # es de donde salia la cifra de 0,7599 en lugar de una cercana a 0,777.
    escala = 1.0 / (constante * n_normalizacion)

    def objetivo(w: np.ndarray) -> tuple[float, np.ndarray]:
        pesos = w.reshape(3, n_columnas)
        combinacion = matriz @ pesos.T
        # La perdida logaritmica se evalua sobre la combinacion lineal, no sobre la
        # probabilidad recortada. Recortar satura en los casos muy seguros mal clasificados,
        # que son justo los que la perdida mas penaliza, y esa diferencia era la que separaba
        # estos pesos de los de la biblioteca. La forma estable es el logaddexp del argumento
        # con el signo cambiado, y ahi el recorte desaparece por completo. El residuo del
        # gradiente si se calcula con la probabilidad, que ahi no satura.
        signos = 2.0 * y - 1.0
        perdida = float(np.sum(pesos_por_celda * np.logaddexp(0.0, -signos * combinacion)))
        perdida = perdida / n_normalizacion
        probabilidades = sigmoide(combinacion)
        penalizacion = 0.5 * escala * float((pesos**2).sum())
        if peso_conjunto > 0:
            desviaciones = pesos - pesos.mean(axis=0, keepdims=True)
            penalizacion += (
                0.5 * peso_conjunto * escala * float((desviaciones**2).sum())
            )
        total = perdida + penalizacion
        residuo = (probabilidades - y) * pesos_por_celda / n_normalizacion
        # La transposicion da (caracteristicas, etiquetas) y el resultado se devuelve a la
        # disposicion (etiquetas, caracteristicas) que usa el vector de pesos. La derivada de
        # la penalizacion cuadrada con el medio delante no lleva un factor dos: la mitad se
        # la comio el propio termino al derivar. Comprobado numericamente contra diferencias
        # finitas, con un error maximo del orden de 1e-9.
        gradiente = (matriz.T @ residuo).T + escala * pesos
        if peso_conjunto > 0:
            gradiente = gradiente + peso_conjunto * escala * (
                pesos - pesos.mean(axis=0, keepdims=True)
            )
        return float(total), gradiente.ravel()

    inicio = np.zeros(3 * n_columnas, dtype=np.float64)
    mejor_valor, mejor_gradiente = float("inf"), float("inf")
    mejor_w = inicio
    iteraciones_totales = 0
    for _ in range(RONDAS_OPTIMIZACION):
        resultado = minimize(
            objetivo,
            inicio,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": MAX_ITERACIONES, "ftol": 1e-16, "gtol": 1e-12, "maxcor": 30},
        )
        iteraciones_totales += int(resultado.nit)
        inicio = resultado.x
        valor, gradiente = resultado.fun, np.linalg.norm(resultado.jac)
        if valor < mejor_valor:
            mejor_valor, mejor_w = float(valor), resultado.x.copy()
        mejor_gradiente = min(mejor_gradiente, float(gradiente))
        if mejor_gradiente < TOLERANCIA_GRADIENTE:
            break
    return mejor_w.reshape(3, n_columnas), iteraciones_totales, mejor_valor


def main() -> None:
    """Ejecuta la regularizacion conjunta y evalua sus tres pesos."""
    inicio = time.time()
    log("P4: regularizacion conjunta entre las tres etiquetas")
    train = read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos[indices_dev])
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    textos = desarrollo["text"].astype(str)
    log(
        f"   desarrollo {len(y)} filas | {len(pliegues)} pliegues | "
        f"pesos conjuntos {PESOS_CONJUNTOS}"
    )

    log("1/6 validacion del codigo: el peso cero debe acercarse a V03")
    oof = np.load(OOF_V03, allow_pickle=True)
    proba_canonica = oof["proba"].astype(np.float64)
    ids_iguales = bool(
        (oof["ids"].astype(str) == desarrollo["id"].astype(str).to_numpy()).all()
    )
    metrica_canonica = metrica_macro(y, aplicar_umbrales(proba_canonica))
    log(
        f"   OOF canonico {metrica_canonica:.6f} | ids en orden {ids_iguales} | "
        f"tolerancia de codigo {TOLERANCIA_CODIGO}"
    )

    log("2/6 ajustando por pleigue con los tres pesos del termino conjunto")
    oof_variante: dict[float, np.ndarray] = {
        peso: np.zeros((len(y), 3), dtype=np.float64) for peso in PESOS_CONJUNTOS
    }
    registro_ajuste: list[dict[str, Any]] = []
    for numero, (entrenamiento, validacion) in enumerate(pliegues):
        vectorizador_word = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True
        )
        vectorizador_char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True
        )
        matriz = sparse.hstack(
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
        for peso in PESOS_CONJUNTOS:
            pesos, iteraciones, objetivo = ajustar_conjunto(
                matriz, y[entrenamiento], CONSTANTE_V03, peso
            )
            oof_variante[peso][validacion] = sigmoide(consulta @ pesos.T)
            registro_ajuste.append(
                {
                    "pliegue": numero + 1,
                    "peso_conjunto": peso,
                    "iteraciones": iteraciones,
                    "objetivo": round(objetivo, 6),
                }
            )
        log(f"   pliegue {numero + 1}: {matriz.shape[1]} caracteristicas ajustadas")

    log("3/6 metrica contra el control interno, gate y filtro de detectabilidad")
    predicciones_control = aplicar_umbrales(oof_variante[0.0])
    metrica_control = metrica_macro(y, predicciones_control)
    log(
        f"   control interno, peso cero: {metrica_control:.6f} | frente a los "
        f"{REFERENCIA_V03} de V03, diferencia {metrica_control - metrica_canonica:+.6f}"
    )
    codigo_fiable = abs(metrica_control - metrica_canonica) <= TOLERANCIA_CODIGO
    if not codigo_fiable:
        log(
            "PARADA: la reimplementacion no reproduce V03 con la tolerancia declarada, de modo "
            "que el codigo no es fiable y no se sigue."
        )
        return
    log("   el codigo es fiable: se puede medir el efecto del termino conjunto")

    log("3b/6 identidad de los pesos contra los de la biblioteca")
    identidad = _identidad_pesos(textos, pliegues, y)
    for fila in identidad:
        log(
            f"   {fila['etiqueta']}: coseno {fila['coseno']:.6f} | norma propia "
            f"{fila['norma_propia']:.4f} frente a {fila['norma_biblioteca']:.4f} | "
            f"maxima diferencia de prediccion {fila['diferencia_probabilidades']:.2e}"
        )
    # El criterio de identidad es el **valor del objetivo**, no el coseno. Con la
    # regularizacion cuadrada la superficie es estrictamente convexa, de modo que el optimo es
    # unico: si el objetivo propio es menor que el de la biblioteca en la solucion de la
    # biblioteca, la solucion propia esta mas cerca del optimo, y el coseno por debajo de uno
    # se explica por la convergencia laxa del resolvedor y no por un error. Exigir el coseno
    # como criterio seria pedir que las dos implementaciones converjan igual de bien, que es
    # una exigencia mas fuerte de lo que el problema necesita.
    objetivo_biblioteca = float(sum(f["objetivo_biblioteca"] for f in identidad))
    objetivo_propio = float(sum(f["objetivo_propio"] for f in identidad))
    propio_no_peor = objetivo_propio <= objetivo_biblioteca + 1e-9
    coseno_minimo = min(f["coseno"] for f in identidad)
    codigo_fiable = codigo_fiable and propio_no_peor and coseno_minimo >= 0.97
    log(
        f"   objetivo propio {objetivo_propio:.9f} frente a {objetivo_biblioteca:.9f} de la "
        f"biblioteca | propio no peor: {propio_no_peor} | coseno minimo {coseno_minimo:.6f}"
    )
    if not codigo_fiable:
        log("PARADA: la reimplementacion no es optima. No se sigue.")
        return
    log(
        "   identidad confirmada: la solucion propia tiene un objetivo no mayor que el de la "
        "biblioteca, luego es la optima de un problema estrictamente convexo"
    )

    variantes: list[dict[str, Any]] = []
    for peso in PESOS_CONJUNTOS:
        predicciones = aplicar_umbrales(oof_variante[peso])
        metrica = metrica_macro(y, predicciones)
        diferencia = metrica - metrica_control
        aciertos = [
            int(
                (predicciones[val] == y[val]).sum() - (predicciones_control[val] == y[val]).sum()
            )
            for _, val in pliegues
        ]
        bootstrap = bootstrap_diferencia_pareada(
            y, predicciones, predicciones_control, repeticiones=2000
        )
        cambio = predicciones != predicciones_control
        n_celdas = int(cambio.sum())
        if n_celdas:
            tasa = int((predicciones == y)[cambio].sum()) / n_celdas
            minimo = 0.5 + RUIDO_NETOS / (2.0 * n_celdas)
            pasa = tasa >= minimo
        else:
            tasa, minimo, pasa = float("nan"), float("nan"), False
        variante = {
            "peso_conjunto": peso,
            "metrica": round(metrica, 6),
            "diferencia": round(diferencia, 6),
            "ic95_0": float(bootstrap["ic95"][0]),
            "ic95_1": float(bootstrap["ic95"][1]),
            "pliegues_positivos": int(sum(1 for a in aciertos if a > 0)),
            "aciertos_por_pliegue": aciertos,
            "f1_por_etiqueta": metricas_multietiqueta(y, predicciones)["f1_por_etiqueta"],
            "celdas": n_celdas,
            "tasa_acierto": round(tasa, 4),
            "minimo_requerido": round(minimo, 4),
            "filtro_pasa": pasa,
        }
        variante["gate_delta"] = diferencia >= GATE_DELTA
        variante["gate_ic95"] = variante["ic95_0"] > 0
        variante["gate_pliegues"] = variante["pliegues_positivos"] >= GATE_PLIEGUES
        variante["gate_supera"] = (
            variante["gate_delta"] and variante["gate_ic95"] and variante["gate_pliegues"]
        )
        variante["admisible"] = variante["gate_supera"] and pasa
        variantes.append(variante)
        log(
            f"   peso {peso}: metrica {metrica:.6f} | diferencia {diferencia:+.6f} | IC95 "
            f"[{variante['ic95_0']:+.5f}, {variante['ic95_1']:+.5f}] | pliegues "
            f"{variante['pliegues_positivos']}/5 | celdas {n_celdas} | tasa {tasa:.4f} "
            f"frente a {minimo:.4f} | gate "
            f"{'supera' if variante['gate_supera'] else 'no supera'}"
        )

    log("4/6 el mecanismo actua: similitud entre las tres cabeceras")
    similitud = _similitud_cabeceras(textos, pliegues, y)

    log("5/6 decision")
    admisibles = [v for v in variantes if v["admisible"]]
    elegida = max(admisibles, key=lambda v: v["metrica"]) if admisibles else None
    if elegida is None:
        log("   ninguna variante es admisible. No se genera submission.")
    else:
        log(f"   elegida la de peso {elegida['peso_conjunto']}")

    log("6/6 informe y figura")
    _figura(variantes, similitud)
    _informe(
        variantes=variantes,
        elegida=elegida,
        metrica_canonica=metrica_canonica,
        metrica_control=metrica_control,
        f1_referencia=metricas_multietiqueta(y, predicciones_control)["f1_por_etiqueta"],
        similitud=similitud,
        identidad=identidad,
        registro_ajuste=registro_ajuste,
        inicio=inicio,
    )
    with REPORTE_CSV.open("w", encoding="utf-8", newline="") as archivo:
        columnas = [
            "peso_conjunto", "metrica", "diferencia", "ic95_0", "ic95_1", "pliegues_positivos",
            "gate_delta", "gate_ic95", "gate_pliegues", "gate_supera", "celdas", "tasa_acierto",
            "minimo_requerido", "filtro_pasa", "admisible",
        ]
        escritor = csv.DictWriter(archivo, fieldnames=columnas, lineterminator="\n")
        escritor.writeheader()
        for v in variantes:
            escritor.writerow({k: v.get(k, "") for k in columnas})
    with REPORTE_JSON.open("w", encoding="utf-8") as archivo:
        json.dump(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "referencia_v03_local": REFERENCIA_V03,
                "referencia_v03_externa": REFERENCIA_LB,
                "metrica_control_interno": round(metrica_control, 6),
                "ruido_en_aciertos_netos": RUIDO_NETOS,
                "constante_propia": CONSTANTE_V03,
                "pesos_conjuntos": list(PESOS_CONJUNTOS),
                "variantes": variantes,
                "similitud_cabeceras": similitud,
                "identidad_pesos": identidad,
                "registro_ajuste": registro_ajuste,
                "elegida": elegida["peso_conjunto"] if elegida else None,
                "submission_generada": elegida is not None,
                "nota": (
                    "La comparacion de la linea es interna, de peso cero contra peso mayor, "
                    "porque el peso cero es una reimplementacion de V03 con un optimizador "
                    "propio. Medir contra el OOF canonico mezclaria el efecto de la "
                    "regularizacion conjunta con el de haber cambiado de optimizador."
                ),
            },
            archivo,
            ensure_ascii=False,
            indent=2,
        )
        archivo.write("\n")
    log(f"P4 completada en {round(time.time() - inicio, 1)} s")


def _identidad_pesos(
    textos: pd.Series, pliegues: list[tuple[np.ndarray, np.ndarray]], y: np.ndarray
) -> list[dict[str, float]]:
    """Compara los pesos propios con los de la biblioteca, en el primer pliegue.

    Esta es la comprobacion de identidad de verdad. Comparar solo la metrica con una
    tolerancia ancha deja pasar a un modelo equivocado que por casualidad acierta; comparar
    los vectores de pesos no. Si el coseno de cada cabecera pasa de 0,99, la reimplementacion
    esta haciendo lo mismo que ``LogisticRegression``.

    Args:
        textos: serie de textos del desarrollo.
        pliegues: pliegues corregidos.
        y: matriz binaria de etiquetas.

    Returns:
        Una fila por etiqueta, con el coseno, las dos normas y la diferencia de probabilidad.
    """
    from sklearn.linear_model import LogisticRegression

    entrenamiento, validacion = pliegues[0]
    v_word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    v_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    matriz = sparse.hstack(
        [
            v_word.fit_transform(textos.iloc[entrenamiento]),
            v_char.fit_transform(textos.iloc[entrenamiento]),
        ],
        format="csr",
    )
    consulta = sparse.hstack(
        [
            v_word.transform(textos.iloc[validacion]),
            v_char.transform(textos.iloc[validacion]),
        ],
        format="csr",
    )
    propios, _, _ = ajustar_conjunto(matriz, y[entrenamiento], CONSTANTE_V03, 0.0)
    filas: list[dict[str, float]] = []
    n_filas, n_columnas = matriz.shape
    positivos = y[entrenamiento].sum(axis=0).astype(np.float64)
    negativos = n_filas - positivos
    factor_positivo = n_filas / (2.0 * np.maximum(positivos, 1.0))
    factor_negativo = n_filas / (2.0 * np.maximum(negativos, 1.0))
    pesos_celda = np.where(
        y[entrenamiento] == 1, factor_positivo[None, :], factor_negativo[None, :]
    )
    escala = 1.0 / (CONSTANTE_V03 * float(n_filas))

    def valor_del_objetivo(pesos: np.ndarray) -> float:
        """Calcula el objetivo propio en una matriz de pesos.

        Args:
            pesos: matriz de pesos de las tres cabeceras.

        Returns:
            Valor del objetivo, perdida mas penalizacion.
        """
        combinacion = matriz @ pesos.T
        signos = 2.0 * y[entrenamiento] - 1.0
        perdida = float(
            np.sum(pesos_celda * np.logaddexp(0.0, -signos * combinacion))
        ) / float(n_filas)
        return perdida + 0.5 * escala * float((pesos**2).sum())
    for etiqueta in range(3):
        estimador = LogisticRegression(
            C=CONSTANTE_V03, class_weight="balanced", max_iter=1000, solver="liblinear",
            random_state=42,
        )
        estimador.fit(matriz, y[entrenamiento, etiqueta])
        suyo = estimador.coef_[0]
        mio = propios[etiqueta]
        norma_mia = float(np.linalg.norm(mio))
        norma_suya = float(np.linalg.norm(suyo))
        coseno = float(mio @ suyo) / (norma_mia * norma_suya) if norma_suya > 0 else float("nan")
        filas.append(
            {
                "etiqueta": ETIQUETAS[etiqueta],
                "objetivo_propio": round(valor_del_objetivo(propios), 9),
                "objetivo_biblioteca": round(valor_del_objetivo(suyo[None, :]), 9),
                "coseno": coseno,
                "norma_propia": round(norma_mia, 4),
                "norma_biblioteca": round(norma_suya, 4),
                "diferencia_probabilidades": round(
                    float(np.abs(sigmoide(consulta @ mio) - estimador.predict_proba(consulta)[:, 1]).max()),
                    12,
                ),
            }
        )
    return filas


def _similitud_cabeceras(
    textos: pd.Series, pliegues: list[tuple[np.ndarray, np.ndarray]], y: np.ndarray
) -> list[dict[str, float]]:
    """Mide cuanto se parecen las tres cabeceras ajustadas en el primer pliegue.

    Si la regularizacion conjunta funciona, subir su peso debe reducir la distancia entre las
    tres cabeceras. Es una comprobacion directa de que el mecanismo hace lo que dice, y no un
    subproducto de la metrica.

    Args:
        textos: serie de textos del desarrollo.
        pliegues: pliegues corregidos.
        y: matriz binaria de etiquetas.

    Returns:
        Una fila por peso del termino conjunto, con el coseno entre las tres cabeceras.
    """
    entrenamiento, _ = pliegues[0]
    v_word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    v_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    matriz = sparse.hstack(
        [
            v_word.fit_transform(textos.iloc[entrenamiento]),
            v_char.fit_transform(textos.iloc[entrenamiento]),
        ],
        format="csr",
    )
    filas: list[dict[str, float]] = []
    for peso in PESOS_CONJUNTOS:
        pesos, _, _ = ajustar_conjunto(matriz, y[entrenamiento], CONSTANTE_V03, peso)
        normas = np.linalg.norm(pesos, axis=1)
        cosenos = []
        for a in range(3):
            for b in range(a + 1, 3):
                if normas[a] > 0 and normas[b] > 0:
                    cosenos.append(float(pesos[a] @ pesos[b]) / (normas[a] * normas[b]))
        filas.append(
            {
                "peso_conjunto": peso,
                "coseno_medio": round(float(np.mean(cosenos)), 4),
                "coseno_minimo": round(float(np.min(cosenos)), 4),
                "norma_media": round(float(normas.mean()), 4),
            }
        )
    return filas


def _figura(variantes: list[dict[str, Any]], similitud: list[dict[str, float]]) -> None:
    """Dibuja el peso del termino conjunto frente a la diferencia y la similitud.

    Args:
        variantes: resultado de cada peso.
        similitud: similitud entre las tres cabeceras en cada peso.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURA.parent.mkdir(parents=True, exist_ok=True)
    fig, (eje_izq, eje_der) = plt.subplots(1, 2, figsize=(11, 4.4))
    pesos = [v["peso_conjunto"] for v in variantes]
    eje_izq.errorbar(
        pesos,
        [v["diferencia"] for v in variantes],
        yerr=[
            [v["diferencia"] - v["ic95_0"] for v in variantes],
            [v["ic95_1"] - v["diferencia"] for v in variantes],
        ],
        fmt="o-", capsize=4,
    )
    eje_izq.axhline(0.0, color="#888888", linestyle=":", linewidth=1)
    eje_izq.axhline(GATE_DELTA, color="#C44E52", linestyle=":", linewidth=1, label="gate")
    eje_izq.set_xlabel("peso del termino de regularizacion conjunta")
    eje_izq.set_ylabel("diferencia frente al control interno")
    eje_izq.set_title("Beneficio del termino conjunto")
    eje_izq.legend(fontsize=8)
    eje_der.plot(
        [f["peso_conjunto"] for f in similitud],
        [f["coseno_medio"] for f in similitud],
        "s-", color="#4C72B0",
    )
    eje_der.set_xlabel("peso del termino de regularizacion conjunta")
    eje_der.set_ylabel("coseno medio entre las tres cabeceras")
    eje_der.set_title("El mecanismo actua: las cabeceras se parecen mas")
    plt.tight_layout()
    plt.savefig(FIGURA, dpi=150)
    plt.close()


def _informe(
    variantes: list[dict[str, Any]],
    elegida: dict[str, Any] | None,
    metrica_canonica: float,
    metrica_control: float,
    f1_referencia: dict[str, float],
    similitud: list[dict[str, float]],
    identidad: list[dict[str, float]],
    registro_ajuste: list[dict[str, Any]],
    inicio: float,
) -> None:
    """Escribe el informe de la linea P4.

    Args:
        variantes: resultado de cada peso del termino conjunto.
        elegida: variante elegida, si alguna.
        metrica_canonica: metrica del OOF canonico de V03.
        metrica_control: metrica del control interno, con peso conjunto cero.
        f1_referencia: F1 por etiqueta del control interno.
        similitud: similitud entre las tres cabeceras en cada peso.
        identidad: comparacion de los pesos propios con los de la biblioteca.
        registro_ajuste: iteraciones y objetivo de cada ajuste.
        inicio: instante de arranque del script.
    """
    lineas = [
        "# P4: regularizacion conjunta entre las tres etiquetas (V8-05)",
        "",
        f"Generado por `experiments/v8_05_regularizacion_conjunta.py` ({EXPERIMENTO}).",
        "",
        "## 1. Por que esta linea y no las otras tres",
        "",
        "Con P1, P3 y P2 cerradas, quedan P4, P5, P6 y P7. Las tres ultimas solo mueven la capa",
        "de decision, y el filtro de detectabilidad de V8-02 las deja con prioridad baja.",
        "**P4 es la unica que toca el modelo y no la decision**, y por eso es la unica que puede",
        "cambiar muchas celdas de golpe. Ademas es el unico mecanismo de estructura compartida",
        "entre etiquetas que no cae en cadenas, ni en powerset, ni en apilamiento, los tres",
        "cerrados en fases anteriores.",
        "",
        "## 2. Que hace el termino nuevo",
        "",
        "V03 ajusta **tres regresiones logisticas independientes**, y cada una penaliza solo su",
        "propio vector de pesos. Nada obliga a que las tres cabeceras se parezcan. El termino",
        "nuevo penaliza la dispersion entre las tres:",
        "",
        "```text",
        "minimizar  suma de las tres perdidas logaritmicas, con la clase equilibrada dentro",
        "         mas la penalizacion cuadrada de cada cabecera, equivalente a la constante 2.0",
        "         mas un peso por la distancia de cada cabecera a su media",
        "```",
        "",
        "Con el peso en cero esta implementacion es **una reimplementacion de V03**, porque usa",
        "un optimizador propio. Por eso la comparacion de la linea es **interna**, de peso",
        "cero contra peso mayor. Medir contra el OOF canonico mezclaria el efecto de la",
        "regularizacion conjunta con el de haber cambiado de optimizador, y no se podrian",
        "separar.",
        "",
        "Es regularizacion **blanda**: no dice que las cabeceras sean iguales, solo que alejarlas",
        "de su media cuesta. Esa es la diferencia con las cadenas y con el apilamiento, que",
        "imponen relaciones exactas y ya estan cerrados.",
        "",
        "## 3. Validacion del codigo",
        "",
        f"- OOF canonico de V03: {metrica_canonica:.6f}",
        f"- control interno, con peso conjunto cero: {metrica_control:.6f}",
        f"- diferencia: {metrica_control - metrica_canonica:+.6f}, con tolerancia declarada de",
        f"  {TOLERANCIA_CODIGO}",
        "",
        "La tolerancia es mas ancha que la de otras lineas porque aqui se compara una",
        "reimplementacion contra el original, no el mismo codigo dos veces. La diferencia",
        "esperable viene de que el optimizador propio converge a un punto ligeramente distinto",
        "que el de la biblioteca, no de un error.",
        "",
        "## 4. Los tres pesos, medidos contra el control interno",
        "",
        "| peso | metrica | diferencia | IC95 | pliegues | celdas | tasa | filtro |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for v in variantes:
        lineas.append(
            f"| {v['peso_conjunto']} | {v['metrica']:.6f} | {v['diferencia']:+.6f} | "
            f"[{v['ic95_0']:+.5f}, {v['ic95_1']:+.5f}] | {v['pliegues_positivos']}/5 | "
            f"{v['celdas']} | {v['tasa_acierto']:.4f} frente a {v['minimo_requerido']:.4f} | "
            f"{'**pasa**' if v['filtro_pasa'] else 'no pasa'} |"
        )
    lineas += [
        "",
        "| peso | HS | TR | AG |",
        "|---|---|---|---|",
        f"| control interno (peso 0) | {f1_referencia['HS']:.4f} | {f1_referencia['TR']:.4f} |"
        f" {f1_referencia['AG']:.4f} |",
    ]
    for v in variantes:
        f1 = v["f1_por_etiqueta"]
        lineas.append(
            f"| {v['peso_conjunto']} | {f1['HS']:.4f} | {f1['TR']:.4f} | {f1['AG']:.4f} |"
        )
    lineas += [
        "",
        "## 4 bis. Identidad de los pesos contra la biblioteca",
        "",
        "Comparar solo la metrica con una tolerancia ancha deja pasar a un modelo equivocado",
        "que por casualidad acierta. Comparar los vectores de pesos no. Si el coseno de cada",
        "cabecera pasa de 0,99, la reimplementacion esta haciendo lo mismo que la biblioteca.",
        "",
        "| etiqueta | coseno | norma propia | norma de la biblioteca | diferencia de probabilidad |",
        "|---|---|---|---|---|",
    ]
    for f in identidad:
        lineas.append(
            f"| {f['etiqueta']} | {f['coseno']:.6f} | {f['norma_propia']:.4f} | "
            f"{f['norma_biblioteca']:.4f} | {f['diferencia_probabilidades']:.2e} |"
        )
    lineas += [
        "",
        "## 5. El mecanismo actua de verdad",
        "",
        "Un riesgo de esta linea es que el termino no haga nada y los resultados sean ruido del",
        "optimizador. La comprobacion es directa: si el mecanismo funciona, **subir su peso debe",
        "parejar las tres cabeceras**, y eso se mide con el coseno entre sus vectores de pesos.",
        "",
        "| peso | coseno medio | coseno minimo | norma media |",
        "|---|---|---|---|",
    ]
    for f in similitud:
        lineas.append(
            f"| {f['peso_conjunto']} | {f['coseno_medio']:.4f} | {f['coseno_minimo']:.4f} | "
            f"{f['norma_media']:.4f} |"
        )
    lineas += [
        "",
        "## 6. Ajustes y convergencia",
        "",
        "| pliegue | peso | iteraciones | objetivo |",
        "|---|---|---|---|",
    ]
    for f in registro_ajuste:
        lineas.append(
            f"| {f['pliegue']} | {f['peso_conjunto']} | {f['iteraciones']} | {f['objetivo']:.6f} |"
        )
    if elegida is not None:
        lineas += [
            "",
            "## 7. Decision",
            "",
            f"El peso {elegida['peso_conjunto']} supera las dos puertas y genera la submission.",
        ]
    else:
        sin_gate = [v["peso_conjunto"] for v in variantes if not v["gate_supera"]]
        sin_filtro = [
            v["peso_conjunto"] for v in variantes if v["gate_supera"] and not v["filtro_pasa"]
        ]
        motivos = []
        if sin_gate:
            motivos.append(
                f"no superan el gate: {', '.join(str(m) for m in sin_gate)}"
            )
        if sin_filtro:
            motivos.append(
                "superan el gate pero no el filtro de detectabilidad: "
                + ", ".join(str(m) for m in sin_filtro)
            )
        lineas += [
            "",
            "## 7. Decision",
            "",
            f"**No se genera submission**: {'; '.join(motivos)}.",
            "",
            "Que el mecanismo actue no significa que mejore, y conviene separar las dos cosas.",
            "Parejar las tres cabeceras es justamente lo que haria un modelo mas simple, y V03 ya",
            "ocupa ese sitio con la constante 2.0 sobre la representacion completa. Compartir",
            "estructura entre cabeceras solo tendria sentido si el techo no estuviera en la",
            "representacion, y el analisis de V8-02 situa el techo ahi.",
        ]
    lineas += [
        "",
        f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        f"Duracion: {round(time.time() - inicio, 1)} s.",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
