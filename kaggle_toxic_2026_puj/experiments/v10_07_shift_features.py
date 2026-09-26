"""V10-07: diagnostico de covariate shift entre entrenamiento y prueba, con discriminador de alta dimensionalidad.

Que se mide y por que
---------------------
V9-02 ya midio un **shift de prevalencia**: la prueba tiene aproximadamente un 0.94 de los
positivos del entrenamiento. Eso actua sobre la capa de decision y V10-05 comprobo que adapting
la capa a ese shift no ayuda.

Aqui se mide la otra mitad del problema, que es distinta: **shift de covariables**, o sea, si los
textos de la prueba ocupan una region diferente del espacio de caracteristicas que los de
entrenamiento. Si fuera asi, las probabilidades de V03 estarian calibradas para una poblacion
que no es la de evaluacion, y ninguna correccion de decision de serviria.

Como se mide
------------
Se entrena un **discriminador** que separa entrenamiento de prueba usando las mismas
caracteristicas TF-IDF que V03, y se mide su AUC **fuera de muestra**. Un AUC de 0.5 significa
que las dos poblaciones son indistinguibles. Un AUC alto significa que hay covariables desplazadas.

Detalles que importan para que la medicion sea honesta:

- El espacio de caracteristicas se ajusta **dentro de cada pliegue de entrenamiento**. Si se
  ajustase sobre el conjunto combinado, el discriminador veria las filas de validacion en el
  vocabulario y el AUC saldria inflado.
- La validacion cruzada es **agrupada por grupos** en las filas de entrenamiento, con los mismos
  grupos fusionados del proyecto, y por identificador unico en las filas de prueba. Asi un
  near-duplicado no aparece a la vez en entrenamiento y validacion del discriminador.
- La particion es **estratificada por la etiqueta de origen**, para que la proporcion de
 train y test sea estable en cada pliegue.

Ademas del AUC se reporta la **distribucion de w(x) = p_prueba / p_entrenamiento** sobre las
caracteristicas mas desplazadas, que es el cociente de densidades que permite ver la direccion del
desplazamiento y no solo su magnitud.

Limites, declarados antes de ver resultados
-------------------------------------------
Este discriminador es **solo diagnostico**. La regla 2 de la fase v10 prohibe trainer con la
prueba, y el cociente de densidades, si se usa, solo puede reponderar la validacion local. Por
mucho que el AUC salga alto, aqui no se entrena nada con test.

Gates de decision, pre-registrados antes de ejecutar
-----------------------------------------------------
- **AUC menor o igual que 0.55**: el shift de caracteristicas es despreciable, la via se
  **CIERRA** y queda solo la correccion de prevalencia de P1.
- **AUC entre 0.55 y 0.65**: shift moderado. El cociente de densidades se usa **solo** para
  reponderar la validacion local, y se reporta su efecto sobre la decision.
- **AUC mayor que 0.65**: shift fuerte. Se reporta y se propone un plan, sin entrenar con test.

Uso
---
    /home/carlos/venv/bin/python experiments/v10_07_shift_features.py
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

from src.data import read_test, read_train
from src.splits import RANDOM_STATE, grupos_fusionados

EXPERIMENTO = "V10-07 (P2, shift de features)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v10_07_shift_features.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v10_07_shift_features.json"
REPORTE_CSV = PROJECT_ROOT / "reports" / "v10_07_shift_features.csv"

PLIEGUES_DISCRIMINADOR = 5
SEMILLA_DISCRIMINADOR = 20260926
C_DISC = 1.0
TOP_CARACTERISTICAS = 20
UMBRAL_CERRAR = 0.55
UMBRAL_FUERTE = 0.65
FRACCION_POSITIVA_ESPERADA = 0.4

PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V10-07] {mensaje}", flush=True)


def construir_espacio(
    textos_entrenamiento: pd.Series, textos_completo: pd.Series
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray]:
    """Ajusta los vectorizadores en el entrenamiento y transforma el conjunto completo.

    Args:
        textos_entrenamiento: textos del pliegue de entrenamiento.
        textos_completo: textos de entrenamiento y prueba juntos.

    Returns:
        Tupla con la matriz de ajuste, la de consulta y los nombres de las caracteristicas en el
        orden en que las concatena el apilado, que es primero las de palabra y despues las de
        caracter.
    """
    vectorizador_word = TfidfVectorizer(**PARAMS_WORD)
    vectorizador_char = TfidfVectorizer(**PARAMS_CHAR)
    ajuste = sparse.hstack(
        [
            vectorizador_word.fit_transform(textos_entrenamiento),
            vectorizador_char.fit_transform(textos_entrenamiento),
        ],
        format="csr",
    )
    consulta = sparse.hstack(
        [
            vectorizador_word.transform(textos_completo),
            vectorizador_char.transform(textos_completo),
        ],
        format="csr",
    )
    nombres = np.concatenate(
        [
            vectorizador_word.get_feature_names_out(),
            vectorizador_char.get_feature_names_out(),
        ]
    )
    return ajuste, consulta, nombres


def auc_y_exactitud(
    y_verdadero: np.ndarray, puntuaciones: np.ndarray, umbral: float = 0.5
) -> tuple[float, float]:
    """Calcula el AUC por el metodo de los rangos y la exactitud en un umbral.

    Args:
        y_verdadero: etiquetas binarias, con 1 para la clase de interes.
        puntuaciones: puntuaciones continuas del discriminador.
        umbral: punto de corte para la exactitud.

    Returns:
        Tupla con el AUC y la exactitud.
    """
    orden = np.argsort(puntuaciones)
    rangos = np.empty(len(puntuaciones), dtype=np.float64)
    rangos[orden] = np.arange(1, len(puntuaciones) + 1, dtype=np.float64)
    positivos = y_verdadero == 1
    n_positivos = int(np.sum(positivos))
    n_negativos = len(y_verdadero) - n_positivos
    if n_positivos == 0 or n_negativos == 0:
        raise ValueError("el AUC necesita las dos clases presentes")
    suma_rangos = float(np.sum(rangos[positivos]))
    auc = (suma_rangos - n_positivos * (n_positivos + 1) / 2.0) / (n_positivos * n_negativos)
    exactitud = float(np.mean((puntuaciones >= umbral).astype(int) == y_verdadero))
    return float(auc), exactitud


def main() -> None:
    """Ejecuta el diagnostico de shift de features entre entrenamiento y prueba."""
    inicio = time.time()
    log("V10-07: diagnostico de shift de features con discriminador de alta dimensionalidad")

    train = read_train()
    test = read_test()
    log(f"   entrenamiento: {len(train)} filas | prueba: {len(test)} filas")

    textos = pd.concat([train["text"].astype(str), test["text"].astype(str)], ignore_index=True)
    origen = np.concatenate(
        [np.zeros(len(train), dtype=int), np.ones(len(test), dtype=int)]
    )
    grupos_entrenamiento = grupos_fusionados(train)
    grupos = np.concatenate(
        [grupos_entrenamiento, -np.arange(1, len(test) + 1, dtype=np.int64)]
    )
    log(
        f"   grupos: {len(np.unique(grupos_entrenamiento))} en entrenamiento y "
        f"{len(test)} identificadores unicos en prueba"
    )

    log(f"1/4 entrenar el discriminador con {PLIEGUES_DISCRIMINADOR} pliegues agrupados")
    particion = StratifiedGroupKFold(
        n_splits=PLIEGUES_DISCRIMINADOR, shuffle=True, random_state=SEMILLA_DISCRIMINADOR
    )
    puntuaciones = np.zeros(len(origen), dtype=np.float64)
    aucs: list[float] = []
    exactitudes: list[float] = []
    for entrenamiento, validacion in particion.split(textos, origen, groups=grupos):
        ajuste, consulta, _ = construir_espacio(
            textos.iloc[entrenamiento].reset_index(drop=True), textos
        )
        discriminador = LogisticRegression(
            C=C_DISC,
            class_weight="balanced",
            max_iter=3000,
            solver="liblinear",
            random_state=RANDOM_STATE,
        )
        discriminador.fit(ajuste, origen[entrenamiento])
        puntuaciones[validacion] = discriminador.predict_proba(consulta[validacion])[:, 1]
        auc, exactitud = auc_y_exactitud(origen[validacion], puntuaciones[validacion])
        aucs.append(auc)
        exactitudes.append(exactitud)
        log(f"   pliegue: AUC {auc:.6f} | exactitud {exactitud:.6f}")

    auc_global, exactitud_global = auc_y_exactitud(origen, puntuaciones)
    log(
        f"   AUC global fuera de muestra: {auc_global:.6f} | exactitud {exactitud_global:.6f} | "
        f"AUC por pliegue de {min(aucs):.6f} a {max(aucs):.6f}"
    )
    proporcion_test_real = float(np.mean(origen == 1))
    proporcion_test_predicha = float(np.mean(puntuaciones >= 0.5))
    log(
        f"   proporcion de prueba real {proporcion_test_real:.4f} frente a predicha por el "
        f"discriminador {proporcion_test_predicha:.4f}"
    )

    log("2/4 ajustar un discriminador sobre todo el conjunto para leer los coeficientes")
    # Ojo con la distincion entre los dos pasos: el AUC que decide el gate es el de arriba, que
    # es fuera de muestra. El discriminador de aqui se ajusta sobre todas las filas porque solo
    # sirve para leer en que direccion se desplaza la distribucion, y sus coeficientes tienen una
    # una carga de optimismo que no afecta al veredicto porque el veredicto no los usa.
    _, consulta_total, nombres = construir_espacio(
        train["text"].astype(str).reset_index(drop=True), textos
    )
    discriminador_total = LogisticRegression(
        C=C_DISC, class_weight="balanced", max_iter=3000, solver="liblinear",
        random_state=RANDOM_STATE,
    )
    discriminador_total.fit(consulta_total, origen)
    coeficientes = discriminador_total.coef_[0]

    # w(x) es el cociente de densidades. Para un discriminador lineal, el log odds de la
    # probabilidad de ser de prueba es el producto escalar con los coeficientes mas el
    # termino independiente, de modo que la razon de chances da directamente el cociente de
    # densidades que se busca.
    log_odds_test = consulta_total @ coeficientes + discriminador_total.intercept_[0]
    log_ratios = log_odds_test
    orden_descendente = np.argsort(log_ratios)[::-1]
    orden_ascendente = np.argsort(log_ratios)

    filas: list[dict[str, Any]] = []
    for etiqueta, indices in (
        ("mas_probables_de_ser_prueba", orden_descendente[:TOP_CARACTERISTICAS]),
        ("mas_probables_de_ser_entrenamiento", orden_ascendente[:TOP_CARACTERISTICAS]),
    ):
        for posicion, indice in enumerate(indices, start=1):
            filas.append(
                {
                    "lado": etiqueta,
                    "puesto": posicion,
                    "caracteristica": str(nombres[indice]),
                    "coeficiente": round(float(coeficientes[indice]), 6),
                    "log_razon_de_odds": round(float(log_ratios[indice]), 6),
                    "vez_que_mas_probable_prueba_que_entrenamiento": round(
                        float(np.exp(log_ratios[indice])), 4
                    ),
                }
            )
    with REPORTE_CSV.open("w", encoding="utf-8", newline="") as manejador:
        escritor = csv.DictWriter(
            manejador,
            fieldnames=[
                "lado",
                "puesto",
                "caracteristica",
                "coeficiente",
                "log_razon_de_odds",
                "vez_que_mas_probable_prueba_que_entrenamiento",
            ],
            lineterminator="\n",
        )
        escritor.writeheader()
        escritor.writerows(filas)
    log(
        f"   {TOP_CARACTERISTICAS} caracteristicas por lado escritas en "
        f"{REPORTE_CSV.relative_to(PROJECT_ROOT)}"
    )

    log("3/4 aplicar el gate de decision pre-registrado")
    if auc_global <= UMBRAL_CERRAR:
        veredicto = "CERRADA"
        lectura = (
            f"El AUC de {auc_global:.6f} no pasa de {UMBRAL_CERRAR}, asi que las poblaciones de "
            "entrenamiento y prueba son **indistinguibles** con las caracteristicas de V03. El "
            "shift de caracteristicas se despreciable y la via queda cerrada: no queda mas que la "
            "correccion de prevalencia, que V10-05 ya mostro que no ayuda."
        )
    elif auc_global <= UMBRAL_FUERTE:
        veredicto = "ABIERTA, shift moderado"
        lectura = (
            f"El AUC de {auc_global:.6f} cae en la banda moderada. El cociente de densidades se "
            "usaria **solo** para reponderar la validacion local, y se reportaria su efecto sobre "
            "la decision. No se entrena con la prueba en ningun caso."
        )
    else:
        veredicto = "ABIERTA, shift fuerte"
        lectura = (
            f"El AUC de {auc_global:.6f} es alto y las dos poblaciones se separan. Se reporta y se "
            "propone un plan, sin entrenar nunca con la prueba. Esto seria el unico punto del "
            "proyecto donde el cuello de botella no es la representacion sino la distribucion, y "
            "merece una linea propia."
        )
    log(f"   veredicto: {veredicto}")

    log("4/4 informe")
    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocolo": (
            "El espacio de caracteristicas se ajusta dentro de cada pliegue de entrenamiento, la "
            "validacion cruzada es agrupada por los grupos fusionados del proyecto en las filas de "
            "entrenamiento y por identificador unico en las de prueba, y la particion es "
            "estratificada por la etiqueta de origen."
        ),
        "nombres": str(nombres.size),
        "auc_global_fuera_de_muestra": round(auc_global, 6),
        "auc_por_pliegue": [round(a, 6) for a in aucs],
        "auc_minimo_por_pliegue": round(min(aucs), 6),
        "auc_maximo_por_pliegue": round(max(aucs), 6),
        "exactitud_global": round(exactitud_global, 6),
        "exactitud_por_pliegue": [round(a, 6) for a in exactitudes],
        "proporcion_de_prueba_real": round(proporcion_test_real, 6),
        "proporcion_de_prueba_predicha": round(proporcion_test_predicha, 6),
        "umbral_para_cerrar": UMBRAL_CERRAR,
        "umbral_para_shift_fuerte": UMBRAL_FUERTE,
        "veredicto": veredicto,
        "caracteristicas_mas_desplazadas": filas[:TOP_CARACTERISTICAS],
        "caracteristicas_mas_probables_de_entrenamiento": filas[-TOP_CARACTERISTICAS:],
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe(informe, filas, veredicto, lectura, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(f"VEREDICTO: {veredicto}")


def _informe(
    informe: dict[str, Any],
    filas: list[dict[str, Any]],
    veredicto: str,
    lectura: str,
    inicio: float,
) -> None:
    """Escribe el informe del diagnostico de shift de features.

    Args:
        informe: veredicto del experimento.
        filas: caracteristicas mas desplazadas.
        veredicto: resultado del gate de decision.
        lectura: interpretacion del veredicto.
        inicio: instante de arranque del script.
    """
    lineas = [
        "# V10-07: diagnostico de shift de features (P2)",
        "",
        f"Generado por `experiments/v10_07_shift_features.py` ({EXPERIMENTO}).",
        "",
        "## 1. Que se mide",
        "",
        "V9-02 midio un **shift de prevalencia**: la prueba tiene cerca de un 0.94 de los",
        "positivos del entrenamiento. Eso actua sobre la capa de decision, y V10-05 comprobo que",
        "adaptar la capa a ese shift no ayuda.",
        "",
        "Aqui se mide la otra mitad, que es distinta: **shift de covariables**, o sea si los",
        "textos de la prueba ocupan otra region del espacio de caracteristicas. Si fuera asi, las",
        "probabilidades de V03 estarian calibradas para una poblacion que no es la de evaluacion.",
        "",
        "El instrumento es un **discriminador** que separa entrenamiento de prueba con las mismas",
        f"caracteristicas de V03, {informe['nombres']} en total, y su AUC **fuera de muestra**.",
        "",
        "Detalles que hacen honesta la medicion: el espacio de caracteristicas se ajusta dentro de",
        "cada pliegue de entrenamiento, la validacion cruzada es agrupada por los grupos",
        "fusionados en las filas de entrenamiento y por identificador unico en las de prueba, y la",
        "particion es estratificada por la etiqueta de origen.",
        "",
        "## 2. Resultado",
        "",
        "| magnitud | valor |",
        "|---|---|",
        f"| AUC global fuera de muestra | **{informe['auc_global_fuera_de_muestra']:.6f}** |",
        (
            f"| AUC por pliegue | de {informe['auc_minimo_por_pliegue']:.6f} a "
            f"{informe['auc_maximo_por_pliegue']:.6f} |"
        ),
        f"| exactitud global | {informe['exactitud_global']:.6f} |",
        f"| proporcion de prueba real | {informe['proporcion_de_prueba_real']:.6f} |",
        f"| proporcion de prueba predicha | {informe['proporcion_de_prueba_predicha']:.6f} |",
        "",
        "El valor de referencia es 0.5, que es el azar: las poblaciones serian indistinguibles.",
        "",
        "## 3. Las caracteristicas mas desplazadas",
        "",
        "El valor es el cociente de densidades, es decir cuantas veces mas probable es esa",
        "caracteristica en la prueba que en el entrenamiento. Mayor que 1 significa que tira hacia",
        "la prueba.",
        "",
        "### Mas probables de ser de la prueba",
        "",
        "| puesto | caracteristica | cociente |",
        "|---|---|---|",
    ]
    for fila in filas[:TOP_CARACTERISTICAS]:
        lineas.append(
            f"| {fila['puesto']} | `{fila['caracteristica'][:60]}` | "
            f"{fila['vez_que_mas_probable_prueba_que_entrenamiento']:.4f} |"
        )
    lineas += [
        "",
        "### Mas probables de ser del entrenamiento",
        "",
        "| puesto | caracteristica | cociente |",
        "|---|---|---|",
    ]
    for fila in filas[-TOP_CARACTERISTICAS:]:
        lineas.append(
            f"| {fila['puesto']} | `{fila['caracteristica'][:60]}` | "
            f"{fila['vez_que_mas_probable_prueba_que_entrenamiento']:.4f} |"
        )
    lineas += [
        "",
        "## 4. Gate de decision, pre-registrado",
        "",
        f"- AUC menor o igual que {UMBRAL_CERRAR}: la via se **cierra**",
        f"- AUC entre {UMBRAL_CERRAR} y {UMBRAL_FUERTE}: shift moderado, el cociente de densidades",
        "  solo repondera la validacion local",
        f"- AUC mayor que {UMBRAL_FUERTE}: shift fuerte, se reporta y se propone plan",
        "",
        f"**VEREDICTO: {veredicto}.**",
        "",
        lectura,
        "",
        "## 5. Lo que este experimento NO puede hacer",
        "",
        "La regla 2 de la fase v10 prohibe entrenar con la prueba. Este discriminador es **solo",
        "diagnostico**: aunque su AUC fuera alto, aqui no se ajustaria ningun modelo con test, y el",
        "cociente de densidades, si se usara, solo podria reponderar la validacion local.",
        "",
        (
            f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
            f"Duracion: {round(time.time() - inicio, 1)} s."
        ),
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
