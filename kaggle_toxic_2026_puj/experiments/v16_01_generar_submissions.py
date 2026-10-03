"""V16-01: generador de las submissions candidatas de la fase v16.

Encargo
-------
Producir, listas para subir a Kaggle, las cinco candidatas que responden a cinco
preguntas distintas sobre transferencia, con control de identidad del campeon y
hash sha256 de cada salida. Nada se reentrena fuera de lo que hace falta y nada
toca el artefacto congelado.

Lo que la fase v16 ha medido antes de elegir estas cinco
--------------------------------------------------------
1. Prevalencia real del test: k = 0.935177, es decir un 6.48 por ciento menos de
   positivos que train, resuelto de forma analitica desde la sonda E00 (plantilla
   todo-uno, LB 0.51494) con la identidad F1 todo-uno = 2p/(1+p). El factor por
   etiqueta que el EM de Saerens estimo sin etiquetas (HS 0.936, TR 0.955,
   AG 0.938) coincide con el estimador analitico, que es la validacion cruzada
   que hace creible la cifra.
2. Pendiente empirica LB/tasa: +0.12011 por la pareja controlada V02 -> V03
   (mismo modelo, mismas features, unica diferencia la regla de decision) y
   +0.18416 en el ajuste descriptivo de los 14 modelos reales. **El signo es
   positivo en las dos**: subir la tasa de positivos predicha sube el LB. La
   hipotesis del encargo, que era bajar la tasa, queda refutada por dos vias
   independientes.
3. Techo de cualquier candidato de baja capacidad: 107 cambios netos de acierto
   mueven el LB 0.003, y el error estandar del LB con 1.154 filas es 0.01219. Solo
   dos de las cinco candidatas superan las 107 celdas.

Las cinco preguntas
-------------------
CH1  Si el unico aumento de tasa controlado del proyecto (V02 -> V03, +0.026 de
     tasa por +0.00313 de LB) extrapola a un paso de la mitad de tamano. Se aplica
     por cuantil, que es invariante a la escala de las probabilidades y por tanto
     inmune al modo de fallo que el registro atribuyo a los ensembles de promedio.

CH2  Si la correccion de prior shift transfiere cuando se aplica a los scores y no
     a la regla de decision. Es la unica candidata que usa informacion de test
     medida fuera de modelo. El artefacto exists en disco (v17) pero nunca se
     envio, y la validacion anidada de la fase v9 le.media solo +0.000574.

CH3  Que bloque de la union word+char carga con la senal que transfiere. Es el
     unico cambio de representacion de la lista, y se ha disenado para que no
     repita los modos de fallo ya pagados: conserva las 27.431 features y el
     C=2.0 del campeon, y solo mueve el peso de mezcla. v05, v06 y v07 fallaron
     cambiando numero de features o C, que es justo lo que aqui no se toca.

CH4  Control negativo: la hipotesis del encargo (bajar la tasa) sometida a la
     misma regla por cuantil. Nadie ha bajado la tasa de forma controlada dentro
     de la familia de V03: las dos sondas historicas que la bajaron cambiaron
     ademas el modelo, asi que su resultado no informa.

CH5  Si la respuesta de la tasa continua mas alla del paso controlado, es decir
     si el optimum esta en la tasa optima de F1 medida y no en la de V03. Es la
     unica candidata ademas de CH3 que puede mover el LB de forma detectable, y
     por tanto la que mas informacion da si se equivoca.

Lo que este script NO hace
--------------------------
No toca experiments/baseline_v03_frozen/ ni ninguna submission existente. Aborta si
el hash del campeon no coincide con el declarado, si la reproduccion celda a celda
falla, o si se intenta escribir sobre un fichero que ya existe. El champion no se
modifica: las candidatas son ficheros nuevos.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import ETIQUETAS
from src.splits import grupos_fusionados, holdout_v2, pliegues_corregidos

DIR_CONGELADO = PROJECT_ROOT / "experiments" / "baseline_v03_frozen"
DIR_SUBMISSIONS = PROJECT_ROOT / "submissions"
DIR_RESULTADOS = PROJECT_ROOT / "resultados"
RUTA_TEST = PROJECT_ROOT / "data" / "raw" / "toxic_test.csv"
RUTA_TRAIN = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
RUTA_OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"

SHA256_V03 = "7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e"
LB_V03 = 0.78306
CORRECTED_CV_V03 = 0.7772114525165721
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])

PARAMS_WORD = {
    "analyzer": "word",
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": True,
}
PARAMS_CHAR = {
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "min_df": 2,
    "sublinear_tf": True,
}
PARAMETROS_LR = {
    "C": 2.0,
    "class_weight": "balanced",
    "max_iter": 1000,
    "solver": "liblinear",
    "random_state": 42,
}

# Prevalencia medida en v16_00 por el identificador analitico de la sonda E00.
K_ANALITICO = 0.935177
# Factores por etiqueta del EM de Saerens (reports/v9_02_prior_shift_em.md). Se
# usan en CH2 porque son decomposicion por etiqueta y el estimador analitico solo
# entrega un escalar.
K_POR_ETIQUETA_EM = np.array([0.936, 0.955, 0.938])

# Tasa optima de F1 medida sobre el OOF a prevalencia de test y la tasa que V03
# realmente alcanza en el mismo OOF. Su cociente es el multiplicador que se
# transporta a las probabilidades de test.
TASA_OPTIMA_OOF = 0.3969
TASA_V03_OOF = 0.3783
MULTIPLICADOR_TRANSPORTADO = TASA_OPTIMA_OOF / TASA_V03_OOF

# Pendiente causal de la pareja controlada V02 -> V03.
PENDIENTE_CAUSAL = (0.78306 - 0.77993) / (0.36452917388792605 - 0.338470)
PENDIENTE_DESCRIPTIVA = 0.18416
F1_POR_ACIERTO_NETO = 0.003 / 107.0
ERROR_ESTANDAR_LB = 0.01219


# ---------------------------------------------------------------------------
# Estructura de una candidata
# ---------------------------------------------------------------------------
@dataclass
class Candidata:
    """Una candidata de la fase v16.

    Attributes:
        clave: identificador corto, usado en el nombre del fichero.
        titulo: nombre descriptivo de la submission.
        pregunta: que pregunta concreta responde sobre transferencia.
        mecanismo: como se construye la matriz de etiquetas.
        argumentos: parametros del mecanismo.
    """

    clave: str
    titulo: str
    pregunta: str
    mecanismo: str
    argumentos: dict[str, object] = field(default_factory=dict)


CANDIDATAS: list[Candidata] = [
    Candidata(
        clave="ch1",
        titulo="v16_ch1_tasa_optima_transportada",
        pregunta=(
            "Si el unico aumento de tasa controlado del proyecto (V02 -> V03, "
            "+0.026 de tasa por +0.00313 de LB) extrapola a un paso de la mitad "
            "de su tamano."
        ),
        mecanismo="cuantil",
        argumentos={
            "tasa_objetivo": None,
            "descripcion": (
                "tasa objetivo = tasa de V03 en test por el multiplicador "
                "transportado desde el OOF"
            ),
        },
    ),
    Candidata(
        clave="ch2",
        titulo="v16_ch2_prior_shift_em",
        pregunta=(
            "Si la correccion de prior shift transfiere cuando se aplica a los "
            "scores y no a la regla de decision, con los factores por etiqueta "
            "del EM sin etiquetas."
        ),
        mecanismo="logit_prior",
        argumentos={"k_por_etiqueta": K_POR_ETIQUETA_EM.tolist()},
    ),
    Candidata(
        clave="ch3",
        titulo="v16_ch3_peso_char_070",
        pregunta=(
            "Que bloque de la union word+char carga con la senal que transfiere, "
            "sin tocar ni el numero de features ni la constante de "
            "regularizacion."
        ),
        mecanismo="peso_bloques",
        argumentos={"peso_word": 0.3, "peso_char": 0.7},
    ),
    Candidata(
        clave="ch4",
        titulo="v16_ch4_control_tasa_baja",
        pregunta=(
            "La hipotesis del encargo (bajar la tasa para igualar la "
            "prevalencia) sometida a control: mismo modelo, misma regla por "
            "cuantil, solo baja la tasa."
        ),
        mecanismo="cuantil",
        argumentos={
            "tasa_objetivo": 0.3567,
            "descripcion": "tasa objetivo 0.3567, un 0.0078 por debajo de V03",
        },
    ),
    Candidata(
        clave="ch5",
        titulo="v16_ch5_tasa_optima_absoluta",
        pregunta=(
            "Si la respuesta de la tasa continua mas alla del paso controlado, "
            "es decir si el optimum esta en la tasa optima de F1 medida y no en "
            "la que V03 alcanza."
        ),
        mecanismo="cuantil",
        argumentos={
            "tasa_objetivo": TASA_OPTIMA_OOF,
            "descripcion": "tasa objetivo igual a la tasa optima de F1 medida en el OOF",
        },
    ),
]


# ---------------------------------------------------------------------------
# Utilidades de datos
# ---------------------------------------------------------------------------
def sha256_de(ruta: Path) -> str:
    """Calcula el sha256 de un fichero.

    Args:
        ruta: camino del fichero.

    Returns:
        Hexadecimal de 64 caracteres.
    """
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def leer_train() -> pd.DataFrame:
    """Lee el train con los identificadores normalizados.

    Returns:
        Marco de 1.731 filas.
    """
    train = pd.read_csv(RUTA_TRAIN, sep=";")
    train["id"] = train["id"].astype(str).str.strip('"')
    return train


def leer_test() -> pd.DataFrame:
    """Lee el test, que es cp1252 y no utf-8.

    Returns:
        Marco de 1.154 filas.
    """
    return pd.read_csv(RUTA_TEST, sep=";", encoding="latin-1")


def ajustar_probabilidades(
    train: pd.DataFrame, test: pd.DataFrame, peso_word: float = 1.0, peso_char: float = 1.0
) -> np.ndarray:
    """Reajusta la union TF-IDF de V03 con la mezcla de bloques indicada.

    Los dos bloques se multiplican por su peso antes de concatenarse, lo que es
    exactamente lo que hstack haria con un ``TransformerWeight``. No se cambia nada
    mas: mismos vectorizadores, mismo numero de features, mismos parametros del
    clasificador.

    Args:
        train: conjunto de entrenamiento.
        test: conjunto de test.
        peso_word: peso del bloque word.
        peso_char: peso del bloque char_wb.

    Returns:
        Matriz de 1.154 por 3 con las probabilidades.
    """
    vectorizador_word = TfidfVectorizer(**PARAMS_WORD)
    vectorizador_char = TfidfVectorizer(**PARAMS_CHAR)
    textos_train = train["text"].astype(str)
    textos_test = test["text"].astype(str)
    bloque_word = vectorizador_word.fit_transform(textos_train)
    bloque_char = vectorizador_char.fit_transform(textos_train)
    ajuste = sparse.hstack(
        [peso_word * bloque_word, peso_char * bloque_char], format="csr"
    )
    consulta = sparse.hstack(
        [
            peso_word * vectorizador_word.transform(textos_test),
            peso_char * vectorizador_char.transform(textos_test),
        ],
        format="csr",
    )
    return np.column_stack(
        [
            LogisticRegression(**PARAMETROS_LR)
            .fit(ajuste, train[etiqueta].to_numpy())
            .predict_proba(consulta)[:, 1]
            for etiqueta in ETIQUETAS
        ]
    )


def probabilidades_congeladas(test: pd.DataFrame) -> np.ndarray:
    """Lee las probabilidades de test del artefacto congelado.

    Args:
        test: conjunto de test, usado solo para alinear filas.

    Returns:
        Matriz de 1.154 por 3 con las probabilidades de V03.
    """
    artefacto = joblib.load(DIR_CONGELADO / "model_v03.joblib")
    if not bool(
        (
            test["id"].to_numpy()
            == pd.read_csv(DIR_CONGELADO / "submission_v03.csv")["id"].to_numpy()
        ).all()
    ):
        raise AssertionError("El orden de filas de test no coincide con la referencia.")
    matriz = artefacto["vectorizador"].transform(test["text"].astype(str))
    return np.column_stack(
        [estimador.predict_proba(matriz)[:, 1] for estimador in artefacto["modelos"]]
    )


def umbral_por_cuantil(probabilidades: np.ndarray, tasa_objetivo: float) -> float:
    """Devuelve el umbral que deja la tasa objetivo por encima.

    Args:
        probabilidades: probabilidades de una etiqueta.
        tasa_objetivo: fraccion de filas que se quieren positivas.

    Returns:
        Umbral equivalente al cuantil ``1 - tasa_objetivo``.
    """
    return float(np.quantile(probabilidades, 1.0 - tasa_objetivo, method="lower"))


def aplicar_logit_prior(probabilidades: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Corrige el prior restando ``log(k)`` al logit de cada etiqueta.

    Es la correccion de prior con la forma que corresponde a un cambio de
    prevalencia manteniendo las pendientes. Se recorta antes de tomar el logit
    porque las probabilidades del champion nunca son 0 ni 1, pero el recorte evita
    que una celda inesperada reviente la transformacion.

    Args:
        probabilidades: matriz de probabilidades.
        k: factor de prevalencia por etiqueta.

    Returns:
        Matriz de probabilidades corregida.
    """
    epsilon = 1e-6
    recortadas = np.clip(probabilidades, epsilon, 1.0 - epsilon)
    logit = np.log(recortadas / (1.0 - recortadas))
    return 1.0 / (1.0 + np.exp(-(logit - np.log(k)[None, :])))


def f1_por_etiqueta(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """F1 binaria de la clase positiva para cada etiqueta.

    Args:
        y: matriz binaria verdadera.
        pred: matriz binaria predicha.

    Returns:
        Vector de tres F1.
    """
    salida = np.empty(y.shape[1], dtype=np.float64)
    for etiqueta in range(y.shape[1]):
        tp = float(np.sum(y[:, etiqueta] * pred[:, etiqueta]))
        fp = float(np.sum((1 - y[:, etiqueta]) * pred[:, etiqueta]))
        fn = float(np.sum(y[:, etiqueta] * (1 - pred[:, etiqueta])))
        denominador = 2 * tp + fp + fn
        salida[etiqueta] = 2 * tp / denominador if denominador > 0 else 0.0
    return salida


def oof_de_la_mezcla(
    train: pd.DataFrame, peso_word: float, peso_char: float
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula el OOF de la mezcla indicada con los pliegues corregidos v2.

    Los vectorizadores se ajustan **dentro** de cada pliegue de entrenamiento. Ajustarlos
    antes, sobre las filas de validacion, seria fuga.

    La particion es la misma que la del OOF canonico, y eso se verifica: el OOF
    recalculado con pesos 1.0 y 1.0 coincide con ``features/oof_v03_final.npz``
    con una diferencia maxima de 0.0.

    Args:
        train: conjunto de entrenamiento completo.
        peso_word: peso del bloque word.
        peso_char: peso del bloque char_wb.

    Returns:
        Tupla con las probabilidades OOF y las etiquetas verdaderas del desarrollo.
    """
    grupos = grupos_fusionados(train)
    desarrollo, _ = holdout_v2(train, grupos)
    marco = train.iloc[desarrollo].reset_index(drop=True)
    pliegues = pliegues_corregidos(marco, grupos[desarrollo])
    textos = marco["text"].astype(str)
    probabilidades = np.zeros((len(marco), 3), dtype=np.float64)
    for entrenamiento, validacion in pliegues:
        vectorizador_word = TfidfVectorizer(**PARAMS_WORD)
        vectorizador_char = TfidfVectorizer(**PARAMS_CHAR)
        ajuste = sparse.hstack(
            [
                peso_word * vectorizador_word.fit_transform(textos.iloc[entrenamiento]),
                peso_char * vectorizador_char.fit_transform(textos.iloc[entrenamiento]),
            ],
            format="csr",
        )
        consulta = sparse.hstack(
            [
                peso_word * vectorizador_word.transform(textos.iloc[validacion]),
                peso_char * vectorizador_char.transform(textos.iloc[validacion]),
            ],
            format="csr",
        )
        for indice, etiqueta in enumerate(ETIQUETAS):
            estimador = LogisticRegression(
                C=2.0, class_weight="balanced", max_iter=2000,
                solver="liblinear", random_state=42,
            )
            estimador.fit(ajuste, marco[etiqueta].to_numpy()[entrenamiento])
            probabilidades[validacion, indice] = estimador.predict_proba(consulta)[:, 1]
    return probabilidades, marco[list(ETIQUETAS)].to_numpy(np.int64)


# ---------------------------------------------------------------------------
# Controles de identidad
# ---------------------------------------------------------------------------
def control_de_identidad(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Verifica tres veces que el pipeline reproduce la submission congelada.

    Los tres controles se hacen antes de generar nada, y cualquiera que falle
    aborta el script. Sin ellos, un cambio silencioso en la receta producing
    ficheros silenciosamente distintos invalidaria toda la comparacion.

    Args:
        train: conjunto de entrenamiento.
        test: conjunto de test.

    Returns:
        Matriz de 1.154 por 3 con las probabilidades de V03.
    """
    hash_real = sha256_de(DIR_CONGELADO / "submission_v03.csv")
    print(f"  control 1, sha256 del artefacto congelado : {hash_real}")
    if hash_real != SHA256_V03:
        raise SystemExit(
            f"El hash del campeon no coincide: {hash_real} frente a {SHA256_V03}."
        )

    referencia = pd.read_csv(DIR_CONGELADO / "submission_v03.csv")
    propias = referencia[list(ETIQUETAS)].to_numpy(np.int64)

    congeladas = probabilidades_congeladas(test)
    coinciden_congelado = int(
        ((congeladas >= UMBRALES_V03[None, :]).astype(np.int64) == propias).sum()
    )
    print(f"  control 2, desde el artefacto congelado  : {coinciden_congelado} de 3462")
    if coinciden_congelado != 3462:
        raise SystemExit("El artefacto congelado no reproduce su propia submission.")

    reentrenadas = ajustar_probabilidades(train, test)
    coinciden_reentreno = int(
        ((reentrenadas >= UMBRALES_V03[None, :]).astype(np.int64) == propias).sum()
    )
    diferencia = float(np.abs(reentrenadas - congeladas).max())
    print(
        f"  control 3, reentreno independiente       : {coinciden_reentreno} de 3462, "
        f"diferencia maxima de probabilidad {diferencia:.2e}"
    )
    if coinciden_reentreno != 3462 or diferencia > 1e-12:
        raise SystemExit("El reentreno independiente no reproduce el campeon.")
    return congeladas


# ---------------------------------------------------------------------------
# Construccion de las candidatas
# ---------------------------------------------------------------------------
def construir_candidata(
    candidata: Candidata,
    probabilidades: np.ndarray,
    oof_relevante: np.ndarray,
    referencia: np.ndarray,
    tasa_v03: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Construye la matriz de etiquetas de una candidata y su regla equivalente.

    Args:
        candidata: definicion de la candidata.
        probabilidades: probabilidades de test de la candidata.
        oof_relevante: probabilidades OOF correspondientes, para medir en desarrollo.
        referencia: matriz de etiquetas de V03.
        tasa_v03: tasa de positivos de V03 en test.

    Returns:
        Tupla con las etiquetas de test, las de OOF y un diccionario de detalle.
    """
    detalle: dict[str, object] = {}
    if candidata.mecanismo == "logit_prior":
        k = np.asarray(candidata.argumentos["k_por_etiqueta"], dtype=np.float64)
        prob_test = aplicar_logit_prior(probabilidades, k)
        prob_oof = aplicar_logit_prior(oof_relevante, k)
        etiquetas_test = (prob_test >= UMBRALES_V03[None, :]).astype(np.int64)
        etiquetas_oof = (prob_oof >= UMBRALES_V03[None, :]).astype(np.int64)
        detalle["umbrales"] = UMBRALES_V03.tolist()
        detalle["k"] = k.tolist()
    elif candidata.mecanismo == "peso_bloques":
        etiquetas_test = (probabilidades >= UMBRALES_V03[None, :]).astype(np.int64)
        etiquetas_oof = (oof_relevante >= UMBRALES_V03[None, :]).astype(np.int64)
        detalle["umbrales"] = UMBRALES_V03.tolist()
        detalle["peso_word"] = candidata.argumentos["peso_word"]
        detalle["peso_char"] = candidata.argumentos["peso_char"]
    else:
        objetivo = candidata.argumentos.get("tasa_objetivo")
        if objetivo is None:
            objetivo = tasa_v03 * MULTIPLICADOR_TRANSPORTADO
        objetivo_por_etiqueta = np.clip(
            referencia.mean(axis=0) * (float(objetivo) / tasa_v03), 0.0, 1.0
        )
        umbrales = np.array(
            [
                umbral_por_cuantil(probabilidades[:, e], objetivo_por_etiqueta[e])
                for e in range(3)
            ]
        )
        etiquetas_test = (probabilidades >= umbrales[None, :]).astype(np.int64)
        etiquetas_oof = (
            oof_relevante
            >= np.array(
                [
                    umbral_por_cuantil(oof_relevante[:, e], objetivo_por_etiqueta[e])
                    for e in range(3)
                ]
            )[None, :]
        ).astype(np.int64)
        detalle["umbrales"] = [float(u) for u in umbrales]
        detalle["tasa_objetivo"] = float(objetivo)
        detalle["tasa_objetivo_por_etiqueta"] = [
            float(t) for t in objetivo_por_etiqueta
        ]
    return etiquetas_test, etiquetas_oof, detalle


def main() -> None:
    """Genera las cinco candidatas y escribe sus artefactos."""
    DIR_RESULTADOS.mkdir(exist_ok=True)
    print("=" * 78)
    print("FASE v16: GENERADOR DE CANDIDATAS")
    print("=" * 78)

    train = leer_train()
    test = leer_test()
    print("Controles de identidad del campeon:")
    probabilidades_v03 = control_de_identidad(train, test)

    referencia = (
        probabilidades_v03 >= UMBRALES_V03[None, :]
    ).astype(np.int64)
    tasa_v03 = float(referencia.mean())

    oof = np.load(RUTA_OOF, allow_pickle=True)
    proba_oof_canonica = oof["proba"].astype(np.float64)
    y_oof = oof["y_true"].astype(np.int64)
    referencia_oof = (proba_oof_canonica >= UMBRALES_V03[None, :]).astype(np.int64)
    f1_referencia_oof = float(f1_por_etiqueta(y_oof, referencia_oof).mean())

    print()
    print(f"corrected_cv de V03 (OOF canonico) : {f1_referencia_oof:.6f}")
    print(f"LB de V03                        : {LB_V03}")
    print(f"tasa de V03 en test              : {tasa_v03:.6f}")
    print(f"multiplicador transportado       : {MULTIPLICADOR_TRANSPORTADO:.6f} "
          f"(tasa optima OOF {TASA_OPTIMA_OOF} / tasa real OOF {TASA_V03_OOF})")
    print(f"tasa objetivo de CH1             : "
          f"{tasa_v03 * MULTIPLICADOR_TRANSPORTADO:.6f}")

    # OOF propio de la mezcla 0.3/0.7, que es la unica candidata que reentrena.
    print()
    print("Recalculando el OOF de la mezcla 0.3 word / 0.7 char (particion corregida):")
    proba_oof_ch3, y_oof_ch3 = oof_de_la_mezcla(train, 0.3, 0.7)
    proba_oof_ch1, _ = oof_de_la_mezcla(train, 1.0, 1.0)
    diferencia_oof = float(np.abs(proba_oof_ch1 - proba_oof_canonica).max())
    print(f"  control 4, OOF con pesos 1.0 reproduce el canonico en "
          f"{diferencia_oof:.2e}")
    if diferencia_oof > 1e-12:
        raise SystemExit("La particion del OOF recalculado no es la canonica.")
    assert bool((y_oof_ch3 == y_oof).all()), "Las etiquetas del OOF no coinciden."

    filas = []
    resumen: dict[str, object] = {}
    for candidata in CANDIDATAS:
        if candidata.mecanismo == "peso_bloques":
            probabilidades = ajustar_probabilidades(
                train, test,
                peso_word=float(candidata.argumentos["peso_word"]),
                peso_char=float(candidata.argumentos["peso_char"]),
            )
            oof_relevante = proba_oof_ch3
        else:
            probabilidades = probabilidades_v03
            oof_relevante = proba_oof_canonica

        etiquetas_test, etiquetas_oof, detalle = construir_candidata(
            candidata, probabilidades, oof_relevante, referencia, tasa_v03
        )

        ruta = DIR_SUBMISSIONS / f"toxic_submission_{candidata.titulo}.csv"
        if ruta.exists():
            raise SystemExit(f"La salida ya existe y no se sobrescribe: {ruta}")
        marco = pd.DataFrame({"id": test["id"].to_numpy()})
        for indice, etiqueta in enumerate(ETIQUETAS):
            marco[etiqueta] = etiquetas_test[:, indice]
        marco.to_csv(ruta, index=False)
        hash_salida = sha256_de(ruta)

        # Verificacion posterior de la escritura: se relee y se cuentan celdas.
        releido = pd.read_csv(ruta)
        if list(releido.columns) != ["id", *ETIQUETAS]:
            raise SystemExit(f"Columnas inesperado en {ruta}: {list(releido.columns)}")
        if bool(releido.isna().to_numpy().any()):
            raise SystemExit(f"Hay nulos en {ruta}.")
        if not bool((releido["id"].to_numpy() == test["id"].to_numpy()).all()):
            raise SystemExit(f"El orden de identificadores no cuadra en {ruta}.")

        celdas = int((etiquetas_test != referencia).sum())
        celdas_por_etiqueta = [int(x) for x in (etiquetas_test != referencia).sum(axis=0)]
        tasa = float(etiquetas_test.mean())
        delta_tasa = tasa - tasa_v03

        mascara_oof = etiquetas_oof != referencia_oof
        if mascara_oof.any():
            acierto = float((y_oof[mascara_oof] == etiquetas_oof[mascara_oof]).mean())
            aciertos_netos = int(
                (
                    (etiquetas_oof[mascara_oof] == y_oof[mascara_oof]).sum()
                    - (referencia_oof[mascara_oof] == y_oof[mascara_oof]).sum()
                )
            )
        else:
            acierto = float("nan")
            aciertos_netos = 0
        f1_oof = float(f1_por_etiqueta(y_oof, etiquetas_oof).mean())

        fila = {
            "clave": candidata.clave,
            "archivo": ruta.name,
            "sha256": hash_salida,
            "pregunta": candidata.pregunta,
            "mecanismo": candidata.mecanismo,
            "tasa_candidata": tasa,
            "delta_tasa": delta_tasa,
            "tasa_HS": float(etiquetas_test.mean(axis=0)[0]),
            "tasa_TR": float(etiquetas_test.mean(axis=0)[1]),
            "tasa_AG": float(etiquetas_test.mean(axis=0)[2]),
            "celdas_distintas": celdas,
            "celdas_por_etiqueta": celdas_por_etiqueta,
            "acierto_entre_modificadas_oof": acierto,
            "aciertos_netos_oof": aciertos_netos,
            "corrected_cv_oof": f1_oof,
            "delta_corrected_cv": f1_oof - f1_referencia_oof,
            "lb_predicho_causal": LB_V03 + PENDIENTE_CAUSAL * delta_tasa,
            "lb_predicho_descriptivo": LB_V03 + PENDIENTE_DESCRIPTIVA * delta_tasa,
            "lb_techo_si_todo_acierta": LB_V03 + celdas * F1_POR_ACIERTO_NETO,
            "supera_presupuesto_de_107_celdas": bool(celdas > 107),
            "detalle_regla": detalle,
        }
        filas.append(fila)
        resumen[candidata.clave] = fila

        print()
        print(f"--- {candidata.clave.upper()}: {candidata.titulo}")
        print(f"    {candidata.pregunta}")
        print(f"    mecanismo            : {candidata.mecanismo} {detalle}")
        print(f"    celdas que cambian   : {celdas} (por etiqueta {celdas_por_etiqueta})")
        print(f"    tasa                 : {tasa:.6f} (delta {delta_tasa:+.6f})")
        print(f"    acierto en modificadas (OOF): {acierto:.4f}, "
              f"aciertos netos {aciertos_netos}")
        print(f"    corrected_cv (OOF)   : {f1_oof:.6f} "
              f"(V03 {f1_referencia_oof:.6f}, delta {f1_oof - f1_referencia_oof:+.6f})")
        print(f"    LB predicho          : causal {fila['lb_predicho_causal']:.5f} | "
              f"descriptivo {fila['lb_predicho_descriptivo']:.5f} | "
              f"techo {fila['lb_techo_si_todo_acierta']:.5f}")
        print(f"    sha256               : {hash_salida}")

    tabla = pd.DataFrame(
        [{k: v for k, v in fila.items() if k != "detalle_regla"} for fila in filas]
    )
    tabla.to_csv(DIR_RESULTADOS / "v16_01_tabla_de_challengers.csv", index=False)
    (DIR_RESULTADOS / "v16_01_challengers.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=True), encoding="ascii"
    )

    print()
    print("=" * 78)
    print("RESUMEN")
    print("=" * 78)
    print(
        f"{'clave':<7}{'mecanismo':<17}{'tasa':>9}{'celdas':>8}{'acierto':>9}"
        f"{'LB causal':>11}{'LB descr':>10}{'LB techo':>10}"
    )
    for fila in filas:
        print(
            f"{fila['clave']:<7}{fila['mecanismo']:<17}{fila['tasa_candidata']:>9.4f}"
            f"{fila['celdas_distintas']:>8d}{fila['acierto_entre_modificadas_oof']:>9.4f}"
            f"{fila['lb_predicho_causal']:>11.5f}{fila['lb_predicho_descriptivo']:>10.5f}"
            f"{fila['lb_techo_si_todo_acierta']:>10.5f}"
        )
    print()
    print(f"Referencia V03: tasa {tasa_v03:.4f}, LB {LB_V03}, "
          f"corrected_cv {f1_referencia_oof:.6f}")
    print(f"Error estandar del LB: {ERROR_ESTANDAR_LB}")
    print()
    print("Artefactos escritos:")
    for fila in filas:
        print(f"  submissions/{fila['archivo']}")
    print("  resultados/v16_01_tabla_de_challengers.csv")
    print("  resultados/v16_01_challengers.json")
    print()
    print("El artefacto congelado no se ha modificado:")
    print(f"  {sha256_de(DIR_CONGELADO / 'submission_v03.csv')}")


if __name__ == "__main__":
    main()
