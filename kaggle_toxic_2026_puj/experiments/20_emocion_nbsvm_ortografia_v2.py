"""experiments/20_emocion_nbsvm_ortografia_v2.py
Fecha: 2026-09-25
Descripcion: P5 del plan v2 sobre el desarrollo (n=1483) con los pliegues
corregidos de D-016 (grupos = near-duplicado UNION pseudo-autor,
StratifiedGroupKFold(5), seed 42). El holdout congelado no se toca en ningun
momento. Tres lineas experimentales en un solo script:
  E2-09 emocion / afecto (H5): marcadores estaticos de risa, afecto,
        diminutivos, vocativos, emoticones de texto, intensificadores, amenaza
        y deseo negativo; lexicos de afecto INDUCIDOS fold-local con log-odds
        suavizado (HS/TR/AG 1-vs-0, min frecuencia 5, >= 3 documentos) y
        polaridad local en ventanas de +-3/5/10 tokens alrededor de menciones y
        de referencias colectivas.
  E2-15 NB-SVM: log-count ratio r = log((p+alpha)/(q+alpha)) fold-local sobre
        word(1,2), char_wb(3,5) y la union; LR y LinearSVC balanced.
  E2-10 multicanal ortografico: canales A raw, B colapso de repeticiones,
        C sin puntuacion interna de palabra, D normalizacion leetspeak,
        E enmascarado de insultos con <INSULT> (lexico inducido fold-local).
Estado del arte aplicado: NB-SVM de Wang y Manning (2012) con log-count ratio;
log-odds informativo de Monroe et al. (2008) reutilizado del canal P2; reglas
de adopcion pre-registradas de D-016; expectativa declarada del plan v2 para P5:
BAJA-MEDIA (se evalua y se documenta con honestidad, sin inflar la decision).

Preguntas que responde
----------------------
E2-09 (H5): aportan la emocion del autor y la polaridad local al target senal
que word/char de V03 no capturan (mecanismo esperado: risas/afecto/vocativo
desactivan el falso positivo por broma o afiliacion; el deseo negativo y la
amenaza marcan agresion)?
E2-15: da el peso log-count ratio un modelo complementario mas barato y diverso?
E2-10: recupera un canal normalizado lexemas ofuscados (repeticiones, puntuacion
interna, leetspeak) que el raw y char_wb pierden, o el raw conserva mejor el
estilo?

Lexicos externos de emocion (documentacion exigida por el plan v2)
-------------------------------------------------------------------
No se usa ningun lexico externo. `data/external/lexicons/` esta vacio y la
sesion prohibe descargas e instalaciones, por lo que solo se usa el lexico
INDUCIDO fold-local (autocontenido, sin licencia de terceros). Alternativas
evaluadas y descartadas: NRC EmoLex/HashTag (descarga manual con confirmacion,
solo ingles), ML-SentiCon/DAL del grupo TASS (descarga manual, licencia no
comercial), VADER-es de pysentimiento (requiere transformers/modelos
preentrenados, prohibidos) y WordNet-Affect via OMW (solo ingles; el mapeo no
cubre jerga de twitter). Detalle en el bloque ``lexicos_externos`` del JSON.

Metodo
------
1. Se reutiliza la anotacion spaCy SIMBOLICA cacheada de P2
   (`features/v2_targeting_annotations.json.gz`, lema/POS/numero/persona/modo)
   y sus detectores estructurales de mencion y de referencia colectiva; no se
   usan vectores, similitud ni modelos preentrenados.
2. Todas las transformaciones (lexicos inductivos, inventario colectivo,
   vectorizadores, escaladores, pesos NB, modelos y umbrales) se ajustan SOLO
   con el pliegue de entrenamiento; el sobreajuste de la seleccion de C se
   declara como cota optimista.
3. Referencias del protocolo v2 sobre el desarrollo: M2 a 0.5 = 0.7718
   (reproducida en el script) y V03 con umbrales congelados = 0.7772.
   Nota de escala en E2-09: el hstack directo V03 + afecto deja al bloque denso
   con norma de fila ~8.4 frente a ~1.4 del TF-IDF, por lo que la LR queda
   dominada por las features densas; se reporta ademas TXT+EMO-NORM (bloque
   denso reescalado por 1/sqrt(d)) como diagnostico y la decision pre-registrada
   se toma solo sobre las variantes declaradas.
4. Regla de adopcion (D-016): adoptar si corrected_cv >= 0.7822 con >= 3/5
   pliegues mejorados frente a V03, o complementariedad con ``ensemble_gain``
   >= 0.0015 e IC bootstrap pareado (2000 remuestreos a nivel de tweet)
   estrictamente positivo. El ``ensemble_gain`` se define igual que en P1-P4
   (media de probabilidades con M2 menos 0.7718) para que los numeros sean
   comparables; la concatenacion aprendida de scores es P6.

Pruebas de robustez
-------------------
Textos limite (vacio, solo URL, solo mencion, RT, comillas desbalanceadas,
elongaciones, leetspeak, insulto enmascarado): ninguna feature nueva produce NaN
ni infinito, los canales no dependen del orden de las filas (se verifica
permutando el desarrollo) y el enmascarado no rompe el tokenizador.

Salidas
-------
`reports/emocion_nbsvm_ortografia_v2.csv` y `.json`,
`reports/figuras/fig_31_emocion_nbsvm_v2.png`,
`features/v2_oof_emocion_nbsvm_ortografia.npz` (canales OOF para el meta de P6),
filas `E2-09-*`, `E2-10-*` y `E2-15-*` en `experiments.csv`, seccion
"## Fase v2 - P5 (2026-09-25)" en `reports/contexto.md` y actualizacion de
`tasks/todo.md` (P5.1-P5.4 y Gate P5).

Ejecucion
---------
    /home/carlos/venv/bin/python experiments/20_emocion_nbsvm_ortografia_v2.py
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import time
import traceback
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse
from scipy.special import expit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import ETIQUETAS, metricas_multietiqueta

OOF_BASE_PATH = PROJECT_ROOT / "features" / "v2_oof_base_corrected.npz"
OOF_PATH = PROJECT_ROOT / "features" / "v2_oof_emocion_nbsvm_ortografia.npz"
REPORTE_CSV = PROJECT_ROOT / "reports" / "emocion_nbsvm_ortografia_v2.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "emocion_nbsvm_ortografia_v2.json"
FIGURA_PATH = PROJECT_ROOT / "reports" / "figuras" / "fig_31_emocion_nbsvm_v2.png"
EXPERIMENTS_CSV = PROJECT_ROOT / "experiments.csv"
CONTEXTO_MD = PROJECT_ROOT / "reports" / "contexto.md"
TODO_MD = PROJECT_ROOT / "tasks" / "todo.md"

Pliegues = list[tuple[NDArray[np.int_], NDArray[np.int_]]]

CONFIG: Final[dict[str, Any]] = {
    "fecha": "2026-09-25",
    "referencia_m2_umbral_05": 0.7718,
    "referencia_v03_umbrales": 0.7772,
    "referencia_v2a_meta": 0.7853,
    "umbrales_v03": {"HS": 0.44, "TR": 0.51, "AG": 0.46},
    "v03_por_etiqueta": {"HS": 0.762, "TR": 0.816, "AG": 0.754},
    "C_variantes": (1.0, 2.0),
    "min_frecuencia_lexico": 5,
    "min_documentos_lexico": 3,
    "umbral_log_odds_afecto": 1.0,
    "ventanas_tokens": (3, 5, 10),
    "alpha_nb_svm": 1.0,
    "C_lr_nb_svm": 1.0,
    "C_svc_nb_svm": 1.0,
    "n_bootstrap": 2000,
    "mejora_minima": 0.005,
    "ensemble_gain_minimo": 0.0015,
    "pliegues_minimos_mejorados": 3,
    "max_iter_lr": 1000,
    "max_iter_svc": 5000,
    "tolerancia_replicacion_f1": 1e-4,
}

# Representacion de texto de V03 (identica a experiments/14, modelo M2).
PARAMS_WORD: Final[dict[str, Any]] = {
    "analyzer": "word",
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": True,
}
PARAMS_CHAR: Final[dict[str, Any]] = {
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "min_df": 2,
    "sublinear_tf": True,
}
PARAMS_COUNT_WORD: Final[dict[str, Any]] = {
    "analyzer": "word",
    "ngram_range": (1, 2),
    "min_df": 2,
}
PARAMS_COUNT_CHAR: Final[dict[str, Any]] = {
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "min_df": 2,
}
PARAMS_LR: Final[dict[str, Any]] = {
    "class_weight": "balanced",
    "max_iter": CONFIG["max_iter_lr"],
    "solver": "liblinear",
    "random_state": 42,
}
PARAMS_SVC: Final[dict[str, Any]] = {
    "class_weight": "balanced",
    "max_iter": CONFIG["max_iter_svc"],
    "random_state": 42,
}

# ===== Inventarios ESTATICOS de marcadores (escritos en este script, sin datos externos) =====

PATRONES_ESTATICOS: Final[dict[str, tuple[re.Pattern[str], ...]]] = {
    "risa": (
        re.compile(r"\b(?:ja(?:ja)+|je(?:je)+|ji(?:ji)+|jo(?:jo)+|ju(?:ju)+)\b"),
        re.compile(r"\bxd+\b"),
        re.compile(r"\b(?:lol|lmao|rofl|risas?|jajas?)\b"),
    ),
    "afecto": (
        re.compile(
            r"\b(?:bebe|bebes|amor|amores|carino|carinos|guapa|guapo|hermosa|hermoso|"
            r"preciosa|precioso|linda|lindo|corazon|corazones|princesa|principe|reina|rey|"
            r"mi vida|mivida|te amo|te quiero|te adoro|socia|socio|amiga|amigo)\b"
        ),
    ),
    "emoticono": (
        re.compile(r"[:;=8]-?[)(\]\[dDpP3/|Ss]"),
        re.compile(r"</?3"),
    ),
    "intensificador": (
        re.compile(
            r"\b(?:muy|muchisimo|muchisima|super|requete|ultra|mega|hiper|tanto|demasiado|"
            r"demasiada|jamas|nunca|recontra)\b"
        ),
        re.compile(r"\b\w+isim[oa]s?\b"),
    ),
    "amenaza": (
        re.compile(
            r"\b(?:te vas a|vas a ver|te voy a|te vamos a|ojo con|cuidadito|te arrepentiras|"
            r"te acordaras|ya veras|vas a llorar|lo vas a pagar|te espera|pagaras|preparate|"
            r"atent[oa]|te cae|se te va a caer)\b"
        ),
    ),
    "deseo_negativo": (
        re.compile(
            r"\b(?:ojala|que se muera|que te mueras|que se pudra|muerete|que te maten|"
            r"ojala se muera|que se joda|mueranse|que te den|te mueras)\b"
        ),
    ),
}
PATRON_GRAFIA_ELONGADA: Final[re.Pattern[str]] = re.compile(r"\b\w*([a-z])\1{2,}\w*\b")
PATRON_DIMINUTIVO: Final[re.Pattern[str]] = re.compile(r"\b[a-z]{4,}(?:ito|ita|itos|itas)\b")
EXCLUSIONES_DIMINUTIVO: Final[frozenset[str]] = frozenset(
    {
        "bonito",
        "bonita",
        "bonitos",
        "bonitas",
        "chiquito",
        "chiquita",
        "gratuito",
        "gratuita",
        "delito",
        "delitos",
        "ejercito",
        "infinito",
        "infinita",
        "grito",
        "gritos",
        "grita",
        "cita",
        "citas",
        "visita",
        "visitas",
        "sito",
        "incognito",
        "maldito",
        "maldita",
        "malditos",
        "malditas",
        "poquito",
        "poquita",
        "ratito",
    }
)
PATRON_MENCION: Final[re.Pattern[str]] = re.compile(r"@\w+")
PATRON_MENCION_INICIAL: Final[re.Pattern[str]] = re.compile(r"^\s*(?:rt\s+)?@\w+", re.IGNORECASE)
PATRON_PALABRA_MAYUSCULA: Final[re.Pattern[str]] = re.compile(r"^[A-Z][a-z]{2,}[.,!?)]*$")
PATRON_TRAS_COMA: Final[re.Pattern[str]] = re.compile(r",\s+[A-Za-z]{2,}")

NOMBRES_ESTATICAS: Final[tuple[str, ...]] = (
    "est_risa_n",
    "est_risa_densidad",
    "est_afecto_n",
    "est_afecto_densidad",
    "est_diminutivo_n",
    "est_diminutivo_densidad",
    "est_vocativo_n",
    "est_vocativo_densidad",
    "est_mencion_n",
    "est_mencion_inicial",
    "est_emoticono_n",
    "est_emoticono_densidad",
    "est_intensificador_n",
    "est_intensificador_densidad",
    "est_intensificador_grafia_n",
    "est_amenaza_n",
    "est_amenaza_densidad",
    "est_deseo_negativo_n",
    "est_deseo_negativo_densidad",
    "est_int_risa_insulto",
    "est_int_afecto_insulto",
    "est_int_vocativo_insulto",
    "est_int_risa_sin_colectivo",
    "est_n_tokens",
)
NOMBRES_LEXICO: Final[tuple[str, ...]] = (
    "lex_suma_HS",
    "lex_suma_TR",
    "lex_suma_AG",
    "lex_media_HS",
    "lex_media_TR",
    "lex_media_AG",
    "lex_max_HS",
    "lex_max_TR",
    "lex_max_AG",
    "lex_margen_HS_AG",
    "lex_margen_HS_TR",
    "lex_margen_TR_AG",
    "lex_tokens_neg_HS",
    "lex_tokens_neg_TR",
    "lex_tokens_neg_AG",
)
NOMBRES_VENTANAS: Final[tuple[str, ...]] = (
    tuple(
        f"ven_{ancla}_k{k}_media_{etiqueta}"
        for ancla in ("mencion", "colectivo")
        for k in CONFIG["ventanas_tokens"]
        for etiqueta in ETIQUETAS
    )
    + tuple(
        f"ven_{ancla}_max_{etiqueta}"
        for ancla in ("mencion", "colectivo")
        for etiqueta in ETIQUETAS
    )
    + tuple(
        f"ven_{ancla}_min_{etiqueta}"
        for ancla in ("mencion", "colectivo")
        for etiqueta in ETIQUETAS
    )
    + ("ven_n_menciones", "ven_n_colectivos")
)
NOMBRES_EMO: Final[tuple[str, ...]] = NOMBRES_ESTATICAS + NOMBRES_LEXICO + NOMBRES_VENTANAS
ANCLAS: Final[tuple[str, ...]] = ("mencion", "colectivo")

# Familias ortograficas del canal D (mapeo documentado; 1 inicial -> l).
TABLA_LEET: Final[dict[int, str]] = str.maketrans(
    {"4": "a", "3": "e", "0": "o", "1": "i", "$": "s"}
)
PATRON_LEET_1_INICIAL: Final[re.Pattern[str]] = re.compile(r"^1")
PATRON_PUNTUACION_INTERNA: Final[re.Pattern[str]] = re.compile(r"(?<=\w)[.\-_*'\"|]+(?=\w)")
PATRON_REPETICIONES: Final[re.Pattern[str]] = re.compile(r"(.)\1{2,}")

VARIANTES_EMO: Final[tuple[str, ...]] = ("EMO-EST", "EMO-LEX", "EMO-ALL", "TXT+EMO")
VARIANTES_EMO_DENSAS: Final[tuple[str, ...]] = ("EMO-EST", "EMO-LEX", "EMO-ALL")
VARIANTES_EMO_DIAGNOSTICO: Final[tuple[str, ...]] = ("TXT+EMO-NORM",)
VARIANTES_NB: Final[dict[str, tuple[str, str]]] = {
    "NB-WORD-LR": ("word", "lr"),
    "NB-CHAR-LR": ("char", "lr"),
    "NB-UNION-LR": ("union", "lr"),
    "NB-WORD-SVC": ("word", "svc"),
    "NB-CHAR-SVC": ("char", "svc"),
    "NB-UNION-SVC": ("union", "svc"),
}
VARIANTES_ORT: Final[dict[str, tuple[str, ...]]] = {
    "ORT-A": ("A",),
    "ORT-B": ("B",),
    "ORT-C": ("C",),
    "ORT-D": ("D",),
    "ORT-AB": ("A", "B"),
    "ORT-ABCD": ("A", "B", "C", "D"),
    "ORT-AE": ("A", "E"),
}
DESCRIPCION_REPRESENTACION: Final[dict[str, str]] = {
    "EMO-EST": "24 features densas de marcadores estaticos de afecto/risa/vocativo",
    "EMO-LEX": "47 features densas: scores del lexico inducido + ventanas +-3/5/10",
    "EMO-ALL": "71 features densas: marcadores estaticos + lexico inducido + ventanas",
    "TXT+EMO": "word(1,2)+char_wb(3,5) de V03 + 71 features densas de emocion",
    "NB-WORD-LR": "counts word(1,2) con log-count ratio fold-local + LR C=1.0",
    "NB-WORD-SVC": "counts word(1,2) con log-count ratio fold-local + LinearSVC C=1.0",
    "NB-CHAR-LR": "counts char_wb(3,5) con log-count ratio fold-local + LR C=1.0",
    "NB-CHAR-SVC": "counts char_wb(3,5) con log-count ratio fold-local + LinearSVC C=1.0",
    "NB-UNION-LR": "counts word(1,2)+char_wb(3,5) con log-count ratio + LR C=1.0",
    "NB-UNION-SVC": "counts word(1,2)+char_wb(3,5) con log-count ratio + LinearSVC C=1.0",
    "ORT-A": "TF-IDF word(1,2) del texto raw (A)",
    "ORT-B": "TF-IDF word(1,2) del canal colapsado (B)",
    "ORT-C": "TF-IDF word(1,2) del canal sin puntuacion interna (C)",
    "ORT-D": "TF-IDF word(1,2) del canal leetspeak (D)",
    "ORT-AB": "TF-IDF word(1,2) de raw + colapsado (A+B)",
    "ORT-ABCD": "TF-IDF word(1,2) de raw + colapsado + sin puntuacion + leetspeak (A+B+C+D)",
    "ORT-AE": "TF-IDF word(1,2) de raw + insulto enmascarado (A+E)",
}
DEFINICIONES_EXPERIMENTO: Final[dict[str, tuple[str, str, str]]] = {
    "EMO-EST": (
        "E2-09-EMO-EST",
        "Aportan los marcadores estaticos de afecto, risa y vocativo por si solos a la deteccion de toxicidad?",
        "El insulto en registro de broma o afiliacion (risa, afecto, vocativo) es el falso positivo dominante (H5)",
    ),
    "EMO-LEX": (
        "E2-09-EMO-LEX",
        "Aportan los lexicos de afecto inducidos fold-local y la polaridad local en ventanas +-3/5/10 por si solos?",
        "La polaridad local alrededor de la mencion o del colectivo separa agresion de afecto (H5/E04)",
    ),
    "EMO-ALL": (
        "E2-09-EMO",
        "Aporta el canal de emocion completo (marcadores + lexico inducido + ventanas) por si solo?",
        "La emocion del autor complementa word/char de V03 (H5)",
    ),
    "TXT+EMO": (
        "E2-09-TXT-EMO",
        "Aporta el canal de emocion completo concatenado a la representacion word+char de V03?",
        "La emocion del autor complementa word/char de V03 (H5)",
    ),
    "NB-WORD-LR": (
        "E2-15-NB-WORD",
        "Aporta el peso log-count ratio (NB-SVM) sobre word(1,2) con LR balanced?",
        "El peso NB da un modelo diverso y complementario (E2-15)",
    ),
    "NB-CHAR-LR": (
        "E2-15-NB-CHAR",
        "Aporta el peso log-count ratio (NB-SVM) sobre char_wb(3,5) con LR balanced?",
        "El peso NB sobre caracteres captura ofuscacion que word(1,2) pierde (E2-15)",
    ),
    "NB-UNION-LR": (
        "E2-15-NBSVM",
        "Aporta el NB-SVM sobre la union word(1,2)+char_wb(3,5) con LR balanced?",
        "El peso log-count ratio sobre la union de V03 da el modelo complementario mas fuerte (E2-15)",
    ),
    "NB-UNION-SVC": (
        "E2-15-NBSVM-SVC",
        "Aporta el NB-SVM sobre la union con LinearSVC balanced?",
        "El margen de la SVC sobre las features NB mejora la diversidad del canal (E2-15)",
    ),
    "ORT-A": (
        "E2-10-ORT-A",
        "Es el canal raw (A) la referencia suficiente de la linea ortografica?",
        "El raw conserva el estilo y las normalizaciones pueden destruir senal (E2-10)",
    ),
    "ORT-ABCD": (
        "E2-10-ORT-ABCD",
        "Recupera la union de canales normalizados (A+B+C+D) lexemas ofuscados que el raw pierde?",
        "Un canal normalizado recupera lexemas ofuscados y el raw conserva estilo (E2-10)",
    ),
    "ORT-AE": (
        "E2-10-ORT-AE",
        "Aporta enmascarar el insulto con <INSULT> (canal E) sobre el raw?",
        "El enmascarado separa la presencia de insulto de su lexema concreto (E2-10)",
    ),
}
SECCION_POR_VARIANTE: Final[dict[str, str]] = {
    **{variante: "E2-09" for variante in VARIANTES_EMO},
    **{variante: "E2-15" for variante in VARIANTES_NB},
    **{variante: "E2-10" for variante in VARIANTES_ORT},
}


def log(mensaje: str) -> None:
    """Imprime un mensaje de progreso con prefijo de fase.

    Args:
        mensaje: texto a imprimir.
    """
    print(f"[P5] {mensaje}", flush=True)


def _cargar_modulo(ruta: Path, nombre: str) -> Any:
    """Carga un script de experimento como modulo sin modificar el archivo.

    Args:
        ruta: ruta del script a importar.
        nombre: nombre interno del modulo en ``sys.modules``.

    Returns:
        Modulo cargado.

    Raises:
        ImportError: si el script no se puede cargar.
    """
    especificacion = importlib.util.spec_from_file_location(nombre, ruta)
    if especificacion is None or especificacion.loader is None:
        raise ImportError(f"No se pudo cargar {ruta}")
    modulo = importlib.util.module_from_spec(especificacion)
    sys.modules[nombre] = modulo
    especificacion.loader.exec_module(modulo)
    return modulo


def cargar_p2() -> Any:
    """Carga ``experiments/16_targeting_v2.py`` para reutilizar su maquinaria P2.

    Se reutilizan la carga del desarrollo y sus pliegues corregidos, la
    anotacion spaCy simbolica cacheada (lema/POS/numero/persona/modo), los
    detectores estructurales de mencion y de colectivo, la induccion fold-local
    del lexico de insulto, el log-odds informativo de Monroe et al. (2008) y el
    bootstrap pareado con la definicion de P1-P4.

    Returns:
        Modulo del experimento 16.

    Raises:
        RuntimeError: si las constantes reutilizadas no coinciden con las esperadas.
    """
    modulo = _cargar_modulo(
        PROJECT_ROOT / "experiments" / "16_targeting_v2.py", "experimento_16_p2"
    )
    if modulo.CONFIG["n_bootstrap"] != CONFIG["n_bootstrap"]:
        raise RuntimeError("El bootstrap de experiments/16 no usa 2000 remuestreos")
    if modulo.CONFIG["min_frecuencia_lexico"] != CONFIG["min_frecuencia_lexico"]:
        raise RuntimeError("La frecuencia minima del lexico de experiments/16 no coincide")
    if modulo.PARAMS_WORD != PARAMS_WORD or modulo.PARAMS_CHAR != PARAMS_CHAR:
        raise RuntimeError("La representacion word/char de experiments/16 no coincide con la esperada")
    return modulo


def cargar_receta_v03() -> Any:
    """Carga ``experiments/14_oof_corregido_v2.py`` (receta M2 de V03).

    Returns:
        Modulo del experimento 14.

    Raises:
        RuntimeError: si la receta M2 no es la esperada (C=2.0, word+char).
    """
    modulo = _cargar_modulo(
        PROJECT_ROOT / "experiments" / "14_oof_corregido_v2.py", "experimento_14_base_p5"
    )
    if modulo.MODELOS["M2_union_lr_c2"]["params"]["C"] != 2.0:
        raise RuntimeError("La receta M2 de experiments/14 no es C=2.0")
    if modulo.PARAMS_WORD != PARAMS_WORD or modulo.PARAMS_CHAR != PARAMS_CHAR:
        raise RuntimeError("La representacion word/char de experiments/14 no coincide con la esperada")
    return modulo


def concatenar_bloques(bloques: Sequence[Any]) -> sparse.csr_matrix:
    """Concatena bloques sparse y densos en una matriz CSR.

    Args:
        bloques: matrices sparse o arreglos densos en el orden deseado.

    Returns:
        Matriz CSR con todos los bloques concatenados por columnas.
    """
    return sparse.hstack([sparse.csr_matrix(bloque) for bloque in bloques], format="csr")


def ajustar_lr(
    valor_c: float,
    x_entrenamiento: sparse.csr_matrix,
    etiquetas_entrenamiento: NDArray[np.int_],
    x_prediccion: sparse.csr_matrix,
) -> NDArray[np.float64]:
    """Ajusta una regresion logistica balanceada y devuelve probabilidades.

    Args:
        valor_c: valor de ``C``.
        x_entrenamiento: matriz del pliegue de entrenamiento.
        etiquetas_entrenamiento: etiqueta binaria del pliegue.
        x_prediccion: matriz a predecir.

    Returns:
        Probabilidades de la clase positiva.
    """
    modelo = LogisticRegression(C=valor_c, **PARAMS_LR)
    modelo.fit(x_entrenamiento, etiquetas_entrenamiento)
    return modelo.predict_proba(x_prediccion)[:, 1]


def ajustar_svc(
    valor_c: float,
    x_entrenamiento: sparse.csr_matrix,
    etiquetas_entrenamiento: NDArray[np.int_],
    x_prediccion: sparse.csr_matrix,
) -> NDArray[np.float64]:
    """Ajusta una LinearSVC balanceada y devuelve la sigmoide del margen.

    Args:
        valor_c: valor de ``C``.
        x_entrenamiento: matriz del pliegue de entrenamiento.
        etiquetas_entrenamiento: etiqueta binaria del pliegue.
        x_prediccion: matriz a predecir.

    Returns:
        Pseudo-probabilidades de la clase positiva (sigmoide del margen).
    """
    modelo = LinearSVC(C=valor_c, **PARAMS_SVC)
    modelo.fit(x_entrenamiento, etiquetas_entrenamiento)
    return expit(modelo.decision_function(x_prediccion))


# ===== SECCION 1: carga del desarrollo y pliegues corregidos =====


def cargar_datos() -> tuple[pd.DataFrame, NDArray[np.int_], Any, Pliegues]:
    """Carga el desarrollo (holdout excluido), sus grupos, pliegues y la anotacion P2.

    Returns:
        Tupla ``(desarrollo, grupos, modulo_p2, pliegues)``.

    Raises:
        RuntimeError: si el desarrollo no tiene 1483 filas o tiene nulos.
    """
    p2 = cargar_p2()
    desarrollo, grupos, pliegues = p2.cargar_desarrollo()
    resumen = p2.verificar_datos(desarrollo, grupos)
    if resumen["n_filas"] != 1483:
        raise RuntimeError(f"El desarrollo debe tener 1483 filas, tiene {resumen['n_filas']}")
    return desarrollo, grupos, p2, pliegues


def evaluar_clave_base(
    desarrollo: pd.DataFrame,
    y_verdadera: NDArray[np.int_],
    pliegues: Pliegues,
    receta_v03: Any,
) -> tuple[NDArray[np.float64], list[dict[str, Any]]]:
    """Reproduce la base M2 (V03) fold a fold para fijar la referencia local.

    Args:
        desarrollo: DataFrame del desarrollo.
        y_verdadera: etiquetas verdaderas ``(n, 3)``.
        pliegues: pliegues corregidos.
        receta_v03: modulo de ``experiments/14``.

    Returns:
        Tupla ``(probabilidades, diagnostico)`` de la base M2.
    """
    textos = desarrollo["text"]
    probabilidades = np.zeros((len(desarrollo), len(ETIQUETAS)), dtype=float)
    diagnostico: list[dict[str, Any]] = []
    for numero, (idx_entrenamiento, idx_validacion) in enumerate(pliegues, start=1):
        x_entrenamiento, x_validacion = receta_v03.construir_matrices(
            textos.iloc[idx_entrenamiento], textos.iloc[idx_validacion], "union"
        )
        for indice_etiqueta in range(len(ETIQUETAS)):
            probabilidades[idx_validacion, indice_etiqueta] = ajustar_lr(
                2.0, x_entrenamiento, y_verdadera[idx_entrenamiento, indice_etiqueta], x_validacion
            )
        diagnostico.append({"pliegue": numero, "dim_union": int(x_entrenamiento.shape[1])})
        log(f"BASE pliegue {numero}: dim_union={x_entrenamiento.shape[1]}")
    return probabilidades, diagnostico


def verificar_replicacion(
    oof_base: NDArray[np.float64],
    y_verdadera: NDArray[np.int_],
    pliegue_de_fila: NDArray[np.int_],
    ids: NDArray[np.int_],
    grupos: NDArray[np.int_],
) -> dict[str, Any]:
    """Verifica que la base M2 reproduzca el OOF congelado y los pliegues.

    El OOF congelado de P0.3 no guarda la asignacion de pliegue, asi que la
    verificacion comprueba ids, etiquetas, grupos y probabilidades de M2, y la
    cobertura completa de los 5 pliegues corregidos (que son deterministas:
    ``StratifiedGroupKFold(5, shuffle=True, random_state=42)`` sobre los grupos
    fusionados).

    Args:
        oof_base: probabilidades de la base M2 recalculadas en este script.
        y_verdadera: etiquetas verdaderas.
        pliegue_de_fila: numero de pliegue de cada fila.
        ids: identificadores de las filas del desarrollo.
        grupos: grupos fusionados alineados con el desarrollo.

    Returns:
        Diccionario con las diferencias maximas y las F1 de referencia.

    Raises:
        RuntimeError: si la cobertura de pliegues es incompleta.
    """
    congelado = np.load(OOF_BASE_PATH)
    modelos = list(congelado["modelos"])
    proba_congelada = congelado["proba"][:, modelos.index("M2_union_lr_c2"), :]
    f1_base = metricas_multietiqueta(y_verdadera, predecir(oof_base, umbrales_constantes(0.5)))
    f1_v03 = metricas_multietiqueta(y_verdadera, predecir(oof_base, CONFIG["umbrales_v03"]))
    conteo_pliegues = {int(numero): int((pliegue_de_fila == numero).sum()) for numero in range(1, 6)}
    if any(conteo == 0 for conteo in conteo_pliegues.values()):
        raise RuntimeError(f"Cobertura de pliegues incompleta: {conteo_pliegues}")
    return {
        "diff_max_probabilidades_m2": float(np.abs(proba_congelada - oof_base).max()),
        "diff_max_etiquetas": float(np.abs(congelado["y_true"] - y_verdadera).max()),
        "diff_max_ids": float(np.abs(congelado["ids"] - ids).max()),
        "diff_max_grupos": float(np.abs(congelado["grupos"] - grupos).max()),
        "n_pliegues": conteo_pliegues,
        "m2_umbral_05": f1_base["f1_media_columnas"],
        "v03_umbrales": f1_v03["f1_media_columnas"],
        "v03_por_etiqueta": f1_v03["f1_por_etiqueta"],
    }


# ===== SECCION 2: diagnostico del desarrollo (EDA dirigido a P5) =====


def diagnostico_desarrollo(desarrollo: pd.DataFrame) -> dict[str, Any]:
    """Resume el desarrollo y la prevalencia de los marcadores estaticos.

    Args:
        desarrollo: DataFrame del desarrollo.

    Returns:
        Diccionario con prevalencias de etiquetas, longitudes y de cada familia
        de marcadores (documento que contiene al menos una marca).
    """
    textos_minuscula = desarrollo["text"].str.lower()
    n_tokens = textos_minuscula.str.split().str.len()
    marcadores = {}
    for familia, patrones in PATRONES_ESTATICOS.items():
        presentes = textos_minuscula.map(
            lambda texto, patrones=patrones: any(patron.search(texto) for patron in patrones)
        )
        marcadores[familia] = float(presentes.mean())
    marcadores["diminutivo"] = float(
        textos_minuscula.map(lambda texto: bool(PATRON_DIMINUTIVO.search(texto))).mean()
    )
    diagnosticos = {
        "n_filas": len(desarrollo),
        "nulos": int(desarrollo[list(ETIQUETAS) + ["text"]].isna().sum().sum()),
        "prevalencia": {etiqueta: float(desarrollo[etiqueta].mean()) for etiqueta in ETIQUETAS},
        "n_tokens": {
            "media": float(n_tokens.mean()),
            "mediana": float(n_tokens.median()),
            "max": int(n_tokens.max()),
            "min": int(n_tokens.min()),
        },
        "prevalencia_marcadores": marcadores,
        "tasa_mencion": float(textos_minuscula.str.contains(PATRON_MENCION).mean()),
        "tasa_elongacion": float(
            textos_minuscula.map(lambda texto: bool(PATRON_GRAFIA_ELONGADA.search(texto))).mean()
        ),
        "tasa_puntuacion_interna": float(
            textos_minuscula.str.contains(PATRON_PUNTUACION_INTERNA).mean()
        ),
    }
    log(
        "desarrollo n={n} nulos={nulos} prevalencia={prev}".format(
            n=diagnosticos["n_filas"],
            nulos=diagnosticos["nulos"],
            prev={clave: round(valor, 3) for clave, valor in diagnosticos["prevalencia"].items()},
        )
    )
    log(
        "marcadores (tasa de documento): "
        + ", ".join(f"{clave}={valor:.3f}" for clave, valor in marcadores.items())
    )
    return diagnosticos


# ===== SECCION 3 (E2-09): emocion y afecto (H5) =====


@dataclass(frozen=True)
class LexicoAfecto:
    """Lexico de afecto inducido SOLO con el pliegue de entrenamiento.

    Attributes:
        scores: ``etiqueta -> lema -> log-odds`` con prior informativo.
        positivos: lemas candidatos con log-odds > umbral por etiqueta.
        negativos: lemas candidatos con log-odds < -umbral por etiqueta.
        candidatos: lemas con frecuencia y documentos minimos del pliegue.
        tamanos: conteos de vocabulario, candidatos, positivos y negativos.
    """

    scores: dict[str, dict[str, float]]
    positivos: dict[str, frozenset[str]]
    negativos: dict[str, frozenset[str]]
    candidatos: frozenset[str]
    tamanos: dict[str, Any]


@dataclass(frozen=True)
class AnalisisEmocion:
    """Features de emocion de un documento.

    Attributes:
        estaticas: marcadores estaticos e interacciones.
        lexico: scores del lexico inducido y polaridad de ventanas.
    """

    estaticas: NDArray[np.float64]
    lexico: NDArray[np.float64]

    def completo(self) -> NDArray[np.float64]:
        """Devuelve el vector completo (estaticas seguido de lexico).

        Returns:
            Arreglo denso ``(len(NOMBRES_EMO),)``.
        """
        return np.concatenate([self.estaticas, self.lexico])


def umbrales_constantes(valor: float) -> dict[str, float]:
    """Devuelve un diccionario de umbrales con el mismo valor por etiqueta.

    Args:
        valor: umbral comun.

    Returns:
        Diccionario ``etiqueta -> umbral``.
    """
    return {etiqueta: valor for etiqueta in ETIQUETAS}


def _stopwords_espanol() -> frozenset[str]:
    """Devuelve las stopwords del espanol de spaCy con respaldo local.

    Returns:
        Conjunto de stopwords en minusculas.
    """
    try:
        from spacy.lang.es.stop_words import STOP_WORDS

        return frozenset(STOP_WORDS)
    except Exception:  # noqa: BLE001 - respaldo defensivo si spaCy no esta disponible
        return frozenset(
            {
                "de", "la", "el", "los", "las", "un", "una", "unos", "unas", "y", "o", "que",
                "en", "a", "al", "del", "por", "para", "con", "sin", "se", "su", "sus", "es",
                "son", "ser", "esta", "este", "esto", "estos", "estas", "como", "mas", "pero",
                "porque", "cuando", "donde", "ya", "tambien", "solo", "hay", "lo", "le", "les",
            }
        )


def inducir_lexico_afecto(
    documentos: Sequence[Any], etiquetas_pliegue: NDArray[np.int_], p2: Any
) -> LexicoAfecto:
    """Induce el lexico de afecto del pliegue de entrenamiento (fold-local).

    Reproduce el procedimiento de Monroe et al. (2008) ya usado en P2
    (``log_odds_informativo``, prior de Dirichlet con el ``alpha`` de
    experiments/16): se cuenta cada lema de contenido (ADJ, NOUN, VERB, INTJ) y
    se calculan los log-odds 1-vs-0 de HS, TR y AG. Un lema entra al lexicon si
    aparece al menos ``min_frecuencia_lexico`` veces y en
    ``min_documentos_lexico`` documentos del pliegue.

    Args:
        documentos: documentos anotados SOLO del pliegue de entrenamiento.
        etiquetas_pliegue: matriz ``(n, 3)`` de etiquetas del pliegue.
        p2: modulo del experimento 16 (aporta ``log_odds_informativo``).

    Returns:
        Lexico de afecto inducido con sus tamanos.
    """
    stopwords = _stopwords_espanol()
    pos_contenido = ("ADJ", "NOUN", "VERB", "INTJ")
    df_lema: Counter = Counter()
    frecuencia: Counter = Counter()
    df_positivo: list[Counter] = [Counter() for _ in ETIQUETAS]
    df_negativo: list[Counter] = [Counter() for _ in ETIQUETAS]
    for posicion, documento in enumerate(documentos):
        vistos: set[str] = set()
        for token in documento.tokens:
            if token.pos not in pos_contenido or token.lema in stopwords or len(token.lema) < 3:
                continue
            frecuencia[token.lema] += 1
            vistos.add(token.lema)
        for lema in vistos:
            df_lema[lema] += 1
        for indice_etiqueta in range(len(ETIQUETAS)):
            destino = df_positivo if etiquetas_pliegue[posicion, indice_etiqueta] else df_negativo
            destino[indice_etiqueta].update(vistos)
    total_frecuencia = sum(frecuencia.values())
    scores: dict[str, dict[str, float]] = {}
    for indice_etiqueta, etiqueta in enumerate(ETIQUETAS):
        n_positivos = int(etiquetas_pliegue[:, indice_etiqueta].sum())
        n_negativos = int(len(etiquetas_pliegue) - n_positivos)
        scores[etiqueta] = p2.log_odds_informativo(
            df_positivo[indice_etiqueta],
            df_negativo[indice_etiqueta],
            n_positivos,
            n_negativos,
            frecuencia,
            total_frecuencia,
        )
    candidatos = frozenset(
        lema
        for lema, apariciones in frecuencia.items()
        if apariciones >= CONFIG["min_frecuencia_lexico"]
        and df_lema[lema] >= CONFIG["min_documentos_lexico"]
    )
    umbral = CONFIG["umbral_log_odds_afecto"]
    positivos = {
        etiqueta: frozenset(lema for lema in candidatos if scores[etiqueta].get(lema, 0.0) > umbral)
        for etiqueta in ETIQUETAS
    }
    negativos = {
        etiqueta: frozenset(lema for lema in candidatos if scores[etiqueta].get(lema, 0.0) < -umbral)
        for etiqueta in ETIQUETAS
    }
    tamanos = {
        "vocabulario": len(frecuencia),
        "candidatos": len(candidatos),
        "positivos": {etiqueta: len(positivos[etiqueta]) for etiqueta in ETIQUETAS},
        "negativos": {etiqueta: len(negativos[etiqueta]) for etiqueta in ETIQUETAS},
    }
    return LexicoAfecto(
        scores=scores,
        positivos=positivos,
        negativos=negativos,
        candidatos=candidatos,
        tamanos=tamanos,
    )


def contar_familia(texto: str, patrones: Sequence[re.Pattern[str]]) -> int:
    """Cuenta las apariciones de una familia de patrones en el texto.

    Args:
        texto: texto en minusculas.
        patrones: patrones compilados de la familia.

    Returns:
        Numero de coincidencias (todas las apariciones).
    """
    return sum(len(patron.findall(texto)) for patron in patrones)


def contar_diminutivos(texto: str) -> int:
    """Cuenta diminutivos por sufijo descartando los falsos amigos documentados.

    Args:
        texto: texto en minusculas.

    Returns:
        Numero de diminutivos.
    """
    return sum(
        1 for palabra in PATRON_DIMINUTIVO.findall(texto) if palabra not in EXCLUSIONES_DIMINUTIVO
    )


def contar_vocativos(texto: str) -> tuple[int, int, int, int]:
    """Cuenta marcas de vocativo: menciones, mencion inicial, mayusculas y tras coma.

    Args:
        texto: texto original (con mayusculas).

    Returns:
        Tupla ``(menciones, mencion_inicial, mayusculas_no_iniciales, tras_coma)``.
    """
    tokens = texto.split()
    menciones = len(PATRON_MENCION.findall(texto))
    mencion_inicial = int(bool(PATRON_MENCION_INICIAL.match(texto)))
    mayusculas = sum(1 for token in tokens[1:] if PATRON_PALABRA_MAYUSCULA.match(token))
    tras_coma = len(PATRON_TRAS_COMA.findall(texto))
    return menciones, mencion_inicial, mayusculas, tras_coma


def features_estaticas(
    texto: str, n_tokens: int, n_insultos: int, tiene_colectivo: bool
) -> NDArray[np.float64]:
    """Calcula el bloque de marcadores estaticos y sus interacciones.

    Las interacciones (risa x insulto, afecto x insulto, vocativo x insulto y
    risa sin colectivo) provienen del mecanismo de falsos positivos descrito en
    `reports/error_analysis.md` (insulto interpersonal en registro de broma o
    afiliacion). Un documento sin tokens (texto vacio) no es un error: las
    densidades se calculan con denominador 1 y ``est_n_tokens`` vale 0.

    Args:
        texto: texto original del documento.
        n_tokens: numero de tokens del documento (puede ser 0).
        n_insultos: insultos del lexico inducido presentes en el documento.
        tiene_colectivo: si el documento tiene una referencia colectiva.

    Returns:
        Arreglo denso en el orden de :data:`NOMBRES_ESTATICAS`.
    """
    minuscula = texto.lower()
    denominador = max(n_tokens, 1)
    conteos = {
        familia: contar_familia(minuscula, patrones)
        for familia, patrones in PATRONES_ESTATICOS.items()
    }
    conteos["diminutivo"] = contar_diminutivos(minuscula)
    menciones, mencion_inicial, mayusculas, tras_coma = contar_vocativos(texto)
    vocativos = menciones + mayusculas + tras_coma
    intensificador_grafia = len(PATRON_GRAFIA_ELONGADA.findall(minuscula))
    risa = conteos["risa"]
    afecto = conteos["afecto"]
    return np.array(
        [
            risa,
            risa / denominador,
            afecto,
            afecto / denominador,
            conteos["diminutivo"],
            conteos["diminutivo"] / denominador,
            vocativos,
            vocativos / denominador,
            menciones,
            mencion_inicial,
            conteos["emoticono"],
            conteos["emoticono"] / denominador,
            conteos["intensificador"],
            conteos["intensificador"] / denominador,
            intensificador_grafia,
            conteos["amenaza"],
            conteos["amenaza"] / denominador,
            conteos["deseo_negativo"],
            conteos["deseo_negativo"] / denominador,
            float(risa * n_insultos),
            float(afecto * n_insultos),
            float(vocativos * n_insultos),
            float(risa * (0 if tiene_colectivo else 1)),
            float(n_tokens),
        ],
        dtype=float,
    )


def posiciones_ancla(
    documento: Any, constantes: dict[str, Any], p2: Any, colectivos: frozenset[str]
) -> tuple[list[int], list[int]]:
    """Localiza menciones y referencias colectivas del documento.

    Args:
        documento: documento anotado.
        constantes: constantes estructurales del script 06.
        p2: modulo del experimento 16 (detectores estructurales).
        colectivos: lemas colectivos minados en el pliegue de entrenamiento.

    Returns:
        Tupla ``(posiciones_mencion, posiciones_colectivo)``.
    """
    posiciones_mencion = [
        indice for indice, token in enumerate(documento.tokens) if p2.es_mencion(token)
    ]
    posiciones_colectivo = [
        indice
        for indice, token in enumerate(documento.tokens)
        if token.lema in colectivos
        or p2.es_colectivo_estructural(documento.tokens, indice, constantes)
    ]
    return posiciones_mencion, posiciones_colectivo


def _media_polaridad_ventana(
    tokens: Sequence[Any],
    posiciones: Sequence[int],
    k: int,
    scores: dict[str, float],
) -> float:
    """Media de polaridad de las ventanas +-k de todas las anclas.

    Args:
        tokens: tokens del documento.
        posiciones: posiciones de las anclas.
        k: semiancho de la ventana en tokens.
        scores: ``lema -> log-odds`` de la etiqueta.

    Returns:
        Media de la polaridad media de cada ventana (0 si no hay anclas o si
        ningun token de las ventanas esta en el lexico).
    """
    medias = []
    for posicion in posiciones:
        inicio = max(0, posicion - k)
        fin = min(len(tokens), posicion + k + 1)
        puntajes = [scores[token.lema] for token in tokens[inicio:fin] if token.lema in scores]
        if puntajes:
            medias.append(float(np.mean(puntajes)))
    return float(np.mean(medias)) if medias else 0.0


def polaridad_ventanas(
    documento: Any,
    scores: dict[str, dict[str, float]],
    posiciones_mencion: Sequence[int],
    posiciones_colectivo: Sequence[int],
) -> NDArray[np.float64]:
    """Calcula la polaridad local en ventanas +-3/5/10 tokens por ancla y etiqueta.

    La polaridad de un token es su log-odds inducido en el pliegue (0 si no esta
    en el lexico). Para cada ancla (mencion, colectivo), cada ventana k y cada
    etiqueta se agrega la media de polaridad de la ventana; tambien se reportan
    el maximo y el minimo entre ventanas y el numero de anclas.

    Args:
        documento: documento anotado.
        scores: ``etiqueta -> lema -> log-odds`` inducidos fold-local.
        posiciones_mencion: posiciones de token de las menciones.
        posiciones_colectivo: posiciones de token de las referencias colectivas.

    Returns:
        Arreglo denso en el orden de :data:`NOMBRES_VENTANAS`.
    """
    tokens = documento.tokens
    ventanas = CONFIG["ventanas_tokens"]
    matrices: dict[str, NDArray[np.float64]] = {}
    for ancla, posiciones in (("mencion", posiciones_mencion), ("colectivo", posiciones_colectivo)):
        matriz = np.zeros((len(ventanas), len(ETIQUETAS)), dtype=float)
        for indice_k, k in enumerate(ventanas):
            for indice_etiqueta, etiqueta in enumerate(ETIQUETAS):
                matriz[indice_k, indice_etiqueta] = _media_polaridad_ventana(
                    tokens, posiciones, k, scores[etiqueta]
                )
        matrices[ancla] = matriz
    valores = [
        matrices[ancla][indice_k, indice_etiqueta]
        for ancla in ANCLAS
        for indice_k in range(len(ventanas))
        for indice_etiqueta in range(len(ETIQUETAS))
    ]
    valores.extend(
        matrices[ancla].max(axis=0)[indice_etiqueta]
        for ancla in ANCLAS
        for indice_etiqueta in range(len(ETIQUETAS))
    )
    valores.extend(
        matrices[ancla].min(axis=0)[indice_etiqueta]
        for ancla in ANCLAS
        for indice_etiqueta in range(len(ETIQUETAS))
    )
    valores.extend([float(len(posiciones_mencion)), float(len(posiciones_colectivo))])
    return np.array(valores, dtype=float)


def features_lexico(documento: Any, lexico: LexicoAfecto) -> NDArray[np.float64]:
    """Calcula los scores del lexico inducido por documento.

    Un documento sin tokens usa denominador 1 en las medias (misma convencion
    que :func:`features_estaticas`).

    Args:
        documento: documento anotado.
        lexico: lexico de afecto inducido fold-local.

    Returns:
        Arreglo denso en el orden de :data:`NOMBRES_LEXICO`.
    """
    n_tokens = len(documento.tokens)
    denominador = max(n_tokens, 1)
    umbral = CONFIG["umbral_log_odds_afecto"]
    sumas = {etiqueta: 0.0 for etiqueta in ETIQUETAS}
    maximos = {etiqueta: 0.0 for etiqueta in ETIQUETAS}
    tokens_negativos = {etiqueta: 0 for etiqueta in ETIQUETAS}
    for token in documento.tokens:
        for etiqueta in ETIQUETAS:
            valor = lexico.scores[etiqueta].get(token.lema)
            if valor is None:
                continue
            sumas[etiqueta] += valor
            maximos[etiqueta] = max(maximos[etiqueta], valor)
            if valor < -umbral:
                tokens_negativos[etiqueta] += 1
    return np.array(
        [
            sumas["HS"],
            sumas["TR"],
            sumas["AG"],
            sumas["HS"] / denominador,
            sumas["TR"] / denominador,
            sumas["AG"] / denominador,
            maximos["HS"],
            maximos["TR"],
            maximos["AG"],
            sumas["HS"] - sumas["AG"],
            sumas["HS"] - sumas["TR"],
            sumas["TR"] - sumas["AG"],
            float(tokens_negativos["HS"]),
            float(tokens_negativos["TR"]),
            float(tokens_negativos["AG"]),
        ],
        dtype=float,
    )


def analizar_emocion(
    documento: Any,
    texto: str,
    lexico: LexicoAfecto,
    lexicos_p2: Any,
    constantes: dict[str, Any],
    p2: Any,
) -> AnalisisEmocion:
    """Construye las features de emocion de un documento.

    Args:
        documento: documento anotado.
        texto: texto original del documento.
        lexico: lexico de afecto inducido fold-local.
        lexicos_p2: lexicos de P2 del pliegue (insultos y colectivos).
        constantes: constantes estructurales del script 06.
        p2: modulo del experimento 16 (detectores estructurales).

    Returns:
        Analisis con los bloques estatico y lexico.
    """
    n_tokens = len(documento.tokens)
    n_insultos = sum(1 for token in documento.tokens if token.lema in lexicos_p2.insulto)
    posiciones_mencion, posiciones_colectivo = posiciones_ancla(
        documento, constantes, p2, lexicos_p2.colectivos
    )
    estaticas = features_estaticas(
        texto, n_tokens, n_insultos, bool(posiciones_colectivo)
    )
    bloque_lexico = np.concatenate(
        [
            features_lexico(documento, lexico),
            polaridad_ventanas(documento, lexico.scores, posiciones_mencion, posiciones_colectivo),
        ]
    )
    return AnalisisEmocion(estaticas=estaticas, lexico=bloque_lexico)


def evaluar_seccion_emocion(
    desarrollo: pd.DataFrame,
    documentos: Sequence[Any],
    y_verdadera: NDArray[np.int_],
    pliegues: Pliegues,
    constantes: dict[str, Any],
    p2: Any,
    receta_v03: Any,
) -> tuple[dict[str, NDArray[np.float64]], list[dict[str, Any]]]:
    """Evalua E2-09 fold a fold (features densas y concatenadas a V03).

    Args:
        desarrollo: DataFrame del desarrollo.
        documentos: documentos anotados del desarrollo.
        y_verdadera: etiquetas verdaderas ``(n, 3)``.
        pliegues: pliegues corregidos ``(entrenamiento, validacion)``.
        constantes: constantes estructurales del script 06.
        p2: modulo del experimento 16.
        receta_v03: modulo de ``experiments/14``.

    Returns:
        Tupla ``(oof, diagnostico)`` con las probabilidades por clave y el
        diagnostico por pliegue.
    """
    textos = desarrollo["text"]
    oof = {
        f"{variante}@C={valor_c}": np.zeros((len(desarrollo), len(ETIQUETAS)), dtype=float)
        for variante in VARIANTES_EMO + VARIANTES_EMO_DIAGNOSTICO
        for valor_c in CONFIG["C_variantes"]
    }
    diagnostico: list[dict[str, Any]] = []
    for numero, (idx_entrenamiento, idx_validacion) in enumerate(pliegues, start=1):
        lexico = inducir_lexico_afecto(
            [documentos[indice] for indice in idx_entrenamiento],
            y_verdadera[idx_entrenamiento],
            p2,
        )
        lexicos_p2 = p2.inducir_lexicos(
            [documentos[indice] for indice in idx_entrenamiento], y_verdadera[idx_entrenamiento]
        )
        analisis_entrenamiento = [
            analizar_emocion(
                documentos[indice], textos.iloc[indice], lexico, lexicos_p2, constantes, p2
            )
            for indice in idx_entrenamiento
        ]
        analisis_validacion = [
            analizar_emocion(
                documentos[indice], textos.iloc[indice], lexico, lexicos_p2, constantes, p2
            )
            for indice in idx_validacion
        ]
        crudos_entrenamiento = {
            "EMO-EST": np.vstack([analisis.estaticas for analisis in analisis_entrenamiento]),
            "EMO-LEX": np.vstack([analisis.lexico for analisis in analisis_entrenamiento]),
            "EMO-ALL": np.vstack([analisis.completo() for analisis in analisis_entrenamiento]),
        }
        crudos_validacion = {
            "EMO-EST": np.vstack([analisis.estaticas for analisis in analisis_validacion]),
            "EMO-LEX": np.vstack([analisis.lexico for analisis in analisis_validacion]),
            "EMO-ALL": np.vstack([analisis.completo() for analisis in analisis_validacion]),
        }
        densas_entrenamiento = {}
        densas_validacion = {}
        for variante in VARIANTES_EMO_DENSAS:
            escalador = StandardScaler().fit(crudos_entrenamiento[variante])
            densas_entrenamiento[variante] = escalador.transform(crudos_entrenamiento[variante])
            densas_validacion[variante] = escalador.transform(crudos_validacion[variante])
        bloques: dict[str, tuple[Any, Any]] = {
            variante: (densas_entrenamiento[variante], densas_validacion[variante])
            for variante in VARIANTES_EMO_DENSAS
        }
        union_entrenamiento, union_validacion = receta_v03.construir_matrices(
            textos.iloc[idx_entrenamiento], textos.iloc[idx_validacion], "union"
        )
        bloques["TXT+EMO"] = (
            concatenar_bloques([union_entrenamiento, densas_entrenamiento["EMO-ALL"]]),
            concatenar_bloques([union_validacion, densas_validacion["EMO-ALL"]]),
        )
        # Diagnostico de escala: el hstack directo deja al bloque denso con norma
        # de fila ~sqrt(71) = 8.4 frente a ~1.4 del bloque TF-IDF, de modo que la
        # LR queda dominada por las features densas. Esta variante reescala el
        # bloque denso por 1/sqrt(d) para que la comparacion no la explique la
        # escala (no interviene en la decision pre-registrada).
        factor_denso = 1.0 / np.sqrt(densas_entrenamiento["EMO-ALL"].shape[1])
        bloques["TXT+EMO-NORM"] = (
            concatenar_bloques([union_entrenamiento, densas_entrenamiento["EMO-ALL"] * factor_denso]),
            concatenar_bloques([union_validacion, densas_validacion["EMO-ALL"] * factor_denso]),
        )
        for variante, (x_entrenamiento, x_validacion) in bloques.items():
            for valor_c in CONFIG["C_variantes"]:
                for indice_etiqueta in range(len(ETIQUETAS)):
                    oof[f"{variante}@C={valor_c}"][idx_validacion, indice_etiqueta] = ajustar_lr(
                        valor_c,
                        sparse.csr_matrix(x_entrenamiento),
                        y_verdadera[idx_entrenamiento, indice_etiqueta],
                        sparse.csr_matrix(x_validacion),
                    )
        diagnostico.append(
            {
                "pliegue": numero,
                "lexico_afecto": lexico.tamanos,
                "lexico_insulto": lexicos_p2.tamanos,
                "dims": {
                    variante: int(matriz.shape[1])
                    for variante, matriz in crudos_entrenamiento.items()
                },
            }
        )
        log(
            f"E2-09 pliegue {numero}: candidatos={lexico.tamanos['candidatos']} "
            f"positivos={lexico.tamanos['positivos']} dims={diagnostico[-1]['dims']}"
        )
    return oof, diagnostico


# ===== SECCION 4 (E2-15): NB-SVM con log-count ratio fold-local =====


def log_count_ratio(
    x_entrenamiento: sparse.csr_matrix, etiquetas_entrenamiento: NDArray[np.int_], alpha: float
) -> NDArray[np.float64]:
    """Calcula el log-count ratio del NB-SVM (Wang y Manning, 2012).

    ``p`` y ``q`` son las sumas de counts de los documentos positivos y negativos
    normalizadas por su total L1 y suavizadas con ``alpha``;
    ``r = log(p / q)``.

    Args:
        x_entrenamiento: matriz de counts del pliegue de entrenamiento.
        etiquetas_entrenamiento: etiqueta binaria del pliegue.
        alpha: suavizado de Laplace.

    Returns:
        Arreglo ``(n_features,)`` con el log-count ratio.

    Raises:
        ValueError: si el pliegue de entrenamiento no tiene ambas clases.
    """
    positivos = etiquetas_entrenamiento == 1
    if not positivos.any() or positivos.all():
        raise ValueError("El pliegue de entrenamiento necesita ambas clases para el NB-SVM")
    p = np.asarray(x_entrenamiento[positivos].sum(axis=0)).ravel() + alpha
    q = np.asarray(x_entrenamiento[~positivos].sum(axis=0)).ravel() + alpha
    return np.log((p / p.sum()) / (q / q.sum()))


def construir_counts(
    textos_entrenamiento: pd.Series, textos_validacion: pd.Series, representacion: str
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Ajusta los vectorizadores de counts del pliegue y transforma validacion.

    Args:
        textos_entrenamiento: textos del pliegue de entrenamiento.
        textos_validacion: textos del pliegue de validacion.
        representacion: ``word``, ``char`` o ``union``.

    Returns:
        Tupla ``(matriz_entrenamiento, matriz_validacion)``.

    Raises:
        ValueError: si la representacion no es conocida.
    """
    if representacion == "word":
        vectorizador = CountVectorizer(**PARAMS_COUNT_WORD)
        return (
            vectorizador.fit_transform(textos_entrenamiento).tocsr(),
            vectorizador.transform(textos_validacion).tocsr(),
        )
    if representacion == "char":
        vectorizador = CountVectorizer(**PARAMS_COUNT_CHAR)
        return (
            vectorizador.fit_transform(textos_entrenamiento).tocsr(),
            vectorizador.transform(textos_validacion).tocsr(),
        )
    if representacion == "union":
        word = CountVectorizer(**PARAMS_COUNT_WORD)
        char = CountVectorizer(**PARAMS_COUNT_CHAR)
        entrenamiento = sparse.hstack(
            [word.fit_transform(textos_entrenamiento), char.fit_transform(textos_entrenamiento)],
            format="csr",
        )
        validacion = sparse.hstack(
            [word.transform(textos_validacion), char.transform(textos_validacion)], format="csr"
        )
        return entrenamiento.tocsr(), validacion.tocsr()
    raise ValueError(f"Representacion desconocida: {representacion}")


def clave_nb(variante: str) -> str:
    """Devuelve la clave OOF de una variante NB-SVM con su hiperparametro fijo.

    Args:
        variante: nombre de la variante (``NB-*``).

    Returns:
        Clave ``VARIANTE@C=x`` con el ``C`` configurado (1.0).
    """
    _, familia = VARIANTES_NB[variante]
    valor_c = CONFIG["C_lr_nb_svm"] if familia == "lr" else CONFIG["C_svc_nb_svm"]
    return f"{variante}@C={valor_c}"


def evaluar_seccion_nb_svm(
    desarrollo: pd.DataFrame, y_verdadera: NDArray[np.int_], pliegues: Pliegues
) -> tuple[dict[str, NDArray[np.float64]], list[dict[str, Any]]]:
    """Evalua la linea NB-SVM fold a fold.

    Args:
        desarrollo: DataFrame del desarrollo.
        y_verdadera: etiquetas verdaderas ``(n, 3)``.
        pliegues: pliegues corregidos ``(entrenamiento, validacion)``.

    Returns:
        Tupla ``(oof, diagnostico)`` con las pseudo-probabilidades por clave y
        el diagnostico por pliegue.
    """
    textos = desarrollo["text"]
    oof = {
        clave_nb(variante): np.zeros((len(desarrollo), len(ETIQUETAS)), dtype=float)
        for variante in VARIANTES_NB
    }
    diagnostico: list[dict[str, Any]] = []
    for numero, (idx_entrenamiento, idx_validacion) in enumerate(pliegues, start=1):
        representaciones = {representacion for representacion, _ in VARIANTES_NB.values()}
        matrices = {
            representacion: construir_counts(
                textos.iloc[idx_entrenamiento], textos.iloc[idx_validacion], representacion
            )
            for representacion in sorted(representaciones)
        }
        for variante, (representacion, familia) in VARIANTES_NB.items():
            x_entrenamiento, x_validacion = matrices[representacion]
            clave = clave_nb(variante)
            for indice_etiqueta in range(len(ETIQUETAS)):
                ratios = log_count_ratio(
                    x_entrenamiento,
                    y_verdadera[idx_entrenamiento, indice_etiqueta],
                    CONFIG["alpha_nb_svm"],
                )
                nb_entrenamiento = x_entrenamiento.multiply(ratios).tocsr()
                nb_validacion = x_validacion.multiply(ratios).tocsr()
                etiqueta = y_verdadera[idx_entrenamiento, indice_etiqueta]
                if familia == "lr":
                    puntajes = ajustar_lr(
                        CONFIG["C_lr_nb_svm"], nb_entrenamiento, etiqueta, nb_validacion
                    )
                else:
                    puntajes = ajustar_svc(
                        CONFIG["C_svc_nb_svm"], nb_entrenamiento, etiqueta, nb_validacion
                    )
                oof[clave][idx_validacion, indice_etiqueta] = puntajes
        dimensiones = {
            representacion: int(matriz[0].shape[1]) for representacion, matriz in matrices.items()
        }
        diagnostico.append({"pliegue": numero, "dims": dimensiones})
        log(f"E2-15 pliegue {numero}: dims={dimensiones}")
    return oof, diagnostico


# ===== SECCION 5 (E2-10): multicanal ortografico =====


def canal_b_colapsar_repeticiones(texto: str) -> str:
    """Colapsa repeticiones de 3 o mas caracteres a 2 (canal B).

    Args:
        texto: texto original.

    Returns:
        Texto con ``(.)\\1{2,}`` reemplazado por ``\\1\\1``.
    """
    return PATRON_REPETICIONES.sub(r"\1\1", texto)


def canal_c_sin_puntuacion_interna(texto: str) -> str:
    """Elimina puntuacion interna de palabra (canal C).

    Args:
        texto: texto original.

    Returns:
        Texto sin separadores interiores en palabras (``p.u.t.a`` -> ``puta``).
    """
    return PATRON_PUNTUACION_INTERNA.sub("", texto)


def leetspeak_token(token: str) -> str:
    """Normaliza leetspeak de un token (canal D).

    El mapeo documentado es ``4->a``, ``3->e``, ``0->o``, ``$->s`` y ``1->i``,
    salvo ``1`` inicial que se mapea a ``l`` (caso ``1a`` -> ``la``). Solo se
    aplica a tokens con al menos una letra para no destruir numeros.

    Args:
        token: token original.

    Returns:
        Token normalizado (o el original si no tiene letras).
    """
    if not any(caracter.isalpha() for caracter in token):
        return token
    inicial = "l" + token[1:] if PATRON_LEET_1_INICIAL.match(token) else token
    return inicial.translate(TABLA_LEET)


def canal_d_leetspeak(texto: str) -> str:
    """Aplica la normalizacion leetspeak token a token (canal D).

    Args:
        texto: texto original.

    Returns:
        Texto con los tokens leetspeak normalizados.
    """
    return " ".join(leetspeak_token(token) for token in texto.split())


def construir_patron_insultos(insultos: frozenset[str]) -> re.Pattern[str] | None:
    """Compila el patron de enmascarado a partir del lexico inducido (canal E).

    Args:
        insultos: lemas de insulto inducidos en el pliegue de entrenamiento.

    Returns:
        Patron con alternation ordenada por longitud descendente, o ``None`` si
        el lexico esta vacio.
    """
    if not insultos:
        return None
    alternativas = "|".join(re.escape(lema) for lema in sorted(insultos, key=len, reverse=True))
    return re.compile(rf"\b(?:{alternativas})\w*\b", re.IGNORECASE)


def canal_e_enmascarar(texto: str, patron_insultos: re.Pattern[str] | None) -> str:
    """Enmascara los insultos del lexico inducido con ``<INSULT>`` (canal E).

    Args:
        texto: texto original.
        patron_insultos: patron compilado del pliegue de entrenamiento.

    Returns:
        Texto con los insultos reemplazados por ``<INSULT>``.
    """
    if patron_insultos is None:
        return texto
    return patron_insultos.sub(" <INSULT> ", texto)


def matriz_canal_tfidf(
    serie: pd.Series, idx_entrenamiento: NDArray[np.int_], idx_validacion: NDArray[np.int_]
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Ajusta el TF-IDF word(1,2) de un canal y transforma validacion.

    Args:
        serie: textos del canal para todo el desarrollo.
        idx_entrenamiento: indices posicionales del pliegue de entrenamiento.
        idx_validacion: indices posicionales del pliegue de validacion.

    Returns:
        Tupla ``(matriz_entrenamiento, matriz_validacion)``.
    """
    vectorizador = TfidfVectorizer(**PARAMS_WORD)
    return (
        vectorizador.fit_transform(serie.iloc[idx_entrenamiento]).tocsr(),
        vectorizador.transform(serie.iloc[idx_validacion]).tocsr(),
    )


def evaluar_seccion_ortografia(
    desarrollo: pd.DataFrame,
    documentos: Sequence[Any],
    y_verdadera: NDArray[np.int_],
    pliegues: Pliegues,
    p2: Any,
) -> tuple[dict[str, NDArray[np.float64]], list[dict[str, Any]]]:
    """Evalua E2-10 fold a fold (canales A-E y sus uniones).

    Args:
        desarrollo: DataFrame del desarrollo.
        documentos: documentos anotados (se usan para inducir el lexico de insulto).
        y_verdadera: etiquetas verdaderas ``(n, 3)``.
        pliegues: pliegues corregidos ``(entrenamiento, validacion)``.
        p2: modulo del experimento 16.

    Returns:
        Tupla ``(oof, diagnostico)`` con las probabilidades por clave y el
        diagnostico por pliegue.
    """
    textos = desarrollo["text"]
    canales_estaticos = {
        "A": textos.str.lower(),
        "B": textos.map(lambda texto: canal_b_colapsar_repeticiones(texto.lower())),
        "C": textos.map(lambda texto: canal_c_sin_puntuacion_interna(texto.lower())),
        "D": textos.map(lambda texto: canal_d_leetspeak(texto.lower())),
    }
    oof = {
        f"{variante}@C={valor_c}": np.zeros((len(desarrollo), len(ETIQUETAS)), dtype=float)
        for variante in VARIANTES_ORT
        for valor_c in CONFIG["C_variantes"]
    }
    diagnostico: list[dict[str, Any]] = []
    for numero, (idx_entrenamiento, idx_validacion) in enumerate(pliegues, start=1):
        lexicos_p2 = p2.inducir_lexicos(
            [documentos[indice] for indice in idx_entrenamiento], y_verdadera[idx_entrenamiento]
        )
        patron_insultos = construir_patron_insultos(lexicos_p2.insulto)
        canales = dict(canales_estaticos)
        canales["E"] = textos.map(
            lambda texto, patron=patron_insultos: canal_e_enmascarar(texto.lower(), patron)
        )
        matrices: dict[str, tuple[sparse.csr_matrix, sparse.csr_matrix]] = {}
        for variante, canales_variante in VARIANTES_ORT.items():
            for canal in canales_variante:
                if canal not in matrices:
                    matrices[canal] = matriz_canal_tfidf(
                        canales[canal], idx_entrenamiento, idx_validacion
                    )
            x_entrenamiento = concatenar_bloques(
                [matrices[canal][0] for canal in canales_variante]
            )
            x_validacion = concatenar_bloques([matrices[canal][1] for canal in canales_variante])
            for valor_c in CONFIG["C_variantes"]:
                for indice_etiqueta in range(len(ETIQUETAS)):
                    oof[f"{variante}@C={valor_c}"][idx_validacion, indice_etiqueta] = ajustar_lr(
                        valor_c,
                        x_entrenamiento,
                        y_verdadera[idx_entrenamiento, indice_etiqueta],
                        x_validacion,
                    )
        diagnostico.append(
            {
                "pliegue": numero,
                "dims_canal": {
                    canal: int(matriz[0].shape[1]) for canal, matriz in matrices.items()
                },
                "n_insultos_enmascarados": len(lexicos_p2.insulto),
            }
        )
        log(
            f"E2-10 pliegue {numero}: dims_canal={diagnostico[-1]['dims_canal']} "
            f"insultos={len(lexicos_p2.insulto)}"
        )
    return oof, diagnostico


# ===== SECCION 6: evaluacion comun, resumen y decision (D-016) =====


def vector_umbrales(umbrales: dict[str, float]) -> NDArray[np.float64]:
    """Convierte el diccionario de umbrales en vector en el orden de etiquetas.

    Args:
        umbrales: umbral por etiqueta.

    Returns:
        Arreglo ``(3,)`` en el orden de :data:`ETIQUETAS`.
    """
    return np.array([umbrales[etiqueta] for etiqueta in ETIQUETAS])


def predecir(proba: NDArray[np.float64], umbrales: dict[str, float]) -> NDArray[np.int_]:
    """Binariza las probabilidades con umbrales por etiqueta.

    Args:
        proba: probabilidades ``(n, 3)``.
        umbrales: umbral por etiqueta.

    Returns:
        Matriz binaria ``(n, 3)``.
    """
    return (proba >= vector_umbrales(umbrales)).astype(int)


def f1_media_rapida(y_verdadera: NDArray[np.int_], prediccion: NDArray[np.int_]) -> float:
    """Calcula la F1 media por columna de forma vectorizada.

    Args:
        y_verdadera: matriz binaria verdadera ``(n, k)``.
        prediccion: matriz binaria predicha ``(n, k)``.

    Returns:
        Media de la F1 binaria de la clase positiva por columna.
    """
    f1_columnas = []
    for indice in range(y_verdadera.shape[1]):
        verdad = y_verdadera[:, indice]
        pred = prediccion[:, indice]
        verdaderos_positivos = float(np.count_nonzero((verdad == 1) & (pred == 1)))
        falsos_positivos = float(np.count_nonzero((verdad == 0) & (pred == 1)))
        falsos_negativos = float(np.count_nonzero((verdad == 1) & (pred == 0)))
        denominador = 2 * verdaderos_positivos + falsos_positivos + falsos_negativos
        f1_columnas.append(0.0 if denominador == 0 else 2 * verdaderos_positivos / denominador)
    return float(np.mean(f1_columnas))


def f1_por_pliegue(
    proba: NDArray[np.float64],
    y_verdadera: NDArray[np.int_],
    pliegue_de_fila: NDArray[np.int_],
    umbrales: dict[str, float],
) -> list[float]:
    """Calcula la F1 de cada pliegue.

    Args:
        proba: probabilidades OOF.
        y_verdadera: etiquetas verdaderas.
        pliegue_de_fila: numero de pliegue de cada fila.
        umbrales: umbral por etiqueta.

    Returns:
        Lista de F1 por pliegue en orden creciente de numero de pliegue.
    """
    prediccion = predecir(proba, umbrales)
    return [
        f1_media_rapida(
            y_verdadera[pliegue_de_fila == numero], prediccion[pliegue_de_fila == numero]
        )
        for numero in sorted(np.unique(pliegue_de_fila).tolist())
    ]


def correlacion_residuos(
    proba_base: NDArray[np.float64], proba: NDArray[np.float64], y_verdadera: NDArray[np.int_]
) -> dict[str, Any]:
    """Calcula la correlacion de errores y de probabilidades frente a la base.

    Args:
        proba_base: probabilidades de la base M2.
        proba: probabilidades de la variante.
        y_verdadera: etiquetas verdaderas.

    Returns:
        Diccionario con la correlacion de residuos (``y - p``) por etiqueta, su
        media y la correlacion media de probabilidades.
    """
    residuos = []
    probabilidades = []
    for indice, _ in enumerate(ETIQUETAS):
        verdad = y_verdadera[:, indice].astype(float)
        residuo_base = verdad - proba_base[:, indice]
        residuo_variante = verdad - proba[:, indice]
        residuos.append(
            float(np.corrcoef(residuo_base, residuo_variante)[0, 1])
            if residuo_base.std() > 0 and residuo_variante.std() > 0
            else 0.0
        )
        probabilidades.append(
            float(np.corrcoef(proba_base[:, indice], proba[:, indice])[0, 1])
            if proba_base[:, indice].std() > 0 and proba[:, indice].std() > 0
            else 0.0
        )
    return {
        "correlacion_residuos": {
            etiqueta: valor for etiqueta, valor in zip(ETIQUETAS, residuos, strict=True)
        },
        "correlacion_residuos_media": float(np.mean(residuos)),
        "correlacion_probabilidades_media": float(np.mean(probabilidades)),
    }


def resumir_clave(
    clave: str,
    seccion: str,
    proba: NDArray[np.float64],
    y_verdadera: NDArray[np.int_],
    pliegue_de_fila: NDArray[np.int_],
    proba_base: NDArray[np.float64],
    f1_pliegues_v03: list[float],
    p2: Any,
) -> dict[str, Any]:
    """Resume una clave frente a las referencias del protocolo v2.

    Args:
        clave: nombre de la clave (``VARIANTE@C=x``).
        seccion: seccion a la que pertenece (``E2-09``, ``E2-10``, ``E2-15``).
        proba: probabilidades OOF de la clave.
        y_verdadera: etiquetas verdaderas.
        pliegue_de_fila: numero de pliegue de cada fila.
        proba_base: probabilidades de la base M2.
        f1_pliegues_v03: F1 por pliegue de la base con los umbrales de V03.
        p2: modulo del experimento 16 (aporta el bootstrap pareado).

    Returns:
        Diccionario con F1 global, por etiqueta, por pliegue, deltas,
        complementariedad y diversidad frente a la base.
    """
    sin_umbral = umbrales_constantes(0.5)
    metrica = metricas_multietiqueta(y_verdadera, predecir(proba, sin_umbral))
    f1_pliegues = f1_por_pliegue(proba, y_verdadera, pliegue_de_fila, sin_umbral)
    proba_ensemble = (proba + proba_base) / 2.0
    metrica_ensemble = metricas_multietiqueta(y_verdadera, predecir(proba_ensemble, sin_umbral))
    ganancia = metrica_ensemble["f1_media_columnas"] - CONFIG["referencia_m2_umbral_05"]
    resumen: dict[str, Any] = {
        "clave": clave,
        "seccion": seccion,
        "variante": clave.split("@")[0],
        "C": float(clave.split("@C=")[1]),
        "corrected_cv": metrica["f1_media_columnas"],
        "f1_por_etiqueta": metrica["f1_por_etiqueta"],
        "f1_pliegues": f1_pliegues,
        "std_pliegues": float(np.std(f1_pliegues)),
        "delta_vs_m2_05": metrica["f1_media_columnas"] - CONFIG["referencia_m2_umbral_05"],
        "delta_vs_v03_umbrales": metrica["f1_media_columnas"] - CONFIG["referencia_v03_umbrales"],
        "delta_vs_v2a_meta": metrica["f1_media_columnas"] - CONFIG["referencia_v2a_meta"],
        "pliegues_mejorados_vs_v03": int(np.sum(np.array(f1_pliegues) > np.array(f1_pliegues_v03))),
        "ensemble_gain": ganancia,
        "corrected_cv_ensemble": metrica_ensemble["f1_media_columnas"],
    }
    resumen.update(correlacion_residuos(proba_base, proba, y_verdadera))
    if ganancia > 0:
        resumen["bootstrap"] = p2.bootstrap_delta(
            proba_base, proba_ensemble, y_verdadera, sin_umbral, sin_umbral
        )
    return resumen


def decidir_seccion(
    resumenes: dict[str, dict[str, Any]], variantes: Sequence[str], seccion: str
) -> dict[str, Any]:
    """Aplica la regla de adopcion pre-registrada en D-016 a una seccion.

    Args:
        resumenes: resumen por clave.
        variantes: variantes de la seccion.
        seccion: identificador de la seccion.

    Returns:
        Diccionario con la mejor variante y la decision.
    """
    candidatos = {
        clave: resumen
        for clave, resumen in resumenes.items()
        if resumen["seccion"] == seccion and resumen["variante"] in set(variantes)
    }
    clave, fila = max(candidatos.items(), key=lambda par: par[1]["corrected_cv"])
    cumple_nivel = (
        fila["corrected_cv"] >= CONFIG["referencia_v03_umbrales"] + CONFIG["mejora_minima"]
    )
    cumple_pliegues = fila["pliegues_mejorados_vs_v03"] >= CONFIG["pliegues_minimos_mejorados"]
    ganancia = fila.get("ensemble_gain", 0.0)
    bootstrap = fila.get("bootstrap", {})
    cumple_complementariedad = bool(
        ganancia >= CONFIG["ensemble_gain_minimo"] and bootstrap.get("ic_inferior", 0.0) > 0
    )
    adoptado = bool((cumple_nivel and cumple_pliegues) or cumple_complementariedad)
    return {
        "seccion": seccion,
        "mejor_clave": clave,
        "corrected_cv": fila["corrected_cv"],
        "criterio_nivel": {
            "objetivo": CONFIG["referencia_v03_umbrales"] + CONFIG["mejora_minima"],
            "cumple": bool(cumple_nivel),
        },
        "criterio_pliegues": {
            "objetivo": CONFIG["pliegues_minimos_mejorados"],
            "pliegues_mejorados": fila["pliegues_mejorados_vs_v03"],
            "cumple": bool(cumple_pliegues),
        },
        "criterio_complementariedad": {
            "objetivo_gain": CONFIG["ensemble_gain_minimo"],
            "ensemble_gain": ganancia,
            "ic_bootstrap": [bootstrap.get("ic_inferior"), bootstrap.get("ic_superior")],
            "cumple": cumple_complementariedad,
        },
        "adoptado": adoptado,
        "justificacion": (
            "Adoptada: supera el nivel exigido con 3/5 pliegues o demuestra complementariedad con IC > 0."
            if adoptado
            else "No adoptada: no alcanza el nivel exigido con 3/5 pliegues ni la complementariedad "
            "exigida; la linea se cierra documentando el resultado."
        ),
    }


# ===== SECCION 7: robustez, figura, artefactos y main =====


def probar_robustez(
    desarrollo: pd.DataFrame,
    p2: Any,
    lexico: LexicoAfecto,
    constantes: dict[str, Any],
) -> dict[str, Any]:
    """Comprueba que las features y canales nuevos son finitos e insensibles al orden.

    Args:
        desarrollo: DataFrame del desarrollo.
        p2: modulo del experimento 16.
        lexico: lexico de afecto del ultimo pliegue (solo para la prueba).
        constantes: constantes estructurales del script 06.

    Returns:
        Diccionario con el resultado de las pruebas.

    Raises:
        RuntimeError: si alguna feature no es finita o el orden cambia el resultado.
    """
    documentos, _ = p2.cargar_o_crear_anotaciones(desarrollo)
    lexicos_p2 = p2.inducir_lexicos(documentos, desarrollo[list(ETIQUETAS)].to_numpy())
    casos = [
        ("vacio", ""),
        ("url", "https://t.co/abc123"),
        ("mencion", "@usuario"),
        ("retuit", "RT @usuario: texto citado"),
        ("comillas", 'dijo "eres un idiota" y se fue'),
        ("elongacion", "jajajajajajajajajaja que gracioso"),
        ("leetspeak", "3r3s un p3nd3j0 m1erda 4bus4dor"),
        ("enmascarado", "eres un <INSULT> de mierda"),
    ]
    patron = construir_patron_insultos(lexicos_p2.insulto)
    resultados: dict[str, Any] = {}
    for nombre, texto in casos:
        anotacion = p2.anotar_heuristico([texto])
        analisis = analizar_emocion(anotacion[0], texto, lexico, lexicos_p2, constantes, p2)
        completo = analisis.completo()
        if not np.all(np.isfinite(completo)):
            raise RuntimeError(f"Features de emocion no finitas en el caso {nombre}")
        resultados[nombre] = {
            "n_features": int(completo.shape[0]),
            "n_tokens": len(anotacion[0].tokens),
            "canal_b": canal_b_colapsar_repeticiones(texto),
            "canal_c": canal_c_sin_puntuacion_interna(texto),
            "canal_d": canal_d_leetspeak(texto),
            "canal_e": canal_e_enmascarar(texto, patron),
        }
    muestras = list(range(0, len(desarrollo), 37))
    textos_muestra = [desarrollo["text"].iloc[indice] for indice in muestras]
    anotaciones = p2.anotar_heuristico(textos_muestra)
    inverso = list(range(len(textos_muestra) - 1, -1, -1))
    directo = np.vstack(
        [
            analizar_emocion(
                anotaciones[posicion], textos_muestra[posicion], lexico, lexicos_p2, constantes, p2
            ).completo()
            for posicion in range(len(textos_muestra))
        ]
    )
    invertido = np.vstack(
        [
            analizar_emocion(
                anotaciones[posicion],
                textos_muestra[posicion],
                lexico,
                lexicos_p2,
                constantes,
                p2,
            ).completo()
            for posicion in inverso
        ]
    )
    if not np.allclose(directo, invertido[::-1]):
        raise RuntimeError("Las features de emocion dependen del orden de las filas")
    resultados["independiente_del_orden"] = True
    log(f"robustez: {len(casos)} casos limite sin NaN/infinito; features independientes del orden")
    return resultados


def generar_figura(
    resumenes: dict[str, dict[str, Any]],
    decisiones: dict[str, dict[str, Any]],
    f1_pliegues_v03: list[float],
) -> None:
    """Genera la figura comparativa de las tres lineas de P5.

    Args:
        resumenes: resumen por clave.
        decisiones: decision por seccion.
        f1_pliegues_v03: F1 por pliegue de V03 con umbrales congelados.
    """
    mejores_por_variante: dict[str, dict[str, Any]] = {}
    for resumen in resumenes.values():
        variante = resumen["variante"]
        if variante not in mejores_por_variante or (
            resumen["corrected_cv"] > mejores_por_variante[variante]["corrected_cv"]
        ):
            mejores_por_variante[variante] = resumen
    ordenadas = sorted(mejores_por_variante.items(), key=lambda par: par[1]["corrected_cv"])
    figura, ejes = plt.subplots(1, 3, figsize=(19, 6))
    colores = {"E2-09": "#1f77b4", "E2-10": "#2ca02c", "E2-15": "#d62728", "BASE": "#7f7f7f"}
    nombres = [clave for clave, _ in ordenadas]
    valores = [resumen["corrected_cv"] for _, resumen in ordenadas]
    colores_barras = [colores[resumen["seccion"]] for _, resumen in ordenadas]
    ejes[0].barh(nombres, valores, color=colores_barras)
    ejes[0].axvline(
        CONFIG["referencia_m2_umbral_05"], color="gray", linestyle="--", label="M2 0.5 (0.7718)"
    )
    ejes[0].axvline(
        CONFIG["referencia_v03_umbrales"], color="black", linestyle=":", label="V03 umbrales (0.7772)"
    )
    ejes[0].axvline(
        CONFIG["referencia_v03_umbrales"] + CONFIG["mejora_minima"],
        color="red",
        linestyle="-.",
        label="nivel exigido (0.7822)",
    )
    ejes[0].set_xlabel("corrected_cv (umbral 0.5)")
    ejes[0].set_title("Variantes de P5 (mejor C por variante)")
    ejes[0].legend(fontsize=8)
    ejes[0].tick_params(axis="y", labelsize=8)

    for seccion, decision in decisiones.items():
        mejor = resumenes[decision["mejor_clave"]]
        ejes[1].plot(
            range(1, 6), mejor["f1_pliegues"], marker="o", label=f"{seccion} {mejor['clave']}"
        )
    ejes[1].plot(
        range(1, 6), f1_pliegues_v03, marker="s", color="black", linestyle="--", label="V03 umbrales"
    )
    ejes[1].set_xlabel("pliegue")
    ejes[1].set_ylabel("F1 media por columna")
    ejes[1].set_title("F1 por pliegue de la mejor variante por seccion")
    ejes[1].legend(fontsize=8)
    ejes[1].grid(alpha=0.3)

    nb = {
        resumen["clave"]: resumen["correlacion_residuos_media"]
        for resumen in resumenes.values()
        if resumen["seccion"] == "E2-15"
    }
    ejes[2].bar(list(nb), list(nb.values()), color="#d62728")
    ejes[2].set_ylabel("correlacion de residuos vs M2")
    ejes[2].set_title("Diversidad del canal NB-SVM")
    ejes[2].tick_params(axis="x", labelrotation=60, labelsize=8)
    for etiqueta in ejes[2].get_xticklabels():
        etiqueta.set_horizontalalignment("right")
    figura.tight_layout()
    FIGURA_PATH.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(FIGURA_PATH, dpi=150, bbox_inches="tight")
    plt.close(figura)
    log(f"figura escrita: {FIGURA_PATH.name}")


def escribir_csv(resumenes: dict[str, dict[str, Any]], mejores: dict[str, str]) -> pd.DataFrame:
    """Escribe el CSV con una fila por clave y marca la mejor de cada variante.

    Args:
        resumenes: resumen por clave.
        mejores: clave ganadora por variante.

    Returns:
        DataFrame escrito a disco.
    """
    filas = []
    for clave, resumen in sorted(resumenes.items()):
        bootstrap = resumen.get("bootstrap", {})
        filas.append(
            {
                "seccion": resumen["seccion"],
                "clave": resumen["clave"],
                "variante": resumen["variante"],
                "C": resumen["C"],
                "corrected_cv": round(resumen["corrected_cv"], 4),
                "f1_HS": round(resumen["f1_por_etiqueta"]["HS"], 4),
                "f1_TR": round(resumen["f1_por_etiqueta"]["TR"], 4),
                "f1_AG": round(resumen["f1_por_etiqueta"]["AG"], 4),
                "std_pliegues": round(resumen["std_pliegues"], 4),
                "f1_pliegues": json.dumps([round(valor, 4) for valor in resumen["f1_pliegues"]]),
                "delta_vs_m2_05": round(resumen["delta_vs_m2_05"], 4),
                "delta_vs_v03_umbrales": round(resumen["delta_vs_v03_umbrales"], 4),
                "delta_vs_v2a_meta": round(resumen["delta_vs_v2a_meta"], 4),
                "pliegues_mejorados_vs_v03": resumen["pliegues_mejorados_vs_v03"],
                "ensemble_gain": round(resumen["ensemble_gain"], 4),
                "corrected_cv_ensemble": round(resumen["corrected_cv_ensemble"], 4),
                "bootstrap_ic_inferior": bootstrap.get("ic_inferior"),
                "bootstrap_ic_superior": bootstrap.get("ic_superior"),
                "bootstrap_fraccion_positiva": bootstrap.get("fraccion_positiva"),
                "correlacion_residuos_media": round(resumen["correlacion_residuos_media"], 4),
                "correlacion_probabilidades_media": round(
                    resumen["correlacion_probabilidades_media"], 4
                ),
                "seleccionada": int(clave == mejores.get(resumen["variante"])),
            }
        )
    tabla = pd.DataFrame(filas)
    REPORTE_CSV.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(REPORTE_CSV, index=False)
    log(f"CSV escrito: {REPORTE_CSV.name} ({len(tabla)} claves)")
    return tabla


def seleccionar_mejores(
    resumenes: dict[str, dict[str, Any]], variantes: Sequence[str]
) -> dict[str, str]:
    """Selecciona la mejor clave de cada variante por corrected_cv.

    Args:
        resumenes: resumen por clave.
        variantes: variantes a considerar.

    Returns:
        Diccionario ``variante -> clave``.
    """
    return {
        variante: max(
            (clave for clave, resumen in resumenes.items() if resumen["variante"] == variante),
            key=lambda clave: resumenes[clave]["corrected_cv"],
        )
        for variante in variantes
    }


def guardar_oof(
    desarrollo: pd.DataFrame,
    grupos: NDArray[np.int_],
    y_verdadera: NDArray[np.int_],
    pliegue_de_fila: NDArray[np.int_],
    oof: dict[str, NDArray[np.float64]],
    canales: Sequence[str],
) -> list[str]:
    """Guarda los canales OOF que pueden alimentar el meta de P6.

    Args:
        desarrollo: DataFrame del desarrollo.
        grupos: grupos fusionados alineados con el desarrollo.
        y_verdadera: etiquetas verdaderas.
        pliegue_de_fila: numero de pliegue de cada fila.
        oof: probabilidades por clave.
        canales: claves de las que se guarda la probabilidad.

    Returns:
        Lista de canales guardados.
    """
    np.savez_compressed(
        OOF_PATH,
        ids=desarrollo["id"].astype(int).to_numpy(),
        y_true=y_verdadera,
        grupos=grupos,
        pliegues=pliegue_de_fila,
        etiquetas=np.array(list(ETIQUETAS)),
        canales=np.array(list(canales)),
        proba=np.stack([oof[canal] for canal in canales], axis=0),
    )
    log(f"OOF escrito: {OOF_PATH.name} ({len(canales)} canales)")
    return list(canales)


def registrar_experimentos(
    resumenes: dict[str, dict[str, Any]],
    mejores: dict[str, str],
    decisiones: dict[str, dict[str, Any]],
    duracion: float,
) -> None:
    """Anade las filas E2-09/E2-10/E2-15 a ``experiments.csv``.

    Args:
        resumenes: resumen por clave.
        mejores: clave ganadora por variante.
        decisiones: decision por seccion.
        duracion: segundos totales del experimento.
    """
    filas = []
    for variante, (identificador, pregunta, hipotesis) in DEFINICIONES_EXPERIMENTO.items():
        clave = mejores[variante]
        resumen = resumenes[clave]
        decision = decisiones[resumen["seccion"]]
        es_svc = variante.endswith("SVC")
        filas.append(
            {
                "experiment_id": identificador,
                "competition": "KAGGLE_TOXIC_2026_PUJ",
                "question": pregunta,
                "hypothesis": hipotesis,
                "preprocessing": (
                    "texto crudo; anotacion spaCy SIMBOLICA cacheada de P2 (lema/POS) para el canal "
                    "de emocion; sin vectores, sin modelos preentrenados y sin lexicos externos"
                ),
                "feature_groups": {
                    "E2-09": "+EMOTION_AFFECT",
                    "E2-10": "+ORTHOGRAPHIC_CHANNEL",
                    "E2-15": "+NB_LOG_COUNT_RATIO",
                }[resumen["seccion"]],
                "representation": DESCRIPCION_REPRESENTACION[variante],
                "model": (
                    f"LinearSVC C={resumen['C']} balanced"
                    if es_svc
                    else f"LogisticRegression C={resumen['C']} balanced"
                ),
                "hyperparameters": json.dumps(
                    {"C": resumen["C"], **(PARAMS_SVC if es_svc else PARAMS_LR)},
                    ensure_ascii=False,
                ),
                "validation": "StratifiedGroupKFold(5) sobre desarrollo, grupos fusionados (D-016)",
                "f1": round(resumen["corrected_cv"], 4),
                "macro_f1": "",
                "weighted_f1": "",
                "precision": "",
                "recall": "",
                "training_time": round(duracion, 2),
                "dimension": "",
                "result": json.dumps(
                    {
                        "corrected_cv": resumen["corrected_cv"],
                        "f1_por_etiqueta": resumen["f1_por_etiqueta"],
                        "pliegues": resumen["f1_pliegues"],
                        "delta_vs_v03_umbrales": resumen["delta_vs_v03_umbrales"],
                        "delta_vs_v2a_meta": resumen["delta_vs_v2a_meta"],
                        "pliegues_mejorados_vs_v03": resumen["pliegues_mejorados_vs_v03"],
                        "ensemble_gain": resumen.get("ensemble_gain"),
                        "bootstrap": resumen.get("bootstrap"),
                        "correlacion_residuos_media": resumen["correlacion_residuos_media"],
                    },
                    ensure_ascii=False,
                ),
                "interpretation": (
                    f"Mejor C={resumen['C']} elegido por corrected_cv interno (cota optimista de "
                    f"seleccion); delta vs V03 {resumen['delta_vs_v03_umbrales']:+.4f} y vs v2-A "
                    f"{resumen['delta_vs_v2a_meta']:+.4f}"
                ),
                "decision": (
                    f"ADOPTADA: {decision['justificacion']}"
                    if clave == decision["mejor_clave"] and decision["adoptado"]
                    else (
                        f"NO adoptada: {resumen['delta_vs_v03_umbrales']:+.4f} vs V03 "
                        f"({resumen['pliegues_mejorados_vs_v03']}/5 pliegues)"
                    )
                ),
                "date": CONFIG["fecha"],
                "script": "experiments/20_emocion_nbsvm_ortografia_v2.py",
                "notes": (
                    "holdout congelado no utilizado; sin fuga (lexicos, vectorizadores, pesos NB y "
                    "escaladores solo del pliegue de entrenamiento)"
                ),
            }
        )
    existente = pd.read_csv(EXPERIMENTS_CSV)
    identificadores = {fila["experiment_id"] for fila in filas}
    existente = existente[~existente["experiment_id"].isin(identificadores)]
    existente.to_csv(EXPERIMENTS_CSV, index=False)
    pd.DataFrame(filas).to_csv(EXPERIMENTS_CSV, mode="a", header=False, index=False)
    log(f"filas registradas: {sorted(identificadores)}")


def escribir_contexto(lineas: list[str]) -> None:
    """Reemplaza o anade la seccion de P5 en ``reports/contexto.md``.

    Args:
        lineas: lineas de la seccion (sin encabezado); maximo 20.

    Raises:
        ValueError: si la seccion excede 20 lineas.
    """
    if len(lineas) > 20:
        raise ValueError("La seccion de contexto no puede superar 20 lineas")
    encabezado = "## Fase v2 - P5 (2026-09-25)"
    bloque = "\n".join([encabezado, *lineas])
    texto = CONTEXTO_MD.read_text(encoding="utf-8")
    if encabezado in texto:
        inicio = texto.index(encabezado)
        fin = texto.find("\n## ", inicio + len(encabezado))
        fin = len(texto) if fin == -1 else fin + 1
        texto = texto[:inicio] + bloque + "\n" + texto[fin:]
    else:
        texto = texto.rstrip("\n") + "\n\n" + bloque + "\n"
    CONTEXTO_MD.write_text(texto, encoding="utf-8")
    log("seccion P5 escrita en reports/contexto.md")


def actualizar_todo(
    resumenes: dict[str, dict[str, Any]],
    mejores: dict[str, str],
    decisiones: dict[str, dict[str, Any]],
) -> None:
    """Actualiza P5.1-P5.4 y el gate P5 en ``tasks/todo.md``.

    Args:
        resumenes: resumen por clave.
        mejores: clave ganadora por variante.
        decisiones: decision por seccion.
    """

    def linea(variante: str) -> str:
        """Devuelve el texto resumido de la mejor clave de una variante.

        Args:
            variante: nombre de la variante.

        Returns:
            Texto compacto con clave, corrected_cv, delta y pliegues.
        """
        resumen = resumenes[mejores[variante]]
        return (
            f"{resumen['clave']} corrected_cv {resumen['corrected_cv']:.4f} "
            f"(delta {resumen['delta_vs_v03_umbrales']:+.4f} vs V03, "
            f"pliegues {resumen['pliegues_mejorados_vs_v03']}/5)"
        )

    reemplazos = [
        (
            "- [pendiente] P5.1 Emocion H5 con lexicones documentados (licencia, cobertura, riesgo)",
            (
                "- [hecho] P5.1 Emocion H5 (sin lexicos externos: `data/external/lexicons/` vacio y la "
                "sesion prohibe descargas; se usa el lexico inducido fold-local documentado en el "
                f"JSON): EMO-EST {linea('EMO-EST')}; EMO-LEX {linea('EMO-LEX')}; "
                f"EMO-ALL {linea('EMO-ALL')}; TXT+EMO {linea('TXT+EMO')}"
            ),
        ),
        (
            "- [pendiente] P5.2 NB-SVM (log-count ratio fold-local) en word/char/char_wb",
            (
                "- [hecho] P5.2 NB-SVM (log-count ratio fold-local) en word/char/union: "
                f"NB-WORD {linea('NB-WORD-LR')}; NB-CHAR {linea('NB-CHAR-LR')}; "
                f"NB-UNION {linea('NB-UNION-LR')}; NB-UNION-SVC {linea('NB-UNION-SVC')}"
            ),
        ),
        (
            "- [pendiente] P5.3 Multicanal ortografico (raw + normalizado + patrones)",
            (
                "- [hecho] P5.3 Multicanal ortografico (raw + normalizado + patrones): "
                f"ORT-A {linea('ORT-A')}; ORT-ABCD {linea('ORT-ABCD')}; ORT-AE {linea('ORT-AE')}"
            ),
        ),
        (
            "- [pendiente] P5.4 Gate P5",
            (
                "- [hecho] P5.4 Gate P5: "
                + (
                    "ADOPTADO"
                    if any(decision["adoptado"] for decision in decisiones.values())
                    else "CERRADO sin mejora"
                )
                + f" (E2-09 {decisiones['E2-09']['mejor_clave']} "
                f"{decisiones['E2-09']['corrected_cv']:.4f}; E2-15 "
                f"{decisiones['E2-15']['mejor_clave']} {decisiones['E2-15']['corrected_cv']:.4f}; "
                f"E2-10 {decisiones['E2-10']['mejor_clave']} "
                f"{decisiones['E2-10']['corrected_cv']:.4f})"
            ),
        ),
    ]
    texto = TODO_MD.read_text(encoding="utf-8")
    for viejo, nuevo in reemplazos:
        if viejo in texto:
            texto = texto.replace(viejo, nuevo)
    ancla = "| P4 | 2026-09-25 |"
    if ancla in texto:
        lineas = texto.split("\n")
        posicion = max(indice for indice, linea in enumerate(lineas) if linea.startswith(ancla))
        for desplazamiento, (seccion, etiqueta) in enumerate(
            (("E2-09", "P5 emocion"), ("E2-15", "P5 NB-SVM"), ("E2-10", "P5 ortografia"))
        ):
            if any(linea.startswith(f"| P5 ({etiqueta}) |") for linea in lineas):
                continue
            decision = decisiones[seccion]
            mejor = resumenes[decision["mejor_clave"]]
            filas_gate = (
                f"| P5 ({etiqueta}) | 2026-09-25 | "
                f"{'SUPERADO' if decision['adoptado'] else 'CERRADO (sin mejora)'}. "
                f"Mejor variante {decision['mejor_clave']} con corrected_cv "
                f"{decision['corrected_cv']:.4f} "
                f"({decision['corrected_cv'] - CONFIG['referencia_v03_umbrales']:+.4f} vs V03 0.7772, "
                f"{mejor['pliegues_mejorados_vs_v03']}/5 pliegues, ensemble_gain "
                f"{mejor.get('ensemble_gain', float('nan')):+.4f}). Evidencia: "
                "`experiments/20_emocion_nbsvm_ortografia_v2.py`, "
                "`reports/emocion_nbsvm_ortografia_v2.json` |"
            )
            lineas.insert(posicion + 1 + desplazamiento, filas_gate)
        texto = "\n".join(lineas)
    TODO_MD.write_text(texto, encoding="utf-8")
    log("tasks/todo.md actualizado (P5.1-P5.4 y gates)")


def escribir_json(
    resumenes: dict[str, dict[str, Any]],
    mejores: dict[str, str],
    decisiones: dict[str, dict[str, Any]],
    verificaciones: dict[str, Any],
    diagnosticos: dict[str, list[dict[str, Any]]],
    robustez: dict[str, Any],
    canales_oof: Sequence[str],
    duracion: float,
) -> None:
    """Escribe el reporte JSON completo del experimento.

    Args:
        resumenes: resumen por clave.
        mejores: clave ganadora por variante.
        decisiones: decision por seccion.
        verificaciones: comprobaciones de replicacion y diagnostico del EDA.
        diagnosticos: diagnostico por pliegue de cada seccion.
        robustez: resultado de las pruebas de robustez.
        canales_oof: canales guardados en el npz.
        duracion: segundos totales.
    """
    reporte = {
        "fecha": CONFIG["fecha"],
        "protocolo": "v2 (D-016): grupos = near-duplicado UNION pseudo-autor, StratifiedGroupKFold(5)",
        "referencias": {
            "m2_umbral_05": CONFIG["referencia_m2_umbral_05"],
            "v03_umbrales": CONFIG["referencia_v03_umbrales"],
            "v03_por_etiqueta": CONFIG["v03_por_etiqueta"],
            "candidato_v2a_meta": CONFIG["referencia_v2a_meta"],
            "nivel_exigido": CONFIG["referencia_v03_umbrales"] + CONFIG["mejora_minima"],
        },
        "lexicos_externos": {
            "usados": [],
            "motivo": (
                "data/external/lexicons/ esta vacio y la sesion prohibe descargas e instalaciones; "
                "se usa solo el lexico de afecto inducido fold-local (autocontenido, sin licencia de "
                "terceros) con log-odds suavizado de Monroe et al. (2008)."
            ),
            "candidatos_evaluados": [
                {
                    "nombre": "NRC Emotion Lexicon / EmoLex (Mohammad y Turney)",
                    "idioma": "ingles",
                    "licencia": "uso libre con atribucion, requiere confirmacion de uso",
                    "motivo_no_uso": "descarga manual con confirmacion; sin cobertura de jerga espanola",
                },
                {
                    "nombre": "ML-SentiCon / DAL (grupo TASS)",
                    "idioma": "espanol",
                    "licencia": "CC BY-NC-SA (no comercial)",
                    "motivo_no_uso": "descarga manual desde el repositorio TASS; licencia no comercial",
                },
                {
                    "nombre": "VADER-es (pysentimiento)",
                    "idioma": "espanol",
                    "licencia": "MIT (codigo)",
                    "motivo_no_uso": "depende de transformers y modelos preentrenados, prohibidos",
                },
                {
                    "nombre": "WordNet-Affect via NLTK/OMW local",
                    "idioma": "ingles",
                    "licencia": "WordNet License (permisiva)",
                    "motivo_no_uso": "solo ingles; el mapeo por OMW no cubre jerga de twitter",
                },
            ],
        },
        "config": {clave: valor for clave, valor in CONFIG.items()},
        "verificaciones": verificaciones,
        "robustez": robustez,
        "diagnostico_por_pliegue": diagnosticos,
        "variantes": {
            variante: {
                "seccion": SECCION_POR_VARIANTE[variante],
                "mejor_clave": mejores[variante],
                "mejor_C": resumenes[mejores[variante]]["C"],
                "por_C": {
                    f"C={valor_c}": round(resumenes[f"{variante}@C={valor_c}"]["corrected_cv"], 4)
                    for valor_c in CONFIG["C_variantes"]
                    if f"{variante}@C={valor_c}" in resumenes
                },
                "resumen": resumenes[mejores[variante]],
            }
            for variante in SECCION_POR_VARIANTE
        },
        "todas_las_claves": {
            clave: {
                "seccion": resumen["seccion"],
                "corrected_cv": round(resumen["corrected_cv"], 4),
                "f1_por_etiqueta": resumen["f1_por_etiqueta"],
                "f1_pliegues": resumen["f1_pliegues"],
                "delta_vs_v03_umbrales": resumen["delta_vs_v03_umbrales"],
                "delta_vs_v2a_meta": resumen["delta_vs_v2a_meta"],
                "pliegues_mejorados_vs_v03": resumen["pliegues_mejorados_vs_v03"],
                "ensemble_gain": resumen["ensemble_gain"],
                "corrected_cv_ensemble": resumen["corrected_cv_ensemble"],
                "correlacion_residuos_media": resumen["correlacion_residuos_media"],
                "correlacion_probabilidades_media": resumen["correlacion_probabilidades_media"],
                "bootstrap": resumen.get("bootstrap"),
            }
            for clave, resumen in sorted(resumenes.items())
        },
        "decisiones": decisiones,
        "canales_oof": list(canales_oof),
        "duracion_segundos": round(duracion, 2),
        "artefactos": [
            str(REPORTE_CSV.relative_to(PROJECT_ROOT)),
            str(REPORTE_JSON.relative_to(PROJECT_ROOT)),
            str(FIGURA_PATH.relative_to(PROJECT_ROOT)),
            str(OOF_PATH.relative_to(PROJECT_ROOT)),
        ],
    }
    REPORTE_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORTE_JSON.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"reporte JSON escrito: {REPORTE_JSON.name}")


def construir_contexto(
    resumenes: dict[str, dict[str, Any]],
    mejores: dict[str, str],
    decisiones: dict[str, dict[str, Any]],
    verificaciones: dict[str, Any],
    duracion: float,
) -> list[str]:
    """Construye las lineas (maximo 20) de la seccion de P5 en contexto.md.

    Args:
        resumenes: resumen por clave.
        mejores: clave ganadora por variante.
        decisiones: decision por seccion.
        verificaciones: comprobaciones de replicacion.
        duracion: segundos totales.

    Returns:
        Lista de lineas sin encabezado.
    """

    def resumen_texto(variante: str) -> str:
        """Devuelve el texto compacto de la mejor clave de una variante.

        Args:
            variante: nombre de la variante.

        Returns:
            Texto con clave, corrected_cv, delta y pliegues.
        """
        resumen = resumenes[mejores[variante]]
        return (
            f"{resumen['clave']} {resumen['corrected_cv']:.4f} "
            f"({resumen['delta_vs_v03_umbrales']:+.4f}, {resumen['pliegues_mejorados_vs_v03']}/5)"
        )

    lineas_decision = [
        f"Gate {seccion}: {'ADOPTADO' if decision['adoptado'] else 'CERRADO sin mejora'} "
        f"({decision['mejor_clave']} {decision['corrected_cv']:.4f}, ensemble_gain "
        f"{resumenes[decision['mejor_clave']]['ensemble_gain']:+.4f})."
        for seccion, decision in decisiones.items()
    ]
    return [
        (
            "P5 (E2-09 emocion/afecto, E2-15 NB-SVM y E2-10 multicanal ortografico) sobre el "
            "desarrollo (n=1483), pliegues corregidos v2, holdout congelado intacto."
        ),
        (
            f"Base reproducida: M2 a 0.5 {verificaciones['m2_umbral_05']:.4f} (dif max vs OOF "
            f"congelado {verificaciones['diff_max_probabilidades_m2']:.1e}); V03 con umbrales "
            f"{verificaciones['v03_umbrales']:.4f}; pliegues identicos al OOF congelado."
        ),
        (
            "Lexicos externos: ninguno. `data/external/lexicons/` vacio y la sesion prohibe "
            "descargas; solo lexico de afecto inducido fold-local (Monroe 2008) documentado en el "
            "JSON (NRC EmoLex, ML-SentiCon, VADER-es y WordNet-Affect evaluados y descartados)."
        ),
        "E2-09 marcadores estaticos: " + resumen_texto("EMO-EST"),
        "E2-09 lexico inducido + ventanas +-3/5/10: " + resumen_texto("EMO-LEX"),
        "E2-09 canal completo solo: " + resumen_texto("EMO-ALL"),
        "E2-09 union con V03: "
        + resumen_texto("TXT+EMO")
        + "; con el bloque denso reescalado (diagnostico de escala, fuera de la decision): "
        + resumen_texto("TXT+EMO-NORM"),
        "E2-15 NB-SVM word: " + resumen_texto("NB-WORD-LR") + "; char: " + resumen_texto("NB-CHAR-LR"),
        "E2-15 NB-SVM union LR: "
        + resumen_texto("NB-UNION-LR")
        + f" (correlacion de residuos vs M2 {resumenes[mejores['NB-UNION-LR']]['correlacion_residuos_media']:.3f})"
        + "; SVC: "
        + resumen_texto("NB-UNION-SVC"),
        "E2-10 raw (A): " + resumen_texto("ORT-A"),
        "E2-10 colapso de repeticiones: "
        + resumen_texto("ORT-B")
        + "; sin puntuacion interna: "
        + resumen_texto("ORT-C"),
        "E2-10 leetspeak: "
        + resumen_texto("ORT-D")
        + "; union normalizada A+B+C+D: "
        + resumen_texto("ORT-ABCD"),
        "E2-10 insulto enmascarado sobre raw: " + resumen_texto("ORT-AE"),
        (
            "Complementariedad: el ensemble_gain es la media de probabilidades con M2 menos 0.7718 "
            "(definicion de P1-P4); la concatenacion aprendida de scores pertenece a P6."
        ),
        *lineas_decision,
        (
            "Canales OOF guardados en `features/v2_oof_emocion_nbsvm_ortografia.npz` para el meta "
            "de P6 (4 canales de emocion, 4 de NB-SVM y 3 ortograficos)."
        ),
        (
            "Sin fuga: lexicos, inventario colectivo, vectorizadores, pesos NB, escaladores y "
            "modelos solo con el pliegue de entrenamiento; umbral y seleccion de C sobre OOF del "
            "desarrollo."
        ),
        (
            f"Duracion: {duracion:.1f} s. Script: experiments/20_emocion_nbsvm_ortografia_v2.py. "
            "Artefactos: reports/emocion_nbsvm_ortografia_v2.csv y .json, "
            "reports/figuras/fig_31_emocion_nbsvm_v2.png."
        ),
    ]


def main() -> None:
    """Ejecuta P5 (E2-09, E2-15, E2-10) y guarda todos los artefactos."""
    inicio = time.perf_counter()
    desarrollo, grupos, p2, pliegues = cargar_datos()
    y_verdadera = desarrollo[list(ETIQUETAS)].to_numpy()
    documentos, modo_anotacion = p2.cargar_o_crear_anotaciones(desarrollo)
    constantes, origen_constantes = p2.cargar_constantes_error_analysis()
    receta_v03 = cargar_receta_v03()
    log(f"anotacion={modo_anotacion} constantes={origen_constantes}")

    diagnosticos_eda = diagnostico_desarrollo(desarrollo)

    pliegue_de_fila = np.zeros(len(desarrollo), dtype=int)
    for numero, (_, idx_validacion) in enumerate(pliegues, start=1):
        pliegue_de_fila[idx_validacion] = numero

    oof_base, _ = evaluar_clave_base(desarrollo, y_verdadera, pliegues, receta_v03)
    verificaciones = verificar_replicacion(
        oof_base,
        y_verdadera,
        pliegue_de_fila,
        desarrollo["id"].astype(int).to_numpy(),
        grupos,
    )
    if verificaciones["diff_max_probabilidades_m2"] > 1e-9:
        raise RuntimeError(
            "La base M2 no reproduce el OOF congelado: "
            f"{verificaciones['diff_max_probabilidades_m2']:.3e}"
        )
    if verificaciones["diff_max_ids"] != 0 or verificaciones["diff_max_grupos"] != 0:
        raise RuntimeError("Los ids o los grupos no coinciden con el OOF congelado")
    f1_pliegues_v03_umbrales = f1_por_pliegue(
        oof_base, y_verdadera, pliegue_de_fila, CONFIG["umbrales_v03"]
    )

    oof_emocion, diag_emocion = evaluar_seccion_emocion(
        desarrollo, documentos, y_verdadera, pliegues, constantes, p2, receta_v03
    )
    lexico_ultimo = inducir_lexico_afecto(
        [documentos[indice] for indice in pliegues[-1][0]], y_verdadera[pliegues[-1][0]], p2
    )
    oof_nb, diag_nb = evaluar_seccion_nb_svm(desarrollo, y_verdadera, pliegues)
    oof_ort, diag_ort = evaluar_seccion_ortografia(
        desarrollo, documentos, y_verdadera, pliegues, p2
    )
    robustez = probar_robustez(desarrollo, p2, lexico_ultimo, constantes)

    clave_base = "BASE@C=2.0"
    oof = {clave_base: oof_base, **oof_emocion, **oof_nb, **oof_ort}
    seccion_por_clave = {
        **{clave: "E2-09" for clave in oof_emocion},
        **{clave: "E2-15" for clave in oof_nb},
        **{clave: "E2-10" for clave in oof_ort},
        clave_base: "BASE",
    }
    resumenes = {
        clave: resumir_clave(
            clave,
            seccion_por_clave[clave],
            proba,
            y_verdadera,
            pliegue_de_fila,
            oof_base,
            f1_pliegues_v03_umbrales,
            p2,
        )
        for clave, proba in oof.items()
    }
    mejores = seleccionar_mejores(
        resumenes,
        list(VARIANTES_EMO)
        + list(VARIANTES_EMO_DIAGNOSTICO)
        + list(VARIANTES_NB)
        + list(VARIANTES_ORT),
    )
    decisiones = {
        seccion: decidir_seccion(resumenes, variantes, seccion)
        for seccion, variantes in (
            ("E2-09", VARIANTES_EMO),
            ("E2-15", list(VARIANTES_NB)),
            ("E2-10", list(VARIANTES_ORT)),
        )
    }

    canales = guardar_oof(
        desarrollo,
        grupos,
        y_verdadera,
        pliegue_de_fila,
        oof,
        [
            mejores["EMO-EST"],
            mejores["EMO-LEX"],
            mejores["EMO-ALL"],
            mejores["TXT+EMO"],
            clave_nb("NB-WORD-LR"),
            clave_nb("NB-CHAR-LR"),
            clave_nb("NB-UNION-LR"),
            clave_nb("NB-UNION-SVC"),
            mejores["ORT-A"],
            mejores["ORT-ABCD"],
            mejores["ORT-AE"],
        ],
    )
    generar_figura(resumenes, decisiones, f1_pliegues_v03_umbrales)
    escribir_csv(resumenes, mejores)
    duracion = time.perf_counter() - inicio
    escribir_json(
        resumenes,
        mejores,
        decisiones,
        {
            **verificaciones,
            "diagnostico_eda": diagnosticos_eda,
            "modo_anotacion": modo_anotacion,
            "origen_constantes": origen_constantes,
        },
        {"E2-09": diag_emocion, "E2-15": diag_nb, "E2-10": diag_ort},
        robustez,
        canales,
        duracion,
    )
    registrar_experimentos(resumenes, mejores, decisiones, duracion)
    escribir_contexto(construir_contexto(resumenes, mejores, decisiones, verificaciones, duracion))
    actualizar_todo(resumenes, mejores, decisiones)
    log(f"tiempo total: {duracion:.1f} s")
    for seccion, decision in decisiones.items():
        log(
            f"{seccion}: mejor {decision['mejor_clave']} corrected_cv {decision['corrected_cv']:.4f} "
            f"({'ADOPTADO' if decision['adoptado'] else 'cerrado sin mejora'})"
        )
    log(f"artefactos: {REPORTE_CSV.name}, {REPORTE_JSON.name}, {FIGURA_PATH.name}, {OOF_PATH.name}")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - se reporta el fallo completo y se corta la ejecucion
        traceback.print_exc()
        sys.exit(1)
