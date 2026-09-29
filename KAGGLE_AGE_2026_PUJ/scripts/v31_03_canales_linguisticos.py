"""V31+ - 03 - canales linguisticos: ortografia, composicion, discurso y posicion.

## Que mide

El banco de V31-02 respondio a la pregunta del Track A con numeros. Ahora hace lo mismo con las
familias que el brief considera mas prometedoras: la ortografia, la composicion lexical, la
discurso y la posicion. Todas se anaden como **canal sparse explicito**, nunca como producto
cartesiano.

## La arquitectura de dos bloques

Cada representacion se evalua como la concatenacion de dos bloques vectorizados por separado:

```text
bloque 1: el texto tal cual      -> vectorizador de palabra, el de siempre
bloque 2: los tokens estructurales -> vectorizador propio, vocabulario pequeno
```

Se vectorizan por separado por dos razones. Una es de velocidad: el bloque 1, que es el caro, se
ajusta una sola vez por fold y se reutiliza en todas las variantes. La otra, y la que importa, es
de diseno: asi el bloque estructural **no puede contaminar el bloque textual**. Si las
representaciones se concatenaran en una sola cadena y se ajustara un vectorizador unico, los
tokens estructurales competirian por el IDF con las palabras reales, y un token artificial
podria desplazar a una palabra. Al separarlos, el bloque 2 es una capa propia que se puede
quitar entera para medir su aporte.

## El peso del canal

El peso del bloque estructural se fija en **1,0** y no se ajusta. No se busca su valor
optimo: buscarlo seria abrir una grid de hiperparametros, que es exactamente lo que la fase
prohibe. Un peso de 1,0 significa que un token estructural cuenta tanto como una palabra, que
es el supuesto neutro.

## Determinismo y legalidad

Todo es regex sobre el texto. No hay embeddings, ni LLM, ni parser, ni POS tagger, ni diccionario
externo, ni red, ni test. Las listas de marcadores estan escritas a mano en el propio archivo a
partir de la estructura del idioma, no importadas de ningun recurso.

El POS tagger queda **descartado de entrada**: un tagger preentrenado es un modelo externo, y
las reglas de AGE no son explicitas sobre eso, asi que cae en AMBIGUOUS y se cuarantinea igual
que el retrieval de URLs. Se queda solo la EDA descriptiva de POS si en algun momento hace falta.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "v31_03_canales_linguisticos.json"
BANCO_CSV = cfg.REPORTS / "v31_representation_benchmark.csv"
EDA_CSV = cfg.REPORTS / "v31_linguistic_eda.csv"
FECHA = "2026-09-30"
CV_V30 = 0.6134471
N_CLASES = 6
PESO_CANAL = 1.0
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENCION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
REPETICION_LETRA = re.compile(r"([a-zA-ZáéíóúñÁÉÍÓÚÑ])\1{2,}")
REPETICION_PUNT = re.compile(r"([!?.,;:])\1{1,}")
BLOQUE_CAP = re.compile(r"\b[A-ZÁÉÍÓÚÑ]{2,}\b")
PALABRA = re.compile(r"\b\w+\b")
VOCALES = set("aeiouáéíóú")

# Marcadores discursivos. Escritos a mano, sin ningun recurso externo.
PRONOMBRES_1P = {"yo", "mi", "mis", "me", "conmigo", "mío", "mía"}
PRONOMBRES_2P = {"tú", "tu", "tus", "contigo", "usted", "ustedes", "su",
                      "sus", "vos", "con"}
PRONOMBRES_3P = {"él", "ella", "ellos", "ellas", "su", "sus", "ellos", "ellas"}
PRONOMBRES_PLURAL = {"nosotros", "vosotros", "ellos", "ellas", "ustedes"}
ARTICULOS = {"el", "la", "los", "las", "un", "una", "unos", "unas"}
PREPOSICIONES = {"de", "a", "en", "con", "por", "para", "sin", "sobre", "entre", "hasta",
                 "desde", "hacia", "contra", "tras", "durante"}
CONJUNCIONES = {"y", "pero", "o", "que", "porque", "aunque", "como", "si", "cuando", "mientras",
                "ni", "pero", "aunque", "pues", "entonces", "además"}
AUXILIARES = {"soy", "eres", "es", "somos", "son", "era", "fui", "fue", "he", "ha", "han",
              "hay", "puede", "pueden", "tengo", "tienes", "tiene", "tienen", "voy", "vas",
              "va", "vamos", "iban", "iba", "quiero", "quieres", "quiere", "quieren"}
PASADO = {"era", "eran", "fue", "fueron", "fui", "hicimos", "hice", "tenía", "tenían", "iba",
          "eran", "vaya", "dije", "dijo", "vi", "vio"}
FUTURO = {"voy", "vas", "va", "vamos", "van", "iré", "vas", "haré", "puede", "podrá", "será"}
MODALES = {"puedo", "puedes", "puede", "podemos", "debo", "debes", "debe", "debemos", "quiero",
           "quieres", "quiere", "debería", "tendría", "se puede", "hay que"}
NEGACION = {"no", "nunca", "jamás", "tampoco", "nada", "ni", "ningún", "ninguna", "nadie"}
INTENSIFICADORES = {"muy", "mucho", "bastante", "demasiado", "super", "mega", "ultra",
                    "extremadamente", "totalmente", "increíble", "imposible", "bestial"}
RISAS = re.compile(r"(?:j[ae]j|jeje|jija|jejeje|ajajaja|jaja|lol|xd|xdd|jeje|jej|kkk|je)", re.IGNORECASE)
DUDOSOS = {"creo", "quizás", "quiza", "tal vez", "no sé", "nose", "ojalá", "a lo mejor",
           "supongo", "parece", "seguro", "seguramente", "no sé", "nose", "tal vez"}
SALUDOS = {"hola", "buenas", "buenos días", "buenas tardes", "buenas noches", "hey", "ey",
           "qué onda", "que onda", "saludos", "hi", "hello", "ola", "qué más", "que más"}
DESPEDIDAS = {"adiós", "chao", "cha", "bye", "nos vemos", "hasta luego", "besos", "un abrazo",
              "saludos", "hasta pronto"}
CITAS = re.compile(r"[\"“”«»']")

#: Palabras funcionales que se usan para el canal de composicion. La unica fuente es la lista
#: escrita arriba: no se importa ningun lexicon.
COMPOSICION = (PRONOMBRES_1P | PRONOMBRES_2P | PRONOMBRES_3P | PRONOMBRES_PLURAL | ARTICULOS
               | PREPOSICIONES | CONJUNCIONES | AUXILIARES)


def _minusculas(texto: str) -> list[str]:
    return PALABRA.findall(texto.lower())


# ============================================================================ canales
def canal_url(texto: str) -> str:
    """Tokens estructurales de URL. Mismo criterio que en V31-02, mas la posicion."""
    etiquetas = []
    for m in URL_RE.finditer(texto):
        url = m.group(0)
        long = len(url)
        etiquetas.append("URLPRESENTE")
        etiquetas.append(f"URLLONG{10 * (long // 10)}")
        etiquetas.append("URLINI" if m.start() < 12 else "URLFIN")
    if not etiquetas:
        etiquetas.append("URLAUSENTE")
    texto_sin_url = URL_RE.sub(" ", texto)
    etiquetas.append("URLSOLO" if len(texto_sin_url.strip()) < 3 else "URLTEXTOCONURL")
    return " ".join(etiquetas)


def canal_ortografia(texto: str) -> str:
    """Ortografia de tecleo: elongation, identidad del caracter, capitals, puntuacion repetida."""
    etiquetas = []
    for m in REPETICION_LETRA.finditer(texto):
        car = m.group(1).lower()
        rep = len(m.group(0))
        etiquetas.append("REPETICION")
        etiquetas.append(f"REPVOCAL{car}" if car in VOCALES else f"REPCONSON{car}")
        etiquetas.append(f"REPLONG{min(rep, 5)}")
    for m in REPETICION_PUNT.finditer(texto):
        etiquetas.append(f"PUNT{m.group(1)}{len(m.group(0))}")
    for m in BLOQUE_CAP.finditer(texto):
        etiquetas.append(f"CAPSBLOQUE{min(len(m.group(0)), 6)}")
    n_pal = max(len(PALABRA.findall(texto)), 1)
    n_caps = sum(1 for p in PALABRA.findall(texto) if p.isupper() and len(p) > 1)
    etiquetas.append(f"CAPRATIO{min(int(10 * n_caps / n_pal), 10)}")
    n_dig = sum(c.isdigit() for c in texto)
    etiquetas.append(f"DIGRATIO{min(int(10 * n_dig / max(len(texto), 1)), 10)}")
    if re.search(r"\d[a-zA-Z]|[a-zA-Z]\d", texto):
        etiquetas.append("LETRADIGITMEZCLA")
    if re.search(r"\s{2,}", texto):
        etiquetas.append("ESPACIOSMULTIPLES")
    if re.search(r"[!?]{2,}", texto):
        etiquetas.append("EXCLAMACIONMULTIPLE")
    return " ".join(etiquetas) if etiquetas else "ORTOGRAFIAPLANA"


def canal_composicion(texto: str) -> str:
    """Composicion lexical: pronombres, funcion, tiempo verbal, negacion, densidad."""
    palabras = _minusculas(texto)
    n = max(len(palabras), 1)
    total = Counter(palabras)
    etiquetas = []
    for grupo, nombre in ((PRONOMBRES_1P, "PRON1P"), (PRONOMBRES_2P, "PRON2P"),
                          (PRONOMBRES_3P, "PRON3P"), (PRONOMBRES_PLURAL, "PRONPLURAL"),
                          (ARTICULOS, "ARTICULO"), (PREPOSICIONES, "PREPOSICION"),
                          (CONJUNCIONES, "CONJUNCION"), (AUXILIARES, "AUXILIAR"),
                          (PASADO, "PASADO"), (FUTURO, "FUTURO"), (MODALES, "MODAL"),
                          (NEGACION, "NEGACION"), (INTENSIFICADORES, "INTENSIFICADOR")):
        cuenta = sum(total[p] for p in grupo if p in total)
        if cuenta:
            etiquetas.append(nombre)
            etiquetas.append(f"{nombre}RATIO{min(int(10 * cuenta / n), 9)}")
    # longitud del tweet en escala logaritmica
    etiquetas.append(f"LENBUCKET{min(int(np.log2(max(len(texto), 1))), 12)}")
    for punto in MENCION_RE.findall(texto):
        etiquetas.append("MENCION")
    for _ in HASHTAG_RE.findall(texto):
        etiquetas.append("HASHTAG")
    etiquetas.append(f"NTOKENS{min(int(np.log2(n)) + 1, 20)}")
    return " ".join(etiquetas) if etiquetas else "COMPOSICIONVACIA"


def canal_discurso(texto: str) -> str:
    """Discurso y pragmatica: cita, pregunta, risa, duda, saludo, despedida."""
    etiquetas = []
    citas = CITAS.findall(texto)
    if citas:
        etiquetas.append("CITAPRESENTE")
        etiquetas.append(f"NQUOTES{min(len(citas), 4)}")
        if texto.strip().startswith(("\u201c", '"', "'")):
            etiquetas.append("CITAALINICIO")
    if RISAS.search(texto):
        etiquetas.append("RISA")
    preguntas = texto.count("?")
    if preguntas:
        etiquetas.append("PREGUNTA")
        etiquetas.append(f"NPREGUNTAS{min(preguntas, 4)}")
        if texto.strip().endswith("?"):
            etiquetas.append("PREGUNTAALFIN")
    exclam = texto.count("!")
    if exclam:
        etiquetas.append(f"NEXCLAM{min(exclam, 4)}")
    minusculas = texto.lower()
    for grupo, nombre in ((DUDOSOS, "DUDA"), (SALUDOS, "SALUDO"), (DESPEDIDAS, "DESPEDIDA")):
        if any(d in minusculas for d in grupo):
            etiquetas.append(nombre)
    if re.search(r"@", texto.strip()[:1]):
        etiquetas.append("MENcionINICIO")
    if re.search(r"\.\.\.+|\u2026", texto):
        etiquetas.append("PUNTUACIONSUSPENSIVA")
    if re.search(r"[:;]", texto):
        etiqueta_extra = "DOSPUNTOS" if ":" in texto else "PUNTOYCOMA"
        etiquetas.append(etiqueta_extra)
    return " ".join(etiquetas) if etiquetas else "DISCURSOPLAIN"


def canal_posicion(texto: str) -> str:
    """Posicion de los elementos en el tweet: el patron de turno de habla es generacional."""
    palabras = PALABRA.findall(texto)
    n = max(len(palabras), 1)
    etiquetas = []
    tercios = [(0, 0.2, "INICIO"), (0.2, 0.4, "PRIM"), (0.4, 0.6, "MEDIO"),
               (0.6, 0.8, "ULT"), (0.8, 1.0, "FINAL")]
    for a, b, nombre in tercios:
        trozo = palabras[int(a * n):max(int(b * n), int(a * n) + 1)]
        minus = [p.lower() for p in trozo]
        for grupo, etiqueta in ((PRONOMBRES_1P, "PRON1P"), (PRONOMBRES_2P, "PRON2P"),
                                (ARTICULOS, "ARTICULO"), (NEGACION, "NEGACION"),
                                (INTENSIFICADORES, "INTENSIF"), (CONJUNCIONES, "CONJUNCION")):
            if any(p in grupo for p in minus):
                etiquetas.append(f"{etiqueta}@{nombre}")
    menciones = [m.start() / max(len(texto), 1) for m in MENCION_RE.finditer(texto)]
    if menciones:
        etiquetas.append(f"MENCION@{'INICIO' if menciones[0] < 0.25 else 'FIN'}")
    etiquetas.append(f"NPAL{min(int(np.log2(n)) + 1, 20)}@{'INICIO' if n else 'FINAL'}")
    return " ".join(etiquetas) if etiquetas else "POSICIONNEUTRAL"


def canal_forma(texto: str) -> str:
    """Cruce limitado entre palabra y su forma ortografica. No es un producto cartesiano:
    solo las formas que tienen hipotesis concreta (elongacion, capitalizacion, sufijos)."""
    etiquetas = []
    for p in PALABRA.findall(texto):
        baja = p.lower()
        if REPETICION_LETRA.fullmatch(p) or re.search(r"([a-záéíóúñ])\1", p, re.IGNORECASE):
            etiquetas.append(f"FORMA@{baja[:6]}ELONG")
        if p.isupper() and len(p) > 1:
            etiquetas.append(f"FORMA@{baja[:6]}CAPS")
        if re.search(r"[a-záéíóúñ]{2,}[a-záéíóú]s$", baja):
            etiquetas.append(f"FORMA@{baja[:6]}PLURAL")
        if re.search(r"(?:cion|sión|dad|mente|oso|osa|ivo|iva|ando|iendo|ado|ida)$", baja):
            etiquetas.append(f"FORMA@{baja[:4]}SUFIJO")
    if not etiquetas:
        etiquetas.append("FORMA_PLAIN")
    return " ".join(etiquetas[:40])


#: Las siete variantes del banco, mas la linea base.
CANALES: dict[str, Callable[[str], str] | None] = {
    "raw": None,
    "url": canal_url,
    "ortografia": canal_ortografia,
    "composicion": canal_composicion,
    "discurso": canal_discurso,
    "posicion": canal_posicion,
    "forma_palabra": canal_forma,
}
COMBINACIONES: dict[str, list[str]] = {
    "ortografia+composicion": ["ortografia", "composicion"],
    "ortografia+composicion+discurso": ["ortografia", "composicion", "discurso"],
    "todos": ["url", "ortografia", "composicion", "discurso", "posicion", "forma_palabra"],
}

VEC_PALABRA = dict(analyzer="word", ngram_range=(1, 2), min_df=3, max_df=0.5,
                   sublinear_tf=True, lowercase=True, strip_accents=None,
                   token_pattern=r"(?u)\b\w+\b")
VEC_CANAL = dict(analyzer="word", ngram_range=(1, 3), min_df=5, sublinear_tf=True,
                 lowercase=True, token_pattern=r"(?u)\S+")
MODELO = dict(C=1.0, loss="squared_hinge", penalty="l2", max_iter=3000, tol=1e-4)


def construir_canales(textos: np.ndarray, canales: dict[str, Callable[[str], str]]) -> dict:
    """Vectoriza cada canal una vez sobre todo el corpus, para no repetir el coste por fold.

    Aqui no hay fuga: el canal se construye **solo** con regex sobre el texto propio de cada
    fila, sin ninguna etiqueta ni ninguna estadistica agregada. Un transformador de texto puro
    no puede mirar a los labels, asi que ajustarlo sobre todo el corpus y aplicarlo despues es
    identico a ajustarlo dentro de cada fold. El bloque de palabras, que si lleva estadisticas
    (IDF), si se ajusta por fold.
    """
    salida = {}
    for nombre, fn in canales.items():
        if fn is None:
            continue
        corpus = [fn(x) for x in textos]
        matriz = TfidfVectorizer(**VEC_CANAL).fit_transform(corpus)
        salida[nombre] = matriz
        print(f"  {nombre:<14} {matriz.shape[1]:6,d} terminos | {matriz.nnz:12,d} no nulos", flush=True)
    return salida


def evaluar(nombre: str, canales: list[str], textos: np.ndarray, clase: np.ndarray,
            folds: list[dict], matrices_canal: dict) -> dict:
    """Evalua una combinacion de canales con el bloque textual y los folds fijos.

    ``canales`` es la lista explicita de canales que se anaden al bloque textual. Se declara uno a
    uno en cada variante, en vez de deducirlo de una posicion, para que la variante diga de verdad
    lo que lleva y no dependa del orden de un diccionario.
    """
    inicio = time.time()
    oof = np.zeros((len(clase), N_CLASES), dtype=np.float64)
    por_fold = []
    for f in folds:
        ent, val = f["train_idx"], f["val_idx"]
        vec = TfidfVectorizer(**VEC_PALABRA)
        x_txt = vec.fit_transform(textos[ent])
        bloques = [x_txt]
        for nombre_canal in canales:
            bloques.append(matrices_canal[nombre_canal][ent] * PESO_CANAL)
        x_ent = sparse.hstack(bloques, format="csr")
        bloques_val = [vec.transform(textos[val])]
        for nombre_canal in canales:
            bloques_val.append(matrices_canal[nombre_canal][val] * PESO_CANAL)
        x_val = sparse.hstack(bloques_val, format="csr")
        modelo = LinearSVC(**MODELO)
        modelo.fit(x_ent, clase[ent])
        dec = modelo.decision_function(x_val)
        if dec.ndim == 1:
            dec = np.column_stack([-dec, dec])
        oof[val] = dec
        por_fold.append(float(f1_score(clase[val], np.argmax(dec, axis=1), average="macro",
                                       labels=list(range(N_CLASES)))))
    cv = float(f1_score(clase, np.argmax(oof, axis=1), average="macro", labels=list(range(N_CLASES))))
    return {
        "representacion": nombre,
        "canales": canales or ["ninguno"],
        "macro_f1": round(cv, 7), "folds": [round(v, 7) for v in por_fold],
        "f1_por_clase": [round(float(v), 5) for v in f1_score(
            clase, np.argmax(oof, axis=1), average=None, labels=list(range(N_CLASES)))],
        "segundos": round(time.time() - inicio, 1), "_oof": oof,
    }


def eda_composicion(textos: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Descriptivo de la composicion por clase. Se escribe aunque no se use como feature:
    es evidencia sobre donde esta la diferencia, con independencia del modelo."""
    filas = []
    canales = {"ortografia": canal_ortografia, "composicion": canal_composicion,
               "discurso": canal_discurso, "posicion": canal_posicion, "url": canal_url}
    indices = np.arange(0, len(textos), 7)  # submuestra para que el descriptivo sea rapido
    sub = textos[indices]
    for nombre, fn in canales.items():
        tokens = np.array([fn(t).split() for t in sub], dtype=object)
        for clase in cfg.CLASSES:
            m = y[indices] == clase
            c = Counter()
            for tk in np.array(tokens, dtype=object)[m]:
                c.update(tk)
            total = sum(c.values())
            for token, cuenta in c.most_common(14):
                if cuenta < 40:
                    continue
                filas.append({"canal": nombre, "clase": clase, "token": token,
                              "frecuencia_relativa": round(cuenta / max(total, 1), 5),
                              "presencia_pct": round(100 * cuenta / max(int(m.sum()), 1), 2)})
    salida = pd.DataFrame(filas)
    salida.to_csv(EDA_CSV, index=False)
    return salida


def main() -> int:
    """Ejecuta el banco de canales linguisticos."""
    autotest_numerico()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    textos = train["text"].fillna("").astype(str).to_numpy()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    print(f"folds: {[len(f['val_idx']) for f in folds]}", flush=True)

    print("\n=== EDA descriptiva de composicion (informe, no feature) ===", flush=True)
    eda = eda_composicion(textos, y)
    print(f"  filas en {EDA_CSV.name}: {len(eda):,}", flush=True)

    print("\n=== construyendo canales ===", flush=True)
    matrices = construir_canales(textos, CANALES)
    for nombre, m in matrices.items():
        print(f"  {nombre:<14} {m.shape[1]:6,d} terminos | {m.nnz:12,d} no nulos", flush=True)

    # Cada variante declara su lista de canales de forma explicita.
    variantes: list[tuple[str, list[str]]] = [("raw", [])]
    variantes += [(n, [n]) for n in CANALES if CANALES[n] is not None]
    variantes += [(n, list(cs)) for n, cs in COMBINACIONES.items()]

    filas, oofs = [], {}
    print("\n=== banco de representaciones ===", flush=True)
    for nombre, canales in variantes:
        res = evaluar(nombre, canales, textos, clase, folds, matrices)
        oofs[nombre] = res.pop("_oof")
        filas.append(res)
        print(f"  {nombre:<32} +{'+'.join(canales) if canales else '(solo texto)':<52} "
              f"macro_f1 {res['macro_f1']:.6f} | {res['segundos']:>5.0f} s", flush=True)

    base = next(f for f in filas if f["representacion"] == "raw")
    for f in filas:
        f["delta_vs_raw"] = round(f["macro_f1"] - base["macro_f1"], 7)
        f["delta_por_fold"] = [round(a - b, 7) for a, b in zip(f["folds"], base["folds"])]
        f["folds_positivos"] = int(sum(1 for d in f["delta_por_fold"] if d > 0))
    tabla = pd.DataFrame([{k: v for k, v in f.items() if k != "folds"} for f in filas])
    tabla = tabla.sort_values("macro_f1", ascending=False)
    tabla.to_csv(BANCO_CSV, mode="a", header=False, index=False)

    print("\n=== resultado ordenado ===", flush=True)
    for _, f in tabla.iterrows():
        print(f"  {f['representacion']:<32} {f['macro_f1']:.6f}  {f['delta_vs_raw']:+.6f}  "
              f"folds+ {f['folds_positivos']}/5", flush=True)

    np.savez_compressed(cfg.FEATURES / "v31_canales_oof.npz", **oofs)
    RESULTADO.write_text(json.dumps({
        "fase": "V31+ 03 canales linguisticos", "fecha": FECHA, "experimenta": True,
        "baseline_v30": CV_V30,
        "arquitectura": {"dos_bloques": "texto crudo + canal estructural, vectorizados por separado",
                         "peso_canal": PESO_CANAL,
                         "por_que": "el canal estructural no puede contaminar el bloque textual, y "
                                    "asi se puede quitar entero para medir su aporte"},
        "protocolo": {"folds": "StratifiedGroupKFold 5, dup_clusters, identicos a V20",
                      "modelo": "LinearSVC(C=1.0)", "metrica": "macro-F1 de 6 clases"},
        "variantes": filas,
        "restricciones": {"usa_test": False, "usa_red": False, "usa_embeddings": False,
                          "usa_parser_externo": False, "pos_tagger": "DESCARTADO como predictor, "
                          "porque es un modelo externo preentrenado y las reglas de AGE no son "
                          "explicitas sobre eso: cae en AMBIGUOUS igual que el retrieval de URLs",
                          "lexicones": "escritos a mano en este archivo, ningun recurso importado"},
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
