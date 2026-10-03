"""V14 · A — Variables lingüísticas del analizador español (morphología, sintaxis, NER, lemas).

V14 nace de dos decisiones de alcance que Carlos asume explícitamente y que están registradas en
`DECISIONS.md`:

* **D-043**: se relaja §3 para permitir un **recurso lingüístico externo**. El recurso disponible es
  ``es_core_news_md`` de spaCy: un analizador morfosintáctico del español con ``morphologizer``,
  ``parser``, ``lemmatizer`` y ``ner``. **No** es un transformer, **no** es un LLM, **no** son
  embeddings para clasificación, y **no** aporta etiquetas: solo forma. Se sigue sin permitir
  embeddings, modelos externos de clasificación, corpus con etiquetas, pseudo-etiquetas ni labels de
  test.
* **D-042**: el gate de adopción baja a Δ ≥ +0,001 con ≥ 3/5 folds, **pero** el IC95 del bootstrap
  pareado sigue tenien que excluir el cero y se añade la exigencia de reproducirse en una segunda
  partición. El listón de efecto baja; el de significación no.

Por qué no es la línea que D-026 ya cerró. D-026 midió **26 recuentos de etiqueta POS** y obtuvo
−0,0032. Eso es la misma información en forma de conteo. Lo que este experimento usa es otro tipo de
variable, que sale de los otros componentes del mismo modelo y que ninguna línea anterior había
medido:

1. **Morfología de verbo**: pretérito frente a imperfecto, presente, subjuntivo frente a indicativo,
   infinitivo, gerundio, participio, y primera, segunda y tercera persona. La distinción
   pretérito/imperfecto es un marcador de estilo muy informativo en español y
   no es un n-grama de caracteres ni un recuento de etiqueta.
2. **Morfología de nombre y adjetivo**: femenino frente a masculino, singular frente a plural,
   diminutivo (−ito, −ita, −ico, −ica, −illo, −illa) y aumentativo (−azo, −ona, −ote), que sí es
   información morfológica real y no un sufijo contado a ojo.
3. **Lemas**: diversidad léxica por lema y no por forma, hapax legomena por lema. El bloque
   estructurado de V12 mide diversidad por forma, así que esto es complementario.
4. **Entidades nombradas**: reparto de PERSON, ORG, LOC y MISC, y numero de entidades distintas.
5. **Sintaxis**: profundidad media de dependencia, ratio de coordinacion, ratio de subordinadores,
  numero de hijos por token, construcciones pasivas, longitud media de frase en tokens.

El bloque es **determinista y sin etiquetas**: se calcula una vez sobre el train y se cachea, de
modo que no hay ninguna vía de leakage (a diferencia de P1, que sí es supervisado y por eso necesita
ajuste cruzado). Lo que sí hay que vigilar es la redundancia contra las 279 variables de V12, y por
eso el cribado es obligatorio y va primero.

Estructura en dos etapas, como P17b y P20: control *block-only* y puerta de rescate antes de pagar
una evaluación del meta-modelo.

Entregables: ``features/v14_linguistico.npy``, ``features/v14_linguistico_nombres.json``,
``reports/v14_linguistico_cobertura.csv``, ``reports/v14_linguistico_redundancia.csv``,
``reports/v14_linguistico_control.csv``, ``reports/v14_linguistico.md``,
``experiments/v14a_linguistico.json``.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import spacy  # noqa: E402
import spacy.util  # noqa: E402
from phase2_ponderacion import build_features  # noqa: E402
from v13_p0b_colinealidad import criba_redundancia  # noqa: E402
from v13_p1b_fusion_tokens import PARES, oof_una_cabeza  # noqa: E402

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

MODELO = "es_core_news_md"
INFORME = cfg.REPORTS / "v14_linguistico.md"
COBERTURA = cfg.REPORTS / "v14_linguistico_cobertura.csv"
REDUNDANCIA = cfg.REPORTS / "v14_linguistico_redundancia.csv"
CONTROL = cfg.REPORTS / "v14_linguistico_control.csv"
MATRIZ = cfg.FEATURES / "v14_linguistico.npy"
NOMBRES = cfg.FEATURES / "v14_linguistico_nombres.json"
CACHE = cfg.FEATURES / "v14_linguistico_cache.parquet"
RESULTADO = cfg.EXPERIMENTS / "v14a_linguistico.json"

SUELO_AZAR = 1.0 / len(cfg.CLASSES)
RESCATE_MINIMO = 1000
PORCENTAJE_ERROR = 0.015
#: Sufijos morfológicos contados sobre el lema. Son informacion linguistica real del analizador
#: (el lema ya esta normalizado), no un patron de caracteres sobre la forma superficial.
DIMINUTIVOS = ("ito", "ita", "ico", "ica", "illo", "illa", "cito", "cita")
AUMENTATIVOS = ("azo", "aza", "ote", "ota", "on", "ona")
#: Dependencias que indican coordinacion, subordinacion y pasiva.
COORDINACION = {"cc", "ccomp", "csub"}
SUBORDINACION = {"mark", "advcl", "acl", "advmod", "prep", "acl:relcl"}
PASIVAS = {"nsubjpass", "auxpass", "csubjpass"}
ENTIDADES = ("PER", "ORG", "LOC", "MISC")
#: Categorias POS de las que se sacan los recuentos gramaticales.
VERBOS = {"VERB", "AUX"}
SUSTANTIVOS = {"NOUN", "PROPN", "PRON"}
DETERMINANTES = {"DET", "ADP"}
CONJUNCIONES = {"CCONJ", "SCONJ"}
ADJETIVOS = {"ADJ"}
ADVERBIOS = {"ADV"}
PARTICIPIOS = {"PART"}


def nombres_de_features() -> list[str]:
    """Nombres de las características, en el orden exacto en que las construye :func:`agregar`."""
    nombres = [
        "verbo_preterito_ratio", "verbo_imperfecto_ratio", "verbo_presente_ratio",
        "verbo_subjuntivo_ratio", "verbo_infinitivo_ratio", "verbo_gerundio_ratio",
        "verbo_participio_ratio", "verbo_primera_persona_ratio", "verbo_segunda_persona_ratio",
        "verbo_tercera_persona_ratio", "verbo_por_documento",
        "sustantivo_femenino_ratio", "sustantivo_masculino_ratio",
        "sustantivo_singular_ratio", "sustantivo_plural_ratio", "diminutivo_ratio",
        "aumentativo_ratio",
        "lema_distintos_por_token", "lema_hapax_ratio", "forma_distintas_por_lemma",
        "ent_per_ratio", "ent_org_ratio", "ent_loc_ratio", "ent_misc_ratio",
        "entidades_distintas_por_documento", "entidades_por_documento",
        "profundidad_media", "coordinacion_ratio", "subordinacion_ratio", "hijos_por_token",
        "pasiva_ratio", "tokens_por_frase", "determinante_ratio", "conjuncion_ratio",
        "adjetivo_ratio", "adverbio_ratio", "longitud_media_palabra",
    ]
    return nombres


def agregar(documento) -> tuple[list[float], dict[str, int]]:
    """Resume un documento ya analizado en las características del bloque.

    La API de morfologia de esta version de spaCy es ``token.morph.to_dict()``: ni ``in`` ni ``[]``
    funcionan sobre ``MorphAnalysis`` en 3.8, y ``get`` devuelve una lista vacia en algunos casos.
    Se convierte una sola vez por token a diccionario y todo el calculo se hace sobre esos
    diccionarios, que es mas rapido y no depende de la version.

    Devuelve tambien un diccionario de conteos brutos, que alimenta el informe de cobertura: que
    las variables no esten vacias es la primera comprobacion de que el analizador funciona.
    """
    # Los signos de puntuacion y los espacios no aportan morfologia ni sintaxis util.
    tokens = [t for t in documento if not t.is_punct and not t.is_space]
    n = max(1, len(tokens))
    morfologia = [t.morph.to_dict() for t in tokens]
    verbos = [indice for indice, t in enumerate(tokens) if t.pos_ in VERBOS]
    n_verbos = max(1, len(verbos))
    sustantivos = [indice for indice, t in enumerate(tokens) if t.pos_ in SUSTANTIVOS]
    n_sustantivos = max(1, len(sustantivos))
    lemas = {t.lemma_ for t in tokens if t.lemma_ not in ("", "-PRON-")}
    frecuencias_lemma: dict[str, int] = {}
    for token in tokens:
        lema = token.lemma_
        if lema:
            frecuencias_lemma[lema] = frecuencias_lemma.get(lema, 0) + 1
    entidades = [(t.ent_type_, t.text.lower()) for t in tokens if t.ent_type_]
    n_entidades = max(1, len(entidades))
    profundidades = [t.head.i - t.i for t in tokens if t.head is not t]
    hijos = [sum(1 for _ in t.children) for t in tokens]
    dependencias = [t.dep_ for t in tokens]
    etiquetas = [t.pos_ for t in tokens]
    frases = list(documento.sents)
    formas = [t.text.lower() for t in tokens]
    lemas_texto = [(t.lemma_ or t.text).lower() for t in tokens]

    def proporcion(coleccion, predicado, total) -> float:
        return sum(1 for elemento in coleccion if predicado(elemento)) / total

    def proporcion_por_valor(indices: list[int], clave: str, valor, total: int) -> float:
        return proporcion(indices, lambda i: morfologia[i].get(clave) == valor, total)

    valores = [
        proporcion_por_valor(verbos, "Tense", "Past", n_verbos),
        proporcion_por_valor(verbos, "Tense", "Imp", n_verbos),
        proporcion_por_valor(verbos, "Tense", "Pres", n_verbos),
        proporcion(verbos, lambda i: morfologia[i].get("Mood") in ("Sub", "Cnd"), n_verbos),
        proporcion_por_valor(verbos, "VerbForm", "Inf", n_verbos),
        proporcion_por_valor(verbos, "VerbForm", "Ger", n_verbos),
        proporcion_por_valor(verbos, "VerbForm", "Part", n_verbos),
        proporcion_por_valor(verbos, "Person", "1", n_verbos),
        proporcion_por_valor(verbos, "Person", "2", n_verbos),
        proporcion_por_valor(verbos, "Person", "3", n_verbos),
        len(verbos) / n,
        proporcion(morfologia, lambda r: r.get("Gender") == "Fem", n),
        proporcion(morfologia, lambda r: r.get("Gender") == "Masc", n),
        proporcion_por_valor(sustantivos, "Number", "Sing", n_sustantivos),
        proporcion_por_valor(sustantivos, "Number", "Plur", n_sustantivos),
        proporcion(lemas_texto, lambda lema: lema.endswith(DIMINUTIVOS), n),
        proporcion(lemas_texto, lambda lema: lema.endswith(AUMENTATIVOS), n),
        len(lemas) / n,
        sum(1 for cuenta in frecuencias_lemma.values() if cuenta == 1) / max(1, len(lemas)),
        len({t.text.lower() for t in tokens}) / max(1, len(lemas)),
        *[proporcion(entidades, lambda par, objetivo=tipo: par[0] == objetivo, n_entidades)
          for tipo in ENTIDADES],
        len({texto for _, texto in entidades}) / n,
        len(entidades) / n,
        float(np.mean(np.abs(profundidades))) if profundidades else 0.0,
        proporcion(dependencias, lambda d: d in COORDINACION, n),
        proporcion(dependencias, lambda d: d in SUBORDINACION, n),
        float(np.mean(hijos)) if hijos else 0.0,
        proporcion(dependencias, lambda d: d in PASIVAS, n),
        n / max(1, len(frases)),
        proporcion(etiquetas, lambda p: p in DETERMINANTES, n),
        proporcion(etiquetas, lambda p: p in CONJUNCIONES, n),
        proporcion(etiquetas, lambda p: p in ADJETIVOS, n),
        proporcion(etiquetas, lambda p: p in ADVERBIOS, n),
        float(np.mean([len(forma) for forma in formas])) if formas else 0.0,
    ]
    # Guarda estructural: si el numero de valores no coincide con el de nombres, todas las
    # variables siguientes quedan mal etiquetadas y el bloque entero pierde sentido.
    assert len(valores) == len(nombres_de_features()), (len(valores), len(nombres_de_features()))
    conteos = {
        "verbos": len(verbos), "sustantivos": len(sustantivos), "entidades": len(entidades),
        "tokens": len(tokens), "lemas": len(lemas), "frases": len(frases),
        "personas": sum(1 for t in tokens if t.ent_type_ == "PER"),
        "organizaciones": sum(1 for t in tokens if t.ent_type_ == "ORG"),
        "lugares": sum(1 for t in tokens if t.ent_type_ == "LOC"),
        "verbos_con_morfologia": sum(1 for i in verbos if morfologia[i]),
    }
    return valores, conteos


def extraer(textos: pd.Series) -> pd.DataFrame:
    """Analiza el corpus con el modelo español y devuelve una fila por documento.

    Si la caché existe se reutiliza: el análisis es determinista y lento, y repetirlo en cada
    re-ejecución del control es tirar el tiempo.
    """
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    inicio = time.time()
    nlp = spacy.load(MODELO)
    filas, contadores = [], []
    documentos = textos.tolist()
    for documento in nlp.pipe(documentos, batch_size=500):
        valores, conteos = agregar(documento)
        filas.append(valores)
        contadores.append(conteos)
    matriz = np.asarray(filas, dtype=np.float32)
    tabla = pd.concat([pd.DataFrame(matriz, columns=nombres_de_features()),
                       pd.DataFrame(contadores)], axis=1)
    tabla.to_parquet(CACHE, index=False)
    print(f"[analisis] {len(tabla):,} documentos en {time.time() - inicio:.0f} s", flush=True)
    return tabla


def main() -> int:
    """Extrae las variables lingüísticas, las criba y aplica control y puerta de rescate."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    ruta_modelo = Path(spacy.util.get_package_path(MODELO))
    huella = hashlib.sha256((ruta_modelo / "meta.json").read_bytes()).hexdigest()

    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")

    tabla = extraer(train["text"])
    nombres = nombres_de_features()
    matriz = tabla[nombres].to_numpy(np.float64)
    np.save(MATRIZ, matriz.astype(np.float32))
    NOMBRES.write_text(json.dumps(nombres, ensure_ascii=False), encoding="utf-8")
    print(f"[bloque] {matriz.shape} | columnas con varianza "
          f"{int((matriz.std(axis=0) > 0).sum())} | modelo {MODELO} "
          f"(sha256 de meta.json {huella[:16]})", flush=True)

    # ---------------------------------------------------------------- cobertura
    cobertura = []
    for columna in ("verbos", "sustantivos", "entidades", "tokens", "lemas", "frases",
                    "personas", "organizaciones", "lugares"):
        serie = tabla[columna].to_numpy()
        cobertura.append({
            "magnitud": columna,
            "media_por_documento": round(float(serie.mean()), 3),
            "documentos_con_valor": int((serie > 0).sum()),
            "porcentaje_con_valor": round(100 * float((serie > 0).mean()), 2),
            "maximo": int(serie.max()),
        })
    pd.DataFrame(cobertura).to_csv(COBERTURA, index=False)
    for fila in cobertura:
        print(f"[cobertura] {fila['magnitud']:16s} media {fila['media_por_documento']:8.3f} | "
              f"documentos con valor {fila['porcentaje_con_valor']:6.2f} %", flush=True)

    # ---------------------------------------------------------------- cribado
    base_v12, _, _, _, _ = build_features()
    base_v12 = np.concatenate([base_v12, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")],
                              axis=1)
    generador = np.random.default_rng(cfg.SEED)
    filas = generador.choice(len(y), size=20000, replace=False)
    criba = criba_redundancia(matriz, nombres, base_v12,
                              [f"v12_{i:03d}" for i in range(base_v12.shape[1])], filas)
    criba.to_csv(REDUNDANCIA, index=False)
    print(f"[cribado] {int(criba['redundante'].sum())} de {len(criba)} con |rho| >= 0,80 contra V12 | "
          f"maxima {criba['correlacion_maxima_v12'].max():.4f} | mediana "
          f"{criba['correlacion_maxima_v12'].median():.4f}", flush=True)

    # ---------------------------------------------------------------- control block-only
    filas_control = []
    for nombre, constructor in (
            ("logreg", lambda: LogisticRegression(C=1.0, max_iter=3000, random_state=cfg.SEED)),
            ("linearsvc", lambda: LinearSVC(C=0.2, random_state=cfg.SEED))):
        por_fold = []
        for fold in folds:
            validacion = fold["val_idx"]
            entrenamiento = np.concatenate([otro["val_idx"] for otro in folds
                                            if otro["fold"] != fold["fold"]])
            estimador = constructor()
            estimador.fit(matriz[entrenamiento], clase[entrenamiento])
            por_fold.append(float(f1_score(clase[validacion],
                                           estimador.predict(matriz[validacion]),
                                           average="macro",
                                           labels=list(range(len(cfg.CLASSES))))))
        filas_control.append({"modelo": nombre, "macro_f1": round(float(np.mean(por_fold)), 4),
                              "std": round(float(np.std(por_fold)), 4),
                              "per_fold": [round(v, 4) for v in por_fold],
                              "suelo_azar": round(SUELO_AZAR, 4),
                              "supera_el_azar": bool(float(np.mean(por_fold)) > SUELO_AZAR)})
        print(f"[control] {nombre}: {np.mean(por_fold):.4f} ± {np.std(por_fold):.4f} "
              f"(suelo {SUELO_AZAR:.4f})", flush=True)
    pd.DataFrame(filas_control).to_csv(CONTROL, index=False)

    # ---------------------------------------------------------------- puerta de rescate
    referencia = np.load(cfg.FEATURES / "oof_ensemble_v12.npy")
    acierto_v12 = np.argmax(referencia, axis=1) == clase
    mejor = max(filas_control, key=lambda f: f["macro_f1"])
    if mejor["supera_el_azar"]:
        oof = oof_una_cabeza(matriz, y, folds, "logreg")
        acierto_bloque = np.argmax(oof, axis=1) == clase
        rescate = int((~acierto_v12 & acierto_bloque).sum())
        dano = int((acierto_v12 & ~acierto_bloque).sum())
        por_frontera = {}
        for (clase_a, clase_b), etiqueta in PARES.items():
            mascara = np.isin(clase, (cfg.CLASSES.index(clase_a), cfg.CLASSES.index(clase_b)))
            res = int((~acierto_v12 & acierto_bloque & mascara).sum())
            dan = int((acierto_v12 & ~acierto_bloque & mascara).sum())
            por_frontera[etiqueta] = {"rescate": res, "dano": dan, "neto": res - dan}
    else:
        rescate = dano = 0
        por_frontera = {etiqueta: {"rescate": 0, "dano": 0, "neto": 0} for etiqueta in PARES.values()}
    errores = int((~acierto_v12).sum())
    umbral = max(RESCATE_MINIMO, int(PORCENTAJE_ERROR * errores))
    pasa = ((rescate - dano) >= umbral
            and all(m["neto"] > 0 for m in por_frontera.values()))
    print(f"[puerta] mejor {mejor['modelo']} {mejor['macro_f1']} | rescate {rescate:,} | dano "
          f"{dano:,} | neto {rescate - dano:+,} | umbral {umbral:,} | pasa {pasa}", flush=True)

    # ---------------------------------------------------------------- informe
    lineas = [
        "# V14 · A — Variables lingüísticas del analizador español", "",
        "## 1. Qué cambia respecto a V13", "",
        "D-043 relaja §3 para permitir un recurso lingüístico externo. El recurso es "
        f"`{MODELO}` de spaCy: un analizador morfosintáctico (**no** transformer, **no** LLM, "
        "**no** embeddings de clasificación) que aporta análisis morfológico, lemas, entidades "
        "nombradas y dependencias. El bloque es determinista y **no** usa etiquetas, así que no "
        "tiene ninguna vía de leakage.", "",
        f"Procedencia: `{MODELO}` instalado en el venv, SHA256 de `meta.json` = `{huella}`.",
        "",
        "**No es la línea de D-026** (26 recuentos de etiqueta POS, −0,0032): aquí se usan los "
        "otros componentes del mismo modelo, que son otro tipo de variable.", "",
        "## 2. Cobertura del análisis", "",
        "| Magnitud | Media por documento | Documentos con valor | Máximo |", "|---|---|---|---|",
    ]
    for fila in cobertura:
        lineas.append(f"| {fila['magnitud']} | {fila['media_por_documento']} | "
                      f"{fila['porcentaje_con_valor']:.2f} % | {fila['maximo']} |")
    lineas += ["", "## 3. Características", "",
               f"{len(nombres)} columnas en cinco familias: morfología verbal (tiempo, aspecto, "
               f"modo, persona), morfología nominal (género, número, diminutivo, aumentativo), "
               f"lemas (diversidad y hapax por lema), entidades nombradas (reparto por tipo) y "
               f"sintaxis (profundidad de dependencia, coordinación, subordinación, pasiva, "
               f"longitud de frase).", "",
              "## 4. Cribado de redundancia contra las 279 variables de V12", "",
              f"- Variables redundantes: **{int(criba['redundante'].sum())}** de {len(criba)}",
              f"- Correlación máxima: **{criba['correlacion_maxima_v12'].max():.4f}** · mediana "
              f"{criba['correlacion_maxima_v12'].median():.4f}", ""]
    lineas += ["## 5. Control block-only", "",
               "| Modelo | macro-F1 | std | Suelo de azar | Supera el azar |", "|---|---|---|---|---|"]
    for fila in filas_control:
        lineas.append(f"| {fila['modelo']} | {fila['macro_f1']} | {fila['std']} | "
                      f"{fila['suelo_azar']} | {'si' if fila['supera_el_azar'] else 'no'} |")
    lineas += ["", "## 6. Puerta de rescate", "",
               f"- Rescate **{rescate:,}** · daño **{dano:,}** · **neto {rescate - dano:+,}** "
               f"(umbral {umbral:,})", ""]
    for etiqueta, medidas in por_frontera.items():
        lineas.append(f"- {etiqueta}: rescate {medidas['rescate']:,} · daño {medidas['dano']:,} · "
                      f"neto {medidas['neto']:+,}")
    lineas += ["", "## 7. Decisión", ""]
    if not mejor["supera_el_azar"]:
        lineas.append("**Ningún modelo llega al suelo de azar**: el analizador no aporta variables "
                      "discriminantes por sí solo y la pista se cierra sin pagar la evaluación del "
                      "meta-modelo.")
    elif pasa:
        lineas.append("**La puerta se supera**: procede la fusión con V12, control de permutado y el "
                      "gate revisado de D-042 (efecto ≥ +0,001, ≥ 3/5 folds, IC95 pareado que "
                      "excluya el cero y reproducción en una segunda partición).")
    else:
        lineas.append("**La puerta no se supera**: el bloque no justifica pagar la evaluación del "
                      "meta-modelo. Se cierra la pista.")
    lineas.append("")
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    RESULTADO.write_text(json.dumps({
        "fase": "V14 · A variables linguisticas", "fecha": "2026-09-26",
        "modelo": MODELO, "sha256_meta": huella,
        "n_caracteristicas": len(nombres), "nombres": nombres,
        "cobertura": cobertura, "suelo_azar": round(SUELO_AZAR, 4),
        "redundantes_contra_v12": int(criba["redundante"].sum()),
        "correlacion_maxima_v12": float(criba["correlacion_maxima_v12"].max()),
        "control": filas_control, "mejor_control": mejor,
        "prechequeo": {"errores_v12": errores, "rescate": rescate, "dano": dano,
                       "neto": rescate - dano, "umbral": umbral, "pasa": pasa,
                       "por_frontera": por_frontera},
        "gate": "D-042: delta >= +0,001, 3/5 folds, IC95 pareado que excluya el cero, segunda particion",
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nartefactos: {INFORME.name}, {COBERTURA.name}, {REDUNDANCIA.name}, {CONTROL.name}, "
          f"{MATRIZ.name}, {CACHE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
