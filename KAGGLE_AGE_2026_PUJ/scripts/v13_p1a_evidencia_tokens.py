"""V13 · P1a — Evidencia de edad por token, calculada dentro del fold (construccion, criba y control).

Primera pista de V13 y la primera que el plan coloca por delante de las variables deterministas por
el hallazgo de P0b: las familias de textometria de la matriz de V12 estan saturadas (VIF mediano de
12,6 a 42,8) mientras que las de evidencia supervisada (3,1) y NB-SVM (0,77) son las unicas con
margen. Anadir mas textometria tiene poco recorrido; anadir **evidencia supervisada calculada
dentro del fold**, que es un tipo de variable distinto, es lo que puede aportar.

Que hay ya y que no. Las 72 variables supervisadas que ya tiene V12 son ``P(clase | entidad)`` para
menciones, hashtags y fuentes (``src.advanced_features.supervised_entity_features``). **No existe**
``P(clase | token)`` para tokens simples, que es lo que se construye aqui. El vocabulario no se
declara: se **deriva del propio fold** por frecuencia de documento, de modo que no es un lexico
externo ni cerrado (regla R-1) sino una estadistica derivada del train.

Protocolo antileakage (§38). Nada se calcula antes de crear los folds:

1. El corpus se tokeniza una sola vez con ``src.lexicons.tokeniza``, el tokenizador del vectorizador
   de V12 (no se anade ningun vectorizador nuevo). El mapa token -> identificador es una
   codificacion: no transporta frecuencias ni etiquetas, asi que construirlo sobre todo el train no
   filtra informacion.
2. Para cada fold exterior, el train exterior se reparte a su vez en 5 folds interiores (agrupados
   por duplicados, igual que los exteriores) y se hace **ajuste cruzado**: las caracteristicas de
   cada fila del train exterior salen de un ``P(clase | token)`` calculado sobre las otras cuatro
   partes. Ninguna fila del train exterior ve su propia etiqueta.
3. Las filas de validacion exteriores reciben el perfil calculado sobre **todo** el train exterior,
   que es el unico conjunto legitimo.
4. Vocabulario y frecuencias de documento se calculan siempre sobre el train interior.
5. El test no se toca en ninguna parte.

Cada banda de frecuencia aporta 18 caracteristicas: el perfil de clase medio del documento (6), el
perfil del token mas informativo (6), su margen, su entropia, la cobertura de tokens conocidos, el
logaritmo de cuantos son, la ventaja de su clase dominante sobre la previa y su confianza.

Entregables: ``features/v13_p1_tokens_fold{0..4}.npy``, ``features/v13_p1_tokens_nombres.json``,
``reports/v13_p1_evidencia_tokens.md``, ``reports/v13_p1_redundancia.csv``,
``reports/v13_p1_control.csv``, ``experiments/v13_p1a_evidencia_tokens.json``.
"""

from __future__ import annotations

import json
import math
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

from phase2_ponderacion import build_features  # noqa: E402
from v13_p0b_colinealidad import criba_redundancia  # noqa: E402

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src import lexicons as lx  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v13_p1_evidencia_tokens.md"
REDUNDANCIA = cfg.REPORTS / "v13_p1_redundancia.csv"
CONTROL = cfg.REPORTS / "v13_p1_control.csv"
RESULTADO = cfg.EXPERIMENTS / "v13_p1a_evidencia_tokens.json"
NOMBRES = cfg.FEATURES / "v13_p1_tokens_nombres.json"

#: Frecuencia de documento minima para entrar en el vocabulario de un fold.
DF_MINIMO = 20
#: Bandas de frecuencia de documento. La banda baja es la informativa: los tokens frecuentes ya
#: estan representados por la textometria y por el TF-IDF de V12.
BANDAS = {"especifica": (DF_MINIMO, 100), "frecuente": (100, 1 << 30)}
#: Suavizado empirico-bayesiano del perfil de clase, el mismo que usa el bloque de entidades.
SUAVIZADO = 20.0
#: Suelo de azar del control block-only: macro-F1 de la clase mayoritaria es 1/6.
SUELO_AZAR = 1.0 / len(cfg.CLASSES)
CARACTERISTICAS_POR_BANDA = 2 * len(cfg.CLASSES) + 6


def tokeniza_todo(textos: pd.Series) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Tokeniza el corpus una vez y lo devuelve en forma dispersa.

    Parametros
    ----------
    textos:
        Serie de textos del train.

    Devuelve
    -------
    tuple
        ``(indices, punteros, vocabulario)``: los identificadores de token de cada ocurrencia en
        orden de documento, los punteros CSR y la lista de tokens indexada por identificador.
    """
    identificadores: list[int] = []
    filas: list[int] = []
    vocabulario: dict[str, int] = {}
    for documento, texto in enumerate(textos.to_numpy()):
        for token in lx.tokeniza(str(texto)):
            indice = vocabulario.get(token)
            if indice is None:
                indice = len(vocabulario)
                vocabulario[token] = indice
            filas.append(documento)
            identificadores.append(indice)
    columnas = np.asarray(identificadores, dtype=np.int64)
    documentacion = np.asarray(filas, dtype=np.int64)
    punteros = np.zeros(len(textos) + 1, dtype=np.int64)
    np.cumsum(np.bincount(documentacion, minlength=len(textos)), out=punteros[1:])
    return (columnas, punteros,
            [token for token, _ in sorted(vocabulario.items(), key=lambda par: par[1])])


def ocurrencias(indice: np.ndarray, punteros: np.ndarray,
                documentos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tokenos de los documentos indicados y su fila local, en una sola pasada.

    Devuelve
    -------
    tuple
        ``(tokens, fila)`` con un par por ocurrencia de token de esos documentos.
    """
    if not len(documentos):
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    longitudes = np.diff(punteros)[documentos]
    total = int(longitudes.sum())
    inicios = np.repeat(punteros[documentos], longitudes)
    dentro = np.arange(total) - np.repeat(np.cumsum(longitudes) - longitudes, longitudes)
    return indice[inicios + dentro], np.repeat(np.arange(len(documentos)), longitudes)


def frecuencia_de_documento(indice: np.ndarray, punteros: np.ndarray, documentos: np.ndarray,
                            n_tokens: int) -> np.ndarray:
    """Frecuencia de documento de cada token: en cuantos de esos documentos aparece al menos una vez.

    No basta con contar ocurrencias: un token repetido cinco veces en el mismo documento debe
    contar una sola vez, que es lo que define la banda de frecuencia.
    """
    tokens, fila = ocurrencias(indice, punteros, documentos)
    if not len(tokens):
        return np.zeros(n_tokens, dtype=np.int64)
    # Clave unica por par (documento local, token): la fila local va de 0 a len(documentos)-1.
    claves = np.unique(fila * n_tokens + tokens)
    return np.bincount(claves % n_tokens, minlength=n_tokens).astype(np.int64)


def perfil_de_clase(indice: np.ndarray, punteros: np.ndarray, documentos: np.ndarray,
                    clase_documento: np.ndarray, seleccion: np.ndarray,
                    n_clases: int) -> np.ndarray:
    """``P(clase | token)`` con suavizado empirico-bayesiano, alineado con ``seleccion``.

    Solo se estiman los tokens de la banda: las filas devueltas corresponden, en el mismo orden,
    a los identificadores de ``seleccion``.
    """
    previa = np.bincount(clase_documento[documentos], minlength=n_clases).astype(float)
    previa /= max(1.0, previa.sum())
    locales = np.full(int(indice.max()) + 1 if len(indice) else 1, -1, dtype=np.int64)
    locales[seleccion] = np.arange(len(seleccion))
    tokens, fila = ocurrencias(indice, punteros, documentos)
    identificados = locales[tokens] if len(tokens) else np.zeros(0, dtype=np.int64)
    conocidos = identificados >= 0
    codigos = (identificados[conocidos] * n_clases
               + clase_documento[documentos][fila[conocidos]])
    conteos = np.bincount(codigos, minlength=len(seleccion) * n_clases)
    conteos = conteos.reshape(len(seleccion), n_clases).astype(float)
    totales = conteos.sum(axis=1, keepdims=True)
    return (conteos + SUAVIZADO * previa[None, :]) / (totales + SUAVIZADO)


def resumir_documentos(indice: np.ndarray, punteros: np.ndarray, documentos: np.ndarray,
                       perfil: np.ndarray, seleccion: np.ndarray, n_clases: int) -> np.ndarray:
    """Resume cada documento en 18 caracteristicas a partir del perfil de sus tokens conocidos.

    La agregacion es vectorizada: se ordenan las ocurrencias por documento y por entropia, de modo
    que la primera de cada grupo es la del token mas informativo. No hay bucles sobre documentos.
    """
    n_documentos = len(documentos)
    salida = np.zeros((n_documentos, CARACTERISTICAS_POR_BANDA))
    if not n_documentos:
        return salida
    locales = np.full(int(indice.max()) + 1 if len(indice) else 1, -1, dtype=np.int64)
    locales[seleccion] = np.arange(len(seleccion))
    tokens, fila = ocurrencias(indice, punteros, documentos)
    identificados = locales[tokens]
    conocidos = identificados >= 0
    total_por_documento = np.bincount(fila, minlength=n_documentos)
    if not conocidos.any():
        return salida
    perfiles = perfil[identificados[conocidos]].astype(float)
    filas = fila[conocidos]
    entropias = -(perfiles * np.log(np.clip(perfiles, 1e-12, None))).sum(axis=1) / math.log(n_clases)
    orden = np.lexsort((entropias, filas))
    filas, perfiles, entropias = filas[orden], perfiles[orden], entropias[orden]
    inicios = np.flatnonzero(np.r_[True, np.diff(filas) != 0])
    cantidades = np.diff(np.r_[inicios, len(filas)])
    grupos = filas[inicios]
    mejores = perfiles[inicios]
    cola = np.sort(mejores, axis=1)
    salida[grupos, :n_clases] = np.add.reduceat(perfiles, inicios, axis=0) / cantidades[:, None]
    salida[grupos, n_clases:2 * n_clases] = mejores
    salida[grupos, 2 * n_clases] = cola[:, -1] - cola[:, -2]
    salida[grupos, 2 * n_clases + 1] = entropias[inicios]
    salida[grupos, 2 * n_clases + 2] = cantidades / np.maximum(total_por_documento[grupos], 1)
    salida[grupos, 2 * n_clases + 3] = np.log1p(cantidades)
    salida[grupos, 2 * n_clases + 4] = mejores.max(axis=1)
    salida[grupos, 2 * n_clases + 5] = np.maximum(mejores.max(axis=1) - 1.0 / n_clases, 0.0)
    return salida


def nombres_de_features() -> list[str]:
    """Nombres de las caracteristicas, en el orden exacto en que las produce el resumen."""
    nombres: list[str] = []
    for banda in BANDAS:
        nombres += [f"{banda}_media_c{indice}" for indice in range(len(cfg.CLASSES))]
        nombres += [f"{banda}_mejor_c{indice}" for indice in range(len(cfg.CLASSES))]
        nombres += [f"{banda}_margen", f"{banda}_entropia", f"{banda}_cobertura",
                    f"{banda}_n_conocidos", f"{banda}_ventaja", f"{banda}_confianza"]
    return nombres


def bloque_por_fold(indice: np.ndarray, punteros: np.ndarray, n_documentos: int,
                    vocabulario: list[str], folds: list[dict], y: np.ndarray,
                    clusters: np.ndarray, clase: np.ndarray) -> tuple[dict[int, np.ndarray], list[dict], list[dict]]:
    """Construye, para cada fold exterior, la matriz de caracteristicas sin leakage.

    Devuelve
    -------
    tuple
        ``(matrices, resumen_por_fold, tokens_destacados)``: las matrices por fold, un resumen de
        cobertura y los tokens que más se apartan de la previa en el train completo (descriptivo,
        sin prueba de significancia todavia).
    """
    n_tokens = len(vocabulario)
    nombres = nombres_de_features()
    matrices: dict[int, np.ndarray] = {}
    resumen: list[dict] = []
    destacados: list[dict] = []
    for fold in folds:
        numero = fold["fold"]
        validacion = fold["val_idx"]
        train_exterior = np.concatenate([otro["val_idx"] for otro in folds
                                         if otro["fold"] != numero])
        interiores = cvmod.make_folds(y[train_exterior], groups=clusters[train_exterior],
                                      n_splits=5, n_repeats=1, scheme="group")
        matriz = np.zeros((n_documentos, len(nombres)), dtype=np.float32)

        for numero_banda, (banda, (bajo, alto)) in enumerate(BANDAS.items()):
            inicio = numero_banda * CARACTERISTICAS_POR_BANDA
            bloque = slice(inicio, inicio + CARACTERISTICAS_POR_BANDA)
            for interior in interiores:
                propio = train_exterior[interior["val_idx"]]
                resto = np.ones(len(train_exterior), dtype=bool)
                resto[interior["val_idx"]] = False
                otros = train_exterior[resto]
                frecuencia = frecuencia_de_documento(indice, punteros, otros, n_tokens)
                seleccion = np.flatnonzero((frecuencia >= bajo) & (frecuencia < alto))
                if not len(seleccion):
                    continue
                perfil = perfil_de_clase(indice, punteros, otros, clase, seleccion,
                                         len(cfg.CLASSES))
                matriz[propio, bloque] = resumir_documentos(indice, punteros, propio, perfil,
                                                            seleccion, len(cfg.CLASSES))
            frecuencia = frecuencia_de_documento(indice, punteros, train_exterior, n_tokens)
            seleccion = np.flatnonzero((frecuencia >= bajo) & (frecuencia < alto))
            if len(seleccion):
                perfil = perfil_de_clase(indice, punteros, train_exterior, clase, seleccion,
                                         len(cfg.CLASSES))
                matriz[validacion, bloque] = resumir_documentos(indice, punteros, validacion,
                                                               perfil, seleccion,
                                                               len(cfg.CLASSES))
                for indice_clase in range(len(cfg.CLASSES)):
                    orden = np.argsort(-perfil[:, indice_clase])[:10]
                    for token in orden:
                        destacados.append({
                            "fold": numero, "banda": banda, "clase": cfg.CLASSES[indice_clase],
                            "token": vocabulario[int(seleccion[token])],
                            "frecuencia": int(frecuencia[seleccion[token]]),
                            "p_clase": round(float(perfil[token, indice_clase]), 4),
                        })
        matrices[numero] = matriz
        con_varianza = int((matriz.std(axis=0) > 0).sum())
        resumen.append({
            "fold": numero,
            "filas_train": int(len(train_exterior)),
            "filas_validacion": int(len(validacion)),
            "columnas_con_varianza": con_varianza,
            "tokens_vocabulario_banda_especifica": int(
                ((frecuencia_de_documento(indice, punteros, train_exterior, n_tokens) >= DF_MINIMO)
                 & (frecuencia_de_documento(indice, punteros, train_exterior, n_tokens) < 100)).sum()),
            "tokens_vocabulario_banda_frecuente": int(
                (frecuencia_de_documento(indice, punteros, train_exterior, n_tokens) >= 100).sum()),
        })
        print(f"[fold {numero}] {matriz.shape[0]:,} filas x {matriz.shape[1]} columnas | "
              f"con varianza {con_varianza} | vocabulario especifico "
              f"{resumen[-1]['tokens_vocabulario_banda_especifica']:,} y frecuente "
              f"{resumen[-1]['tokens_vocabulario_banda_frecuente']:,}", flush=True)
        np.save(cfg.FEATURES / f"v13_p1_tokens_fold{numero}.npy", matriz)
    return matrices, resumen, destacados


def main() -> int:
    """Construye el bloque de evidencia por token, lo criba contra V12 y lo mide en control."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")

    indice, punteros, vocabulario = tokeniza_todo(train["text"])
    print(f"[tokenizacion] {len(y):,} documentos | {len(indice):,} ocurrencias | "
          f"{len(vocabulario):,} tokens distintos", flush=True)
    matrices, resumen, destacados = bloque_por_fold(indice, punteros, len(y), vocabulario, folds,
                                                   y, clusters, clase)
    nombres = nombres_de_features()
    NOMBRES.write_text(json.dumps(nombres, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(destacados).to_csv(cfg.REPORTS / "v13_p1_tokens_destacados.csv", index=False)

    # ------------------------------------------------------------------ cribado contra V12
    base, _, _, _, _ = build_features()
    base = np.concatenate([base, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    generador = np.random.default_rng(cfg.SEED)
    filas = generador.choice(len(y), size=20000, replace=False)
    nombres_base = [f"v12_{indice:03d}" for indice in range(base.shape[1])]
    criba = criba_redundancia(matrices[0], nombres, base, nombres_base, filas)
    criba.to_csv(REDUNDANCIA, index=False)
    print(f"[cribado] {int(criba['redundante'].sum())} de {len(criba)} variables con |rho| >= 0,80 "
          f"contra V12 | maxima {criba['correlacion_maxima_v12'].max():.4f} | mediana "
          f"{criba['correlacion_maxima_v12'].median():.4f}", flush=True)

    # ------------------------------------------------------------------ control block-only
    filas_control = []
    for nombre, constructor in (
            ("logreg", lambda: LogisticRegression(C=1.0, max_iter=3000, random_state=cfg.SEED)),
            ("linearsvc", lambda: LinearSVC(C=0.2, random_state=cfg.SEED))):
        por_fold = []
        for fold in folds:
            validacion = fold["val_idx"]
            train_exterior = np.concatenate([otro["val_idx"] for otro in folds
                                             if otro["fold"] != fold["fold"]])
            estimador = constructor()
            estimador.fit(matrices[fold["fold"]][train_exterior], clase[train_exterior])
            por_fold.append(float(f1_score(clase[validacion],
                                           estimador.predict(matrices[fold["fold"]][validacion]),
                                           average="macro",
                                           labels=list(range(len(cfg.CLASSES))))))
        filas_control.append({
            "modelo": nombre, "macro_f1": round(float(np.mean(por_fold)), 4),
            "std": round(float(np.std(por_fold)), 4),
            "per_fold": [round(valor, 4) for valor in por_fold],
            "suelo_azar": round(SUELO_AZAR, 4),
            "supera_el_azar": bool(float(np.mean(por_fold)) > SUELO_AZAR),
        })
        print(f"[control] {nombre}: {np.mean(por_fold):.4f} ± {np.std(por_fold):.4f} "
              f"(suelo {SUELO_AZAR:.4f})", flush=True)
    pd.DataFrame(filas_control).to_csv(CONTROL, index=False)

    # ------------------------------------------------------------------ informe
    lineas = [
        "# V13 · P1a — Evidencia de edad por token, dentro del fold", "",
        "## 1. Que se construye y por que va ahora", "",
        "Las 72 variables supervisadas que ya tiene V12 son `P(clase | entidad)` para menciones, "
        "hashtags y fuentes. **No existe `P(clase | token)` para tokens simples.** Es la primera "
        "pista de V13 y va primera porque P0b showed que las familias de textometria estan "
        "saturadas (VIF mediano 12,6 a 42,8) y las de evidencia supervisada (3,1) y NB-SVM (0,77) "
        "son las unicas con margen.", "",
        f"El vocabulario no se declara: se **deriva del propio fold** por frecuencia de documento "
        f"(minimo {DF_MINIMO} documentos). No es un lexico externo ni cerrado, es una estadistica "
        f"derivada del train, que es lo que permite R-1.", "",
        "## 2. Protocolo antileakage (§38)", "",
        "1. Tokenizado unico con `src.lexicons.tokeniza`, el del vectorizador de V12: no se anade "
        "ningun vectorizador nuevo. El mapa token -> identificador es una codificacion, no "
        "transporta frecuencias ni etiquetas.",
        "2. Para cada fold exterior, el train exterior se reparte en 5 folds interiores (agrupados "
        "por duplicados) y las caracteristicas de cada fila de train salen de un "
        "`P(clase | token)` calculado sobre las otras cuatro partes: ninguna fila ve su propia "
        "etiqueta.",
        "3. Las filas de validacion reciben el perfil calculado sobre todo el train exterior.",
        "4. Vocabulario y frecuencias de documento se calculan siempre sobre el train interior.",
        "5. El test no se toca en ninguna parte.", "",
        "## 3. Cobertura y vocabulario por fold", "",
        "| Fold | Filas train | Filas validacion | Columnas con varianza | Tokens banda especifica | "
        "Tokens banda frecuente |", "|---|---|---|---|---|---|",
    ]
    for fila in resumen:
        lineas.append(f"| {fila['fold']} | {fila['filas_train']:,} | {fila['filas_validacion']:,} | "
                      f"{fila['columnas_con_varianza']} de {len(nombres)} | "
                      f"{fila['tokens_vocabulario_banda_especifica']:,} | "
                      f"{fila['tokens_vocabulario_banda_frecuente']:,} |")
    lineas += ["", "## 4. Cribado de redundancia contra las 279 variables de V12", "",
               f"- Variables redundantes (|rho| >= 0,80): **{int(criba['redundante'].sum())}** de "
               f"{len(criba)}",
               f"- Correlacion maxima: **{criba['correlacion_maxima_v12'].max():.4f}** "
               f"(`{criba.loc[criba['correlacion_maxima_v12'].idxmax(), 'v12_mas_correlacionada']}`)",
               f"- Mediana: {criba['correlacion_maxima_v12'].median():.4f}", "",
               "## 5. Control block-only", "",
               "Mismo protocolo que la fase de lexicones: LogReg y LinearSVC sobre el bloque, "
               "corrected CV agrupado. El suelo de azar es la clase mayoritaria (1/6).", "",
               "| Modelo | macro-F1 | std | Suelo de azar | Supera el azar |",
               "|---|---|---|---|---|"]
    for fila in filas_control:
        lineas.append(f"| {fila['modelo']} | {fila['macro_f1']:.4f} | {fila['std']:.4f} | "
                      f"{fila['suelo_azar']:.4f} | {'si' if fila['supera_el_azar'] else 'no'} |")
    lineas += ["", "## 6. Decision", ""]
    if all(fila["supera_el_azar"] for fila in filas_control):
        lineas.append("El bloque **supera el suelo de azar en los dos modelos**: hay senal en "
                      "`P(clase | token)`. Procede P1b (fusion con V12, control de permutado y "
                      "gate §24).")
    elif any(fila["supera_el_azar"] for fila in filas_control):
        lineas.append("Solo uno de los dos modelos supera el suelo de azar: senal debil y "
                      "dependiente del clasificador. Procede P1b, donde el control de permutado "
                      "es decisivo.")
    else:
        lineas.append("El bloque **no supera el suelo de azar**: no hay senal en "
                      "`P(clase | token)` y la pista se cierra sin fusionar.")
    lineas += ["",
               f"Caracteristicas: {len(nombres)} ({len(BANDAS)} bandas de frecuencia de "
               f"documento: " + ", ".join(
                   f"{banda} df en [{bajo}, {alto if alto < (1 << 29) else 'inf'})"
                   for banda, (bajo, alto) in BANDAS.items()) + ").", "",
               "Los tokens mas asociados a cada clase estan en "
               "`reports/v13_p1_tokens_destacados.csv`. Son **descriptivos**: la prueba de "
               "significancia con permutado (regla §27) se hace en P1b, sobre el bloque completo y "
               "no sobre tokens sueltos.", ""]
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    RESULTADO.write_text(json.dumps({
        "fase": "V13 · P1a evidencia por token", "fecha": "2026-09-25",
        "n_tokens_vocabulario": len(vocabulario), "n_ocurrencias": int(len(indice)),
        "n_caracteristicas": len(nombres), "caracteristicas_por_banda":
            CARACTERISTICAS_POR_BANDA,
        "bandas": {banda: [bajo, alto if alto < (1 << 29) else None]
                   for banda, (bajo, alto) in BANDAS.items()},
        "df_minimo": DF_MINIMO, "suavizado": SUAVIZADO, "suelo_azar": round(SUELO_AZAR, 4),
        "por_fold": resumen, "redundantes_contra_v12": int(criba["redundante"].sum()),
        "correlacion_maxima_v12": float(criba["correlacion_maxima_v12"].max()),
        "correlacion_mediana_v12": float(criba["correlacion_maxima_v12"].median()),
        "control": filas_control,
        "supera_el_azar": bool(all(fila["supera_el_azar"] for fila in filas_control)),
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nartefactos: {INFORME.name}, {REDUNDANCIA.name}, {CONTROL.name} y las cinco matrices "
          f"en features/v13_p1_tokens_fold*.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
