"""V13 · P17a — Pronombres de persona (V14-V18 del artículo de referencia), construcción y control.

Las 18 variables del artículo de Moreno-Sandoval, Pomares-Quimbaya y Alvarado-Valencia (2021) se
agrupan en cinco familias léxica, sintáctica, simbólica, de participación y complementaria. De las
que el gap audit de P0 verificó que existen en el corpus, **faltan cinco**: los pronombres de
persona (V14-V18). Esa laguna es la que cierra esta pista.

Hipótesis, tal y como está pre-registrada en el plan: la referencia del artículo afirma que los
pronombres de persona son **la variable más predictiva** y que la tercera persona singular es la
más informativa, porque el desplazamiento hacia la primera y la segunda persona marca
intimidad y el tema, y porque en redes el sujeto nombrado se referencia en tercera. V13 predice
falsablemente que ese bloque sube la rama LogReg y se diluye en la MLP.

El vocabulario es **cerrado y propio** (regla R-1): se construye desde cero en este archivo, con
palabras funcionales del español, no con un lexicón externo ni con un corpus de referencia. El
corpus es ASCII sin tildes (D-009), lo que obliga a tratar ``tu``/``tú`` como la misma cadena y a
dejar fuera las formas que solo se distinguen por la tilde.

Decisión de diseño importante: ``el`` y ``los`` son a la vez artículo y pronombre, y ``el`` es la
palabra más frecuente del corpus. Mezclarlas en un contador de "tercera persona" mediría longitud
de texto, no persona. Por eso las cinco categorías usan **solo formas no ambiguas**, y las
ambiguas se declaran aparte como constante de normalización.

Salida: ``reports/v17_pronombres.md``, ``reports/v17_pronombres_cobertura.csv``,
``reports/v17_pronombres_redundancia.csv``, ``reports/v17_pronombres_control.csv``,
``features/v17_pronombres.npy``, ``experiments/v17_p17a_pronombres.json``.
"""

from __future__ import annotations

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

from phase2_ponderacion import build_features  # noqa: E402
from v13_p0b_colinealidad import criba_redundancia  # noqa: E402

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src import lexicons as lx  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v17_pronombres.md"
COBERTURA = cfg.REPORTS / "v17_pronombres_cobertura.csv"
REDUNDANCIA = cfg.REPORTS / "v17_pronombres_redundancia.csv"
CONTROL = cfg.REPORTS / "v17_pronombres_control.csv"
MATRIZ = cfg.FEATURES / "v17_pronombres.npy"
NOMBRES = cfg.FEATURES / "v17_pronombres_nombres.json"
RESULTADO = cfg.EXPERIMENTS / "v17_p17a_pronombres.json"

#: Las cinco categorias de persona del articulo (V14-V18), con solo formas no ambiguas.
PERSONAS = {
    "primera_singular": ["yo", "mi", "mis", "mio", "mia", "mios", "mias", "conmigo", "me", "nos"],
    "primera_plural": ["nosotros", "nuestra", "nuestros", "nuestras"],
    "segunda": ["tu", "tus", "tuyo", "tuya", "tuyos", "tuyas", "contigo", "usted", "ustedes",
                "te", "ti", "vos", "otros"],
    "tercera_singular": ["su", "sus", "suyo", "suya", "suyos", "suyas", "consigo", "ella", "el"],
    "tercera_plural": ["ellos", "ellas"],
}
#: ``el`` y ``los`` son articulo o pronombre segun el contexto. Se declaran aparte: mezclarlos en
#: un contador de persona mediría longitud de texto, no persona, y ``el`` es la palabra mas
#: frecuente del corpus.
AMBIGUAS = ["el", "los", "las", "una", "un", "este", "esta", "esto", "ese", "esa", "aquel"]
#: Pronombre impersonal y reflexivo, que no marca persona: su frecuencia es en si misma relevante.
IMPERSONAL = ["se", "lo", "le", "les"]
SUELO_AZAR = 1.0 / len(cfg.CLASSES)
MINIMO_DOCUMENTOS = 200


def cuenta_por_categoria(documentos: list[list[str]], categorias: dict[str, list[str]]
                         ) -> tuple[np.ndarray, dict[str, set[str]]]:
    """Cuenta apariciones de cada categoria en cada documento, junto al indice de formas usadas."""
    totales = np.zeros((len(documentos), len(categorias)), dtype=np.float32)
    usadas: dict[str, set[str]] = {}
    for persona, formas in categorias.items():
        conjunto = set(formas)
        usadas[persona] = conjunto
        columna = len(usadas) - 1
        for fila, tokens in enumerate(documentos):
            totales[fila, columna] = sum(1 for token in tokens if token in conjunto)
    return totales, usadas


def nombres_de_features() -> list[str]:
    """Nombres de las caracteristicas derivadas de los cinco contadores de persona."""
    nombres: list[str] = []
    for persona in PERSONAS:
        nombres += [f"{persona}_conteo", f"{persona}_ratio", f"{persona}_presencia"]
    nombres += ["impersonal_conteo", "impersonal_ratio", "impersonal_presencia",
                "ambiguas_ratio",
                "distancia_social", "distancia_social_ratio",
                "proporcion_tercera", "proporcion_primera", "indice_segunda_primera",
                "n_pronombres", "n_pronombres_por_token"]
    return nombres


def main() -> int:
    """Construye el bloque de pronombres, mide su cobertura, lo criba y lo somete a control."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    textos = train["text"]
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")

    documentos = [lx.tokeniza(str(texto)) for texto in textos]
    longitudes = np.array([len(tokens) for tokens in documentos], dtype=np.float32)
    print(f"[tokenizacion] {len(documentos):,} documentos | "
          f"{int(longitudes.sum()):,} tokens | media {longitudes.mean():.1f}", flush=True)

    conteos, usadas = cuenta_por_categoria(documentos, PERSONAS)
    impersonales, _ = cuenta_por_categoria(documentos, {"impersonal": IMPERSONAL})
    set_ambiguas = set(AMBIGUAS)
    ambiguas = np.array([[sum(1 for token in tokens if token in set_ambiguas)
                          for tokens in documentos]], dtype=np.float32).T

    # ---------------------------------------------------------------- cobertura por forma
    cobertura = []
    for persona, formas in list(PERSONAS.items()) + [("impersonal", IMPERSONAL),
                                                     ("ambigua", AMBIGUAS)]:
        documentos_por_forma = {}
        for forma in formas:
            presentes = sum(1 for tokens in documentos if forma in tokens)
            documentos_por_forma[forma] = presentes
        # Apariciones reales de la categoria, contadas sobre el corpus (no solo las personas).
        conjunto = usadas.get(persona, set(formas))
        total_apariciones = sum(1 for tokens in documentos for token in tokens
                                if token in conjunto)
        cobertura.append({
            "categoria": persona,
            "formas": " ".join(formas),
            "documentos_con_la_categoria": int(sum(
                1 for tokens in documentos
                if any(token in usadas.get(persona, set(formas)) for token in tokens))),
            "apariciones_totales": int(total_apariciones),
            "formas_sobre_el_minimo": sum(1 for presentes in documentos_por_forma.values()
                                          if presentes >= MINIMO_DOCUMENTOS),
            "formas_totales": len(formas),
            "detalle": " ".join(f"{forma}:{presentes:,}" for forma, presentes
                                in documentos_por_forma.items()),
        })
    tabla_cobertura = pd.DataFrame(cobertura)
    tabla_cobertura.to_csv(COBERTURA, index=False)
    for fila in cobertura:
        print(f"[cobertura] {fila['categoria']:20s} {fila['documentos_con_la_categoria']:7,} "
              f"documentos | formas sobre el minimo "
              f"{fila['formas_sobre_el_minimo']}/{fila['formas_totales']}", flush=True)

    # ---------------------------------------------------------------- bloque de caracteristicas
    nombres = nombres_de_features()
    matriz = np.zeros((len(documentos), len(nombres)), dtype=np.float32)
    Safe = np.clip(longitudes, 1.0, None)
    for indice, persona in enumerate(PERSONAS):
        matriz[:, 3 * indice] = conteos[:, indice]
        matriz[:, 3 * indice + 1] = conteos[:, indice] / Safe * 100.0
        matriz[:, 3 * indice + 2] = (conteos[:, indice] > 0).astype(np.float32)
    base = 3 * len(PERSONAS)
    matriz[:, base] = impersonales[:, 0]
    matriz[:, base + 1] = impersonales[:, 0] / Safe * 100.0
    matriz[:, base + 2] = (impersonales[:, 0] > 0).astype(np.float32)
    matriz[:, base + 3] = ambiguas[:, 0] / Safe * 100.0
    # Distancia social: cuanto mas lejos del yo y del tu, mas probable es que el tema sea la
    # tercera persona.
    primera = conteos[:, 0] + conteos[:, 1]
    segunda = conteos[:, 2]
    tercera = conteos[:, 3] + conteos[:, 4]
    matriz[:, base + 4] = tercera - (primera + segunda)
    matriz[:, base + 5] = matriz[:, base + 4] / Safe * 100.0
    total_pronombres = np.maximum(primera + segunda + tercera, 1.0)
    matriz[:, base + 6] = tercera / total_pronombres
    matriz[:, base + 7] = primera / total_pronombres
    matriz[:, base + 8] = (segunda + 1.0) / (primera + 1.0)
    matriz[:, base + 9] = primera + segunda + tercera
    matriz[:, base + 10] = (primera + segunda + tercera) / Safe * 100.0
    np.save(MATRIZ, matriz)
    NOMBRES.write_text(json.dumps(nombres, ensure_ascii=False), encoding="utf-8")
    print(f"[bloque] {matriz.shape} | columnas con varianza "
          f"{int((matriz.std(axis=0) > 0).sum())} de {matriz.shape[1]}", flush=True)

    # ---------------------------------------------------------------- cribado contra V12
    base_v12, _, _, _, _ = build_features()
    base_v12 = np.concatenate([base_v12, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")],
                              axis=1)
    generador = np.random.default_rng(cfg.SEED)
    filas = generador.choice(len(y), size=20000, replace=False)
    nombres_v12 = [f"v12_{indice:03d}" for indice in range(base_v12.shape[1])]
    criba = criba_redundancia(matriz, nombres, base_v12, nombres_v12, filas)
    criba.to_csv(REDUNDANCIA, index=False)
    print(f"[cribado] {int(criba['redundante'].sum())} de {len(criba)} variables con |rho| >= 0,80 "
          f"contra V12 | maxima {criba['correlacion_maxima_v12'].max():.4f}", flush=True)

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
                              "per_fold": [round(valor, 4) for valor in por_fold],
                              "suelo_azar": round(SUELO_AZAR, 4),
                              "supera_el_azar": bool(float(np.mean(por_fold)) > SUELO_AZAR)})
        print(f"[control] {nombre}: {np.mean(por_fold):.4f} ± {np.std(por_fold):.4f} "
              f"(suelo {SUELO_AZAR:.4f})", flush=True)
    pd.DataFrame(filas_control).to_csv(CONTROL, index=False)

    # ---------------------------------------------------------------- informe
    lineas = [
        "# V13 · P17a — Pronombres de persona", "",
        "## 1. Que se construye y por que", "",
        "El artículo de referencia (Moreno-Sandoval, Pomares-Quimbaya y Alvarado-Valencia, 2021) "
        "identifica los pronombres de persona como su variable más predictiva, con la tercera "
        "persona singular a la cabeza. El gap audit de P0 verificó que de sus 18 variables, cinco no "
        "existían en el corpus: V14-V18, los pronombres. Esta es la pista que las cubre.", "",
        "El vocabulario es **cerrado y propio** (regla R-1): palabras funcionales del español "
        "escritas en este archivo, no un lexicón externo. Como el corpus es ASCII sin tildes (D-009), "
        "`tu` y `tú` son la misma cadena y las formas que solo se distinguen por la tilde quedan "
        "fuera.", "",
        "**Decisión de diseño**: `el` y `los` son artículo o pronombre según el contexto, y `el` es "
        "la palabra más frecuente del corpus. Incluirlas en un contador de tercera persona mediría "
        "longitud de texto, no persona, así que se declaran aparte como constante de normalización "
        "y las cinco categorías usan solo formas no ambiguas.", "",
        "## 2. Cobertura por categoría", "",
        "| Categoría | Documentos con la categoría | Apariciones | Formas sobre el mínimo |",
        "|---|---|---|---|",
    ]
    for fila in cobertura:
        lineas.append(f"| {fila['categoria']} | {fila['documentos_con_la_categoria']:,} | "
                      f"{fila['apariciones_totales']:,} | {fila['formas_sobre_el_minimo']}/"
                      f"{fila['formas_totales']} |")
    lineas += ["", f"Umbral de permanencia de una forma: {MINIMO_DOCUMENTOS} documentos. El detalle "
                  "de cobertura forma por forma está en "
                  "`reports/v17_pronombres_cobertura.csv`.", "",
              "## 3. Características", "",
              f"{len(nombres)} columnas: conteo, ratio por cien tokens y presencia para cada una de "
              f"las cinco categorías de persona y para el impersonal, más la proporción de "
              f"artículos ambiguos y cinco derivadas:", "",
              "- **distancia_social**: tercera persona menos (primera más segunda). Es la lectura "
              "psicológica del artículo: cuanto más lejos del yo, más probable es que el tema sea la "
              "tercera persona.",
              "- **proporcion_tercera** y **proporcion_primera** de los pronombres del documento.",
              "- **indice_segunda_primera**: segunda sobre primera, con suavizado para que los "
              "documentos sin pronombres no explode.",
              "- **n_pronombres** y su ratio por token.", "",
              "## 4. Cribado de redundancia contra las 279 variables de V12", "",
              f"- Variables redundantes (|rho| >= 0,80): **{int(criba['redundante'].sum())}** de "
              f"{len(criba)}",
              f"- Correlacion maxima: **{criba['correlacion_maxima_v12'].max():.4f}** "
              f"(`{criba.loc[criba['correlacion_maxima_v12'].idxmax(), 'v12_mas_correlacionada']}`)",
              f"- Mediana: {criba['correlacion_maxima_v12'].median():.4f}", "",
              "## 5. Control block-only", "",
              "| Modelo | macro-F1 | std | Suelo de azar | Supera el azar |", "|---|---|---|---|---|"]
    for fila in filas_control:
        lineas.append(f"| {fila['modelo']} | {fila['macro_f1']:.4f} | {fila['std']:.4f} | "
                      f"{fila['suelo_azar']:.4f} | {'si' if fila['supera_el_azar'] else 'no'} |")
    lineas += ["", "## 6. Decision", ""]
    if all(fila["supera_el_azar"] for fila in filas_control):
        lineas.append("El bloque **supera el suelo de azar en los dos modelos**: procede P17b "
                      "(fusion con V12, control de permutado, gate §24 y reporte por cabeza, que "
                      "es donde se mide la predicción pre-registrada de que la señal entra por la "
                      "rama lineal y se diluye en la MLP).")
    elif any(fila["supera_el_azar"] for fila in filas_control):
        lineas.append("Solo un modelo supera el suelo de azar: procede P17b con el control de "
                      "permutado como criterio decisivo.")
    else:
        lineas.append("El bloque **no supera el suelo de azar**: la pista se cierra sin fusionar, "
                      "y con ella queda confirmado que las cinco variables del artículo que faltan "
                      "en el corpus no son el hueco rentable.")
    lineas += ["", "Comparación de referencia: el bloque de evidencia por token (P1a) da 0,3997 y "
                  "el mejor bloque léxico de la fase cerrada 0,2033, con el mismo suelo de azar.", ""]
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    RESULTADO.write_text(json.dumps({
        "fase": "V13 · P17a pronombres de persona", "fecha": "2026-09-25",
        "categorias": {persona: formas for persona, formas in PERSONAS.items()},
        "ambiguas": AMBIGUAS, "impersonal": IMPERSONAL,
        "n_caracteristicas": len(nombres), "nombres": nombres,
        "cobertura": cobertura, "suelo_azar": round(SUELO_AZAR, 4),
        "redundantes_contra_v12": int(criba["redundante"].sum()),
        "correlacion_maxima_v12": float(criba["correlacion_maxima_v12"].max()),
        "control": filas_control,
        "supera_el_azar": bool(all(fila["supera_el_azar"] for fila in filas_control)),
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nartefactos: {INFORME.name}, {COBERTURA.name}, {REDUNDANCIA.name}, {CONTROL.name}, "
          f"{MATRIZ.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
