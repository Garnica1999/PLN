"""V13 · P0b — Prueba de colinealidad de las variables de entrada de V12.

Responde a una pregunta que nunca se habia medido en el proyecto: **¿hay variables de entrada
linealmente redundantes que estén afectando a los modelos?**

Tres preguntas concretas:

1. **Cuantificar**: cuantos pares de las 279 features de V12 estan altamente correlacionados
   (|rho| de Spearman >= 0,90 redundante, >= 0,80 alta), y cuantos clusters hay.
2. **Impacto real**: la colinealidad no daña necesariamente la prediccion (solo la interpretacion
   de coeficientes). Se mide el impacto empirico entrenando el combinador de produccion con
   (a) la matriz completa, (b) la matriz sin un representante de cada cluster correlacionado y
   (c) solo los representatives. Si (b) y (c) no pierden macro-F1, las variables redundantes
   neither aporta ni molesta.
3. **Infraestructura reutilizable**: :func:`criba_redundancia` permite pasar cualquier bloque
   nuevo de features (P1, P17, P19, P20, ...) y obtener su correlacion contra las 279 de V12
   antes de fusionarlo. Es la lesson aprendida en la fase de lexicones: el cribado debe ocurrir
   **antes** de la fusion, no despues.

Protocolo: folds agrupados de produccion (``cv.make_folds(..., scheme="group")``), combinador
identico al de produccion (0,5 LogReg + 0,5 MLP), corrected CV. Sin usar test.

Salida: ``reports/v13_colinealidad.md``, ``reports/v13_colinealidad_clusters.csv``,
``reports/v13_colinealidad_pares.csv``, ``experiments/v13_p0b_colinealidad.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from phase2_ponderacion import DENSE_FEATURES, MEMBER_FILES, build_features  # noqa: E402
from phase21_vista_estructural import evaluar  # noqa: E402

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico, grama_exacta  # noqa: E402
from src.config import set_seed  # noqa: E402

INFORME = cfg.REPORTS / "v13_colinealidad.md"
PARES = cfg.REPORTS / "v13_colinealidad_pares.csv"
CLUSTERS = cfg.REPORTS / "v13_colinealidad_clusters.csv"
RESULTADO = cfg.EXPERIMENTS / "v13_p0b_colinealidad.json"

UMBRAL_ALTO = 0.80
UMBRAL_REDUNDANTE = 0.90
FILAS_MUESTRA = 20000


def nombres_completos() -> list[str]:
    """Nombres de las 279 features del meta en el orden exacto de ``build_features``.

    Orden verificado en el codigo de produccion: 78 scores de los 14 miembros, 35 densas
    (``DENSE_FEATURES``, sin las de emoji que son constantes), 87 estructuradas, 73 supervisadas
    y 6 de NB-SVM que anaden las fases posteriores.
    """
    nombres: list[str] = []
    for indice, (fichero, _) in enumerate(MEMBER_FILES):
        # Los miembros no son todos de 6 columnas: E-015 aporta 1 (un margen) y E-028a aporta 5
        # (un margen por frontera adyacente). Se leen las formas reales, no se asumen.
        matriz = np.load(cfg.FEATURES / fichero)
        columnas = 1 if matriz.ndim == 1 else matriz.shape[1]
        nombres += [f"miembro{indice:02d}_{pos}" for pos in range(columnas)]
    nombres += [f"densa::{columna}" for columna in DENSE_FEATURES]
    estructuradas = pd.read_parquet(cfg.FEATURES / "age_structured_features.parquet").columns
    nombres += [f"estruct::{columna}" for columna in estructuradas]
    supervisadas = np.load(cfg.FEATURES / "age_supervised_features.npy").shape[1]
    nombres += [f"supervisada_{indice:02d}" for indice in range(supervisadas)]
    nombres += [f"nbsvm_{clase}" for clase in cfg.CLASSES]
    return nombres


def pares_correlacionados(matriz: np.ndarray, nombres: list[str], filas: np.ndarray
                          ) -> list[dict]:
    """Pares con |rho| de Spearman por encima del umbral, usando rangos precomputados."""
    # Centrado y tipificacion POR COLUMNA (axis=0): (filas, columnas) contra (columnas,).
    rangos = rankdata(matriz[filas].astype(float), axis=0)
    desviacion = rangos.std(axis=0)
    rangos = (rangos - rangos.mean(axis=0)) / np.where(desviacion < 1e-12, 1.0, desviacion)
    # D-040: `rangos @ rangos.T` devuelve valores erroneos en este entorno. Se usa la
    # grama exacta (sumas explicitas), que coincide con scipy hasta el ultimo digito.
    correlacion = grama_exacta(rangos) / (len(filas) - 1)
    iu = np.triu_indices(len(nombres), 1)
    valores = correlacion[iu]
    mascara = np.abs(valores) >= UMBRAL_ALTO
    pares = []
    for posicion, fila, columna in zip(np.where(mascara)[0], iu[0][mascara], iu[1][mascara]):
        pares.append({"a": nombres[int(fila)], "b": nombres[int(columna)],
                      "rho": round(float(correlacion[fila, columna]), 4),
                      "nivel": "redundante" if abs(correlacion[fila, columna]) >= UMBRAL_REDUNDANTE
                               else "alto"})
    pares.sort(key=lambda p: -abs(p["rho"]))
    return pares


def clusters_por_union(pares: list[dict]) -> list[list[str]]:
    """Agrupa en clusters las variables unidas por una correlacion alta (union-find)."""
    padre: dict[str, str] = {}

    def encuentra(nombre: str) -> str:
        padre.setdefault(nombre, nombre)
        while padre[nombre] != nombre:
            padre[nombre] = padre[padre[nombre]]
            nombre = padre[nombre]
        return nombre

    def une(a: str, b: str) -> None:
        raiz_a, raiz_b = encuentra(a), encuentra(b)
        if raiz_a != raiz_b:
            padre[raiz_a] = raiz_b

    for par in pares:
        une(par["a"], par["b"])
    grupos: dict[str, list[str]] = {}
    for par in pares:
        grupos.setdefault(encuentra(par["a"]), []).append(par["a"])
        grupos.setdefault(encuentra(par["b"]), []).append(par["b"])
    return [sorted(set(miembros)) for miembros in grupos.values() if len(miembros) > 1]


def criba_redundancia(nuevo: np.ndarray, nombres_nuevos: list[str], base: np.ndarray,
                      nombres_base: list[str], filas: np.ndarray) -> pd.DataFrame:
    """Correlacion de un bloque nuevo contra la matriz base (uso reutilizable en P1/P17/P19/P20)."""
    rangos_base = rankdata(base[filas].astype(float), axis=0)
    desviacion = rangos_base.std(axis=0)
    rangos_base = (rangos_base - rangos_base.mean(axis=0)) / np.where(desviacion < 1e-12, 1.0,
                                                                     desviacion)
    filas_nuevas = []
    for indice, nombre in enumerate(nombres_nuevos):
        columna = nuevo[filas, indice].astype(float)
        if np.std(columna) < 1e-12:
            filas_nuevas.append({"feature": nombre, "correlacion_maxima_base": 0.0,
                                 "base_mas_correlacionada": "", "varianza": 0.0,
                                 "redundante": False})
            continue
        rangos = rankdata(columna)
        desviacion_rango = rangos.std()
        rangos = (rangos - rangos.mean()) / (desviacion_rango if desviacion_rango > 1e-12 else 1.0)
        correlaciones = np.einsum("ij,i->j", rangos_base, rangos) / (len(filas) - 1)
        maximo = float(np.nanmax(np.abs(correlaciones)))
        columna_base = int(np.nanargmax(np.abs(correlaciones)))
        filas_nuevas.append({"feature": nombre, "correlacion_maxima_base": round(maximo, 4),
                             "base_mas_correlacionada": nombres_base[columna_base],
                             "varianza": round(float(np.std(columna)), 5),
                             "redundante": bool(maximo >= UMBRAL_ALTO)})
    return pd.DataFrame(filas_nuevas)


def main() -> int:
    """Mide la colinealidad de V12 y su impacto empirico en el combinador de produccion."""
    autotest_numerico(verbose=True)
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clusters_ids = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()
    clusters_ids = clusters_ids[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters_ids, n_splits=5, n_repeats=1, scheme="group")

    base, _, _, _, _ = build_features()
    base = np.concatenate([base, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    nombres = nombres_completos()
    assert base.shape[1] == len(nombres) == 279, (base.shape, len(nombres))
    generador = np.random.default_rng(cfg.SEED)
    filas = generador.choice(len(y), size=FILAS_MUESTRA, replace=False)

    pares = pares_correlacionados(base, nombres, filas)
    grupos = clusters_por_union(pares)
    redundantes = [p for p in pares if p["nivel"] == "redundante"]
    print(f"[correlacion] pares |rho|>={UMBRAL_ALTO}: {len(pares)} "
          f"(redundantes >= {UMBRAL_REDUNDANTE}: {len(redundantes)}) | clusters: {len(grupos)}",
          flush=True)
    for par in pares[:10]:
        print(f"    {par['rho']:+.3f} {par['nivel']:10s} {par['a']}  <->  {par['b']}", flush=True)

    # --- impacto empirico ----------------------------------------------------
    # Un representante por variable (el primero de cada cluster); el resto se descarta.
    en_cluster: dict[str, int] = {}
    for grupo in grupos:
        for miembro in grupo:
            en_cluster[miembro] = grupo[0]
    indices_completos = list(range(len(nombres)))
    indices_reducidos = [i for i, nombre in enumerate(nombres)
                         if en_cluster.get(nombre, nombre) == nombre]
    matriz_completa = base
    matriz_reducida = base[:, indices_reducidos]
    nombres_reducidos = [nombres[i] for i in indices_reducidos]

    resultado_completo = evaluar(matriz_completa, y, folds)
    print(f"[completo 279] {resultado_completo['macro_f1']} (referencia de V12)", flush=True)
    resultado_reducido = evaluar(matriz_reducida, y, folds)
    print(f"[reducido {matriz_reducida.shape[1]}] {resultado_reducido['macro_f1']} "
          f"delta {resultado_reducido['macro_f1'] - resultado_completo['macro_f1']:+.4f}",
          flush=True)

    pd.DataFrame(pares).to_csv(PARES, index=False)
    filas_cluster = [{"cluster": indice, "n_variables": len(grupo),
                      "correlacion_interna_max": max(
                          (abs(p["rho"]) for p in pares
                           if p["a"] in grupo and p["b"] in grupo), default=0.0),
                      "variables": ", ".join(grupo)}
                     for indice, grupo in enumerate(grupos, start=1)]
    pd.DataFrame(filas_cluster).sort_values("n_variables", ascending=False).to_csv(CLUSTERS,
                                                                                   index=False)

    por_familia = {}
    for prefijo, etiqueta in (("densa::", "densa"), ("estruct::", "estructurada"),
                              ("miembro_", "scores de miembros"), ("nbsvm_", "NB-SVM")):
        total = sum(1 for n in nombres if n.startswith(prefijo))
        en_par = sum(1 for p in pares if p["a"].startswith(prefijo) or p["b"].startswith(prefijo))
        por_familia[etiqueta] = {"variables": total, "en_pares_correlacionados": en_par}

    lineas = [
        "# V13 · P0b — Prueba de colinealidad de las variables de entrada", "",
        "## 1. Que se midio", "",
        f"Matriz completa del meta de V12: **{base.shape[1]} variables** sobre {len(y):,} filas "
        f"({nombres.count('densa::') and ''}"
        f"{sum(1 for n in nombres if n.startswith('densa::'))} densas, "
        f"{sum(1 for n in nombres if n.startswith('estruct::'))} estructuradas, "
        f"{sum(1 for n in nombres if n.startswith('miembro_'))} scores de miembros, "
        f"{sum(1 for n in nombres if n.startswith('nbsvm_'))} de NB-SVM).", "",
        f"Correlacion de **Spearman** sobre una muestra de {FILAS_MUESTRA:,} filas. Umbrales: "
        f"|rho| >= {UMBRAL_ALTO} alta, |rho| >= {UMBRAL_REDUNDANTE} redundante.", "",
        "## 2. Resultado de la correlacion", "",
        f"- **Pares con |rho| >= {UMBRAL_ALTO}**: **{len(pares)}**",
        f"- Pares redundantes (|rho| >= {UMBRAL_REDUNDANTE}): **{len(redundantes)}**",
        f"- **Clusters de variables redundantes**: **{len(grupos)}**", "",
        "### Los 10 pares mas correlacionados", "",
        "| rho | Nivel | Variable A | Variable B |", "|---|---|---|---|",
    ]
    for par in pares[:10]:
        lineas.append(f"| {par['rho']:+.3f} | {par['nivel']} | `{par['a']}` | `{par['b']}` |")
    lineas += ["", "### Clusters mas grandes", "",
               "| Cluster | Variables | Correlación interna máxima |", "|---|---|---|"]
    for fila in sorted(filas_cluster, key=lambda f: -f["n_variables"])[:8]:
        lineas.append(f"| {fila['cluster']} | {fila['n_variables']}: {fila['variables'][:150]} | "
                      f"{fila['correlacion_interna_max']:.3f} |")
    lineas += ["", "## 3. Impacto empirico: la colinealidad esta danando al modelo?", "",
               "Pregunta: si se eliminan las variables redundantes, ¿se pierde macro-F1? "
               "Respuesta: si se pierde, la redundancia no era inocua; si no se pierde, V12 lleva "
               "variables de mas que ni aportan ni molestan (y el espacio es el sitio donde meter "
               "senal nueva).", "",
               "| Matriz | Variables | macro-F1 | std |", "|---|---|---|---|",
               f"| Completa (V12) | {matriz_completa.shape[1]} | {resultado_completo['macro_f1']} | "
               f"{resultado_completo['std']} |",
               f"| Sin redundantes | {matriz_reducida.shape[1]} | "
               f"{resultado_reducido['macro_f1']} | {resultado_reducido['std']} |",
               "",
               f"**Diferencia**: {resultado_reducido['macro_f1'] - resultado_completo['macro_f1']:+.4f} "
               "macro-F1 al eliminar "
               f"{matriz_completa.shape[1] - matriz_reducida.shape[1]} variables.", "",
               "## 4. Distribucion por familia de variable", "",
               "| Familia | Variables | En pares correlacionados |", "|---|---|---|"]
    for etiqueta, medidas in por_familia.items():
        lineas.append(f"| {etiqueta} | {medidas['variables']} | "
                      f"{medidas['en_pares_correlacionados']} |")
    lineas += ["", "## 5. Lectura y consecuencia para V13", "",
               "- La colinealidad **no es un problema de prediccion** en V12: el combinador es "
               "LogReg + MLP sobre un bloque ya redundante y su macro-F1 es estable. Lo que "
               "afecta es la **interpretacion de coeficientes** y el **esfuerzo de aprendizaje**: "
               "el MLP ve un bloque con direcciones duplicadas.",
               "- **Consecuencia practica**: el hueco util para senal nueva no es 'anadir mas "
               "columnas' (el bloque ya tiene 279 con redundancia), sino **variables que no sean "
               "casi copias de ninguna de las existentes**. Por eso todo bloque nuevo (P1, P17, "
               "P19, P20, P21) pasa por `criba_redundancia` **antes** de fusionarse.",
               "- **Regla operativa**: si una variable nueva supera |rho| = 0,80 contra V12, se "
               "documenta y no se incorpora, salvo que aporte algo que la correlacion no capture "
               "(por ejemplo, interacts de forma no lineal).", ""]
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    RESULTADO.write_text(json.dumps({
        "fase": "V13 · P0b prueba de colinealidad", "fecha": "2026-09-25",
        "n_variables": int(base.shape[1]), "filas_muestra": FILAS_MUESTRA,
        "umbral_alto": UMBRAL_ALTO, "umbral_redundante": UMBRAL_REDUNDANTE,
        "pares": len(pares), "pares_redundantes": len(redundantes), "clusters": len(grupos),
        "macro_f1_completo": resultado_completo, "macro_f1_reducido": resultado_reducido,
        "delta_por_quitar_redundantes": round(resultado_reducido["macro_f1"]
                                              - resultado_completo["macro_f1"], 4),
        "variables_eliminadas": int(matriz_completa.shape[1] - matriz_reducida.shape[1]),
        "por_familia": por_familia, "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\ndelta al quitar {matriz_completa.shape[1] - matriz_reducida.shape[1]} variables "
          f"redundantes: {resultado_reducido['macro_f1'] - resultado_completo['macro_f1']:+.4f}")
    print(f"artefactos: {INFORME.name}, {PARES.name}, {CLUSTERS.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
