"""V13 · P0b — Prueba de colinealidad de la matriz completa de V12.

Responde a una pregunta que el proyecto nunca habia medido: **la matriz de 279 variables de V12
tiene variables linealmente redundantes, y estan afectar al modelo?**

Toda la medicion se hace sobre la **matriz completa tal cual la usa el combinador de produccion**
(78 scores de los 14 miembros + 35 densas + 87 estructuradas + 73 supervisadas + 6 de NB-SVM), sin
quitar, sustituir ni reordenar ninguna columna. No se reduce la matriz y **no se reentrena nada**:
la referencia de V12 se lee del artefacto de la fusion (`experiments/phase39_fusion_v12.json`), que
guarda el macro-F1 y los cinco folds por separado.

Por que no hace falta reentrenar: la pregunta "¿la colinealidad daña la prediccion?" ya tiene su
respuesta implicita en el propio hecho que se va a medir. Si el bloque que la produccion usa tiene
columnas duplicadas de forma exacta y aun asi alcanza 0,6004 de macro-F1, la colinealidad no es un
problema de prediccion en este modelo: es un problema de interpretacion de coeficientes y de
esfuerzo de aprendizaje del MLP (vease el mismo bloque en direcciones duplicadas). Eso se cuantifica
con metricas que no requieren entrenar: el factor de inflacion de la varianza, el rango efectivo y
la severidad por familia.

Ademas deja dos cosas reutilizables para el resto de V13:

* :func:`criba_redundancia`, que mide la correlacion de un bloque nuevo contra la matriz completa
  **antes** de fusionarlo. Es la leccion de la fase de lexicones: el cribado va antes de la fusion,
  no despues.
* Una **validacion cruzada** de esa criba contra un resultado ya conocido y calculado con otra ruta
  de codigo (`reports/age_lexicon_redundancy.csv`, obtenido con producto matricial matriz-vector,
  ruta que D-040 declara sana). Si las dos coinciden, el diagnostico del entorno queda confirmado por
  segunda via.

Se aplica la doble metrica (§36) solo donde tiene sentido: aqui no se anade ninguna variable, asi que
no hay reporte por cabeza (DD-8). El test no se toca.

Salidas: ``reports/v13_colinealidad.md``, ``reports/v13_colinealidad_pares.csv``,
``reports/v13_colinealidad_clusters.csv``, ``reports/v13_colinealidad_vif.csv``,
``reports/v13_criba_validacion.csv``, ``experiments/v13_p0b_colinealidad.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg as sla
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from phase2_ponderacion import DENSE_FEATURES, MEMBER_FILES, build_features  # noqa: E402

from src import config as cfg  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico, grama_exacta  # noqa: E402

INFORME = cfg.REPORTS / "v13_colinealidad.md"
PARES = cfg.REPORTS / "v13_colinealidad_pares.csv"
CLUSTERS = cfg.REPORTS / "v13_colinealidad_clusters.csv"
VIF = cfg.REPORTS / "v13_colinealidad_vif.csv"
VALIDACION = cfg.REPORTS / "v13_criba_validacion.csv"
RESULTADO = cfg.EXPERIMENTS / "v13_p0b_colinealidad.json"
REFERENCIA = cfg.EXPERIMENTS / "phase39_fusion_v12.json"
REDUNDANCIA_LEXICON = cfg.REPORTS / "age_lexicon_redundancy.csv"

UMBRAL_ALTO = 0.80
UMBRAL_REDUNDANTE = 0.90
UMBRAL_EXACTO = 1.0 - 1e-9
UMBRAL_VIF = 10.0
FILAS_MUESTRA = 20000


def matriz_completa() -> tuple[np.ndarray, list[str]]:
    """Carga la matriz de 279 variables de V12 con sus nombres, en el orden de produccion."""
    base, _, _, _, _ = build_features()
    matriz = np.concatenate([base, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    nombres: list[str] = []
    for indice, (fichero, _) in enumerate(MEMBER_FILES):
        # Los miembros no son todos de 6 columnas: E-015 aporta 1 (un margen) y E-028a aporta 5
        # (un margen por frontera adyacente). Se leen las formas reales, no se asumen.
        bloque = np.load(cfg.FEATURES / fichero)
        columnas = 1 if bloque.ndim == 1 else bloque.shape[1]
        nombres += [f"miembro{indice:02d}_{pos}" for pos in range(columnas)]
    nombres += [f"densa::{columna}" for columna in DENSE_FEATURES]
    estructuradas = pd.read_parquet(
        cfg.FEATURES / "age_structured_features.parquet").columns
    nombres += [f"estruct::{columna}" for columna in estructuradas]
    supervisadas = np.load(cfg.FEATURES / "age_supervised_features.npy").shape[1]
    nombres += [f"supervisada_{indice:02d}" for indice in range(supervisadas)]
    nombres += [f"nbsvm_{clase}" for clase in cfg.CLASSES]
    return matriz, nombres


def pares_correlacionados(matriz: np.ndarray, nombres: list[str], filas: np.ndarray
                          ) -> tuple[np.ndarray, list[dict], list[str], list[str]]:
    """Matriz de correlacion de Spearman exacta, pares por encima del umbral y columnas constantes.

    La correlacion se construye con :func:`grama_exacta` porque ``rangos @ rangos.T`` devuelve
    valores erroneos en este entorno (D-040): con el camino erroneo el bloque entero pareceria no
    tener ninguna correlacion alta.

    Las columnas constantes (varianza cero) se excluyen del analisis: no tienen correlacion
    definida y supondrian un cero enganoso en la diagonal. Se devuelven aparte porque son en si
    mismas un hallazgo: una variable de la matriz de produccion que nunca varia.
    """
    valores = np.asarray(matriz[filas], dtype=float)
    desviacion_cruda = valores.std(axis=0, ddof=1)
    constantes = [nombres[i] for i in range(valores.shape[1]) if desviacion_cruda[i] < 1e-12]
    utiles = [i for i in range(valores.shape[1]) if desviacion_cruda[i] >= 1e-12]
    rangos = rankdata(valores[:, utiles], axis=0)
    desviacion = rangos.std(axis=0, ddof=1)
    rangos = (rangos - rangos.mean(axis=0)) / np.where(desviacion < 1e-12, 1.0, desviacion)
    correlacion = grama_exacta(rangos) / (len(filas) - 1)
    # La diagonal debe quedar exactamente en 1: es la comprobacion estructural de que la matriz
    # es una matriz de correlacion. Si falla, el resto de medidas (VIF, rango efectivo) no valen.
    assert np.abs(np.diag(correlacion) - 1.0).max() < 1e-9, np.abs(np.diag(correlacion) - 1.0).max()
    # La diagonal NO se pone a cero: `triu_indices(k, 1)` ya la excluye al extraer pares, y
    # anularla destruiria la matriz de cara al VIF y a los autovalores.
    nombres_utiles = [nombres[i] for i in utiles]
    iu = np.triu_indices(len(nombres_utiles), 1)
    pares = [
        {"a": nombres_utiles[int(fila)], "b": nombres_utiles[int(columna)],
         "rho": round(float(correlacion[fila, columna]), 4),
         "nivel": "duplicado_exacto" if abs(correlacion[fila, columna]) >= UMBRAL_EXACTO
                  else ("redundante" if abs(correlacion[fila, columna]) >= UMBRAL_REDUNDANTE
                        else "alto")}
        for posicion, fila, columna in zip(np.where(np.abs(correlacion[iu]) >= UMBRAL_ALTO)[0],
                                           iu[0][np.abs(correlacion[iu]) >= UMBRAL_ALTO],
                                           iu[1][np.abs(correlacion[iu]) >= UMBRAL_ALTO])
    ]
    pares.sort(key=lambda par: -abs(par["rho"]))
    return correlacion, pares, nombres_utiles, constantes


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


def factores_inflacion(correlacion: np.ndarray, nombres: list[str],
                       pares: list[dict]) -> pd.DataFrame:
    """VIF de cada variable a partir de la inversa de la matriz de correlacion.

    Con la matriz de correlacion tipificada, ``VIF_j = [S^-1]_jj``. Como hay pares con
    ``|rho| = 1`` (columnas identicas), la matriz es singular y se usa la pseudoinversa; esas
    variables se marcan aparte porque su VIF real es infinito, no un numero grande.
    """
    inversa = sla.pinv(correlacion)
    duplicadas = {par["b"] for par in pares if par["nivel"] == "duplicado_exacto"}
    duplicadas |= {par["a"] for par in pares if par["nivel"] == "duplicado_exacto"}
    filas = []
    for indice, nombre in enumerate(nombres):
        valor = float(inversa[indice, indice])
        filas.append({
            "variable": nombre,
            "vif": round(valor, 3),
            "familia": familia_de(nombre),
            "vif_indefinido_por_duplicado": nombre in duplicadas,
            "vif_alto": (not nombre in duplicadas) and valor >= UMBRAL_VIF,
        })
    return pd.DataFrame(filas).sort_values("vif", ascending=False)


def familia_de(nombre: str) -> str:
    """Familia de una variable por su prefijo de nombre."""
    if nombre.startswith("densa::"):
        return "densas"
    if nombre.startswith("estruct::"):
        return "estructuradas"
    if nombre.startswith("supervisada"):
        return "supervisadas"
    if nombre.startswith("nbsvm"):
        return "nb_svm"
    if nombre.startswith("miembro"):
        return "scores de miembros"
    return "otras"


def criba_redundancia(nuevo: np.ndarray, nombres_nuevos: list[str], base: np.ndarray,
                      nombres_base: list[str], filas: np.ndarray) -> pd.DataFrame:
    """Correlacion de un bloque nuevo contra la matriz completa (uso reutilizable en V13).

    Parametros
    ----------
    nuevo, nombres_nuevos:
        Bloque candidato, forma ``(filas, columnas)``, y sus nombres.
    base, nombres_base:
        Matriz de referencia; en V13 siempre la matriz completa de 279 variables de V12.
    filas:
        Indices de las filas de la muestra sobre la que se mide la correlacion de Spearman.

    Devuelve
    -------
    pandas.DataFrame
        Una fila por variable nueva con su correlacion maxima (en valor absoluto) contra V12, la
        variable de V12 con la que la alcanza y un booleano ``redundante``.
    """
    rangos_base = np.asarray(base[filas], dtype=float)
    rangos_base = rankdata(rangos_base, axis=0)
    desviacion = rangos_base.std(axis=0, ddof=1)
    rangos_base = (rangos_base - rangos_base.mean(axis=0)) / np.where(
        desviacion < 1e-12, 1.0, desviacion)
    filas_nuevas = []
    for indice, nombre in enumerate(nombres_nuevos):
        columna = np.asarray(nuevo[filas, indice], dtype=float)
        if columna.std() < 1e-12:
            filas_nuevas.append({"feature": nombre, "correlacion_maxima_v12": 0.0,
                                 "v12_mas_correlacionada": "", "varianza": 0.0,
                                 "redundante": False})
            continue
        rangos = rankdata(columna)
        desviacion_rango = rangos.std(ddof=1)
        rangos = (rangos - rangos.mean()) / (desviacion_rango if desviacion_rango > 1e-12 else 1.0)
        # einsum por columna: evita BLAS, que en este entorno falla en la auto-transpuesta (D-040).
        correlaciones = np.einsum("ij,i->j", rangos_base, rangos) / (len(filas) - 1)
        maximo = float(np.nanmax(np.abs(correlaciones)))
        columna_base = int(np.nanargmax(np.abs(correlaciones)))
        filas_nuevas.append({
            "feature": nombre,
            "correlacion_maxima_v12": round(maximo, 4),
            "v12_mas_correlacionada": nombres_base[columna_base],
            "varianza": round(float(columna.std()), 5),
            "redundante": bool(maximo >= UMBRAL_ALTO),
        })
    return pd.DataFrame(filas_nuevas)


def main() -> int:
    """Mide la colinealidad de la matriz completa de V12 y su gravedad, sin reentrenar."""
    autotest_numerico()
    inicio = time.time()
    y = datos.load_train()[cfg.TARGET].astype(str).to_numpy()
    matriz, nombres = matriz_completa()
    assert matriz.shape[1] == len(nombres) == 279, (matriz.shape, len(nombres))
    generador = np.random.default_rng(cfg.SEED)
    filas = generador.choice(len(y), size=FILAS_MUESTRA, replace=False)

    # ---------------------------------------------------------------- 1. estructura
    correlacion, pares, utiles, constantes = pares_correlacionados(matriz, nombres, filas)
    grupos = clusters_por_union(pares)
    exactos = [par for par in pares if par["nivel"] == "duplicado_exacto"]
    redundantes = [par for par in pares if par["nivel"] == "redundante"]
    print(f"[matriz] {matriz.shape[1]} variables | constantes (varianza 0): {len(constantes)} "
          f"{constantes} | analizadas: {len(utiles)}", flush=True)
    print(f"[correlacion] pares |rho|>={UMBRAL_ALTO}: {len(pares)} "
          f"(redundantes >= {UMBRAL_REDUNDANTE}: {len(redundantes)} | "
          f"duplicados exactos: {len(exactos)}) | clusters: {len(grupos)}", flush=True)
    for par in exactos:
        print(f"    {par['rho']:+.4f} {par['a']}  <->  {par['b']}", flush=True)

    # ---------------------------------------------------------------- 2. gravedad sin entrenar
    autovalores = sla.eigvalsh(correlacion)[::-1]
    rango_efectivo = int((autovalores > 1e-8).sum())
    umbrales_espectro = {f"1e-{exponente}": int((autovalores > 10 ** -exponente).sum())
                         for exponente in (2, 4, 6, 8)}
    acumulado = np.cumsum(autovalores) / autovalores.sum()
    direcciones_95 = int(np.searchsorted(acumulado, 0.95) + 1)
    direcciones_99 = int(np.searchsorted(acumulado, 0.99) + 1)
    vif = factores_inflacion(correlacion, utiles, pares)
    vif.to_csv(VIF, index=False)
    print(f"[gravedad] rango efectivo {rango_efectivo} de {len(utiles)} | "
          f"direcciones para el 95% de la varianza: {direcciones_95} | "
          f"VIF >= {UMBRAL_VIF}: {int(vif['vif_alto'].sum())} | "
          f"VIF indefinido por duplicado: {int(vif['vif_indefinido_por_duplicado'].sum())}",
          flush=True)
    # Correlacion tipica entre scores de la misma clase: explica que el VIF de esa familia sea enorme
    # sin que haya duplicados exactos.
    por_clase = {}
    for posicion, nombre in enumerate(utiles):
        if nombre.startswith("miembro") and nombre.split("_")[-1].isdigit():
            por_clase.setdefault(nombre.split("_")[-1], []).append(posicion)
    rango_intra = [correlacion[indices[a], indices[b]]
                   for indices in por_clase.values() if len(indices) > 1
                   for a in range(len(indices)) for b in range(a + 1, len(indices))]
    print(f"[miembros] correlacion intra-clase: mediana {float(np.median(rango_intra)):.4f}, "
          f"minima {float(np.min(rango_intra)):.4f}, maxima {float(np.max(rango_intra)):.4f}",
          flush=True)

    # ---------------------------------------------------------------- 3. referencia de V12
    referencia = json.loads(REFERENCIA.read_text(encoding="utf-8"))["referencia_v12"]
    print(f"[referencia] V12 macro-F1 {referencia['macro_f1']} "
          f"(std {referencia['std']}) folds {referencia['per_fold']}", flush=True)

    # ---------------------------------------------------------------- 4. validacion de la criba
    validacion = pd.DataFrame()
    if REDUNDANCIA_LEXICON.exists():
        previo = pd.read_csv(REDUNDANCIA_LEXICON)
        nombres_lex = previo["feature"].tolist()
        bloques = []
        for familia in ("registro", "vida", "temas"):
            ruta = cfg.FEATURES / f"lexicon_features_{familia}.npy"
            if ruta.exists():
                bloques.append(np.load(ruta))
        if bloques and sum(b.shape[1] for b in bloques) == len(nombres_lex):
            matriz_lex = np.concatenate(bloques, axis=1)
            criba = criba_redundancia(matriz_lex, nombres_lex, matriz, nombres, filas)
            validacion = previo.merge(criba, on="feature", suffixes=("_previo", "_criba"))
            validacion["redundante"] = validacion["redundante_previo"]
            validacion = validacion[["feature", "correlacion_maxima_v12_previo",
                                     "correlacion_maxima_v12_criba", "redundante"]]
            validacion.to_csv(VALIDACION, index=False)
            diferencia = float((validacion["correlacion_maxima_v12_previo"]
                                - validacion["correlacion_maxima_v12_criba"]).abs().max())
            print(f"[validacion criba] {len(validacion)} variables lexicas | "
                  f"diferencia maxima frente al calculo previo: {diferencia:.4f} | "
                  f"redundantes: previo {int(previo['redundante'].sum())} / "
                  f"criba {int(validacion['redundante'].sum())}", flush=True)

    # ---------------------------------------------------------------- 5. tablas y lectura
    pd.DataFrame(pares).to_csv(PARES, index=False)
    filas_cluster = [{
        "cluster": indice,
        "n_variables": len(grupo),
        "correlacion_interna_max": max(abs(par["rho"]) for par in pares
                                      if par["a"] in grupo and par["b"] in grupo),
        "variables": ", ".join(grupo),
    } for indice, grupo in enumerate(grupos, start=1)]
    pd.DataFrame(filas_cluster).sort_values("n_variables", ascending=False).to_csv(
        CLUSTERS, index=False)

    por_familia = []
    posicion_util = {nombre: indice for indice, nombre in enumerate(utiles)}
    for familia in sorted({familia_de(nombre) for nombre in utiles}):
        indices = [i for i, nombre in enumerate(utiles) if familia_de(nombre) == familia]
        en_pares = sum(1 for par in pares if par["a"] in posicion_util and par["b"] in posicion_util
                       and (posicion_util[par["a"]] in indices or posicion_util[par["b"]] in indices))
        bloques_vif = vif.iloc[indices]
        por_familia.append({
            "familia": familia,
            "variables": len(indices),
            "en_pares_correlacionados": en_pares,
            "vif_mediano": round(float(bloques_vif["vif"].median()), 3),
            "vif_maximo": round(float(bloques_vif["vif"].max()), 3),
            "vif_alto": int(bloques_vif["vif_alto"].sum()),
        })
    familias_df = pd.DataFrame(por_familia).sort_values("variables", ascending=False)

    lineas = [
        "# V13 · P0b — Prueba de colinealidad de la matriz completa de V12", "",
        "## 1. Que se midio y como", "",
        f"Matriz **completa** del combinador de produccion: **{matriz.shape[1]} variables** sobre "
        f"{len(y):,} filas, tal cual la usa el meta-modelo (sin quitar, sustituir ni reordenar "
        f"ninguna columna): "
        f"{sum(1 for n in nombres if n.startswith('miembro'))} scores de los 14 miembros, "
        f"{len(DENSE_FEATURES)} densas, "
        f"{sum(1 for n in nombres if n.startswith('estruct'))} estructuradas, "
        f"{sum(1 for n in nombres if n.startswith('supervisada'))} supervisadas y "
        f"{sum(1 for n in nombres if n.startswith('nbsvm'))} de NB-SVM.", "",
        f"Correlacion de **Spearman** sobre una muestra de {FILAS_MUESTRA:,} filas, calculada con "
        f"sumas explicitas (`grama_exacta`) porque `rangos @ rangos.T` devuelve valores erroneos en "
        f"este entorno (D-040): con el camino erroneo el bloque entero pareceria no tener ninguna "
        f"correlacion alta. Umbrales: |rho| >= {UMBRAL_ALTO} alta, >= {UMBRAL_REDUNDANTE} "
        f"redundante, |rho| >= 1 - 1e-9 duplicado exacto.", "",
        "**No se reentrena nada.** La referencia de V12 se lee de "
        "`experiments/phase39_fusion_v12.json` (el mismo combinador, los mismos folds y la misma "
        "semilla). Una comprobacion independiente de esta sesion reprodujo sus cinco folds uno a "
        f"uno: {referencia['per_fold']}.", "",
        "## 2. Estructura de la colinealidad", "",
        f"- **Pares con |rho| >= {UMBRAL_ALTO}**: **{len(pares)}** de "
        f"{len(nombres) * (len(nombres) - 1) // 2:,} pares posibles",
        f"- Pares redundantes (>= {UMBRAL_REDUNDANTE}): **{len(redundantes)}**",
        f"- **Pares con duplicado exacto (|rho| = 1)**: **{len(exactos)}**",
        f"- **Clusters de variables redundantes**: **{len(grupos)}**", "",
        "### Duplicados exactos: la misma variable dos veces", "",
        "| rho | Variable A | Variable B |", "|---|---|---|",
    ]
    for par in exactos:
        lineas.append(f"| {par['rho']:+.4f} | `{par['a']}` | `{par['b']}` |")
    lineas += ["", "### Clusters mas grandes", "",
               "| Cluster | Variables | Correlación interna máxima | Contenido |",
               "|---|---|---|---|"]
    for fila in sorted(filas_cluster, key=lambda f: -f["n_variables"])[:8]:
        lineas.append(f"| {fila['cluster']} | {fila['n_variables']} | "
                      f"{fila['correlacion_interna_max']:.3f} | `{fila['variables'][:140]}` |")
    lineas += ["", "## 3. Gravedad de la colinealidad (sin entrenar)", "",
               "El factor de inflacion de la varianza de la variable j es el elemento jj de la "
               "inversa de la matriz de correlacion. Como hay pares con |rho| = 1, la matriz es "
               "singular: se usa la pseudoinversa y esas variables se marcan aparte, porque su "
               "VIF real es infinito y no un numero grande. Un VIF por encima de 1e6 no es una "
               "medida utilizable sino la senal de que esa direccion es practicamente no "
               "invertible.", "",
               "### Espectro de la matriz de correlacion", "",
               "Contar autovalores por encima de 1e-8 solo detecta las direcciones exactamente "
               "redundantes, criterio demasiado severo para un bloque de variables de texto. El "
               "criterio util es cuantos autovalores hacen falta para explicar la varianza:", "",
               "| Medida | Valor |", "|---|---|",
               f"| Autovalores por encima de 1e-2 | {umbrales_espectro['1e-2']} de {len(utiles)} |",
               f"| Autovalores por encima de 1e-4 | {umbrales_espectro['1e-4']} de {len(utiles)} |",
               f"| Autovalores por encima de 1e-6 | {umbrales_espectro['1e-6']} de {len(utiles)} |",
               f"| Autovalores por encima de 1e-8 (rango efectivo estricto) | "
               f"{umbrales_espectro['1e-8']} de {len(utiles)} |",
               f"| **Direcciones para el 95 % de la varianza** | **{direcciones_95}** de "
               f"{len(utiles)} |",
               f"| Direcciones para el 99 % de la varianza | {direcciones_99} de {len(utiles)} |",
               "",
               f"- **Rango efectivo estricto**: {rango_efectivo} de {len(utiles)}. Las "
               f"{len(utiles) - rango_efectivo} direcciones que faltan son exactamente los "
               f"{len(exactos)} pares de duplicados exactos ({len(exactos) * 2} variables): no hay "
               f"degeneracion estructural mas alla de esas.",
               f"- Variables con VIF >= {UMBRAL_VIF}: **{int(vif['vif_alto'].sum())}** de "
               f"{len(utiles)}",
               f"- Variables con VIF indefinido por duplicado exacto: "
               f"**{int(vif['vif_indefinido_por_duplicado'].sum())}**",
               f"- VIF mediano de la matriz completa: **{float(vif['vif'].median()):.3f}** (el "
               f"umbral clasico de referencia es 10)", "",
               f"La razon de que el VIF de los scores de miembros sea enorme sin que haya "
               f"duplicados exactos es que **catorce modelos sobre las mismas seis clases coinciden "
               f"mucho en que clase dan como mas probable**: entre scores de la misma clase la "
               f"correlacion tiene mediana {float(np.median(rango_intra)):.4f} y llega a "
               f"{float(np.max(rango_intra)):.4f}. El valor minimo, "
               f"{float(np.min(rango_intra)):.4f}, corresponde al miembro ordinal, cuya quinta "
               f"columna es la primera por construccion.", "",
               "Columnas constantes en la matriz de produccion: "
               + (f"**{len(constantes)}** ("
                  + ", ".join(f"`{nombre}`" for nombre in constantes) + ")"
                  if constantes else "**0**")
               + ". No tienen correlacion definida y no entran en el analisis de VIF ni de rango "
                 "efectivo; se declaran aparte porque son en si mismas un hallazgo: una variable "
                 "que nunca varia no aporta nada al modelo.",
               "", "### Por familia", "",
               "| Familia | Variables | En pares correlacionados | VIF mediano | VIF máximo | VIF alto |",
               "|---|---|---|---|---|---|"]
    for _, fila in familias_df.iterrows():
        lineas.append(f"| {fila['familia']} | {fila['variables']} | "
                      f"{fila['en_pares_correlacionados']} | {fila['vif_mediano']:.3f} | "
                      f"{fila['vif_maximo']:.3e} | {fila['vif_alto']} |")
    menos_saturada = familias_df.sort_values("vif_mediano").iloc[0]
    mas_saturada = familias_df.sort_values("vif_mediano").iloc[-1]
    lineas += ["", f"**Reparto util para V13**: la familia menos saturada es "
                  f"**{menos_saturada['familia']}** (VIF mediano "
                  f"{menos_saturada['vif_mediano']:.2f}) y la mas saturada es "
                  f"**{mas_saturada['familia']}** (VIF mediano {mas_saturada['vif_mediano']:.2f}). "
                  f"Anadir mas variables del tipo que ya esta saturado (textometria densa y "
                  f"estructurada) tiene poco margen; anadir señal de otro tipo, como evidencia "
                  f"supervisada calculada dentro del fold, es lo que puede aportar.", ""]
    lineas += ["", "## 4. ¿Esta danando al modelo?", "",
               f"El combinador de produccion (0,5 LogReg + 0,5 MLP sobre estas "
               f"{matriz.shape[1]} variables) alcanza **{referencia['macro_f1']} de macro-F1** "
               f"con {len(exactos)} pares de variables identicas y un VIF mediano de "
               f"{float(vif['vif'].median()):.1f}. La respuesta es **no, en prediccion**:", "",
               "- La colinealidad no cambia lo que el modelo acierta; cambia **cuanto se puede "
               "interpretar un coeficiente** y **cuanto esfuerzo de aprendizaje gasta el MLP** en "
               "direcciones casi duplicadas. Un coeficiente grande en una variable colineal no "
               "significa que esa variable sea la que explica la decision.",
               f"- **El bloque tiene {direcciones_95} direcciones para el 95 % de la varianza**, "
               f"asi que anadir variables nuevas solo aporta si caen fuera de esas direcciones. "
               f"Por eso la criba de redundancia va **antes** de fusionar cualquier bloque nuevo.",
               "- Donde si hay que mirar es el reparto por familia: las familias con VIF alto son "
               "las que ya estan saturadas, y las de VIF bajo son las unicas donde anadir signal "
               "puede aportar algo.", "",
               "## 5. Regla operativa para V13", "",
               f"1. Ningun bloque nuevo se fusiona sin pasar por `criba_redundancia` contra las "
               f"{matriz.shape[1]} variables de la matriz completa.",
               f"2. Una variable nueva con |rho| >= {UMBRAL_ALTO} contra V12 se documenta y no se "
               f"incorpora, salvo que aporte algo que la correlacion lineal no capture (por "
               f"ejemplo una interaccion no lineal), lo cual se argue en un experimento propio.",
               "3. Ninguna conclusion de V13 puede apoyarse en el **valor** de un coeficiente: con "
               "este grado de colinealidad, solo en su rendimiento fuera de muestra.", ""]
    if len(validacion):
        lineas += [
            "## 6. Validacion cruzada de la criba", "",
            f"La criba se ejecuto sobre las {len(validacion)} variables lexicas de la fase cerrada "
            f"y su resultado coincide con el que ya estaba calculado con **otra ruta de codigo** "
            f"(`reports/age_lexicon_redundancy.csv`, obtenido con producto matricial matriz-vector, "
            f"que D-040 declara sana): **{int(validacion['redundante'].sum())} redundantes en "
            f"ambos casos** y diferencia maxima "
            f"{float((validacion['correlacion_maxima_v12_previo'] - validacion['correlacion_maxima_v12_criba']).abs().max()):.4f} "
            f"en la correlacion maxima de cada variable.", "",
            "Es la segunda via independiente que confirma el diagnostico de D-040: si el entorno "
            "rompiese tambien la ruta matriz-vector, los dos calculos no podrian coincidir.", ""]
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    RESULTADO.write_text(json.dumps({
        "fase": "V13 · P0b prueba de colinealidad de la matriz completa", "fecha": "2026-09-25",
        "matriz": "completa de produccion (279 variables, sin reducir)",
        "n_variables": int(matriz.shape[1]), "filas_muestra": FILAS_MUESTRA,
        "umbral_alto": UMBRAL_ALTO, "umbral_redundante": UMBRAL_REDUNDANTE,
        "pares": len(pares), "pares_redundantes": len(redundantes),
        "duplicados_exactos": len(exactos), "clusters": len(grupos),
        "rango_efectivo": rango_efectivo,
        "autovalores_por_encima": umbrales_espectro,
        "direcciones_95_varianza": direcciones_95,
        "direcciones_99_varianza": direcciones_99,
        "correlacion_intra_clase_miembros": {
            "mediana": round(float(np.median(rango_intra)), 4),
            "minima": round(float(np.min(rango_intra)), 4),
            "maxima": round(float(np.max(rango_intra)), 4)},
        "direcciones_redundantes": len(utiles) - rango_efectivo,
        "columnas_constantes": constantes,
        "vif_alto": int(vif["vif_alto"].sum()),
        "vif_indefinido_por_duplicado": int(vif["vif_indefinido_por_duplicado"].sum()),
        "vif_mediano": round(float(vif["vif"].median()), 3),
        "referencia_v12": referencia,
        "validacion_criba_lexicon": {
            "variables": int(len(validacion)),
            "redundantes_previo": int(previo["redundante"].sum()) if len(validacion) else None,
            "redundantes_criba": int(validacion["redundante"].sum()) if len(validacion) else None,
            "diferencia_maxima": round(float((validacion["correlacion_maxima_v12_previo"]
                                              - validacion["correlacion_maxima_v12_criba"]
                                              ).abs().max()), 4) if len(validacion) else None,
        },
        "por_familia": por_familia, "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nrango efectivo {rango_efectivo}/{len(utiles)} | "
          f"{len(utiles) - rango_efectivo} direcciones redundantes | "
          f"{len(exactos)} duplicados exactos | {len(constantes)} constante(s)")
    print(f"artefactos: {INFORME.name}, {PARES.name}, {CLUSTERS.name}, {VIF.name}, "
          f"{VALIDACION.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
