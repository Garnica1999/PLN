"""V15 · A — Correlación entre variables de entrada y correlación con la salida, medidas in-fold.

Petición que motiva la fase: evaluar si hay variables de entrada redundantes y cuáles se asocian a
la salida, para eliminar las redundantes. La pregunta tiene una trampa estadística y la fase está diseñada
para evitarla: **elegir variables por su correlación con la etiqueta, calculada sobre todo el train,
es una forma de fuga de información**, aunque no se toque el test. Si se mide la asociación sobre las
mismas filas con las que luego se evalúa, el criterio de selección ya ha visto la validación.

Por eso aquí la asociación y la correlación se calculan **dentro de cada fold de entrenamiento** y se
agregan sobre los cinco folds. Cada fold ve solo sus filas de train, y el número que se publica es el
promedio de cinco mediciones independientes. La estructura de clusters se calcula igual, sobre las filas
de train de cada fold, y se compara entre folds para medir su estabilidad: una partición de variables
que cambia al cambiar de fold no es una partición que se pueda usar.

**Tres medidas, con propósitos distintos.**

1. **Correlación entrada-entrada** (rho de Spearman): identifica redundancia. Se agrupan en clusters
   por union-find con umbral |rho| >= 0,90 y se mide el tamaño de los clusters, los duplicados
   exactos y la estabilidad de la partición entre folds.
2. **Asociación entrada-salida**: la salida es categórica con seis clases, así que una correlación
   de Pearson no es la herramienta adecuada. Se usa la **razón de correlación eta cuadrado**, que es
   la variación de la variable explicada por la clase, en [0, 1]. Se reporta por variable y en
   promedio sobre folds, y en dos versiones: en **escala bruta** y en **ranking dentro del fold**,
   porque el orden relativo es lo que importa para elegir un representante de cluster y el valor
   absoluto depende de la escala.
3. **Asociación dentro de cada cluster**: para cada cluster, qué variable es la más asociada a la
   salida. Es el criterio de elección del representante, y por eso se mide antes que nada.

El script **no poda todavía**: solo deja el diagnóstico y, sobre todo, la partición in-fold que
V15b usará para construir el candidato podado sin fuga.

Entregables: ``reports/v15_correlacion_entradas.csv``, ``reports/v15_clusters.csv``,
``reports/v15_asociacion_salida.csv``, ``reports/v15_estabilidad.csv``,
``reports/v15_correlacion.md``, ``features/v15_clusters_fold{0..4}.npy`` (índice de representante por
cluster y fold), ``experiments/v15a_correlacion.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from phase2_ponderacion import DENSE_FEATURES, MEMBER_FILES, build_features  # noqa: E402

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico, grama_exacta  # noqa: E402

CORRELACION = cfg.REPORTS / "v15_correlacion_entradas.csv"
CLUSTERS = cfg.REPORTS / "v15_clusters.csv"
ASOCIACION = cfg.REPORTS / "v15_asociacion_salida.csv"
ESTABILIDAD = cfg.REPORTS / "v15_estabilidad.csv"
INFORME = cfg.REPORTS / "v15_correlacion.md"
RESULTADO = cfg.EXPERIMENTS / "v15a_correlacion.json"

UMBRAL_CLUSTER = 0.90
FILAS_MUESTRA = 20000


def nombres_completos() -> list[str]:
    """Nombres de las 279 variables de V12, en el orden exacto de ``build_features``."""
    nombres: list[str] = []
    for indice, (fichero, _) in enumerate(MEMBER_FILES):
        bloque = np.load(cfg.FEATURES / fichero)
        columnas = 1 if bloque.ndim == 1 else bloque.shape[1]
        nombres += [f"v12_miembro{indice:02d}_{pos}" for pos in range(columnas)]
    nombres += [f"v12_densa::{columna}" for columna in DENSE_FEATURES]
    estructuradas = pd.read_parquet(
        cfg.FEATURES / "age_structured_features.parquet").columns
    nombres += [f"v12_estruct::{columna}" for columna in estructuradas]
    supervisadas = np.load(cfg.FEATURES / "age_supervised_features.npy").shape[1]
    nombres += [f"v12_supervisada_{indice:02d}" for indice in range(supervisadas)]
    nombres += [f"v12_nbsvm_{clase}" for clase in cfg.CLASSES]
    return nombres


def matriz_completa() -> tuple[np.ndarray, list[str]]:
    """Las 279 variables de V12 más las 33 variables lingüísticas de V14, con sus nombres."""
    base, _, _, _, _ = build_features()
    matriz = np.concatenate([base, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    nombres = nombres_completos()
    assert len(nombres) == matriz.shape[1] == 279, (len(nombres), matriz.shape)
    linguisticas = json.loads((cfg.FEATURES / "v14_linguistico_nombres.json").read_text(
        encoding="utf-8"))
    retiradas = ("longitud_media_palabra", "lema_distintos_por_token", "lema_hapax_ratio",
                 "tokens_por_frase")
    bloque = np.load(cfg.FEATURES / "v14_linguistico.npy").astype(np.float64)
    columnas = [i for i, nombre in enumerate(linguisticas) if nombre not in retiradas]
    return (np.hstack([matriz, bloque[:, columnas]]),
            nombres + [f"v14_{linguisticas[i]}" for i in columnas])


def razon_correlacion(matriz: np.ndarray, clase: np.ndarray) -> np.ndarray:
    """Eta cuadrado de cada variable con la clase, para una salida categórica.

    ``eta^2 = variacion entre grupos / variacion total``, con las medias de grupo por la etiqueta
    codificada. Es la generalization de ``r^2`` a una salida de clases y vale entre 0 y 1. Se
    calcula en la escala bruta de cada variable, que es lo que hace falta para comparar columnas de
    naturaleza distinta.
    """
    total = matriz.var(axis=0)
    explicada = np.zeros(matriz.shape[1])
    for etiqueta in np.unique(clase):
        filas = matriz[clase == etiqueta]
        if len(filas):
            explicada += len(filas) * (filas.mean(axis=0) - matriz.mean(axis=0)) ** 2
    explicada = explicada / max(1, len(matriz))
    salida = np.zeros(matriz.shape[1])
    con_varianza = total > 1e-12
    salida[con_varianza] = explicada[con_varianza] / total[con_varianza]
    return salida


def clusters_por_union(pares: list[tuple[int, int]]) -> list[list[int]]:
    """Agrupa en clusters las variables unidas por correlación alta (union-find)."""
    padre: dict[int, int] = {}

    def encuentra(indice: int) -> int:
        padre.setdefault(indice, indice)
        while padre[indice] != indice:
            padre[indice] = padre[padre[indice]]
            indice = padre[indice]
        return indice

    def une(a: int, b: int) -> None:
        raiz_a, raiz_b = encuentra(a), encuentra(b)
        if raiz_a != raiz_b:
            padre[raiz_a] = raiz_b

    for a, b in pares:
        une(a, b)
    grupos: dict[int, list[int]] = {}
    for a, b in pares:
        grupos.setdefault(encuentra(a), []).append(a)
        grupos.setdefault(encuentra(b), []).append(b)
    return [sorted(set(miembros)) for miembros in grupos.values() if len(miembros) > 1]


def main() -> int:
    """Mide la correlación entre entradas y su asociación con la salida, dentro de cada fold."""
    autotest_numerico()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])
    clusters_ids = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters_ids, n_splits=5, n_repeats=1, scheme="group")

    matriz, nombres_original = matriz_completa()
    nombres = nombres_original
    print(f"[matriz] {matriz.shape[1]} variables ({len([n for n in nombres if n.startswith('v12_')])} "
          f"de V12 y {len([n for n in nombres if n.startswith('v14_')])} lingüísticas de V14)",
          flush=True)

    # ------------------------------------------------------- 1. correlación entrada-entrada
    generador = np.random.default_rng(cfg.SEED)
    filas = generador.choice(len(y), size=FILAS_MUESTRA, replace=False)
    rangos = np.asarray(matriz[filas], dtype=float)
    desviacion_cruda = rangos.std(axis=0, ddof=1)
    # Las columnas constantes no tienen correlacion definida: su diagonal no puede ser 1 y
    # estarian sonando como si fueran independientes de todo. Se declaran aparte y se podan por
    # definicion, que es lo que corresponde a una variable que nunca varia.
    constantes = [i for i in range(matriz.shape[1]) if desviacion_cruda[i] < 1e-12]
    analizables = [i for i in range(matriz.shape[1]) if desviacion_cruda[i] >= 1e-12]
    print(f"[matriz] columnas constantes: {len(constantes)} "
          f"{[nombres[i] for i in constantes]}", flush=True)
    rangos = rangos[:, analizables]
    rangos = pd.DataFrame(rangos).rank(axis=0).to_numpy()
    desviacion = rangos.std(axis=0, ddof=1)
    rangos = (rangos - rangos.mean(axis=0)) / np.where(desviacion < 1e-12, 1.0, desviacion)
    correlacion = grama_exacta(rangos) / (len(filas) - 1)
    assert np.abs(np.diag(correlacion) - 1.0).max() < 1e-9, np.abs(np.diag(correlacion) - 1.0).max()
    nombres = [nombres[i] for i in analizables]
    iu = np.triu_indices(len(nombres), 1)
    valores = correlacion[iu]
    for umbral, etiqueta in ((0.80, "alto"), (UMBRAL_CLUSTER, "redundante"),
                             (1 - 1e-9, "duplicado")):
        print(f"[correlacion] pares |rho| >= {umbral}: {int((np.abs(valores) >= umbral).sum())}"
              f" ({etiqueta})", flush=True)
    pares_alto = [(int(a), int(b)) for a, b, v in zip(iu[0], iu[1], valores)
                  if abs(v) >= UMBRAL_CLUSTER]
    grupos = clusters_por_union(pares_alto)
    print_sizes = sorted((len(g) for g in grupos), reverse=True)
    print(f"[clusters] {len(grupos)} clusters con |rho| >= {UMBRAL_CLUSTER} | tamaños máximos "
          f"{print_sizes[:8]} | variables en clusters "
          f"{sum(len(g) for g in grupos)} de {len(nombres)}", flush=True)

    filas_correlacion = []
    for a, b, valor in zip(iu[0], iu[1], valores):
        if abs(valor) < 0.80:
            continue
        filas_correlacion.append({"a": nombres[a], "b": nombres[b],
                                  "rho": round(float(valor), 4),
                                  "nivel": "duplicado_exacto" if abs(valor) >= 1 - 1e-9
                                  else ("redundante" if abs(valor) >= UMBRAL_CLUSTER
                                        else "alto")})
    tabla_correlacion = pd.DataFrame(filas_correlacion).sort_values("rho", key=lambda s: -s.abs())
    tabla_correlacion.to_csv(CORRELACION, index=False)

    # ------------------------------------------------------- 2. asociación con la salida
    eta_global = razon_correlacion(matriz, clase)
    orden_global = np.argsort(-eta_global)
    print(f"[asociacion] eta^2: mediana {float(np.median(eta_global)):.4f} | max "
          f"{float(eta_global.max()):.4f} ({nombres[int(orden_global[0])]}) | "
          f"variables con eta^2 < 0,01: {int((eta_global < 0.01).sum())}", flush=True)

    # in-fold: la seleccion no puede usar las filas de validacion
    eta_folds = np.zeros((len(folds), matriz.shape[1]))
    clusters_fold: list[list[list[int]]] = []
    pares_fold: list[set[tuple[int, int]]] = []
    for fold in folds:
        entrenamiento = np.concatenate([otro["val_idx"] for otro in folds
                                        if otro["fold"] != fold["fold"]])
        eta_folds[fold["fold"]] = razon_correlacion(matriz[entrenamiento], clase[entrenamiento])
        rangos_fold = pd.DataFrame(np.asarray(matriz[entrenamiento], dtype=float)).rank(
            axis=0).to_numpy()
        desviacion_fold = rangos_fold.std(axis=0, ddof=1)
        rangos_fold = (rangos_fold - rangos_fold.mean(axis=0)) / np.where(
            desviacion_fold < 1e-12, 1.0, desviacion_fold)
        correlacion_fold = grama_exacta(rangos_fold) / (len(entrenamiento) - 1)
        iu_fold = np.triu_indices(len(nombres), 1)
        valores_fold = correlacion_fold[iu_fold]
        pares_fold_local = [(int(a), int(b)) for a, b, v in zip(iu_fold[0], iu_fold[1], valores_fold)
                                if abs(v) >= UMBRAL_CLUSTER]
        clusters_fold.append(clusters_por_union(pares_fold_local))
        pares_fold.append({(min(a, b), max(a, b)) for a, b in pares_fold_local})
        print(f"[fold {fold['fold']}] {len(clusters_fold[-1])} clusters in-fold | "
              f"asociacion mediana {float(np.median(eta_folds[fold['fold']])):.4f}", flush=True)

    eta_infold = eta_folds.mean(axis=0)
    filas_asociacion = [{
        "variable": nombres[i],
        "es_v12": nombres[i].startswith("v12_"),
        "eta2_global": round(float(eta_global[i]), 5),
        "eta2_infold_medio": round(float(eta_infold[i]), 5),
        "desviacion_entre_folds": round(float(eta_folds[:, i].std()), 5),
        "ranking_infold": int((np.argsort(-eta_infold) == i).argmax()) + 1,
    } for i in range(len(nombres))]
    tabla_asociacion = pd.DataFrame(filas_asociacion).sort_values("eta2_infold_medio",
                                                                 ascending=False)
    tabla_asociacion.to_csv(ASOCIACION, index=False)

    # ------------------------------------------------------- 3. estabilidad y representantes
    filas_clusters, filas_estabilidad = [], []
    for indice, grupo in enumerate(grupos, start=1):
        miembros = [nombres[i] for i in grupo]
        # El representante es el de mayor asociacion **in-fold**: entre variables redundantes
        # se conserva la que mas informa de la salida, no la primera que aparece.
        representante = grupo[int(np.argmax(eta_infold[grupo]))]
        filas_clusters.append({
            "cluster": indice, "n_variables": len(grupo),
            "correlacion_interna_max": max(
                abs(float(correlacion[a, b])) for a in grupo for b in grupo if a != b),
            "representante": nombres[representante],
            "eta2_del_representante": round(float(eta_infold[representante]), 5),
            "eta2_maximo_del_cluster": round(float(max(eta_infold[i] for i in grupo)), 5),
            "variables": ", ".join(miembros),
        })
    pd.DataFrame(filas_clusters).sort_values("n_variables", ascending=False).to_csv(
        CLUSTERS, index=False)

    # Estabilidad de la **estructura de pares**: es la medida que responde a si la redundancia se
    # reproduce al cambiar de fold. Comparar la igualdad exacta de los miembros de cada cluster es
    # un criterio tan estricto que da 0 practicamente siempre, porque union-find encadena las
    # relaciones de transitive y un solo par que cambie reordena el cluster entero.
    jaccard_pares = []
    for a in range(len(pares_fold)):
        for b in range(a + 1, len(pares_fold)):
            interseccion = pares_fold[a] & pares_fold[b]
            union = pares_fold[a] | pares_fold[b]
            jaccard_pares.append(len(interseccion) / max(1, len(union)))
    estabilidad_pares = float(np.mean(jaccard_pares)) if jaccard_pares else 0.0
    print(f"[estabilidad de pares] jaccard medio entre folds: {estabilidad_pares:.4f} | "
          f"minimo {float(np.min(jaccard_pares)):.4f}", flush=True)

    for indice, grupo in enumerate(grupos, start=1):
        coeficientes = []
        for lista in clusters_fold:
            for otro in lista:
                comun = set(grupo) & set(otro)
                if len(comun) == len(grupo) == len(otro):
                    coeficientes.append(1.0)
                elif len(comun) == 0:
                    coeficientes.append(0.0)
                else:
                    coeficientes.append(len(comun) / len(set(grupo) | set(otro)))
        filas_estabilidad.append({"cluster": indice, "n_variables": len(grupo),
                                  "jaccard_medio_entre_folds": round(float(np.mean(coeficientes)), 4),
                                  "jaccard_minimo": round(float(np.min(coeficientes)), 4),
                                  "aparece_completo_en_los_5_folds":
                                      bool(np.mean(coeficientes) == 1.0)})
    tabla_estabilidad = pd.DataFrame(filas_estabilidad)
    tabla_estabilidad.to_csv(ESTABILIDAD, index=False)
    estables = int(tabla_estabilidad["aparece_completo_en_los_5_folds"].sum())
    print(f"[estabilidad] {estables} de {len(tabla_estabilidad)} clusters aparecen completos en los "
          f"5 folds | jaccard medio {float(tabla_estabilidad['jaccard_medio_entre_folds'].mean()):.4f}",
          flush=True)

    # Indice de variables conservadas por fold, que es lo que usara V15b. Todo el criterio es
    # in-fold: la pertenencia al cluster se calcula con las filas de train de ese fold y el
    # representante es el de mayor eta2 de ese fold, de modo que ninguna decision de poda ve
    # las filas de validacion.
    for fold in folds:
        numero = fold["fold"]
        clusters_del_fold = clusters_fold[numero]
        en_cluster = {indice for grupo in clusters_del_fold for indice in grupo}
        representantes = {grupo[int(np.argmax([eta_folds[numero][i] for i in grupo]))]
                        for grupo in clusters_del_fold}
        conservadas = [i for i in range(len(nombres))
                       if i not in en_cluster or i in representantes]
        np.save(cfg.FEATURES / f"v15_conservadas_fold{numero}.npy",
                np.asarray(conservadas, dtype=np.int64))
    print(f"[guardado] indice de variables conservadas por fold en "
          f"features/v15_conservadas_fold*.npy", flush=True)

    lineas = [
        "# V15 · A — Correlación entre entradas y asociación con la salida", "",
        "## 1. Qué se mide y por qué dentro de cada fold", "",
        f"Matriz completa: **{len(nombres)} variables** (279 de V12 y "
        f"{len([n for n in nombres if n.startswith('v14_')])} lingüísticas de V14).",
        "",
        "Elegir variables por su correlación con la etiqueta calculada sobre todo el train es una "
        "forma de fuga aunque no se toque el test: el criterio de selección ya habría visto la "
        "validación. Por eso la estructura de clusters y la asociación se calculan **dentro de cada "
        "fold de entrenamiento** y se agregan. Además se mide la **estabilidad** de los clusters entre "
        "folds: una partición que cambia al cambiar de fold no sirve para podar.", "",
        "## 2. Correlación entre variables de entrada", "",
        f"- Pares con |rho| >= 0,80: **{int((np.abs(valores) >= 0.80).sum())}**",
        f"- Pares con |rho| >= {UMBRAL_CLUSTER}: **{len(pares_alto)}**",
        f"- Pares con duplicado exacto (|rho| = 1): "
        f"**{int((np.abs(valores) >= 1 - 1e-9).sum())}**",
        f"- Clusters con |rho| >= {UMBRAL_CLUSTER}: **{len(grupos)}**, que cubren "
        f"**{sum(len(g) for g in grupos)}** variables de {len(nombres)}", "",
        "### Los diez pares más correlacionados", "",
        "| rho | Nivel | Variable A | Variable B |", "|---|---|---|---|",
    ]
    for _, fila in tabla_correlacion.head(10).iterrows():
        lineas.append(f"| {fila['rho']:+.4f} | {fila['nivel']} | `{fila['a']}` | `{fila['b']}` |")
    lineas += ["", "## 3. Asociación con la salida (eta cuadrado)", "",
               "La salida es categórica con seis clases, así que se usa **eta cuadrado** (variación "
               "explicada por la clase) en lugar de correlación de Pearson.", "",
               f"- Mediana de eta²: **{float(np.median(eta_global)):.4f}**",
               f"- Máximo: **{float(eta_global.max()):.4f}** (`{nombres[int(orden_global[0])]}`)",
               f"- Variables con eta² < 0,01: **{int((eta_global < 0.01).sum())}** de {len(nombres)}",
               f"- Discrepancia entre la asociación global y la in-fold: mediana "
               f"{float(np.median(np.abs(eta_global - eta_infold))):.6f}", "",
               "### Las diez variables más asociadas a la salida", "",
               "| Variable | eta² in-fold | eta² global | Desviación entre folds |",
               "|---|---|---|---|"]
    for _, fila in tabla_asociacion.head(10).iterrows():
        lineas.append(f"| `{fila['variable']}` | {fila['eta2_infold_medio']} | "
                      f"{fila['eta2_global']} | {fila['desviacion_entre_folds']} |")
    lineas += ["", "### Las diez menos asociadas", "",
               "| Variable | eta² in-fold | eta² global |", "|---|---|---|"]
    for _, fila in tabla_asociacion.tail(10).iloc[::-1].iterrows():
        lineas.append(f"| `{fila['variable']}` | {fila['eta2_infold_medio']} | {fila['eta2_global']} |")
    lineas += ["", "## 4. Estabilidad de los clusters", "",
               f"- Clusters que aparecen **completos en los cinco folds**: **{estables}** de "
               f"{len(tabla_estabilidad)}",
               f"- Jaccard medio de los **miembros** de cluster entre folds: "
               f"**{float(tabla_estabilidad['jaccard_medio_entre_folds'].mean()):.4f}** (criterio muy "
               f"estricto: exige que el cluster tenga exactamente los mismos miembros, y un solo par "
               f"que cambie reordena el cluster entero por la transitividad de union-find)",
               f"- Jaccard medio de la **estructura de pares** entre folds: "
               f"**{estabilidad_pares:.4f}** (mínimo {float(np.min(jaccard_pares)):.4f}). Esta es la "
               f"medida que responde a si la redundancia se reproduce al cambiar de fold.", "",
               "### Los clusters más grandes y su representante", "",
               "| Cluster | Variables | Correlación interna máxima | Representante | eta² del "
               "representante |", "|---|---|---|---|---|"]
    for _, fila in pd.DataFrame(filas_clusters).sort_values("n_variables", ascending=False
                                                           ).head(8).iterrows():
        lineas.append(f"| {fila['cluster']} | {fila['n_variables']} | "
                      f"{fila['correlacion_interna_max']:.3f} | `{fila['representante']}` | "
                      f"{fila['eta2_del_representante']} |")
    lineas += ["", "## 5. Lo que hace V15b con esto", "",
               "El criterio de poda será: **una variable por cluster**, y se queda la de mayor eta² "
               "**dentro del fold**, no la primera del cluster. Las variables que no están en ningún "
               "cluster se conservan todas. Con eso el número de variables pasa de "
               f"{len(nombres)} a {len(nombres) - sum(len(g) for g in grupos) - len(grupos)} y la "
               "decisión se evalúa contra V12 con el gate D-042, incluida la segunda partición.", ""]
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    RESULTADO.write_text(json.dumps({
        "fase": "V15 · A correlacion y asociacion in-fold", "fecha": "2026-09-26",
        "n_variables": len(analizables), "nombres": nombres,
        "columnas_constantes": [nombres_original[i] for i in constantes],
        "filas_muestra": FILAS_MUESTRA,
        "umbral_cluster": UMBRAL_CLUSTER,
        "pares_0_80": int((np.abs(valores) >= 0.80).sum()),
        "pares_redundantes": len(pares_alto),
        "duplicados_exactos": int((np.abs(valores) >= 1 - 1e-9).sum()),
        "clusters": len(grupos),
        "variables_en_clusters": int(sum(len(g) for g in grupos)),
        "tamanos_maximos": print_sizes[:10],
        "eta2_mediana": round(float(np.median(eta_global)), 5),
        "eta2_maxima": round(float(eta_global.max()), 5),
        "variables_eta2_bajo_0_01": int((eta_global < 0.01).sum()),
        "clusters_estables": estables,
        "jaccard_medio_miembros": round(float(tabla_estabilidad["jaccard_medio_entre_folds"].mean()), 4),
        "jaccard_medio_pares": round(estabilidad_pares, 4),
        "jaccard_minimo_pares": round(float(np.min(jaccard_pares)), 4),
        "variables_tras_poda": int(len(nombres) - sum(len(g) for g in grupos) - len(grupos)),
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nartefactos: {INFORME.name}, {CORRELACION.name}, {CLUSTERS.name}, {ASOCIACION.name}, "
          f"{ESTABILIDAD.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
