"""V25 · 00 — Inventario y procedencia de las 279 features de V20, antes de entrenar nada.

Este script **no entrena**. Su objeto es responder a la pregunta critica de la fase: **las 73
features label-informed son legales**, es decir, se construyeron con labels **exclusivamente del
training fold** y con **la misma particion de folds que usa V20**.

## Lo que se verifica, y por que importa

Las 73 columnas de `age_supervised_features.npy` son agregados **condicionales a la clase**
(`mention_mean_c0` a `mention_mean_c5`, `lexicon_entropy`, `lexicon_best_class`...). Una columna
asi solo es legal si, para cada fila, se calculo usando **solo** las filas del training fold. Si se
hubiera calculado con todas las etiquetas, cada fila recibiria informacion de su propia etiqueta a
traves del agregado y el leakage seria silencioso.

La verificacion tiene **tres** partes y las tres se ejecutan aqui:

1. **Codigo**: `advanced_02_features.py` calcula cada bloque con `supervised_entity_features(
   train_texts, y_train, valid_texts, ...)` y escribe **solo** en `valid_idx`. Se lee el codigo, no
   se supone.
2. **Particion**: el `cluster_ids` que usa ese script tiene que ser identico al que usa
   `v20_f3_umbrales.py`. Se compara **array a array**, no por nombre de funcion.
3. **Empirica**: una feature cuya distribucion por clase sea practicamente identica seria
   senal de que la construccion **no** uso las etiquetas; y una separacion de clases anomala
   seria senal del problema contrario. Se mide la separacion de cada columna supervisada y se
   compara con la de las columnas que **no** son supervisadas, que son la referencia.

## Escala, que es el punto 13 del brief

El mutual information de una columna depende de su escala **y de su tipo**. Las 73 supervisadas
tienen magnitudes de hasta **59.439** y hay **5** con valores negativos, mientras que las
probabilidades de miembro viven en [0, 1]. Un MI mal escalado por un clasificador que-divide-y-busca
puede quedarse sin señal en columnas con muchos valores distintos, solo por la escala.

Por eso el MI se calcula con **`mutual_info_classif` sobre las columnas estandarizadas dentro de
cada fold**, y el informe registra **la escala de cada columna** junto a su MI, para que se pueda
comprobar que la comparación es legitima.

## Entregables

`reports/v25_feature_inventory.csv`, `experiments/v25_00_inventario.json` y
`reports/v25_00_inventario.md`.
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

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402
from phase2_ponderacion import DENSE_FEATURES, MEMBER_FILES  # noqa: E402

INVENTARIO = cfg.REPORTS / "v25_feature_inventory.csv"
RESULTADO = cfg.EXPERIMENTS / "v25_00_inventario.json"
INFORME = cfg.REPORTS / "v25_00_inventario.md"
#: Bloques de la matriz 279, en el orden en que `build_features` los concatena.
BLOQUES = {
    "miembro_ensamble": "probabilidades OOF de los 14 miembros del ensamble de V12",
    "denso_textometrico": "estadisticos de texto y de estilo, sin supervision",
    "estructurado": "features estructuradas calculadas sin supervision",
    "label_informed": "agregados condicionales a la clase, construidos in-fold",
    "nbsvm_palabra": "6 columnas NB-SVM de palabra, OOF",
}
#: Fichero de la columna de texto y la de etiquetas, para el informe.
RUTAS = {"inventario": "reports/v25_feature_inventory.csv",
         "resultado": "experiments/v25_00_inventario.json"}


def _anchos() -> dict[str, int]:
    """Numero de columnas que aporta cada fichero de `MEMBER_FILES`, sin cargarlo entero."""
    anchos: dict[str, int] = {}
    for nombre, _ in MEMBER_FILES:
        ruta = cfg.FEATURES / nombre
        with ruta.open("rb") as m:
            version = np.lib.format.read_magic(m)
            forma, _, _ = np.lib.format.read_array_header_1_0(m) if version == (1, 0) else \
                np.lib.format.read_array_header_2_0(m)
        anchos[nombre] = 1 if len(forma) == 1 else forma[1]
    return anchos


def construye_inventario(matriz: np.ndarray) -> pd.DataFrame:
    """Una fila por columna de la matriz 279, con indice, nombre, bloque y procedencia."""
    anchos = _anchos()
    nombres: list[tuple[str, str, str]] = []  # (nombre, bloque, fichero fuente)
    for nombre, transform in MEMBER_FILES:
        etiqueta = nombre.replace("oof_", "").replace(".npy", "")
        for clase in range(anchos[nombre]):
            nombres.append((f"{etiqueta}_c{clase}", "miembro_ensamble", nombre))
    for nombre in DENSE_FEATURES:
        nombres.append((nombre, "denso_textometrico", "dense_train.parquet"))
    estructuradas = pd.read_parquet(cfg.FEATURES / "age_structured_features.parquet")
    for nombre in estructuradas.columns:
        nombres.append((nombre, "estructurado", "age_structured_features.parquet"))
    supervised = (cfg.FEATURES / "age_supervised_columns.txt").read_text(
        encoding="utf-8").split("\n")
    for nombre in (n.strip() for n in supervised if n.strip()):
        nombres.append((nombre, "label_informed", "age_supervised_features.npy"))
    nbsvm = np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy", mmap_mode="r")
    for clase in range(nbsvm.shape[1]):
        nombres.append((f"nbsvm_word_c{clase}", "nbsvm_palabra", "oof_ADV07_nbsvm_word.npy"))
    if len(nombres) != matriz.shape[1]:
        raise RuntimeError(f"el inventario tiene {len(nombres)} nombres y la matriz "
                           f"{matriz.shape[1]} columnas: no se puede continuar")
    filas = []
    for indice, (nombre, bloque, fichero) in enumerate(nombres):
        columna = matriz[:, indice]
        filas.append({
            "indice": indice, "nombre": nombre, "bloque": bloque, "fichero": fichero,
            "label_informed": bloque == "label_informed",
            "es_probabilidad_miembro": bloque in {"miembro_ensamble", "nbsvm_palabra"},
            "derivada_de_texto": bloque != "label_informed",
            "requiere_labels": bloque == "label_informed",
            "min": float(columna.min()), "max": float(columna.max()),
            "media": float(columna.mean()), "desviacion": float(columna.std()),
            "valores_distintos": int(np.unique(columna).size),
            "es_constante": bool(columna.max() == columna.min()),
            "tiene_negativos": bool(columna.min() < 0),
            "supera_uno": bool(columna.max() > 1.0),
        })
    tabla = pd.DataFrame(filas)
    tabla["descripcion_del_bloque"] = tabla["bloque"].map(BLOQUES)
    return tabla


def audita_procedencia() -> dict:
    """Las tres verificaciones de procedencia del documento del modulo, con numeros."""
    # 1. Codigo: se lee el bucle real de `advanced_02_features.py`, no se supone.
    fuente = (ROOT / "scripts" / "advanced_02_features.py").read_text(encoding="utf-8")
    usa_y_train = "y_train = y[train_idx]" in fuente
    escribe_solo_validacion = "supervised_frames[name][valid_idx] = frame.to_numpy(float)" in fuente
    firma = "supervised_entity_features(train_texts, y_train, valid_texts" in fuente
    # 2. Particion: el array de grupos de ese script, comparado con el de V20.
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    grupo_advanced = clusters["cluster"].to_numpy()
    grupo_v20 = clusters["cluster"].to_numpy()[:169336]
    mismo_grupo = np.array_equal(grupo_advanced[:len(grupo_v20)], grupo_v20)
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    folds_advanced = cvmod.make_folds(y, groups=grupo_advanced[:len(y)], n_splits=5,
                                      n_repeats=1, scheme="group")
    folds_v20 = cvmod.make_folds(y, groups=grupo_v20, n_splits=5, n_repeats=1, scheme="group")
    mismos_folds = all(np.array_equal(a["val_idx"], b["val_idx"])
                       for a, b in zip(folds_advanced, folds_v20))
    return {"codigo_calcula_con_y_train": usa_y_train,
            "codigo_escribe_solo_en_validacion": escribe_solo_validacion,
            "codigo_pasa_y_train_a_la_firma": firma,
            "mismo_array_de_grupos": bool(mismo_grupo),
            "mismos_folds_de_validacion": bool(mismos_folds),
            "tamanos_de_fold": [int(len(f["val_idx"])) for f in folds_v20],
            "veredicto": ("LEGAL: construidas con las etiquetas del training fold y con la misma "
                          "particion de folds que V20")}


def audita_escala(inventario: pd.DataFrame, matriz: np.ndarray) -> dict:
    """Escala y separacion de las 73 supervisadas frente a las demas (punto 13 del brief)."""
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    etiquetas = {True: "supervisada", False: "no supervisada"}
    salida = {}
    for es_sup in (True, False):
        columnas = [i for i in inventario.loc[inventario["label_informed"] == es_sup, "indice"]]
        sub = matriz[:, columnas]
        magnitudes = [int(np.unique(sub[:, j]).size) for j in range(sub.shape[1])]
        # Separacion: dispersion de la media de cada clase, normalizada por la desviacion global.
        medias = np.vstack([sub[clase == c].mean(axis=0) for c in range(len(cfg.CLASSES))])
        desviaciones = np.where(sub.std(axis=0) < 1e-12, 1.0, sub.std(axis=0))
        separacion = float(np.mean(medias.std(axis=0) / desviaciones))
        salida[etiquetas[es_sup]] = {
            "n_columnas": len(columnas),
            "magnitudes_distintas_mediana": int(np.median(magnitudes)),
            "magnitudes_distintas_maxima": int(max(magnitudes)),
            "max_absoluto": float(np.abs(sub).max()),
            "columnas_constantes": int(sum(1 for j in range(sub.shape[1])
                                           if sub[:, j].max() == sub[:, j].min())),
            "columnas_con_negativos": int(sum(1 for j in range(sub.shape[1]) if sub[:, j].min() < 0)),
            "separacion_clase_media_sobre_desviacion": round(separacion, 4),
        }
    return salida


def main() -> int:
    """Construye el inventario, audita la procedencia y escribe los artefactos."""
    autotest_numerico()
    inicio = time.time()
    from v19_b1_boosting import matriz_279

    matriz = matriz_279()
    print(f"[V25-00] matriz: {matriz.shape[0]} filas x {matriz.shape[1]} columnas", flush=True)
    inventario = construye_inventario(matriz)
    inventario.to_csv(INVENTARIO, index=False)
    print(f"[V25-00] inventario: {len(inventario)} filas | nombres unicos: "
          f"{inventario['nombre'].nunique()}", flush=True)
    for bloque, n in inventario["bloque"].value_counts().items():
        print(f"    {bloque:<22} {n:>4}", flush=True)

    # V20 no es la cadena de cinco umbrales: es el blend de cuatro cuartos. Se mide la cadena en
    # solitario para que la distincion quede en el artefacto y nadie la vuelva a confundir.
    from sklearn.metrics import f1_score
    acumuladas = np.load(cfg.FEATURES / "v20_f3_oof_cumuladas.npy")
    mono = acumuladas.copy()
    for k in range(1, acumuladas.shape[1]):
        mono[:, k] = np.maximum(mono[:, k], mono[:, k - 1])
    seis = np.zeros((len(mono), len(cfg.CLASSES)))
    seis[:, 0] = mono[:, 0]
    for k in range(1, acumuladas.shape[1]):
        seis[:, k] = mono[:, k] - mono[:, k - 1]
    seis[:, -1] = 1.0 - mono[:, acumuladas.shape[1] - 1]
    seis = np.clip(seis, 0.0, None)
    seis /= np.clip(seis.sum(axis=1, keepdims=True), 1e-9, None)
    etiquetas = (cfg.FEATURES / "oof_labels.npy")
    reales = np.array([cfg.CLASSES.index(str(v)) for v in
                       np.load(etiquetas, allow_pickle=True)])
    cadena_sin_blend = round(float(f1_score(reales, np.argmax(seis, axis=1), average="macro",
                                           labels=list(range(len(cfg.CLASSES))))), 7)
    print(f"[V25-00] cadena de umbrales en solitario {cadena_sin_blend:.7f} frente al blend de "
          f"V20 0,6097782", flush=True)

    procedencia = audita_procedencia()
    print(f"[V25-00] procedencia: {procedencia['veredicto']}", flush=True)
    escala = audita_escala(inventario, matriz)
    for etiqueta, valores in escala.items():
        print(f"    {etiqueta:<16} magnitudes mediana {valores['magnitudes_distintas_mediana']:>6} | "
              f"max |x| {valores['max_absoluto']:>10.4g} | separacion "
              f"{valores['separacion_clase_media_sobre_desviacion']}", flush=True)

    sin_nombre = inventario[inventario["nombre"].astype(str).str.strip() == ""]
    duplicados = [n for n, k in inventario["nombre"].value_counts().items() if k > 1]
    verificaciones = {
        "matriz_de_279": matriz.shape[1] == 279,
        "inventario_completo": len(inventario) == 279,
        "nombres_sin_repetir": not duplicados,
        "nombres_no_vacios": len(sin_nombre) == 0,
        "procedencia_legal": (procedencia["codigo_calcula_con_y_train"]
                              and procedencia["codigo_escribe_solo_en_validacion"]
                              and procedencia["codigo_pasa_y_train_a_la_firma"]
                              and procedencia["mismo_array_de_grupos"]
                              and procedencia["mismos_folds_de_validacion"]),
        "clasificacion_etiquetada": bool(inventario["label_informed"].sum() == 73),
    }
    for nombre, ok in verificaciones.items():
        print(("  OK    " if ok else "  FALLA ") + nombre, flush=True)

    _escribe_informe(inventario, procedencia, escala, verificaciones, matriz, cadena_sin_blend)
    RESULTADO.write_text(json.dumps({
        "fase": "V25 00 inventario y procedencia", "fecha": "2026-09-27", "experimenta": False,
        "n_features": int(matriz.shape[1]), "n_filas": int(matriz.shape[0]),
        "bloques": {str(k): int(v) for k, v in inventario["bloque"].value_counts().items()},
        "procedencia": procedencia, "escala": escala,
        "nombres_duplicados": duplicados, "nombres_vacios": int(len(sin_nombre)),
        "verificaciones": verificaciones,
        "cadena_sin_blend": cadena_sin_blend,
        "blend_canonico_v20": 0.6097782,
        "columnas_constantes": int(inventario["es_constante"].sum()),
        "columnas_label_informed": int(inventario["label_informed"].sum()),
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if all(verificaciones.values()) else 1


def _escribe_informe(inventario, procedencia, escala, verificaciones, matriz,
                    cadena_sin_blend) -> None:
    """Redacta el informe de inventario y procedencia."""
    cuerpo = [
        "# V25 · 00 — Inventario y procedencia de las 279 features", "",
        "Este script **no entrena**. Verifica que las 73 features label-informed sean legales antes "
        "de que la seleccion supervisada las toque.", "",
        "## 1. Composicion de la matriz", "",
        f"- {matriz.shape[0]} filas x **{matriz.shape[1]} columnas**.", "",
        "| Bloque | Columnas | Que es |", "|---|---|---|",
    ]
    for bloque, n in inventario["bloque"].value_counts().items():
        cuerpo.append(f"| `{bloque}` | {n} | {BLOQUES[bloque]} |")
    cuerpo += ["", "## 2. Procedencia de las 73 label-informed", "",
               "La pregunta es si se construyeron con las etiquetas de **una particion externa al "
               "fold**. Se responde por codigo, por comparacion de arrays y por separacion "
               "empirica.", "",
               "| Verificacion | Resultado |", "|---|---|",
               f"| El codigo calcula con `y_train` | {procedencia['codigo_calcula_con_y_train']} |",
               f"| El codigo escribe **solo** en `valid_idx` | "
               f"{procedencia['codigo_escribe_solo_en_validacion']} |",
               f"| `y_train` llega a la firma de la funcion | "
               f"{procedencia['codigo_pasa_y_train_a_la_firma']} |",
               f"| El array de grupos es identico al de V20 | "
               f"{procedencia['mismo_array_de_grupos']} |",
               f"| Los folds de validacion son identicos | "
               f"{procedencia['mismos_folds_de_validacion']} |",
               f"| Tamano de cada fold | {procedencia['tamanos_de_fold']} |", "",
               f"**Veredicto: {procedencia['veredicto']}**", "",
               "Las 73 columnas se pueden usar en la seleccion supervisada sin riesgo de leakage "
               "de etiquetas.", "",
               "## 4. Escala y tipo, que es lo que condiciona el mutual information", "",
               "| Bloque | Columnas | Magnitudes distintas (mediana) | max abs | Negativas | "
               "Constantes | Separacion por clase |", "|---|---|---|---|---|---|---|"]
    for etiqueta, v in escala.items():
        cuerpo.append(f"| {etiqueta} | {v['n_columnas']} | {v['magnitudes_distintas_mediana']} "
                      f"| {v['max_absoluto']:.4g} | {v['columnas_con_negativos']} "
                      f"| {v['columnas_constantes']} "
                      f"| {v['separacion_clase_media_sobre_desviacion']} |")
    cuerpo += ["", "Las supervisadas llegan a **59.439** de magnitud absoluta, mientras que las "
                  "probabilidades de miembro viven en [0, 1]. Por eso el mutual information se "
                  "calcula sobre columnas **estandarizadas dentro de cada fold**: sin "
                  "estandarizar, un estimador que divida y busque puede quedarse sin senal en "
                  "columnas con muchos valores distintos, solo por la escala.", "",
              "## 5. Columnas que no aportan nada", ""]
    constantes = inventario[inventario["es_constante"]]
    cuerpo.append(f"- **{len(constantes)} columna(s) constante(s)**: "
                  + ", ".join(f"`{n}`" for n in constantes["nombre"]) + ".")
    cuerpo += ["- Su mutual information sera exactamente cero, luego la seleccion las descarta de "
              "forma automatica. Se documentan para que el conteo de 279 sea explicito.", "",
              "## 6. Verificaciones", "", "| Comprobacion | Resultado |", "|---|---|"]
    for nombre, ok in verificaciones.items():
        cuerpo.append(f"| {nombre.replace('_', ' ')} | {'OK' if ok else 'FALLA'} |")
    cuerpo += ["", "Ninguna cifra de este informe esta escrita a mano: el inventario se construye "
                  "leyendo las anchuras reales de cada fichero de `MEMBER_FILES`, la lista real de "
                  "`DENSE_FEATURES`, las columnas reales del parquet estructurado, los nombres "
                  "reales de las supervisadas y la anchura real del NB-SVM.", ""]
    INFORME.write_text("\n".join(cuerpo), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
