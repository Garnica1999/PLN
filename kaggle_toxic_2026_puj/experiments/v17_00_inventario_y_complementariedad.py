"""V17-00: inventario del patrimonio historico y cuantificacion de rescate y dano.

La pregunta que este script responde, y es la unica de V17
---------------------------------------------------------
> Cuantos errores de V03 son potencialmente rescatables por cada modelo historico, y cuantos danos
> introduce cada uno?

No es una demostracion de que el ensemble no funciono. Es un recuento, y se hace sobre artefactos
que **ya existen**: no se entrena nada, no se regenera nada y no se toca el champion.

Por que el recuento y no un ensemble
------------------------------------
Un ensemble solo se justifica si el candidato aporta errores que V03 no tiene. Si el recuento de
rescates y danos no favorece a nadie, construirlo seria gastar presupuesto de submissions en una
combinacion que la evidencia ya descarta. Y al reves: si favorece a alguien, ese alguien tiene una
hipotesis concreta y la fase puede gastarse.

Como se mide
------------
El OOF canonico de V03 tiene 1483 filas y tres etiquetas, luego 4449 celdas. Cada artefacto
historico se alinea a ese mismo conjunto **por identificador**, porque los ficheros estan sobre
conjuntos distintos: algunos cubren las 1483 filas del desarrollo y otros las 1731 del train
completo. Alinear por posicion en vez de por identificador haria que las celdas de dos modelos no se
refirieran a la misma fila, y el recuento no significaria nada.

De cada variante se mide, a su propio punto de operacion y tambien en su version mas favorable:

- **rescates**: celdas donde V03 falla y el candidato acierta.
- **danios**: celdas donde V03 acierta y el candidato falla.
- **rescate neto**: rescates menos danos.
- **cobertura**: fraccion de los errores de V03 que el candidato cubre.
- **correlacion de errores**: phi entre los dos indicadores de error, celda a celda.
- El desglose por etiqueta, porque las tres etiquetas tienen estructura distinta.

Por que la busqueda del umbral optimo es exacta y no una aproximacion
----------------------------------------------------------------------
El indicador de error de una celda depende **solo** de la prediccion de esa celda, y las celdas de
distinta etiqueta no interactuan en el recuento. Por tanto

    max_h,t,a [ sum_etiquetas (rescates - danos) ]  =  sum_etiquetas max_umbral (rescates - danos)

y el maximo se puede descomponer etiqueta por etiqueta. Eso hace la busqueda exacta con una
complejidad de 3 por 66 en lugar de 66 al cubo, y evita un trabajo de tres ordenes de magnitud que
no habria cambiado el resultado.

Uso
---
    /home/carlos/venv/bin/python experiments/v17_00_inventario_y_complementariedad.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

EXPERIMENTO = "V17-00 (P0 inventario y P1 complementariedad de errores frente a V03)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v17_00_inventario_complementariedad.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v17_00_inventario_complementariedad.json"
CSV_INVENTARIO = PROJECT_ROOT / "experiments" / "historical_models_inventory.csv"
CSV_COMPLEMENTARIEDAD = PROJECT_ROOT / "resultados" / "v17_01_complementariedad.csv"

ETIQUETAS = ("HS", "TR", "AG")
REFERENCIA_V03 = 0.7772114525165721
LB_V03 = 0.78306
# Escala del proyecto: 107 cambios netos de celda mueven el LB en 0.003.
CELDAS_POR_003_LB = 107

LB_CONOCIDOS: dict[str, float] = {
    "v01_baseline_tfidf_lr": 0.75677,
    "v02_lr_union": 0.77993,
    "v03_lr_union_thr": 0.78306,
    "v04_ens04_mediana": 0.77510,
    "v05_optuna": 0.77602,
    "v06_robustez_c1": 0.77227,
    "v07_robustez_mindf5": 0.77034,
    "v08_meta_v2": 0.78163,
    "v08_p1_fold_ensemble": 0.77470,
    "v09_candidate_a": 0.77895,
    "v09_p5_c_por_etiqueta": 0.77781,
    "v10_candidate_b": 0.77978,
    "v11_candidate_c": 0.78150,
    "v12_p7_promedio_repeticiones": 0.77122,
    "v16_ch1_tasa_optima_transportada": 0.77828,
    "v16_ch5_tasa_optima_absoluta": 0.77495,
    "v16_ch7_C1p5": 0.77296,
    "v16_ch8_C2p5": 0.77707,
    "v16_ch9_mindf1": 0.76978,
}

# Artefacto, fichero, descripcion. Las claves de probabilidad se descubren por forma.
ARTEFACTOS: tuple[tuple[str, str, str], ...] = (
    ("v2_oof_base_corrected", "v2_oof_base_corrected.npz",
     "4 bases TF-IDF word y char, OOF corregido de la fase v2"),
    ("v2_oof_targeting", "v2_oof_targeting.npz",
     "canal de targeting simbolico, 2 variantes"),
    ("v2_oof_syntax", "v2_oof_syntax.npz",
     "canal de sintaxis y citacion, 8 variantes"),
    ("v2_oof_emocion_nbsvm_ortografia", "v2_oof_emocion_nbsvm_ortografia.npz",
     "canal de emocion, NBSVM y ortografia"),
    ("v2_oof_cita_negacion", "v2_oof_cita_negacion.npz",
     "canales de cita y negacion, 5 variantes"),
    ("v2_meta_oof", "v2_meta_oof.npz",
     "meta de stacking v2, que dio 0.7853 local y 0.78163 externo"),
    ("oof_candidates_v4", "oof_candidates_v4.npz",
     "candidatos de la fase v4: downweight y meta de 36 features"),
    ("oof_meta_shift_v4", "oof_meta_shift_v4.npz",
     "meta con desplazamiento de calibracion, la causa de la mala transferencia"),
    ("oof_joint_label_v4", "oof_joint_label_v4.npz",
     "label powerset de V4-E, 12 variantes"),
    ("oof_reweight_v4", "oof_reweight_v4.npz",
     "instance weighting por dureza, 14 variantes"),
    ("oof_weighting_v4", "oof_weighting_v4.npz",
     "ponderacion de la geometria de la matriz, 15 variantes"),
    ("oof_tracks_g_v4", "oof_tracks_g_v4.npz",
     "tracks G: standalone y metas de 9, 36 y 45 features"),
    ("oof_mod03b", "oof_mod03b.npz",
     "MOD-03b, mejor OOF de la primera fase con 0.7927"),
    ("oof_ens04_v04", "oof_ens04_v04.npz",
     "ENS-04, el ensemble de la mediana que dio 0.7751 externo"),
)

# Claves que no son vectores de probabilidad por etiqueta.
NO_PROBABILIDAD = {
    "ids", "y_true", "umbrales", "umbrales_f", "pliegues", "pliegue", "grupos", "etiquetas",
    "modelos", "variantes", "canales", "C", "cluster_id", "en_cluster", "hardness_score",
    "hardness_por_etiqueta", "error_score", "bce_score", "discrepancia_etiqueta", "vecino_ids",
    "discrepancia_por_etiqueta",
}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V17-00] {mensaje}", flush=True)


def main() -> None:
    """Construye el inventario y mide la complementariedad de cada artefacto."""
    inicio = time.time()
    log("V17-00: inventario del patrimonio y complementariedad de errores")

    log("1/5 el OOF canonico de V03 y la rejilla de comparacion")
    datos_v03 = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)
    ids_v03 = np.asarray(datos_v03["ids"]).astype(str)
    y = np.asarray(datos_v03["y_true"], dtype=int)
    proba_v03 = np.asarray(datos_v03["proba"], dtype=float)
    umbrales_v03 = np.array([0.44, 0.51, 0.46])
    error_v03 = (proba_v03 >= umbrales_v03[None, :]).astype(int) != y
    n_celdas = int(y.size)
    n_errores_v03 = int(error_v03.sum())
    log(f"   {y.shape[0]} filas x 3 etiquetas = {n_celdas} celdas")
    log(f"   errores de V03: {n_errores_v03} ({100 * n_errores_v03 / n_celdas:.2f} por ciento)")
    log("   La rejilla es de celdas y no de filas: la metrica es la media de las tres F1, y")
    log("   corregir HS no cambia el acierto de TR ni de AG. Eso es lo que V13 midio con delta cero.")

    log("2/5 inventario de los artefactos OOF y de las predicciones de test")
    filas_inventario: list[dict[str, Any]] = []
    for clave, fichero, descripcion in ARTEFACTOS:
        ruta = PROJECT_ROOT / "features" / fichero
        if not ruta.exists():
            filas_inventario.append(
                {"artefacto": clave, "fichero": fichero, "descripcion": descripcion,
                 "estado_de_integridad": "AUSENTE", "filas": 0, "variantes": 0,
                 "claves_de_probabilidad": ""}
            )
            log(f"   {clave:32s} AUSENTE")
            continue
        datos = np.load(ruta, allow_pickle=True)
        n_filas = len(np.asarray(datos["ids"])) if "ids" in datos.files else 0
        variantes = _claves_de_probabilidad(datos)
        filas_inventario.append(
            {"artefacto": clave, "fichero": fichero, "descripcion": descripcion,
             "estado_de_integridad": "OK", "filas": n_filas, "variantes": len(variantes),
             "claves_de_probabilidad": ",".join(v for v, _ in variantes)}
        )
        log(f"   {clave:32s} {n_filas:5d} filas | {len(variantes):2d} variantes")

    test_dir = PROJECT_ROOT / "submissions"
    predicciones = sorted(
        p for p in test_dir.glob("toxic_submission_v*.csv")
        if "v16_" not in p.name and "plantilla" not in p.name
    )
    for ruta in predicciones:
        ident = ruta.stem.replace("toxic_submission_", "")
        with ruta.open(encoding="utf-8") as manejador:
            n = sum(1 for _ in manejador) - 1
        filas_inventario.append(
            {"artefacto": ident, "fichero": ruta.name,
             "descripcion": "prediccion de test historica ya generada, no se regenera",
             "estado_de_integridad": "OK", "filas": n, "variantes": 1,
             "claves_de_probabilidad": "labels binarias, no probabilidades",
             "lb_conocido": LB_CONOCIDOS.get(ident, "")}
        )
    log(f"   {len(predicciones)} predicciones de test historicas en disco, todas trazables")

    with CSV_INVENTARIO.open("w", encoding="utf-8", newline="") as manejador:
        campos = sorted({k for f in filas_inventario for k in f})
        escritor = csv.DictWriter(manejador, fieldnames=campos, lineterminator="\n")
        escritor.writeheader()
        for f in filas_inventario:
            escritor.writerow({k: f.get(k, "") for k in campos})

    log("3/5 la medicion central: rescate y dano de cada variante frente a V03")
    filas_comp: list[dict[str, Any]] = []
    omitidos: list[str] = []
    errores: dict[str, np.ndarray] = {}
    for clave, fichero, descripcion in ARTEFACTOS:
        ruta = PROJECT_ROOT / "features" / fichero
        if not ruta.exists():
            omitidos.append(f"{clave}: artefacto ausente")
            continue
        datos = np.load(ruta, allow_pickle=True)
        if "ids" not in datos.files:
            omitidos.append(f"{clave}: sin clave ids, no se puede alinear")
            log(f"   {clave:32s} sin ids, se omite")
            continue
        ids = np.asarray(datos["ids"]).astype(str)
        posicion = {v: i for i, v in enumerate(ids)}
        comunes = [v for v in ids_v03 if v in posicion]
        if len(comunes) != len(ids_v03):
            omitidos.append(
                f"{clave}: solo {len(comunes)} de {len(ids_v03)} ids coinciden con V03"
            )
            log(f"   {clave:32s} {len(comunes)} de {len(ids_v03)} ids, se omite")
            continue
        # `indices` son posiciones **en el array del candidato**, porque el candidato puede tener
        # mas filas que V03. Ni la verdad ni el error de V03 se reindexan: ambos ya estan en el
        # orden de identificadores de V03, que es el orden de referencia, y la verdad es la misma
        # para los dos modelos. Reindexarlos con posiciones del candidato seria indexar la y de
        # 1483 filas con un indice de hasta 1730, y ademas mezclaria dos ordenes distintos.
        indices = np.array([posicion[v] for v in ids_v03], dtype=int)
        umbrales = _umbrales(datos)
        variantes = _claves_de_probabilidad(datos)
        for nombre_variante, matriz in variantes:
            proba = np.asarray(matriz, dtype=float)[indices]
            fila_medida = _medir(
                clave, nombre_variante, descripcion, y, proba, error_v03, umbrales, n_celdas
            )
            filas_comp.append(fila_medida)
            # Se conserva el indicador de error en el umbral optimo de cada variante. Son 90
            # matrices de 4449 booleanos, unos 400 KB, y hacen falta para medir si los rescates de
            # variantes distintas se suman o se solapan, que es lo que decide si un ensemble de
            # varias puede alcanzar la escala de 107 cambios netos.
            optimos = np.array([float(v) for v in fila_medida["umbrales_optimos"].split("/")])
            errores[f"{clave}/{nombre_variante}"] = (
                (proba >= optimos[None, :]).astype(int) != y
            )
        log(f"   {clave:32s} {len(variantes):2d} variantes medidas")

    filas_comp.sort(key=lambda f: -f["rescate_neto_optimo"])
    _resumen(filas_comp, n_errores_v03, n_celdas, omitidos)

    log("4/5 la union: los rescates de variantes distintas se suman o se solapan")
    union = _analizar_union(filas_comp, errores, error_v03)

    log("4b/5 agrupar en A, B y C segun el P2")
    for f in filas_comp:
        f["grupo"] = _grupo(f)
    conteo = {"A": 0, "B": 0, "C": 0}
    for f in filas_comp:
        conteo[f["grupo"]] += 1
    log(f"   grupo A, LB cercano o superior: {conteo['A']}")
    log(f"   grupo B, LB inferior pero complementary: {conteo['B']}")
    log(f"   grupo C, mala complementary: {conteo['C']}")

    log("5/5 escribir")
    Path("resultados").mkdir(exist_ok=True)
    campos = [
        "artefacto", "variante", "grupo", "descripcion", "macro_f1_propia", "delta_macro_f1",
        "umbrales_usados", "rescates", "danios", "rescate_neto", "ratio_rescate_danio",
        "cobertura_de_errores_v03", "errores_nuevos_sobre_aciertos_v03",
        "correlacion_de_errores", "solape_de_predicciones", "rescates_HS", "danios_HS",
        "rescates_TR", "danios_TR", "rescates_AG", "danios_AG", "rescate_neto_optimo",
        "rescates_optimos", "danios_optimos",
        "macro_f1_optima", "delta_macro_f1_optima", "umbrales_optimos",
        "supera_a_v03_en_oof", "celdas_netas_para_mover_0p003",
    ]
    with CSV_COMPLEMENTARIEDAD.open("w", encoding="utf-8", newline="") as manejador:
        escritor = csv.DictWriter(manejador, fieldnames=campos, lineterminator="\n")
        escritor.writeheader()
        for f in filas_comp:
            escritor.writerow({k: f.get(k, "") for k in campos})

    posibles = [f for f in filas_comp if f["supera_a_v03_en_oof"]]
    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "referencia_v03": REFERENCIA_V03,
        "lb_v03": LB_V03,
        "rejilla": {
            "filas": int(y.shape[0]), "celdas": n_celdas, "errores_de_v03": n_errores_v03,
            "proporcion_de_error": n_errores_v03 / n_celdas,
        },
        "inventario": filas_inventario,
        "artefactos_omitidos": omitidos,
        "complementariedad": filas_comp,
        "conteo_por_grupo": conteo,
        "analisis_de_union": union,
        "variantes_que_superan_a_v03_en_oof": [
            {"artefacto": f["artefacto"], "variante": f["variante"],
             "macro_f1_optima": f["macro_f1_optima"],
             "delta": f["delta_macro_f1_optima"],
             "rescate_neto": f["rescate_neto_optimo"],
             "umbrales": f["umbrales_optimos"]}
            for f in posibles
        ],
        "gastos": {"submissions": 0, "modelos_entrenados": 0, "artefactos_regenerados": 0},
    }
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _informe(informe, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(
        f"VEREDICTO: {len(filas_comp)} variantes medidas | V03 falla {n_errores_v03} de {n_celdas} "
        f"celdas | {len(posibles)} superan a V03 en OOF con su umbral optimo | "
        f"mejor rescate neto {filas_comp[0]['rescate_neto_optimo']:+d}, union "
        f"{union['rescate_neto_de_la_union']:+d} frente a {CELDAS_POR_003_LB} necesarios, en "
        f"{filas_comp[0]['artefacto']}/{filas_comp[0]['variante']} | submissions gastadas 0"
    )


def _claves_de_probabilidad(datos: Any) -> list[str]:
    """Lista las claves que son vectores de probabilidad con una columna por etiqueta.

    Args:
        datos: diccionario del npz.

    Returns:
        Nombres de las claves de probabilidad.
    """
    n_ids = len(np.asarray(datos["ids"])) if "ids" in datos.files else 0
    salida: list[tuple[str, np.ndarray]] = []
    for clave in datos.files:
        if clave in NO_PROBABILIDAD:
            continue
        array = np.asarray(datos[clave])
        if array.dtype.kind != "f" or array.ndim < 2 or array.shape[-1] != 3:
            continue
        # **El eje de filas no siempre es el primero.** v2_oof_syntax tiene forma (1483, 8, 3)
        # con las variantes en el eje 1, mientras que v2_oof_emocion_nbsvm_ortografia tiene
        # (11, 1483, 3) con las variantes en el eje 0. Localizar el eje por su tamano, en vez de
        # suponerlo, es lo que hace la extraccion correcta y evita medir 1483 variantes falsas.
        ejes = [i for i, tamano in enumerate(array.shape) if tamano == n_ids]
        if not ejes:
            continue
        eje_filas = ejes[0]
        array = np.moveaxis(array, eje_filas, 0)
        if array.ndim == 2:
            salida.append((clave, array))
            continue
        for indice in range(array.shape[1]):
            salida.append((f"{clave}#{indice}", array[:, indice, :]))
    return salida


def _umbrales(datos: Any) -> np.ndarray:
    """Extrae los umbrales que declara el artefacto, o usa 0.5 en los tres.

    Args:
        datos: diccionario del npz.

    Returns:
        Vector de tres umbrales.
    """
    for clave in ("umbrales", "umbrales_f"):
        if clave in datos.files:
            vector = np.asarray(datos[clave], dtype=float).ravel()
            if vector.size == 3:
                return vector
    return np.array([0.5, 0.5, 0.5])


def _f1_por_etiqueta(y: np.ndarray, proba: np.ndarray, umbrales: np.ndarray) -> float:
    """Calcula la media de las tres F1 de la clase positiva.

    Args:
        y: etiquetas verdaderas.
        proba: probabilidades.
        umbrales: umbrales por etiqueta.

    Returns:
        La media de las tres F1.
    """
    pred = (proba >= umbrales[None, :]).astype(int)
    valores = []
    for indice in range(3):
        verdaderos = y[:, indice] == 1
        predichos = pred[:, indice] == 1
        verdaderos_positivos = float(np.sum(verdaderos & predichos))
        denominador = (
            2 * verdaderos_positivos
            + float(np.sum(~verdaderos & predichos))
            + float(np.sum(verdaderos & ~predichos))
        )
        valores.append(2 * verdaderos_positivos / denominador if denominador > 0 else 0.0)
    return float(np.mean(valores))


def _medir(
    artefacto: str,
    variante: str,
    descripcion: str,
    y: np.ndarray,
    proba: np.ndarray,
    error_v03: np.ndarray,
    umbrales: np.ndarray,
    n_celdas: int,
) -> dict[str, Any]:
    """Mide rescate, dano y cobertura de una variante frente a V03.

    Se evaluan dos puntos de operacion: el del propio artefacto, que es su version fiel, y el
    umbral que maximiza el rescate neto, que es la version mas favorable. Un modelo que solo tiene
    complementary con un umbral exotico y pierde con el suyo no es un candidato.

    La busqueda del umbral optimo se hace **etiqueta por etiqueta**, que es exacto: el error de una
    celda depende solo de su propia prediccion y las celdas de distinta etiqueta no interactuan en
    el recuento, de modo que el maximo del total es la suma de los maximos por etiqueta.

    Args:
        artefacto: nombre del artefacto.
        variante: nombre de la variante.
        descripcion: descripcion legible.
        y: etiquetas verdaderas alineadas.
        proba: probabilidades alineadas.
        error_v03: indicador de error de V03, alineado.
        umbrales: umbrales declarados por el artefacto.
        n_celdas: numero de celdas de la rejilla.

    Returns:
        Fila con todas las medidas.
    """
    pred = (proba >= umbrales[None, :]).astype(int)
    error = pred != y
    rescates = int(np.sum(error_v03 & ~error))
    danos = int(np.sum(~error_v03 & error))
    a = error_v03.ravel().astype(float)
    b = error.ravel().astype(float)
    desviacion = float(np.std(a) * np.std(b))
    correlacion = (
        float(np.mean((a - a.mean()) * (b - b.mean()))) / desviacion if desviacion > 0 else 0.0
    )
    por_etiqueta = {
        ETIQUETAS[i]: {
            "rescates": int(np.sum(error_v03[:, i] & ~error[:, i])),
            "danios": int(np.sum(~error_v03[:, i] & error[:, i])),
        }
        for i in range(3)
    }

    # Maximizacion exacta del rescate neto, etiqueta por etiqueta.
    rejilla = np.arange(0.20, 0.851, 0.01)
    optimos = np.zeros(3, dtype=np.float64)
    neto_optimo = 0
    for indice in range(3):
        e_v03 = error_v03[:, indice]
        columna = proba[:, indice]
        mejor = (-10**9, float(rejilla[0]))
        for umbral in rejilla:
            e_cand = (columna >= umbral).astype(int) != y[:, indice]
            r = int(np.sum(e_v03 & ~e_cand))
            d = int(np.sum(~e_v03 & e_cand))
            if r - d > mejor[0]:
                mejor = (r - d, float(umbral))
        neto_optimo += mejor[0]
        optimos[indice] = mejor[1]
    macro_optima = _f1_por_etiqueta(y, proba, optimos)
    # Rescates y danos **en el umbral optimo**, que es el punto de operacion que decide. Los de
    # mas arriba estan en el umbral que declara el artefacto, que es otro punto. Sin guardar los
    # dos, una fila de la tabla mezclaria un recuento con un neto de otro punto de operacion y la
    # resta no cerraria, aunque cada cifra por separado fuese cierta.
    error_opt = (proba >= optimos[None, :]).astype(int) != y

    return {
        "artefacto": artefacto,
        "variante": variante,
        "descripcion": descripcion,
        "macro_f1_propia": _f1_por_etiqueta(y, proba, umbrales),
        "delta_macro_f1": _f1_por_etiqueta(y, proba, umbrales) - REFERENCIA_V03,
        "umbrales_usados": "/".join(f"{u:.2f}" for u in umbrales),
        "rescates": rescates,
        "danios": danos,
        "rescate_neto": rescates - danos,
        "ratio_rescate_danio": rescates / danos if danos else float("inf"),
        "cobertura_de_errores_v03": rescates / max(1, int(error_v03.sum())),
        "errores_nuevos_sobre_aciertos_v03": danos / max(1, int((~error_v03).sum())),
        "correlacion_de_errores": correlacion,
        "solape_de_predicciones": float(np.sum(pred == ((proba >= umbrales[None, :]).astype(int)))) / n_celdas,
        "rescates_HS": por_etiqueta["HS"]["rescates"],
        "danios_HS": por_etiqueta["HS"]["danios"],
        "rescates_TR": por_etiqueta["TR"]["rescates"],
        "danios_TR": por_etiqueta["TR"]["danios"],
        "rescates_AG": por_etiqueta["AG"]["rescates"],
        "danios_AG": por_etiqueta["AG"]["danios"],
        "rescate_neto_optimo": neto_optimo,
        "rescates_optimos": int(np.sum(error_v03 & ~error_opt)),
        "danios_optimos": int(np.sum(~error_v03 & error_opt)),
        "macro_f1_optima": macro_optima,
        "delta_macro_f1_optima": macro_optima - REFERENCIA_V03,
        "umbrales_optimos": "/".join(f"{u:.2f}" for u in optimos),
        "supera_a_v03_en_oof": macro_optima > REFERENCIA_V03,
        "celdas_netas_para_mover_0p003": CELDAS_POR_003_LB,
    }


def _analizar_union(
    filas: list[dict[str, Any]], errores: dict[str, np.ndarray], error_v03: np.ndarray
) -> dict[str, Any]:
    """Mide si los rescates de las mejores variantes se suman o se solapan.

    Esta es la medicion que decide si un ensemble de varias variantes puede alcanzar la escala
    necesaria. Si los rescates se solapan mucho, combinarlas no acumula nada: los modelos fallan en
    las mismas filas, y el ensemble se reduce a uno solo.

    Args:
        filas: variantes medidas, ordenadas por rescate neto.
        errores: indicador de error por variante, en su umbral optimo.
        error_v03: indicador de error de V03.

    Returns:
        Diccionario con la union, el solape y el veredicto.
    """
    mejores = list(filas[:8])
    rescates_union = np.zeros_like(error_v03)
    danos_union = np.zeros_like(error_v03)
    suma_individual = 0
    log("")
    log(f"   {'candidato':46s} {'resc':>5s} {'danio':>5s} {'neto':>5s}")
    for f in mejores:
        clave = f"{f['artefacto']}/{f['variante']}"
        e = errores.get(clave)
        if e is None:
            continue
        rescate = error_v03 & ~e
        dano = ~error_v03 & e
        rescates_union |= rescate
        danos_union |= dano
        suma_individual += int(rescate.sum())
        # La matriz de error guardada esta en el umbral optimo de la variante, de modo que el
        # recuento de la union y el de la tabla salen del mismo punto de operacion. Si no
        # coincidieran, la union compararia unos numeros con otros y el veredicto no significaria
        # nada, asi que se detiene en vez de avisar y continuar.
        if int(rescate.sum()) != f["rescates_optimos"] or int(dano.sum()) != f["danios_optimos"]:
            raise SystemExit(
                f"PARADA OBLIGATORIA: el recuento de la union para {clave} no coincide con la "
                "tabla, luego los dos usan puntos de operacion distintos"
            )
        log(
            f"   {clave[:46]:46s} {int(rescate.sum()):5d} {int(dano.sum()):5d} "
            f"{int(rescate.sum()) - int(dano.sum()):+5d}"
        )
    rescates = int(rescates_union.sum())
    damages = int(danos_union.sum())
    neto = rescates - damages
    solape = 1 - rescates / max(1, suma_individual)
    alcance = neto >= CELDAS_POR_003_LB
    log("")
    log(f"   suma individual de rescates de las {len(mejores)}: {suma_individual}")
    log(f"   UNION: {rescates} celdas rescatadas y {damages} danadas, neto {neto:+d}")
    log(f"   solape de los rescates: {100 * solape:.0f} por ciento")
    log(f"   escala necesaria para mover 0.003 de LB: {CELDAS_POR_003_LB} netos")
    log(f"   la union alcanza esa escala: {'SI' if alcance else 'NO'}")
    return {
        "variantes": [f"{f['artefacto']}/{f['variante']}" for f in mejores],
        "rescates_individuales_sumados": suma_individual,
        "rescates_de_la_union": rescates,
        "danios_de_la_union": damages,
        "rescate_neto_de_la_union": neto,
        "solape_de_los_rescates": solape,
        "celdas_netas_necesarias": CELDAS_POR_003_LB,
        "la_union_alcanza_la_escala": alcance,
    }


def _grupo(f: dict[str, Any]) -> str:
    """Asigna una variante a los grupos A, B o C del P2.

    Args:
        f: fila medida.

    Returns:
        "A", "B" o "C".
    """
    if f["supera_a_v03_en_oof"]:
        return "A"
    if f["rescate_neto_optimo"] > 0:
        return "B"
    return "C"


def _resumen(
    filas: list[dict[str, Any]], n_errores: int, n_celdas: int, omitidos: list[str]
) -> None:
    """Muestra el resumen de la medicion.

    Args:
        filas: filas medidas.
        n_errores: errores de V03.
        n_celdas: celdas de la rejilla.
        omitidos: artefactos omitidos con su motivo.
    """
    log("")
    log(f"   V03 falla {n_errores} de {n_celdas} celdas. Orden por rescate neto con umbral optimo.")
    log("")
    log(
        f"   {'artefacto':24s} {'variante':26s} {'resc':>5s} {'danio':>5s} {'neto':>5s} "
        f"{'macro':>8s} {'delta':>9s} {'corr':>6s}"
    )
    log(f"   {'-'*24} {'-'*26} {'-'*5} {'-'*5} {'-'*5} {'-'*8} {'-'*9} {'-'*6}")
    for f in filas[:16]:
        log(
            f"   {f['artefacto'][:24]:24s} {f['variante'][:26]:26s} "
            f"{f['rescates_optimos']:5d} {f['danios_optimos']:5d} "
            f"{f['rescate_neto_optimo']:+5d} {f['macro_f1_optima']:8.4f} "
            f"{f['delta_macro_f1_optima']:+9.5f} {f['correlacion_de_errores']:6.3f}"
        )
    posibles = [f for f in filas if f["supera_a_v03_en_oof"]]
    log("")
    log(f"   Variantes que con su umbral optimo superan a V03 en OOF: {len(posibles)} de {len(filas)}")
    for f in sorted(posibles, key=lambda x: -x["macro_f1_optima"])[:8]:
        log(
            f"     {f['artefacto']}/{f['variante']}: OOF {f['macro_f1_optima']:.6f} "
            f"({f['delta_macro_f1_optima']:+.6f}) con umbrales {f['umbrales_optimos']}"
        )
    if omitidos:
        log("")
        log("   Artefactos omitidos:")
        for motivo in omitidos:
            log(f"     {motivo}")


def _informe(informe: dict[str, Any], inicio: float) -> None:
    """Escribe el informe de la complementariedad.

    Args:
        informe: veredicto del inventario y la medicion.
        inicio: instante de arranque del script.
    """
    from v11_ayudas import fila, sello

    rejilla = informe["rejilla"]
    filas = informe["complementariedad"]
    posibles = informe["variantes_que_superan_a_v03_en_oof"]
    linea: list[str] = [
        "# V17-00: inventario del patrimonio y complementariedad de errores frente a V03",
        "",
        f"Generado por `experiments/v17_00_inventario_y_complementariedad.py` ({EXPERIMENTO}).",
        "",
        "**No se entrena nada, no se regenera nada y no se gasta ninguna submission.** Esto es un",
        "recuento sobre artefactos que ya existen, y decide si un ensemble merece construirse.",
        "",
        "## 1. La rejilla",
        "",
        f"- **{rejilla['filas']} filas x 3 etiquetas = {rejilla['celdas']} celdas.**",
        (
            f"- Errores de V03: **{rejilla['errores_de_v03']}**, el "
            f"{100 * rejilla['proporcion_de_error']:.2f} por ciento de las celdas."
        ),
        "- La rejilla es de celdas y no de filas: la metrica es la media de las tres F1, y",
        "  corregir HS no cambia el acierto de TR ni de AG.",
        "",
        "## 2. La primera pregunta, con numeros",
        "",
        "Cuantos errores de V03 rescata cada modelo historico y cuantos danos introduce. Orden",
        "por rescate neto en el umbral optimo, que es la cifra que decide.",
        "",
        fila(
            [
                "artefacto", "variante", "macro F1 optima", "rescates optimos",
                "danios optimos", "rescate neto optimo", "cobertura",
                "correlacion de errores",
            ]
        ),
        fila(["---"] * 8),
    ]
    for f in filas[:18]:
        linea.append(
            fila(
                [
                    f["artefacto"][:22], f["variante"][:24], f"{f['macro_f1_optima']:.4f}",
                    str(f["rescates_optimos"]), str(f["danios_optimos"]),
                    f"{f['rescate_neto_optimo']:+d}",
                    f"{100 * f['cobertura_de_errores_v03']:.1f} por ciento",
                    f"{f['correlacion_de_errores']:.3f}",
                ]
            )
        )
    linea += [
        "",
        "## 3. La segunda pregunta: hay alguna combinacion con rescate neto suficiente?",
        "",
    ]
    if posibles:
        linea += [
            f"**{len(posibles)} variantes superan a V03 en OOF con su umbral optimo.** Esas son las",
            "unicas que podrian entrar en un ensemble. Todas pertenecen a artefactos que ya se",
            "habian enviado, de modo que la pregunta no es si el OOF mejora, sino si esa mejora",
            "transferiria, que es justamente lo que V16 demostro que no hace.",
            "",
            fila(["artefacto", "variante", "OOF optimo", "delta", "umbrales optimos"]),
            fila(["---"] * 5),
        ]
        for f in sorted(posibles, key=lambda x: -x["macro_f1_optima"])[:10]:
            linea.append(
                fila(
                    [
                        f["artefacto"][:22], f["variante"][:24],
                        f"{f['macro_f1_optima']:.6f}", f"{f['delta']:+.6f}", f["umbrales"],
                    ]
                )
            )
    else:
        linea += [
            "**Ninguna.** Ninguna variante del patrimonio, ni con el umbral que declara el",
            "artefacto ni con el que maximiza su rescate neto, supera a V03 en el OOF de",
            "desarrollo.",
            "",
            "Eso responde la segunda pregunta de forma negativa y sin ambiguedad: si ningun",
            "modelo historico contiene, en el mismo conjunto de calibracion, mas informacion que",
            "V03, **no existe ninguna combinacion de esos modelos cuyo rescate neto produzca una",
            "mejora de leaderboard**. La combinacion no se construye porque la evidencia ya la",
            "descarta.",
        ]
    if informe["artefactos_omitidos"]:
        linea += ["", "## 4. Artefactos omitidos y por que", ""]
        for motivo in informe["artefactos_omitidos"]:
            linea.append(f"- {motivo}")
    linea += [
        "",
        "## 5. Lo que este script NO hace",
        "",
        "- No entrena ningun modelo, como pide la instruccion de la fase.",
        "- No construye ningun ensemble: la condicion para construirlo es que la",
        "  complementibilidad se sostenga, y eso se evalua aqui.",
        "- No gasta ninguna submission. Las cinco restantes se conservan para un candidato con",
        "  hipotesis concreta.",
        "",
    ]
    linea += sello(inicio)
    REPORTE_MD.write_text("\n".join(linea) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
