"""V30-01: perfil del error residual de V03, orientado a encontrar el siguiente eje.

Que responde
------------
La pregunta de la fase, dicha con las palabras de la directiva:

> Una vez descartados ruido de etiquetas, especializacion, matching y familias no lineales, en
> que regiones concretas del texto esta perdiendo V03 y que informacion de esas regiones todavia
> no estamos explotando?

Contexto que lo motiva
----------------------
`LABEL_NOISE = CLOSED` (V30-00). Cuatro experimentos historicos ya cerraron la via de la
robustez frente a etiquetas: V4-B01 (que medida de dureza ordena la dificultad), V4-B02 (las
poblaciones dificil-coherente y posible-ruido son separables), V4-B03 (reponderar por dureza,
unica positiva +0,0036 que no cumple gates), V4-B05 (downweight y remove sobre contradiccion
textual, ninguna mueve la metrica), V6-07 (ponderar pares inconsistentes, cerrado negativo) y
V8-08 (corrector por margenes, negativo). No se repite ninguno.

Lo que este script hace es otra cosa: **no toca el modelo**. Describe donde falla V03, con
cuantiles y no solo con medias, por label, con interacciones y con estabilidad entre folds, para
localizar regiones concretas y despues preguntar que informacion de esas regiones no se esta
usando.

Las poblaciones historicas, sin redefinir
-----------------------------------------
`experiments/28_hardness_v4.py` (lineas 593-598) define:

    corte = percentile(hardness_score, 80)
    poblacion = "Posible ruido"  si cluster_contradictorio > 0
                "Dificil coherente" si hardness_score >= corte
                "Facil" en otro caso

donde `hardness_score = 1 - |(p - umbral) * (2y - 1)|` por etiqueta, con los umbrales congelados
de V03 (0,44 / 0,51 / 0,46), y `cluster_contradictorio` marca los ejemplos en un cluster de
texto repetido con anotaciones contradictorias. V28-03 midio exactamente esa ultima condicion:
43 filas del desarrollo. Se reconstruye igual, sin tocar la definicion.

Lo que este script NO hace
--------------------------
- No entrena ningun modelo.
- No modifica V03, sus umbrales ni su representacion.
- No construye un predictor con las caracteristicas.
- No lee el test.
- No produce submission.

Uso
---
    /home/carlos/venv/bin/python experiments/v30_01_perfil_error_residual.py
"""

from __future__ import annotations

import json
import re
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.splits import grupos_fusionados

EXPERIMENTO = "V30-01 (perfil del error residual de V03, orientado a encontrar el siguiente eje)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v30_01_perfil_error_residual.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v30_01_perfil_error_residual.md"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
OOF_CANONICO = PROJECT_ROOT / "features" / "oof_v03_final.npz"

ETIQUETAS = ("HS", "TR", "AG")
UMBRALES_V03 = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
CORRECTED_CV_V03 = 0.7772114525165721
RE_URL = re.compile(r"https?://|www\.")
RE_MENCION = re.compile(r"@\w+")
RE_HASHTAG = re.compile(r"#\w+")
RE_ALFA = re.compile("[A-Za-z\u00c0-\u00ff]")
RE_DIGITO = re.compile(r"\d")
RE_PUNCT = re.compile(r"[!\u00bf?.,;:()\[\]{}\"'`\-_/\\|+*=<>$%&]")
RE_MAYUS = re.compile("[A-Z\u00c0-\u00dc]")
RE_REPETIDO = re.compile(r"(.)\1{2,}")
CUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V30-01] {mensaje}", flush=True)


def memoria_pico_mb() -> float:
    """Resident set maximum del proceso, en megabytes.

    Returns:
        La memoria maxima residente en MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def rasgos_de(texto: str) -> dict[str, float]:
    """Calcula los rasgos textuales de un documento.

    Todos derivan exclusivamente del campo `text`. Ninguno usa etiquetas, identificadores ni
    metadatos externos, de modo que el diagnostico no puede filtrar informacion del objetivo.

    Args:
        texto: contenido del documento.

    Returns:
        Diccionario con un valor por rasgo.
    """
    n = max(len(texto), 1)
    tokens = texto.split()
    n_tokens = max(len(tokens), 1)
    n_alfa = len(RE_ALFA.findall(texto))
    menciones = RE_MENCION.findall(texto)
    hashtags = RE_HASHTAG.findall(texto)
    pos_mencion = [m.start() / n for m in RE_MENCION.finditer(texto)]
    pos_hashtag = [m.start() / n for m in RE_HASHTAG.finditer(texto)]
    unicas = len({t.lower().strip(".,;:!?") for t in tokens})
    return {
        "caracteres": float(len(texto)),
        "tokens": float(n_tokens),
        "alfabeticos": float(n_alfa),
        "frac_alfabeticos": n_alfa / n,
        "digitos": float(len(RE_DIGITO.findall(texto))),
        "urls": float(len(RE_URL.findall(texto))),
        "menciones": float(len(menciones)),
        "hashtags": float(len(hashtags)),
        "puntuacion": float(len(RE_PUNCT.findall(texto))),
        "frac_puntuacion": len(RE_PUNCT.findall(texto)) / n,
        "mayusculas": float(len(RE_MAYUS.findall(texto))),
        "frac_mayusculas": len(RE_MAYUS.findall(texto)) / max(n_alfa, 1),
        "caracteres_repetidos": float(len(RE_REPETIDO.findall(texto))),
        "densidad_lexica": unicas / n_tokens,
        "longitud_media_token": n / n_tokens,
        "frac_mayus_completas": sum(1 for t in tokens if t.isupper()) / n_tokens,
        "pos_primera_mencion": float(pos_mencion[0]) if pos_mencion else -1.0,
        "pos_primera_hashtag": float(pos_hashtag[0]) if pos_hashtag else -1.0,
        "tiene_url": float(bool(RE_URL.search(texto))),
        "tiene_mencion": float(bool(menciones)),
        "tiene_hashtag": float(bool(hashtags)),
        "es_muy_corto": float(len(tokens) <= 4),
        "es_largo": float(len(tokens) >= 25),
    }


def cuantiles(serie: pd.Series) -> dict[str, float]:
    """Cuantiles de una serie, con nombre explicito.

    Args:
        serie: serie numerica.

    Returns:
        Diccionario con los cuantiles pedidos.
    """
    valores = np.percentile(serie.to_numpy(dtype=float), [q * 100 for q in CUANTILES])
    return {f"p{int(q * 100)}": float(v) for q, v in zip(CUANTILES, valores, strict=True)}


def main() -> None:
    """Ejecuta el diagnostico del error residual de V03.

    Returns:
        None.
    """
    inicio = time.time()
    log(f"{EXPERIMENTO}. NINGUN modelo, NINGUNA submission, test no leido")

    train = read_train()
    particion = pd.read_csv(PARTICION)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    oof = np.load(OOF_CANONICO, allow_pickle=True)
    y = np.asarray(oof["y_true"], dtype=np.int64)
    proba = np.asarray(oof["proba"], dtype=np.float64)
    pred = np.asarray(oof["pred"], dtype=np.int64)
    pliegues = np.asarray(oof["pliegues"], dtype=np.int64)
    ids = np.asarray(oof["ids"], dtype=np.int64)
    mapa = dict(zip(desarrollo["id"].astype(int), desarrollo["text"]))
    textos = [mapa[int(i)] for i in ids]
    log(f"   desarrollo {len(textos)} filas | {len(np.unique(pliegues))} pliegues | "
        f"corrected_cv de referencia {CORRECTED_CV_V03:.6f}")

    log("1/7 poblaciones historicas, reconstruidas sin redefinir")
    grupos = grupos_fusionados(desarrollo)
    mapa_grupo = dict(zip(desarrollo["id"].astype(int), grupos))
    grupos_oof = np.array([mapa_grupo[int(i)] for i in ids])
    # cluster_contradictorio: el grupo fusionado tiene anotaciones contradictorias en alguna
    # etiqueta. Es la condicion que midio V28-03 (43 filas) y la que usa 28_hardness_v4.py.
    contradictorio = np.zeros(len(ids), dtype=bool)
    for grupo in np.unique(grupos_oof):
        sel = grupos_oof == grupo
        if any(np.unique(y[sel, j]).size > 1 for j in range(3)):
            contradictorio[sel] = True
    # hardness_score = 1 - |(p - umbral) * (2y - 1)|, media por etiqueta, con umbrales congelados.
    umbrales = np.array([UMBRALES_V03[et] for et in ETIQUETAS])
    hardness = 1.0 - np.abs((proba - umbrales[None, :]) * (2 * y - 1))
    hardness_score = hardness.mean(axis=1)
    corte = float(np.percentile(hardness_score, 80))
    poblacion = np.where(
        contradictorio,
        "Posible ruido",
        np.where(hardness_score >= corte, "Dificil coherente", "Facil"),
    )
    for nombre in ("Facil", "Dificil coherente", "Posible ruido"):
        n = int((poblacion == nombre).sum())
        log(f"   {nombre:<20} {n:>4} ({n / len(ids):.4f})")
    log(f"   corte de dureza (percentil 80): {corte:.4f}")

    log("2/7 grupos de error por label y perfil textual con cuantiles")
    tabla = pd.DataFrame([rasgos_de(t) for t in textos])
    tabla["__pliegue"] = pliegues
    tabla["__poblacion"] = poblacion
    tabla["__hardness"] = hardness_score
    for j, et in enumerate(ETIQUETAS):
        tabla[f"__correct_{et.lower()}"] = pred[:, j] == y[:, j]
        tabla[f"__fp_{et.lower()}"] = (pred[:, j] == 1) & (y[:, j] == 0)
        tabla[f"__fn_{et.lower()}"] = (pred[:, j] == 0) & (y[:, j] == 1)
        tabla[f"__error_{et.lower()}"] = pred[:, j] != y[:, j]
    rasgos = [c for c in tabla.columns if not c.startswith("__")]
    log(f"   {len(rasgos)} rasgos textuales, todos derivados de text")

    perfil: dict[str, Any] = {}
    for rasgo in rasgos:
        perfil[rasgo] = {
            "global": cuantiles(tabla[rasgo]),
            "por_poblacion": {
                nombre: cuantiles(tabla.loc[tabla["__poblacion"] == nombre, rasgo])
                for nombre in ("Facil", "Dificil coherente", "Posible ruido")
            },
        }
    # Diferencia de distribucion entre poblaciones, que es lo que dice si son distintas.
    separacion: dict[str, Any] = {}
    for rasgo in rasgos:
        facil = tabla.loc[tabla["__poblacion"] == "Facil", rasgo].to_numpy(dtype=float)
        dificil = tabla.loc[tabla["__poblacion"] == "Dificil coherente", rasgo].to_numpy(dtype=float)
        if facil.std() < 1e-12 and dificil.std() < 1e-12:
            continue
        _u, pvalor = stats.mannwhitneyu(facil, dificil, alternative="two-sided")
        pooled = np.sqrt((facil.var(ddof=1) + dificil.var(ddof=1)) / 2.0)
        d_cohen = (dificil.mean() - facil.mean()) / pooled if pooled > 1e-12 else 0.0
        separacion[rasgo] = {
            "d_cohen_dificil_menos_facil": float(d_cohen),
            "p_mannwhitney": float(pvalor),
            "media_facil": float(facil.mean()),
            "media_dificil": float(dificil.mean()),
        }
    ordenados = sorted(separacion.items(), key=lambda kv: -abs(kv[1]["d_cohen_dificil_menos_facil"]))
    log("   separacion Facil vs Dificil coherente, los ocho rasgos de mayor efecto:")
    for nombre, d in ordenados[:8]:
        log(
            f"      {nombre:<26} d {d['d_cohen_dificil_menos_facil']:+.3f} | "
            f"facil {d['media_facil']:.3f} vs dificil {d['media_dificil']:.3f} | "
            f"p {d['p_mannwhitney']:.2e}"
        )

    log("3/7 perfil por label: las tres etiquetas tienen errores distintos?")
    por_label: dict[str, Any] = {}
    for j, et in enumerate(ETIQUETAS):
        col = f"__error_{et.lower()}"
        bloque: dict[str, Any] = {
            "n_errores": int(tabla[col].sum()),
            "tasa_de_error": float(tabla[col].mean()),
            "n_fp": int(tabla[f"__fp_{et.lower()}"].sum()),
            "n_fn": int(tabla[f"__fn_{et.lower()}"].sum()),
            "fp_entre_predichos": float(tabla[f"__fp_{et.lower()}"].sum() / max(pred[:, j].sum(), 1)),
            "fn_entre_positivos": float(
                tabla[f"__fn_{et.lower()}"].sum() / max(y[:, j].sum(), 1)
            ),
            "rasgos_de_mayor_efecto": [],
        }
        efectos = []
        for rasgo in rasgos:
            mal = tabla.loc[tabla[col], rasgo].to_numpy(dtype=float)
            bien = tabla.loc[~tabla[col], rasgo].to_numpy(dtype=float)
            if mal.std() < 1e-12 and bien.std() < 1e-12:
                continue
            _u, pvalor = stats.mannwhitneyu(mal, bien, alternative="two-sided")
            pooled = np.sqrt((mal.var(ddof=1) + bien.var(ddof=1)) / 2.0)
            d_cohen = (mal.mean() - bien.mean()) / pooled if pooled > 1e-12 else 0.0
            efectos.append((rasgo, d_cohen, pvalor, mal.mean(), bien.mean()))
        efectos.sort(key=lambda e: -abs(e[1]))
        bloque["rasgos_de_mayor_efecto"] = [
            {"rasgo": r, "d_cohen": float(d), "p": float(p),
             "media_si_error": float(m), "media_si_correcto": float(b)}
            for r, d, p, m, b in efectos[:6]
        ]
        por_label[et] = bloque
        log(
            f"   {et}: errores {bloque['n_errores']} ({bloque['tasa_de_error']:.4f}) | "
            f"FP {bloque['n_fp']} FN {bloque['n_fn']}"
        )
        for e in efectos[:3]:
            log(
                f"      {e[0]:<26} d {e[1]:+.3f} | error {e[3]:.3f} vs correcto {e[4]:.3f} | "
                f"p {e[2]:.2e}"
            )

    log("4/7 interaccion: P(V03 wrong | region de texto)")
    regiones = _definir_regiones(tabla)
    interaccion: dict[str, Any] = {}
    for nombre, mask in regiones.items():
        m = int(mask.sum())
        if m == 0:
            continue
        entrada: dict[str, Any] = {"n_filas": m, "fraccion": m / len(tabla)}
        for j, et in enumerate(ETIQUETAS):
            col = f"__error_{et.lower()}"
            tasa = float(tabla.loc[mask, col].mean())
            resto = float(tabla.loc[~mask, col].mean())
            entrada[et] = {
                "tasa_error_en_region": tasa,
                "tasa_error_fuera": resto,
                "exceso": tasa - resto,
            }
        interaccion[nombre] = entrada
        linea = " ".join(
            f"{et} {entrada[et]['tasa_error_en_region']:.4f} (fuera {entrada[et]['tasa_error_fuera']:.4f})"
            for et in ETIQUETAS
        )
        log(f"   {nombre:<34} filas {m:>4} | {linea}")

    log("5/7 cruces predefinidos, no un grid exhaustivo")
    cruces = _cruces(tabla)
    for nombre, d in cruces.items():
        filas = d["filas"]
        log(f"   {nombre}: {len(filas)} combinaciones")
        # Solo se imprimen las combinaciones mas extremas: un crucero interesa por donde se
        # separa de la media del cruce, no por el listado completo.
        if "condicion" in filas[0]:
            media = float(np.mean([f["tasa_error"]["HS"] for f in filas]))
            orden = sorted(filas, key=lambda f: -abs(f["tasa_error"]["HS"] - media))
            for f in orden[:2]:
                log(
                    f"      {f['tramo']:<8} {f['condicion']:<3} n {f['n_filas']:>4} | "
                    + " ".join(
                        f"{et} {f['tasa_error'][et]:.4f}" for et in ETIQUETAS
                    )
                )

    log("6/7 estabilidad OOF: en cuantos de los 5 pliegues aparece cada patron")
    estabilidad = _estabilidad(tabla, regiones, rasgos)
    for nombre, d in estabilidad.items():
        log(f"   {nombre:<34} {d['resumen']}")

    log("7/7 informe")
    informe = {
        "familia": "RESIDUAL_ERROR_PROFILE",
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "DIAGNOSTIC_ONLY",
        "pregunta": (
            "en que regiones concretas del texto esta perdiendo V03 y que informacion de esas "
            "regiones todavia no estamos explotando"
        ),
        "alcance": (
            "no se entrena ningun modelo, no se modifica V03, no se construye predictor y no se "
            "produce submission. Las caracteristicas describen el error, no alimentan un clasificador"
        ),
        "poblaciones_historicas": {
            "definicion": (
                "la de experiments/28_hardness_v4.py lineas 593-598: Posible ruido si "
                "cluster_contradictorio > 0; Dificil coherente si hardness_score >= percentil 80; "
                "Facil en otro caso. hardness_score = 1 - |(p - umbral) * (2y - 1)| por etiqueta "
                "con los umbrales congelados de V03"
            ),
            "umbral_de_dureza_p80": corte,
            "conteo": {nombre: int((poblacion == nombre).sum())
                       for nombre in ("Facil", "Dificil coherente", "Posible ruido")},
            "cluster_contradictorio_filas": int(contradictorio.sum()),
        },
        "rasgos": rasgos,
        "cuantiles": CUANTILES,
        "perfil_por_poblacion": perfil,
        "separacion_facil_vs_dificil": separacion,
        "por_label": por_label,
        "interaccion_por_region": interaccion,
        "cruces": cruces,
        "estabilidad_por_pliegue": estabilidad,
        "gastos": {
            "submissions": 0,
            "modelos_entrenados": 0,
            "memoria_pico_mb": round(memoria_pico_mb(), 1),
            "segundos": round(time.time() - inicio, 1),
        },
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe_md(informe)
    log("   informe escrito en reports/v30_01_perfil_error_residual.{json,md}")
    log(f"   memoria pico {memoria_pico_mb():.0f} MB | tiempo {round(time.time() - inicio, 1)} s")
    log("VEREDICTO: diagnostico completado, 0 modelos, 0 submissions")


def _definir_regiones(tabla: pd.DataFrame) -> dict[str, pd.Series]:
    """Regiones del espacio textual, predefinidas y robustas.

    Se usan cortes por cuantil en vez de umbrales magicos, para que cada region tenga volumen
    comparable y las tasas sean comparables entre si.

    Args:
        tabla: tabla de rasgos por fila.

    Returns:
        Diccionario con la mascara booleana de cada region.
    """
    tokens = tabla["tokens"].to_numpy(dtype=float)
    t1, t2 = np.percentile(tokens, [33.33, 66.67])
    densidad = tabla["densidad_lexica"].to_numpy(dtype=float)
    d1, d2 = np.percentile(densidad, [33.33, 66.67])
    return {
        "decil 1 de longitud (mas corto)": tabla["tokens"] <= np.percentile(tokens, 10),
        "decil 10 de longitud (mas largo)": tabla["tokens"] >= np.percentile(tokens, 90),
        "texto corto (tercio inferior de tokens)": tabla["tokens"] <= t1,
        "texto largo (tercio superior de tokens)": tabla["tokens"] >= t2,
        "con URL": tabla["tiene_url"] > 0,
        "sin URL": tabla["tiene_url"] == 0,
        "con mencion": tabla["tiene_mencion"] > 0,
        "sin mencion": tabla["tiene_mencion"] == 0,
        "con hashtag": tabla["tiene_hashtag"] > 0,
        "sin hashtag": tabla["tiene_hashtag"] == 0,
        "densidad lexica baja (tercio inferior)": tabla["densidad_lexica"] <= d1,
        "densidad lexica alta (tercio superior)": tabla["densidad_lexica"] >= d2,
        "casi todo en mayusculas (frac > 0,7)": tabla["frac_mayusculas"] > 0.70,
        "alta puntuacion (> 0,10 por caracter)": tabla["frac_puntuacion"] > 0.10,
        "con caracteres repetidos": tabla["caracteres_repetidos"] > 0,
        "mencion en el primer tercio": (tabla["pos_primera_mencion"] >= 0)
        & (tabla["pos_primera_mencion"] < 0.33),
        "hashtag en el primer tercio": (tabla["pos_primera_hashtag"] >= 0)
        & (tabla["pos_primera_hashtag"] < 0.33),
    }


def _cruces(tabla: pd.DataFrame) -> dict[str, Any]:
    """Cruces predefinidos entre longitud y los rasgos de presencia.

    Se fija la lista por adelantado para no buscar subgrupos por azar. Cada cruce compara la
    tasa de error de las combinaciones y senala donde esta se aleja del promedio.

    Args:
        tabla: tabla de rasgos por fila.

    Returns:
        Diccionario con los cruces y sus filas.
    """
    tokens = tabla["tokens"].to_numpy(dtype=float)
    t1, t2 = np.percentile(tokens, [33.33, 66.67])
    tramo = np.where(tokens <= t1, "corto", np.where(tokens >= t2, "largo", "medio"))
    salida: dict[str, Any] = {}
    for nombre, serie in (
        ("longitud x URL", tabla["tiene_url"] > 0),
        ("longitud x mencion", tabla["tiene_mencion"] > 0),
        ("longitud x hashtag", tabla["tiene_hashtag"] > 0),
    ):
        filas = []
        for tramo_nombre in ("corto", "medio", "largo"):
            for valor in (False, True):
                mask = (tramo == tramo_nombre) & (serie == valor)
                m = int(mask.sum())
                if m == 0:
                    continue
                tasas = {
                    et: float(tabla.loc[mask, f"__error_{et.lower()}"].mean())
                    for et in ETIQUETAS
                }
                filas.append({
                    "tramo": tramo_nombre,
                    "condicion": "si" if valor else "no",
                    "n_filas": m,
                    "tasa_error": tasas,
                })
        salida[nombre] = {"filas": filas}
    # longitud x label: la tasa de error de cada label por tramo de longitud.
    filas_label = []
    for tramo_nombre in ("corto", "medio", "largo"):
        mask = tramo == tramo_nombre
        m = int(mask.sum())
        if m == 0:
            continue
        filas_label.append({
            "tramo": tramo_nombre,
            "n_filas": m,
            "tasa_error": {
                et: float(tabla.loc[mask, f"__error_{et.lower()}"].mean())
                for et in ETIQUETAS
            },
        })
    salida["longitud x label"] = {"filas": filas_label}
    return salida


def _estabilidad(
    tabla: pd.DataFrame, regiones: dict[str, pd.Series], rasgos: list[str]
) -> dict[str, Any]:
    """Comprueba en cuantos pliegues aparece cada patron.

    Un fenomeno presente en un solo pliegue no es una estructura. Para cada region y para el
    rasgo dominante de cada label se cuenta en cuantos de los cinco pliegues la tasa de error de
    la region supera a la del resto, y se registra la direccion y la magnitud.

    Args:
        tabla: tabla de rasgos por fila.
        regiones: mascaras de las regiones predefinidas.
        rasgos: lista de rasgos disponibles.

    Returns:
        Diccionario con la estabilidad de cada patron.
    """
    pliegues = sorted(tabla["__pliegue"].unique())
    salida: dict[str, Any] = {}
    for nombre, mask in regiones.items():
        if mask.sum() < 20 or (~mask).sum() < 20:
            continue
        apariciones = 0
        deltas = []
        for pliegue in pliegues:
            sel = tabla["__pliegue"] == pliegue
            dentro = tabla.loc[sel & mask, "__error_hs"].mean()
            fuera = tabla.loc[sel & ~mask, "__error_hs"].mean()
            deltas.append(float(dentro - fuera))
            if dentro > fuera:
                apariciones += 1
        salida[nombre] = {
            "pliegues_con_la_misma_direccion": apariciones,
            "de_total": len(pliegues),
            "delta_por_pliegue": deltas,
            "resumen": (
                f"{apariciones}/{len(pliegues)} pliegues con la misma direccion, "
                f"delta medio {float(np.mean(deltas)):+.4f}"
            ),
        }
    return salida


def _informe_md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.

    Returns:
        None.
    """
    p = informe["poblaciones_historicas"]
    sep = informe["separacion_facil_vs_dificil"]
    pl = informe["por_label"]
    g = informe["gastos"]
    lineas = [
        "# V30-01: perfil del error residual de V03",
        "",
        f"Generado por `experiments/v30_01_perfil_error_residual.py` ({EXPERIMENTO}).",
        "",
        f"**Estado `{informe['estado']}`. Alcance: {informe['alcance']}**",
        "",
        "## 1. Las poblaciones historicas, reconstruidas sin redefinir",
        "",
        f"{p['definicion']}.",
        "",
        "| poblacion | filas | fraccion |",
        "|---|---:|---:|",
    ]
    for nombre, n in p["conteo"].items():
        lineas.append(f"| {nombre} | {n} | {n / sum(p['conteo'].values()):.4f} |")
    lineas += [
        "",
        f"Umbral de dureza (percentil 80): {p['umbral_de_dureza_p80']:.4f}.",
        f"Filas en cluster contradictorio: {p['cluster_contradictorio_filas']}.",
        "",
        "## 2. Se separan Facil y Dificil coherente en algo?",
        "",
        "Cohen d es la diferencia de medias en desviaciones estandar. Se listan los ocho rasgos de",
        "mayor efecto absoluto.",
        "",
        "| rasgo | d (dificil - facil) | media facil | media dificil | p |",
        "|---|---:|---:|---:|---:|",
    ]
    for nombre, d in sorted(sep.items(), key=lambda kv: -abs(kv[1]["d_cohen_dificil_menos_facil"]))[:8]:
        lineas.append(
            f"| {nombre} | {d['d_cohen_dificil_menos_facil']:+.3f} | {d['media_facil']:.3f} | "
            f"{d['media_dificil']:.3f} | {d['p_mannwhitney']:.2e} |"
        )
    lineas += [
        "",
        "## 3. Perfil por label",
        "",
        "| label | errores | tasa | FP | FN | FP / predichos | FN / positivos |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for et in ETIQUETAS:
        d = pl[et]
        lineas.append(
            f"| {et} | {d['n_errores']} | {d['tasa_de_error']:.4f} | {d['n_fp']} | {d['n_fn']} | "
            f"{d['fp_entre_predichos']:.4f} | {d['fn_entre_positivos']:.4f} |"
        )
    lineas += [
        "",
        "Los tres rasgos de mayor efecto para cada label:",
        "",
    ]
    for et in ETIQUETAS:
        lineas.append(f"### {et}")
        lineas.append("")
        lineas.append("| rasgo | d | media si error | media si correcto | p |")
        lineas.append("|---|---:|---:|---:|---:|")
        for e in pl[et]["rasgos_de_mayor_efecto"][:3]:
            lineas.append(
                f"| {e['rasgo']} | {e['d_cohen']:+.3f} | {e['media_si_error']:.3f} | "
                f"{e['media_si_correcto']:.3f} | {e['p']:.2e} |"
            )
        lineas.append("")
    lineas += [
        "## 4. P(V03 wrong | region de texto)",
        "",
        "Las regiones se cortan por cuantil para que tengan volumen comparable. El exceso es la",
        "tasa de error dentro de la region menos la tasa fuera.",
        "",
        "| region | filas | " + " | ".join(f"{et} dentro" for et in ETIQUETAS) + " |",
        "|---|---:|" + "---:|" * 3,
    ]
    for nombre, d in informe["interaccion_por_region"].items():
        celdas = " | ".join(f"{d[et]['tasa_error_en_region']:.4f}" for et in ETIQUETAS)
        lineas.append(f"| {nombre} | {d['n_filas']} | {celdas} |")
    lineas += [
        "",
        "## 5. Cruces predefinidos",
        "",
    ]
    for nombre, d in informe["cruces"].items():
        lineas.append(f"### {nombre}")
        lineas.append("")
        lineas.append("| tramo | condicion | filas | " + " | ".join(ETIQUETAS) + " |")
        lineas.append("|---|---|---:|" + "---:|" * 3)
        for f in d["filas"]:
            # El cruce longitud x label no tiene condicion: su segunda dimension es la propia
            # etiqueta, y no se inventa un valor para que la tabla quede cuadrada.
            condicion = f.get("condicion", "(tramo unico)")
            celdas = " | ".join(f"{f['tasa_error'][et]:.4f}" for et in ETIQUETAS)
            lineas.append(f"| {f['tramo']} | {condicion} | {f['n_filas']} | {celdas} |")
        lineas.append("")
    lineas += [
        "## 6. Estabilidad entre pliegues",
        "",
        "Un fenomeno presente en un solo pliegue no es una estructura. Se cuenta en cuantos de los",
        "cinco pliegues la tasa de error de la region supera a la del resto.",
        "",
        "| region | pliegues con la misma direccion | delta medio |",
        "|---|---:|---:|",
    ]
    for nombre, d in informe["estabilidad_por_pliegue"].items():
        lineas.append(
            f"| {nombre} | {d['pliegues_con_la_misma_direccion']}/{d['de_total']} | "
            f"{float(np.mean(d['delta_por_pliegue'])):+.4f} |"
        )
    lineas += [
        "",
        "## 7. Lo que este script no hace",
        "",
        "- No entrena ningun modelo.",
        "- No modifica V03, sus umbrales ni su representacion.",
        "- No construye un predictor con las caracteristicas.",
        "- No lee el test.",
        "- No produce submission.",
        "",
        (f"Coste: {g['segundos']} s, memoria {g['memoria_pico_mb']:.0f} MB, "
        f"{g['modelos_entrenados']} modelos, {g['submissions']} submissions."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
