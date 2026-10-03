"""V28-01: diagnostico de los errores de V03 por etiqueta, condicionado por rasgos del texto.

Que responde
------------
Si `LABEL_SPECIALIZATION` esta cerrado, la pregunta util ya no es "que modelo conviene", sino
**donde falla V03 y por que**. Este script no entrena nada: solo mide. Su valor es triple.

1. Que tipos de ejemplo concentran los errores de V03, por etiqueta.
2. Si hay **heterogeneidad entre HS, TR y AG**: si las tres etiquetas fallan sobre textos de
   naturaleza distinta, hay margen para especialistas. Si fallan sobre lo mismo, no lo hay.
3. Si existe un subconjunto suficientemente grande y distinguible donde un modelo distinto
   tendria sentido, que es el requisito para que una hipotesis de nueva informacion valga.

Las caracteristicas
-------------------
Todas se derivan **exclusivamente del campo `text`**, sin informacion externa, sin etiquetas y
sin los metadatos que la libreria de splits ya calcula. Se listan en `RASGOS` y cada una tiene su
funcion, con una nota de por que se incluye.

Lo que este script NO hace
--------------------------
No construye un modelo con estas caracteristicas. Solo las usa para **describir** los errores.
Convertirlas en features de un clasificador es una fase posterior y distinta, y este informe
deja escrita la condicion que habria que cumplir para que valiera la pena.

Ausencia de leakage
-------------------
Solo se leen `text` y las etiquetas del OOF canonico de V03. El test no se lee. Ninguna
caracteristica usa las etiquetas, de modo que el diagnostico no puede filtrar informacion.

Uso
---
    /home/carlos/venv/bin/python experiments/v28_01_diagnostico_errores.py
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

EXPERIMENTO = "V28-01 (diagnostico de errores de V03 por etiqueta, condicionado por rasgos del texto)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v28_01_diagnostico_errores.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v28_01_diagnostico_errores.md"
OOF_CANONICO = PROJECT_ROOT / "features" / "oof_v03_final.npz"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"

ETIQUETAS = ("HS", "TR", "AG")
CORRECTED_CV_V03 = 0.7772114525165721
UMBRAL_MINIMO = 0.06


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V28-01] {mensaje}", flush=True)


def memoria_pico_mb() -> float:
    """Resident set maximum del proceso, en megabytes.

    Returns:
        La memoria maxima residente en MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


RE_MENCION = re.compile(r"@\w+")
RE_HASHTAG = re.compile(r"#\w+")
RE_URL = re.compile(r"https?://|www\.")
RE_ALFA = re.compile("[A-Za-zA-Z\\u00C0-\\u00FF]")
RE_DIGITO = re.compile(r"\d")
# El signo de interrogacion de apertura se escribe como escape para que el fuente siga
# siendo ASCII puro sin cambiar el comportamiento de la clase de caracteres.
RE_PUNCT = re.compile(r"[!\u00bf?.,;:()\[\]{}\"'`\-_/\\|+*=<>$%&]")
RE_MAYUS = re.compile("[A-Z\\u00C0-\\u00DC]")
RE_REPETIDO = re.compile(r"(.)\1{2,}")


def rasgos_de(texto: str) -> dict[str, float]:
    """Calcula los rasgos textuales de un documento.

    Todos los rasgos derivan **solo** del campo `text`. Ninguno usa etiquetas, identificadores
    ni metadatos externos, de modo que el diagnostico no puede filtrar informacion del
    objetivo.

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


RASGOS = sorted(rasgos_de("texto de prueba @user #tag http://x.com 123 !!! ABC").keys())


def main() -> None:
    """Ejecuta el diagnostico de errores por etiqueta.

    Returns:
        None.
    """
    inicio = time.time()
    log(f"{EXPERIMENTO}. NINGUNA submission, NINGUN modelo en este script")

    log("1/6 datos y OOF canonico de V03")
    train = read_train()
    particion = pd.read_csv(PARTICION)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    oof = np.load(OOF_CANONICO, allow_pickle=True)
    y_oof = np.asarray(oof["y_true"], dtype=np.int64)
    pred_v03 = np.asarray(oof["pred"], dtype=np.int64)
    ids_oof = np.asarray(oof["ids"], dtype=np.int64)
    mapa = dict(zip(desarrollo["id"].astype(int), desarrollo["text"]))
    textos = pd.Series([mapa[int(i)] for i in ids_oof], name="text")
    log(f"   desarrollo {len(desarrollo)} | filas del OOF {len(textos)} | {len(RASGOS)} rasgos")
    log("   el test no se lee; ninguna caracteristica usa etiquetas")

    log("2/6 calculo de rasgos")
    tabla = pd.DataFrame([rasgos_de(t) for t in textos.tolist()], columns=RASGOS)
    log(f"   {tabla.shape[0]} filas x {tabla.shape[1]} rasgos")

    log("3/6 error de V03 por etiqueta y su estructura")
    estructura: dict[str, Any] = {}
    for j, etiqueta in enumerate(ETIQUETAS):
        y = y_oof[:, j]
        p = pred_v03[:, j]
        error = p != y
        tp = int(np.count_nonzero((p == 1) & (y == 1)))
        fp = int(np.count_nonzero((p == 1) & (y == 0)))
        fn = int(np.count_nonzero((p == 0) & (y == 1)))
        tn = int(np.count_nonzero((p == 0) & (y == 0)))
        comunes = int(np.count_nonzero(error))
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        estructura[etiqueta] = {
            "f1_v03": float(f1),
            "n_positivos_reales": int(np.count_nonzero(y)),
            "n_positivos_predichos": int(np.count_nonzero(p)),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_errores": comunes,
            "tasa_de_error": float(comunes / len(y)),
            "error_entre_positivos": float(fn / max(np.count_nonzero(y), 1)),
            "error_entre_negativos": float(fp / max(np.count_nonzero(y == 0), 1)),
        }
        log(
            f"   {etiqueta}: F1 {f1:.4f} | errores {comunes} ({comunes / len(y):.4f}) | "
            f"FN {fn} sobre {np.count_nonzero(y)} positivos ({fn / max(np.count_nonzero(y), 1):.4f}) | "
            f"FP {fp}"
        )
    # Interseccion de errores: que etiquetas fallan a la vez.
    matrices_error = np.stack([(pred_v03[:, j] != y_oof[:, j]) for j in range(3)], axis=1)
    n_ninguno = int(np.count_nonzero(~matrices_error.any(axis=1)))
    n_todas = int(np.count_nonzero(matrices_error.all(axis=1)))
    n_exactamente_una = int(np.count_nonzero(matrices_error.sum(axis=1) == 1))
    log(
        f"   interseccion: 0 etiquetas {n_ninguno} | 1 etiqueta {n_exactamente_una} | "
        f"3 etiquetas {n_todas}"
    )
    estructura["interseccion_de_errores"] = {
        "ninguna_etiqueta": n_ninguno,
        "exactamente_una": n_exactamente_una,
        "exactamente_dos": int(np.count_nonzero(matrices_error.sum(axis=1) == 2)),
        "las_tres": n_todas,
        "fractions": {
            "ninguna_etiqueta": n_ninguno / len(y_oof),
            "exactamente_una": n_exactamente_una / len(y_oof),
            "exactamente_dos": float(np.count_nonzero(matrices_error.sum(axis=1) == 2)) / len(y_oof),
            "las_tres": n_todas / len(y_oof),
        },
        "solapamiento_de_errores_entre_etiquetas": _solapamiento(matrices_error),
    }

    log("4/6 que rasgo distingue V03 wrong de V03 correct, por etiqueta")
    tabla["__error_hs"] = matrices_error[:, 0].astype(int)
    tabla["__error_tr"] = matrices_error[:, 1].astype(int)
    tabla["__error_ag"] = matrices_error[:, 2].astype(int)
    discriminacion: dict[str, Any] = {}
    for j, etiqueta in enumerate(ETIQUETAS):
        col = f"__error_{etiqueta.lower()}"
        mal = tabla.loc[tabla[col] == 1]
        bien = tabla.loc[tabla[col] == 0]
        por_rasgo: dict[str, Any] = {}
        for rasgo in RASGOS:
            a = mal[rasgo].to_numpy(dtype=float)
            b = bien[rasgo].to_numpy(dtype=float)
            if a.std() < 1e-12 and b.std() < 1e-12:
                continue
            _u, pval = stats.mannwhitneyu(a, b, alternative="two-sided")
            # SciPy 1.17 devuelve un objeto con `.statistic`; en versiones antiguas era un float.
            bruto = stats.pointbiserialr((tabla[col] == 1).astype(int), tabla[rasgo])
            punto = float(getattr(bruto, "statistic", bruto))
            pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
            d_cohen = (a.mean() - b.mean()) / pooled if pooled > 1e-12 else 0.0
            por_rasgo[rasgo] = {
                "media_error": float(a.mean()),
                "media_correcto": float(b.mean()),
                "diferencia": float(a.mean() - b.mean()),
                "d_cohen": float(d_cohen),
                "correlacion_punto_biserial": punto,
                "p_mannwhitney": float(pval),
            }
        orden = sorted(por_rasgo.items(), key=lambda kv: -abs(kv[1]["d_cohen"]))
        discriminacion[etiqueta] = {
            "rasgos_ordenados_por_efecto": orden[:8],
            "n_rasgos_con_p_menor_005": sum(
                1 for v in por_rasgo.values() if v["p_mannwhitney"] < 0.005
            ),
        }
        log(f"   {etiqueta}: rasgos con p < 0.005 = {discriminacion[etiqueta]['n_rasgos_con_p_menor_005']}")
        for nombre, d in orden[:5]:
            log(
                f"      {nombre:<24} error {d['media_error']:>8.3f} vs correcto "
                f"{d['media_correcto']:>8.3f} | d {d['d_cohen']:+.3f} | p {d['p_mannwhitney']:.2e}"
            )

    log("5/6 heterogeneidad: se parecen los errores de las tres etiquetas?")
    # Correlacion de los indicadores de error entre etiquetas: si HS y TR fallan juntos mas
    # de lo que el azar, no hay margen para un especialista por etiqueta.
    heterogeneidad: dict[str, Any] = {}
    for a in range(3):
        for b in range(a + 1, 3):
            ea = matrices_error[:, a].astype(float)
            eb = matrices_error[:, b].astype(float)
            r = float(np.corrcoef(ea, eb)[0, 1])
            n11 = int(np.count_nonzero(ea * eb))
            esperado = float(ea.mean() * eb.mean() * len(ea))
            jaccard = n11 / max(int(np.count_nonzero((ea + eb) > 0)), 1)
            heterogeneidad[f"{ETIQUETAS[a]}_{ETIQUETAS[b]}"] = {
                "correlacion_phi": r,
                "n_ambos_error": n11,
                "esperado_si_independientes": esperado,
                "exceso_sobre_lo_esperado": n11 - esperado,
                "jaccard": float(jaccard),
            }
            log(
                f"   {ETIQUETAS[a]}-{ETIQUETAS[b]}: phi {r:+.4f} | ambos {n11} frente a "
                f"{esperado:.0f} esperados | Jaccard {jaccard:.4f}"
            )
    # la heterogeneidad va al nivel superior del informe, no anidada en la estructura

    log("6/6 subconjuntos donde un especialista podria tener sentido")
    # Region del espacio: errores donde la mencion o el hashtag aparecen al principio del
    # texto, que es donde el contexto suele estar implicito. Se exige un minimo de filas.
    subconjuntos = _subconjuntos(tabla, matrices_error)
    for s in subconjuntos:
        log(
            f"   {s['nombre']:<44} filas {s['n_filas']:>4} ({s['n_filas'] / len(y_oof):.4f}) | "
            f"error por etiqueta: "
            + " ".join(f"{e} {s['tasa_error'][e]:.4f}" for e in ETIQUETAS)
        )
    estructura["subconjuntos"] = subconjuntos

    informe = {
        "familia": "ERROR_DIAGNOSTIC",
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "DIAGNOSTIC_ONLY",
        "pregunta": (
            "donde falla V03, si hay heterogeneidad entre HS, TR y AG, y si existe un "
            "subconjunto distinguible donde un modelo distinto tendria sentido"
        ),
        "alcance": (
            "no se entrena ningun modelo. Los rasgos se usan solo para DESCRIBIR los errores, "
            "no como features de un clasificador"
        ),
        "rasgos": RASGOS,
        "estructura_del_error": estructura,
        "heterogeneidad_entre_etiquetas": heterogeneidad,
        "discriminacion_por_rasgo": discriminacion,
        "informe": {
            "experimento": EXPERIMENTO,
            "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "v03_corrected_cv": CORRECTED_CV_V03,
        },
        "gastos": {
            "submissions": 0,
            "modelos_entrenados": 0,
            "memoria_pico_mb": round(memoria_pico_mb(), 1),
            "segundos": round(time.time() - inicio, 1),
        },
    }
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _informe_md(informe)
    log("   informe escrito en reports/v28_01_diagnostico_errores.{json,md}")
    log(f"   memoria pico {memoria_pico_mb():.0f} MB | tiempo {round(time.time() - inicio, 1)} s")
    log("VEREDICTO: diagnostico completado, 0 modelos, 0 submissions")


def _solapamiento(matrices_error: np.ndarray) -> dict[str, Any]:
    """Cuanto se solapan los errores entre pares de etiquetas.

    Args:
        matrices_error: matriz booleana de errores, una columna por etiqueta.

    Returns:
        Diccionario con la matriz de interseccion y el solapamiento esperado.
    """
    saliente: dict[str, Any] = {}
    n = matrices_error.shape[0]
    for a in range(3):
        for b in range(a + 1, 3):
            ea = matrices_error[:, a]
            eb = matrices_error[:, b]
            n11 = int(np.count_nonzero(ea & eb))
            esperado = float(ea.mean() * eb.mean() * n)
            saliente[f"{ETIQUETAS[a]}_{ETIQUETAS[b]}"] = {
                "ambos": n11,
                "esperado_bajo_independencia": esperado,
                "razon_de_coincidencia": n11 / esperado if esperado > 0 else 0.0,
            }
    return saliente


def _subconjuntos(tabla: pd.DataFrame, matrices_error: np.ndarray) -> list[dict[str, Any]]:
    """Define regiones del espacio textual y mide el error de V03 dentro de cada una.

    Args:
        tabla: tabla de rasgos por fila.
        matrices_error: matriz booleana de errores, una columna por etiqueta.

    Returns:
        Lista de subconjuntos con su tamano y su tasa de error.
    """
    n = len(tabla)
    masks: dict[str, np.ndarray] = {
        "tweets muy cortos (<= 4 tokens)": tabla["es_muy_corto"].to_numpy() > 0,
        "tweets largos (>= 25 tokens)": tabla["es_largo"].to_numpy() > 0,
        "con URL": tabla["tiene_url"].to_numpy() > 0,
        "con mencion": tabla["tiene_mencion"].to_numpy() > 0,
        "con hashtag": tabla["tiene_hashtag"].to_numpy() > 0,
        "con @ en el primer tercio": (tabla["pos_primera_mencion"].to_numpy() >= 0)
        & (tabla["pos_primera_mencion"].to_numpy() < 0.33),
        "sin ninguna marca (@, #, URL)": (tabla["tiene_mencion"].to_numpy() == 0)
        & (tabla["tiene_hashtag"].to_numpy() == 0)
        & (tabla["tiene_url"].to_numpy() == 0),
        "alta puntuacion (> 0.10 por caracter)": tabla["frac_puntuacion"].to_numpy() > 0.10,
        "casi todos mayusculas (frac > 0.7)": tabla["frac_mayusculas"].to_numpy() > 0.70,
        "densidad lexica alta (> 0.90)": tabla["densidad_lexica"].to_numpy() > 0.90,
    }
    salida: list[dict[str, Any]] = []
    for nombre, mask in masks.items():
        m = int(np.count_nonzero(mask))
        if m == 0:
            continue
        tasa = {
            ETIQUETAS[j]: float(np.count_nonzero(matrices_error[mask, j]) / m)
            for j in range(3)
        }
        salida.append({
            "nombre": nombre,
            "n_filas": m,
            "fraccion": m / n,
            "tasa_error": tasa,
            "tasa_error_media": float(np.mean(list(tasa.values()))),
            "suficientemente_grande": bool(m >= n * UMBRAL_MINIMO),
        })
    salida.sort(key=lambda s: -s["n_filas"])
    return salida


def _informe_md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.

    Returns:
        None.
    """
    e = informe["estructura_del_error"]
    disc = informe["discriminacion_por_rasgo"]
    het = informe["heterogeneidad_entre_etiquetas"]
    inter = e["interseccion_de_errores"]
    lineas = [
        "# V28-01: diagnostico de errores de V03 por etiqueta",
        "",
        f"Generado por `experiments/v28_01_diagnostico_errores.py` ({EXPERIMENTO}).",
        "",
        f"**Estado `{informe['estado']}`. Alcance: {informe['alcance']}**",
        "",
        "## 1. La estructura del error por etiqueta",
        "",
        "| etiqueta | F1 V03 | errores | tasa | FN / positivos | FP | TP | TN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for etiqueta in ETIQUETAS:
        d = e[etiqueta]
        lineas.append(
            f"| {etiqueta} | {d['f1_v03']:.4f} | {d['n_errores']} | {d['tasa_de_error']:.4f} | "
            f"{d['fn']} / {d['n_positivos_reales']} ({d['error_entre_positivos']:.4f}) | "
            f"{d['fp']} | {d['tp']} | {d['tn']} |"
        )
    f = inter["fractions"]
    lineas += [
        "",
        "## 2. Los errores de las tres etiquetas, juntos y separados",
        "",
        "| numero de etiquetas falladas | filas | fraccion |",
        "|---:|---:|---:|",
        f"| 0 | {inter['ninguna_etiqueta']} | {f['ninguna_etiqueta']:.4f} |",
        f"| 1 | {inter['exactamente_una']} | {f['exactamente_una']:.4f} |",
        f"| 2 | {inter['exactamente_dos']} | {f['exactamente_dos']:.4f} |",
        f"| 3 | {inter['las_tres']} | {f['las_tres']:.4f} |",
        "",
        "El punto que decide si tiene sentido un especialista por etiqueta es cuanto se solapan",
        "los errores. Si HS y TR fallan sobre los mismos textos, un especialista por etiqueta",
        "no puede ganar nada, porque solo puede cambiar las decisiones donde ambos fallan.",
        "",
        "| par | correlacion phi | ambos fallan | esperados si independientes | exceso | Jaccard |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for clave, d in het.items():
        lineas.append(
            f"| {clave} | {d['correlacion_phi']:+.4f} | {d['n_ambos_error']} | "
            f"{d['esperado_si_independientes']:.0f} | {d['exceso_sobre_lo_esperado']:+.0f} | "
            f"{d['jaccard']:.4f} |"
        )
    lineas += [
        "",
        "## 3. Que rasgo del texto distingue V03 wrong de V03 correct",
        "",
        "Cohen d es la diferencia tipada de medias en desviaciones estandar. Se listan los cinco",
        "rasgos de mayor efecto absoluto para cada etiqueta.",
        "",
    ]
    for etiqueta in ETIQUETAS:
        lineas.append(f"### {etiqueta}  (F1 {e[etiqueta]['f1_v03']:.4f})")
        lineas.append("")
        lineas.append("| rasgo | media si error | media si correcto | d | p |")
        lineas.append("|---|---:|---:|---:|---:|")
        for nombre, d in disc[etiqueta]["rasgos_ordenados_por_efecto"][:5]:
            lineas.append(
                f"| {nombre} | {d['media_error']:.4f} | {d['media_correcto']:.4f} | "
                f"{d['d_cohen']:+.4f} | {d['p_mannwhitney']:.2e} |"
            )
        lineas.append("")
    lineas += [
        "## 4. Regiones del espacio textual",
        "",
        "Tamano de cada region y tasa de error de V03 dentro. Un subconjunto solo justifica un",
        f"especialista si es grande (al menos el {int(UMBRAL_MINIMO * 100)} % del desarrollo) y ademas",
        "difiere de la tasa global.",
        "",
        "| region | filas | fraccion | " + " | ".join(f"error {e}" for e in ETIQUETAS) + " | suficiente |",
        "|---|---:|---:|" + "---:|" * 3 + ":-:|",
    ]
    for s in e["subconjuntos"]:
        marcas = " | ".join(f"{s['tasa_error'][et]:.4f}" for et in ETIQUETAS)
        lineas.append(
            f"| {s['nombre']} | {s['n_filas']} | {s['fraccion']:.4f} | {marcas} | "
            f"{'si' if s['suficientemente_grande'] else 'no'} |"
        )
    g = informe["gastos"]
    lineas += [
        "",
        "## 5. Lo que este diagnostico no hace",
        "",
        "- No entrena ningun modelo. Los rasgos describen los errores, no alimentan un clasificador.",
        "- No lee el test.",
        "- No usa etiquetas para calcular ningun rasgo, de modo que no puede filtrar informacion.",
        "",
        (f"Coste: {g['segundos']} s, memoria {g['memoria_pico_mb']:.0f} MB, "
        f"{g['modelos_entrenados']} modelos, {g['submissions']} submissions."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
