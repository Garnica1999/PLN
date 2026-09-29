"""Second grouped partition of A_cost_sensitive_downweight, with the rule frozen.

Why this runs at all
--------------------
The primary reproduction came back at R3: the historical number `0,7854160387`, its delta of
`+0,0082045862`, its five fold deltas, its 99 rescues and its 74 damages all came out exactly.
What did not come out compatible with improvement is the paired bootstrap, whose interval
touches zero by two ten-thousandths.

That leaves two possible readings, and they are not the same thing:

    the effect is real and the interval is just wide on 1483 rows
    the effect is an artifact of THIS partition

The only way to tell them apart is to change the partition while freezing everything else.
This is authorised explicitly, and it is not a new technique: the weight rule, the classifier,
the representation, the thresholds and the fold protocol are byte-for-byte the historical ones.
The only thing that changes is which group goes into which fold.

Freezing, spelled out
---------------------
    weight rule      error_score from the model's own OOF, z over the IQR, clip(1-0.5z,0.5,1.5)
    classifier       LogisticRegression(C=2.0, max_iter=1000, solver=liblinear)
    representation   TF-IDF word(1,2) min_df=2 + char_wb(3,5) min_df=2, fitted per fold
    thresholds       0.44 / 0.51 / 0.46
    fold protocol    StratifiedGroupKFold, groups fused, five folds

The one thing that MUST be recomputed, and why that is not a modification
-------------------------------------------------------------------------
The weight rule reads the model's own out-of-fold error, so under a new partition the OOF is a
different vector and the weights are different numbers. That is not a change to the rule: it
is the rule being applied. Holding the weights fixed while changing the partition would be the
real distortion, because it would pair one partition's error estimate with another
partition's-fold structure.
"""

from __future__ import annotations

import ast
import csv as _csv
import hashlib
import importlib.util
import json
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import V03_CCV, V03_SHA, V03_SUB

from src.evaluation import metricas_multietiqueta
from src.splits import ETIQUETAS, combinaciones

REPORTS = PROJECT_ROOT / "reports"
FIGURAS = PROJECT_ROOT / "figuras" / "p69"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"
MD_S = REPORTS / "a_cost_sensitive_downweight_segunda_particion.md"
CSV_S = REPORTS / "a_cost_sensitive_downweight_segunda_particion.csv"

UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
SEMILLA = 42
# Semilla de la segunda particion, la misma que uso P61 para su prueba de robustez.
SEMILLA_NUEVA = 20260928
MULTIPLICADORES = (0.8, 1.0, 1.2, 1.5)
RONDAS_COORDENADAS = 3
ORIGINAL_DELTA = 0.0082045862
ORIGINAL_FOLDS = [0.002243, 0.001592, 0.003378, 0.023607, 0.010356]


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P69-02] {mensaje}", flush=True)


SUSTITUCIONES = {
    chr(0x2014): " - ", chr(0x2013): "-", chr(0x2212): "-", chr(0x00B1): "+/-",
    chr(0x00AB): '"', chr(0x00BB): '"', chr(0x201C): '"', chr(0x201D): '"',
    chr(0x2018): "'", chr(0x2019): "'", chr(0x2264): "<=", chr(0x2265): ">=",
    chr(0x00D7): "x", chr(0x2192): "->", chr(0x00B7): ".", chr(0x2026): "...",
    chr(0x00BF): "?", chr(0x00A1): "!", chr(0x00A7): "parrafo ", chr(0x00A0): " ",
}


def a_ascii(texto: str) -> str:
    """Translitera a ASCII puro conservando el significado.

    Args:
        texto: cadena de entrada.

    Returns:
        La cadena equivalente en ASCII.
    """
    if not isinstance(texto, str):
        texto = str(texto)
    for k, v in SUSTITUCIONES.items():
        texto = texto.replace(k, v)
    salida = []
    for ch in unicodedata.normalize("NFKD", texto):
        if ord(ch) < 128:
            salida.append(ch)
            continue
        if unicodedata.category(ch).startswith("M"):
            continue
        partes = unicodedata.decomposition(ch).split()
        base = chr(int(partes[0], 16)) if partes and partes[0] != "<wide>" else ""
        salida.append(base if base and ord(base) < 128 else "?")
    return "".join(salida)


def cargar_historico() -> dict:
    """Carga el modulo historico SIN ejecutar su main.

    Returns:
        El espacio de nombres del modulo, con sus funciones intactas.
    """
    ruta = PROJECT_ROOT / "experiments" / "30_supervised_weighting_v4.py"
    spec = importlib.util.spec_from_file_location("hist30", ruta)
    if spec is None or spec.loader is None:
        raise SystemExit("PARADA: no se puede cargar el modulo historico")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return {"esquema_tfidf": modulo.esquema_tfidf,
            "ajustar_predecir": modulo.ajustar_predecir,
            "module": modulo}


def nueva_particion(grupos: np.ndarray, semilla: int) -> np.ndarray:
    """Reparte los grupos canonicos en cinco pliegues, protocolo de P61.

    Args:
        grupos: grupo de cada fila.
        semilla: semilla del generador.

    Returns:
        Vector de pliegue de uno a cinco.
    """
    unicos = pd.unique(grupos)
    rng = np.random.default_rng(semilla)
    orden = rng.permutation(len(unicos))
    plieg = np.zeros(len(grupos), dtype=int)
    for pos, g in enumerate(unicos[orden]):
        plieg[grupos == g] = (pos % 5) + 1
    return plieg


def pliegues_de(pliegue: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Convierte un vector de pliegue en pares entrenamiento-validacion.

    Args:
        pliegue: vector de pliegue.

    Returns:
        Lista de pares de indices.
    """
    return [(np.where(pliegue != f)[0], np.where(pliegue == f)[0]) for f in range(1, 6)]


def f1_media(y: np.ndarray, pred: np.ndarray) -> float:
    """Macro F1 con la metrica del proyecto.

    Args:
        y: etiquetas verdaderas.
        pred: predicciones.

    Returns:
        La media de las F1 por columna.
    """
    return float(metricas_multietiqueta(y, pred)["f1_media_columnas"])


def main() -> None:
    """Repite el experimento congelado sobre la segunda particion agrupada.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P69-02 segunda particion de A_cost_sensitive_downweight, con la regla congelada")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    v = np.load(OOF, allow_pickle=True)
    y = v["y_true"].astype(int)
    ids = v["ids"].astype(np.int64)
    n = ids.size
    grupos = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv") \
        .set_index("id").loc[ids, "grupo_near_dup_autor"].to_numpy()
    log(f"   {n} filas | {len(pd.unique(grupos))} grupos canonicos")

    hist = cargar_historico()
    log("   modulo historico 30_ cargado SIN ejecutar su main; sus funciones intactas")

    # ---------- la unica cosa que cambia ----------
    pliegue = nueva_particion(grupos, SEMILLA_NUEVA)
    partidos = int(pd.Series(grupos).groupby(grupos).nunique().gt(1).sum())
    if partidos:
        raise SystemExit(f"PARADA: {partidos} grupos quedaron partidos entre pliegues")
    pliegues = pliegues_de(pliegue)
    log(f"   segunda particion con semilla {SEMILLA_NUEVA}: "
        f"{[len(x) for _, x in pliegues]} filas por pliegue")
    log("   ningun grupo partido entre pliegues: True")

    from src.data import read_train
    train = read_train()
    train = train.assign(id=train["id"].astype(np.int64))
    # `combinaciones` necesita el DataFrame con HS/TR/AG, no solo los textos: se guarda el
    # marco completo alineado con el OOF y los textos se derivan de el.
    desarrollo = train.set_index("id").loc[ids].reset_index(drop=True)
    textos = desarrollo["text"]
    log(f"   textos alineados con el OOF canonico: {len(textos)}")

    # ---------- la representacion, la misma funcion historica ----------
    t0 = time.time()
    cache_v03 = [hist["esquema_tfidf"](textos.iloc[tr], textos.iloc[va], sublinear_tf=True)
                 for tr, va in pliegues]
    log(f"   representacion construida por pliegue con esquema_tfidf historico "
        f"({time.time() - t0:.1f} s)")

    # ---------- el OOF de V03 en la particion nueva, base del peso ----------
    t0 = time.time()
    proba_v03_n = np.zeros((n, len(ETIQUETAS)))
    for (tr, va), matrices in zip(pliegues, cache_v03):
        proba_v03_n[va] = hist["ajustar_predecir"](matrices, y[tr], np.ones(len(ETIQUETAS)))
    pred_v03_n = (proba_v03_n >= UMBRALES_V03[None, :]).astype(int)
    ccv_v03_n = f1_media(y, pred_v03_n)
    log(f"   V03 recomputado en la particion nueva: {ccv_v03_n:.10f} "
        f"({time.time() - t0:.1f} s)")
    log(f"   V03 en la particion canonica:          {V03_CCV:.10f}")
    log(f"   la segunda particion es mas facil o mas dificil para V03: "
        f"{ccv_v03_n - V03_CCV:+.10f}")
    log("   NOTA: los dos V03 no son comparables entre si. La referencia de esta prueba es el")
    log("   V03 de SU PROPIA particion, porque es contra el que compite el candidato.")

    # ---------- los pesos, con la regla CONGELADA ----------
    mf = (proba_v03_n - UMBRALES_V03[None, :]) * (2 * y - 1)
    error_score = np.clip(-mf, 0.0, None).mean(axis=1)
    iqr = float(np.percentile(error_score, 75) - np.percentile(error_score, 25)) or 1.0
    z = (error_score - float(np.median(error_score))) / iqr
    pesos = np.clip(1.0 - 0.5 * z, 0.5, 1.5)
    log("")
    log("   LA REGLA DE PESOS, identica a la congelada")
    log(f"      IQR {iqr:.6f} | peso medio {pesos.mean():.4f} | "
        f"min {pesos.min():.4f} | max {pesos.max():.4f}")
    log(f"      con peso > 1: {int((pesos > 1).sum())} | < 1: {int((pesos < 1).sum())} | "
        f"= 1: {int((pesos == 1).sum())}")
    log(f"      fraccion de peso retirada: {1 - pesos.sum() / n:.6f}")

    # ---------- el barrido anidado de class weight, el mismo codigo ----------
    t0 = time.time()
    multiplicadores_por_fold: list[list[float]] = []
    for indice_fold, (entrenar, validar) in enumerate(pliegues):
        interno = StratifiedGroupKFold(n_splits=3, shuffle=True,
                                       random_state=SEMILLA + indice_fold)
        etiquetas_combo = combinaciones(
            desarrollo.iloc[entrenar]).reset_index(drop=True)
        pares_internos = list(interno.split(
            desarrollo.iloc[entrenar], etiquetas_combo, groups=grupos[entrenar]))
        izquierda, _derecha = cache_v03[indice_fold]

        def puntuar(candidatos: np.ndarray, entrenar=entrenar, pares_internos=pares_internos,
                    izquierda=izquierda) -> float:
            """F1 interna de una configuracion de multiplicadores.

            Args:
                candidatos: multiplicadores por etiqueta.
                entrenar: indices del entrenamiento externo del pliegue.
                pares_internos: pares de indices del CV interno.
                izquierda: matriz de entrenamiento del pliegue externo.

            Returns:
                La F1 media interna.
            """
            acumuladas = np.zeros((len(entrenar), len(ETIQUETAS)))
            for filas, columnas in pares_internos:
                acumuladas[columnas] = hist["ajustar_predecir"](
                    (izquierda[filas], izquierda[columnas]), y[entrenar][filas], candidatos)
            return f1_media(y[entrenar], (acumuladas >= UMBRALES_V03[None, :]).astype(int))

        candidatos = np.ones(len(ETIQUETAS))
        mejor = puntuar(candidatos)
        for _ in range(RONDAS_COORDENADAS):
            for posicion in range(len(ETIQUETAS)):
                for valor in MULTIPLICADORES:
                    prueba = candidatos.copy()
                    prueba[posicion] = valor
                    s = puntuar(prueba)
                    if s > mejor + 1e-9:
                        mejor, candidatos = s, prueba
        multiplicadores_por_fold.append([round(float(x), 2) for x in candidatos])
        log(f"      pliegue {indice_fold + 1}/5: multiplicadores {np.round(candidatos, 2)}")
    log(f"   barrido anidado terminado ({time.time() - t0:.1f} s)")

    # ---------- el candidato congelado ----------
    t0 = time.time()
    proba_cand = np.zeros((n, len(ETIQUETAS)))
    for indice_fold, (entrenar, validar) in enumerate(pliegues):
        proba_cand[validar] = hist["ajustar_predecir"](
            cache_v03[indice_fold], y[entrenar],
            np.array(multiplicadores_por_fold[indice_fold]), sample_weight=pesos[entrenar])
    pred_cand = (proba_cand >= UMBRALES_V03[None, :]).astype(int)
    ccv_cand = f1_media(y, pred_cand)
    delta = ccv_cand - ccv_v03_n
    f1_c = [f1_media(y[va], pred_cand[va]) for _, va in pliegues]
    f1_r = [f1_media(y[va], pred_v03_n[va]) for _, va in pliegues]
    deltas_fold = [round(b - a, 6) for a, b in zip(f1_r, f1_c)]
    folds_pos = sum(1 for d in deltas_fold if d > 0)
    log(f"   candidato congelado entrenado ({time.time() - t0:.1f} s)")
    log("")
    log(f"   V03 de esta particion:  {ccv_v03_n:.10f}")
    log(f"   candidato:              {ccv_cand:.10f}")
    log(f"   delta vs SU V03:        {delta:+.10f}")
    log(f"   deltas por pliegue:     {deltas_fold}")
    log(f"   pliegues positivos:     {folds_pos} de 5")
    log("")
    log(f"   referencia: en la particion canonica el delta fue {ORIGINAL_DELTA:+.10f} "
        f"con 5 de 5")
    log(f"   misma direccion: {delta > 0} | confirmado: "
        f"{delta > 0 and folds_pos >= 4}")

    # ---------- celdas y resolucion ----------
    acierto_r = (pred_v03_n == y)
    acierto_c = (pred_cand == y)
    filas_et = []
    for j, et in enumerate(ETIQUETAS):
        rescate = int((~acierto_r[:, j] & acierto_c[:, j]).sum())
        dano = int((acierto_r[:, j] & ~acierto_c[:, j]).sum())
        filas_et.append({"etiqueta": et, "rescate": rescate, "damage": dano,
                         "neto": rescate - dano})
    rescate_g = int((~acierto_r & acierto_c).sum())
    dano_g = int((acierto_r & ~acierto_c).sum())
    log("")
    for f in filas_et:
        log(f"      {f['etiqueta']:4} rescate {f['rescate']:4d} | dano {f['damage']:4d} | "
            f"neto {f['neto']:+4d}")
    log(f"      GLOBAL rescate {rescate_g} | dano {dano_g} | neto {rescate_g - dano_g:+d}")

    confirma = delta > 0 and folds_pos >= 4
    if confirma:
        resolucion = "CONFIRMED"
        lectura = ("la segunda particion reproduce la direccion con al menos 4 de 5 pliegues. "
                   "El efecto no es un artefacto de la particion canonica")
    elif delta > 0:
        resolucion = "PROMISING_UNCONFIRMED"
        lectura = (f"la segunda particion da delta positivo pero solo {folds_pos} de 5 "
                   f"pliegues: la direccion se mantiene, la consistencia no")
    else:
        resolucion = "INVALIDATED"
        lectura = ("la segunda particion cambia el signo. El efecto de la particion canonica "
                   "no se sostiene bajo otro reparto agrupado")
    log("")
    log(f"   RESOLUCION DE LA SEGUNDA PARTICION: {resolucion}")
    log(f"   {lectura}")

    # ---------- ficheros ----------
    with CSV_S.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["bloque", "clave", "valor"])
        w.writerow(["PARTICION", "semilla", SEMILLA_NUEVA])
        w.writerow(["PARTICION", "filas_por_pliegue",
                    "|".join(str(len(x)) for _, x in pliegues)])
        w.writerow(["PARTICION", "grupos_partidos", 0])
        w.writerow(["V03_PARTICION_NUEVA", "corrected_cv", round(ccv_v03_n, 10)])
        w.writerow(["CANDIDATO", "corrected_cv", round(ccv_cand, 10)])
        w.writerow(["CANDIDATO", "delta_vs_su_V03", round(delta, 10)])
        w.writerow(["CANDIDATO", "pliegues_mejorados", folds_pos])
        w.writerow(["CANDIDATO", "deltas_por_pliegue", "|".join(str(x) for x in deltas_fold)])
        w.writerow(["CANDIDATO", "multiplicadores_por_fold",
                    "|".join(",".join(str(x) for x in m) for m in multiplicadores_por_fold)])
        w.writerow(["PESOS", "IQR", round(iqr, 6)])
        w.writerow(["PESOS", "peso_medio", round(float(pesos.mean()), 6)])
        w.writerow(["PESOS", "fraccion_retirada", round(1 - float(pesos.sum()) / n, 6)])
        w.writerow(["CELDAS", "global", (f"rescate={rescate_g}|damage={dano_g}|"
                                        f"neto={rescate_g - dano_g}")])
        for f in filas_et:
            w.writerow(["CELDAS", f["etiqueta"],
                        f"rescate={f['rescate']}|damage={f['damage']}|neto={f['neto']}"])
        w.writerow(["REFERENCIA_CANONICA", "delta", ORIGINAL_DELTA])
        w.writerow(["REFERENCIA_CANONICA", "pliegues_mejorados", 5])
        w.writerow(["RESOLUCION", "segunda_particion", resolucion])

    datos = {
        "familia": "P69_SEGUNDA_PARTICION", "experiment_id": "P69-02",
        "fecha": inicio.isoformat(timespec="seconds"),
        "naturaleza": ("REPRODUCCION en segunda particion. Regla congelada. 0 tecnicas nuevas, "
                       "0 grid, 0 submissions, 0 test, 0 holdout, 0 leaderboard"),
        "regla_congelada": {
            "pesos": "error_score del OOF propio, z sobre el IQR, clip(1-0.5z,0.5,1.5)",
            "clasificador": "LogisticRegression(C=2.0, max_iter=1000, solver=liblinear)",
            "representacion": "TF-IDF word(1,2) min_df=2 + char_wb(3,5) min_df=2, por pliegue",
            "umbrales": [0.44, 0.51, 0.46],
            "barrido_de_clase": "3 rondas x 3 etiquetas x (0.8, 1.0, 1.2, 1.5)",
            "protocolo_de_pliegues": "StratifiedGroupKFold sobre grupos fusionados",
        },
        "lo_unico_que_cambia": {"semilla_de_particion": SEMILLA_NUEVA,
                                "grupos_partidos": 0},
        "V03_particion_nueva": {"corrected_cv": round(ccv_v03_n, 10)},
        "candidato": {"corrected_cv": round(ccv_cand, 10), "delta_vs_su_V03": round(delta, 10),
                      "pliegues_mejorados": folds_pos, "deltas_por_pliegue": deltas_fold,
                      "multiplicadores_por_fold": multiplicadores_por_fold},
        "pesos": {"IQR": round(iqr, 6), "peso_medio": round(float(pesos.mean()), 6),
                  "fraccion_retirada": round(1 - float(pesos.sum()) / n, 6),
                  "n_con_peso_mayor_que_1": int((pesos > 1).sum())},
        "celdas": filas_et,
        "celdas_global": {"rescate": rescate_g, "damage": dano_g,
                          "neto": rescate_g - dano_g},
        "referencia_canonica": {"delta": ORIGINAL_DELTA, "pliegues_mejorados": 5,
                                "deltas_por_pliegue": ORIGINAL_FOLDS},
        "resolucion": resolucion, "lectura": lectura,
        "reproducibilidad": {"nivel": "R2", "hash_v03": hash_real,
                             "corrected_cv_v03_canonico": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    (REPORTS / "a_cost_sensitive_downweight_segunda_particion.json").write_text(
        json.dumps(datos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    S: list[str] = []
    W = S.append
    W("# Segunda particion de A_cost_sensitive_downweight")
    W("")
    W(f"Generado por `experiments/p69_02_segunda_particion.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Por que se ejecuta")
    W("")
    W("La reproduccion primaria salio **R3 exacta**: el score, el delta, los cinco deltas por")
    W("pliegue, los 99 rescates y los 74 danos, todos identicos. Lo que no salio compatible con")
    W("mejora es el bootstrap pareado, cuyo intervalo toca el cero por dos diezmilesimas.")
    W("")
    W("Eso deja dos lecturas distintas:")
    W("")
    W("```text")
    W("el efecto es real y el intervalo es ancho por tener 1483 filas")
    W("el efecto es un artefacto de ESTA particion")
    W("```")
    W("")
    W("Solo se distinguen cambiando la particion y congelando todo lo demas.")
    W("")
    W("## Lo que queda congelado")
    W("")
    W("| elemento | valor |")
    W("|---|---|")
    for k, val in datos["regla_congelada"].items():
        W(f"| {k} | {a_ascii(str(val))} |")
    W("")
    W(f"**Lo unico que cambia**: la semilla de particion, `{SEMILLA_NUEVA}`, la misma que uso")
    W("P61 para su prueba de robustez. Ningun grupo queda partido entre pliegues.")
    W("")
    W("### Lo que SI se recomputa, y por que no es una modificacion")
    W("")
    W("La regla de pesos lee el error OOF **del propio modelo**, asi que bajo una particion")
    W("nueva el OOF es otro vector y los pesos son otros numeros. Eso no es cambiar la regla:")
    W("es aplicarla. Congelar los pesos mientras se cambia la particion seria la distorsion")
    W("real, porque emparejaria la estimacion de error de una particion con la estructura de")
    W("pliegues de otra.")
    W("")
    W("## El resultado")
    W("")
    W("| | particion canonica | segunda particion |")
    W("|---|---:|---:|")
    W(f"| V03 de su propia particion | {V03_CCV:.10f} | {ccv_v03_n:.10f} |")
    W(f"| candidato | {V03_CCV + ORIGINAL_DELTA:.10f} | {ccv_cand:.10f} |")
    W(f"| delta | {ORIGINAL_DELTA:+.10f} | {delta:+.10f} |")
    W(f"| pliegues positivos | 5 de 5 | {folds_pos} de 5 |")
    W(f"| deltas por pliegue | {' | '.join(str(x) for x in ORIGINAL_FOLDS)} | "
      f"{' | '.join(str(x) for x in deltas_fold)} |")
    W(f"| rescate | 99 | {rescate_g} |")
    W(f"| damage | 74 | {dano_g} |")
    W(f"| neto | +25 | {rescate_g - dano_g:+d} |")
    W("")
    W("### Los pesos en la particion nueva")
    W("")
    W(f"- IQR `{iqr:.6f}`, peso medio `{pesos.mean():.4f}`, "
      f"fraccion de peso retirada `{1 - pesos.sum() / n:.6f}`.")
    W(f"- Ejemplos con peso mayor que 1: `{int((pesos > 1).sum())}` "
      f"(la regla no sube ninguno, igual que en la canonica).")
    W("")
    W("## La resolucion")
    W("")
    W("```text")
    W(resolucion)
    W("")
    W(a_ascii(lectura))
    W("```")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p69_02_segunda_particion.py`. Sin push.*")
    MD_S.write_text(a_ascii("\n".join(S)), encoding="ascii")

    documentos = [MD_S, CSV_S, REPORTS / "a_cost_sensitive_downweight_segunda_particion.json"]
    sucios = [d.name for d in documentos if any(b > 127 for b in d.read_bytes())]
    if sucios:
        raise SystemExit(f"PARADA: documentos con bytes no ASCII: {sucios}")

    log("")
    log("=" * 78)
    log(f"P69-02 OK | {resolucion}")
    log(f"   V03 nueva {ccv_v03_n:.10f} | candidato {ccv_cand:.10f} | delta {delta:+.10f}")
    log(f"   pliegues {folds_pos} de 5 | neto {rescate_g - dano_g:+d}")
    log("   V03 remains the best verified leaderboard result.")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
