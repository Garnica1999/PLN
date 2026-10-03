"""Third grouped partition of A_cost_sensitive_downweight: the pre-registered confirmation.

The protocol of this run was written and committed BEFORE any result existed:
`docs/p69_especificacion_congelada.md`, commit `915991b`. The seed, the fold method, the
decision rule and the permutation design were all fixed in that document.

That ordering is the whole point. A partition run AFTER seeing another partition's result is
not independent evidence, whatever the seed looks like. The only thing that makes this
confirmatory rather than adaptive is that the specification came first and can be checked.

What is frozen, and it is everything except the fold assignment:

    V03                  frozen at 0,7772114525165721
    representation       TF-IDF word(1,2) min_df=2 + char_wb(3,5) min_df=2, per fold
    classifier           LogisticRegression(C=2.0, max_iter=1000, solver=liblinear)
    thresholds           0.44 / 0.51 / 0.46
    weight rule          error_score from the own OOF, z over IQR, clip(1-0.5z, 0.5, 1.5)
    class sweep          3 rounds x 3 labels x (0.8, 1.0, 1.2, 1.5)

What is recomputed, and it has to be: the weight rule reads the model's own out-of-fold error,
so a new partition gives a new OOF and therefore new weight numbers. Freezing the weights
while changing the partition would pair one partition's error estimate with another
partition's fold structure.
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

from src.splits import ETIQUETAS, combinaciones

REPORTS = PROJECT_ROOT / "reports"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"
MD_C = REPORTS / "a_cost_sensitive_downweight_confirmatory.md"
CSV_P = REPORTS / "a_cost_sensitive_downweight_partitions.csv"
JSON_C = REPORTS / "a_cost_sensitive_downweight_confirmatory.json"

UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
SEMILLA = 42
# Fijadas en docs/p69_especificacion_congelada.md, commit 915991b, ANTES de este resultado.
SEMILLA_TERCERA = 20261001
MULTIPLICADORES = (0.8, 1.0, 1.2, 1.5)
RONDAS_COORDENADAS = 3
# Las dos particiones anteriores, del registro.
CANONICA = {"nombre": "canonica", "V03": 0.7772114525, "delta": 0.0082045862,
            "pliegues": 5, "rescate": 99, "damage": 74,
            "deltas": [0.002243, 0.001592, 0.003378, 0.023607, 0.010356]}
SEGUNDA = {"nombre": "segunda", "V03": 0.7674405718, "delta": 0.0105253130,
           "pliegues": 5, "rescate": 86, "damage": 74,
           "deltas": [0.026302, 0.004643, 0.002975, 0.020868, 0.003209]}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P69-04] {mensaje}", flush=True)


def cargar_p69_02() -> dict:
    """Carga las utilidades ya verificadas del script de la segunda particion.

    Returns:
        Diccionario con las funciones.
    """
    ruta = PROJECT_ROOT / "experiments" / "p69_02_segunda_particion.py"
    spec = importlib.util.spec_from_file_location("p69_02", ruta)
    if spec is None or spec.loader is None:
        raise SystemExit("PARADA: no se puede cargar p69_02")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return {"cargar_historico": modulo.cargar_historico,
            "nueva_particion": modulo.nueva_particion,
            "pliegues_de": modulo.pliegues_de,
            "f1_media": modulo.f1_media}


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


def main() -> None:
    """Ejecuta la tercera particion con el protocolo pre-registrado.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P69-04 tercera particion confirmatoria")
    log("   protocolo pre-registrado en docs/p69_especificacion_congelada.md, commit 915991b")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    u = cargar_p69_02()
    v = np.load(OOF, allow_pickle=True)
    y = v["y_true"].astype(int)
    ids = v["ids"].astype(np.int64)
    n = ids.size
    grupos = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv") \
        .set_index("id").loc[ids, "grupo_near_dup_autor"].to_numpy()
    log(f"   {n} filas | {len(pd.unique(grupos))} grupos canonicos")

    pliegue = u["nueva_particion"](grupos, SEMILLA_TERCERA)
    partidos = int(pd.Series(grupos).groupby(grupos).nunique().gt(1).sum())
    if partidos:
        raise SystemExit(f"PARADA: {partidos} grupos quedaron partidos")
    pliegues = u["pliegues_de"](pliegue)
    log(f"   tercera particion, semilla {SEMILLA_TERCERA}: "
        f"{[len(x) for _, x in pliegues]} filas por pliegue")
    log("   ningun grupo partido entre pliegues: True")

    hist = u["cargar_historico"]()
    from src.data import read_train
    train = read_train().assign(id=lambda d: d["id"].astype(np.int64))
    desarrollo = train.set_index("id").loc[ids].reset_index(drop=True)
    textos = desarrollo["text"]

    t0 = time.time()
    cache = [hist["esquema_tfidf"](textos.iloc[tr], textos.iloc[va], sublinear_tf=True)
             for tr, va in pliegues]
    proba_v03 = np.zeros((n, len(ETIQUETAS)))
    for (tr, va), m in zip(pliegues, cache):
        proba_v03[va] = hist["ajustar_predecir"](m, y[tr], np.ones(len(ETIQUETAS)))
    pred_v03 = (proba_v03 >= UMBRALES_V03[None, :]).astype(int)
    ccv_v03 = u["f1_media"](y, pred_v03)
    log(f"   V03 recomputado en la tercera particion: {ccv_v03:.10f} "
        f"({time.time() - t0:.1f} s)")

    mf = (proba_v03 - UMBRALES_V03[None, :]) * (2 * y - 1)
    error_score = np.clip(-mf, 0.0, None).mean(axis=1)
    iqr = float(np.percentile(error_score, 75) - np.percentile(error_score, 25)) or 1.0
    z = (error_score - float(np.median(error_score))) / iqr
    pesos = np.clip(1.0 - 0.5 * z, 0.5, 1.5)
    log(f"   regla de pesos congelada | IQR {iqr:.6f} | peso medio {pesos.mean():.4f} | "
        f">1: {int((pesos > 1).sum())} | <1: {int((pesos < 1).sum())}")

    t0 = time.time()
    multiplicadores: list[list[float]] = []
    for indice_fold, (entrenar, validar) in enumerate(pliegues):
        interno = StratifiedGroupKFold(n_splits=3, shuffle=True,
                                       random_state=SEMILLA + indice_fold)
        etiquetas_combo = combinaciones(desarrollo.iloc[entrenar]).reset_index(drop=True)
        pares = list(interno.split(desarrollo.iloc[entrenar], etiquetas_combo,
                                   groups=grupos[entrenar]))
        izquierda, _d = cache[indice_fold]

        def puntuar(candidatos: np.ndarray, entrenar=entrenar, pares=pares,
                    izquierda=izquierda) -> float:
            """F1 interna de una configuracion de multiplicadores.

            Args:
                candidatos: multiplicadores por etiqueta.
                entrenar: indices del entrenamiento externo.
                pares: pares del CV interno.
                izquierda: matriz de entrenamiento del pliegue externo.

            Returns:
                La F1 media interna.
            """
            acum = np.zeros((len(entrenar), len(ETIQUETAS)))
            for filas, columnas in pares:
                acum[columnas] = hist["ajustar_predecir"](
                    (izquierda[filas], izquierda[columnas]), y[entrenar][filas], candidatos)
            return u["f1_media"](y[entrenar], (acum >= UMBRALES_V03[None, :]).astype(int))

        cand = np.ones(len(ETIQUETAS))
        mejor = puntuar(cand)
        for _ in range(RONDAS_COORDENADAS):
            for posicion in range(len(ETIQUETAS)):
                for valor in MULTIPLICADORES:
                    prueba = cand.copy()
                    prueba[posicion] = valor
                    s = puntuar(prueba)
                    if s > mejor + 1e-9:
                        mejor, cand = s, prueba
        multiplicadores.append([round(float(x), 2) for x in cand])
        log(f"      pliegue {indice_fold + 1}/5: multiplicadores {np.round(cand, 2)}")
    log(f"   barrido anidado terminado ({time.time() - t0:.1f} s)")

    proba_cand = np.zeros((n, len(ETIQUETAS)))
    for indice_fold, (entrenar, validar) in enumerate(pliegues):
        proba_cand[validar] = hist["ajustar_predecir"](
            cache[indice_fold], y[entrenar],
            np.array(multiplicadores[indice_fold]), sample_weight=pesos[entrenar])
    pred_cand = (proba_cand >= UMBRALES_V03[None, :]).astype(int)
    ccv_cand = u["f1_media"](y, pred_cand)
    delta = ccv_cand - ccv_v03
    deltas_fold = [round(u["f1_media"](y[va], pred_cand[va])
                         - u["f1_media"](y[va], pred_v03[va]), 6) for _, va in pliegues]
    folds_pos = sum(1 for d in deltas_fold if d > 0)

    ar = (pred_v03 == y)
    ac = (pred_cand == y)
    filas_et = []
    for j, et in enumerate(ETIQUETAS):
        r = int((~ar[:, j] & ac[:, j]).sum())
        dd = int((ar[:, j] & ~ac[:, j]).sum())
        filas_et.append({"etiqueta": et, "rescate": r, "damage": dd, "neto": r - dd,
                         "V03_ok_cand_ko": dd, "V03_ko_cand_ok": r,
                         "ambos_ok": int((ar[:, j] & ac[:, j]).sum()),
                         "ambos_ko": int((~ar[:, j] & ~ac[:, j]).sum())})
    resc = int((~ar & ac).sum())
    dan = int((ar & ~ac).sum())

    log("")
    log(f"   V03:       {ccv_v03:.10f}")
    log(f"   candidato: {ccv_cand:.10f}")
    log(f"   delta:     {delta:+.10f}")
    log(f"   deltas por pliegue: {deltas_fold}")
    log(f"   pliegues positivos: {folds_pos} de 5")
    log(f"   rescate {resc} | dano {dan} | neto {resc - dan:+d}")
    log("")
    for f in filas_et:
        log(f"      {f['etiqueta']:4} rescate {f['rescate']:4d} | dano {f['damage']:4d} | "
            f"neto {f['neto']:+4d}")

    # ---------- la tabla de las tres particiones ----------
    tercera = {"nombre": "tercera", "V03": round(ccv_v03, 10), "delta": round(delta, 10),
               "pliegues": folds_pos, "rescate": resc, "damage": dan,
               "deltas": deltas_fold}
    todas = [CANONICA, SEGUNDA, tercera]
    deltas_tres = np.array([t["delta"] for t in todas])
    parciales_pos = int((deltas_tres > 0).sum())
    log("")
    log("   LAS TRES PARTICIONES")
    log(f"      {'particion':12}{'V03':>16}{'delta':>16}{'folds+':>8}{'rescate':>9}"
        f"{'damage':>8}{'neto':>7}")
    for t, d in zip(todas, (CANONICA["deltas"], SEGUNDA["deltas"], deltas_fold)):
        log(f"      {t['nombre']:12}{t['V03']:>16.10f}{t['delta']:>+16.10f}"
            f"{t['pliegues']:>6}/5{t['rescate']:>9}{t['damage']:>8}"
            f"{t['rescate'] - t['damage']:>+7}")
    log("")
    log(f"      media    {deltas_tres.mean():+.10f}")
    log(f"      mediana  {np.median(deltas_tres):+.10f}")
    log(f"      minimo   {deltas_tres.min():+.10f}")
    log(f"      maximo   {deltas_tres.max():+.10f}")
    log(f"      particiones positivas: {parciales_pos} de 3")

    # ---------- la regla de decision pre-registrada ----------
    log("")
    log("   REGLA DE DECISION PRE-REGISTRADA")
    if parciales_pos == 3 and folds_pos >= 4:
        caso, resolucion = "CASO A", "PROMISING_UNCONFIRMED"
        lectura = ("tres de tres particiones positivas y la tercera con "
                   f"{folds_pos} de 5 pliegues. Se pasa al test de permutacion")
    elif parciales_pos == 2:
        caso, resolucion = "CASO B", "MARGINAL"
        lectura = ("dos de tres particiones positivas: la tercera no confirma con claridad. "
                   "Se detiene la investigacion de esta hipotesis")
    else:
        caso, resolucion = "CASO C", "MARGINAL_UNSTABLE"
        lectura = ("la tercera particion cambia el signo. La mejora depende de la particion")
    log(f"      {caso}: {resolucion}")
    log(f"      {lectura}")
    ejecutar_permutacion = parciales_pos == 3 and folds_pos >= 4 and delta > 0
    log(f"      procede el test de permutacion: {ejecutar_permutacion}")

    # ---------- ficheros ----------
    with CSV_P.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["particion", "V03", "candidato", "delta", "folds_positivos", "rescate",
                    "damage", "neto", "deltas_por_pliegue"])
        for t in todas:
            w.writerow([t["nombre"], t["V03"], round(t["V03"] + t["delta"], 10), t["delta"],
                        t["pliegues"], t["rescate"], t["damage"], t["rescate"] - t["damage"],
                        "|".join(str(x) for x in t["deltas"])])
        w.writerow(["RESUMEN", "media", "", round(float(deltas_tres.mean()), 10),
                    "", "", "", "", ""])
        w.writerow(["RESUMEN", "mediana", "",
                    round(float(np.median(deltas_tres)), 10), "", "", "", "", ""])
        w.writerow(["RESUMEN", "minimo", "", round(float(deltas_tres.min()), 10),
                    "", "", "", "", ""])
        w.writerow(["RESUMEN", "maximo", "", round(float(deltas_tres.max()), 10),
                    "", "", "", "", ""])
        w.writerow(["RESUMEN", "particiones_positivas", parciales_pos, "", "", "", "", "", ""])
        w.writerow(["TERCERA", "semilla", SEMILLA_TERCERA, "", "", "", "", "", ""])
        w.writerow(["TERCERA", "multiplicadores", "|".join(
            ",".join(str(x) for x in mm) for mm in multiplicadores), "", "", "", "", "", ""])
        w.writerow(["TERCERA", "pesos_IQR", round(iqr, 6), "", "", "", "", "", ""])
        w.writerow(["TERCERA", "pesos_medio", round(float(pesos.mean()), 6), "", "", "", "",
                    "", ""])
        for f in filas_et:
            w.writerow(["TERCERA_ETIQUETA", f["etiqueta"], "", "", "", f["rescate"],
                        f["damage"], f["neto"],
                        (f"V03ok_candko={f['V03_ok_cand_ko']}|V03ko_candok={f['V03_ko_cand_ok']}"
                        f"|ambos_ok={f['ambos_ok']}|ambos_ko={f['ambos_ko']}")])
        w.writerow(["DECISION", caso, resolucion, "", "", "", "", "", ""])

    salida = {
        "familia": "P69_TERCERA_PARTICION", "experiment_id": "P69-04",
        "fecha": inicio.isoformat(timespec="seconds"),
        "naturaleza": ("CONFIRMACION pre-registrada. 1 particion. 0 tecnicas nuevas, 0 grid, "
                       "0 submissions, 0 test, 0 holdout, 0 leaderboard"),
        "pre_registro": {"documento": "docs/p69_especificacion_congelada.md",
                         "commit": "915991b",
                         "escrito_antes_de_ejecutar": True},
        "semilla_tercera_particion": SEMILLA_TERCERA,
        "particiones": todas,
        "resumen": {"media": round(float(deltas_tres.mean()), 10),
                    "mediana": round(float(np.median(deltas_tres)), 10),
                    "minimo": round(float(deltas_tres.min()), 10),
                    "maximo": round(float(deltas_tres.max()), 10),
                    "particiones_positivas": parciales_pos},
        "tercera": {"V03": round(ccv_v03, 10), "candidato": round(ccv_cand, 10),
                    "delta": round(delta, 10), "pliegues_positivos": folds_pos,
                    "deltas_por_pliegue": deltas_fold, "rescate": resc, "damage": dan,
                    "neto": resc - dan, "por_etiqueta": filas_et,
                    "multiplicadores_por_fold": multiplicadores,
                    "pesos": {"IQR": round(iqr, 6),
                              "peso_medio": round(float(pesos.mean()), 6),
                              "n_con_peso_mayor_que_1": int((pesos > 1).sum()),
                              "n_con_peso_menor_que_1": int((pesos < 1).sum()),
                              "fraccion_retirada": round(1 - float(pesos.sum()) / n, 6)}},
        "caso": caso, "resolucion": resolucion, "lectura": lectura,
        "procede_permutacion": bool(ejecutar_permutacion),
        "reproducibilidad": {"nivel": "R2", "hash_v03": hash_real,
                             "corrected_cv_v03_canonico": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_C.write_text(json.dumps(salida, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")

    M: list[str] = []
    W = M.append
    W("# Tercera particion de A_cost_sensitive_downweight")
    W("")
    W(f"Generado por `experiments/p69_04_tercera_particion.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## El protocolo estaba escrito antes")
    W("")
    W("La semilla, el metodo de pliegues, la regla de decision y el diseno de la permutacion se")
    W("fijaron en `docs/p69_especificacion_congelada.md`, commit `915991b`, **antes** de que")
    W("existiera ningun resultado de esta particion.")
    W("")
    W("Ese orden es el punto entero. Una particion ejecutada **despues** de ver el resultado de")
    W("otra no es evidencia independiente, por mucho que la semilla parezca arbitraria. Lo unico")
    W("que la hace confirmatoria en vez de adaptativa es que la especificacion vino antes y se")
    W("puede comprobar.")
    W("")
    W("```text")
    W(f"semilla de la tercera particion   {SEMILLA_TERCERA}")
    W("escrito y commiteado en           docs/p69_especificacion_congelada.md, 915991b")
    W("```")
    W("")
    W("## Las tres particiones")
    W("")
    W("| particion | V03 | candidato | delta | folds+ | rescate | damage | neto |")
    W("|---|---:|---:|---:|---:|---:|---:|---:|")
    for t in todas:
        W(f"| {t['nombre']} | {t['V03']:.10f} | {t['V03'] + t['delta']:.10f} | "
          f"{t['delta']:+.10f} | {t['pliegues']}/5 | {t['rescate']} | {t['damage']} | "
          f"{t['rescate'] - t['damage']:+d} |")
    W("")
    W("```text")
    W(f"media     {deltas_tres.mean():+.10f}")
    W(f"mediana   {np.median(deltas_tres):+.10f}")
    W(f"minimo    {deltas_tres.min():+.10f}")
    W(f"maximo    {deltas_tres.max():+.10f}")
    W(f"positivas {parciales_pos} de 3")
    W("```")
    W("")
    W("### Deltas por pliegue")
    W("")
    for t in todas:
        W("```text")
        W(f"{t['nombre']:10} " + " ".join(f"{x:+.6f}" for x in t["deltas"]))
        W("```")
    W("")
    W("## La tercera particion, en detalle")
    W("")
    W(f"- V03 de su propia particion: `{ccv_v03:.10f}`.")
    W(f"- Candidato: `{ccv_cand:.10f}`. Delta `{delta:+.10f}`.")
    W(f"- Pliegues positivos: `{folds_pos}` de 5.")
    W(f"- Rescate `{resc}`, dano `{dan}`, neto `{resc - dan:+d}`.")
    W("")
    W("| etiqueta | V03 ok / cand ko | V03 ko / cand ok | ambos ok | ambos ko | rescate | "
      "damage | neto |")
    W("|---|---:|---:|---:|---:|---:|---:|---:|")
    for f in filas_et:
        W(f"| {f['etiqueta']} | {f['V03_ok_cand_ko']} | {f['V03_ko_cand_ok']} | "
          f"{f['ambos_ok']} | {f['ambos_ko']} | {f['rescate']} | {f['damage']} | "
          f"{f['neto']:+d} |")
    W("")
    W("## La regla de decision, la pre-registrada")
    W("")
    W("```text")
    W(f"{caso}")
    W(f"{resolucion}")
    W("")
    W(a_ascii(lectura))
    W("```")
    W("")
    W(f"Procede el test de permutacion: **{ejecutar_permutacion}**.")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p69_04_tercera_particion.py`. Sin push.*")
    MD_C.write_text(a_ascii("\n".join(M)), encoding="ascii")

    documentos = [MD_C, CSV_P, JSON_C]
    sucios = [d.name for d in documentos if any(b > 127 for b in d.read_bytes())]
    if sucios:
        raise SystemExit(f"PARADA: documentos con bytes no ASCII: {sucios}")

    log("")
    log("=" * 78)
    log(f"P69-04 OK | {caso} | {resolucion}")
    log(f"   tercera: V03 {ccv_v03:.10f} | candidato {ccv_cand:.10f} | delta {delta:+.10f}")
    log(f"   pliegues {folds_pos} de 5 | neto {resc - dan:+d}")
    log(f"   positivas {parciales_pos} de 3 | media {deltas_tres.mean():+.10f}")
    log(f"   procede permutacion: {ejecutar_permutacion}")
    log("   V03 remains the best verified leaderboard result.")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
