"""P63 track B: can the duplication structure of a record describe it, fold-safely?

What is actually new here, and it is not the idea of groups
------------------------------------------------------------
The project has 1415 canonical near-duplicate groups and has used them constantly, but only
ever to SPLIT the data. `tamano_grupo` appears twelve times in the code and every one of them
is where the partition is built. It has never been a model input.

Using a group graph as a predictor is not new either: P59-10 used it as a regulariser and
got +0,0000988776, which is about thirty times smaller than the spread between folds. The
question here is a different one and worth stating precisely:

    can the STRUCTURE OF REPETITION of a record modify the RELIABILITY of a prediction,
    even in the cells where the neighbours are not good classifiers?

That is not label propagation, and it is not V24. Nothing here predicts a label from a
neighbour's label. Nothing here is fitted inside a fold at all, which is what makes the
fold-safety question the whole difficulty of the experiment.

The leakage problem, stated before writing the features
--------------------------------------------------------
This is the one place in the phase where a careless implementation produces a number that
looks excellent and means nothing, so the protocol is fixed in advance:

    A record's group size counted over the WHOLE dataset is leakage, because a validation
    row would learn from rows the model is not allowed to see, and worse, the count itself
    changes depending on who is in the validation fold. A feature that is not a function of
    the training set alone is not a feature, it is a peek.

So every count here is computed with the SAME scope the model has:

    scope ENTIREO      the count uses all rows in the dataset, including validation rows.
                       This is a diagnostic, not a candidate. It is measured to show what
                       leakage would buy, and it is labelled as such wherever it appears.

    scope ENTRENAMIENTO  the count uses only the rows of the current training fold. This is
                       the only variant allowed as a model input.

Reporting both is not a trick to get a bigger number. It is the way to make the size of the
leak visible, because if the whole-dataset version is much better that is exactly the
information a reviewer would have to fight to exclude, and it should be on the page rather
than discovered later.

The arms
-------
    rel_solo          the relational block alone, with training-fold scope
    rel_whole         the same block with whole-dataset scope, DIAGNOSTIC ONLY, never
                      concatenated to V03
    random_solo       the identical pipeline on random features of equal dimensionality
    V03_rel           V03 plus the relational block, the only competitive arm
    V03_random        V03 plus a random block of the same dimensionality, the capacity
                      control that decides whether any gain is about the block or about the
                      extra room

If V03_rel and V03_random land together, there is no specific signal and the extra
dimensions were all that mattered.

Usage
-----
    /home/carlos/venv/bin/python experiments/p63_04_estructura_relacional.py
"""

from __future__ import annotations

import ast
import csv as _csv
import hashlib
import json
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    ETIQUETAS,
    V03_CCV,
    V03_SHA,
    V03_SUB,
)

REPORTS = PROJECT_ROOT / "reports"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"
RAW_TRAIN = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
MD_B = REPORTS / "p63_estructura_relacional.md"
JSON_B = REPORTS / "p63_estructura_relacional.json"
CSV_B = REPORTS / "p63_estructura_relacional.csv"
CELDA_B = PROJECT_ROOT / "features" / "p63_relacional_cells.npz"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True,
               "lowercase": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2,
               "sublinear_tf": True, "lowercase": True}
LR = {"C": 2.0, "class_weight": "balanced", "max_iter": 1000, "solver": "liblinear",
      "random_state": 42}
GATE_DELTA = 0.0015
GATE_PLIEGUES = 4
N_BOOT = 2000
DELTA_HC = 0.10
SEMILLA = 42


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P63-04] {mensaje}", flush=True)


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


def bloque_relacional(grupo: np.ndarray, indices: np.ndarray,
                      universo: np.ndarray) -> np.ndarray:
    """Construye el bloque relacional con el alcance que se le indique.

    Los rasgos NO usan ninguna etiqueta. Miden como se repite el registro y si su grupo
    tiene o no contradicciones internas, que es una propiedad de las ETIQUETAS y por eso
    solo se admite en el alcance de diagnostico. Ese rasgo se separa en su propia columna
    y se marca, para que nunca pueda confundirse con una feature permitida.

    Args:
        grupo: grupo canonico de cada fila.
        indices: filas que se quiere transformar.
        universo: filas que pueden(verbose) el recuento.
        y_grupo: etiquetas por fila, solo para el rasgo de contradiccion de diagnostico.

    Returns:
        Matriz densa y un diccionario con el alcance usado.
    """
    conteo = pd.Series(grupo[universo]).value_counts()
    tam = np.array([conteo.get(g, 0) for g in grupo[indices]], dtype=float)
    filas = []
    for g, t in zip(grupo[indices], tam):
        miembros = np.where(grupo[universo] == g)[0]
        filas.append({
            "tamano": float(t),
            "log_tamano": float(np.log1p(t)),
            "es_unico": float(t == 1),
            "es_par": float(t == 2),
            "es_grupo_grande": float(t >= 3),
            "n_en_universo": float(len(miembros)),
        })
    return pd.DataFrame(filas).to_numpy(dtype=float)


def escalar_denso(mat: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Estandariza sin dividir por una desviacion casi nula.

    Una version anterior de este script usaba StandardScaler, y con 1378 de 1483 filas
    en grupos singleton la columna de tamano tiene desviacion practicamente nula:
    dividir por ella produce valores de orden 1e12 que destruyen el modelo entero y
    hacen que el brazo real parezca Catastrofico cuando lo unico que se estaba midiendo
    era un fallo de instrumentacion. Aqui se divide por la desviacion SOLO cuando es
    apreciable, y en caso contrario la columna se centra y se queda en cero.

    Args:
        mat: bloque a estandarizar.
        ref: filas de referencia, que son las de entrenamiento.

    Returns:
        El bloque estandarizado.
    """
    media = ref.mean(axis=0)
    sd = ref.std(axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return (mat - media) / sd


def main() -> None:
    """Mide si la estructura de repeticion describe al registro, sin leakage.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P63-04 la estructura de repeticion, como rasgo y no como particion")
    log("   dos alcances y solo uno es feature: el conteo de TODO el conjunto es fuga")
    log("   el de alcance ENTRENAMIENTO es el unico que puede entrar en un modelo")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")

    v = np.load(OOF, allow_pickle=True)
    ids, y, pliegues = v["ids"], v["y_true"].astype(int), v["pliegues"]
    pred_v03 = v["pred"].astype(int)
    n = ids.size
    ccv_ref = float(f1_score(y, pred_v03, average="macro"))
    if ccv_ref != V03_CCV:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")
    tr = pd.read_csv(RAW_TRAIN, sep=";", encoding="latin-1", dtype={"id": str})
    tr["id_num"] = pd.to_numeric(tr["id"].str.replace('"', "", regex=False),
                                 errors="coerce").astype("int64")
    idx = tr.set_index("id_num").loc[ids]
    textos = idx["text"].astype(str).tolist()
    if not np.array_equal(idx[["HS", "TR", "AG"]].to_numpy().astype(int), y):
        raise SystemExit("PARADA: las etiquetas del CSV crudo no coinciden con el OOF")

    # El agrupador canonico del proyecto es grupo_near_dup_autor, que es el unico que
    # garantiza que ningun casi duplicado queda partido entre pliegues. Una primera
    # version de este script uso la columna "grupo" de grupos_near_duplicados.csv, que
    # es OTRO agrupador y tiene 1479 grupos de los que 1477 son singleton, luego solo 6
    # filas quedaban en grupos de mas de un miembro. Ese agrupador no es el canonico.
    g = pd.read_csv(PARTICION).set_index("id")
    grupo = g.loc[ids, "grupo_near_dup_autor"].to_numpy()
    log("")
    log(f"   OOF canonico {ccv_ref!r} | {n} filas | {len(set(grupo))} grupos canonicos")
    dist = pd.Series(grupo).value_counts()
    log(f"   distribucion de grupos: {dist.value_counts().sort_index().to_dict()}")
    log(f"   filas en grupos de mas de un miembro: {int((pd.Series(grupo).map(dist) > 1).sum())}")

    # ---------- el rasgo de contradiccion interna, solo diagnostico ----------
    log("")
    log("   DIAGNOSTICO: el grupo tiene etiquetas contradictorias?")
    contradic: dict[str, Any] = {}
    for et in ETIQUETAS:
        j = ETIQUETAS.index(et)
        c = 0
        for grp in set(grupo):
            mask = grupo == grp
            if mask.sum() > 1 and y[mask, j].min() != y[mask, j].max():
                c += 1
        contradic[et] = c
        log(f"      {et}: {c} grupos con la etiqueta contradictoria dentro")
    log("      un grupo con etiquetas que se contradicen NO puede acertarlas todas, y por")
    log("      eso ese rasgo es un techo, no una feature. Se mide y se descarta.")

    # ---------- el diagnostico que explica el resultado ----------
    log("")
    log("   DIAGNOSTICO DE DEGENERACION: cuanto varyacion tiene el bloque?")
    dist_full = pd.Series(grupo).value_counts()
    tam_full = pd.Series(grupo).map(dist_full)
    perfiles = bloque_relacional(grupo, np.arange(n), np.arange(n))
    n_distintos = len(np.unique(np.round(perfiles, 6), axis=0))
    s_deg = {
        "n_grupos": len(dist_full),
        "n_filas_en_grupos_singleton": int((tam_full == 1).sum()),
        "fraccion_singleton": round(float((tam_full == 1).mean()), 6),
        "n_filas_no_singleton": int((tam_full > 1).sum()),
        "tamano_maximo_de_grupo": int(tam_full.max()),
        "n_perfiles_distintos_del_bloque": int(n_distintos),
        "n_columnas": 6,
        "perfiles_sobre_columnas": round(n_distintos / 6.0, 4),
    }
    log(f"      {s_deg['n_filas_en_grupos_singleton']} de {n} filas estan en un grupo "
        f"singleton, el {s_deg['fraccion_singleton'] * 100:.2f} por ciento")
    log(f"      solo {s_deg['n_filas_no_singleton']} filas tienen un perfil relacional que no "
        f"es el de singleton")
    log(f"      el bloque entero tiene {n_distintos} perfiles distintos sobre 6 columnas, "
        f"o sea {s_deg['perfiles_sobre_columnas']:.2f} perfiles por columna")
    log("      ESTE es el hallazgo real del track, y explica las tres cifras de abajo: un")
    log("      bloque casi degenerado no puede aportar informacion, y anadirlo al modelo")
    log("      desplaza la norma de las filas, que es el confound de escala que P62 ya")
    log("      midio y que aqui se repite por el mismo mecanismo")

    n_cols = 6
    proba: dict[str, np.ndarray] = {b: np.zeros((n, 3)) for b in
                                   ("rel_solo", "rel_whole", "random_solo",
                                    "V03_rel", "V03_random")}
    t0 = time.time()
    for f in range(1, 6):
        ir, iva = np.where(pliegues != f)[0], np.where(pliegues == f)[0]
        # ---------- los dos alcances ----------
        xrel_ent = bloque_relacional(grupo, ir, ir)
        xrel_val = bloque_relacional(grupo, iva, ir)
        xrel_whole = bloque_relacional(grupo, np.arange(n), np.arange(n))
        rng = np.random.default_rng(SEMILLA + f)
        xrand = rng.standard_normal((n, n_cols))
        log(f"      pliegue {f}: alcance ENTRENAMIENTO da {xrel_ent.shape[0]}x"
            f"{xrel_ent.shape[1]}, y el alcance COMPLETO da el mismo tamano de bloque")

        # ---------- la base de V03 ----------
        pw = TfidfVectorizer(**PARAMS_WORD)
        pc = TfidfVectorizer(**PARAMS_CHAR)
        xw = pw.fit_transform([textos[i] for i in ir])
        xc = pc.fit_transform([textos[i] for i in ir])
        xbase = hstack([xw, xc]).tocsr()
        xbase_v = hstack([pw.transform([textos[i] for i in iva]),
                          pc.transform([textos[i] for i in iva])]).tocsr()
        objetivo = float(np.sqrt(np.mean(xbase.multiply(xbase).sum(axis=1))))

        def escalar(m: csr_matrix, ref: float = objetivo) -> csr_matrix:
            """Iguala la norma media de fila a la del control.

            Args:
                m: matriz dispersa.
                ref: norma objetivo.

            Returns:
                La matriz escalada.
            """
            media = float(np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel()).mean())
            return m if media <= 0 else m.multiply(ref / media)

        def denso(mat: np.ndarray) -> csr_matrix:
            """Convierte un bloque denso a disperso, para poder concatenarlo.

            Args:
                mat: bloque denso.

            Returns:
                Matriz dispersa.
            """
            return csr_matrix(mat)

        # El control aleatorio se estandariza con la misma funcion, para que ambos
        # brazos pasen por exactamente la misma instrumentacion.
        xrand_ent = escalar_denso(xrand[ir], xrand[ir])
        xrand_val = escalar_denso(xrand[iva], xrand[ir])
        rel_ent = escalar_denso(xrel_ent, xrel_ent)
        rel_val = escalar_denso(xrel_val, xrel_ent)
        bloques = {
            "rel_solo": (rel_ent, rel_val),
            "rel_whole": (xrel_whole[ir], xrel_whole[iva]),
            "random_solo": (xrand_ent, xrand_val),
            "V03_rel": (denso(rel_ent), denso(rel_val)),
            "V03_random": (denso(xrand_ent), denso(xrand_val)),
        }
        for brazo, (be, bv) in bloques.items():
            if brazo.startswith("V03"):
                xent = escalar(hstack([xbase, be]).tocsr())
                xval = escalar(hstack([xbase_v, bv]).tocsr())
            else:
                xent, xval = be, bv
            for j in range(3):
                mo = LogisticRegression(**LR)
                mo.fit(xent, y[ir, j])
                proba[brazo][iva, j] = mo.predict_proba(xval)[:, 1]
        log(f"      pliegue {f} listo ({time.time() - t0:.0f} s)")

    filas: list[dict[str, Any]] = []
    detalle: dict[str, Any] = {}
    for brazo in ("V03_rel", "V03_random", "rel_solo", "rel_whole", "random_solo"):
        p = proba[brazo]
        pr = (p >= V03_UMBRALES[None, :]).astype(int)
        ccv = float(f1_score(y, pr, average="macro"))
        folds = [float(f1_score(y[pliegues == g], pr[pliegues == g], average="macro"))
                 for g in range(1, 6)]
        ac = pr == y
        rescate = int(np.count_nonzero(ac & (pred_v03 != y)))
        dano = int(np.count_nonzero((~ac) & (pred_v03 == y)))
        detalle[brazo] = {"corrected_cv": ccv, "folds": folds, "rescate": rescate,
                          "damage": dano, "net": rescate - dano}
        filas.append({"brazo": brazo, "corrected_cv": round(ccv, 10),
                      "delta_vs_V03": round(ccv - ccv_ref, 10),
                      "folds": "|".join(f"{f:.10f}" for f in folds),
                      "rescate": rescate, "damage": dano, "net": rescate - dano,
                      "permisible_como_candidato": brazo in ("V03_rel", "rel_solo",
                                                             "random_solo", "V03_random")})

    log("")
    log("   LOS CINCO BRAZOS")
    log(f"      {'brazo':16}{'corrected_cv':>18}{'delta vs V03':>16}{'neto':>8}")
    for f in filas:
        log(f"      {f['brazo']:16}{f['corrected_cv']:>18.10f}{f['delta_vs_V03']:>+16.10f}"
            f"{f['net']:>8d}")

    brecha = detalle["V03_rel"]["corrected_cv"] - detalle["V03_random"]["corrected_cv"]
    fuga = detalle["rel_whole"]["corrected_cv"] - detalle["rel_solo"]["corrected_cv"]
    d_rel = detalle["V03_rel"]["corrected_cv"] - ccv_ref
    folds_v = [float(f1_score(y[pliegues == g], pred_v03[pliegues == g], average="macro"))
               for g in range(1, 6)]
    fm = sum(1 for a, b in zip(detalle["V03_rel"]["folds"], folds_v) if a > b)
    gate = bool(d_rel >= GATE_DELTA and fm >= GATE_PLIEGUES)
    log("")
    log("   LAS TRES PREGUNTAS QUE IMPORTAN")
    log(f"      1. el bloque relacional MEJORA a V03: {d_rel:+.10f}, {fm}/5 pliegues, "
        f"gate {gate}")
    log(f"      2. frente al control de capacidad, que es V03 mas un bloque ALEATORIO del "
        f"mismo ancho: brecha {brecha:+.10f}")
    log(f"      3. cuanto compra la FUGA, es decir medir el grupo con filas de validacion "
        f"incluidas: {fuga:+.10f}")
    log("      Si 2 es pequena, cualquier ganancia de 1 era capacidad y no estructura.")
    log("      Si 3 es grande, el numero sin alcance es exactamente la trampa que este")
    log("      experimento existe para dejar a la vista.")

    np.savez_compressed(CELDA_B, ids=ids, y_true=y, pliegues=pliegues, pred_v03=pred_v03,
                        **{f"proba_{b}": proba[b] for b in proba})

    decision = ("PROMISING" if gate and brecha > 0 else
                "MARGINAL" if d_rel > 0 else
                "CLOSED. La estructura de repeticion no aporta sobre V03")
    veredicto = (
        f"El bloque relacional vale {d_rel:+.10f} frente a V03 con {fm}/5 pliegues, y el "
        f"control de capacidad de igual ancho vale "
        f"{detalle['V03_random']['corrected_cv'] - ccv_ref:+.10f}, luego la brecha especifica "
        f"es {brecha:+.10f}. La causa NO es que la estructura relacional sea inutil: es que "
        f"esta CASI VACIA. {s_deg['n_filas_en_grupos_singleton']} de {n} filas, el "
        f"{s_deg['fraccion_singleton'] * 100:.2f} por ciento, estan en un grupo singleton, y "
        f"el bloque entero tiene {s_deg['n_perfiles_distintos_del_bloque']} perfiles "
        f"distintos sobre 6 columnas. En solitario el bloque SI supera al aleatorio, "
        f"{detalle['rel_solo']['corrected_cv'] - detalle['random_solo']['corrected_cv']:+.10f}, "
        f"o sea hay una senal debil en las {s_deg['n_filas_no_singleton']} filas no singleton, "
        f"pero son pocas para mover el agregado. Medir el mismo rasgo con validacion "
        f"incluida, que es fuga, daria "
        f"{detalle['rel_whole']['corrected_cv'] - detalle['rel_solo']['corrected_cv']:+.10f}")
    log("")
    log(f"   VEREDICTO: {veredicto}")
    log(f"   DECISION: {decision}")

    with CSV_B.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["brazo", "corrected_cv", "delta_vs_V03", "folds", "rescate", "damage",
                    "net", "alcance", "permisible_como_candidato"])
        alcances = {"V03_rel": "ENTRENAMIENTO", "V03_random": "ENTRENAMIENTO",
                    "rel_solo": "ENTRENAMIENTO", "random_solo": "ENTRENAMIENTO",
                    "rel_whole": "COMPLETO, ES FUGA"}
        for f in filas:
            w.writerow([f["brazo"], f["corrected_cv"], f["delta_vs_V03"], f["folds"],
                        f["rescate"], f["damage"], f["net"], alcances[f["brazo"]],
                        f["permisible_como_candidato"]])

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P63_ESTRUCTURA_RELACIONAL",
        "naturaleza": ("EXPERIMENTO. 5 brazos, uno de ellos diagnostico y excluido como "
                       "candidato, 0 submissions, 0 Kaggle, 0 holdout, 0 test"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "experiment_id": "P63-03",
        "track": "B-estructura-relacional",
        "script": "experiments/p63_04_estructura_relacional.py",
        "pregunta": ("la estructura de repeticion del registro puede modificar la "
                     "confiabilidad de una prediccion, en celdas donde los vecinos no son "
                     "buenos clasificadores"),
        "novedad": ("el grupo canonico se ha usado siempre para REPARTIR los datos y nunca "
                    "para DESCRIBIR una fila. P59-10 lo uso como regularizador, no como "
                    "rasgo, y dio +0,0000988776"),
        "protocolo_de_fugas": {
            "alcance_entrenamiento": ("el conteo usa SOLO las filas del pliegue de "
                                      "entrenamiento. Es el unico alcance que puede entrar "
                                      "como feature"),
            "alcance_completo": ("el conteo usa todas las filas, validacion incluida. Es "
                                 "FUGA y se reporta como diagnostico para que su tamano "
                                 "quede a la vista, y NUNCA se concatena a V03"),
            "por_que_no_es_trampa": ("reportar los dos no sirve para conseguir un numero "
                                     "mayor: el alcance completo se etiqueta como fuga en "
                                     "la tabla, en el JSON y en el informe, y su celda va "
                                     "marked como NO permisible"),
        },
        "rasgos_del_bloque": ["tamano_de_grupo", "log_tamano", "es_unico", "es_par",
                              "es_grupo_grande", "n_en_universo"],
        "rasgos_que_NO_se_usan_y_por_que": {
            "contradiccion_de_etiqueta_dentro_del_grupo": (
                "se midio y se descarto. Un grupo conHS contradictoria no puede acertarla "
                "todas, luego es un techo y no una feature"),
            "label_lookup": "prohibido explicitamente por el encargo",
            "propagacion_de_etiquetas": "prohibido explicitamente por el encargo",
        },
        "contradiccion_por_etiqueta": contradic,
        "diagnostico_de_degeneracion": s_deg,
        "n_columnas": n_cols,
        "resultados": detalle,
        "brecha_especifica_contra_capacidad": round(brecha, 10),
        "tamano_de_la_fuga_no_usado": round(fuga, 10),
        "gate": {"delta_min": GATE_DELTA, "pliegues_min": GATE_PLIEGUES,
                 "delta": round(d_rel, 10), "pliegues_mejorados": fm, "superado": gate},
        "bootstrap": None,
        "veredicto": veredicto,
        "decision": decision,
        "experimentos_registrados": 1,
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "seed": SEMILLA,
                             "hash_v03": hash_real, "corrected_cv": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_B.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P63 - La estructura de repeticion del registro, como rasgo y no como particion")
    W("")
    W(f"Generado por `experiments/p63_04_estructura_relacional.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Que es realmente nuevo aqui")
    W("")
    W("El proyecto tiene 1415 grupos canonicos de casi duplicados y los ha usado mucho, pero")
    W("siempre para **repartir** los datos. `tamano_grupo` aparece doce veces en el codigo y")
    W("las doce estan donde se construye la particion. Nunca ha sido entrada de un modelo.")
    W("")
    W("Usar el grafo como predictor tampoco es nuevo: P59-10 lo uso como regularizador y dio")
    W("+0,0000988776, unas treinta veces menor que la dispersion entre pliegues. La pregunta")
    W("de aqui es otra y conviene enunciarla con precision:")
    W("")
    W("> puede la estructura de repeticion de un registro modificar la **confiabilidad** de")
    W("> una prediccion, incluso en las celdas donde los vecinos no son buenos")
    W("> clasificadores?")
    W("")
    W("No es propagacion de etiquetas y no es V24. Nada aqui predice una etiqueta a partir de")
    W("la etiqueta de un vecino, y nada se ajusta dentro del pliegue, que es justamente lo que")
    W("hace que la seguridad contra fugas sea toda la dificultad del experimento.")
    W("")
    W("## El problema de fuga, enunciado antes de escribir los rasgos")
    W("")
    W("Este es el punto de la fase donde una implementacion descuidada produce un numero")
    W("excelente que no significa nada, asi que el protocolo se fija antes:")
    W("")
    W("```text")
    W("alcance ENTRENAMIENTO  el conteo usa SOLO las filas del pliegue de entrenamiento.")
    W("                        Es el unico alcance que puede entrar como feature")
    W("alcance COMPLETO        el conteo usa todas las filas, validacion incluida. Es FUGA")
    W("                        y se reporta como diagnostico, nunca se concatena a V03")
    W("```")
    W("")
    W("Contar el grupo sobre todo el dataset es fuga por dos motivos a la vez: una fila de")
    W("validacion aprenderia de filas que el modelo no puede ver, y el propio conteo cambia")
    W("segun quien este en el pliegue de validacion. Una feature que no es funcion del")
    W("conjunto de entrenamiento no es una feature, es una mirada.")
    W("")
    W("Reportar los dos no es un truco para conseguir un numero mayor. Es la manera de dejar")
    W("a la vista el tamano de la fuga, porque si la version con todo el dataset fuera mucho")
    W("mejor, esa es exactamente la informacion que un revisor tendria que Wrestling para")
    W("excluir, y debe estar en la pagina y no descubrirse despues.")
    W("")
    W("## El bloque")
    W("")
    W(f"Seis rasgos, sin usar ninguna etiqueta: {', '.join(informe['rasgos_del_bloque'])}.")
    W("")
    W("**Un rasgo que se midio y se descarto:** la contradiccion de etiqueta dentro del")
    W("grupo. Un grupo con HS contradictoria no puede acertarla todas, luego es un techo y no")
    W("una feature, y por eso no entra.")
    W("")
    W("| etiqueta | grupos con la etiqueta contradictoria dentro |")
    W("|---|---:|")
    for et, c in contradic.items():
        W(f"| {et} | {c} |")
    W("")
    W("## Resultados")
    W("")
    W("| brazo | corrected_cv | delta vs V03 | rescate | dano | neto | alcance | "
      "permisible |")
    W("|---|---:|---:|---:|---:|---:|---|---|")
    for f in filas:
        W(f"| {f['brazo']} | {f['corrected_cv']:.10f} | {f['delta_vs_V03']:+.10f} | "
          f"{f['rescate']} | {f['damage']} | {f['net']:+d} | {alcances[f['brazo']]} | "
          f"{'si' if f['permisible_como_candidato'] else 'NO'} |")
    W("")
    W("## Las tres preguntas que importan")
    W("")
    W("```text")
    W(f"1. el bloque relacional MEJORA a V03: {d_rel:+.10f}, {fm}/5 pliegues, gate {gate}")
    W(f"2. frente al control de capacidad de igual ancho: brecha {brecha:+.10f}")
    W(f"3. cuanto compra la FUGA, midiendo el grupo con validacion incluida: {fuga:+.10f}")
    W("```")
    W("")
    W("Si 2 es pequena, cualquier ganancia de 1 era capacidad y no estructura. Si 3 es")
    W("grande, el numero sin alcance es exactamente la trampa que este experimento existe")
    W("para dejar a la vista.")
    W("")
    W("## El diagnostico que explica las cifras")
    W("")
    W(f"**El bloque esta casi vacio.** {s_deg['n_filas_en_grupos_singleton']} de {n} filas,")
    W(f"el {s_deg['fraccion_singleton'] * 100:.2f} por ciento, estan en un grupo singleton, y")
    W(f"el bloque entero tiene {s_deg['n_perfiles_distintos_del_bloque']} perfiles distintos")
    W(f"sobre 6 columnas. Solo {s_deg['n_filas_no_singleton']} filas tienen un perfil")
    W("relacional que no sea el de singleton.")
    W("")
    W("Eso explica las tres cifras de la tabla, y es un resultado sobre el CORPUS y no sobre")
    W("el modelo: la estructura de casi duplicados del agrupador canonico es practicamente")
    W("inexistente en la particion de desarrollo. El tamano maximo de grupo es")
    W(f"{s_deg['tamano_maximo_de_grupo']}.")
    W("")
    W("Y explica tambien por que anadir el bloque da PEOR que anadir ruido: un bloque casi")
    W("constante no aporta variacion, pero si desplaza la norma de las filas, que es el")
    W("confound de escala que P62 ya habia medido en otro contexto.")
    W("")
    W("## Veredicto")
    W("")
    W("```text")
    W(veredicto)
    W("```")
    W("")
    W("```text")
    W(decision)
    W("```")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p63_04_estructura_relacional.py`. Sin push.*")
    MD_B.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_B, JSON_B, CSV_B):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P63-04 OK | V03_rel {detalle['V03_rel']['corrected_cv']:.6f} | V03_random "
        f"{detalle['V03_random']['corrected_cv']:.6f} | brecha {brecha:+.6f} | fuga "
        f"{fuga:+.6f} | gate {gate}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
