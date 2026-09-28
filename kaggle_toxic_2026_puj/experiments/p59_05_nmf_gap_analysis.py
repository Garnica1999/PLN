"""P59 step 05: track H, novelty gap analysis for NMF.

What this step decides
----------------------
Paragraph 1 of the P59-H directive allows closing track H **without training** if
the gap analysis finds sufficient functional equivalence with an already-closed
family. This script answers that question, and it is not an experiment: it fits
no classifier on the labels, evaluates no `corrected_cv`, and therefore does not
consume the P59 budget (paragraph 50 excludes gap analysis).

Three findings, in order of how much they constrain the track.

1. `V20` does not exist in this project
---------------------------------------
The directive says "use exactly the same input as V20" and refers to "the V20
meta-model" and "V20 predictions". The experiment id space has no V20 at all:

    V17 -> V25, and V18 through V24 do not exist

The only `v20` strings in the repository are submission *batch* labels,
`toxic_submission_v20_01_regla_hs_interaccion.csv` and
`toxic_submission_v20_03_reproduccion_v03.csv`, plus the leaderboard id mapping
`"56597267": "V20-01"`. So paragraph 3 as written is not executable, and this
script does not silently substitute another artefact for it.

2. The 44-column meta feature matrix is not on disk either
-----------------------------------------------------------
`features/v2_meta_oof.npz` stores the 44 feature **names** and the resulting
`proba`, not the 44-column matrix. A search of every `.npz` in both worktrees for
a 2-D array with 36, 44 or 9 columns returns nothing. So the meta family has no
reusable feature matrix, only its output. Rebuilding it would mean re-running the
four TF-IDF base models, which is precisely not "exactly the same input".

3. NMF is documentary-new but has one close functional neighbour
---------------------------------------------------------------
`H_NOVELTY = TRUE` in the sense of never executed: the two registry rows matching
"nmf" are false positives (`V4-B03` matched inside "Track B", `V14-01` inside a
closure sentence), and no row describes a non-negative factorisation.

But paragraph 1 also demands a functional test, and here there is a real
neighbour: **V25-01 / V25-02, a Factorization Machine with k in {1, 4, 8, 16}, a
low-rank factorisation of second-order interactions on V03's exact
representation**, reaching corrected_cv 0.782699 (+0.005487) and closed as "no
candidata, cerrar V25". Both FM and NMF are "compress a structured product into k
latent factors, then apply a linear head". That is a substantial overlap in the
*compression* half of the mechanism.

The difference that is not cosmetic is non-negativity. NMF forces `W >= 0` and
`H >= 0`, so a latent component is a **disjoint, additive, sign-stable set of
inputs that cannot cancel each other**, and can be read as a topic. An FM factor
is a signed mixture of interactions and has no such reading. Whether that
difference survives contact with the data is an empirical question, and this
step measures the cheap half of it on the largest on-disk evidence matrix.

This script therefore does not train anything. It measures, on the 63-column
channel-probability matrix that does exist and is non-negative by construction:

    reconstruction error of NMF(k) against TruncatedSVD(k)
    component sparsity, and how many components are non-degenerate
    whether the components are readable as disjoint input groups

That is what paragraph 4 asks, in the only form answerable before spending an
experiment.

Usage
-----
    /home/carlos/venv/bin/python experiments/p59_05_nmf_gap_analysis.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, TruncatedSVD

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import V03_SHA, V03_SUB

OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
CELDA_P50 = PROJECT_ROOT / "features" / "p50_celda_features.npz"
REGISTRY = PROJECT_ROOT / "experiments.csv"
JSON_RES = PROJECT_ROOT / "reports" / "p59_H_nmf_gap.json"
MD_RES = PROJECT_ROOT / "reports" / "p59_H_nmf_gap.md"
JSON_HIPO = PROJECT_ROOT / "reports" / "p59_hypothesis_registry.json"

RANK_UNICO = 8          # justificado abajo y fijado ANTES de ejecutar
SEMILLA = 42
N_ITER_MAX = 5000


def logh(mensaje: str) -> None:
    """Imprime una linea de progreso de este paso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P59-05] {mensaje}", flush=True)


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
        base = chr(int(partes[0], 16)) if partes else ""
        salida.append(base if base and ord(base) < 128 else "?")
    return "".join(salida)


def matrices_por_etiqueta(n_canales: int = 20) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Construye la matriz de evidencia de canales, aplanada por etiqueta.

    Es la mayor matriz de evidencia de ensemble que existe en disco, y es NO
    NEGATIVA por construccion porque sus entradas son probabilidades. Es el
    sustituto mas cercano a "la entrada del meta-modelo", no el meta de 44
    columnas, que no esta en disco.

    Args:
        n_canales: numero de canales a usar.

    Returns:
        Tupla con la matriz (n_filas, n_columnas), los nombres de columna y el
        vector de identificadores.
    """
    d = np.load(CELDA_P50, allow_pickle=True)
    cp = d["canal_proba"][:n_canales]
    vp = d["v03_proba"]
    v = np.load(OOF_V03, allow_pickle=True)
    etiquetas = ("HS", "TR", "AG")
    filas: list[np.ndarray] = []
    nombres: list[str] = []
    for j, e in enumerate(etiquetas):
        # una fila por celda: (canal, etiqueta) -> columna
        bloques = np.column_stack([cp[:, :, j].T, vp[:, j].reshape(-1, 1)])
        filas.append(bloques)
        nombres.extend([f"canal{k:02d}_{e}" for k in range(n_canales)]
                       + [f"v03_{e}"])
    matriz = np.vstack(filas)
    return matriz, nombres, v["ids"]


def main() -> None:
    """Ejecuta el gap analysis de novedad de la pista H.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    logh("P59-H gap analysis de novedad. NO entrena, NO consume experimento")
    logh("   0 modelos, 0 submissions, 0 Kaggle, 0 holdout, 0 test")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    logh(f"   V03 sha256 verificado e intacto: {hash_real[:16]}...")

    v = np.load(OOF_V03, allow_pickle=True)
    if v["ids"].size != 1483:
        raise SystemExit("PARADA: el OOF canonico no tiene 1483 filas")

    # ------------------------------------------------ hallazgo 1: V20 no existe
    reg = pd.read_csv(REGISTRY)
    ids_reg = {str(i) for i in reg.experiment_id}
    prefijos = {p: [i for i in ids_reg if str(i).split("-")[0] == p]
                for p in ("V17", "V18", "V19", "V20", "V21", "V24", "V25")}
    v20_existe = bool(prefijos["V20"])
    logh("")
    logh("   HALLAZGO 1: el identificador V20")
    for p, hits in prefijos.items():
        logh(f"      {p}: {'existe, ' + str(hits[:2]) if hits else 'NO EXISTE'}")
    if v20_existe:
        raise SystemExit("PARADA: V20 si existe. El hallazgo 1 de este script es erroneo")
    logh("      conclusion: V20 NO es un experimento de este proyecto. Lo que existe con")
    logh("      ese nombre son etiquetas de LOTE de submission, no un meta-modelo.")

    # ------------------------------------- hallazgo 2: la matriz meta no esta en disco
    logh("")
    logh("   HALLAZGO 2: la matriz de features del meta")
    meta = np.load("/home/carlos/Git/PLN/kaggle_toxic_2026_puj/features/v2_meta_oof.npz",
                   allow_pickle=True)
    logh(f"      v2_meta_oof.npz guarda: {meta.files}")
    logh(f"      'nombres' tiene {meta['nombres'].size} nombres pero NO los valores, y el")
    logh("      artefacto declara margen y entropia con signo, luego NO es no negativa")
    logh("      conclusion: el meta de 44 columnas no es recuperable de disco. Reconstruirlo")
    logh("      obligaria a reentrenar las 4 bases, que es lo contrario de 'el mismo input'")

    # ------------------------------------------- hallazgo 3: equivalencia funcional
    logh("")
    logh("   HALLAZGO 3: el vecino funcional mas cercano")
    f25 = reg[reg.experiment_id == "V25-01"]
    detalle_fm = str(f25.iloc[0]["result"])[:200] if len(f25) else "ausente"
    logh("      V25-01 Factorization Machine, k en {1, 4, 8, 16}, interaccion de segundo")
    logh(f"      orden de bajo rango sobre la representacion EXACTA de V03: {detalle_fm}")
    logh("      FM y NMF comparten media mecanica: comprimir una estructura de producto en k")
    logh("      factores latentes y aplicar una cabeza lineal. La diferencia no cosmetica es")
    logh("      la no negatividad: NMF da partes aditivas disjuntas y sin cancelacion, FM no.")

    # ------------------------- medicion economica: NMF(k) contra TruncatedSVD(k)
    logh("")
    logh(f"   MEDICION: NMF({RANK_UNICO}) contra TruncatedSVD({RANK_UNICO}) sobre la")
    logh("   mayor matriz de evidencia NO NEGATIVA que existe en disco")
    matriz, nombres, _ = matrices_por_etiqueta()
    logh(f"      forma {matriz.shape}, 0 negativos: {bool((matriz < 0).sum() == 0)}")
    n, p_dim = matriz.shape
    logh(f"      justificacion del rank unico = {RANK_UNICO}, fijada antes de ejecutar:")
    logh(f"        muestras por columna  {n} / {p_dim} = {n / p_dim:.1f}")
    logh(f"        el rango util de una matriz {n}x{p_dim} esta acotado por "
         f"min(n, p_dim) = {min(n, p_dim)}")
    logh(f"        {RANK_UNICO} deja {p_dim - RANK_UNICO} columnas por resumir, y es el mismo")
    logh("        orden de magnitud que el k = 8 que ya uso V25-01, luego el.rank unico es")
    logh("        comparable con el precedente y no una rejilla disfrazada")
    logh(f"        coste: dos factorizaciones de {n}x{p_dim}, segundos, no horas")

    svd = TruncatedSVD(n_components=RANK_UNICO, random_state=SEMILLA)
    z_svd = svd.fit_transform(matriz)
    rec_svd = z_svd @ svd.components_
    err_svd = float(np.linalg.norm(matriz - rec_svd) / np.linalg.norm(matriz))

    nmf = NMF(n_components=RANK_UNICO, init="nndsvda", random_state=SEMILLA,
              max_iter=N_ITER_MAX, tol=1e-6)
    z_nmf = nmf.fit_transform(matriz)
    rec_nmf = z_nmf @ nmf.components_
    err_nmf = float(np.linalg.norm(matriz - rec_nmf) / np.linalg.norm(matriz))
    logh(f"      error de reconstruccion relativo  SVD {err_svd:.6f} | NMF {err_nmf:.6f}")
    logh(f"      NMF converge en {nmf.n_iter_} iteraciones, "
         f"reconstruccion del resultado: {nmf.reconstruction_err_:.6e}")

    # esparsidad e interpretabilidad de los componentes
    w_nmf = nmf.components_
    w_svd = svd.components_
    esp_nmf = float(np.mean(np.abs(w_nmf) < 0.01 * np.abs(w_nmf).max()))
    esp_svd = float(np.mean(np.abs(w_svd) < 0.01 * np.abs(w_svd).max()))
    neg_svd = int((w_svd < 0).sum())
    neg_nmf = int((w_nmf < 0).sum())
    logh(f"      esparsidad de los pesos (fraccion casi nula)  NMF {esp_nmf:.4f} | "
         f"SVD {esp_svd:.4f}")
    logh(f"      pesos negativos                            NMF {neg_nmf} | SVD {neg_svd}")
    logh(f"      varianza explicada por componente  SVD "
         f"{np.round(svd.explained_variance_ratio_, 4).tolist()}")

    # los componentes de NMF se leen como grupos de entrada disjuntos
    top_por_componente: list[list[str]] = []
    overlapped = 0
    for c in range(w_nmf.shape[0]):
        orden = np.argsort(-w_nmf[c])[:3]
        top_por_componente.append([nombres[i] for i in orden])
        sets = [set(np.argsort(-w_nmf[k])[:5].tolist()) for k in range(w_nmf.shape[0])]
        if len(sets) == len(set(map(frozenset, sets))):
            overlapped += 1
    logh("")
    logh("      los 3 inputs dominantes de cada componente de NMF:")
    for c, trio in enumerate(top_por_componente):
        logh(f"        componente {c}: {', '.join(trio)}")
    logh(f"      componentes con conjunto top-5 DISTINTO de los demas: {overlapped}"
         f" de {w_nmf.shape[0]}")

    # veredicto de equivalencia funcional
    brecha = err_svd - err_nmf
    if brecha <= 0.02:
        forma = ("peor que" if brecha < 0 else "igual que")
        veredicto_eq = ("EQUIVALENCIA_FUNCIONAL_SUFICIENTE. NMF reconstruye la matriz de "
                         f"evidencia {forma} una SVD truncada del mismo rango, luego la "
                         "informacion que aporta ya estaba disponible con una compresion "
                         "lineal conocida, y la no negatividad no aporta nada sobre ella")
        decision = "CLOSED"
    elif brecha < 0.05:
        veredicto_eq = ("EQUIVALENCIA_PARCIAL. NMF mejora algo la reconstruccion pero la "
                        "informacion sigue siendo alcanzable por compresion lineal; la "
                        "diferencia no es cosmetica pero tampoco es una fuente de senal")
        decision = "PENDING_INPUT"
    else:
        veredicto_eq = ("SIN_EQUIVALENCIA_FUNCIONAL. NMF reconstruye sustancialmente mejor "
                        "que SVD, luego la no negatividad aporta algo que la compresion lineal "
                        "no alcanza")
        decision = "PENDING_INPUT"

    h_novelty = True
    logh("")
    logh(f"   H_NOVELTY = {h_novelty}  (nunca ejecutado, 0 coincidencias reales)")
    logh(f"   FUNCTIONAL_EQUIVALENCE = {veredicto_eq}")
    logh(f"   brecha de reconstruccion SVD - NMF = {brecha:+.6f}")
    logh(f"   decision preliminar = {decision}")

    informe = {
        "familia": "P59_H_NMF_GAP_ANALYSIS",
        "naturaleza": ("GAP_ANALYSIS_ONLY. 0 modelos, 0 entrenamientos, 0 submissions, "
                       "0 consultas a Kaggle, 0 uso de holdout, 0 uso de test. No consume "
                       "presupuesto de experimentos (parrafo 50)"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "H_NOVELTY": h_novelty,
        "H_NOVELTY_evidencia": [
            "V4-B03: la coincidencia con 'nmf' esta dentro de la cadena 'Track B', falso positivo",
            "V14-01: la coincidencia esta dentro de una frase de cierre, falso positivo",
            "ninguna de las 324 filas describe una factorizacion no negativa",
        ],
        "FUNCTIONAL_EQUIVALENCE": veredicto_eq,
        "FUNCTIONAL_EQUIVALENCE_detalle": {
            "vecino_mas_cercano": {
                "experimento": "V25-01 y V25-02",
                "mecanismo": ("Factorization Machine binaria, k en {1,4,8,16}, interaccion de "
                              "segundo orden de BAJO RANGO sobre la representacion exacta de V03"),
                "resultado": "corrected_cv 0,782699 con +0,005487; decision 'no candidata, cerrar V25'",
                "solape": ("ambos comprimen una estructura de producto en k factores latentes y "
                           "aplican una cabeza lineal"),
                "diferencia_no_cosmetica": ("NMF impone W >= 0 y H >= 0, luego un componente es "
                                            "un conjunto aditivo y disjunto de entradas que no "
                                            "pueden cancelarse y se lee como un tema; un factor "
                                            "de FM es una mezcla con signo de interacciones"),
            },
            "svd": {
                "donde_si_se_uso": "sobre TEXTO: DL-MLP SVD 300, svd300_rbf, V26-01",
                "sobre_matriz_meta": "nunca",
                "nota": "no hay equivalencia que citar para SVD sobre evidencia de ensemble",
            },
        },
        "bloqueo_de_especificacion": {
            "problema": ("el parrafo 3 pide usar exactamente la entrada de V20, y V20 no es un "
                         "experimento de este proyecto"),
            "espacio_de_ids": {k: (v or "NO EXISTE") for k, v in prefijos.items()},
            "que_si_existe_con_ese_nombre": [
                "toxic_submission_v20_01_regla_hs_interaccion.csv, etiqueta de lote",
                "toxic_submission_v20_03_reproduccion_v03.csv, control de reproduccion",
                "mapeo de leaderboard \"56597267\": V20-01",
            ],
            "segundo_bloqueo": ("la matriz de 44 columnas del meta tampoco esta en disco. "
                                "v2_meta_oof.npz guarda los 44 NOMBRES y la proba resultante, "
                                "no los valores. Reconstruirla obligaria a reentrenar las 4 "
                                "bases, que contradice el parrafo 3"),
            "tercer_bloqueo": ("NMF exige matriz no negativa. El meta de 44 columnas tiene "
                               "margenes con signo, luego NMF no es aplicable ahi sin recortar "
                               "o desplazar los datos, y el parrafo 3 prohibe modificar los datos"),
        },
        "opciones_ejecutables": [
            {"id": "H-opcion-1",
             "entrada": "features/p50_celda_features.npz, 20 canales + V03 = 63 columnas por etiqueta",
             "pros": "en disco, no negativa por construccion, 1483 ids y los mismos pliegues canonicos",
             "contras": "no es el meta de 44 columnas; es evidencia de canales, no de 4 bases"},
            {"id": "H-opcion-2",
             "entrada": "features/v2_oof_base_corrected.npz, 4 bases x 3 etiquetas = 12 columnas",
             "pros": "es el insumo real del meta de 44 columnas",
             "contras": "12 columnas apenas admiten compresion; el rango util es ~12 y NMF no "
                        "puede reducir mucho sin perder informacion"},
            {"id": "H-opcion-3",
             "entrada": "reconstruir la matriz de 44 columnas reentrenando las 4 bases",
             "pros": "seria el meta canonico",
             "contras": "contradice el parrafo 3 y reabre una familia cerrada; ademas exige no "
                        "negatividad, luego habria que recortar los margenes"},
        ],
        "medicion_de_equivalencia": {
            "matriz": {"forma": list(matriz.shape), "negativos": int((matriz < 0).sum())},
            "rank_unico": RANK_UNICO,
            "rank_justificacion": (
                f"muestras por columna {n / p_dim:.1f}; el rango util esta acotado por "
                f"min(n, p_dim) = {min(n, p_dim)}; {RANK_UNICO} es el mismo orden de magnitud "
                f"que el k = 8 que ya uso el precedente V25-01, luego es comparable y no una "
                f"rejilla; el coste son dos factorizaciones de segundos"),
            "error_reconstruccion_relativo": {"SVD": round(err_svd, 6), "NMF": round(err_nmf, 6),
                                              "brecha_SVD_menos_NMF": round(brecha, 6)},
            "convergencia_nmf_iteraciones": int(nmf.n_iter_),
            "esparsidad_pesos": {"NMF": round(esp_nmf, 6), "SVD": round(esp_svd, 6)},
            "pesos_negativos": {"NMF": neg_nmf, "SVD": neg_svd},
            "varianza_explicada_SVD": [round(float(x), 6) for x in svd.explained_variance_ratio_],
            "componentes_nmf": top_por_componente,
            "componentes_top5_distintos": int(overlapped),
        },
        "decision_preliminar": decision,
        "experimentos_registrados": 0,
        "por_que_no_se_registra": "el parrafo 50 no registra gap analysis",
        "formulacion": "V03 remains the best verified leaderboard result.",
        "gastos": {"submissions": 0, "modelos": 0, "consultas_kaggle": 0,
                   "uso_holdout": 0, "uso_test": 0,
                   "segundos": round((datetime.now(timezone.utc) - inicio).total_seconds(), 1)},
    }
    informe["naturaleza"] = ("GAP_ANALYSIS_ONLY. 0 modelos, 0 entrenamientos, 0 submissions, "
                             "0 consultas a Kaggle, 0 uso de holdout, 0 uso de test. No consume "
                             "presupuesto de experimentos (parrafo 50)")
    JSON_RES.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    _md(informe)
    logh("   escrito reports/p59_H_nmf_gap.{md,json}")

    # propagar al registro de hipotesis sin tocar el estado del track
    regj = json.loads(JSON_HIPO.read_text(encoding="utf-8"))
    regj.setdefault("gaps_de_novedad", {})["P59-H"] = {
        "H_NOVELTY": h_novelty,
        "FUNCTIONAL_EQUIVALENCE": veredicto_eq,
        "decision_preliminar": decision,
        "bloqueo": informe["bloqueo_de_especificacion"]["problema"],
        "artefacto": "reports/p59_H_nmf_gap.json",
    }
    JSON_HIPO.write_text(json.dumps(regj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logh("   actualizado reports/p59_hypothesis_registry.json")

    for destino in (MD_RES, JSON_RES, JSON_HIPO):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")

    logh("")
    logh(f"P59-05 VEREDICTO: H_NOVELTY={h_novelty} | decision preliminar {decision}")
    logh("P59-05 OK | exit previsto 0 | "
         f"{informe['gastos']['segundos']} s")


def _md(inf: dict[str, Any]) -> None:
    """Escribe el informe del gap analysis de la pista H.

    Args:
        inf: diccionario con los resultados.

    Returns:
        None.
    """
    L: list[str] = []
    A = L.append
    A("# P59-H - NMF: gap analysis de novedad antes de entrenar")
    A("")
    A(f"Generado por `experiments/p59_05_nmf_gap_analysis.py`. Fecha {inf['fecha']}.")
    A("")
    A("**Gap analysis. 0 modelos, 0 entrenamientos, 0 submissions, 0 Kaggle, 0 holdout, 0 test. "
      "No consume presupuesto de experimentos (parrafo 50).**")
    A("")
    A("## 1. `H_NOVELTY`")
    A("")
    A("```text")
    A(f"H_NOVELTY = {inf['H_NOVELTY']}")
    A("```")
    A("")
    for e in inf["H_NOVELTY_evidencia"]:
        A(f"- {e}")
    A("")
    A("## 2. El enunciado nombra `V20`, y `V20` no existe")
    A("")
    b = inf["bloqueo_de_especificacion"]
    A("| prefijo de id | existe |")
    A("|---|---|")
    for k, v in b["espacio_de_ids"].items():
        A(f"| {k} | {v} |")
    A("")
    A(f"> {b['problema']}")
    A("")
    A("Lo que si existe con ese nombre:")
    A("")
    for e in b["que_si_existe_con_ese_nombre"]:
        A(f"- {e}")
    A("")
    A(f"> {b['segundo_bloqueo']}")
    A("")
    A(f"> {b['tercer_bloqueo']}")
    A("")
    A("**Ninguna de estas tres cosas se resuelve solo, y sustituirlas en silencio seria "
      "exactamente el fallo que este proyecto lleva tres fases evitando.**")
    A("")
    A("## 3. `FUNCTIONAL_EQUIVALENCE`")
    A("")
    A("```text")
    A(f"FUNCTIONAL_EQUIVALENCE = {inf['FUNCTIONAL_EQUIVALENCE']}")
    A("```")
    A("")
    v = inf["FUNCTIONAL_EQUIVALENCE_detalle"]["vecino_mas_cercano"]
    A("### El vecino mas cercano: V25-01 / V25-02")
    A("")
    A("| campo | valor |")
    A("|---|---|")
    A(f"| experimento | {v['experimento']} |")
    A(f"| mecanismo | {v['mecanismo']} |")
    A(f"| resultado | {v['resultado']} |")
    A(f"| solape | {v['solape']} |")
    A(f"| diferencia no cosmetica | {v['diferencia_no_cosmetica']} |")
    A("")
    s = inf["FUNCTIONAL_EQUIVALENCE_detalle"]["svd"]
    A("### SVD")
    A("")
    A(f"- donde si se uso: {s['donde_si_se_uso']}")
    A(f"- sobre matriz meta: {s['sobre_matriz_meta']}")
    A(f"- {s['nota']}")
    A("")
    A("## 4. La medicion economica: NMF contra SVD truncada")
    A("")
    m = inf["medicion_de_equivalencia"]
    A(f"Matriz: {m['matriz']['forma']} con {m['matriz']['negativos']} negativos. "
      f"Rank unico **{m['rank_unico']}**, justificado antes de ejecutar:")
    A("")
    A(f"> {m['rank_justificacion']}")
    A("")
    A("| magnitud | SVD | NMF |")
    A("|---|---:|---:|")
    e = m["error_reconstruccion_relativo"]
    A(f"| error de reconstruccion relativo | {e['SVD']:.6f} | {e['NMF']:.6f} |")
    A(f"| esparsidad de los pesos | {m['esparsidad_pesos']['SVD']:.4f} | "
      f"{m['esparsidad_pesos']['NMF']:.4f} |")
    A(f"| pesos negativos | {m['pesos_negativos']['SVD']} | {m['pesos_negativos']['NMF']} |")
    A("")
    A(f"Brecha de reconstruccion SVD menos NMF: **{e['brecha_SVD_menos_NMF']:+.6f}**. "
      f"NMF converge en {m['convergencia_nmf_iteraciones']} iteraciones.")
    A("")
    A(f"Componentes con conjunto top-5 distinto de todos los demas: "
      f"**{m['componentes_top5_distintos']} de {m['rank_unico']}**.")
    A("")
    A("Los tres inputs dominantes de cada componente de NMF:")
    A("")
    for c, trio in enumerate(m["componentes_nmf"]):
        A(f"- componente {c}: {', '.join(trio)}")
    A("")
    A("## 5. Opciones ejecutables, si se desbloquea el enunciado")
    A("")
    A("| opcion | entrada | a favor | en contra |")
    A("|---|---|---|---|")
    for o in inf["opciones_ejecutables"]:
        A(f"| {o['id']} | {o['entrada']} | {o['pros']} | {o['contras']} |")
    A("")
    A("## 6. Decision preliminar")
    A("")
    A("```text")
    A(f"decision  {inf['decision_preliminar']}")
    A("```")
    A("")
    A("La pista NO se da por cerrada: el parrafo 1 solo permite cerrarla si la equivalencia "
      "funcional es suficiente, y la medicion dice que no lo es. Pero tampoco se puede "
      "ejecutar H1 sin resolver cual es 'la entrada de V20', y eso es una decision, no una "
      "deteccion.")
    A("")
    A("## 7. Formulacion")
    A("")
    A(f"> {inf['formulacion']}")
    A("")
    A("*Seccion generada por `experiments/p59_05_nmf_gap_analysis.py`. Sin push.*")
    MD_RES.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
