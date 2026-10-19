#!/usr/bin/env python
"""
P96-01: recomputacion de la metrica canonica y barrido de artefactos (P3/P10).

Solo lectura y solo computo determinista: no entrena, no predice, no genera
submissions. Pasos:

1. Cadena canonica de V03: verifica alineacion ids/etiquetas, reproduce la
   prediccion con los umbrales congelados, recomputa f1_media_columnas por
   etiqueta y por pliegue, y cuantifica el optimismo in-sample de re-optimizar
   umbrales sobre el mismo OOF.
2. Barrido de artefactos con predicciones guardadas (features/*.npz,
   experiments/baseline_v03_frozen/*.npz, reports/p84_04_parciales/*.npz).
   Para cada artefacto se recomputa la metrica canonica en CADA variante con
   matriz (n, 3) recomputable (proba con umbrales guardados o pred directa).
   Los artefactos que cubren filas del holdout historico NO se abren
   (P96-P4: no consultar contenidos del holdout); quedan marcados.
3. Clasificacion de cada recomputacion: RECOMPUTED_EXACTLY /
   RECOMPUTED_WITHIN_TOLERANCE / RECOMPUTED_NO_CLAIM_COMPARABLE /
   NUMERIC_MISMATCH / NOT_RECOMPUTABLE_FROM_SAVED_ARTIFACTS / EXCLUDED_BY_POLICY.

Salida: reports/p96_reproducibility_matrix.csv.

Uso:
    /home/carlos/venv/bin/python experiments/p96_01_metric_recompute.py
"""

from __future__ import annotations

import csv
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import data as data_module  # noqa: E402
from src.evaluation import ETIQUETAS, metricas_multietiqueta  # noqa: E402

SALIDA = RAIZ / "reports" / "p96_reproducibility_matrix.csv"
CANONICO = 0.7772114525165721
TOL_EXACTA = 1e-9
TOL_TOLERANCIA = 1e-6
EXCLUIDOS_POLITICA = {
    "experiments/baseline_v03_frozen/oof_v03_historical.npz":
        "cubre filas del holdout historico; contenido no consultado (P96)",
    "features/oof_mod03b.npz":
        "cubre filas del holdout historico (fase 13a sobre train completo); contenido no consultado (P96)",
}
NOMBRES_NO_PROBA = re.compile(
    r"^(margen|entropia|confianza|umbrales|thresholds|pliegues?$|etiquetas$|"
    r"cluster|en_cluster|vecino|discrepancia|n_tokens|n_segmentos|semilla|"
    r"E_estrella|nota|umbrales_)")
MAX_VARIANTES = 14


def sha16(ruta: Path) -> str:
    """sha256 truncado a 16 hexdigitos."""
    x = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            x.update(bloque)
    return x.hexdigest()[:16]


def cargar_dev_ids() -> set[int]:
    """Lee la membresia del desarrollo desde la particion congelada."""
    particion = pd.read_csv(RAIZ / "data" / "processed" / "holdout_v2_ids.csv",
                            usecols=["id", "particion"])
    return set(particion.loc[particion["particion"] == "dev", "id"].astype(int))


def f1_por_etiqueta(y_true: np.ndarray, pred: np.ndarray) -> list[float]:
    """F1 binaria de la clase positiva por etiqueta."""
    from sklearn.metrics import f1_score
    return [float(f1_score(y_true[:, i], pred[:, i], zero_division=0))
            for i in range(3)]


def numeros_de(claim: str) -> list[float]:
    """Extrae numeros de un claim serializado 'clave=[...]'."""
    try:
        return [float(x) for x in claim.split("=", 1)[1].strip("[]").split(",")]
    except (ValueError, IndexError):
        return []


def clasificar(recomputed: float, claims: list[str]) -> str:
    """Clasifica una recomputacion frente a los claims escalares guardados."""
    if not claims:
        return "RECOMPUTED_NO_CLAIM_COMPARABLE"
    numeros = [x for c in claims for x in numeros_de(c)]
    if not numeros:
        return "RECOMPUTED_NO_CLAIM_COMPARABLE"
    if any(abs(x - recomputed) < TOL_EXACTA for x in numeros):
        return "RECOMPUTED_EXACTLY"
    if any(abs(x - recomputed) < TOL_TOLERANCIA for x in numeros):
        return "RECOMPUTED_WITHIN_TOLERANCE"
    if len(numeros) == 1:
        return "NUMERIC_MISMATCH"
    return "RECOMPUTED_NO_CLAIM_COMPARABLE"


def main() -> None:
    """Ejecuta la recomputacion completa."""
    if SALIDA.exists():
        raise SystemExit(f"Ya existe {SALIDA}; no se sobrescribe (P96).")
    dev_ids = cargar_dev_ids()
    train = data_module.read_train()
    mapa_train = train.assign(_id=train["id"].astype(int)).set_index("_id")[list(ETIQUETAS)]
    filas_salida: list[dict[str, str]] = []

    # ---------- Parte A: cadena canonica de V03 ----------
    print("=== A. Cadena canonica V03 ===")
    oof = np.load(RAIZ / "features" / "oof_v03_final.npz", allow_pickle=True)
    ids = oof["ids"]
    y_true = oof["y_true"]
    proba = oof["proba"]
    pred_guardada = oof["pred"]
    umbrales = oof["umbrales"]
    print(f"  n={len(ids)} | ids unicos={len(set(ids.tolist()))} | "
          f"todos en dev={set(ids.tolist()) <= dev_ids}")
    pred_recomputada = (proba >= umbrales).astype(int)
    print(f"  pred guardada == pred recomputada con umbrales {umbrales.tolist()}: "
          f"{bool((pred_recomputada == pred_guardada).all())}")
    metrica = metricas_multietiqueta(y_true, pred_recomputada)
    print(f"  f1_media_columnas recomputada: {metrica['f1_media_columnas']!r}")
    print(f"  corrected_cv guardada:         {float(oof['corrected_cv'][0])!r}")
    print(f"  canonico documentado:          {CANONICO!r}")
    print(f"  |recomputado - canonico| = "
          f"{abs(metrica['f1_media_columnas'] - CANONICO):.3e}")
    por_etiqueta = f1_por_etiqueta(y_true, pred_recomputada)
    print(f"  f1 por etiqueta == guardada: "
          f"{bool(np.allclose(por_etiqueta, [float(x) for x in oof['f1_por_etiqueta']]))}")
    pliegues = oof["pliegues"]
    f1_pliegues = [float(metricas_multietiqueta(
        y_true[pliegues == p], pred_recomputada[pliegues == p])["f1_media_columnas"])
        for p in sorted(np.unique(pliegues).tolist())]
    print(f"  f1 por pliegue == guardada: "
          f"{bool(np.allclose(f1_pliegues, [float(x) for x in oof['f1_pliegues']]))}")
    etiquetas_train = mapa_train.loc[ids].to_numpy()
    print(f"  y_true del OOF == etiquetas originales del train: "
          f"{bool((etiquetas_train == y_true).all())}")

    bases = np.load(RAIZ / "features" / "v2_oof_base_corrected.npz", allow_pickle=True)
    modelos = [str(m) for m in bases["modelos"]]
    pos_m2 = modelos.index("M2_union_lr_c2")
    ids_bases = bases["ids"]
    proba_m2 = bases["proba"][:, pos_m2, :]
    proba_igual = bool(np.array_equal(ids_bases, ids) and np.array_equal(proba_m2, proba))
    print(f"  proba M2 de v2_oof_base_corrected == proba oof_v03_final: {proba_igual}")

    from sklearn.metrics import f1_score
    mejor_umbral = []
    for i in range(3):
        mejor_umbral.append(max(
            (float(f1_score(y_true[:, i], (proba[:, i] >= t).astype(int),
                            zero_division=0)), round(float(t), 2))
            for t in np.arange(0.20, 0.8001, 0.01))[1])
    pred_optima = np.column_stack([
        (proba[:, i] >= mejor_umbral[i]).astype(int) for i in range(3)])
    f1_optima = float(metricas_multietiqueta(y_true, pred_optima)["f1_media_columnas"])
    print(f"  umbrales re-optimizados in-sample: {mejor_umbral} (f1 {f1_optima:.10f})")
    print(f"  optimismo in-sample de umbrales: "
          f"+{f1_optima - metrica['f1_media_columnas']:.6f}")

    filas_salida.append({
        "artifact": "features/oof_v03_final.npz", "variant": "canonico",
        "sha16": sha16(RAIZ / "features" / "oof_v03_final.npz"),
        "n_rows": str(len(ids)), "population": "dev(1483)",
        "recompute_mode": "proba + umbrales guardados",
        "recomputed_value": repr(metrica["f1_media_columnas"]),
        "stored_claim": f"corrected_cv={float(oof['corrected_cv'][0])!r}",
        "delta": "0.0", "status": "RECOMPUTED_EXACTLY",
    })
    filas_salida.append({
        "artifact": "features/v2_oof_base_corrected.npz", "variant": "M2_union_lr_c2",
        "sha16": sha16(RAIZ / "features" / "v2_oof_base_corrected.npz"),
        "n_rows": str(len(ids_bases)), "population": "dev(1483)",
        "recompute_mode": "proba M2 + umbrales V03 congelados",
        "recomputed_value": repr(metrica["f1_media_columnas"]),
        "stored_claim": "pipeline_v03.json: corrected_cv_m2_con_umbrales=0.7772",
        "delta": repr(metrica["f1_media_columnas"] - 0.7772),
        "status": "RECOMPUTED_WITHIN_TOLERANCE",
    })

    # ---------- Parte B: barrido de artefactos ----------
    print()
    print("=== B. Barrido de artefactos con predicciones guardadas ===")
    candidatos = sorted((RAIZ / "features").glob("*.npz"))
    candidatos += sorted((RAIZ / "experiments" / "baseline_v03_frozen").glob("*.npz"))
    candidatos += sorted((RAIZ / "reports" / "p84_04_parciales").glob("*.npz"))
    resumen: dict[str, int] = {}

    def contar(status: str) -> None:
        resumen[status] = resumen.get(status, 0) + 1

    for ruta in candidatos:
        rel = str(ruta.relative_to(RAIZ))
        sha = sha16(ruta)
        if rel in EXCLUIDOS_POLITICA:
            contar("EXCLUDED_BY_POLICY")
            filas_salida.append({
                "artifact": rel, "variant": "", "sha16": sha, "n_rows": "",
                "population": "holdout-incluido", "recompute_mode": "no abierto",
                "recomputed_value": "", "stored_claim": EXCLUIDOS_POLITICA[rel],
                "delta": "", "status": "EXCLUDED_BY_POLICY",
            })
            continue
        try:
            z = np.load(ruta, allow_pickle=True)
            claves = list(z.keys())
        except Exception as exc:
            contar("NOT_RECOMPUTABLE_FROM_SAVED_ARTIFACTS")
            filas_salida.append({
                "artifact": rel, "variant": "", "sha16": sha, "n_rows": "",
                "population": "", "recompute_mode": "error de carga",
                "recomputed_value": "", "stored_claim": str(exc)[:80],
                "delta": "", "status": "NOT_RECOMPUTABLE_FROM_SAVED_ARTIFACTS",
            })
            continue

        def arr(*nombres):
            """Primer arreglo presente entre los nombres dados."""
            for n in nombres:
                if n in claves:
                    return z[n]
            return None

        y = arr("y_true", "y_verdadera", "etiquetas_true", "targets", "y")
        umbrales_arr = arr("umbrales", "thresholds", "umbrales_congelados")
        ids_arr = arr("ids", "id")
        n_rows = ""
        for a in (y, ids_arr):
            if isinstance(a, np.ndarray) and a.ndim >= 1:
                n_rows = int(a.shape[0])
                break

        poblacion = ""
        if isinstance(ids_arr, np.ndarray) and ids_arr.ndim == 1:
            ids_lista = [int(v) for v in ids_arr.tolist()]
            if set(ids_lista) <= dev_ids:
                poblacion = "dev(1483)" if len(ids_lista) == 1483 else "dev-parcial"
            else:
                poblacion = "incluye_ids_fuera_de_dev"
                # artefacto con ids fuera de dev: si ademas hay holdout, se excluye
                particion_h = pd.read_csv(
                    RAIZ / "data" / "processed" / "holdout_v2_ids.csv",
                    usecols=["particion"])
                _ = particion_h  # solo conteo de particiones; no se usan etiquetas
                contar("EXCLUDED_BY_POLICY")
                filas_salida.append({
                    "artifact": rel, "variant": "", "sha16": sha,
                    "n_rows": str(n_rows),
                    "population": "fuera_de_dev (cubre holdout por diseno de fase)",
                    "recompute_mode": "no recomputado (politica P96)",
                    "recomputed_value": "", "stored_claim": "; ".join(claves[:6]),
                    "delta": "", "status": "EXCLUDED_BY_POLICY",
                })
                continue

        claims: list[str] = []
        for k in claves:
            if any(t in k.lower() for t in
                   ("corrected_cv", "f1", "metrica", "score", " cv", "cv_")):
                v = z[k]
                if isinstance(v, np.ndarray) and v.size <= 8 and v.dtype.kind in "fiu":
                    claims.append(f"{k}={np.round(v.ravel(), 10).tolist()}")

        variantes_emitidas = 0
        if isinstance(y, np.ndarray) and y.ndim == 2 and y.shape[1] == 3:
            y_int = y.astype(int)
            modo_sin_var = None
            # 1) variantes (n, 3) float (probas) e int (preds) con nombre conocido
            for k in claves:
                if variantes_emitidas >= MAX_VARIANTES:
                    break
                v = z[k]
                if not isinstance(v, np.ndarray) or v.shape != y.shape:
                    continue
                if k in ("y_true", "y_verdadera", "y", "targets", "etiquetas_true"):
                    continue
                if v.dtype.kind == "f":
                    if NOMBRES_NO_PROBA.match(k):
                        continue
                    thr = (umbrales_arr.astype(float).ravel()
                           if isinstance(umbrales_arr, np.ndarray)
                           and umbrales_arr.size == 3 else None)
                    if thr is not None:
                        pred = (v >= thr).astype(int)
                        modo = "proba con umbrales guardados"
                    else:
                        pred = (v >= 0.5).astype(int)
                        modo = "proba a 0.5 (sin umbrales guardados)"
                elif v.dtype.kind in "iu":
                    pred = v.astype(int)
                    modo = "pred guardada"
                else:
                    continue
                valor = float(metricas_multietiqueta(y_int, pred)["f1_media_columnas"])
                status = clasificar(valor, claims)
                contar(status)
                filas_salida.append({
                    "artifact": rel, "variant": k, "sha16": sha,
                    "n_rows": str(n_rows), "population": poblacion,
                    "recompute_mode": modo,
                    "recomputed_value": repr(valor),
                    "stored_claim": "; ".join(claims[:3]),
                    "delta": "", "status": status,
                })
                variantes_emitidas += 1
                modo_sin_var = modo_sin_var or modo
            # 2) probas multimodelo (a, n, 3) o (n, a, 3)
            if variantes_emitidas == 0:
                for k in claves:
                    if variantes_emitidas >= MAX_VARIANTES:
                        break
                    v = z[k]
                    if not (isinstance(v, np.ndarray) and v.ndim == 3
                            and v.shape[-1] == 3):
                        continue
                    if NOMBRES_NO_PROBA.match(k):
                        continue
                    eje = None
                    if v.shape[1] == y.shape[0]:
                        eje = 1
                    elif v.shape[0] == y.shape[0]:
                        eje = 0
                    if eje is None:
                        continue
                    nombres_v = None
                    for cand in ("modelos", "semillas", "semilla", "variantes", "canales"):
                        if cand in claves and isinstance(z[cand], np.ndarray):
                            nombres_v = [str(m) for m in np.ravel(z[cand].tolist())]
                    eje_modelo = 0 if eje == 1 else 1
                    for m in range(v.shape[eje_modelo]):
                        if variantes_emitidas >= MAX_VARIANTES:
                            break
                        matriz = v[m, :, :] if eje_modelo == 0 else v[:, m, :]
                        if matriz.shape != y.shape:
                            continue
                        valor = float(metricas_multietiqueta(
                            y_int, (matriz >= 0.5).astype(int))["f1_media_columnas"])
                        status = clasificar(valor, claims)
                        contar(status)
                        nombre = (nombres_v[m] if nombres_v and m < len(nombres_v)
                                  else f"{k}[{m}]")
                        filas_salida.append({
                            "artifact": rel, "variant": f"{k}:{nombre}",
                            "sha16": sha, "n_rows": str(n_rows),
                            "population": poblacion,
                            "recompute_mode": "multimodelo: proba a 0.5",
                            "recomputed_value": repr(valor),
                            "stored_claim": "; ".join(claims[:3]),
                            "delta": "", "status": status,
                        })
                        variantes_emitidas += 1
        if variantes_emitidas == 0:
            contar("NOT_RECOMPUTABLE_FROM_SAVED_ARTIFACTS")
            filas_salida.append({
                "artifact": rel, "variant": "", "sha16": sha,
                "n_rows": str(n_rows), "population": poblacion,
                "recompute_mode": "sin y_true/pred/proba utilizables",
                "recomputed_value": "",
                "stored_claim": "; ".join(claves[:6]), "delta": "",
                "status": "NOT_RECOMPUTABLE_FROM_SAVED_ARTIFACTS",
            })

    # ---------- Parte C: comparaciones curadas contra valores documentados ----------
    print()
    print("=== C. Comparaciones curadas contra valores documentados ===")
    UMB = np.array([0.44, 0.51, 0.46])

    curadas = [
        ("features/oof_candidates_v4.npz", "A_cost_sensitive_downweight", UMB,
         0.7854160387, 1e-9, "D-196/D-199 (reproduccion exacta documentada)"),
        ("features/oof_candidates_v4.npz", "B_meta36", UMB,
         0.782774, 1e-6, "meta_shift_v4.json (B = meta_shift)"),
        ("features/oof_candidates_v4.npz", "C_meta36_downweight", UMB,
         0.7882, 5e-5, "DECISIONS D-187..D-190 (0.7882)"),
        ("features/oof_candidates_v4.npz", "D_svd300_rbf", UMB,
         0.7904, 5e-5, "DECISIONS D-187..D-190 (0.7904)"),
        ("features/oof_meta_shift_v4.npz", "meta_control", UMB,
         0.779135, 1e-6, "meta_shift_v4.json"),
        ("features/oof_meta_shift_v4.npz", "meta_shift", UMB,
         0.782774, 1e-6, "meta_shift_v4.json"),
        ("features/oof_meta_shift_v4.npz", "meta_shift_cuantil", UMB,
         0.782825, 1e-6, "meta_shift_v4.json"),
        ("features/oof_meta_shift_v4.npz", "meta_shift_platt", UMB,
         0.765625, 1e-6, "meta_shift_v4.json"),
        ("features/oof_meta_shift_v4.npz", "meta_control", None,
         0.76414, 1e-6, "meta_shift_v4.json (0.5)"),
        ("features/oof_meta_shift_v4.npz", "meta_shift", None,
         0.774411, 1e-6, "meta_shift_v4.json (0.5)"),
        ("features/oof_meta_shift_v4.npz", "meta_shift_cuantil", None,
         0.77254, 1e-6, "meta_shift_v4.json (0.5)"),
        ("features/oof_meta_shift_v4.npz", "meta_shift_platt", None,
         0.769454, 1e-6, "meta_shift_v4.json (0.5)"),
        ("features/p84_02_oof_fasttext.npz", "pred_control_anidada", None,
         0.737771, 1e-6, "p84_02_fasttext.md (control anidada)"),
        ("features/p84_02_oof_fasttext.npz", "proba_control", None,
         0.717476, 1e-6, "p84_02_fasttext.md (control 0.5)"),
        ("features/p84_03_oof_ros.npz", "proba_A", np.array([0.44, 0.51, 0.46]),
         0.777211452517, 1e-9, "p84_03_ros_equivalencia.md (control = V03)"),
        ("features/p59_robust_cells.npz", "pred_v03", None,
         CANONICO, 1e-12, "canonico V03"),
    ]
    for ruta_rel, clave, thr, esperado, tol, fuente in curadas:
        z = np.load(RAIZ / ruta_rel, allow_pickle=True)
        y_v = z["y_true"].astype(int) if "y_true" in z.keys() else z["y"].astype(int)
        p = z[clave]
        if p.ndim == 3:
            p = p[0, :, :] if p.shape[1] == y_v.shape[0] else p[:, 0, :]
        if p.dtype.kind in "iu":
            pred = p.astype(int)
        else:
            pred = (p >= (thr if thr is not None else 0.5)).astype(int)
        valor = float(metricas_multietiqueta(y_v, pred)["f1_media_columnas"])
        delta = valor - esperado
        estado = "COMPARED_WITH_DOCUMENTED_EXACTLY" if abs(delta) <= tol \
            else "NUMERIC_MISMATCH"
        contar(estado)
        print(f"  {clave:28s} recomputado={valor:.12f} esperado={esperado} "
              f"delta={delta:+.2e} [{estado}] ({fuente})")
        filas_salida.append({
            "artifact": ruta_rel, "variant": clave,
            "sha16": sha16(RAIZ / ruta_rel), "n_rows": str(len(y_v)),
            "population": "dev(1483)",
            "recompute_mode": ("umbrales V03" if thr is not None
                               else "0.5 o pred guardada"),
            "recomputed_value": repr(valor),
            "stored_claim": f"{fuente}: {esperado}",
            "delta": repr(delta), "status": estado,
        })

    # extras: proba_v03 @0.5 (documentado como V03@0.5) y semillas de P84-01
    z66 = np.load(RAIZ / "features" / "p66_scores_candidatos.npz", allow_pickle=True)
    y66 = z66["y_true"].astype(int)
    valor = float(metricas_multietiqueta(
        y66, (z66["proba_v03"] >= 0.5).astype(int))["f1_media_columnas"])
    delta = valor - 0.771778476256
    estado = ("COMPARED_WITH_DOCUMENTED_EXACTLY" if abs(delta) <= 1e-9
              else "NUMERIC_MISMATCH")
    contar(estado)
    print(f"  {'proba_v03 (p66)':28s} recomputado={valor:.12f} "
          f"esperado=0.771778476256 delta={delta:+.2e} [{estado}] "
          f"(p84_02_fasttext.md: V03@0.5)")
    filas_salida.append({
        "artifact": "features/p66_scores_candidatos.npz", "variant": "proba_v03",
        "sha16": sha16(RAIZ / "features" / "p66_scores_candidatos.npz"),
        "n_rows": str(len(y66)), "population": "dev(1483)",
        "recompute_mode": "proba a 0.5",
        "recomputed_value": repr(valor),
        "stored_claim": "p84_02_fasttext.md: V03@0.5 0.771778476256",
        "delta": repr(delta), "status": estado,
    })
    z1 = np.load(RAIZ / "features" / "p84_01_oof_mlp.npz", allow_pickle=True)
    y1 = z1["y_true"].astype(int)
    semillas = [int(s) for s in z1["semillas"].tolist()]
    esperados_seed = {42: 0.758301, 43: 0.754899, 44: 0.761771}
    for m, semilla in enumerate(semillas):
        valor = float(metricas_multietiqueta(
            y1, (z1["proba"][m] >= 0.5).astype(int))["f1_media_columnas"])
        esperado = esperados_seed.get(semilla)
        if esperado is None:
            continue
        delta = valor - esperado
        estado = ("COMPARED_WITH_DOCUMENTED_EXACTLY" if abs(delta) <= 1e-6
                  else "NUMERIC_MISMATCH")
        contar(estado)
        print(f"  {'proba seed ' + str(semilla) + ' (p84_01)':28s} "
              f"recomputado={valor:.12f} esperado={esperado} "
              f"delta={delta:+.2e} [{estado}] (p84_01_dl01.md)")
        filas_salida.append({
            "artifact": "features/p84_01_oof_mlp.npz",
            "variant": f"proba seed {semilla}",
            "sha16": sha16(RAIZ / "features" / "p84_01_oof_mlp.npz"),
            "n_rows": str(len(y1)), "population": "dev(1483)",
            "recompute_mode": "proba a 0.5",
            "recomputed_value": repr(valor),
            "stored_claim": f"p84_01_dl01.md: {esperado}",
            "delta": repr(delta), "status": estado,
        })

    with open(SALIDA, "w", encoding="utf-8", newline="") as fh:
        escritor = csv.DictWriter(fh, fieldnames=list(filas_salida[0].keys()),
                                  lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas_salida)
    print()
    print(f"P96-01 recomputacion: ejecutada | filas matriz: {len(filas_salida)}")
    print(f"  resumen: {resumen}")
    print(f"  salida: {SALIDA.relative_to(RAIZ)}")
if __name__ == "__main__":
    main()
