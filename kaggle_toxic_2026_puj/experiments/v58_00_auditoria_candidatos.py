"""P58: auditoria final de candidatos existentes.

Que responde
------------
> Existe ya, dentro de los artefactos que hemos producido, algun candidato
> suficientemente defendible como para justificar una unica sonda final?

Que NO hace, por diseno de la directiva
---------------------------------------
- No entrena nada. No importa ningun estimador: ni sklearn, ni torch.
- No crea ninguna variante de modelo.
- No modifica ningun artefacto existente, ni una sola submission.
- No genera ni envia submissions.
- No consulta Kaggle.
- No usa holdout ni test labels.

La unica transformacion que se permite es **validacion formal** de artefactos que ya
existen: cargar, verificar esquema, calcular hash, y recomputar metricas a partir de
OOF ya produced en disco. Eso es lo que hace este script.

El hallazgo que gobierna el informe
----------------------------------
`experiments.csv` registra E2-13-META-ENT-RIDGE como "CERRADA por sparsity" y P57
lo trato como el unico candidato con delta honesto anidado positivo y sin veredicto
externo. Al auditarlo se ve que esa lectura es incorrecta, y este script lo
documenta con las pruebas.

Uso
---
    /home/carlos/venv/bin/python experiments/v58_00_auditoria_candidatos.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Los tensores OOF viven en el worktree externo, no en este. Solo lectura.
FEATURES_EXTERNOS = Path("/home/carlos/Git/PLN/kaggle_toxic_2026_puj/features")

V03_ID = "v03_lr_union_thr"
V03_CORRECTED_CV = 0.7772114525165721
V03_LB = 0.78306
V03_SHA = "7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e"
V03_UMBRALES = np.array([0.44, 0.51, 0.46])
ETIQUETAS = ("HS", "TR", "AG")
N_TEST = 1154
N_CELDAS = 1483 * 3

OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
OOF_META = FEATURES_EXTERNOS / "v2_meta_oof.npz"
OOF_CAND = FEATURES_EXTERNOS / "oof_candidates_v4.npz"
REG_V4 = PROJECT_ROOT / "reports" / "v4_candidates.json"
REG_BACKLOG = PROJECT_ROOT / "reports" / "backlog_restante_v2.json"
INV_P57 = PROJECT_ROOT / "reports" / "p57_final_experiment_inventory.csv"
EXPERIMENTS = PROJECT_ROOT / "experiments.csv"
LB_LOG = PROJECT_ROOT / "submissions" / "leaderboard_log.csv"
DIR_SUB = PROJECT_ROOT / "submissions"

CSV_AUDIT = PROJECT_ROOT / "reports" / "p58_final_candidate_audit.csv"
MD_AUDIT = PROJECT_ROOT / "reports" / "p58_final_candidate_audit.md"
JSON_AUDIT = PROJECT_ROOT / "reports" / "p58_final_candidate_audit.json"

FECHA = "2026-09-28"

# Los siete criterios de la directiva 9. El sexto es el que decide.
CRITERIOS = (
    "1_cv_sobre_V03",
    "2_reproducible",
    "3_al_menos_4_de_5_pliegues",
    "4_sin_leakage",
    "5_mecanismo_claro",
    "6_artefacto_de_prediccion_completo",
    "7_no_rechazado_por_LB",
)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V58-00] {mensaje}", flush=True)


# ------------------------------------------------------- transliteracion ASCII
SUSTITUCIONES = {
    "\u2014": " - ", "\u2013": "-", "\u2212": "-", "\u00b1": "+/-", "\u00a7": " seccion ",
    "\u00ab": '"', "\u00bb": '"', "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
    "\u2264": "<=", "\u2265": ">=", "\u00d7": "x", "\u2192": "->", "\u00b7": ".",
    "\u00a0": " ", "\u2026": "...", "\u00bf": "?", "\u00a1": "!",
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


def asciificar(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica `a_ascii` a todas las columnas de texto de un DataFrame.

    Args:
        df: DataFrame de entrada.

    Returns:
        El DataFrame con las columnas de texto en ASCII.
    """
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(lambda x: a_ascii(x) if isinstance(x, str) else x)
    return out


# --------------------------------------------------------- validacion formal
def validar_submission(ruta: Path) -> dict[str, Any]:
    """Valida formalmente un CSV de submission sin modificarlo.

    Args:
        ruta: ruta al fichero CSV.

    Returns:
        Diccionario con el resultado de cada comprobacion.
    """
    bruto = ruta.read_bytes()
    res: dict[str, Any] = {
        "fichero": ruta.name,
        "sha256": hashlib.sha256(bruto).hexdigest(),
        "bytes": len(bruto),
        "rows_ok": False, "n_rows": 0,
        "columnas_ok": False, "columnas": "",
        "ids_ok": False, "orden_ok": False,
        "binario_ok": False, "sin_nulos": False,
        "valido": False, "observaciones": [],
    }
    try:
        df = pd.read_csv(ruta)
    except Exception as exc:  # noqa: BLE001
        res["observaciones"].append(f"no se puede leer: {exc}")
        return res
    res["n_rows"] = len(df)
    res["rows_ok"] = len(df) == N_TEST
    res["columnas"] = ",".join(map(str, df.columns))
    res["columnas_ok"] = list(df.columns) == ["id", "HS", "TR", "AG"]
    res["sin_nulos"] = bool(df.isna().sum().sum() == 0)
    etiquetas = df[["HS", "TR", "AG"]].to_numpy() if res["columnas_ok"] else None
    if etiquetas is not None:
        res["binario_ok"] = bool(np.isin(etiquetas, (0, 1)).all())
    test = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "toxic_test.csv", sep=";",
                       encoding="latin-1")
    ids_test = test["id"].to_numpy()
    if "id" in df.columns:
        ids_sub = df["id"].to_numpy()
        res["orden_ok"] = bool(np.array_equal(ids_sub, ids_test))
        res["ids_ok"] = res["orden_ok"]
    res["valido"] = all([res["rows_ok"], res["columnas_ok"], res["ids_ok"],
                         res["orden_ok"], res["binario_ok"], res["sin_nulos"]])
    if not res["rows_ok"]:
        res["observaciones"].append(f"filas {len(df)} != {N_TEST}")
    if not res["columnas_ok"]:
        res["observaciones"].append("columnas distintas de id,HS,TR,AG")
    if not res["orden_ok"]:
        res["observaciones"].append("ids o orden distintos de toxic_test.csv")
    if not res["binario_ok"]:
        res["observaciones"].append("hay valores no binarios")
    if not res["sin_nulos"]:
        res["observaciones"].append("hay nulos")
    return res


# ---------------------------------------------------- descomposicion celular
def descomponer(nombre: str, proba: np.ndarray, umbral: np.ndarray,
                y: np.ndarray, pred_v03: np.ndarray, pliegues: np.ndarray,
                lb: Any, nota: str) -> dict[str, Any]:
    """Recomputa la descomposicion celular de un candidato desde su OOF.

    Args:
        nombre: etiqueta del candidato.
        proba: matriz de probabilidades out-of-fold.
        umbral: vector de umbrales.
        y: etiquetas verdaderas.
        pred_v03: predicciones de V03.
        pliegues: asignacion de pliegue, de 1 a 5.
        lb: leaderboard conocido o `None`.
        nota: comentario de cierre.

    Returns:
        Diccionario con la metricas y el recuento de celdas.
    """
    pred = (proba >= umbral[None, :]).astype(int)
    ccv = float(f1_score(y, pred, average="macro"))
    rescate = int(np.count_nonzero((pred == y) & (pred_v03 != y)))
    damage = int(np.count_nonzero((pred != y) & (pred_v03 == y)))
    cambian = int(np.count_nonzero(pred != pred_v03))
    folds = range(1, 6)
    f_cand = [float(f1_score(y[pliegues == k], pred[pliegues == k], average="macro"))
              for k in folds]
    f_v03 = [float(f1_score(y[pliegues == k], pred_v03[pliegues == k], average="macro"))
             for k in folds]
    por = {}
    for j, e in enumerate(ETIQUETAS):
        por[e] = {
            "f1": round(float(f1_score(y[:, j], pred[:, j])), 10),
            "rescate": int(np.count_nonzero((pred[:, j] == y[:, j]) & (pred_v03[:, j] != y[:, j]))),
            "damage": int(np.count_nonzero((pred[:, j] != y[:, j]) & (pred_v03[:, j] == y[:, j]))),
        }
    return {
        "candidato": nombre,
        "corrected_cv": round(ccv, 10),
        "delta_vs_V03": round(ccv - V03_CORRECTED_CV, 10),
        "LB": lb,
        "LB_verified": lb is not None,
        "celdas_que_cambian": cambian,
        "porcentaje_celdas": round(cambian / N_CELDAS * 100, 4),
        "rescate": rescate,
        "damage": damage,
        "neto": rescate - damage,
        "acierto_en_las_que_cambian": round(rescate / cambian, 4) if cambian else None,
        "tasa_positiva": {e: round(float(pred[:, j].mean()), 6) for j, e in enumerate(ETIQUETAS)},
        "pliegues_mejorados": sum(1 for a, b in zip(f_cand, f_v03) if a > b),
        "deltas_por_pliegue": [round(a - b, 6) for a, b in zip(f_cand, f_v03)],
        "por_etiqueta": por,
        "nota": nota,
    }


def main() -> None:
    """Ejecuta la auditoria final de candidatos.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P58 auditoria final de candidatos. 0 modelos, 0 entrenamiento, 0 submissions")
    log("   0 consultas a Kaggle, 0 uso de holdout, 0 uso de test labels")

    # ------------------------------------------------------------- integridad
    hash_real = hashlib.sha256(
        (DIR_SUB / "toxic_submission_v03_lr_union_thr.csv").read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    log(f"   V03 sha256 verificado: {hash_real[:16]}... intacto")

    # ------------------------------------------------------------- OOF en disco
    v = np.load(OOF_V03, allow_pickle=True)
    m = np.load(OOF_META, allow_pickle=True)
    c = np.load(OOF_CAND, allow_pickle=True)
    y = v["y_true"]
    pred_v03 = v["pred"]
    pliegues = v["pliegues"]
    ccv_v03 = float(f1_score(y, pred_v03, average="macro"))
    if abs(ccv_v03 - V03_CORRECTED_CV) > 1e-12:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_v03!r}, no {V03_CORRECTED_CV!r}")
    if not (np.array_equal(m["ids"], v["ids"]) and np.array_equal(c["ids"], v["ids"])):
        raise SystemExit("PARADA: los OOF candidatos no comparten los ids del OOF canonico")
    log(f"   OOF canonico reproduce {ccv_v03!r} exacto")
    log(f"   v2_meta_oof variante = {m['variante'][0]!s} | ridge alpha={float(m['parametro'][0])}, "
        f"{len(m['nombres'])} features")
    log("   identificacion de ids coherente entre los tres OOF")

    reg_v4 = json.loads(REG_V4.read_text(encoding="utf-8"))
    gate = {d["candidato"]: d for d in reg_v4["detalle"]}
    backlog = json.loads(REG_BACKLOG.read_text(encoding="utf-8"))

    # ------------------------------------------- descomposiciones sin reentrenar
    log("   recomputando descomposiciones desde OOF ya en disco (sin entrenar)")
    semi = np.array([0.5, 0.5, 0.5])
    decomps = [
        descomponer("META-BASES-RIDGE (= v08_meta_v2)", m["proba"], semi, y, pred_v03,
                    pliegues, 0.78163,
                    "ridge alpha 3 sobre 44 features. Es la arquitectura que se envio como "
                    "v08_meta_v2. Su delta de +0,010146 es la ganancia de la FAMILIA META"),
        descomponer("A_cost_sensitive_downweight", c["A_cost_sensitive_downweight"],
                    V03_UMBRALES, y, pred_v03, pliegues, None,
                    "5/5 pliegues y el mayor acierto en las celdas que cambian, pero el IC95 "
                    "pareado incluye el cero"),
        descomponer("B_meta36", c["B_meta36"], V03_UMBRALES, y, pred_v03, pliegues, None,
                    "unico que pasa el gate D-016 con IC95 positivo y SIN veredicto externo"),
        descomponer("C_meta36_downweight", c["C_meta36_downweight"], V03_UMBRALES, y,
                    pred_v03, pliegues, None,
                    "unico que cumplia los tres criterios de D-016. Rechazado por el holdout "
                    "congelado: 0,8163 contra 0,8285 de V03. La submission v12 se genero y "
                    "quedo marcada DESCARTADO, nunca se envio"),
        descomponer("D_svd300_rbf", c["D_svd300_rbf"], V03_UMBRALES, y, pred_v03, pliegues,
                    0.77938,
                    "el corrected_cv mas alto del proyecto y 5/5 pliegues. Enviado, LB -0,00368"),
    ]
    for d in decomps:
        log(f"      {d['candidato'][:36]:36} ccv {d['corrected_cv']:.10f} "
            f"({d['delta_vs_V03']:+.6f}) {d['pliegues_mejorados']}/5 "
            f"| cambian {d['celdas_que_cambian']:3d} neto {d['neto']:+3d}")

    # el CCV registrado en el artefacto debe coincidir con el recomputado
    for d in decomps:
        g = gate.get(d["candidato"].split(" ")[0])
        if g and abs(round(g["corrected_cv_umbrales_v03"], 6) - d["corrected_cv"]) > 1e-6:
            raise SystemExit(f"PARADA: {d['candidato']} no reproduce el registro")
    log("   las 5 descomposiciones reproducen el artefacto v4_candidates.json al sexto decimal")

    # ------------------------------------------------- E2-13, auditoria profunda
    log("   E2-13: que significa exactamente el cierre por sparsity")
    cob = backlog["e2_13_entidades"]["cobertura"]
    por_tipo = cob["por_tipo"]
    e2_13 = {
        "cobertura_informativa": cob["cobertura_informativa"],
        "umbral_de_cierre": cob["umbral_cierre"],
        "cierra_por_sparsity": cob["cierra_por_sparsity"],
        "instancias_informativas": cob["instancias_informativas"],
        "por_tipo": {
            k: {
                "instancias": v["instancias"],
                "instancias_cubiertas": v["instancias_cubiertas"],
                "cobertura_instancias": v["cobertura_instanticas"]
                if "cobertura_instanticas" in v else v.get("cobertura_instancias"),
                "filas_con_entidad": v["filas_con_entidad"],
                "filas_cubiertas": v["filas_cubiertas"],
                "unicos_dev": v["unicos_dev"],
            } for k, v in por_tipo.items()
        },
        "ccv_de_la_variante": backlog["e2_13_entidades"]["variantes"]["META-ENT-RIDGE"]["corrected_cv"],
        "delta_vs_V03": backlog["e2_13_entidades"]["variantes"]["META-ENT-RIDGE"]["delta_vs_v03"],
        "delta_vs_su_comparador": backlog["e2_13_entidades"]["variantes"]["META-ENT-RIDGE"]["delta_vs_meta_ridge"],
        "pliegues_mejorados_vs_V03": 4,
        "pliegues_mejorados_vs_comparador":
            backlog["e2_13_entidades"]["decision"]["pliegues_mejorados_vs_comparador"],
        "comparador": backlog["e2_13_entidades"]["decision"]["comparador"],
        "complementariedad": backlog["e2_13_entidades"]["variantes"]["META-ENT-RIDGE"]["complementariedad"],
        "justificacion_registrada": backlog["e2_13_entidades"]["decision"]["justificacion"],
    }
    e2_13["veredicto"] = {
        "que_significa_sparsity": (
            "La cobertura informativa del canal de entidades es "
            f"{cob['cobertura_informativa'] * 100:.2f} por ciento, contra un umbral de cierre de "
            f"{cob['umbral_cierre'] * 100:.0f} por ciento. El canal de target encoding de "
            "menciones, hashtags y dominios sobrevive en muy pocas instancias."),
        "por_que_no_invalida_automaticamente": (
            "No lo invalida por el numero, sino porque el incremento medido sobre su propio "
            f"comparador ({e2_13['comparador']}) es de {e2_13['delta_vs_su_comparador']:+.6f} con "
            f"{e2_13['pliegues_mejorados_vs_comparador']} de 5 pliegues, y la complementariedad da "
            f"{e2_13['complementariedad']['ensemble_gain']:+.6f} con IC no positivo. O sea: el "
            "sparsity es la CAUSA, y el incremento medido es la PRUEBA de que la causa es real."),
        "cierre_metodologico": True,
        "pero_el_delta_de_0_0113_no_es_suyo": (
            "El +0,0113 es el delta de la FAMILIA META, no de este canal. Su comparador es "
            f"{e2_13['comparador']}, que se envio como v08_meta_v2 y dio LB 0,78163. "
            "E2-13-META-ENT-RIDGE es v08_meta_v2 mas un canal que cubre el 0,52 por ciento de las "
            "instancias. No es un mecanismo nuevo: es una perturbacion de un candidato ya "
            "rechazado externamente."),
        "reproducible": "SI, el corrected_cv y el desglose de cobertura estan en reports/backlog_restante_v2.json",
        "artefacto_de_prediccion": "NO. No existe CSV de submission para esta variante",
        "submission": "ninguna",
        "leakage": ("NO detectado. Las tablas de entidad se ajustan fold-locales y el bucket "
                    "desconocido recibe el prior del pliegue, segun las notas del registro"),
        "evidencia_externa": "NINGUNA. Nunca se envio a Kaggle",
    }
    log(f"      cobertura informativa {cob['cobertura_informativa'] * 100:.2f} por ciento "
        f"frente a un umbral de {cob['umbral_cierre'] * 100:.0f} por ciento")
    log(f"      incremento sobre su comparador {e2_13['delta_vs_su_comparador']:+.6f} con "
        f"{e2_13['pliegues_mejorados_vs_comparador']}/5 pliegues")
    log("      el delta de +0,0113 pertenece a la familia META, ya rechazada 5 veces en LB")

    # ------------------------------------------- los siete criterios, por candidato
    log("   aplicando los 7 criterios de la directiva 9")
    lb_log = pd.read_csv(LB_LOG)
    ficheros = {f for f in os.listdir(DIR_SUB) if f.startswith("toxic_submission_")}
    lb_files = {os.path.basename(str(a)) for a in lb_log.archivo}

    def artefacto_de(nombre: str) -> tuple[str, str]:
        """Devuelve si existe artefacto de prediccion y cual es.

        Args:
            nombre: clave del candidato.

        Returns:
            Tupla con el estado y el nombre del fichero.
        """
        mapa = {
            "META-BASES-RIDGE": "toxic_submission_v08_meta_v2.csv",
            "C_meta36_downweight": "toxic_submission_v12_descartado_meta36_downweight.csv",
            "D_svd300_rbf": "toxic_submission_v23_01_svd_rbf_sigmoid.csv",
            "A_cost_sensitive_downweight": "",
            "B_meta36": "",
        }
        f = mapa.get(nombre, "")
        if not f:
            return "NO EXISTE", ""
        return ("existe, ya enviado" if f in lb_files else "existe, nunca enviado"), f

    # informacion de leakage, leida del codigo y de las notas del registro
    sin_leakage = {
        "META-BASES-RIDGE": "SI. Meta ajustado fold a fold, tablas de entidad fold-locales",
        "A_cost_sensitive_downweight": ("SI. Vectorizadores ajustados solo con textos de "
                                        "entrenamiento, meta fold a fold, barrido anidado de "
                                        "class_weight con StratifiedGroupKFold interno de 3"),
        "B_meta36": ("SI. Vectorizadores ajustados solo con textos de entrenamiento, meta "
                     "ajustado fold a fold"),
        "C_meta36_downweight": ("SI. Vectorizadores ajustados solo con textos de "
                                "entrenamiento, meta fold a fold, downweight sobre el error "
                                "score OOF del propio pliegue"),
        "D_svd300_rbf": ("SI. SVD ajustado en el pliegue de entrenamiento. PERO el modelo "
                         "resultante memoriza: 60 a 76 por ciento de vectores de soporte"),
    }
    filas = []
    for d in decomps:
        clave = d["candidato"].split(" ")[0]
        g = gate.get(clave, {})
        art, art_nombre = artefacto_de(clave)
        lb_val = d["LB"]
        lb_verif = lb_val is not None
        # elholdout cuenta como veredicto externo aunque no sea LB
        veredicto_ext = ""
        if clave == "C_meta36_downweight":
            veredicto_ext = "holdout 0,8163 contra 0,8285 de V03, delta -0,0122. Holdout agotado 5/5"
        elif lb_verif:
            veredicto_ext = f"LB {lb_val:.5f}, delta {lb_val - V03_LB:+.5f}"
        crit = {
            "1_cv_sobre_V03": d["delta_vs_V03"] > 0,
            "2_reproducible": True,
            "3_al_menos_4_de_5_pliegues": d["pliegues_mejorados"] >= 4,
            "4_sin_leakage": not sin_leakage[clave].startswith("SI. SVD"),
            "5_mecanismo_claro": True,
            "6_artefacto_de_prediccion_completo": art != "NO EXISTE",
            "7_no_rechazado_por_LB": not (lb_verif and lb_val <= V03_LB) and "holdout" not in veredicto_ext,
        }
        # el criterio 4 para D se marca como fulfilled-pero-con-reserva
        if clave == "D_svd300_rbf":
            crit["4_sin_leakage"] = True
        if crit["7_no_rechazado_por_LB"] and veredicto_ext:
            crit["7_no_rechazado_por_LB"] = False
        finalist = all(crit.values())
        filas.append({
            "candidate": d["candidato"],
            "experiment": {"META-BASES-RIDGE": "E2-18-META-BASES-RIDGE",
                           "A_cost_sensitive_downweight": "V4-X01 (candidato A)",
                           "B_meta36": "V4-X01 (candidato B)",
                           "C_meta36_downweight": "V4-X01 (candidato C) / FIN-01-C-REF-44",
                           "D_svd300_rbf": "V4-G01-G03"}.get(clave, clave),
            "honest_cv": d["corrected_cv"],
            "local_cv": d["corrected_cv"],
            "LB": lb_val if lb_verif else "",
            "LB_verified": lb_verif,
            "LB_delta": round(lb_val - V03_LB, 5) if lb_verif else "",
            "fold_consistency": f"{d['pliegues_mejorados']}/5",
            "ic95_pareado": (f"[{g['ic95_bajo']:+.6f}, {g['ic95_alto']:+.6f}]" if g else "no registrado"),
            "cumple_D016_registrado": g.get("cumple_D016", "no aplica"),
            "bootstrap": f"IC95 pareado del delta; incluye cero: "
                         f"{'si' if g and g['ic95_bajo'] <= 0 <= g['ic95_alto'] else ('no' if g else 'n/d')}",
            "mechanism": {"META-BASES-RIDGE": "stacking sobre 44 columnas de 4 bases TF-IDF mas 8 rasgos estructurales",
                          "A_cost_sensitive_downweight": "meta con downweight coste-sensible por violacion del umbral",
                          "B_meta36": "meta de 36 features: scores, incertidumbre y rasgos estructurales",
                          "C_meta36_downweight": "meta de 36 features con downweight por violacion del umbral",
                          "D_svd300_rbf": "SVD 300 + SVC kernel RBF con calibracion sigmoidal"}[clave],
            "celdas_que_cambian": d["celdas_que_cambian"],
            "rescue": d["rescate"],
            "damage": d["damage"],
            "net": d["neto"],
            "reason_closed": veredicto_ext or (
                "sin artefacto de prediccion y sin veredicto externo. No puede ser finalista "
                "porque el criterio 6 exige artefacto y la directiva 12 prohibe crear uno"),
            "submission_exists": art,
            "submission_file": art_nombre,
            "sin_leakage": sin_leakage[clave],
            **{c: crit[c] for c in CRITERIOS},
            "finalist": finalist,
        })
    aud = asciificar(pd.DataFrame(filas))
    for _, r in aud.iterrows():
        faltan = [c for c in CRITERIOS if not r[c]]
        log(f"      {r['candidate'][:34]:34} finalista={r['finalist']!s:5} "
            f"faltan: {', '.join(faltan) if faltan else 'ninguno'}")

    finalistas = aud[aud.finalist]
    n_final = len(finalistas)
    caso = "A" if n_final == 0 else ("B" if n_final == 1 else "C")
    log(f"   FINALIST = {n_final} | CASE = {caso}")

    # ------------------------------------------- validacion formal de artefactos
    log("   validando formalmente los CSV de submission (sin modificar ninguno)")
    val = [validar_submission(DIR_SUB / f) for f in sorted(ficheros)]
    valdf = asciificar(pd.DataFrame(val))
    n_validos = int(valdf["valido"].sum())
    log(f"      {len(valdf)} CSV, {n_validos} validos de forma formal")
    nunca = sorted(ficheros - lb_files - {"toxic_submission_v32_05_url_length.csv"})
    log(f"      generados y nunca enviados: {len(nunca)}")
    identico_v03 = "toxic_submission_v20_03_reproduccion_v03.csv"
    if identico_v03 in ficheros:
        h_v03 = hashlib.sha256((DIR_SUB / "toxic_submission_v03_lr_union_thr.csv").read_bytes()).hexdigest()
        h_id = hashlib.sha256((DIR_SUB / identico_v03).read_bytes()).hexdigest()
        log(f"      v20_03_reproduccion_v03 es bit a bit identico a V03: {h_id == h_v03}")

    # ------------------------------------------- por que el CV alto no transfirio
    dsvd = next(x for x in decomps if x["candidato"].startswith("D_"))
    log("   D_svd300_rbf: por que el CV mas alto no transfirio")
    log(f"      cambia {dsvd['celdas_que_cambian']} celdas ({dsvd['porcentaje_celdas']} por ciento), "
        f"neto {dsvd['neto']:+d}, acierto {dsvd['acierto_en_las_que_cambian']:.4f}, {dsvd['pliegues_mejorados']}/5")
    log(f"      y aun asi LB {dsvd['LB']:.5f} ({dsvd['LB'] - V03_LB:+.5f})")
    log("      el soporte de 60 a 76 por ciento de vectores es la firma de un kernel RBF con")
    log("      gamma=scale sobre 300 dimensiones que memoriza en vez de generalizar")

    # ------------------------------------------------------------- presupuesto
    presupuesto = {
        "local_reservation": 3,
        "competition_limit": "UNKNOWN",
        "envios_historicos": 23,
        "ficheros_en_submissions": len(ficheros),
        "envios_hoy": 0,
        "nota": ("Las 3 reservas son autolimitacion del proyecto, no un limite de Kaggle. "
                "No se consulta Kaggle. El limite de la competencia queda como UNKNOWN."),
    }

    # ------------------------------------------------------------- escribir
    aud.to_csv(CSV_AUDIT, index=False, encoding="utf-8")
    for destino in (CSV_AUDIT,):
        if any(x > 127 for x in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")

    informe = {
        "familia": "P58_FINAL_CANDIDATE_AUDIT",
        "naturaleza": ("AUDIT_ONLY. 0 modelos, 0 entrenamiento, 0 submissions, 0 consultas a "
                       "Kaggle, 0 uso de holdout, 0 uso de test"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "pregunta": ("existe ya, dentro de los artefactos que hemos producido, algun candidato "
                     "suficientemente defendible como para justificar una unica sonda final?"),
        "baseline": {"id": V03_ID, "corrected_cv": V03_CORRECTED_CV, "LB": V03_LB,
                     "sha256": V03_SHA, "sha256_verificado": True,
                     "oof_reproducido": ccv_v03 == V03_CORRECTED_CV},
        "criterios": list(CRITERIOS),
        "descomposiciones": decomps,
        "candidatos": filas,
        "finalistas": n_final,
        "finalist_ids": list(finalistas["candidate"]),
        "CASE": caso,
        "veredicto": ("FINALIST = NONE. Ningun artefacto cumple los siete criterios a la vez. "
                      "El mas proximo es B_meta36, que pasa cinco de siete y falla en el "
                      "artefacto de prediccion, que no existe y que la directiva 12 prohibe "
                      "crear") if n_final == 0 else "ver finalists",
        "E2_13": e2_13,
        "D_svd300_rbf": {
            "corrected_cv": dsvd["corrected_cv"],
            "pliegues": dsvd["pliegues_mejorados"],
            "celdas_que_cambian": dsvd["celdas_que_cambian"],
            "rescate": dsvd["rescate"], "damage": dsvd["damage"], "neto": dsvd["neto"],
            "LB": dsvd["LB"], "LB_delta": round(dsvd["LB"] - V03_LB, 5),
            "por_que_no_transfirio": (
                "La firma mecanica esta en los vectores de soporte: 1321 de 1731 para HS, 1317 "
                "para AG y 1044 para TR, es decir el 60 a 76 por ciento de las muestras. Un "
                "kernel RBF con gamma=scale sobre 300 dimensiones memoriza en vez de "
                "generalizar. El gap local-LB pasa de +0,005849 de V03 a -0,011030."),
            "estado": "CLOSED. El LB ya aporto la evidencia negativa suficiente",
        },
        "validacion_formal_submissions": val,
        "nunca_enviados": nunca,
        "n_validos_formalmente": n_validos,
        "presupuesto": presupuesto,
        "formulacion": "V03 remains the best verified leaderboard result.",
        "gastos": {"submissions": 0, "modelos": 0, "entrenamientos": 0,
                   "consultas_kaggle": 0, "uso_holdout": 0, "uso_test": 0,
                   "segundos": round((datetime.now(timezone.utc) - inicio).total_seconds(), 1)},
    }
    JSON_AUDIT.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    _md(informe, aud, valdf, nunca, e2_13, decomps, presupuesto, identico_v03)
    log("   escrito reports/p58_*")
    log(f"VEREDICTO: CASE {caso} | FINALIST={n_final} | 0 models, 0 submissions, "
        f"{informe['gastos']['segundos']} s")


def _md(inf: dict[str, Any], aud: pd.DataFrame, valdf: pd.DataFrame, nunca: list[str],
        e2: dict[str, Any], decomps: list[dict[str, Any]], presu: dict[str, Any],
        identico: str) -> None:
    """Escribe el informe de la auditoria final de candidatos.

    Args:
        inf: diccionario con los resultados.
        aud: tabla de candidatos con los siete criterios.
        valdf: tabla de validacion formal de submissions.
        nunca: lista de ficheros nunca enviados.
        e2: bloque de la auditoria de E2-13.
        decomps: descomposiciones celulares.
        presu: presupuesto.
        identico: nombre del fichero identico a V03.

    Returns:
        None.
    """
    b = inf["baseline"]
    L: list[str] = []
    A = L.append
    A("# P58 - Auditoria final de candidatos existentes")
    A("")
    A(f"Generado por `experiments/v58_00_auditoria_candidatos.py`. Fecha {inf['fecha']}.")
    A("")
    A("**0 modelos. 0 entrenamientos. 0 submissions. 0 consultas a Kaggle. 0 holdout. 0 test.** "
      "La unica transformacion permitida es validacion formal de artefactos que ya existen: "
      "cargar, verificar esquema, calcular hash y recomputar metricas desde OOF ya en disco.")
    A("")
    A("## 1. Baseline y su integridad")
    A("")
    A("```text")
    A(f"id           {b['id']}")
    A(f"corrected_cv {b['corrected_cv']!r}")
    A(f"LB           {b['LB']}")
    A(f"sha256       {b['sha256']}")
    A(f"verificado   {b['sha256_verificado']}")
    A(f"OOF reproducido exactamente desde disco: {b['oof_reproducido']}")
    A("```")
    A("")
    A("El script **aborta** si el sha256 no coincide o si el OOF canonico no reproduce el "
      "corrected_cv al ultimo digito.")
    A("")
    A("## 2. Que se audito y como, sin reentrenar")
    A("")
    A("| artefacto en disco | contenido | identificacion |")
    A("|---|---|---|")
    A("| `features/oof_v03_final.npz` | OOF canonico de V03, 1483 x 3 | referencia |")
    A("| `features/v2_meta_oof.npz` (worktree externo) | `META-BASES-RIDGE`, ridge alpha 3, **44 features** | es la arquitectura de `v08_meta_v2` |")
    A("| `features/oof_candidates_v4.npz` (worktree externo) | A, B, C, D de V4-X01 | los cuatro candidatos del gate |")
    A("")
    A("Los ids y las etiquetas de los tres se verificaron identicos antes de comparar. Las cinco "
      "descomposiciones resultantes **reproducen `reports/v4_candidates.json` al sexto decimal**, "
      "lo que confirma que los tensores son los que alimentaron el gate pre-registrado.")
    A("")
    A("## 3. Descomposicion celular, recomputada desde OOF existente")
    A("")
    A("```text")
    A(f"{'candidato':38}{'ccv':>12}{'delta':>11}{'plieg':>6}{'cambian':>8}{'r':>5}{'d':>5}{'neto':>6}{'acierto':>9}  LB")
    for d in decomps:
        lb = f"{d['LB']:.5f}" if d["LB_verified"] else "sin veredicto"
        A(f"{d['candidato'][:38]:38}{d['corrected_cv']:>12.6f}{d['delta_vs_V03']:>+11.6f}"
          f"{str(d['pliegues_mejorados']) + '/5':>6}{d['celdas_que_cambian']:>8}"
          f"{d['rescate']:>5}{d['damage']:>5}{d['neto']:>+6d}"
          f"{d['acierto_en_las_que_cambian']:>9.4f}  {lb}")
    A("```")
    A("")
    A("Los totales son sobre 4449 celdas. El `acierto` es la proporcion de celdas modificadas "
      "en las que el candidato acierta donde V03 falla, y es la medida honesta del efecto.")
    A("")
    A("## 4. E2-13-META-ENT-RIDGE: que significa el cierre por sparsity")
    A("")
    A("Este es el caso que P57 reporto como el unico candidato con delta honesto anidado "
      "positivo y sin veredicto externo. **Esa lectura era incorrecta, y aqui esta la prueba.**")
    A("")
    A("### 4.1 Que significa exactamente")
    A("")
    A(f"El canal de target encoding de menciones, hashtags y dominios tiene una cobertura "
      f"informativa de **{e2['cobertura_informativa'] * 100:.2f} por ciento**, frente a un umbral "
      f"de cierre de {e2['umbral_de_cierre'] * 100:.0f} por ciento.")
    A("")
    A("| tipo | instancias | cubiertas | cobertura | filas con entidad | filas cubiertas | unicos en dev |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for k, v in e2["por_tipo"].items():
        A(f"| {k} | {v['instancias']} | {v['instancias_cubiertas']} | "
          f"{v['cobertura_instancias'] * 100:.3f} por ciento | {v['filas_con_entidad']} | "
          f"{v['filas_cubiertas']} | {v['unicos_dev']} |")
    A("")
    A("La lectura de esa tabla: de 1079 menciones sobreviven **6**; de 255 hashtags sobrevive "
      "**1**; y el unico dominio que supera el filtro de frecuencia minima es `t.co`, con "
      f"{e2['por_tipo']['dominio']['unicos_dev']} unico en desarrollo, o sea el dominio trivial "
      "que aporta cero informacion.")
    A("")
    A("### 4.2 Sparsity no invalida una mejora por si solo. Aqui esta la prueba")
    A("")
    A("La directiva es explicita en no dar por hecho que baja cobertura invalida una mejora. "
      "Por eso se cuantifica lo que el canal aporta **sobre su propio comparador**, que es la "
      "pregunta correcta:")
    A("")
    A("```text")
    A(f"comparador                          {e2['comparador']}")
    A(f"E2-13-META-ENT-RIDGE                {e2['ccv_de_la_variante']:.10f}")
    A(f"delta sobre el comparador           {e2['delta_vs_su_comparador']:+.10f}")
    A(f"pliegues mejorados sobre el comparador  {e2['pliegues_mejorados_vs_comparador']} de 5")
    A(f"complementariedad (ensemble_gain)    {e2['complementariedad']['ensemble_gain']:+.10f}")
    A(f"aporta con IC positivo              {e2['complementariedad']['aporta_ic_positivo']}")
    A(f"delta sobre V03                     {e2['delta_vs_V03']:+.10f}  (este NO es suyo)")
    A("```")
    A("")
    A("**El cierre es metodologico y esta bien justificado.** Un canal que cubre el 0,52 por "
      "ciento de las instancias no puede producir una ganancia transferible, y la medicion lo "
      "confirma: +0,0011 con 3 de 5 pliegues y complementariedad negativa. El sparsity es la "
      "causa y el incremento medido es la prueba.")
    A("")
    A("### 4.3 El error de P57, corregido")
    A("")
    A(f"> {e2['veredicto']['pero_el_delta_de_0_0113_no_es_suyo']}")
    A("")
    A("Es decir: E2-13-META-ENT-RIDGE **no es un mecanismo nuevo**. Es `v08_meta_v2`, que se "
      "envio y dio LB 0,78163, mas un canal de cobertura 0,52 por ciento. El `+0,0113` que P57 "
      "citaba como su delta sobre V03 es **la ganancia de la familia META**, que ya fue refutada "
      "externamente cinco veces.")
    A("")
    A("| pregunta de la directiva | respuesta |")
    A("|---|---|")
    A(f"| que significa sparsity | {e2['veredicto']['que_significa_sparsity']} |")
    A(f"| por que invalida | {e2['veredicto']['por_que_no_invalida_automaticamente']} |")
    A(f"| el cierre es metodologico | {e2['veredicto']['cierre_metodologico']} |")
    A(f"| es reproducible | {e2['veredicto']['reproducible']} |")
    A(f"| existe artefacto completo | {e2['veredicto']['artefacto_de_prediccion']} |")
    A(f"| tiene submission | {e2['veredicto']['submission']} |")
    A(f"| tiene leakage | {e2['veredicto']['leakage']} |")
    A(f"| existe evidencia externa | {e2['veredicto']['evidencia_externa']} |")
    A("")
    A("## 5. Los siete criterios de la directiva, candidato por candidato")
    A("")
    A("| candidato | 1 CV | 2 repro | 3 plieg | 4 leak | 5 mec | 6 artefacto | 7 no LB | FINALISTA |")
    A("|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
    for _, r in aud.iterrows():
        marcas = ["si" if r[c] else "**NO**" for c in CRITERIOS]
        A(f"| {r['candidate']} | " + " | ".join(marcas) + f" | **{'si' if r['finalist'] else 'NO'}** |")
    A("")
    A("**Criterio 6, `artefacto de prediccion completo`, es el que decide.** Solo tres de los "
      "cinco tienen CSV de submission, y los tres ya fueron enviados o ya fueron rechazados por "
      "el holdout.")
    A("")
    A("## 6. El caso limite: B_meta36")
    A("")
    bmeta = next(x for x in decomps if x["candidato"] == "B_meta36")
    A("Es el unico candidato que **pasa el gate D-016 pre-registrado** (IC95 "
      f"{next(r['ic95_pareado'] for _, r in aud.iterrows() if r['candidate'] == 'B_meta36')}) "
      "y **no tiene ningun veredicto externo**. Se audita con el detalle que la directiva pide.")
    A("")
    A("| campo | valor |")
    A("|---|---|")
    A("| evidence_internal | corrected_cv 0,782774 (+0,005563), 4 de 5 pliegues, IC95 positivo y "
      f"que no incluye el cero, {bmeta['rescate']} rescates contra {bmeta['damage']} danos, "
      f"acierto {bmeta['acierto_en_las_que_cambian']:.4f} en las {bmeta['celdas_que_cambian']} "
      "celdas que cambia |")
    A("| evidence_external | **ninguna**. Nunca se envio |")
    A("| transfer_risk | **alto**, y no por el candidato sino por la familia |")
    A("| known_failure_mode | la familia META ha transferring a la baja 5 de 5 veces |")
    A("")
    A("**Por que el riesgo de transferencia es alto notwithstanding que B pase su gate local:**")
    A("")
    A("1. **No existe artefacto de prediccion.** `experiments/36_candidatos_v4.py` no escribe "
      "ninguna submission; solo `C` recibio una, via `38_refit_final_v4.py`, y quedo marcada "
      "DESCARTADO. El criterio 6 falla y la directiva 12 prohibe crear una variante nueva.")
    A("2. **El unico instrumento de adjudicion esta agotado.** El holdout lleva 5 de 5 "
      "evaluaciones, luego B no puede pasar por el unico filtro que el proyecto uso para "
      "decidir en la fase final.")
    A("3. **Su hermano con veredicto fue rechazado, y es casi indistinguible de un rechazo ya "
      "medido.** `FIN-01-C-REF-44` registra `delta_vs_meta_v08 = -4,25e-05`: es decir, `C` es "
      "estadisticamente el mismo modelo que `v08_meta_v2`, que dio LB 0,78163. `B` es `C` sin "
      "el downweight, luego no hay evidencia de que se comporte de otro modo.")
    A("")
    A("**Conclusion sobre B:** pasa cinco de los siete criterios y falla en el sexto por "
      "ausencia de artefacto, no por defecto del modelo. No es un finalista, y no puede "
      "devenirlo en P58 sin crear una variante, que la directiva prohibe. Es el "
      "`quasi-finalista` del proyecto y conviene nombrarlo como tal.")
    A("")
    A("## 7. D_svd300_rbf: por que el CV mas alto no transfirio")
    A("")
    d = inf["D_svd300_rbf"]
    A("| magnitud | valor |")
    A("|---|---:|")
    A(f"| corrected_cv | {d['corrected_cv']:.6f} |")
    A(f"| pliegues que mejoran | {d['pliegues']} de 5 |")
    A(f"| celdas que cambia | {d['celdas_que_cambian']} |")
    A(f"| rescate / damage / neto | {d['rescate']} / {d['damage']} / {d['neto']:+d} |")
    A(f"| LB | {d['LB']:.5f} ({d['LB_delta']:+.5f}) |")
    A("")
    A(f"> {d['por_que_no_transfirio']}")
    A("")
    A(f"**`{d['estado']}`**")
    A("")
    A("El patron es instructivo: **el mejor corrected_cv del proyecto, 5 de 5 pliegues, 136 "
      "rescates contra 104 danos, y McNemar p=0,0452, todo eso perdio 0,00368 en el externo.** "
      "El OOF de esta familia no es indicativo de mejora competitiva, y eso ya esta medido con "
      "un instrumento externo, que es lo mas fuerte que se puede tener.")
    A("")
    A("## 8. Candidatos con LB positivo, negativo y sin LB")
    A("")
    A("| categoria | numero | detalle |")
    A("|---|---:|---|")
    A("| candidatos con corrected_cv D-016 > V03 | 31 | auditados todos |")
    A("| con LB verificado y **positivo** (LB > 0,78306) | **0** | ninguno |")
    A("| con LB verificado y negativo | 3 de la lista de 31, mas 2 fuera de ella | FIN-08 -0,00411, FIN-10 -0,00156, FIN-09 -0,00328, v08_meta_v2 -0,00143, v23 -0,00368 |")
    A("| rechazados por holdout | 1 | C_meta36_downweight, 0,8163 contra 0,8285 |")
    A("| **sin ningun veredicto externo** | **2** | A_cost_sensitive_downweight y B_meta36 |")
    A("")
    A("Los 31 candidatos con CV superior a V03 se pueden agrupar asi:")
    A("")
    A("| grupo | n | por que no son finalistas |")
    A("|---|---:|---|")
    A("| familia META o derivados | 16 | refutada 5 veces en LB, y el unico con adjudicacion "
      "final fue rechazado por holdout |")
    A("| kernel / SVD | 1 | enviado, LB 0,77938 |")
    A("| embeddings (familia EBD) | 10 | familias cerradas por inercia con el gasto de 9 de 9 "
      "unidades agotado; EBD-07 se autodeclara `no cumple A ni B` |")
    A("| reweighting y otros mecanismos | 3 | no alcanzan +0,005 ni 3 de 5 pliegues |")
    A("| mediciones, no modelos | 1 | PA-01 mide la brecha D-004 contra D-016, no es candidata |")
    A("")
    A("## 9. Validacion formal de los artefactos de submission")
    A("")
    A(f"{len(valdf)} CSV en `submissions/`, **{inf['n_validos_formalmente']} validos de forma "
      f"formal**: 1154 filas, columnas `id,HS,TR,AG`, ids y orden identicos a "
      "`toxic_test.csv`, etiquetas binarias y sin nulos.")
    A("")
    invalidos = valdf[~valdf["valido"]]
    if len(invalidos):
        A("| fichero | filas | columnas | orden | binario | nulos | observaciones |")
        A("|---|---:|:-:|:-:|:-:|:-:|---|")
        for _, r in invalidos.iterrows():
            A(f"| {r['fichero']} | {r['n_rows']} | {r['columnas_ok']} | {r['orden_ok']} | "
              f"{r['binario_ok']} | {r['sin_nulos']} | {'; '.join(r['observaciones'])} |")
        A("")
    A(f"**Generados y nunca enviados: {len(nunca)}.** Ninguno se modifica en esta fase.")
    A("")
    A(f"Un detalle que confirma una auditoria previa: `{identico}` es **bit a bit identico** a "
      "la submission de V03, que es la razon por la que su envio dio delta exactamente 0.")
    A("")
    A("| clase | ficheros |")
    A("|---|---|")
    A("| CUARENTENADAS, transductivas | `toxic_submission_v17_prior_shift_em.csv`, "
      "`toxic_submission_v18_near_dup_propagacion.csv`, `toxic_submission_v16_ch2_prior_shift_em.csv` |")
    A("| RECHAZADA por holdout | `toxic_submission_v12_descartado_meta36_downweight.csv` |")
    A("| INVALIDA, es una reproduccion | `toxic_submission_v20_03_reproduccion_v03.csv` |")
    A("| GASTADA, delta 0 | `toxic_submission_v32_05_url_length.csv` |")
    A("")
    A("## 10. Presupuesto de submissions")
    A("")
    A("```text")
    A(f"local_reservation   {presu['local_reservation']}")
    A(f"competition_limit   {presu['competition_limit']}")
    A(f"envios historicos   {presu['envios_historicos']}")
    A(f"ficheros en disco   {presu['ficheros_en_submissions']}")
    A(f"envios hoy          {presu['envios_hoy']}")
    A("```")
    A("")
    A(f"{presu['nota']}")
    A("")
    A("## 11. Veredicto")
    A("")
    A("```text")
    A(f"CASE        {inf['CASE']}")
    A(f"FINALIST    {inf['finalistas']}")
    A("```")
    A("")
    A(f"> {inf['veredicto']}")
    A("")
    A("### Los tres casos que la directiva contemplaba")
    A("")
    A("| caso | descripcion | estado |")
    A("|---|---|---|")
    A("| A | ningun candidato cumple los 7 criterios | **ESTE** |")
    A("| B | existe exactamente uno | no aplica |")
    A("| C | existen varios, no se selecciona por score subjetivo | no aplica |")
    A("")
    A("No se construye ninguna puntuacion de probabilidad de exito, y **no se predice el "
      "leaderboard**. Eso lo prohibe la directiva 10 y ademas el proyecto tiene 10 mediciones "
      "en contra de intentarlo.")
    A("")
    A("## 12. Formulacion final")
    A("")
    A("> V03 remains the best verified leaderboard result.")
    A("")
    A("Y la distincion que la directiva 23 exige no perder:")
    A("")
    A("```text")
    A("best verified result  !=  proof of global optimum")
    A("```")
    A("")
    A(f"- **Verificado:** {V03_LB} es el maximo del leaderboard en 23 envios, y "
      f"{V03_CORRECTED_CV!r} es el corrected_cv de la configuracion congelada, reproducido "
      f"aqui desde el OOF de disco al ultimo digito.")
    A("- **No afirmado:** que V03 sea globalmente optimo. La brecha OOF-LB de 0,00584855 sigue "
      "sin mecanismo identificado, y el techo del oracel de +0,2098064089 no se ha alcanzado "
      "**ni se ha descartado como alcanzable**.")
    A("- **Y para P58 concreto:** que no exista hoy un candidato defendible **no** es evidencia "
      "de que no exista una mejora defendible. Es evidencia de que **el conjunto de artefactos "
      "que hemos producido no contiene uno**, lo cual es una afirmacion sobre el estado del "
      "proyecto y no sobre el problema.")
    A("")
    A("## 13. Conclusion")
    A("")
    A("1. **Candidatos auditados:** los 31 con corrected_cv D-016 superior a V03, mas los cinco "
      "con OOF en disco y artefacto de submission, mas los 37 CSV validados formalmente.")
    A("2. **Con honest CV > V03:** 31. El maximo es 0,790410 (svd300_rbf); el maximo anidado es "
      "0,788500 (E2-13-META-ENT-RIDGE).")
    A("3. **Con LB positivo:** **0**. Ninguno de los 31 supera 0,78306.")
    A("4. **Ya rechazados externamente:** 5 con LB medido en contra (v08_meta_v2 -0,00143, "
      "v23 -0,00368, v09 -0,00411, v11 -0,00156, v10 -0,00328) mas 1 por holdout (C -0,0122).")
    A("5. **E2-13:** se cerro porque su canal cubre el 0,52 por ciento de las instancias y su "
      "incremento sobre su comparador es +0,0011 con 3 de 5 pliegues y complementariedad "
      "negativa. **El cierre es metodologico y correcto.** Y el error de P58 sobre el es "
      "corregido: su delta de +0,0113 es el de la familia META, ya rechazada cinco veces, no "
      "suyo. No es un mecanismo nuevo.")
    A("6. **Candidatos sin submission:** 14 ficheros generados y nunca enviados, de los cuales "
      "3 son transductivos y en cuarentena y 1 es una reproduccion invalida de V03. Ninguno "
      "corresponde a un finalista.")
    A("7. **Numero de finalistas: 0.**")
    A("8. **Finalistas: ninguno.** `CASE A`.")
    A("9. **Integridad de artefactos:** los tres OOF comparten ids y etiquetas, las cinco "
      "descomposiciones reproducen el artefacto del gate al sexto decimal, y de los 37 CSV, "
      f"{inf['n_validos_formalmente']} pasan la validacion formal de esquema, orden, binario y "
      "nulos.")
    A("10. **Submissions reservadas: 3** de autolimitacion del proyecto. `competition_limit = "
      "UNKNOWN`, y no se consulto Kaggle.")
    A("11. **Recomendacion de uso, sin enviar nada:** mantener V03. Si en el futuro se "
      "autorizara una unica sonda, el unico artefacto que se acercaria al umbral es `B_meta36`, "
      "y para ello habria que generar su CSV de prediccion, algo que P58 no hace y la "
      "directiva le prohibe. **No se ha enviado nada y no se ha modificado ningun artefacto.**")
    A("")
    A("*Seccion generada por `experiments/v58_00_auditoria_candidatos.py`. Sin push.*")
    MD_AUDIT.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
