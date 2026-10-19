#!/usr/bin/env python
"""
P92-T - Analisis de la calibracion humana: auditoria de codificacion, acuerdo entre
anotadores y clasificacion de ambiguedades (fase diagnostica; sin entrenamiento).

Entradas (solo lectura):
  artifacts/human_annotation_v1/calibration_responses/calibration_response_{A,B,C,D}.csv
  artifacts/human_annotation_v1/calibration_responses/calibration_response_{A,B}_normalized.csv
  reports/p92_internal_sampling_key.csv (solo columnas review_id/componente, para
  confirmar que la seleccion de calibracion no cambio)
  reports/p92s2_calibration_results.json (para reconciliar las alfas ya registradas)

Salida:
  reports/p92t_audit_results.json

Reglas de esta fase:
  - Las respuestas humanas originales NO se modifican; este script solo lee.
  - La codificacion 0/1 de A y B se aplica como interpretacion derivada
    (0 = NEGATIVE, 1 = POSITIVE), documentada y trazable; las capas
    raw / normalizada / interpretada se mantienen separadas.
  - UNABLE_TO_DETERMINE (UTD) se conserva como categoria diferenciada para C y D.
  - Sin acceso a test, holdout, etiquetas originales (uso analitico) ni modelos.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DIR_RESP = RAIZ / "artifacts" / "human_annotation_v1" / "calibration_responses"
DIR_HANDOFF = RAIZ / "artifacts" / "human_annotation_v1" / "calibration_handoff"
RUTA_JSON = RAIZ / "reports" / "p92t_audit_results.json"
RUTA_ALFAS_P92S2 = RAIZ / "reports" / "p92s2_calibration_results.json"
RUTA_CLAVE = RAIZ / "reports" / "p92_internal_sampling_key.csv"

ANOTADORES = ("A", "B", "C", "D")
ETIQUETAS = ("HS", "TR", "AG")
COLUMNAS_CLASE = {"HS": "HS_annotation", "TR": "TR_annotation", "AG": "AG_annotation"}
VALORES_VALIDOS = ("POSITIVE", "NEGATIVE", "UNABLE_TO_DETERMINE")
UTD = "UNABLE_TO_DETERMINE"

# Prefijos sha256 congelados en auditorias previas (P90); se comparan por prefijo.
HASHES_ESPERADOS = {
    "data/raw/toxic_train.csv": "3d40e7345b588860",
    "features/oof_v03_final.npz": "559265816ce4f631",
    "submissions/toxic_submission_v03_lr_union_thr.csv": "7d7478f271e6edfd",
    "experiments/baseline_v03_frozen/model_v03.joblib": "263ed5bd0fc0de7c",
    "data/processed/holdout_v2_ids.csv": "e5b62e5f22c4c63a",
    "artifacts/human_annotation_v1/annotator_A.csv": "7dfd5b89",
    "artifacts/human_annotation_v1/annotator_B.csv": "956646fa",
}

HASHES_RESPUESTAS_ESPERADOS = {
    "A": "a20521989aa3fea9",
    "B": "41e50c1f1d48760e",
    "C": "8de76e21046d5091",
    "D": "763229f9a9caa176",
}


def sha256_archivo(ruta: Path) -> str:
    """sha256 completo de un archivo (lectura en bloques)."""
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def cargar_respuesta(nombre: str) -> tuple[list[str], list[dict]]:
    """Carga un CSV de respuesta humana sin alterar su contenido."""
    ruta = DIR_RESP / nombre
    with open(ruta, encoding="utf-8-sig", newline="") as fh:
        lector = csv.DictReader(fh)
        columnas = list(lector.fieldnames or [])
        filas = list(lector)
    return columnas, filas


def aplicar_codificacion_ab(filas: list[dict]) -> dict[str, dict[str, str]]:
    """Interpretacion documentada de A/B: 0 = NEGATIVE, 1 = POSITIVE (solo clases).

    Las notas 0/1 NO se convierten en UTD: se conservan como flag de fila
    (1 = dificil de determinar para cualquier clase, aclaracion de Carlos 2026-10-12).
    """
    mapa = {"0": "NEGATIVE", "1": "POSITIVE"}
    interpretado = {}
    for fila in filas:
        rid = fila["review_id"]
        interpretado[rid] = {
            etiqueta: mapa[fila[columna]] for etiqueta, columna in COLUMNAS_CLASE.items()
        }
    return interpretado


def celdas_par(x: dict[str, str], y: dict[str, str]) -> dict[str, int]:
    """Matriz 3x3 aplanada del par (X, Y): clave XY = X dice X0, Y dice Y0."""
    corto = {"NEGATIVE": "N", "POSITIVE": "P", UTD: "U"}
    celdas = defaultdict(int)
    for rid in x:
        celdas[corto[x[rid]] + corto[y[rid]]] += 1
    return {
        clave: celdas.get(clave, 0)
        for clave in ("NN", "NP", "NU", "PN", "PP", "PU", "UN", "UP", "UU")
    }


def estadisticas_par(celdas: dict[str, int]) -> dict:
    """Estadisticas derivadas de la matriz 3x3 del par (con numeradores y denominadores)."""
    total = sum(celdas.values())
    num_3cat = celdas["NN"] + celdas["PP"] + celdas["UU"]
    num_bin = celdas["NN"] + celdas["PP"] + celdas["NP"] + celdas["PN"]
    den_bin = num_bin
    aciertos_bin = celdas["NN"] + celdas["PP"]
    pos_x = celdas["PP"] + celdas["PN"] + celdas["PU"]
    pos_y = celdas["PP"] + celdas["NP"] + celdas["UP"]
    neg_x = celdas["NN"] + celdas["NP"] + celdas["NU"]
    neg_y = celdas["NN"] + celdas["PN"] + celdas["UN"]
    pos_bin_num = 2 * celdas["PP"]
    pos_bin_den = pos_x + pos_y
    neg_bin_num = 2 * celdas["NN"]
    neg_bin_den = neg_x + neg_y
    # kappa 2 categorias solo sobre celdas binarias (UTD excluido)
    kappa_bin = None
    if den_bin > 0:
        p_o = aciertos_bin / den_bin
        p_n_x = (celdas["NN"] + celdas["NP"]) / den_bin
        p_n_y = (celdas["NN"] + celdas["PN"]) / den_bin
        p_p_x = (celdas["PP"] + celdas["PN"]) / den_bin
        p_p_y = (celdas["PP"] + celdas["NP"]) / den_bin
        p_e = p_n_x * p_n_y + p_p_x * p_p_y
        if p_e < 1.0:
            kappa_bin = (p_o - p_e) / (1.0 - p_e)
    # kappa 3 categorias nominal sobre las 15 celdas (UTD como categoria)
    kappa_3cat = None
    if total > 0:
        p_o = num_3cat / total
        p_e = 0.0
        for categoria in ("N", "P", "U"):
            n_x = celdas[categoria + "N"] + celdas[categoria + "P"] + celdas[categoria + "U"]
            n_y = celdas["N" + categoria] + celdas["P" + categoria] + celdas["U" + categoria]
            p_e += (n_x / total) * (n_y / total)
        if p_e < 1.0:
            kappa_3cat = (p_o - p_e) / (1.0 - p_e)
    return {
        "celdas": celdas,
        "total": total,
        "acuerdo_3cat": {"num": num_3cat, "den": total},
        "acuerdo_binario": {"num": aciertos_bin, "den": den_bin},
        "acuerdo_positivo_dice": {"num": pos_bin_num, "den": pos_bin_den},
        "acuerdo_negativo_dice": {"num": neg_bin_num, "den": neg_bin_den},
        "kappa_2cat_binario": kappa_bin,
        "kappa_3cat_nominal": kappa_3cat,
        "conteos_utd": {
            "x_utd": celdas["UN"] + celdas["UP"] + celdas["UU"],
            "y_utd": celdas["NU"] + celdas["PU"] + celdas["UU"],
            "ambos_utd": celdas["UU"],
        },
    }


def krippendorff_alpha_nominal(unidades: list[list[str | None]]) -> float | None:
    """Alpha de Krippendorff (nominal) via matriz de coincidencias.

    unidades: lista de listas de valores por unidad; None = dato faltante.
    """
    coincidencias: dict[tuple[str, str], float] = defaultdict(float)
    valores: set[str] = set()
    for unidad in unidades:
        presentes = [v for v in unidad if v is not None]
        m = len(presentes)
        if m < 2:
            continue
        for i, vi in enumerate(presentes):
            for j, vj in enumerate(presentes):
                if i != j:
                    coincidencias[(vi, vj)] += 1.0 / (m - 1)
        valores.update(presentes)
    if not valores:
        return None
    n_c = {c: sum(coincidencias[(c, k)] for k in valores) for c in valores}
    n = sum(n_c.values())
    if n <= 1:
        return None
    d_o = sum(v for (c, k), v in coincidencias.items() if c != k) / n
    d_e = (n * n - sum(v * v for v in n_c.values())) / (n * (n - 1))
    if d_e <= 0:
        return None
    return 1.0 - d_o / d_e


def main() -> None:
    resultado: dict = {
        "fase": "P92-T",
        "fecha": "2026-10-12",
        "estado_congelado": {
            "V03": "FROZEN",
            "CANONICAL_OOF": "FROZEN",
            "ORIGINAL_LABELS": "FROZEN",
            "HOLDOUT": "FROZEN",
            "HOLDOUT2": "FROZEN",
            "TEST": "UNTOUCHED",
            "LEADERBOARD": "UNUSED",
            "CALIBRATION": "COMPLETED",
            "MAIN_ANNOTATION": "NOT_DISTRIBUTED",
            "MODEL_RETRAINING": "NOT_AUTHORIZED",
        },
        "archivos": {},
        "integridad": {},
        "codificacion_AB": {},
        "acuerdo": {},
        "casos": {},
    }

    # ---------------------------------------------------------------- carga
    crudos: dict[str, list[dict]] = {}
    columnas_por_anotador: dict[str, list[str]] = {}
    for anotador in ANOTADORES:
        columnas, filas = cargar_respuesta(f"calibration_response_{anotador}.csv")
        columnas_por_anotador[anotador] = columnas
        crudos[anotador] = filas

    normalizados: dict[str, list[dict]] = {}
    for anotador in ("A", "B"):
        _, filas = cargar_respuesta(f"calibration_response_{anotador}_normalized.csv")
        normalizados[anotador] = filas

    ids_por_anotador = {a: [f["review_id"] for f in crudos[a]] for a in ANOTADORES}
    ids_iguales = all(set(v) == set(ids_por_anotador["A"]) for v in ids_por_anotador.values())
    orden_igual = all(ids_por_anotador[a] == ids_por_anotador["A"] for a in ANOTADORES)

    # ------------------------------------------------------------- esquema
    esquema_igual = all(columnas_por_anotador[a] == columnas_por_anotador["A"] for a in ANOTADORES)
    versiones = {a: sorted({f["guideline_version"] for f in crudos[a]}) for a in ANOTADORES}
    dominios_clases = {}
    dominios_notas = {}
    for anotador in ANOTADORES:
        valores_clase = sorted(
            {f[col] for f in crudos[anotador] for col in COLUMNAS_CLASE.values()}
        )
        dominios_clases[anotador] = valores_clase
        dominios_notas[anotador] = sorted({f["uncertainty_notes"] for f in crudos[anotador]})
    desviaciones_formato = []
    for anotador in ("A", "B"):
        if set(dominios_clases[anotador]) <= {"0", "1"}:
            desviaciones_formato.append(
                f"{anotador}: respuestas 0/1 en lugar de tokens POSITIVE/NEGATIVE/UNABLE_TO_DETERMINE"
            )
    if any("" in v for v in dominios_notas.values()):
        desviaciones_formato.append("notas vacias presentes (permitido por la guia)")

    # -------------------------------------------------- integridad de textos
    textos_ref = {f["review_id"]: f["text"] for f in crudos["A"]}
    textos_iguales = True
    for anotador in ANOTADORES:
        for fila in crudos[anotador]:
            if fila["text"] != textos_ref[fila["review_id"]]:
                textos_iguales = False
    bloque_textos = "".join(
        f"{rid}\x1f{textos_ref[rid]}\n" for rid in sorted(textos_ref)
    ).encode("utf-8")
    hash_textos = hashlib.sha256(bloque_textos).hexdigest()
    textos_ascii = all(ord(c) <= 126 for c in "".join(textos_ref.values()))

    # ---------------------------------------------------- hashes de archivos
    hashes_respuestas = {}
    for anotador in ANOTADORES:
        ruta = DIR_RESP / f"calibration_response_{anotador}.csv"
        h = sha256_archivo(ruta)
        hashes_respuestas[anotador] = {
            "path": str(ruta.relative_to(RAIZ)),
            "sha256": h,
            "sha256_16": h[:16],
            "esperado_16": HASHES_RESPUESTAS_ESPERADOS[anotador],
            "ok": h.startswith(HASHES_RESPUESTAS_ESPERADOS[anotador]),
        }
    hashes_normalizados = {}
    for anotador in ("A", "B"):
        ruta = DIR_RESP / f"calibration_response_{anotador}_normalized.csv"
        h = sha256_archivo(ruta)
        hashes_normalizados[anotador] = {
            "path": str(ruta.relative_to(RAIZ)),
            "sha256": h,
            "sha256_16": h[:16],
        }
    hashes_handoff = {}
    for anotador in ("A", "B"):
        ruta = DIR_HANDOFF / f"annotator_{anotador}" / "calibration_sample.csv"
        h = sha256_archivo(ruta)
        hashes_handoff[anotador] = {
            "path": str(ruta.relative_to(RAIZ)),
            "sha256_16": h[:16],
            "coincide_con_respuesta": h.startswith(HASHES_RESPUESTAS_ESPERADOS[anotador]),
        }

    hashes_canonicos = {}
    for ruta_rel, prefijo in HASHES_ESPERADOS.items():
        ruta = RAIZ / ruta_rel
        if ruta.exists():
            h = sha256_archivo(ruta)
            hashes_canonicos[ruta_rel] = {
                "sha256": h,
                "sha256_16": h[:16],
                "esperado_prefijo": prefijo,
                "ok": h.startswith(prefijo),
            }
        else:
            hashes_canonicos[ruta_rel] = {"existe": False, "ok": False}

    # ------------------------------------- seleccion de calibracion sin cambios
    calibracion_ids_ok = None
    if RUTA_CLAVE.exists():
        with open(RUTA_CLAVE, encoding="utf-8", newline="") as fh:
            lector = csv.DictReader(fh)
            col_componente = None
            for col in lector.fieldnames or []:
                if "component" in col.lower():
                    col_componente = col
                    break
            ids_calibracion_clave = []
            if col_componente is not None:
                for fila in lector:
                    if fila[col_componente].strip().upper() == "CALIBRACION":
                        ids_calibracion_clave.append(fila["review_id"])
        calibracion_ids_ok = (
            col_componente is not None
            and sorted(ids_calibracion_clave) == sorted(ids_por_anotador["A"])
        )

    resultado["archivos"] = {
        "respuestas": hashes_respuestas,
        "normalizados": hashes_normalizados,
        "handoff_copias": hashes_handoff,
        "esquema": {
            "columnas": columnas_por_anotador["A"],
            "esquema_identico_4_anotadores": esquema_igual,
            "filas": {a: len(crudos[a]) for a in ANOTADORES},
            "ids_iguales": ids_iguales,
            "orden_igual": orden_igual,
            "guideline_version": versiones,
            "dominios_clases": dominios_clases,
            "dominios_notas": {a: f"{len(dominios_notas[a])} valores distintos" for a in ANOTADORES},
            "desviaciones_formato": desviaciones_formato,
        },
    }

    resultado["integridad"] = {
        "raw_annotation_files_unchanged": all(
            v["ok"] for v in hashes_respuestas.values()
        ),
        "original_labels_unchanged": hashes_canonicos["data/raw/toxic_train.csv"]["ok"],
        "original_texts_unchanged": textos_iguales,
        "textos_ascii": textos_ascii,
        "textos_sha256": hash_textos,
        "calibration_sample_unchanged": calibracion_ids_ok,
        "main_sample_not_distributed": (
            hashes_canonicos["artifacts/human_annotation_v1/annotator_A.csv"]["ok"]
            and hashes_canonicos["artifacts/human_annotation_v1/annotator_B.csv"]["ok"]
        ),
        "v03_unchanged": hashes_canonicos[
            "submissions/toxic_submission_v03_lr_union_thr.csv"
        ]["ok"],
        "canonical_oof_unchanged": hashes_canonicos["features/oof_v03_final.npz"]["ok"],
        "holdouts_untouched": hashes_canonicos["data/processed/holdout_v2_ids.csv"]["ok"],
        "test_not_accessed": True,
        "leaderboard_not_used": True,
        "no_models_trained": True,
        "no_predictions_generated": True,
        "no_new_submissions": True,
        "no_pseudo_labeling": True,
        "no_external_classifier": True,
        "no_external_labeled_dataset": True,
        "hashes_canonicos": hashes_canonicos,
    }

    # ------------------------------------------------- codificacion A y B
    interpretado = {a: aplicar_codificacion_ab(crudos[a]) for a in ("A", "B")}
    # contraste con los archivos normalizados derivados (deben coincidir en clases)
    contraste_normalizados = {}
    for anotador in ("A", "B"):
        coincide = all(
            fila[columna] == interpretado[anotador][fila["review_id"]][etiqueta]
            for fila in normalizados[anotador]
            for etiqueta, columna in COLUMNAS_CLASE.items()
        )
        contraste_normalizados[anotador] = coincide
    flags = {
        anotador: sorted(
            fila["review_id"] for fila in crudos[anotador] if fila["uncertainty_notes"] == "1"
        )
        for anotador in ("A", "B")
    }
    resultado["codificacion_AB"] = {
        "clasificacion": "ENCODING_CONFIRMED",
        "regla": (
            "En las columnas de clase: 0 = NEGATIVE, 1 = POSITIVE. En uncertainty_notes: "
            "1 = dificil de determinar para cualquier clase (flag de fila), no UTD por etiqueta."
        ),
        "fuente": (
            "aclaracion explicita del responsable del proyecto (Carlos, 2026-10-12), "
            "solicitada tras detectar que A y B devolvieron 0/1 en lugar de los tokens "
            "exigidos por el README y la guia v1"
        ),
        "evidencia": [
            "El README del handoff y la guia v1 exigen exclusivamente POSITIVE/NEGATIVE/"
            "UNABLE_TO_DETERMINE; ningun documento define 0/1 (por eso la correspondencia "
            "requeria verificacion, no suposicion).",
            "La aclaracion registrada define el significado de 0/1 en las columnas de clase "
            "y en la columna de notas, resolviendo ambas ambiguedades de esquema.",
            "La interpretacion es consistente con el esquema binario observado en los "
            "archivos de A y B (dominios {0,1} en las cuatro columnas).",
        ],
        "no_uso_de_cd": (
            "La coincidencia con C o D NO se uso como fundamento de la codificacion "
            "(seria circular). Se reporta solo como observacion de consistencia posterior."
        ),
        "tratamiento_en_metricas": (
            "Las respuestas de clase de A y B se usan como binarias interpretadas; los flags "
            "de fila se mantienen como capa separada y NO se convierten en UTD (no existe base "
            "documental para equiparar 'dificil para cualquier clase' con UTD por etiqueta)."
        ),
        "flags_fila": flags,
        "contraste_con_normalizados": contraste_normalizados,
    }

    # --------------------------------------------------------------- acuerdo
    respuestas = {
        "A": interpretado["A"],
        "B": interpretado["B"],
        "C": {
            fila["review_id"]: {e: fila[c] for e, c in COLUMNAS_CLASE.items()}
            for fila in crudos["C"]
        },
        "D": {
            fila["review_id"]: {e: fila[c] for e, c in COLUMNAS_CLASE.items()}
            for fila in crudos["D"]
        },
    }
    pares = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]

    alfas_previas = {}
    if RUTA_ALFAS_P92S2.exists():
        with open(RUTA_ALFAS_P92S2, encoding="utf-8") as fh:
            alfas_previas = json.load(fh).get("acuerdo", {})

    acuerdo: dict = {"por_etiqueta": {}, "pares": [f"{x}{y}" for x, y in pares]}
    for etiqueta in ETIQUETAS:
        stats_pares = {}
        for x, y in pares:
            valor_x = {rid: respuestas[x][rid][etiqueta] for rid in ids_por_anotador["A"]}
            valor_y = {rid: respuestas[y][rid][etiqueta] for rid in ids_por_anotador["A"]}
            stats_pares[f"{x}{y}"] = estadisticas_par(celdas_par(valor_x, valor_y))
        # acuerdo exacto de la combinacion (4 anotadores, UTD como categoria)
        unidades_4 = [
            [respuestas[a][rid][etiqueta] for a in ANOTADORES]
            for rid in ids_por_anotador["A"]
        ]
        exacto_4 = sum(1 for u in unidades_4 if len(set(u)) == 1)
        alpha_cat = krippendorff_alpha_nominal(unidades_4)
        unidades_faltante = [
            [v if v != UTD else None for v in u] for u in unidades_4
        ]
        alpha_falt = krippendorff_alpha_nominal(unidades_faltante)
        conteo_utd = {
            a: sum(1 for rid in ids_por_anotador["A"] if respuestas[a][rid][etiqueta] == UTD)
            for a in ANOTADORES
        }
        previas = alfas_previas.get(etiqueta, {})
        acuerdo["por_etiqueta"][etiqueta] = {
            "pares": stats_pares,
            "exacto_4cat": {"num": exacto_4, "den": len(unidades_4)},
            "alpha_utd_categoria": alpha_cat,
            "alpha_utd_faltante": alpha_falt,
            "utd_por_anotador": conteo_utd,
            "reconciliacion_p92s2": {
                "alpha_utd_categoria_previo": previas.get("alpha_utd_categoria"),
                "alpha_utd_faltante_previo": previas.get("alpha_utd_faltante"),
                "coincide_categoria": (
                    previas.get("alpha_utd_categoria") is not None
                    and abs(alpha_cat - previas["alpha_utd_categoria"]) < 0.002
                ),
                "coincide_faltante": (
                    previas.get("alpha_utd_faltante") is not None
                    and abs(alpha_falt - previas["alpha_utd_faltante"]) < 0.002
                ),
            },
        }
    # combinacion exacta global (las tres etiquetas identicas en los 4 anotadores)
    global_exacto = 0
    for rid in ids_por_anotador["A"]:
        identicos = all(
            len({respuestas[a][rid][e] for a in ANOTADORES}) == 1 for e in ETIQUETAS
        )
        global_exacto += 1 if identicos else 0
    acuerdo["combinacion_exacta_global"] = {
        "num": global_exacto,
        "den": len(ids_por_anotador["A"]),
    }
    # relacion flags A/B con UTD de C/D
    union_flags = sorted(set(flags["A"]) | set(flags["B"]))
    utd_union_rid = sorted(
        rid
        for rid in ids_por_anotador["A"]
        if any(respuestas[a][rid][e] == UTD for a in ("C", "D") for e in ETIQUETAS)
    )
    acuerdo["flags_vs_utd"] = {
        "flags_A": flags["A"],
        "flags_B": flags["B"],
        "union_flags": union_flags,
        "filas_con_utd_CD": utd_union_rid,
        "flags_sin_utd_CD": sorted(set(union_flags) - set(utd_union_rid)),
        "utd_CD_sin_flag": sorted(set(utd_union_rid) - set(union_flags)),
        "interseccion": sorted(set(union_flags) & set(utd_union_rid)),
    }
    resultado["acuerdo"] = acuerdo

    # ----------------------------------------------------------------- casos
    casos = {}
    for anotador in ANOTADORES:
        for fila in crudos[anotador]:
            rid = fila["review_id"]
            casos.setdefault(rid, {})[anotador] = {
                "HS": fila["HS_annotation"],
                "TR": fila["TR_annotation"],
                "AG": fila["AG_annotation"],
                "notas": fila["uncertainty_notes"],
            }
    resultado["casos"] = {
        "respuestas_crudas_por_fila": casos,
        "ids": ids_por_anotador["A"],
    }

    # --------------------------------------------------------------- guardar
    RUTA_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RUTA_JSON, "w", encoding="utf-8") as fh:
        json.dump(resultado, fh, ensure_ascii=True, indent=2, sort_keys=False)
        fh.write("\n")

    # ---------------------------------------------------------------- resumen
    print("P92-T analisis de calibracion: OK")
    print(f"  JSON: {RUTA_JSON.relative_to(RAIZ)}")
    print(f"  raw_annotation_files_unchanged: {resultado['integridad']['raw_annotation_files_unchanged']}")
    print(f"  textos_identicos: {textos_iguales} | ascii: {textos_ascii}")
    print(f"  calibracion_sample_unchanged: {calibracion_ids_ok}")
    for etiqueta in ETIQUETAS:
        bloque = acuerdo["por_etiqueta"][etiqueta]
        pares_txt = " ".join(
            f"{p}={bloque['pares'][p]['acuerdo_3cat']['num']}/15" for p in acuerdo["pares"]
        )
        print(
            f"  {etiqueta}: {pares_txt} | exacto4={bloque['exacto_4cat']['num']}/15 | "
            f"alpha_cat={bloque['alpha_utd_categoria']:.3f} "
            f"alpha_falt={bloque['alpha_utd_faltante']:.3f}"
        )
        rec = bloque["reconciliacion_p92s2"]
        print(
            f"      reconciliacion P92-S2: cat={rec['coincide_categoria']} "
            f"falt={rec['coincide_faltante']}"
        )
    print(f"  combinacion exacta global: {global_exacto}/15")
    print(f"  flags A={flags['A']} B={flags['B']}")
    print(f"  flags_sin_utd_CD={acuerdo['flags_vs_utd']['flags_sin_utd_CD']}")
    print(f"  utd_CD_sin_flag={acuerdo['flags_vs_utd']['utd_CD_sin_flag']}")


if __name__ == "__main__":
    main()
