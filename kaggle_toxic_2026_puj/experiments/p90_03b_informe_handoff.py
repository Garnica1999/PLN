"""P90-03 parte 2: informe de diagnostico, gap analysis, manifiesto y handoff.

Lee `reports/p90_03_audit_results.json` y las tablas `reports/p90_03_*.csv` y produce:

- `reports/p90_03_feature_contribution_diagnostic.md` (informe, 23 secciones);
- `reports/p90_03_hypothesis_gap_analysis.csv` (hipotesis contra historial);
- `reports/p90_03_hypothesis_candidates.csv` (candidatos futuros; negativo explicito);
- `reports/p90_03_key_feature_associations.csv` (asociaciones descriptivas dev);
- `reports/p90_03_artifact_manifest.json` (manifiesto de reproducibilidad);
- `reports/p90_03_chatgpt_handoff.md` (sintesis autocontenida).

Sin entrenamiento, sin test, sin holdout, sin submissions, sin commits.

Uso:
    /home/carlos/venv/bin/python experiments/p90_03b_informe_handoff.py
"""

from __future__ import annotations

import ast
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

RAIZ: Path = Path(__file__).resolve().parents[1]

RUTA_RESULTS = RAIZ / "reports" / "p90_03_audit_results.json"
RUTA_BY_CELL = RAIZ / "reports" / "p90_03_error_contribution_by_cell.csv"
RUTA_SUMMARY = RAIZ / "reports" / "p90_03_feature_contribution_summary.csv"
RUTA_SIGNATURES = RAIZ / "reports" / "p90_03_error_signature_profiles.csv"
RUTA_CONTROL_CMP = RAIZ / "reports" / "p90_03_error_control_comparison.csv"
RUTA_SLICE_SUM = RAIZ / "reports" / "p90_03_slice_contribution_summary.csv"
RUTA_TRAIN = RAIZ / "data" / "raw" / "toxic_train.csv"
RUTA_OOF = RAIZ / "features" / "oof_v03_final.npz"

RUTA_INFORME = RAIZ / "reports" / "p90_03_feature_contribution_diagnostic.md"
RUTA_GAP = RAIZ / "reports" / "p90_03_hypothesis_gap_analysis.csv"
RUTA_CAND = RAIZ / "reports" / "p90_03_hypothesis_candidates.csv"
RUTA_ASSOC = RAIZ / "reports" / "p90_03_key_feature_associations.csv"
RUTA_MANIFEST = RAIZ / "reports" / "p90_03_artifact_manifest.json"
RUTA_HANDOFF = RAIZ / "reports" / "p90_03_chatgpt_handoff.md"

FRASES_CLAVE = ("perra", "zorra", "mi perra", "la puta", "hijo", "escoria", "acoso",
                "rabe", "sudaca", "callate", "eres una")


def leer_csv(ruta: Path, encoding: str = "utf-8") -> list[dict[str, str]]:
    """Lee un CSV respetando saltos de linea embebidos."""
    with ruta.open(encoding=encoding, newline="") as fh:
        return list(csv.DictReader(fh))


def escribir_csv(ruta: Path, filas: list[dict[str, Any]]) -> None:
    """Escribe un CSV con terminador LF."""
    if not filas:
        raise SystemExit(f"STOP: {ruta.name} quedaria vacio")
    with ruta.open("w", encoding="utf-8", newline="") as fh:
        escritor = csv.DictWriter(fh, fieldnames=list(filas[0].keys()), lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)


def escribir_texto(ruta: Path, texto: str) -> None:
    """Escribe texto exigiendo ASCII puro."""
    if any(ord(c) > 127 for c in texto):
        raise SystemExit(f"STOP: bytes > 127 en {ruta.name}")
    ruta.write_text(texto, encoding="utf-8")


def tabla(cabeceras: list[str], filas: list[list[Any]]) -> str:
    """Construye una tabla markdown."""
    lineas = ["| " + " | ".join(cabeceras) + " |",
              "|" + "|".join("---" for _ in cabeceras) + "|"]
    for fila in filas:
        lineas.append("| " + " | ".join(str(c) for c in fila) + " |")
    return "\n".join(lineas)


def asociaciones_descriptivas() -> list[dict[str, Any]]:
    """Asociaciones descriptivas (dev) de frases clave con las etiquetas (solo diagnostico)."""
    oof = np.load(RUTA_OOF, allow_pickle=True)
    ids = [int(i) for i in oof["ids"]]
    y = oof["y_true"].astype(int)
    dev = set(ids)
    textos: dict[int, str] = {}
    with RUTA_TRAIN.open(encoding="utf-8") as fh:
        for f in csv.DictReader(fh, delimiter=";"):
            i = int(str(f["id"]).strip('"'))
            if i in dev:
                textos[i] = f["text"]
    filas = []
    for frase in FRASES_CLAVE:
        idxs = [k for k, i in enumerate(ids) if frase in textos[i].lower()]
        n = len(idxs)
        filas.append({
            "frase": frase, "docs_dev": n,
            "HS_1": sum(1 for k in idxs if y[k, 0] == 1),
            "TR_1": sum(1 for k in idxs if y[k, 1] == 1),
            "AG_1": sum(1 for k in idxs if y[k, 2] == 1),
            "sin_etiqueta_000": sum(1 for k in idxs if y[k].sum() == 0),
            "nota": "descriptivo; sin construir lexico nuevo ni cambiar features",
        })
    return filas


def construir_gap(d: dict[str, Any]) -> list[dict[str, Any]]:
    """Tabla de gap analysis: hipotesis de P90.03 contra el historial experimental."""
    ag = d["agregados"]
    firmas = d["firmas"]
    filas = [
        {
            "hypothesis_id": "FP-H1",
            "hypothesis": "Una expresion vulgar recibe una contribucion demasiado fuerte",
            "error_mechanism": "lexico de misoginia (perra/zorra/callate) con coeficiente alto y sin contrapeso activo",
            "evidence_from_v03": (f"top1 positivo del lexico en FP: HS {ag['HS|FP']['top1_pos_es_lexico']}/173, "
                                  f"TR {ag['TR|FP']['top1_pos_es_lexico']}/82, AG {ag['AG|FP']['top1_pos_es_lexico']}/135; "
                                  "coef final perra 3.23/4.34/3.25 (auxiliar)"),
            "related_historical_experiments": "V30/V4-B03/P69 (noise, reweighting, downweight por violacion); term weighting; HIB-H11",
            "historical_result": "reweighting y downweight descartados; HIB-H11 -0.0213; term weighting cerrado",
            "historical_status": "CLOSED",
            "is_genuinely_new": "NO",
            "reason": "el mecanismo es la familia de reweighting/term-weighting ya ejecutada y cerrada",
            "potential_test": "downweight pre-registrado por familia de features (idea futura; no ejecutar)",
            "governance_status": "PERMITTED pero duplicado",
            "final_decision": "CLOSED",
        },
        {
            "hypothesis_id": "FP-H2",
            "hypothesis": "Las features de insulto no distinguen usos afiliativos",
            "error_mechanism": "marcadores afiliativos (mi perra, la puta) tienen coeficiente NEGATIVO aprendido pero no cubren los FP afiliativos",
            "evidence_from_v03": (f"mi perra como top-negativo en FP: HS 9, TR 8, AG 7; la puta negativo en 20/98 docs; "
                                  f"interpersonal en top10: HS {ag['HS|FP']['interpersonal_en_top10_pos']}/173, "
                                  f"AG {ag['AG|FP']['interpersonal_en_top10_pos']}/135"),
            "related_historical_experiments": "HIB-H11 (bloques bromista/interpersonal); targeting D-018; contextual_selection",
            "historical_result": "HIB-H11 -0.0213 descartada; targeting sin mejora standalone; contextual_selection CLOSED (P0=V03)",
            "historical_status": "CLOSED",
            "is_genuinely_new": "NO",
            "reason": "el modelo ya aprendio la senal contraria; los bloques densos de contexto fueron probados y descartados",
            "potential_test": "gating de contexto por instancia (idea futura; familia cerrada)",
            "governance_status": "PERMITTED pero duplicado",
            "final_decision": "CLOSED",
        },
        {
            "hypothesis_id": "FP-H3",
            "hypothesis": "Las features de char_wb disparan sobre variaciones superficiales ambiguas",
            "error_mechanism": "conflicto word-char y dominancia char en una parte de los FP",
            "evidence_from_v03": (f"conflicto word-char en FP: HS {ag['HS|FP']['conflicto_wc']}/173, TR {ag['TR|FP']['conflicto_wc']}/82, "
                                  f"AG {ag['AG|FP']['conflicto_wc']}/135; char dominante: HS {ag['HS|FP']['char_dominante']}, "
                                  f"TR {ag['TR|FP']['char_dominante']}, AG {ag['AG|FP']['char_dominante']}"),
            "related_historical_experiments": "REP-05/REP-06 (barrido word/char/union); ENS-01..04 (pesos por representacion); Optuna 09 (parametros word_/char_)",
            "historical_result": "union adoptada; pesos por representacion descartados (ENS); busqueda de parametros cerrada",
            "historical_status": "DUPLICATE",
            "is_genuinely_new": "NO",
            "reason": "ponderar o elegir canales es la familia REP/ENS ya ejecutada; el conflicto es una propiedad del modelo union",
            "potential_test": "reponderacion de bloques con protocolo anidado (idea futura; duplicada de ENS)",
            "governance_status": "PERMITTED pero duplicado",
            "final_decision": "DUPLICATE",
        },
        {
            "hypothesis_id": "FP-H4",
            "hypothesis": "Las menciones o nombres propios interactuan con insultos de forma incorrecta",
            "error_mechanism": "interaccion mencion x insulto",
            "evidence_from_v03": (f"menciones en el top10 positivo: HS {ag['HS|FP']['mencion_en_top10_pos']}/173, "
                                  f"TR {ag['TR|FP']['mencion_en_top10_pos']}/82, AG {ag['AG|FP']['mencion_en_top10_pos']}/135; "
                                  "las menciones no figuran entre las contribuciones mas fuertes"),
            "related_historical_experiments": "targeting D-018; P50/P51 selector; P68 enrutado por estructura",
            "historical_result": "sin mejora standalone; canales preservados solo para stacking; selector y enrutado cerrados",
            "historical_status": "CLOSED",
            "is_genuinely_new": "NO",
            "reason": "la evidencia de contribuciones no respalda la hipotesis y la familia esta cerrada",
            "potential_test": "no procede",
            "governance_status": "PERMITTED",
            "final_decision": "CLOSED",
        },
        {
            "hypothesis_id": "FP-H5",
            "hypothesis": "El modelo confunde referencias literales o no dirigidas con agresion",
            "error_mechanism": "lexico positivo sin marcadores de direccion activos",
            "evidence_from_v03": (f"FP sin mencion en el texto: HS {ag['HS|FP']['sin_mencion_texto']}/173, "
                                  f"TR {ag['TR|FP']['sin_mencion_texto']}/82, AG {ag['AG|FP']['sin_mencion_texto']}/135"),
            "related_historical_experiments": "contextual_selection (P0=V03); P19 cita/negacion; P66 negacion (no adoptado)",
            "historical_result": "ninguna politica contextual supero a V03; cita/negacion cerrado",
            "historical_status": "CLOSED",
            "is_genuinely_new": "NO",
            "reason": "la correccion contextual fue la familia contextual_selection, cerrada con P0 como mejor politica",
            "potential_test": "no procede",
            "governance_status": "PERMITTED",
            "final_decision": "CLOSED",
        },
        {
            "hypothesis_id": "FP-H6",
            "hypothesis": "La estructura del mensaje contiene evidencia que word/char no representa suficientemente",
            "error_mechanism": "puntuacion/elongaciones/mayusculas como senal parcial",
            "evidence_from_v03": (f"puntuacion en top10: HS {ag['HS|FP']['puntuacion_en_top10']}, TR {ag['TR|FP']['puntuacion_en_top10']}, "
                                  f"AG {ag['AG|FP']['puntuacion_en_top10']}; elongacion en texto: HS {ag['HS|FP']['elongacion_texto']}"),
            "related_historical_experiments": "structural_features V15-01; HIB-H11/H12/HIB-UNION bloques densos; P68",
            "historical_result": "V15-01 0.775860 no adoptada; bloques densos descartados (-0.02 a -0.03)",
            "historical_status": "CLOSED",
            "is_genuinely_new": "NO",
            "reason": "las features estructurales densas se probaron repetidamente y no superaron a V03",
            "potential_test": "no procede",
            "governance_status": "PERMITTED",
            "final_decision": "CLOSED",
        },
        {
            "hypothesis_id": "FP-H7",
            "hypothesis": "Los errores dependen de citas, discusiones o discurso sobre el propio lenguaje",
            "error_mechanism": "cita/meta-discurso",
            "evidence_from_v03": (f"FP con comillas: HS {ag['HS|FP']['cita_texto']}/173, TR {ag['TR|FP']['cita_texto']}/82, "
                                  f"AG {ag['AG|FP']['cita_texto']}/135; el canal char aprende comillas como positivas en AG (coef aux 2.87)"),
            "related_historical_experiments": "P19 sintaxis cita/negacion; P66 negacion; taxonomia v4 QUOTED (16 errores)",
            "historical_result": "cerrado; rescate de negacion no transferible (P66)",
            "historical_status": "CLOSED",
            "is_genuinely_new": "NO",
            "reason": "minoritario y ya investigado (cita/negacion)",
            "potential_test": "no procede",
            "governance_status": "PERMITTED",
            "final_decision": "CLOSED",
        },
        {
            "hypothesis_id": "FN-H1",
            "hypothesis": "Las features de insulto no se activan en agresion indirecta",
            "error_mechanism": "ausencia de lexico entre las contribuciones positivas; canal char dominante",
            "evidence_from_v03": (f"sin lexico en top10 positivo: HS {ag['HS|FN']['sin_lexico_en_top10_pos']}/143, "
                                  f"TR {ag['TR|FN']['sin_lexico_en_top10_pos']}/82, AG {ag['AG|FN']['sin_lexico_en_top10_pos']}/142; "
                                  f"char dominante: HS {ag['HS|FN']['char_dominante']}, AG {ag['AG|FN']['char_dominante']}"),
            "related_historical_experiments": "HIB-H14 (lexico inducido x colectiva); embeddings; representacion completa",
            "historical_result": "HIB-H14 -0.0297 descartada; familias de representacion cerradas",
            "historical_status": "CLOSED",
            "is_genuinely_new": "NO",
            "reason": "la recuperacion por lexico inducido fue probada y descartada; el espacio de representacion esta cerrado",
            "potential_test": "no procede",
            "governance_status": "PERMITTED",
            "final_decision": "CLOSED",
        },
        {
            "hypothesis_id": "FN-H2",
            "hypothesis": "El modelo no representa suficientemente referencias colectivas",
            "error_mechanism": "colectivos con peso bajo y tokens colectivos con asociacion anti-toxica aprendida",
            "evidence_from_v03": (f"colectivo en top10: HS {ag['HS|FN']['colectivo_en_top10_pos']}/143, AG {ag['AG|FN']['colectivo_en_top10_pos']}/142; "
                                  "rabe: 149 docs dev, 114 sin etiqueta; rabe como top-negativo en FN AG (6 celdas)"),
            "related_historical_experiments": "HIB-H14 (colectiva); joint_labels; colectivos v4; targeting",
            "historical_result": "bloques de colectiva descartados; factorizacion conjunta cerrada (D-045)",
            "historical_status": "CLOSED",
            "is_genuinely_new": "NO",
            "reason": "la senal colectiva ya se probo como bloque denso y como cruce; no supero a V03",
            "potential_test": "no procede",
            "governance_status": "PERMITTED",
            "final_decision": "CLOSED",
        },
        {
            "hypothesis_id": "FN-H3",
            "hypothesis": "Las generalizaciones sobre grupos no producen las combinaciones lexicales aprendidas",
            "error_mechanism": "evidencia positiva debil y acumulativa sin n-gramas hostiles aprendidos",
            "evidence_from_v03": (f"firma acumulacion en FN: HS {firmas['HS|FN|acumulacion de muchas features']}/143, "
                                  f"AG {firmas['AG|FN|acumulacion de muchas features']}/142; 'pocas dominantes' ~0"),
            "related_historical_experiments": "HIB-H14/HIB-UNION; embeddings; kernels; VW; familias de representacion",
            "historical_result": "todas descartadas; el espacio de representacion TF-IDF/embeddings esta cerrado",
            "historical_status": "CLOSED",
            "is_genuinely_new": "NO",
            "reason": "no hay combinacion aprendible demostrable sin reabrir una familia cerrada",
            "potential_test": "no procede",
            "governance_status": "PERMITTED",
            "final_decision": "CLOSED",
        },
        {
            "hypothesis_id": "FN-H4",
            "hypothesis": "La ausencia de mencion explicita debilita la puntuacion de TR",
            "error_mechanism": "TR FN sin destinatario explicito",
            "evidence_from_v03": (f"TR FN sin mencion en texto: {ag['TR|FN']['sin_mencion_texto']}/82; "
                                  "menciones en top10 positivo de TR FN: 0; TR FN con HS correcta: 34/82"),
            "related_historical_experiments": "HIB-H12 (objetivo individual vs colectivo); targeting D-018; P87 (anotacion TR)",
            "historical_result": "HIB-H12 -0.0091 descartada; P87 BLOCKED_BY_RULES (revision humana sin autorizacion)",
            "historical_status": "CLOSED",
            "is_genuinely_new": "NO",
            "reason": "el mecanismo de destinatario se probo como bloque; la revision de anotacion esta bloqueada por reglas",
            "potential_test": "no procede en esta fase",
            "governance_status": "PARCIAL: revision de anotacion BLOCKED_BY_RULES",
            "final_decision": "CLOSED",
        },
        {
            "hypothesis_id": "FN-H5",
            "hypothesis": "La estructura retorica, cita o relato altera las combinaciones de n-gramas",
            "error_mechanism": "registro noticioso/cita con evidencia negativa de palabras funcionales y https",
            "evidence_from_v03": (f"url en texto: HS FN {ag['HS|FN']['url_texto']}/143, AG FN {ag['AG|FN']['url_texto']}/142; "
                                  "'https' entre los top-negativos de FN; cita: HS {ag['HS|FN']['cita_texto']}"),
            "related_historical_experiments": "P19 cita/negacion; retrieval; diagnosticos de registro noticioso (taxonomia v4)",
            "historical_result": "cerrado; sin mecanismo adoptable",
            "historical_status": "CLOSED",
            "is_genuinely_new": "NO",
            "reason": "registro y cita ya caracterizados; sin linea nueva demostrable",
            "potential_test": "no procede",
            "governance_status": "PERMITTED",
            "final_decision": "CLOSED",
        },
        {
            "hypothesis_id": "FN-H6",
            "hypothesis": "La variacion ortografica modifica las features activas",
            "error_mechanism": "formas elongadas/variantes con senal debil",
            "evidence_from_v03": (f"elongacion en texto: HS FN {ag['HS|FN']['elongacion_texto']}/143, AG FN {ag['AG|FN']['elongacion_texto']}/142 "
                                  "(minoritario)"),
            "related_historical_experiments": "augmentation V6-08; fuzzy_matching V29; PRE-00..09",
            "historical_result": "familias cerradas; sin mejora",
            "historical_status": "CLOSED",
            "is_genuinely_new": "NO",
            "reason": "la variacion ortografica fue el objetivo explicito del canal char y de las familias cerradas",
            "potential_test": "no procede",
            "governance_status": "PERMITTED",
            "final_decision": "CLOSED",
        },
        {
            "hypothesis_id": "FN-H7",
            "hypothesis": "El canal word y el canal char aportan evidencia contradictoria",
            "error_mechanism": "conflicto entre canales",
            "evidence_from_v03": (f"conflicto word-char en FN: HS {ag['HS|FN']['conflicto_wc']}/143, AG {ag['AG|FN']['conflicto_wc']}/142, "
                                  f"TR {ag['TR|FN']['conflicto_wc']}/82; mismo signo: TR 82/82"),
            "related_historical_experiments": "ENS-01..04 (pesos por representacion); REP-05/REP-06",
            "historical_result": "pesos por representacion descartados; union adoptada",
            "historical_status": "DUPLICATE",
            "is_genuinely_new": "NO",
            "reason": "resolver conflictos de canal es ponderar representaciones (ENS), ya ejecutado",
            "potential_test": "no procede",
            "governance_status": "PERMITTED pero duplicado",
            "final_decision": "DUPLICATE",
        },
        {
            "hypothesis_id": "FN-H8",
            "hypothesis": "Conflicto de evidencia: varias features negativas dominan sobre evidencia hostil",
            "error_mechanism": "acumulacion con evidencia negativa (stopwords, registro, tokens anti-toxicos) dominante",
            "evidence_from_v03": (f"neg domina (ratio>=0.3): HS {ag['HS|FN']['neg_domina_ratio_0.3']}/143, "
                                  f"TR {ag['TR|FN']['neg_domina_ratio_0.3']}/82, AG {ag['AG|FN']['neg_domina_ratio_0.3']}/142"),
            "related_historical_experiments": "reweighting/noise V30; hardness v4 (BCE cuestionada); P50/P51; calibracion",
            "historical_result": "familias cerradas; ninguna intervencion supero el gate",
            "historical_status": "CLOSED",
            "is_genuinely_new": "NO",
            "reason": "el conflicto de evidencia es la estructura normal del modelo; su correccion mapea a familias cerradas",
            "potential_test": "no procede",
            "governance_status": "PERMITTED",
            "final_decision": "CLOSED",
        },
        {
            "hypothesis_id": "X1",
            "hypothesis": "Los errores de TR estan asociados a la estructura de etiquetas (TR dentro de HS)",
            "error_mechanism": "acoplamiento TR-HS",
            "evidence_from_v03": ("TR FN con HS correcta: 34/82; TR FP con HS error simultaneo: 67/82; "
                                  "identidad TR=>HS: 0 violaciones (1483 filas)"),
            "related_historical_experiments": "joint_labels (E2-01*, label powerset); D-044/D-045 (factorizacion condicional)",
            "historical_result": "familia joint_labels CLOSED; factorizacion cierta pero no accionable",
            "historical_status": "CLOSED",
            "is_genuinely_new": "NO",
            "reason": "la estructura TR-HS ya se exploro con factorizacion y label powerset",
            "potential_test": "no procede",
            "governance_status": "PERMITTED",
            "final_decision": "CLOSED",
        },
        {
            "hypothesis_id": "X2",
            "hypothesis": "Los 466 errores alejados del threshold responden a un mecanismo especifico",
            "error_mechanism": "acumulacion y conflicto (nunca pocas features dominantes)",
            "evidence_from_v03": ("lejos: 466/757; pocas dominantes en lejanos: 0 (HS/AG/TR FP y FN casi todos); "
                                  "conflicto word-char en lejanos: HS FP 63/112, AG FP 27/77"),
            "related_historical_experiments": "threshold_transfer P55/P56; P66/P50/P51 diagnosticos; familias de representacion",
            "historical_result": "thresholds por si solos no bastan (ya documentado); sin mecanismo nuevo",
            "historical_status": "UNRESOLVED",
            "is_genuinely_new": "NO",
            "reason": "no se demuestra un mecanismo distinto de las familias cerradas; queda como limite diagnostico",
            "potential_test": "no procede sin autorizacion de una familia nueva",
            "governance_status": "PERMITTED (solo diagnostico)",
            "final_decision": "UNRESOLVED",
        },
        {
            "hypothesis_id": "X3",
            "hypothesis": "Las regiones de bajo rendimiento (URLs, hashtags, colectivos) comparten una firma propia",
            "error_mechanism": "mas features activas y mezcla de canales distinta en los errores de esas regiones",
            "evidence_from_v03": ("URLs HS: 248.9 vs 185.0 features activas (dentro vs fuera); colectivos HS: 272.7 vs 177.5; "
                                  "dominantes siguen siendo acumulacion"),
            "related_historical_experiments": "P68 enrutado por estructura; contextual_selection; diagnosticos v4",
            "historical_result": "enrutado por estructura CLOSED; sin mejora",
            "historical_status": "CLOSED",
            "is_genuinely_new": "NO",
            "reason": "el enrutado por region fue probado y cerrado; las firmas observadas no constituyen un mecanismo nuevo",
            "potential_test": "no procede",
            "governance_status": "PERMITTED",
            "final_decision": "CLOSED",
        },
    ]
    return filas


def construir_informe(x: dict[str, Any]) -> str:
    """Construye el informe principal (23 secciones)."""
    d = x["d"]
    v = d["validacion"]
    ag = d["agregados"]
    far = d["far"]
    firmas = d["firmas"]
    coef = d["coeficientes"]
    resumen = x["resumen"]
    slices = x["slices"]
    asoc = x["asoc"]

    filas_resumen = [[f["label"], f["error_type"], f["n_errors"], f["near_errors"], f["intermediate_errors"],
                      f["far_errors"], f["mean_word_contribution"], f["mean_char_contribution"],
                      f["mean_abs_word_contribution"], f["mean_abs_char_contribution"],
                      f["share_word_char_same_sign"], f["dominant_feature_families"]] for f in resumen]
    filas_asoc = [[f["frase"], f["docs_dev"], f["HS_1"], f["TR_1"], f["AG_1"], f["sin_etiqueta_000"]] for f in asoc]
    filas_coef = [[et, f"{coef[et]['intercept']:.4f}", f"{coef[et]['share_masa_word']:.4f}",
                   coef[et]["top_pos"][:170], coef[et]["top_neg"][:170]] for et in ("HS", "TR", "AG")]
    filas_firmas = []
    for et in ("HS", "TR", "AG"):
        for tipo in ("FP", "FN"):
            for clave in (f"{et}|{tipo}|conflicto word-char", f"{et}|{tipo}|acumulacion de muchas features",
                          f"{et}|{tipo}|conflicto positivas-negativas", f"{et}|{tipo}|pocas features dominantes",
                          f"{et}|{tipo}|sin evidencia hostil activada", f"{et}|{tipo}|lexico dominante sin contexto",
                          f"{et}|{tipo}|no interpretable con los artefactos disponibles"):
                if clave in firmas:
                    filas_firmas.append([et, tipo, clave.split("|", 2)[2], firmas[clave]])
    filas_far = []
    for et in ("HS", "TR", "AG"):
        for tipo in ("FP", "FN"):
            f = far[f"{et}|{tipo}"]
            filas_far.append([et, tipo, f["n"], f["acumulacion"], f["conflicto_wc"], f["mismo_signo"],
                              f["top_features_pos"][:90]])
    filas_slice = [[f["slice"], f["label"], f["n_errors_inside"], f["mean_word_inside"], f["mean_char_inside"],
                    f["mean_active_inside"], f["mean_active_outside"]] for f in slices
                   if f["slice"] in ("URLs", "hashtags", "colectivos_superficie", "menciones",
                                     "mencion_inicial", "mayusculas_sostenidas", "elongaciones",
                                     "segunda_persona", "imperativo", "cita")]

    secciones = [f"""# P90-03 - Contribuciones de features de V03, mecanismos de error y gap analysis

Generado por `experiments/p90_03_diagnostico_contribuciones.py` y `experiments/p90_03b_informe_handoff.py`.
Commit auditado: `{d['git_commit']}`. Fecha: 2026-10-10. Sin entrenamiento, sin test, sin holdout,
sin submissions, sin commits.

## 1. Resumen ejecutivo

La fase responde la pregunta central con cuatro resultados: (1) **la atribucion exacta OOF esta
BLOCKED_BY_MISSING_ARTIFACTS**: los modelos y vectorizadores por pliegue que generaron las
predicciones OOF no estan serializados (`experiments/14_oof_corregido_v2.py` los ajusta en
memoria y guarda solo probabilidades); (2) el joblib final de V03 (todo el train) se uso como
**evidencia AUXILIAR claramente separada**, con validacion matematica exacta
(sigmoid(z) == predict_proba: {v['sigmoid_vs_predict_proba_max']:.1e}; reconstruccion de z:
residuo maximo {v['reconstruccion_z_max_residuo']:.1e}); (3) las firmas de error son
**acumulativas, no de pocas features dominantes** (0 celdas "pocas dominantes" entre los 466
errores lejanos; 76-86 por ciento de los FN son "acumulacion"), con **conflicto word-char en
43-50 por ciento de los FP de HS/AG** y **canal char dominante en 56-62 por ciento de los FN de
HS/AG**; (4) **ninguna hipotesis genuinamente nueva**: los 18 mecanismos evaluados mapean a
familias cerradas o duplicadas del historial (HIB-H11/H12/H14, ENS, REP, targeting,
contextual_selection, reweighting, structural, augmentation, cita/negacion, P68).
Decision final: **D + A** (atribucion exacta bloqueada; sin hipotesis nueva). La evidencia
auxiliar se interpreta solo como estructura del modelo final, nunca como causa de los errores OOF.

## 2. Alcance y compliance

Sin entrenamiento, sin reconstruir modelos por pliegue, sin recalibrar, sin thresholds, sin
submissions, sin test, sin holdout, sin corpus externos, sin pseudo-labeling, sin operaciones
Git destructivas. Solo se leyeron artefactos de desarrollo autorizados, el modelo final
congelado y los textos dev. Las contribuciones por celda son auxiliares; las distancias,
probabilidades y margenes OOF son canonicos.

## 3. Artefactos utilizados

{tabla(["artefacto", "sha256[:16]", "uso"], [
    ["features/oof_v03_final.npz", d["hashes"]["oof"][:16], "OOF canonico (proba, pred, pliegues, umbrales)"],
    ["features/v2_oof_base_corrected.npz", d["hashes"]["v2base"][:16], "linaje M2 (verificado exacto)"],
    ["experiments/baseline_v03_frozen/model_v03.joblib", d["hashes"]["modelo"][:16], "modelo final auxiliar (vectorizador + 3 LR)"],
    ["data/raw/toxic_train.csv", d["hashes"]["train"][:16], "textos de desarrollo"],
    ["reports/v03_error_cells.csv", d["hashes"]["cells"][:16], "757 celdas error-etiqueta"],
    ["reports/v03_error_controls.csv", d["hashes"]["controls"][:16], "757 controles 1:1"],
    ["reports/v03_error_review_sample.csv", d["hashes"]["sample"][:16], "muestra de revision"],
    ["reports/eda/fase2_lexico_inducido_agresividad.csv", d["hashes"]["lexico"][:16], "lexico in-sample (categorizacion descriptiva)"],
])}

## 4. Disponibilidad de los modelos y vectorizadores por fold

**No disponibles.** El OOF corregido v2 se genero ajustando por pliegue un `TfidfVectorizer`
word + `TfidfVectorizer` char y una `LogisticRegression` por etiqueta, en memoria, guardando
solo `ids, y_true, proba (1483 x 4 modelos x 3), modelos, etiquetas, grupos`. No existe ningun
artefacto joblib/pkl por pliegue en el repositorio (busqueda exhaustiva). Consecuencia:
`EXACT_OOF_ATTRIBUTION = BLOCKED_BY_MISSING_ARTIFACTS`. Artefactos que faltan: modelos ajustados
por pliegue (5 x 3 LR) y sus vectorizadores (5 x word + 5 x char), o equivalentes (coeficientes
y vocabularios por pliegue).

## 5. Limitaciones de atribucion OOF

El modelo final se entreno con todo el train (1731 filas, incluido el desarrollo). Su
probabilidad in-sample no reproduce la decision OOF: el modelo final coincide en direccion con
la prediccion OOF solo en {x['acuerdo']['HS|FP']} de HS FP, {x['acuerdo']['TR|FP']} de TR FP,
{x['acuerdo']['AG|FP']} de AG FP, {x['acuerdo']['HS|FN']} de HS FN, {x['acuerdo']['TR|FN']} de
TR FN y {x['acuerdo']['AG|FN']} de AG FN. Por tanto **ninguna contribucion por celda se
interpreta como causa del error OOF**; las tablas por celda llevan
`attribution_status = AUXILIARY_FULL_MODEL_ATTRIBUTION`. La distorsion media |p_aux - p_oof| es
{v['dif_aux_vs_oof_media']:.3f} (p90 {v['dif_aux_vs_oof_p90']:.3f}).

## 6. Metodologia de descomposicion

Para cada fila i y etiqueta j: `z = intercepto + sum(x_k * coef_k)`, `p = sigmoid(z)`,
`margen = z - logit(threshold)`. La contribucion de la feature k es `x_k * coef_k`. Se separan
los bloques word (`bloque_0__`, {v['n_word_features']} features) y char_wb (`bloque_1__`,
{v['n_char_features']} features). Categorias descriptivas de features (documentadas en el
script): URLs, menciones, hashtags, puntuacion, numeros, insultos y vulgaridad (lexico
in-sample existente), expresiones interpersonales, colectivos, elongaciones, char con espacio,
char alfabetico, otras. Firmas de cobertura: pocas features dominantes (top3 >= 50 por ciento de
|contribucion|), acumulacion (top3 < 25 por ciento), conflicto word-char (signos opuestos),
conflicto positivas-negativas (ratio >= 0.3), lexico dominante sin contexto (FP) / sin evidencia
hostil activada (FN), no interpretable. Los controles se comparan por solapamiento de top-5
features (Jaccard >= 0.4) y banderas estructurales (>= 5 de 7 iguales).

## 7. Validacion matematica

- `sigmoid(z) == predict_proba` en las 1483 x 3: maximo {v['sigmoid_vs_predict_proba_max']:.1e} (exacto).
- `intercepto + suma(contribuciones) == z`: residuo maximo {v['reconstruccion_z_max_residuo']:.1e}.
- Linaje del OOF: M2 de `v2_oof_base_corrected.npz` reordenado por ids reproduce la proba canonica con diferencia 0.0.
- Estratos historicos verificados: HS 40/87/189, TR 13/40/111, AG 31/80/166 (total 84/207/466).
- `reconstruction_error` por celda: maximo 3.3e-15.

## 8. Analisis de errores HS

- HS FP (173): media word {x['resumen'][0]['mean_word_contribution']}, media char {x['resumen'][0]['mean_char_contribution']};
  conflicto word-char {ag['HS|FP']['conflicto_wc']}/173; acumulacion {firmas['HS|FP|acumulacion de muchas features']};
  top-1 positivo del lexico {ag['HS|FP']['top1_pos_es_lexico']}/173 (perra 32, zorra 9, callate 8...);
  top-negativos recurrentes: en, la, **mi perra (9)**, la puta (7).
- HS FN (143): sin lexico en top10 {ag['HS|FN']['sin_lexico_en_top10_pos']}/143; char dominante {ag['HS|FN']['char_dominante']}/143;
  acumulacion {firmas['HS|FN|acumulacion de muchas features']}; top-negativos: la, de, en, **https (9)**;
  con url en texto {ag['HS|FN']['url_texto']}/143 (43 por ciento).
- Nota: las medias de canal del modelo final en FP (word negativo) reflejan que el modelo final a
  menudo no comete el error (seccion 5); la lectura valida es la estructura de features, no la direccion.

## 9. Analisis de errores TR

- TR FP (82): top-1 del lexico {ag['TR|FP']['top1_pos_es_lexico']}/82 (82 por ciento); conflicto word-char {ag['TR|FP']['conflicto_wc']}/82;
  top-positivos: perra (27), zorra (9), eres (8); top-negativos: la puta (8), **mi perra (8)**.
- TR FN (82): mismo signo word-char 82/82 (100 por ciento); sin mencion en texto {ag['TR|FN']['sin_mencion_texto']}/82 (61 por ciento);
  acumulacion {firmas['TR|FN|acumulacion de muchas features']}; top-positivos: puta (12), perra (10), sudaca (7).
- Estructura: TR FN con HS correcta 34/82; TR FP con HS error simultaneo 67/82 (los errores de TR
  viven mayoritariamente junto a los de HS).

## 10. Analisis de errores AG

- AG FP (135): conflicto word-char {ag['AG|FP']['conflicto_wc']}/135 (50 por ciento); top-1 del lexico {ag['AG|FP']['top1_pos_es_lexico']}/135;
  top-positivos: perra (30), zorra (10), callate (6); top-negativos: mi perra (7), la puta (5).
- AG FN (142): char dominante {ag['AG|FN']['char_dominante']}/142 (56 por ciento); acumulacion {firmas['AG|FN|acumulacion de muchas features']}/142;
  top-negativos: en, la, el, **rabe (6)**, polla (6), https (5); con url {ag['AG|FN']['url_texto']}/142.

## 11. Word versus char

{tabla(["etiqueta", "tipo", "n", "word medio", "char medio", "|word|", "|char|", "mismo signo", "dominantes"], filas_resumen)}

Masa de coeficientes del modelo final (|coef|): word {coef['HS']['share_masa_word']:.2f} / char
{1 - coef['HS']['share_masa_word']:.2f} (HS); {coef['TR']['share_masa_word']:.2f} / {1 - coef['TR']['share_masa_word']:.2f} (TR);
{coef['AG']['share_masa_word']:.2f} / {1 - coef['AG']['share_masa_word']:.2f} (AG). El canal char
concentra ~70 por ciento de la masa de pesos pero solo 4024 features word existen: las decisiones
mezclan ambos; el conflicto word-char es una firma sustantiva de los FP de HS/AG y el canal char
domina en los FN de HS/AG. Es una propiedad estructural del modelo union (auxiliar), no una causa
demostrada del error OOF.

## 12. Falsos positivos

{tabla(["hipotesis", "evidencia cuantificada (auxiliar)", "estado"], [
    ["H1 vulgaridad fuerte", f"top1 lexico HS {ag['HS|FP']['top1_pos_es_lexico']}/173, TR {ag['TR|FP']['top1_pos_es_lexico']}/82, AG {ag['AG|FP']['top1_pos_es_lexico']}/135; coef perra 3.23/4.34/3.25", "CLOSED (reweighting/term weighting)"],
    ["H2 afiliativo", "mi perra/la puta con coef NEGATIVO (mi perra: 18 docs, 12 no toxicos; la puta: 98 docs, 88 no toxicos)", "CLOSED (HIB-H11; contextual_selection)"],
    ["H3 char_wb superficial", f"conflicto word-char: HS {ag['HS|FP']['conflicto_wc']}, TR {ag['TR|FP']['conflicto_wc']}, AG {ag['AG|FP']['conflicto_wc']}", "DUPLICATE (ENS/REP)"],
    ["H4 menciones", f"menciones en top10: HS {ag['HS|FP']['mencion_en_top10_pos']}, TR {ag['TR|FP']['mencion_en_top10_pos']}, AG {ag['AG|FP']['mencion_en_top10_pos']}", "CLOSED (targeting/P68); sin respaldo"],
    ["H5 literales/no dirigidas", f"sin mencion en texto: HS {ag['HS|FP']['sin_mencion_texto']}/173, AG {ag['AG|FP']['sin_mencion_texto']}/135", "CLOSED (contextual_selection)"],
    ["H6 estructura", f"puntuacion en top10: HS {ag['HS|FP']['puntuacion_en_top10']}, AG {ag['AG|FP']['puntuacion_en_top10']}", "CLOSED (structural_features/HIB)"],
    ["H7 citas", f"comillas: HS {ag['HS|FP']['cita_texto']}, TR {ag['TR|FP']['cita_texto']}, AG {ag['AG|FP']['cita_texto']}", "CLOSED (P19/P66)"],
])}

## 13. Falsos negativos

{tabla(["hipotesis", "evidencia cuantificada (auxiliar)", "estado"], [
    ["H1 indirecta sin insulto", f"sin lexico top10: HS {ag['HS|FN']['sin_lexico_en_top10_pos']}/143, TR {ag['TR|FN']['sin_lexico_en_top10_pos']}/82, AG {ag['AG|FN']['sin_lexico_en_top10_pos']}/142", "CLOSED (HIB-H14)"],
    ["H2 colectivas", f"colectivo top10: HS {ag['HS|FN']['colectivo_en_top10_pos']}, AG {ag['AG|FN']['colectivo_en_top10_pos']}; rabe: 149 docs, 114 no toxicos", "CLOSED (HIB-H14/joint_labels)"],
    ["H3 generalizaciones", f"acumulacion HS {firmas['HS|FN|acumulacion de muchas features']}/143, AG {firmas['AG|FN|acumulacion de muchas features']}/142", "CLOSED (representacion)"],
    ["H4 TR sin mencion", f"sin mencion {ag['TR|FN']['sin_mencion_texto']}/82; TR FN con HS correcta 34/82", "CLOSED (HIB-H12); anotacion BLOCKED"],
    ["H5 retorica/cita/relato", f"url: HS {ag['HS|FN']['url_texto']}, AG {ag['AG|FN']['url_texto']}; https top-negativo", "CLOSED (cita/negacion; registro)"],
    ["H6 ortografia", f"elongacion: HS {ag['HS|FN']['elongacion_texto']}, AG {ag['AG|FN']['elongacion_texto']}", "CLOSED (augmentation/fuzzy)"],
    ["H7 word-char contradictorio", f"conflicto: HS {ag['HS|FN']['conflicto_wc']}, TR {ag['TR|FN']['conflicto_wc']}, AG {ag['AG|FN']['conflicto_wc']}", "DUPLICATE (ENS)"],
    ["H8 conflicto evidencia", f"neg domina: HS {ag['HS|FN']['neg_domina_ratio_0.3']}/143, TR {ag['TR|FN']['neg_domina_ratio_0.3']}/82, AG {ag['AG|FN']['neg_domina_ratio_0.3']}/142", "CLOSED (reweighting/hardness)"],
])}

## 14. Errores alejados del threshold

{tabla(["etiqueta", "tipo", "n lejos", "acumulacion", "conflicto wc", "mismo signo", "top features positivas"], filas_far)}

En los 466 errores lejanos no hay ninguna celda dominada por pocas features (0 "pocas
dominantes"): son acumulativos. Los FP lejanos de las tres etiquetas tienen a **perra** como
top-1 positivo en todas las folds (HS: f1 5, f2 zorra 3, f3 8, f4 7, f5 4; AG: 3/5/6/7/4;
TR: 3/4/4/6/3). Los FN lejanos son acumulativos (64-75 de 77-89). No se denomina a los errores
lejanos "sin informacion": la lectura es que la evidencia disponible es insuficiente o
contradictoria en el modelo auxiliar, sin mecanismo nuevo demostrado.

## 15. Comparacion con controles

757 relaciones error-control sobre {d['controles']['unicos']} controles unicos (reuso maximo 8;
284 con reuso 1). Clasificacion (reglas descriptivas): parecido 18, lexicalmente_similar 7,
estructuralmente_parecido 378, sin_control_comparable 354. El criterio estructural (>=5 de 7
banderas) es debil: la media de banderas iguales es 4.69 y las banderas mayoritarias (mayusculas,
mencion) coinciden por frecuencia. El criterio lexical (Jaccard top-5 >= 0.4) casi nunca se
cumple (25/757): los errores y sus controles emparejados por distancia no comparten las mismas
features dominantes (esperable en un espacio disperso de 27431 dimensiones). Subset de la
muestra de revision: {d['muestra']['relaciones_de_la_muestra']} relaciones (100 errores), con la
misma estructura. Los controles no son semanticamente equivalentes por construccion.

## 16. Diferencias por slice

{tabla(["slice", "label", "errores dentro", "word medio", "char medio", "features activas dentro", "features activas fuera"], filas_slice)}

Las regiones de bajo rendimiento (URLs, hashtags, colectivos_superficie) tienen errores con
**mas features activas** que el resto (URLs HS 248.9 vs 185.0; colectivos HS 272.7 vs 177.5) y
mezcla de canales distinta, pero las firmas dominantes siguen siendo acumulacion/conflicto: no
aparece una firma propia que no exista fuera. Interpretacion descriptiva; grupos pequenos
(hashtags TR 12 errores) con incertidumbre alta. El enrutado por region fue cerrado (P68).

## 17. Estructura de etiquetas

Identidad TR=>HS verificada (0 violaciones en 1483). TR FN: 48/82 con error HS simultaneo y
34/82 con HS correcta; TR FP: 67/82 con error HS simultaneo y 15/82 con HS correcta. Los errores
de TR acompanan mayoritariamente a los de HS; la factorizacion condicional ya fue explorada y
cerrada (D-044/D-045, familia joint_labels). No se impone ninguna relacion nueva.

## 18. Cruce con resultados historicos

{tabla(["etiqueta", "intercepto", "share masa word", "top positivos", "top negativos"], filas_coef)}

La tabla completa de gap analysis (18 mecanismos) esta en `p90_03_hypothesis_gap_analysis.csv`.
Resumen: 15 CLOSED (incluida FN-H4, cuyo apendice de anotacion queda BLOCKED_BY_RULES),
2 DUPLICATE (FP-H3, FN-H7), 1 UNRESOLVED (X2), 0 POTENTIALLY_NEW. Los precedentes mas directos: HIB-H11 (bloques
bromista/interpersonal, -0.0213), HIB-H12 (objetivo individual/colectivo, -0.0091), HIB-H14
(lexico inducido x colectiva, -0.0297), HIB-UNION (-0.0316), ENS-01..04 (pesos por
representacion), REP-05/06 (union adoptada), contextual_selection (P0=V03), reweighting V30/P69,
structural V15-01, augmentation V6-08, fuzzy V29, cita/negacion P19, P68 enrutado.

## 19. Hipotesis cerradas

FP-H1, FP-H2, FP-H4, FP-H5, FP-H6, FP-H7, FN-H1, FN-H2, FN-H3, FN-H4, FN-H5, FN-H6, FN-H8,
X1, X3 (15 filas CLOSED) y las dos DUPLICATE (FP-H3, FN-H7): todas con experimento historico
equivalente identificado y resultado conocido.

## 20. Hipotesis complementarias

No se identifica ninguna linea complementaria nueva con evidencia suficiente para experimentar.
Las observaciones de P90-00/P90-02 (casebook, slices, grupos) siguen siendo el diagnostico
valido; P90-03 agrega la estructura de contribuciones auxiliares y la confirma.

## 21. Posibles nuevas hipotesis

**Ninguna.** La novedad exigida (mecanismo o procedimiento no evaluado) no se cumple en ninguna
de las firmas observadas: reweighting por familia, gating contextual, ponderacion de canales,
bloques estructurales y enrutado por region tienen experimentos historicos ejecutados y
cerrados. La unica linea realmente no ejecutada es la atribucion exacta OOF, bloqueada por
artefactos faltantes (no por reglas), y la revision humana de anotacion, bloqueada por reglas
(P87). No se propone experimentar.

## 22. Limitaciones

- Atribucion exacta OOF bloqueada (modelos por pliegue no serializados); la evidencia auxiliar
  del modelo final no explica las decisiones OOF (acuerdo 1.4-32.9 por ciento).
- El modelo final vio el desarrollo en entrenamiento: sus probabilidades y contribuciones
  in-sample estan infladas; solo se usan para estructura de features.
- Lexico in-sample (150 lemas) usado solo para categorizar descriptivamente.
- Controles no independientes, reutilizados y emparejados por distancia; los umbrales de
  similitud (Jaccard 0.4; 5/7 banderas) son convenciones descriptivas declaradas.
- Los grupos pequenos de slices tienen incertidumbre alta; sin inferencia poblacional.
- Distancia al threshold no es confianza calibrada (heredado de P90.02).

## 23. Decision final

Clasificacion del diagnostico: **VERIFIED** (validacion matematica, linaje, estratos,
reconstruccion); **SUPPORTED** (estructura acumulativa del error; lexico de misoginia como top-1
positivo en FP; evidencia negativa anti-toxica aprendida; char dominante en FN; conflicto
word-char en FP); **TENTATIVE** (lecturas de registro/contexto en FN); **UNRESOLVED** (mecanismo
especifico de los 466 errores lejanos).

Hipotesis: 15 CLOSED, 2 DUPLICATE, 0 POTENTIALLY_NEW, 1 UNRESOLVED (X2).

**Recomendacion combinada: D. EXACT_OOF_ATTRIBUTION_BLOCKED + A. NO_NEW_HYPOTHESIS.**
La atribucion exacta no es posible sin reentrenar los modelos por pliegue (prohibido); ninguna
hipotesis nueva, concreta, falsable y no duplicada sobrevive al cruce con el historial. V03 se
mantiene FROZEN; no se recomienda reemplazarlo ni experimentar en esta fase.
"""]

    secciones.append(f"""
## Anexo - Asociaciones descriptivas de frases clave (desarrollo)

{tabla(["frase", "docs dev", "HS=1", "TR=1", "AG=1", "sin etiqueta (000)"], filas_asoc)}

Lectura descriptiva (sin construir lexico nuevo): "perra" y "zorra" son mayoritariamente
toxicas en el corpus (155/197 y 91/104 con HS=1) y reciben los coeficientes positivos mas altos;
"mi perra" (12/18 sin etiqueta), "la puta" (88/98), "hijo" (106/113), "escoria" (33/37),
"acoso" (49/56) y "rabe" (114/149) son mayoritariamente no toxicas y reciben coeficientes
negativos. Esta estructura explica el mecanismo observado: los FP ocurren cuando el lexico
positivo domina sin contrapeso afiliativo, y los FN cuando tokens con asociacion anti-toxica
aprendida (rabe, acoso, hijo, la puta) restan puntuacion en textos que si son toxicos.
""")

    secciones.append(f"""
## Anexo - Firmas por etiqueta y tipo (recuentos)

{tabla(["etiqueta", "tipo", "firma", "n"], filas_firmas)}
""")

    return "\n".join(secciones)


def construir_handoff(x: dict[str, Any]) -> str:
    """Construye el handoff autocontenido para ChatGPT."""
    d = x["d"]
    v = d["validacion"]
    ag = d["agregados"]
    far = d["far"]
    firmas = d["firmas"]
    resumen = x["resumen"]
    asoc = x["asoc"]
    filas_resumen = [[f["label"], f["error_type"], f["n_errors"], f["near_errors"], f["intermediate_errors"],
                      f["far_errors"], f["mean_word_contribution"], f["mean_char_contribution"],
                      f["share_word_char_same_sign"]] for f in resumen]
    filas_asoc = [[f["frase"], f["docs_dev"], f["HS_1"], f["TR_1"], f["AG_1"], f["sin_etiqueta_000"]] for f in asoc]
    return f"""# P90-03 - Handoff autocontenido: contribuciones de features de V03 y mecanismos de error

Commit auditado: `{d['git_commit']}`. Fecha: 2026-10-10. Sin entrenamiento, sin test, sin
holdout, sin submissions, sin commits. Este documento contiene los hallazgos completos; las
tablas viven en `reports/p90_03_*.csv`.

## 1. Baseline y estado Git

V03 FROZEN: TF-IDF word(1,2)+char_wb(3,5), min_df=2, sublinear_tf, lowercase, strip_accents=None;
LogisticRegression C=2.0 balanced liblinear x3; umbrales 0.44/0.51/0.46; corrected_cv
0.7772114525165721; LB 0.78306. Git: rama main, HEAD {d['git_commit']}, cambios locales de
P90.02 preservados, sin commits ni push en esta fase.

## 2. Artefactos utilizados y hashes

OOF canonico `features/oof_v03_final.npz` ({d['hashes']['oof'][:16]}); padre v2
`features/v2_oof_base_corrected.npz` ({d['hashes']['v2base'][:16]}, linaje M2 verificado exacto);
modelo final `experiments/baseline_v03_frozen/model_v03.joblib` ({d['hashes']['modelo'][:16]});
textos `data/raw/toxic_train.csv` ({d['hashes']['train'][:16]}); casebook/celdas/controles/muestra
({d['hashes']['cells'][:16]}/{d['hashes']['controls'][:16]}/{d['hashes']['sample'][:16]});
lexico in-sample ({d['hashes']['lexico'][:16]}).

## 3. Disponibilidad de modelos OOF por fold

**AUSENTES.** `experiments/14_oof_corregido_v2.py` ajusta por pliegue vectorizadores y LR en
memoria y guarda solo probabilidades. EXACT_OOF_ATTRIBUTION = BLOCKED_BY_MISSING_ARTIFACTS.
Disponible solo el modelo final (todo el train) como AUXILIARY_FULL_MODEL_ATTRIBUTION.

## 4. Metodologia de contribuciones

z = intercepto + sum(x_k*coef_k); p = sigmoid(z); margen = z - logit(threshold); contribucion_k =
x_k*coef_k; bloques word ({v['n_word_features']}) y char_wb ({v['n_char_features']}); categorias
descriptivas de features y firmas de cobertura (pocas dominantes, acumulacion, conflicto
word-char, conflicto pos-neg, lexico sin contexto / sin evidencia hostil, no interpretable).

## 5. Validacion de la reconstruccion del score

sigmoid(z) == predict_proba: maximo {v['sigmoid_vs_predict_proba_max']:.1e} (exacto);
intercepto + suma(contribuciones) == z: residuo maximo {v['reconstruccion_z_max_residuo']:.1e};
reconstruction_error por celda <= 3.3e-15; estratos historicos verificados (HS 40/87/189,
TR 13/40/111, AG 31/80/166). El modelo final coincide en direccion con la prediccion OOF en
{x['acuerdo']['HS|FP']}, {x['acuerdo']['TR|FP']}, {x['acuerdo']['AG|FP']} (FP) y
{x['acuerdo']['HS|FN']}, {x['acuerdo']['TR|FN']}, {x['acuerdo']['AG|FN']} (FN): por eso la
atribucion por celda es auxiliar y no causal.

## 6. Hallazgos por etiqueta

{tabla(["etiqueta", "tipo", "n", "cerca", "intermedio", "lejos", "word medio", "char medio", "mismo signo"], filas_resumen)}

- HS FP: top-1 positivo del lexico {ag['HS|FP']['top1_pos_es_lexico']}/173; conflicto word-char
  {ag['HS|FP']['conflicto_wc']}/173; top-negativos en, la, mi perra (9), la puta (7).
- HS FN: sin lexico top10 {ag['HS|FN']['sin_lexico_en_top10_pos']}/143; char dominante
  {ag['HS|FN']['char_dominante']}/143; https entre los top-negativos (9).
- TR FP: top-1 lexico {ag['TR|FP']['top1_pos_es_lexico']}/82 (82 por ciento); perra 27, zorra 9.
- TR FN: sin mencion {ag['TR|FN']['sin_mencion_texto']}/82; mismo signo 82/82; puta 12, perra 10, sudaca 7.
- AG FP: conflicto word-char {ag['AG|FP']['conflicto_wc']}/135 (50 por ciento); perra 30.
- AG FN: char dominante {ag['AG|FN']['char_dominante']}/142; rabe (6) y https (5) entre top-negativos.

## 7. Firmas de falsos positivos

Acumulacion {firmas['HS|FP|acumulacion de muchas features']}/{firmas['TR|FP|acumulacion de muchas features']}/{firmas['AG|FP|acumulacion de muchas features']}
(HS/TR/AG); conflicto word-char {firmas['HS|FP|conflicto word-char']}/{firmas['TR|FP|conflicto word-char']}/{firmas['AG|FP|conflicto word-char']};
conflicto pos-neg {firmas['HS|FP|conflicto positivas-negativas']}/{firmas['TR|FP|conflicto positivas-negativas']}/{firmas['AG|FP|conflicto positivas-negativas']};
pocas dominantes ~0. Mecanismo descriptivo: lexico de misoginia (perra/zorra/callate/eres una)
con coeficientes positivos altos sin contrapeso activo; los marcadores afiliativos
("mi perra", "la puta") ya tienen coeficiente NEGATIVO aprendido (asociacion descriptiva:
mi perra 12/18 y la puta 88/98 sin etiqueta en dev) pero insuficiente en estos FP.

## 8. Firmas de falsos negativos

Acumulacion {firmas['HS|FN|acumulacion de muchas features']}/{firmas['TR|FN|acumulacion de muchas features']}/{firmas['AG|FN|acumulacion de muchas features']};
char dominante HS/AG {ag['HS|FN']['char_dominante']}/{ag['AG|FN']['char_dominante']}; neg domina
(ratio>=0.3) {ag['HS|FN']['neg_domina_ratio_0.3']}/{ag['TR|FN']['neg_domina_ratio_0.3']}/{ag['AG|FN']['neg_domina_ratio_0.3']}.
Evidencia negativa recurrente: palabras funcionales (en, la, de, el), https y tokens con
asociacion anti-toxica aprendida (rabe, acoso, hijo, la puta). El canal char concentra la senal
positiva disponible (56-62 por ciento) pero insuficiente.

## 9. Word frente a char

Masa de |coef|: word ~0.30 / char ~0.70 en las tres etiquetas. Conflicto word-char: FP
{ag['HS|FP']['conflicto_wc']}/{ag['TR|FP']['conflicto_wc']}/{ag['AG|FP']['conflicto_wc']};
FN {ag['HS|FN']['conflicto_wc']}/{ag['TR|FN']['conflicto_wc']}/{ag['AG|FN']['conflicto_wc']}.
Mismo signo: FN TR 82/82; FN AG 138/142. El conflicto de canales es firma de una parte
sustancial de los FP; ponderar canales es familia ENS/REP ya ejecutada (DUPLICATE).

## 10. Errores alejados del threshold

466/757 (61.6 por ciento). Lejanos por etiqueta/tipo: HS FP {far['HS|FP']['n']}, HS FN {far['HS|FN']['n']},
TR FP {far['TR|FP']['n']}, TR FN {far['TR|FN']['n']}, AG FP {far['AG|FP']['n']}, AG FN {far['AG|FN']['n']}.
Cero celdas "pocas dominantes"; "perra" es el top-1 positivo de FP lejanos en las cinco folds;
FN lejanos acumulativos (64-75 de 77-89). Sin mecanismo nuevo; la distancia no es calibracion.

## 11. Comparacion con controles

757 relaciones / {d['controles']['unicos']} controles unicos (reuso max 8): parecido 18,
lexicalmente_similar 7, estructuralmente_parecido 378, sin_control_comparable 354. El criterio
estructural es debil (media 4.69 banderas iguales); el lexical casi nunca se cumple (25/757).
No son semanticamente equivalentes; lectura cualitativa.

## 12. Patrones por slice

URLs/colectivos: los errores dentro tienen mas features activas (248.9 vs 185.0; 272.7 vs 177.5
para HS) y mezcla de canales distinta, pero sin firma propia nueva; hashtags con grupos pequenos
(12-37 errores). Enrutado por region cerrado (P68).

## 13. Hipotesis contrastadas

18 mecanismos: 15 CLOSED (FN-H4 incluida, con su apendice de anotacion TR en
BLOCKED_BY_RULES), 2 DUPLICATE (FP-H3 char/canales, FN-H7 conflicto de canales),
1 UNRESOLVED (X2 mecanismo de lejanos). 0 POTENTIALLY_NEW.
Tabla completa: `p90_03_hypothesis_gap_analysis.csv`.

## 14. Experimentos historicos relacionados

HIB-H11 -0.0213 (bromista/interpersonal), HIB-H12 -0.0091 (individual/colectivo), HIB-H14
-0.0297 (lexico x colectiva), HIB-UNION -0.0316; ENS-01..04 (pesos por representacion,
descartados); REP-05/06 (union adoptada); contextual_selection (P0=V03); reweighting V30/P69;
structural V15-01; augmentation V6-08; fuzzy V29; cita/negacion P19; P66 negacion no adoptada;
P68 enrutado; joint_labels cerrado (D-045).

## 15. Hipotesis genuinamente nuevas

**Ninguna.** No se propone candidato (archivo de candidatos con fila NO_CANDIDATE explicita).
La unica linea no ejecutada es la atribucion exacta OOF (bloqueada por artefactos) y la revision
de anotacion (bloqueada por reglas, P87).

## 16. Limitaciones

Atribucion exacta bloqueada; evidencia auxiliar in-sample (no causal); lexico in-sample solo
descriptivo; controles no independientes; grupos pequenos; distancia no calibrada.

## 17. Decision final

Clasificacion: VERIFIED (validacion), SUPPORTED (firmas acumulativas, lexico dominante en FP,
evidencia negativa anti-toxica, char dominante en FN, conflicto word-char en FP), TENTATIVE
(registro/contexto), UNRESOLVED (mecanismo de lejanos).
**D. EXACT_OOF_ATTRIBUTION_BLOCKED + A. NO_NEW_HYPOTHESIS.** V03 FROZEN.

## Anexo - Asociaciones descriptivas de frases clave (dev)

{tabla(["frase", "docs", "HS=1", "TR=1", "AG=1", "000"], filas_asoc)}
"""


def auditar_self() -> dict[str, Any]:
    """Comprobacion AST del propio script (operacion y fichero concretos)."""
    arbol = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    docstrings = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            cuerpo = getattr(nodo, "body", [])
            if cuerpo and isinstance(cuerpo[0], ast.Expr) and isinstance(cuerpo[0].value, ast.Constant):
                if isinstance(cuerpo[0].value.value, str):
                    docstrings.add(id(cuerpo[0].value))
    puro = re.compile(r"^[\w./\\-]+\.\w{2,5}$|^[\w./\\-]+/$")
    tokens = []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Constant) and isinstance(nodo.value, str) and id(nodo) not in docstrings:
            v = nodo.value
            if puro.match(v) and any(p in v.lower() for p in
                                     ("toxic_test", "test.csv", "leaderboard", "holdout", "submission")):
                tokens.append(v)
    modulos = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            modulos |= {a.name.split(".")[0] for a in nodo.names}
        elif isinstance(nodo, ast.ImportFrom) and nodo.module:
            modulos.add(nodo.module.split(".")[0])
    return {"tokens": sorted(set(tokens)), "modulos": sorted(modulos)}


def main() -> int:
    """Genera informe, gap analysis, candidatos, manifiesto y handoff."""
    d = json.loads(RUTA_RESULTS.read_text(encoding="utf-8"))
    resumen = leer_csv(RUTA_SUMMARY)
    slices = leer_csv(RUTA_SLICE_SUM)
    asoc = asociaciones_descriptivas()
    escribir_csv(RUTA_ASSOC, asoc)

    # Acuerdo auxiliar vs OOF por label/tipo (metrica de honestidad).
    celdas = leer_csv(RUTA_BY_CELL)
    acuerdo = {}
    for et in ("HS", "TR", "AG"):
        for tipo in ("FP", "FN"):
            sel = [f for f in celdas if f["label"] == et and f["error_type"] == tipo]
            ok = sum(1 for f in sel if (float(f["aux_probability"]) >= float(f["threshold"])) == (int(f["y_pred"]) == 1))
            acuerdo[f"{et}|{tipo}"] = f"{ok}/{len(sel)} ({100.0 * ok / len(sel):.1f}%)"
    x = {"d": d, "resumen": resumen, "slices": slices, "asoc": asoc, "acuerdo": acuerdo}

    informe = construir_informe(x)
    handoff = construir_handoff(x)
    escribir_texto(RUTA_INFORME, informe + "\n")
    escribir_texto(RUTA_HANDOFF, handoff + "\n")

    gap = construir_gap(d)
    escribir_csv(RUTA_GAP, gap)
    escribir_csv(RUTA_CAND, [{
        "hypothesis_id": "NO_CANDIDATE",
        "hypothesis": "Sin candidatos: ninguna hipotesis genuinamente nueva, concreta, falsable y no duplicada",
        "mechanism": "n/a",
        "evidence_from_v03": "18 mecanismos evaluados: 15 CLOSED, 2 DUPLICATE, 1 UNRESOLVED; 0 POTENTIALLY_NEW",
        "compatible_errors": "n/a", "counterexamples": "n/a", "folds": "n/a",
        "channel": "n/a", "historical_family": "n/a",
        "why_not_duplicate": "n/a", "why_could_improve": "n/a", "overfitting_risk": "n/a",
        "minimal_future_experiment": "ninguno propuesto en esta fase",
        "success_criteria": "n/a", "stop_criteria": "n/a", "compliance": "n/a",
        "final_decision": "NO_CANDIDATE",
    }])

    manifiesto = {
        "audit_id": "P90-03",
        "git_commit": d["git_commit"],
        "branch": subprocess.run(["git", "branch", "--show-current"], capture_output=True,
                                 text=True, cwd=RAIZ).stdout.strip(),
        "input_artifacts": ["features/oof_v03_final.npz", "features/v2_oof_base_corrected.npz",
                            "experiments/baseline_v03_frozen/model_v03.joblib", "data/raw/toxic_train.csv",
                            "reports/v03_error_casebook.csv", "reports/v03_error_cells.csv",
                            "reports/v03_error_controls.csv", "reports/v03_error_review_sample.csv",
                            "reports/eda/fase2_lexico_inducido_agresividad.csv"],
        "input_hashes": d["hashes"],
        "folds": "5 (StratifiedGroupKFold, grupos fusionados, D-004/D-016); asignacion en el OOF canonico",
        "thresholds": {"HS": 0.44, "TR": 0.51, "AG": 0.46},
        "attribution_method": "descomposicion lineal z = intercepto + sum(x_k*coef_k); contribucion_k = x_k*coef_k",
        "feature_names_source": "FeatureUnion.get_feature_names_out() del joblib final (bloques word/char)",
        "model_artifacts": "experiments/baseline_v03_frozen/model_v03.joblib (hash completo en input_hashes)",
        "vectorizer_artifacts": "incluido en el mismo joblib (FeatureUnion word + char_wb)",
        "reconstruction_tolerance": "1e-9 para z; 1e-12 para sigmoid vs predict_proba",
        "validation_results": {
            "sigmoid_vs_predict_proba_max": d["validacion"]["sigmoid_vs_predict_proba_max"],
            "reconstruccion_z_max_residuo": d["validacion"]["reconstruccion_z_max_residuo"],
            "estratos_verificados": d["validacion"]["estratos_verificados"],
            "acuerdo_auxiliar_vs_oof": acuerdo,
        },
        "output_artifacts": [
            "reports/p90_03_feature_contribution_diagnostic.md", "reports/p90_03_chatgpt_handoff.md",
            "reports/p90_03_artifact_manifest.json", "reports/p90_03_audit_results.json",
            "reports/p90_03_error_contribution_by_cell.csv", "reports/p90_03_feature_contribution_summary.csv",
            "reports/p90_03_error_signature_profiles.csv", "reports/p90_03_error_control_comparison.csv",
            "reports/p90_03_slice_contribution_summary.csv", "reports/p90_03_hypothesis_gap_analysis.csv",
            "reports/p90_03_hypothesis_candidates.csv", "reports/p90_03_key_feature_associations.csv",
            "experiments/p90_03_diagnostico_contribuciones.py", "experiments/p90_03b_informe_handoff.py",
            "notebooks/10_final_pipeline.ipynb",
        ],
        "unresolved_issues": [
            "EXACT_OOF_ATTRIBUTION_BLOCKED: faltan modelos y vectorizadores por pliegue",
            "X2: mecanismo especifico de los 466 errores lejanos no demostrado",
            "evidencia auxiliar in-sample: no causal sobre las decisiones OOF",
        ],
    }
    escribir_texto(RUTA_MANIFEST, json.dumps(manifiesto, ensure_ascii=False, indent=1) + "\n")

    for ruta in (RUTA_INFORME, RUTA_HANDOFF, RUTA_MANIFEST, RUTA_GAP, RUTA_CAND, RUTA_ASSOC):
        if any(b > 127 for b in ruta.read_bytes()):
            raise SystemExit(f"STOP: bytes > 127 en {ruta.name}")
    auditoria = auditar_self()
    permitidos = {"toxic_train.csv", "holdout_v2_ids.csv", "oof_v03_final.npz",
                  "v2_oof_base_corrected.npz", "model_v03.joblib",
                  "fase2_lexico_inducido_agresividad.csv", "v03_error_casebook.csv",
                  "v03_error_cells.csv", "v03_error_controls.csv", "v03_error_review_sample.csv",
                  "p90_03b_informe_handoff.py", "p90_03_diagnostico_contribuciones.py",
                  "test.csv", "submission", "submissions", "leaderboard", "holdout"}
    raros = [vv for vv in auditoria["tokens"] if vv not in permitidos]
    if raros:
        raise SystemExit(f"STOP: tokens de ruta inesperados: {raros}")
    prohibidos = {"transformers", "sentence_transformers", "openai", "anthropic"} & set(auditoria["modulos"])
    if prohibidos:
        raise SystemExit(f"STOP: imports prohibidos: {sorted(prohibidos)}")
    print("P90-03 parte 2 completada:")
    print(f"  informe: {RUTA_INFORME.name} ({len(informe.splitlines())} lineas)")
    print(f"  handoff: {RUTA_HANDOFF.name} ({len(handoff.splitlines())} lineas)")
    print(f"  gap: {len(gap)} filas | candidatos: 1 fila NO_CANDIDATE | manifiesto OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
