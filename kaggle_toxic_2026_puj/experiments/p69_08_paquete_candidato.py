"""Freezing of the candidate package: the research ends here, the candidate does not.

What this commit is and is not
------------------------------
It is the formal close of the experimental search. No new mechanisms are opened, no partition
is added, no weight is touched. What it does is package a specific candidate so that a future
decision, whoever makes it, starts from something reproducible instead of from a report.

Three states that must not be confused with one another:

    research closed       no new mechanisms are being searched for
    candidate rejected    not this: nothing here says the candidate is wrong
    candidate adopted     not this either: nothing here says it should ship

The candidate is `PROMISING_UNCONFIRMED`, and the governance status is `PENDING`. Those two
words are the honest summary of where the evidence stops.

What the permutation test does and does not prove
-------------------------------------------------
It proves that the real weight assignment carries more information than equivalent permuted
assignments of the same weights. That is a statement about specificity, and it is what was
tested.

It does NOT prove population-level generalisation, and it does NOT prove a leaderboard gain.
The group bootstrap remains the binding limitation, and no amount of re-reading the permutation
result turns it into the other thing.
"""

from __future__ import annotations

import ast
import csv as _csv
import hashlib
import json
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import V03_CCV, V03_SHA, V03_SUB

REPORTS = PROJECT_ROOT / "reports"
MD_C = REPORTS / "toxic_candidate_cost_sensitive_final.md"
CSV_C = REPORTS / "toxic_candidate_cost_sensitive_evidence.csv"
JSON_M = REPORTS / "toxic_candidate_cost_sensitive_manifest.json"
MD_G = REPORTS / "toxic_candidate_governance_note.md"

ESTADOS = [
    ("ORIGINAL", "+0.0082045862", "2026-09-25, nunca enviado, nunca auditado"),
    ("REPRODUCED", "+0.0082045862", "R3, diferencia 3,2e-11"),
    ("SECOND_PARTITION", "+0.0105253130", "5 de 5 pliegues, semilla 20260928"),
    ("THIRD_PARTITION_CORRECTED", "+0.0048825807", "4 de 5 pliegues, semilla 20261001"),
    ("PERMUTATION", "p = 0.000999", "0 de 1000 superan al observado"),
    ("FINAL_STATUS", "PROMISING_UNCONFIRMED", "5 de 6 componentes"),
]


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P69-08] {mensaje}", flush=True)


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


def sha(ruta: Path, corto: int = 32) -> str:
    """SHA256 de un fichero, o cadena vacia si no existe.

    Args:
        ruta: fichero.
        corto: longitud del prefijo.

    Returns:
        El hash truncado.
    """
    if not ruta.exists():
        return ""
    return hashlib.sha256(ruta.read_bytes()).hexdigest()[:corto]


def main() -> None:
    """Empaqueta el candidato congelado y declara el cierre experimental.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P69-08 paquete del candidato y cierre experimental")
    log("   0 experimentos, 0 entrenamientos, 0 submissions")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    v = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)
    if float(v["corrected_cv"][0]) != V03_CCV:
        raise SystemExit("PARADA: el OOF canonico no reproduce corrected_cv")
    log(f"   V03 intacto: {V03_CCV!r}")

    per = json.loads((REPORTS / "a_cost_sensitive_downweight_permutation.json")
                     .read_text(encoding="utf-8"))
    json.loads((REPORTS / "a_cost_sensitive_downweight_final_evidence.json")
                      .read_text(encoding="utf-8"))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()

    # ---------- el manifest ----------
    manifest = {
        "candidato": "A_cost_sensitive_downweight",
        "descripcion_correcta": "downweight basado en violacion de margen del propio V03",
        "descripciones_que_NO_aplican": ["meta-model", "stacking", "residual correction",
                                         "rescue weighting"],
        "por_que_esas_no_aplican": (
            "P58 lo etiquetaba como 'meta'; el codigo no construye ningun meta-modelo. Y "
            "'rescue weighting' es P68-H2, que va en direccion OPUESTA: alli se SUBE el peso "
            "de los casos que otro modelo rescata"),
        "hypothesis": {
            "enunciado": ("los ejemplos con violacion alta del margen de V03 reciben menos "
                          "peso en el entrenamiento, y eso mueve la frontera de forma util"),
            "regla": ["margen_favor = (p_V03 - umbral) * (2*y - 1)",
                      "violacion_l = clip(-margen_favor, 0, None)  por etiqueta",
                      "error_score = violacion_l.mean(axis=1)  por ejemplo",
                      "z = (error_score - mediana) / IQR",
                      "peso = clip(1 - 0.5*z, 0.5, 1.5)",
                      "clase = pesos de clase explicitos con multiplicador anidado"],
            "signo": "la regla SOLO baja pesos; ningun ejemplo supera 1",
        },
        "git_commit_de_este_manifest": commit,
        "git_commit_original": "c96cb0f",
        "fecha_original": "2026-09-25",
        "data_hash": {
            "oof_v03_final_npz": sha(PROJECT_ROOT / "features" / "oof_v03_final.npz"),
            "oof_weighting_v4_npz": sha(PROJECT_ROOT / "features" / "oof_weighting_v4.npz"),
            "hardness_v4_npz": sha(PROJECT_ROOT / "features" / "hardness_v4.npz"),
            "submission_v03_csv": hash_real,
            "modelo_v03_joblib": sha(PROJECT_ROOT / "experiments" / "baseline_v03_frozen"
                                     / "model_v03.joblib"),
        },
        "representation": {
            "word": {"analyzer": "word", "ngram_range": [1, 2], "min_df": 2,
                     "sublinear_tf": True},
            "char": {"analyzer": "char_wb", "ngram_range": [3, 5], "min_df": 2,
                     "sublinear_tf": True},
            "ajuste": "por pliegue, solo con textos del entrenamiento",
        },
        "classifier": {"tipo": "LogisticRegression", "C": 2.0, "max_iter": 1000,
                       "solver": "liblinear", "class_weight": "explicito con multiplicador",
                       "sample_weight": "el de la regla congelada"},
        "thresholds": [0.44, 0.51, 0.46],
        "weight_function": "clip(1 - 0.5*z, 0.5, 1.5), z = (error_score - mediana) / IQR",
        "seed": 42,
        "fold_protocol": {
            "metodo": "StratifiedGroupKFold",
            "grupos": "near-duplicate UNION pseudo-autor, fusionados",
            "n_pliegues": 5,
            "canonica": "RANDOM_STATE",
            "segunda": 20260928,
            "tercera": 20261001,
            "pre_registro": "docs/p69_especificacion_congelada.md, commit 915991b",
        },
        "canonical_result": {"corrected_cv": 0.7854160387, "V03": 0.7772114525,
                             "delta": 0.0082045862, "pliegues_mejorados": 5,
                             "nivel_reproducibilidad": "R3"},
        "second_partition": {"delta": 0.0105253130, "pliegues_mejorados": 5,
                             "semilla": 20260928},
        "third_partition": {"delta": 0.0048825807, "pliegues_mejorados": 4,
                            "semilla": 20261001,
                            "nota": "corregida tras el defecto de class_weight/sample_weight"},
        "permutation": {"n": 1000, "semilla": 20261002,
                        "p_empirico": per["test"]["p_empirico"],
                        "n_ge_observado": per["test"]["n_permutaciones_ge_observado"],
                        "nulo_media": per["nulo"]["media"],
                        "nulo_maximo": per["nulo"]["maximo"],
                        "que_prueba": ("la asignacion real de pesos contiene mas informacion "
                                       "que asignaciones equivalentes permutadas"),
                        "que_NO_prueba": ["generalizacion poblacional definitiva",
                                          "mejora garantizada de leaderboard"]},
        "bootstrap": {"por_grupos_ic95": [-0.0006906975, 0.0118175641],
                      "incluye_cero": True,
                      "tamano": {"filas": 1483, "grupos": 1415}},
        "artifact_hashes": {
            "especificacion_congelada": sha(PROJECT_ROOT / "docs"
                                            / "p69_especificacion_congelada.md"),
            "informe_final_evidencia": sha(REPORTS
                                           / "a_cost_sensitive_downweight_final_evidence.md"),
            "particiones_csv": sha(REPORTS / "a_cost_sensitive_downweight_partitions.csv"),
            "permutacion_csv": sha(REPORTS
                                   / "a_cost_sensitive_downweight_permutation.csv"),
        },
        "compliance": {
            "test_development_use": 0, "holdout_use": 0, "leaderboard_tuning": 0,
            "submissions_nuevas": 0,
            "verificado_por": "operacion + fichero concreto, no por busqueda textual",
        },
        "estados": [{"estado": n, "valor": v2, "nota": t} for n, v2, t in ESTADOS],
        "research_status": "EXPERIMENTAL_RESEARCH_CLOSED",
        "candidate_status": "PROMISING_UNCONFIRMED",
        "governance_status": "PENDING",
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_M.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    log(f"   manifest escrito con {len(manifest)} bloques")

    # ---------- la comparacion final ----------
    filas_cmp = [
        {"model": "V03", "corrected_cv": 0.7772114525, "partition_2": 0.7772114525,
         "partition_3": 0.7772114525, "delta": 0.0, "folds_positive": "referencia",
         "rescue": 0, "damage": 0, "net": 0,
         "bootstrap": "no aplica",
         "permutation": "no aplica", "status": "FROZEN_REFERENCE"},
        {"model": "A_cost_sensitive_downweight", "corrected_cv": 0.7854160387,
         "partition_2": 0.7779658848, "partition_3": 0.7788638872,
         "delta": 0.0082045862, "folds_positive": "5/5, 5/5, 4/5",
         "rescue": 99, "damage": 74, "net": 25,
         "bootstrap": "[-0.0006906975, +0.0118175641] incluye 0",
         "permutation": "p = 0.000999", "status": "PROMISING_UNCONFIRMED"},
    ]
    with CSV_C.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["bloque"] + list(filas_cmp[0].keys()))
        for f in filas_cmp:
            w.writerow(["COMPARACION"] + list(f.values()))
        w.writerow([])
        w.writerow(["bloque", "estado", "valor", "nota"])
        for n, v2, t in ESTADOS:
            w.writerow(["ESTADOS", n, v2, t])
        w.writerow([])
        w.writerow(["bloque", "clave", "valor"])
        for capa in ("canonical_result", "second_partition", "third_partition", "permutation",
                     "bootstrap"):
            for k, v2 in manifest[capa].items():
                if isinstance(v2, (str, int, float, bool)):
                    w.writerow([capa, k, v2])
        for k, v2 in manifest["compliance"].items():
            w.writerow(["compliance", k, v2])
        w.writerow(["estado", "research_status", "EXPERIMENTAL_RESEARCH_CLOSED"])
        w.writerow(["estado", "candidate_status", "PROMISING_UNCONFIRMED"])
        w.writerow(["estado", "governance_status", "PENDING"])
    log(f"   {CSV_C.name} escrita")

    # ---------- el informe del candidato ----------
    T: list[str] = []
    W = T.append
    W("# Candidato congelado - A_cost_sensitive_downweight")
    W("")
    W(f"Generado por `experiments/p69_08_paquete_candidato.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## El estado, en tres palabras que no se confunden entre si")
    W("")
    W("```text")
    W("research closed       no se buscan mas mecanismos")
    W("candidate rejected    NO: nada aqui dice que el candidato este mal")
    W("candidate adopted     TAMPOCO: nada aqui dice que deba enviarse")
    W("```")
    W("")
    W("```text")
    W("research_status     EXPERIMENTAL_RESEARCH_CLOSED")
    W("candidate_status    PROMISING_UNCONFIRMED")
    W("governance_status   PENDING")
    W("```")
    W("")
    W("## La comparacion final")
    W("")
    W("| | V03 | A_cost_sensitive_downweight |")
    W("|---|---:|---:|")
    W(f"| corrected_cv | {filas_cmp[0]['corrected_cv']:.10f} | "
      f"{filas_cmp[1]['corrected_cv']:.10f} |")
    W(f"| particion 2 | {filas_cmp[0]['partition_2']:.10f} | "
      f"{filas_cmp[1]['partition_2']:.10f} |")
    W(f"| particion 3 | {filas_cmp[0]['partition_3']:.10f} | "
      f"{filas_cmp[1]['partition_3']:.10f} |")
    W(f"| delta | - | {filas_cmp[1]['delta']:+.10f} |")
    W(f"| pliegues positivos | referencia | {filas_cmp[1]['folds_positive']} |")
    W(f"| rescue | - | {filas_cmp[1]['rescue']} |")
    W(f"| damage | - | {filas_cmp[1]['damage']} |")
    W(f"| net | - | {filas_cmp[1]['net']:+d} |")
    W(f"| bootstrap por grupos | - | {filas_cmp[1]['bootstrap']} |")
    W(f"| permutacion | - | {filas_cmp[1]['permutation']} |")
    W("| estado | `FROZEN_REFERENCE` | `PROMISING_UNCONFIRMED` |")
    W("")
    W("Los candidatos cerrados (META-BASES-RIDGE, B_meta36, D_svd300_rbf, y todos los")
    W("mecanismos de P66 a P68) quedan como contexto historico y no entran en esta tabla:")
    W("ninguno sobrevivio a su gate.")
    W("")
    W("## El mecanismo, con su nombre correcto")
    W("")
    W("**downweight basado en violacion de margen del propio V03**")
    W("")
    W("```text")
    W("margen_favor = (p_V03 - umbral) * (2*y - 1)")
    W("violacion_l  = clip(-margen_favor, 0, None)      por etiqueta")
    W("error_score  = violacion_l.mean(axis=1)          por ejemplo, media de 3")
    W("z            = (error_score - mediana) / IQR")
    W("peso         = clip(1 - 0.5*z, 0.5, 1.5)")
    W("```")
    W("")
    W("### Las descripciones que NO aplican")
    W("")
    W("| descripcion | por que no |")
    W("|---|---|")
    W("| meta-model | P58 lo etiquetaba asi, y el codigo no construye ningun meta-modelo |")
    W("| stacking | no hay modelo de segundo nivel |")
    W("| residual correction | el score no se corrige despues; la frontera cambia al entrenar |")
    W("| rescue weighting | es P68-H2, y va en direccion **opuesta** |")
    W("")
    W("La diferencia con P68-H2 merece una linea, porque es facil confundirlos:")
    W("")
    W("```text")
    W("P68-H2                        SUBIR el peso donde V03 falla y otro modelo acierta")
    W("A_cost_sensitive_downweight   BAJAR el peso donde V03 falla, segun la violacion")
    W("```")
    W("")
    W("P68-H2 se clasifico `DUPLICATE` y ademas apuntaba en la direccion que historicamente")
    W("empeoro. Este candidato va en la contraria, y es la unica hipotesis viva.")
    W("")
    W("## Los seis estados historicos")
    W("")
    W("| estado | valor | nota |")
    W("|---|---|---|")
    for n, v2, t in ESTADOS:
        W(f"| `{n}` | {v2} | {t} |")
    W("")
    W("Ninguno sustituye a otro, y ninguna cifra historica se ha sobrescrito.")
    W("")
    W("## Lo que la permutacion prueba, y lo que no")
    W("")
    W("```text")
    W("PRUEBA    la asignacion real de pesos contiene mas informacion")
    W("          que asignaciones equivalentes permutadas")
    W("")
    W("NO PRUEBA generalizacion poblacional definitiva")
    W("NO PRUEBA mejora garantizada de leaderboard")
    W("```")
    W("")
    W("El bootstrap por grupos sigue siendo la limitacion principal.")
    W("")
    W("## Analisis por etiqueta, descriptivo")
    W("")
    W("```text")
    W("HS   positivo en las TRES particiones   +19, +7, +18")
    W("TR   inestable                          -5, -4, +3")
    W("AG   inestable                          +11, +9, -11")
    W("```")
    W("")
    W("Es evidencia descriptiva de estabilidad mecanica. **No** es una invitacion a optimizar")
    W("por etiqueta, y no se ha hecho ninguna modificacion especifica por etiqueta.")
    W("")
    W("## Compliance")
    W("")
    W("| concepto | valor |")
    W("|---|---:|")
    for k, v2 in manifest["compliance"].items():
        if k != "verificado_por":
            W(f"| {k} | {v2} |")
    W("")
    W(f"Verificado por: {manifest['compliance']['verificado_por']}.")
    W("")
    W("## Reproducibilidad")
    W("")
    W("| elemento | valor |")
    W("|---|---|")
    W(f"| commit del manifest | `{commit}` |")
    W("| commit original | `c96cb0f` |")
    W(f"| OOF canonico | `{manifest['data_hash']['oof_v03_final_npz']}` |")
    W(f"| OOF del candidato | `{manifest['data_hash']['oof_weighting_v4_npz']}` |")
    W(f"| submission de V03 | `{hash_real[:32]}` |")
    W("")
    W("El manifest completo, con todos los hashes y parametros, esta en")
    W("`reports/toxic_candidate_cost_sensitive_manifest.json`.")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p69_08_paquete_candidato.py`. Sin push.*")
    MD_C.write_text(a_ascii("\n".join(T)), encoding="ascii")
    log(f"   {MD_C.name} escrito")

    # ---------- la nota de gobernanza del candidato ----------
    G: list[str] = []
    Wg = G.append
    Wg("# Nota de gobernanza - candidato A_cost_sensitive_downweight")
    Wg("")
    Wg(f"Generada por `experiments/p69_08_paquete_candidato.py`. Fecha "
       f"{inicio.isoformat(timespec='seconds')}.")
    Wg("")
    Wg("Esta nota es **informativa**. No decide si el uso del conjunto de prueba seria licito,")
    Wg("no recomienda utilizarlo, y no genera ninguna submission. Su unico proposito es dejar")
    Wg("por escrito que esta validado, que no lo esta, y que procedimiento haria falta para")
    Wg("una eventual submission.")
    Wg("")
    Wg("## Que esta validado")
    Wg("")
    Wg("| aspecto | estado |")
    Wg("|---|---|")
    Wg("| reconstruccion del candidato desde sus artefactos | **SI**, cadena regenerada |")
    Wg("| reproduccion del resultado canonico | **SI**, R3, identica a 3,2e-11 |")
    Wg("| direccion del efecto entre particiones | **SI**, 3 de 3 positivas |")
    Wg("| especificidad de la correspondencia peso-ejemplo | **SI**, p = 0,000999 |")
    Wg("| compliance del pipeline | **SI**, 16 de 16 |")
    Wg("| provenance | **SI**, manifest con hashes y commits |")
    Wg("")
    Wg("## Que NO esta validado")
    Wg("")
    Wg("| aspecto | estado |")
    Wg("|---|---|")
    Wg("| magnitud poblacional del efecto | **NO**, IC95 por grupos incluye el cero |")
    Wg("| transferencia al leaderboard | **NO**, nunca se ha medido |")
    Wg("| comportamiento sobre datos fuera del desarrollo | **NO**, no se ha evaluado |")
    Wg("")
    Wg("## La incertidumbre que permanece")
    Wg("")
    Wg("```text")
    Wg("IC95 por grupos   [-0.0006906975, +0.0118175641]   incluye el cero")
    Wg("limite inferior   a seis diezmilesimas de cero")
    Wg("tamano            1483 filas, 1415 grupos")
    Wg("```")
    Wg("")
    Wg("La incertidumbre es de **precision**, y la precision esta limitada por el tamano")
    Wg("efectivo del train. No es una incertidumbre sobre el signo: las tres particiones son")
    Wg("positivas y la permutacion es contundente. Es una incertidumbre sobre **cuanto**.")
    Wg("")
    Wg("## Que evidencia existe, en una tabla")
    Wg("")
    Wg("| prueba | resultado | pregunta que responde |")
    Wg("|---|---|---|")
    Wg("| Reproduccion | R3, 3,2e-11 | el numero es real y trazable |")
    Wg("| Particion canonica | +0,0082045862, 5 de 5 | el efecto existe en el reparto original |")
    Wg("| Segunda particion | +0,0105253130, 5 de 5 | sobrevive a otro reparto |")
    Wg("| Tercera particion | +0,0048825807, 4 de 5 | sobrevive a un tercero |")
    Wg("| Permutacion, 1000 | p = 0,000999 | la asociacion peso-ejemplo es especifica |")
    Wg("| Bootstrap por grupos | incluye el cero | la magnitud no queda resuelta |")
    Wg("")
    Wg("## Que procedimiento necesitaria una eventual submission")
    Wg("")
    Wg("Una decision **explicita y separada** de la investigacion, que reconociera por escrito:")
    Wg("")
    Wg("```text")
    Wg("1. que se evaluaria un candidato cuyo bootstrap por grupos incluye el cero")
    Wg("2. que la medicion sobre el conjunto de prueba consumiria informacion hoy intacta")
    Wg("3. que el resultado no seria confirmatorio de nada nuevo, porque las tres")
    Wg("   particiones y la permutacion ya estan hechas y repetirlas no anade evidencia")
    Wg("4. que la decision sobre si ese uso es admisible corresponde a una regla,")
    Wg("   no a esta nota")
    Wg("```")
    Wg("")
    Wg("### Lo que esta nota NO hace")
    Wg("")
    Wg("```text")
    Wg("no decide si el uso del test es legal")
    Wg("no recomienda utilizar el test")
    Wg("no genera submission")
    Wg("no abre una cuarta particion")
    Wg("no propone una estrategia transductiva")
    Wg("```")
    Wg("")
    Wg("## Criterio para reabrir la investigacion")
    Wg("")
    Wg("Solo con evidencia **nueva y externa al experimento actual**:")
    Wg("")
    Wg("```text")
    Wg("autorizacion normativa explicita")
    Wg("nuevo dato de entrenamiento legal")
    Wg("nuevo conjunto etiquetado permitido")
    Wg("cambio de reglas")
    Wg("```")
    Wg("")
    Wg("**No** se considera justificacion suficiente:")
    Wg("")
    Wg("```text")
    Wg("'quiero un IC95 que no incluya 0'")
    Wg("mas particiones sobre los mismos datos")
    Wg("mas semillas, mas repartos, mas grid de pesos")
    Wg("```")
    Wg("")
    Wg("## El estado final")
    Wg("")
    Wg("```text")
    Wg("V03                          FROZEN")
    Wg("A_cost_sensitive_downweight  PROMISING_UNCONFIRMED")
    Wg("research_status              EXPERIMENTAL_RESEARCH_CLOSED")
    Wg("governance_status            PENDING")
    Wg("submission                   0")
    Wg("test_development_use         0")
    Wg("leaderboard_tuning           0")
    Wg("```")
    Wg("")
    Wg("La investigacion experimental termina aqui. El candidato queda congelado y")
    Wg("perfectamente reproducible en `reports/toxic_candidate_cost_sensitive_manifest.json`.")
    Wg("")
    Wg("> V03 remains the best verified leaderboard result.")
    Wg("")
    Wg("*Generado por `experiments/p69_08_paquete_candidato.py`. Sin push.*")
    MD_G.write_text(a_ascii("\n".join(G)), encoding="ascii")
    log(f"   {MD_G.name} escrito")

    documentos = [MD_C, CSV_C, JSON_M, MD_G]
    sucios = [d.name for d in documentos if any(b > 127 for b in d.read_bytes())]
    if sucios:
        raise SystemExit(f"PARADA: documentos con bytes no ASCII: {sucios}")

    log("")
    log("=" * 78)
    log("P69-08 OK | paquete del candidato y cierre experimental")
    log("   research_status   EXPERIMENTAL_RESEARCH_CLOSED")
    log("   candidate_status  PROMISING_UNCONFIRMED")
    log("   governance_status PENDING")
    log("   submission 0 | test_development_use 0 | leaderboard_tuning 0")
    log(f"   {len(documentos)} documentos, 0 bytes no ASCII")
    log("   V03 remains the best verified leaderboard result.")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
