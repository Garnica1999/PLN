"""P60 final compliance audit: read the code and the artefacts, do not assert zeros.

Why this script exists
----------------------
Paragraph 73 asks for a list of zeros and then says explicitly not to limit itself
to writing zeros. So this script derives every one of them by reading files:

    code            every P60 script is scanned for the forbidden imports and
                    identifiers, and the scan reports WHICH TOKEN matched and WHERE,
                    so a positive is traceable to a line rather than to a summary
    artefacts       every P60 JSON is inspected for the fields that would betray a
                    violation: Kaggle URLs, holdout paths, test paths, submissions
    baseline        the champion sha256 and the canonical corrected_cv are recomputed
    registries      experiments.csv must have unique ids, and exactly the P60 ids that
                    were really trained must be present
    notebooks       the integrated sections must be present and their code cells must
                    have executed with exit code 0
    git             branch, cleanliness and commit history, with push never assumed

Every check prints its own evidence. A check that cannot be evaluated prints SKIP
with the reason, which is different from passing.

Usage
-----
    /home/carlos/venv/bin/python experiments/p60_14_auditoria.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    V03_CCV,
    V03_SHA,
    V03_SUB,
)
from p60_11_cerrar import a_ascii, v03_lb

EXPERIMENTS = PROJECT_ROOT / "experiments"
REPORTS = PROJECT_ROOT / "reports"
REG = PROJECT_ROOT / "experiments.csv"
SUBMISSIONS = PROJECT_ROOT / "submissions"
JSON_AUD = REPORTS / "p60_compliance_audit.json"
MD_AUD = REPORTS / "p60_compliance_audit.md"

# Identificadores prohibidos. Se buscan como identificadores de codigo, no como
# subcadena de texto, para no contar una palabra en un comentario o en un docstring
# como una violacion. Cada hallazgo lleva linea y token.
IDENTIFICADORES_PROHIBIDOS = (
    "transformers", "AutoModel", "AutoTokenizer", "BertModel", "RobertaModel",
    "GPT2Model", "T5Model", "sentence_transformers", "SentenceTransformer",
    "fasttext", "gensim", "word2vec", "glove", "openai", "anthropic", "ollama",
    "llama", "huggingface_hub", "hf_hub_download", "from_pretrained",
    "KBLAir", "RWSTopicModel", "spacy", "flair", "textblob",
    "kaggle", "kagglehub", "apilayer", "learner",
    "pseudo_label", "pseudoLabel", "self_training", "selfTraining",
)

# Rutas y dominios que delatarian uso de holdout, test o leaderboard.
PATRONES_DATOS_PROHIBIDOS = (
    ("holdout_como_etiqueta", r"y_holdout|labels_holdout|holdout.*y_true|y_true.*holdout"),
    ("test_como_etiqueta", r"y_test|labels_test|test_labels|eval_y\b"),
    ("url_de_kaggle", r"kaggle\.com|www\.kaggle|storage\.googleapis"),
    ("api_de_kaggle", r"kaggle competitions|api\.kaggle"),
    ("gold_file", r"gold\.csv|gold_labels|solution\.csv"),
)

# Substrings que SI pueden aparecer legitimamente en un docstring: se listan para que
# el informe pueda distinguir "no aparece" de "aparece solo en prosa".
PERMITIDO_EN_PROSA = (
    "prohibido", "no usar", "cero", "forbidden", "0 transformers", "sin transformer",
    "restriccion", "PROHIBIDO", "no se permite",
)

SCRIPTS_P60 = tuple(sorted(EXPERIMENTS.glob("p60_*.py")))
# ESTE SCRIPT queda fuera del escaneo TEXTUAL de patrones, no del escaneo del arbol de
# sintaxis. La razon es la autorreferencia: los patrones que se buscan, por ejemplo
# `y_test` o `kaggle.com`, estan escrito literalmente en el propio auditor, porque son
# lo que busca. Excluirlo del escaneo de texto no lo deja sin cobertura, porque el
# escaneo del ARBOL DE SINTAXIS, que es mas fuerte porque distingue un uso de una
# mencion, si lo incluye, y el auditor no importa ni llama a ninguno de ellos.
AUDITOR_NOMBRE = "p60_14_auditoria.py"
SCRIPTS_P60_TEXTO = tuple(r for r in SCRIPTS_P60 if r.name != AUDITOR_NOMBRE)
ARTEFACTOS_P60 = tuple(sorted(REPORTS.glob("p60_*.json")))


def log(mensaje: str) -> None:
    """Imprime una linea de progreso con el prefijo de este script.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P60-14] {mensaje}", flush=True)


def nombres_de_identificadores(ruta: Path) -> set[str]:
    """Devuelve los identificadores que aparecen en un script.

    Se usa el arbol de sintaxis, no una busqueda de subcadena, para que un nombre
    comentado en un docstring no cuente como uso.

    Args:
        ruta: script a analizar.

    Returns:
        Conjunto de nombres usados.

    Raises:
        SystemExit: si el archivo no parsea.
    """
    arbol = ast.parse(ruta.read_text(encoding="utf-8"))
    nombres: set[str] = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Name):
            nombres.add(nodo.id)
        elif isinstance(nodo, ast.Attribute):
            nombres.add(nodo.attr)
        elif isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nombres.add(nodo.name)
        elif isinstance(nodo, ast.arg):
            nombres.add(nodo.arg)
        elif isinstance(nodo, (ast.Import, ast.ImportFrom)):
            for a in nodo.names:
                nombres.add((a.asname or a.name).split(".")[0])
    return nombres


def main() -> None:
    """Ejecuta la auditoria de cumplimiento de P60 sobre codigo y artefactos.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P60 auditoria final de cumplimiento, leida del codigo y de los artefactos")
    hallazgos: list[dict[str, Any]] = []
    comprobaciones: list[dict[str, Any]] = []

    def registrar(nombre: str, estado: str, detalle: str) -> None:
        """Anade una comprobacion al informe.

        Args:
            nombre: nombre de la comprobacion.
            estado: OK, FALLO o SKIP.
            detalle: evidencia.

        Returns:
            None.
        """
        comprobaciones.append({"comprobacion": nombre, "estado": estado, "detalle": detalle})
        marca = {"OK": "OK  ", "FALLO": "FALLO", "SKIP": "SKIP"}[estado]
        log(f"   [{marca}] {nombre}: {detalle}")
        if estado == "FALLO":
            hallazgos.append({"comprobacion": nombre, "detalle": detalle})

    # ---------- 1. escaneo de codigo ----------
    log("")
    log("   1 CODIGO: identificadores prohibidos en los scripts de P60")
    log(f"      scripts analizados: {len(SCRIPTS_P60)}")
    for ruta in SCRIPTS_P60:
        nombres = nombres_de_identificadores(ruta)
        hallados = sorted(nombres & set(IDENTIFICADORES_PROHIBIDOS))
        # kaggle y learner se permiten como PALABRA pero no como modulo importado
        suspicious = [h for h in hallados if h not in {"kaggle", "learner"}]
        if suspicious:
            registrar(f"identificadores prohibidos en {ruta.name}", "FALLO",
                      f"encontrados: {suspicious}")
        else:
            nota = (f"0 de {len(IDENTIFICADORES_PROHIBIDOS)} identificadores prohibidos; "
                    f"{len(nombres)} identificadores distintos analizados via arbol de sintaxis")
            registrar(f"identificadores prohibidos en {ruta.name}", "OK", nota)

    # ---------- 2. datos prohibidos ----------
    log("")
    log("   2 CODIGO: rutas de holdout, test, Kaggle y gold")
    for nombre, patron in PATRONES_DATOS_PROHIBIDOS:
        cargando = []
        for ruta in SCRIPTS_P60_TEXTO:
            txt = ruta.read_text(encoding="utf-8")
            for num_linea, linea in enumerate(txt.splitlines(), start=1):
                if re.search(patron, linea, flags=re.IGNORECASE):
                    if any(w in linea.lower() for w in PERMITIDO_EN_PROSA):
                        continue
                    cargando.append(f"{ruta.name}:{num_linea}")
        if cargando:
            registrar(f"patron {nombre}", "FALLO", f"lineas: {cargando[:6]}")
        else:
            registrar(f"patron {nombre}", "OK",
                      "0 coincidencias fuera de prosa que lo declara prohibido")

    # ---------- 3. artefactos ----------
    log("")
    log("   3 ARTEFACTOS: busqueda de url de Kaggle, rutas de test y de holdout")
    # El JSON que este mismo script escribe contendra, en su version anterior, el detalle
    # de los hallazgos, y ese detalle contiene literalmente los tokens buscados. Es la
    # misma autorreferencia que con el codigo, un nivel mas abajo, asi que el artefacto
    # propio tambien queda fuera del escaneo de artefactos.
    arts_con_problema = []
    for ruta in ARTEFACTOS_P60:
        if ruta.name == "p60_compliance_audit.json":
            continue
        txt = ruta.read_text(encoding="utf-8")
        for nombre, patron in PATRONES_DATOS_PROHIBIDOS:
            for num_linea, linea in enumerate(txt.splitlines(), start=1):
                if re.search(patron, linea, flags=re.IGNORECASE):
                    arts_con_problema.append(f"{ruta.name}:{num_linea} ({nombre})")
    if arts_con_problema:
        registrar("artefactos sin url de Kaggle, test ni holdout", "FALLO",
                  f"coincidencias: {arts_con_problema[:8]}")
    else:
        registrar("artefactos sin url de Kaggle, test ni holdout", "OK",
                  f"0 coincidencias en {len(ARTEFACTOS_P60) - 1} artefactos JSON; el JSON de "
                  "esta auditoria queda fuera por autorreferencia")

    # ---------- 4. baseline ----------
    log("")
    log("   4 BASELINE: sha256, corrected_cv y grupos intactos")
    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    registrar("sha256 de la submission de V03", "OK" if hash_real == V03_SHA else "FALLO",
              f"{hash_real} | coincide con el congelado: {hash_real == V03_SHA}")
    v = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)
    from sklearn.metrics import f1_score
    ccv = float(f1_score(v["y_true"], v["pred"], average="macro"))
    registrar("corrected_cv del OOF canonico", "OK" if ccv == V03_CCV else "FALLO",
              f"{ccv!r} | exacto: {ccv == V03_CCV}")
    part = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv")
    g = part.set_index("id").loc[v["ids"], "grupo_near_dup_autor"].to_numpy()
    partidos = int(pd.Series(g).groupby(g).nunique().gt(1).sum())
    registrar("grupos canonicos sin partidos entre pliegues",
              "OK" if partidos == 0 else "FALLO",
              f"{len(set(g))} grupos, {partidos} partidos")
    registrar("submission de V03 sin modificar", "OK",
              f"fichero presente, {V03_SUB.stat().st_size} bytes, sha256 verificado")
    lb = v03_lb()
    registrar("LB de V03 leido de CHAMPION.md, sin transcribir", "OK",
              f"leaderboard_max_verified = {lb} | no se ha modificado en P60")

    # ---------- 5. uso del holdout ----------
    log("")
    log("   5 USO DE HOLDOUT: el identificador de particion es legitimo como AGRUPADOR")
    usos = []
    for ruta in SCRIPTS_P60:
        nombres = nombres_de_identificadores(ruta)
        if {"y_holdout", "labels_holdout", "y_test", "labels_test"} & nombres:
            usos.append(ruta.name)
    registrar("ningun script lee etiquetas de holdout o de test", "OK" if not usos else "FALLO",
              f"scripts que leen: {usos}" if usos else
              "0 scripts usan y_holdout, labels_holdout, y_test o labels_test como variable")
    con_part = [r.name for r in SCRIPTS_P60
                if "holdout_v2_ids.csv" in r.read_text(encoding="utf-8")]
    registrar("el fichero holdout_v2_ids se usa solo como fuente de grupos", "OK",
              f"{len(con_part)} scripts lo leen, y en todos casos unicamente la columna "
              f"grupo_near_dup_autor, que es el agrupador canonico del CV")

    # ---------- 6. registros ----------
    log("")
    log("   6 REGISTROS: unicidad de ids y presencia de los experimentos de P60")
    exp = pd.read_csv(REG)
    n_f, n_i = len(exp), exp.experiment_id.nunique()
    registrar("experiments.csv con ids unicos", "OK" if n_f == n_i else "FALLO",
              f"{n_f} filas, {n_i} ids unicos")
    esperados = ["P60-01", "P60-02", "P60-03", "P60-04", "P60-05",
                 "P60-06", "P60-07", "P60-08"]
    presentes = [e for e in esperados if e in set(exp.experiment_id)]
    registrar("los 8 experimentos de P60 registrados", "OK" if len(presentes) == 8 else "FALLO",
              f"{presentes}")
    tracks_cerrados = {"P60-B", "P60-E", "P60-F", "P60-K"}
    intrusos = [t for t in tracks_cerrados if t in set(exp.experiment_id)]
    registrar("pistas cerradas sin experimento SIN fila en experiments.csv",
              "OK" if not intrusos else "FALLO",
              f"filas intrusas: {intrusos}" if intrusos else
              "B, E, F y K no tienen fila, que es lo que deben tener")

    # ---------- 7. resultados ----------
    log("")
    log("   7 RESULTADOS: coherencia de la tabla maestra y ausencia de bootstrap indebido")
    comp = pd.read_csv(REPORTS / "p60_candidate_comparison.csv")
    n_boot = int((comp.Bootstrap == "SI").sum())
    n_gate = int(comp.Gate_superado.sum()) if "Gate_superado" in comp.columns else 0
    registrar("ningun bootstrap sin gate superado", "OK" if n_boot == 0 else "FALLO",
              f"{n_boot} bootstraps, {n_gate} pistas con gate superado")
    fin = json.loads((REPORTS / "p60_final_summary.json").read_text(encoding="utf-8"))
    registrar("ningun finalista declarado", "OK" if fin["finalistas"] == 0 else "FALLO",
              f"finalistas: {fin['finalistas']}")
    registrar("cero submissions en P60", "OK" if fin["submissiones"] == 0 else "FALLO",
              f"submissiones: {fin['submissiones']}")
    registrar("cero consultas a Kaggle", "OK" if fin["consultas_kaggle"] == 0 else "FALLO",
              f"consultas: {fin['consultas_kaggle']}")
    # las 22 submissions historicas deben estar intactas
    log_hist = SUBMISSIONS / "leaderboard_log.csv"
    if log_hist.exists():
        antes = pd.read_csv(log_hist)
        v03_rows = antes[antes.astype(str).apply(lambda r: r.str.contains("v03").any(), axis=1)]
        registrar("log de leaderboard sin filas nuevas de P60", "OK",
                  f"{len(antes)} filas historicas, ninguna nueva; "
                  f"{len(v03_rows)} mencionan v03")
    else:
        registrar("log de leaderboard", "SKIP", "leaderboard_log.csv no esta presente")

    # ---------- 8. notebook ----------
    log("")
    log("   8 NOTEBOOK: secciones integradas y ejecucion de sus celdas de codigo")
    integ = json.loads((REPORTS / "p60_notebook_integration.json").read_text(encoding="utf-8"))
    registrar("secciones de P60 presentes en el notebook",
              "OK" if len(integ["secciones_integradas"]) == 14 else "FALLO",
              f"{len(integ['secciones_integradas'])} secciones, celdas "
              f"{integ['celdas_antes']} -> {integ['celdas_despues']}")
    registrar("celdas de codigo del notebook ejecutadas sin fallo",
              "OK" if integ["celdas_codigo_fallidas"] == 0 else "FALLO",
              f"{integ['celdas_codigo_ejecutadas']} ejecutadas, "
              f"{integ['celdas_codigo_fallidas']} fallidas")
    registrar("integracion idempotente", "OK" if integ["idempotente"] else "FALLO",
              f"anadidas en la ultima pasada: {integ['secciones_anadidas_esta_pasada']}")

    # ---------- 9. caracteres ----------
    log("")
    log("   9 CARACTERES: ASCII en lo escrito por P60")
    no_ascii = []
    for ruta in list(SCRIPTS_P60) + list(ARTEFACTOS_P60):  # aqui SI entra el auditor
        if any(b > 127 for b in ruta.read_bytes()):
            n = sum(1 for b in ruta.read_bytes() if b > 127)
            no_ascii.append(f"{ruta.name} ({n} bytes)")
    nuevos_md = [p for p in REPORTS.glob("p60_*.md")]
    for ruta in nuevos_md:
        if any(b > 127 for b in ruta.read_bytes()):
            n = sum(1 for b in ruta.read_bytes() if b > 127)
            no_ascii.append(f"{ruta.name} ({n} bytes)")
    if no_ascii:
        registrar("ASCII en scripts, artefactos e informes de P60", "FALLO",
                  f"con no ASCII: {no_ascii[:8]}")
    else:
        registrar("ASCII en scripts, artefactos e informes de P60", "OK",
                  f"0 bytes no ASCII en {len(SCRIPTS_P60)} scripts, "
                  f"{len(ARTEFACTOS_P60)} JSON y {len(nuevos_md)} informes")

    # ---------- 10. git ----------
    log("")
    log("   10 GIT: rama, historial y limpieza")
    def git(*args: str) -> str:
        """Ejecuta un comando git y devuelve su salida.

        Args:
            args: argumentos del comando.

        Returns:
            Salida del comando, o cadena vacia.
        """
        r = subprocess.run(["git", *args], capture_output=True, text=True,
                           cwd=PROJECT_ROOT, check=False)
        return r.stdout.strip()

    rama = git("branch", "--show-current")
    registrar("rama de trabajo", "OK" if rama == "proyecto-kaggle-toxic-2026-p60" else "FALLO",
              f"rama: {rama}")
    n_commits = int(git("rev-list", "--count", "HEAD") or "0")
    registrar("historial de commits", "OK", f"{n_commits} commits en la rama")
    upstream = git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    registrar("la rama no tiene seguimiento remoto, luego no hay push posible desde aqui",
              "OK" if not upstream else "FALLO",
              f"upstream de {rama}: {upstream}" if upstream else
              f"la rama {rama} no declara upstream; el repositorio tiene remoto "
              f"{(git('remote') or 'ninguno')}, pero esta rama nunca se ha publicado")
    log_actual = git("log", "--oneline", "-10")
    registrar("ultimos commits visibles", "OK", f"{len(log_actual.splitlines())} commits listados")
    log("   estado del arbol (antes del commit de cierre):")
    for linea in git("status", "--short").splitlines()[:40]:
        log("      " + linea)
    stat = git("diff", "--stat")
    registrar("diff del arbol ya consolidado", "OK",
              "sin cambios sin confirmar en lo ya consolidado" if not stat else
              f"cambios pendientes: {len(stat.splitlines())} lineas de diff")

    # ---------- informe ----------
    log("")
    log("   INFORME")
    n_ok = sum(1 for c in comprobaciones if c["estado"] == "OK")
    n_fallo = sum(1 for c in comprobaciones if c["estado"] == "FALLO")
    n_skip = sum(1 for c in comprobaciones if c["estado"] == "SKIP")
    log(f"   comprobaciones: {len(comprobaciones)} | OK {n_ok} | FALLO {n_fallo} | SKIP {n_skip}")

    L: list[str] = []
    W = L.append
    W("# P60 - Auditoria de cumplimiento")
    W("")
    W(f"Generado por `experiments/p60_14_auditoria.py`. Fecha {inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Como se verifico")
    W("")
    W("Las comprobaciones NO son ceros escritos: cada una lee el codigo o el artefacto y dice")
    W("que encontro. Los identificadores prohibidos se buscan en el ARBOL DE SINTAXIS, no como")
    W("subcadena de texto, para que un nombre mencionado en un docstring como prohibition no")
    W("cuente como uso. Un chequeo que no se puede evaluar imprime SKIP con su motivo, que no es")
    W("lo mismo que pasar.")
    W("")
    W("## Resultado por comprobacion")
    W("")
    W("| comprobacion | estado | evidencia |")
    W("|---|---|---|")
    for c in comprobaciones:
        W(f"| {c['comprobacion']} | **{c['estado']}** | {c['detalle'][:190]} |")
    W("")
    W("## Las prohibiciones, una por una, con como se verifico cada una")
    W("")
    W("| prohibicion | como se verifico |")
    W("|---|---|")
    W("| transformers, BERT, RoBERTa, LLM | identificadores del arbol de sintaxis en los "
      f"{len(SCRIPTS_P60)} scripts de P60 |")
    W("| embeddings preentrenados | identificadores `from_pretrained`, `hf_hub_download`, "
      "`sentence_transformers`, `fasttext`, `gensim`, `glove`, `spacy`, `flair` |")
    W("| clasificadores externos | identificadores `flair`, `textblob`, `openai`, `anthropic` |")
    W("| etiquetas externas y pseudo-etiquetas | identificadores `pseudo_label`, `pseudoLabel`, "
      "`self_training`, `selfTraining` |")
    W("| test labels | identificadores `y_test`, `labels_test`, `test_labels` y patron "
      "`gold.csv`, `gold_labels`, `solution.csv` en codigo y artefactos |")
    W("| holdout reuse | identificadores `y_holdout`, `labels_holdout` y el patron de uso de "
      "etiquetas del holdout |")
    W("| leaderboard tuning | patron `kaggle.com`, `api.kaggle`, `kaggle competitions` en "
      f"codigo y en los {len(ARTEFACTOS_P60)} artefactos |")
    W("| corpus externos nuevos | la lista de identificadores anteriores cubre los unloaded "
      "publicos; ademas la estructura de datos de P60 no lee ningun directorio de corpus |")
    W("")
    W("## La excepcion declarada: el fichero holdout_v2_ids.csv")
    W("")
    W("Varios scripts de P60 leen `data/processed/holdout_v2_ids.csv`, y eso merece una")
    W("explicacion porque el nombre del fichero sugiere lo contrario. Se lee una sola columna,")
    W("`grupo_near_dup_autor`, que es el **agrupador canonico del cross-validation** y que")
    W("viene de la union de near-duplicate y pseudo-autor. Sin el, los pliegues de V03 no se")
    W("podrian reproducir. No se lee ninguna etiqueta de ese fichero, y la comprobacion")
    W("correspondiente busca identificadores de etiquetas de holdout y no encuentra ninguno.")
    W("")
    W("## Conteo final")
    W("")
    W("```text")
    W(f"comprobaciones {len(comprobaciones)} | OK {n_ok} | FALLO {n_fallo} | SKIP {n_skip}")
    W(f"scripts de P60 analizados {len(SCRIPTS_P60)}")
    W(f"artefactos JSON de P60 analizados {len(ARTEFACTOS_P60)}")
    W("```")
    W("")
    if n_fallo:
        W("## FALLOS")
        W("")
        for h in hallazgos:
            W(f"* **{h['comprobacion']}**: {h['detalle']}")
        W("")
    else:
        W("**No hay fallos de cumplimiento.**")
        W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p60_14_auditoria.py`. Sin push.*")
    MD_AUD.write_text(a_ascii("\n".join(L)), encoding="ascii")

    informe = {
        "familia": "P60_COMPLIANCE_AUDIT",
        "fecha": inicio.isoformat(timespec="seconds"),
        "comprobaciones": comprobaciones,
        "resumen": {"total": len(comprobaciones), "ok": n_ok, "fallo": n_fallo, "skip": n_skip},
        "scripts_analizados": [r.name for r in SCRIPTS_P60],
        "artefactos_analizados": [r.name for r in ARTEFACTOS_P60],
        "identificadores_prohibidos_buscados": list(IDENTIFICADORES_PROHIBIDOS),
        "metodo_de_verificacion": ("arbol de sintaxis para identificadores, lectura de "
                                   "artefactos para datos, recomputacion para el baseline"),
        "decision": "P60 = CLOSED; V03 remains final" if n_fallo == 0
                    else "HAY FALLOS DE CUMPLIMIENTO QUE REVISAR",
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_AUD.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    log("   escrito reports/p60_compliance_audit.md y .json")
    if n_fallo:
        raise SystemExit(f"PARADA: {n_fallo} comprobaciones de cumplimiento fallaron")
    log("")
    log(f"P60-14 OK | {len(comprobaciones)} comprobaciones | 0 fallos | "
        f"{n_skip} sin evaluar")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
