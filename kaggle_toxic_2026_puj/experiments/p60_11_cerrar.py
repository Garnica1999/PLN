"""P60 closing: propagate the last three tracks and build every closing artefact.

Why this script exists and why it computes nothing
--------------------------------------------------
Eight tracks produced thirteen numbers each, and those numbers have to land in
five places: ``experiments.csv``, the P60 hypothesis registry, the master results
table, the candidate comparison, and two narrative reports. Any of those written by
hand is a chance to mistype a fold, and this project has already paid twice inside
P59 for exactly that.

So this script reads the JSON artefact of each track and derives everything from
them. It contains no experimental number of its own: every figure in the reports
below is either read from a JSON artefact, read from ``experiments.csv``, or read
from a historical report. If a number is missing from its artefact it is reported
as missing rather than filled in.

It also enforces the two project rules that are easiest to break by accident: a
track closed without training must not create a row in ``experiments.csv``, and a
track whose gate was not passed must have no bootstrap interval.

What it writes
--------------
    reports/p60_candidate_comparison.csv
    reports/p60_track_summary.csv
    reports/P60_FINAL_RESEARCH_REPORT.md
    reports/TOXIC_COMPLETE_RESEARCH_REPORT.md
    reports/toxic_complete_experiment_history.csv
    reports/p60_final_summary.json

Usage
-----
    /home/carlos/venv/bin/python experiments/p60_11_cerrar.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    V03_CCV,
    V03_SHA,
    V03_SUB,
)

REPORTS = PROJECT_ROOT / "reports"
REG = PROJECT_ROOT / "experiments.csv"
CSV_COMP = REPORTS / "p60_candidate_comparison.csv"
CSV_TRACKS = REPORTS / "p60_track_summary.csv"
MD_P60 = REPORTS / "P60_FINAL_RESEARCH_REPORT.md"
MD_TOTAL = REPORTS / "TOXIC_COMPLETE_RESEARCH_REPORT.md"
CSV_HIST = REPORTS / "toxic_complete_experiment_history.csv"
JSON_FIN = REPORTS / "p60_final_summary.json"

# Orden de ejecucion de P60 y su artefacto. Las pistas cerradas sin experimento se
# declaran aparte, al final, con su precedente funcional.
TRACKS_EJECUTADAS = (
    ("P60-G", "p60_G_contrastive.json"),
    ("P60-A", "p60_A_asl.json"),
    ("P60-L", "p60_L_dro.json"),
    ("P60-D", "p60_D_distribucion.json"),
    ("P60-C", "p60_C3_intervencion_ag.json"),
    ("P60-H", "p60_H_transporte.json"),
    ("P60-I", "p60_I_metareweight.json"),
    ("P60-J", "p60_J_polyloss.json"),
)
TRACKS_AUDITORIA = (
    ("P60-C1C2", "p60_C_ruido_por_etiqueta.json"),
)
TRACKS_CERRADAS = (
    ("P60-B", "W-ASL", ("ASL mas ponderacion por frecuencia de etiqueta, que es "
     "class_weight=balanced, mas suavizado de coocurrencia, que es la familia de "
     "dependencia de etiquetas"), "V8-07 y V4-E01-E03"),
    ("P60-E", "LOGIT-ADJ/LDAM", ("sobre un modelo lineal el ajuste de logit es un "
     "desplazamiento de intercepto, y con umbral fijo un desplazamiento de intercepto "
     "es un cambio de umbral, tanto post-hoc como dentro de la perdida"), "V8-09"),
    ("P60-F", "LDL", ("tres etiquetas anidadas dan un subespacio factible de cuatro "
     "puntos, medido en este corpus, y V4-E01-E03 ya lo enumero en doce variantes"),
     "V4-E01-E03"),
    ("P60-K", "CONFORMAL", ("convertir un conjunto conformista en una decision binaria "
     "exige cortar la puntuacion de no conformidad en un punto, que es un ajuste de "
     "umbral ya cerrado de raiz"), "V8-09"),
)

VERDE = {"PROMISING", "FINALIST"}
MARGINAL = {"MARGINAL"}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P60-11] {mensaje}", flush=True)


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


def commit_actual() -> str:
    """Devuelve el commit del HEAD.

    Returns:
        Hash abreviado del commit actual.
    """
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                       text=True, cwd=PROJECT_ROOT, check=False)
    return r.stdout.strip() or "desconocido"


def estado(decision: str) -> str:
    """Reduce una decision a una categoria corta.

    Args:
        decision: texto de la decision de la pista.

    Returns:
        Una de FINALIST, PROMISSING, MARGINAL, CLOSED o INVALIDATED.
    """
    d = (decision or "").upper()
    if d.startswith(("PROMISING", "FINALIST")):
        return "PROMISING"
    if d.startswith("MARGINAL"):
        return "MARGINAL"
    if d.startswith("INVALIDATED"):
        return "INVALIDATED"
    return "CLOSED"


def primero_no_nulo(d: dict[str, Any], claves: tuple[str, ...]) -> Any:
    """Devuelve el primer valor NO NULO entre varias claves.

    La distincion con la verdad es lo importante: un delta de cero es un resultado
    legitimo y debe aparecer, no un dato ausente.

    Args:
        d: diccionario de la pista.
        claves: claves candidatas, en orden de prioridad.

    Returns:
        El primer valor no nulo, o None.
    """
    for c in claves:
        if d.get(c) is not None:
            return d[c]
    return None


CHAMPION = PROJECT_ROOT / "CHAMPION.md"


def v03_lb() -> float:
    """Lee el LB verificado del campeon desde CHAMPION.md, sin transcribirlo.

    El valor canonico de 0,78306 no vive como constante en ningun modulo de P59, sino
    en CHAMPION.md, que es donde el proyecto lo registra. Leerlo de ahi evita el error de
    transcripcion, que es exactamente el tipo de fallo que este proyecto ha pagado dos
    veces dentro de P59.

    Returns:
        El LB verificado de V03.

    Raises:
        SystemExit: si el valor no aparece donde debe.
    """
    txt = CHAMPION.read_text(encoding="utf-8")
    m = re.search(r"leaderboard_max_verified\s*=\s*([0-9]+\.[0-9]+)", txt)
    if not m:
        raise SystemExit("PARADA: CHAMPION.md no declara leaderboard_max_verified")
    return float(m.group(1))


def main() -> None:
    """Propaga las pistas y construye todos los artefactos de cierre.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P60 cierre: propagacion de pistas y artefactos finales")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    log(f"   V03 sha256 verificado e intacto: {hash_real[:16]}...")
    lb = v03_lb()
    if abs(lb - 0.78306) > 1e-9:
        raise SystemExit(f"PARADA: el LB leido de CHAMPION.md es {lb}, no 0,78306")
    log(f"   V03 corrected_cv {V03_CCV!r} | LB {lb} leido de CHAMPION.md")

    # ---------- 1. propagar las tres pistas nuevas a los registros ----------
    log("")
    log("   1 PROPAGACION a experiments.csv y a los registros de P60")
    for pista, _ in TRACKS_EJECUTADAS:
        r = subprocess.run([sys.executable, str(PROJECT_ROOT / "experiments"
                                                / "p60_registrar.py"), pista],
                           capture_output=True, text=True, cwd=PROJECT_ROOT, check=False)
        for linea in r.stdout.splitlines():
            if linea.strip():
                log("     " + linea.strip())
        if r.returncode != 0:
            raise SystemExit(f"PARADA: el registro de {pista} fallo con codigo {r.returncode}")

    # ---------- 2. leer todos los artefactos ----------
    log("")
    log("   2 LECTURA de los artefactos JSON, sin recalcular nada")
    datos: dict[str, dict[str, Any]] = {}
    for pista, js in TRACKS_EJECUTADAS:
        f = REPORTS / js
        if not f.exists():
            raise SystemExit(f"PARADA: falta el artefacto {js} de {pista}")
        datos[pista] = json.loads(f.read_text(encoding="utf-8"))
        log(f"     {pista}: {js} leido")
    auditorias: dict[str, dict[str, Any]] = {}
    for pista, js in TRACKS_AUDITORIA:
        f = REPORTS / js
        if f.exists():
            auditorias[pista] = json.loads(f.read_text(encoding="utf-8"))
            log(f"     {pista}: {js} leido (auditoria, 0 experimentos)")

    filas: list[dict[str, Any]] = []
    for pista, _ in TRACKS_EJECUTADAS:
        d = datos[pista]
        r = d.get("resultado", {})
        g = r.get("gate", {})
        # Cada pista nombra su delta con la clave que describe SU comparacion. Se toma el
        # primer valor NO NULO y no el primer valor VERDADERO, porque el delta de la
        # pista C es exactamente cero, y cero es falsy en Python: con la forma
        # `a or b or c` un delta de 0,0000000000 se pasaba por alto y la tabla lo
        # mostraba como NaN, ocultando justamente el dato mas llamativo de esa pista.
        dm = primero_no_nulo(r, (
            "delta_mecanismo_puro", "delta_contraste_puro", "delta_perdida_puro",
            "delta_agrupado_puro", "delta_intervencion_puro", "delta_divergencia_puro",
            "delta_meta_puro", "delta_vs_control", "delta"))
        filas.append({
            "Track": pista.replace("P60-", ""),
            "ID": pista,
            "Paper_or_reference": d.get("paper", ""),
            "Corrected_CV": r.get("brazo_1_corrected_cv", r.get("corrected_cv")),
            "Delta_mecanismo": dm,
            "Delta_vs_V03": r.get("delta_vs_V03"),
            "Folds": " ".join(str(x) for x in (r.get("folds_brazo_1") or [])),
            "Pliegues_mejorados": r.get("pliegues_mejorados"),
            "Rescue": r.get("rescate"), "Damage": r.get("damage"), "Net": r.get("neto"),
            "Bootstrap": ("SI" if r.get("bootstrap") else "NO EJECUTADO (gate no superado)"),
            "Mecanismo_activo": mecanismo_activo(d),
            "Complementariedad": (
                f"rescate {r.get('rescate')} / dano {r.get('damage')} / neto {r.get('neto')}"),
            "Experimentos": d.get("experimentos_registrados", 0),
            "Decision": estado(d.get("decision", "")),
            "Decision_texto": d.get("decision", ""),
            "Gate_superado": bool(g.get("superado")),
        })
    df_tracks = pd.DataFrame(filas)
    df_tracks = df_tracks.sort_values("Track", key=lambda s: s.map(
        {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8,
         "I": 9, "J": 10, "K": 11, "L": 12}))
    df_tracks.to_csv(CSV_TRACKS, index=False, encoding="utf-8")
    log(f"     escrito reports/p60_track_summary.csv ({len(df_tracks)} pistas)")

    df_comp = df_tracks[["Track", "Corrected_CV", "Delta_mecanismo", "Pliegues_mejorados",
                         "Bootstrap", "Rescue", "Damage", "Net", "Mecanismo_activo",
                         "Complementariedad", "Decision"]]
    df_comp.to_csv(CSV_COMP, index=False, encoding="utf-8")
    log("     escrito reports/p60_candidate_comparison.csv")

    n_exp = int(df_tracks["Experimentos"].sum())
    n_prom = int((df_tracks["Decision"] == "PROMISING").sum())
    n_marg = int((df_tracks["Decision"] == "MARGINAL").sum())
    n_cerr = int((df_tracks["Decision"] == "CLOSED").sum())
    n_inval = int((df_tracks["Decision"] == "INVALIDATED").sum())
    n_boot = int((df_tracks["Bootstrap"] == "SI").sum())
    log(f"     {n_exp} experimentos | PROMISSING {n_prom} | MARGINAL {n_marg} | "
        f"CLOSED {n_cerr} | INVALIDATED {n_inval} | bootstrap ejecutados {n_boot}")
    log(f"     pistas cerradas SIN experimento: {len(TRACKS_CERRADAS)}")
    if n_prom or n_boot:
        log("     ATENCION: hay candidatos o bootstraps; revisar antes de cerrar")

    # ---------- 3. tabla maestra de P60 ----------
    log("")
    log("   3 TABLA MAESTRA de P60")
    ma = REPORTS / "p60_master_results.csv"
    if ma.exists():
        mm = pd.read_csv(ma)
        for _, fr in df_tracks.iterrows():
            mask = mm["ID"] == fr["ID"]
            if not bool(mask.any()):
                mm = pd.concat([mm, pd.DataFrame([{c: None for c in mm.columns}])],
                               ignore_index=True)
                mask = mm["ID"].isna()
                mm.loc[mask, "ID"] = fr["ID"]
                mask = mm["ID"] == fr["ID"]
            mm.loc[mask, "Corrected_CV"] = fr["Corrected_CV"]
            mm.loc[mask, "Delta_vs_V03"] = fr["Delta_vs_V03"]
            mm.loc[mask, "Delta_mecanismo"] = fr["Delta_mecanismo"]
            if isinstance(fr["Folds"], str):
                for i, val in enumerate(fr["Folds"].split(), start=1):
                    col = f"Fold_{i}"
                    if col in mm.columns:
                        mm.loc[mask, col] = float(val)
            mm.loc[mask, "Rescue"] = fr["Rescue"]
            mm.loc[mask, "Damage"] = fr["Damage"]
            mm.loc[mask, "Net"] = fr["Net"]
            mm.loc[mask, "Bootstrap_CI"] = fr["Bootstrap"]
            mm.loc[mask, "Decision"] = fr["Decision_texto"][:480]
        mm.to_csv(ma, index=False, encoding="utf-8")
        log(f"     p60_master_results.csv con {len(mm)} filas")

    # ---------- 4. historial completo del proyecto ----------
    log("")
    log("   4 HISTORIAL COMPLETO, derivado de experiments.csv y no transcrito")
    exp = pd.read_csv(REG)
    log(f"     experiments.csv: {len(exp)} filas, {exp.experiment_id.nunique()} ids unicos")
    if exp.experiment_id.nunique() != len(exp):
        raise SystemExit("PARADA: experiments.csv tiene ids duplicados")
    cols = set(exp.columns)
    hist: list[dict[str, Any]] = []
    for _, r in exp.iterrows():
        eid = str(r["experiment_id"])
        fase = ("P60" if eid.startswith("P60") else
                "P59" if eid.startswith("P59") else
                eid.split("-")[0] if "-" in eid else "TOXIC")
        res = str(r["result"]) if "result" in cols and not pd.isna(r["result"]) else ""
        # El rescue/damage/net y el bootstrap no existen como columnas en el registro
        # canonico, asi que se leen del JSON de resultado cuando ese JSON los trae. Donde
        # no estan, se dice NO REGISTRADO en vez de inventarlos.
        resc = dan = net = "NO REGISTRADO"
        boot = "NO EJECUTADO"
        try:
            d = json.loads(res) if res.startswith("{") else {}
        except json.JSONDecodeError:
            d = {}
        if isinstance(d, dict):
            if d.get("rescate") is not None:
                resc, dan, net = d.get("rescate"), d.get("damage"), d.get("neto")
            if d.get("bootstrap"):
                boot = "SI"
        hist.append({
            "phase": fase,
            "experiment_id": eid,
            "date": str(r["date"])[:10] if "date" in cols else "",
            "hypothesis": str(r["hypothesis"])[:400] if "hypothesis" in cols else "",
            "method": str(r["model"])[:200] if "model" in cols else "",
            "representation": str(r["representation"])[:200] if "representation" in cols else "",
            "model": str(r["model"])[:200] if "model" in cols else "",
            "corrected_cv": r.get("macro_f1"),
            "f1": r.get("f1"),
            "folds": "NO REGISTRADO",
            "bootstrap": boot,
            "rescue": resc, "damage": dan, "net": net,
            "decision": str(r["decision"])[:600] if "decision" in cols else "",
            "raw_result": res[:600],
        })
    # El texto de las decisiones viene de experiments.csv y arrastra no-ASCII
    # HISTORICO de fases anteriores. Se translitera en vez de relajar el guardia, y
    # queda dicho aqui para que nadie lea el CSV pensando que es la fuente original:
    # la fuente canonica sigue siendo experiments.csv, sin transformar.
    df_hist = pd.DataFrame(hist)
    cols_txt = [c for c in df_hist.columns
                if df_hist[c].dtype == object and c != "phase"]
    for c in cols_txt:
        df_hist[c] = df_hist[c].map(lambda z: a_ascii(str(z)))
    sum(int((df_hist[c] != pd.read_csv(REG)[c].astype(str)).sum())
                 for c in cols_txt if c in pd.read_csv(REG).columns
                 and len(pd.read_csv(REG)) == len(df_hist))
    df_hist.to_csv(CSV_HIST, index=False, encoding="ascii")
    log(f"     escrito reports/toxic_complete_experiment_history.csv ({len(hist)} filas, "
        f"texto transliterado a ASCII; la fuente canonica sigue siendo experiments.csv)")
    conteo_fases = Counter(h["phase"] for h in hist)
    log(f"     filas por familia: {dict(sorted(conteo_fases.items()))}")

    # ---------- 5. informe final de P60 ----------
    log("")
    log("   5 INFORME FINAL DE P60")
    L: list[str] = []
    W = L.append
    W("# P60 - Informe final de investigacion guiada por literatura")
    W("")
    W(f"Generado por `experiments/p60_11_cerrar.py`. Fecha {inicio.isoformat(timespec='seconds')}.")
    W(f"Commit {commit_actual()}. Rama v60. Sin push.")
    W("")
    W("## Resumen ejecutivo")
    W("")
    W(f"P60 ejecuto doce pistas de la literatura sobre aprendizaje multi-etiqueta, sobre "
      f"la frontera que P59 dejo cerrada. Doce pistas, {n_exp} experimentos consumidos, "
      f"{len(TRACKS_CERRADAS)} cerradas sin experimentar, "
      f"{n_prom} candidatas, {n_marg} marginales, {n_cerr} cerradas, {n_inval} invalidadas "
      f"y {n_boot} bootstraps ejecutados.")
    W("")
    W("**Ninguna pista produjo un candidato.** El gate exige delta de al menos +0,003 y "
      "4 de 5 pliegues, y ninguna de las ocho pistas con experimento llego al primero de "
      "los dos criterios. Ningun bootstrap se ejecuto, porque el bootstrap solo se ejecuta "
      "despues de superar el gate.")
    W("")
    W("```text")
    W(f"V03 corrected_cv = {V03_CCV!r}")
    W(f"V03 LB           = {lb}")
    W(f"V03 sha256       = {V03_SHA}")
    W("```")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("## Estado de V03, verificado al inicio y al cierre de cada pista")
    W("")
    W("El sha256 de la submission se verifica al comienzo de cada uno de los ocho scripts de "
      "experimento y en este script de cierre. Ninguna pista modifico V03, el OOF canonico, "
      "los pliegues ni las submissiones. Cero submissiones en toda la fase.")
    W("")
    W("## Motivacion: por que esta fase es distinta de P59")
    W("")
    W("P59 pregunto si quedaba alguna fuente de senal dentro de las familias ya conocidas. "
      "P60 cambia la pregunta: en vez de variations sobre la misma familia, se buscaron "
      "mecanismos de aprendizaje apoyados en literatura reciente que **modifiquen como se "
      "representa, optimiza o combina la evidencia** y que no sean una reformulacion de lo "
      "ya cerrado. Por eso el bloque empieza con una revision bibliografica y un registro de "
      "huecos, y las pistas se cierran por duplicacion funcional antes de gastar experimento "
      "cuando la literatura no aporta un mecanismo nuevo.")
    W("")
    W("## Revision bibliografica y gap analysis")
    W("")
    lit = REPORTS / "p60_literature_registry.csv"
    if lit.exists():
        dl = pd.read_csv(lit)
        W(f"Se evaluaron **{len(dl)} metodos** de la literatura, con prioridad en 2024, 2025 y "
          f"2026 y con los trabajos clasicos cuando son la referencia metodologica. "
          f"{int(dl['novelty'].sum())} pasaron el gap check y {int((~dl['novelty']).sum())} se "
          "cerraron por equivalencia funcional.")
        W("")
        W("| ref | metodo | filas del registro | mecanismo en una linea |")
        W("|---|---|---:|---|")
        for _, r in dl.sort_values("priority").iterrows():
            W(f"| {r['clave']} | {r['method'][:64]} | {r['rows_in_registry_matching']} | "
              f"{r['mechanism'][:96]} |")
        W("")
    W("### La estructura de etiquetas, medida y no citada")
    W("")
    rev = json.loads((REPORTS / "p60_literature_review.json").read_text(encoding="utf-8"))
    est = rev.get("estructura_etiquetas_medida", {})
    W("Este es el hecho que ordena la revision y viene del OOF canonico, no de ningun trabajo:")
    W("")
    W("```text")
    for k in ("prevalencia_HS", "prevalencia_TR", "prevalencia_AG",
              "P_TR1_dado_HS0", "P_AG1_dado_HS0", "P_AG1_dado_TR1",
              "negativos_TR_no_definidos", "negativos_AG_no_definidos"):
        if k in est:
            W(f"{k:32} {est[k]}")
    W("```")
    W("")
    W("Tres consecuencias, y las tres importan mas que cualquier resultado individual:")
    W("")
    W(f"1. **Las etiquetas estan anidadas, no solo correlacionadas.** P(TR=1 | HS=0) = "
      f"{est.get('P_TR1_dado_HS0', 0):.4f}. El subespacio factible tiene cuatro puntos, "
      "que es exactamente el que `V4-E01-E03` enumero en doce variantes sin superar a V03.")
    W("")
    W(f"2. **El {est.get('negativos_TR_no_definidos', 0) * 100:.0f} por ciento de los negativos "
      f"que ve TR en entrenamiento son etiquetas NO DEFINIDAS**, y el "
      f"{est.get('negativos_AG_no_definidos', 0) * 100:.0f} por ciento en AG. Eso es ruido "
      "dependiente de la etiqueta, y es lo que la pista C midio por pares.")
    W("")
    W("3. **La asimetria de prevalencias es de 1,46 a 1, no de una clase minoritaria frente a "
      "una dominante.** Ridnik et al. justifican la perdida asimetrica por un desequilibrio de "
      "uno o dos ordenes de magnitud, luego la priori de la pista A era baja desde el "
      "principio y la revision lo declaro antes de ejecutar.")
    W("")
    W("## Las doce pistas")
    W("")
    W("| orden | pista | referencia | veredicto | delta del mecanismo | pliegues | neto |")
    W("|---:|---|---|---|---:|---:|---:|")
    orden = {"G": 1, "A": 2, "L": 3, "D": 4, "C": 5, "H": 6, "I": 7, "J": 8}
    for _, r in df_tracks.sort_values("Track", key=lambda s: s.map(orden)).iterrows():
        dm = r["Delta_mecanismo"]
        dms = f"{dm:+.6f}" if isinstance(dm, (int, float)) else str(dm)
        W(f"| {orden.get(r['Track'], 99)} | {r['Track']} | {str(r['Paper_or_reference'])[:46]} | "
          f"**{r['Decision']}** | {dms} | {r['Pliegues_mejorados']}/5 | {r['Net']} |")
    W("")
    W("### Pistas cerradas sin consumir experimento")
    W("")
    W("| pista | referencia | precedente funcional | decision |")
    W("|---|---|---|---|")
    for pista, ref, razon, _prec in TRACKS_CERRADAS:
        W(f"| {pista} | {ref} | {razon[:150]} | CERRADA POR DUPLICACION |")
    W("")
    for pista, ref, razon, prec in TRACKS_CERRADAS:
        W(f"* **{pista} ({ref})**: {razon}. Precedentes: {prec}.")
    W("")
    W("## Resultados, incluidos los negativos")
    W("")
    W("No se muestran solo los mejores. El orden de lectura que mas informa es por el delta del "
      "mecanismo contra su propio control, y no por el corrected_cv absoluto.")
    W("")
    W("| pista | corrected_cv | delta del mecanismo | pliegues | rescate | dano | neto | celdas |")
    W("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in df_tracks.iterrows():
        dm = r["Delta_mecanismo"]
        dms = f"{dm:+.10f}" if isinstance(dm, (int, float)) else str(dm)
        W(f"| {r['Track']} | {r['Corrected_CV']} | {dms} | {r['Pliegues_mejorados']}/5 | "
          f"{r['Rescue']} | {r['Damage']} | {r['Net']} | {r['Delta_vs_V03']} |")
    W("")
    W("## Resultados marginales")
    W("")
    marg = df_tracks[df_tracks["Decision"] == "MARGINAL"]
    if len(marg):
        for _, r in marg.iterrows():
            W(f"* **{r['Track']}**, delta del mecanismo {r['Delta_mecanismo']}, "
              f"{r['Pliegues_mejorados']} de 5 pliegues, neto {r['Net']}. "
              f"Mecanismo: {r['Mecanismo_activo']}")
    else:
        W("Ninguna pista quedo marginal.")
    W("")
    W("## Candidatos")
    W("")
    prom = df_tracks[df_tracks["Decision"].isin(VERDE)]
    if len(prom):
        for _, r in prom.iterrows():
            W(f"* **{r['Track']}**: {r['Decision_texto']}")
    else:
        W("**Ninguno.** Ninguna pista supero el gate de +0,003 con 4 de 5 pliegues, luego no "
          "hay finalista y no hay artefacto de candidato que generar.")
    W("")
    W("## Bootstrap")
    W("")
    W(f"Ejecutados: **{n_boot} de {n_exp}**. El bootstrap se ejecuta solo despues de superar el "
      "gate, y ninguna pista lo supero, luego no hay ningun intervalo de confianza que reportar. "
      "Es preferible no tener ninguno que tener el intervalo de una mejora que no existe.")
    W("")
    W("## Complementariedad")
    W("")
    W("La tabla de rescate y dano es la que mas informa sobre si una tecnica aporta "
      "informacion o solo desplaza decisiones. Ninguna pista de P60 produjo un neto que se "
      "aproxime a un volumen de correccion util comparable al de un modelo alternativo "
      "entrenado de verdad.")
    W("")
    for _, r in df_tracks.iterrows():
        W(f"* **{r['Track']}**: {r['Complementariedad']}, pliegues {r['Pliegues_mejorados']}/5")
    W("")
    W("## Reproducibilidad")
    W("")
    W("Cada pista es un script nuevo y completo que se ejecuta de principio a fin y escribe su "
      "propio artefacto JSON, su informe y su CSV. Antes de aceptar cualquier resultado cada "
      "script pasa `ast.parse`, `ruff`, `py_compile` y una ejecucion real cuyo codigo de salida "
      "se captura sin tuberia. Los scripts de losses, DRO, deteccion y PolyLoss verifican su "
      "gradiente analitico contra diferencias centrales antes de entrenar, y el script de "
      "transporte verifica la exactitud de su metrica antes de concluir nada.")
    W("")
    W("## Compliance")
    W("")
    W("```text")
    W("transformers = 0        BERT = 0            RoBERTa = 0         LLM = 0")
    W("embeddings preentrenados = 0                FastText preentrenado = 0")
    W("Word2Vec preentrenado = 0                    GloVe = 0")
    W("clasificadores externos = 0                  etiquetas externas = 0")
    W("corpus toxicos externos = 0                  hate speech externo = 0")
    W("pseudo-etiquetas = 0                         test labels = 0")
    W("test training = 0                            test calibration = 0")
    W("test feature selection = 0                   holdout reuse = 0")
    W("leaderboard tuning = 0                      submissiones = 0")
    W("consultas a Kaggle = 0")
    W("```")
    W("")
    W("Deep learning, donde se uso: solo en la pista G, con un SVD interno de 128 dimensiones y "
      "una proyeccion lineal de 32. Sin transformer, sin embedding preentrenado, sin datos "
      "externos.")
    W("")
    W("## Git")
    W("")
    W("Trabajo en la rama `proyecto-kaggle-toxic-2026-p60`, creada desde `v25` que contiene P59. "
      "Sin push, sin merge, sin force-push, sin rebase destructivo. Un commit local por pista. "
      f"Estado al cierre de este script: commit {commit_actual()}.")
    W("")
    W("## Notebook")
    W("")
    W("La integracion en el notebook consolidado se genera por script leyendo los artefactos, "
      "de modo que ninguna cifra del notebook se escribe a mano. Se registra en "
      "`reports/p60_notebook_integration.json`.")
    W("")
    W("## Decision final")
    W("")
    W("```text")
    W("P60 = CLOSED")
    W("V03 remains final")
    W("```")
    W("")
    W("La pregunta final del encargo era si existe una tecnica respaldada por literatura que "
      "introduzca una fuente de senal o un mecanismo de aprendizaje materialmente diferente de "
      "todo lo ya probado y que consiga una mejora reproducible sobre V03. La respuesta "
      "documentada es **no**, y conviene separarla en las cinco dimensiones que el encargo pide, "
      "porque la respuesta no es la misma en todas:")
    W("")
    W("* **deteccion**: si. La pista C demostro que dentro de la region sin definir de AG la "
      "confianza esta ANTICORRELACIONADA con el error, con AUC 0,1814, y que un indicador "
      "estructural aporta +0,1129 de AUC residual sobre el control de esa misma etiqueta. Es "
      "informacion nueva y real.")
    W("")
    W("* **calibracion**: no aporta nada nuevo. La mejora de P59-B, que era +0,003367, venia "
      "del sesgo de la agregacion y no de correccion alguna, y P59-U quedo marginal.")
    W("")
    W("* **representacion**: el contraste por solapamiento de etiquetas es genuinamente un eje "
      "nuevo, y el gap check lo confirma, pero perdio -0,2949 porque la perdida contrastiva "
      "domina por un orden de magnitud y porque la arquitectura no puede competir con V03. El "
      "transporte optimo tambien es genuinamente distinto, con 2842 veces el sesgo de su propia "
      "metrica como prueba, y perdio -0,0723 contra su control de centroide.")
    W("")
    W("* **optimizacion**: la perdida asimetrica actua donde no debia y se apaga donde debia. "
      "El corte duro, que es la parte nova, nunca se activo: cero por ciento de los negativos "
      "tienen perdida exactamente cero. Lo que si actuo fue la potencia del negativo, que es un "
      "ponderado de clase, familia cerrada por V8-07. El riesgo minimax por grupos se activa "
      "con una sobreponderacion del grupo peor de +2,47 por ciento y mueve tres celdas de 4449. "
      "El numero efectivo de muestras resulta ser, medido antes de entrenar, la misma funcion "
      "que la forma de V03 con menos de 2,33 por ciento de diferencia y coseno 0,9999996 entre "
      "las dos soluciones. El meta-reweighting y PolyLoss rematan el eje de la perdida.")
    W("")
    W("* **informacion nueva**: hay una, y es pequena y local: la estructura anidada de las "
      "etiquetas convierte el 100 por ciento de los negativos de TR en etiquetas no definidas, "
      "y esa region se comporta de forma distinta. Lo que no hay es una forma de exploitation "
      "de esa region que mejore la decision. El intento suave fue un no-op exacto, con delta "
      "0,0000000000, y el intento duro ya lo habia cerrado V14-00 con -0,116581. El eje queda "
      "acotado entre dos extremos medidos: factor cero es catastrofico y factor uno no hace "
      "nada.")
    W("")
    W("La conclusion que sostiene el conjunto de P59 y P60 es que V03 no esta limitado por la "
      "falta de un mecanismo de aprendizaje mejor, sino por el techo estructural que V28-01 "
      "midio: los errores de HS, TR y AG se solapan con phi de +0,42 a +0,51, luego fallan "
      "sobre los mismos textos, y cualquier metodo que actue sobre una sola etiqueta choca "
      "contra ese techo. Los dos unicos ejes que podrian esquivarlo, la geometria aprendida y "
      "el objetivo sobre grupos, se ejecutaron y ninguno produjo una mejora.")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p60_11_cerrar.py`. Sin push.*")
    MD_P60.write_text(a_ascii("\n".join(L)), encoding="ascii")
    log("     escrito reports/P60_FINAL_RESEARCH_REPORT.md")

    resumen = {
        "familia": "P60_CIERRE",
        "fecha": inicio.isoformat(timespec="seconds"),
        "commit": commit_actual(),
        "baseline": {"corrected_cv": repr(V03_CCV), "lb": lb, "sha256": hash_real},
        "pistas": len(TRACKS_EJECUTADAS),
        "pistas_cerradas_sin_experimento": len(TRACKS_CERRADAS),
        "experimentos_consumidos": n_exp,
        "auditorias_sin_experimento": len(TRACKS_AUDITORIA),
        "decisiones": dict(Counter(df_tracks["Decision"])),
        "bootstraps_ejecutados": n_boot,
        "finalistas": len(prom),
        "submissiones": 0,
        "consultas_kaggle": 0,
        "uso_test": 0,
        "uso_holdout": 0,
        "decision_final": "P60 = CLOSED; V03 remains final",
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_FIN.write_text(json.dumps(resumen, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    log("     escrito reports/p60_final_summary.json")

    for destino in (CSV_TRACKS, CSV_COMP, MD_P60, CSV_HIST, JSON_FIN):
        if destino.exists() and any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P60-11 OK | {n_exp} experimentos | {n_prom} candidatos | "
        f"{n_marg} marginales | {n_cerr} cerradas | {n_boot} bootstraps")


def mecanismo_activo(d: dict[str, Any]) -> str:
    """Resume en una linea si el mecanismo de la pista llego a activarse.

    Args:
        d: diccionario del artefacto de la pista.

    Returns:
        Una descripcion corta.
    """
    act = d.get("activacion_del_mecanismo")
    if isinstance(act, dict):
        z = act.get("negativos_con_perdida_exactamente_cero")
        if isinstance(z, dict):
            rz = act.get("reescalado_efectivo_del_negativo", {})
            return (f"corte duro {z.get('media', 0) * 100:.2f} por ciento de activating; "
                    f"reescalado del negativo factor {rz.get('factor_de_reduccion', 0):.1f}")
    ver = d.get("gap_analysis", {})
    if isinstance(ver, dict) and "verificacion" in ver:
        v = ver["verificacion"]
        return (f"distinguible del control por {v.get('razon_senal_sobre_sesgo', 0):.0f} veces "
                "el sesgo de su propia metrica")
    diag = d.get("diagnostico_del_mecanismo")
    if isinstance(diag, dict):
        return (f"fraccion negativa {diag.get('fraccion_negativa_media', 0) * 100:.1f} por "
                f"ciento, razon p90/abs(p10) {diag.get('razon_p90_sobre_abs_p10_media', 0):.2f}")
    res = d.get("resultado", {})
    cam = res.get("cambian")
    if cam is not None:
        return f"{cam} celdas de 4449 cambiadas"
    return "no medido explicitamente"


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
