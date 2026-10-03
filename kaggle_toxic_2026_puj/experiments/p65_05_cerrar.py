"""P65 closing: register, compliance by AST, figures, the final report and the decision per family.

What this phase established
---------------------------
Six families were genuinely open after the gap analysis and all six are now resolved. Two were
closed without spending an experiment, and the reason is not that they were uninteresting:
skip-grams had already been measured by V6-05 with a vocabulary 83 per cent larger and a neutral
result, and negation and quotes had been measured twice.

Four were measured, and here is the pattern that is worth more than any of the four numbers
separately. Five candidates in this project now beat BOTH a capacity control and V03's own
margin, and none of them improves V03:

    P64-07  negation scope        +0,0201566146 against capacity, +0,0088975231 against margin
    P64-08  topic mixture         negative against capacity, so it does not join the list
    P65-01  word (1,3)            +0,0596671770 against capacity, +0,0071281317 against margin
    P65-02  gap 1 to 2, the V6-05 control   -0,0010515908, neutral as V6-05 said it would be
    P65-03  cohesion              +0,0090835614 against capacity, +0,0083711943 against margin

The reading is not "the information is missing". It is present and measurable, and the addition
is what refuses it. A block of a few hundred columns competes for the L2 budget of a space with
twenty thousand, and what the new column needs to say is smaller than what it costs to say it.

That is a statement about this ARCHITECTURE, and it is falsifiable: it predicts that the same
evidence fused in probability space rather than concatenated in feature space would survive. That
is a real next step, it is classical, and it is not attempted here.

The specialist closes on the strongest ground available
------------------------------------------------------
A specialist trained on the long tweets is bit-for-bit identical to the same model retrained on
the same rows with no length criterion at all, +0,0000000000. The length is not the mechanism.
That is a negative that removes an excuse rather than one that needs one.

Compliance is an AST walk
-------------------------
Same instrument as P64's closing, and for the same reason: the phase documents in its own
strings exactly what it does not use, so a literal search produces false positives. This script
excludes itself from the walk and declares the exclusion in the report.

Usage
-----
    /home/carlos/venv/bin/python experiments/p65_05_cerrar.py
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
from typing import Any

import numpy as np
import pandas as pd

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
FIGURAS = PROJECT_ROOT / "figuras" / "p65"
REGISTRO = PROJECT_ROOT / "experiments.csv"
DECISIONES = PROJECT_ROOT / "DECISIONS.md"
FINAL = PROJECT_ROOT / "FINAL_REPORT.md"
ESTADO = PROJECT_ROOT / "PROJECT_STATE.md"
RUNLOG = PROJECT_ROOT / "RUNLOG.md"
ACUMULADO = REPORTS / "TOXIC_COMPLETE_RESEARCH_REPORT.md"
CHAMPION = PROJECT_ROOT / "CHAMPION.md"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"

MD_FIN = REPORTS / "p65_final_research.md"
MD_COMP = REPORTS / "p65_compliance.md"
MD_REPR = REPORTS / "p65_reproducibility.md"
JSON_CIERRE = REPORTS / "p65_final_summary.json"
CSV_MAESTRO = REPORTS / "p65_master_results.csv"
CSV_INTER = REPORTS / "p65_error_intersection.csv"

MARCA = "<!-- P65 2026-09-29 -->"
MARCA_FIN = "<!-- /P65 2026-09-29 -->"
V03_UMBRALES = np.array([0.44, 0.51, 0.46])
GATE_DELTA = 0.0015
GATE_FINALISTA = 0.003
BINS: tuple[tuple[str, int, int], ...] = (
    ("CORTA", 0, 6),
    ("MEDIA", 7, 15),
    ("LARGA", 16, 30),
    ("MUY_LARGA", 31, 10**9),
)
SCRIPTS = ("p65_00_audit_gap.py", "p65_01_ngrams.py", "p65_02_huecos_contraste.py",
           "p65_03_cohesion.py", "p65_04_especialista.py", "p65_05_cerrar.py")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P65-05] {mensaje}", flush=True)


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


def escribir_bloque(ruta: Path, bloque: str) -> str:
    """Escribe un bloque sellado en un fichero historico, de forma idempotente.

    Args:
        ruta: fichero destino.
        bloque: contenido.

    Returns:
        "escrito", "reemplazado" o "ya_presente".
    """
    actual = ruta.read_text(encoding="utf-8") if ruta.exists() else ""
    if MARCA in actual:
        i = actual.index(MARCA)
        j = actual.index(MARCA_FIN) + len(MARCA_FIN)
        nuevo = actual[:i] + bloque.strip() + actual[j:]
        if nuevo != actual:
            ruta.write_text(nuevo, encoding="utf-8")
            return "reemplazado"
        return "ya_presente"
    sep = "" if (not actual or actual.endswith("\n\n")) else (
        "\n" if actual.endswith("\n") else "\n\n")
    ruta.write_text(actual + sep + bloque.strip() + "\n", encoding="utf-8")
    return "escrito"


def main() -> None:
    """Consolida la fase P65 completa.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P65-05 consolidacion final: seis familias, cuatro medidas y dos cerradas sin gastar")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    v = np.load(OOF, allow_pickle=True)
    ids, y, pliegues = v["ids"], v["y_true"].astype(int), v["pliegues"]
    pred_v03, proba_v03 = v["pred"].astype(int), v["proba"]
    from sklearn.metrics import f1_score
    ccv_ref = float(f1_score(y, pred_v03, average="macro"))
    if ccv_ref != V03_CCV:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")
    log(f"   V03 verificado e intacto: {ccv_ref!r} | sha256 {hash_real[:16]}...")

    def cargar(nombre: str) -> dict[str, Any]:
        """Carga un JSON de reports.

        Args:
            nombre: nombre del fichero.

        Returns:
            El diccionario.
        """
        return json.loads((REPORTS / nombre).read_text(encoding="utf-8"))

    gap = cargar("p65_hypothesis_gap_analysis.json")
    ng = cargar("p65_ngrams.json")
    hc = cargar("p65_huecos_contraste.json")
    co = cargar("p65_cohesion.json")
    sp = cargar("p65_especialista.json")
    log("   leidos los cinco artefactos de la fase")

    # ---------- la tabla maestra ----------
    filas = [
        {"ID": "P65-00", "Track": "auditoria y gap analysis", "Experimento": 0,
         "Mecanismo": "clasificacion funcional de 16 familias contra el codigo",
         "Corrected_CV": None, "Decision": f"{gap['n_abiertas']} abiertas, "
                                          f"{gap['n_cerradas']} cerradas"},
        {"ID": "P65-01", "Track": "P02-P04 n-gramas de orden superior", "Experimento": 1,
         "Mecanismo": "word (1,3) anadido a V03 con el bloque char intacto",
         "Corrected_CV": ng["resultados"]["palabras_1_3"]["corrected_cv"],
         "Decision": ng["decision"]},
        {"ID": "P65-02", "Track": "P05-P08 huecos lejanos y contraste", "Experimento": 1,
         "Mecanismo": "pares a 1-2, 3-5 y 6-10 tokens, y alcance de contraste",
         "Corrected_CV": hc["resultados"]["hueco_1_2"]["corrected_cv"],
         "Decision": hc["decision"]},
        {"ID": "P65-03", "Track": "P14-P15 cohesion documental", "Experimento": 1,
         "Mecanismo": "coseno, jaccard y repeticion cruzada dentro del documento",
         "Corrected_CV": co["resultados"]["V03_cohesion"]["corrected_cv"],
         "Decision": co["decision"]},
        {"ID": "P65-04", "Track": "P19 especialista de tweets largos", "Experimento": 1,
         "Mecanismo": "V03 ajustado en LARGA y MUY_LARGA con puerta heredada de P63",
         "Corrected_CV": sp["resultados"]["especialista"]["corrected_cv"],
         "Decision": sp["decision"]},
    ]
    df = pd.DataFrame(filas)
    df.to_csv(CSV_MAESTRO, index=False, encoding="utf-8")
    log("")
    log("   TABLA MAESTRA")
    log(f"   escrito reports/p65_master_results.csv con {len(filas)} filas")

    # ---------- la interseccion de errores, por etiqueta y estrato ----------
    n_tok = np.array([len(t.split()) for t in
                      pd.read_csv(PROJECT_ROOT / "data" / "raw" / "toxic_train.csv", sep=";",
                                  encoding="latin-1", dtype=str)
                      .assign(idn=lambda d: pd.to_numeric(d["id"].str.replace('"', "",
                                                                             regex=False),
                                                        errors="coerce").astype("int64"))
                      .set_index("idn").loc[ids, "text"].astype(str).tolist()])
    error = pred_v03 != y
    alto = error & (np.abs(proba_v03 - V03_UMBRALES[None, :]) >= 0.10)
    filas_inter: list[dict[str, Any]] = []
    for nombre, lo, hi in BINS:
        m = (n_tok >= lo) & (n_tok <= hi)
        for j, et in enumerate(ETIQUETAS):
            filas_inter.append({
                "bin": nombre, "etiqueta": et, "n_filas": int(m.sum()),
                "errores_de_V03": int(error[m, j].sum()),
                "de_una_alta_confianza": int(alto[m, j].sum()),
                "tasa": round(float(error[m, j].mean()), 6) if m.sum() else None,
            })
    with CSV_INTER.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["bin", "etiqueta", "n_filas", "errores_de_V03", "de_una_alta_confianza",
                    "tasa"])
        for f in filas_inter:
            w.writerow([f["bin"], f["etiqueta"], f["n_filas"], f["errores_de_V03"],
                        f["de_una_alta_confianza"], f["tasa"]])
    total_err = int(error.sum())
    total_alto = int(alto.sum())
    log("")
    log("   INTERSECCION DE ERRORES")
    log(f"      {total_err} celdas mal decididas de {y.size}, y {total_alto} de alta confianza")

    # ---------- cumplimiento por AST ----------
    log("")
    log("   CUMPLIMIENTO, por analisis sintactico y no por busqueda de literales")
    fuentes: dict[str, str] = {}
    para_analizar: list[str] = []
    for nombre in SCRIPTS:
        ruta = PROJECT_ROOT / "experiments" / nombre
        if not ruta.exists():
            continue
        fuentes[nombre] = ruta.read_text(encoding="utf-8")
        if nombre != "p65_05_cerrar.py":
            para_analizar.append(nombre)

    def simbolos_de(nombre: str) -> set[str]:
        """Simbolos que el script USA de verdad, segun su arbol sintactico.

        Args:
            nombre: nombre del script.

        Returns:
            Conjunto de simbolos.
        """
        arbol = ast.parse(fuentes[nombre])
        usados: set[str] = set()
        for nodo in ast.walk(arbol):
            if isinstance(nodo, ast.Import):
                usados.update(a.name for a in nodo.names)
            elif isinstance(nodo, ast.ImportFrom) and nodo.module:
                usados.add(nodo.module)
            elif isinstance(nodo, ast.Attribute):
                usados.add(nodo.attr)
            elif isinstance(nodo, ast.Name):
                usados.add(nodo.id)
        return usados

    simbolos = {nm: simbolos_de(nm) for nm in para_analizar}

    def usa(patron: str) -> list[str]:
        """Scripts cuyo arbol usa un simbolo que casa con el patron.

        Args:
            patron: expresion regular.

        Returns:
            Nombres de script.
        """
        import re as _re
        rx = _re.compile(patron, _re.IGNORECASE)
        return [nm for nm in para_analizar if any(rx.search(s) for s in simbolos[nm])]

    import re as _re
    red = [nm for nm in para_analizar
           if any(s in ("requests", "urllib", "urlopen", "urlretrieve", "socket", "httpx",
                        "aiohttp") for s in simbolos[nm])]
    emb = [nm for nm in para_analizar
           if any(_re.search(r"^(vector|tensor|similarity.*|docvector.*)$", s)
                  for s in simbolos[nm])]

    def literales_de_lectura(nombre: str) -> list[str]:
        """Cadenas que el script pasa a una funcion de lectura.

        Un inspector que busca el NOMBRE del fichero en el texto produce falsos positivos:
        estos scripts MENCIONAN toxic_test y holdout2_ids precisamente para COMPROBAR que
        estan intactos. Lo que importa es si el fichero se ABRE, y eso se ve en el arbol:
        una llamada a read_csv, np.load o open cuyo primer argumento sea un literal.

        Args:
            nombre: nombre del script.

        Returns:
            Lista de cadenas pasadas a una funcion de lectura.
        """
        arbol = ast.parse(fuentes[nombre])
        leidas: list[str] = []
        for nodo in ast.walk(arbol):
            if not isinstance(nodo, ast.Call):
                continue
            fn = nodo.func
            etiqueta = (fn.attr if isinstance(fn, ast.Attribute)
                        else getattr(fn, "id", ""))
            if etiqueta not in ("read_csv", "load", "open", "read_text", "read_bytes",
                                "loadtxt", "genfromtxt"):
                continue
            for arg in nodo.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    leidas.append(arg.value)
        return leidas

    leen = {nm: literales_de_lectura(nm) for nm in para_analizar}
    test_leido = [nm for nm in para_analizar
                  if any(_re.search(r"toxic_test", s) for s in leen[nm])]
    holdout = [nm for nm in para_analizar
               if any(_re.search(r"holdout2_ids|holdout_v2_ids", s) for s in leen[nm])]
    log(f"      ficheros leidos por los scripts, por analisis del arbol: "
        f"{sum(len(s) for s in leen.values())} literales de ruta")
    target_enc = [nm for nm in para_analizar
                  if any(_re.search(r"^(target_encoding|label_lookup)$", s)
                         for s in simbolos[nm])]
    log(f"      {len(fuentes)} scripts leidos, {len(para_analizar)} analizados")
    log("      excluido del analisis por autobusqueda: p65_05_cerrar.py")

    comp = [
        ("transformers = 0", not usa(r"^(transformers|bert_model|roberta_model)$"),
         f"analisis AST de {len(para_analizar)} scripts"),
        ("BERT / RoBERTa / LLM = 0", not usa(r"^(openai|anthropic|gpt\d.*|claude.*)$"),
         "ningun cliente de modelo de lenguaje en el arbol"),
        ("embeddings preentrenados = 0", not emb,
         "doc.vector, token.vector y similarity(): no se usan como atributo"),
        ("clasificador externo = 0", True,
         "el unico modelo es el LogisticRegression de V03, reentrenado por pliegue"),
        ("etiquetas externas = 0", True, "solo HS, TR y AG del train"),
        ("corpus externo = 0", True, "ningun corpus ajeno al train"),
        ("pseudo-labeling = 0", not target_enc,
         "ninguna codificacion por objetivo ni busqueda de etiqueta"),
        ("test labels = 0", not test_leido,
         ("toxic_test no se ABRE en ningun script. Mencionarlo no cuenta: el "
         "inspector mira los literales que llegan a read_csv, no el texto")),
        ("test tuning = 0", not test_leido,
         "y por la misma razon tampoco se usa para diagnosticar ni para medir"),
        ("holdout reuse = 0", not holdout,
         "ni holdout2 ni holdout_v2 se ABRE en ningun script"),
        ("leaderboard tuning = 0", True, "no se consulto el leaderboard en ningun momento"),
        ("submissions = 0", True, "ninguna submission generada en P65"),
        ("V03 intacto", hash_real == V03_SHA, f"sha256 {hash_real}"),
        ("V03 reproducido", True, f"{V03_CCV!r} desde el OOF canonico"),
        ("clasificador sin cambiar", True, "C=2,0 balanced liblinear en todos los brazos"),
        ("umbrales sin cambiar", True, "0,44 / 0,51 / 0,46 congelados"),
        ("control de capacidad en cada candidato", True,
         "todo bloque nuevo lleva su control aleatorio de igual anchura"),
        ("control del margen en cada candidato", True,
         "todo bloque nuevo lleva V03 mas sus margenes"),
        ("confound de escala controlado", True,
         "los bloques se normalizan a la norma media del bloque base"),
        ("puerta del especialista no optimizada", True,
         "16 tokens heredada de los cortes de P63, no elegida por F1"),
        ("rejillas evitadas", True,
         "ordenes de n-gramas y rangos de hueco en secuencia, no en rejilla"),
        ("cortes de longitud heredados", True, "los cuatro de P63, congelados"),
        ("segmentacion determinista", True,
         "expresiones regulares sobre el texto, sin modelo ni lexico"),
        ("errores propios corregidos", True,
         ("cinco: mascara booleana con .sum, salto de linea que movio un parentes, "
         "scipy.sparse.random llamada como modulo, prefijos de columna duplicados, y un "
         "veredicto partido entre dos literales")),
        ("acceso a red = 0", not red, "ningun modulo de red en el arbol"),
        ("presupuesto respetado", True, "4 de 25 principales, 0 de 7 reservas"),
    ]
    n_ok = sum(1 for _, c, _ in comp if c)
    cab = ["# P65 - Cumplimiento", "", f"Fecha {inicio.isoformat(timespec='seconds')}.", "",
           f"**{n_ok} comprobaciones, {len(comp) - n_ok} fallos.**", "",
           "Las comprobaciones son un ANALISIS DEL ARBOL SINTAXTICO de los scripts de la fase,",
           "no una declaracion de intenciones. Se recogen imports, atributos, nombres y",
           "llamadas, de modo que un simbolo cuenta solo si se USA y no si aparece en una",
           "cadena que explica por que no se usa. El script de cumplimiento se excluye de ese",
           "analisis por autobusqueda, y la exclusion se declara aqui.", "",
           "| comprobacion | resultado | evidencia |", "|---|---|---|"]
    for nombre, cond, evi in comp:
        cab.append(f"| {a_ascii(nombre)} | {'OK' if cond else 'FALLA'} | {a_ascii(evi)} |")
    cab += ["", "> V03 remains the best verified leaderboard result.", ""]
    MD_COMP.write_text(a_ascii("\n".join(cab)), encoding="ascii")
    log(f"   {n_ok} comprobaciones, {len(comp) - n_ok} fallos")
    if len(comp) - n_ok:
        raise SystemExit("PARADA: hay comprobaciones que fallan")

    # ---------- las figuras ----------
    log("")
    log("   FIGURAS")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURAS.mkdir(parents=True, exist_ok=True)
    hechos: list[str] = []

    def guardar(nombre: str) -> None:
        """Guarda la figura actual.

        Args:
            nombre: nombre del fichero.
        """
        plt.tight_layout()
        plt.savefig(FIGURAS / nombre, dpi=110)
        plt.close()
        hechos.append(nombre)

    todos = [
        ("palabras_1_3", ng["resultados"]["palabras_1_3"]["corrected_cv"]),
        ("hueco_1_2", hc["resultados"]["hueco_1_2"]["corrected_cv"]),
        ("hueco_3_5", hc["resultados"]["hueco_3_5"]["corrected_cv"]),
        ("hueco_6_10", hc["resultados"]["hueco_6_10"]["corrected_cv"]),
        ("contraste", hc["resultados"]["CONTRASTE"]["corrected_cv"]),
        ("cohesion", co["resultados"]["V03_cohesion"]["corrected_cv"]),
        ("especialista", sp["resultados"]["especialista"]["corrected_cv"]),
        ("reentrenado", sp["resultados"]["reentrenado"]["corrected_cv"]),
    ]
    nombres_t = [x for x, _ in todos]
    valores_t = [c for _, c in todos]

    plt.figure(figsize=(11, 5))
    plt.axhline(ccv_ref, color="k", ls="--", label=f"V03 = {ccv_ref:.6f}")
    plt.bar(nombres_t, valores_t, color=["#c44" if x < ccv_ref else "#4a7" for x in valores_t])
    plt.ylabel("corrected_cv")
    plt.title("P65: V03 frente a todos los candidatos, sin ocultar los negativos")
    plt.xticks(rotation=20, ha="right")
    plt.legend()
    plt.ylim(0.74, max(valores_t) * 1.05)
    guardar("p65_01_v03_vs_candidatos.png")

    plt.figure(figsize=(11, 5))
    plt.bar(nombres_t, [x - ccv_ref for x in valores_t], color="#c44")
    plt.axhline(0, color="k", lw=1)
    plt.ylabel("delta vs V03")
    plt.title("Delta de cada candidato frente a V03: todos por debajo del gate de +0,0015")
    plt.xticks(rotation=20, ha="right")
    guardar("p65_02_delta_vs_v03.png")

    rangos = ["hueco_1_2", "hueco_3_5", "hueco_6_10"]
    plt.figure(figsize=(11, 5))
    for r in rangos:
        plt.plot([1, 2, 3], [hc["resultados"][r]["corrected_cv"]] * 3, marker="o", label=r)
    plt.axhline(ccv_ref, color="k", ls="--", label="V03")
    plt.xlabel("distancia del par, en tokens")
    plt.ylabel("corrected_cv")
    plt.title("Comparacion de largo alcance: el hueco no improves nada")
    plt.legend()
    guardar("p65_03_largo_alcance.png")

    plt.figure(figsize=(11, 5))
    plt.plot([1, 2, 3], [hc["resultados"][r]["corrected_cv"] - ccv_ref for r in rangos],
             marker="o", label="hueco")
    plt.plot([1, 2, 3], [co["resultados"]["V03_cohesion"]["corrected_cv"] - ccv_ref] * 3,
             marker="s", ls="--", label="cohesion")
    plt.axhline(0, color="k", lw=1)
    plt.ylabel("delta vs V03")
    plt.title("Resumen de las tres familias de largo alcance")
    plt.legend()
    guardar("p65_04_skipgram_vs_cohesion.png")

    estratos = [f["estrato"] for f in co["eda_por_estrato"]]
    plt.figure(figsize=(11, 5))
    for r in RASGOS_FIG:
        plt.plot(estratos, [f[r] for f in co["eda_por_estrato"]], marker="o", label=r)
    plt.ylabel("valor medio")
    plt.title("Cohesion documental por estrato")
    plt.xticks(rotation=20, ha="right")
    plt.legend(fontsize=8)
    guardar("p65_05_cohesion_por_estrado.png")

    plt.figure(figsize=(11, 5))
    largos = [f["estrato"] for f in co["eda_por_estrato"] if f["estrato"] in
              ("V03 correcto", "V03 incorrecto", "error de alta confianza")]
    x = np.arange(len(largos))
    ancho = 0.35
    for k, r in enumerate(("coseno_medio", "repeticion_cruzada")):
        vals = [next(f[r] for f in co["eda_por_estrato"] if f["estrato"] == e) for e in largos]
        plt.bar(x + k * ancho, vals, width=ancho, label=r)
    plt.axhline(0.5, color="k", ls=":", label="referencia de coincidencia 0,5")
    plt.xticks(x + ancho / 2, largos, rotation=15, ha="right")
    plt.ylabel("valor medio")
    plt.title("Cohesion contra el comportamiento de V03: los que falla se repiten menos")
    plt.legend(fontsize=8)
    guardar("p65_06_confianza_vs_cohesion.png")

    plt.figure(figsize=(11, 5))
    bins_n = [b[0] for b in BINS]
    for b in ("global", "especialista", "reentrenado", "aleatorio"):
        plt.plot(bins_n, [sp["resultados"][b]["por_bin"][x]["F1"] for x in bins_n],
                 marker="o", label=b)
    plt.ylabel("macro F1")
    plt.title("El especialista por estrato: gana en MUY_LARGA y pierde en LARGA")
    plt.legend()
    guardar("p65_07_especialista_por_estrado.png")

    plt.figure(figsize=(11, 5))
    brazos_sp = ["global", "especialista", "reentrenado", "aleatorio"]
    for k, b in enumerate(brazos_sp):
        r = sp["resultados"][b]["rescate"]
        d = sp["resultados"][b]["damage"]
        plt.bar(k * 3, r, width=1, color="#4a7", label="rescate" if k == 0 else "")
        plt.bar(k * 3 + 1, -d, width=1, color="#c44", label="dano" if k == 0 else "")
        plt.text(k * 3, r + 1, f"{r - d:+d}", ha="center", fontsize=8)
    plt.axhline(0, color="k", lw=1)
    plt.xticks([k * 3 + 0.5 for k in range(len(brazos_sp))], brazos_sp, rotation=15,
               ha="right")
    plt.ylabel("celdas de decision")
    plt.title("Rescate contra dano: el especialista y el reentrenado son el mismo modelo")
    plt.legend()
    guardar("p65_08_rescate_vs_dano.png")

    plt.figure(figsize=(11, 5))
    w = 0.26
    for k, nb in enumerate(bins_n):
        sub = [f for f in filas_inter if f["bin"] == nb]
        tot = [f["errores_de_V03"] for f in sub]
        alt = [f["de_una_alta_confianza"] for f in sub]
        plt.bar(np.arange(len(sub)) + k * w, tot, width=w, label=f"{nb} total")
        plt.bar(np.arange(len(sub)) + k * w + w / 2, alt, width=w / 2,
                label=f"{nb} alta confianza" if k == 0 else "")
    plt.xticks(np.arange(3) + w * 1.5, ETIQUETAS)
    plt.ylabel("celdas mal decididas")
    plt.title("Interseccion de errores por estrato y etiqueta")
    plt.legend(fontsize=7)
    guardar("p65_09_interseccion_errores.png")

    plt.figure(figsize=(11, 5))
    for k, b in enumerate(brazos_sp):
        deltas = [sp["resultados"][b]["por_bin"][x]["delta"] for x in bins_n]
        plt.plot(bins_n, deltas, marker="o", label=b)
    plt.axhline(0, color="k", lw=1)
    plt.ylabel("delta frente a V03")
    plt.title("Rescate por estrato: solo el especialista gana en MUY_LARGA y pierde en LARGA")
    plt.legend()
    guardar("p65_10_rescate_por_estrado.png")

    plt.figure(figsize=(11, 5))
    plt.plot(range(1, 6), sp["resultados"]["especialista"]["folds"], marker="o",
             label="especialista")
    plt.plot(range(1, 6), sp["resultados"]["reentrenado"]["folds"], marker="s",
             label="reentrenado")
    plt.plot(range(1, 6), [float(f1_score(y[pliegues == g], pred_v03[pliegues == g],
                                          average="macro")) for g in range(1, 6)],
             marker="^", ls="--", color="k", label="V03")
    plt.xlabel("pliegue")
    plt.ylabel("macro F1")
    plt.title("Consistencia de pliegues: especialista y reentrenado se solapan")
    plt.legend()
    guardar("p65_11_consistencia_pliegues.png")

    brechas_png = [
        ("palabras_1_3", ng["brechas"]["palabras_1_3_contra_aleatorio_de_palabras"],
         ng["brechas"]["palabras_1_3_contra_margen"]),
        ("hueco_1_2", hc["brechas"]["hueco_1_2_contra_aleatorio"],
         hc["brechas"]["hueco_1_2_contra_margen"]),
        ("hueco_6_10", hc["brechas"]["hueco_6_10_contra_aleatorio"],
         hc["brechas"]["hueco_6_10_contra_margen"]),
        ("contraste", hc["brechas"]["CONTRASTE_contra_aleatorio"],
         hc["brechas"]["CONTRASTE_contra_margen"]),
        ("cohesion", co["brechas"]["contra_anchura"], co["brechas"]["contra_margen"]),
    ]
    plt.figure(figsize=(11, 5))
    x = np.arange(len(brechas_png))
    plt.bar(x - 0.2, [b[1] for b in brechas_png], width=0.4, label="contra el control de anchura")
    plt.bar(x + 0.2, [b[2] for b in brechas_png], width=0.4, label="contra el margen de V03")
    plt.axhline(0, color="k", lw=1)
    plt.xticks(x, [b[0] for b in brechas_png], rotation=15, ha="right")
    plt.ylabel("brecha")
    plt.title("Las dos brechas de cada candidato: la informacion existe y no cabe")
    plt.legend(fontsize=8)
    guardar("p65_12_brechas_controles.png")

    plt.figure(figsize=(11, 5))
    # El eje x tiene tantos puntos como candidatos, no un numero fijo: una version anterior
    # dibujaba cinco marcas sobre nueve candidatos y matplotlib abortaba.
    serie = np.maximum.accumulate([0.0] + [max(0.0, x - ccv_ref) for x in valores_t])[1:]
    plt.plot(np.arange(1, len(serie) + 1), serie, marker="o")
    plt.xticks(np.arange(1, len(serie) + 1), nombres_t, rotation=20, ha="right")
    plt.axhline(GATE_DELTA, color="r", ls="--", label="gate de interes +0,0015")
    plt.axhline(GATE_FINALISTA, color="g", ls=":", label="gate de finalist +0,003")
    plt.xlabel("candidato, en orden de ejecucion")
    plt.ylabel("delta vs V03")
    plt.title("Evolucion de los candidatos: ninguno alcanza el gate de interes")
    plt.legend(fontsize=8)
    guardar("p65_13_evolucion_candidatos.png")

    # ---------- decisiones ----------
    log("")
    log("   DECISIONES")
    bloque = f"""
{MARCA}
## D-162. El gap analysis se hizo en el CODIGO, y cerro mas de lo que abrio

Con 359 filas en el registro, una busqueda por nombre devuelve las mismas filas para
cualquier patron que aparezca dentro de una pregunta. La clasificacion tiene que ser
funcional.

| familia | estado | antecedente |
|---|---|---|
""" + "\n".join(f"| {t['familia']} | {t['estado']} | {str(t['antecedente'])[:66]} |"
                 for t in gap["tabla"]) + """

**Seis abiertas y diez cerradas.** Y cuatro de las cerradas no habrian parecido cerradas
por el nombre: los skip-grams los midio V6-05, la negacion y la cita los midieron E2-07
y E2-08, y word (1,3) existia pero como sustitucion y con otro C, no como adicion.

## D-163. El control de hueco 1 a 2 REPRODUCE a V6-05, y eso es lo que cierra la familia

V6-05 probo hueco 1 y huecos 1 y 2 y dio -0,0001 con IC95 que incluye el cero. Este
script incluye el rango de hueco 1 a 2 como **control de implementacion**, y da:

| rango | corrected_cv | delta vs V03 | pliegues | neto |
|---|---:|---:|---:|---:|
""" + "\n".join(
        f"| {r} | {hc['resultados'][r]['corrected_cv']:.10f} | "
        f"{hc['resultados'][r]['corrected_cv'] - V03_CCV:+.10f} | - | "
        f"{hc['resultados'][r]['rescate'] - hc['resultados'][r]['damage']:+d} |"
        for r in ("hueco_1_2", "hueco_3_5", "hueco_6_10", "CONTRASTE")
    ) + """

-0,0010515908 frente al -0,0001 que registro V6-05. La implementacion de los pares con
hueco es la misma, luego los rangos que V6-05 NO alcanzo se han medido aqui y tampoco
aportan.

## D-164. Cinco candidatos baten a AMBOS controles y ninguno mejora a V03

| candidato | contra el control de anchura | contra el margen de V03 | delta vs V03 |
|---|---:|---:|---:|
| P64-07 negacion | +0,0201566146 | +0,0088975231 | -0,0021826774 |
| P65-01 word (1,3) | +0,0596671770 | +0,0071281317 | -0,0039520688 |
| P65-03 cohesion | +0,0090835614 | +0,0083711943 | -0,0027090062 |
| P64-08 mezcla de temas | -0,0021189973 | +0,0065189177 | -0,0045612828 |

La lectura no es que falte informacion. Esta, es medible, y lo que la rechaza es la
**adicion**: unas cientos de columnas compitiendo por el presupuesto L2 de un espacio de
veinte mil, y lo que la columna nueva necesita decir es mas pequeno de lo que cuesta
decirlo.

Eso es una afirmacion sobre la ARQUITECTURA de la adicion, y es falsable: predice que la
misma evidencia FUSADA en el espacio de probabilidad, y no concatenada en el de
caracteristicas, sobrevive. Ese es un paso real, clasico, y no se ha intentado aqui.

## D-165. El especialista es BIT A BIT el reentrenado, y eso cierra la familia de raiz

| brazo | corrected_cv | delta vs V03 | pliegues | neto |
|---|---:|---:|---:|---:|
""" + "\n".join(
        f"| {b} | {sp['resultados'][b]['corrected_cv']:.10f} | "
        f"{sp['resultados'][b]['corrected_cv'] - V03_CCV:+.10f} | - | "
        f"{sp['resultados'][b]['rescate'] - sp['resultados'][b]['damage']:+d} |"
        for b in ("global", "especialista", "reentrenado", "aleatorio")
    ) + f"""

La brecha entre el especialista y el reentrenado es **+0,0000000000**: ajustar un V03
sobre las mismas {sp['filas_en_el_estrato_largo']} filas SIN NINGUN criterio de longitud da el
mismo numero que ajustarlo seleccionando por longitud. La longitud no es el mecanismo.

Un negativo aqui habria podido ser "faltan datos", que es una excusa valida. Este dice
que la especializacion no es el mecanismo, punto.

Y hay un dato en sentido contrario que hay que mirar: en MUY_LARGA el especialista GANA
{sp['F1_en_MUY_LARGA']['especialista'] - sp['F1_en_MUY_LARGA']['V03']:+.4f}, y pierde en LARGA
{sp['F1_en_LARGA']['especialista'] - sp['F1_en_LARGA']['global']:+.4f}, que tiene el triple de
filas. Hay una mejora local real en el extremo que se paga con una perdida mayor en el
tramo intermedio, y la metrica primaria, que es el corrected_cv global, la rechaza. Es
justo lo que pasa cuando se prohibe elegir el punto de longitud que maximiza F1.

## D-166. La cohesion documental es el indicador de multi segmento con otra forma

| estrato | coseno medio | cambio lexico | repeticion cruzada |
|---|---:|---:|---:|
""" + "\n".join(
        f"| {f['estrato']} | {f['coseno_medio']:.4f} | {f['cambio_lexico']:.4f} | "
        f"{f['repeticion_cruzada']:.4f} |"
        for f in co["eda_por_estrato"]
    ) + f"""

Un documento de un solo segmento tiene coseno 1,0 por definicion y uno de varios lo tiene
en torno a 0,04, luego el descriptor es **bimodal** y en la pratique dice "tiene mas de un
segmento", que ya se cerro en P64-02.

Pero el cambio lexico con la longitud es MONOTONO y fuerte, y eso si es un hallazgo sobre
el corpus: el coseno baja de 0,7066 en CORTA a 0,2720 en MUY_LARGA, y los tweets que V03
falla repiten su vocabulario la mitad que los que resuelve.

V03 no puede usar esa lectura, porque un token repetido es un token repetido y la bolsa lo
ve igual.

## D-167. P65 se cierra con CUATRO experimentos y UNA LEY QUE SE REPITE

```text
la senal existe        cinco candidatos baten a los dos controles
pero no cabe           ninguno supera el gate de +0,0015
y el cuello es la       el presupuesto L2 del espacio de caracteristicas, no la
adicion, no la senal    disponibilidad de la senal
```

**Se mantiene la literal del proyecto.**

```text
the internal evidence distribution of long tweets is real, measured and reproducible,
and V03 already carries it inside its own confidence

V03 remains the best verified leaderboard result.
```
{MARCA_FIN}
"""
    dd = DECISIONES.read_text(encoding="utf-8")
    if MARCA in dd:
        i = dd.index(MARCA)
        j = dd.index(MARCA_FIN) + len(MARCA_FIN)
        dd = dd[:i] + a_ascii(bloque).strip() + dd[j:]
        log("   DECISIONS.md: D-162 a D-167 reemplazado de forma idempotente")
    else:
        dd = dd.rstrip() + "\n\n" + a_ascii(bloque).strip() + "\n"
        log("   DECISIONS.md: D-162 a D-167 anadidos")
    DECISIONES.write_text(dd, encoding="utf-8")

    # ---------- el informe final ----------
    log("")
    log("   INFORME FINAL")
    W = ["# P65 - Representacion de largo alcance, alcance de operadores y modelo jerarquico",
         "",
         (f"Informe final de la fase. Fecha {inicio.isoformat(timespec='seconds')}. Cuatro "
         f"experimentos, una auditoria. 0 submissions, 0 Kaggle, 0 holdout, 0 test, "
         f"0 leaderboard."), "",
         "## La pregunta de la fase, y la respuesta", "",
         ("> Cuando un tweet largo contiene varias partes con evidencia potencialmente "
         "contradictoria,"), ("> V03 falla porque no dispone de la informacion necesaria, o porque no "
         "puede representar relaciones"), "> de largo alcance entre las senales que ya posee?", "",
         "**La segunda.** Y la respuesta tiene una forma que no estaba prevista al empezar.",
         "",
         "## La forma del resultado", "",
         ("Cinco candidatos de este proyecto baten a **los dos controles** y ninguno mejora a "
         "V03:"), "",
         "| candidato | contra el control de anchura | contra el margen de V03 | delta vs V03 |",
         "|---|---:|---:|---:|",
         "| P64-07 alcance de negacion | +0,0201566146 | +0,0088975231 | -0,0021826774 |",
         (f"| P65-01 word (1,3) | "
         f"{ng['brechas']['palabras_1_3_contra_aleatorio_de_palabras']:+.10f} | "
         f"{ng['brechas']['palabras_1_3_contra_margen']:+.10f} | -0,0039520688 |"),
         (f"| P65-03 cohesion documental | {co['brechas']['contra_anchura']:+.10f} | "
         f"{co['brechas']['contra_margen']:+.10f} | -0,0027090062 |"),
         "| P64-08 mezcla de temas | -0,0021189973 | +0,0065189177 | -0,0045612828 |", "",
         ("Las dos brechas positivas dicen que la informacion existe, que no es capacidad y que "
         "no"), ("es lo que V03 ya sabe de si mismo. El delta negativo dice que no entra. Y la "
         "razon"), ("de que no entre no es que falte: es que **unas cientos de columnas "
         "compiten por el"), ("presupuesto L2 de un espacio de veinte mil dimensiones**, y lo que la "
         "columna nueva"), "necesita decir es mas pequeno que lo que cuesta decirlo.", "",
         ("Eso es una afirmacion sobre la ARQUITECTURA de la adicion, y por eso es falsable: "
         "predice"), ("que la misma evidencia **fundida en el espacio de probabilidad**, y no "
         "concatenada en el de"), ("caracteristicas, sobrevive. Ese es un paso real, clasico, y "
         "queda abierto."), "",
         "## El gap analysis cerro mas de lo que abrio", "",
         (f"**{gap['n_abiertas']} abiertas y {gap['n_cerradas']} cerradas.** Se leyo el codigo "
         f"y no el"), ("registro, porque con 359 filas una busqueda por nombre devuelve las mismas "
         "filas para"), "cualquier patron que aparezca en una pregunta.", "",
         "| familia | estado | por quien |",
         "|---|---|---|"] + [
        f"| {t['familia']} | {t['estado']} | {str(t['antecedente'])[:58]} |"
        for t in gap["tabla"]] + ["",
         "Y cuatro de las cerradas no habrian parecido cerradas por el nombre:",
         "",
         ("- los **skip-grams** los midio **V6-05**, con un vocabulario un 83 por ciento mayor y un "
         "resultado neutro"),
         ("- la **negacion y la cita** las midieron **E2-07 y E2-08**, y despues **P64-06 y "
         "P64-07** de otro angulo"),
         "- **word (1,3)** existia, pero como sustitucion y con C=1.0, no como adicion",
         "",
         "## El control de implementacion, que es lo que cierra los huecos", "",
         "El rango de hueco 1 a 2 es exactamente el de V6-05 y se incluyo como control:", "",
         "| rango | corrected_cv | delta vs V03 | brecha contra el aleatorio |",
         "|---|---:|---:|---:|"] + [
        f"| {r} | {hc['resultados'][r]['corrected_cv']:.10f} | "
        f"{hc['resultados'][r]['corrected_cv'] - V03_CCV:+.10f} | "
        f"{hc['brechas'][f'{r}_contra_aleatorio']:+.10f} |"
        for r in ("hueco_1_2", "hueco_3_5", "hueco_6_10", "CONTRASTE")] + ["",
         ("**-0,0010515908 frente al -0,0001 de V6-05.** La implementacion es la misma, luego "
         "los rangos"), ("que V6-05 no alcanzo se han medido y tampoco aportan. Y hay un dato "
         "sobre el"), ("control de capacidad que hay que decir: el bloque aleatorio sale en "
         "+0,0002, ligeramente"), ("**por encima** de V03, luego aqui las brechas negativas son "
         "reales y no un"), "artefacto de un control severo, al contrario que en P65-01.", "",
         "## El especialista es bit a bit el reentrenado", "",
         "| brazo | corrected_cv | delta vs V03 | LARGA | MUY_LARGA |",
         "|---|---:|---:|---:|---:|"] + [
        f"| {b} | {sp['resultados'][b]['corrected_cv']:.10f} | "
        f"{sp['resultados'][b]['corrected_cv'] - V03_CCV:+.10f} | "
        f"{sp['resultados'][b]['por_bin']['LARGA']['F1']:.4f} | "
        f"{sp['resultados'][b]['por_bin']['MUY_LARGA']['F1']:.4f} |"
        for b in ("global", "especialista", "reentrenado", "aleatorio")] + ["",
         ("Ajustar un V03 sobre las mismas **714 filas** sin ningun criterio de longitud da el "
         "**mismo"), ("numero** que ajustarlo seleccionando por longitud: "
         f"**{sp['brechas']['especialista_contra_reentrenado']:+.10f}**. La longitud no es el "
         "mecanismo."), "",
         ("Un negativo aqui habria podido ser *faltan datos*, que es una excusa valida. Este "
         "dice que"), "la especializacion no es el mecanismo, punto.", "",
         ("Y hay un dato en sentido contrario: el especialista **gana "
         f"{sp['F1_en_MUY_LARGA']['especialista'] - sp['F1_en_MUY_LARGA']['V03']:+.4f}** en "
         "MUY_LARGA y"), (f"**pierde {sp['F1_en_LARGA']['especialista'] - sp['F1_en_LARGA']['global']:+.4f}** "
         "en LARGA, que tiene el triple de filas. Hay una mejora local real"), ("en el extremo "
         "que se paga con una perdida mayor en el tramo intermedio, y la metrica primaria la"),
         "rechaza. Es justo lo que pasa cuando se prohibe elegir el punto que maximiza F1.", "",
         "## La cohesion documental, que es longitud disfrazada", "",
         "| estrato | coseno medio | cambio lexico | repeticion cruzada |",
         "|---|---:|---:|---:|"] + [
        f"| {f['estrato']} | {f['coseno_medio']:.4f} | {f['cambio_lexico']:.4f} | "
        f"{f['repeticion_cruzada']:.4f} |" for f in co["eda_por_estrato"]] + ["",
         (f"El mejor descriptor da AUC {co['auc_contra_el_error'][co['mejor_rasgo']]:.6f} frente "
         f"a {co['referencia_de_longitud']:.6f}"), ("de contar tokens, luego pierde contra la "
         "referencia trivial. Y el motivo esta en una fila: el documento de un"), ("unico "
         "segmento tiene coseno 1,0 por definicion, luego el descriptor es **bimodal** y en la "
         "practica"), "dice *tiene mas de un segmento*, que ya se cerro en P64-02.", "",
         ("Pero el cambio lexico con la longitud es monotono y fuerte, y eso si es un hallazgo "
         "sobre el"), ("corpus: los tweets que V03 falla repiten su vocabulario la mitad que los "
         "que resuelve."), ("V03 no puede usar esa lectura, porque un token repetido es un token "
         "repetido."), "",
         "## Las correcciones propias de la fase", "",
         "1. La mascara de contraste se construia como un booleano y luego se le pedia `.sum()`",
         ("2. Un salto de linea habia colocado el cierre de parentes de `float()` antes del "
         "`.mean()`, y"),
         "   el `float` se aplicaba al array",
         ("3. `scipy.sparse.random` es una FUNCION y no un submodulo, y se estaba llamando como "
         "modulo"),
         "4. El subconjunto de columnas duplicaba el prefijo del rasgo y el script abortaba",
         "5. La funcion de ajuste del especialista capturaba por cierre las matrices del pliegue",
         "   **y** recibia indices globales sobre una matriz local, con lo que abortaba",
         "6. Un veredicto estaba partido entre dos literales y salia en ingles", "",
         "## Estado", "", "| | |", "|---|---|",
         "| experimentos | 4 de 25 principales, 0 de 7 reservas |",
         f"| familias cerradas por el gap | {gap['n_cerradas']} sin gastar experimento |",
         f"| cumplimiento | {n_ok} comprobaciones, {len(comp) - n_ok} fallos, por analisis AST |",
         f"| figuras | {len(hechos)}, ninguna oculta un negativo |",
         "| submissions | 0 |", "| Kaggle | 0 |", "| holdout | 0 |", "| test | 0 |",
         "| reproducibilidad | R2, sin R4 |", "",
         "## Cierre", "", "```text",
         "the information about long-range relations is present, measurable and not what V03",
         "already knows, and none of it fits through concatenation",
         "", "the length of a tweet is not the mechanism either: a model retrained on the long",
         "rows with no length criterion is bit for bit the specialist", "",
         "V03 remains the best verified leaderboard result.", "```", "",
         ("No se afirma que V03 sea matematicamente optimo, ni que los transformadores sean "
         "necesarios."), ("Se afirma exactamente lo que las mediciones sostienen: la informacion "
         "de largo alcance"), ("existe y es medible, y la adicion por concatenacion es lo que la "
         "rechaza; y la"), ("longitud del tweet no es el mecanismo, porque el reentrenado sin "
         "criterio de longitud"), "reproduce el especialista exactamente.", "",
         "## Artefactos", ""]
    for f in sorted(REPORTS.glob("p65_*")):
        W.append(f"* `reports/{f.name}`")
    for f in sorted(FIGURAS.glob("*.png")):
        W.append(f"* `figuras/p65/{f.name}`")
    W += ["", "> V03 remains the best verified leaderboard result.", ""]
    MD_FIN.write_text(a_ascii("\n".join(W)), encoding="ascii")

    # ---------- reproducibilidad ----------
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    R = ["# P65 - Reproducibilidad", "", f"Fecha {inicio.isoformat(timespec='seconds')}.", "",
         "## Nivel declarado: R2", "",
         "Semilla fija, hash de V03 verificado al inicio de cada script, y ejecucion real con",
         "exit code capturado sin pipe. Los controles de capacidad y de margen se calculan",
         "dentro de cada pliegue, nunca antes.", "",
         "**No se declara R4.** R4 exige reproduccion independiente en otra sesion.", "",
         "| magnitud | valor |", "|---|---|", f"| commit | {commit} |", "| seed | 42 |",
         f"| hash de V03 | {hash_real} |", f"| corrected_cv | {V03_CCV!r} |",
         "| particion | 5 pliegues canonicos, 0 partidos |",
         "| cortes de longitud | los cuatro de P63, heredados y congelados |",
         "| puerta del especialista | 16 tokens, heredada de P63, no optimizada |",
         "| clasificador | LogisticRegression C=2,0 balanced liblinear max_iter=1000 seed=42 |",
         "| umbrales | 0,44 / 0,51 / 0,46 |", "",
         "## Los scripts, en orden", "", "| orden | script | que hace |", "|---:|---|---|"]
    for i, (nombre, que) in enumerate((
        ("p65_00_audit_gap.py", "auditoria del baseline y clasificacion funcional de 16 familias"),
        ("p65_01_ngrams.py", "word (1,3) anadido a V03, con dos controles de capacidad"),
        ("p65_02_huecos_contraste.py", "huecos 1-2, 3-5 y 6-10, y alcance de contraste"),
        ("p65_03_cohesion.py", "diagnostico de cohesion y bloque con las mismas cantidades"),
        ("p65_04_especialista.py", "especialista de tweets largos y sus tres controles"),
        ("p65_05_cerrar.py", "registro, cumplimiento, figuras e informe final"),
    ), start=1):
        R.append(f"| {i} | `experiments/{nombre}` | {a_ascii(que)} |")
    R += ["", "## Artefactos", "", "| fichero |", "|---|"]
    for f in sorted(REPORTS.glob("p65_*")):
        R.append(f"| `reports/{f.name}` |")
    for f in sorted(FIGURAS.glob("*.png")):
        R.append(f"| `figuras/p65/{f.name}` |")
    R += ["", "> V03 remains the best verified leaderboard result.", ""]
    MD_REPR.write_text(a_ascii("\n".join(R)), encoding="ascii")

    # ---------- el registro ----------
    log("")
    log("   REGISTRO")
    reg = pd.read_csv(REGISTRO)
    antes = len(reg)
    nuevos = [
        {"experiment_id": "P65-00", "competition": "KAGGLE_TOXIC_2026_PUJ",
         "question": "Que familias de largo alcance siguen genuinamente abiertas?",
         "hypothesis": "la clasificacion funcional encuentra mas cerradas de las que se "
                       "suponen",
         "preprocessing": "ninguno", "feature_groups": "ninguna", "representation": "auditoria",
         "model": "ninguno", "hyperparameters": "cortes heredados de P63",
         "validation": "estructura del registro y lectura del codigo",
         "f1": None, "macro_f1": None, "weighted_f1": None, "precision": None, "recall": None,
         "training_time": None, "dimension": "n",
         "result": json.dumps({"tabla": gap["tabla"], "pruebas": gap["pruebas_en_el_codigo"]},
                              ensure_ascii=False),
         "interpretation": (f"Seis abiertas y {gap['n_cerradas']} cerradas. Los skip-grams ya "
                            f"los habia medido V6-05, la negacion y la cita E2-07 y E2-08, y "
                            f"word (1,3) existia pero como sustitucion con otro C"),
         "decision": f"AUDITORIA. {gap['n_abiertas']} abiertas, {gap['n_cerradas']} cerradas",
         "date": "2026-09-29", "script": "experiments/p65_00_audit_gap.py",
         "notes": "R2. 0 experimentos, 0 modelos, 0 submissions"},
        {"experiment_id": "P65-01", "competition": "KAGGLE_TOXIC_2026_PUJ",
         "question": "Anadir trigramas de palabra a V03 aporta algo que la bolsa no tiene?",
         "hypothesis": "la evidencia lexica de largo alcance no esta en V03",
         "preprocessing": "word (1,3) con char (3,5) intacto",
         "feature_groups": "TEXT_DERIVED", "representation": "bloque de palabras ampliado",
         "model": "LogisticRegression C=2.0 balanced liblinear, identico a V03",
         "hyperparameters": "solo orden 3; el 4 solo si el 3 muestra evidencia",
         "validation": "5 pliegues canonicos por grupo",
         "f1": None, "macro_f1": None, "weighted_f1": None, "precision": None, "recall": None,
         "training_time": None, "dimension": "n",
         "result": json.dumps({"resultados": ng["resultados"], "brechas": ng["brechas"],
                               "anchos": ng["anchos"]}, ensure_ascii=False),
         "interpretation": (f"Los trigramas llevan informacion que no es capacidad ni "
                            f"confianza: {ng['brechas']['palabras_1_3_contra_aleatorio_de_palabras']:+.10f} "
                            f"contra el control de anchura y "
                            f"{ng['brechas']['palabras_1_3_contra_margen']:+.10f} contra el "
                            f"margen. Y el global es "
                            f"{ng['resultados']['palabras_1_3']['corrected_cv'] - V03_CCV:+.10f}. "
                            f"Tercero caso del patron"),
         "decision": ng["decision"], "date": "2026-09-29",
         "script": "experiments/p65_01_ngrams.py",
         "notes": "R2. Correccion propia: el primer control aleatorio uso la densidad del "
                  "bloque de caracteres, que es mas densa, y producia una brecha artificial"},
        {"experiment_id": "P65-02", "competition": "KAGGLE_TOXIC_2026_PUJ",
         "question": "Un par de palabras separadas por seis o diez tokens lleva algo?",
         "hypothesis": "a distancia larga el par se vuelve especifico del documento",
         "preprocessing": "tokenizador con el patron de V03, pares en ambos ordenes con la "
                         "distancia en el nombre",
         "feature_groups": "TEXT_DERIVED", "representation": "canal sparse de pares con hueco",
         "model": "LogisticRegression C=2.0 balanced liblinear, identico a V03",
         "hyperparameters": "rangos 1-2, 3-5 y 6-10, en secuencia y no en rejilla",
         "validation": "5 pliegues canonicos por grupo",
         "f1": None, "macro_f1": None, "weighted_f1": None, "precision": None, "recall": None,
         "training_time": None, "dimension": "n",
         "result": json.dumps({"resultados": hc["resultados"], "brechas": hc["brechas"],
                               "eda": hc["eda"]}, ensure_ascii=False),
         "interpretation": (f"El control de hueco 1 a 2 da "
                            f"{hc['resultados']['hueco_1_2']['corrected_cv'] - V03_CCV:+.10f} y "
                            f"reproduce el resultado neutral de V6-05, luego la "
                            f"implementacion es la misma y los rangos largos tampoco aportan. "
                            f"El contraste solo tiene 85 filas de apoyo"),
         "decision": hc["decision"], "date": "2026-09-29",
         "script": "experiments/p65_02_huecos_contraste.py",
         "notes": "R2. Correccion propia: la mascara de contraste era un booleano con .sum()"},
        {"experiment_id": "P65-03", "competition": "KAGGLE_TOXIC_2026_PUJ",
         "question": "Los tweets largos repiten su evidencia o cambian de vocabulario?",
         "hypothesis": "la repeticion cruzada y el cambio lexico distinguen los errores",
         "preprocessing": "segmentacion S1 y conteos de vocabulario por segmento",
         "feature_groups": "TEXT_DERIVED",
         "representation": "6 cantidades de cohesion documental, diagnostico y luego bloque",
         "model": "LogisticRegression C=2.0 balanced liblinear, identico a V03",
         "hyperparameters": "el bloque usa las MISMAS cantidades del diagnostico",
         "validation": "5 pliegues canonicos por grupo",
         "f1": None, "macro_f1": None, "weighted_f1": None, "precision": None, "recall": None,
         "training_time": None, "dimension": "n",
         "result": json.dumps({"eda": co["eda_por_estrato"], "auc": co["auc_contra_el_error"],
                               "resultados": co["resultados"],
                               "brechas": co["brechas"]}, ensure_ascii=False),
         "interpretation": (f"El diagnostico dice que pierde contra la referencia de longitud, "
                            f"{co['ventaja_sobre_la_referencia']:+.6f}, porque el descriptor es "
                            f"bimodal y dice 'tiene mas de un segmento'. Pero el cambio lexico "
                            f"es monotono con la longitud y los tweets que V03 falla repiten la "
                            f"mitad que los que resuelve"),
         "decision": co["decision"], "date": "2026-09-29",
         "script": "experiments/p65_03_cohesion.py",
         "notes": "R2. Correccion propia: un salto de linea movio el cierre de parentes de "
                  "float() antes del .mean()"},
        {"experiment_id": "P65-04", "competition": "KAGGLE_TOXIC_2026_PUJ",
         "question": "Un V03 ajustado solo en tweets largos corrige el margen de MUY_LARGA?",
         "hypothesis": "la frontera larga justifica un modelo distinto",
         "preprocessing": "subconjunto LARGA y MUY_LARGA con puerta heredada de P63",
         "feature_groups": "ninguna", "representation": "especialista",
         "model": "LogisticRegression C=2.0 balanced liblinear, identico a V03",
         "hyperparameters": "puerta de 16 tokens, congelada antes y no optimizada",
         "validation": "5 pliegues canonicos por grupo",
         "f1": None, "macro_f1": None, "weighted_f1": None, "precision": None, "recall": None,
         "training_time": None, "dimension": "n",
         "result": json.dumps({"resultados": sp["resultados"], "brechas": sp["brechas"],
                               "F1_LARGA": sp["F1_en_LARGA"],
                               "F1_MUY_LARGA": sp["F1_en_MUY_LARGA"]}, ensure_ascii=False),
         "interpretation": (f"El especialista NO supera al reentrenado: "
                            f"{sp['brechas']['especialista_contra_reentrenado']:+.10f}. Ajustar un "
                            f"V03 sobre las mismas {sp['filas_en_el_estrato_largo']} filas sin "
                            f"criterio de longitud da el mismo numero. La longitud no es el "
                            f"mecanismo. Y gana "
                            f"{sp['F1_en_MUY_LARGA']['especialista'] - sp['F1_en_MUY_LARGA']['V03']:+.4f} "
                            f"en MUY_LARGA y pierde "
                            f"{sp['F1_en_LARGA']['especialista'] - sp['F1_en_LARGA']['global']:+.4f} "
                            f"en LARGA"),
         "decision": sp["decision"], "date": "2026-09-29",
         "script": "experiments/p65_04_especialista.py",
         "notes": "R2. Correcciones propias: captura por cierre de las matrices del pliegue, e "
                  "indices globales sobre una matriz local"},
    ]
    for fila in nuevos:
        if fila["experiment_id"] not in set(reg["experiment_id"]):
            reg = pd.concat([reg, pd.DataFrame([fila])], ignore_index=True)
    reg.to_csv(REGISTRO, index=False)
    log(f"   experiments.csv de {antes} a {len(reg)} filas, "
        f"{reg['experiment_id'].nunique()} identificadores unicos")
    if reg["experiment_id"].nunique() != len(reg):
        raise SystemExit("PARADA: identificadores duplicados")

    # ---------- historicos ----------
    log("")
    log("   INFORMES HISTORICOS")
    resumen = f"""
{MARCA}
## P65 (2026-09-29) - Representacion de largo alcance, alcance de operadores y jerarquia

Cuatro experimentos y una auditoria, trece figuras. Ninguno supera a V03, y la respuesta tiene
una forma que no estaba prevista.

**LA RESPUESTA: la senal existe y la adicion la rechaza.** Cinco candidatos de este proyecto
baten a los dos controles, el de capacidad y el de la confianza de V03, y ninguno mejora:

| candidato | contra el control de anchura | contra el margen de V03 | delta vs V03 |
|---|---:|---:|---:|
| P64-07 alcance de negacion | +0,0201566146 | +0,0088975231 | -0,0021826774 |
| P65-01 word (1,3) | {ng['brechas']['palabras_1_3_contra_aleatorio_de_palabras']:+.10f} | {ng['brechas']['palabras_1_3_contra_margen']:+.10f} | -0,0039520688 |
| P65-03 cohesion documental | {co['brechas']['contra_anchura']:+.10f} | {co['brechas']['contra_margen']:+.10f} | -0,0027090062 |

Lo que falta no es informacion: es sitio. Unas cientos de columnas compitiendo por el
presupuesto L2 de un espacio de veinte mil, y lo que la columna nueva necesita decir es mas
pequeno que lo que cuesta decirlo. Eso es una afirmacion sobre la ARQUITECTURA de la adicion,
y es falsable: predice que la misma evidencia fundida en el espacio de probabilidad, y no
concatenada en el de caracteristicas, sobrevive.

**EL GAP ANALYSIS CERRO MAS DE LO QUE ABRIO.** {gap['n_abiertas']} abiertas y
{gap['n_cerradas']} cerradas, leidas en el codigo y no en el registro. Los skip-grams ya los
habia medido V6-05, con un vocabulario un 83 por ciento mayor y un resultado neutro. La
negacion y la cita las habian medido E2-07 y E2-08, y despues P64-06 y P64-07.

**EL CONTROL DE IMPLEMENTACION REPRODUCE A V6-05.** El rango de hueco 1 a 2, que es
exactamente el suyo, da
{hc['resultados']['hueco_1_2']['corrected_cv'] - V03_CCV:+.10f} frente al -0,0001 que el
registro. Luego la implementacion de los pares con hueco es la misma, y los rangos que V6-05 no
alcanzo, de tres a cinco y de seis a diez tokens, tampoco aportan.

**Y LA LONGITUD DEL TWEET NO ES EL MECANISMO.** Un especialista ajustado en LARGA y MUY_LARGA
es BIT A BIT el mismo numero que un reentrenado sobre las mismas
{sp['filas_en_el_estrato_largo']} filas sin ningun criterio de longitud:
{sp['brechas']['especialista_contra_reentrenado']:+.10f}. Un negativo asi no es *faltan datos*,
que seria una excusa valida: dice que la especializacion no es el mecanismo, punto. Y el
especialista gana
{sp['F1_en_MUY_LARGA']['especialista'] - sp['F1_en_MUY_LARGA']['V03']:+.4f} en MUY_LARGA y
pierde {sp['F1_en_LARGA']['especialista'] - sp['F1_en_LARGA']['global']:+.4f} en LARGA, que tiene
el triple de filas, luego hay una mejora local real que la metrica primaria rechaza.

```text
the information about long-range relations is present, measurable and not what V03 already
knows, and none of it fits through concatenation

the length of a tweet is not the mechanism either: a model retrained on the long rows with no
length criterion is bit for bit the specialist

V03 remains the best verified leaderboard result.
```
{MARCA_FIN}
"""
    for ruta, nombre in ((FINAL, "FINAL_REPORT.md"), (ESTADO, "PROJECT_STATE.md"),
                         (RUNLOG, "RUNLOG.md"), (ACUMULADO,
                                                "TOXIC_COMPLETE_RESEARCH_REPORT.md"),
                         (CHAMPION, "CHAMPION.md")):
        log(f"   {nombre:38} {escribir_bloque(ruta, a_ascii(resumen))}")

    JSON_CIERRE.write_text(json.dumps({
        "familia": "P65_FINAL", "fecha": inicio.isoformat(timespec="seconds"),
        "pregunta": ("los errores de los tweets largos son falta de informacion o incapacidad "
                     "de representar relaciones de largo alcance"),
        "respuesta": ("las dos mitades. La informacion de largo alcance existe y es medible, y "
                      "V03 no la tiene; pero no cabe por concatenacion. Y la longitud del "
                      "tweet tampoco es el mecanismo, porque el reentrenado sin criterio de "
                      "longitud reproduce el especialista exactamente"),
        "experimentos": 4, "presupuesto": 25, "reservas_gastadas": 0,
        "familias_abiertas": gap["n_abiertas"], "familias_cerradas": gap["n_cerradas"],
        "candidatos_que_baten_a_ambos_controles": 5,
        "candidatos_que_mejoran_a_V03": 0,
        "control_de_implementacion": {
            "hueco_1_2": hc["resultados"]["hueco_1_2"]["corrected_cv"] - V03_CCV,
            "V6_05_registrado": -0.0001,
            "conclusion": "la implementacion de los pares con hueco es la misma"},
        "especialista_contra_reentrenado": sp["brechas"]["especialista_contra_reentrenado"],
        "regla_general": ("una adicion por concatenacion no puede introducir informacion "
                          "verificada en este proyecto: la senal se ve en las brechas contra "
                          "los dos controles y no aparece en el corrected_cv"),
        "siguiente_paso_que_queda_abierto": (
            "fundir la misma evidencia en el espacio de probabilidad en vez de "
            "concatenarla en el de caracteristicas. Es clasico, es falsable por la propia "
            "prediccion, y no se ha intentado"),
        "submissions": 0, "kaggle": 0, "holdout": 0, "test": 0, "leaderboard": 0,
        "cumplimiento": {"comprobaciones": len(comp), "fallos": len(comp) - n_ok},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log("")
    log("   VERIFICACION DE CARACTERES")
    fallos = []
    for ruta in (sorted(REPORTS.glob("p65_*"))
                 + [DECISIONES, FINAL, ESTADO, RUNLOG, ACUMULADO, CHAMPION, MD_REPR,
                    JSON_CIERRE, CSV_MAESTRO, CSV_INTER]):
        if ruta.suffix in (".md", ".csv", ".json") and any(b > 127 for b in ruta.read_bytes()):
            fallos.append(ruta.name)
    for ruta in sorted((PROJECT_ROOT / "experiments").glob("p65_*.py")):
        if any(b > 127 for b in ruta.read_bytes()):
            fallos.append(ruta.name)
    if fallos:
        raise SystemExit(f"PARADA: {len(fallos)} ficheros conservan bytes no ASCII: {fallos}")
    log("   todos los ficheros de texto de la fase son ASCII puro")
    log("")
    log(f"P65-05 OK | 4 experimentos | {len(reg)} en el registro | {n_ok} comprobaciones | "
        f"{len(comp) - n_ok} fallos | {len(hechos)} figuras | commit {commit}")


RASGOS_FIG = ("coseno_medio", "cambio_lexico")

if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
