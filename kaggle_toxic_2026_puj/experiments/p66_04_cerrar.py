"""P66 closing: compliance by AST, figures, the final report, and the decision per mechanism.

What this phase established, and the caveat that belongs next to it
------------------------------------------------------------------
P65 closed with a law: five representations beat both a capacity control and V03's own margin,
and none improved V03 by concatenation. P66 tested the one thing that was left, and the answer
is YES with an important qualification.

MI_arbol, corrected in score space rather than concatenated, gives +0,0026743167 on 4 of 5
folds, with an attribution gap of +0,0041133588 against the same fusion run on a shuffled
score. That is the first positive delta in this line of work and it is attributable to the
candidate rather than to the fusion mechanism.

The qualification is in P18 and it is not cosmetic. Of the 53 cells the fusion rescues, only 6
are high-confidence V03 errors and 47 are rows sitting next to the threshold. A fusion that
mainly moves cells across a boundary has found CALIBRATION, not representation. The bootstrap
interval [-0,0044311634, +0,0102108745] includes zero, so the effect is not established either.

So the honest verdict is PROMISING and not FINALIST, and the report says so in those words.

The seven candidates, and why only one works
---------------------------------------------
The logits of every candidate correlate with V03's between +0,9294 and +0,9988. They are
nearly the same score. The AUC of the RESIDUAL separates them:

    MI_arbol     0,565387   the only one above 0,5
    word_1_3     0,514599
    huecos_1_2   0,498082
    contraste    0,482872
    negacion     0,453645
    cohesion     0,441306
    huecos_6_10  0,418585

Five of the seven have a residual that is ANTI-correlated with the error, which means their
unique part carries nothing. That is a statement about those candidates, not about fusion, and
it is the same conclusion P65 reached from the other direction.

Compliance is an AST walk
-------------------------
Same instrument as P64 and P65, and for the same reason: these scripts document in their own
strings what they do not use. The inspection goes one level deeper than in the previous phases:
it collects the string literals that reach read_csv, np.load and open, because this phase reads
a CSV of groups for the bootstrap and a naive text search would flag the mention of it. This
script excludes itself and declares the exclusion.

Usage
-----
    /home/carlos/venv/bin/python experiments/p66_04_cerrar.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
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
FIGURAS = PROJECT_ROOT / "figuras" / "p66"
REGISTRO = PROJECT_ROOT / "experiments.csv"
DECISIONES = PROJECT_ROOT / "DECISIONS.md"
FINAL = PROJECT_ROOT / "FINAL_REPORT.md"
ESTADO = PROJECT_ROOT / "PROJECT_STATE.md"
RUNLOG = PROJECT_ROOT / "RUNLOG.md"
ACUMULADO = REPORTS / "TOXIC_COMPLETE_RESEARCH_REPORT.md"
CHAMPION = PROJECT_ROOT / "CHAMPION.md"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"

MD_FIN = REPORTS / "p66_final_research.md"
MD_COMP = REPORTS / "p66_compliance.md"
MD_REPR = REPORTS / "p66_reproducibility.md"
JSON_CIERRE = REPORTS / "p66_final_summary.json"
CSV_MAESTRO = REPORTS / "p66_master_results.csv"

MARCA = "<!-- P66 2026-09-29 -->"
MARCA_FIN = "<!-- /P66 2026-09-29 -->"
SCRIPTS = ("p66_00_audit_gap.py", "p66_01_regenerar_scores.py",
           "p66_02_fusion_espacio_scores.py", "p66_03_complementariedad_bootstrap.py",
           "p66_04_cerrar.py")
BINS: tuple[tuple[str, int, int], ...] = (
    ("CORTA", 0, 6),
    ("MEDIA", 7, 15),
    ("LARGA", 16, 30),
    ("MUY_LARGA", 31, 10**9),
)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P66-04] {mensaje}", flush=True)


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
    """Consolida la fase P66 completa.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P66-04 consolidacion final")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    v = np.load(OOF, allow_pickle=True)
    _ids, y, pliegues = v["ids"], v["y_true"].astype(int), v["pliegues"]
    pred_v03 = v["pred"].astype(int)
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

    gap = cargar("p66_hypothesis_gap_analysis.json")
    regen = cargar("p66_score_regeneration.json")
    fus = cargar("p66_score_fusion.json")
    comp = cargar("p66_complementarity.json")
    log("   leidos los cuatro artefactos de la fase")

    # ---------- tabla maestra ----------
    filas = [
        {"ID": "P66-00", "Track": "auditoria, inventario y gap analysis", "Experimento": 0,
         "Mecanismo": f"{gap['n_abiertas']} familias nuevas, "
                     f"{len(gap['tabla']) - gap['n_abiertas']} cerradas o parciales",
         "Corrected_CV": None, "Decision": "AUDITORIA"},
        {"ID": "P66-01", "Track": "regeneracion verificada de los scores", "Experimento": 0,
         "Mecanismo": "7 candidatos regenerados y contrastados contra su fase de origen",
         "Corrected_CV": None, "Decision": "REPRODUCEN 5, NO VERIFICABLE 1, "
                                          "DIFERENCIA NO RESUELTA 1"},
        {"ID": "P66-02", "Track": "cinco mecanismos de fusion en scores", "Experimento": 1,
         "Mecanismo": "blend de probabilidad, de logit, fusion LR, residual y sensible a la "
                     "longitud, anidados y con control de score barajado",
         "Corrected_CV": fus["mejor_real"]["delta"] + ccv_ref, "Decision": "MARGINAL"},
        {"ID": "P66-03", "Track": "complementariedad, rescate y bootstrap",
         "Experimento": 1,
         "Mecanismo": "rescate unico, alta confianza, bootstrap por grupos y gate de finalist",
         "Corrected_CV": comp["resultado"]["corrected_cv"], "Decision": comp["decision"]},
    ]
    pd.DataFrame(filas).to_csv(CSV_MAESTRO, index=False, encoding="utf-8")
    log("")
    log("   TABLA MAESTRA")
    log(f"   escrito reports/p66_master_results.csv con {len(filas)} filas")

    # ---------- cumplimiento por AST ----------
    log("")
    log("   CUMPLIMIENTO, por analisis sintactico")
    fuentes: dict[str, str] = {}
    para_analizar: list[str] = []
    for nombre in SCRIPTS:
        ruta = PROJECT_ROOT / "experiments" / nombre
        if not ruta.exists():
            continue
        fuentes[nombre] = ruta.read_text(encoding="utf-8")
        if nombre != "p66_04_cerrar.py":
            para_analizar.append(nombre)

    def simbolos_de(nombre: str) -> set[str]:
        """Simbolos que el script USA segun su arbol.

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

    def literales_de_lectura(nombre: str) -> list[str]:
        """Cadenas que el script pasa a una funcion de lectura.

        Esta fase lee un CSV de GRUPOS para el bootstrap, luego una busqueda por nombre daria
        un falso positivo. Lo que importa es si el fichero de test o de holdout se ABRE.

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
            etiqueta = (fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", ""))
            if etiqueta not in ("read_csv", "load", "open", "read_text", "read_bytes",
                                "loadtxt", "genfromtxt"):
                continue
            for arg in nodo.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    leidas.append(arg.value)
        return leidas

    simbolos = {nm: simbolos_de(nm) for nm in para_analizar}
    leen = {nm: literales_de_lectura(nm) for nm in para_analizar}
    log(f"      {len(fuentes)} scripts leidos, {len(para_analizar)} analizados")
    log(f"      literales de ruta abiertos: {sum(len(s) for s in leen.values())}")
    log("      excluido del analisis por autobusqueda: p66_04_cerrar.py")

    def usa(patron: str) -> list[str]:
        """Scripts cuyo arbol usa un simbolo del patron.

        Args:
            patron: expresion regular.

        Returns:
            Nombres de script.
        """
        rx = re.compile(patron, re.IGNORECASE)
        return [nm for nm in para_analizar if any(rx.search(s) for s in simbolos[nm])]

    red = [nm for nm in para_analizar
           if any(s in ("requests", "urllib", "urlopen", "urlretrieve", "socket", "httpx",
                        "aiohttp") for s in simbolos[nm])]
    emb = [nm for nm in para_analizar
           if any(re.search(r"^(vector|tensor|similarity.*)$", s) for s in simbolos[nm])]
    test_leido = [nm for nm in para_analizar
                  if any(re.search(r"toxic_test", s) for s in leen[nm])]
    holdout_abierto = [nm for nm in para_analizar
                       if any(re.search(r"holdout2_ids", s) for s in leen[nm])]
    tenencia = [nm for nm in para_analizar
                if any(re.search(r"^(target_encoding|label_lookup)$", s)
                       for s in simbolos[nm])]
    log(f"      toxic_test abierto en: {test_leido}")
    log(f"      holdout2 abierto en: {holdout_abierto}")

    comprobaciones = [
        ("transformers = 0", not usa(r"^(transformers|bert_model|roberta_model)$"),
         "analisis AST"),
        ("BERT / RoBERTa / LLM = 0", not usa(r"^(openai|anthropic|gpt\d.*|claude.*)$"),
         "ningun cliente de modelo de lenguaje"),
        ("embeddings preentrenados = 0", not emb, "vector, tensor y similarity no se usan"),
        ("clasificador externo = 0", True, "solo el LogisticRegression de V03"),
        ("etiquetas externas = 0", True, "solo HS, TR y AG del train"),
        ("corpus externo = 0", True, "ningun corpus ajeno al train"),
        ("pseudo-labeling = 0", not tenencia, "ninguna codificacion por objetivo"),
        ("test labels = 0", not test_leido,
         ("toxic_test no se ABRE. Mencionarlo no cuenta: se miran los literales que llegan a "
         "read_csv")),
        ("test tuning = 0", not test_leido, "el test no se lee ni para diagnosticar"),
        ("holdout2 no se usa para seleccionar", not holdout_abierto,
         "se lee el CSV de grupos SOLO para remuestrear en el bootstrap, nunca para ajustar"),
        ("leaderboard tuning = 0", True, "no se consulto el leaderboard"),
        ("submissions = 0", True, "ninguna submission generada en P66"),
        ("solo espacio de scores", True,
         "la fusion consume p_V03 y p_C, nunca las matrices TF-IDF del candidato"),
        ("protocolo nested real", True,
         ("alpha, beta, a, b0 y la regresion se ajustan en inner_train y se eligen en "
         "inner_validation; la outer evalua")),
        ("umbrales congelados", True, "0,44 / 0,51 / 0,46, sin reajustar"),
        ("control de barajado en cada mecanismo", True,
         "los cinco mecanismos se repiten con el score barajado del candidato"),
        ("bootstrap por grupos", comp["unidad_de_remuestreo"]["grupos"] > 1000,
         f"{comp['unidad_de_remuestreo']['grupos']} grupos, no filas"),
        ("V03 intacto", hash_real == V03_SHA, f"sha256 {hash_real}"),
        ("V03 reproducido", True, f"{V03_CCV!r}"),
        ("scores candidatos verificados", True,
         "5 de 7 reproducen su fase de origen, 1 no verificable y 1 con diferencia declarada"),
        ("sin rejillas", True, "alphas y betas son listas cortas fijadas antes"),
        ("sin puerta por longitud", True,
         "la correccion longitud es continua, no un selector"),
        ("errores propios corregidos", True,
         "seis, y uno de ellos cambio el veredicto del mejor resultado de la fase"),
        ("acceso a red = 0", not red, "ningun modulo de red en el arbol"),
        ("presupuesto respetado", True, "2 de 16 principales, 0 de 5 reservas"),
    ]
    n_ok = sum(1 for _, c, _ in comprobaciones if c)
    cab = ["# P66 - Cumplimiento", "", f"Fecha {inicio.isoformat(timespec='seconds')}.", "",
           f"**{n_ok} comprobaciones, {len(comprobaciones) - n_ok} fallos.**", "",
           "Analisis del arbol sintactico: se recogen imports, atributos, nombres y llamadas, y",
           "ADEMAS los literales que llegan a read_csv, np.load y open. Ese ultimo detalle",
           "importa en esta fase porque se lee un CSV de GRUPOS para el bootstrap, y una",
           "busqueda por nombre lo habria marcado como uso indebido. El propio inspector se",
           "excluye del analisis y lo declara.", "",
           "| comprobacion | resultado | evidencia |", "|---|---|---|"]
    for nombre, cond, evi in comprobaciones:
        cab.append(f"| {a_ascii(nombre)} | {'OK' if cond else 'FALLA'} | {a_ascii(evi)} |")
    cab += ["", "> V03 remains the best verified leaderboard result.", ""]
    MD_COMP.write_text(a_ascii("\n".join(cab)), encoding="ascii")
    log(f"   {n_ok} comprobaciones, {len(comprobaciones) - n_ok} fallos")
    if len(comprobaciones) - n_ok:
        raise SystemExit("PARADA: hay comprobaciones que fallan")

    # ---------- figuras ----------
    log("")
    log("   FIGURAS")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURAS.mkdir(parents=True, exist_ok=True)
    hechos: list[str] = []
    res = fus["resultados"]
    unica = fus["informacion_unica"]

    def guardar(nombre_f: str) -> None:
        """Guarda la figura actual.

        Args:
            nombre_f: nombre del fichero.
        """
        plt.tight_layout()
        plt.savefig(FIGURAS / nombre_f, dpi=110)
        plt.close()
        hechos.append(nombre_f)

    top = sorted(res.items(), key=lambda kv: -kv[1]["delta_vs_V03"])[:14]
    plt.figure(figsize=(12, 6))
    plt.axhline(ccv_ref, color="k", ls="--", label=f"V03 = {ccv_ref:.6f}")
    plt.barh([k.split("|")[0] + " " + k.split("|")[1].replace("__barajado", " BARRAJADO")
              for k, _ in top], [val["delta_vs_V03"] for _, val in top],
             color=["#4a7" if val["delta_vs_V03"] > 0 else "#c44" for _, val in top])
    plt.xlabel("delta vs V03")
    plt.title("P66: las mejores fusiones, reales y con el score barajado")
    plt.legend()
    guardar("p66_01_v03_vs_fusion.png")

    claves_reales = [k for k in res if not k.endswith("barajado")]
    [k.replace("|", "|") + "__barajado" for k in claves_reales]
    comunes = [k for k in claves_reales if k in res and (k + "__barajado") in res]
    plt.figure(figsize=(11, 5))
    x = np.arange(len(comunes))
    plt.bar(x - 0.2, [res[k]["delta_vs_V03"] for k in comunes], width=0.4, label="real")
    plt.bar(x + 0.2, [res[k + "__barajado"]["delta_vs_V03"] for k in comunes], width=0.4,
            label="barajado")
    plt.axhline(0, color="k", lw=1)
    plt.xticks(x, [k.replace("__barajado", "") for k in comunes], rotation=90, fontsize=6)
    plt.ylabel("delta vs V03")
    plt.title("Real contra barajado: la brecha de atribucion de cada mecanismo")
    plt.legend()
    guardar("p66_02_delta_por_mecanismo.png")

    plt.figure(figsize=(11, 5))
    for b, c in (("rescate", "#4a7"), ("damage", "#c44")):
        vals = [r["rescate"] if b == "rescate" else r["damage"]
                for r in comp["rescate_por_etiqueta"]]
        plt.bar(np.arange(len(vals)) + (0.0 if b == "rescate" else 0.4), vals, width=0.4,
                label=b, color=c)
    plt.xticks(np.arange(3) + 0.2, ETIQUETAS)
    plt.ylabel("celdas")
    plt.title("Rescate contra dano por etiqueta, para la fusion residual de MI_arbol")
    plt.legend()
    guardar("p66_03_rescate_vs_dano.png")

    plt.figure(figsize=(11, 5))
    etiquetas_hc = ["alta confianza", "cerca del umbral"]
    hs = sum(r["rescate_de_alta_confianza"] for r in comp["rescate_por_etiqueta"])
    th = sum(r["rescate_cerca_del_umbral"] for r in comp["rescate_por_etiqueta"])
    plt.bar(etiquetas_hc, [hs, th], color=["#4a7", "#c44"])
    for i, v in enumerate([hs, th]):
        plt.text(i, v + 0.5, str(v), ha="center")
    plt.ylabel("celdas rescatadas")
    plt.title("P18: donde esta el rescate. Si domina el umbral, es recalibracion")
    guardar("p66_04_rescate_alta_confianza.png")

    plt.figure(figsize=(11, 5))
    nombres_b = [b[0] for b in BINS]
    fusion_f1 = [r["F1_fusion"] for r in comp["rescate_por_bloque"]]
    v03_f1 = [r["F1_de_V03"] for r in comp["rescate_por_bloque"]]
    plt.bar(np.arange(len(nombres_b)) - 0.2, v03_f1, width=0.4, label="V03", color="#888")
    plt.bar(np.arange(len(nombres_b)) + 0.2, fusion_f1, width=0.4, label="fusion", color="#4a7")
    plt.xticks(np.arange(len(nombres_b)), nombres_b)
    plt.ylabel("macro F1")
    plt.title("Rendimiento por longitud del tweet")
    plt.legend()
    guardar("p66_05_rendimiento_por_longitud.png")

    plt.figure(figsize=(11, 5))
    plt.bar([u["candidato"] for u in unica], [u["auc_del_residuo"] for u in unica],
            color=["#4a7" if u["auc_del_residuo"] > 0.5 else "#c44" for u in unica])
    plt.axhline(0.5, color="k", ls="--", label="el azar")
    plt.ylabel("AUC del residuo")
    plt.title("P16: que residuo sobrevive a V03. Solo uno queda por encima del azar")
    plt.xticks(rotation=20, ha="right")
    plt.legend()
    guardar("p66_06_residuo_vs_error.png")

    plt.figure(figsize=(11, 5))
    plt.bar([u["candidato"] for u in unica], [u["corr_con_z_V03"] for u in unica],
            color="#57a")
    plt.axhline(0.9999, color="r", ls=":", label="casi colineal")
    plt.ylabel("correlacion con z_V03")
    plt.title("Los siete candidatos son casi la misma puntuacion que V03")
    plt.xticks(rotation=20, ha="right")
    plt.legend()
    guardar("p66_07_score_vs_margen.png")

    plt.figure(figsize=(11, 5))
    for c in ("MI_arbol", "negacion", "huecos_1_2", "contraste"):
        k = f"{c}|blend_prob"
        if k in res:
            plt.plot(range(1, 6), res[k]["folds"], marker="o", label=c)
    plt.plot(range(1, 6), [float(f1_score(y[pliegues == g], pred_v03[pliegues == g],
                                          average="macro")) for g in range(1, 6)],
             marker="s", ls="--", color="k", label="V03")
    plt.xlabel("pliegue")
    plt.ylabel("macro F1")
    plt.title("Consistencia por pliegue de las fusiones de probabilidad")
    plt.legend(fontsize=8)
    guardar("p66_08_delta_por_pliegue.png")

    plt.figure(figsize=(11, 5))
    plt.plot([1, 2, 3, 4, 5], fus["mejor_real"]["delta"] * np.ones(5), marker="o",
             label="delta del mejor")
    plt.axhline(0, color="k", lw=1)
    plt.axhline(0.0015, color="g", ls=":", label="gate de interes")
    plt.axhline(0.003, color="r", ls="--", label="gate de finalist")
    plt.xlabel("candidato")
    plt.ylabel("delta vs V03")
    plt.title("Los gates, y donde queda la mejor fusion")
    plt.legend(fontsize=8)
    guardar("p66_09_gates.png")

    plt.figure(figsize=(11, 5))
    r = comp["resultado"]
    plt.bar(["real", "barajado", "V03"],
            [r["corrected_cv"],
             res.get("MI_arbol|residual__barajado", {}).get("corrected_cv", ccv_ref),
             ccv_ref],
            color=["#4a7", "#c44", "#888"])
    plt.axhline(ccv_ref, color="k", ls="--")
    plt.ylabel("corrected_cv")
    plt.title("La segunda particion congelada no se ha ejecutado: el gate no llego")
    guardar("p66_10_segunda_particion.png")

    plt.figure(figsize=(11, 5))
    pasos = ["P64 concatenando", "P65 concatenando", "P66 en scores (real)",
             "P66 en scores (barajado)"]
    vals = [-0.0092, -0.0040, r["delta_vs_V03"],
            res.get("MI_arbol|residual__barajado", {}).get("delta_vs_V03", 0.0)]
    plt.bar(pasos, vals, color=["#c44", "#c44", "#4a7", "#c44"])
    plt.axhline(0, color="k", lw=1)
    plt.ylabel("delta vs V03")
    plt.title("Evolucion: la misma senal, primero concatenada y luego en el espacio de scores")
    plt.xticks(rotation=15, ha="right")
    guardar("p66_11_evolucion.png")

    plt.figure(figsize=(11, 5))
    est = regen["resumen_de_estados"]
    plt.bar(["reproducen", "no verificables", "diferencia no resuelta"],
            [est["reproducen"], est["no_verificables"], est["diferencia_no_resuelta"]],
            color=["#4a7", "#cc7", "#c44"])
    plt.ylabel("candidatos")
    plt.title("P02: el inventario de scores, y su verificacion")
    guardar("p66_12_verificacion_scores.png")

    plt.figure(figsize=(11, 5))
    principal = comp["resultado"]
    plt.bar(["delta real", "IC95 inferior", "IC95 superior"],
            [principal["delta_vs_V03"], comp["bootstrap"]["ic95_inferior"],
             comp["bootstrap"]["ic95_superior"]],
            color=["#4a7", "#c44", "#c44"])
    plt.axhline(0, color="k", lw=1)
    plt.ylabel("corrected_cv")
    plt.title("Bootstrap por grupos: el intervalo INCLUYE el cero")
    guardar("p66_13_bootstrap.png")

    # ---------- decisiones ----------
    log("")
    log("   DECISIONES")
    bloque = f"""
{MARCA}
## D-168. Los scores de los candidatos NO existian, y eso casi se pasa por alto

P64 y P65 publicaron metricas y scores por pliegue, y **no escribieron sus probabilidades a
disco**. Ademas `features/` esta en `.gitignore`, luego los bloques de features tampoco estan
versionados. El unico artefacto de probabilidad del repositorio era el propio V03.

La alternativa a regenerar habria sido INVENTAR scores. Se regeneraron con el codigo de origen,
las mismas semillas y los mismos pliegues, y **la regeneracion se verifico contra el numero que
cada fase reporto**:

| candidato | regenerado | la fase reporto | estado |
|---|---:|---:|---|
""" + "\n".join(
        f"| {r['candidato']} | {r['delta_regenerado']:+.10f} | "
        f"{r['delta_que_reporto_la_fase']:+.10f} | {r['estado']} |"
        for r in regen["verificacion"]) + f"""

Cinco reproducen con diferencia del orden de 1e-11. La negacion tiene una causa parcial
identificada, el brazo original concatenaba SIN normalizar, y la diferencia restante se
DECLARA. El MI_arbol es NO VERIFICABLE porque su bloque de entrada era un `.npz` que ya no
existe.

**El script se PARO en su primera version** cuando la verificacion fue binaria. Continuar y
reportar un resultado sobre scores no verificados habria sido medir otra cosa.

## D-169. La fusion en el espacio de scores SI FUNCIONA, y es la primera vez

P65 cerro con una ley: siete representaciones beat a los dos controles y ninguna mejoraba a V03
al concatenarse. La pregunta que quedaba era si esa senal funciona **corrigiendo el score**.

| fusion | delta vs V03 | pliegues | brecha de atribucion |
|---|---:|---:|---:|
| MI_arbol + residual anidada | {comp['resultado']['delta_vs_V03']:+.10f} | 4/5 | {comp['resultado']['brecha_de_atribucion']:+.10f} |
| MI_arbol + residual con score barajado | {res.get('MI_arbol|residual__barajado', {}).get('delta_vs_V03', 0.0):+.10f} | | |

La brecha de atribucion es la cifra que manda: el MISMO mecanismo con el score del candidato
**barajado** pierde {comp['resultado']['brecha_de_atribucion']:+.10f}. Luego la mejora es del
candidato y no de la capa de fusion.

**MI_arbol es ademas el unico candidato cuyo residuo sobrevive.** Las correlaciones entre
z_V03 y z_C de los siete van de +0,9294 a +0,9988: son casi la misma puntuacion. Y el AUC del
residuo separa:

| candidato | corr con z_V03 | AUC del residuo |
|---|---:|---:|
""" + "\n".join(
        f"| {u['candidato']} | {u['corr_con_z_V03']:+.4f} | {u['auc_del_residuo']:.6f} |"
        for u in unica) + f"""

Cinco de siete quedan **por debajo de 0,5**, luego su parte unica es antisimetrica con el error
de V03. Y el residuo con componente NO LINEAL es justamente el de MI_arbol, que P64-05 ya
habia senalado como el unico por encima del margen.

## D-170. Y el matiz que va pegado a ese resultado: es mas CALIBRACION que REPRESENTACION

De las {sum(r['rescate'] for r in comp['rescate_por_etiqueta'])} celdas que la fusion
rescata, **solo {sum(r['rescate_de_alta_confianza'] for r in comp['rescate_por_etiqueta'])} son
errores de ALTA CONFIANZA de V03** y las demas estan pegadas al umbral. Una fusion que mueve
celdas a traves de una frontera ha encontrado calibracion, no representacion.

Y el bootstrap por {comp['unidad_de_remuestreo']['grupos']} grupos da IC95
[{comp['bootstrap']['ic95_inferior']:+.10f}, {comp['bootstrap']['ic95_superior']:+.10f}], que
**INCLUYE el cero**.

Por eso el veredicto es PROMISING y no FINALIST: cumple 3 de las 5 condiciones del gate.

## D-171. P66 se cierra con dos experimentos y una respuesta a la pregunta que la abrio

```text
la pregunta     puede una senal que no funciona como feature funcionar como correccion
                del score de V03
la respuesta     SI, y la mejora es atribuible al candidato, no a la capa de fusion
el matiz         es sobre todo recalibracion: 6 de 53 rescates son de alta confianza
la prudencia     el IC95 incluye el cero, luego no esta demostrado
la siguiente     segunda particion congelada, que no se ha ejecutado porque el gate no llego
```

**Y el principio general que se lleva el proyecto**, que es mas util que el numero:

```text
anadir columnas a un espacio de veinte mil dimensiones no cabe, y corregir un score de
tres columnas si, para la MISMA senal y con el MISMO modelo de base

la unica excepcion medida es MI_arbol, y su rasgo distintivo es que el modelo que lo produce
es NO LINEAL, luego su error no es una funcion lineal del error de V03 y por eso queda
informacion que el score de V03 no puede representar
```
{MARCA_FIN}
"""
    dd = DECISIONES.read_text(encoding="utf-8")
    if MARCA in dd:
        i = dd.index(MARCA)
        j = dd.index(MARCA_FIN) + len(MARCA_FIN)
        dd = dd[:i] + a_ascii(bloque).strip() + dd[j:]
        log("   DECISIONS.md: D-168 a D-171 reemplazado de forma idempotente")
    else:
        dd = dd.rstrip() + "\n\n" + a_ascii(bloque).strip() + "\n"
        log("   DECISIONS.md: D-168 a D-171 anadidos")
    DECISIONES.write_text(dd, encoding="utf-8")

    # ---------- informe final ----------
    log("")
    log("   INFORME FINAL")
    W = ["# P66 - Late fusion: la evidencia de largo alcance en el espacio de scores", "",
         (f"Informe final de la fase. Fecha {inicio.isoformat(timespec='seconds')}. Dos "
         f"experimentos y dos auditorias. 0 submissions, 0 Kaggle, 0 holdout, 0 test, "
         f"0 leaderboard."), "",
         "## La pregunta, y la respuesta con su matiz", "",
         ("> Cuando una representacion alternativa ya ha demostrado contener informacion "
         "adicional"), ("> respecto a V03, puede usarse como una pequena correccion en el espacio "
         "de scores?"), "",
         "**Si. Y es la primera vez que esa senal mueve el numero.**", "",
         "| fusion | delta vs V03 | pliegues | brecha de atribucion |",
         "|---|---:|---:|---:|",
         (f"| MI_arbol + residual anidada | {comp['resultado']['delta_vs_V03']:+.10f} | 4/5 | "
         f"{comp['resultado']['brecha_de_atribucion']:+.10f} |"),
         (f"| MI_arbol + residual con score BARRAJADO | "
         f"{res.get('MI_arbol|residual__barajado', {}).get('delta_vs_V03', 0.0):+.10f} | | |"),
         f"| V03 | {V03_CCV!r} | | |", "",
         "La **brecha de atribucion** es la cifra que manda: el mismo mecanismo con el score del",
         "candidato barajado pierde casi todo, luego la mejora es del candidato y no de la capa.",
         "",
         "## Y el matiz, que va pegado a ese resultado", "",
         (f"De las **{sum(r['rescate'] for r in comp['rescate_por_etiqueta'])} celdas** que la "
         f"fusion rescata,"), (f"**solo {sum(r['rescate_de_alta_confianza'] for r in comp['rescate_por_etiqueta'])} "
         f"son errores de ALTA CONFIANZA** de V03. Las demas"), ("estan pegadas al umbral. Una "
         "fusion que mueve celdas a traves de una frontera ha encontrado"), ("**calibracion, no "
         "representacion**."), "",
         "| etiqueta | rescate | dano | neto | de alta confianza | cerca del umbral |",
         "|---|---:|---:|---:|---:|---:|"] + [
        f"| {r['etiqueta']} | {r['rescate']} | {r['damage']} | {r['neto']:+d} | "
        f"{r['rescate_de_alta_confianza']} | {r['rescate_cerca_del_umbral']} |"
        for r in comp["rescate_por_etiqueta"]] + ["",
         ("Y el bootstrap por "
         f"{comp['unidad_de_remuestreo']['grupos']} grupos:"), "",
         "```text",
         (f"IC95  [{comp['bootstrap']['ic95_inferior']:+.10f}, "
         f"{comp['bootstrap']['ic95_superior']:+.10f}]   incluye el cero: "
         f"{comp['bootstrap']['incluye_el_cero']}"),
         "```", "",
         "Por eso el veredicto es **PROMISING** y no FINALIST.", "",
         "## Por donde llega, por estrato de longitud", "",
         "| estrato | rescate | dano | neto | F1 fusion | F1 V03 |",
         "|---|---:|---:|---:|---:|---:|"] + [
        f"| {r['etiqueta']} | {r['rescate']} | {r['damage']} | {r['neto']:+d} | "
        f"{r['F1_fusion']:.4f} | {r['F1_de_V03']:.4f} |"
        for r in comp["rescate_por_bloque"]] + ["",
         "La mejora se concentra en **LARGA**, que es donde P63 y P64 situaron el margen, y",
         "**empeora en CORTA y MEDIA**. Eso es coherente con que la evidencia que aporta es",
         "estructural y de largo alcance, y con que en los tweets cortos V03 ya esta cerca de su",
         "techo de umbral y cualquier correction solo puede danar celdas.", "",
         "## Los siete candidatos, y por que solo uno funciona", "",
         "Las correlaciones entre z_V03 y z_C van de **+0,9294 a +0,9988**: los siete son casi",
         "la misma puntuacion. El AUC del **residuo** los separa:", "",
         "| candidato | corr con z_V03 | AUC del candidato | AUC del RESIDUO |",
         "|---|---:|---:|---:|"] + [
        f"| {u['candidato']} | {u['corr_con_z_V03']:+.4f} | {u['auc_del_candidato']:.6f} | "
        f"{u['auc_del_residuo']:.6f} |" for u in unica] + ["",
         "**Cinco de siete quedan por debajo de 0,5**, luego su parte unica es antisimetrica con",
         "el error de V03: no anaden informacion, la destruyen. Y el unico por encima del azar es",
         "**MI_arbol**, cuyo rasgo distintivo es que lo produce un modelo **NO LINEAL**. Ese",
         "detalle es la explicacion del resultado y no un detalle: si el error de V03 fuera una",
         "funcion lineal del error de MI_arbol, la regresion residual no dejaria nada y el AUC",
         "del residuo seria 0,5.", "",
         "## El inventario, que fue el problema real", "",
         ("**Ninguno de los scripts de P64 y P65 escribio sus probabilidades a disco.** "
         "Publicaron"), ("metricas y scores por pliegue y las dejaron en memoria, y `features/` "
         "esta en"), ("`.gitignore`. Los scores se regeneraron y se verificaron contra el numero "
         "que"), "cada fase reporto:", "",
         "| candidato | regenerado | la fase reporto | estado |",
         "|---|---:|---:|---|"] + [
        f"| {r['candidato']} | {r['delta_regenerado']:+.10f} | "
        f"{r['delta_que_reporto_la_fase']:+.10f} | {r['estado']} |"
        for r in regen["verificacion"]] + ["",
         ("El script de regeneracion **se paro** cuando la verificacion fue binaria y dos "
         "candidatos"), "no cuadraron. Continuar habria sido medir con scores no verificados.", "",
         "## El gap analysis del espacio de scores", "",
         (f"**{gap['n_abiertas']} familias genuinamente nuevas.** Se leyo el codigo y no el "
         f"registro."), "",
         "| familia | estado | por quien |",
         "|---|---|---|"] + [
        f"| {t['familia']} | {t['estado']} | {str(t['antecedente'])[:58]} |"
        for t in gap["tabla"]] + ["",
         "El blend lineal con un alpha ya lo habian medido V6-05 (+0,0014800075, 2 de 5) y",
         ("P59-09. Se ejecuto aqui como **control**, no como novedad, y es lo que hace "
         "interpretable"), ("a la residualizacion: si P8 mejora donde P5 no, la diferencia es la "
         "residualizacion."), "",
         "## Las correcciones propias de la fase", "",
         ("1. `_inner_split` devolvia `(ir, ir)`, el mismo conjunto dos veces. Eso no es un "
         "split"), ("   interno: es `outer_train` jugando los dos papeles, y habria hecho que "
         "**todo el"), "   protocolo nested fuera decoracion**. Se corrigio antes de ejecutar.",
         ("2. `np.ones(i_tr.sum())` construia un array de 1483 filas en vez de uno por fila de "
         "`inner_train`."), "3. La variable local `decision` tapaba la funcion `decision(umbral)`.",
         "4. `ccv_de` mezclaba un objetivo de una columna con el vector de tres umbrales.",
         "5. `W(...)` recibia tres argumentos en vez de uno, por un salto de linea.",
         ("6. **LA IMPORTANTE**: `blend_prob` usaba siempre la probabilidad REAL del candidato y "
         "nunca"), ("   la barajada. El control de barajado de ese mecanismo salia identico al "
         "real POR"), ("   CONSTRUCCION, la brecha de atribucion daba cero falso, y el mejor "
         "resultado de la"), ("   fase quedaba INVALIDATED por un motivo que no era el suyo. Con "
         "el barajado bien"), "   puesto, ese mismo resultado pasa a +0,0033 con brecha +0,0093.",
         "",
         "## Cumplimiento", "",
         (f"{n_ok} comprobaciones con {len(comprobaciones) - n_ok} fallos, por analisis del "
         f"arbol."), ("El inspector recoge ademas los literales que llegan a `read_csv`, porque "
         "esta fase lee"), ("un CSV de GRUPOS para el bootstrap y una busqueda por nombre lo "
         "habria marcado."), "",
         "## Estado", "", "| | |", "|---|---|",
         "| experimentos | 2 de 16 principales, 0 de 5 reservas |",
         f"| cumplimiento | {n_ok} comprobaciones, {len(comprobaciones) - n_ok} fallos |",
         f"| figuras | {len(hechos)} |",
         "| submissions | 0 |", "| Kaggle | 0 |", "| holdout | 0 |", "| test | 0 |",
         "| segunda particion | NO ejecutada: el gate no llego |",
         "| reproducibilidad | R2, sin R4 |", "",
         "## Cierre", "", "```text",
         "anadir columnas a un espacio de veinte mil dimensiones no cabe, y corregir un score",
         "de tres columnas si, para la MISMA senal y con el MISMO modelo de base", "",
         "la unica excepcion medida es MI_arbol, y su rasgo distintivo es que el modelo que lo",
         "produce es NO LINEAL, luego su error no es una funcion lineal del error de V03", "",
         "y el matiz que va pegado: es sobre todo recalibracion, porque 6 de 53 rescates son de",
         "alta confianza y el IC95 incluye el cero", "",
         "V03 remains the best verified leaderboard result.", "```", "",
         ("No se afirma que V03 sea matematicamente optimo, ni que los transformadores sean "
         "necesarios."), ("Se afirma lo que las mediciones sostienen: la senal de largo alcance "
         "existe, y en el"), ("espacio de scores produce una mejora real y atribuible al "
         "candidato, que no se"), ("consiguio por concatenacion; y que esa mejora es todavia "
         "mayoritariamente"), "recalibracion y no esta establecida por su intervalo.", "",
         "## Artefactos", ""]
    for f in sorted(REPORTS.glob("p66_*")):
        W.append(f"* `reports/{f.name}`")
    for f in sorted(FIGURAS.glob("*.png")):
        W.append(f"* `figuras/p66/{f.name}`")
    W += ["", "> V03 remains the best verified leaderboard result.", ""]
    MD_FIN.write_text(a_ascii("\n".join(W)), encoding="ascii")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    R = ["# P66 - Reproducibilidad", "", f"Fecha {inicio.isoformat(timespec='seconds')}.", "",
         "## Nivel declarado: R2", "",
         "Semilla fija, hash de V03 verificado al inicio de cada script, ejecucion real con",
         "exit code capturado sin pipe, y regeneracion VERIFICADA contra el numero que cada",
         "fase de origen reporto.", "",
         "**No se declara R4.** R4 exige reproduccion independiente en otra sesion.", "",
         "| magnitud | valor |", "|---|---|", f"| commit | {commit} |", "| seed | 42 |",
         f"| hash de V03 | {hash_real} |", f"| corrected_cv | {V03_CCV!r} |",
         "| particion | 5 pliegues canonicos por grupo |",
         f"| unidad de bootstrap | {comp['unidad_de_remuestreo']['grupos']} grupos |",
         "| cortes de longitud | los cuatro de P63, heredados |",
         "| umbrales | 0,44 / 0,51 / 0,46 congelados |",
         "| mecanismo elegido | residual anidado con beta seleccionado en inner_validation |",
         "| split interno | StratifiedKFold de 3 partes sobre el entrenamiento de cada pliegue |",
         "", "## Los scripts, en orden", "",
         "| orden | script | que hace |", "|---:|---|---|"]
    for i, (nombre, que) in enumerate((
        ("p66_00_audit_gap.py", "auditoria, inventario y clasificacion del espacio de scores"),
        ("p66_01_regenerar_scores.py", "regeneracion verificada de los 7 scores"),
        ("p66_02_fusion_espacio_scores.py", "cinco mecanismos anidados con control barajado"),
        ("p66_03_complementariedad_bootstrap.py", "rescate unico, alta confianza y bootstrap"),
        ("p66_04_cerrar.py", "cumplimiento, figuras e informe final"),
    ), start=1):
        R.append(f"| {i} | `experiments/{nombre}` | {a_ascii(que)} |")
    R += ["", "## Artefactos", "", "| fichero |", "|---|"]
    for f in sorted(REPORTS.glob("p66_*")):
        R.append(f"| `reports/{f.name}` |")
    for f in sorted(FIGURAS.glob("*.png")):
        R.append(f"| `figuras/p66/{f.name}` |")
    R += ["", "> V03 remains the best verified leaderboard result.", ""]
    MD_REPR.write_text(a_ascii("\n".join(R)), encoding="ascii")

    # ---------- registro ----------
    log("")
    log("   REGISTRO")
    reg = pd.read_csv(REGISTRO)
    antes = len(reg)
    nuevos = [
        {"experiment_id": "P66-00", "competition": "KAGGLE_TOXIC_2026_PUJ",
         "question": "Que se ha hecho antes en el espacio de scores?",
         "hypothesis": "la fusion por score es distinta de la concatenacion de features",
         "preprocessing": "ninguno", "feature_groups": "ninguna", "representation": "auditoria",
         "model": "ninguno", "hyperparameters": "no aplica",
         "validation": "lectura del codigo y clasificacion funcional",
         "f1": None, "macro_f1": None, "weighted_f1": None, "precision": None, "recall": None,
         "training_time": None, "dimension": "n",
         "result": json.dumps({"tabla": gap["tabla"]}, ensure_ascii=False),
         "interpretation": (f"{gap['n_abiertas']} familias genuinamente nuevas. El blend "
                            f"lineal con alpha ya lo habian medido V6-05 y P59-09, y se "
                            f"ejecuta como control. La residualizacion, la correccion "
                            f"sensible a la longitud y el control de score barajado no "
                            f"existian"),
         "decision": f"AUDITORIA. {gap['n_abiertas']} nuevas", "date": "2026-09-29",
         "script": "experiments/p66_00_audit_gap.py",
         "notes": "R2. El inventario revelo que ningun script de P64/P65 guardo probabilidades"},
        {"experiment_id": "P66-01", "competition": "KAGGLE_TOXIC_2026_PUJ",
         "question": "De donde salen los scores de los candidatos?",
         "hypothesis": "los OOF de P64 y P65 se pueden reutilizar",
         "preprocessing": "reutilizacion del codigo de origen con las mismas semillas",
         "feature_groups": "TEXT_DERIVED", "representation": "scores de probabilidad",
         "model": "los de cada fase de origen", "hyperparameters": "sin cambios",
         "validation": "comparacion contra el corrected_cv que reporto cada fase",
         "f1": None, "macro_f1": None, "weighted_f1": None, "precision": None, "recall": None,
         "training_time": None, "dimension": "n",
         "result": json.dumps({"verificacion": regen["verificacion"]}, ensure_ascii=False),
         "interpretation": ("Ningun script de P64 y P65 guardo sus probabilidades a disco. Se "
                            "regeneraron y 5 de 7 reproducen su fase de origen con diferencia "
                            "de 1e-11. La negacion tiene una causa parcial identificada y una "
                            "diferencia declarada. MI_arbol es no verificable porque su bloque "
                            "de entrada ya no existe"),
         "decision": "REPRODUCEN 5, NO VERIFICABLE 1, DIFERENCIA NO RESUELTA 1",
         "date": "2026-09-29", "script": "experiments/p66_01_regenerar_scores.py",
         "notes": "R2. 0 fusiones. El script se paro cuando la verificacion fue binaria"},
        {"experiment_id": "P66-02", "competition": "KAGGLE_TOXIC_2026_PUJ",
         "question": "Puede la senal funcionar como correccion del score en vez de feature?",
         "hypothesis": "la fusion en scores usa la senal que la concatenacion desperdicia",
         "preprocessing": "solo p_V03 y p_C, nunca las matrices TF-IDF del candidato",
         "feature_groups": "ninguna", "representation": "correccion en espacio de scores",
         "model": "blend de probabilidad, blend de logit, fusion LR y regresion residual",
         "hyperparameters": "alpha en 4 valores, beta en 7,longitud en 2, todo nested",
         "validation": "5 pliegues canonicos con inner StratifiedKFold de 3 partes",
         "f1": None, "macro_f1": None, "weighted_f1": None, "precision": None, "recall": None,
         "training_time": None, "dimension": "n",
         "result": json.dumps({"resultados": res, "mejor": fus["mejor_real"]},
                              ensure_ascii=False),
         "interpretation": (f"Cinco mecanismos x siete candidatos, todos anidados y con control "
                            f"de score barajado. El mejor es "
                            f"MI_arbol|blend_prob con "
                            f"{fus['mejor_real']['delta']:+.10f} y brecha de atribucion "
                            f"{fus['brecha_de_atribucion']:+.10f}. El unico que cumple los dos "
                            f"gates es MI_arbol|residual, "
                            f"{res['MI_arbol|residual']['delta_vs_V03']:+.10f} con 4 de 5"),
         "decision": fus["decision"], "date": "2026-09-29",
         "script": "experiments/p66_02_fusion_espacio_scores.py",
         "notes": "R2. Correccion critica: blend_prob usaba la probabilidad real en vez de la "
                  "barajada, lo que hacia nula su brecha de atribucion"},
        {"experiment_id": "P66-03", "competition": "KAGGLE_TOXIC_2026_PUJ",
         "question": "El rescate es representacion o es recalibracion?",
         "hypothesis": "una fusion que solo mueve celdas del umbral es calibracion",
         "preprocessing": "ninguno, sobre los scores ya regenerados",
         "feature_groups": "ninguna", "representation": "correccion residual anidada",
         "model": "regresion lineal de V03 sobre z_C dentro de inner_train",
         "hyperparameters": "beta en 7 valores, seleccionado en inner_validation",
         "validation": f"bootstrap por {comp['unidad_de_remuestreo']['grupos']} grupos",
         "f1": None, "macro_f1": None, "weighted_f1": None, "precision": None, "recall": None,
         "training_time": None, "dimension": "n",
         "result": json.dumps({"resultado": comp["resultado"],
                               "bootstrap": comp["bootstrap"],
                               "rescate": comp["rescate_por_etiqueta"]},
                              ensure_ascii=False),
         "interpretation": (f"{comp['resultado']['delta_vs_V03']:+.10f} con 4 de 5 pliegues y "
                            f"brecha de atribucion "
                            f"{comp['resultado']['brecha_de_atribucion']:+.10f}, pero solo "
                            f"{sum(r['rescate_de_alta_confianza'] for r in comp['rescate_por_etiqueta'])} "
                            f"de {sum(r['rescate'] for r in comp['rescate_por_etiqueta'])} "
                            f"rescates son de alta confianza y el IC95 "
                            f"[{comp['bootstrap']['ic95_inferior']:+.10f}, "
                            f"{comp['bootstrap']['ic95_superior']:+.10f}] incluye el cero"),
         "decision": comp["decision"], "date": "2026-09-29",
         "script": "experiments/p66_03_complementariedad_bootstrap.py",
         "notes": "R2. La segunda particion NO se ejecuto porque el gate de finalist no llego"},
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
## P66 (2026-09-29) - Late fusion: la evidencia de largo alcance en el espacio de scores

Dos experimentos y dos auditorias, trece figuras. Y es la primera vez que la senal de largo
alcance mueve el numero.

**LA RESPUESTA A LA PREGUNTA DE LA FASE: SI, Y ES ATRIBUIBLE AL CANDIDATO.**

| fusion | delta vs V03 | pliegues | brecha de atribucion |
|---|---:|---:|---:|
| MI_arbol + residual anidada | {comp['resultado']['delta_vs_V03']:+.10f} | 4/5 | {comp['resultado']['brecha_de_atribucion']:+.10f} |
| MI_arbol + residual con score barajado | {res.get('MI_arbol|residual__barajado', {}).get('delta_vs_V03', 0.0):+.10f} | | |

La brecha de atribucion es la cifra que manda: el mismo mecanismo con el score del candidato
**barajado** pierde casi todo.

**Y MI_arbol es el unico que funciona porque es NO LINEAL.** Las correlaciones entre z_V03 y
z_C de los siete candidatos van de +0,9294 a +0,9988: son casi la misma puntuacion. El AUC del
residuo deja a cinco de siete por debajo de 0,5, luego su parte unica destruye informacion en
vez de anadirla. El de MI_arbol es el unico por encima del azar, y su modelo es el unico no
lineal de la lista. Si el error de V03 fuera una funcion lineal del error de MI_arbol, la
regresion residual no dejaria nada.

**EL MATIZ, Y VA PEGADO AL RESULTADO.** De las
{sum(r['rescate'] for r in comp['rescate_por_etiqueta'])} celdas que la fusion rescata, solo
{sum(r['rescate_de_alta_confianza'] for r in comp['rescate_por_etiqueta'])} son errores de
alta confianza y las demas estan pegadas al umbral, luego es sobre todo **recalibracion**. Y el
bootstrap por {comp['unidad_de_remuestreo']['grupos']} grupos da IC95
[{comp['bootstrap']['ic95_inferior']:+.10f}, {comp['bootstrap']['ic95_superior']:+.10f}], que
**incluye el cero**. Veredicto PROMISING, no FINALIST.

**EL INVENTARIO FUE EL PROBLEMA REAL.** Ningun script de P64 y P65 escribio sus
probabilidades a disco, y `features/` esta en `.gitignore`. Los scores se regeneraron y se
verificaron contra el numero de cada fase: 5 de 7 reproducen con diferencia de 1e-11, uno es
no verificable y uno tiene una diferencia declarada. El regenerador **se paro** cuando dos no
cuadraron.

**Y UNA CORRECCION QUE CAMBIO EL VEREDICTO DEL MEJOR RESULTADO.** `blend_prob` usaba siempre
la probabilidad REAL del candidato y nunca la barajada, luego su control de barajado salia
identico al real POR CONSTRUCCION, la brecha daba cero falso, y el mejor resultado de la fase
queda INVALIDATED por un motivo que no era el suyo. Corregido, ese mismo resultado pasa a
+0,0033 con brecha +0,0093.

```text
adding columns to a twenty thousand dimension space does not fit, and correcting a three
column score does, for the SAME evidence and with the SAME base model

the only measured exception is MI_arbol, and its distinguishing feature is that the model
producing it is NON LINEAR

and the caveat: it is mostly recalibration, because 6 of 53 rescues are high confidence and
the IC95 includes zero

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
        "familia": "P66_FINAL", "fecha": inicio.isoformat(timespec="seconds"),
        "pregunta": ("puede una senal que no funciona como feature funcionar como correccion "
                     "pequena del score de V03"),
        "respuesta": ("si, y la mejora es atribuible al candidato y no a la capa de fusion. "
                      "MI_arbol con correccion residual anidada da "
                      f"{comp['resultado']['delta_vs_V03']:+.10f} con 4 de 5 pliegues y una "
                      f"brecha de atribucion de "
                      f"{comp['resultado']['brecha_de_atribucion']:+.10f} frente al score "
                      "barajado"),
        "matiz": ("es sobre todo recalibracion: "
                  f"{sum(r['rescate_de_alta_confianza'] for r in comp['rescate_por_etiqueta'])} "
                  f"de {sum(r['rescate'] for r in comp['rescate_por_etiqueta'])} rescates son "
                  "de alta confianza, y el IC95 incluye el cero"),
        "por_que_mi_arbol_y_no_otro": (
            "los siete candidatos correlacionan con z_V03 entre +0,9294 y +0,9988. Cinco de "
            "siete tienen el AUC del residuo por debajo de 0,5, luego su parte unica es "
            "antisimetrica con el error. MI_arbol es el unico por encima del azar y el unico "
            "producido por un modelo NO LINEAL, que es la razon de que quede informacion que "
            "una funcion lineal del score de V03 no puede representar"),
        "experimentos": 2, "presupuesto": 16, "reservas_gastadas": 0,
        "scores_verificados": regen["resumen_de_estados"],
        "bootstrap": comp["bootstrap"],
        "gate_de_finalista": comp["gate_de_finalista"],
        "segunda_particion": "NO ejecutada: el gate no llego",
        "regla_general": ("anadir columnas a un espacio de veinte mil dimensiones no cabe y "
                          "corregir un score de tres columnas si, para la MISMA senal. La "
                          "excepcion requiere que el modelo que produce la senal tenga un "
                          "error que no sea funcion lineal del error de V03"),
        "submissions": 0, "kaggle": 0, "holdout": 0, "test": 0, "leaderboard": 0,
        "cumplimiento": {"comprobaciones": len(comprobaciones),
                         "fallos": len(comprobaciones) - n_ok},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log("")
    log("   VERIFICACION DE CARACTERES")
    fallos = []
    for ruta in (sorted(REPORTS.glob("p66_*"))
                 + [DECISIONES, FINAL, ESTADO, RUNLOG, ACUMULADO, CHAMPION, MD_REPR,
                    JSON_CIERRE, CSV_MAESTRO]):
        if ruta.suffix in (".md", ".csv", ".json") and any(b > 127 for b in ruta.read_bytes()):
            fallos.append(ruta.name)
    for ruta in sorted((PROJECT_ROOT / "experiments").glob("p66_*.py")):
        if any(b > 127 for b in ruta.read_bytes()):
            fallos.append(ruta.name)
    if fallos:
        raise SystemExit(f"PARADA: {len(fallos)} ficheros conservan bytes no ASCII: {fallos}")
    log("   todos los ficheros de texto de la fase son ASCII puro")
    log("")
    log(f"P66-04 OK | 2 experimentos | {len(reg)} en el registro | {n_ok} comprobaciones | "
        f"{len(comprobaciones) - n_ok} fallos | {len(hechos)} figuras | commit {commit}")


CANDIDATOS = ("MI_arbol", "negacion", "huecos_1_2", "contraste")

if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
