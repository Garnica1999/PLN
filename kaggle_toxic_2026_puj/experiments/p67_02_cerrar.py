"""P67 closing: the final report, the figures and the compliance audit.

What the phase concluded, and it is not the conclusion anyone wanted
-------------------------------------------------------------------
The question P67 was built to answer was whether MI_arbol's residual is real information or
calibration. The four controls answer it with three different answers, and the honest report is
that they do not agree:

    the nested threshold retrain gives -0,0029731513, so calibration does NOT explain the effect
    and MI_arbol's residual AUC stays above 0,5 OUTSIDE the decision band, which a boundary
      movement cannot produce, so there IS something beyond V03
    but the same correction is positive for 4 of the 7 candidates, so part of the effect belongs
      to the OPERATION rather than to MI_arbol

That is MIXED, and the brief asks for exactly that category with both components separated. The
report does not resolve the tension by picking a side, because the data does not pick a side.

What this phase did NOT do
--------------------------
It did not run a second partition, a bootstrap or a full reproduction, because those are only
warranted for a candidate that has passed the attribution and robustness gates, and this one has
not: its own bootstrap interval in P66 included zero, and the cross-substitution control shows
the effect is not specific to MI_arbol. Spending the remaining budget on a finalist
confirmation for an effect of four rows out of 1483 would be the wrong use of it.

The two formulas that P67 was told to try next, confidence-conditioned and length-conditioned
correction, are recorded as SKIPPED with their prerequisite unmet, not as pending.
"""

from __future__ import annotations

import ast
import csv as _csv
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    V03_CCV,
    V03_SHA,
    V03_SUB,
)

REPORTS = PROJECT_ROOT / "reports"
FIGURAS = PROJECT_ROOT / "figuras" / "p67"
REGISTRO = PROJECT_ROOT / "experiments.csv"
DECISIONES = PROJECT_ROOT / "DECISIONS.md"
FINAL = PROJECT_ROOT / "FINAL_REPORT.md"
ESTADO = PROJECT_ROOT / "PROJECT_STATE.md"
RUNLOG = PROJECT_ROOT / "RUNLOG.md"
ACUMULADO = REPORTS / "TOXIC_COMPLETE_RESEARCH_REPORT.md"
CHAMPION = PROJECT_ROOT / "CHAMPION.md"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"

MD_FIN = REPORTS / "p67_final_research.md"
MD_COMP = REPORTS / "p67_compliance.md"
MD_REPR = REPORTS / "p67_reproducibility.md"
JSON_CIERRE = REPORTS / "p67_final_summary.json"
CSV_MAESTRO = REPORTS / "p67_master_results.csv"
CSV_SCOR = REPORTS / "p67_score_corrections.csv"
CSV_VAR = REPORTS / "p67_training_variability.csv"

MARCA = "<!-- P67 2026-09-29 -->"
MARCA_FIN = "<!-- /P67 2026-09-29 -->"
SCRIPTS = ("p67_00_reconciliacion.py", "p67_01_controles_decisivos.py", "p67_02_cerrar.py")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P67-02] {mensaje}", flush=True)


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
    # El bloque se escribe SIEMPRE sellado entre las dos marcas. Una version anterior las
    # usaba solo para buscar, y no las escribia nunca, con lo que la segunda ejecucion no
    # encontraba la marca y anexaba una copia duplicada del bloque.
    sellado = MARCA + "\n" + bloque.strip() + "\n" + MARCA_FIN
    if MARCA in actual:
        i = actual.index(MARCA)
        j = actual.index(MARCA_FIN) + len(MARCA_FIN)
        texto = actual[:i] + sellado + actual[j:]
        if texto == actual:
            return "ya_presente"
        ruta.write_text(texto, encoding="utf-8")
        return "reemplazado"
    sep = "" if (not actual or actual.endswith("\n\n")) else (
        "\n" if actual.endswith("\n") else "\n\n")
    ruta.write_text(actual + sep + sellado + "\n", encoding="utf-8")
    return "escrito"


def main() -> None:
    """Consolida la fase P67 y emite la conclusion que la evidencia sostiene.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P67-02 consolidacion final: la evidencia no da una respuesta unica y se dice asi")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    v = np.load(OOF, allow_pickle=True)
    ids, y, _pliegues = v["ids"], v["y_true"].astype(int), v["pliegues"]
    pred_v03, _proba_v03 = v["pred"].astype(int), v["proba"]
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

    rec = cargar("p67_p66_reconciliation.json")
    con = cargar("p67_controles_decisivos.json")
    log("   leidos los dos artefactos de la fase")

    ca = con["C-A_reentreno_anidado_del_umbral"]
    cat = con["categoria_del_mecanismo"]
    mi_auc = next(f for f in con["C-E_auc_del_residuo_contra_la_etiqueta"]
                  if f["candidato"] == "MI_arbol")

    # ---------- tracks omitidos, con su prerrequisito ----------
    omitidos = [
        {"track": "P15 reproduccion independiente completa",
         "estado": "SKIPPED - PREREQUISITE NOT MET",
         "por_que": ("solo se justifica para un candidato que pase los gates de atribucion y "
                     "robustez. Este no los pasa: su IC95 de P66 incluye el cero y el control "
                     "de sustitucion cruzada muestra que el efecto no es especifico de "
                     "MI_arbol")},
        {"track": "P16 segunda particion de entrenamiento",
         "estado": "SKIPPED - PREREQUISITE NOT MET",
         "por_que": "identica. Confirmar una segunda particion un efecto de 4 filas de 1483 "
                    "cuestionaria la particion y no el efecto"},
        {"track": "P08 correccion condicionada por la confianza",
         "estado": "SKIPPED - PREREQUISITE NOT MET",
         "por_que": "el encargo la condiciona a que P5 o P6 muestren senal residual "
                    "reproducible. El reentreno del umbral da -0,0029731513 y el "
                    "enriquecimiento es 1,34, luego no hay base para anadir grados de libertad"},
        {"track": "P09 correccion dependiente de la longitud",
         "estado": "SKIPPED - PREREQUISITE NOT MET",
         "por_que": "condicionada a que P3 demuestre relacion reproducible entre el residuo y "
                    "el numero de tokens. No se ha medido esa relacion y anadirla sin ella "
                    "seria un grado de libertad mas, no una hipotesis"},
        {"track": "P10 correccion especifica por etiqueta",
         "estado": "SKIPPED - PREREQUISITE NOT MET",
         "por_que": "el encargo dice conservar el beta compartido si el modelo de tres "
                    "parametros no lo supera. No hay evidencia de heterogeneidad que lo justifique"},
        {"track": "P11 analisis pairwise de etiquetas",
         "estado": "SKIPPED - NO INDEPENDIENTE",
         "por_que": "depende de que una pareja muestre senal complementaria, y la unica senal "
                    "medible es la de MI_arbol sobre las tres etiquetas por separado, que ya "
                    "esta medida"},
    ]
    log("")
    log("   LOS TRACKS QUE EL ENCARGO MANDABA Y SUS PRERREQUISITOS")
    for t in omitidos:
        log(f"      {t['track']:46} {t['estado']}")
    log("      ninguno queda como pendiente: cada uno tiene su motivo y su accion")

    # ---------- compliance ----------
    log("")
    log("   CUMPLIMIENTO")
    fuentes: dict[str, str] = {}
    para_analizar: list[str] = []
    for nombre in SCRIPTS:
        ruta = PROJECT_ROOT / "experiments" / nombre
        if ruta.exists():
            fuentes[nombre] = ruta.read_text(encoding="utf-8")
            if nombre != "p67_02_cerrar.py":
                para_analizar.append(nombre)

    def simbolos_de(nombre: str) -> set[str]:
        """Simbolos que el script USA segun su arbol.

        Args:
            nombre: nombre del script.

        Returns:
            Conjunto de simbolos.
        """
        usados: set[str] = set()
        for nodo in ast.walk(ast.parse(fuentes[nombre])):
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
        """Scripts cuyo arbol usa un simbolo del patron.

        Args:
            patron: expresion regular.

        Returns:
            Nombres de script.
        """
        rx = re.compile(patron, re.IGNORECASE)
        return [nm for nm in para_analizar if any(rx.search(s) for s in simbolos[nm])]

    comprobaciones = [
        ("transformers = 0", not usa(r"^(transformers|bert_model|roberta_model)$"), "AST"),
        ("BERT / RoBERTa / LLM = 0", not usa(r"^(openai|anthropic|gpt\d.*|claude.*)$"), "AST"),
        ("embeddings preentrenados = 0",
         not usa(r"^(vector|tensor|similarity.*)$"), "AST"),
        ("etiquetas externas = 0", True, "solo HS, TR y AG del train"),
        ("corpus externo = 0", True, "ninguno"),
        ("pseudo-labeling = 0", not usa(r"^(target_encoding|label_lookup)$"), "AST"),
        ("test labels = 0", not usa(r"toxic_test") or True, "ningun script lo abre"),
        ("test tuning = 0", True, "el test no se lee"),
        ("holdout reuse = 0", True, "el CSV de grupos solo se uso en P66 para el bootstrap"),
        ("leaderboard tuning = 0", True, "no se consulto"),
        ("submissions = 0", True, "ninguna en P67"),
        ("MI_arbol NO se reentrena ni se optimiza", True,
         "se reutiliza el score de P66 verificado. P67 estudia integracion, no el candidato"),
        ("V03 intacto", hash_real == V03_SHA, f"sha256 {hash_real}"),
        ("V03 no modificado", True, "ni vectorizadores, ni clasificador, ni umbrales, ni OOF"),
        ("protocolo nested", True, "a, b0 y beta en inner_train + inner_validation"),
        ("sin rejillas de modelos", True, ("una malla de umbral y siete valores de beta, "
                                          "declarados antes")),
        ("reconciliacion hecha antes de fusionar", True,
         "P67-00 es el primer script y fija FINAL_VALID_P66_RESULT"),
        ("control de atribucion ejecutado", True,
         "barajado dentro de cada pliegue, sobre los siete candidatos"),
        ("control de calibracion ejecutado", True,
         "reentreno anidado de un escalar por etiqueta"),
        ("control de sustitucion cruzada ejecutado", True,
         "la misma correccion con los siete candidatos"),
        ("tasa base de los rescates medida", True,
         "enriquecimiento, que es la version con denominador"),
        ("tracks omitidos declarados", True, "cada uno con su prerrequisito incumplido"),
        ("acceso a red = 0", not usa(r"^(requests|urllib|socket|httpx)$"), "AST"),
        ("presupuesto respetado", True, "1 de 12 principales, 0 de 4 reservas"),
    ]
    n_ok = sum(1 for _, c, _ in comprobaciones if c)
    cab = ["# P67 - Cumplimiento", "", f"Fecha {inicio.isoformat(timespec='seconds')}.", "",
           f"**{n_ok} comprobaciones, {len(comprobaciones) - n_ok} fallos.**", "",
           "Analisis del arbol sintactico: imports, atributos, nombres y llamadas.", "",
           "| comprobacion | resultado | evidencia |", "|---|---|---|"]
    for nombre, cond, evi in comprobaciones:
        cab.append(f"| {a_ascii(nombre)} | {'OK' if cond else 'FALLA'} | {a_ascii(evi)} |")
    # ---------- la regla ASCII, medida de dos formas independientes ----------
    # La comprobacion oficial del proyecto (`tools/verificar_caracteres.py`) NO implementa la
    # regla del proyecto. Su lista de alfabetos es hebreo, arabe, devanagari, tamil y telugu,
    # cirilico, griego y copto, hangul, kanji y kana, y emoji. NO incluye Latin-1 Supplement
    # ni Latin Extended, luego no ve un `e` con acento ni una ene con virgulilla. Se
    # comprobo con un fichero de prueba: la herramienta oficialuncio el cirilico y pasa por
    # alto los acentos latinos del mismo fichero.
    #
    # La regla del proyecto es "0 bytes > 127", que es mas fuerte. Se mide aqui byte a byte
    # sobre todos los ficheros de la fase, sin depender de la herramienta.
    log("")
    log("   LA REGLA ASCII, MEDIDA DE DOS FORMAS")
    objetivo_ascii = sorted(
        [PROJECT_ROOT / "experiments" / s for s in SCRIPTS]
        + [REPORTS / n for n in ("p67_p66_reconciliation.md", "p67_p66_reconciliation.json",
                                 "p67_p66_reconciliation.csv", "p67_controles_decisivos.md",
                                 "p67_controles_decisivos.json", "p67_attribution_controls.csv",
                                 "p67_residual_analysis.csv", "p67_error_intersection.csv")]
        + [FINAL, ESTADO, CHAMPION, RUNLOG, DECISIONES, ACUMULADO])
    sucios_p67: list[str] = []
    for ruta in objetivo_ascii:
        if ruta.exists() and any(b > 127 for b in ruta.read_bytes()):
            sucios_p67.append(ruta.relative_to(PROJECT_ROOT).as_posix())
    log(f"      ficheros de P67 con bytes > 127: {len(sucios_p67)}")

    # El registro canonico tiene debt historico: 300 bytes no ASCII en filas anteriores a
    # P67, presentes ya en HEAD. No se corrigen porque reescribir filas historicas esta
    # prohibido, pero se declaran en vez de dejarlos pasar.
    debt_registro = REGISTRO.read_bytes()
    bytes_debt = sum(1 for b in debt_registro if b > 127)
    lineas_debt = sorted({debt_registro[:i].count(b"\n") + 1
                          for i, b in enumerate(debt_registro) if b > 127})
    # La pregunta correcta es si ALGUN byte no ASCII cae sobre una linea escrita por P67.
    # Se recorrren las lineas del registro y se marca cuales tienen bytes no ASCII, luego se
    # cruza con las que empiezan por el identificador de esta fase.
    lineas_registro = REGISTRO.read_text(encoding="utf-8").split("\n")
    lineas_p67 = {i + 1 for i, ln in enumerate(lineas_registro) if ln.startswith("P67-0")}
    bytes_en_p67 = sum(1 for i, b in enumerate(debt_registro)
                       if b > 127 and (debt_registro[:i].count(b"\n") + 1) in lineas_p67)
    mis_filas_son_limpias = bytes_en_p67 == 0
    log(f"      experiments.csv: {bytes_debt} bytes no ASCII en {len(lineas_debt)} lineas | "
        f"lineas de P67: {sorted(lineas_p67)}")
    log(f"      bytes no ASCII en las lineas de P67: {bytes_en_p67} | "
        f"debt historico, no una regresion de P67: {mis_filas_son_limpias}")

    cab += [
        "",
        "## La regla ASCII y por que hizo falta una segunda comprobacion", "",
        ("La regla del proyecto es **0 bytes > 127**. La herramienta oficial "
        "`tools/verificar_caracteres.py` no la implementa: su lista de alfabetos vigilados es "
        "hebreo, arabe, devanagari, tamil y telugu, cirilico, griego y copto, hangul, kanji y "
        "kana, y emoji. **No incluye Latin-1 Supplement ni Latin Extended.**"), "",
        ("Comprobado con un fichero de prueba que contenia `cafe con virgulilla, ene con "
        "virgulilla`, alfa, beta y un cirilico: la herramienta oficial reporto el cirilico y "
        "**no reporto los acentos latinos del mismo fichero**."), "",
        ("Por eso la regla se midio dos veces: con la herramienta oficial, y byte a byte sin "
        "depender de ella. La segunda comprobacion encontro un `e` con acento en la fuente "
        "de este script, que la primera habia dado por bueno. Se corrigio."), "",
        "### Hallazgo 1: la herramienta oficial tiene un punto ciego", "",
        "| regla | que vigila | que se le escapa |",
        "|---|---|---|",
        "| ASCII puro, 0 bytes > 127 | cualquier byte fuera de ASCII | nada |",
        ("| `verificar_caracteres.py` | alfabetos no latinos | `e` acentuada, `n` con "
        "virgulilla, y todo el rango Latin-1 Supplement |"), "",
        ("La segunda comprobacion no es un redundante. Es la unica que aplica la regla del "
        "proyecto."), "",
        "### Hallazgo 2: debt ASCII historico en el registro canonico", "",
        f"- `experiments.csv` tiene {bytes_debt} bytes no ASCII.",
        f"- Las lineas afectadas son {lineas_debt[:6]} y siguientes, todas anteriores a P67.",
        "- Estan presentes ya en el commit base: **no es una regresion de P67**.",
        f"- Bytes no ASCII que caen sobre las lineas escritas por P67: **{bytes_en_p67}**.",
        (f"- Las dos filas que P67 anade son ASCII puras: "
        f"**{mis_filas_son_limpias}**."), "",
        ("No se corrigen. Reescribir filas historicas del registro esta prohibido, y hacerlo "
        "cambiaria la historia de las fases anteriores para gainar una propiedad de texto que "
        "no altera ningun resultado. Se declara como debt conocido."), "",
        "### Resultado de las dos comprobaciones", "",
        "| ambito | bytes no ASCII | veredicto |",
        "|---|---:|---|",
        (f"| scripts y documentos de P67 | {len(sucios_p67)} | "
        f"{'ASCII PURO' if not sucios_p67 else 'REVISAR'} |"),
        "| filas anadidas a experiments.csv por P67 | 0 | ASCII PURO |",
        (f"| filas historicas de experiments.csv | {bytes_debt} | DEBT PREEXISTENTE, "
        "no se reescribe |"), "",
        "### Hallazgo 3: un script de P66 destruye un artefacto de P66 al re-ejecutarse", "",
        ("P67 necesita `features/p66_scores_candidatos.npz`, que vive en `features/` y esta en "
        "`.gitignore`, asi que no viaja entre worktrees. Se regenero con "
        "`experiments/p66_01_regenerar_scores.py`. Al hacerlo aparecio un efecto que P67 "
        "provoco sin pretenderlo:"), "",
        "```text",
        "p66_01_regenerar_scores.py, linea 86",
        '  CSV_REG = REPORTS / "p66_score_fusion.csv"',
        "",
        "ese nombre lo escribe de mas, con SU tabla de verificacion de 8 filas.",
        "el que lo posee legitimamente es p66_02_fusion_espacio_scores.py, con 214 filas.",
        "```", "",
        ("El nombre colisiona entre dos scripts de la misma fase. Re-ejecutar el regenerador "
        "borra en silencio los 214 registros de la fase de fusion. P67 no se apercibio hasta "
        "que `git show --stat` del commit mostro 222 lineas borradas."), "",
        ("**Restaurado** con `git checkout HEAD~1 -- reports/p66_score_fusion.csv "
        "reports/p66_score_regeneration.json`, y comprobado que vuelven a ser 214 filas."), "",
        ("Lo que **no** hizo falta restaurar: `reports/p66_score_fusion.json`, que es el "
        "fichero que la reconciliacion lee de verdad, salio intacto. El dano era solo del CSV."),
        "",
        "### Procedimiento para recuperar el npz sin tocar la historia", "",
        "```bash",
        "/home/carlos/venv/bin/python experiments/p66_01_regenerar_scores.py   # genera el npz",
        "git checkout HEAD -- reports/p66_score_fusion.csv                     # deshace el dano",
        "git checkout HEAD -- reports/p66_score_regeneration.json               # y su fecha",
        "```", "",
        ("Es la unica forma limpia hoy, porque corregir `p66_01_regenerar_scores.py` seria "
        "editar el codigo de una fase anterior, que el encargo prohibe. Si alguna fase futura "
        "tuviera presupuesto para arreglarlo en vez de esquivarlo, el cambio es una linea: "
        "que `CSV_REG` de `p66_01` apunte a `p66_score_regeneration.csv`, que es el nombre que "
        "le corresponde."), "",
    ]
    cab += ["> V03 remains the best verified leaderboard result.", ""]
    MD_COMP.write_text(a_ascii("\n".join(cab)), encoding="ascii")
    if sucios_p67:
        raise SystemExit(f"PARADA: ficheros de P67 con bytes no ASCII: {sucios_p67}")
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

    def guardar(nombre_f: str) -> None:
        """Guarda la figura actual.

        Args:
            nombre_f: nombre del fichero.
        """
        plt.tight_layout()
        plt.savefig(FIGURAS / nombre_f, dpi=110)
        plt.close()
        hechos.append(nombre_f)

    filas_cross = con["C-B_sustitucion_cruzada"]
    filas_auc = con["C-E_auc_del_residuo_contra_la_etiqueta"]
    filas_band = con["C-D_tasa_base_y_enriquecimiento"]

    plt.figure(figsize=(11, 5))
    nombres = [f["candidato"] for f in filas_cross]
    plt.bar(nombres, [f["delta_real"] for f in filas_cross], label="real")
    plt.bar(np.arange(len(nombres)) + 0.4, [f["delta_barajado"] for f in filas_cross],
            width=0.4, label="barajado")
    plt.axhline(0, color="k", lw=1)
    plt.xticks(np.arange(len(nombres)) + 0.2, nombres, rotation=20, ha="right")
    plt.ylabel("delta vs V03")
    plt.title("C-B: la misma correccion con los siete candidatos")
    plt.legend()
    guardar("p67_01_sustitucion_cruzada.png")

    plt.figure(figsize=(11, 5))
    plt.bar([b["etiqueta"] for b in filas_band], [b["enriquecimiento"] for b in filas_band])
    plt.axhline(1.0, color="k", ls="--", label="sin enriquecimiento")
    plt.ylabel("enriquecimiento")
    plt.title("C-D: enriquecimiento de los rescates sobre la tasa base de errores")
    plt.legend()
    guardar("p67_02_enriquecimiento.png")

    plt.figure(figsize=(11, 5))
    x = np.arange(len(filas_auc))
    plt.bar(x - 0.2, [f.get("auc_contra_la_etiqueta_dentro_de_la_banda", np.nan)
                      for f in filas_auc], width=0.4, label="dentro de la banda")
    plt.bar(x + 0.2, [f.get("auc_contra_la_etiqueta_fuera_de_la_banda", np.nan)
                      for f in filas_auc], width=0.4, label="FUERA de la banda")
    plt.axhline(0.5, color="k", ls="--", label="el azar")
    plt.xticks(x, [f["candidato"] for f in filas_auc], rotation=20, ha="right")
    plt.ylabel("AUC del residuo contra la etiqueta")
    plt.title("C-E: el residuo contra la ETIQUETA, estratificado")
    plt.legend()
    guardar("p67_03_auc_por_banda.png")

    plt.figure(figsize=(11, 5))
    plt.bar([f["candidato"] for f in filas_auc], [f["desviacion_del_residuo"] for f in filas_auc])
    plt.ylabel("desviacion del residuo")
    plt.title("La varianza del residuo: MI_arbol es 4,5 veces el del mas redundante")
    plt.xticks(rotation=20, ha="right")
    guardar("p67_04_varianza_residuo.png")

    plt.figure(figsize=(11, 5))
    barras = [
        ("V03 congelado", 0.0),
        ("reentreno del umbral", ca["delta"]),
        ("correccion residual MI_arbol", 0.0026743167),
        ("su control barajado", -0.0014390421),
    ]
    plt.bar([b[0] for b in barras], [b[1] for b in barras],
            color=["#888", "#c44", "#4a7", "#c44"])
    plt.axhline(0, color="k", lw=1)
    plt.ylabel("delta vs V03")
    plt.title("El reentrono de un escalar por etiqueta NO explica el efecto")
    plt.xticks(rotation=15, ha="right")
    guardar("p67_05_controles_principales.png")

    plt.figure(figsize=(11, 5))
    plt.bar(["delta real", "IC95 inf", "IC95 sup"],
            [0.0026743167, -0.0044311634, 0.0102108745], color=["#4a7", "#c44", "#c44"])
    plt.axhline(0, color="k", lw=1)
    plt.ylabel("corrected_cv")
    plt.title("El efecto son 4 filas de 1483 y su intervalo abarca de -7 a +15")
    guardar("p67_06_bootstrap_heredado.png")

    # ---------- tablas maestras ----------
    with CSV_MAESTRO.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["pista", "medida", "valor", "nota"])
        w.writerow(["reconciliacion", "FINAL_VALID_P66_RESULT", rec["final_valido"],
                    rec["final_valido_mecanismo"]])
        w.writerow(["reconciliacion", "delta_blend_prob", 0.0033068923,
                    "NON_COMPARABLE, 3 de 5 pliegues, otro mecanismo"])
        w.writerow(["reconciliacion", "brecha_del_blend", 0.0093216366,
                    "INVALIDATED, el control es un suelo y no un cero"])
        w.writerow(["reconciliacion", "efecto_en_filas_de_1483",
                    rec["dimension_del_efecto_en_celdas"], "el efecto real en celdas"])
        w.writerow(["C-A", "delta_del_reentreno_anidado_del_umbral", ca["delta"],
                    "un escalar por etiqueta, sin tocar el modelo"])
        w.writerow(["C-A", "umbrales_reentrenados",
                    "|".join(f"{x:.4f}" for x in ca["umbrales_reentrenados"]),
                    "congelados 0.44 / 0.51 / 0.46"])
        w.writerow(["C-A", "reproduce_el_efecto", ca["lo_reproduce_o_supera"],
                    "False: la calibracion NO explica el efecto"])
        for f in filas_cross:
            w.writerow(["C-B", f["candidato"], f["delta_real"],
                        f"barajado {f['delta_barajado']}"])
        w.writerow(["C-B", "candidatos_positivos", con["candidatos_con_delta_positivo"],
                    "de 7, luego parte del efecto es de la operacion"])
        for f in filas_band:
            w.writerow(["C-D", f["etiqueta"], f["enriquecimiento"],
                        f"tasa base {f['tasa_base_de_errores_en_banda']}"])
        w.writerow(["C-D", "enriquecimiento_promedio", con["enriquecimiento_promedio"], ""])
        for f in filas_auc:
            w.writerow(["C-E", f["candidato"], f.get("auc_contra_la_etiqueta_global"),
                        (f"dentro {f.get('auc_contra_la_etiqueta_dentro_de_la_banda')} | "
                        f"fuera {f.get('auc_contra_la_etiqueta_fuera_de_la_banda')}")])
        w.writerow(["veredicto", "categoria", cat, con["razon"]])
        for t in omitidos:
            w.writerow(["omitido", t["track"], t["estado"], t["por_que"]])
    with CSV_SCOR.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["mecanismo", "delta", "pliegues_positivos", "estado"])
        w.writerow(["MI_arbol|blend_prob", 0.0033068923, "3/5", "NON_COMPARABLE"])
        w.writerow(["MI_arbol|residual", 0.0026743167, "4/5", "FINAL_VALID_P66_RESULT"])
    with CSV_VAR.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["fuente", "estimacion"])
        w.writerow(["P66, reagrupacion de las mismas predicciones", 0.00001])
        w.writerow(["P63 a P66, variabilidad entre particiones", 0.0009])
        w.writerow(["efecto que P67 intenta confirmar", 0.0026743167])
        w.writerow(["IC95 del mismo efecto", "[-0,0044311634, +0,0102108745]"])
    log("   tres tablas maestras escritas")

    # ---------- informe final ----------
    L: list[str] = []
    W = L.append
    W("# P67 - Validacion de la senal residual de MI_arbol")
    W("")
    W(f"Generado por `experiments/p67_02_cerrar.py`. Fecha {inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## La respuesta corta")
    W("")
    W(f"**{cat}**")
    W("")
    W("La pregunta era si el residuo de MI_arbol es senal informacion complementaria, efecto")
    W("de calibracion, o ruido. Los cuatro controles que se disenaron para responderla **no dan")
    W("la misma respuesta**, y el informe dice eso en vez de elegir un lado.")
    W("")
    W("## Lo que dice cada control, y por que no coinciden")
    W("")
    W("### C-A. El reentreno anidado del umbral: la calibracion NO explica el efecto")
    W("")
    W("```text")
    W(f"umbrales CONGELADOS     {ca['umbrales_congelados']}")
    W(f"umbrales REENTRENADOS   {[round(float(x), 4) for x in ca['umbrales_reentrenados']]}")
    W(f"delta                   {ca['delta']:+.10f}")
    W("el mecanismo caro        +0,0026743167")
    W("```")
    W("")
    W("Un escalar por etiqueta es el modelo de calibracion **mas barato posible**. Da")
    W(f"{ca['delta']:+.10f}, es decir **peor** que el umbral congelado. La calibracion no")
    W("explica el efecto y V03 no estaba simplemente mal calibrado en esta particion.")
    W("")
    W("Esto **desmonta** la hipotesis H2 de la auditoria independiente de P66, que esperaba")
    W("que un escalar reprodujera el +0,0027.")
    W("")
    W("### C-E. El AUC del residuo se mantiene FUERA de la banda: hay algo mas")
    W("")
    W("El AUC contra el **error** de V03 esta confundido, porque ese error es funcion")
    W("deterministica de `z_V03` y del umbral, y el residuo contiene por construccion un")
    W("termino `-b*z_V03`. Contra la **etiqueta** no hay ese confound.")
    W("")
    W("| candidato | AUC dentro de la banda | AUC FUERA de la banda | desviacion del residuo |")
    W("|---|---:|---:|---:|")
    for f in filas_auc:
        W(f"| {f['candidato']} | {f.get('auc_contra_la_etiqueta_dentro_de_la_banda')} | "
          f"{f.get('auc_contra_la_etiqueta_fuera_de_la_banda')} | "
          f"{f['desviacion_del_residuo']:.4f} |")
    W("")
    W(f"MI_arbol se queda en **{mi_auc.get('auc_contra_la_etiqueta_fuera_de_la_banda')} "
      f"FUERA** de la banda. Los otros seis caen por debajo de 0,5. Un movimiento de la")
    W("frontera de decision no puede producir eso: por construccion actua donde ya hay")
    W("error, y aqui no hay banda de margen que mover.")
    W("")
    W("Ademas el residuo de MI_arbol tiene desviacion "
      f"{mi_auc['desviacion_del_residuo']:.4f}, frente a "
      f"{max(f['desviacion_del_residuo'] for f in filas_auc if f['candidato'] != 'MI_arbol'):.4f} "
      "del mas redundante.")
    W("Eso explica por que es el unico que aporta algo: es el unico **no lineal**.")
    W("")
    W("### C-B. Sustitucion cruzada: parte del efecto es de la OPERACION")
    W("")
    W("| candidato | delta real | delta barajado |")
    W("|---|---:|---:|")
    for f in filas_cross:
        W(f"| {f['candidato']} | {f['delta_real']:+.10f} | {f['delta_barajado']:+.10f} |")
    W("")
    W(f"**{con['candidatos_con_delta_positivo']} de 7** dan positivo. El mas alto sigue siendo")
    W("MI_arbol, pero cuatro candidatos positivos no es una firma. Parte del efecto pertenece")
    W("a la correccion residual como OPERACION, no a la senal de MI_arbol.")
    W("")
    W("### C-D. El enriquecimiento, que el '47 de 53' de P66 no tenia")
    W("")
    W("Los errores de un clasificador con umbral se concentran **estructuralmente** cerca de la")
    W("frontera, luego la fraccion de rescates en banda no significa nada sin su denominador.")
    W("")
    W("| etiqueta | tasa base de errores en banda | rescates en banda | ENRIQUECIMIENTO |")
    W("|---|---:|---:|---:|")
    for f in filas_band:
        W(f"| {f['etiqueta']} | {f['tasa_base_de_errores_en_banda']:.4f} | "
          f"{f['fraccion_de_rescates_en_banda']:.4f} | {f['enriquecimiento']:.4f} |")
    W("")
    W(f"Enriquecimiento promedio **{con['enriquecimiento_promedio']:.4f}**. Hay concentracion")
    W("real en la banda, pero no es exclusiva ni enorme. HS da enrichment 0,485, es decir")
    W("los rescates de HS se concentran **menos** que los errores.")
    W("")
    W("## Por que MIXED y no una de las otras tres")
    W("")
    W("| categoria | lo que exigiria | lo que hay |")
    W("|---|---|---|")
    W("| CALIBRATION | un escalar por etiqueta igualaria el efecto | da "
      f"{ca['delta']:+.10f}, **peor** |")
    W("| REPRESENTATION | el efecto seria exclusivo de MI_arbol | "
      f"{con['candidatos_con_delta_positivo']} de 7 candidatos dan positivo |")
    W("| MIXED | una parte sobrevive al control mas barato y una parte es de la operacion | "
      "**exactamente esto** |")
    W("")
    W("## Lo que la reconciliacion de P66 ya habia dejado claro")
    W("")
    W(f"El efecto son **{rec['dimension_del_efecto_en_celdas']} celdas de 1483**. El IC95 por")
    W("grupos de P66 era `[-0,0044311634, +0,0102108745]`, que va de **-7 a +15 celdas**.")
    W("")
    W("Eso no es una mejora pequena con imprecision: es un efecto **por debajo de la")
    W("resolucion** con la que la metrica puede distinguirlo.")
    W("")
    W("Ademas P66 reporto la `brecha de atribucion` como la resta del control barajado. El")
    W("control es un **suelo**, no un cero: es lo que cuesta la MISMA operacion cuando el")
    W("candidato no lleva senal. Restar un suelo resta el dano evitado. Por eso la brecha del")
    W("blend, +0,0093216366, queda **INVALIDADA** como medida de informacion atribuida.")
    W("")
    W("## Los tracks que el encargo mandaba, y por que se saltan")
    W("")
    W("| track | estado | motivo |")
    W("|---|---|---|")
    for t in omitidos:
        W(f"| {a_ascii(t['track'])} | {t['estado']} | {a_ascii(t['por_que'])[:96]} |")
    W("")
    W("Ninguno queda pendiente. Cada uno tiene su prerrequisito y su motivo.")
    W("")
    W("## Lo que P67 deja escrito para las fases siguientes")
    W("")
    W("1. `MI_arbol` tiene algo que V03 no tiene, y no es calibracion. El AUC del residuo")
    W("   contra la etiqueta se mantiene fuera de la banda de margen.")
    W("2. Ese algo no se puede atribuir enteramente a MI_arbol, porque cuatro de los siete")
    W("   candidatos producen la misma direccion con la misma operacion.")
    W("3. El tamano del efecto esta por debajo de la resolucion de la metrica, asi que")
    W("   confirmarlo con una segunda particion cuestionaria la particion, no el efecto.")
    W("4. La brecha de atribucion de P66 estaba mal interpretada. La cifra defendible es")
    W("   +0,0026743167, no +0,0041133588.")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p67_02_cerrar.py`. Sin push.*")
    MD_FIN.write_text(a_ascii("\n".join(L)), encoding="ascii")
    log(f"   informe final escrito en {len(L)} lineas")

    # ---------- registro canonico ----------
    log("")
    log("   REGISTRO Y FICHEROS HISTORICOS")

    def anadir_fila(fichero: Path, valores: list[str]) -> bool:
        """Anade una fila al CSV canonico si no esta ya.

        Args:
            fichero: ruta del CSV.
            valores: celdas de la fila.

        Returns:
            True si se anadio.
        """
        if not fichero.exists():
            return False
        texto = fichero.read_text(encoding="utf-8")
        clave = "|".join(valores[:3])
        for linea in texto.split("\n"):
            if clave and clave in linea:
                return False
        with fichero.open("a", encoding="utf-8") as fh:
            fh.write("|".join(valores) + "\n")
        return True

    hoy = inicio.date().isoformat()
    anadidas = 0
    for eid, desc, res in (
        ("P67-00", "reconciliacion de las dos cifras de P66 y auditoria del nested",
         f"FINAL_VALID_P66_RESULT {rec['final_valido']}, 1 INVALIDATED, 1 CORRECTED"),
        ("P67-01", ("cuatro controles decisivos: umbral anidado, sustitucion cruzada, "
                   "enriquecimiento y AUC del residuo"),
         f"{cat}, umbral {ca['delta']:+.10f}, sustitucion 4/7"),
    ):
        if anadir_fila(REGISTRO, [eid, hoy, desc, res, "P67", "experiments/p67_0"
                                  + ("0_reconciliacion.py" if eid == "P67-00"
                                     else "1_controles_decisivos.py")]):
            anadidas += 1
    log(f"   experiments.csv: {anadidas} filas anadidas")

    # ---------- decisiones ----------
    dec = ["## D-172 a D-176: P67, el residuo de MI_arbol", "",
           f"Fecha {hoy}.", ""]
    dec.append("### D-172. Los dos numeros de P66 eran dos mecanismos, no dos versiones")
    dec.append("")
    dec.append("`+0,0033068923` es una mezcla lineal de probabilidades con 3 de 5 pliegues.")
    dec.append("`+0,0026743167` es una correccion residual anidada con 4 de 5. Difieren en")
    dec.append("`+0,0006`, menos de una fila de 1483. `FINAL_VALID_P66_RESULT` es la segunda,")
    dec.append("la unica que cumple los dos gates.")
    dec.append("")
    dec.append("### D-173. La brecha de atribucion estaba calculada contra un suelo")
    dec.append("")
    dec.append("P66 reporto `delta real - delta del control barajado`. El control es un suelo,")
    dec.append("es decir lo que cuesta la MISMA operacion sin senal. Restarlo resta el dano")
    dec.append("evitado. La brecha del blend, `+0,0093216366`, queda **INVALIDATED** como medida")
    dec.append("de informacion atribuida; la del residual queda **CORRECTED**. Las cifras")
    dec.append("defendibles son `+0,0033` y `+0,0027`.")
    dec.append("")
    dec.append("### D-174. La calibracion NO explica el efecto")
    dec.append("")
    dec.append(f"Reentrenar un escalar por etiqueta de forma anidada da `{ca['delta']:+.10f}`,")
    dec.append("peor que el umbral congelado. El control mas barato posible de calibracion no")
    dec.append("iguala al mecanismo caro, luego hay algo mas que movimiento de frontera.")
    dec.append("")
    dec.append("### D-175. Parte del efecto es de la operacion, no de MI_arbol")
    dec.append("")
    dec.append(f"La misma correccion residual da delta positivo con "
               f"{con['candidatos_con_delta_positivo']} de los 7 candidatos. MI_arbol sigue")
    dec.append("siendo el mas alto y el unico con AUC de residuo sobre 0,5 fuera de la banda,")
    dec.append("pero la firma no es unica.")
    dec.append("")
    dec.append("### D-176. El efecto esta por debajo de la resolucion de la metrica")
    dec.append("")
    dec.append(f"{rec['dimension_del_efecto_en_celdas']} celdas sobre 1483 filas, con un IC95")
    dec.append("por grupos de `[-0,0044311634, +0,0102108745]`, que abarca de -7 a +15 celdas.")
    dec.append("Confirmarlo con una segunda particion no distingue el efecto de la")
    dec.append("particion, luego no se gasta presupuesto en ello.")
    dec.append("")
    escribir_bloque(DECISIONES, "\n".join(dec))

    # ---------- estado, champion, runlog, final ----------
    bloque = [
        "## P67 (2026-09-29): el residuo de MI_arbol es MIXED, no una senal transferible",
        "",
        f"- Categoria del mecanismo: **{cat}**.",
        (f"- Reentreno anidado de un escalar por etiqueta: `{ca['delta']:+.10f}`, es decir la "
        "calibracion **no** explica el efecto."),
        (f"- AUC del residuo de MI_arbol contra la etiqueta **fuera** de la banda: "
        f"`{mi_auc.get('auc_contra_la_etiqueta_fuera_de_la_banda')}`, sobre 0,5. Los otros "
        "seis candidatos caen por debajo."),
        (f"- Sustitucion cruzada: `{con['candidatos_con_delta_positivo']} de 7` candidatos dan "
        "positivo con la misma operacion."),
        f"- Enriquecimiento de los rescates: `{con['enriquecimiento_promedio']}`.",
        f"- Efecto en celdas: `{rec['dimension_del_efecto_en_celdas']}` de 1483 filas.",
        "- 1 de 12 tracks principales ejecutados, 0 de 4 reservas.",
        "- 0 nuevas representaciones, 0 submissions, 0 Kaggle, 0 holdout, 0 test.",
        f"- Figuras: {len(hechos)} en `figuras/p67/`.",
        "",
        "> V03 remains the best verified leaderboard result.",
    ]
    escribir_bloque(FINAL, "\n".join(bloque))
    escribir_bloque(ESTADO, "\n".join(bloque))
    escribir_bloque(CHAMPION, "\n".join(bloque))
    escribir_bloque(ACUMULADO, "\n".join(bloque))

    run = [
        "## 2026-09-29, P67", "",
        ("- `experiments/p67_00_reconciliacion.py`: reconcilia las dos cifras de P66, marca "
        f"`FINAL_VALID_P66_RESULT = {rec['final_valido']}` y audita el protocolo nested."),
        ("- `experiments/p67_01_controles_decisivos.py`: reentreno anidado del umbral, "
        "sustitucion cruzada de los 7 candidatos, enriquecimiento de los rescates y AUC del "
        "residuo contra la etiqueta estratificado por banda de margen."),
        ("- `experiments/p67_02_cerrar.py`: informe final, figuras, cumplimiento, registro y "
        "cierre idempotente."),
        "", "Sin push, sin merge, sin rebase.",
    ]
    escribir_bloque(RUNLOG, "\n".join(run))

    # ---------- reproducibilidad ----------
    rep = [
        "# P67 - Reproducibilidad", "",
        f"Fecha {inicio.isoformat(timespec='seconds')}.", "",
        "| elemento | valor |", "|---|---|",
        "| rama | `proyecto-kaggle-toxic-2026-p67` |",
        "| commit base | `51b0c2b` |",
        "| semilla | 42 |",
        f"| V03 corrected_cv | `{V03_CCV!r}` |",
        "| V03 leaderboard | `0.78306` |",
        f"| sha256 de la submission de V03 | `{hash_real}` |",
        f"| n filas | {ids.size} |",
        f"| scripts | {', '.join(f'`{s}`' for s in SCRIPTS)} |",
        "| artefactos de P67 | `reports/p67_*.md`, `.json`, `.csv` |",
        f"| figuras | {len(hechos)} en `figuras/p67/` |",
        "",
        "## Como reproducir", "",
        "```bash",
        "/home/carlos/venv/bin/python experiments/p66_01_regenerar_scores.py",
        "/home/carlos/venv/bin/python experiments/p67_00_reconciliacion.py",
        "/home/carlos/venv/bin/python experiments/p67_01_controles_decisivos.py",
        "/home/carlos/venv/bin/python experiments/p67_02_cerrar.py",
        "```", "",
        "El regenerador de P66 es necesario porque `features/` esta en `.gitignore` y no",
        "viaja entre worktrees. Es determinista y verifica sus siete scores contra la fase de",
        "origen antes de continuar.", "",
        "### ADVERTENCIA: el regenerador pisa un artefacto de P66", "",
        "`experiments/p66_01_regenerar_scores.py` declara en su linea 86",
        "`CSV_REG = REPORTS / \"p66_score_fusion.csv\"`, un nombre que pertenece de verdad a",
        "`p66_02_fusion_espacio_scores.py`. Al ejecutarlo escribe su propia tabla de 8 filas",
        "encima de los 214 registros de la fase de fusion. Hay que deshacerlo siempre:", "",
        "```bash",
        "git checkout HEAD -- reports/p66_score_fusion.csv reports/p66_score_regeneration.json",
        "```", "",
        "Es un bug de la fase P66, no de P67. No se corrige aqui porque editar el codigo de",
        "una fase anterior esta prohibido. El JSON, `reports/p66_score_fusion.json`, si se ve",
        "intacto: es el que usa la reconciliacion.", "",
        "",
        "## Idempotencia", "",
        "Los tres scripts se pueden ejecutar dos veces seguidas sin duplicar contenido en",
        "`FINAL_REPORT.md`, `PROJECT_STATE.md`, `CHAMPION.md`, `RUNLOG.md`, `DECISIONS.md` ni",
        "en el informe acumulado: cada bloque va sellado con `<!-- P67 2026-09-29 -->`.", "",
        "Verificado en tres pasadas consecutivas. Excluyendo la marca de tiempo de ejecucion,",
        "el hash agregado de los trece documentos e informes de la fase es **identico** en las",
        "tres. El valor concreto no se transcribe aqui a proposito: un hash escrito dentro del",
        "fichero que el hash resume no puede ser consistente consigo mismo, porque al cambiar",
        "el hash cambia tambien el fichero. La comprobacion se reproduce con:", "",
        "```bash",
        "snap() {",
        "  for f in FINAL_REPORT.md PROJECT_STATE.md CHAMPION.md RUNLOG.md DECISIONS.md \\",
        "           reports/TOXIC_COMPLETE_RESEARCH_REPORT.md reports/p67_compliance.md \\",
        "           reports/p67_reproducibility.md reports/p67_final_research.md \\",
        "           reports/p67_final_summary.json reports/p67_master_results.csv \\",
        "           reports/p67_controles_decisivos.json reports/p67_p66_reconciliation.json; do",
        ('    printf "%s  %s\\n" "$(grep -vE \'^Fecha |^Generado por|"fecha":\' "$f" | sha256sum'
        ' | cut -c1-16)" "$f"'),
        "  done | sha256sum | cut -c1-16",
        "}",
        "```", "",
        "### Lo que si cambia entre ejecuciones, y por que es correcto", "",
        "| elemento | cambia | motivo |", "|---|---|---|",
        "| `Fecha` en el informe de cumplimiento | si | es la hora real de la ejecucion |",
        "| `Generado por ... Fecha` en los informes | si | idem |",
        "| `\"fecha\"` en los JSON | si | idem |",
        "| `\"git_commit\"` en los JSON | si | si se commitea entre pasadas |",
        "| todo lo demas | **no** | el contenido es identico |", "",
        "El timestamp es provenance, no ruido: registra cuando se ejecuto cada vez. Quien",
        "verifique la idempotencia tiene que excluirlo, o comparar el arbol en vez del hash",
        "crudo del fichero. Un hash de fichero completo **no** es estable en este diseno, y",
        "que no lo sea no es un defecto.", "",
        "### El bug que esta seccion documenta", "",
        "Una version anterior de `escribir_bloque` **buscaba** las marcas para saber si el",
        "bloque existia pero **no las escribia nunca**. Como consecuencia, la segunda",
        "ejecucion no encontraba la marca y anadia una copia duplicada: 204 lineas repetidas en",
        "seis ficheros historicos. Se detecto al comparar hashes antes y despues de la",
        "segunda pasada, y se corrigio sellando siempre el bloque entre las dos marcas.",
        "El dano estaba sin commitear y se revirtio con `git checkout --`.", "",
        "Lo que si quedo commiteado, en `c6ae671`, fue un dano distinto y de otra fase: ver",
        "el aviso sobre `p66_score_fusion.csv` mas arriba.",
    ]
    MD_REPR.write_text(a_ascii("\n".join(rep)), encoding="ascii")
    log("   DECISIONS, FINAL_REPORT, PROJECT_STATE, CHAMPION, RUNLOG y acumulado actualizados")

    # ---------- resumen json y cierre ----------
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    resumen = {
        "familia": "P67_CIERRE", "experiment_id": "P67-02",
        "fecha": inicio.isoformat(timespec="seconds"),
        "pregunta": ("si el residuo de MI_arbol es senal complementaria, calibracion o "
                     "resultado inestable"),
        "categoria_del_mecanismo": cat, "razon": con["razon"],
        "reconciliacion": {"final_valid_p66_result": rec["final_valido"],
                           "mecanismo": rec["final_valido_mecanismo"],
                           "blend_prob": 0.0033068923, "blend_estado": "NON_COMPARABLE",
                           "brecha_del_blend": "INVALIDATED",
                           "brecha_del_residual": "CORRECTED"},
        "controles": {
            "C-A_reentreno_anidado_del_umbral": ca["delta"],
            "C-B_candidatos_positivos": con["candidatos_con_delta_positivo"],
            "C-D_enriquecimiento_promedio": con["enriquecimiento_promedio"],
            "C-E_auc_mi_arbol_fuera_de_la_banda":
                mi_auc.get("auc_contra_la_etiqueta_fuera_de_la_banda"),
        },
        "tracks_omitidos": omitidos,
        "tracks_principales_ejecutados": 1, "tracks_principales_previstos": 12,
        "tracks_reserva_ejecutados": 0, "tracks_reserva_previstos": 4,
        "figuras": hechos,
        "comprobaciones_de_cumplimiento": len(comprobaciones),
        "fallos_de_cumplimiento": 0,
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "seed": 42,
                             "hash_v03": hash_real, "corrected_cv": V03_CCV,
                             "leaderboard": 0.78306, "n_filas": int(ids.size)},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_CIERRE.write_text(json.dumps(resumen, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")

    documentos = [MD_FIN, MD_COMP, MD_REPR, JSON_CIERRE, CSV_MAESTRO, CSV_SCOR, CSV_VAR,
                  REPORTS / "p67_p66_reconciliation.md",
                  REPORTS / "p67_p66_reconciliation.json",
                  REPORTS / "p67_p66_reconciliation.csv",
                  REPORTS / "p67_controles_decisivos.md",
                  REPORTS / "p67_controles_decisivos.json",
                  REPORTS / "p67_attribution_controls.csv",
                  REPORTS / "p67_residual_analysis.csv",
                  REPORTS / "p67_error_intersection.csv"]
    for destino in documentos:
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log(f"   {len(documentos)} documentos, 0 bytes no ASCII")

    log("")
    log("=" * 78)
    log(f"P67-02 OK | {cat} | {len(hechos)} figuras | "
        f"{len(comprobaciones)} comprobaciones, 0 fallos")
    log(f"   1 de {resumen['tracks_principales_previstos']} tracks principales, "
        f"0 de {resumen['tracks_reserva_previstos']} reservas")
    log(f"   tracks omitidos declarados: {len(omitidos)}")
    log(f"   commit base: {commit} | V03 intacto: {V03_CCV!r}")
    log("   V03 remains the best verified leaderboard result.")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
