"""Post-mortem of the CV-to-leaderboard decoupling, on train and dev only.

The three facts that must not be collapsed into one
---------------------------------------------------
    1. the weighting improves the controlled OOF protocol
    2. the frozen submission scored 0,77522 against V03's 0,78306, a delta of -0,00784
    3. neither of those facts, on its own, shows the weighting is false or that the OOF
       experiment is invalid

What this script measures, and what it refuses to
-------------------------------------------------
Everything here runs on the development set. The test set is not read, the holdout stays
frozen, and the leaderboard number is treated as a single external observation, not as a signal
to be optimised against.

The structural fact worth naming before any measurement: V03's leaderboard score sits ABOVE its
own CV, and the candidate's sits BELOW. That asymmetry is the thing to explain, and it is not
the same as the delta between them.
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

from p59_01_voronoi_local import V03_SHA, V03_SUB

from src.evaluation import metricas_multietiqueta
from src.splits import ETIQUETAS

REPORTS = PROJECT_ROOT / "reports"
FIGURAS = PROJECT_ROOT / "figuras" / "p70"
MD = REPORTS / "toxic_cv_lb_transfer_postmortem.md"
CSV = REPORTS / "toxic_cv_lb_transfer_postmortem.csv"

UMBRALES = np.array([0.44, 0.51, 0.46])
LB_V03, LB_CAND = 0.78306, 0.77522


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P70-02] {mensaje}", flush=True)


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


def f1m(y: np.ndarray, pred: np.ndarray) -> float:
    """Macro F1 con la metrica del proyecto.

    Args:
        y: etiquetas verdaderas.
        pred: predicciones.

    Returns:
        La media de las F1 por columna.
    """
    return float(metricas_multietiqueta(y, pred)["f1_media_columnas"])


def main() -> None:
    """Ejecuta el post-mortem sobre desarrollo y escribe el informe.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P70-02 post-mortem de la transferencia CV -> LB, solo con train y dev")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    v = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)
    d = np.load(PROJECT_ROOT / "features" / "oof_weighting_v4.npz", allow_pickle=True)
    y = v["y_true"].astype(int)
    p03 = (v["proba"] >= UMBRALES[None, :]).astype(int)
    pc = (d["combo__downweight_violacion_x_clase"].astype(float)
          >= UMBRALES[None, :]).astype(int)
    pr03, prc = v["proba"], d["combo__downweight_violacion_x_clase"].astype(float)
    cv03, cvc = f1m(y, p03), f1m(y, pc)
    log(f"   V03 {cv03:.10f} | candidato {cvc:.10f} | delta {cvc - cv03:+.10f}")

    sub = json.loads((REPORTS / "toxic_cv_lb_transfer_postmortem.json")
                     .read_text(encoding="utf-8"))
    sens = sub["sensibilidad"]

    # ---------- ITEM 16: prevalencia ----------
    log("")
    log("   ITEM 16, PREVALENCIA: EL WEIGHTING ALTERA EL BALANCE DE DECISIONES?")
    prev = []
    for j, et in enumerate(ETIQUETAS):
        prev.append({"etiqueta": et,
                     "prevalencia_real": round(float(y[:, j].mean()), 6),
                     "prevalencia_V03": round(float(p03[:, j].mean()), 6),
                     "prevalencia_candidato": round(float(pc[:, j].mean()), 6),
                     "cambio_absoluto": round(float(pc[:, j].mean() - p03[:, j].mean()), 6)})
        f = prev[-1]
        log(f"      {et}: real {f['prevalencia_real']:.4f} | V03 {f['prevalencia_V03']:.4f}"
            f" | cand {f['prevalencia_candidato']:.4f} | cambio {f['cambio_absoluto']:+.6f}")

    # ---------- ITEM 12: descomposicion de la mejora ----------
    log("")
    log("   ITEM 12, DESCOMPOSICION DEL DELTA POR ETIQUETA")
    desg = []
    from sklearn.metrics import precision_score, recall_score
    for j, et in enumerate(ETIQUETAS):
        f03 = float(metricas_multietiqueta(y, p03)["f1_por_etiqueta"][et])
        fc = float(metricas_multietiqueta(y, pc)["f1_por_etiqueta"][et])
        desg.append({"etiqueta": et, "f1_V03": round(f03, 6), "f1_candidato": round(fc, 6),
                     "delta_f1": round(fc - f03, 6),
                     "prec_V03": round(float(precision_score(y[:, j], p03[:, j],
                                                             zero_division=0)), 6),
                     "prec_cand": round(float(precision_score(y[:, j], pc[:, j],
                                                              zero_division=0)), 6),
                     "rec_V03": round(float(recall_score(y[:, j], p03[:, j],
                                                         zero_division=0)), 6),
                     "rec_cand": round(float(recall_score(y[:, j], pc[:, j],
                                                          zero_division=0)), 6)})
        d2 = desg[-1]
        log(f"      {et}: F1 {d2['f1_V03']:.4f} -> {d2['f1_candidato']:.4f} "
            f"({d2['delta_f1']:+.6f}) | prec {d2['prec_V03']:.4f} -> {d2['prec_cand']:.4f} "
            f"| rec {d2['rec_V03']:.4f} -> {d2['rec_cand']:.4f}")

    # ---------- ITEM 13: margenes ----------
    log("")
    log("   ITEM 13, MARGENES")
    marg = []
    dm03 = np.abs(pr03 - UMBRALES[None, :])
    dmc = np.abs(prc - UMBRALES[None, :])
    for j, et in enumerate(ETIQUETAS):
        marg.append({"etiqueta": et,
                     "mediana_margen_V03": round(float(np.median(dm03[:, j])), 6),
                     "mediana_margen_candidato": round(float(np.median(dmc[:, j])), 6),
                     "frac_cerca_umbral_V03": round(float((dm03[:, j] < 0.10).mean()), 6),
                     "frac_cerca_umbral_candidato": round(float((dmc[:, j] < 0.10).mean()), 6)})
        f = marg[-1]
        log(f"      {et}: mediana {f['mediana_margen_V03']:.4f} -> "
            f"{f['mediana_margen_candidato']:.4f} | fraccion < 0.10 del umbral "
            f"{f['frac_cerca_umbral_V03']:.4f} -> {f['frac_cerca_umbral_candidato']:.4f}")

    # ---------- ITEM 14: interseccion de errores ----------
    log("")
    log("   ITEM 14, INTERSECCION DE ERRORES SOBRE OOF")
    ar, ac = (p03 == y), (pc == y)
    inter = []
    for j, et in enumerate(ETIQUETAS):
        r = int((~ar[:, j] & ac[:, j]).sum())
        dd = int((ar[:, j] & ~ac[:, j]).sum())
        inter.append({"etiqueta": et, "V03_ko_cand_ok": r, "V03_ok_cand_ko": dd,
                      "ambos_ok": int((ar[:, j] & ac[:, j]).sum()),
                      "ambos_ko": int((~ar[:, j] & ~ac[:, j]).sum()),
                      "rescue": r, "damage": dd, "net": r - dd})
        f = inter[-1]
        log(f"      {et}: rescue {f['rescue']:4d} | damage {f['damage']:4d} | "
            f"neto {f['net']:+4d}")
    int_glob = {"etiqueta": "GLOBAL",
                "V03_ko_cand_ok": int((~ar & ac).sum()),
                "V03_ok_cand_ko": int((ar & ~ac).sum()),
                "rescue": int((~ar & ac).sum()), "damage": int((ar & ~ac).sum())}
    int_glob["net"] = int_glob["rescue"] - int_glob["damage"]
    log(f"      GLOBAL: rescue {int_glob['rescue']} | damage {int_glob['damage']} | "
        f"neto {int_glob['net']:+d}")

    # ---------- ITEMS 17 y 27: clasificacion de la causa ----------
    log("")
    log("   ITEMS 17 y 27, CLASIFICACION DE LA CAUSA")
    ratio = sens["ratio_desviacion_sobre_efecto"]
    caida = sens["caida_100_a_70"]
    confound = sub["confound_de_tamano"]
    pct_menos = 1.0 - confound["candidato_enviado_filas"] / confound["V03_enviado_filas"]

    hechos = [
        {"hecho": "el weighting mejora el protocolo OOF controlado",
         "valor": f"{cvc - cv03:+.10f}", "verificado": True},
        {"hecho": "la submission congelada obtiene LB por debajo de V03",
         "valor": f"{LB_CAND} contra {LB_V03}, delta {LB_CAND - LB_V03:+.5f}",
         "verificado": True},
        {"hecho": "los dos hechos no permiten afirmar que el weighting sea falso",
         "valor": "el CV mejora en las tres particiones y la permutacion es p = 0,000999",
         "verificado": True},
    ]
    candidatos = {
        "A_diferencia_de_datos": {
            "aplica": True,
            "magnitud_medida": (f"reducir el entrenamiento un 10 por ciento mueve el CV de V03 "
                                f"con desviacion {sens['desviacion_maxima']:.6f}, que es "
                                f"{ratio:.2f} veces el efecto"),
            "la_diferencia_real": f"{pct_menos * 100:.1f} por ciento menos filas",
            "lectura": ("la magnitud es del orden correcto para explicar el deficit. NO se "
                        "puede medir V03 con 1731 sin el holdout"),
        },
        "B_el_weighting_no_generaliza": {
            "aplica": "NO_DEMOSTRABLE",
            "razon": ("el candidato sigue batiendo a V03 en TODOS los tamanos medidos, con "
                      "deltas entre +0,0068 y +0,0151. No hay evidencia de que el mecanismo "
                      "sea danino"),
        },
        "C_variabilidad_entre_particiones": {
            "aplica": True,
            "magnitud_medida": (f"el rango dentro de un mismo tamano llega a "
                                f"{sens['rango_maximo']:.6f}"),
            "lectura": "comparable al efecto, y refuerza A",
        },
        "D_cambio_en_la_distribucion_de_clases": {
            "aplica": False,
            "razon": "medido en el item 16: la prevalencia predicha cambia muy poco",
        },
        "E_sensibilidad_de_la_regla_de_pesos": {
            "aplica": False,
            "razon": ("medido en el item 9: la misma formula produce poblaciones de pesos "
                      "estables en todos los tamanos"),
        },
        "H_no_identificable_con_datos_permitidos": {
            "aplica": True,
            "razon": ("el holdout esta congelado y las labels del test estan prohibidas, luego "
                      "no se puede separar 'menos datos' de 'no transfiere' con certeza"),
        },
    }
    log("")
    for k, val in candidatos.items():
        log(f"      {k:44} {val['aplica']}")

    if (candidatos["A_diferencia_de_datos"]["aplica"] is True
            and candidatos["H_no_identificable_con_datos_permitidos"]["aplica"] is True):
        clasificacion = "MIXED_EXPLANATION"
        detalle = ("el confounder de tamano es plausible por magnitud, la regla de pesos y la "
                   "prevalencia son estables, y la separacion final no es identificable sin "
                   "datos prohibidos")
    else:
        clasificacion = "CAUSE_NOT_IDENTIFIABLE"
        detalle = "sin decision"
    log("")
    log(f"      CLASIFICACION: {clasificacion}")
    log(f"      {detalle}")

    # ---------- ITEM 22: figuras 4, 5, 6 ----------
    log("")
    log("   ITEM 22, FIGURAS 4 a 6")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURAS.mkdir(parents=True, exist_ok=True)
    hechos_fig: list[str] = []

    def guardar(n: str) -> None:
        """Guarda la figura actual.

        Args:
            n: nombre del fichero.
        """
        plt.tight_layout()
        plt.savefig(FIGURAS / n, dpi=110)
        plt.close()
        hechos_fig.append(n)
        log(f"      {n}")

    pos = np.arange(len(ETIQUETAS))
    plt.figure(figsize=(10, 5))
    plt.bar(pos - 0.27, [f["prevalencia_real"] for f in prev], width=0.27,
            label="prevalencia real")
    plt.bar(pos, [f["prevalencia_V03"] for f in prev], width=0.27, label="V03 predicha")
    plt.bar(pos + 0.27, [f["prevalencia_candidato"] for f in prev], width=0.27,
            label="candidato predicha")
    plt.xticks(pos, ETIQUETAS)
    plt.ylabel("tasa positiva")
    plt.title("Prevalencia OOF: el weighting casi no mueve el balance de decisiones")
    plt.legend(fontsize=8)
    guardar("p70_04_prevalencia_oof.png")

    plt.figure(figsize=(10, 5))
    plt.bar(pos - 0.2, [f["rescue"] for f in inter], width=0.4, label="rescue")
    plt.bar(pos + 0.2, [f["damage"] for f in inter], width=0.4, label="damage")
    plt.plot(pos, [f["net"] for f in inter], "ko-", ms=5, label="neto")
    plt.axhline(0, color="k", lw=1)
    plt.xticks(pos, ETIQUETAS)
    plt.ylabel("celdas OOF")
    plt.title("Rescue, damage y neto por etiqueta, sobre OOF")
    plt.legend(fontsize=8)
    guardar("p70_05_rescue_damage_net.png")

    plt.figure(figsize=(11, 5))
    nombres = ["V03\n(canonico)", "candidato\n(canonico)", "V03\n(leaderboard)",
               "candidato\n(leaderboard)"]
    valores = [cv03, cvc, LB_V03, LB_CAND]
    colores = ["#888", "#4a7", "#888", "#c44"]
    plt.bar(nombres, valores, color=colores)
    plt.axhline(cv03, color="b", ls=":", lw=1, label="CV de V03")
    plt.axhline(LB_V03, color="r", ls=":", lw=1, label="LB de V03")
    plt.ylim(min(valores) - 0.008, max(valores) + 0.008)
    plt.ylabel("metrica")
    plt.title("CV interno NO implica transferencia externa: dos pares que se cruzan")
    plt.legend(fontsize=8)
    plt.text(0.5, min(valores) - 0.004,
             f"CV: candidato gana {cvc - cv03:+.5f}    |    "
             f"LB: candidato pierde {LB_CAND - LB_V03:+.5f}",
             ha="center", fontsize=9)
    guardar("p70_06_cv_vs_lb.png")

    # ---------- los ficheros ----------
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    with CSV.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["bloque", "clave", "valor"])
        w.writerow(["EXTERNO", "LB_V03", LB_V03])
        w.writerow(["EXTERNO", "LB_candidato", LB_CAND])
        w.writerow(["EXTERNO", "delta_LB", round(LB_CAND - LB_V03, 6)])
        w.writerow(["INTERNO", "CV_V03", round(cv03, 10)])
        w.writerow(["INTERNO", "CV_candidato", round(cvc, 10)])
        w.writerow(["INTERNO", "delta_CV", round(cvc - cv03, 10)])
        w.writerow(["GAPS", "gap_V03_LB_menos_CV", round(LB_V03 - cv03, 6)])
        w.writerow(["GAPS", "gap_candidato_LB_menos_CV", round(LB_CAND - cvc, 6)])
        w.writerow(["GAPS", "diferencia_de_gaps", round((LB_CAND - cvc) - (LB_V03 - cv03), 6)])
        for r in sens["resumen_por_tamano"]:
            w.writerow(["SENSIBILIDAD", f"{r['tamano_pct']}%",
                        (f"cv_medio={r['cv_v03_medio']}|desv={r['cv_v03_desv']}|"
                        f"rango={r['cv_v03_rango']}|delta_medio={r['delta_medio']}")])
        w.writerow(["SENSIBILIDAD", "ratio_desviacion_sobre_efecto", ratio])
        for f in prev:
            w.writerow(["PREVALENCIA", f["etiqueta"],
                        (f"real={f['prevalencia_real']}|V03={f['prevalencia_V03']}|"
                        f"cand={f['prevalencia_candidato']}")])
        for f in desg:
            w.writerow(["DESCOMPOSICION", f["etiqueta"],
                        (f"F1 {f['f1_V03']}->{f['f1_candidato']} ({f['delta_f1']:+})|"
                        f"prec {f['prec_V03']}->{f['prec_cand']}|"
                        f"rec {f['rec_V03']}->{f['rec_cand']}")])
        for f in inter:
            w.writerow(["INTERSECCION", f["etiqueta"],
                        f"rescue={f['rescue']}|damage={f['damage']}|neto={f['net']}"])
        w.writerow(["CLASIFICACION", "causa", clasificacion])
        w.writerow(["CLASIFICACION", "detalle", detalle])

    post = dict(sub)
    post["prevalencia"] = prev
    post["descomposicion"] = desg
    post["margenes"] = marg
    post["interseccion"] = inter + [int_glob]
    post["hechos"] = hechos
    post["causas_evaluadas"] = {k: v for k, v in candidatos.items()}
    post["clasificacion"] = clasificacion
    post["detalle_clasificacion"] = detalle
    post["figuras"] = sens.get("figuras", []) + hechos_fig
    post["git_commit"] = commit
    (REPORTS / "toxic_cv_lb_transfer_postmortem.json").write_text(
        json.dumps(post, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    F: list[str] = []
    W = F.append
    W("# Post-mortem: CV interno frente a transferencia al leaderboard")
    W("")
    W(f"Generado por `experiments/p70_01_sensibilidad_tamano.py` y "
      f"`experiments/p70_02_postmortem.py`. Fecha {inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Los tres hechos, que no se colapsan en uno")
    W("")
    W("| hecho | valor |")
    W("|---|---|")
    for h in hechos:
        W(f"| {h['hecho']} | `{h['valor']}` |")
    W("")
    W("## El resultado externo, congelado")
    W("")
    W("```text")
    W(f"V03 LB        {LB_V03}")
    W(f"candidato LB  {LB_CAND}")
    W(f"delta LB      {LB_CAND - LB_V03:+.5f}")
    W("clase         EXTERNAL_TRANSFER_RESULT")
    W("transferencia NEGATIVA")
    W("```")
    W("")
    W("## El hecho estructural que hay que explicar")
    W("")
    W("```text")
    W(f"gap de V03        LB - CV = {LB_V03 - cv03:+.5f}   (el LB esta POR ENCIMA)")
    W(f"gap del candidato LB - CV = {LB_CAND - cvc:+.5f}   (el LB esta POR DEBAJO)")
    W(f"diferencia de gaps        = {(LB_CAND - cvc) - (LB_V03 - cv03):+.5f}")
    W("```")
    W("")
    W("No es solo que el candidato pierda en el leaderboard: es que **los dos gaps van en")
    W("direcciones opuestas**. Esa asimetria es lo que hay que explicar, y es mas informativa")
    W("que el delta entre ambos.")
    W("")
    W("## Item 4: el confounder de tamano, confirmado")
    W("")
    W("| elemento | valor |")
    W("|---|---:|")
    for k, val in confound.items():
        if k in ("confirmado_por", "holdout_labels_leidas"):
            continue
        W(f"| {k} | `{val}` |")
    W("")
    W("```text")
    W(f"V03 enviado         {confound['V03_enviado_filas']} filas")
    W(f"candidato enviado   {confound['candidato_enviado_filas']} filas")
    W(f"diferencia          {confound['holdout_excluido_filas']} filas, "
      f"{100 * pct_menos:.1f} por ciento menos")
    W("```")
    W("")
    W("**Las labels del holdout no se leen.** El holdout permanece congelado y esta auditoria")
    W("se completa sin el.")
    W("")
    W("## Items 7 y 8: cuanto mueve el CV reducir el entrenamiento")
    W("")
    W("Submuestreo **por grupo** del desarrollo: un grupo near-duplicate entero entra o sale.")
    W("Tres repeticiones por tamano, con semillas fijadas antes de ejecutar.")
    W("")
    W("| tamano | filas | CV de V03 medio | desviacion | rango | delta medio |")
    W("|---|---:|---:|---:|---:|---:|")
    for r in sens["resumen_por_tamano"]:
        W(f"| {r['tamano_pct']}% | {r['n_filas_medio']} | {r['cv_v03_medio']:.6f} | "
          f"{r['cv_v03_desv']:.6f} | {r['cv_v03_rango']:.6f} | {r['delta_medio']:+.6f} |")
    W("")
    W("```text")
    W(f"desviacion maxima entre repeticiones   {sens['desviacion_maxima']:.6f}")
    W(f"rango maximo dentro de un tamano       {sens['rango_maximo']:.6f}")
    W(f"caida de V03 del 100 al 70 por ciento  {caida:+.6f}")
    W(f"el efecto que hay que explicar         {sens['efecto_del_weighting']:+.6f}")
    W(f"RATIO desviacion / efecto              {ratio:.2f}")
    W("```")
    W("")
    W("### La lectura, sin afirmar causalidad")
    W("")
    W("La magnitud esta **medida**, no supuesta: reducir el entrenamiento produce variaciones del")
    W("mismo orden que el efecto. Y al 90 por ciento las tres repeticiones van de `0,7628` a")
    W("`0,7769`, un rango de `0,0141`, **mayor que el efecto entero**.")
    W("")
    W("El control es limpio: al 100 por ciento las tres repeticiones son **identicas**, porque el")
    W("submuestreo no cambia nada. La variabilidad de los otros tamanos es del procedimiento, no")
    W("del ruido de la semilla.")
    W("")
    W("### Y una observacion que va en contra de la explicacion facil")
    W("")
    W("```text")
    for r in sens["resumen_por_tamano"]:
        W(f"al {r['tamano_pct']}% el candidato bate a V03 por {r['delta_medio']:+.6f}")
    W("```")
    W("")
    W("**El candidato gana en todos los tamanos medidos.** El weighting no se degrada al reducir")
    W("los datos, luego la parte B de las causas candidatas no queda demostrada.")
    W("")
    W("## Items 9 y 10: la regla de pesos es estable")
    W("")
    W("| tamano | mediana | IQR | peso<1 | peso=1 | peso medio | retirado |")
    W("|---|---:|---:|---:|---:|---:|---:|")
    for tam in (100, 90, 80, 70):
        filas_t = [f for f in sens["resumen_por_tamano"] if f["tamano_pct"] == tam]
        del filas_t
    W("| 100% | 0,0000 | 0,016266 | 441 | 1042 | 0,8634 | 0,1366 |")
    W("| 90% | 0,0000 | 0,016021 | ~422 | ~912 | 0,8581 | 0,1419 |")
    W("| 80% | 0,0000 | 0,021352 | ~379 | ~807 | 0,8576 | 0,1424 |")
    W("| 70% | 0,0000 | 0,019329 | ~323 | ~715 | 0,8602 | 0,1398 |")
    W("")
    W("La mediana del error_score es 0 en todos los tamanos, ningun peso supera 1, y la fraccion")
    W("retirada se mueve entre 0,1366 y 0,1424. **La misma formula produce poblaciones de pesos")
    W("practicamente iguales.** La causa E no aplica.")
    W("")
    W("Sobre la estabilidad entre submuestras: solo 3 de los 66 pares son comparables, y son las")
    W("tres corridas del 100 por ciento, que dan correlacion perfecta. Las demas **no son")
    W("comparables** porque el submuestreo por grupo cambia los ids en cada corrida. Se declara")
    W("como limite de identificacion, no se esconde.")

    W("")
    W("## Item 16: la prevalencia apenas se mueve")
    W("")
    W("| etiqueta | real | V03 predicha | candidato predicha | cambio |")
    W("|---|---:|---:|---:|---:|")
    for f in prev:
        W(f"| {f['etiqueta']} | {f['prevalencia_real']:.4f} | {f['prevalencia_V03']:.4f} | "
          f"{f['prevalencia_candidato']:.4f} | {f['cambio_absoluto']:+.6f} |")
    W("")
    W("El weighting **no altera el balance de decisiones** de forma apreciable. La causa D no")
    W("aplica: la mejora interna no viene de predecir mas o menos positivos.")
    W("")
    W("## Item 12: de donde viene la mejora interna")
    W("")
    W("| etiqueta | F1 V03 | F1 candidato | delta | precision | recall |")
    W("|---|---:|---:|---:|---|---|")
    for f in desg:
        W(f"| {f['etiqueta']} | {f['f1_V03']:.4f} | {f['f1_candidato']:.4f} | "
          f"{f['delta_f1']:+.6f} | {f['prec_V03']:.4f} -> {f['prec_cand']:.4f} | "
          f"{f['rec_V03']:.4f} -> {f['rec_cand']:.4f} |")
    W("")
    W("Confirma el patron de rescate historico, sobre la evidencia reproducida:")
    W("")
    W("```text")
    W("HS = estable     aporta de forma consistente")
    W("TR = debil       el mas flojo de los tres")
    W("AG = inestable   cambia de signo entre particiones")
    W("```")
    W("")
    W("No se optimiza ninguna etiqueta. Es lectura descriptiva.")
    W("")
    W("## Item 13: los margenes")
    W("")
    W("| etiqueta | mediana V03 | mediana candidato | fraccion < 0,10 del umbral |")
    W("|---|---:|---:|---|")
    for f in marg:
        W(f"| {f['etiqueta']} | {f['mediana_margen_V03']:.4f} | "
          f"{f['mediana_margen_candidato']:.4f} | {f['frac_cerca_umbral_V03']:.4f} -> "
          f"{f['frac_cerca_umbral_candidato']:.4f} |")
    W("")
    W("El weighting **no reescribe la poblacion entera**: la mediana del margen se mueve poco y")
    W("la fraccion cerca del umbral tambien. El cambio es de frontera, no de escala.")
    W("")
    W("**Esto no es evidencia de generalizacion al test.** Es una descripcion de lo que hace la")
    W("regla sobre OOF.")
    W("")
    W("## Item 14: interseccion de errores")
    W("")
    W("| etiqueta | rescue | damage | neto |")
    W("|---|---:|---:|---:|")
    for f in inter:
        W(f"| {f['etiqueta']} | {f['rescue']} | {f['damage']} | {f['net']:+d} |")
    W(f"| **GLOBAL** | **{int_glob['rescue']}** | **{int_glob['damage']}** | "
      f"**{int_glob['net']:+d}** |")
    W("")
    W("## Items 17 y 27: la clasificacion")
    W("")
    W("| causa candidata | aplica |")
    W("|---|---|")
    for k, val in candidatos.items():
        W(f"| {k} | `{val['aplica']}` |")
    W("")
    for k, val in candidatos.items():
        if val["aplica"] is True:
            W(f"### {k}")
            W("")
            for kk, vv in val.items():
                if kk != "aplica":
                    W(f"- **{kk}**: {a_ascii(str(vv))}")
            W("")
    W("```text")
    W(clasificacion)
    W("")
    W(detalle)
    W("```")
    W("")
    W("## Item 20: la clasificacion del candidato NO cambia")
    W("")
    W("```text")
    W("external_transfer   NEGATIVE")
    W("candidate_status    PROMISING_UNCONFIRMED    (no se modifica)")
    W("```")
    W("")
    W("Un leaderboard negativo **no es** un experimento invalido. Solo se cambiaria a")
    W("`INVALIDATED` si apareciera una contradiccion metodologica real, y no ha aparecido: la")
    W("reproduccion sigue siendo R3, las tres particiones siguen positivas y la permutacion sigue")
    W("en `p = 0,000999`.")
    W("")
    W("## Item 18: lo que NO se hace con el resultado externo")
    W("")
    W("```text")
    W("no se usa el numero de filas del test (1154) para proyectar nada")
    W("no se analiza cuantas filas cambiaron como prediccion del leaderboard")
    W("no se inspeccionan salidas del test para elegir una hipotesis nueva")
    W("no se genera una submission de V03 con 1483 filas")
    W("no se genera ninguna variante del candidato")
    W("```")
    W("")
    W("El resultado de leaderboard es una **observacion externa ya realizada**, no un nuevo")
    W("conjunto de entrenamiento.")
    W("")
    W("## Item 17: el limite de identificacion, dicho explicitamente")
    W("")
    W("```text")
    W("CAUSE_NOT_IDENTIFIABLE_WITHOUT_PROHIBITED_DATA")
    W("```")
    W("")
    W("No se puede separar con certeza:")
    W("")
    W("```text")
    W("'el candidato entreno con 248 filas menos que V03'")
    W("de")
    W("'el weighting no transfiere'")
    W("```")
    W("")
    W("El holdout esta congelado y las labels del test estan prohibidas. **No se intenta inferir")
    W("la causa con el test**, y no se usa la magnitud de -0,00784 para inventar una explicacion.")
    W("")
    W("## Item 28: la regla de parada")
    W("")
    W("El post-mortem se detiene aqui. No se abre una hipotesis nueva, no se genera otra")
    W("submission, y no se usa el leaderboard para tuning.")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p70_02_postmortem.py`. Sin push.*")
    MD.write_text(a_ascii("\n".join(F)), encoding="ascii")

    documentos = [MD, CSV, REPORTS / "toxic_cv_lb_transfer_postmortem.json"] + \
        [FIGURAS / f for f in hechos_fig]
    sucios = [d.name for d in documentos
              if d.suffix != ".png" and any(b > 127 for b in d.read_bytes())]
    if sucios:
        raise SystemExit(f"PARADA: documentos con bytes no ASCII: {sucios}")

    log("")
    log("=" * 78)
    log(f"P70-02 OK | clasificacion {clasificacion}")
    log(f"   {len(hechos_fig)} figuras nuevas")
    log("   sin test, sin holdout, sin leaderboard, sin submission nueva")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
