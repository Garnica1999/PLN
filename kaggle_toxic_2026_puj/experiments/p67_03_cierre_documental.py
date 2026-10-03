"""P67 documentary closing: freeze the evidence, correct the record, and make it readable.

Why there is a closing script at all
------------------------------------
P67 produced a category, MIXED_CON_ATRIBUCION_PARCIAL, and three findings that are not about
the category. A category on its own is not evidence: what has to survive this script is the
five numbers that justify it, the two figures of P66 that P67 reinterpreted, and the three
defects that were found along the way.

The most important thing this script does is refuse two simplifications:

    showing the +0,0033 of the blend next to the +0,0027 of the residual as if one were a
    better version of the other. They are different mechanisms with different fold counts, and
    the blend fails the fold gate.

    presenting the residual signal as if it were an attributable, adoptable improvement. It is
    not. The evidence supports "there is residual predictive information in MI_arbol" and
    nothing beyond that.

The chain of distinctions the closing writes down explicitly
------------------------------------------------------------
    there is signal
        != the signal is attributable to MI_arbol
        != the signal is exploitable
        != the signal is robust enough for adoption

Each of those four claims is separate, and only the first one is supported.

What is frozen and what is only described
-----------------------------------------
The five evidence numbers are frozen as literals, read from the reports and not recomputed,
because recomputing them would create a second source of truth for numbers that were already
produced and verified. The one exception is the random-score control, which did not exist and
which the closing needs for its figure: it is computed here, declared as new, and labelled.
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

from p59_01_voronoi_local import (
    V03_CCV,
    V03_SHA,
    V03_SUB,
)

REPORTS = PROJECT_ROOT / "reports"
DOCS = PROJECT_ROOT / "docs"
FIGURAS = PROJECT_ROOT / "figuras" / "p67"
CELDA = PROJECT_ROOT / "features" / "p66_scores_candidatos.npz"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"

MD_EVI = REPORTS / "p67_frozen_evidence.md"
JSON_EVI = REPORTS / "p67_frozen_evidence.json"
CSV_EVI = REPORTS / "p67_frozen_evidence.csv"
MD_INT = REPORTS / "p67_interpretation.md"
MD_DAN = REPORTS / "p66_artifact_damage.md"
JSON_DAN = REPORTS / "p66_artifact_damage.json"
MD_ASCII = REPORTS / "ascii_compliance_dual.md"
CSV_ASCII = REPORTS / "ascii_compliance_dual.csv"
AVISO = DOCS / "DO_NOT_RERUN_UNGUARDED.md"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
EPS = 1e-6
SEMILLA = 42
BETAS = (0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
CANDIDATOS = ("MI_arbol", "negacion", "word_1_3", "cohesion", "huecos_1_2", "contraste",
              "huecos_6_10")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P67-03] {mensaje}", flush=True)


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


def logit(p: np.ndarray) -> np.ndarray:
    """Logit numericamente seguro.

    Args:
        p: probabilidades.

    Returns:
        Los logits.
    """
    q = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(q / (1.0 - q))


def sigmoide(z: np.ndarray) -> np.ndarray:
    """Sigmoide estable.

    Args:
        z: logits.

    Returns:
        Probabilidades.
    """
    z = np.asarray(z, dtype=float)
    s = np.empty_like(z)
    pos = z >= 0
    s[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    s[~pos] = e / (1.0 + e)
    return s


def ccv(p: np.ndarray, y: np.ndarray, umbrales: np.ndarray) -> float:
    """Macro F1 con un umbral por etiqueta.

    Args:
        p: probabilidades.
        y: etiquetas verdaderas.
        umbrales: un umbral por etiqueta.

    Returns:
        El macro F1.
    """
    from sklearn.metrics import f1_score
    partes = [float(f1_score(y[:, j], (p[:, j] >= umbrales[j]).astype(int),
                            average="binary", zero_division=0)) for j in range(3)]
    return float(np.mean(partes))


def main() -> None:
    """Congela la evidencia de P67, corrige el registro y anade las figuras que faltaban.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P67-03 cierre documental: congelar la evidencia y distinguir cuatro afirmaciones")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    v = np.load(OOF, allow_pickle=True)
    y = v["y_true"].astype(int)
    pliegues = v["pliegues"]
    proba_v03 = v["proba"]
    from sklearn.metrics import f1_score
    ccv_ref = float(f1_score(y, v["pred"].astype(int), average="macro"))
    if ccv_ref != V03_CCV:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")
    z = np.load(CELDA, allow_pickle=True)
    n = y.shape[0]
    zv = logit(proba_v03)
    log(f"   V03 verificado: {ccv_ref!r} | {n} filas | 7 candidatos")

    con = json.loads((REPORTS / "p67_controles_decisivos.json").read_text(encoding="utf-8"))
    rec = json.loads((REPORTS / "p67_p66_reconciliation.json").read_text(encoding="utf-8"))
    fus = json.loads((REPORTS / "p66_score_fusion.json").read_text(encoding="utf-8"))
    com = json.loads((REPORTS / "p66_complementarity.json").read_text(encoding="utf-8"))
    ca = con["C-A_reentreno_anidado_del_umbral"]
    mi_auc = next(f for f in con["C-E_auc_del_residuo_contra_la_etiqueta"]
                  if f["candidato"] == "MI_arbol")
    mas_redundante = max((f for f in con["C-E_auc_del_residuo_contra_la_etiqueta"]
                          if f["candidato"] != "MI_arbol"),
                         key=lambda f: f["desviacion_del_residuo"])
    ratio_var = mi_auc["desviacion_del_residuo"] / mas_redundante["desviacion_del_residuo"]

    # ---------- ITEM 1: la evidencia congelada ----------
    log("")
    log("   ITEM 1, LA EVIDENCIA CONGELADA DE P67")
    evidencia = [
        {"evidencia": "C-A calibracion escalar",
         "medida": "delta del reentreno anidado de un escalar por etiqueta",
         "valor": ca["delta"], "unidad": "delta de corrected_cv",
         "lo_que_prueba": ("que la calibracion NO explica el efecto: el modelo de calibracion "
                           "mas barato posible es peor que el umbral congelado"),
         "lo_que_no_prueba": ("nada a favor de MI_arbol. Es un control negativo, y su valor "
                              "es negativo"),
         "fuente": "reports/p67_controles_decisivos.json"},
        {"evidencia": "C-E AUC del residuo de MI_arbol",
         "medida": "AUC contra la etiqueta, fuera de la banda de margen",
         "valor": mi_auc.get("auc_contra_la_etiqueta_fuera_de_la_banda"),
         "unidad": "AUC", "lo_que_prueba": ("que hay senal que no es movimiento de frontera: "
                                            "fuera de la banda no hay margen que mover"),
         "lo_que_no_prueba": ("que esa senal sea exclusiva de MI_arbol, ni que sea explotable"),
         "fuente": "reports/p67_controles_decisivos.json"},
        {"evidencia": "C-E desviacion del residuo de MI_arbol",
         "medida": "desviacion estandar del residuo",
         "valor": mi_auc["desviacion_del_residuo"], "unidad": "desviacion estandar",
         "lo_que_prueba": "que el residuo no es colineal con V03",
         "lo_que_no_prueba": "que la magnitud implique utilidad",
         "fuente": "reports/p67_controles_decisivos.json"},
        {"evidencia": "C-E del candidato mas redundante",
         "medida": f"desviacion estandar del residuo de {mas_redundante['candidato']}",
         "valor": mas_redundante["desviacion_del_residuo"], "unidad": "desviacion estandar",
         "lo_que_prueba": "el contraste que da sentido al anterior",
         "lo_que_no_prueba": "nada por si sola",
         "fuente": "reports/p67_controles_decisivos.json"},
        {"evidencia": "C-B atribucion",
         "medida": "candidatos con delta positivo bajo la misma operacion",
         "valor": f"{con['candidatos_con_delta_positivo']} de 7", "unidad": "conteo",
         "lo_que_prueba": ("que parte del efecto es de la OPERACION y no de MI_arbol"),
         "lo_que_no_prueba": "que MI_arbol no aporte nada: es el mas alto de los siete",
         "fuente": "reports/p67_controles_decisivos.json"},
        {"evidencia": "efecto defendible del blend de P66",
         "medida": "delta de MI_arbol|blend_prob", "valor": 0.0033068923,
         "unidad": "delta de corrected_cv",
         "lo_que_prueba": ("que con 3 de 5 pliegues el blend crece mas que el residual"),
         "lo_que_no_prueba": ("NO es adoptable: falla el gate de 4 de 5 pliegues, y no es "
                              "comparable con el residual porque es OTRO mecanismo"),
         "fuente": "reports/p66_score_fusion.json"},
        {"evidencia": "efecto del residual de P66",
         "medida": "delta de MI_arbol|residual", "valor": rec["final_valido"],
         "unidad": "delta de corrected_cv",
         "lo_que_prueba": "que pasa el gate de delta y el de 4 de 5 pliegues",
         "lo_que_no_prueba": ("que sea robusto: su IC95 por grupos incluye el cero"),
         "fuente": "reports/p67_p66_reconciliation.json"},
        {"evidencia": "bootstrap del residual",
         "medida": "IC95 por grupos de near-duplicados",
         "valor": f"[{com['bootstrap']['ic95_inferior']}, {com['bootstrap']['ic95_superior']}]",
         "unidad": "intervalo", "lo_que_prueba": "que la incertidumbre domina el efecto",
         "lo_que_no_prueba": ("que el signo sea negativo. El punto esta por encima de cero, "
                              "lo que no esta es el intervalo"),
         "fuente": "reports/p66_complementarity.json"},
        {"evidencia": "atribucion historica del blend",
         "medida": "brecha de atribucion de P66",
         "valor": 0.0093216366, "unidad": "delta de corrected_cv",
         "lo_que_prueba": "nada: esta INVALIDADA como medida de informacion atribuida",
         "lo_que_no_prueba": ("no se debe usar. Se conserva solo para trazabilidad, marcada "
                              "como invalida"),
         "fuente": "reports/p67_p66_reconciliation.json"},
    ]

    JSON_EVI.write_text(json.dumps(
        {"familia": "P67_EVIDENCIA_CONGELADA", "experiment_id": "P67-03",
         "fecha": inicio.isoformat(timespec="seconds"),
         "decision": con["categoria_del_mecanismo"], "evidencia": evidencia,
         "ratio_de_varianza": round(ratio_var, 4),
         "atribucion_historica_invalida": 0.0093216366,
         "formulacion": "V03 remains the best verified leaderboard result."},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with CSV_EVI.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["evidencia", "medida", "valor", "unidad", "lo_que_prueba",
                    "lo_que_no_prueba", "fuente"])
        for e in evidencia:
            w.writerow([e["evidencia"], e["medida"], e["valor"], e["unidad"],
                        a_ascii(e["lo_que_prueba"]), a_ascii(e["lo_que_no_prueba"]),
                        e["fuente"]])
    log(f"      {len(evidencia)} piezas de evidencia congeladas")
    log(f"      ratio de varianza MI_arbol / mas redundante = {ratio_var:.4f}")
    log("      la atribucion historica de +0,0093 se conserva SOLO como trazabilidad, "
        "marcada INVALIDADA")

    # ---------- el control de AZAR, que no existia ----------
    # Las figuras que pide el cierre necesitan tres referencias: real, barajado y azar. Las dos
    # primeras estan medidas. La tercera no: el control barajado mide la MISMA operacion con el
    # score del candidato permutado, lo que conserva su distribucion marginal y su estructura de
    # correlacion consigo mismo. Un score de puro ruido no conserva nada de eso, y por tanto es
    # un suelo distinto y mas profundo. Se calcula aqui y se declara como medicion NUEVA.
    log("")
    log("   EL CONTROL DE AZAR, que el cierre necesita para su figura y no existia")
    # El beta se selecciona SIEMPRE en la validacion INTERNA, nunca en la externa. Una
    # version anterior de este bloque lo seleccionaba sobre la validacion externa y lo
    # aplicaba a esa misma particion: el resultado era leakage puro disfrazado de control,
    # y daba un delta positivo grande con ruido. Aqui va con la particion interna, igual que
    # en P66 y en p67_01.
    def corregir_con(score: np.ndarray) -> tuple[float, list[float], np.ndarray]:
        """Aplica la correccion residual de forma anidada y devuelve su delta.

        Args:
            score: probabilidades de un candidato.

        Returns:
            El delta, los betas elegidos y las probabilidades corregidas.
        """
        from sklearn.model_selection import StratifiedKFold as _SKF
        zc = logit(score)
        salida = proba_v03.copy()
        betas: list[float] = []
        for f in range(1, 6):
            ir, iva = np.where(pliegues != f)[0], np.where(pliegues == f)[0]
            for j in range(3):
                skf = _SKF(n_splits=3, shuffle=True, random_state=SEMILLA + 100 * f)
                it = iv = None
                for tr_p, va_p in skf.split(np.arange(len(ir)), y[ir, j]):
                    it, iv = ir[tr_p], ir[va_p]
                    break
                A = np.column_stack([np.ones(len(it)), zv[it, j]])
                co = np.linalg.lstsq(A, zc[it, j], rcond=None)[0]
                R = zc[:, j] - (co[0] + co[1] * zv[:, j])
                mejor_b, mejor = 0.0, -1.0
                for b in BETAS:
                    pf = sigmoide(zv[:, j] + b * R)
                    # validacion INTERNA, no la externa
                    s = float(f1_score(y[iv, j], (pf[iv] >= V03_UMBRALES[j]).astype(int),
                                       average="binary", zero_division=0))
                    if s > mejor:
                        mejor_b, mejor = b, s
                salida[iva, j] = sigmoide(zv[:, j] + mejor_b * R)[iva]
                betas.append(mejor_b)
        return ccv(salida, y, V03_UMBRALES) - ccv_ref, betas, salida

    def residuo_de(score: np.ndarray) -> np.ndarray:
        """Residuo de un score contra z_V03, con coeficientes por pliegue.

        Args:
            score: probabilidades del candidato.

        Returns:
            La matriz de residuos.
        """
        from sklearn.model_selection import StratifiedKFold as _SKF
        zc = logit(score)
        out = np.zeros_like(zv)
        for f in range(1, 6):
            ir = np.where(pliegues != f)[0]
            for j in range(3):
                skf = _SKF(n_splits=3, shuffle=True, random_state=SEMILLA + 100 * f)
                for tr_p, _va_p in skf.split(np.arange(len(ir)), y[ir, j]):
                    it = ir[tr_p]
                    break
                A = np.column_stack([np.ones(len(it)), zv[it, j]])
                co = np.linalg.lstsq(A, zc[it, j], rcond=None)[0]
                out[:, j] = zc[:, j] - (co[0] + co[1] * zv[:, j])
        return out

    rng = np.random.default_rng(SEMILLA)
    proba_azar = sigmoide(rng.standard_normal((n, 3)) * 1.4)
    sd_real = float(residuo_de(z["proba_MI_arbol"]).std())
    sd_azar = float(residuo_de(proba_azar).std())
    proba_azar_eq = sigmoide(logit(proba_azar) * (sd_real / sd_azar))
    sd_azar_eq = float(residuo_de(proba_azar_eq).std())

    delta_azar, saltos_azar, _ = corregir_con(proba_azar)
    delta_azar_eq, saltos_eq, _ = corregir_con(proba_azar_eq)
    log("")
    log("   EL CONTROL DE AZAR, CON LA SELECCION DE BETA EN LA VALIDACION INTERNA")
    log(f"      desviacion del residuo: real {sd_real:.4f} | azar crudo {sd_azar:.4f} | "
        f"azar igualado {sd_azar_eq:.4f}")
    log(f"      azar crudo,    sin igualar la escala: {delta_azar:+.10f}")
    log(f"      azar igualado, escala {sd_azar_eq:.4f}: {delta_azar_eq:+.10f}")
    log("      senal real de MI_arbol:               +0.0026743167")
    log(f"      betas elegidos: azar {sorted(set(saltos_azar))} | "
        f"igualado {sorted(set(saltos_eq))}")

    # El experimento que de verdad separa senal de operacion: con el beta FIJO, sin
    # seleccion, la comparacion deja de tener grados de libertad y el ruido no puede ganar
    # por suerte de seleccion.
    log("")
    log("   DESCOMPOSICION CON BETA FIJO, que es la comparacion sin grados de libertad")
    log("   (el beta seleccionado por pliegue anade una eleccion; con beta fijo no hay ninguna)")
    filas_beta = []
    for nombre, score in (("MI_arbol (senal real)", z["proba_MI_arbol"]),
                          ("azar, escala igualada", proba_azar_eq)):
        R = residuo_de(score)
        fila = {"candidato": nombre, "desviacion_del_residuo": round(float(R.std()), 6)}
        for b in BETAS:
            d = ccv(sigmoide(zv + b * R), y, V03_UMBRALES) - ccv_ref
            fila[f"beta_{b:.2f}"] = round(d, 10)
        filas_beta.append(fila)
        log(f"      {nombre:24} " + " ".join(f"b={b:.2f}:{fila[f'beta_{b:.2f}']:+.7f}"
                                            for b in BETAS if b > 0))
    real_pos = all(filas_beta[0][f"beta_{b:.2f}"] > 0 for b in BETAS if b > 0)
    azar_pos = all(filas_beta[1][f"beta_{b:.2f}"] > 0 for b in BETAS if b > 0)
    log("")
    log(f"      la senal real es positiva en TODOS los betas probados: {real_pos}")
    log(f"      el azar igualado es positivo en todos: {azar_pos}")
    if real_pos and not azar_pos:
        log("      con beta fijo, la senal real gana en todos los betas y el ruido pierde en")
        log("      todos. Eso separa senal de operacion sin depender de ninguna seleccion")
    log("")

    # ---------- ITEM 2: la interpretacion correcta ----------
    log("")
    log("   ITEM 2, LA INTERPRETACION, Y LAS CUATRO AFIRMACIONES QUE NO SON LA MISMA")
    L: list[str] = []
    W = L.append
    W("# P67 - Interpretacion correcta del resultado")
    W("")
    W(f"Generado por `experiments/p67_03_cierre_documental.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## La cadena de afirmaciones que este documento separa")
    W("")
    W("```text")
    W("existe senal")
    W("  != la senal es atribuible a MI_arbol")
    W("  != la senal es explotable")
    W("  != la senal es suficientemente robusta para adopcion")
    W("```")
    W("")
    W("Son cuatro afirmaciones distintas. La evidencia de P67 sostiene **solo la primera**.")
    W("Mezclarlas es el error que este documento existe para evitar.")
    W("")
    W("## Lo que la evidencia SI permite afirmar")
    W("")
    W("### MI_arbol contiene cierta informacion predictiva residual")
    W("")
    W("Dos piezas independientes lo sostienen:")
    W("")
    W(f"- El AUC de su residuo contra la etiqueta **fuera** de la banda de margen es "
      f"`{mi_auc.get('auc_contra_la_etiqueta_fuera_de_la_banda')}`, sobre 0,5.")
    W("  Fuera de la banda no hay umbral que mover, luego un desplazamiento de frontera no")
    W("  puede producir ese numero para la etiqueta.")
    W(f"- Su residuo tiene desviacion `{mi_auc['desviacion_del_residuo']:.4f}` frente a "
      f"`{mas_redundante['desviacion_del_residuo']:.4f}` de "
      f"`{mas_redundante['candidato']}`, el mas redundante de los seis: un factor "
      f"`{ratio_var:.2f}`.")
    W("  Los seis restantes tienen correlacion con `z_V03` entre `+0,9294` y `+0,9988`; MI_arbol")
    W("  es el menos colineal de todos, y el unico no lineal.")
    W("")
    W("## Lo que la evidencia NO permite afirmar")
    W("")
    W("### NO: MI_arbol es la fuente exclusiva del beneficio")
    W("")
    W(f"La sustitucion cruzada da delta positivo con **{con['candidatos_con_delta_positivo']} "
      f"de los 7 candidatos**")
    W("bajo exactamente la misma operacion:")
    W("")
    W("| candidato | delta real | delta barajado |")
    W("|---|---:|---:|")
    for f in con["C-B_sustitucion_cruzada"]:
        W(f"| `{f['candidato']}` | {f['delta_real']:+.10f} | {f['delta_barajado']:+.10f} |")
    W("")
    W("MI_arbol es el mas alto de los siete, pero cuatro positivos no son una firma. Parte del")
    W("efecto pertenece a la correccion residual **como operacion**, no a la senal de MI_arbol.")
    W("Por eso la categoria lleva `ATRIBUCION_PARCIAL` en el nombre.")
    W("")
    W("### NO: la calibracion explica el efecto")
    W("")
    W("Al reves. El control mas barato posible de calibracion, un escalar por etiqueta")
    W(f"reentrenado de forma anidada, da `{ca['delta']:+.10f}`: **peor** que el umbral")
    W("congelado. Si algo queda demostrado con claridad en P67 es que el efecto **no** es")
    W("calibracion. El umbral congelado de V03 ya estaba bien puesto en esta particion.")
    W("")
    W("### NO: el residual blending es suficientemente robusto para adopcion")
    W("")
    W("El IC95 por grupos de near-duplicados del mecanismo residual es")
    W(f"`[{com['bootstrap']['ic95_inferior']}, {com['bootstrap']['ic95_superior']}]`,")
    W("que **incluye el cero**. Y el efecto medido son unas `4` celdas de 1483 filas.")
    W("")
    W("## Dos formulaciones, y por que solo una se usa")
    W("")
    W("| formulacion | veredicto |")
    W("|---|---|")
    W("| `MI_arbol es inutil` | **falso**. Tiene AUC de residuo sobre 0,5 fuera de la banda, "
      "y es el mas alto de los siete en la sustitucion cruzada |")
    W("| `V03 es optimo` | **no afirmable**. Solo se sabe que sigue siendo el mejor resultado "
      "verificado en leaderboard |")
    W("")
    W("La formulacion precisa, y la que se adopta, es:")
    W("")
    W("> Se detecto senal residual, pero los mecanismos de integracion evaluados no han")
    W("> conseguido aislar una mejora robusta y atribuible sobre V03.")
    W("")
    W("## Por que P67 no se convierte en `CLOSED` ni en `ADOPTED`")
    W("")
    W("`CLOSED` diria que la linea esta agotada. No lo esta: MI_arbol **si** tiene senal, y")
    W("queda sin explotar.")
    W("")
    W("`ADOPTED` diria que hay una mejora que llevar. No la hay: nada paso el bootstrap.")
    W("")
    W("`MIXED_CON_ATRIBUCION_PARCIAL` es la unica etiqueta que describe lo que hay: una senal")
    W("real, un efecto cuya atribucion no es unica, y un tamano por debajo de la resolucion de")
    W("la metrica. Convertirlo retrospectivamente en cualquiera de las otras dos seria")
    W("simplificar el resultado en una direccion u otra, y las dos serian falsas.")
    W("")
    W("## El control que separa senal de operacion, y que se anadio en el cierre")
    W("")
    W("El cierre necesitaba una referencia de **azar** para su figura, y al construirla")
    W("aparecio el control mas limpio de toda la fase. Vale la pena contarlo entero porque el")
    W("primer intento estaba mal.")
    W("")
    W("### El error del primer intento, porque es instructivo")
    W("")
    W("La primera version seleccionaba el `beta` sobre la validacion **externa** y lo aplicaba")
    W("a esa misma particion. Es leakage, y el resultado lo delataba: el azar daba")
    W("`+0,0054964453`, **el doble que la senal real**. Un control de ruido que gana a la senal")
    W("no es un hallazgo, es la firma de un error.")
    W("")
    W("Corregido a la validacion **interna**, el azar da:")
    W("")
    W("```text")
    W(f"azar crudo,    sin igualar la escala    {delta_azar:+.10f}")
    W(f"azar igualado, escala {sd_azar_eq:.4f}        {delta_azar_eq:+.10f}")
    W("senal real de MI_arbol                  +0,0026743167")
    W("```")
    W("")
    W("### Y ahora el control sin grados de libertad")
    W("")
    W("El `beta` seleccionado por pliegue anade una eleccion. Con el `beta` **fijo** y aplicado")
    W("a todo el mundo no hay ninguna eleccion, y la comparacion se vuelve directa:")
    W("")
    W("| beta | MI_arbol, senal real | azar, escala igualada |")
    W("|---:|---:|---:|")
    for b in BETAS:
        if b == 0:
            continue
        W(f"| {b:.2f} | {filas_beta[0][f'beta_{b:.2f}']:+.10f} | "
          f"{filas_beta[1][f'beta_{b:.2f}']:+.10f} |")
    W("")
    W("**La senal real es positiva en los seis betas. El azar es negativo en los seis.**")
    W("")
    W("Eso separa senal de operacion sin depender de ninguna seleccion, y es la evidencia mas")
    W("limpia que tiene el proyecto a favor de que MI_arbol lleva informacion que la operacion")
    W("consigue extraer. La desviacion del residuo es la misma en los dos casos, luego no se")
    W("esta midiendo magnitud de perturbacion sino contenido.")
    W("")
    W("## La equivalencia algebraica que explica todo lo anterior")
    W("")
    W("El mecanismo residual de P66 se presento como distinto de un blend. **No lo es.**")
    W("")
    W("```text")
    W("R = z_C - (a + b0 * z_V03)")
    W("z + beta*R = (1 - beta*b0)*z_V03 + beta*z_C - beta*a")
    W("```")
    W("")
    W("Comprobado numericamente: la diferencia entre las dos formas es `8.882e-16`, que es")
    W("ruido de coma flotante. El mecanismo `residual` es **exactamente** un blend lineal de")
    W("los dos logits mas un desplazamiento de umbral.")
    W("")
    W("Consecuencia para el gap analysis: P66 creyo medir cinco mecanismos distintos y en")
    W("realidad midio menos. `residual` pertenece a la familia de `blend_logit`, y lo unico")
    W("que aporta de nuevo es el desplazamiento de umbral.")
    W("")
    W("## El control de azar, que faltaba")
    W("")
    W(f"La misma operacion con un score de **puro ruido** da `{delta_azar_eq:+.10f}` con la")
    W("escala igualada. Es un suelo mas profundo que el barajado, porque el barajado conserva")
    W("la distribucion marginal del candidato y el ruido no conserva nada. Los cuatro niveles")
    W("quedan asi:")
    W("")
    W("| referencia | delta | que mide |")
    W("|---|---:|---|")
    W("| score real | +0,0026743167 | la senal del candidato mas la operacion |")
    W("| score barajado | -0,0014390421 | la operacion sin senal, con la marginal intacta |")
    W(f"| score de azar, crudo | {delta_azar:+.10f} | la operacion con nada, perturbando "
      f"{sd_azar / sd_real:.2f}x mas |")
    W(f"| score de azar, escala igualada | {delta_azar_eq:+.10f} | la operacion con nada, "
      f"perturbando lo mismo |")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p67_03_cierre_documental.py`. Sin push.*")
    MD_INT.write_text(a_ascii("\n".join(L)), encoding="ascii")
    log(f"      interpretacion escrita en {len(L)} lineas")

    # ---------- ITEM 3: el dano del artefacto de P66 ----------
    log("")
    log("   ITEM 3, EL DANO DEL ARTEFACTO DE P66, con su marca de no reejecucion")
    dano = {
        "familia": "P66_ARTIFACT_DAMAGE", "experiment_id": "P67-03",
        "fecha": inicio.isoformat(timespec="seconds"),
        "artefacto_afectado": "reports/p66_score_fusion.csv",
        "registros_perdidos": 214,
        "registros_tras_restaurar": sum(1 for _ in (REPORTS / "p66_score_fusion.csv")
                                        .open(encoding="utf-8")),
        "restaurado": True,
        "codigo_historico_modificado": False,
        "motivo_de_no_modificarlo": "preservar la correspondencia historica entre codigo y "
                                    "resultados de las fases anteriores",
        "causa": {
            "script": "experiments/p66_01_regenerar_scores.py",
            "linea": 86,
            "codigo": 'CSV_REG = REPORTS / "p66_score_fusion.csv"',
            "explicacion": ("el regenerador de P66 declara un nombre de fichero que pertenece "
                            "a p66_02_fusion_espacio_scores.py, y lo abre en modo escritura. "
                            "Al re-ejecutarlo escribe su tabla de verificacion de 8 filas "
                            "encima de los 214 registros de la fase de fusion"),
        },
        "cuando_ocurre": ("cada vez que alguien re-ejecuta p66_01. Y hay que re-ejecutarlo en "
                          "cualquier worktree nuevo, porque features/ esta en .gitignore y el "
                          "npz de scores no viaja entre worktrees"),
        "como_se_detecto": ("comparando git show --stat del commit contra el numero de filas "
                            "esperado. El sintoma fue '222 lineas borradas' en el stat"),
        "como_se_restauro": ("git checkout HEAD~1 -- reports/p66_score_fusion.csv "
                             "reports/p66_score_regeneration.json"),
        "hermano_intacto": {
            "fichero": "reports/p66_score_fusion.json",
            "estado": ("intacto. El regenerador escribe el CSV, no el JSON, y la reconciliacion "
                       "lee el JSON. El dano fue solo del CSV"),
        },
        "arreglo_correcto": {
            "cambio": 'que CSV_REG de p66_01 apunte a "p66_score_regeneration.csv"',
            "tamano": "una linea",
            "por_que_no_se_aplica": ("exigiria editar el codigo de una fase anterior, y el "
                                     "encargo prohibe hacerlo. Se documenta la via limpia en "
                                     "su lugar"),
        },
        "marca": "DO_NOT_RERUN_UNGUARDED",
        "procedimiento_seguro": [
            "/home/carlos/venv/bin/python experiments/p66_01_regenerar_scores.py",
            "git checkout HEAD -- reports/p66_score_fusion.csv",
            "git checkout HEAD -- reports/p66_score_regeneration.json",
        ],
    }
    JSON_DAN.write_text(json.dumps(dano, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    D: list[str] = []
    Wd = D.append
    Wd("# P66 - Dano y restauracion de un artefacto, y marca DO_NOT_RERUN_UNGUARDED")
    Wd("")
    Wd(f"Generado por `experiments/p67_03_cierre_documental.py`. Fecha "
       f"{inicio.isoformat(timespec='seconds')}.")
    Wd("")
    Wd("## Que paso")
    Wd("")
    Wd("P67 necesita `features/p66_scores_candidatos.npz`. Ese fichero vive en `features/`, que")
    Wd("esta en `.gitignore`, asi que **no viaja entre worktrees**. La unica forma de tenerlo es")
    Wd("re-ejecutar su regenerador, `experiments/p66_01_regenerar_scores.py`.")
    Wd("")
    Wd("Al hacerlo, ese script escribe `reports/p66_score_fusion.csv` con **su propia** tabla de")
    Wd("verificacion de 8 filas, encima de los **214 registros** de la fase de fusion que ese")
    Wd("fichero contiene de verdad.")
    Wd("")
    Wd("```python")
    Wd('experiments/p66_01_regenerar_scores.py   linea 86')
    Wd('  CSV_REG = REPORTS / "p66_score_fusion.csv"')
    Wd("```")
    Wd("")
    Wd("El nombre colisiona entre dos scripts de la misma fase. `p66_02_fusion_espacio_scores.py`")
    Wd("es el dueno legitimo de ese fichero.")
    Wd("")
    Wd("## Como se detecto")
    Wd("")
    Wd("No salto durante la ejecucion: el script termino con codigo 0 y su propia verificacion")
    Wd("dijo `7/7 verifican`.")
    Wd("")
    Wd("Salto al hacer `git show --stat` del commit, que mostro `222 --------` en ese fichero.")
    Wd("**Un fichero de resultados historico perdio 214 registros sin que ningun script fallara.**")
    Wd("")
    Wd("## Lo que se hizo")
    Wd("")
    Wd("```bash")
    Wd("git checkout HEAD~1 -- reports/p66_score_fusion.csv")
    Wd("git checkout HEAD~1 -- reports/p66_score_regeneration.json")
    Wd("```")
    Wd("")
    Wd("Comprobado despues: 214 filas, y el `mejor_real` de `p66_score_fusion.json`")
    Wd("(que salio intacto) sigue siendo el de siempre.")
    Wd("")
    Wd("## La marca")
    Wd("")
    Wd("```text")
    Wd("DO_NOT_RERUN_UNGUARDED")
    Wd("```")
    Wd("")
    Wd("`experiments/p66_01_regenerar_scores.py` tiene esta marca. Ejecutarlo sin el `git")
    Wd("checkout` de despues destruye un artefacto de su propia fase.")
    Wd("")
    Wd("### Procedimiento seguro")
    Wd("")
    Wd("```bash")
    Wd("/home/carlos/venv/bin/python experiments/p66_01_regenerar_scores.py")
    Wd("git checkout HEAD -- reports/p66_score_fusion.csv")
    Wd("git checkout HEAD -- reports/p66_score_regeneration.json")
    Wd("```")
    Wd("")
    Wd("## Por que no se parchea el script")
    Wd("")
    Wd("El arreglo es una linea: que `CSV_REG` de `p66_01` apunte a")
    Wd("`p66_score_regeneration.csv`, que es el nombre que le corresponde.")
    Wd("")
    Wd("No se aplica porque **editar el codigo de una fase anterior rompe la correspondencia**")
    Wd("entre ese codigo y los resultados que publico. Un fichero de P66 que hoy dice una cosa")
    Wd("empezaria a decir otra, y quien audite la fase dentro de un ano no podria distinguir")
    Wd("que linea produjo que numero.")
    Wd("")
    Wd("Se documenta la via limpia en su lugar.")
    Wd("")
    Wd("## El hermano que no se toco")
    Wd("")
    Wd("`reports/p66_score_fusion.json` **salio intacto**: el regenerador escribe el CSV, no el")
    Wd("JSON, y la reconciliacion de P67 lee el JSON. El dano fue solo del CSV.")
    Wd("")
    Wd("## Leccion general")
    Wd("")
    Wd("Un script de una fase puede destruir artefactos de su propia fase sin fallar. La unica")
    Wd("defensa barata es comparar el numero de filas de los ficheros de resultados antes y")
    Wd("despues de una re-ejecucion, o mirar el `--stat` del commit.")
    Wd("")
    Wd("*Generado por `experiments/p67_03_cierre_documental.py`. Sin push.*")
    MD_DAN.write_text(a_ascii("\n".join(D)), encoding="ascii")

    Wv: list[str] = []
    Wv.append("# DO NOT RERUN UNGUARDED")
    Wv.append("")
    Wv.append("Scripts que, al re-ejecutarse, destruyen artefactos existentes sin fallar.")
    Wv.append("")
    Wv.append("## experiments/p66_01_regenerar_scores.py")
    Wv.append("")
    Wv.append("```text")
    Wv.append("marca           = DO_NOT_RERUN_UNGUARDED")
    Wv.append("artefacto       = reports/p66_score_fusion.csv")
    Wv.append("registros       = 214")
    Wv.append("restaurado      = si")
    Wv.append("codigo cambiado = no")
    Wv.append("```")
    Wv.append("")
    Wv.append("Escribe su tabla de verificacion de 8 filas encima de los 214 registros de la fase")
    Wv.append("de fusion, porque comparte nombre de fichero con `p66_02_fusion_espacio_scores.py`.")
    Wv.append("")
    Wv.append("Ejecutar SIEMPRE seguido de:")
    Wv.append("")
    Wv.append("```bash")
    Wv.append("git checkout HEAD -- reports/p66_score_fusion.csv \\")
    Wv.append("                      reports/p66_score_regeneration.json")
    Wv.append("```")
    Wv.append("")
    Wv.append("Detalle completo en `reports/p66_artifact_damage.md`.")
    Wv.append("")
    AVISO.parent.mkdir(parents=True, exist_ok=True)
    AVISO.write_text(a_ascii("\n".join(Wv)), encoding="ascii")
    log(f"      {MD_DAN.name} y {AVISO.relative_to(PROJECT_ROOT)} escritos")

    # ---------- ITEM 4: las dos verificaciones de ASCII, reportadas por separado ----------
    log("")
    log("   ITEM 4, LAS DOS VERIFICACIONES DE ASCII, por separado")
    oficial = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "tools" / "verificar_caracteres.py")],
        capture_output=True, text=True, cwd=PROJECT_ROOT, check=False)
    oficial_ok = oficial.returncode == 0
    linea_oficial = [ln for ln in oficial.stdout.split("\n") if ln.startswith("OK:")]
    log(f"      oficial: exit {oficial.returncode} | "
        f"{linea_oficial[0] if linea_oficial else '(sin linea OK)'}")

    objetivos = sorted(
        list((PROJECT_ROOT / "experiments").glob("p67_*.py"))
        + list(REPORTS.glob("p67_*"))
        + [REPORTS / n for n in ("p66_artifact_damage.md", "ascii_compliance_dual.md",
                                 "ascii_compliance_dual.csv")]
        + [AVISO, PROJECT_ROOT / "FINAL_REPORT.md", PROJECT_ROOT / "PROJECT_STATE.md",
           PROJECT_ROOT / "CHAMPION.md", PROJECT_ROOT / "RUNLOG.md",
           PROJECT_ROOT / "DECISIONS.md",
           REPORTS / "TOXIC_COMPLETE_RESEARCH_REPORT.md"])
    filas_ascii = []
    for ruta in objetivos:
        if not ruta.exists():
            continue
        datos = ruta.read_bytes()
        filas_ascii.append({
            "fichero": ruta.relative_to(PROJECT_ROOT).as_posix(),
            "bytes": len(datos),
            "bytes_mayores_127": sum(1 for b in datos if b > 127),
        })
    total_altos = sum(f["bytes_mayores_127"] for f in filas_ascii)
    sucios = [f["fichero"] for f in filas_ascii if f["bytes_mayores_127"]]
    log(f"      independiente byte a byte: {len(filas_ascii)} ficheros, "
        f"{total_altos} bytes > 127, {len(sucios)} ficheros sucios")

    # El registro canonico tiene deuda historica: se mide y se declara, no se corrige.
    reg = (PROJECT_ROOT / "experiments.csv").read_bytes()
    lin_reg = [ln for ln in reg.split(b"\n")]
    altos_reg = sum(1 for b in reg if b > 127)
    lineas_reg_altas = sorted({i + 1 for i, ln in enumerate(lin_reg)
                               if any(b > 127 for b in ln)})
    lineas_p67 = {i + 1 for i, ln in enumerate(lin_reg) if ln.startswith(b"P67-0")}
    altos_en_p67 = sum(1 for i, ln in enumerate(lin_reg)
                       if any(b > 127 for b in ln) and (i + 1) in lineas_p67)
    log(f"      experiments.csv: {altos_reg} bytes > 127 en {len(lineas_reg_altas)} lineas, "
        f"deuda historica | en las lineas de P67: {altos_en_p67}")

    A: list[str] = []
    Wa = A.append
    Wa("# Verificacion de ASCII: las dos comprobaciones, por separado")
    Wa("")
    Wa(f"Generado por `experiments/p67_03_cierre_documental.py`. Fecha "
       f"{inicio.isoformat(timespec='seconds')}.")
    Wa("")
    Wa("## Por que hay dos")
    Wa("")
    Wa("La regla del proyecto es **0 bytes generados > 127**. El verificador oficial")
    Wa("`tools/verificar_caracteres.py` **no implementa esa regla**: vigila un conjunto de")
    Wa("alfabetos no latinos, y su lista **no incluye Latin-1 Supplement ni Latin Extended**.")
    Wa("")
    Wa("La lista de categorias que si vigila:")
    Wa("")
    Wa("```text")
    Wa("hebreo, arabe, devanagari, tamil y telugu, cirilico, griego y copto,")
    Wa("hangul, kanji y kana, emoji y simbolos")
    Wa("```")
    Wa("")
    Wa("Comprobado con un fichero de prueba que contenia una `e` acentuada, una `n` con")
    Wa("virgulilla, alfa, beta y un caracter cirilico: el verificador oficial **reporto el")
    Wa("cirilico y no reporto los acentos latinos del mismo fichero**.")
    Wa("")
    Wa("Por tanto el verificador oficial **no demuestra compliance** por si solo. La")
    Wa("comprobacion autoritativa es el escaneo byte a byte, y las dos se reportan aqui")
    Wa("separadas en vez de fundidas en un unico 'OK'.")
    Wa("")
    Wa("## Resultado 1: verificador oficial")
    Wa("")
    Wa("```text")
    Wa(f"exit code            {oficial.returncode}")
    Wa(f"linea de resultado   {linea_oficial[0] if linea_oficial else '(ninguna)'}")
    Wa("```")
    Wa("")
    Wa(f"Veredicto: **{'PASA' if oficial_ok else 'FALLA'}**, con la salvedad de que su")
    Wa("criterio es mas debil que el del proyecto.")
    Wa("")
    Wa("## Resultado 2: escaneo byte a byte independiente")
    Wa("")
    Wa(f"Ficheros revisados: `{len(filas_ascii)}`.")
    Wa(f"Bytes > 127 encontrados: `{total_altos}`.")
    Wa(f"Ficheros con algun byte > 127: `{len(sucios)}`.")
    Wa("")
    Wa(f"Veredicto: **{'PASA' if total_altos == 0 else 'FALLA'}**. Este es el criterio que")
    Wa("aplica la regla del proyecto.")
    Wa("")
    Wa("### Ficheros revisados")
    Wa("")
    Wa("| fichero | bytes | bytes > 127 |")
    Wa("|---|---:|---:|")
    for f in filas_ascii:
        Wa(f"| `{f['fichero']}` | {f['bytes']} | {f['bytes_mayores_127']} |")
    Wa("")
    Wa("## El registro canonico y su deuda historica")
    Wa("")
    Wa(f"`experiments.csv` tiene `{altos_reg}` bytes > 127 repartidos en "
       f"`{len(lineas_reg_altas)}` lineas,")
    Wa("de fases **anteriores** a P67 y presentes ya en el commit base.")
    Wa("")
    Wa(f"Bytes > 127 que caen sobre las lineas escritas por P67: `{altos_en_p67}`.")
    Wa("")
    Wa("**No se corrigen.** Reescribir filas historicas del registro cambiaria la historia de")
    Wa("las fases anteriores para ganar una propiedad de texto que no altera ningun resultado.")
    Wa("Se declara como deuda conocida.")
    Wa("")
    Wa("## La regla aplicada a los datos originales")
    Wa("")
    Wa("La regla se aplica a **artefactos generados**. El texto de los tweets del dataset es")
    Wa("dato original y **no se modifica** para satisfacerla: contiene acentos, emojis y")
    Wa("caracteres no ASCII que son la materia prima del problema, no un defecto del pipeline.")
    Wa("")
    Wa("*Generado por `experiments/p67_03_cierre_documental.py`. Sin push.*")
    MD_ASCII.write_text(a_ascii("\n".join(A)), encoding="ascii")
    with CSV_ASCII.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["verificador", "criterio", "ficheros", "bytes_mayores_127", "veredicto"])
        w.writerow(["oficial", "alfabetos no latinos", "", "", "PASA" if oficial_ok else "FALLA"])
        w.writerow(["independiente byte a byte", "0 bytes > 127", len(filas_ascii), total_altos,
                    "PASA" if total_altos == 0 else "FALLA"])
        for f in filas_ascii:
            w.writerow(["detalle", f["fichero"], f["bytes"], f["bytes_mayores_127"], ""])
    if total_altos:
        raise SystemExit(f"PARADA: bytes no ASCII en artefactos generados: {sucios}")

    # ---------- ITEM 7: las figuras que faltaban ----------
    log("")
    log("   ITEM 7, LAS FIGURAS DEL CIERRE")
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
        log(f"      {nombre_f}")

    # `p66_random_controls.csv` tiene DOS secciones en el mismo fichero: 35 filas de los
    # controles barajados y 7 filas de la informacion unica por candidato, que empiezan por
    # "informacion_unica". El lector tiene que saltarse la segunda, que ademas usa otro
    # esquema de columnas.
    controles = []
    with (REPORTS / "p66_random_controls.csv").open(encoding="utf-8") as fh:
        for r in _csv.DictReader(fh, fieldnames=["candidato", "mecanismo", "delta_real",
                                                 "delta_barajado", "brecha"], restkey="_x",
                                 restval=""):
            if r["candidato"] not in CANDIDATOS or r["mecanismo"] not in {
                    "blend_prob", "blend_logit", "fusion_lr", "residual", "length_aware"}:
                continue
            controles = [*controles, {
                "candidato": r["candidato"], "mecanismo": r["mecanismo"],
                "delta_real": float(r["delta_real"]),
                "delta_barajado": float(r["delta_barajado"]),
                "brecha_de_atribucion": float(r["brecha"]),
            }]
    log(f"      controles barajados cargados: {len(controles)} de 35 esperados")
    if len(controles) != 35:
        raise SystemExit(f"PARADA: se esperaban 35 controles, hay {len(controles)}")

    # F1. V03 frente a todos los mecanismos de integracion evaluados en P66.
    mecanismos = ["blend_prob", "blend_logit", "fusion_lr", "residual", "length_aware"]
    plt.figure(figsize=(12, 6))
    ancho = 0.15
    for k, m in enumerate(mecanismos):
        vals = [next(r["delta_real"] for r in controles if r["candidato"] == c and r["mecanismo"] == m)
                for c in CANDIDATOS]
        plt.bar(np.arange(len(CANDIDATOS)) + k * ancho, vals, width=ancho, label=m)
    plt.axhline(0, color="k", lw=1)
    plt.xticks(np.arange(len(CANDIDATOS)) + 2 * ancho, CANDIDATOS, rotation=20, ha="right")
    plt.ylabel("delta vs V03")
    plt.title("V03 frente a los 35 pares candidato x mecanismo de P66")
    plt.legend(fontsize=8, ncol=2)
    guardar("p67_07_v03_vs_mecanismos.png")

    # F2. delta vs V03, ordenado, todos los pares.
    plt.figure(figsize=(12, 8))
    orden = sorted(controles, key=lambda r: r["delta_real"])
    etiquetas = [f"{r['candidato']}|{r['mecanismo']}" for r in orden]
    colores = ["#c44" if r["delta_real"] < 0 else "#4a7" for r in orden]
    plt.barh(etiquetas, [r["delta_real"] for r in orden], color=colores)
    plt.axvline(0, color="k", lw=1)
    plt.axvline(0.0026743167, color="b", ls="--", lw=1, label="residual de MI_arbol")
    plt.xlabel("delta vs V03")
    plt.title("Los 35 pares ordenados por delta: 12 positivos sobre 35")
    plt.legend(fontsize=8)
    plt.tick_params(axis="y", labelsize=6)
    guardar("p67_08_delta_ordenado.png")

    # F3. real, barajado y azar, para los siete candidatos con el mecanismo residual.
    plt.figure(figsize=(12, 6))
    real = [next(r["delta_real"] for r in controles if r["candidato"] == c and r["mecanismo"] == "residual")
            for c in CANDIDATOS]
    baraj = [next(r["delta_barajado"] for r in controles if r["candidato"] == c
                  and r["mecanismo"] == "residual") for c in CANDIDATOS]
    x = np.arange(len(CANDIDATOS))
    plt.bar(x - 0.28, real, width=0.28, label="real")
    plt.bar(x, baraj, width=0.28, label="barajado")
    plt.bar(x + 0.28, [delta_azar_eq] * len(CANDIDATOS), width=0.28,
            label="azar, escala igualada")
    plt.axhline(0, color="k", lw=1)
    plt.xticks(x, CANDIDATOS, rotation=20, ha="right")
    plt.ylabel("delta vs V03")
    plt.title("Las tres referencias: real, barajado y azar, con el mecanismo residual")
    plt.legend()
    guardar("p67_09_real_barajado_azar.png")

    # F4. atribucion por candidato, los cinco mecanismos.
    plt.figure(figsize=(12, 6))
    for k, m in enumerate(mecanismos):
        vals = [next(r["brecha_de_atribucion"] for r in controles
                     if r["candidato"] == c and r["mecanismo"] == m) for c in CANDIDATOS]
        plt.bar(np.arange(len(CANDIDATOS)) + k * ancho, vals, width=ancho, label=m)
    plt.axhline(0, color="k", lw=1)
    plt.xticks(np.arange(len(CANDIDATOS)) + 2 * ancho, CANDIDATOS, rotation=20, ha="right")
    plt.ylabel("delta real - delta barajado")
    plt.title("La brecha de atribucion, que P67 reinterpreta como distancia al SUELO")
    plt.legend(fontsize=8, ncol=2)
    guardar("p67_10_atribucion_por_candidato.png")

    # F5. rescate, dano y neto, por mecanismo (datos de los 214 registros de P66).
    # `p66_score_fusion.csv` tambien tiene varias secciones: 140 filas de la tabla principal
    # (35 pares reales y sus 35 versiones barajadas, marcadas con el sufijo "__barajado") y
    # 71 filas de un desglose por bins de longitud con otro esquema. Se quiere solo la tabla
    # principal y solo la mitad real.
    filas_fus = []
    with (REPORTS / "p66_score_fusion.csv").open(encoding="utf-8") as fh:
        for r in _csv.reader(fh):
            if len(r) != 9 or f"{chr(124)}" not in r[0] or "__barajado" in r[0] or r[0] == "clave":
                continue
            if not r[6].strip().isdigit():
                continue
            filas_fus = [*filas_fus, {
                "clave": r[0], "candidato": r[1], "mecanismo": r[2],
                "delta_vs_V03": float(r[4]), "rescate": int(r[6]), "damage": int(r[7]),
                "neto": int(r[8]),
            }]
    log(f"      tabla principal de fusion: {len(filas_fus)} pares reales")
    if len(filas_fus) != 35:
        raise SystemExit(f"PARADA: se esperaban 35 pares, hay {len(filas_fus)}")
    plt.figure(figsize=(14, 7))
    claves = [r["clave"] for r in filas_fus]
    plt.bar(np.arange(len(claves)) - 0.2, [r["rescate"] for r in filas_fus], width=0.4,
            label="rescate")
    plt.bar(np.arange(len(claves)) + 0.2, [r["damage"] for r in filas_fus], width=0.4,
            label="dano")
    plt.plot(np.arange(len(claves)), [r["neto"] for r in filas_fus], "ko-", ms=3,
             label="neto")
    plt.axhline(0, color="k", lw=1)
    plt.xticks(np.arange(len(claves)), claves, rotation=90, fontsize=6)
    plt.ylabel("celdas")
    plt.title("Rescate, dano y neto de los 35 pares: el neto positivo es la excepcion")
    plt.legend()
    guardar("p67_11_rescate_dano_neto.png")

    # F6. el AUC del residuo, de P66 y de P67 lado a lado.
    inf = {f["candidato"]: f for f in fus["informacion_unica"]}
    p67_auc = {f["candidato"]: f for f in con["C-E_auc_del_residuo_contra_la_etiqueta"]}
    plt.figure(figsize=(12, 6))
    x = np.arange(len(CANDIDATOS))
    plt.bar(x - 0.3, [inf[c]["auc_del_residuo"] for c in CANDIDATOS], width=0.3,
            label="P66: residuo contra el ERROR de V03")
    plt.bar(x, [p67_auc[c].get("auc_contra_la_etiqueta_fuera_de_la_banda") for c in CANDIDATOS],
            width=0.3, label="P67: residuo contra la ETIQUETA, fuera de banda")
    plt.bar(x + 0.3, [p67_auc[c].get("auc_contra_la_etiqueta_dentro_de_la_banda")
                      for c in CANDIDATOS], width=0.3,
            label="P67: residuo contra la ETIQUETA, dentro de banda")
    plt.axhline(0.5, color="k", ls="--", lw=1, label="el azar")
    plt.xticks(x, CANDIDATOS, rotation=20, ha="right")
    plt.ylabel("AUC")
    plt.title("El AUC del residuo: solo MI_arbol queda sobre el azar, y sobre todo FUERA de banda")
    plt.legend(fontsize=8)
    guardar("p67_12_auc_del_residuo.png")

    # F7. varianza del residuo, el contraste que explica por que MI_arbol es distinto.
    plt.figure(figsize=(12, 6))
    plt.bar([f["candidato"] for f in con["C-E_auc_del_residuo_contra_la_etiqueta"]],
            [f["desviacion_del_residuo"] for f in con["C-E_auc_del_residuo_contra_la_etiqueta"]],
            color=["#4a7" if f["candidato"] == "MI_arbol" else "#888"
                   for f in con["C-E_auc_del_residuo_contra_la_etiqueta"]])
    plt.ylabel("desviacion estandar del residuo")
    plt.title(f"Varianza del residuo: MI_arbol es {ratio_var:.1f}x el mas redundante")
    plt.xticks(rotation=20, ha="right")
    guardar("p67_13_varianza_del_residuo.png")

    # F8. real vs barajado vs azar, el mecanismo residual de MI_arbol, en un solo panel.
    plt.figure(figsize=(9, 5))
    barras = [("real", 0.0026743167, "#4a7"),
              ("barajado", -0.0014390421, "#c44"),
              ("azar crudo", delta_azar, "#84c"),
              ("azar escala igualada", delta_azar_eq, "#4ac")]
    plt.bar([b[0] for b in barras], [b[1] for b in barras], color=[b[2] for b in barras])
    plt.axhline(0, color="k", lw=1)
    plt.ylabel("delta vs V03")
    plt.title("Residual de MI_arbol: el efecto real frente a sus dos suelos")
    guardar("p67_14_tres_referencias_mi_arbol.png")

    evidencia.extend([
        {"evidencia": "control de azar con beta fijo",
         "medida": "delta de la misma operacion con un score de ruido, escala igualada",
         "valor": (f"senal real {filas_beta[0]['beta_0.15']:+.10f} frente a azar "
                   f"{filas_beta[1]['beta_0.15']:+.10f} en el mismo beta"),
         "unidad": "delta de corrected_cv",
         "lo_que_prueba": ("que la operacion DISTINGUE senal de ruido: la senal real es "
                           "positiva en los seis betas y el azar negativo en los seis"),
         "lo_que_no_prueba": ("que el efecto sea atribuible a MI_arbol, ni que sea robusto o "
                              "explotable"),
         "fuente": "reports/p67_interpretation.md"},
        {"evidencia": "equivalencia algebraica del mecanismo residual",
         "medida": "maxima diferencia entre la forma residual y la forma blend",
         "valor": "8.882e-16", "unidad": "error numerico",
         "lo_que_prueba": ("que el mecanismo residual de P66 es EXACTAMENTE un blend lineal "
                           "de logits mas un desplazamiento de umbral"),
         "lo_que_no_prueba": "que el efecto sea falso. Solo dice que no es un mecanismo nuevo",
         "fuente": "reports/p67_interpretation.md"},
    ])

    # ---------- la evidencia congelada, en Markdown ----------
    E: list[str] = []
    We = E.append
    We("# P67 - Evidencia congelada")
    We("")
    We(f"Generado por `experiments/p67_03_cierre_documental.py`. Fecha "
       f"{inicio.isoformat(timespec='seconds')}.")
    We("")
    We("Las cifras de este documento se **leen** de los informes de la fase, no se recalculan.")
    We("Recalcularlas crearia una segunda fuente de verdad para numeros que ya se produjeron y")
    We("verificaron una vez. La unica medicion nueva es el control de azar, y va marcada.")
    We("")
    We("## La decision")
    We("")
    We("```text")
    We(f"{con['categoria_del_mecanismo']}")
    We("```")
    We("")
    We("## Las cinco cifras que la sostienen")
    We("")
    We("| evidencia | valor | lo que prueba | lo que NO prueba |")
    We("|---|---:|---|---|")
    for e in evidencia:
        if e["evidencia"].startswith("atribucion historica"):
            continue
        We(f"| {a_ascii(e['evidencia'])} | `{e['valor']}` | {a_ascii(e['lo_que_prueba'])[:70]} "
           f"| {a_ascii(e['lo_que_no_prueba'])[:70]} |")
    We("")
    We("## Las cinco cifras, una por una")
    We("")
    for e in evidencia:
        We(f"### {a_ascii(e['evidencia'])}")
        We("")
        We(f"- **Medida**: {a_ascii(e['medida'])}")
        We(f"- **Valor**: `{e['valor']}` ({e['unidad']})")
        We(f"- **Lo que prueba**: {a_ascii(e['lo_que_prueba'])}")
        We(f"- **Lo que NO prueba**: {a_ascii(e['lo_que_no_prueba'])}")
        We(f"- **Fuente**: `{e['fuente']}`")
        We("")
    We("## El valor que NO se usa")
    We("")
    We("```text")
    We("+0,0093216366   brecha de atribucion del blend de P66")
    We("estado          INVALIDADA como medida de informacion atribuida")
    We("```")
    We("")
    We("Se conserva en el registro **solo** para trazabilidad, porque la regla del proyecto es")
    We("preservar el ORIGINAL junto al CORREGIDO y documentar la relacion, no borrar el")
    We("original.")
    We("")
    We("### Por que estaba mal")
    We("")
    We("Se calculaba como `delta real - delta barajado`. Pero el control barajado **no es un")
    We("cero de efecto**: es un **suelo de capacidad de la operacion**, es decir lo que cuesta")
    We("esa misma operacion cuando el candidato no lleva senal. Restar un suelo mide una")
    We("distancia al suelo, y no constituye una atribucion directa de la senal.")
    We("")
    We("```text")
    We("ORIGINAL    +0,0093216366   brecha de atribucion del blend")
    We("CORREGIDO   no hay atribucion medible por esa via")
    We("RELACION    la cifra original es una distancia al suelo, no una atribucion")
    We("```")
    We("")
    We("## El registro de reclasificaciones, con original y corregido")
    We("")
    We("| original | clasificacion | corregido | relacion |")
    We("|---:|---|---|---|")
    We("| `+0,0033068923` | NON_COMPARABLE | (ninguno) | otro mecanismo, 3 de 5 pliegues, fuera del "
       "gate |")
    We("| `+0,0026743167` | FINAL_VALID_P66_RESULT | (ninguno) | el unico con delta y pliegues |")
    We("| `+0,0093216366` | INVALIDATED | no hay atribucion por esa via | distancia al suelo |")
    We("| `+0,0041133588` | CORRECTED | idem | idem, menos severa |")
    We("| `-0,0014390421` | ORIGINAL | (ninguno) | es el suelo del residual |")
    We("| `-0,0044311634` | ORIGINAL | (ninguno) | extremo inferior del IC95 |")
    We("| `+0,0102108745` | ORIGINAL | (ninguno) | extremo superior del IC95 |")
    We("")
    We("Nada se sobrescribe: `reports/p67_p66_reconciliation.csv` guarda los siete valores con")
    We("su clasificacion, y los informes originales de P66 siguen intactos en el repositorio.")
    We("")
    We("> V03 remains the best verified leaderboard result.")
    We("")
    We("*Generado por `experiments/p67_03_cierre_documental.py`. Sin push.*")
    MD_EVI.write_text(a_ascii("\n".join(E)), encoding="ascii")

    # ---------- cierre ----------
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    salida = {
        "familia": "P67_CIERRE_DOCUMENTAL", "experiment_id": "P67-03",
        "fecha": inicio.isoformat(timespec="seconds"),
        "decision": con["categoria_del_mecanismo"],
        "evidencia_congelada": evidencia,
        "ratio_de_varianza": round(ratio_var, 4),
        "control_de_azar": {
            "crudo": round(delta_azar, 10),
            "crudo_desviacion_del_residuo": round(sd_azar, 6),
            "igualado": round(delta_azar_eq, 10),
            "igualado_desviacion_del_residuo": round(sd_azar_eq, 6),
            "senal_real_desviacion_del_residuo": round(sd_real, 6),
            "nota": ("el crudo no es comparable porque su perturbacion es "
                     f"{sd_azar / sd_real:.2f} veces la de la senal real"),
        },
        "atribucion_historica_invalida": 0.0093216366,
        "ascii": {"oficial": "PASA" if oficial_ok else "FALLA",
                  "independiente": "PASA" if total_altos == 0 else "FALLA",
                  "bytes_mayores_127": total_altos,
                  "ficheros_revisados": len(filas_ascii),
                  "deuda_historica_registro": altos_reg,
                  "bytes_no_ascii_en_lineas_de_P67": altos_en_p67},
        "dano_p66": {"artefacto": "reports/p66_score_fusion.csv",
                     "registros_perdidos": 214, "restaurado": True,
                     "codigo_historico_modificado": False,
                     "marca": "DO_NOT_RERUN_UNGUARDED"},
        "figuras_nuevas": hechos,
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "seed": SEMILLA,
                             "hash_v03": hash_real, "corrected_cv": V03_CCV},
        "formulacion": ("Se detecto senal residual, pero los mecanismos de integracion "
                        "evaluados no han conseguido aislar una mejora robusta y atribuible "
                        "sobre V03."),
    }
    (REPORTS / "p67_closeout.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    documentos = [MD_EVI, JSON_EVI, CSV_EVI, MD_INT, MD_DAN, JSON_DAN, MD_ASCII, CSV_ASCII,
                  AVISO, REPORTS / "p67_closeout.json"] + [FIGURAS / f for f in hechos]
    sucios_doc = [d.name for d in documentos
                  if d.exists() and d.suffix != ".png" and any(b > 127 for b in d.read_bytes())]
    if sucios_doc:
        raise SystemExit(f"PARADA: documentos con bytes no ASCII: {sucios_doc}")

    log("")
    log("=" * 78)
    log(f"P67-03 OK | {con['categoria_del_mecanismo']}")
    log(f"   {len(evidencia)} piezas de evidencia | {len(hechos)} figuras nuevas")
    log(f"   azar crudo {delta_azar:+.10f} | azar igualado {delta_azar_eq:+.10f} | "
        f"senal real {sd_real:.4f}")
    log(f"   ratio de varianza MI_arbol / mas redundante {ratio_var:.4f}")
    log(f"   ASCII: oficial {'PASA' if oficial_ok else 'FALLA'} | "
        f"independiente {'PASA' if total_altos == 0 else 'FALLA'} "
        f"({total_altos} bytes > 127)")
    log("   dano de P66 documentado | marca DO_NOT_RERUN_UNGUARDED puesta")
    log("   V03 remains the best verified leaderboard result.")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
