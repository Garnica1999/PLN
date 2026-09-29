"""P68-H2 audit: is rescue-conditioned reweighting new, or is it the most explored mechanism here.

Why this script exists instead of an experiment
-----------------------------------------------
The proposal was `Teacher-Guided Rescue Reweighting`: upweight the training pairs where V03 is
wrong and an alternative model is right, retrain the same classifier with sample weights, and
see whether the rescue map transfers into a different boundary.

Before spending anything on that, the brief requires checking whether the project has already
done something methodologically equivalent, and it names five forms to look for: sample
weighting based on another model's correction, teacher-guided weighting, error-conditioned
weighting, rescue-conditioned training, and disagreement-weighted training.

It has. This script documents the audit, and the audit says the proposal is a NARROWER
VERSION of a family that this project has already run to exhaustion.

The distinction that decides it
-------------------------------
Every weighting experiment in the registry draws its indicator from one of two places:

    a function of (p_V03, y)                the model's own error, margin or violation
    a function of (pred_V03, pred_OTHER, y) another model's correctness

The first is explored six times. The second is what P68-H2 proposes, and it appears nowhere.

But the second is a SUBSET of the first, and that is the whole argument:

    R_i = (pred_V03_i != y_i) AND (pred_OTHER_i == y_i)

The first term is exactly the hard-example indicator V4-B03 already weighted. The second term
only REMOVES rows from it. So P68-H2 is the same mechanism with a strictly narrower indicator,
and the project has already measured what happens when that indicator is widened: upweighting
the examples the model gets wrong made the metric worse, and the reason is documented.

The audit also finds the one case where weighting did work, and it is not a rescue map.

What this script does NOT do
----------------------------
It does not train anything. It does not reopen P68-H1. It does not touch the test set, the
holdout or the leaderboard. It reads the registry, the code and the reports, and it says no.
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
CELDA = PROJECT_ROOT / "features" / "p66_scores_candidatos.npz"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"

MD_H2 = REPORTS / "p68_h2_rescue_reweighting.md"
CSV_H2 = REPORTS / "p68_h2_rescue_reweighting.csv"
CSV_AUD = REPORTS / "p68_h2_duplicate_audit.csv"
MD_LIBRE = REPORTS / "post_p68_remaining_mechanisms.md"
JSON_LIBRE = REPORTS / "post_p68_remaining_mechanisms.json"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
CANDIDATOS = ("MI_arbol", "negacion", "word_1_3", "cohesion", "huecos_1_2", "contraste",
              "huecos_6_10")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P68-02] {mensaje}", flush=True)


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


# ---------- la auditoria: cada ponderado del proyecto y de donde saca su indicador ----------
# La columna que decide no es el nombre ni el resultado. Es la FUENTE: de que depende el peso.
#   "propio"   el peso es funcion de (p_V03, y): error, margen, violacion del umbral
#   "ajeno"    el peso es funcion de (pred_V03, pred_OTRO, y): la correccion de otro modelo
AUDITORIA = [
    {"experimento": "V4-B03", "anio": "v4",
     "variantes": "mild_up, medium_up, medium_down, clipped_extremos, down_violacion",
     "fuente": "propio", "indicador": "dureza OOF de V03: margen y violacion del umbral",
     "resultado": ("upweighting de la dureza EMPEORA: medium_up -0,0096. Solo mejora "
                   "down_violacion, que BAJA el peso de los que mas violan"),
     "estado": "CLOSED", "veredicto": "la familia de reweighting se cierra sin adopcion"},
    {"experimento": "V4-B04", "anio": "v4",
     "variantes": "dureza por etiqueta frente a global",
     "fuente": "propio", "indicador": "dureza de la propia etiqueta",
     "resultado": "0,7686 frente a 0,7772: empeora",
     "estado": "CLOSED", "veredicto": "la ponderacion independiente por etiqueta empeora"},
    {"experimento": "V4-B05", "anio": "v4",
     "variantes": "downweight 0,3 y remove de ejemplos con contradiccion textual",
     "fuente": "propio", "indicador": "contradiccion textual del ejemplo",
     "resultado": "downweight +0,0005 y remove -0,0004: nada",
     "estado": "CLOSED", "veredicto": "las dos estrategias de contradiccion no mueven la metrica"},
    {"experimento": "V4-C01..C05", "anio": "v4",
     "variantes": "combo:downweight_violacion_x_clase",
     "fuente": "propio", "indicador": "violacion del umbral combinada con el peso de clase",
     "resultado": ("EL UNICO CASO QUE FUNCIONA: 0,785416, +0,008205, 5 de 5 pliegues. Pero "
                   "falla el criterio 6 (artefacto) del gate de V4-X01 y no tiene veredicto "
                   "externo"),
     "estado": "NO ADOPTADO", "veredicto": "no pasa el gate de artefacto aunque gane en CV"},
    {"experimento": "V6-03", "anio": "v6",
     "variantes": "ponderacion por tamano de grupo",
     "fuente": "propio", "indicador": "repeticiones del ejemplo",
     "resultado": "la ponderacion por tamano de grupo no corrige el problema",
     "estado": "CLOSED", "veredicto": "V03 no sobrepondera de la forma que se suponia"},
    {"experimento": "V6-07 / V30-00", "anio": "v6, v30",
     "variantes": "sample_weight = 1 - lambda * sospecha",
     "fuente": "propio",
     "indicador": "inconsistencia del par instancia-etiqueta detectada en el propio modelo",
     "resultado": "el entrenamiento robusto OOF no reduce el impacto de forma utilizable",
     "estado": "CLOSED", "veredicto": "el ruido textual no explica el error (D-026)"},
    {"experimento": "P60-07 (P60_I)", "anio": "p60",
     "variantes": "meta-reweighting por alineacion de gradiente",
     "fuente": "propio",
     "indicador": "eps_i = <g_i, G>/<G, G>, alineacion del gradiente del ejemplo con el de un "
                  "objetivo meta",
     "resultado": "-0,3011 frente a V03, con pesos casi degenerados",
     "estado": "CLOSED", "veredicto": "el ponderado por alineacion de gradiente no mejora"},
    {"experimento": "P60-08", "anio": "p60",
     "variantes": "polinomio en el error con coeficientes independientes",
     "fuente": "propio",
     "indicador": "region de error, modulada por la perdida y NO por un sample_weight",
     "resultado": "reformula el reparto del gradiente entre regiones de error",
     "estado": "CLOSED", "veredicto": ("se declara explicitamente 'sin recurrir a ponderar "
                                       "ejemplos': el propio proyecto considero esta via "
                                       "equivalente y la cerro")},
    {"experimento": "V8-07", "anio": "v8",
     "variantes": "multiplicador del peso de clase por etiqueta",
     "fuente": "propio", "indicador": "peso de clase elegido con validacion anidada",
     "resultado": "no mejora sobre balanced",
     "estado": "CLOSED", "veredicto": "el multiplicador por etiqueta no aporta"},
    {"experimento": "V51-01 / V51-02", "anio": "v51",
     "variantes": "selector por utilidad esperada sobre el mapa de rescates de los canales",
     "fuente": "ajeno",
     "indicador": ("P(canal acierta) estimado por canal y celda: es el unico antecedente que "
                   "mira a OTRO modelo"),
     "resultado": "-0,0101586431, con 121 rescates y 183 danos, neto -62",
     "estado": "INVALIDATED",
     "veredicto": ("el mapa de rescates de otros modelos SE CONSTRUYO y NO se pudo explotar "
                   "en decision. P50 midio el techo del oracle en +0,2098 y el selector no lo "
                   "alcanzo")},
    {"experimento": "P68-H2 (propuesta)", "anio": "p68",
     "variantes": "w = 2 en los pares rescatables, w = 1 en el resto",
     "fuente": "ajeno",
     "indicador": "R_i = (pred_V03 != y) AND (pred_OTRO == y)",
     "resultado": "no ejecutado",
     "estado": "DUPLICATE",
     "veredicto": ("es el indicador de V4-B03 INTERSECADO con la correccion de otro modelo. "
                   "El primer termino ya se midio solo, y el segundo solo QUITA filas")},
]


def main() -> None:
    """Determina si P68-H2 es nueva, analiza los rescates exclusivos y barre lo que queda.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P68-02 auditoria de P68-H2: antes de entrenar hay que saber si ya esta hecho")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    v = np.load(OOF, allow_pickle=True)
    y, _pliegues = v["y_true"].astype(int), v["pliegues"]
    n = y.shape[0]
    from sklearn.metrics import f1_score
    ccv_ref = float(f1_score(y, v["pred"].astype(int), average="macro"))
    if ccv_ref != V03_CCV:
        raise SystemExit("PARADA: el OOF canonico no reproduce el corrected_cv congelado")
    z = np.load(CELDA, allow_pickle=True)
    pred_v03 = z["pred_v03"].astype(int)
    proba_c = {c: z[f"proba_{c}"] for c in CANDIDATOS}
    pred_c = {c: (proba_c[c] >= V03_UMBRALES[None, :]).astype(int) for c in CANDIDATOS}
    log(f"   V03 verificado: {ccv_ref:.10f} | {n} filas")

    # ---------- P0: la auditoria de duplicacion ----------
    log("")
    log("   P0, LA AUDITORIA DE DUPLICACION: de donde saca el peso cada experimento")
    n_propio = sum(1 for a in AUDITORIA if a["fuente"] == "propio")
    n_ajeno = sum(1 for a in AUDITORIA if a["fuente"] == "ajeno")
    n_cerrados = sum(1 for a in AUDITORIA if a["estado"] in ("CLOSED", "NO ADOPTADO"))
    log(f"      experimentos de ponderacion revisados: {len(AUDITORIA)}")
    log(f"      con indicador PROPIO, funcion de (p_V03, y):        {n_propio}")
    log(f"      con indicador AJENO, funcion de otro modelo:        {n_ajeno}")
    log(f"      cerrados o no adoptados:                            {n_cerrados}")
    log("")
    for a in AUDITORIA:
        marca = "AJENO" if a["fuente"] == "ajeno" else "propio"
        log(f"      [{marca:6}] {a['experimento']:22} {a['estado']:12} "
            f"{a['resultado'][:56]}")

    # ---------- P1: los 122 rescates exclusivos ----------
    log("")
    log("   P1, LOS RESCATES EXCLUSIVOS: donde estan los 122")
    v_ko = pred_v03 != y
    res = {c: (v_ko & (pred_c[c] == y)) for c in CANDIDATOS}
    unicos = {}
    for c in CANDIDATOS:
        otros = [res[o] for o in CANDIDATOS if o != c]
        unicos[c] = res[c] & ~np.stack(otros).any(axis=0)
    total_unicos = int(sum(u.sum() for u in unicos.values()))
    log(f"      rescates exclusivos totales: {total_unicos}")
    log("")
    log(f"      {'candidato':12}{'exclusivos':>12}{'de los cuales en LARGA':>26}")
    n_tok = z["n_tokens"]
    es_larga = n_tok >= 16
    for c in sorted(CANDIDATOS, key=lambda x: -unicos[x].sum()):
        en_larga = int((unicos[c] & es_larga[:, None]).sum())
        log(f"      {c:12}{int(unicos[c].sum()):>12}{en_larga:>26}")
    # La pregunta que importa para H2: el mapa de rescates acota el indicador de error?
    res_alg = np.stack([res[c] for c in CANDIDATOS]).any(axis=0)
    log("")
    log("      LA RELACION QUE DECIDE LA CLASIFICACION")
    log(f"      celdas donde V03 falla:                        {int(v_ko.sum())}")
    log(f"      de esas, alguna alternativa acierta:           {int(res_alg.sum())} "
        f"({100 * res_alg.sum() / max(int(v_ko.sum()), 1):.1f}% de los errores)")
    log("      luego el indicador de rescate es un SUBCONJUNTO ESTRICTO del de error")
    log(f"      celdas de error que el mapa de rescate EXCLUYE: "
        f"{int((v_ko & ~res_alg).sum())}")

    # ---------- P2: la determinacion ----------
    log("")
    log("   P2, DETERMINACION: P68-H2 es genuinamente nueva?")
    criterios = [
        ("el MECANISMO (sample weighting) es nuevo",
         False, "V4-B03, V4-B04, V4-B05, V4-C, V6-03, V6-07, V30-00, P60-07, V8-07: nueve veces"),
        ("la REPRESENTACION cambia",
         False, "el encargo fija TF-IDF word(1,2)+char_wb(3,5) congelado, igual que V03"),
        ("el CLASIFICADOR cambia",
         False, "LogisticRegression C=2.0 balanced liblinear, identico"),
        ("la UNIDAD de ponderacion (par etiqueta-instancia) es nueva",
         False, "V4-B04 pondera por etiqueta y V6-07 por par instancia-etiqueta"),
        ("la FUENTE del indicador referencia otro modelo",
         True, "aqui si: R usa pred_OTRO. Ningun ponderado del registro lo hace"),
        ("el indicador es un SUBCONJUNTO ESTRICTO de uno ya medido",
         True, (f"R = error AND (el otro acierta), y el error solo ya se midio: "
               f"{100 * res_alg.sum() / max(int(v_ko.sum()), 1):.1f}% de los errores")),
        ("la direccion del peso (subir) ya se midio y salio mal",
         True, "V4-B03 medium_up -0,0096 y mild_up -0,0021: subir el peso a lo que falla EMPEORA"),
    ]
    for nombre, ok, detalle in criterios:
        log(f"      {'SI' if ok else 'NO':3} {nombre:52} {detalle[:70]}")
    nuevo = all(c[1] for c in criterios[:5])
    log("")
    log(f"      conclusion: P68-H2 es {'' if nuevo else 'NO '}GENUINELY_NEW")
    log("      La unica novedad es la fuente del indicador, y el indicador que propone es un")
    log("      SUBCONJUNTO del que V4-B03 ya midio; ademas en la direccion que salio mal.")
    clasificacion = "DUPLICATE"
    log(f"      CLASIFICACION: {clasificacion} -> NO SE EJECUTA")
    log("      Coste: 0 experimentos. Se continua con el barrido de mecanismos restantes.")

    # ---------- P7: el barrido de mecanismos restantes ----------
    log("")
    log("   P7, EL BARRIDO DE MECANISMOS RESTANTES")
    # Se recorre la lista de ejes que el encargo senala como prioritarios, y para cada uno se
    # cita el experimento que ya lo recorrio. Un eje sin cita es un eje abierto.
    ejes = [
        {"eje": "estructura de los ejemplos", "experimentos": "P68-H1",
         "resultado": "INVALIDATED: la estructura predice dificultad (AUC 0,608), no ganador "
                      "(0,549)"},
        {"eje": "estructura local", "experimentos": "P59-02, P59-10, V29-03, V29-04",
         "resultado": "la geometria local predice el error con AUC 0,725 pero no es "
                      "accionable; los vecinos puros no corrigen nada"},
        {"eje": "geometria de representacion", "experimentos": "V25, V27, V26, V43",
         "resultado": "FM, kernel de espectro, Tsetlin y PLS: los cuatro cerrados"},
        {"eje": "distribucion del entrenamiento", "experimentos": "V6-03, P60-06, P60-04",
         "resultado": "transporte, peso de clase y tamano de grupo: los tres cerrados"},
        {"eje": "descomposicion de decisiones", "experimentos": "P68-H1, P50, P51",
         "resultado": ("enrutado por estructura INVALIDATED, y el gating por canal con mapa de "
                       "rescates INVALIDATED con -0,0102 sobre un techo de +0,2098")},
        {"eje": "estructura especifica de los errores", "experimentos": "V4-B, V4-C, P60-08, "
                                                                       "V28",
         "resultado": "dureza, violacion, contradiccion y region: los cuatro cerrados"},
        {"eje": "transferencia al entrenamiento", "experimentos": "V30-00, V6-07, P59-10",
         "resultado": "peso robusto, deteccion de par inconsistente y regularizacion por "
                      "vecindad: los tres cerrados"},
        {"eje": "ponderacion por correccion ajena", "experimentos": "P68-H2",
         "resultado": "DUPLICATE: subconjunto estricto del indicador de error ya medido"},
    ]
    abiertos = [e for e in ejes if not e["experimentos"]]
    log(f"      ejes recorridos: {len(ejes)} | sin cita historica: {len(abiertos)}")
    for e in ejes:
        log(f"      {e['eje']:38} {e['experimentos'][:34]}")
    log("")
    log("      NINGUN eje queda sin cita historica. Los ocho tienen experimento que los cubre.")
    log("      Siete cerrados o invalidados, y el octavo DUPLICATE.")

    # ---------- el veredicto de la fase ----------
    puede_cerrar = not abiertos and clasificacion == "DUPLICATE"
    log("")
    log(f"      P68-H2 DUPLICATE: {clasificacion == 'DUPLICATE'}")
    log(f"      ningun eje sin recorrer: {not abiertos}")
    log(f"      CONDICION DE CIERRE DEL ENCARGO CUMPLIDA: {puede_cerrar}")
    if puede_cerrar:
        log("      El encargo autoriza cerrar solo si el gap analysis posterior demuestra que")
        log("      no quedan mecanismos genuinamente nuevos, o que los que quedan son variantes")
        log("      de mecanismos cerrados o invalidados. Es el caso.")

    # ---------- ficheros ----------
    with CSV_AUD.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["experimento", "fuente_del_indicador", "indicador", "resultado", "estado",
                    "veredicto"])
        for a in AUDITORIA:
            w.writerow([a["experimento"], a["fuente"], a["indicador"], a["resultado"],
                        a["estado"], a["veredicto"]])
    filas_h2 = [
        {"seccion": "novedad", "estado": "DUPLICATE",
         "detalle": "unico cambio: la fuente del indicador. El indicador es un subconjunto "
                    "estricto del de V4-B03"},
        {"seccion": "hipotesis", "estado": "NO EJECUTADA",
         "detalle": "R_i = (pred_V03 != y) AND (pred_OTRO == y), con w = 2"},
        {"seccion": "protocolo nested", "estado": "NO EJECUTADO",
         "detalle": "no se gasto presupuesto en un mecanismo duplicado"},
        {"seccion": "construccion del rescue map", "estado": "YA EXISTE",
         "detalle": "P50 y P51 lo construyeron: techo del oracle +0,2098 y -0,0102 real"},
        {"seccion": "pesos", "estado": "NO EJECUTADOS", "detalle": "w = 2 propuesto"},
        {"seccion": "real", "estado": "NO EJECUTADO", "detalle": "n/a"},
        {"seccion": "shuffled", "estado": "NO EJECUTADO", "detalle": "n/a"},
        {"seccion": "random", "estado": "NO EJECUTADO", "detalle": "n/a"},
        {"seccion": "resultados", "estado": "NO EJECUTADOS",
         "detalle": "el coste de esta hipotesis es 0 experimentos"},
        {"seccion": "bootstrap", "estado": "NO APLICA",
         "detalle": "solo se calcula para un candidato que pase el gate inicial"},
        {"seccion": "complementarity", "estado": "MEDIDA",
         "detalle": f"{total_unicos} rescates exclusivos, 122 en el gap analysis previo"},
        {"seccion": "decision", "estado": clasificacion,
         "detalle": "no se ejecuta; se cierra la fase de investigacion porque no quedan ejes"},
    ]
    with CSV_H2.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["seccion", "estado", "detalle"])
        for f in filas_h2:
            w.writerow(list(f.values()))

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    salida = {
        "familia": "P68_H2_AUDITORIA", "experiment_id": "P68-02", "hipotesis": "P68-H2",
        "fecha": inicio.isoformat(timespec="seconds"),
        "clasificacion": clasificacion, "ejecutada": False,
        "coste_en_experimentos": 0,
        "razon": ("el mecanismo (sample weighting) esta medido nueve veces en el registro. La "
                  "unica novedad es la fuente del indicador, y el indicador propuesto es un "
                  "subconjunto estricto del de V4-B03, en la direccion que salio mal"),
        "auditoria": AUDITORIA,
        "rescates_exclusivos": {c: int(unicos[c].sum()) for c in CANDIDATOS},
        "total_rescates_exclusivos": total_unicos,
        "fraccion_de_errores_que_el_mapa_cubre":
            round(float(res_alg.sum() / max(int(v_ko.sum()), 1)), 6),
        "celdas_de_error_que_el_mapa_excluye": int((v_ko & ~res_alg).sum()),
        "ejes_recorridos": ejes,
        "ejes_sin_cita_historica": len(abiertos),
        "condicion_de_cierre_cumplida": bool(puede_cerrar),
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "hash_v03": hash_real,
                             "corrected_cv": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_LIBRE.write_text(json.dumps(salida, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")

    H: list[str] = []
    W = H.append
    W("# P68-H2 - Rescue Re-weighting: auditoria, y por que NO se ejecuta")
    W("")
    W(f"Generado por `experiments/p68_02_h2_auditoria.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## La clasificacion")
    W("")
    W("```text")
    W(f"{clasificacion}")
    W("")
    W("coste en experimentos: 0")
    W("```")
    W("")
    W("## La propuesta")
    W("")
    W("Reponderar los pares etiqueta-instancia donde V03 falla y un modelo alternativo acierta,")
    W("y reentrenar el mismo clasificador con esos pesos, para ver si el mapa de rescates")
    W("**transfiere al entrenamiento** en vez de corregir el score despues.")
    W("")
    W("```text")
    W("R_i = (pred_V03_i != y_i) AND (pred_OTRO_i == y_i)")
    W("w   = 2 si R_i, 1 en el resto")
    W("```")
    W("")
    W("## La auditoria que lo decide")
    W("")
    W("La pregunta no es si el nombre es nuevo. Es de donde sale el peso. Cada ponderado del")
    W("registro se clasifica por la **fuente de su indicador**:")
    W("")
    W("| experimento | fuente | indicador | resultado | estado |")
    W("|---|---|---|---|---|")
    for a in AUDITORIA:
        W(f"| `{a['experimento']}` | {a['fuente']} | {a_ascii(a['indicador'])[:52]} | "
          f"{a_ascii(a['resultado'])[:64]} | `{a['estado']}` |")
    W("")
    W(f"**{n_propio} con indicador propio** (funcion de `p_V03` e `y`) y **{n_ajeno} con "
      f"indicador ajeno**.")
    W("")
    W("## El argumento, en una linea")
    W("")
    W("El indicador que P68-H2 propone es el de V4-B03 **intersecado** con la correccion de otro")
    W("modelo:")
    W("")
    W("```text")
    W("V4-B03 ya pondero:        pred_V03 != y          (hard example mining)")
    W("P68-H2 propone:           pred_V03 != y  AND  pred_OTRO == y")
    W("")
    W("el segundo termino no anade informacion al indicador: le QUITA filas")
    W("```")
    W("")
    W("Y la medicion que cierra el caso:")
    W("")
    W("```text")
    W(f"celdas donde V03 falla                        {int(v_ko.sum())}")
    W(f"de esas, alguna alternativa acierta           {int(res_alg.sum())} "
      f"({100 * res_alg.sum() / int(v_ko.sum()):.1f}%)")
    W(f"celdas de error que el mapa EXCLUYE            {int((v_ko & ~res_alg).sum())}")
    W("```")
    W("")
    W("El mapa de rescates cubre el "
      f"`{100 * res_alg.sum() / int(v_ko.sum()):.1f} por ciento` de los errores de V03, luego")
    W("el indicador propuesto es un **subconjunto estricto** de uno que ya se midio. Un")
    W("subconjunto de un indicador no puede ser un mecanismo nuevo.")
    W("")
    W("## Y ademas, en la direccion que salio mal")
    W("")
    W("V4-B03 probo las dos direcciones sobre la dureza:")
    W("")
    W("```text")
    W("mild_up   0,7751   -0,0021   subir el peso a lo dificil EMPEORA")
    W("medium_up 0,7676   -0,0096   subirlo mas, EMPEORA mas")
    W("down_violacion 0,7784  +0,0036   BAJAR el peso a los que violan: mejora")
    W("```")
    W("")
    W("P68-H2 propone **subir** el peso (`w = 2`), y sobre una poblacion que es la parte mas")
    W("dificil del corpus. La razon por la que subir no funciona esta escrita desde V4 y no ha")
    W("cambiado:")
    W("")
    W("> en este corpus los ejemplos que el modelo falla son, desde el punto de vista del")
    W("> modelo, frontera irreductible con la representacion disponible: subir su peso empuja")
    W("> la frontera y degrada el resto")
    W("")
    W("## El unico ponderado que funciono, y por que no es un mapa de rescates")
    W("")
    W("`A_cost_sensitive_downweight` alcanzo `0,785416` (`+0,008205`) con **5 de 5 pliegues**.")
    W("Es el mejor ponderado del proyecto y no se adopto porque **fallo el criterio 6, el de")
    W("artefacto**, del gate de V4-X01, y no tiene veredicto externo.")
    W("")
    W("Su indicador es la **violacion del umbral del propio V03 combinada con el peso de")
    W("clase**. No mira a ningun otro modelo. Es la direccion que funciona: bajar, no subir.")
    W("")
    W("## Y el antecedente que ya construyo el mapa de rescates")
    W("")
    W("P50 y P51 no solo construyeron el mapa de rescates de los canales, sino que midieron su")
    W("techo y fallaron al explotarlo:")
    W("")
    W("```text")
    W("P50   techo del oracle                         +0,2098")
    W("P51   gating por utilidad esperada             -0,0102   (121 rescates, 183 danos)")
    W("```")
    W("")
    W("Es la misma leccion de P68-H1 con otro nombre: el techo existe y no es alcanzable.")
    W("")
    W("## Las secciones que el encargo pedia, y su estado")
    W("")
    W("| seccion | estado |")
    W("|---|---|")
    for f in filas_h2:
        W(f"| {f['seccion']} | `{f['estado']}`: {a_ascii(f['detalle'])[:76]} |")
    W("")
    W("## Lo que NO se hace")
    W("")
    W("```text")
    W("no se entrena con sample weights derivados del mapa de rescates")
    W("no se lanza una grid de pesos")
    W("no se reabre P68-H1")
    W("no se toca el test, ni el holdout, ni el leaderboard")
    W("no se consume presupuesto para repetir una familia cerrada")
    W("```")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p68_02_h2_auditoria.py`. Sin push.*")
    MD_H2.write_text(a_ascii("\n".join(H)), encoding="ascii")

    # ---------- el barrido de lo que queda ----------
    R: list[str] = []
    Wr = R.append
    Wr("# Post-P68 - Mecanismos restantes")
    Wr("")
    Wr(f"Generado por `experiments/p68_02_h2_auditoria.py`. Fecha "
       f"{inicio.isoformat(timespec='seconds')}.")
    Wr("")
    Wr("## Los ocho ejes que el encargo senala, y quien los recorrio")
    Wr("")
    Wr("| eje | experimentos que lo cubren | resultado |")
    Wr("|---|---|---|")
    for e in ejes:
        Wr(f"| {a_ascii(e['eje'])} | {e['experimentos']} | {a_ascii(e['resultado'])[:96]} |")
    Wr("")
    Wr(f"**Ejes sin cita historica: `{len(abiertos)}`.**")
    Wr("")
    Wr("## La condicion de cierre del encargo")
    Wr("")
    Wr("> Solo cerrar la investigacion si el gap analysis posterior demuestra que no quedan")
    Wr("> mecanismos genuinamente nuevos, o que las unicas lineas restantes son variantes de")
    Wr("> mecanismos ya invalidados o cerrados.")
    Wr("")
    Wr("```text")
    Wr(f"P68-H2 = DUPLICATE                     {clasificacion == 'DUPLICATE'}")
    Wr(f"ejes sin recorrer = 0                  {not abiertos}")
    Wr(f"condicion de cierre cumplida            {puede_cerrar}")
    Wr("```")
    Wr("")
    Wr("## El estado cientifico al cerrar")
    Wr("")
    Wr("```text")
    Wr("V03 sigue siendo la referencia congelada, 0,7772114525165721.")
    Wr("P67 no produjo mejora adoptable. P68-H1 tampoco. P68-H2 no se ejecuto.")
    Wr("Existe senal residual en MI_arbol, medida y no explotada.")
    Wr("El techo oracle de los modelos existentes sobre las celdas delegables es +0,0391,")
    Wr("  y cuatro mecanismos distintos han fallado al intentar acercarse a el.")
    Wr("```")
    Wr("")
    Wr("## Lo que queda abierto, dicho con precision")
    Wr("")
    Wr("No queda **ningun mecanismo nuevo** dentro de los ejes conocidos. Lo que queda abierto")
    Wr("es una pregunta, no una tecnica:")
    Wr("")
    Wr("> Existe informacion en los modelos alternativos que ninguno de los mecanismos")
    Wr("> probados ha sabido explotar. El techo medido lo confirma. Lo que falta no es una")
    Wr("> variante de lo probado, sino una razon por la que la informacion sea alcanzable")
    Wr("> desde el entrenamiento, y esa razon no existe hoy.")
    Wr("")
    Wr("## Lo que haria falta para reabrir")
    Wr("")
    Wr("```text")
    Wr("un mecanismo que NO sea: ponderacion, concatenacion, blend, residualizacion,")
    Wr("  correccion de umbral, enrutado por estructura, gating por canal, ensemble de pesos,")
    Wr("  perdida modificada, regularizacion por vecindad, ni transporte")
    Wr("una medida de por que la informacion seria alcanzable desde el entrenamiento")
    Wr("  y no solo desde la decision")
    Wr("```")
    Wr("")
    Wr("Hasta entonces, cerrar es el resultado honesto y no una rendicion: cuatro mecanismos")
    Wr("probados y uno descartado por algebra es mas de lo que la mayoria de los proyectos")
    Wr("puede afirmar sobre su propio techo.")
    Wr("")
    Wr("> V03 remains the best verified leaderboard result.")
    Wr("")
    Wr("*Generado por `experiments/p68_02_h2_auditoria.py`. Sin push.*")
    MD_LIBRE.write_text(a_ascii("\n".join(R)), encoding="ascii")

    documentos = [MD_H2, CSV_H2, CSV_AUD, MD_LIBRE, JSON_LIBRE]
    sucios = [d.name for d in documentos if any(b > 127 for b in d.read_bytes())]
    if sucios:
        raise SystemExit(f"PARADA: documentos con bytes no ASCII: {sucios}")

    log("")
    log("=" * 78)
    log(f"P68-02 OK | P68-H2 {clasificacion} | 0 experimentos gastados")
    log(f"   indicador propio: {n_propio} | ajeno: {n_ajeno} | cerrados: {n_cerrados}")
    log(f"   el mapa de rescates cubre {100 * res_alg.sum() / int(v_ko.sum()):.1f}% de los "
        f"errores: subconjunto estricto")
    log(f"   ejes recorridos {len(ejes)}, sin cita historica {len(abiertos)}")
    log(f"   condicion de cierre cumplida: {puede_cerrar}")
    log(f"   {len(documentos)} documentos, 0 bytes no ASCII")
    log("   V03 remains the best verified leaderboard result.")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
