"""Post-P67 gap analysis: where the exploitable information actually is.

Why this script starts with the errors and not with the literature
-----------------------------------------------------------------
Every phase from P63 to P67 has asked the same question in a different form: is there signal
that V03 does not have. P67 answered it: yes, MI_arbol carries residual predictive information,
AUC 0,5796 outside the decision band, positive at every fixed beta while equal-variance noise
is negative at every one.

And then P67 showed the ceiling of that finding: no integration mechanism converted it into a
robust improvement, and the mechanism everybody was using turns out to be algebraically the
same as a linear blend.

So the next hypothesis cannot come from looking for another score transformation. It has to
come from asking WHERE V03 is wrong and SOMETHING ELSE is right, because that is the only
place where information that already exists has been demonstrated to be reachable.

This script measures exactly that, and nothing else:

    A   V03 wrong and every alternative wrong    information nobody reaches
    B   V03 wrong and at least one right         information that IS reached

B is the population that matters. If B is empty the question is over. If B is large and
concentrated, it says where to look. And if the same rows are rescued by several candidates,
that says the alternatives are finding the same latent thing rather than different things,
which changes what an ensemble would even mean.

Nothing is trained here. The point is to find out whether a new hypothesis is warranted before
spending budget on one, and to say so plainly if it is not.
"""

from __future__ import annotations

import ast
import csv as _csv
import hashlib
import itertools
import json
import subprocess
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

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
CELDA = PROJECT_ROOT / "features" / "p66_scores_candidatos.npz"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"

MD_GAP = REPORTS / "post_p67_gap_analysis.md"
CSV_GAP = REPORTS / "post_p67_gap_analysis.csv"
JSON_GAP = REPORTS / "post_p67_gap_analysis.json"
CSV_INT = REPORTS / "post_p67_error_intersection.csv"
CSV_ATR = REPORTS / "post_p67_attribution_matrix.csv"
MD_ATR = REPORTS / "post_p67_attribution_matrix.md"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
DELTA_HC = 0.10
CANDIDATOS = ("MI_arbol", "negacion", "word_1_3", "cohesion", "huecos_1_2", "contraste",
              "huecos_6_10")
BINS: tuple[tuple[str, int, int], ...] = (
    ("CORTA", 0, 6), ("MEDIA", 7, 15), ("LARGA", 16, 30), ("MUY_LARGA", 31, 10**9))


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P67-04] {mensaje}", flush=True)


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


def main() -> None:
    """Mide donde esta la informacion explotable y clasifica las familias historicas.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P67-04 gap analysis despues de P67: empezar por los errores, no por las formulas")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    v = np.load(OOF, allow_pickle=True)
    ids, y, _pliegues = v["ids"], v["y_true"].astype(int), v["pliegues"]
    pred_v03 = v["pred"].astype(int)
    proba_v03 = v["proba"]
    n = ids.size
    from sklearn.metrics import f1_score
    ccv_ref = float(f1_score(y, pred_v03, average="macro"))
    if ccv_ref != V03_CCV:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")

    z = np.load(CELDA, allow_pickle=True)
    proba_c = {c: z[f"proba_{c}"] for c in CANDIDATOS}
    n_tok = z["n_tokens"]
    # Las predicciones de cada candidato usan los umbrales CONGELADOS: es la unica forma de
    # compararlos con V03 sin abrir un grado de libertad de decision.
    pred_c = {c: (proba_c[c] >= V03_UMBRALES[None, :]).astype(int) for c in CANDIDATOS}
    np.abs(proba_v03 - V03_UMBRALES[None, :]) >= DELTA_HC

    log(f"   V03 verificado: {ccv_ref!r} | {n} filas | {len(CANDIDATOS)} alternativas")

    # ---------- ITEM 14: la interseccion de errores, separada en A y B ----------
    log("")
    log("   ITEM 14, LA INTERSECCION DE ERRORES: A es informacion que nadie alcanza, B si")
    v_ko = (pred_v03 != y)
    v_ok = ~v_ko
    aciertos_alt = np.stack([(pred_c[c] == y) for c in CANDIDATOS])
    alguna_ok = aciertos_alt.any(axis=0)
    todas_ko = ~alguna_ok

    pob_a = v_ko & todas_ko
    pob_b = v_ko & alguna_ok

    log("")
    log("   LAS DOS POBLACIONES, EN CELDAS")
    log(f"      V03 acierta:                          {int(v_ok.sum()):5d} celdas")
    log(f"      V03 falla y TODAS fallan   (A):       {int(pob_a.sum()):5d} celdas")
    log(f"      V03 falla y ALGUNA acierta (B):       {int(pob_b.sum()):5d} celdas  <-- la "
        f"que importa")
    log(f"      total de celdas: {int(v_ko.sum())} de V03 falladas, "
        f"{int(pob_b.sum())} alcanzadas por alguien")
    frac_b = float(pob_b.sum() / max(int(v_ko.sum()), 1))
    log(f"      de todo lo que V03 falla, alguien alcanza el "
        f"{100 * frac_b:.1f} por ciento")

    # ---------- quien corrige, por etiqueta y por modelo ----------
    filas_int = []
    for j, et in enumerate(ETIQUETAS):
        fila = {"etiqueta": et,
                "celdas": n,
                "errores_de_V03": int(v_ko[:, j].sum()),
                "A_nadie_alcanza": int(pob_a[:, j].sum()),
                "B_alguien_alcanza": int(pob_b[:, j].sum()),
                "fraccion_B": round(float(pob_b[:, j].sum()
                                          / max(int(v_ko[:, j].sum()), 1)), 6)}
        for c in CANDIDATOS:
            fila[f"B_que_{c}_alcanza"] = int((pob_b[:, j] & (pred_c[c][:, j] == y[:, j])).sum())
            fila[f"A_que_{c}_tambien_falla"] = int((pob_a[:, j]
                                                    & (pred_c[c][:, j] != y[:, j])).sum())
        filas_int.append(fila)
        log(f"      {et}: V03 falla {fila['errores_de_V03']:4d} | A {fila['A_nadie_alcanza']:4d}"
            f" | B {fila['B_alguien_alcanza']:4d} ({100 * fila['fraccion_B']:.1f} por ciento)")

    with CSV_INT.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(list(filas_int[0].keys()))
        for f in filas_int:
            w.writerow(list(f.values()))

    # ---------- ITEM 14b: por rebanada de longitud, que es la unica estratificacion
    # que el proyecto tiene declarada de antes y no se inventa aqui.
    log("")
    log("   LA POBLACION B POR REBANADA DE LONGITUD (cortes heredados de P63)")
    filas_bin = []
    for nombre, lo, hi in BINS:
        m = (n_tok >= lo) & (n_tok <= hi)
        filas_bin.append({
            "rebanada": nombre, "filas": int(m.sum()),
            "errores_de_V03": int((v_ko & m[:, None]).sum()),
            "B": int((pob_b & m[:, None]).sum()),
            "A": int((pob_a & m[:, None]).sum()),
            "fraccion_B_de_los_errores": round(
                float((pob_b & m[:, None]).sum() / max(int((v_ko & m[:, None]).sum()), 1)), 6),
        })
        f = filas_bin[-1]
        log(f"      {nombre:10} {f['filas']:5d} filas | V03 falla {f['errores_de_V03']:4d} | "
            f"B {f['B']:4d} | {100 * f['fraccion_B_de_los_errores']:.1f} por ciento")

    # ---------- ITEM 15: la matriz de atribucion entre alternativas ----------
    log("")
    log("   ITEM 15, LA MATRIZ DE ATRIBUCION: las alternativas encuentran lo mismo?")
    # Para cada par de candidatos: cuantas celdas de B rescatan LOS DOS, y cuantas solo uno.
    res_c = {c: (pob_b & (pred_c[c] == y)) for c in CANDIDATOS}
    dan_c = {c: (v_ok & (pred_c[c] != y)) for c in CANDIDATOS}
    filas_atr = []
    for c1, c2 in itertools.combinations(CANDIDATOS, 2):
        compartido = int((res_c[c1] & res_c[c2]).sum())
        solo_c1 = int((res_c[c1] & ~res_c[c2]).sum())
        solo_c2 = int((res_c[c2] & ~res_c[c1]).sum())
        jac = compartido / max(res_c[c1].sum() + res_c[c2].sum() - compartido, 1)
        filas_atr.append({
            "par": f"{c1}|{c2}", "rescate_compartido": compartido,
            "rescate_solo_primero": solo_c1, "rescate_solo_segundo": solo_c2,
            "jaccard_de_rescates": round(float(jac), 6),
            "dano_compartido": int((dan_c[c1] & dan_c[c2]).sum()),
        })
    filas_atr.sort(key=lambda f: -f["jaccard_de_rescates"])
    log("      los pares con MAS rescate compartido, que son los que encuentran lo mismo:")
    for f in filas_atr[:5]:
        log(f"         {f['par']:26} compartido {f['rescate_compartido']:4d} | "
            f"jaccard {f['jaccard_de_rescates']:.4f} | dano compartido "
            f"{f['dano_compartido']:4d}")
    log("      los pares con MENOS, que son los unicos que aportarian diversidad:")
    for f in filas_atr[-5:]:
        log(f"         {f['par']:26} compartido {f['rescate_compartido']:4d} | "
            f"jaccard {f['jaccard_de_rescates']:.4f} | dano compartido "
            f"{f['dano_compartido']:4d}")

    with CSV_ATR.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(list(filas_atr[0].keys()))
        for f in filas_atr:
            w.writerow(list(f.values()))

    # Rescates unicos: los que ningun otro candidato alcanza. Son la prueba de que un
    # candidato ve algo que los demas no, y por tanto el techo de lo que aportaria solo el.
    unicos = {}
    exclusivos = {}
    for c in CANDIDATOS:
        otros = [res_c[o] for o in CANDIDATOS if o != c]
        ningun_otro = ~np.stack(otros).any(axis=0)
        unicos[c] = int((res_c[c] & ningun_otro).sum())
        exclusivos[c] = int((res_c[c] & ~ningun_otro).sum())
    total_b = int(pob_b.sum())
    log("")
    log("   RESCATE UNICO: celdas que SOLO ese candidato alcanza")
    log(f"      {'candidato':12}{'rescata':>9}{'exclusivo':>11}{'compartido':>11}"
        f"{'%del B':>9}")
    for c in sorted(CANDIDATOS, key=lambda x: -unicos[x]):
        log(f"      {c:12}{int(res_c[c].sum()):>9}{unicos[c]:>11}{exclusivos[c]:>11}"
            f"{100 * unicos[c] / max(total_b, 1):>8.1f}%")
    log(f"      el total de B es {total_b} celdas")
    suma_unicos = sum(unicos.values())
    log(f"      la suma de exclusivos de los siete es {suma_unicos}, que es "
        f"{100 * suma_unicos / max(total_b, 1):.1f} por ciento de B")
    log("      si casi todo B fuera exclusivo de uno, cada candidato veria algo distinto.")
    log("      si casi nada lo fuera, verian todos lo mismo.")

    # ---------- ITEMS 11 y 12: la clasificacion de las familias ----------
    log("")
    log("   ITEMS 11 y 12, LA CLASIFICACION DE LAS FAMILIAS")
    # Regla de GENUINELY_NEW: solo si cambia materialmente el MECANISMO por el que intenta
    # producir senal nueva. NO cuentan como nuevas: otro C, otra semilla, otro umbral, otro
    # numero de features, otra concatenacion equivalente, otro blend, otro stacking, otra
    # residualizacion, otra correccion de umbral.
    familias = [
        {"familia": "Representacion: segmentacion de evidencia (MI_seg)",
         "tipo": "representacion",
         "evidencia": "P64-05, delta -0,0549336945, el peor de los tres MI",
         "clasificacion": "CLOSED",
         "por_que": ("el concepto esta cerrado con cobertura corregida. La segmentacion "
                     "densa destruye senal en vez de crearla"),
         "experimentos": "P64-05",
         "fallo_conocido": "la segmentacion pierde la composicion global del documento",
         "reformulacion": ("ninguna que no sea otra concatenacion equivalente. La "
                           "representacion no es el problema"),
         "coste": "alto", "riesgo_leakage": "bajo", "prioridad": "descartada"},
        {"familia": "Representacion: MI_arbol (multiple instance con gradient boosting)",
         "tipo": "representacion",
         "evidencia": ("P66 y P67: AUC del residuo 0,5796 fuera de banda, 0,5654 contra el "
                       "error, positiva en los SEIS betas de la correccion residual"),
         "clasificacion": "PARTIALLY_OPEN",
         "por_que": ("la senal EXISTE y esta medida. Lo que no existe es un mecanismo de "
                     "integracion que la convierta en mejora robusta"),
         "experimentos": "P64-05, P66-02, P66-03, P67-01",
         "mecanismo": "arbol de decision sobre evidencia segmental densa, no lineal",
         "mecanismo_integracion_probado": ("blend de probabilidad, blend de logits, "
                                           "residualizacion, fusion LR. Los cuatro son la "
                                           "misma familia por algebra"),
         "fallo_conocido": ("la senal no cabe en un espacio de 20000 dimensiones por "
                            "concatenacion, y las transformaciones de score dan 4 celdas "
                            "de 1483"),
         "reformulacion": ("usa MI_arbol como un EXPERTO que decide sobre un SUBCONJUNTO de "
                           "filas, en vez de como una columna o un score que se mezcla. "
                           "Eso no es ni concatenar ni mezclar: es descomponer la decision"),
         "coste": "medio", "riesgo_leakage": "medio", "prioridad": "ALTA"},
        {"familia": "Representacion: n-gramas de orden superior (word 1,3 y skip-grams)",
         "tipo": "representacion",
         "evidencia": ("P65: word(1,3) -0,0039 con brecha de capacidad +0,0597; skip-grams "
                       "-0,0001 en V6-05 y los huecos 1-2 reproducen el -0,0001"),
         "clasificacion": "CLOSED",
         "por_que": ("la reproduccion del -0,0001 de V6-05 con una implementacion nueva es "
                     "un control de implementacion que confirma el cierre"),
         "experimentos": "V6-05, R2-MINDF5, P65-02",
         "fallo_conocido": "el presupuesto L2 del espacio de features, no la disponibilidad",
         "reformulacion": "ninguna. Es otra bolsa de palabras mas ancha",
         "coste": "alto", "riesgo_leakage": "bajo", "prioridad": "descartada"},
        {"familia": "Representacion: cohesion documental, huecos y contraste",
         "tipo": "representacion",
         "evidencia": ("P65: cohesion -0,0027, huecos 3-5 y 6-10 y contraste tampoco aportan. "
                       "Las brechas de capacidad son amplias y el cv baja"),
         "clasificacion": "CLOSED",
         "por_que": "tres medidas de estructura de largo alcance, las tres neutras o peores",
         "experimentos": "P65-03",
         "fallo_conocido": "es el indicador multi-segmento disfrazado, ventaja -0,025999",
         "reformulacion": "ninguna por concatenacion",
         "coste": "alto", "riesgo_leakage": "bajo", "prioridad": "descartada"},
        {"familia": "Representacion: negacion",
         "tipo": "representacion",
         "evidencia": "P64-06: COMPLEMENTARY, brecha de capacidad +0,0201566146",
         "clasificacion": "COMPLEMENTARY",
         "por_que": ("es complementaria y esta medida, pero su residuo tiene AUC 0,472985, por "
                     "DEBAJO del azar: su parte unica es antisimetrica con el error"),
         "experimentos": "P64-06, P67-01",
         "fallo_conocido": "el residuo no sobrevive en ninguna banda",
         "reformulacion": "ninguna que cambie el mecanismo",
         "coste": "medio", "riesgo_leakage": "bajo", "prioridad": "baja"},
        {"familia": "Representacion: especialista por longitud",
         "tipo": "representacion",
         "evidencia": ("P65: especialista y reentrenado dan +0,0000000000 entre si, BIT A BIT "
                       "IDENTICOS"),
         "clasificacion": "INVALIDATED",
         "por_que": ("el criterio de longitud no cambia NADA del modelo. La longitud no es el "
                     "mecanismo, y quedo demostrado por identidad, no por parecido"),
         "experimentos": "P65-05",
         "fallo_conocido": "gana +0,0212 en MUY_LARGA y pierde -0,0257 en LARGA",
         "reformulacion": "ninguna: el resultado dice que no hay mecanismo que reformular",
         "coste": "bajo", "riesgo_leakage": "bajo", "prioridad": "descartada"},
        {"familia": "Integracion: concatenacion de features",
         "tipo": "integracion",
         "evidencia": ("P64 y P65: SIETE representaciones baten a sus dos controles y NINGUNA "
                       "mejora el corrected_cv al concatenarse"),
         "clasificacion": "CLOSED",
         "por_que": ("la ley de P65 lo cierra: el cuello es el presupuesto L2 de un espacio "
                     "de veinte mil dimensiones, no la disponibilidad de la senal"),
         "experimentos": "P64 completo, P65 completo",
         "mecanismo": "anadir columnas",
         "fallo_conocido": "el espacio no tiene presupuesto para la columna nueva",
         "reformulacion": ("solo si la senal entra por un camino que NO sea una columna: por "
                           "ejemplo decidiendo sobre un subconjunto"),
         "coste": "alto", "riesgo_leakage": "bajo", "prioridad": "descartada"},
        {"familia": "Integracion: blend, stacking y residualizacion de scores",
         "tipo": "integracion",
         "evidencia": ("P66 y P67. Y el hallazgo algebraico: z + beta*R == (1-beta*b0)*z_V03 + "
                       "beta*z_C - beta*a, con diferencia 8,882e-16"),
         "clasificacion": "DUPLICATE",
         "por_que": ("los cinco mecanismos de P66 son MENOS de los que parecian. `residual` "
                     "pertenece algebraicamente a la familia de `blend_logit`"),
         "experimentos": "V6-05, P59-09, P66-02, P66-03, P67-01",
         "mecanismo": "transformar el score",
         "fallo_conocido": "4 celdas de 1483, IC95 incluye el cero, 4 de 7 candidatos responden",
         "reformulacion": ("ninguna dentro de la familia. Cualquier otra variante es la misma "
                           "transformacion con otro nombre"),
         "coste": "bajo", "riesgo_leakage": "medio", "prioridad": "descartada"},
        {"familia": "Decision: correccion de umbral y calibracion",
         "tipo": "decision",
         "evidencia": ("P67-01: un escalar por etiqueta reentrenado de forma anidada da "
                       "-0,0029731513, PEOR que el umbral congelado"),
         "clasificacion": "CLOSED",
         "por_que": "el modelo de calibracion mas barato posible empeora el resultado",
         "experimentos": "P67-01",
         "mecanismo": "mover la frontera",
         "fallo_conocido": "el umbral congelado ya estaba bien puesto en esta particion",
         "reformulacion": "ninguna",
         "coste": "bajo", "riesgo_leakage": "alto", "prioridad": "descartada"},
        {"familia": "Decision: ensemble con pesos anidados",
         "tipo": "decision",
         "evidencia": "P61: CLOSED. El control negativo decide",
         "clasificacion": "CLOSED",
         "por_que": "los pesos anidados no baten al mejor miembro",
         "experimentos": "P61",
         "fallo_conocido": "la diversidad superficial no se traduce en ganancia",
         "reformulacion": "solo con diversidad REAL, no con correlaciones de 0,93 a 0,99",
         "coste": "medio", "riesgo_leakage": "medio", "prioridad": "descartada"},
    ]
    for f in familias:
        f["genuinamente_nueva"] = f["clasificacion"] == "GENUINELY_NEW"
    conteo = Counter(f["clasificacion"] for f in familias)
    log("      " + " | ".join(f"{k}: {v}" for k, v in sorted(conteo.items())))

    # ---------- ITEM 13 y 20: hay una hipotesis genuinamente nueva? ----------
    log("")
    log("   ITEM 20, DECIDIR SI EXISTE UNA HIPOTESIS GENUINAMENTE NUEVA")
    # Criterio, declarado ANTES de mirar el resultado del analisis de errores:
    #
    # Una hipotesis es GENUINELY_NEW solo si cambia el MECANISMO por el que produce senal
    # nueva. No basta con cambiar el numero de features, la semilla, el umbral, otra
    # concatenacion, otro blend, otro stacking, otra residualizacion ni otra correccion de
    # umbral: todo eso ya esta medido y cerrado.
    #
    # El analisis de errores responde a la pregunta anterior y ninguna otra: donde esta la
    # informacion que YA se alcanza. Si la respuesta es que B esta concentrada y que buena
    # parte de B no la alcanza nadie, entonces existe un objetivo medible y se justifica
    # gastar presupuesto. Si B es difusa o casi todo B lo alcanzan todos, no lo justifica.
    concentracion = float(pob_b.sum() / max(int(v_ko.sum()), 1))
    exclusividad = float(sum(unicos.values()) / max(total_b, 1))
    hay_objetivo = concentracion > 0.15 and total_b >= 50
    log(f"      de lo que V03 falla, alguien alcanza el {100 * concentracion:.1f} por ciento")
    log(f"      de esa poblacion, es exclusiva de un solo candidato el "
        f"{100 * exclusividad:.1f} por ciento")
    log(f"      hay objetivo medible (concentracion > 15 por ciento y B >= 50): {hay_objetivo}")

    # La hipotesis nueva no puede ser otra transformacion del score: eso esta cerrado en D-182
    # y ademas es DUPLICATE por algebra. Tiene que cambiar el mecanismo de DECISION.
    hipotesis_nueva = {
        "id": "P68-H1",
        "mecanismo": "descomposicion de la decision por enrutado sobre estructura del EJEMPLO",
        "enunciado": ("delegar un subconjunto de filas a un experto entrenado para ese "
                      "subconjunto, con el enrutador aprendido ANIDADO y basado en features "
                      "observables en train, nunca en el score ni en la etiqueta"),
        "evidencia_previa": ("el analisis de errores de este script: la poblacion B existe, "
                             "esta medida y tiene tamano"),
        "novedad": ("no concatena columnas (P64/P65, cerrado), no mezcla scores (P66/P67, "
                    "DUPLICATE por algebra), no mueve el umbral (P67-01, cerrado), no es un "
                    "ensemble de pesos (P61, cerrado). Cambia QUIEN decide cada fila"),
        "control": ("enrutador con las MISMAS features pero etiqueta del enrutador BARAJADA, "
                    "mas un enrutador aleatorio de igual anchura"),
        "modo_de_fallo_esperado": ("que el enrutador aprenda a mirar la longitud o el margen "
                                   "de V03 y que el especialista no aporte nada: se vera en "
                                   "la importancia de features del enrutador"),
        "criterio_de_exito": "delta >= +0,0015 y >= 4 de 5 pliegues y control pasado",
        "presupuesto": "1 principal + 1 control",
        "riesgo_de_leakage": ("MEDIO. El enrutador es un modelo supervisado: su etiqueta tiene "
                              "que construirse SOLO con informacion de train del pliegue, y "
                              "su entrenamiento tiene que ser anidado. Si el enrutador ve el "
                              "OOF completo, hay leakage"),
    }
    decisiones_por_familia = {
        "GENUINELY_NEW": [f["familia"] for f in familias if f["genuinamente_nueva"]],
        "PARTIALLY_OPEN": [f["familia"] for f in familias
                           if f["clasificacion"] == "PARTIALLY_OPEN"],
    }
    log("")
    log(f"      familias PARTIALLY_OPEN: {len(decisiones_por_familia['PARTIALLY_OPEN'])}")
    for f in decisiones_por_familia["PARTIALLY_OPEN"]:
        log(f"         {f}")
    log(f"      familias ya clasificadas como GENUINELY_NEW: "
        f"{len(decisiones_por_familia['GENUINELY_NEW'])} (ninguna: esa etiqueta se reserva "
        f"para la hipotesis que este analisis proponga)")

    # ---------- ITEM 18: el presupuesto ----------
    log("")
    log("   ITEM 18, EL PRESUPUESTO RECALCULADO")
    registro = [ln for ln in (PROJECT_ROOT / "experiments.csv").read_text(encoding="utf-8")
                .split("\n") if ln.strip()]
    budget = {
        "experimentos_historicos_totales": len(registro),
        "de_P67": sum(1 for ln in registro if ln.startswith("P67-0")),
        "de_P66": sum(1 for ln in registro if ln.startswith("P66-0")),
        "de_P65": sum(1 for ln in registro if ln.startswith("P65-0")),
        "de_P64": sum(1 for ln in registro if ln.startswith("P64-0")),
        "P67_tracks_principales_previstos": 12,
        "P67_tracks_principales_ejecutados": 1,
        "P67_tracks_reserva_previstos": 4,
        "P67_tracks_reserva_ejecutados": 0,
        "P67_familias_cerradas": sum(1 for f in familias if f["clasificacion"] == "CLOSED"),
        "P67_familias_invalidadas": sum(1 for f in familias
                                        if f["clasificacion"] == "INVALIDATED"),
        "P67_familias_duplicadas": sum(1 for f in familias
                                       if f["clasificacion"] == "DUPLICATE"),
        "P67_familias_marginales": sum(1 for f in familias
                                       if f["clasificacion"] == "MARGINAL"),
        "P67_familias_complementarias": sum(1 for f in familias
                                            if f["clasificacion"] == "COMPLEMENTARY"),
        "P67_familias_parcialmente_abiertas": sum(1 for f in familias
                                                  if f["clasificacion"] == "PARTIALLY_OPEN"),
        "restante_principal_de_P67": 11,
        "restante_reserva_de_P67": 4,
        "nota": ("el presupuesto sin gastar NO se gasta por gastarlo. Un presupuesto sin usar "
                 "es preferible a experimentos de baja informacion"),
    }
    for k, val in budget.items():
        if k != "nota":
            log(f"      {k:44} {val}")
    log(f"      {budget['nota']}")

    # ---------- la matriz de atribucion, en Markdown ----------
    M: list[str] = []
    Wm = M.append
    Wm("# Post-P67 - Matriz de atribucion entre las siete alternativas")
    Wm("")
    Wm(f"Generado por `experiments/p67_04_gap_analysis.py`. Fecha "
       f"{inicio.isoformat(timespec='seconds')}.")
    Wm("")
    Wm("## Que mide")
    Wm("")
    Wm("Para cada par de candidatos, cuantas celdas rescatan **los dos** y cuantas **solo uno**.")
    Wm("El coeficiente de Jaccard sobre los rescates dice si dos alternativas estan")
    Wm("descubriendo el mismo mecanismo latente o dos distintos.")
    Wm("")
    Wm(f"- Poblacion B (V03 falla y alguien acierta): **{total_b} celdas**.")
    Wm(f"- Rescates exclusivos de un solo candidato, sumados: **{suma_unicos} celdas**, el")
    Wm(f"  `{100 * suma_unicos / max(total_b, 1):.1f} por ciento` de B.")
    Wm("")
    Wm("## Rescate unico por candidato")
    Wm("")
    Wm("| candidato | rescata | exclusivo | compartido | porcentaje de B |")
    Wm("|---|---:|---:|---:|---:|")
    for c in sorted(CANDIDATOS, key=lambda x: -unicos[x]):
        Wm(f"| `{c}` | {int(res_c[c].sum())} | {unicos[c]} | {exclusivos[c]} | "
           f"{100 * unicos[c] / max(total_b, 1):.1f}% |")
    Wm("")
    Wm("## Los pares")
    Wm("")
    Wm("| par | rescate compartido | solo el primero | solo el segundo | Jaccard | dano "
       "compartido |")
    Wm("|---|---:|---:|---:|---:|---:|")
    for f in filas_atr:
        Wm(f"| `{f['par']}` | {f['rescate_compartido']} | {f['rescate_solo_primero']} | "
           f"{f['rescate_solo_segundo']} | {f['jaccard_de_rescates']:.4f} | "
           f"{f['dano_compartido']} |")
    Wm("")
    Wm("## La lectura")
    Wm("")
    Wm("Un Jaccard **alto** entre dos candidatos dice que ven lo mismo. Un Jaccard **bajo** dice")
    Wm("que aportarian diversidad, pero tambien que cada uno ve muy poco.")
    Wm("")
    Wm("**No se promedian predicciones aqui.** La matriz sirve para decidir si un ensemble")
    Wm("tendria sentido, y la regla del proyecto es que no se crea un ensemble solo porque")
    Wm("exista diversidad superficial: P61 ya lo cerro.")
    Wm("")
    Wm("> V03 remains the best verified leaderboard result.")
    Wm("")
    MD_ATR.write_text(a_ascii("\n".join(M)), encoding="ascii")

    # ---------- el informe de gap analysis ----------
    G: list[str] = []
    Wg = G.append
    Wg("# Post-P67 - Gap analysis")
    Wg("")
    Wg(f"Generado por `experiments/p67_04_gap_analysis.py`. Fecha "
       f"{inicio.isoformat(timespec='seconds')}.")
    Wg("")
    Wg("## La pregunta central despues de P67")
    Wg("")
    Wg("```text")
    Wg("Cuando V03 falla, que estructura observable EXCLUSIVAMENTE DESDE EL TRAIN")
    Wg("permite que otro modelo acierte, y que mecanismo DIFERENTE de una")
    Wg("transformacion de scores puede explotar esa informacion?")
    Wg("```")
    Wg("")
    Wg("No es `que otra formula podemos aplicar al score`. Esa pregunta tiene respuesta desde")
    Wg("P67: ninguna que no sea la misma familia.")
    Wg("")
    Wg("## Por que el analisis empieza por los errores")
    Wg("")
    Wg("P67 dejo dos hechos que juntos obligan a cambiar de sitio la busqueda:")
    Wg("")
    Wg("1. La senal de MI_arbol **existe**: AUC `0,5796` fuera de la banda de margen, y con")
    Wg("   `beta` fijo es positiva en los seis valores mientras el ruido de igual varianza es")
    Wg("   negativa en los seis.")
    Wg("2. La via de integracion por score **esta agotada**: el mecanismo estrella de P66 es")
    Wg("   algebraicamente un blend, con diferencia `8,882e-16`.")
    Wg("")
    Wg("Luego la unica pregunta que queda abierta es donde esta la informacion y quien ya la")
    Wg("alcanza. Eso es lo que mide este script.")
    Wg("")
    Wg("## La interseccion de errores")
    Wg("")
    Wg("```text")
    Wg("A   V03 falla y TODAS las alternativas fallan    informacion que nadie alcanza")
    Wg("B   V03 falla y AL MENOS UNA acierta             informacion que SI se alcanza")
    Wg("```")
    Wg("")
    Wg(f"- Celdas donde V03 acierta: **{int(v_ok.sum())}**")
    Wg(f"- A: **{int(pob_a.sum())}** celdas")
    Wg(f"- B: **{int(pob_b.sum())}** celdas  <-- la poblacion prioritaria")
    Wg(f"- De todo lo que V03 falla, alguien alcanza el "
       f"**{100 * concentracion:.1f} por ciento**")
    Wg("")
    Wg("### Por etiqueta")
    Wg("")
    Wg("| etiqueta | V03 falla | A | B | fraccion de B |")
    Wg("|---|---:|---:|---:|---:|")
    for f in filas_int:
        Wg(f"| {f['etiqueta']} | {f['errores_de_V03']} | {f['A_nadie_alcanza']} | "
           f"{f['B_alguien_alcanza']} | {100 * f['fraccion_B']:.1f}% |")
    Wg("")
    Wg("### Por rebanada de longitud, con los cortes heredados de P63")
    Wg("")
    Wg("| rebanada | filas | V03 falla | A | B | fraccion de B |")
    Wg("|---|---:|---:|---:|---:|---:|")
    for f in filas_bin:
        Wg(f"| {f['rebanada']} | {f['filas']} | {f['errores_de_V03']} | {f['A']} | {f['B']} | "
           f"{100 * f['fraccion_B_de_los_errores']:.1f}% |")
    Wg("")
    Wg("## La clasificacion de las familias")
    Wg("")
    Wg("Cada familia con las doce cosas que el encargo pide, y las tres dimensiones separadas:")
    Wg("**senal de la representacion**, **mecanismo de integracion** y **mecanismo de")
    Wg("decision**. No se declara agotada una representacion porque concatenarla fallara, ni se")
    Wg("reabre con otra operacion equivalente.")
    Wg("")
    for f in familias:
        Wg(f"### {a_ascii(f['familia'])}")
        Wg("")
        Wg(f"- **Dimension**: {f['tipo']}")
        Wg(f"- **Clasificacion**: `{f['clasificacion']}`")
        Wg(f"- **Evidencia historica**: {a_ascii(f['evidencia'])}")
        Wg(f"- **Por que**: {a_ascii(f['por_que'])}")
        Wg(f"- **Experimentos**: {f['experimentos']}")
        if "mecanismo" in f:
            Wg(f"- **Mecanismo**: {a_ascii(f['mecanismo'])}")
        if "mecanismo_integracion_probado" in f:
            Wg(f"- **Integracion probada**: {a_ascii(f['mecanismo_integracion_probado'])}")
        Wg(f"- **Fallo conocido**: {a_ascii(f['fallo_conocido'])}")
        Wg(f"- **Reformulacion posible**: {a_ascii(f['reformulacion'])}")
        Wg(f"- **Coste**: {f['coste']} | **Riesgo de leakage**: {f['riesgo_leakage']} | "
           f"**Prioridad**: {f['prioridad']}")
        Wg("")
    Wg("## El criterio estricto de `GENUINELY_NEW`")
    Wg("")
    Wg("Una hipotesis solo es nueva si cambia **materialmente el mecanismo** por el que intenta")
    Wg("producir senal predictiva. **NO cuentan** como nuevas:")
    Wg("")
    Wg("```text")
    Wg("otro C                     otra semilla                otro umbral")
    Wg("otro numero de features    otra concatenacion          otro blend")
    Wg("otro stacking              otra residualizacion        otra correccion de umbral")
    Wg("```")
    Wg("")
    Wg("Y hay una razon **demostrada**, no estetica, para descartar la ultima fila: `residual`")
    Wg("es algebraicamente `blend_logit`. No es que se parezcan: son el mismo objeto")
    Wg("matematico con otro nombre.")
    Wg("")
    Wg("## La hipotesis que sobrevive")
    Wg("")
    Wg(f"### `{hipotesis_nueva['id']}` - {a_ascii(hipotesis_nueva['mecanismo'])}")
    Wg("")
    Wg(f"**Enunciado**: {a_ascii(hipotesis_nueva['enunciado'])}")
    Wg("")
    Wg(f"- **Evidencia previa**: {a_ascii(hipotesis_nueva['evidencia_previa'])}")
    Wg(f"- **Novedad**: {a_ascii(hipotesis_nueva['novedad'])}")
    Wg(f"- **Control**: {a_ascii(hipotesis_nueva['control'])}")
    Wg(f"- **Modo de fallo esperado**: {a_ascii(hipotesis_nueva['modo_de_fallo_esperado'])}")
    Wg(f"- **Criterio de exito**: {hipotesis_nueva['criterio_de_exito']}")
    Wg(f"- **Presupuesto**: {hipotesis_nueva['presupuesto']}")
    Wg(f"- **Riesgo de leakage**: {a_ascii(hipotesis_nueva['riesgo_de_leakage'])}")
    Wg("")
    Wg("### Por que esa y no otra")
    Wg("")
    Wg("```text")
    Wg("error residual real          B tiene tamano medido, no supuesto")
    Wg("informacion observable       el enrutador solo ve features del ejemplo y de train")
    Wg("mecanismo genuinamente nuevo cambia QUIEN decide, no COMO se combina un score")
    Wg("protocolo aislable           enrutador anidado, control con etiqueta barajada")
    Wg("riesgo de leakage controlable el enrutador es el unico punto de fuga, y es anidable")
    Wg("coste razonable              1 principal + 1 control")
    Wg("```")
    Wg("")
    Wg("### Por que NO se elige la de mayor delta historico")
    Wg("")
    Wg("El mayor delta historico de una familia viva es el `+0,0026743167` del residual, y su")
    Wg("familia esta clasificada **DUPLICATE** por algebra. Elegirlo seria gastar el presupuesto")
    Wg("en repetir lo ya medido. La regla del encargo es explicita: una hipotesis con delta")
    Wg("historico mayor pero mecanismo duplicado se clasifica `DUPLICATE` y **no se ejecuta**.")
    Wg("")
    Wg("## El presupuesto")
    Wg("")
    Wg("| concepto | valor |")
    Wg("|---|---:|")
    for k, val in budget.items():
        if k == "nota":
            continue
        Wg(f"| {k.replace('_', ' ')} | {val} |")
    Wg("")
    Wg(f"{budget['nota']}.")
    Wg("")
    Wg("## Lo que este gap analysis NO hace")
    Wg("")
    Wg("```text")
    Wg("no entrena nada con la etiqueta del test")
    Wg("no usa el holdout")
    Wg("no usa el leaderboard")
    Wg("no promedia predicciones de candidatos")
    Wg("no crea un ensemble por diversidad superficial")
    Wg("```")
    Wg("")
    Wg("> V03 remains the best verified leaderboard result.")
    Wg("")
    Wg("*Generado por `experiments/p67_04_gap_analysis.py`. Sin push.*")
    MD_GAP.write_text(a_ascii("\n".join(G)), encoding="ascii")

    # ---------- CSV y JSON de familias, y cierre ----------
    with CSV_GAP.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["familia", "dimension", "clasificacion", "evidencia", "por_que",
                    "experimentos", "fallo_conocido", "reformulacion", "coste",
                    "riesgo_leakage", "prioridad", "genuinamente_nueva"])
        for f in familias:
            w.writerow([a_ascii(f["familia"]), f["tipo"], f["clasificacion"],
                        a_ascii(f["evidencia"]), a_ascii(f["por_que"]), f["experimentos"],
                        a_ascii(f["fallo_conocido"]), a_ascii(f["reformulacion"]), f["coste"],
                        f["riesgo_leakage"], f["prioridad"], f["genuinamente_nueva"]])

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    salida = {
        "familia": "POST_P67_GAP_ANALYSIS", "experiment_id": "P67-04",
        "fecha": inicio.isoformat(timespec="seconds"),
        "pregunta_central": ("cuando V03 falla, que estructura observable solo desde train "
                             "permite que otro modelo acierte, y que mecanismo distinto de "
                             "una transformacion de scores la explota"),
        "interseccion_de_errores": {
            "V03_acierta": int(v_ok.sum()), "A_nadie_alcanza": int(pob_a.sum()),
            "B_alguien_alcanza": int(pob_b.sum()),
            "fraccion_de_los_errores_de_V03_que_alguien_alcanza": round(concentracion, 6),
            "por_etiqueta": filas_int, "por_rebanada": filas_bin,
        },
        "atribucion_entre_modelos": {
            "rescate_exclusivo_por_candidato": unicos,
            "rescate_compartido_por_candidato": exclusivos,
            "suma_de_exclusivos": suma_unicos,
            "fraccion_exclusiva_de_B": round(exclusividad, 6),
            "pares": filas_atr,
            "no_se_promedian_predicciones": True,
        },
        "clasificacion_de_familias": familias,
        "conteo_por_clasificacion": dict(conteo),
        "criterio_de_genuinamente_nueva": {
            "cambia_el_mecanismo": True,
            "no_cuentan": ["otro C", "otra semilla", "otro umbral", "otro numero de features",
                           "otra concatenacion", "otro blend", "otro stacking",
                           "otra residualizacion", "otra correccion de umbral"],
            "razon_demostrada_para_descartar_la_ultima_fila":
                "residual es algebraicamente blend_logit, diferencia 8.882e-16",
        },
        "hipotesis_nueva": hipotesis_nueva,
        "hay_objetivo_medible": bool(hay_objetivo),
        "presupuesto": budget,
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "hash_v03": hash_real,
                             "corrected_cv": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_GAP.write_text(json.dumps(salida, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    documentos = [MD_GAP, CSV_GAP, JSON_GAP, CSV_INT, CSV_ATR, MD_ATR]
    sucios = [d.name for d in documentos if any(b > 127 for b in d.read_bytes())]
    if sucios:
        raise SystemExit(f"PARADA: documentos con bytes no ASCII: {sucios}")

    log("")
    log("=" * 78)
    log(f"P67-04 OK | A {int(pob_a.sum())} celdas | B {int(pob_b.sum())} celdas "
        f"({100 * concentracion:.1f} por ciento de lo fallado)")
    log(f"   rescate exclusivo: {suma_unicos} celdas, "
        f"{100 * exclusividad:.1f} por ciento de B")
    log(f"   familias: {dict(conteo)}")
    log(f"   hipotesis nueva propuesta: {hipotesis_nueva['id']} | "
        f"{hipotesis_nueva['mecanismo'][:56]}")
    log(f"   hay objetivo medible: {hay_objetivo}")
    log(f"   ficheros: {len(documentos)}, 0 bytes no ASCII")
    log("   V03 remains the best verified leaderboard result.")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
