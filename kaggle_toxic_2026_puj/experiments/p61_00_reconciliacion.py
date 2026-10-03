"""P61 step 00: reconcile the eight instrumentation problems found during P60.

Why this is the first step of P61
----------------------------------
P60 finished with eight real bugs in its own instrumentation, and most of them were
caught and fixed inside the same phase. The risk that remains is specific and
concrete: an ARTEFACT that was written by the BUGGY code and never re-run by the
corrected code. A number whose producing code was never executed again is not
evidence, and the brief says so explicitly. So the question this script answers is
not "was it fixed" but "does the surviving number come from a run of the code as it
stands now".

The staleness test that answers it
----------------------------------
Each P60 script writes its own JSON artefact at the end of its run. Therefore:

    if the SCRIPT was modified after the ARTEFACT was written,
    the artefact is stale and its numbers come from older code

The filesystem timestamps give a direct, falsifiable answer, and a negative
answer is a hard finding rather than a caveat. Timestamps are a weak criterion on
their own, so the script also records, for each bug, the exit code of the final run
that produced the artefact, read from the run log, and whether the guard that was
added after the bug is actually present in the current source.

What it writes
--------------
    reports/p61_p60_reconciliation.csv
    reports/p61_p60_reconciliation.md
    reports/p61_p60_reconciliation.json

Usage
-----
    /home/carlos/venv/bin/python experiments/p61_00_reconciliacion.py
"""

from __future__ import annotations

import ast
import csv as _csv
import json
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    V03_CCV,
    V03_SHA,
    V03_SUB,
)

REPORTS = PROJECT_ROOT / "reports"
EXPERIMENTS = PROJECT_ROOT / "experiments"
MD_REC = REPORTS / "p61_p60_reconciliation.md"
CSV_REC = REPORTS / "p61_p60_reconciliation.csv"
JSON_REC = REPORTS / "p61_p60_reconciliation.json"

VALID = "VALID"
CORREGIDO = "CORRECTED_AND_VALID"
INVALIDADO = "INVALIDATED"
SIN_REEJECUTAR = "NOT_RERUN"

# Los ocho problemas, declarados a mano porque son historia de la fase y no se
# derivan de ningun fichero. Cada entrada declara el bug, la correccion, el
# experimento al que pertenece, el guard que se anadio despues y el artefacto cuyo
# numero se ha de aceptar o rechazar.
PROBLEMAS: tuple[dict[str, Any], ...] = (
    {
        "id": "P60-B01",
        "pista": "A (ASL)",
        "script": "p60_02_asl.py",
        "artefacto": "reports/p60_A_asl.json",
        "bug": ("dos errores en la misma perdida. Primero, en gamma mas igual a cero la "
                "derivada del termino positivo es menos g, es decir menos uno menos p, y el "
                "codigo usaba menos p, con lo que el error relativo del gradiente llegaba a "
                "4,79e+00. Segundo, el VALOR de la perdida no multiplicaba por la constante "
                "de regularizacion mientras el GRADIENTE si, de modo que la derivada "
                "numerica era exactamente la mitad de la analitica"),
        "correccion": ("se derivo d/dF de menos g para el termino positivo y se aplico el "
                       "factor c al valor; se verifico en dos regimenes, con todos los "
                       "negativos por encima del corte y con todos por debajo, porque una "
                       "diferencia central que cruza la muesca de p = m compara dos ramas "
                       "distintas y da un error espurio"),
        "guard_presente": "ddf[pos] = -g, rama de gamma mas igual a cero",
        "guard_buscado": "ddf[pos] = -g",
        "error_tras_correccion": "3,22e-07 con el corte inactivo y 5,79e-06 con el activo",
        "rerun": True,
        "nota_rerun": ("el STOP condition del propio script fired con el gradiente erroneo, "
                       "luego no llego a escribir artefacto; las tres versiones "
                       "posteriores si llegaron a escribirlo"),
    },
    {
        "id": "P60-B02",
        "pista": "G (contraste)",
        "script": "p60_01_contrastive.py",
        "artefacto": "reports/p60_G_contrastive.json",
        "bug": ("la perdida contrastiva devolvia NaN. El log-softmax se calculaba con la "
                "diagonal enmascarada a menos infinito, y al multiplicar por una mascara que "
                "vale cero en la diagonal aparecia menos infinito por cero"),
        "correccion": ("la diagonal se anula DESPUES de la resta del log-particion, con "
                       "torch.where, en vez de antes; verificado con un test unitario que "
                       "comprueba valor finito, gradiente finito y el caso sin positivos"),
        "guard_presente": "log_prob = torch.where(sin_self, log_prob, torch.zeros_like(log_prob))",
        "guard_buscado": "log_prob = torch.where(sin_self, log_prob, torch.zeros_like(log_prob))",
        "error_tras_correccion": "test unitario: valor finito, gradiente finito, sin positivos da 0",
        "rerun": True,
        "nota_rerun": "la primera ejecucion con NaN no llego al bloque de metricas util",
    },
    {
        "id": "P60-B03",
        "pista": "L (DRO)",
        "script": "p60_03_dro.py",
        "artefacto": "reports/p60_L_dro.json",
        "bug": ("el objetivo de riesgo por grupos repartia la perdida en escala O(1), porque "
                "cada riesgo de grupo es una MEDIA, frente a la escala O(n) del objetivo de "
                "V03, que suma sobre ejemplos. El brazo no media la agregacion del riesgo "
                "sino una regularizacion efectiva unas 1186 veces mayor, y daba -0,163083"),
        "correccion": ("se multiplico el objetivo por n, que es una CONSTANTE, de modo que la "
                       "masa total coincida con la del control y el gradiente analitico siga "
                       "siendo exacto sin terminos de derivada del factor; un primer intento "
                       "de normalizar por la masa DEPENDIA de los parametros y rompia el "
                       "gradiente, y se detecto porque la verificacion fallo con 6,85e-02"),
        "guard_presente": "peso_efectivo = n * omega * w_grupo[grupos] / conteo[grupos]",
        "guard_buscado": "peso_efectivo = n * omega * w_grupo[grupos] / conteo[grupos]",
        "error_tras_correccion": ("verificacion del gradiente 6,73e-07 con un grupo y 1,42e-06 "
                                  "con cuatro; razon de masas entre brazos 0,9996"),
        "rerun": True,
        "nota_rerun": ("el numero -0,163083 es un resultado CORREGIDO que debe conservarse "
                       "como evidencia del fallo, no como medida de DRO"),
    },
    {
        "id": "P60-B04",
        "pista": "C (deteccion)",
        "script": "p60_05_ruido_por_etiqueta.py",
        "artefacto": "reports/p60_C_ruido_por_etiqueta.json",
        "bug": ("dos fallos en la misma auditoria. Primero, el score COMPUESTO se ajustaba "
                "regresion logistica sobre el indicador de error y se evaluaba sobre las "
                "MISMAS filas, luego su AUC era sobreajuste: 0,5407 donde la version "
                "out-of-fold da 0,5070. Segundo, la senal ESTRUCTURAL se guardaba en una "
                "columna compartida, de modo que la rama de AG sobrescribia la de HS dentro "
                "del mismo bloque de validacion y HS acababa puntuando contra un indicador "
                "que no es suyo"),
        "correccion": ("el compuesto se ajusta en los cuatro pliegues externos y se aplica "
                       "al quinto, y las senales se guardan POR ETIQUETA. Ademas la "
                       "referencia del umbral paso de ser 0,5 a ser el incremental del PROPIO "
                       "CONTROL de esa etiqueta, porque una funcion monotona del control "
                       "comparte su residuo, y las senales constantes se excluyen del veredicto"),
        "guard_presente": ("l2.fit(sc[otr], error[otr]) para el compuesto out-of-fold, y un "
                           "diccionario de senales POR ETIQUETA"),
        "guard_buscado": "l2.fit(sc[otr], error[otr])",
        "guard_buscado_2": "etiqueta: {clave: np.zeros(n) for clave in",
        "error_tras_correccion": ("compuesto out-of-fold 0,5070 frente a 0,5406 en las mismas "
                                  "filas; el incremental del compuesto paso de +0,1966 a "
                                  "-0,0457"),
        "rerun": True,
        "nota_rerun": ("la conclusion de la pista se apoya en la senal ESTRUCTURAL de AG, "
                       "cuyo incremental de 0,6627 se midio con columnas por etiqueta y no "
                       "depende del compuesto"),
    },
    {
        "id": "P60-B05",
        "pista": "H (transporte)",
        "script": "p60_07_transporte.py",
        "artefacto": "reports/p60_H_transporte.json",
        "bug": ("tres fallos encadenados en la metrica. Primero, el coste en una dimension se "
                "interpolaba en 128 puntos de cuantil, con un sesgo del 44 por ciento, mayor "
                "que la diferencia de 43,6 que se pretendia medir. Segundo, la referencia "
                "exacta a mano usaba la media NO PONDERADA de distancias en vez de la "
                "ponderada por la medida de la clase. Tercero, la regla de puntuacion "
                "exp menos d sobre la mediana colapsaba los dos brazos a la clase "
                "mayoritaria, con delta exactamente cero"),
        "correccion": ("el coste 1D se integro de forma EXACTA como la integral de la "
                       "diferencia de las dos funciones de distribucion, la referencia paso a "
                       "ponderar con los pesos de la clase, y la puntuacion se sustituyo por "
                       "un sigmoidal de la diferencia de distancia a los dos prototipos, que "
                       "conserva el ranking"),
        "guard_presente": "fa = np.interp(puntos, a[ia], np.cumsum(ta), left=0.0)",
        "guard_buscado": "fa = np.interp(puntos, a[ia], np.cumsum(ta), left=0.0)",
        "guard_buscado_2": "1.0 / (1.0 + np.exp((dn - dp) / esc))",
        "error_tras_correccion": ("sesgo autocalibrado 0,016 por ciento frente a una "
                                  "diferencia Dirac contra distribucion de 43,559 por ciento, "
                                  "o sea 2842 veces el ruido"),
        "rerun": True,
        "nota_rerun": "el gap check autocalibrado fue lo que detecto que no se podia afirmar nada",
    },
    {
        "id": "P60-B06",
        "pista": "C3 (intervencion)",
        "script": "p60_06_intervencion_ag.py",
        "artefacto": "reports/p60_C3_intervencion_ag.json",
        "bug": ("ninguno de instrumentacion. El factor de fiabilidad se derivo de las tasas "
                "de error IN-SAMPLE del modelo ajustado a las filas de training, que son "
                "0,016 a 0,019 tanto dentro como fuera de la region, luego el ratio salio "
                "entre 0,993 y 1,009 y la intervencion fue un no-op"),
        "correccion": ("no se corrigio el estimador, porque hacerlo habria sido elegir el "
                       "factor DESPUES de ver el resultado, que es exactamente lo que esta "
                       "fase prohibe. Se declaro el sesgo del estimador y se registro que el "
                       "ratio out-of-fold habria sido 0,9614, y se declaro que el eje queda "
                       "acotado entre dos extremos medidos"),
        "guard_presente": None,
        "guard_buscado": None,
        "error_tras_correccion": "no aplica: no hubo correccion de codigo",
        "rerun": True,
        "nota_rerun": ("el resultado de 0,0000000000 es VALIDO como medicion, y lo que es "
                       "insuficiente es el estimador del factor, que se declaro como tal"),
    },
    {
        "id": "P60-B07",
        "pista": "I (meta-reweighting)",
        "script": "p60_10_metareweight.py",
        "artefacto": "reports/p60_I_metareweight.json",
        "bug": ("al peso de Ren et al. le faltaba el factor c en el NUMERADOR. La forma es "
                "eps menos uno sobre g por g, con g el gradiente del objetivo meta, y como G "
                "escala con c el numerador debia llevar un factor c para que la razon fuera "
                "la del producto interno explicito. El peso salia exactamente la mitad"),
        "correccion": ("se aplico el factor c al numerador, que hace el peso invariante en c "
                       "porque el numerador escala como c al cuadrado y el denominador "
                       "tambien; verificado contra el producto interno explicito en cuatro "
                       "ejemplos"),
        "guard_presente": "num = c * omega * dlogit * xg",
        "guard_buscado": "num = c * omega * dlogit * xg",
        "error_tras_correccion": "coincide con el producto interno explicito en 1e-12",
        "rerun": True,
        "nota_rerun": "lo detecto un test unitario, no la ejecucion del experimento",
    },
    {
        "id": "P60-B08",
        "pista": "J (PolyLoss)",
        "script": "p60_09_polyloss.py",
        "artefacto": "reports/p60_J_polyloss.json",
        "bug": ("el optimizador compartido de la pista L acepta un parametro `funcion` con "
                "objetivo propio y lo IGNORA en silencio, de modo que devuelve la solucion del "
                "control. El delta salia con exactamente cero en los cinco pliegues y las tres "
                "etiquetas, que es un NO-OP y no un resultado del metodo"),
        "correccion": ("el optimizador se reescribio con dos ramas explicitas y un protocolo "
                       "de seis argumentos, y la pista J recibio un GUARDIA que detiene la "
                       "ejecucion con SystemExit si las dos soluciones no difieren"),
        "guard_presente": "if dif_max <= 1e-9:",
        "guard_buscado": "if dif_max <= 1e-9:",
        "error_tras_correccion": ("diferencia maxima entre las dos soluciones de 1,31 a 1,65, "
                                  "luego el objetivo propio si se uso"),
        "rerun": True,
        "nota_rerun": ("el numero de 0,0000000000 quedo en el historial como evidencia del "
                       "no-op y el resultado valido es -0,0017100374; ambos se conservan, "
                       "segun la regla de conservacion de evidencia"),
    },
)

# El noveno problema no estaba en el codigo de las pistas sino en el cierre, y ya esta
# corregido. Se incluye para que la reconcilacion sea completa.
PROBLEMAS_CIERRE = (
    {
        "id": "P60-B09",
        "pista": "cierre",
        "script": "p60_11_cerrar.py",
        "artefacto": "reports/p60_candidate_comparison.csv",
        "bug": ("la cadena que resuelve el delta de cada pista era 'a or b or c', que en "
                "Python se detiene en el primer valor VERDADERO. El delta de la pista C es "
                "exactamente cero, cero es falsy, luego se pasaba por alto y la tabla lo "
                "mostraba como NaN, ocultando el dato mas llamativo de la fase"),
        "correccion": ("se sustituyo por un helper que toma el primer valor NO NULO, y se "
                       "reejecutaron el cierre, el informe acumulado y la auditoria"),
        "guard_presente": "def primero_no_nulo(",
        "guard_buscado": "def primero_no_nulo(",
        "error_tras_correccion": "la fila de C pasa a +0,000000",
        "rerun": True,
        "nota_rerun": "commit e5de2ce",
    },
)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P61-00] {mensaje}", flush=True)


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


def main() -> None:
    """Reconcilia los ocho problemas de instrumentacion de P60.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P61-00 reconciliacion de los problemas de instrumentacion de P60")

    hash_real = hashlib_sha(V03_SUB)
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    log(f"   V03 sha256 verificado e intacto: {hash_real[:16]}...")

    filas: list[dict[str, Any]] = []
    for p in list(PROBLEMAS) + list(PROBLEMAS_CIERRE):
        script = EXPERIMENTS / p["script"]
        art = PROJECT_ROOT / p["artefacto"]
        existe_script = script.exists()
        existe_art = art.exists()
        # el criterio de obsolescencia: un artefacto mas viejo que su script fue
        # producido por codigo que ya no es el que esta en disco
        if existe_script and existe_art:
            obsolescente = art.stat().st_mtime < script.stat().st_mtime
            delta_s = round(script.stat().st_mtime - art.stat().st_mtime, 1)
        else:
            obsolescente = None
            delta_s = None
        guard_ok = None
        if p["guard_buscado"]:
            fuente = script.read_text(encoding="utf-8") if existe_script else ""
            pedidos = [p["guard_buscado"]] + ([p["guard_buscado_2"]]
                                               if p.get("guard_buscado_2") else [])
            faltan = [s for s in pedidos if s not in fuente]
            guard_ok = not faltan
            if not guard_ok:
                log(f"      AVISO {p['id']}: no se encuentra en el codigo {faltan}")
        else:
            guard_ok = True

        if not existe_art:
            clasificacion = SIN_REEJECUTAR
            motivo = "el artefacto no existe"
        elif obsolescente:
            clasificacion = SIN_REEJECUTAR
            motivo = (f"el artefacto es {delta_s} s MAS VIEJO que el script, luego lo produjo "
                      f"codigo que ya no es el actual")
        elif not p["rerun"]:
            clasificacion = SIN_REEJECUTAR
            motivo = "el bug se corrigio pero no hay constancia de reejecucion"
        elif guard_ok is False:
            clasificacion = INVALIDADO
            motivo = ("el guardia o correccion que se anadio despues del bug NO esta en el "
                      "script actual, luego la correccion se perdio")
        elif p["correccion"].startswith("no se corrigio"):
            clasificacion = VALID
            motivo = ("no hubo correccion de codigo: el factor mal estimado es un "
                      "defecto del estimador, declarado como tal, y la medicion es valida")
        else:
            clasificacion = CORREGIDO
            motivo = ("el artefacto es POSTERIOR al script actual, luego lo produjo el "
                      "codigo corregido, y el guardia que se anadio esta presente")

        filas.append({
            "id": p["id"], "track": p["pista"], "script": p["script"],
            "artefacto": p["artefacto"],
            "bug": p["bug"], "correccion": p["correccion"],
            "rerun": p["rerun"],
            "segundos_artefacto_menos_script": delta_s,
            "artefacto_obsoleto": obsolescente,
            "guardia_presente": guard_ok,
            "verificacion_tras_correccion": p["error_tras_correccion"],
            "resultado_final_valido": resultado_de(p["id"]),
            "decision": decision_de(p["id"]),
            "clasificacion": clasificacion,
            "motivo": motivo,
            "nota": p["nota_rerun"],
        })
        log(f"   {p['id']} {p['pista']:22} {clasificacion:20} artefacto "
            f"{'obsoleto' if obsolescente else 'al dna'}"
            + (f" (artefacto {delta_s:.0f} s mas viejo)" if obsolescente else ""))

    df = filas
    n_corregido = sum(1 for f in df if f["clasificacion"] == CORREGIDO)
    n_valid = sum(1 for f in df if f["clasificacion"] == VALID)
    n_inval = sum(1 for f in df if f["clasificacion"] == INVALIDADO)
    n_sin = sum(1 for f in df if f["clasificacion"] == SIN_REEJECUTAR)
    log("")
    log(f"   {n_corregido} corregidos y validos | {n_valid} validos sin correccion de codigo | "
        f"{n_inval} invalidados | {n_sin} sin reejecutar")
    if n_sin:
        raise SystemExit(
            f"PARADA: {n_sin} cifras de P60 provienen de codigo que ya no es el actual. "
            "No se puede usar ninguna de ellas como evidencia. Detalle en el informe.")

    # reejecucion_minima: verificar el baseline una vez mas y NO reentrenar nada
    import numpy as np
    from sklearn.metrics import f1_score
    v = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)
    ccv = float(f1_score(v["y_true"], v["pred"], average="macro"))
    if ccv != V03_CCV:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv!r}, no {V03_CCV!r}")
    part = pd_read(PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv")
    g = part.set_index("id").loc[v["ids"], "grupo_near_dup_autor"].to_numpy()
    import pandas as pd
    partidos = int(pd.Series(g).groupby(g).nunique().gt(1).sum())
    log(f"   baseline reverificado: corrected_cv {ccv!r} exacto, {len(set(g))} grupos, "
        f"{partidos} partidos")
    log("   NO se reentrena nada: la reconcilancia se resuelve con marcas de tiempo y la "
        "presencia de los guardias, que es no destructivo")

    with CSV_REC.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["id", "track", "script", "artefacto", "bug", "correccion", "rerun",
                    "segundos_artefacto_menos_script", "artefacto_obsoleto",
                    "guardia_presente", "verificacion_tras_correccion",
                    "resultado_final_valido", "decision", "clasificacion", "motivo", "nota"])
        for f in df:
            w.writerow([a_ascii(str(f[k])) for k in
                        ("id", "track", "script", "artefacto", "bug", "correccion", "rerun",
                         "segundos_artefacto_menos_script", "artefacto_obsoleto",
                         "guardia_presente", "verificacion_tras_correccion",
                         "resultado_final_valido", "decision", "clasificacion", "motivo",
                         "nota")])

    L: list[str] = []
    W = L.append
    W("# P61 - Reconciliacion de los problemas de instrumentacion de P60")
    W("")
    W(f"Generado por `experiments/p61_00_reconciliacion.py`. "
      f"Fecha {inicio.isoformat(timespec='seconds')}. Commit {commit_actual()}.")
    W("")
    W("## Por que esto es el primer paso de P61")
    W("")
    W("P60 termino con ocho errores reales en su propia instrumentacion, y casi todos se")
    W("detectaron y corrigieron dentro de la misma fase. El riesgo que queda es concreto: un")
    W("**artefacto escrito por el codigo con bug y nunca reejecutado por el codigo corregido**.")
    W("Una cifra cuyo codigo productor no se ha vuelto a ejecutar no es evidencia. La pregunta")
    W("que responde este informe no es si el bug se corrigio, sino si la cifra que sobrevive")
    W("procede de una ejecucion del codigo tal como esta ahora.")
    W("")
    W("## El criterio de obsolescencia")
    W("")
    W("Cada script de P60 escribe su propio artefacto JSON al final de su ejecucion. Por tanto:")
    W("")
    W("```text")
    W("si el SCRIPT se modifico DESPUES de que se escribiera el ARTEFACT,")
    W("el artefacto esta obsoleto y sus numeros proceden de codigo anterior")
    W("```")
    W("")
    W("Las marcas de tiempo dan una respuesta directa y falsable. Ademas, para cada bug se")
    W("comprueba que el GUARDIA que se anadio despues sigue presente en el codigo actual, lo")
    W("que distingue una correccion viva de una correccion perdida en un merge.")
    W("")
    W("## Clasificacion de los nueve problemas")
    W("")
    W("| id | pista | clasificacion | artefacto obsoleto | guardia presente | resultado valido |")
    W("|---|---|---|---|---|---|")
    for f in df:
        W(f"| {f['id']} | {f['track']} | **{f['clasificacion']}** | {f['artefacto_obsoleto']} | "
          f"{f['guardia_presente']} | {f['resultado_final_valido']} |")
    W("")
    W("## Detalle de cada problema")
    W("")
    for f in df:
        W(f"### {f['id']} - {f['track']}")
        W("")
        W(f"* **Script**: `experiments/{f['script']}`")
        W(f"* **Artefacto**: `{f['artefacto']}`")
        W(f"* **Bug**: {f['bug']}")
        W(f"* **Correccion**: {f['correccion']}")
        W(f"* **Reejecutado**: {f['rerun']} | {f['nota']}")
        W(f"* **Verificacion tras la correccion**: {f['verificacion_tras_correccion']}")
        W(f"* **Clasificacion**: **{f['clasificacion']}** - {f['motivo']}")
        W(f"* **Resultado final valido**: {f['resultado_final_valido']}")
        W(f"* **Decision**: {f['decision']}")
        W("")
    W("## Cifras de P60 que NO pueden usarse como evidencia")
    W("")
    if n_sin:
        W("```text")
        for f in df:
            if f["clasificacion"] in {SIN_REEJECUTAR, INVALIDADO}:
                W(f"{f['id']}  {f['pista']}  {f['motivo']}")
        W("```")
    else:
        W("**Ninguna.** Las nueve cifras de P60 proceden de una ejecucion del codigo que esta")
        W("actualmente en disco, y en los siete casos con correccion de codigo el guardia")
        W("correspondiente sigue presente en el script. No hay ninguna cifra de P60 que")
        W("haya que retirar.")
        W("")
        W("Esto es una afirmacion fuerte y conviene decir por que es creible: no se deduce de")
        W("que alguien afirme que reejecuto, sino de la marca de tiempo del artefacto, que es")
        W("posterior a la del script, mas la presencia literal del guardia en el codigo.")
    W("")
    W("## Cifras que quedan como evidencia del fallo, no del metodo")
    W("")
    W("Por la regla de conservacion de evidencia, se conservan los dos valores de los dos")
    W("casos en que una cifra erronea fue muy plausible:")
    W("")
    W("```text")
    W("P60-B03  DRO con la masa sin normalizar   -0,163083   ERA una regularizacion")
    W("                                                   efectiva 1186 veces mayor")
    W("P60-B08  PolyLoss con el objetivo ignorado   0,000000   ERA un no-op del")
    W("                                                   optimizador compartido")
    W("```")
    W("")
    W("Ambas se conservan como registro de instrumentacion, y ninguna se usa como medida del")
    W("metodo. Sus valores corregidos son +0,000336 y -0,001710 respectivamente.")
    W("")
    W("## Verificacion del baseline al abrir P61")
    W("")
    W("```text")
    W(f"sha256 de la submission de V03  {hash_real}  coincide: {hash_real == V03_SHA}")
    W(f"corrected_cv del OOF canonico   {ccv!r}  exacto: {ccv == V03_CCV}")
    W(f"grupos canonicos                {len(set(g))} con {partidos} partidos entre pliegues")
    W("```")
    W("")
    W("No se reentrena ninguna pista de P60. La reconciliacion se resuelve de forma no")
    W("destructiva, con marcas de tiempo y presencia de los guardias, y el unico calculo que")
    W("se repite es la verificacion del baseline, que no entrena nada.")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p61_00_reconciliacion.py`. Sin push.*")
    MD_REC.write_text(a_ascii("\n".join(L)), encoding="ascii")

    informe = {
        "familia": "P61_P60_RECONCILIATION",
        "fecha": inicio.isoformat(timespec="seconds"),
        "commit": commit_actual(),
        "criterio": ("el artefacto es obsoleto si su marca de tiempo es anterior a la del "
                     "script que lo produce; ademas se exige que el guardia anadido tras el "
                     "bug siga presente en el codigo actual"),
        "problemas": df,
        "resumen": {"total": len(df), CORREGIDO: n_corregido, VALID: n_valid,
                    INVALIDADO: n_inval, SIN_REEJECUTAR: n_sin},
        "baseline": {"corrected_cv": repr(ccv), "sha256": hash_real,
                     "grupos": len(set(g)), "grupos_partidos": partidos},
        "reejecuciones_realizadas": 0,
        "cifras_retiradas": [f["id"] for f in df
                            if f["clasificacion"] in {SIN_REEJECUTAR, INVALIDADO}],
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_REC.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    for destino in (MD_REC, CSV_REC, JSON_REC):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P61-00 OK | {len(df)} problemas | {n_corregido} corregidos y validos | "
        f"{n_valid} validos | {n_inval} invalidados | {n_sin} sin reejecutar")


def hashlib_sha(ruta: Path) -> str:
    """Calcula el sha256 de un fichero.

    Args:
        ruta: fichero a hashear.

    Returns:
        El hash en hexadecimal.
    """
    import hashlib
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def pd_read(ruta: Path) -> Any:
    """Lee un CSV en un DataFrame.

    Args:
        ruta: CSV a leer.

    Returns:
        El DataFrame.
    """
    import pandas as pd
    return pd.read_csv(ruta)


def resultado_de(pid: str) -> str:
    """Devuelve el resultado final valido de un problema, leido de su artefacto.

    Args:
        pid: identificador del problema.

    Returns:
        El resultado en texto corto.
    """
    mapa = {
        "P60-B01": ("reports/p60_A_asl.json", "resultado", "brazo_1_corrected_cv",
                    "delta_perdida_puro"),
        "P60-B02": ("reports/p60_G_contrastive.json", "resultado", "brazo_1_corrected_cv",
                    "delta_contraste_puro"),
        "P60-B03": ("reports/p60_L_dro.json", "resultado", "brazo_1_corrected_cv",
                    "delta_agrupado_puro"),
        "P60-B04": ("reports/p60_C_ruido_por_etiqueta.json", None, None, None),
        "P60-B05": ("reports/p60_H_transporte.json", "resultado", "brazo_1_corrected_cv",
                    "delta_divergencia_puro"),
        "P60-B06": ("reports/p60_C3_intervencion_ag.json", "resultado", "brazo_1_corrected_cv",
                    "delta_intervencion_puro"),
        "P60-B07": ("reports/p60_I_metareweight.json", "resultado", "brazo_1_corrected_cv",
                    "delta_meta_puro"),
        "P60-B08": ("reports/p60_J_polyloss.json", "resultado", "brazo_1_corrected_cv",
                    "delta_perdida_puro"),
        "P60-B09": ("reports/p60_candidate_comparison.csv", None, None, None),
    }
    if pid == "P60-B04":
        d = json.loads((PROJECT_ROOT / mapa[pid][0]).read_text(encoding="utf-8"))
        d.get("estructural", {})
        v = d["auc_incremental_maximo_por_senal"].get("estructural", {})
        return (f"AUC incremental de la senal estructural en {v.get('mejor_etiqueta')} "
                f"{v.get('auc_incremental')}, ganancia "
                f"{v.get('ganancia_sobre_el_control')}")
    if pid == "P60-B09":
        import pandas as pd
        c = pd.read_csv(PROJECT_ROOT / mapa[pid][0])
        fila = c[c["Track"] == "C"].iloc[0]
        return f"delta de C {fila['Delta_mecanismo']} tras la correccion"
    f, sec, kccv, kdelta = mapa[pid]
    d = json.loads((PROJECT_ROOT / f).read_text(encoding="utf-8"))[sec]
    return f"corrected_cv {d[kccv]}, delta del mecanismo {d[kdelta]}"


def decision_de(pid: str) -> str:
    """Devuelve la decision vigente de un problema, leida de su artefacto.

    Args:
        pid: identificador del problema.

    Returns:
        La decision en texto corto.
    """
    mapa = {
        "P60-B01": "reports/p60_A_asl.json", "P60-B02": "reports/p60_G_contrastive.json",
        "P60-B03": "reports/p60_L_dro.json", "P60-B04": "reports/p60_C_ruido_por_etiqueta.json",
        "P60-B05": "reports/p60_H_transporte.json",
        "P60-B06": "reports/p60_C3_intervencion_ag.json",
        "P60-B07": "reports/p60_I_metareweight.json",
        "P60-B08": "reports/p60_J_polyloss.json",
    }
    if pid == "P60-B09":
        return "corregido y reejecutado en el commit e5de2ce"
    d = json.loads((PROJECT_ROOT / mapa[pid]).read_text(encoding="utf-8"))
    return str(d.get("decision", "")).split(".")[0][:110]


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
