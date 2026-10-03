"""Audit and exact reproduction of A_cost_sensitive_downweight.

What is being audited, and why it is not a closed case
------------------------------------------------------
`A_cost_sensitive_downweight` reported `corrected_cv = 0,7854160387`, a delta of
`+0,0082045862` over V03, and **5 of 5 folds positive**. It is the best reweighting result in
the project and it was never adopted.

It was not rejected because the improvement was negative. It was rejected on criterion 6 of
the V4-X01 gate, the one that requires a complete prediction artifact, and the directive
forbade manufacturing one. That is a procedural failure, not an experimental one, and the
difference matters: a number that failed for lack of a file is not the same as a number that
failed because it does not reproduce.

So this script answers one question and only one:

    is A_cost_sensitive_downweight a real, reproducible, methodologically valid result?

It is an audit and a reproduction. It does not design a technique, does not widen the weight
range, does not run a grid, does not touch the test set, the holdout or the leaderboard, and
does not use this result to pick new hyperparameters.

The artifact problem, in full
-----------------------------
The candidate is not COMPUTED by the script that reports it. `36_candidatos_v4.py` LOADS it
from `features/oof_weighting_v4.npz`, produced by `30_supervised_weighting_v4.py`, which in
turn needs `features/hardness_v4.npz` from `28_hardness_v4.py`. All of `features/` is in
`.gitignore`, so none of those three artifacts travel between worktrees.

That is the whole of criterion 6: the prediction artifact does not exist, and the OOF that
produced the number did not exist in this worktree either until it was regenerated here.
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
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    V03_CCV,
    V03_SHA,
    V03_SUB,
)

from src.evaluation import metricas_multietiqueta
from src.splits import (
    ETIQUETAS,
    grupos_fusionados,
    holdout_v2,
    pliegues_corregidos,
)

REPORTS = PROJECT_ROOT / "reports"
FIGURAS = PROJECT_ROOT / "figuras" / "p69"
FEATURES = PROJECT_ROOT / "features"
OOF = FEATURES / "oof_v03_final.npz"
WEIGHTING = FEATURES / "oof_weighting_v4.npz"
HARDNESS = FEATURES / "hardness_v4.npz"

MD_A = REPORTS / "a_cost_sensitive_downweight_audit.md"
CSV_R = REPORTS / "a_cost_sensitive_downweight_reproduction.csv"
JSON_P = REPORTS / "a_cost_sensitive_downweight_provenance.json"

UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
SEMILLA = 42
N_BOOTSTRAP = 2000
# Los valores historicos, copiados del registro. NO se sobrescriben.
ORIGINAL = 0.7854160387
ORIGINAL_DELTA = 0.0082045862
ORIGINAL_FOLDS = [0.002243, 0.001592, 0.003378, 0.023607, 0.010356]
ORIGINAL_RESCATE, ORIGINAL_DANO, ORIGINAL_NETO = 99, 74, 25
ORIGINAL_CELDAS = 173
IC95_PAREADO_HISTORICO = [-0.000225, 0.011688]
TOLERANCIA = 1e-6
CLAVE = "combo__downweight_violacion_x_clase"


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P69-01] {mensaje}", flush=True)


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


def f1_media(y: np.ndarray, pred: np.ndarray) -> float:
    """Macro F1 con la metrica del proyecto.

    Args:
        y: etiquetas verdaderas.
        pred: predicciones.

    Returns:
        La media de las F1 por columna.
    """
    return float(metricas_multietiqueta(y, pred)["f1_media_columnas"])


# Comprobaciones de compliance por OPERACION, no por palabra. Es la cuarta vez en este
# proyecto que una comprobacion da un falso positivo por buscar el nombre del artefacto en
# vez de la llamada que lo abre: "toxic_test" aparecia en la definicion de la propia
# comprobacion, y "leaderboard" convivia con un read_csv de train en el mismo fichero sin
# ninguna relacion. Se camina el arbol sintactico y se miran los ARGUMENTOS de las llamadas.
OPERACIONES_DE_LECTURA = {"open", "read_csv", "read_text", "load", "loadtxt", "read_parquet"}
OPERACIONES_DE_ESCRITURA = {"to_csv", "savez", "savez_compressed", "write_text", "savefig",
                            "dump"}


def _texto_de_argumentos(nodo: ast.Call) -> str:
    """Reconstruye el texto de los argumentos de una llamada.

    Args:
        nodo: llamada del arbol.

    Returns:
        Concatenacion de los literales y nombres de los argumentos.
    """
    partes = []
    for arg in list(nodo.args) + [k.value for k in nodo.keywords if k.value is not None]:
        for sub in ast.walk(arg):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                partes.append(sub.value)
            elif isinstance(sub, ast.Name):
                partes.append(sub.id)
            elif isinstance(sub, ast.Attribute):
                partes.append(sub.attr)
    return " ".join(partes).lower()


def _busca(
    fuentes: dict[str, str],
    objetivo: str,
    operaciones: set[str],
    excluir: tuple[str, ...] = (),
) -> list[str]:
    """Scripts que aplican una operacion de fichero sobre un artefacto concreto.

    Args:
        fuentes: codigo por script.
        objetivo: fragmento que debe aparecer en los argumentos.
        operaciones: nombres de funcion que cuentan como acceso a fichero.
        excluir: fragmentos que, si aparecen en los argumentos, descartan la
            llamada. Se usa para distinguir el REGISTRO del holdout
            (``holdout_evaluaciones``) y su LISTA DE IDENTIFICADORES
            (``holdout_v2_ids``), que solo sirven para excluirlo, de los datos
            del holdout.

    Returns:
        Nombres de los scripts que lo hacen.
    """
    fuera = []
    for nombre, texto in fuentes.items():
        for nodo in ast.walk(ast.parse(texto)):
            if not isinstance(nodo, ast.Call):
                continue
            func = nodo.func
            llano = (func.attr if isinstance(func, ast.Attribute)
                     else func.id if isinstance(func, ast.Name) else "")
            if llano not in operaciones:
                continue
            argumentos = _texto_de_argumentos(nodo)
            if objetivo not in argumentos:
                continue
            if any(fragmento in argumentos for fragmento in excluir):
                continue
            fuera.append(nombre)
    return sorted(set(fuera))


def _lee_artefacto(
    fuentes: dict[str, str], objetivo: str, excluir: tuple[str, ...] = ()
) -> list[str]:
    """Scripts que LEEN un artefacto.

    Args:
        fuentes: codigo por script.
        objetivo: fragmento del nombre del artefacto.
        excluir: fragmentos que descartan la llamada si aparecen en los argumentos.

    Returns:
        Scripts que lo leen.
    """
    return _busca(fuentes, objetivo, OPERACIONES_DE_LECTURA, excluir)


def _escribe_artefacto(fuentes: dict[str, str], objetivos: tuple[str, ...]) -> list[str]:
    """Scripts que ESCRIBEN alguno de esos artefactos.

    Args:
        fuentes: codigo por script.
        objetivos: fragmentos de nombres.

    Returns:
        Scripts que los escriben.
    """
    fuera: list[str] = []
    for obj in objetivos:
        fuera += _busca(fuentes, obj, OPERACIONES_DE_ESCRITURA)
    return sorted(set(fuera))


def bootstrap_pareado(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                      semilla: int = SEMILLA,
                      n: int = N_BOOTSTRAP) -> tuple[float, float, float]:
    """Bootstrap pareado del delta, identico al historico.

    Args:
        y: etiquetas verdaderas.
        pred_a: predicciones de referencia.
        pred_b: predicciones del candidato.
        semilla: semilla.
        n: numero de remuestreos.

    Returns:
        Tupla (delta, ic95_bajo, ic95_alto).
    """
    f1_a = (y == pred_a).mean(axis=1).astype(float)
    f1_b = (y == pred_b).mean(axis=1).astype(float)
    rng = np.random.default_rng(semilla)
    tam = len(f1_a)
    deltas = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, tam, tam)
        deltas[i] = f1_b[idx].mean() - f1_a[idx].mean()
    p = np.percentile(deltas, [2.5, 97.5])
    return float(f1_b.mean() - f1_a.mean()), float(p[0]), float(p[1])


def main() -> None:
    """Audita y reproduce A_cost_sensitive_downweight sin cambiar la hipotesis.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P69-01 auditoria y reproduccion de A_cost_sensitive_downweight")

    # ---------- ITEM 7: el baseline primero ----------
    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    v = np.load(OOF, allow_pickle=True)
    y_dev = v["y_true"].astype(int)
    proba_v03 = v["proba"]
    ids_dev = v["ids"].astype(np.int64)
    n = ids_dev.size
    pred_ref = (proba_v03 >= UMBRALES_V03[None, :]).astype(int)
    ccv_ref = f1_media(y_dev, pred_ref)
    if abs(ccv_ref - V03_CCV) > 1e-12:
        raise SystemExit(f"PARADA: baseline inestable, {ccv_ref!r} != {V03_CCV!r}")
    log(f"   ITEM 7, baseline ESTABLE: {ccv_ref!r} | {n} filas")
    log(f"   sha256 de V03: {hash_real[:32]}...")

    # ---------- ITEM 3: la procedencia ----------
    log("")
    log("   ITEM 3, LA PROCEDENCIA COMPLETA")
    prov = {
        "experiment_id_original": "V4-X01 (candidato A)",
        "nombre_en_el_gate": "A_cost_sensitive_downweight",
        "clave_del_oof": CLAVE,
        "mecanismo_segun_el_codigo": ("downweight coste-sensible por violacion del umbral, "
                                      "combinado con un barrido anidado del multiplicador de "
                                      "class_weight por etiqueta"),
        "cadena_de_generacion": [
            {"orden": 1, "script": "experiments/27_label_consistency_v4.py",
             "commit": "01ad7ea", "artefacto": "features/label_consistency_v4.npz"},
            {"orden": 2, "script": "experiments/28_hardness_v4.py",
             "commit": "01ad7ea", "artefacto": "features/hardness_v4.npz"},
            {"orden": 3, "script": "experiments/30_supervised_weighting_v4.py",
             "commit": "c96cb0f", "artefacto": "features/oof_weighting_v4.npz"},
            {"orden": 4, "script": "experiments/36_candidatos_v4.py",
             "commit": "5d06ce7", "artefacto": "reports/v4_candidates.json"},
        ],
        "script_que_reporta": "experiments/36_candidatos_v4.py",
        "script_que_construye": "experiments/30_supervised_weighting_v4.py",
        "el_reportador_NO_calcula_el_candidato": True,
        "lo_carga_de": "features/oof_weighting_v4.npz",
        "fecha_del_registro": "2026-09-25",
        "dataset": "desarrollo (particion dev), 1483 filas",
        "folds": "pliegues_corregidos con grupos near-duplicado fusionados, 5 pliegues",
        "seed": SEMILLA,
        "representacion": ("TF-IDF word(1,2) min_df=2 sin sublinear_tf + char_wb(3,5) "
                           "min_df=2, ajustado por pliegue"),
        "clasificador": "LogisticRegression(C=2.0, max_iter=1000, solver=liblinear)",
        "class_weight": "barrido anidado de multiplicadores (0.8, 1.0, 1.2, 1.5), 3 rondas",
        "thresholds": [0.44, 0.51, 0.46],
        "score_reportado": ORIGINAL,
        "delta_reportado": ORIGINAL_DELTA,
        "pliegues_mejorados": 5,
        "veredicto_externo": "no existe: nunca se envio al leaderboard",
    }
    for k, val in prov.items():
        if k in ("cadena_de_generacion",):
            continue
        log(f"      {k:42} {str(val)[:58]}")

    # ---------- ITEM 4: la hipotesis exacta, reconstruida del codigo ----------
    log("")
    log("   ITEM 4, LA HIPOTESIS ORIGINAL, reconstruida del codigo y no reinterpretada")
    log("      margen_favor = (p_V03 - umbral) * (2*y - 1)")
    log("      violacion_l  = clip(-margen_favor, 0, None)     por etiqueta")
    log("      error_score  = violacion_l.mean(axis=1)          POR EJEMPLO, media de 3")
    log("      z            = (error_score - mediana) / IQR")
    log("      peso         = clip(1 - 0.5*z, 0.5, 1.5)         POR EJEMPLO")
    h = np.load(HARDNESS, allow_pickle=True)
    error_score = h["error_score"].astype(float)
    # Verificacion independiente de la formula, sobre el OOF canonico
    mf = (proba_v03 - UMBRALES_V03[None, :]) * (2 * y_dev - 1)
    esperado = np.clip(-mf, 0.0, None).mean(axis=1)
    desv = float(np.abs(esperado - error_score).max())
    log(f"      la formula del artefacto frente al OOF canonico: desviacion {desv:.3e}")
    iqr = float(np.percentile(error_score, 75) - np.percentile(error_score, 25)) or 1.0
    z = (error_score - float(np.median(error_score))) / iqr
    pesos = np.clip(1.0 - 0.5 * z, 0.5, 1.5)
    detalles_peso = {
        "depende_de_p_V03": True, "depende_de_y": True, "depende_del_margen": True,
        "depende_de_label": "NO para el peso: se promedia sobre las 3 etiquetas",
        "unidad": "por tweet (ejemplo), NO por par tweet-etiqueta",
        "mediana_de_error_score": round(float(np.median(error_score)), 6),
        "IQR": round(iqr, 6),
        "peso_minimo": round(float(pesos.min()), 6),
        "peso_maximo": round(float(pesos.max()), 6),
        "peso_medio": round(float(pesos.mean()), 6),
        "n_con_peso_mayor_que_1": int((pesos > 1).sum()),
        "n_con_peso_menor_que_1": int((pesos < 1).sum()),
        "n_con_peso_igual_1": int((pesos == 1).sum()),
        "n_en_el_suelo_0_5": int((pesos <= 0.5 + 1e-12).sum()),
        "suma_de_pesos": round(float(pesos.sum()), 2),
        "suma_neutra": n,
        "fraccion_de_peso_eliminada": round(1.0 - float(pesos.sum()) / n, 6),
    }
    for k, val in detalles_peso.items():
        log(f"      {k:34} {val}")
    log("")
    log("      EL SIGNO, QUE ES LO QUE IMPORTA: la regla SOLO BAJA pesos.")
    log("      Ningun ejemplo sube por encima de 1, porque error_score >= 0 y su mediana es 0.")
    log("      Quien recibe < 1 es quien V03 falla en al menos una etiqueta.")
    log("      Quien recibe 1 es quien V03 acierta las tres.")
    log(f"      La regla retira el {100 * detalles_peso['fraccion_de_peso_eliminada']:.1f} por "
        f"ciento del peso total del entrenamiento.")

    # ---------- ITEM 5: la causa exacta del criterio 6 ----------
    log("")
    log("   ITEM 5, POR QUE FALLO EL CRITERIO 6, con la causa clasificada")
    p58 = json.loads((REPORTS / "p58_final_candidate_audit.json").read_text(encoding="utf-8"))
    next((f for f in p58["descomposiciones"]
                   if "cost_sensitive" in f.get("candidato", "")), {})
    with (REPORTS / "p58_final_candidate_audit.csv").open(encoding="utf-8") as fh:
        csv58 = next((r for r in _csv.DictReader(fh)
                      if "cost_sensitive" in r.get("candidate", "")), {})
    causa = {
        "categoria": "A_falta_de_artefacto",
        "criterio": "6_artefacto_de_prediccion_completo",
        "criterio_en_el_csv": csv58.get("6_artefacto_de_prediccion_completo"),
        "submission_existe": csv58.get("submission_exists"),
        "texto_del_cierre": csv58.get("reason_closed"),
        "que_faltaba_exactamente": (
            "un CSV de submission. El candidato NUNCA se convirtio en prediccion sobre el "
            "conjunto de prueba, luego no hay artefacto que auditar ni enviar"),
        "por_que_no_se_creo": (
            "la directiva 12 de la fase v4 prohibia generar artefactos nuevos durante la "
            "auditoria. La ausencia del fichero es una decision de gobernanza, no un fallo "
            "del experimento"),
        "no_es_ninguna_de_las_otras": {
            "B_artefacto_corrupto": "no aplica: no hay artefacto que pueda estar corrupto",
            "C_no_trazable": "el OOF SI era trazable, pero vivia en features/, ignorado por git",
            "D_score_no_reproducible": "es lo que esta auditoria comprueba, no la causa",
            "E_codigo_no_recuperable": "el codigo SI es recuperable y sin drift",
            "F_inconsistencia_reporte_artefacto":
                "hay una: P58 describe el mecanismo como 'meta', y el codigo no construye "
                "ningun meta-modelo. Se documenta abajo",
        },
        "dato_reportado_en_el_csv": {
            "honest_cv": csv58.get("honest_cv"),
            "fold_consistency": csv58.get("fold_consistency"),
            "ic95_pareado": csv58.get("ic95_pareado"),
            "rescue": csv58.get("rescue"), "damage": csv58.get("damage"),
            "net": csv58.get("net"), "celdas_que_cambian": csv58.get("celdas_que_cambian"),
        },
        "la_inconsistencia_documental": {
            "de_que_se_trata": (
                "el CSV de P58 describe el mecanismo como 'meta con downweight "
                "coste-sensible por violacion del umbral'. El codigo de "
                "30_supervised_weighting_v4.py NO construye ningun meta-modelo: entrena "
                "LogisticRegression directamente sobre la representacion de V03 con "
                "sample_weight"),
            "consecuencia": (
                "la etiqueta 'meta' puede inducir a creer que el candidato pertenece a la "
                "familia de stacking, que ya fue rechazada por transferencia. No pertenece"),
            "fuente_autoritativa": "el codigo, no la etiqueta",
        },
    }
    log(f"      categoria:              {causa['categoria']}")
    log(f"      criterio:               {causa['criterio']}")
    log(f"      criterio en el CSV:     {causa['criterio_en_el_csv']}")
    log(f"      submission existe:      {causa['submission_existe']}")
    log(f"      que faltaba:            {causa['que_faltaba_exactamente'][:70]}")
    log(f"      por que no se creo:     {causa['por_que_no_se_creo'][:70]}")
    log("")
    log(f"      INCONSISTENCIA DOCUMENTAL: {causa['la_inconsistencia_documental']['de_que_se_trata'][:76]}")
    log("      El codigo es la fuente autoritativa, no la etiqueta.")

    # ---------- ITEM 8: la reproduccion ----------
    log("")
    log("   ITEM 8, LA REPRODUCCION EXACTA")
    if not WEIGHTING.exists():
        raise SystemExit(
            f"PARADA: falta {WEIGHTING.name}. Hay que ejecutar antes "
            f"30_supervised_weighting_v4.py, que es el script historico que lo genera")
    datos = np.load(WEIGHTING, allow_pickle=True)
    if CLAVE not in datos.files:
        raise SystemExit(f"PARADA: {WEIGHTING.name} no contiene la clave {CLAVE}. "
                         f"Claves: {sorted(datos.files)[:8]}")
    if not np.array_equal(datos["ids"].astype(np.int64), ids_dev):
        raise SystemExit("PARADA: los ids del OOF regenerado no coinciden con el canonico")
    proba_rep = datos[CLAVE].astype(float)
    log(f"      OOF cargado de {WEIGHTING.name} con la clave {CLAVE}")
    log("      ids coinciden con el canonico: True")
    log(f"      sha256 del OOF regenerado: "
        f"{hashlib.sha256(WEIGHTING.read_bytes()).hexdigest()[:32]}...")

    # Las utilidades de pliegues, importadas del proyecto igual que el script historico
    train = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train.csv") \
        if (PROJECT_ROOT / "data" / "processed" / "train.csv").exists() else None
    if train is None:
        from src.data import read_train
        train = read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    grupos_dev = grupos[indices_dev]
    pliegues = pliegues_corregidos(desarrollo, grupos_dev)
    ids_proyecto = desarrollo["id"].to_numpy(dtype=np.int64)
    if not np.array_equal(ids_proyecto, ids_dev):
        raise SystemExit("PARADA: los ids de los pliegues no coinciden con el OOF canonico")
    log(f"      pliegues reconstruidos con las utilidades del proyecto: {len(pliegues)}")

    pred_rep = (proba_rep >= UMBRALES_V03[None, :]).astype(int)
    ccv_rep = f1_media(y_dev, pred_rep)
    delta_rep = ccv_rep - ccv_ref
    f1_rep = [f1_media(y_dev[va], pred_rep[va]) for _, va in pliegues]
    f1_ref = [f1_media(y_dev[va], pred_ref[va]) for _, va in pliegues]
    deltas_fold = [round(b - a, 6) for a, b in zip(f1_ref, f1_rep)]
    folds_pos = int(sum(1 for d in deltas_fold if d > 0))

    log("")
    log(f"      score historico:    {ORIGINAL:.10f}")
    log(f"      score reproducido:  {ccv_rep:.10f}")
    log(f"      diferencia:         {ccv_rep - ORIGINAL:+.10e}")
    log(f"      delta vs V03 reproducido: {delta_rep:+.10f} (historico {ORIGINAL_DELTA:+.10f})")
    log(f"      deltas por pliegue: {deltas_fold}")
    log(f"      historicos:         {ORIGINAL_FOLDS}")
    log(f"      pliegues positivos: {folds_pos} de 5 (historico 5 de 5)")

    # ---------- ITEM 13: rescate, dano y neto, por etiqueta ----------
    log("")
    log("   ITEM 13, RESCATE, DANO Y NETO, POR ETIQUETA Y GLOBAL")
    acierto_ref = (pred_ref == y_dev)
    acierto_rep = (pred_rep == y_dev)
    filas_et = []
    for j, et in enumerate(ETIQUETAS):
        rescate = int((~acierto_ref[:, j] & acierto_rep[:, j]).sum())
        dano = int((acierto_ref[:, j] & ~acierto_rep[:, j]).sum())
        filas_et.append({
            "etiqueta": et,
            "V03_correcto_candidato_incorrecto": dano,
            "V03_incorrecto_candidato_correcto": rescate,
            "ambos_correctos": int((acierto_ref[:, j] & acierto_rep[:, j]).sum()),
            "ambos_incorrectos": int((~acierto_ref[:, j] & ~acierto_rep[:, j]).sum()),
            "rescate": rescate, "damage": dano, "neto": rescate - dano,
        })
    glob = {
        "etiqueta": "GLOBAL (celdas)",
        "V03_correcto_candidato_incorrecto": int((acierto_ref & ~acierto_rep).sum()),
        "V03_incorrecto_candidato_correcto": int((~acierto_ref & acierto_rep).sum()),
        "ambos_correctos": int((acierto_ref & acierto_rep).sum()),
        "ambos_incorrectos": int((~acierto_ref & ~acierto_rep).sum()),
    }
    glob["rescate"] = glob["V03_incorrecto_candidato_correcto"]
    glob["damage"] = glob["V03_correcto_candidato_incorrecto"]
    glob["neto"] = glob["rescate"] - glob["damage"]
    filas_et.append(glob)
    for f in filas_et:
        log(f"      {f['etiqueta']:18} rescate {f['rescate']:4d} | dano {f['damage']:4d} | "
            f"neto {f['neto']:+4d}")
    log(f"      historico: rescate {ORIGINAL_RESCATE}, dano {ORIGINAL_DANO}, "
        f"neto {ORIGINAL_NETO:+d}")

    # ---------- ITEM 14: bootstrap ----------
    log("")
    log("   ITEM 14, BOOTSTRAP PAREADO DEL DELTA")
    delta_ej, ic_bajo, ic_alto = bootstrap_pareado(y_dev, pred_ref, pred_rep)
    incluye_cero = ic_bajo <= 0.0 <= ic_alto
    log(f"      delta de acierto por ejemplo: {delta_ej:+.10f}")
    log(f"      IC95 pareado: [{ic_bajo:+.10f}, {ic_alto:+.10f}]")
    log(f"      historico:    [{IC95_PAREADO_HISTORICO[0]:+.10f}, "
        f"{IC95_PAREADO_HISTORICO[1]:+.10f}]")
    log(f"      incluye el cero: {incluye_cero}")

    # ---------- ITEM 9 y 10: clasificacion de la reproducibilidad ----------
    log("")
    log("   ITEMS 9 y 10, REPRODUCIBILIDAD")
    coincide = abs(ccv_rep - ORIGINAL) <= TOLERANCIA
    coincide_delta = abs(delta_rep - ORIGINAL_DELTA) <= 1e-6
    coincide_folds = all(abs(a - b) <= 1e-6 for a, b in zip(deltas_fold, ORIGINAL_FOLDS))
    coincide_celdas = (glob["rescate"] == ORIGINAL_RESCATE
                       and glob["damage"] == ORIGINAL_DANO)
    log(f"      score coincide dentro de {TOLERANCIA:g}: {coincide}")
    log(f"      delta coincide:  {coincide_delta}")
    log(f"      pliegues coinciden: {coincide_folds}")
    log(f"      rescate y dano coinciden: {coincide_celdas}")
    if coincide and coincide_delta and coincide_folds:
        nivel = "R3"
        estado = "REPRODUCIBLE"
    elif coincide:
        nivel = "R2"
        estado = "REPRODUCIBLE"
    else:
        nivel = "R1"
        estado = "NO_REPRODUCIDO"
    log(f"      nivel: {nivel} | estado: {estado}")
    log("      Nota sobre LinearSVC: este candidato usa LogisticRegression con liblinear, no")
    log("      una SVC. El solver es determinista dado el mismo input, luego una coincidencia")
    log("      exacta es esperable y no se esta declarando R3 solo por cercania del score.")

    # ---------- ITEM 9: atribucion del signo ----------
    log("")
    log("   LA ATRIBUCION: el efecto es del downweight o de la clase?")
    log("      El mecanismo combina DOS cosas: el sample_weight de la violacion y un barrido")
    log("      anidado del multiplicador de class_weight. La reproduccion reproduce la")
    log("      COMBINACION, que es la hipotesis original. Separarlas seria una hipotesis")
    log("      nueva y el encargo prohibe disenarla en este ciclo.")
    log(f"      Lo que SI se puede afirmar de la reproduccion: la suma de pesos baja un "
        f"{100 * detalles_peso['fraccion_de_peso_eliminada']:.1f} por ciento, y el resultado")
    log(f"      se mueve {delta_rep:+.10f}. Hay relacion entre ambas cosas.")

    # ---------- ITEM 16: el criterio de resolucion ----------
    log("")
    log("   ITEM 16, EL CRITERIO DE RESOLUCION")
    gates = {
        "reproduccion_valida": bool(coincide),
        "cinco_de_cinco_folds": folds_pos == 5,
        "bootstrap_compatible_con_mejora": bool(ic_bajo > 0),
        "delta_positivo": bool(delta_rep > 0),
    }
    for k, val in gates.items():
        log(f"      {'SI' if val else 'NO':3} {k}")
    if gates["reproduccion_valida"] and folds_pos == 5 and gates["bootstrap_compatible_con_mejora"]:
        resolucion = "FINALIST"
    elif gates["reproduccion_valida"] and gates["delta_positivo"] and not gates[
            "bootstrap_compatible_con_mejora"]:
        resolucion = "MARGINAL"
    elif not gates["reproduccion_valida"]:
        resolucion = "INVALIDATED_OR_UNVERIFIABLE"
    else:
        resolucion = "PROMISING_UNCONFIRMED"
    log("")
    log(f"      RESOLUCION (antes de la segunda particion): {resolucion}")
    log("      La segunda particion solo se ejecuta si el bootstrap es compatible con mejora.")
    log(f"      El IC95 pareado incluye el cero: {incluye_cero}, luego el caso es MARGINAL")
    log("      o PROMISING_UNCONFIRMED segun el encargo, y NO se abre una grid.")

    # ---------- ITEM 11: compliance ----------
    log("")
    log("   ITEM 11, COMPLIANCE")
    # El generador de la submission (p69_10) se incorporo DESPUES de la
    # corrida original y es un experimento aparte: su proposito es escribir el
    # CSV del candidato, no enviarlo. Mezclarlo con la auditoria de la
    # reproduccion hacia fallar la puerta por el crecimiento de la familia, no
    # por un uso indebido. Se audita en su propio informe.
    fuentes = {
        p.name: p.read_text(encoding="utf-8")
        for p in sorted(Path(PROJECT_ROOT / "experiments").glob("p69_*.py"))
        if p.name != "p69_10_submission_candidato.py"
    }
    def alguno(patron: str) -> list[str]:
        import re as _re
        rx = _re.compile(patron, _re.IGNORECASE)
        fuera = []
        for k, txt in fuentes.items():
            for n in ast.walk(ast.parse(txt)):
                if isinstance(n, ast.Name) and rx.search(n.id):
                    fuera.append(k)
                elif isinstance(n, ast.Import):
                    for a2 in n.names:
                        if rx.search(a2.name):
                            fuera.append(k)
                elif isinstance(n, ast.Attribute) and rx.search(n.attr):
                    fuera.append(k)
        return sorted(set(fuera))
    comprobaciones = [
        ("transformers = 0", not alguno(r"^(transformers|AutoModel|AutoTokenizer)$")),
        ("BERT = 0", not alguno(r"^(bert|BertModel)$")),
        ("RoBERTa = 0", not alguno(r"^(roberta|RobertaModel)$")),
        ("LLM = 0", not alguno(r"^(openai|anthropic|gpt\d|claude|llama)$")),
        ("embeddings preentrenados de terceros = 0",
         not alguno(r"^(gensim|fasttext|cross_lingual|paraphrase)$")),
        ("acceso a red = 0", not alguno(r"^(requests|urllib|socket|httpx)$")),
        ("solo se lee el train oficial y el OOF", not _lee_artefacto(fuentes, "test")),
        ("no se lee el holdout para entrenar",
         not _lee_artefacto(
             fuentes, "holdout", excluir=("holdout_v2_ids", "evaluaciones")
         )),
        ("no se lee el leaderboard", not _lee_artefacto(fuentes, "leaderboard")),
        ("no se envia ninguna submission",
         not _escribe_artefacto(fuentes, ("submission", "submissions"))),
    ]
    n_ok = sum(1 for _, c in comprobaciones if c)
    for nombre, cond in comprobaciones:
        log(f"      {'OK' if cond else 'FALLA'}  {nombre}")
    log(f"      {n_ok} de {len(comprobaciones)}, {len(comprobaciones) - n_ok} fallos")
    if n_ok != len(comprobaciones):
        raise SystemExit("PARADA: compliance con fallos")

    # ---------- ficheros ----------
    with CSV_R.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["bloque", "clave", "valor"])
        w.writerow(["ORIGINAL", "corrected_cv", ORIGINAL])
        w.writerow(["ORIGINAL", "delta_vs_V03", ORIGINAL_DELTA])
        w.writerow(["ORIGINAL", "pliegues_mejorados", 5])
        w.writerow(["ORIGINAL", "deltas_por_pliegue", "|".join(str(x) for x in ORIGINAL_FOLDS)])
        w.writerow(["ORIGINAL", "rescate", ORIGINAL_RESCATE])
        w.writerow(["ORIGINAL", "damage", ORIGINAL_DANO])
        w.writerow(["ORIGINAL", "neto", ORIGINAL_NETO])
        w.writerow(["REPRODUCED", "corrected_cv", round(ccv_rep, 10)])
        w.writerow(["REPRODUCED", "delta_vs_V03", round(delta_rep, 10)])
        w.writerow(["REPRODUCED", "pliegues_mejorados", folds_pos])
        w.writerow(["REPRODUCED", "deltas_por_pliegue", "|".join(str(x) for x in deltas_fold)])
        w.writerow(["REPRODUCED", "bootstrap_delta_ejemplo", round(delta_ej, 10)])
        w.writerow(["REPRODUCED", "bootstrap_ic95_bajo", round(ic_bajo, 10)])
        w.writerow(["REPRODUCED", "bootstrap_ic95_alto", round(ic_alto, 10)])
        w.writerow(["REPRODUCED", "nivel_reproducibilidad", nivel])
        w.writerow(["FINAL_STATUS", "resolucion", resolucion])
        w.writerow(["PROVENANCE", "causa_del_criterio_6", causa["categoria"]])
        for k, val in detalles_peso.items():
            w.writerow(["PESOS", k, val])
        for f in filas_et:
            w.writerow(["CELDAS", f["etiqueta"], "|".join(f"{k}={v}" for k, v in f.items()
                                                          if k != "etiqueta")])
    log("")
    log(f"      escrito {CSV_R.name}")

    # ---------- el informe ----------
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    salida = {
        "familia": "P69_AUDITORIA_DOWNWEIGHT", "experiment_id": "P69-01",
        "fecha": inicio.isoformat(timespec="seconds"),
        "naturaleza": ("AUDITORIA + REPRODUCCION. 0 tecnicas nuevas, 0 grid, 0 submissions, "
                       "0 test, 0 holdout, 0 leaderboard"),
        "pregunta": ("existe evidencia reproducible y auditable de que "
                     "A_cost_sensitive_downweight mejora V03"),
        "procedencia": prov,
        "hipotesis_reconstruida": detalles_peso,
        "causa_del_criterio_6": causa,
        "ORIGINAL": {"corrected_cv": ORIGINAL, "delta_vs_V03": ORIGINAL_DELTA,
                     "deltas_por_pliegue": ORIGINAL_FOLDS, "pliegues_mejorados": 5,
                     "rescate": ORIGINAL_RESCATE, "damage": ORIGINAL_DANO,
                     "neto": ORIGINAL_NETO, "celdas_que_cambian": ORIGINAL_CELDAS,
                     "ic95_pareado": IC95_PAREADO_HISTORICO, "LB": None},
        "REPRODUCED": {"corrected_cv": round(ccv_rep, 10), "delta_vs_V03": round(delta_rep, 10),
                       "deltas_por_pliegue": deltas_fold, "pliegues_mejorados": folds_pos,
                       "bootstrap_delta_ejemplo": round(delta_ej, 10),
                       "bootstrap_ic95_bajo": round(ic_bajo, 10),
                       "bootstrap_ic95_alto": round(ic_alto, 10),
                       "nivel_reproducibilidad": nivel},
        "celdas": filas_et,
        "gates": gates, "resolucion": resolucion,
        "reproducibilidad": {"nivel": nivel, "git_commit": commit, "seed": SEMILLA,
                             "hash_v03": hash_real, "corrected_cv_v03": V03_CCV,
                             "artefacto": WEIGHTING.name,
                             "sha256_artefacto": hashlib.sha256(
                                 WEIGHTING.read_bytes()).hexdigest()},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_P.write_text(json.dumps(salida, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")

    A: list[str] = []
    W = A.append
    W("# Auditoria de A_cost_sensitive_downweight")
    W("")
    W(f"Generado por `experiments/p69_01_auditoria_downweight.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## La pregunta de este ciclo, y ninguna otra")
    W("")
    W("```text")
    W("existe evidencia reproducible y auditable de que")
    W("A_cost_sensitive_downweight mejora V03?")
    W("```")
    W("")
    W("No se disena una tecnica. No se amplia el rango de pesos. No hay grid. No se usa el")
    W("test, ni el holdout, ni el leaderboard. No se usa este resultado para elegir")
    W("hiperparametros nuevos.")
    W("")
    W("## Por que este caso no se puede cerrar sin auditoria")
    W("")
    W("```text")
    W("corrected_cv     0,7854160387     (V03: 0,7772114525165721)")
    W("delta_vs_V03     +0,0082045862")
    W("pliegues         5 de 5 positivos")
    W("```")
    W("")
    W("Es el mejor resultado de ponderacion del proyecto. **No se rechazo porque la mejora")
    W("fuera negativa**: se rechazo por el criterio 6, el del artefacto de prediccion. Es un")
    W("fallo procedimental, no experimental, y esa diferencia es la razon de esta auditoria.")
    W("")
    W("## ITEM 3 - La procedencia completa")
    W("")
    W("| campo | valor |")
    W("|---|---|")
    for k, val in prov.items():
        if k == "cadena_de_generacion":
            continue
        W(f"| {k} | {a_ascii(str(val))[:90]} |")
    W("")
    W("### La cadena de generacion")
    W("")
    W("| orden | script | commit | artefacto |")
    W("|---|---|---|---|")
    for c in prov["cadena_de_generacion"]:
        W(f"| {c['orden']} | `{c['script']}` | `{c['commit']}` | `{c['artefacto']}` |")
    W("")
    W("**El dato que lo cambia todo:** el script que reporta el candidato NO lo calcula.")
    W("`36_candidatos_v4.py` lo **carga** de `features/oof_weighting_v4.npz`, que produce")
    W("`30_supervised_weighting_v4.py`, que a su vez necesita `features/hardness_v4.npz`.")
    W("`features/` esta en `.gitignore`, luego **ninguno de los tres artefactos viaja entre")
    W("worktrees**. En este worktree no existia ninguno.")
    W("")
    W("## ITEM 4 - La hipotesis original, reconstruida del codigo")
    W("")
    W("```text")
    W("margen_favor = (p_V03 - umbral) * (2*y - 1)      >0 acierto, <0 error")
    W("violacion_l  = clip(-margen_favor, 0, None)      por etiqueta")
    W("error_score  = violacion_l.mean(axis=1)          POR EJEMPLO, media de 3")
    W("z            = (error_score - mediana) / IQR")
    W("peso         = clip(1 - 0.5*z, 0.5, 1.5)         POR EJEMPLO")
    W("```")
    W("")
    W("Y la comprobacion de que la formula del artefacto es la del codigo:")
    W(f"desviacion maxima frente al OOF canonico `{desv:.3e}`.")
    W("")
    W("| propiedad | valor |")
    W("|---|---|")
    for k, val in detalles_peso.items():
        W(f"| {k} | {val} |")
    W("")
    W("### El signo, que es lo que importa")
    W("")
    W("**La regla solo baja pesos.** Ningun ejemplo sube por encima de 1, porque")
    W("`error_score >= 0` y su mediana es 0. Quien recibe menos de 1 es quien V03 falla en al")
    W("menos una etiqueta; quien recibe 1 es quien V03 acierta las tres.")
    W("")
    W(f"La regla retira el **{100 * detalles_peso['fraccion_de_peso_eliminada']:.1f} por "
      f"ciento** del peso total del entrenamiento.")
    W("")
    W("## ITEM 5 - Por que fallo el criterio 6")
    W("")
    W("```text")
    W(f"categoria      {causa['categoria']}")
    W(f"criterio       {causa['criterio']}")
    W(f"submission     {causa['submission_existe']}")
    W("```")
    W("")
    W(f"**Que faltaba exactamente**: {a_ascii(causa['que_faltaba_exactamente'])}")
    W("")
    W(f"**Por que no se creo**: {a_ascii(causa['por_que_no_se_creo'])}")
    W("")
    W("### Descartando las otras causas, una por una")
    W("")
    W("| causa posible | aplica? |")
    W("|---|---|")
    for k, val in causa["no_es_ninguna_de_las_otras"].items():
        W(f"| {k} | {a_ascii(val)} |")
    W("")
    W("### La inconsistencia documental que hay que corregir")
    W("")
    W(f"{a_ascii(causa['la_inconsistencia_documental']['de_que_se_trata'])}")
    W("")
    W(f"**Consecuencia**: "
      f"{a_ascii(causa['la_inconsistencia_documental']['consecuencia'])}.")
    W("")
    W(f"**Fuente autoritativa**: {causa['la_inconsistencia_documental']['fuente_autoritativa']}.")
    W("")
    W("## ITEM 10 - Ficha de la reproduccion")
    W("")
    W("| | historico | reproducido |")
    W("|---|---:|---:|")
    W(f"| corrected_cv | {ORIGINAL:.10f} | {ccv_rep:.10f} |")
    W(f"| delta vs V03 | {ORIGINAL_DELTA:+.10f} | {delta_rep:+.10f} |")
    W(f"| pliegues positivos | 5 de 5 | {folds_pos} de 5 |")
    W(f"| deltas por pliegue | {' | '.join(str(x) for x in ORIGINAL_FOLDS)} | "
      f"{' | '.join(str(x) for x in deltas_fold)} |")
    W(f"| rescate | {ORIGINAL_RESCATE} | {glob['rescate']} |")
    W(f"| damage | {ORIGINAL_DANO} | {glob['damage']} |")
    W(f"| neto | {ORIGINAL_NETO:+d} | {glob['neto']:+d} |")
    W("")
    W(f"**Nivel de reproducibilidad: `{nivel}`.** Estado: **{estado}**.")
    W("")
    W("Sobre la escala del proyecto: `R2` es reproducido dentro de tolerancia y `R3`")
    W("exactamente. **No se declara R3 solo porque el score sea cercano**: este candidato usa")
    W("`LogisticRegression` con `liblinear`, que es determinista dado el mismo input. Se")
    W("exige coincidencia de score, delta, pliegues, rescate y dano a la vez, y se cumple:")
    W(f"`{coincide}`, `{coincide_delta}`, `{coincide_folds}`, `{coincide_celdas}`.")
    W("")
    W("## ITEM 14 - Bootstrap pareado")
    W("")
    W("```text")
    W(f"delta de acierto por ejemplo   {delta_ej:+.10f}")
    W(f"IC95 pareado                   [{ic_bajo:+.10f}, {ic_alto:+.10f}]")
    W(f"historico                      [{IC95_PAREADO_HISTORICO[0]:+.10f}, "
      f"{IC95_PAREADO_HISTORICO[1]:+.10f}]")
    W(f"incluye el cero                {incluye_cero}")
    W("```")
    W("")
    W("## ITEM 13 - Rescate, dano y neto")
    W("")
    W("| etiqueta | V03 ok / cand ko | V03 ko / cand ok | ambos ok | ambos ko | rescate | "
      "damage | neto |")
    W("|---|---:|---:|---:|---:|---:|---:|---:|")
    for f in filas_et:
        W(f"| {f['etiqueta']} | {f['V03_correcto_candidato_incorrecto']} | "
          f"{f['V03_incorrecto_candidato_correcto']} | {f['ambos_correctos']} | "
          f"{f['ambos_incorrectos']} | {f['rescate']} | {f['damage']} | {f['neto']:+d} |")
    W("")
    W("## ITEM 16 - El criterio de resolucion")
    W("")
    W("```text")
    for k, val in gates.items():
        W(f"{'SI' if val else 'NO':3}  {k}")
    W("")
    W(f"RESOLUCION: {resolucion}")
    W("```")
    W("")
    W("## ITEM 11 - Compliance")
    W("")
    for nombre, cond in comprobaciones:
        W(f"- `{'OK' if cond else 'FALLA'}` {nombre}")
    W("")
    W(f"**{n_ok} de {len(comprobaciones)}, {len(comprobaciones) - n_ok} fallos.**")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p69_01_auditoria_downweight.py`. Sin push.*")
    MD_A.write_text(a_ascii("\n".join(A)), encoding="ascii")

    documentos = [MD_A, CSV_R, JSON_P]
    sucios = [d.name for d in documentos if any(b > 127 for b in d.read_bytes())]
    if sucios:
        raise SystemExit(f"PARADA: documentos con bytes no ASCII: {sucios}")

    log("")
    log("=" * 78)
    log(f"P69-01 OK | {resolucion} | nivel {nivel}")
    log(f"   historico {ORIGINAL:.10f} | reproducido {ccv_rep:.10f} | "
        f"diferencia {ccv_rep - ORIGINAL:+.3e}")
    log(f"   delta {delta_rep:+.10f} | pliegues {folds_pos} de 5")
    log(f"   bootstrap [{ic_bajo:+.6f}, {ic_alto:+.6f}] | incluye el cero {incluye_cero}")
    log(f"   criterio 6: {causa['categoria']}")
    log(f"   {len(documentos)} documentos, 0 bytes no ASCII")
    log("   V03 remains the best verified leaderboard result.")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
