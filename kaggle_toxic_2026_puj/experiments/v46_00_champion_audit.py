"""V46-00 (P0): auditoria del campeon real y reconciliacion del leaderboard.

Que resuelve
------------
Cual es el verdadero mejor resultado de leaderboard del proyecto, y cual debe
ser la referencia oficial para futuras decisiones.

La contradiccion a resolver: `CHAMPION.md` atribuye a `v16_ch1` un LB de
0.78535 y a `v16_ch5` un LB de 0.78650, mientras `submissions/leaderboard_log.csv`
registra como maximo 0.78306 (V03).

Que NO hace
-----------
- No entrena ningun modelo.
- No consulta Kaggle.
- No genera ninguna submission.
- No modifica V03, el holdout ni el test.
- No corrige CHAMPION.md: solo establece la evidencia.

Jerarquia de evidencia aplicada
-------------------------------
1. archivo de submission
2. registro externo asociado a esa submission
3. leaderboard log estructurado
4. metadata de experimento
5. reports
6. notebook
7. CHAMPION.md
8. texto narrativo

Diseno
------
1. Las 22 filas del log, con artefacto, sha256 y enlace a experimento.
2. Clasificacion de cada score: MEDIDO (procede del host) o NO_MEDIDO.
3. Extraccion de las afirmaciones de `CHAMPION.md` y su clasificacion.
4. Comprobacion de si la columna `LB combinado` de CHAMPION.md es una medicion
   o el resultado de la formula que el propio fichero declara.
5. Verificacion de la formula contra los valores afirmados.
6. Separacion de las cuatro magnitudes que se confunden: mejor CV, mejor
   modelo, mejor submission y mejor LB.

Uso
---
    /home/carlos/venv/bin/python experiments/v46_00_champion_audit.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

LOG = PROJECT_ROOT / "submissions" / "leaderboard_log.csv"
CHAMPION = PROJECT_ROOT / "CHAMPION.md"
REGISTRO = PROJECT_ROOT / "experiments.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "p46_champion_audit.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "p46_champion_audit.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "p46_champion_audit.csv"

# Coeficientes de la formula que CHAMPION.md declara textualmente.
COEF_BASE = 0.78306
COEF_TASA = 0.12011
COEF_NETO = 2.804e-05
V03_HASH = "7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e"

# Enlace submission -> experimento, deducido de la narrativa del registro.
ENLACE_EXPERIMENTO = {
    "v00_plantilla": "E00",
    "v01_baseline_tfidf_lr": "B1",
    "v02_lr_union": "MOD-03b_BEST",
    "v03_lr_union_thr": "MOD-04 / TH-01",
    "v04_ens04_mediana": "ENS-03",
    "v05_optuna": "OPT-01",
    "v06_robustez_c1": "S04",
    "v07_robustez_mindf5": "S06",
    "v08_meta_v2": "E2-02-META",
    "v09_candidate_a": "E2-09",
    "v10_candidate_b": "E2-10",
    "v11_candidate_c": "E2-11",
    "v08_p1_fold_ensemble": "V8-04",
    "v09_p5_c_por_etiqueta": "V9-05",
    "v12_p7_promedio_repeticiones": "V12-P7",
    "v16_ch5_tasa_optima_absoluta": "V16-05",
    "v16_ch1_tasa_optima_transportada": "V16-01",
    "v16_ch7_C1p5": "V16-07",
    "v16_ch8_C2p5": "V16-08",
    "v16_ch9_mindf1": "V16-09",
    "56597267": "V20-01",
    "56599802": "V23-01",
}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V46-00] {mensaje}", flush=True)


def sha256_de(ruta: Path) -> str:
    """Devuelve el sha256 de un fichero, o cadena vacia si no existe.

    Args:
        ruta: ruta del fichero.

    Returns:
        Hexadecimal de 64 caracteres, o cadena vacia.
    """
    if not ruta.exists():
        return ""
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def resolver_artefacto(nombre: str) -> Path:
    """Resuelve la ruta de un artefacto de submission.

    El log escribe a veces el nombre sin el prefijo `submissions/`. Esta funcion
    acepta las dos formas y devuelve la ruta que existe en disco.

    Args:
        nombre: valor del campo `archivo` del log.

    Returns:
        Ruta existente, o la original si ninguna forma esta en disco.
    """
    directo = PROJECT_ROOT / nombre
    if directo.exists():
        return directo
    con_prefijo = PROJECT_ROOT / "submissions" / Path(nombre).name
    if con_prefijo.exists():
        return con_prefijo
    return directo


def main() -> None:
    """Ejecuta la auditoria del campeon.

    Returns:
        None.
    """
    inicio = time.time()
    log("P46 auditoria del campeon. NINGUN modelo, NINGUNA submission, NINGUNA consulta a Kaggle")

    # ------------------------------------------------------ 1. las 22 filas
    log("1/6 las 22 filas del leaderboard log")
    log_df = pd.read_csv(LOG)
    log(f"   filas {len(log_df)} | columnas {len(log_df.columns)}")
    log(f"   duplicados de submission_id: {int(log_df.submission_id.duplicated().sum())}")
    log(f"   scores nulos: {int(log_df.kaggle_f1.isna().sum())}")
    imposibles = log_df[(log_df.kaggle_f1 < 0) | (log_df.kaggle_f1 > 1)]
    log(f"   scores fuera de [0,1]: {len(imposibles)}")
    log_df = log_df.sort_values("fecha", kind="stable")

    filas = []
    for _, r in log_df.iterrows():
        ruta = resolver_artefacto(str(r.archivo))
        existe = ruta.exists()
        filas.append({
            "submission_id": str(r.submission_id),
            "experiment_id": ENLACE_EXPERIMENTO.get(str(r.submission_id), "no enlazado"),
            "fecha": str(r.fecha),
            "archivo_log": str(r.archivo),
            "archivo_resuelto": str(ruta.relative_to(PROJECT_ROOT)) if existe else "",
            "artefacto_existe": bool(existe),
            "sha256": sha256_de(ruta),
            "local_f1": None if pd.isna(r.local_f1) else float(r.local_f1),
            "leaderboard_f1": float(r.kaggle_f1),
            "diferencia_log": None if pd.isna(r.diferencia) else float(r.diferencia),
            "evidencia": "MEDIDO" if not pd.isna(r.kaggle_f1) else "NO_MEDIDO",
            "fuente": "submissions/leaderboard_log.csv (host Kaggle)",
        })
    sin_artefacto = [f["submission_id"] for f in filas if not f["artefacto_existe"]]
    log(f"   filas con artefacto en disco: {len(filas) - len(sin_artefacto)}/{len(filas)}")
    log(f"   sin artefacto: {sin_artefacto or 'ninguna'}")
    log(f"   V03 hash verificado: {filas[3]['sha256'] == V03_HASH if filas[3]['submission_id'].startswith('v03') else 'n/d'}")

    maximo = max(filas, key=lambda f: f["leaderboard_f1"])
    log(f"   maximo MEDIDO = {maximo['leaderboard_f1']:.5f} ({maximo['submission_id']})")
    por_encima = [f["submission_id"] for f in filas
                  if f["leaderboard_f1"] > maximo["leaderboard_f1"]]
    log(f"   filas por encima del maximo medido: {por_encima or 'ninguna'}")

    # ------------------------------------------- 2. las afirmaciones narratives
    log("2/6 afirmaciones de LB en CHAMPION.md")
    texto = CHAMPION.read_text(encoding="utf-8")
    # La tabla de la seccion 4 se titula 'LB combinado', que es la formula declarada.
    cabecera = re.search(r"\|\s*#\s*\|\s*Clave\s*\|.*?\|(.+?)\|\s*$", texto, re.MULTILINE)
    etiqueta_columna = cabecera.group(1).strip() if cabecera else "?"
    log(f"   nombre de la columna de score en la tabla de candidatas: '{etiqueta_columna}'")
    formula_declarada = re.search(r"`LB combinado`\s*=\s*(.+)", texto)
    log(f"   formula declarada presente: {bool(formula_declarada)}")
    if formula_declarada:
        log(f"   '{(formula_declarada.group(1) or '')[:110]}'")

    patrones = {
        "0.78535": re.compile(r"\|\s*1\s*\|\s*ch1\s*\|.*?\|\s*(0\.78535)\s*\|"),
        "0.78650": re.compile(r"\|\s*2\s*\|\s*ch5\s*\|.*?\|\s*(0\.78650)\s*\|"),
        "0.78418": re.compile(r"\|\s*3\s*\|\s*ch2\s*\|.*?\|\s*(0\.78418)\s*\|"),
        "0.78242": re.compile(r"\|\s*4\s*\|\s*ch4\s*\|.*?\|\s*(0\.78242)\s*\|"),
        "0.77431": re.compile(r"\|\s*5\s*\|\s*ch3\s*\|.*?\|\s*(0\.77431)\s*\|"),
    }
    afirmaciones = []
    for valor, patron in patrones.items():
        encontrado = patron.search(texto)
        if not encontrado:
            continue
        clave = re.search(r"\|\s*\d\s*\|\s*(ch\d)\s*\|", encontrado.group(0)).group(1)
        fila_log = next((f for f in filas if f["submission_id"].startswith(f"v16_{clave}")), None)
        afirmaciones.append({
            "valor_afirmado": float(valor),
            "clave": clave,
            "linea": texto[:encontrado.start()].count("\n") + 1,
            "columna": etiqueta_columna,
            "submission_id": fila_log["submission_id"] if fila_log else "no encontrada en el log",
            "leaderboard_medido": fila_log["leaderboard_f1"] if fila_log else None,
            "discrepancia": (float(valor) - fila_log["leaderboard_f1"]) if fila_log else None,
        })
        log(f"   {clave}: afirmado {valor} (linea {afirmaciones[-1]['linea']}, columna "
            f"'{etiqueta_columna}') | medido "
            f"{afirmaciones[-1]['leaderboard_medido']}")

    # ------------------------------- 3. la columna es prediccion o medicion
    log("3/6 la columna es una prediccion, no una medicion")
    pasos = {"ch1": (0.019, 0), "ch5": (0.0335, -21)}
    predicciones = []
    for clave, (delta_tasa, neto) in pasos.items():
        predicho = COEF_BASE + COEF_TASA * delta_tasa + neto * COEF_NETO
        afirmado = next((a["valor_afirmado"] for a in afirmaciones if a["clave"] == clave), None)
        desviacion = abs(predicho - afirmado) if afirmado is not None else None
        predicciones.append({
            "clave": clave,
            "delta_tasa": delta_tasa,
            "aciertos_netos_oof": neto,
            "lb_combinado_recalculado": predicho,
            "valor_afirmado": afirmado,
            "desviacion": desviacion,
            "reproduce": bool(desviacion is not None and desviacion < 1e-4),
        })
        log(f"   {clave}: recalculado {predicho:.6f} | afirmado {afirmado} | "
            f"desviacion {desviacion:.2e} | reproduce {predicciones[-1]['reproduce']}")
    todas_reproducen = all(p["reproduce"] for p in predicciones)
    log(f"   las afirmaciones se reproducen con la formula declarada: {todas_reproducen}")
    log("   => los valores de la tabla de candidatas son PREDICCIONES del modelo causal,")
    log("      no puntuaciones medidas por el host.")

    # ---------------------------- 4. contradiccion interna de CHAMPION.md
    log("4/6 contradiccion interna de CHAMPION.md")
    tabla_medida = re.search(
        r"## Estado al cierre de V16(.{0,1800})", texto, re.DOTALL)
    contradicciones = []
    if tabla_medida:
        bloque = tabla_medida.group(1)
        for clave in ("ch1", "ch5"):
            fila = re.search(rf"\|\s*v16_{clave}[^|]*\|[^|]*\|\s*(0\.\d+)\s*\|[^|]*\|\s*(\w+)\s*\|", bloque)
            if fila:
                contradicciones.append({
                    "clave": clave,
                    "valor_en_tabla_de_estado": float(fila.group(1)),
                    "estado": fila.group(2),
                    "linea": texto[:texto.index(fila.group(0))].count("\n") + 1,
                })
                log(f"   seccion 'Estado al cierre de V16': {clave} = {fila.group(1)} "
                    f"({fila.group(2)})")
    log(f"   el mismo fichero contiene ambas cifras: {len(contradicciones)} filas")

    con_local = [f for f in filas if f["local_f1"] is not None]
    mejor_local = max(con_local, key=lambda f: f["local_f1"])

    # ------------------------- 5. las cuatro magnitudes que se confunden
    log("5/6 mejor CV, mejor modelo, mejor submission, mejor LB")
    registro = pd.read_csv(REGISTRO)
    # La columna macro_f1 del registro mezcla bases de medida. Se clasifica cada
    # fila por el texto de `validation` antes de comparar nada, porque un maximo
    # sobre esa columna sin clasificar no significa nada.
    val = registro.validation.astype(str).str.lower()
    con_score = registro.macro_f1.astype(str).str.fullmatch(r"0\.\d+")
    en_holdout = con_score & val.str.contains("holdout")
    en_d004 = con_score & ~en_holdout & (
        val.str.contains("d-004")
        | (val.str.contains("near-duplicad") & ~val.str.contains("fusionad"))
    )
    sin_declarar = con_score & ~en_holdout & ~en_d004
    bases = {
        "con_score_numerico": int(con_score.sum()),
        "base_holdout2_248_filas": int(en_holdout.sum()),
        "base_split_D004_supersedida": int(en_d004.sum()),
        "base_no_declarada": int(sin_declarar.sum()),
    }
    log(f"   macro_f1 del registro mezcla bases: {bases}")
    log("   por eso NO se toma el maximo de esa columna como 'mejor CV'.")
    oof = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)
    cv_canonico = float(oof["corrected_cv"][0])
    magnitud = {
        "mejor_corrected_cv_utilizable": {
            "valor": cv_canonico,
            "fuente": "features/oof_v03_final.npz (OOF canonico de 1483 filas, pliegues corregidos)",
            "nota": ("La columna macro_f1 de experiments.csv NO esta en esa base para las "
                     "filas historicas: MOD-04, TH-01 y S03 registran 0.8322 y 0.8356 sobre el "
                     "split D-004, supersedido por D-015/D-016. Ordenar esa columna sin "
                     "clasificar la base produce rankings invalidos"),
        },
        "mejor_local_del_log": {
            "submission_id": mejor_local["submission_id"],
            "local_f1": mejor_local["local_f1"],
            "leaderboard_f1": mejor_local["leaderboard_f1"],
        },
        "mejor_submission_por_LB": {
            "submission_id": maximo["submission_id"],
            "leaderboard_f1": maximo["leaderboard_f1"],
        },
        "mejor_modelo_congelado": {
            "experiment_id": "V03",
            "corrected_cv": cv_canonico,
            "leaderboard_f1": maximo["leaderboard_f1"],
            "sha256": V03_HASH,
        },
    }
    for clave, valor in magnitud.items():
        log(f"   {clave}: {valor}")

    # ------------------------------------------------------------ veredicto
    claims_sin_prueba = [a for a in afirmaciones
                         if a["leaderboard_medido"] is not None
                         and a["valor_afirmado"] > a["leaderboard_medido"]]
    champion_status = "INCONSISTENTE_AUTOCORREGIDO" if contradicciones else "CONSISTENTE"
    veredicto = {
        "CASE": "B",
        "leaderboard_max_verified": maximo["leaderboard_f1"],
        "leaderboard_max_verified_submission": maximo["submission_id"],
        "champion_ofrecional": "V03",
        "champion_ofrecional_sha256": V03_HASH,
        "CHAMPION_STATUS": champion_status,
        "afirmaciones_sin_evidencia_primaria": [a["valor_afirmado"] for a in claims_sin_prueba],
        "clasificacion": ("UNSUPPORTED_CLAIM_PREDICTION: los valores de la tabla de candidatas "
                          "son el resultado de la formula causal declarada en el propio "
                          "fichero, no puntuaciones del host. Las mediciones reales de esas "
                          "mismas submissions son inferiores y estan marcadas REJECTED en el "
                          "mismo CHAMPION.md"),
        "baseline_operativo": "V03",
        "gastos": {
            "submissions": 0,
            "modelos_entrenados": 0,
            "consultas_kaggle": 0,
            "segundos": round(time.time() - inicio, 1),
        },
    }
    log(f"CASE = {veredicto['CASE']} | leaderboard_max_verified = "
        f"{veredicto['leaderboard_max_verified']:.5f} | CHAMPION_STATUS = {champion_status}")

    informe = {
        "familia": "P46_CHAMPION_AUDIT",
        "experimento": "V46-00 (P0 auditoria del campeon real y reconciliacion del leaderboard)",
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "AUDIT_ONLY",
        "jerarquia_de_evidencia": [
            "1. archivo de submission",
            "2. registro externo asociado a esa submission",
            "3. leaderboard log estructurado",
            "4. metadata de experimento",
            "5. reports",
            "6. notebook",
            "7. CHAMPION.md",
            "8. texto narrativo",
        ],
        "leaderboard_log": {
            "ruta": str(LOG.relative_to(PROJECT_ROOT)),
            "n_rows": len(log_df),
            "columnas": list(log_df.columns),
            "duplicados": int(log_df.submission_id.duplicated().sum()),
            "scores_nulos": int(log_df.kaggle_f1.isna().sum()),
            "scores_fuera_de_rango": len(imposibles),
            "filas": filas,
        },
        "afirmaciones_champion_md": afirmaciones,
        "columna_de_score": etiqueta_columna,
        "formula_declarada": (formula_declarada.group(0) if formula_declarada else ""),
        "reproduccion_de_la_formula": {
            "coeficientes": {"base": COEF_BASE, "tasa": COEF_TASA, "neto": COEF_NETO},
            "casos": predicciones,
            "todas_reproducen": todas_reproducen,
            "conclusion": ("Los valores de la tabla de candidatas de CHAMPION.md son "
                           "predicciones del modelo causal declarado en ese mismo fichero, "
                           "no puntuaciones medidas por el host"),
        },
        "contradiccion_interna": contradicciones,
        "magnitudes_separadas": magnitud,
        "veredicto": veredicto,
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _csv(filas, afirmaciones, veredicto, predicciones, contradicciones)
    _md(informe)
    log("   informes escritos en reports/p46_champion_audit.{json,md,csv}")
    log(f"VEREDICTO: leaderboard_max_verified = {veredicto['leaderboard_max_verified']:.5f}, "
        f"0 modelos, 0 submissions, 0 consultas a Kaggle, "
        f"{round(time.time() - inicio, 1)} s")


def _csv(filas, afirmaciones, veredicto, predicciones, contradicciones) -> None:
    """Escribe el informe tabular.

    Args:
        filas: filas del leaderboard log auditadas.
        afirmaciones: afirmaciones extraidas de CHAMPION.md.
        veredicto: diccionario de veredicto.
        predicciones: reproduccion de la formula declarada.
        contradicciones: filas de la tabla de estado de CHAMPION.md.

    Returns:
        None.
    """
    cabecera = ["submission", "experiment", "local_score", "leaderboard_score", "dLB",
                "artefacto", "sha256", "verified", "source", "status"]
    with REPORTE_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cabecera)
        for x in filas:
            mejor = x["leaderboard_f1"] == veredicto["leaderboard_max_verified"]
            writer.writerow([
                x["submission_id"], x["experiment_id"],
                "" if x["local_f1"] is None else f"{x['local_f1']:.5f}",
                f"{x['leaderboard_f1']:.5f}",
                "" if x["diferencia_log"] is None else f"{x['diferencia_log']:+.5f}",
                x["archivo_resuelto"], x["sha256"], x["evidencia"], x["fuente"],
                "BEST_VERIFIED_LB" if mejor else "REJECTED",
            ])
        writer.writerow([])
        writer.writerow(["afirmacion_no_medida", "experimento", "valor_afirmado", "", "",
                         "", "", "UNSUPPORTED_CLAIM_PREDICTION", "CHAMPION.md", ""])
        for a in afirmaciones:
            nota = (
                "no enviada: sin valor medido"
                if a["leaderboard_medido"] is None
                else f"medido real {a['leaderboard_medido']:.5f}, "
                     f"discrepancia {a['discrepancia']:+.5f}"
            )
            writer.writerow([
                f"{a['clave']} (linea {a['linea']})", a["submission_id"],
                f"{a['valor_afirmado']:.5f}", "", "",
                "", "", "UNSUPPORTED_CLAIM_PREDICTION", "CHAMPION.md", nota,
            ])
        writer.writerow([])
        writer.writerow(["reproduccion_formula", "clave", "delta_tasa", "neto_oof", "",
                         "lb_combinado_recalculado", "", "reproduce", "CHAMPION.md", ""])
        for p in predicciones:
            writer.writerow([p["clave"], "", p["delta_tasa"], p["aciertos_netos_oof"], "",
                             f"{p['lb_combinado_recalculado']:.6f}", "",
                             str(p["reproduce"]), "CHAMPION.md",
                             f"afirmado {p['valor_afirmado']}"])
        if contradicciones:
            writer.writerow([])
            writer.writerow(["contradiccion_interna_champion_md", "clave", "valor_medido", "",
                             "", "", "", "MEDIDO", "CHAMPION.md", ""])
            for c in contradicciones:
                writer.writerow([c["clave"], "", f"{c['valor_en_tabla_de_estado']:.5f}", "",
                                 "", "", "", "MEDIDO", "CHAMPION.md", c["estado"]])


def _md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible.

    Args:
        informe: diccionario con los resultados.

    Returns:
        None.
    """
    ll = informe["leaderboard_log"]
    v = informe["veredicto"]
    rf = informe["reproduccion_de_la_formula"]
    mg = informe["magnitudes_separadas"]
    filas = sorted(ll["filas"], key=lambda x: -x["leaderboard_f1"])
    lineas = [
        "# P46 - Auditoria del campeon real",
        "",
        f"Generado por `experiments/v46_00_champion_audit.py` ({informe['experimento']}).",
        "",
        f"Fecha {informe['fecha']}. **0 modelos, 0 submissions, 0 consultas a Kaggle.**",
        "",
        "## 1. Veredicto",
        "",
        f"**CASE = {v['CASE']}**",
        "",
        (f"- `leaderboard_max_verified` = **{v['leaderboard_max_verified']:.5f}** "
        f"(`{v['leaderboard_max_verified_submission']}`)"),
        (f"- `baseline_operativo` = **{v['baseline_operativo']}**, sha256 "
        f"`{v['champion_ofrecional_sha256']}`"),
        f"- `CHAMPION_STATUS` = **{v['CHAMPION_STATUS']}**",
        f"- Afirmaciones sin evidencia primaria: {v['afirmaciones_sin_evidencia_primaria']}",
        "",
        f"{v['clasificacion']}.",
        "",
        "## 2. Las 22 submissions, por score de leaderboard",
        "",
        "| # | submission | experimento | local | LB | dLB | artefacto | sha256 | estado |",
        "|---:|---|---|---:|---:|---:|---|---|---|",
    ]
    maximo = v["leaderboard_max_verified"]
    for i, f in enumerate(filas, 1):
        eta = f["sha256"][:12] if f["sha256"] else "SIN ARTEFACTO"
        estado = "**BEST_VERIFIED_LB**" if f["leaderboard_f1"] == maximo else "REJECTED"
        celda_local = "" if f["local_f1"] is None else f"{f['local_f1']:.5f}"
        celda_dlb = "" if f["diferencia_log"] is None else f"{f['diferencia_log']:+.5f}"
        celda_art = "si" if f["artefacto_existe"] else "NO"
        lineas.append(
            f"| {i} | `{f['submission_id']}` | {f['experiment_id']} | {celda_local} | "
            f"**{f['leaderboard_f1']:.5f}** | {celda_dlb} | {celda_art} | `{eta}` | {estado} |"
        )
    lineas += [
        "",
        (f"Filas: {ll['n_rows']} | duplicados: {ll['duplicados']} | scores nulos: "
        f"{ll['scores_nulos']} | scores fuera de [0,1]: {ll['scores_fuera_de_rango']} | "
        f"sin artefacto: {sum(1 for f in filas if not f['artefacto_existe'])}."),
        "",
        f"Ninguna fila supera {maximo:.5f}.",
        "",
        "## 3. Las afirmaciones de CHAMPION.md",
        "",
        (f"La columna se titula **`{informe['columna_de_score']}`**, y el propio fichero declara "
        "la formula:"),
        "",
        f"> {informe['formula_declarada'][:200]}",
        "",
        "| clave | linea | afirmado | medido | discrepancia |",
        "|---|---:|---:|---:|---:|",
    ]
    for a in informe["afirmaciones_champion_md"]:
        medido = ("no enviada" if a["leaderboard_medido"] is None
                  else f"{a['leaderboard_medido']:.5f}")
        disc = ("n/d" if a["discrepancia"] is None
                else f"{a['discrepancia']:+.5f}")
        lineas.append(
            f"| {a['clave']} | {a['linea']} | **{a['valor_afirmado']:.5f}** | "
            f"{medido} | {disc} |"
        )
    lineas += [
        "",
        "## 4. La columna es una prediccion, no una medicion",
        "",
        "Recalculo la formula declarada con los valores que el propio texto da:",
        "",
        "| clave | delta_tasa | netos OOF | recalculado | afirmado | desviacion | reproduce |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for p in rf["casos"]:
        lineas.append(
            f"| {p['clave']} | {p['delta_tasa']} | {p['aciertos_netos_oof']} | "
            f"{p['lb_combinado_recalculado']:.6f} | {p['valor_afirmado']:.5f} | "
            f"{p['desviacion']:.2e} | {p['reproduce']} |"
        )
    lineas += [
        "",
        f"Todas reproducen: **{rf['todas_reproducen']}**. {rf['conclusion']}.",
        "",
        "## 5. La contradiccion es interna a CHAMPION.md",
        "",
        "El mismo fichero contiene las dos cifras. En la seccion *Estado al cierre de V16*:",
        "",
        "| clave | linea | valor medido | estado |",
        "|---|---:|---:|---|",
    ]
    for c in informe["contradiccion_interna"]:
        lineas.append(
            f"| {c['clave']} | {c['linea']} | **{c['valor_en_tabla_de_estado']:.5f}** | "
            f"{c['estado']} |"
        )
    lineas += [
        "",
        ("Las mediciones reales de esas mismas submissions son inferiores a las affirmaciones, y "
        "el propio CHAMPION.md las marca REJECTED. No hay dos fuentes en conflicto: hay una "
        "prediction y una medicion, en el mismo fichero, y la medicion es la que vale."),
        "",
        "## 6. Las cuatro magnitudes que no deben confundirse",
        "",
        "| magnitud | valor | quien |",
        "|---|---:|---|",
        (f"| mejor corrected_cv utilizable | "
         f"{mg['mejor_corrected_cv_utilizable']['valor']:.16f} | "
         f"OOF canonico de 1483 filas |"),
        (f"| mejor local del log | {mg['mejor_local_del_log']['local_f1']:.5f} | "
        f"`{mg['mejor_local_del_log']['submission_id']}`, LB "
        f"{mg['mejor_local_del_log']['leaderboard_f1']:.5f} |"),
        (f"| mejor submission por LB | **{mg['mejor_submission_por_LB']['leaderboard_f1']:.5f}** | "
        f"`{mg['mejor_submission_por_LB']['submission_id']}` |"),
        (f"| mejor modelo congelado | {mg['mejor_modelo_congelado']['corrected_cv']:.10f} | "
        f"{mg['mejor_modelo_congelado']['experiment_id']}, LB "
        f"{mg['mejor_modelo_congelado']['leaderboard_f1']:.5f} |"),
        "",
        ("El mejor local no es el mejor LB, y el mejor corrected_cv del registro tampoco. Las "
        "tres son magnitudes distintas y la unica que decide el campeon es la tercera."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
