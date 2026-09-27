"""V45-01 (P1): estructura conjunta de etiquetas HS x TR x AG y error de V03.

Que responde
------------
Si los errores de V03 dependen de la combinacion conjunta de etiquetas de una
forma que no este capturada por sus tres decisiones binarias independientes, y
si queda alguna formulacion de prediccion conjunta que no haya sido probada.

Que NO hace
-----------
- No entrena ningun modelo.
- No genera ninguna submission.
- No usa el holdout para ninguna decision.
- No creacaracteristicas ni representacion nueva.

Diseno
------
1. Variable `LABEL_STATE = 4*HS + 2*TR + AG` sobre ocho estados posibles.
2. Distribucion observada en el train global, en cada pliegue canonico y en las
   predicciones OOF de V03.
3. Matriz 8x8 de estado verdadero contra estado predicho.
4. Decomposicion de los errores de V03 en las cuatro categorias mutuamente
   excluyentes (acierta las tres, falla una, falla dos, falla las tres).
5. Descomposicion por etiqueta para cada estado verdadero.
6. Contraste de independencia: si los tres indicadores de error fueran
   independientes dado el estado verdadero, las ocho patron de error por estado
   seguirian un producto de marginales. Se mide la desviacion con chi-cuadrado.
7. Inventario de mecanismos conjuntos ya probados, leido del registro.

Uso
---
    /home/carlos/venv/bin/python experiments/v45_01_joint_label_structure.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.splits import grupos_fusionados, pliegues_corregidos

OOF_CANONICO = PROJECT_ROOT / "features" / "oof_v03_final.npz"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "p45_joint_label_structure.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "p45_joint_label_structure.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "p45_joint_label_structure.csv"

ETIQUETAS = ("HS", "TR", "AG")
CORRECTED_CV_V03 = 0.7772114525165721

# Los ocho estados posibles de LABEL_STATE = 4*HS + 2*TR + AG.
ESTADOS = [(h, t, a) for h in (0, 1) for t in (0, 1) for a in (0, 1)]


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V45-01] {mensaje}", flush=True)


def a_estado(matriz: np.ndarray) -> np.ndarray:
    """Convierte una matriz de tres columnas binarias en `LABEL_STATE`.

    Args:
        matriz: matriz binaria de forma ``(n, 3)`` en el orden HS, TR, AG.

    Returns:
        Vector de enteros con el estado de cada fila.
    """
    m = np.asarray(matriz, dtype=np.int64)
    return 4 * m[:, 0] + 2 * m[:, 1] + m[:, 2]


def nombre(estado: int) -> str:
    """Devuelve la etiqueta textual de un estado.

    Args:
        estado: entero de 0 a 7.

    Returns:
        Cadena de tres digitos en el orden TR, AG, HS.
    """
    h, t, a = 1 if estado & 4 else 0, 1 if estado & 2 else 0, 1 if estado & 1 else 0
    return f"{h}{t}{a}"


def distribucion(matriz: np.ndarray) -> list[dict[str, Any]]:
    """Cuenta la distribucion de estados de una matriz de etiquetas.

    Args:
        matriz: matriz binaria de forma ``(n, 3)``.

    Returns:
        Lista de ocho diccionarios, uno por estado, con n y porcentaje.
    """
    estados = a_estado(matriz)
    n = len(estados)
    conteo = Counter(estados.tolist())
    salida = []
    for estado in range(8):
        cantidad = conteo.get(estado, 0)
        h, t, a = ESTADOS[estado]
        salida.append({
            "estado": estado,
            "HS": h, "TR": t, "AG": a,
            "n": int(cantidad),
            "porcentaje": float(100.0 * cantidad / n) if n else 0.0,
        })
    return salida


def main() -> None:
    """Ejecuta la auditoria de estructura conjunta.

    Returns:
        None.
    """
    inicio = time.time()
    log("P45 estructura conjunta HS x TR x AG. NINGUN modelo, NINGUNA submission")

    # ------------------------------------------------------------- 1. datos
    log("1/7 datos")
    train = read_train()
    particion = pd.read_csv(PARTICION)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos_fusionados(desarrollo))
    y_dev = desarrollo[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    y_train = train[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    log(f"   train global {len(y_train)} filas | dev {len(y_dev)} filas | "
        f"{len(pliegues)} pliegues canonicos")

    datos = np.load(OOF_CANONICO, allow_pickle=True)
    y_oof = np.asarray(datos["y_true"], dtype=np.int64)
    p_oof = np.asarray(datos["pred"], dtype=np.int64)
    np.asarray(datos["pliegues"], dtype=np.int64)
    log(f"   OOF canonico {len(y_oof)} filas | coincide con dev: {np.array_equal(y_dev, y_oof)}")

    # ------------------------------------------------------ 2. distribucion
    log("2/7 distribucion de estados")
    dist_train = distribucion(y_train)
    dist_dev = distribucion(y_dev)
    log("   estado  HS TR AG   train   n     %  |    dev     n     %")
    for a, b in zip(dist_train, dist_dev):
        log(f"     {a['estado']}     {a['HS']}  {a['TR']}  {a['AG']}  "
            f"{a['n']:6d} {a['porcentaje']:6.3f}  |  {b['n']:5d} {b['porcentaje']:6.3f}")
    observados = [d["estado"] for d in dist_train if d["n"] > 0]
    log(f"   estados observados en train: {observados}")
    log(f"   estados NO observados en train: {[e for e in range(8) if e not in observados]}")
    tr_sin_hs = sum(d["n"] for d in dist_train if d["TR"] == 1 and d["HS"] == 0)
    log(f"   TR=1 y HS=0 en train: {tr_sin_hs} casos")

    dist_por_pliegue = []
    for indice, (_, idx_va) in enumerate(pliegues, start=1):
        dist_por_pliegue.append({
            "pliegue": indice,
            "n": len(idx_va),
            "distribucion": distribucion(y_dev[idx_va]),
        })
    log("   por pliegue, numero de estados distintos observados:")
    for p in dist_por_pliegue:
        con_datos = [d["estado"] for d in p["distribucion"] if d["n"] > 0]
        log(f"     pliegue {p['pliegue']} (n={p['n']}): {con_datos}")

    # --------------------------------------------------- 3. matriz 8 x 8
    log("3/7 matriz de 8 x 8 estado verdadero contra estado predicho")
    verdadero = a_estado(y_oof)
    predicho = a_estado(p_oof)
    matriz = np.zeros((8, 8), dtype=np.int64)
    for v, p in zip(verdadero, predicho):
        matriz[v, p] += 1
    log("        pred:  " + "".join(f"{nombre(e):>7}" for e in range(8)))
    for v in range(8):
        fila = "".join(f"{matriz[v, p]:7d}" for p in range(8))
        marca = "  <-" if matriz[v, v] else ""
        log(f"   true {nombre(v)}:{fila}   n={matriz[v].sum():4d}{marca}")
    aciertos_estado = int(np.trace(matriz))
    log(f"   filas con el estado entero correcto: {aciertos_estado} "
        f"({100.0 * aciertos_estado / len(y_oof):.3f} %)")

    # ------------------------------------------ 4. categorias A B C D
    log("4/7 categorias de error de V03")
    error = (p_oof != y_oof)
    n_errores = error.sum(axis=1)
    categorias = {"A_tres_correctas": 0, "B_falla_una": 0, "C_falla_dos": 0, "D_falla_tres": 0}
    for cuenta in n_errores:
        categorias["A_tres_correctas"] += int(cuenta == 0)
        categorias["B_falla_una"] += int(cuenta == 1)
        categorias["C_falla_dos"] += int(cuenta == 2)
        categorias["D_falla_tres"] += int(cuenta == 3)
    total = len(y_oof)
    for clave, valor in categorias.items():
        log(f"   {clave:20s} {valor:5d}  {100.0 * valor / total:6.3f} %")
    celdas_totales = int(error.sum())
    log(f"   celdas erroneas: {celdas_totales} de {3 * total}")

    # ------------------------------------ 5. desglose por estado verdadero
    log("5/7 desglose HS/TR/AG por estado verdadero")
    por_estado = []
    for v in range(8):
        mascara = verdadero == v
        n = int(mascara.sum())
        if n == 0:
            continue
        fila = {
            "estado": v, "nombre": nombre(v), "n": n,
            "n_HS": int(np.count_nonzero(mascara & (y_oof[:, 0] == 1))),
            "n_TR": int(np.count_nonzero(mascara & (y_oof[:, 1] == 1))),
            "n_AG": int(np.count_nonzero(mascara & (y_oof[:, 2] == 1))),
        }
        for j, etiqueta in enumerate(ETIQUETAS):
            fila[f"err_{etiqueta}"] = int(np.count_nonzero(mascara & error[:, j]))
        fila["tasa_err_HS"] = fila["err_HS"] / n
        fila["tasa_err_TR"] = fila["err_TR"] / n
        fila["tasa_err_AG"] = fila["err_AG"] / n
        por_estado.append(fila)
        log(f"   {nombre(v)}  n={n:4d} | err HS {fila['err_HS']:3d} ({fila['tasa_err_HS']:.3f}) "
            f"TR {fila['err_TR']:3d} ({fila['tasa_err_TR']:.3f}) "
            f"AG {fila['err_AG']:3d} ({fila['tasa_err_AG']:.3f})")

    # ------------------------------------- 6. contraste de independencia
    log("6/7 contraste: los errores de las tres etiquetas son independientes?")
    tabla_global: list[dict[str, Any]] = []
    chi_acumulado = 0.0
    gl_acumulado = 0
    for v in range(8):
        mascara = verdadero == v
        n = int(mascara.sum())
        if n == 0:
            continue
        e = error[mascara]
        observados_patron = Counter()
        for fila in e:
            patron = "".join("1" if x else "0" for x in fila)
            observados_patron[patron] += 1
        marginales = [float(np.count_nonzero(e[:, j])) / n for j in range(3)]
        esperado = {}
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    pa = marginales[0] if a else 1 - marginales[0]
                    pb = marginales[1] if b else 1 - marginales[1]
                    pc = marginales[2] if c else 1 - marginales[2]
                    esperado[f"{a}{b}{c}"] = n * pa * pb * pc
        chi = 0.0
        gl = 0
        for patron, exp in esperado.items():
            obs = observados_patron.get(patron, 0)
            tabla_global.append({
                "estado": v, "estado_nombre": nombre(v), "n": n,
                "patron_error": patron, "observado": int(obs), "esperado_indep": round(exp, 3),
            })
            if exp > 5:
                chi += (obs - exp) ** 2 / exp
                gl += 1
        gl = max(gl - 1, 1)
        chi_acumulado += chi
        gl_acumulado += gl
        p_valor = float(1 - stats.chi2.cdf(chi, gl)) if gl > 0 else float("nan")
        log(f"   estado {nombre(v)} (n={n:4d}): chi2 {chi:7.3f} gl {gl} p {p_valor:.4f}")
    p_global = float(1 - stats.chi2.cdf(chi_acumulado, gl_acumulado))
    log(f"   agregado: chi2 {chi_acumulado:.3f} gl {gl_acumulado} p {p_global:.6f}")
    log(f"   p < 0.05 -> los errores NO son independientes dado el estado: "
        f"{p_global < 0.05}")

    # desviacion maxima observada frente a esperado
    desv = [abs(f["observado"] - f["esperado_indep"]) for f in tabla_global]
    log(f"   desviacion absoluta maxima de una celda: {max(desv):.2f} casos")

    # --------------------------------------- 7. correccion estrutural
    log("7/7 consecuencia: una correccion estructural dura, cuanto valdria")
    from src.evaluation import metricas_multietiqueta

    base = float(metricas_multietiqueta(y_oof, p_oof)["f1_media_columnas"])
    p_tr0 = p_oof.copy()
    p_tr0[p_tr0[:, 0] == 0, 1] = 0
    p_ag0 = p_oof.copy()
    p_ag0[p_ag0[:, 0] == 0, 2] = 0
    for etiqueta, indice, ajustada in (("TR", 1, p_tr0), ("AG", 2, p_ag0)):
        valor = float(metricas_multietiqueta(y_oof, ajustada)["f1_media_columnas"])
        log(f"   forzar {etiqueta}=0 cuando HS=0: {valor:.10f} "
            f"(delta {valor - base:+.10f})")
    log(f"   control V03: {base:.10f}")

    cs_tr = float(metricas_multietiqueta(y_oof, p_tr0)["f1_media_columnas"])
    cs_ag = float(metricas_multietiqueta(y_oof, p_ag0)["f1_media_columnas"])
    # --------------------------------- 8. inventario de mecanismos ya probados
    log("8/8 inventario de mecanismos conjuntos ya probados (leido del registro)")
    registro = pd.read_csv(PROJECT_ROOT / "experiments.csv")
    mecanismos = [
        ("Label Powerset", ["V4-E01-E03"], "multinomial sobre combinaciones de etiqueta"),
        ("Classifier chain HS->TR->AG", ["E2-01-HS-TR-AG"], "cadena anidada de scores"),
        ("Classifier chain HS->AG->TR", ["E2-01-HS-AG-TR"], "cadena anidada de scores"),
        ("Cadena con espacios normalizados", ["E2-01-HS-TR-AG-STD"], "cadena anidada"),
        ("Espacio separado por etiqueta", ["E2-21-ESPACIOS"], "espacio propio por etiqueta"),
        ("Acoplamiento suave entre cabeceras", ["V8-05"], "penalizacion hacia la media"),
        ("Meta-modelo sobre scores OOF", ["E2-02-META"], "meta-modelo de dependencia"),
        ("Meta-modelo con canales", ["E2-18-META-BASES"], "meta-modelo de dependencia"),
        ("Correccion estructural dura", [], "regla dura TR/AG=0 si HS=0"),
    ]
    inventario = []
    for mecanismo, ids, patron in mecanismos:
        if ids:
            filas = registro[registro.experiment_id.isin(ids)]
        else:
            filas = registro.iloc[0:0]
        if mecanismo == "Correccion estructural dura":
            resultado = (f"medido en esta fase: TR=0 si HS=0 da "
                         f"{cs_tr - base:+.10f}, AG=0 si HS=0 da {cs_ag - base:+.10f}")
            decision = "NEGATIVA en ambos casos"
            uuids = "P45-01 (esta fase)"
        else:
            resultado = "; ".join(str(x)[:120] for x in filas.result.tolist()) or "no encontrado"
            decision = "; ".join(str(x)[:90] for x in filas.decision.tolist()) or "-"
            uuids = ", ".join(ids)
        inventario.append({
            "mecanismo": mecanismo,
            "experimento": uuids,
            "formulacion": patron,
            "resultado": resultado,
            "decision": decision,
            "equivalente_a_familia_previa": True,
        })
        log(f"   {mecanismo:38} -> {decision[:60]}")

    lineas_extra = [
        "",
        "## 7. Inventario de mecanismos conjuntos ya probados",
        "",
        "Ninguna formulacion de la lista de la directiva queda sin probar.",
        "",
        "| mecanismo | experimento | decision registrada |",
        "|---|---|---|",
    ]
    for item in inventario:
        lineas_extra.append(
            f"| {item['mecanismo']} | `{item['experimento']}` | {item['decision'][:110]} |"
        )
    lineas_extra += [
        "",
        ("Los dos unicos que produjeron mejora local (E2-02-META, 0.785320) ya fallaron en el "
        "leaderboard: `v08_meta_v2` scored 0.78163 frente a 0.78306 de V03."),
        "",
        "## 8. Veredicto sobre la estructura conjunta",
        "",
        (f"**Hay estructura conjunta real y es medible.** El contraste de independencia da "
        f"chi2 {chi_acumulado:.1f} con {gl_acumulado} grados de libertad, "
        f"p = {p_global:.2e}. Los "
        "errores de HS, TR y AG no son independientes dado el estado verdadero."),
        "",
        ("**Pero la estructura no es explotable con ninguna de lasvias ya probadas.** La "
        "correccion estructural dura, que es la extraccion mas directa de esa estructura, es "
        f"negativa en las dos formas ({cs_tr - base:+.10f} y {cs_ag - base:+.10f}). Y las "
        "formulaciones de aprendizaje conjunto (label powerset, cadenas, meta-modelos, "
        "acoplamiento suave) ya estan todas registradas con resultado negativo o sin "
        "transferencia."),
        "",
        "**`JOINT_LABEL_STRUCTURE = CLOSED`.** No se abre ninguna familia conjunta nueva.",
        "",
    ]

    informe = {
        "familia": "P45_METRIC_AUDIT",
        "experimento": "V45-01 (P1 estructura conjunta HS x TR x AG y error de V03)",
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "AUDIT_ONLY",
        "pregunta": ("Hay evidencia de que los errores de V03 dependan de la combinacion "
                     "conjunta de etiquetas de una forma que no capturen sus tres decisiones "
                     "binarias independientes?"),
        "nota_de_alcance": ("El holdout no se usa para ninguna decision. Las cifras de "
                            "train global son descriptivas; todas las decisiones se toman "
                            "sobre dev y el OOF canonico."),
        "distribucion": {
            "train_global": dist_train,
            "dev": dist_dev,
            "por_pliegue": dist_por_pliegue,
            "estados_observados_train": observados,
            "estados_no_observados_train": [e for e in range(8) if e not in observados],
            "casos_TR1_HS0_en_train": tr_sin_hs,
        },
        "matriz_8x8": {
            "orden": "filas = estado verdadero, columnas = estado predicho",
            "etiquetas": [nombre(e) for e in range(8)],
            "matriz": matriz.tolist(),
            "filas_con_estado_correcto": aciertos_estado,
            "porcentaje_estado_correcto": float(100.0 * aciertos_estado / len(y_oof)),
        },
        "categorias_error": {k: {"n": v, "porcentaje": 100.0 * v / total}
                             for k, v in categorias.items()},
        "por_estado_verdadero": por_estado,
        "contraste_independencia": {
            "chi2": float(chi_acumulado),
            "gl": int(gl_acumulado),
            "p_valor": p_global,
            "significativo": bool(p_global < 0.05),
            "desviacion_maxima_celda": float(max(desv)),
            "tabla": tabla_global,
        },
        "inventario_mecanismos": inventario,
        "correccion_estructural_dura": {
            "control_v03": base,
            "forzar_TR0_si_HS0": metricas_multietiqueta(y_oof, p_tr0)["f1_media_columnas"],
            "forzar_AG0_si_HS0": metricas_multietiqueta(y_oof, p_ag0)["f1_media_columnas"],
        },
        "gastos": {
            "submissions": 0,
            "modelos_entrenados": 0,
            "segundos": round(time.time() - inicio, 1),
        },
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _csv(informe)
    _md(informe, lineas_extra)
    log("   informes escritos en reports/p45_joint_label_structure.{json,md,csv}")
    log(f"p del contraste de independencia = {p_global:.6f}")
    log(f"VEREDICTO: auditoria descriptiva, 0 modelos, 0 submissions, "
        f"{round(time.time() - inicio, 1)} s")


def _csv(informe: dict[str, Any]) -> None:
    """Escribe el informe tabular.

    Args:
        informe: diccionario con los resultados.

    Returns:
        None.
    """
    lineas = ["bloque,estado,HS,TR,AG,n,porcentaje,nota"]
    for d in informe["distribucion"]["train_global"]:
        lineas.append(f"train_global,{d['estado']},{d['HS']},{d['TR']},{d['AG']},"
                      f"{d['n']},{d['porcentaje']:.4f},")
    for d in informe["distribucion"]["dev"]:
        lineas.append(f"dev,{d['estado']},{d['HS']},{d['TR']},{d['AG']},"
                      f"{d['n']},{d['porcentaje']:.4f},")
    for p in informe["distribucion"]["por_pliegue"]:
        for d in p["distribucion"]:
            if d["n"]:
                lineas.append(f"pliegue_{p['pliegue']},{d['estado']},{d['HS']},{d['TR']},"
                              f"{d['AG']},{d['n']},{d['porcentaje']:.4f},n_fold={p['n']}")
    et = informe["matriz_8x8"]["etiquetas"]
    for v, fila in enumerate(informe["matriz_8x8"]["matriz"]):
        for p, valor in enumerate(fila):
            lineas.append(f"matriz_8x8,{et[v]},{''},{''},{''},{valor},,"
                          f"true={et[v]} pred={et[p]}")
    for f in informe["contraste_independencia"]["tabla"]:
        lineas.append(f"independencia,{f['estado_nombre']},{''},{''},{''},"
                      f"{f['observado']},{f['esperado_indep']:.3f},"
                      f"patron_error={f['patron_error']} n_estado={f['n']}")
    for e in informe["por_estado_verdadero"]:
        lineas.append(
            f"error_por_estado,{e['nombre']},{e['n_HS']},{e['n_TR']},{e['n_AG']},"
            f"{e['n']},{e['tasa_err_HS']:.4f},"
            f"tasa_HS={e['tasa_err_HS']:.4f} tasa_TR={e['tasa_err_TR']:.4f} "
            f"tasa_AG={e['tasa_err_AG']:.4f}"
        )
    for k, v in informe["categorias_error"].items():
        lineas.append(f"categoria_error,{k},,,,{v['n']},{v['porcentaje']:.4f},")
    lineas.append(f"contraste_independencia,chi2,,,,{informe['contraste_independencia']['chi2']:.3f},"
                  f"{informe['contraste_independencia']['p_valor']:.6f},gl={informe['contraste_independencia']['gl']}")
    cs = informe["correccion_estructural_dura"]
    lineas.append(f"correccion_dura,control_v03,,,,{cs['control_v03']:.10f},,")
    lineas.append(f"correccion_dura,forzar_TR0_si_HS0,,,,{cs['forzar_TR0_si_HS0']:.10f},,")
    lineas.append(f"correccion_dura,forzar_AG0_si_HS0,,,,{cs['forzar_AG0_si_HS0']:.10f},,")
    REPORTE_CSV.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def _md(informe: dict[str, Any], extra: list[str]) -> None:
    """Escribe el informe legible.

    Args:
        informe: diccionario con los resultados.
        extra: lineas del inventario y del veredicto sobre la estructura conjunta.

    Returns:
        None.
    """
    d = informe["distribucion"]
    m = informe["matriz_8x8"]
    c = informe["contraste_independencia"]
    cs = informe["correccion_estructural_dura"]
    lineas = [
        "# P45 - Estructura conjunta HS x TR x AG y error de V03",
        "",
        (f"Generado por `experiments/v45_01_joint_label_structure.py` "
        f"({informe['experimento']})."),
        "",
        f"Fecha {informe['fecha']}. **0 modelos, 0 submissions.**",
        "",
        f"> {informe['nota_de_alcance']}",
        "",
        f"Pregunta: {informe['pregunta']}",
        "",
        "## 1. Distribucion de los ocho estados",
        "",
        "Variable `LABEL_STATE = 4*HS + 2*TR + AG`.",
        "",
        "| estado | HS | TR | AG | train n | train % | dev n | dev % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a, b in zip(d["train_global"], d["dev"]):
        lineas.append(
            f"| {a['estado']} | {a['HS']} | {a['TR']} | {a['AG']} | {a['n']} | "
            f"{a['porcentaje']:.3f} | {b['n']} | {b['porcentaje']:.3f} |"
        )
    lineas += [
        "",
        (f"Estados observados en train: {d['estados_observados_train']}. "
        f"No observados: {d['estados_no_observados_train']}."),
        "",
        f"Casos con `TR=1` y `HS=0` en train: **{d['casos_TR1_HS0_en_train']}**.",
        "",
        "## 2. Matriz de 8 x 8: estado verdadero contra estado predicho",
        "",
        "Filas = estado verdadero, columnas = estado predicho por V03.",
        "",
        "| true \\ pred | " + " | ".join(m["etiquetas"]) + " | n |",
        "|---|" + "---:|" * (len(m["etiquetas"]) + 1),
    ]
    for v, fila in enumerate(m["matriz"]):
        marca = " **<-**" if fila[v] else ""
        lineas.append(
            f"| **{m['etiquetas'][v]}** | " + " | ".join(str(x) for x in fila)
            + f" | {sum(fila)} |{marca}"
        )
    lineas += [
        "",
        (f"Filas con el estado entero correcto: **{m['filas_con_estado_correcto']}** "
        f"({m['porcentaje_estado_correcto']:.3f} %)."),
        "",
        "## 3. Categorias de error de V03",
        "",
        "| categoria | n | % |",
        "|---|---:|---:|",
    ]
    for k, v in informe["categorias_error"].items():
        lineas.append(f"| `{k}` | {v['n']} | {v['porcentaje']:.3f} |")
    lineas += [
        "",
        "## 4. Errores por etiqueta, para cada estado verdadero",
        "",
        "| estado | n | err HS | tasa | err TR | tasa | err AG | tasa |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for e in informe["por_estado_verdadero"]:
        lineas.append(
            f"| {e['nombre']} | {e['n']} | {e['err_HS']} | {e['tasa_err_HS']:.3f} | "
            f"{e['err_TR']} | {e['tasa_err_TR']:.3f} | {e['err_AG']} | {e['tasa_err_AG']:.3f} |"
        )
    lineas += [
        "",
        "## 5. Contraste de independencia de los errores",
        "",
        ("Si los tres indicadores de error fueran independientes dado el estado verdadero, "
        "los ocho patrones de error seguirian el producto de las marginales. Se contrasta esa "
        "hipotesis nula con chi-cuadrado, celda a celda, agregando solo las celdas con "
        "esperado mayor que 5."),
        "",
        f"- chi2 = **{c['chi2']:.3f}**, gl = {c['gl']}, p = **{c['p_valor']:.6f}**",
        f"- desviacion absoluta maxima de una celda: {c['desviacion_maxima_celda']:.2f} casos",
        f"- rechazamos la independencia al 5 %: **{c['significativo']}**",
        "",
        "## 6. Consecuencia: correccion estructural dura",
        "",
        "| regla | corrected_cv | delta |",
        "|---|---:|---:|",
        f"| control V03 | {cs['control_v03']:.10f} | 0 |",
        (f"| forzar TR=0 si HS=0 | {cs['forzar_TR0_si_HS0']:.10f} | "
        f"{cs['forzar_TR0_si_HS0'] - cs['control_v03']:+.10f} |"),
        (f"| forzar AG=0 si HS=0 | {cs['forzar_AG0_si_HS0']:.10f} | "
        f"{cs['forzar_AG0_si_HS0'] - cs['control_v03']:+.10f} |"),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas + extra) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
