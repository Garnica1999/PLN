"""Integra la fase P45 (auditoria de metrica y estructura conjunta) en el notebook.

Anade las secciones 40 y 41 al final de `notebooks/10_final_pipeline.ipynb`. Las
celdas de codigo leen los JSON en tiempo de ejecucion y no llevan outputs
embebidos, de modo que ninguna cifra experimental esta escrita a mano en el
notebook.

Ademas escribe `reports/p45_results.csv`, que consolida las cifras de las dos
auditorias en una sola tabla.

Uso
---
    /home/carlos/venv/bin/python experiments/v45_02_integrar_notebook.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
INFORME_METRICA = PROJECT_ROOT / "reports" / "p45_metric_audit.json"
INFORME_JOINT = PROJECT_ROOT / "reports" / "p45_joint_label_structure.json"
RESULTADOS = PROJECT_ROOT / "reports" / "p45_results.csv"

TITULO_METRICA = """\
## 40. P45 - Auditoria de la metrica (2026-09-27)

La pregunta previa a cualquier otra: la magnitud local `corrected_cv` que ha
dirigido todas las decisiones del proyecto, es exactamente la que puntua Kaggle?

Sin resolver esto, cualquier delta posterior comparativo careceria de base.
"""

TITULO_JOINT = """\
## 41. P45 - Auditoria de la estructura conjunta HS x TR x AG (2026-09-27)

La segunda pregunta: existe una formulacion de prediccion conjunta que NO sea
equivalente a Label Powerset, classifier chains, factorizacion condicional ni
post-procesamiento estructural ya probados?

`LABEL_STATE = 4*HS + 2*TR + AG`.
"""

CODIGO_METRICA = """\
from pathlib import Path
import json

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
with (RAIZ / "reports" / "p45_metric_audit.json").open(encoding="utf-8") as f:
    d = json.load(f)

r = d["reproduccion_v03"]
a = d["agregacion"]
i = d["identidad_macro"]
h = d["hipotesis_average"]
t = d["test_de_impacto"]
e = d["discriminante_escala_lb"]

print("40.1  REPRODUCCION DE V03")
print(f"  corrected_cv calculado : {r['corrected_cv_calculado']:.16f}")
print(f"  corrected_cv objetivo  : {r['corrected_cv_objetivo']:.16f}")
print(f"  diferencia             : {r['diferencia']:+.3e}  (tolerancia {r['tolerancia']:.0e})")
for k in ("HS", "TR", "AG"):
    c = r["conteos"][k]
    print(f"  {k}: TP {c['TP']:4d} FP {c['FP']:4d} FN {c['FN']:4d} TN {c['TN']:4d} "
          f"| F1 {r['f1_por_etiqueta'][k]:.6f}")
print()
print("40.2  MEDIA DE PLIEGUES FRENTE A OOF GLOBAL")
print(f"  media de los 5 pliegues : {a['media_de_pliegues']:.10f}")
print(f"  global sobre el OOF     : {a['global_oof']:.10f}")
print(f"  diferencia              : {a['diferencia']:+.10f}  (no intercambiables)")
print()
print("40.3  IDENTIDAD CON f1_score(average='macro')")
print(f"  f1_score macro  : {i['f1_score_macro']:.16f}")
print(f"  f1_media_columnas: {i['f1_media_columnas']:.16f}")
print(f"  {i['casos_aleatorios']} casos aleatorios, {i['discrepancias']} discrepancias")
print()
print("40.4  LAS CUATRO HIPOTESIS DEL PARAMETRO average")
for k, v in h["valores"].items():
    print(f"  {k:10s} {v:.10f}")
print(f"  dispersion: {h['dispersion']:.10f}")
print()
print("40.5  DISCRIMINANTE DE ESCALA CONTRA samples")
print(f"  LB de V03                 : {e['kaggle_f1_de_v03']:.5f}")
print(f"  macro_3 sobre el OOF      : {e['macro_3_oof_de_v03']:.5f}")
print(f"  samples sobre el OOF      : {e['muestra_oof_de_v03']:.5f}")
print(f"  brecha contra samples     : {e['brecha_samples']:+.5f}")
print(f"  samples EXCLUIDO          : {d['samples_excluido_por_escala']}")
print(f"  pearson local vs kaggle   : {e['pearson_local_vs_kaggle']:.6f}  (debil, y se dice)")
print()
print("40.6  TEST DE IMPACTO: EL RANKING DE UMBRALES")
print(f"  {t['configuraciones']} configuraciones de umbral, sin usar test ni holdout")
for k, s in t["spearman"].items():
    print(f"  macro_3 vs {k:9s} spearman {s['spearman_vs_macro3']:.6f} "
          f"| misma mejor config {s['mejor_config_identica']}")
print()
print("40.7  VEREDICTO")
print(f"  METRIC_MATCH = {d['METRIC_MATCH']}")
print("  SI SOSTIENE: la metrica local es exactamente la macro de 3 etiquetas,")
print("               corrected_cv se reproduce al ultimo digito, y samples queda excluido.")
print("  NO SOSTIENE: que el host use 'macro' de forma directa; macro/micro/weighted")
print("               siguen sin distinguirse, aunque es inmaterial para el ranking.")
"""

CODIGO_JOINT = """\
from pathlib import Path
import json

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
with (RAIZ / "reports" / "p45_joint_label_structure.json").open(encoding="utf-8") as f:
    d = json.load(f)

dist = d["distribucion"]
m = d["matriz_8x8"]
cat = d["categorias_error"]
c = d["contraste_independencia"]
cs = d["correccion_estructural_dura"]

print("41.1  DISTRIBUCION DE LOS OCHO ESTADOS")
print("   estado  HS TR AG        n      %")
for f in dist["train_global"]:
    print(f"   {f['estado']}      {f['HS']}  {f['TR']}  {f['AG']}  {f['n']:6d}  {f['porcentaje']:6.3f}")
print(f"  estados no observados: {dist['estados_no_observados_train']}")
print(f"  casos TR=1 y HS=0     : {dist['casos_TR1_HS0_en_train']}")
print()
print("41.2  MATRIZ 8x8: ESTADO VERDADERO CONTRA ESTADO PREDICHO")
print("          pred: " + "".join(f"{x:>7}" for x in m["etiquetas"]))
for v, fila in enumerate(m["matriz"]):
    print(f"   true {m['etiquetas'][v]}: " + "".join(f"{x:7d}" for x in fila)
          + f"   n={sum(fila):4d}")
print(f"  estado entero correcto: {m['filas_con_estado_correcto']} "
      f"({m['porcentaje_estado_correcto']:.3f} pct)")
print()
print("41.3  CATEGORIAS DE ERROR DE V03")
for k, v in cat.items():
    print(f"  {k:20s} {v['n']:5d}  {v['porcentaje']:6.3f} pct")
print()
print("41.4  CONTRASTE DE INDEPENDENCIA DE LOS ERRORES")
print(f"  chi2 {c['chi2']:.3f}  gl {c['gl']}  p {c['p_valor']:.3e}")
print(f"  desviacion maxima de una celda: {c['desviacion_maxima_celda']:.2f} casos")
print(f"  los errores NO son independientes dado el estado: {c['significativo']}")
print()
print("41.5  LA ESTRUCTURA ES REAL, PERO NO ES EXPLOTABLE")
print(f"  control V03                    : {cs['control_v03']:.10f}")
print(f"  forzar TR=0 si HS=0            : {cs['forzar_TR0_si_HS0']:.10f} "
      f"({cs['forzar_TR0_si_HS0'] - cs['control_v03']:+.10f})")
print(f"  forzar AG=0 si HS=0            : {cs['forzar_AG0_si_HS0']:.10f} "
      f"({cs['forzar_AG0_si_HS0'] - cs['control_v03']:+.10f})")
print()
print("41.6  INVENTARIO DE MECANISMOS CONJUNTOS YA PROBADOS")
for it in d["inventario_mecanismos"]:
    print(f"  {it['mecanismo']:38s} {it['experimento']:26s} {it['decision'][:56]}")
print()
print("41.7  VEREDICTO")
print("  JOINT_LABEL_STRUCTURE = CLOSED")
print("  Hay estructura conjunta medible, pero ninguna via Known la explota:")
print("  la correccion dura es negativa y las Learning conjunta ya estan cerradas.")
"""


def markdown(texto: str) -> dict[str, Any]:
    """Construye una celda markdown.

    Args:
        texto: fuente de la celda.

    Returns:
        La celda como diccionario de notebook.
    """
    return {"cell_type": "markdown", "metadata": {}, "source": texto.splitlines(keepends=True)}


def codigo(texto: str) -> dict[str, Any]:
    """Construye una celda de codigo sin outputs.

    Args:
        texto: fuente de la celda.

    Returns:
        La celda como diccionario de notebook.
    """
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": texto.splitlines(keepends=True),
    }


def escribir_results(metrica: dict[str, Any], joint: dict[str, Any]) -> None:
    """Escribe la tabla consolidada de resultados de P45.

    Args:
        metrica: diccionario de la auditoria de metrica.
        joint: diccionario de la auditoria de estructura conjunta.

    Returns:
        None.
    """
    filas: list[list[Any]] = [["bloque", "magnitud", "valor", "nota"]]
    r = metrica["reproduccion_v03"]
    filas.append(["metrica", "corrected_cv_calculado", f"{r['corrected_cv_calculado']:.16f}",
                  f"objetivo {r['corrected_cv_objetivo']:.16f}"])
    filas.append(["metrica", "diferencia_reproduccion", f"{r['diferencia']:.3e}",
                  f"tolerancia {r['tolerancia']:.0e}"])
    for etiqueta in ("HS", "TR", "AG"):
        filas.append(["metrica", f"f1_{etiqueta}", f"{r['f1_por_etiqueta'][etiqueta]:.10f}",
                      (f"TP {r['conteos'][etiqueta]['TP']} FP {r['conteos'][etiqueta]['FP']} "
                      f"FN {r['conteos'][etiqueta]['FN']} TN {r['conteos'][etiqueta]['TN']}")])
    a = metrica["agregacion"]
    filas.append(["metrica", "media_de_pliegues", f"{a['media_de_pliegues']:.10f}", "no intercambiable"])
    filas.append(["metrica", "global_oof", f"{a['global_oof']:.10f}", "el que reporta el proyecto"])
    i = metrica["identidad_macro"]
    filas.append(["metrica", "f1_score_macro", f"{i['f1_score_macro']:.16f}", "identidad demostrada"])
    filas.append(["metrica", "discrepancias_aleatorios", i["discrepancias"],
                  f"{i['casos_aleatorios']} casos aleatorios"])
    for nombre, valor in metrica["hipotesis_average"]["valores"].items():
        filas.append(["metrica", f"hypothesis_{nombre}", f"{valor:.10f}", "candidata del host"])
    e = metrica["discriminante_escala_lb"]
    filas.append(["metrica", "lb_v03", f"{e['kaggle_f1_de_v03']:.5f}", "del log local"])
    filas.append(["metrica", "brecha_samples", f"{e['brecha_samples']:+.5f}", "discriminante de escala"])
    filas.append(["metrica", "samples_excluido", metrica["samples_excluido_por_escala"], ""])
    for nombre, s in metrica["test_de_impacto"]["spearman"].items():
        filas.append(["metrica", f"spearman_macro3_vs_{nombre}",
                      f"{s['spearman_vs_macro3']:.6f}", "ranking de umbrales"])
    filas.append(["metrica", "METRIC_MATCH", metrica["METRIC_MATCH"], ""])
    for f in joint["distribucion"]["train_global"]:
        filas.append(["joint", f"estado_{f['estado']}", f["n"], f"train {f['porcentaje']:.3f} pct"])
    filas.append(["joint", "estados_no_observados", str(joint["distribucion"]["estados_no_observados_train"]),
                  "TR=1 implica HS=1"])
    filas.append(["joint", "casos_TR1_HS0", joint["distribucion"]["casos_TR1_HS0_en_train"], "train global"])
    m = joint["matriz_8x8"]
    filas.append(["joint", "filas_estado_correcto", m["filas_con_estado_correcto"],
                  f"{m['porcentaje_estado_correcto']:.3f} pct"])
    for k, v in joint["categorias_error"].items():
        filas.append(["joint", k, v["n"], f"{v['porcentaje']:.3f} pct"])
    c = joint["contraste_independencia"]
    filas.append(["joint", "chi2_independencia", f"{c['chi2']:.3f}", f"gl {c['gl']}"])
    filas.append(["joint", "p_independencia", f"{c['p_valor']:.3e}", "rechaza independencia"])
    cs = joint["correccion_estructural_dura"]
    filas.append(["joint", "control_v03", f"{cs['control_v03']:.10f}", ""])
    filas.append(["joint", "forzar_TR0_si_HS0", f"{cs['forzar_TR0_si_HS0']:.10f}",
                  f"{cs['forzar_TR0_si_HS0'] - cs['control_v03']:+.10f}"])
    filas.append(["joint", "forzar_AG0_si_HS0", f"{cs['forzar_AG0_si_HS0']:.10f}",
                  f"{cs['forzar_AG0_si_HS0'] - cs['control_v03']:+.10f}"])
    for it in joint["inventario_mecanismos"]:
        filas.append(["joint", f"mecanismo_{it['mecanismo']}", it["experimento"], it["decision"][:120]])
    with RESULTADOS.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(filas)
    print(f"  reports/p45_results.csv escrito con {len(filas) - 1} filas")


def main() -> None:
    """Anade las secciones 40 y 41 al notebook consolidado.

    Returns:
        None.
    """
    if not INFORME_METRICA.exists() or not INFORME_JOINT.exists():
        raise SystemExit("PARADA: faltan los informes de P45")
    metrica = json.loads(INFORME_METRICA.read_text(encoding="utf-8"))
    joint = json.loads(INFORME_JOINT.read_text(encoding="utf-8"))

    escribir_results(metrica, joint)

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    marcas = " ".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    if "P45 - Auditoria de la metrica" in marcas:
        print("  las secciones 40 y 41 ya existen, no se duplican")
        return
    notebook["cells"].extend([
        markdown(TITULO_METRICA), codigo(CODIGO_METRICA),
        markdown(TITULO_JOINT), codigo(CODIGO_JOINT),
    ])
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas")
    print("  secciones 40 y 41 integradas en el notebook consolidado")


if __name__ == "__main__":
    main()
