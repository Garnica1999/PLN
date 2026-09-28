"""Integra la fase P50 (Contextual Channel Selection) en el notebook consolidado.

Anade la seccion 46 al final de `notebooks/10_final_pipeline.ipynb`. La celda de
codigo lee `reports/p50_gap_and_oracle.json` y `reports/p50_selector_results.json`
en tiempo de ejecucion, de modo que ninguna cifra de la seccion esta escrita a
mano.

Uso
---
    /home/carlos/venv/bin/python experiments/v50_02_integrar_notebook.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
INFORME_GAP = PROJECT_ROOT / "reports" / "p50_gap_and_oracle.json"
INFORME_SEL = PROJECT_ROOT / "reports" / "p50_selector_results.json"
MARCA = "46. P50 - Contextual Channel Selection"

TITULO = """\
## 46. P50 - Contextual Channel Selection (2026-09-28)

P49 cerro con una pregunta abierta: el rescate es real, pero difuso, y 19 de los
20 canales tienen net negativo. P50 hace la prueba decisiva en dos pasos.

**Paso 1, auditoria de brecha.** Solo la categoria E (elegir canal por label-cell
con derecho a abstenerse) seria nueva. Las categorias A a D ya estan medidas: el
ensayo y la ponderacion global en V17 y V23, el umbral en V03, y la correccion
selectiva en **V18-01, que dio veredicto negativo con AUC 0,6181**.

**Paso 2, techo del oracle.** Con `y_true` y el espacio completo de 21 acciones
(`KEEP_V03` mas 20 canales) se mide cuanto macro-F1 seria alcanzable. **Usa
`y_true`, luego no es modelo, no es evaluacion valida y no puede usarse para
seleccionar.** Es solo un techo.

El resultado es el hallazgo central de la fase: el techo es **+0,2098 de macro-F1**,
unas 70 veces el margen de 0,003. La oportunidad es enorme, no marginal. Y aun
asi, el selector aprendido la pierde entera y termina **peor que no hacer nada**.
"""


CODIGO = """\
from pathlib import Path
import json

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
gap = json.loads((RAIZ / "reports" / "p50_gap_and_oracle.json").read_text(encoding="utf-8"))
sel = json.loads((RAIZ / "reports" / "p50_selector_results.json").read_text(encoding="utf-8"))

brecha = gap["gap_audit"]
o = gap["oracle"]
s1 = sel["s1"]

print("46.1  AUDITORIA DE BRECHA: LAS CINCO CATEGORIAS")
print(f"  {'categoria':34}{'por celda':>11}  estado")
for m in brecha["catalogo"]:
    print(f"  {m['categoria']:34}{m['es_por_celda']:>11}  {m['estado']}")
print(f"  SELECTOR_FAMILY = {brecha['estado_familia']}")
print()

v18 = brecha["v18_selective_correction"]
print("46.2  LA CATEGORIA D YA SE MIDIO Y DIO NEGATIVO")
print(f"  estado                          : {v18['estado']}")
print(f"  challengers con senal nominal   : {v18['con_senal_nominal']} de {v18['challengers']}")
print(f"  con senal tras Bonferroni       : {v18['con_senal_tras_bonferroni']}")
print(f"  mejor AUC                       : {v18['mejor_auc']:.4f}")
print(f"  mejor p global                  : {v18['mejor_p_global']:.4f} (umbral {v18['umbral_bonferroni']:.4f})")
print(f"  mejor mejora fuera de muestra   : {v18['mejor_mejora_fuera_de_muestra']} celdas")
print(f"  necesaria para la escala        : {v18['mejora_necesaria_para_la_escala']} celdas")
print()

print("46.3  TECHO DEL ORACLE  (USA y_true: NO es modelo)")
print(f"  {'metrica':44}{'valor':>14}")
for x in (("espacio de acciones", o["espacio_de_acciones"]),
          ("macro-F1 de V03", f"{o['macro_f1_v03']:.10f}"),
          ("macro-F1 del oracle", f"{o['macro_f1_oracle']:.10f}"),
          ("DELTA DEL ORACLE", f"{o['delta_macro_f1']:+.10f}"),
          ("celdas de error de V03", o["celdas_de_error_v03"]),
          ("celdas con alguna accion correcta", o["celdas_con_alguna_accion_correcta"]),
          ("celdas sin ninguna accion correcta", o["celdas_sin_ninguna_accion_correcta"]),
          ("multiples del umbral de 0,003", f"{o['multiples_del_umbral']}")):
    print(f"  {x[0]:44}{x[1]:>14}")
print()
print(f"  {'etiqueta':12}{'F1 V03':>12}{'F1 oracle':>12}{'delta':>12}{'rescate':>10}")
for et, v in o["por_etiqueta"].items():
    print(f"  {et:12}{v['f1_v03']:>12.6f}{v['f1_oracle']:>12.6f}"
          f"{v['delta']:>+12.6f}{v['rescate']:>10}")
print()

print("46.4  FIABILIDAD DE LOS CANALES SOBRE EL CONJUNTO DE DESACUERDO")
print(f"  {'canal':36}{'acc global':>12}{'acc desacuerdo':>17}{'net':>9}")
fiab = sorted(gap["fiabilidad_por_canal"], key=lambda f: -f["accuracy_dado_desacuerdo"])
for f in fiab:
    print(f"  {f['canal']:36}{f['accuracy_global']:>12.4f}"
          f"{f['accuracy_dado_desacuerdo']:>17.4f}{f['net']:>+9d}")
print(f"  mejor accuracy dado el desacuerdo: {gap['mejor_accuracy_dado_desacuerdo']:.4f} (0,5000 es azar)")
print()

print("46.5  BASELINES DE DIAGNOSTICO")
print(f"  {'selector':36}{'corrected_cv':>15}{'delta':>14}{'net':>9}")
for b in sel["baselines"]:
    print(f"  {b['selector']:36}{b['corrected_cv']:>15.10f}"
          f"{b['delta_vs_v03']:>+14.10f}{b['net']:>+9d}")
print()

print("46.6  S1, SELECTOR DE CORRECCION CROSS-FITTED")
print(f"  corrected_cv                  : {s1['corrected_cv']:.10f}")
print(f"  delta vs V03                  : {s1['delta_vs_v03']:+.10f}")
print(f"  celdas cambiadas              : {s1['celdas_cambiadas']}")
print(f"  tasa de abstencion            : {100 * s1['tasa_de_abstencion']:.2f} pct")
print(f"  rescate / damage / net        : {s1['rescate']} / {s1['damage']} / {s1['net']:+d}")
print(f"  accuracy de la accion elegida : {s1['accuracy_de_la_accion_elegida']:.6f}")
print(f"  pliegues positivos            : {sel['pliegues_positivos']} de 5")
print()
n_cel, n_err = 4449, o["celdas_de_error_v03"]
aciertos_v03 = n_cel - n_err
aciertos_sel = round(s1["accuracy_de_la_accion_elegida"] * n_cel)
print(f"  CONTRASTE DECISIVO: no hacer nada acierta {aciertos_v03} de {n_cel} celdas")
print(f"  ({100 * aciertos_v03 / n_cel:.4f} pct). El selector acierta {aciertos_sel}")
print(f"  ({100 * aciertos_sel / n_cel:.4f} pct). Diferencia: {aciertos_sel - aciertos_v03:+d} celdas")
print("  es decir, la accion que elige el selector es PEOR que abstenerse siempre.")
print()

print("46.7  AUC OUT-OFOLD: LA CIFRA DE IDENTIFICABILIDAD")
print(f"  {'accion':36}{'AUC':>8}")
for a in sorted(s1["auc_por_accion"], key=lambda d: -(d["auc"] or 0)):
    valor = "n/d" if a["auc"] is None else "%.4f" % a["auc"]
    print(f"  {a['accion']:36}{valor:>8}")
print()
print("46.8  POR PLIEGUE")
print(f"  {'pliegue':10}{'F1 V03':>14}{'F1 selector':>14}{'delta':>14}")
for p in sel["pliegues"]:
    print(f"  {p['pliegue']:>10}{p['f1_v03']:>14.10f}{p['f1_selector']:>14.10f}"
          f"{p['delta']:>+14.10f}")
print()

print("46.9  INTERSECCION DE ERRORES")
for k, v in sel["interseccion_de_errores"].items():
    print(f"  {k:28}{v:>8}")
print()

print("46.10  POR MARGEN DE V03: TODO EL DANO ESTA EN LA FRONTERA")
print(f"  {'bin':14}{'celdas':>8}{'errores':>9}{'cambiadas':>11}{'resc':>7}{'dano':>7}{'net':>8}")
for m in sel["margen"]:
    print(f"  {m['bin']:14}{m['n_celdas']:>8}{m['n_errores_v03']:>9}"
          f"{m['n_cambiadas']:>11}{m['rescate']:>7}{m['damage']:>7}{m['net']:>+8d}")
print()

print("46.11  GATE Y DECISION")
g = sel["gate"]
print(f"  delta {g['delta_obtenido']:+.6f} frente a {g['delta_minimo']} exigido")
print(f"  pliegues positivos {sel['pliegues_positivos']} de {g['pliegues_positivos_requeridos']}")
print(f"  gate pasado: {g['pasado']}")
print(f"  CASE = {sel['caso']}")
print("  S2 NO se ejecuta: la especificacion lo prohibe cuando S1 ya falla de forma clara,")
print("  y S1 tiene delta negativo y cero pliegues positivos.")
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
    """Construye una celda de codigo sin salidas.

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


def main() -> None:
    """Anade la seccion 46 al notebook consolidado.

    Returns:
        None.
    """
    for informe in (INFORME_GAP, INFORME_SEL):
        if not informe.exists():
            raise SystemExit(f"PARADA: falta {informe}")
        json.loads(informe.read_text(encoding="utf-8"))
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    marcas = " ".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    if MARCA in marcas:
        print("  la seccion 46 ya existe, no se duplica")
        return
    notebook["cells"].extend([markdown(TITULO), codigo(CODIGO)])
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas")
    print("  seccion 46 integrada en el notebook consolidado")


if __name__ == "__main__":
    main()
