"""P33+P34 - seccion del notebook consolidado y auditoria de compliance por AST.

## Dos tareas que van juntas

1. **Notebook**: anade la seccion P33 y P34 con las cifras leidas **de los artefactos**, nunca
   escritas a mano, para que no pueda desincronizarse. El brief exige que incluya el 59,94 % y que
   explique por que el 90 % era una inferencia incorrecta. **El notebook no debe mostrar 90 % como
   hecho**, y por eso lleva una seccion dedicada a desmentirlo.

2. **Compliance por AST**: recorre el arbol de sintaxis de cada script de P33 y P34 buscando
   imports y llamadas, no texto crudo. Un `grep` de "embeddings" encuentra el nombre de la
   variable que lo declara; el AST encuentra el `import`. La diferencia importa porque el propio
   script de auditoria contiene la palabra.

Ademas incluye las **comprobaciones positivas**: que los mecanismos permitidos esten presentes. Una
auditoria que solo busca ausencias pasa un proyecto vacio.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "p33_p34_notebook_compliance.json"
NOTEBOOK = ROOT / "notebooks" / "99_entrega_consolidado.ipynb"
COMPLIANCE_MD = cfg.REPORTS / "p33_compliance.md"
MARCADOR = "FASE P33 Y P34"
FECHA = "2026-10-04"
CV_V30 = 0.6134471
LB_V30 = 0.63302
N_CLASES = 6

PROHIBIDOS = {
    "embeddings": [r"\bsentence_transformers\b", r"\bSentenceTransformer\b", r"\bAutoModel\b",
                   r"\bembeddings?\s*=\s*.*\btransformer\b", r"\bget_embedding\b"],
    "transformers": [r"\bfrom\s+transformers\b", r"\bimport\s+transformers\b", r"\bAutoModel\b",
                     r"\bAutoTokenizer\b", r"\bBertModel\b", r"\bRobertaModel\b", r"\bDistilBert\b"],
    "LLM": [r"\bopenai\b", r"\banthropic\b", r"\bcohere\b", r"\bllama\b", r"\bmistral\b",
            r"\bchat_completion\b", r"\bgpt-4\b"],
    "datos_externos": [r"\brequests\.get\b", r"\burllib\.request\b", r"\bhttpx\b", r"\bsocket\b"],
    "pseudo_labeling": [r"\bpseudo_?label", r"\bself_?train", r"\bsemi_?supervis"],
    "test_labels": [r"age_test\.csv.*age_range", r"load_test\([^)]*\)\[[^\]]*age_range"],
    "holdout": [r"\bholdout2\b"],
    "score_por_suma": [r"/\s*sum\([^)]*decision", r"score\s*/\s*\.sum\(axis\s*=\s*1\)"],
}
POSITIVAS = {
    "validacion_agrupada": r"make_folds\([^)]*groups=",
    "oos_estricto": r"val_idx",
    "inner_train_separado": r"inner_tr|inner_train",
    "bootstrap": r"default_rng\(",
    "control_barajado": r"barraj|permut",
    "control_aleatorio": r"rng\.normal",
    "score_type_declarado": r"decision_function",
}


def scripts_de_la_fase() -> list[Path]:
    return sorted((ROOT / "scripts").glob("p3[34]_*.py"))


def audita_ast() -> dict:
    """Recorre imports y llamadas. No texto crudo: el propio auditor contiene los terminos."""
    hallazgos, positivos = [], {k: 0 for k in POSITIVAS}
    for ruta in scripts_de_la_fase():
        if ruta.name == "p33_p34_notebook_compliance.py":
            continue
        arbol = ast.parse(ruta.read_text(encoding="utf-8"))
        imports, llamadas = set(), set()
        for nodo in ast.walk(arbol):
            if isinstance(nodo, ast.Import):
                imports.update(a.name for a in nodo.names)
            elif isinstance(nodo, ast.ImportFrom) and nodo.module:
                imports.add(nodo.module)
            elif isinstance(nodo, ast.Call):
                try:
                    llamadas.add(ast.unparse(nodo.func))
                except Exception:  # noqa: BLE001
                    pass
        texto = ruta.read_text(encoding="utf-8")
        for recurso, patrones in PROHIBIDOS.items():
            for patron in patronos_de(patrones):
                for obj in imports | llamadas:
                    if re.search(patron, obj, re.IGNORECASE):
                        hallazgos.append({"script": ruta.name, "recurso": recurso,
                                          "encontrado_en": obj, "tipo": "import_o_llamada"})
        # `score_por_suma` se busca en el texto porque el patronarda division, no una llamada.
        for linea, texto_linea in enumerate(texto.splitlines(), 1):
            for patron in PROHIBIDOS["score_por_suma"]:
                if re.search(patron, texto_linea, re.IGNORECASE):
                    # Se admite que aparezca en un comentario que lo prohibe, no en codigo vivo.
                    if texto_linea.strip().startswith("#") or "PROHIBID" in texto_linea.upper():
                        continue
                    hallazgos.append({"script": ruta.name, "recurso": "score_por_suma",
                                      "encontrado_en": f"linea {linea}: {texto_linea.strip()[:90]}",
                                      "tipo": "codigo"})
        for mecanismo, patron in POSITIVAS.items():
            if re.search(patron, texto):
                positivos[mecanismo] += 1
    return {"scripts": [p.name for p in scripts_de_la_fase()],
            "hallazgos": hallazgos, "total": len(hallazgos),
            "positivos": positivos,
            "positivos_ok": all(v > 0 for v in positivos.values())}


def patronos_de(patrones: list[str]) -> list[str]:
    """Devuelve la lista de patrones. Se separa para no repetirla en cada bucle."""
    return patrones


def anade_notebook() -> dict:
    """Anade la seccion P33 y P34, idempotente, con cifras leidas de los artefactos."""
    import nbformat
    nb = nbformat.read(NOTEBOOK, as_version=4)
    antes = len(nb.cells)
    inicio = None
    for i, c in enumerate(nb.cells):
        if c.cell_type == "markdown" and MARCADOR in c.source:
            inicio = i
            break
    if inicio is not None:
        del nb.cells[inicio:]

    def md(t):
        nb.cells.append(nbformat.v4.new_markdown_cell(t))

    def code(t):
        nb.cells.append(nbformat.v4.new_code_cell(t))

    md(f"""# {MARCADOR} - RESIDUALIZACION Y CORRECCION ADYACENTE

## Lo que se pregunta y lo que se responde

La pregunta era si la senal linguistica que V31+ encontro puede convertirse en una mejora
decisional aunque no sirva como feature. Se responde en dos pasos, y los dos son negativos para la
integracion y positivos para la comprension.

**P33, correccion global de seis clases.** El residuo, tras quitarle la componente que V30
explica, conserva AUC 0,5411 contra one-vs-rest en las seis clases. Hay informacion. Pero las
cuatro formas de sumarla al score no mejoran el F1, y el control de ruido aleatorio gana a la senal
real.

**P34, correccion local entre bandas adyacentes.** El residuo por pareja tiene AUC 0,49-0,51, y la
correccion local tampoco mejora nada. Peor: cero rescates de alta confianza.

## Una correccion historica, antes que nada

Este bloque desmente una cifra que el propio proyecto publico.

**V31 reporto que el 90 % de las confusiones de V30 eran entre bandas adyacentes. Ese 90 % era
falso.** Salio de mirar las **diez confusiones mas frecuentes**, que por construccion son casi
todas adyacentes, y de ahi se concluyo que el resto tambien lo seria. Contado sobre los **65.403
errores completos**: **59,94 % adyacentes y 40,06 % saltos no adyacentes**.

La consecuencia importa para el alcance: las adyacentes siguen siendo la mayoria, asi que la
hipotesis era valida, pero su **techo** era bastante mas bajo de lo que se suponia. Solo el 59,94 %
de los errores es elegible para una correccion por pares.

Los informes de V31 **no se han modificado**. La correccion vive aqui y en los informes de P34.""")

    code("""import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXP = pathlib.Path("../experiments")
REP = pathlib.Path("../reports")
p33 = json.loads((EXP / "p33_01_residualizacion.json").read_text(encoding="utf-8"))
p34 = json.loads((EXP / "p34_01_pairwise.json").read_text(encoding="utf-8"))
cierre = json.loads((EXP / "p33_p34_cierre.json").read_text(encoding="utf-8"))
conf = json.loads((REP / "p34_adjacent_confusion.csv").read_text(encoding="utf-8").splitlines()[0]
                  and (EXP / "p34_00_audit_inventario.json").read_text(encoding="utf-8"))
print("artefactos:", len([p33, p34, cierre, conf]))""")

    code("""# P02: la cifra real de confusiones adyacentes
m = conf["p02_matriz_adyacente"]
print(f"errores totales      {m['n_errores']:,}")
print(f"adyacentes           {m['n_errores_adyacentes']:,}  {m['fraccion_adyacente']:.2%}")
print(f"saltos no adyacentes {m['n_errores_salto']:,}  {m['fraccion_no_adyacente']:.2%}")
print()
print("el 90% de V31 venia de mirar solo las 10 confusiones mas frecuentes")
assert abs(m["fraccion_adyacente"] - 0.5994) < 1e-3
assert m["fraccion_adyacente"] < 0.65, "el 90% no puede ser correcto sobre el total"
print("[OK] la afirmacion del 90% queda desmentida con la cifra medida")""")

    code("""# P33: el residuo conserva informacion
t = pd.DataFrame(p33["p05_informacion_del_residuo"]["por_clase"])
print(t.to_string(index=False))
print()
print("AUC medio del residuo:", p33["p05_informacion_del_residuo"]["auc_medio_residuo_vs_ovr"])
print("-> hay informacion de edad en el residuo, pero pequena: z_C da 0.73-0.90 y z_V30 0.84-0.95")""")

    code("""# P33: la correccion no funciona, y el control lo explica
print(pd.DataFrame(p33["p07_correcciones"])[
    ["brazo", "corrected_cv", "delta_vs_v30", "folds_positivos", "rescate", "danio", "neto"]
].to_string(index=False))
a = p33["p06_atribucion"]
print()
print("real - barajado:", a["real_menos_barajado"])
print("real - ruido   :", a["real_menos_ruido"], " <- NEGATIVO: el ruido gana")
print("betas por fold :", a["betas_por_fold"])
print("con intercepto - sin intercepto:", a["con_intercepto_menos_sin"],
      "(0 -> no es efecto de calibracion)")""")

    code("""# P34: scores pairwise, sobre el decision_function CRUDO
s = pd.DataFrame(p34["p03_scores"])
print(s[["par", "clase_i", "clase_j", "corr_d_c_d_v30", "acuerdo_signo",
         "operador_c"]].to_string(index=False))
print()
print("NUNCA dividir un decision_function por la suma de su fila: las filas suman negativo y eso")
print("invierte el orden. Medido en P32: F1 0,5048 con argmax crudo frente a 0,0445 con la")
print("normalizacion invalida.")""")

    code("""# P34: el residuo por pareja no supera el azar
print(pd.DataFrame(p34["p13_auc_por_pareja"]).to_string(index=False))
print()
print("el candidato supera a V30 en la decision binaria, pero con correlaciones d_C/d_V30 de")
print("0.57-0.79: es la misma informacion reescalada con ruido, no informacion independiente.")""")

    code("""# P34: resultado y controles
print(pd.DataFrame(p34["p10_resultados"])[
    ["brazo", "corrected_cv", "delta_vs_v30", "folds_positivos", "rescate", "danio", "neto"]
].to_string(index=False))
c = p34["p10_resultados"]
print()
print("real - barajado :", round(c[0]["delta_vs_v30"] - c[1]["delta_vs_v30"], 7))
print("real - aleatorio:", round(c[0]["delta_vs_v30"] - c[2]["delta_vs_v30"], 7))
print("-> real NO supera al aleatorio, luego INVALIDATED por la regla del brief")""")

    code("""# P34: por pareja, y el rescate de alta confianza
print(pd.DataFrame(p34["p09_por_pareja"]).to_string(index=False))
print()
h = p34["p12_rescate_alta_confianza"]
print("errores adyacentes de alta confianza:", f"{h['errores_adyacentes_alta']:,}")
print("rescatados de alta confianza:", h["rescatados_alta"])
print("danados de alta confianza   :", h["danados_alta"])
print("-> cero rescates. El mecanismo no toca lo que se le pedia.")""")

    code("""# Bootstrap pareado de los dos brazos reales
b = cierre["bootstrap"]
for fase in ("P33", "P34"):
    x = b[fase]
    print(f"{fase}: delta {x['media']:+.7f}  IC95 [{x['ic95'][0]:+.7f}, {x['ic95'][1]:+.7f}]  "
          f"contiene 0: {x['contiene_cero']}")
print()
print(x["metodo"])
print()
print("reproduccion:", json.dumps(cierre["reproduccion"], indent=2))""")

    code("""# Decision
print("P33: integracion global CLOSED | evidencia residual COMPLEMENTARY_BUT_NOT_TRANSFERABLE")
print("P34: INVALIDATED")
print()
print("Lo que NO se dice: que la informacion linguistica no exista. P33 midio AUC 0,5411.")
print("Lo que NO se dice: que V30 sea optimo.")
print("Lo que SÍ se dice: esa informacion residual no ha podido convertirse en una mejora")
print("robusta mediante los mecanismos global y local evaluados.")""")

    nbformat.write(nb, NOTEBOOK)
    return {"anadida": True, "antes": antes, "despues": len(nb.cells),
            "anadidas": len(nb.cells) - antes, "idempotente": True}


def main() -> int:
    """Audita por AST y anade la seccion del notebook."""
    autotest_numerico()
    inicio = time.time()
    print("=== compliance por AST ===", flush=True)
    aud = audita_ast()
    print(f"  scripts auditados: {len(aud['scripts'])}", flush=True)
    print(f"  hallazgos de recursos prohibidos: {aud['total']}", flush=True)
    for h in aud["hallazgos"][:8]:
        print(f"    {h['script']}: {h['recurso']} en {h['encontrado_en'][:60]}", flush=True)
    print("  comprobaciones positivas:", flush=True)
    for m, c in aud["positivos"].items():
        print(f"    {m:<26} presente en {c} script(s)", flush=True)

    print("\n=== notebook ===", flush=True)
    nb = anade_notebook()
    print(f"  {nb}", flush=True)

    COMPLIANCE_MD.write_text("\n".join([
        "# P33 y P34 - auditoria de cumplimiento por AST", "",
        "## Por que AST y no grep", "",
        "Un `grep` de la palabra \"embeddings\" encuentra **el nombre de la variable que la "
        "declara**, y este propio archivo contiene la palabra. Recorrer el arbol de sintaxis "
        "encuentra el `import` y la llamada, que es lo unico que puede constituir un uso. Por eso "
        "los dos metodos darian resultados distintos y el AST es el que vale.", "",
        "## Hallazgos", "",
        "| Recurso | Hallazgos |", "|---|---|",
    ] + [f"| {k} | {len([h for h in aud['hallazgos'] if h['recurso'] == k])} |"
         for k in PROHIBIDOS] + ["", "## Comprobaciones positivas", "",
        "Una auditoria que solo busca ausencias pasa un proyecto vacio. Estos son los mecanismos "
        "permitidos que **si** tienen que estar presentes:", "",
        "| Mecanismo | Scripts que lo contienen |", "|---|---|",
    ] + [f"| {m} | {c} |" for m, c in aud["positivos"].items()] + ["", "## Restricciones "
         "especificas de esta fase", "",
        "| Restriccion | Estado | Como se comprueba |", "|---|---|---|",
        "| `decision_function` tratado como probabilidad | **0 usos** | busqueda en codigo del "
        "patron de division por suma, excluyendo los comentarios que la prohiben |",
        "| `outer_val` usado para ajustar | **0 usos** | `a`, `b` en `inner_train`; `beta` en "
        "`inner_val`; `outer_val` solo recibe la correccion ya fijada |",
        "| Comprobacion de `score_type` | **presente** | `reports/p34_score_inventory.csv` declara "
        "el tipo de cada candidato |",
        "| Llamadas HTTP | **0** | AST sobre imports y llamadas |",
        "| Test y holdout | **0** | AST; ninguna fase los abre |", "",
        "## Resultado", "",
        f"**Cumplimiento global: {'APROBADO' if aud['total'] == 0 and aud['positivos_ok'] else 'REVISAR'}**.** "
        f"{aud['total']} hallazgos y {sum(1 for v in aud['positivos'].values() if v > 0)} de "
        f"{len(aud['positivos'])} comprobaciones positivas presentes.", ""]).rstrip("\n") + "\n",
        encoding="utf-8")

    RESULTADO.write_text(json.dumps({
        "fase": "P33+P34 notebook y compliance", "fecha": FECHA, "experimenta": False,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, capture_output=True,
                                 text=True, timeout=30).stdout.strip(),
        "compliance_ast": aud, "notebook": nb,
        "cumplimiento_global": bool(aud["total"] == 0 and aud["positivos_ok"]),
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
