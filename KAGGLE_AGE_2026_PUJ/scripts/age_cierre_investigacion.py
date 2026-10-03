"""Cierre de la investigacion de AGE: documentacion, notebook y registro.

## La conclusion

```text AGE_RESEARCH_CLOSED ```

No quedan mecanismos de aprendizaje metodologicamente defendibles y no recorridos dentro de las
restricciones del proyecto. La afirmacion es sobre el espacio auditado, y el espacio esta
enumerado: 39 mecanismos, 6 familias de representacion, 10 de modelo, la entrada completa y las
cinco vias de integracion del score.

## Dos correcciones que la auditoria encontro, y se declaran

1. **La particion A/B del gap analysis tenia un enmascarado defectuoso.** Reporto A = 27.007 y el
   valor correcto es **27.430**. El defecto venia de excluir a los miembros de V30 de su propio
   rescate con un encadenado de mascaras enrevesado; aqui se hace comparando por prefijo del
   nombre, que es directo y verificable. La diferencia es de 423 filas y no cambia ninguna
   conclusion, pero se declara.

2. **La definicion de "rescate exclusivo" era distinta.** El inventario previo registro 122; con la
   definicion de esta auditoria -una sola de las 26 fuentes acierta la fila- hay **8.029**. Los dos
   numeros miden cosas distintas y ninguno es erroneo. Se declara cual se usa y por que.

## Lo que este script NO hace

No toca V30, no entrena nada, no usa el test, no toca el holdout y no consulta el leaderboard.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "age_cierre_investigacion.json"
NOTEBOOK = ROOT / "notebooks" / "99_entrega_consolidado.ipynb"
MARCADOR = "CIERRE DE LA INVESTIGACION DE AGE"
FECHA = "2026-10-07"
CV_V30 = 0.6134471
CV_V30_SEG = 0.6143347
LB_V30 = 0.63302
GRUPO_A = 27430
GRUPO_B = 37973
UNICOS = 8358
COMPARTIDOS = 30038
EXCLUSIVOS = 8029

D = """
---

## D-069 - Cierre de la investigacion de AGE: `AGE_RESEARCH_CLOSED`

- **Observation**: G14 cerro el **input gap** (no hay campos de entrada legal sin usar). Quedaba
  abierta la pregunta distinta del **mechanism gap**: si existe alguna forma de APRENDER de los
  mismos inputs que no se haya probado. La auditoria se hizo sobre 39 mecanismos buscados en el
  codigo **por operacion**, no por nombre de libreria.

- **La medicion que fundamenta la decision** refina la particion A/B del gap analysis y anade una
  distincion que faltaba:

  | Grupo | Filas | % |
  |---|---|---|
  | A: nadie corrige | 27.430 | 41,9 % |
  | B: alguien corrige | 37.973 | 58,1 % |
  | - B1 rescate **unico** | 8.358 | 12,8 % |
  | - B2 rescate **compartido** | 30.038 | 45,9 % |

  Mediana de fuentes por error rescatado: **5**, con un maximo de 25. **B2 domina**, y eso importa
  porque un rescate compartido significa que la informacion esta en varias fuentes a la vez y el
  problema es de integracion; solo un rescate unico apuntaria a un problema de mecanismo.

- **Los rescates exclusivos no tienen estructura comun**, y se midio en vez de suponerse: 8.029
  filas repartidas entre **23 fuentes distintas**, con la mas frecuente concentrando solo el
  **20,5 %**. Longitud media 81,6 frente a 84,2 de todos los errores; confianza de V30 0,429 frente
  a 0,442; fraccion con URL 41,5 % frente a 41,3 %. **Indistinguibles en toda dimension medida.**
  Veredicto: heterogeneidad, que es la firma de ruido de modelo y no de un canal sin explotar.

- **De 39 mecanismos, 23 estan usados y 9 cerrados.** Los cinco que no aparecen en el codigo se
  cierran con la medicion delante, no por inercia: **focal loss**, **hard example mining** y
  **label smoothing** apuntan a los ejemplos dificiles y a la frontera, que es la poblacion A
  (27.430 filas sin ninguna fuente que las corrija) y donde P34 midio que el ruido aleatorio gana a
  la senal real. **ranking objective** no mueve la metrica evaluada. **bootstrap de grupos** es
  reduccion de varianza con otra unidad de remuestreo, y el bagging del meta-modelo ya se cerro en
  -0,000312.

- **Las cinco vias de integracion del score estan cerradas con control de atribucion**:

  | Via | Resultado |
  |---|---|
  | feature anadida al meta-espacio (V31-06) | +0,0002948 sobre el control barajado |
  | miembro de la mezcla (V31-05) | +0,0000642 sobre el reoptimizado |
  | residuo global (P33) | -0,0002152; el ruido aleatorio gana |
  | residuo local por pares (P34) | -0,0002039; cero rescates de alta confianza |
  | residuo con pendiente (G12) | +0,0003018 sin superar a la constante, y baja el AUC |

  En las cinco el control gana o empata. **No es que la integracion se intentara poco: se intento
  cinco veces con la misma conclusion.**

- **Decision**: **`AGE_RESEARCH_CLOSED`**. V30 permanece como mejor resultado validado: 0,6134471
  de corrected_cv, 0,6143347 en la particion alterna y **0,63302** en el leaderboard.

- **Dos correcciones que la auditoria encontro, y se declaran**: (a) la particion del gap analysis
  reportaba A = 27.007 y el valor correcto es **27.430**; el defecto estaba en un enmascarado
  enrevesado para excluir a los miembros de V30 de su propio rescate. (b) La definicion de "rescate
  exclusivo" del inventario previo daba 122; con la de aqui da **8.029**. Los dos numeros miden
  cosas distintas y se declara cual se usa.

- **Consequences**: la conclusion es **sobre el espacio auditado**, y el espacio esta enumerado: 39
  mecanismos, 6 familias de representacion, 10 familias de modelo, la entrada completa y las cinco
  vias de integracion. Lo que **no** se afirma: que V30 sea optimo; que no exista ninguna mejora
  posible; que no exista informacion linguistica (P33 midio AUC 0,5411 en el residuo). Lo que si se
  afirma, con la medicion a la vista, es que **no quedan actualmente mecanismos de aprendizaje
  metodologicamente defendibles y no recorridos dentro de las restricciones del proyecto**. Para que
  algo nuevo funcionara haria falta informacion que no esta en los datos: metadatos de cuenta
  (prohibidos por ser una tarea de texto), contenido de URLs (no hay ninguna de status) o modelos
  semanticos (prohibidos por las reglas).
"""

ESTADO = f"""
---

## CIERRE DE LA INVESTIGACION DE AGE (2026-10-07) - `AGE_RESEARCH_CLOSED`

- **Baseline final**: V30 = **{CV_V30}** corrected_cv, **{CV_V30_SEG}** en la particion alterna,
  **{LB_V30}** en el leaderboard. Hash del submission `f543a715...` sin cambio.
- **Input gap**: cerrado por G14 (`NOT_ACTIONABLE`).
- **Mechanism gap**: cerrado por esta auditoria. **39 mecanismos auditados**, 23 usados, 9
  cerrados, 1 invalidado, 1 no accionable y 5 sin usar que se cierran con evidencia.
- **Particion de errores**: A = **{GRUPO_A:,}** (41,9 %) sin ninguna fuente que los corrija; B =
  **{GRUPO_B:,}** (58,1 %) con al menos una. De B, **{UNICOS:,}** son rescates unicos y
  **{COMPARTIDOS:,}** compartidos: **la informacion esta repartida**, no encerrada en una fuente.
- **Rescates exclusivos**: **{EXCLUSIVOS:,}** entre **23 fuentes**, concentracion 20,5 %, sin
  estructura comun en longitud, confianza ni URL. Es ruido de modelo.
- **Cinco vias de integracion cerradas con control**: feature, miembro de mezcla, residuo global,
  residuo local y residuo con pendiente. En las cinco el control gana o empata.
- **Conclusion**: no quedan mecanismos de aprendizaje metodologicamente defendibles y no recorridos
  dentro de las restricciones del proyecto. Ver D-069.
- **Correcciones declaradas**: A = 27.430, no 27.007; rescates exclusivos = 8.029 con esta
  definicion, no 122 con la del inventario previo.
"""

RUNLOG = f"""

---

## Cierre de la investigacion de AGE · 2026-10-07

- **Conclusion:** `AGE_RESEARCH_CLOSED`.
- **Mecanismos auditados:** 39. Usados 23, cerrados 9, invalidado 1, no accionable 1, sin usar 5
  (todos cerrados con evidencia).
- **Experimentos de entrenamiento: 0.** Es una auditoria.
- **Particion de errores:** A = {GRUPO_A:,} (41,9 %), B = {GRUPO_B:,} (58,1 %). B1 unico =
  {UNICOS:,}; B2 compartido = {COMPARTIDOS:,}. Mediana de fuentes por rescate: 5.
- **Rescates exclusivos:** {EXCLUSIVOS:,} entre 23 fuentes, sin estructura comun.
- **Correcciones declaradas:** A correcto = 27.430 (el gap analysis reportaba 27.007); rescates
  exclusivos = 8.029 con esta definicion.
- **Compliance:** 0 hallazgos. **Artefactos de V30 tocados:** 0. **Submissions: 0.**
"""


def _anade(ruta: Path, marca: str, bloque: str) -> str:
    if not ruta.exists():
        return f"{ruta.name}: no existe"
    texto = ruta.read_text(encoding="utf-8")
    if marca in texto:
        return f"{ruta.name}: ya estaba"
    ruta.write_text((texto if texto.endswith("\n") else texto + "\n") + bloque, encoding="utf-8")
    return f"{ruta.name}: +{len(bloque.splitlines())} lineas"


def anade_experiments() -> dict:
    """Registra la auditoria en experiments.csv, declarando que no entrena nada."""
    ruta = ROOT / "experiments.csv"
    lineas = ruta.read_text(encoding="utf-8").splitlines()
    columnas = [c.strip() for c in lineas[0].split(",")]
    existentes = {ln.split(",", 1)[0] for ln in lineas[1:] if ln}
    if "AUDIT-COVERAGE" in existentes:
        return {"hecho": True, "ya_estaba": True}
    reg = {
        "experiment_id": "AUDIT-COVERAGE", "fecha": FECHA, "bloque": "26.0",
        "etapa": "auditoria de cobertura metodologica",
        "pregunta": "queda algun mecanismo de aprendizaje genuinamente nuevo que pueda convertir "
                    "informacion ya disponible en una correccion robusta?",
        "hipotesis": "G14 cerro el input gap pero no necesariamente el mechanism gap",
        "features": "ninguna: **0 experimentos de entrenamiento**, es una auditoria",
        "preprocessing": "sin cambios", "representacion": "n/a", "modelo": "n/a",
        "hiperparametros": "n/a",
        "validacion": "auditoria sobre el OOF congelado de V30 y las 26 fuentes disponibles; sin "
                      "entrenamiento",
        "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "", "micro_f1_mean": "",
        "accuracy_mean": "",
        "resultado": "39 mecanismos auditados: 23 USED, 9 CLOSED, 1 INVALIDATED, 1 NOT_ACTIONABLE, "
                     "5 NOT USED. Particion: A=27.430 (41,9%), B=37.973 (58,1%); B1 unico 8.358, "
                     "B2 compartido 30.038. Rescates exclusivos: 8.029 entre 23 fuentes, "
                     "concentracion 20,5%, sin estructura comun en longitud ni confianza ni URL",
        "interpretacion": "los cinco mecanismos sin usar apuntan a los ejemplos dificiles y a la "
                          "frontera, que es la poblacion A donde ninguna de 26 fuentes tiene "
                          "senal, y a la zona donde P34 midio que el ruido gana. Las cinco vias de "
                          "integracion del score estan cerradas con control de atribucion",
        "decision": "AGE_RESEARCH_CLOSED: no quedan mecanismos metodologicamente defendibles y no "
                    "recorridos dentro de las restricciones del proyecto",
        "duracion_seg": "2.0",
        "artefactos": "experiments/age_auditoria_cobertura.json | "
                      "reports/age_final_methodology_audit.md | "
                      "reports/age_methodology_coverage_matrix.csv",
    }
    faltan = [c for c in columnas if c not in reg]
    if faltan:
        raise ValueError(f"faltan columnas: {faltan}")

    def fila(r):
        vals = [str(r[c]) for c in columnas]
        return ",".join(f'"{v}"' if ("," in v or '"' in v) else v for v in vals)

    with ruta.open("a", encoding="utf-8") as f:
        f.write(fila(reg) + "\n")
    return {"hecho": True, "filas_tras": len(ruta.read_text(encoding="utf-8").splitlines()) - 1}


def anade_notebook() -> dict:
    import nbformat
    nb = nbformat.read(NOTEBOOK, as_version=4)
    antes = len(nb.cells)
    for i, c in enumerate(nb.cells):
        if c.cell_type == "markdown" and MARCADOR in c.source:
            del nb.cells[i:]
            break

    def md(t):
        nb.cells.append(nbformat.v4.new_markdown_cell(t))

    def code(t):
        nb.cells.append(nbformat.v4.new_code_cell(t))

    md(f"""# {MARCADOR}

## `AGE_RESEARCH_CLOSED`

G14 cerro el **input gap**. Esta seccion cierra el **mechanism gap**: si existe alguna forma
distinta de *aprender* de los mismos inputs que no se haya probado.

La respuesta, con la evidencia delante, es que no.

## La particion de los errores, refinada

El gap analysis dejo dos grupos. Aqui se anade una distincion que faltaba y que es la que decide la
pregunta: **un rescate compartido significa que la informacion esta en varias fuentes a la vez, y
el problema es de integracion. Un rescate unico significa que solo esa fuente la tiene, y el
problema es de mecanismo.** Solo la segunda justificaria buscar un mecanismo nuevo.""")

    code("""import json, pathlib
import pandas as pd
EXP = pathlib.Path("../experiments")
a = json.loads((EXP / "age_auditoria_cobertura.json").read_text(encoding="utf-8"))
p = a["particion"]
print("=== particion de los 65.403 errores de V30 ===")
print(f"A  nadie corrige        {p['A_nadie_corrige']:>7,}  {p['A_nadie_corrige']/p['errores_v30']:.1%}")
print(f"B  alguien corrige      {p['B_alguien_corrige']:>7,}  {p['B_alguien_corrige']/p['errores_v30']:.1%}")
print(f"   B1 rescate UNICO     {p['B1_rescate_unico']:>7,}")
print(f"   B2 rescate COMPARTIDO{p['B2_rescate_compartido']:>7,}")
print()
print("mediana de fuentes por error rescatado:", p['mediana_de_fuentes_por_error_rescatado'])
print("maximo de fuentes que rescatan una fila:", p['max_fuentes_que_rescatan_una_fila'])
print()
print("B2 domina: la informacion esta REPARTIDA, no encerrada en una fuente")
assert p['B2_rescate_compartido'] > p['B1_rescate_unico']
print("[OK]")""")

    code("""# Los rescates exclusivos, que era la pregunta central: ¿hay estructura comun?
e = a["rescates_exclusivos"]
print(f"rescates exclusivos: {e['n']:,}")
print(f"fuentes distintas: {e['fuentes_distintas']}")
print(f"concentracion de la fuente mas frecuente: {e['concentracion_fuente_top']:.1%}")
print()
print("| medida | exclusivos | todos los errores |")
print("|---|---|---|")
print(f"| longitud media | {e['longitud_media']} | {e['longitud_media_de_todos_los_errores']} |")
print(f"| confianza V30 | {e['confianza_media_v30']} | {e['confianza_media_de_todos_los_errores']} |")
print(f"| fraccion con URL | {e['fraccion_con_url']:.1%} | {e['fraccion_con_url_en_todos_los_errores']:.1%} |")
print()
print("veredicto:", e["estructura"])
print(e["lectura"])
assert e["concentracion_fuente_top"] < 0.5
print()
print("[OK] no hay estructura comun: es ruido de modelo, no un canal sin explotar")""")

    md("""## La cobertura de mecanismos

Los 39 mecanismos se buscaron en el codigo **por operacion**, no por nombre de libreria: lo que se
registra es que el mecanismo se haya ejecutado, no que la libreria este instalada.""")

    code("""# La matriz vive en su propio artefacto: el de errores mide la particion y el de cobertura
# lleva los 39 mecanismos. Se leen los dos, y no se mezcla uno con el otro.
cob = json.loads((EXP / "age_cobertura_metodologica.json").read_text(encoding="utf-8"))
m = pd.DataFrame(cob["matriz"])
print(m["status"].value_counts().to_string())
print()
sin_usar = m[m["status"] == "NOT USED"]
for _, f in sin_usar.iterrows():
    print(f"{f['ID']} {f['mechanism']:26s} {f['novelty']}")
print()
print("los cinco se cierran con la medicion delante, no por inercia:")
for _, f in sin_usar.iterrows():
    print(f"  - {f['mechanism']}: {f['why'][:110]}")""")

    md("""## Lo que esta conclusion NO dice

- **No** dice que V30 sea optimo.
- **No** dice que no exista ninguna mejora posible.
- **No** dice que no exista informacion linguistica: P33 midio AUC 0,5411 en el residuo.

Dice algo mas estrecho: **no quedan actualmente mecanismos de aprendizaje metodologicamente
defendibles y no recorridos dentro de las restricciones del proyecto.**

## Para que algo nuevo funcionara

No es un mecanismo: haria falta **informacion que no esta en los datos disponibles**.

| Fuente que lo aportaria | Estado |
|---|---|
| Metadatos de la cuenta (fecha de registro, ubicacion, historial) | prohibidos: la tarea es de texto |
| El contenido de las URLs | no hay ninguna URL de status; las 1.685 truncadas no resuelven |
| Modelos semanticos (embeddings, transformers) | prohibidos por las reglas de la competencia |

## Estado final

| | |
|---|---|
| V30 corrected_cv | **0,6134471** |
| V30 particion alterna | 0,6143347 |
| V30 leaderboard | **0,63302** |
| Submissions de esta investigacion | 0 |
| Mecanismos auditados | 39 |
| Genuinamente nuevos | **0** |""")

    nbformat.write(nb, NOTEBOOK)
    return {"anadida": True, "antes": antes, "despues": len(nb.cells),
            "anadidas": len(nb.cells) - antes, "idempotente": True}


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    print("=== documentacion persistente ===", flush=True)
    for ruta, marca, bloque in (
        (ROOT / "DECISIONS.md", "## D-069", D),
        (ROOT / "PROJECT_STATE.md", "CIERRE DE LA INVESTIGACION DE AGE", ESTADO),
        (ROOT / "RUNLOG.md", "Cierre de la investigacion de AGE", RUNLOG),
        (ROOT / "FINAL_REPORT.md", "CIERRE DE LA INVESTIGACION DE AGE", ESTADO),
    ):
        print("  " + _anade(ruta, marca, bloque), flush=True)
    print("\n=== experiments.csv ===", flush=True)
    print("  " + str(anade_experiments()), flush=True)
    print("\n=== notebook ===", flush=True)
    print("  " + str(anade_notebook()), flush=True)
    RESULTADO.write_text(json.dumps({
        "fase": "cierre de la investigacion", "fecha": FECHA, "experimenta": False,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, capture_output=True,
                                 text=True, timeout=30).stdout.strip(),
        "conclusion": "AGE_RESEARCH_CLOSED",
        "documentos": ["DECISIONS.md (D-069)", "PROJECT_STATE.md", "RUNLOG.md", "FINAL_REPORT.md",
                       "experiments.csv (AUDIT-COVERAGE)", "notebook"],
        "experimentos_de_entrenamiento": 0,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
