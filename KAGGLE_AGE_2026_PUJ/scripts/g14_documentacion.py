"""G14 - documentacion persistente, notebook y registro del experimento.

## Que hace

G14 no es un experimento de entrenamiento: es una auditoria. Pero deja una decision que hay que
propagar a los cuatro documentos persistentes del proyecto, a `experiments.csv` y al notebook. Todo
se hace aqui, de forma idempotente, para que no haya dos versiones de la misma conclusion.

## Por que G14 entra en `experiments.csv` si no entrena nada

Porque el brief lo pide y porque la fila es informativa: registra que se **audito** la entrada y
que el resultado fue `NOT_ACTIONABLE`. Una auditoria que no deja fila es una auditoria que en tres
meses alguien repite desde cero. La fila declara explicitamente que no hubo entrenamiento, para no
inflar el presupuesto.
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

RESULTADO = cfg.EXPERIMENTS / "g14_documentacion.json"
NOTEBOOK = ROOT / "notebooks" / "99_entrega_consolidado.ipynb"
MARCADOR = "FASE G14"
FECHA = "2026-10-06"
CV_V30 = 0.6134471
LB_V30 = 0.63302
GRUPO_A = 27007
GRUPO_B = 38396

D_G15 = """
---

## D-068 - G14: no existe informacion de entrada legal sin utilizar, y el `id` no es rescatable

- **Observation**: cerradas las vias de correccion en score-space (P33, P34, G12), la unica linea
  con margen estructural que quedaba era G14: comprobar si existia un campo de entrada legal,
  presente en prediccion y que ningun modelo hubiera consumido. La duda concreta estaba en el
  campo `id`: si fuera un *snowflake* de Twitter, codificaria la fecha de publicacion, que es
  informacion real, legal y disponible en test.

- **El `id` no es un snowflake, y la prueba es geometrica antes que estadistica**: va de **1 a
  169.336** en train y de **1 a 112.892** en test, monotono y sin huecos, con **6 digitos** en el
  maximo. Un id de tweet tiene 18-19. Y hay un dato que cierra el caso por si solo: **los 112.892
  ids del test aparecen tambien en train**, es decir el 100 %, para filas distintas. El `id` 1 es
  un tweet distinto en cada split. Un campo cuyo valor identifica a dos filas con salidas
  distintas no puede ser feature: no aporta informacion sobre la fila.

- **La correlacion con la clase es -0,00084.** Nada, como corresponde a un indice de fila.

- **Auditoria del resto del esquema**: el ZIP de la competicion contiene exactamente tres ficheros
  (`age_train.csv`, `age_test.csv`, `age_submission.csv`), sin campos ocultos. Los tres campos son
  `text` (en uso en las nueve familias de modelo), `age_range` (target, prohibido como entrada) e
  `id` (administrativo). La matriz campo-por-modelo da: `text` USED o DERIVED en 9 de 9, `id`
  NOT_USED en 9 de 9, `age_range` PROHIBITED en 9 de 9.

- **Traza de `text`**: ninguna dimension queda en el limbo. Las que el TF-IDF no captura (case,
  puntuacion, elongaciones, posicion) fueron cubiertas por los diez canales de V31-03 y su
  integracion se cerro sin aporte en V31-06. Eso es el **caso C** del brief, informacion ya
  contenida o ya probada, no un canal nuevo.

- **Decision**: **G14 = NOT_ACTIONABLE**, y se declara por que no es `CLOSED`: `CLOSED` significa
  que habia un canal y estaba cubierto; `NOT_ACTIONABLE` significa que no hay canal, ni cubierto ni
  descubierto. **No se intenta rescatar el `id`** con modulo, hash, posicion, patrones de digitos,
  buckets ni sinonimos, como el brief prohibe expresamente: eso no seria encontrar informacion,
  seria ajustar una correlacion espuria sobre 169.336 filas.

- **Consequences**: la fase de investigacion queda con sus seis lineas cerradas (correccion de
  score global, local, pendiente, canal linguistico, capacidad del meta-modelo, diversidad de
  ensembles) y con la **entrada enumerada por completo**. La conclusion que se puede sostener es
  que **no existe actualmente informacion de entrada legal, observable en prediccion y no
  utilizada por los modelos, que pueda abrir una nueva linea experimental**. No es lo mismo que
  decir que V30 sea optimo: es decir que las vias que el protocolo permite estan recorridas. Y
  queda la cifra que mide el techo: de los 65.403 errores de V30, **27.007 no los corrige ninguna
  de las 26 fuentes disponibles**, y G14 anade que tampoco hay un campo sin explotar que pudiera
  corregirlos.
"""

ESTADO = """
---

## G14 - AUDITORIA DE INFORMACION DE ENTRADA (2026-10-06) - NOT_ACTIONABLE

- **Pregunta**: existe algun campo de entrada legal, presente en prediccion y que ningun modelo
  haya consumido?
- **Esquema real**: el ZIP contiene exactamente `age_train.csv` (`id`, `text`, `age_range`),
  `age_test.csv` (`id`, `text`) y `age_submission.csv`. Sin campos ocultos.
- **El `id` no es un snowflake**: 1..169.336 en train y 1..112.892 en test, **6 digitos** en el
  maximo, monotono y sin huecos. Y **el 100 % de los ids del test reaparece en train** para filas
  distintas: es un indice de fila por split, no un identificador de contenido. Correlacion con la
  clase: **-0,00084**.
- **Matriz campo-por-modelo**: `text` USED/DERIVED 9 de 9; `id` NOT_USED 9 de 9; `age_range`
  PROHIBITED 9 de 9.
- **Traza de `text`**: toda dimension no capturada por el TF-IDF (case, puntuacion, elongaciones,
  posicion) fue cubierta por los diez canales de V31-03 y cerrada por V31-06. Caso C, no canal
  nuevo.
- **Decision**: **NOT_ACTIONABLE**. No se rescata el `id` con modulo, hash, posicion ni patrones de
  digitos. Ver D-068.
- **Estado de la investigacion**: seis lineas cerradas y la **entrada enumerada por completo**. El
  techo esta medido: 27.007 de los 65.403 errores de V30 no los corrige ninguna de las 26 fuentes,
  y no hay campo sin explotar que pudiera hacerlo.
"""

RUNLOG = """

---

## Cierre de G14 - auditoria de informacion de entrada · 2026-10-06

- **Resultado:** `G14 = NOT_ACTIONABLE`. No existe campo de entrada legal, disponible en
  prediccion y sin usar.
- **Experimentos de entrenamiento: 0.** Es una auditoria, y la fila de `experiments.csv` lo declara.
- **El `id`:** indice de fila por split, 6 digitos, 100 % de los ids del test reaparecen en train,
  correlacion -0,00084. No se rescata.
- **Traza de `text`:** toda dimension no capturada por el TF-IDF ya fue probada por V31-03 y
  cerrada por V31-06.
- **Techo medido:** 27.007 de los 65.403 errores de V30 no los corrige ninguna de las 26 fuentes
  disponibles, y no hay campo de entrada que pudiera corregirlos.
- **Compliance:** 0 hallazgos. **Artefactos de V30 tocados:** 0.
"""


def _anade(ruta: Path, marca: str, bloque: str) -> str:
    """Anade un bloque si la marca no esta. Idempotente."""
    if not ruta.exists():
        return f"{ruta.name}: no existe"
    texto = ruta.read_text(encoding="utf-8")
    if marca in texto:
        return f"{ruta.name}: ya estaba"
    return f"{ruta.name}: +{len(bloque.splitlines())} lineas" if (
        ruta.write_text((texto if texto.endswith("\n") else texto + "\n") + bloque,
                        encoding="utf-8") or True) else ""


def anade_experiments() -> dict:
    """Registra G14 en experiments.csv como auditoria, sin contar como experimento."""
    ruta = ROOT / "experiments.csv"
    lineas = ruta.read_text(encoding="utf-8").splitlines()
    columnas = [c.strip() for c in lineas[0].split(",")]
    existentes = {ln.split(",", 1)[0] for ln in lineas[1:] if ln}
    registro = {
        "experiment_id": "G14-AUDIT", "fecha": FECHA, "bloque": "25.0",
        "etapa": "G14 auditoria de informacion de entrada",
        "pregunta": "existe algun campo de entrada legal, presente en prediccion y que ningun "
                    "modelo haya consumido?",
        "hipotesis": "el campo `id` podria codificar una fecha si fuera un id de tweet (snowflake)",
        "features": "ninguna: **0 experimentos de entrenamiento**, es una auditoria",
        "preprocessing": "sin cambios",
        "representacion": "n/a",
        "modelo": "n/a",
        "hiperparametros": "n/a",
        "validacion": "inspeccion directa de los ficheros y del pipeline; sin validacion cruzada "
                      "porque no hay modelo",
        "macro_f1_mean": "",
        "macro_f1_std": "", "weighted_f1_mean": "", "micro_f1_mean": "", "accuracy_mean": "",
        "resultado": "esquema: `id`, `text`, `age_range` en train y `id`, `text` en test. El ZIP "
                     "contiene exactamente esos 3 ficheros. `id` va de 1 a N por split, 6 digitos, "
                     "**el 100% de los ids del test reaparece en train** para filas distintas, "
                     "correlacion -0,00084 con la clase. `text` USED o DERIVED en 9 de 9 familias; "
                     "`id` NOT_USED en 9 de 9; `age_range` PROHIBITED en 9 de 9",
        "interpretacion": "el `id` es un indice administrativo de fila, no un identificador de "
                          "contenido: no puede aportar informacion sobre la fila. Y las "
                          "dimensiones de `text` no capturadas por el TF-IDF ya fueron probadas "
                          "por V31-03 y cerradas por V31-06",
        "decision": "NOT_ACTIONABLE. No se rescata el `id` con modulo, hash, posicion ni patrones "
                    "de digitos",
        "duracion_seg": "1.0",
        "artefactos": "experiments/g14_auditoria_input.json | "
                      "reports/g14_input_information_audit.md | reports/g14_decision.md",
    }
    if "G14-AUDIT" in existentes:
        return {"hecho": True, "ya_estaba": True}
    faltan = [c for c in columnas if c not in registro]
    if faltan:
        raise ValueError(f"faltan columnas en el registro de G14: {faltan}")

    def fila(reg):
        vals = [str(reg[c]) for c in columnas]
        return ",".join(f'"{v}"' if ("," in v or '"' in v) else v for v in vals)

    with ruta.open("a", encoding="utf-8") as f:
        f.write(fila(registro) + "\n")
    return {"hecho": True, "filas_tras": len(ruta.read_text(encoding="utf-8").splitlines()) - 1}


def anade_notebook() -> dict:
    """Anade la seccion G14 con los pasos del brief, leyendo de los artefactos."""
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

    md(f"""# {MARCADOR} - AUDITORIA DE INFORMACION DE ENTRADA

## La pregunta

Cerradas las vias de correccion en score-space (P33, P34, G12), la unica linea con margen
estructural que quedaba era esta: **existe algun campo de entrada legal, presente en prediccion y
que ningun modelo haya consumido?**

La duda concreta estaba en `id`. Si fuera un *snowflake* de Twitter, codificaria la fecha de
publicacion: informacion real, legal y disponible en test.

## La respuesta, y es no

`G14 = NOT_ACTIONABLE`. El esquema tiene tres campos y ninguno esta en la situacion buscada. El
`id` es un **indice de fila**, y la prueba es geometrica antes que estadistica.""")

    code("""import json, pathlib
import pandas as pd
EXP = pathlib.Path("../experiments")
g14 = json.loads((EXP / "g14_auditoria_input.json").read_text(encoding="utf-8"))
a = g14["auditoria_id"]
print("=== el campo id ===")
print(f"train  1 .. {a['train']['max']:,}  monotono {a['train']['monotono']}")
print(f"test   1 .. {a['test']['max']:,}  monotono {a['test']['monotono']}")
print(f"digitos del maximo: {a['digitos_del_maximo']}  -> un snowflake de Twitter tiene 18-19")
print(f"interseccion train-test: {a['interseccion_train_test']:,} "
      f"({a['fraccion_del_test_que_reaparece_en_train']:.0%} del test)")
print(f"correlacion con la clase: {a['correlacion_pearson_con_la_clase']:+.6f}")
print()
assert a["parece_snowflake_de_twitter"] is False
assert a["fraccion_del_test_que_reaparece_en_train"] == 1.0
assert abs(a["correlacion_pearson_con_la_clase"]) < 0.01
print("[OK] el id es un indice administrativo, no una fuente de informacion")""")

    code("""# El esquema completo, y el uso real de cada campo
print(pd.read_csv("../reports/g14_input_schema_inventory.csv")[
    ["campo", "split", "tipo", "filas", "cardinalidad", "rol", "decision"]
].to_string(index=False))
print()
print(pd.read_csv("../reports/g14_field_x_model_matrix.csv")
      .groupby(["input_field", "estado"]).size().to_string())""")

    code("""# Traza de text: ninguna dimension queda en el limbo
print(pd.read_csv("../reports/g14_text_traceability.csv")[
    ["dimension", "se_conserva", "probado_en", "estado"]
].to_string(index=False, max_colwidth=44))""")

    code("""# El techo de lo que queda
print("grupo A (nadie corrige):", f"{g14['grupo_a_y_grupo_b']['A_nadie_corrige']:,}")
print("grupo B (alguien corrige):", f"{g14['grupo_a_y_grupo_b']['B_alguien_corrige']:,}")
print()
print("las dos vias estan cerradas por motivos DISTINTOS y conviene no confundirlas:")
print("  - el grupo A no esta cerrado por falta de explotacion")
print("  - esta cerrado porque la informacion no existe en la entrada disponible")
print()
print("G14 =", g14["decision"])""")

    md("""## Lo que no se hace

El brief anticipa este caso y prohibe rescatarlo. **No** se busca modulo, hash, posicion, patrones
de digitos, buckets ni sinonimos. La razon es que un numero de fila no contiene informacion; lo que
ofrece es la oportunidad de ajustar una correlacion espuria sobre 169.336 filas, que es justo lo
que el protocolo corregido del proyecto lleva veinte fases evitando.

## Estado tras G14

| Linea | Estado |
|---|---|
| Correccion de score global (P33) | CLOSED |
| Correccion de score local por pares (P34) | INVALIDATED |
| Correccion por pendiente en el margen (G12) | CLOSED, y baja el AUC |
| Canal linguistico como feature (V31) | CLOSED |
| Capacidad del meta-modelo (V29, V30) | CLOSED |
| Diversidad de ensembles historicos (V29) | CLOSED |
| **Informacion de entrada nueva (G14)** | **NOT_ACTIONABLE** |

Con las seis lineas cerradas y **la entrada enumerada por completo**, la conclusion sostenible es
que no existe actualmente informacion de entrada legal, observable en prediccion y no utilizada,
que pueda abrir una nueva linea experimental. No es decir que V30 sea optimo: es decir que las vias
que el protocolo permite estan recorridas.""")

    nbformat.write(nb, NOTEBOOK)
    return {"anadida": True, "antes": antes, "despues": len(nb.cells),
            "anadidas": len(nb.cells) - antes, "idempotente": True}


def main() -> int:
    """Propaga la decision de G14 a la documentacion y al notebook."""
    autotest_numerico()
    inicio = time.time()
    print("=== documentacion persistente ===", flush=True)
    for ruta, marca, bloque in (
        (ROOT / "DECISIONS.md", "## D-068", D_G15),
        (ROOT / "PROJECT_STATE.md", "G14 - AUDITORIA DE INFORMACION DE ENTRADA", ESTADO),
        (ROOT / "RUNLOG.md", "Cierre de G14", RUNLOG),
        (ROOT / "FINAL_REPORT.md", "G14 - NOT_ACTIONABLE", ESTADO),
    ):
        print("  " + _anade(ruta, marca, bloque), flush=True)
    print("\n=== experiments.csv ===", flush=True)
    print("  " + str(anade_experiments()), flush=True)
    print("\n=== notebook ===", flush=True)
    print("  " + str(anade_notebook()), flush=True)

    RESULTADO.write_text(json.dumps({
        "fase": "G14 documentacion", "fecha": FECHA, "experimenta": False,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, capture_output=True,
                                 text=True, timeout=30).stdout.strip(),
        "decision": "NOT_ACTIONABLE",
        "documentos": ["DECISIONS.md (D-068)", "PROJECT_STATE.md", "RUNLOG.md", "FINAL_REPORT.md",
                       "experiments.csv (G14-AUDIT)", "notebook"],
        "experimentos_de_entrenamiento": 0,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
