"""Revision bibliografica: documentacion persistente y seccion del notebook.

## Lo que hace

Propaga el resultado `NO_NEW_METHOD` a los documentos persistentes y anade la seccion de
literatura al notebook. **No entrena nada y no toca V30.**

## La decision de no ejecutar C1, y por que se documenta

El brief pide decidir si existe una razon cientifica para **reabrir** la investigacion. La
revision encuentra un unico candidato con hueco de ejecucion real, y su expectativa honesta es
que **no cruce el gate**. Eso es un experimento de falsacion, y el brief no lo pide: pide decidir
si reabrir. La respuesta es no, y queda escrita con las dos razones: la evidencia publicada
empuja contra la metrica, y el mecanismo es un superconjunto de algo ya descartado.

Se deja el candidato **registrado y no ejecutado**, con su control ya disenado, para que la
decision se pueda revisar sin rehacer la busqueda.
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

RESULTADO = cfg.EXPERIMENTS / "age_revision_documentacion.json"
NOTEBOOK = ROOT / "notebooks" / "99_entrega_consolidado.ipynb"
MARCADOR = "REVISION BIBLIOGRAFICA"
FECHA = "2026-10-08"
CV_V30 = 0.6134471
LB_V30 = 0.63302

D = """
---

## D-070 - Revision bibliografica: `NO_NEW_METHOD`

- **Observation**: cerrada la linea experimental, la pregunta que quedaba era si existe literatura
  que revele un mecanismo no probado. Se reviso el estado del arte de author profiling, ordinal
  classification y edad a partir de texto (ACL Anthology, EMNLP, COLING, IEEE TKDE, PRL,
  Information Fusion, CLEO/PAN, arXiv 2020-2026).

- **El hallazgo central es un desajuste de metrica, y decide la revision**:

  ```text
  macro-F1 es NOMINAL: trata todos los errores igual.
  La literatura ordinal optimiza metricas DISTANCIA-CONSCIENTES: MAE, QWK, RPS, 1-off.
  ```

  Dos fuentes convergentes lo dicen de forma explicita: **Vargas et al., IEEE TKDE 2026** (12
  datasets ordinales, 20 configuraciones) encuentra que el etiquetado suave mejora la
  generalizacion ordinal pero **la cross-entropy categorica rinde mejor en promedio en metricas
  nominales**; y **Kasai et al., arXiv 2405.11775** (texto, comparativa explicita de perdidas)
  afirma que **hay un trade-off: la mejora ordinal se paga con metricas nominales**.

- **La validacion contra el repositorio encontro huecos de ejecucion reales**, y se declaran en
  vez de taparlos: AGE **nunca corrio** un target suave ni `label smoothing` (**0 scripts, 0
  menciones**), ni EMD/Wasserstein (**0 scripts**), ni una proyeccion de features fuera de
  direcciones de topico (**0 scripts**). El proyecto entero tiene **solo dos perdidas custom**:
  margen (V18-P3) y destilacion (V18-P5). Es decir, la `label smoothing` que la auditoria dijo
  "cerrada" lo estaba **por argumento, no por medicion**.

- **Dos recuentos iniciales fueron falsos positivos de la propia busqueda**, y se declaran:
  `complement` matcheaba "complementario" en el texto en espanol (391 veces) y `sord` matcheaba
  `tensordot` (15 veces).

- **Los cinco candidatos, y por que ninguno justifica reabrir**:

  | ID | Mecanismo | Veredicto |
  |---|---|---|
  | C1 | perdidas distancia-conscientes (SORD, triangular/beta, OLL/MLL) | unico con hueco real, pero la evidencia empuja contra macro-F1 y es un **superconjunto de label smoothing** |
  | C2 | ruido ordinal explicito (Ord-NLL) | el eje mas no auditado, y **colapsa a C1 en expectativa**: su propio paper demuestra la equivalencia |
  | C3 | regularizacion unimodal (ORCU) | su ganancia medida es de **calibracion**; no supera a su componente de soft labels |
  | C4 | separacion estilo/contenido (DePro, INLP) | la ganancia es **estrictamente OOD** (DePro pierde in-distribution) y el CV de AGE es IID |
  | C5 | geometria ordinal (CPL, Mahalanobis) | ganancia atada a un extractor preentrenado (VGG-16) |

- **La duplicacion mas contundente es algebraica**: `EMD2(one-hot) = suma_k (F_pred(k) -
  1[y<=k])^2`, es decir el objetivo que la literatura presenta como ordinal distinto **ES** la
  familia acumulativa CORAL/all-threshold con perdida L2, cerrada en V22/V23/H10/D-058.

- **Decision**: **`NO_NEW_METHOD`**. Ningun candidato cumple a la vez las tres condiciones del
  brief: compatible con el reglamento, no equivalente a algo cerrado, y con evidencia de mejora de
  macro-F1 en clasificacion de edad a partir de texto con modelos clasicos. **C1 queda registrado y
  NO ejecutado**, con su control ya disenado, porque su expectativa honesta es no cruzar el gate y
  el brief pide decidir si reabrir, no gastar en una falsacion.

- **Consequences**: el mapa de mecanismos queda auditado **dos veces**: contra el codigo del
  proyecto (39 mecanismos) y contra la literatura 2020-2026. La revision aporta cinco piezas de
  evidencia de cierre que antes no existian: el argumento algebraico de EMD, la distincion entre
  calibracion y forma unimodal, la equivalencia en expectativa Ord-NLL/soft labels, la advertencia
  documentada del trade-off nominal/ordinal, y el eje de ruido ordinal con su contra-argumento de
  mismo-proceso. Lo que **no** se afirma: que V30 sea optimo, ni que no exista ninguna mejora
  posible. Lo que si: la literatura no ofrece un mecanismo que AGE no haya cubierto o que no
  empuje contra su metrica.
"""

ESTADO = f"""
---

## REVISION BIBLIOGRAFICA (2026-10-08) - `NO_NEW_METHOD`

- **Baseline**: V30 = **{CV_V30}** corrected_cv, leaderboard **{LB_V30}**, sin cambios.
- **Pregunta**: existe un mecanismo publicado, compatible con el reglamento, genuinamente distinto
  de las 39 familias auditadas y con evidencia de mejora en macro-F1 sobre texto?
- **Hallazgo central**: **macro-F1 es una metrica nominal y la literatura ordinal optimiza metricas
  distancia-conscientes**. Vargas et al. (IEEE TKDE 2026) y Kasai et al. (arXiv 2405.11775)
  confirman el trade-off de forma explicita: la mejora ordinal se paga con metricas nominales.
- **Huecos de ejecucion reales, declarados**: AGE nunca corrio targets suaves ni label smoothing
  (0 scripts), ni EMD (0), ni proyeccion de topicos (0). Solo hay 2 perdidas custom en todo el
  proyecto: margen (V18-P3) y destilacion (V18-P5).
- **Cinco candidatos evaluados, ninguno justifica reabrir**: C1 (perdidas distancia-conscientes),
  C2 (ruido ordinal), C3 (unimodal), C4 (estilo/contenido), C5 (geometria). C1 es el unico con
  hueco real; se registra y **no se ejecuta**.
- **Duplicacion algebraica**: EMD2 es la familia acumulativa CORAL con perdida L2.
- **Registros**: 5 candidatos, 21 duplicaciones, 25 metodos rechazados con motivo.
- **Conclusion**: no existe razon cientifica para reabrir. Ver D-070.
"""

RUNLOG = """

---

## Revision bibliografica · 2026-10-08

- **Resultado:** `NO_NEW_METHOD`.
- **Entrenamiento: 0.** Submissions: 0. Test y holdout: sin tocar.
- **Hallazgo central:** desajuste de metrica. macro-F1 es nominal; la literatura ordinal optimiza
  metricas distancia-conscientes y el trade-off esta documentado en texto.
- **Huecos de ejecucion declarados:** 0 scripts con targets suaves, 0 con EMD, 0 con proyeccion de
  topicos. La "label smoothing cerrada" lo estaba por argumento, no por medicion.
- **Candidatos:** 5 evaluados, 1 con hueco real (C1), registrado y no ejecutado.
- **Duplicaciones:** 21 pares. La mas contundente es algebraica: EMD2 = familia acumulativa CORAL.
- **Rechazados:** 25 metodos con motivo documentado.
- **Compliance:** 0 hallazgos. **Artefactos de V30 tocados:** 0.
"""


def _anade(ruta: Path, marca: str, bloque: str) -> str:
    if not ruta.exists():
        return f"{ruta.name}: no existe"
    texto = ruta.read_text(encoding="utf-8")
    if marca in texto:
        return f"{ruta.name}: ya estaba"
    ruta.write_text((texto if texto.endswith("\n") else texto + "\n") + bloque, encoding="utf-8")
    return f"{ruta.name}: +{len(bloque.splitlines())} lineas"


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

    md("""# REVISION BIBLIOGRAFICA ORIENTADA A MECANISMOS

## `NO_NEW_METHOD`

Cerrada la linea experimental, quedaba una pregunta: existe literatura que revele un mecanismo
que AGE no haya probado? Se reviso el estado del arte de author profiling, clasificacion ordinal y
edad a partir de texto (ACL Anthology, EMNLP, COLING, IEEE TKDE, PR, Information Fusion, PAN,
arXiv 2020-2026).

La respuesta es no, y la razon es un desajuste de metrica.

## El hallazgo central

```text
macro-F1 es una metrica NOMINAL: trata todos los errores igual.
La literatura ordinal optimiza metricas DISTANCIA-CONSCIENTES: MAE, QWK, RPS, 1-off.
```

Dos fuentes convergentes lo dicen de forma explicita:

- **Vargas et al., IEEE TKDE 2026** (12 datasets ordinales, 20 configuraciones): el etiquetado
  suave mejora la generalizacion ordinal, pero **la cross-entropy categorica rinde mejor en
  promedio en metricas nominales**.
- **Kasai et al., arXiv 2405.11775** (texto): *"CE performs best in terms of nominal metrics; OLL
  performs best in terms of ordinal metrics; hay un trade-off"*.

## La validacion contra el repositorio, que encontro huecos reales

Esto es incomodo y se declara: la validacion encontro que algunos mecanismos que la auditoria
llamo "cerrados" lo estaban **por argumento, no por medicion**.""")

    code("""import json, pathlib
import pandas as pd
EXP = pathlib.Path("../experiments")
rev = json.loads((EXP / "age_revision_bibliografica.json").read_text(encoding="utf-8"))
v = rev["validacion_contra_repositorio"]
print("=== ¿AGE ejecuto alguna vez estos mecanismos? ===")
print(f"soft targets o label smoothing : {v['soft_targets_o_label_smoothing_ejecutados']} scripts")
print(f"EMD / Wasserstein              : {v['emd_wasserstein_ejecutados']} scripts")
print(f"proyeccion de topicos          : {v['proyeccion_de_topicos_ejecutada']} scripts")
print(f"perdidas custom en el proyecto : {v['perdidas_custom_en_todo_el_proyecto']}")
for p in v["cuales"]:
    print(f"   - {p}")
print()
print("falsos positivos de mi propia busqueda, declarados:")
for f in v["falsos_positivos_declarados"]:
    print(f"   - {f}")
print()
print("CORAL cerrado:", v["coral_cerrado"], "| intercepto optimo:", v["intercepto_optimo"])
assert v["soft_targets_o_label_smoothing_ejecutados"] == 0
print()
print("[OK] huecos de ejecucion confirmados: no se corrieron")""")

    code("""# Los cinco candidatos y su veredicto
c = pd.DataFrame(rev["candidatos"])
for _, f in c.iterrows():
    print(f"{f['id']}: {f['nombre'][:78]}")
    print(f"   vector: {f['vector']}")
    print(f"   -> {f['recomendacion'][:150]}")
    print()""")

    code("""# Duplicaciones y rechazados
dup = pd.read_csv("../reports/age_literature_gap_registry.csv")
exc = pd.read_csv("../reports/age_literature_excluded_methods.csv")
print(f"{len(dup)} duplicaciones paper -> metodo existente en AGE")
print(f"{len(exc)} metodos rechazados con motivo")
print()
print("las tres duplicaciones que deciden:")
for m in ("EMD2", "Ord-NLL", "SORD"):
    fila = dup[dup["paper_method"].str.contains(m, case=False, na=False)]
    if len(fila):
        f = fila.iloc[0]
        print(f"  {f['paper_method'][:52]} -> {f['AGE_existing_method'][:48]}")
        print(f"     {f['reason'][:130]}")""")

    md("""## La duplicacion que es algebraica

`EMD2(one-hot) = suma_k (F_pred(k) - 1[y<=k])^2`

Es decir: el objetivo que la literatura suele presentar como ordinal distinto **ES** la familia
acumulativa CORAL/all-threshold con perdida L2, cerrada en V22/V23/H10/D-058. No es una variante
parecida, es la misma descomposicion con otra perdida.

## Y la que cierra el ultimo eje

Ord-NLL (Takezaki et al., IEEE Access 2026) es el metodo mas reciente de aprendizaje con ruido
ordinal, y su propio paper demuestra (Corollary 1) que **su minimizador esperado es un target
suave**. Es decir, colapsa al candidato C1 en expectativa. El eje de ruido ordinal, que era el
unico no auditado, queda cerrado por equivalencia y no por opinion.

## Decision

```text
NO_NEW_METHOD
```

Ningun candidato cumple a la vez las tres condiciones: compatible con el reglamento, no
equivalente a algo cerrado, y con evidencia de mejora de macro-F1 en clasificacion de edad a
partir de texto con modelos clasicos.

**C1 queda registrado y NO ejecutado.** Su expectativa honesta es no cruzar el gate, y el brief
pide decidir si reabrir, no gastar presupuesto en una falsacion. El control queda disenado por si
la decision se revisa.

## Lo que la revision NO dice

- No dice que V30 sea optimo.
- No dice que no exista ninguna mejora posible.
- No dice que la literatura no tenga nada util: dice que lo que tiene **ya esta cubierto** o
  **empuja contra la metrica**.

## El mapa, ahora auditado dos veces

| Eje | Estado |
|---|---|
| Representacion | 6 familias, todas USED o CLOSED |
| Familias de modelo | 10, todas USED |
| Objetivos de entrenamiento | 39 mecanismos auditados; 5 sin usar, cerrados con evidencia |
| Estructura de etiquetas | ordinal CLOSED, pairwise INVALIDATED |
| Integracion en score-space | 5 vias cerradas con control de atribucion |
| Entrada | enumerada por completo (G14) |
| Ruido ordinal | cerrado por equivalencia en expectativa |
| Sesgo de topicos | ganancia estrictamente OOD; el CV de AGE es IID |
| Geometria de clases | ganancia atada a extractor preentrenado |""")

    nbformat.write(nb, NOTEBOOK)
    return {"anadida": True, "antes": antes, "despues": len(nb.cells),
            "anadidas": len(nb.cells) - antes, "idempotente": True}


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    print("=== documentacion persistente ===", flush=True)
    for ruta, marca, bloque in (
        (ROOT / "DECISIONS.md", "## D-070", D),
        (ROOT / "PROJECT_STATE.md", "REVISION BIBLIOGRAFICA", ESTADO),
        (ROOT / "RUNLOG.md", "Revision bibliografica", RUNLOG),
        (ROOT / "FINAL_REPORT.md", "REVISION BIBLIOGRAFICA", ESTADO),
    ):
        print("  " + _anade(ruta, marca, bloque), flush=True)
    print("\n=== notebook ===", flush=True)
    print("  " + str(anade_notebook()), flush=True)
    RESULTADO.write_text(json.dumps({
        "fase": "revision bibliografica: documentacion", "fecha": FECHA, "experimenta": False,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, capture_output=True,
                                 text=True, timeout=30).stdout.strip(),
        "conclusion": "NO_NEW_METHOD",
        "documentos": ["DECISIONS.md (D-070)", "PROJECT_STATE.md", "RUNLOG.md", "FINAL_REPORT.md",
                       "notebook"],
        "candidatos_registrados_no_ejecutados": ["C1", "C2", "C3", "C4", "C5"],
        "entrenamiento": 0,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
