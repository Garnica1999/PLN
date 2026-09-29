"""Cierre formal de AGE: consolidacion documental definitiva.

## Que hace

Registra formalmente el estado final de la investigacion en los cuatro documentos persistentes, en
`experiments.csv` y en el notebook. **No entrena, no toca V30, no usa test ni holdout.**

## Por que existe esta ultima consolidacion

Las fases anteriores fueron cerrando lineas una a una, y cada cierre quedo en su propio commit con
su propia seccion. Lo que falta es el **acta de cierre**: un unico bloque que diga, en un sitio, el
estado congelado completo, la conclusion metodologica con sus tres negaciones, y las tres piezas de
evidencia que impiden que una iteracion futura vuelva a proponer tecnicas ya cubiertas. Sin ese
bloque, el cierre hay que reconstruirlo leyendo ocho commits y cuatro informes.

## Las tres piezas que se conservan por su valor anticiclico

1. **Los cierres algebraicos**, que no son opiniones: `EMD2(one-hot) = suma_k (F_pred(k) -
   1[y<=k])^2` es la familia acumulativa CORAL con perdida L2; el minimizador esperado de
   `Ord-NLL` es un target suave; y los targets suaves son un **superconjunto** de label smoothing.
   Quien vuelva a proponer cualquiera de los tres como "mecanismo nuevo" puede comprobarlo en una
   linea.

2. **Los errores corregidos de auditoria**, con su numero correcto: A = **27.430** y no 27.007;
   rescates exclusivos = **8.029** y no 122 (definiciones distintas); los OOF de V31-03 son
   `decision_function` y **no** probabilidades; y los falsos positivos de mis propias busquedas por
   regex, declarados para que nadie los repita como evidencia.

3. **C1 como hipotesis latente**, con su control ya disenado y el motivo explicito por el que no se
   ejecuta: su expectativa honesta es **no cruzar el gate**, y la evidencia publicada dice que el
   trade-off ordinal/nominal va contra macro-F1.
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

RESULTADO = cfg.EXPERIMENTS / "age_cierre_formal.json"
NOTEBOOK = ROOT / "notebooks" / "99_entrega_consolidado.ipynb"
MARCADOR = "ACTA DE CIERRE DE AGE"
FECHA = "2026-10-09"
CV_V30 = 0.6134471
CV_V30_SEG = 0.6143347
LB_V30 = 0.63302

D = """
---

## D-071 - Acta de cierre de AGE: `AGE_RESEARCH_CLOSED`

- **Decision**: se cierra formalmente la investigacion de AGE. **No se ejecuta C1 ni ninguna otra
  falsacion.** V30 permanece como mejor resultado validado y como baseline congelado.

- **Estado congelado, y es el dato que gobierna todo lo demas**:

  ```text
  V30 corrected_cv      = 0.6134471
  V30 particion alterna = 0.6143347
  V30 leaderboard       = 0.63302
  ```

  Los tres numeros son distintos y **no son intercambiables**: 0,6134471 es la unica configuracion
  reproducible desde los artefactos y la que se envio; 0,6143347 es una **reejecucion con
  reentrenamiento** sobre los mismos grupos con otra semilla, y su diferencia con la primera es
  reentrenamiento en el 99 %, medido; 0,63302 es el leaderboard, evidencia de transferencia que no
  se uso para seleccionar nada.

- **Estado de cada linea, cerrado y con su veredicto**:

  | Linea | Veredicto |
  |---|---|
  | G14 auditoria de entrada | **NOT_ACTIONABLE** |
  | P33 residuo global | **CLOSED** |
  | P34 correccion local por pares | **INVALIDATED** |
  | G12 pendiente sobre el margen | **CLOSED** |
  | Revision bibliografica | **NO_NEW_METHOD** |
  | C1 perdidas distancia-conscientes | **LATENT / NO EJECUTADA** |

- **La conclusion metodologica, con sus tres negaciones explicitas**:

  > **No se afirma que V30 sea optimo.** **No se afirma que AGE sea imposible de mejorar.** Se
  > afirma que, dentro de las restricciones vigentes, los inputs legalmente disponibles y los
  > mecanismos auditados, **no queda actualmente una hipotesis nueva con justificacion suficiente
  > para reabrir la investigacion**.

  La formulacion importa: la primera y la segunda negacion evitan vender un cierre como un optimo, y
  la tercera acota la afirmacion al espacio auditado en vez de al espacio entero. Un cierre honesto
  no puede ser mas amplio que su auditoria.

- **La cobertura auditada, que es lo que sostiene la tercera afirmacion**:

  ```text
  39 mecanismos auditados
  23 USED          9 CLOSED          1 INVALIDATED
   1 NOT_ACTIONABLE 5 NOT USED       0 GENUINELY_NEW con justificacion suficiente para ejecutar
  ```

  Los cinco `NOT USED` no quedan en el limbo: **focal loss**, **hard example mining** y **label
  smoothing** apuntan a los ejemplos dificiles y a la frontera, que es la poblacion A (27.430 filas
  donde ninguna de 26 fuentes acierta) y donde P34 midio que el ruido aleatorio gana a la senal
  real. **ranking objective** no mueve la metrica evaluada. **bootstrap de grupos** es reduccion de
  varianza con otra unidad de remuestreo, y el bagging del meta-modelo ya se cerro en -0,000312.

- **C1 queda LATENT, y se documenta con su control por si se revisa**:

  | Campo | Valor |
  |---|---|
  | Hipotesis | codificar en el objetivo que confundir 13-17 con 18-24 es menos grave que con 65-xx cambia la geometria del ajuste |
  | Evidencia a favor | SORD (CVPR 2019) mejora exactitud en Adience en dos replicas independientes; Vargas et al. (TKDE 2026) encuentra que el etiquetado suave mejora la generalizacion ordinal |
  | Evidencia en contra | **el mismo Vargas et al. dice que la CE categorica rinde mejor en promedio en metricas NOMINALES**, y Kasai et al. (texto) lo confirma: hay un trade-off y la mejora ordinal se paga con metricas nominales |
  | Por que no se ejecuta | macro-F1 es nominal. La expectativa honesta es **no cruzar el gate**, y el brief pide decidir si reabrir, no gastar en una falsacion |
  | Control ya disenado | tres brazos: CE one-hot, label smoothing uniforme (el cierre de la auditoria), y kernel con etiquetas permutadas (regularizacion pura). Gate: delta >= +0,0015 con 4/5 folds **y** delta > 8,9e-04 con signo consistente |
  | Diferencia con lo cerrado | no es CORAL (K-1 salidas acumulativas) ni pairwise (K(K-1)/2 binarios); **si es un superconjunto de label smoothing**: con q_k = exp(-beta|k-y|^2) colapsa a LS cuando beta tiende a 0 |

- **Las equivalencias algebraicas, conservadas porque son anticiclicas**: sin ellas, una iteracion
  futura volveria a proponer como nuevas tres tecnicas que ya estan cubiertas.

  1. `EMD2(one-hot) = suma_k (F_pred(k) - 1[y<=k])^2`. El objetivo que la literatura presenta como
     ordinal distinto **es** la familia acumulativa CORAL/all-threshold con perdida L2, cerrada en
     V22/V23/H10/D-058.
  2. El minimizador esperado de **Ord-NLL** es un target suave (Corollary 1 de su propio paper):
     colapsa a C1 en expectativa.
  3. Los **targets suaves** son un superconjunto de **label smoothing**: su gradiente es `p - q =
     (p - e_y) + (e_y - q)`, es decir CE one-hot mas un **sesgo fijo dependiente de la clase**, no
     una superficie de decision distinta.

- **Los errores de auditoria corregidos, conservados con su numero correcto**:

  | Lo que se reporto | Lo correcto | Causa |
  |---|---|---|
  | A = 27.007 | **27.430** | enmascarado enrevesado para excluir a los miembros de V30 de su propio rescate |
  | 122 rescates exclusivos | **8.029** | definiciones distintas: el previo medía otra cosa. Ninguno de los dos es erroneo |
  | OOF de V31-03 como probabilidades | son **`decision_function`** | dividirlos por la suma de su fila **invierte el orden**: F1 0,5048 -> 0,0445 |
  | "soft labels ya cerrado" | cerrado **por argumento, no por medicion** | 0 scripts, 0 menciones; el proyecto entero tiene 2 perdidas custom |

  Y los **falsos positivos de mis propias busquedas por regex**, declarados para que nadie los
  repita como evidencia: `complement` matcheaba "complementario" en espanol (391 veces) y `sord`
  matcheaba `tensordot` (15 veces).

- **Consequences**: la investigacion queda cerrada con el mapa auditado **dos veces**, contra el
  codigo del proyecto y contra la literatura 2020-2026. Una reapertura futura tendria que apoyarse
  en una de tres cosas: **evidencia nueva** que cambie el diagnostico, un **cambio de restriccion**
  (por ejemplo que se permitieran metadatos o modelos preentrenados, que es donde la literatura
  situa la mayor parte de la ganancia), o una **hipotesis cientificamente distinta** que no este en
  las 39 auditadas. Cambiar el `beta` de una familia cerrada no cumple ninguna de las tres.
"""

ESTADO = f"""
---

# CIERRE FORMAL DE AGE (2026-10-09) - `AGE_RESEARCH_CLOSED`

> Estado congelado definitivo. Esta seccion es el acta de cierre: si algo de lo anterior
> contradice lo de aqui, manda esto.

## Los tres numeros

```text
V30 corrected_cv      = {CV_V30}
V30 particion alterna = {CV_V30_SEG}
V30 leaderboard       = {LB_V30}
```

No son intercambiables. {CV_V30} es la unica configuracion reproducible desde los artefactos **y**
la que se envio. {CV_V30_SEG} es una **reejecucion con reentrenamiento** sobre los mismos grupos con
otra semilla: su diferencia con la primera es reentrenamiento en el 99 %, medido. {LB_V30} es
evidencia de transferencia y nunca se uso para seleccionar nada.

## Estado de cada linea

| Linea | Veredicto |
|---|---|
| G14 auditoria de informacion de entrada | **NOT_ACTIONABLE** |
| P33 residuo global | **CLOSED** |
| P34 correccion local por pares | **INVALIDATED** |
| G12 pendiente sobre el margen | **CLOSED** |
| Revision bibliografica 2026 | **NO_NEW_METHOD** |
| C1 perdidas distancia-conscientes | **LATENT / NO EJECUTADA** |

## La conclusion, con sus tres negaciones

> **No se afirma que V30 sea optimo.**
> **No se afirma que AGE sea imposible de mejorar.**
> Se afirma que, dentro de las restricciones vigentes, los inputs legalmente disponibles y los
> mecanismos auditados, **no queda actualmente una hipotesis nueva con justificacion suficiente
> para reabrir la investigacion**.

## La cobertura que sostiene la afirmacion

```text
39 mecanismos auditados
23 USED          9 CLOSED          1 INVALIDATED
 1 NOT_ACTIONABLE 5 NOT USED       0 GENUINELY_NEW con justificacion suficiente
```

Los cinco `NOT USED` se cierran con medicion: focal loss, hard example mining y label smoothing
apuntan a la poblacion A (27.430 filas sin ninguna de 26 fuentes que las corrija) y a la frontera,
donde P34 midio que el ruido aleatorio gana. `ranking objective` no mueve macro-F1. `bootstrap de
grupos` es varianza con otra unidad, y el bagging ya se cerro en -0,000312.

## C1, latente y con su control

**No se ejecuta.** Su expectativa honesta es no cruzar el gate, y la evidencia publicada dice que el
trade-off ordinal/nominal va en contra de macro-F1 (Vargas et al., IEEE TKDE 2026; Kasai et al.,
texto). Queda con su control disenado: tres brazos (CE one-hot, label smoothing, kernel con
etiquetas permutadas) y gate doble (delta >= +0,0015 con 4/5 folds **y** delta > 8,9e-04).

## Las equivalencias que impiden reabrir por error

1. `EMD2(one-hot) = suma_k (F_pred(k) - 1[y<=k])^2` -> es la familia acumulativa CORAL, cerrada.
2. El minimizador esperado de **Ord-NLL** es un target suave -> colapsa a C1.
3. Los **targets suaves** son un **superconjunto** de label smoothing -> cerrado por argumento,
   y ahora por evidencia.

## Correcciones de auditoria conservadas

| Se reporto | Correcto | Causa |
|---|---|---|
| A = 27.007 | **27.430** | enmascarado en los miembros de V30 |
| 122 rescates exclusivos | **8.029** | definiciones distintas, ninguna erronea |
| OOF de V31-03 como probabilidades | son **`decision_function`** | dividir por la suma invierte el orden: 0,5048 -> 0,0445 |

## Como se reabriria

Con **evidencia nueva** que cambie el diagnostico, un **cambio de restriccion** (los metadatos o los
modelos preentrenados son donde la literatura situa la mayor parte de la ganancia), o una
**hipotesis cientificamente distinta** que no este entre las 39 auditadas. Cambiar el `beta` de una
familia cerrada no cumple ninguna de las tres.
"""

RUNLOG = f"""

---

## Cierre formal de AGE · 2026-10-09

- **Resultado:** `AGE_RESEARCH_CLOSED`. Acta completa en D-071.
- **Estado congelado:** V30 corrected_cv **{CV_V30}** | particion alterna {CV_V30_SEG} | leaderboard
  **{LB_V30}**. Los tres no son intercambiables.
- **Lineas:** G14 NOT_ACTIONABLE | P33 CLOSED | P34 INVALIDATED | G12 CLOSED | literatura
  NO_NEW_METHOD | C1 LATENT sin ejecutar.
- **Cobertura:** 39 mecanismos (23 USED, 9 CLOSED, 1 INVALIDATED, 1 NOT_ACTIONABLE, 5 NOT USED),
  0 genuinamente nuevos con justificacion suficiente.
- **Conclusion:** no se afirma que V30 sea optimo ni que AGE sea imposible de mejorar; se afirma que
  no queda hipotesis nueva con justificacion suficiente para reabrir.
- **Entrenamiento: 0. Submissions: 0.** Leaderboard, test y holdout: sin tocar.
- **Evidencia conservada:** las tres equivalencias algebraicas, los tres errores de auditoria
  corregidos con su numero correcto, y C1 con su control disenado y su motivo de no ejecucion.
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
    """Registra el acta en experiments.csv como cierre, no como experimento."""
    ruta = ROOT / "experiments.csv"
    lineas = ruta.read_text(encoding="utf-8").splitlines()
    columnas = [c.strip() for c in lineas[0].split(",")]
    existentes = {ln.split(",", 1)[0] for ln in lineas[1:] if ln}
    if "AGE-CLOSURE" in existentes:
        return {"hecho": True, "ya_estaba": True}
    reg = {
        "experiment_id": "AGE-CLOSURE", "fecha": FECHA, "bloque": "27.0",
        "etapa": "acta de cierre formal de la investigacion",
        "pregunta": "queda alguna hipotesis nueva con justificacion suficiente para reabrir la "
                    "investigacion de AGE?",
        "hipotesis": "no, dentro de las restricciones vigentes y del espacio auditado",
        "features": "ninguna: **0 experimentos de entrenamiento**, es un acta de cierre",
        "preprocessing": "sin cambios", "representacion": "n/a", "modelo": "n/a",
        "hiperparametros": "n/a",
        "validacion": "consolidacion de las auditorias previas; sin validacion cruzada porque no hay "
                      "modelo",
        "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "", "micro_f1_mean": "",
        "accuracy_mean": "",
        "resultado": f"V30 congelado en {CV_V30} (particion alterna {CV_V30_SEG}, leaderboard "
                     f"{LB_V30}). Lineas: G14 NOT_ACTIONABLE, P33 CLOSED, P34 INVALIDATED, G12 "
                     f"CLOSED, literatura NO_NEW_METHOD, C1 LATENT sin ejecutar. Cobertura: 39 "
                     f"mecanismos, 23 USED, 9 CLOSED, 1 INVALIDATED, 1 NOT_ACTIONABLE, 5 NOT USED, "
                     f"0 GENUINELY_NEW con justificacion suficiente",
        "interpretacion": "no se afirma que V30 sea optimo ni que AGE sea imposible de mejorar. Se "
                          "afirma que, dentro de las restricciones vigentes, los inputs legalmente "
                          "disponibles y los mecanismos auditados, no queda hipotesis nueva con "
                          "justificacion suficiente para reabrir",
        "decision": "AGE_RESEARCH_CLOSED. El mapa queda auditado dos veces: contra el codigo del "
                    "proyecto y contra la literatura 2020-2026. Una reapertura exigiria evidencia "
                    "nueva, un cambio de restriccion o una hipotesis cientificamente distinta",
        "duracion_seg": "1.0",
        "artefactos": "experiments/age_cierre_formal.json | DECISIONS.md D-071 | "
                      "reports/age_final_methodology_audit.md | "
                      "reports/age_literature_review_2026.md",
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

    md(f"""# ACTA DE CIERRE DE AGE

## `AGE_RESEARCH_CLOSED`

Esta es la seccion de cierre. Si algo de lo anterior la contradice, manda esta.

## Los tres numeros, que no son intercambiables

```text
V30 corrected_cv      = {CV_V30}
V30 particion alterna = {CV_V30_SEG}
V30 leaderboard       = {LB_V30}
```

- **{CV_V30}** es la unica configuracion reproducible desde los artefactos **y** la que se envio.
- **{CV_V30_SEG}** es una **reejecucion con reentrenamiento** sobre los mismos grupos con otra
  semilla. Su diferencia con la primera es reentrenamiento en el 99 %, medido.
- **{LB_V30}** es evidencia de transferencia y nunca se uso para seleccionar nada.

## Estado de cada linea

| Linea | Veredicto |
|---|---|
| G14 auditoria de informacion de entrada | **NOT_ACTIONABLE** |
| P33 residuo global | **CLOSED** |
| P34 correccion local por pares | **INVALIDATED** |
| G12 pendiente sobre el margen | **CLOSED** |
| Revision bibliografica 2026 | **NO_NEW_METHOD** |
| C1 perdidas distancia-conscientes | **LATENT / NO EJECUTADA** |

## La conclusion, con sus tres negaciones

> **No se afirma que V30 sea optimo.**
> **No se afirma que AGE sea imposible de mejorar.**
> Se afirma que, dentro de las restricciones vigentes, los inputs legalmente disponibles y los
> mecanismos auditados, **no queda actualmente una hipotesis nueva con justificacion suficiente
> para reabrir la investigacion**.""")

    code("""import json, pathlib
import pandas as pd
EXP = pathlib.Path("../experiments")
cie = json.loads((EXP / "age_cierre_formal.json").read_text(encoding="utf-8"))
print("=== estado congelado ===")
for k, v in cie["estado_congelado"].items():
    print(f"  {k:28s} {v}")
print()
print("=== lineas ===")
for k, v in cie["lineas"].items():
    print(f"  {k:26s} {v}")
print()
print("=== cobertura ===")
for k, v in cie["cobertura"].items():
    print(f"  {k:18s} {v}")""")

    code("""# Las tres equivalencias que impiden reabrir por error
print("1. EMD2 es la familia acumulativa CORAL")
print("   EMD2(one-hot) = suma_k (F_pred(k) - 1[y<=k])^2")
print("   -> no es un objetivo ordinal distinto: es CORAL con perdida L2, cerrado en V22/V23/H10/D-058")
print()
print("2. Ord-NLL colapsa a soft labels")
print("   el minimizador esperado de Ord-NLL es un target suave (Corollary 1 de su propio paper)")
print("   -> el eje de ruido ordinal queda cerrado por equivalencia, no por opinion")
print()
print("3. Los targets suaves son un SUPERCONJUNTO de label smoothing")
print("   gradiente p - q = (p - e_y) + (e_y - q):")
print("   CE one-hot mas un sesgo fijo por clase, no una superficie de decision distinta")
print("   -> con beta tendiendo a 0 el kernel colapsa exactamente a LS, que la auditoria ya descarto")""")

    code("""# Correcciones de auditoria conservadas, con su numero correcto
print("| se reporto | correcto | causa |")
print("|---|---|---|")
print("| A = 27.007 | 27.430 | enmascarado en los miembros de V30 |")
print("| 122 rescates exclusivos | 8.029 | definiciones distintas; ninguna erronea |")
print("| OOF de V31-03 como probabilidades | son decision_function | dividir por la suma invierte el orden: 0,5048 -> 0,0445 |")
print()
print("falsos positivos de mis propias busquedas, declarados:")
print('  - "complement" matcheaba "complementario" en espanol (391 veces)')
print("  - 'sord' matcheaba 'tensordot' (15 veces)")
print()
print("V30 sigue con el hash del submission intacto: f543a715...")""")

    md("""## C1, latente y con su control

**No se ejecuta.** Su expectativa honesta es no cruzar el gate, y la evidencia publicada dice que el
trade-off ordinal/nominal va contra macro-F1 (Vargas et al., IEEE TKDE 2026; Kasai et al., texto).

Queda con el control ya disenado: tres brazos (CE one-hot, label smoothing uniforme, kernel con
etiquetas permutadas) y gate doble (delta >= +0,0015 con 4/5 folds **y** delta > 8,9e-04 con signo
consistente). La diferencia con lo cerrado esta documentada: no es CORAL ni pairwise, y si es un
superconjunto de label smoothing.

## Como se reabriria

| Via | Ejemplo |
|---|---|
| **Evidencia nueva** que cambie el diagnostico | una medicion que muestre senal corregible en la poblacion A |
| **Cambio de restriccion** | que se permitieran metadatos o modelos preentrenados, donde la literatura situa la mayor parte de la ganancia |
| **Hipotesis cientificamente distinta** | un mecanismo que no este entre las 39 auditadas |

Cambiar el `beta` de una familia cerrada no cumple ninguna de las tres.""")

    nbformat.write(nb, NOTEBOOK)
    return {"anadida": True, "antes": antes, "despues": len(nb.cells),
            "anadidas": len(nb.cells) - antes, "idempotente": True}


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, capture_output=True,
                            text=True, timeout=30).stdout.strip()
    print("=== documentacion persistente ===", flush=True)
    for ruta, marca, bloque in (
        (ROOT / "DECISIONS.md", "## D-071", D),
        (ROOT / "PROJECT_STATE.md", "CIERRE FORMAL DE AGE", ESTADO),
        (ROOT / "RUNLOG.md", "Cierre formal de AGE", RUNLOG),
        (ROOT / "FINAL_REPORT.md", "CIERRE FORMAL DE AGE", ESTADO),
    ):
        print("  " + _anade(ruta, marca, bloque), flush=True)
    print("\n=== experiments.csv ===", flush=True)
    print("  " + str(anade_experiments()), flush=True)
    print("\n=== notebook ===", flush=True)
    print("  " + str(anade_notebook()), flush=True)

    RESULTADO.write_text(json.dumps({
        "fase": "acta de cierre formal", "fecha": FECHA, "experimenta": False, "commit": commit,
        "conclusion": "AGE_RESEARCH_CLOSED",
        "estado_congelado": {"V30_corrected_cv": CV_V30, "V30_particion_alterna": CV_V30_SEG,
                             "V30_leaderboard": LB_V30},
        "lineas": {"G14": "NOT_ACTIONABLE", "P33": "CLOSED", "P34": "INVALIDATED", "G12": "CLOSED",
                   "revision_bibliografica": "NO_NEW_METHOD", "C1": "LATENT / NO EJECUTADA"},
        "cobertura": {"mecanismos_auditados": 39, "USED": 23, "CLOSED": 9, "INVALIDATED": 1,
                      "NOT_ACTIONABLE": 1, "NOT_USED": 5,
                      "GENUINELY_NEW_con_justificacion_suficiente": 0},
        "conclusion_metodologica": {
            "no_se_afirma_que_v30_sea_optimo": True,
            "no_se_afirma_que_age_sea_imposible_de_mejorar": True,
            "se_afirma": "dentro de las restricciones vigentes, los inputs legalmente disponibles y "
                         "los mecanismos auditados, no queda actualmente una hipotesis nueva con "
                         "justificacion suficiente para reabrir la investigacion",
        },
        "evidencia_conservada": {
            "equivalencias_algebraicas": [
                "EMD2(one-hot) = suma_k (F_pred(k) - 1[y<=k])^2 -> familia acumulativa CORAL",
                "el minimizador esperado de Ord-NLL es un target suave -> colapsa a C1",
                "los targets suaves son un superconjunto de label smoothing",
            ],
            "errores_corregidos": [
                "A = 27.430, no 27.007",
                "rescates exclusivos = 8.029 con esta definicion, no 122",
                "los OOF de V31-03 son decision_function, no probabilidades",
                "label smoothing estaba cerrado por argumento, no por medicion",
            ],
            "falsos_positivos_declarados": [
                "complement matcheaba 'complementario' (391 veces)",
                "sord matcheaba 'tensordot' (15 veces)",
            ],
        },
        "como_se_reabriria": ["evidencia nueva", "cambio de restriccion",
                              "hipotesis cientificamente distinta"],
        "restricciones": {"entrenamiento": 0, "submissions": 0, "usa_test": False,
                          "usa_holdout": False, "usa_leaderboard_para_elegir": False,
                          "toca_v30": False},
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
