"""Documentacion del hallazgo del plug-in por clase.

## Que registra

`PROMISING_CONFIRMED` en los cinco documentos persistentes y en el notebook, mas la correccion al
registro historico de `V20-F3-UMBRALES`.

## Lo que este hallazgo cambia, y lo que no

**Cambia** el estado de la investigacion: hay un candidato que cruza el gate de PROMISING en las dos
particiones, con controles favorables, y que **no es** ninguna de las familias cerradas. Es una
mejora de la **regla de decision**, no del modelo ni de la representacion.

**No cambia** V30 como baseline: el modelo sigue siendo el mismo. Lo que cambia es como se decide
con su salida.
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

RESULTADO = cfg.EXPERIMENTS / "age_plugin_documentacion.json"
NOTEBOOK = ROOT / "notebooks" / "99_entrega_consolidado.ipynb"
MARCADOR = "PLUG-IN POR CLASE PARA MACRO-F1"
FECHA = "2026-10-11"
CV_V30 = 0.6134471
CV_CANON = 0.6154643
CV_ALT = 0.6164700
LB_V30 = 0.63302

D = """
---

## D-072 - Plug-in por clase: `PROMISING_CONFIRMED`, y una correccion al registro historico

- **Observation**: la auditoria de cobertura habia cerrado la investigacion con 39 mecanismos. Al
  revisar el problema con mirada nueva aparecio un objeto que **no se habia probado nunca**: la
  **regla de decision**. No la representacion, no el modelo, no la correccion del score: como se
  convierte la probabilidad en clase.

- **La medicion que lo motiva**: las seis clases de train son **perfectamente balanceadas** (0,160 a
  0,170) y la prediccion de V30 **no lo esta** (0,146 a 0,187). Su F1 por clase va de **0,4992**
  (25-34) a **0,7351** (65-xx). `argmax` maximiza accuracy, no macro-F1, y cuando el reparto
  predicho se desvia hacia las clases faciles hay margen que `argmax` no aprovecha.

- **El candidato**: `argmax_c(w_c * p_c)` con `w = [1.0000, 1.1749, 1.2721, 1.1630, 1.0010,
  0.8128]`, derivados por ascenso por coordenadas sobre validacion interna de la particion canonica.
  Sube 25-34, 18-24 y 35-49; baja 65-xx; deja 13-17 y 50-64 intactas.

- **Las dos particiones**:

  | Particion | V30 | plug-in | delta | folds+ | IC95 | alta conf. |
  |---|---|---|---|---|---|---|
  | canonica | 0,6134471 | 0,6154643 | **+0,0020172** | 5/5 | [+0,00101, +0,00295] | **0** |
  | alternativa | 0,6144120 | 0,6164700 | **+0,0020580** | 5/5 | [+0,00112, +0,00306] | **0** |

  **La diferencia entre las dos particiones es 0,0000408.** Reproduce casi a la cuarta cifra. El
  diagnostico `nested` (pesos re-derivados dentro de la particion alternativa) da **+0,0022884**:
  el mecanismo aparece solo.

- **La prediccion falsable, confirmada dos veces**: P34 establecio que una mejora que solo toca la
  frontera es de tipo calibracion. Aqui la prediccion era falsable —un multiplicador por clase no
  puede alterar el orden cuando el margen es grande— y se cumplio en las dos particiones: **cero**
  cambios en filas de alta confianza. Las 11.225 filas que cambian tienen confianza media 0,3357
  frente a 0,6136 del conjunto.

- **El matiz que se declara**: el control `permutado` cambio de signo entre particiones (−0,00279 en
  la canonica, +0,00144 en la alternativa), asi que `real − permutado` baja de +0,0048 a +0,00062.
  La lectura: parte del efecto es generica —cualquier perturbacion que diversifique el reparto ayuda
  algo a macro-F1— y la asignacion correcta de cada multiplicador a su clase anade unos **+0,0006**
  sobre eso. **El permutado no es un control negativo perfecto**: lo que aisla es la diferencia, no
  el nivel.

- **La fuga que se declaro en el pre-registro, y como queda**: los pesos congelados se derivaron de
  la union de cinco ajustes que cubren las 169.336 filas, asi que no son fuera de muestra para esas
  filas. En la canonica puntuaban **por encima** de los anidados (+0,00254 frente a +0,00202), que
  es la firma de haber visto los datos. **En la particion alternativa eso se invierte**
  (`congelado − nested = −0,0002304`): el efecto no se infla bajo un reparto nuevo.

- **La clasificacion, con las tres respuestas separadas**: **no** es una mejora de representacion,
  **no** es una mejora del modelo, **si** es una mejora de la regla de decision. El candidato sube
  macro-F1 **bajando accuracy** de 0,61464 a 0,61254: sacrifica aciertos para subir el recall de las
  clases debiles, que es lo que macro-F1 premia.

- **Correccion al registro historico**: `V20-F3-UMBRALES` quedo etiquetado en el gap analysis como
  *"correccion de umbral"* y **no lo era**. Fue una **cadena de cinco discriminadores binarios
  anidados con monotonia**, es decir una arquitectura ordinal. La regla de decision post-hoc no se
  habia probado nunca, y ese error mio de clasificacion contribuyo a que la linea pareciera cerrada.
  Los informes historicos no se modifican.

- **Decision**: **`PROMISING_CONFIRMED`**, no `FINALIST` y no `ADOPTED`. El delta esta por encima del
  gate de PROMISING (+0,0015) y **por debajo del de FINALIST** (+0,003). Y la particion alternativa
  usa los **mismos grupos** repartidos de otra forma, asi que es una reejecucion y no una particion
  independiente: ese limite viene de que `dup_clusters` cubre el dataset entero, y esta declarado
  desde P00. **No se ha generado ninguna submission y el leaderboard no se ha consultado.**

- **Consequences**: la conclusion de cierre `AGE_RESEARCH_CLOSED` queda **matizada por este
  hallazgo**. La formulacion correcta pasa a ser: no quedan mecanismos de aprendizaje sin recorrer,
  **pero quedaba uno de decision**, y produce una mejora confirmada de ~+0,0020 en CV que no habia
  sido medida. Eso no invalida la auditoria —los 39 mecanismos siguen auditados— sino que muestra
  que su enumeracion estaba incompleta: auditaba *como aprender* y no *como decidir*.
"""

ESTADO = f"""
---

## PLUG-IN POR CLASE PARA MACRO-F1 (2026-10-11) - `PROMISING_CONFIRMED`

- **Que es**: cambiar la **regla de decision** de `argmax` a `argmax_c(w_c * p_c)` con
  `w = [1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128]`. **No** cambia el modelo, la
  representacion ni los artefactos de V30.
- **Por que**: las clases de train son perfectamente balanceadas (0,160-0,170) y la prediccion de
  V30 no (0,146-0,187). `argmax` maximiza accuracy, no macro-F1.

| Particion | V30 | plug-in | delta | folds+ | IC95 | alta conf. |
|---|---|---|---|---|---|---|
| canonica | {CV_V30} | {CV_CANON} | **+0,0020172** | 5/5 | [+0,00101, +0,00295] | **0** |
| alternativa | 0,6144120 | {CV_ALT} | **+0,0020580** | 5/5 | [+0,00112, +0,00306] | **0** |

- **Diferencia entre particiones: 0,0000408.** Y el `nested` (pesos re-derivados en la particion
  alternativa) da +0,0022884: el mecanismo aparece solo.
- **Prediccion falsable confirmada dos veces**: cero cambios en filas de alta confianza. Un
  multiplicador no puede alterar el orden con margen grande. Es un cambio de regla, no de
  calibracion.
- **Matiz declarado**: el control `permutado` cambio de signo entre particiones; parte del efecto es
  generica (diversificar el reparto) y la asignacion correcta anade ~+0,0006. El permutado no es un
  control negativo perfecto.
- **Fuga descartada**: `congelado - nested = -0,0002304` en la particion nueva; los congelados no se
  inflan.
- **Clasificacion**: NO mejora de representacion, NO mejora del modelo, **SI mejora de la regla de
  decision**. Sube macro-F1 bajando accuracy (0,61464 -> 0,61254).
- **Estado**: no `FINALIST` (delta bajo +0,003) y no `ADOPTED` (la particion alternativa es
  reejecucion, no independiente). **Submission: ninguna. Leaderboard: no consultado.**
- **Correccion al registro**: `V20-F3-UMBRALES` era una **arquitectura ordinal** (cadena de
  discriminadores binarios anidados), no una regla de decision. Ver D-072.
"""

RUNLOG = f"""

---

## Plug-in por clase para macro-F1 · 2026-10-11

- **Resultado:** `PROMISING_CONFIRMED`.
- **Candidato:** `argmax_c(w_c * p_c)`, `w = [1.00, 1.17, 1.27, 1.16, 1.00, 0.81]`.
- **Canonica:** {CV_V30} -> {CV_CANON} (**+0,0020172**, 5/5).
- **Alternativa:** 0,6144120 -> {CV_ALT} (**+0,0020580**, 5/5).
- **Diferencia entre particiones:** 0,0000408.
- **Nested (mecanismo en limpio):** +0,0022884, 5/5.
- **Alta confianza:** **cero** cambios en las dos particiones. Prediccion falsable confirmada.
- **Matiz:** el control `permutado` paso de -0,00279 a +0,00144; parte del efecto es generica.
- **Pre-registro:** commit `bd08aba`. Pesos congelados, ninguna reoptimizacion.
- **Entrenamiento:** 0 modelos nuevos (solo la reejecucion de V30 en la particion alternativa, que
  es el protocolo ya definido). **Submissions: 0.** Test y holdout: sin tocar.
"""


def _anade(ruta: Path, marca: str, bloque: str) -> str:
    if not ruta.exists():
        return f"{ruta.name}: no existe"
    texto = ruta.read_text(encoding="utf-8")
    if marca in texto:
        return f"{ruta.name}: ya estaba"
    cuerpo = (texto if texto.endswith("\n") else texto + "\n") + bloque
    cuerpo = "\n".join(l.rstrip() for l in cuerpo.splitlines()) + "\n"
    ruta.write_text(cuerpo, encoding="utf-8")
    return f"{ruta.name}: +{len(bloque.splitlines())} lineas"


def anade_experiments() -> dict:
    ruta = ROOT / "experiments.csv"
    lineas = ruta.read_text(encoding="utf-8").splitlines()
    columnas = [c.strip() for c in lineas[0].split(",")]
    existentes = {ln.split(",", 1)[0] for ln in lineas[1:] if ln}
    nuevas = [
        {"experiment_id": "AGE-PLUGIN-CANON", "fecha": FECHA, "bloque": "28.0",
         "etapa": "plug-in por clase en la particion canonica",
         "pregunta": "mejora macro-F1 una regla de decision por clase en vez de argmax?",
         "hipotesis": "las clases estan balanceadas y la prediccion no: argmax favorece a las "
                      "faciles y macro-F1 premia el recall de las debiles",
         "features": "ninguna: es una regla de decision sobre las probabilidades existentes de V30",
         "preprocessing": "sin cambios", "representacion": "probabilidades de V30",
         "modelo": "ninguno: no se entrena nada",
         "hiperparametros": "{pesos: [1.0, 1.1749, 1.2721, 1.163, 1.001, 0.8128]}",
         "validacion": "StratifiedGroupKFold 5 sobre dup_clusters, semilla 42; pesos ajustados en "
                       "los otros cuatro folds (anidado)",
         "macro_f1_mean": "0.615464", "macro_f1_std": "", "weighted_f1_mean": "", "micro_f1_mean": "",
         "accuracy_mean": "0.61166",
         "resultado": "delta +0.0020172 | 5/5 folds | bootstrap IC95 [+0.0010143, +0.0029480] sin "
                      "cero | permutado -0.0027902 | aleatorio -0.0046860 | cero cambios en alta "
                      "confianza | accuracy BAJA de 0.61377 a 0.61166",
         "interpretacion": "la firma teorica se cumple: sube macro-F1 y baja accuracy, porque se "
                           "sacrifican aciertos en las clases faciles para subir el recall de las "
                           "debiles. Los controles permutado y aleatorio son ambos negativos, y no "
                           "hay ni un cambio en filas de alta confianza: es una regla de decision, "
                           "no calibracion",
         "decision": "PROMISING en la canonica; pasa a la particion alternativa",
         "duracion_seg": "75.0",
         "artefactos": "experiments/age_umbrales_validacion.json | "
                       "reports/age_class_thresholds.csv"},
        {"experiment_id": "AGE-PLUGIN-ALT", "fecha": FECHA, "bloque": "28.0",
         "etapa": "plug-in por clase en la particion alternativa (confirmatorio, congelado)",
         "pregunta": "se mantiene la mejora cuando cambia el reparto de entrenamiento/validacion?",
         "hipotesis": "si el mecanismo es real y no un ajuste a la particion, debe reproducirse",
         "features": "ninguna",
         "preprocessing": "sin cambios",
         "representacion": "probabilidades de V30 reentrenado en la particion alternativa",
         "modelo": "ninguno nuevo: V20 y los dos LightGBM reentrenados con la configuracion "
                   "CONGELADA de V30, que es el protocolo ya definido",
         "hiperparametros": "los pesos CONGELADOS del pre-registro bd08aba; no se reajustan",
         "validacion": "StratifiedGroupKFold 5 sobre los mismos dup_clusters, semilla 987654 "
                       "(heredada de V30-05, para no elegir semilla)",
         "macro_f1_mean": "0.616470", "macro_f1_std": "", "weighted_f1_mean": "", "micro_f1_mean": "",
         "accuracy_mean": "0.61254",
         "resultado": "V30 0.6144120 -> plug-in 0.6164700 | delta +0.0020580 | 5/5 folds | "
                      "bootstrap IC95 [+0.0011204, +0.0030622] sin cero | permutado +0.0014417 | "
                      "aleatorio -0.0161893 | cero cambios en alta confianza | nested +0.0022884",
         "interpretacion": "la diferencia entre las dos particiones es 0.0000408: reproduce casi a "
                           "la cuarta cifra. El nested (pesos re-derivados en la particion nueva) "
                           "da +0.0022884, luego el mecanismo aparece solo. El permutado cambio de "
                           "signo, asi que parte del efecto es generica y la asignacion correcta "
                           "anade unos +0.0006",
         "decision": "PROMISING_CONFIRMED. No FINALIST: el delta esta bajo +0.003 y la particion "
                     "alternativa es reejecucion, no independiente",
         "duracion_seg": "2088.0",
         "artefactos": "experiments/age_plugin_confirmacion.json | "
                       "reports/age_plugin_macro_f1_partitions.csv | "
                       "reports/age_plugin_macro_f1_final_decision.md"},
    ]
    faltan = [c for c in columnas if c not in nuevas[0]]
    if faltan:
        raise ValueError(f"faltan columnas: {faltan}")

    def fila(r):
        vals = [str(r[c]) for c in columnas]
        return ",".join(f'"{v}"' if ("," in v or '"' in v) else v for v in vals)

    anadidas = 0
    for reg in nuevas:
        if reg["experiment_id"] in existentes:
            continue
        with ruta.open("a", encoding="utf-8") as f:
            f.write(fila(reg) + "\n")
        anadidas += 1
    return {"anadidas": anadidas,
            "filas_tras": len(ruta.read_text(encoding="utf-8").splitlines()) - 1}


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

    md("""# PLUG-IN POR CLASE PARA MACRO-F1

## `PROMISING_CONFIRMED`

Las 30 fases anteriores buscaron **como aprender** mejor. Nadie comprobo **como decidir** con lo
aprendido. Y `argmax` no es la regla optima cuando la metrica es macro-F1.

## El desajuste que lo motiva

Las seis clases de train son **perfectamente balanceadas**. La prediccion de V30 **no**.""")

    code("""import json, pathlib
import pandas as pd
EXP = pathlib.Path("../experiments")
val = json.loads((EXP / "age_umbrales_validacion.json").read_text(encoding="utf-8"))
porc = json.loads((EXP / "age_umbrales_por_clase.json").read_text(encoding="utf-8"))
conf = json.loads((EXP / "age_plugin_confirmacion.json").read_text(encoding="utf-8"))
cie = json.loads((EXP / "age_plugin_cierre.json").read_text(encoding="utf-8"))
print("=== el desajuste ===")
for r in val["per_class"]:
    print(f"  {r['clase']:>6}  F1 base {r['f1_argmax']:.4f} -> plug {r['f1_plug_in']:.4f}  "
          f"delta {r['delta_f1']:+.4f}  recall {r['recall_argmax']:.4f} -> {r['recall_plug_in']:.4f}")
print()
print(f"  accuracy: {val['firma_teorica']['accuracy_argmax']:.5f} -> "
      f"{val['firma_teorica']['accuracy_plug_in']:.5f}  (BAJA, como predice la teoria)")
brazo = next(x for x in porc["resultados"] if x["brazo"] == "plug_in")
print(f"  macro-F1: 0.6134471 -> {brazo['corrected_cv']:.7f}  (SUBE)")
assert val["firma_teorica"]["delta_accuracy"] < 0, "la firma teorica exige que baje la accuracy"
print()
print("[OK] firma teorica correcta: sube macro-F1 bajando accuracy")""")

    code("""# Las dos particiones
print(pd.read_csv("../reports/age_plugin_macro_f1_partitions.csv")[
    ["particion","V30","plug_in","delta","folds_positivos","bootstrap_ic95_lo","bootstrap_ic95_hi","cambios_alta_confianza"]
].to_string(index=False))
print()
print(f"diferencia entre las dos particiones: {cie['diferencia_entre_particiones']:.7f}")
print(f"nested (pesos re-derivados en la alternativa): {cie['nested']['delta']:+.7f} "
      f"({cie['nested']['folds_positivos']}/5)")
print()
print("las DOS particiones cumplen: delta >= +0,0015, 5/5 folds, IC95 sin cero,")
print("y CERO cambios en alta confianza")""")

    code("""# La prediccion falsable: un multiplicador NO puede tocar alta confianza
print("canonica    -> filas que cambian:", f"{brazo['filas_que_cambian']:,}",
      "| de alta confianza:", val["alta_confianza"]["n_alta_que_cambian"])
print("alternativa -> filas que cambian:", f"{conf['resultados'][0]['filas_que_cambian']:,}",
      "| de alta confianza:", conf["alta_confianza"]["n_alta_que_cambian"])
print()
print("confianza media de las que cambian (alternativa):",
      conf["alta_confianza"]["confianza_media_de_las_que_cambian"])
print()
assert val["alta_confianza"]["n_alta_que_cambian"] == 0
assert conf["alta_confianza"]["n_alta_que_cambian"] == 0
print("[OK] cero en las dos. Es una regla de decision, no calibracion.")""")

    md("""## El matiz que se declara

**El control `permutado` cambió de signo entre particiones**: −0,00279 en la canónica, +0,00144 en
la alternativa. `real − permutado` pasa de +0,0048 a +0,00062.

Parte del efecto es **genérica** —cualquier perturbación que diversifique el reparto ayuda algo a
macro-F1— y la asignación correcta de cada multiplicador a su clase añade unos **+0,0006** sobre
eso. El permutado **no es un control negativo perfecto**: lo que aísla es la diferencia, no el
nivel. El aleatorio sí se hunde (−0,0162 con la misma forma de pesos).

## La fuga que se declaró, y cómo queda

Los pesos congelados se derivaron de la unión de cinco ajustes que cubren las 169.336 filas, así que
no son fuera de muestra para esas filas. En la canónica puntuaban **por encima** de los anidados
(+0,00254 vs +0,00202): la firma de haber visto los datos.

**En la partición alternativa eso se invierte** (`congelado − nested = −0,0002304`). El efecto no se
infla bajo un reparto nuevo.

## La clasificación, con las tres respuestas separadas

| Pregunta | Respuesta |
|---|---|
| ¿Mejora de representación? | **NO** |
| ¿Mejora del modelo? | **NO** |
| ¿Mejora de la regla de decisión? | **SÍ** |

## Estado

`PROMISING_CONFIRMED`, no `FINALIST` (el delta está bajo +0,003) y no `ADOPTED` (la partición
alternativa es reejecución, no independiente: usa los mismos `dup_clusters`).

**Ninguna submission generada. Leaderboard no consultado. Test y holdout sin tocar.**

## Corrección al registro histórico

`V20-F3-UMBRALES` quedó etiquetado como *"corrección de umbral"* y **no lo era**: fue una **cadena de
cinco discriminadores binarios anidados con monotonía**, o sea una arquitectura ordinal. La regla de
decisión post-hoc **no se había probado nunca**. Los informes históricos no se modifican.""")

    nbformat.write(nb, NOTEBOOK)
    return {"anadida": True, "antes": antes, "despues": len(nb.cells),
            "anadidas": len(nb.cells) - antes, "idempotente": True}


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    print("=== documentacion persistente ===", flush=True)
    for ruta, marca, bloque in (
        (ROOT / "DECISIONS.md", "## D-072", D),
        (ROOT / "PROJECT_STATE.md", "PLUG-IN POR CLASE PARA MACRO-F1", ESTADO),
        (ROOT / "RUNLOG.md", "Plug-in por clase para macro-F1", RUNLOG),
        (ROOT / "FINAL_REPORT.md", "PLUG-IN POR CLASE PARA MACRO-F1", ESTADO),
    ):
        print("  " + _anade(ruta, marca, bloque), flush=True)
    print("\n=== experiments.csv ===", flush=True)
    print("  " + str(anade_experiments()), flush=True)
    print("\n=== notebook ===", flush=True)
    print("  " + str(anade_notebook()), flush=True)
    RESULTADO.write_text(json.dumps({
        "fase": "documentacion del plug-in por clase", "fecha": FECHA, "experimenta": False,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, capture_output=True,
                                 text=True, timeout=30).stdout.strip(),
        "estado": "PROMISING_CONFIRMED",
        "documentos": ["DECISIONS.md (D-072)", "PROJECT_STATE.md", "RUNLOG.md", "FINAL_REPORT.md",
                       "experiments.csv (AGE-PLUGIN-CANON, AGE-PLUGIN-ALT)", "notebook"],
        "reports": ["age_plugin_macro_f1_confirmatory.md", "age_plugin_macro_f1_partitions.csv",
                    "age_plugin_macro_f1_final_decision.md"],
        "submission_generada": False, "leaderboard_consultado": False,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
