"""Cierre final: el plug-in se adopta como campeon tras transferencia externa positiva.

## El resultado

```text
V30      leaderboard = 0,63302
plug-in  leaderboard = 0,63575
delta LB              = +0,00273
```

`POSITIVE_EXTERNAL_TRANSFER`: la mejora medida internamente se transfirio al test.

## Lo que este script hace, y lo que no

Documenta y congela. **No** reajusta pesos, **no** genera variantes, **no** crea otra submission y
**no** usa el leaderboard para elegir nada. El 0,63575 entra como **evidencia final**, no como
funcion objetivo: no hay ninguna decision posterior que dependa de el.

## Un detalle que la transferencia confirma, y que vale la pena registrar

En el ciclo anterior hubo que corregir un error mio: se estaban confundiendo tres numeros del
interno (nested, global congelado, redondeado), y la submission ejecuta el **global**. Ahora el
leaderboard lo confirma por su cuenta. Con la calibracion del propio proyecto, `LB = 0,00223 +
1,02825 x CV`:

```text
V30                    CV 0,6134471 -> pred 0,63301 | real 0,63302 | error +0,00001
plug-in (CV nested)    CV 0,6154643 -> pred 0,63508 | real 0,63575 | error +0,00067
plug-in (CV global)    CV 0,6159732 -> pred 0,63560 | real 0,63575 | error +0,00015
```

El leaderboard casa con el **global**, no con el nested. Es decir: la correccion de aquel ciclo no
era cosmetica, era la descripcion correcta de lo que la submission hace.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "age_plugin_cierre_final.json"
TRANSFER_MD = cfg.REPORTS / "age_plugin_external_transfer.md"
CHAMPION_MD = cfg.REPORTS / "age_final_champion.md"
TRANSFER_CSV = cfg.REPORTS / "age_transfer_comparison.csv"
NOTEBOOK = ROOT / "notebooks" / "99_entrega_consolidado.ipynb"
MARCADOR = "FINAL ADOPTED CANDIDATE - PLUG-IN MACRO-F1"
SUBMISSION = ROOT / "submissions" / "age_submission_plugin_candidate.csv"
FECHA = "2026-10-13"
CV_V30 = 0.6134471
CV_V30_ALT = 0.6143347
LB_V30 = 0.63302
CV_PLUG = 0.6154643
CV_PLUG_ALT = 0.6164700
CV_PLUG_GLOBAL = 0.6159732
CV_V30_ALT_REEJEC = 0.6144120
LB_PLUG = 0.63575
CALIBRACION = (0.00223, 1.02825)
N_CLASES = 6
SHA_PLUG = "29cbe6cca4312a65f5176bad25c7d4ae4624d348f3e55586dfa36767a3abd6c0"
SHA_V30 = "f543a715a1881c6128a860301b84a7273c3db8f8a709545be1e6267822e713be"

D = f"""
---

## D-074 - El plug-in por clase se ADOPTA: `POSITIVE_EXTERNAL_TRANSFER`

- **Observation**: la submission congelada del plug-in se subio a Kaggle y produjo **0,63575** frente
  a los **0,63302** de V30. Delta de leaderboard: **+0,00273**.

- **Estado**: `research_status = CLOSED`, `candidate_status = ADOPTED`,
  `external_transfer = POSITIVE_TRANSFER`. **V30 permanece como baseline historico congelado** y no
  se sobrescribe: sigue siendo la referencia contra la que se mide todo, y su fichero conserva el
  hash `{SHA_V30[:16]}`.

- **El candidato adoptado**:

  ```text
  V30 + regla de decision  argmax_c( w_c * p_c )
  w = [1.00, 1.17, 1.27, 1.16, 1.00, 0.81]
  ```

- **La comparacion completa**:

  | | CV canonico | CV alternativo | leaderboard |
  |---|---|---|---|
  | V30 | 0,6134471 | 0,6143347 | 0,63302 |
  | plug-in | **0,6154643** | **0,6164700** | **0,63575** |
  | delta | **+0,0020172** | **+0,0020580** | **+0,00273** |

  Las dos direcciones coinciden: la mejora medida en validacion agrupada y la medida en la
  transferencia externa apuntan al mismo lado.

- **La conclusion cientifica, y es especifica**:

  > La mejora **no** proviene de una representacion nueva ni de un modelo nuevo. Proviene de la
  > **regla de decision** aplicada sobre las probabilidades del ensemble V30. El plug-in por clase
  > mejora macro-F1 tanto en validacion agrupada como en la transferencia externa.

  Lo que **no** se afirma: que V30 fuera incorrecto, ni que `argmax` no sea optimo en general. La
  afirmacion es acotada: **para este problema y esta configuracion de probabilidades**, la regla
  `argmax(w * p)` produjo una mejora reproducible y transferible en macro-F1.

- **Un detalle que la transferencia confirma, y corrige el registro**: en el ciclo anterior hubo que
  corregir un error mio de clasificacion —se confundian tres numeros del interno (nested, global
  congelado, redondeado) y la submission ejecuta el **global**—. El leaderboard lo confirma por su
  cuenta con la calibracion del proyecto:

  | | CV | LB predicho | LB real | error |
  |---|---|---|---|---|
  | V30 | 0,6134471 | 0,63301 | 0,63302 | **+0,00001** |
  | plug-in (CV nested) | 0,6154643 | 0,63508 | 0,63575 | +0,00067 |
  | plug-in (CV global) | 0,6159732 | 0,63560 | 0,63575 | **+0,00015** |

  El leaderboard casa con el **global**, que es lo que la submission ejecuta. La correccion de
  aquel ciclo no era cosmetica: era la descripcion correcta de la operacion.

- **La correccion historica que sigue vigente**: `V20-F3-UMBRALES` queda documentado como
  **arquitectura ordinal** —una cadena de cinco discriminadores binarios anidados con monotonia— y
  **no** como una evaluacion de regla de decision post-hoc. La familia de reglas de decision **no
  habia sido auditada correctamente** antes de esta fase, y ese error mio de clasificacion
  contribuyo a que la linea pareciera cerrada. Los informes historicos no se han modificado.

- **Decision**: la investigacion de AGE queda **cerrada**. No se abre ninguna grid, ningun peso
  nuevo, ningun modelo nuevo, ninguna representacion nueva, ninguna submission nueva. **El
  leaderboard no se usa para nada mas**: entra como evidencia final y no hay ninguna decision
  posterior que dependa de el. P33, P34, G12, G14 y la revision bibliografica no se reabren salvo
  evidencia externa nueva o cambio de reglas.

- **Consequences**: AGE tiene un candidato validado internamente en dos particiones, confirmado
  externamente y adoptado. El camino que lo produjo no fue una representacion mejor —se probaron y
  cerraron 39 mecanismos de aprendizaje— sino darse cuenta de que **faltaba auditar como se decide
  con lo aprendido**. La auditoria de 39 mecanismos sigue siendo valida; lo que muestra esto es que
  su enumeracion estaba incompleta.
"""

ESTADO = f"""
---

# CAMPEON FINAL DE AGE (2026-10-13)

```text
V30     = HISTORICAL_FROZEN_BASELINE   LB = {LB_V30}
PLUG-IN = ADOPTED                      LB = {LB_PLUG}
plug-in delta LB = +0,00273

research_status   = CLOSED
candidate_status  = ADOPTED
external_transfer = POSITIVE_TRANSFER
```

## El candidato

```text
V30 + regla de decision   argmax_c( w_c * p_c )
w = [1.00, 1.17, 1.27, 1.16, 1.00, 0.81]
```

**No** cambia el modelo, la representacion, los hiperparametros ni la semilla. Cambia **como se
convierte la probabilidad en clase**.

## La comparacion

| | CV canonico | CV alternativo | leaderboard |
|---|---|---|---|
| V30 | {CV_V30} | {CV_V30_ALT} | {LB_V30} |
| **plug-in** | **{CV_PLUG}** | **{CV_PLUG_ALT}** | **{LB_PLUG}** |
| delta | **+0,0020172** | **+0,0020580** | **+0,00273** |

## Reproduccion

```text
submissions/age_submission_plugin_candidate.csv
SHA256 {SHA_PLUG}
112.892 filas | columnas id, age_range | 10/10 checks
```

```text
submissions/age_submission_v30_final_candidate.csv
SHA256 {SHA_V30}
```

Los dos ficheros quedan **sin sobrescribir**.

## La conclusion, con su alcance exacto

La mejora **no** proviene de una representacion nueva ni de un modelo nuevo: proviene de la **regla
de decision**. Y la afirmacion esta acotada a proposito: **para este problema y esta configuracion de
probabilidades**, `argmax(w * p)` produjo una mejora reproducible y transferible en macro-F1. No se
afirma que V30 fuera incorrecto ni que `argmax` no sea optimo en general.

## Lo que no se hace a partir de aqui

No se usa el 0,63575 para optimizar pesos, crear variantes, seleccionar semillas ni buscar modelos.
No se genera otra submission. No se reabre P33, P34, G12, G14 ni la revision bibliografica salvo
evidencia externa nueva o cambio de reglas.
"""

RUNLOG = f"""

---

## Cierre final: el plug-in se adopta · 2026-10-13

- **Leaderboard del plug-in: {LB_PLUG}** frente a {LB_V30} de V30. Delta **+0,00273**.
- **`POSITIVE_EXTERNAL_TRANSFER`**.
- **Estado final:** `V30 = HISTORICAL_FROZEN_BASELINE`, `PLUG-IN = ADOPTED`,
  `research_status = CLOSED`.
- **Comparacion:** CV canonico +0,0020172 | CV alternativo +0,0020580 | LB +0,00273. Las tres
  direcciones coinciden.
- **Calibracion del proyecto:** el LB casa con el CV **global** (error +0,00015) y no con el nested
  (+0,00067), lo que confirma que la correccion de los tres numeros del ciclo anterior era la
  descripcion correcta de la operacion.
- **Submissions nuevas: 0.** **Tuning post-leaderboard: 0.** El leaderboard entra como evidencia
  final, no como funcion objetivo.
- **Los dos ficheros de submission quedan sin sobrescribir.**
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


def figura_transferencia() -> str:
    """Validacion frente a transferencia externa, con la misma direccion visible."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    ax1.bar(["V30", "plug-in"], [CV_V30, CV_PLUG], color=["#90a4ae", "#2e7d32"])
    # El IC95 es del DELTA, no del CV: hay que trasladarlo al nivel del CV sumando V30.
    lo = CV_V30 + 0.0010143
    hi = CV_V30 + 0.0029480
    ax1.errorbar([1], [CV_PLUG], yerr=[[CV_PLUG - lo], [hi - CV_PLUG]],
                 fmt="none", ecolor="#1b5e20", capsize=6, lw=1.6,
                 label="IC95 bootstrap del delta")
    ax1.set_ylabel("macro-F1 en validacion agrupada")
    ax1.set_title(f"Validacion: +{CV_PLUG - CV_V30:.5f}", fontsize=10)
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(fontsize=7)
    ax2.bar(["V30", "plug-in"], [LB_V30, LB_PLUG], color=["#90a4ae", "#2e7d32"])
    ax2.set_ylabel("leaderboard de Kaggle")
    ax2.set_title(f"Transferencia externa: +{LB_PLUG - LB_V30:.5f}", fontsize=10)
    ax2.grid(axis="y", alpha=0.25)
    fig.suptitle("La mejora aparece en validacion Y en test: la misma direccion", fontsize=11)
    fig.tight_layout()
    salida = cfg.FIGURES / "age_transfer_comparison.png"
    fig.savefig(salida, dpi=110)
    plt.close(fig)
    return salida.name


def main() -> int:
    """Cierra el ciclo: transfiere, adopta, documenta y no toca nada mas."""
    autotest_numerico()
    inicio = time.time()
    marca = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sha_v30 = hashlib.sha256(
        (ROOT / "submissions" / "age_submission_v30_final_candidate.csv").read_bytes()
    ).hexdigest()
    sha_plug = hashlib.sha256(SUBMISSION.read_bytes()).hexdigest()
    print("=== integridad de los dos ficheros ===", flush=True)
    print(f"  V30     {sha_v30[:16]} | intacto: {sha_v30 == SHA_V30}", flush=True)
    print(f"  plug-in {sha_plug[:16]} | intacto: {sha_plug == SHA_PLUG}", flush=True)
    if sha_plug != SHA_PLUG:
        print("[STOP] la submission del plug-in cambio de hash. No se documenta como adoptada.",
              flush=True)
        return 1

    print("\n=== tabla de transferencia ===", flush=True)
    # Hay DOS valores legitimos de V30 en la particion alternativa, y no son el mismo:
    #   0,6143347  lo reporto V30-05 en su fase
    #   0,6144120  lo midio la reejecucion confirmatoria, en la MISMA sesion que el plug-in
    # La comparacion internamente consistente es contra 0,6144120. Se declaran las dos.
    tabla = pd.DataFrame([
        {"metodo": "V30", "CV_canonico": CV_V30, "CV_alternativo": CV_V30_ALT, "LB": LB_V30,
         "delta_CV_canonico": 0.0, "delta_CV_alternativo": 0.0, "delta_LB": 0.0,
         "rol": "baseline historico congelado"},
        {"metodo": "plug-in", "CV_canonico": CV_PLUG, "CV_alternativo": CV_PLUG_ALT, "LB": LB_PLUG,
         "delta_CV_canonico": round(CV_PLUG - CV_V30, 7),
         "delta_CV_alternativo": round(CV_PLUG_ALT - CV_V30_ALT_REEJEC, 7),
         "delta_LB": round(LB_PLUG - LB_V30, 5), "rol": "campeon adoptado"},
    ])
    tabla.to_csv(TRANSFER_CSV, index=False)
    print(tabla.to_string(index=False), flush=True)
    fig = figura_transferencia()
    print(f"\n  figura: {fig}", flush=True)

    a, b = CALIBRACION
    calib = []
    for nombre, cv, real in (("V30", CV_V30, LB_V30), ("plug-in (nested)", CV_PLUG, LB_PLUG),
                             ("plug-in (global)", CV_PLUG_GLOBAL, LB_PLUG)):
        pred = a + b * cv
        calib.append({"metodo": nombre, "CV": cv, "LB_predicho": round(pred, 5),
                      "LB_real": real, "error": round(real - pred, 5)})
    print("\n  calibracion del proyecto:", flush=True)
    for c in calib:
        print(f"    {c['metodo']:20s} CV {c['CV']:.7f} -> pred {c['LB_predicho']:.5f} | "
              f"real {c['LB_real']:.5f} | error {c['error']:+.5f}", flush=True)

    TRANSFER_MD.write_text(_md_transfer(tabla, calib, sha_plug), encoding="utf-8")
    CHAMPION_MD.write_text(_md_champion(tabla, sha_plug, sha_v30), encoding="utf-8")
    for f in (TRANSFER_MD, CHAMPION_MD):
        print(f"  {f.name}: {len(f.read_text(encoding='utf-8').splitlines())} lineas", flush=True)

    RESULTADO.write_text(json.dumps({
        "fase": "cierre final: transferencia externa positiva", "fecha": FECHA,
        "experimenta": False,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent,
                                 capture_output=True, text=True, timeout=30).stdout.strip(),
        "comparacion": tabla.to_dict("records"), "calibracion": calib,
        "estado": {"V30": "HISTORICAL_FROZEN_BASELINE", "plug_in": "ADOPTED",
                   "research_status": "CLOSED", "candidate_status": "ADOPTED",
                   "external_transfer": "POSITIVE_TRANSFER"},
        "leaderboards": {"V30": LB_V30, "plug_in": LB_PLUG, "delta": round(LB_PLUG - LB_V30, 5)},
        "submission": {"path": str(SUBMISSION.relative_to(ROOT)), "sha256": sha_plug,
                       "filas": 112892, "score": LB_PLUG, "fecha_de_registro": marca},
        "correccion_historica_vigente": {
            "V20-F3-UMBRALES": "arquitectura ordinal (cadena de cinco discriminadores binarios "
                               "anidados con monotonia), NO una regla de decision post-hoc",
            "consecuencia": "la familia de reglas de decision no habia sido auditada correctamente "
                            "antes de esta fase",
        },
        "no_se_hace": ["grid nueva", "pesos nuevos", "modelo nuevo", "representacion nueva",
                       "submission nueva", "tuning con el leaderboard", "reabrir P33/P34/G12/G14"],
        "submissions_nuevas": 0, "tuning_post_leaderboard": 0,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s", flush=True)
    return 0


def _md_transfer(tabla, calib, sha) -> str:
    return "\n".join([
        "# Transferencia externa del plug-in por clase", "",
        "## El resultado", "",
        f"```text\nV30      leaderboard = {LB_V30}\nplug-in  leaderboard = {LB_PLUG}\n"
        f"delta                 = +{LB_PLUG - LB_V30:.5f}\n```", "",
        "**`POSITIVE_EXTERNAL_TRANSFER`.** La mejora medida internamente se transfirio al test.", "",
        "## La comparacion, con las tres columnas", "",
        "| Metodo | CV canonico | CV alternativo | Leaderboard | delta CV can. | delta CV alt. | "
        "delta LB |", "|---|---|---|---|---|---|---|",
    ] + [f"| {r['metodo']} | {r['CV_canonico']} | {r['CV_alternativo']} | {r['LB']} | "
         f"{r['delta_CV_canonico']:+.7f} | {r['delta_CV_alternativo']:+.7f} | "
         f"{r['delta_LB']:+.5f} |" for _, r in tabla.iterrows()] + [
        "", "**Las tres direcciones coinciden.** La validacion agrupada y la transferencia externa "
        "apuntan al mismo lado, que es lo que hace util el resultado.", "",
        "### Dos referencias legitimas para el delta alternativo", "",
        "No son el mismo numero y conviene no confundirlas:", "",
        "| Referencia de V30 en la particion alternativa | Valor | delta del plug-in |",
        "|---|---|---|",
        f"| lo que reporto V30-05 en su fase | {CV_V30_ALT} | "
        f"+{CV_PLUG_ALT - CV_V30_ALT:.7f} |",
        f"| lo que midio la reejecucion confirmatoria, en la misma sesion | {CV_V30_ALT_REEJEC} | "
        f"**+{CV_PLUG_ALT - CV_V30_ALT_REEJEC:.7f}** |", "",
        "La diferencia entre las dos referencias es **0,0000773**, y viene de que LightGBM no es "
        "reproducible bit a bit entre ejecuciones con distinta carga de CPU: el proyecto ya lo "
        "documento en V30. **La comparacion internamente consistente es la segunda**, porque el "
        "plug-in se midio en la misma sesion que esa reejecucion. La tabla de arriba usa esa.", "",
        "## El contraste con la calibracion del proyecto", "",
        "Con `LB = 0,00223 + 1,02825 x CV`, la calibracion que el propio proyecto ajusto en fases "
        "tempranas:", "",
        "| | CV | LB predicho | LB real | error |", "|---|---|---|---|---|",
    ] + [f"| {c['metodo']} | {c['CV']} | {c['LB_predicho']} | {c['LB_real']} | "
         f"**{c['error']:+.5f}** |" for c in calib] + [
        "", "El leaderboard del plug-in casa con el **CV global** (error `+0,00015`) mucho mejor que "
        "con el nested (`+0,00067`). Y el global es exactamente lo que la submission ejecuta: los "
        "pesos congelados aplicados de forma global.", "",
        "Eso importa porque en el ciclo anterior hubo que **corregir un error mio** de "
        "clasificacion: se estaban confundiendo tres numeros del interno y el sanity check **paro la "
        "construccion** antes de generar nada. La transferencia confirma que aquella correccion no "
        "era cosmetica, sino la descripcion correcta de la operacion.", "",
        "## Filas modificadas, como analisis descriptivo", "",
        "```text\nfilas que cambian respecto de argmax puro   7.408   (6,56 %)\n"
        "confianza media de las que cambian            0,3361\nconfianza media del conjunto   "
        "                0,6247\ncambios con confianza >= 0,60                 0\n```", "",
        "**Cero cambios en alta confianza.** Es la prediccion falsable que P34 establecio y que se "
        "cumplio en las dos particiones internas **y tambien en test**: un multiplicador por clase no "
        "puede alterar el orden cuando el margen es grande.", "",
        "**Este numero es descriptivo y no se usa para predecir el leaderboard.** El 0,63575 entra "
        "como evidencia final, no como funcion objetivo.", "",
        "## El cambio de distribucion, que es el mecanismo", "",
        "```text\n25-34   ~0,152  ->  0,1786\n65-xx             ->  0,1532\n```", "",
        "La regla redistribuye decisiones hacia las clases con peor F1. 25-34 era la peor (F1 0,4992) "
        "y la mas infra-predicha; 65-xx era la mejor (F1 0,7351) y estaba sobre-predicha. Los valores "
        "se documentan como estan y **no se reajustan**.", "",
        "## Alcance", "",
        f"La submission es `submissions/age_submission_plugin_candidate.csv`, SHA256 `{sha}`, y "
        f"**queda sin sobrescribir**. Este analisis **no se extrapola** a otras competiciones ni "
        f"datasets: la afirmacion vale para este problema y esta configuracion de probabilidades.", "",
    ]).rstrip("\n") + "\n"


def _md_champion(tabla, sha_plug, sha_v30) -> str:
    return "\n".join([
        "# Campeon final de AGE", "",
        f"```text\nV30     = HISTORICAL_FROZEN_BASELINE   LB = {LB_V30}\n"
        f"PLUG-IN = ADOPTED                      LB = {LB_PLUG}\n"
        f"plug-in delta LB = +{LB_PLUG - LB_V30:.5f}\n\n"
        f"research_status   = CLOSED\ncandidate_status  = ADOPTED\n"
        f"external_transfer = POSITIVE_TRANSFER\n```", "",
        "## 1. El problema detectado", "",
        "Treinta fases buscaron **como aprender** mejor: representaciones, familias de modelo, "
        "objetivos, estructura de etiquetas, integracion en score-space. La auditoria de cobertura "
        "cerro 39 mecanismos con evidencia, y **ninguno** produjo una mejora robusta.", "",
        "Lo que faltaba era otra cosa: **como decidir** con lo aprendido. Las seis clases de este "
        "problema son **perfectamente balanceadas** (0,160 a 0,170) y la prediccion del ensemble "
        "**no lo estaba** (0,146 a 0,187). `argmax` maximiza accuracy, no macro-F1, y cuando el "
        "reparto predicho se desvia hacia las clases faciles hay margen que `argmax` no aprovecha.", "",
        "## 2. La hipotesis", "",
        "> `argmax_c(w_c * p_c)` con pesos por clase puede subir macro-F1 sin tocar el modelo.", "",
        "## 3. La derivacion de los pesos", "",
        "Por cada fold `k` de la particion canonica se resolvio, por ascenso por coordenadas en "
        "log-espacio sobre los otros cuatro folds:", "",
        "```text\nw_k = argmax_w macro_F1( y[idx_k], argmax_c( w_c * p_c[idx_k] ) )\n"
        "idx_k = { i : fold(i) != k }\n```", "",
        "La clase 0 se ancla en 1,0 porque multiplicar todos los pesos por una constante no cambia "
        "ningun `argmax`: solo hay 5 grados de libertad reales. Los pesos adoptados son **la media "
        "aritmetica de los cinco**:", "",
        "```text\nw = [1,0000, 1,1749, 1,2721, 1,1630, 1,0010, 0,8128]\n"
        "    13-17   18-24   25-34   35-49   50-64   65-xx\n```", "",
        "## 4. La validacion canonica", "",
        "```text\nV30      0,6134471\nplug-in  0,6154643   delta +0,0020172   5/5 folds\n"
        "IC95 bootstrap  [+0,0010143, +0,0029480]   sin cero\n```", "",
        "## 5. La validacion alternativa", "",
        "```text\nV30      0,6144120\nplug-in  0,6164700   delta +0,0020580   5/5 folds\n"
        "IC95 bootstrap  [+0,0011204, +0,0030622]   sin cero\n```", "",
        "**Diferencia entre las dos particiones: 0,0000408.** Y el diagnostico `nested` —pesos "
        "re-derivados dentro de la particion alternativa, que es el test limpio del mecanismo— da "
        "**+0,0022884** con 5/5: el mecanismo aparece solo.", "",
        "## 6. El bootstrap", "",
        "Pareado, 2.000 remuestreos, unidad la fila, semilla 20261010 fijada en el pre-registro "
        "(`bd08aba`) y **no cambiada** para mejorar el intervalo. En las dos particiones el IC95 "
        "excluye el cero.", "",
        "## 7. Los sanity checks", "",
        "1. **Orden de clases demostrado, no asumido**: sobre el OOF, para cada clase real `i` la "
        "columna `i` es la que mas se activa. Diagonal dominante en las seis filas. Si no cayera, "
        "el script no genera submission.",
        "2. **Reproduccion del interno antes de tocar el test**: el script paro la construccion al "
        "detectar que se comparaba contra el numero equivocado de los tres posibles (nested, global, "
        "redondeado).",
        "3. **El pipeline es el de V30, no uno reconstruido**: se **importan** las funciones de "
        "`v30_03_submission.py` en vez de reimplementarlas.",
        "4. **Cero cambios en alta confianza** en las dos particiones y en test.", "",
        "## 8. La submission", "",
        f"```text\nsubmissions/age_submission_plugin_candidate.csv\nSHA256 {sha_plug}\n"
        f"112.892 filas | columnas id, age_range | 10/10 checks\n```", "",
        f"El fichero de V30 conserva su hash `{sha_v30[:16]}` y **no se ha sobrescrito**.", "",
        "## 9. El leaderboard", "",
        f"```text\nV30      {LB_V30}\nplug-in  {LB_PLUG}\ndelta    +{LB_PLUG - LB_V30:.5f}\n```", "",
        "## 10. La transferencia", "",
        "**`POSITIVE_EXTERNAL_TRANSFER`.** Las tres medidas —CV canonico, CV alternativo y "
        "leaderboard— apuntan en la misma direccion.", "",
        "Y un contraste que conviene registrar: con la calibracion del proyecto, el leaderboard casa "
        "con el **CV global** (error +0,00015) mucho mejor que con el nested (+0,00067), que es "
        "exactamente lo que la submission ejecuta.", "",
        "## 11. La decision", "",
        "**`ADOPTED`.** Y la conclusion, con su alcance exacto:", "",
        "> La mejora **no** proviene de una representacion nueva ni de un modelo nuevo. Proviene de "
        "la **regla de decision** aplicada sobre las probabilidades del ensemble V30.", "",
        "**No se afirma** que V30 fuera incorrecto, ni que `argmax` no sea optimo en general. La "
        "afirmacion es acotada: para **este problema y esta configuracion de probabilidades**, "
        "`argmax(w * p)` produjo una mejora reproducible y transferible en macro-F1.", "",
        "## 12. Lo que no se hace a partir de aqui", "",
        "No se usa el 0,63575 para optimizar pesos, crear variantes, seleccionar semillas ni buscar "
        "modelos. No se genera otra submission. No se reabre P33, P34, G12, G14 ni la revision "
        "bibliografica salvo evidencia externa nueva o cambio de reglas.", "",
        "## 13. La correccion historica que sigue vigente", "",
        "`V20-F3-UMBRALES` fue una **arquitectura ordinal** —una cadena de cinco discriminadores "
        "binarios anidados con monotonia— y **no** una evaluacion de regla de decision post-hoc. La "
        "familia de reglas de decision **no habia sido auditada correctamente** antes de esta fase, "
        "y ese error de clasificacion contribuyo a que la linea pareciera cerrada. Los informes "
        "historicos no se han modificado.", "",
        "## 14. El camino, en una linea", "",
        "39 mecanismos de **aprendizaje** auditados y cerrados, y el resultado vino de un mecanismo "
        "de **decision**. La auditoria era valida; su enumeracion estaba incompleta.", "",
    ]).rstrip("\n") + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
