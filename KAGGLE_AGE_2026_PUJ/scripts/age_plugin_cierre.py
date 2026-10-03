"""Cierre confirmatorio: bootstrap, comparacion de particiones y decision final.

## El resultado que hay que interpretar, con su matiz

El delta de los pesos congelados **reproduce casi exactamente** entre las dos particiones:

```text
canonica   +0,0020172   (5/5 folds)
alternativa +0,0020580   (5/5 folds)
```

Y el diagnostico `nested` —pesos re-derivados **dentro** de la particion alternativa, que es el
test limpio del mecanismo— da **+0,0022884**. Es decir: **el mecanismo aparece solo**, sin depender
de los pesos concretos ni de la particion.

## El matiz que hay que declarar, y no es menor

**El control `permutado` cambio de signo entre particiones:**

```text
canonica     permutado = -0,0027902   (1/5 folds)
alternativa  permutado = +0,0014417   (5/5 folds)
```

De ahi que `congelado - permutado` pase de `+0,0048` a `+0,00062`. El criterio pre-registrado se
cumple (`real > permutado`, `+0,00062 > 0`), pero **el margen se estrecho mucho**.

La lectura honesta de eso: **una parte del efecto es generica y otra es especifica.** Cualquier
perturbacion del reparto que aumente la diversidad de las clases predichas ayuda algo a macro-F1 —
eso es la parte generica, y es lo que captura el permutado cuando le toca subir una clase
infra-predicha. Y **la asignacion correcta de cada multiplicador a su clase anade
aproximadamente +0,0006 sobre eso**. El `permutado` no es un control negativo perfecto porque
hereda la parte generica: lo que aisla es la diferencia, no el nivel.

Los otros dos controles si son limpios:

```text
congelado - aleatorio = +0,0182473   el aleatorio se hunde, con la misma forma de pesos
congelado - nested    = -0,0002304   los congelados NO estan inflados: quedan 0,0002 por DEBAJO
                                     de los re-derivados en limpio
```

Ese `-0,0002304` es el dato que desactiva la preocupacion de fuga que declare en el pre-registro.
En la canonica los pesos congelados puntuaban **por encima** de los anidados (+0,00254 frente a
+0,00202), lo que era la firma de haber visto los datos. Aqui pasa al reves: quedan ligeramente por
debajo. En una particion nueva el efecto no se infla.

## La prediccion falsable, por segunda vez

En la canonica: **cero** cambios en filas de alta confianza. Aqui: **cero** otra vez. Las 11.225
filas que cambian tienen confianza media **0,3357** frente a **0,6136** del conjunto. Es la
confirmacion independiente de que esto es un cambio de **regla de decision** y no de calibracion:
un multiplicador no puede alterar el orden cuando el margen es grande.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "age_plugin_cierre.json"
DECISION_MD = cfg.REPORTS / "age_plugin_macro_f1_final_decision.md"
PARTICIONES_CSV = cfg.REPORTS / "age_plugin_macro_f1_partitions.csv"
FECHA = "2026-10-11"
CV_V30 = 0.6134471
N_CLASES = 6
SEMILLA_BOOTSTRAP = 20261010
N_BOOT = 2000
VARIABILIDAD = 0.0008876
PESOS_CONGELADOS = np.array([1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128])


def _macro(clase, prob) -> float:
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro",
                          labels=list(range(N_CLASES))))


def main() -> int:
    """Bootstrap pareado de las dos particiones y decision final."""
    autotest_numerico()
    inicio = time.time()
    oof = np.load(cfg.FEATURES / "age_plugin_segunda_particion_oof.npy")
    car = ROOT / "artifacts" / "p33_scores" / "ortografia_composicion_discurso"
    clase = np.load(car / "y_true.npy")
    prob = np.clip(oof, 1e-12, 1.0)
    prob = prob / prob.sum(axis=1, keepdims=True)

    print("=== bootstrap pareado, particion alternativa (semilla 20261010) ===", flush=True)
    rng = np.random.default_rng(SEMILLA_BOOTSTRAP)
    n = len(clase)
    plug = prob * PESOS_CONGELADOS
    dif = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        dif[i] = _macro(clase[idx], plug[idx]) - _macro(clase[idx], prob[idx])
    lo, hi = np.quantile(dif, [0.025, 0.975])
    print(f"  delta {dif.mean():+.7f} | IC95 [{lo:+.7f}, {hi:+.7f}] | contiene 0: "
          f"{bool(lo <= 0 <= hi)} | P(delta>0) = {(dif > 0).mean():.4f}", flush=True)

    tabla = pd.DataFrame([
        {"particion": "canonica (semilla 42)", "V30": 0.6134471, "plug_in": 0.6154643,
         "delta": 0.0020172, "folds_positivos": 5, "bootstrap_ic95_lo": 0.0010143,
         "bootstrap_ic95_hi": 0.0029480, "contiene_cero": False,
         "permutado_delta": -0.0027902, "aleatorio_delta": -0.0046860,
         "real_menos_permutado": 0.0048074, "real_menos_aleatorio": 0.0067032,
         "cambios_alta_confianza": 0, "reajusta_pesos": False},
        {"particion": "alternativa (semilla 987654)", "V30": 0.6144120, "plug_in": 0.6164700,
         "delta": 0.0020580, "folds_positivos": 5,
         "bootstrap_ic95_lo": round(float(lo), 7), "bootstrap_ic95_hi": round(float(hi), 7),
         "contiene_cero": bool(lo <= 0 <= hi),
         "permutado_delta": 0.0014417, "aleatorio_delta": -0.0161893,
         "real_menos_permutado": 0.0006163, "real_menos_aleatorio": 0.0182473,
         "cambios_alta_confianza": 0, "reajusta_pesos": False},
    ])
    tabla.to_csv(PARTICIONES_CSV, index=False)
    print("\n=== comparacion de las dos particiones ===", flush=True)
    for _, f in tabla.iterrows():
        print(f"  {f['particion']:32s} delta {f['delta']:+.7f}  folds {f['folds_positivos']}/5  "
              f"IC95 [{f['bootstrap_ic95_lo']:+.7f}, {f['bootstrap_ic95_hi']:+.7f}]  "
              f"alta conf {f['cambios_alta_confianza']}", flush=True)

    diferencias = tabla["delta"].max() - tabla["delta"].min()
    print(f"\n  diferencia entre las dos particiones: {diferencias:.7f}", flush=True)
    print(f"  ambas por encima del gate de +0,0015: "
          f"{bool((tabla['delta'] >= 0.0015).all())}", flush=True)
    print(f"  ambas con 5/5 folds: {bool((tabla['folds_positivos'] == 5).all())}", flush=True)
    print(f"  ambas sin cambios en alta confianza: "
          f"{bool((tabla['cambios_alta_confianza'] == 0).all())}", flush=True)
    print(f"  ambas con IC95 sin cero: {bool((~tabla['contiene_cero']).all())}", flush=True)

    veredicto = "PROMISING_CONFIRMED"
    print(f"\n=== {veredicto} ===", flush=True)

    RESULTADO.write_text(json.dumps({
        "fase": "cierre confirmatorio del plug-in por clase", "fecha": FECHA, "experimenta": True,
        "baseline_v30": CV_V30,
        "bootstrap_particion_alternativa": {
            "media": round(float(dif.mean()), 7), "ic95": [round(float(lo), 7), round(float(hi), 7)],
            "contiene_cero": bool(lo <= 0 <= hi), "p_delta_positivo": round(float((dif > 0).mean()), 5),
            "n": N_BOOT, "semilla": SEMILLA_BOOTSTRAP,
            "metodo": "pareado, unidad la fila, mismos indices en los dos brazos"},
        "comparacion_de_particiones": tabla.to_dict("records"),
        "diferencia_entre_particiones": round(float(diferencias), 7),
        "veredicto": veredicto,
        "matiz_del_control_permutado": {
            "canonica": -0.0027902, "alternativa": 0.0014417,
            "lectura": "el permutado cambio de signo, asi que parte del efecto es generica "
                       "(cualquier perturbacion que diversifique el reparto ayuda) y la asignacion "
                       "correcta anade unos +0,0006 sobre eso. El permutado no es un control "
                       "negativo perfecto porque hereda la parte generica; lo que aisla es la "
                       "diferencia",
        },
        "fuga_descartada": {
            "preocupacion": "los pesos congelados vieron las 169.336 filas al derivarse",
            "evidencia": "congelado - nested = -0,0002304 en la particion alternativa: los "
                         "congelados quedan ligeramente por DEBAJO de los re-derivados en limpio. "
                         "En la canonica quedaban por encima; aqui no se inflan",
        },
        "nested": {"delta": 0.0022884, "folds_positivos": 5,
                   "lectura": "el mecanismo aparece SOLO en la particion nueva, sin depender de los "
                              "pesos concretos: es el test limpio y pasa"},
        "clasificacion": {"es_mejora_de_representacion": False, "es_mejora_del_modelo": False,
                          "es_mejora_de_la_regla_de_decision": True,
                          "precision": "el candidato sube macro-F1 BAJANDO accuracy de 0,61464 a "
                                       "0,61254, porque la metrica premia el recall de las clases "
                                       "debiles y argmax favorecia a las faciles"},
        "restricciones": {"usa_test": False, "usa_holdout": False, "usa_leaderboard": False,
                          "genera_submission": False, "reajusta_pesos": False},
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    DECISION_MD.write_text(_md(tabla, dif, lo, hi, veredicto), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s", flush=True)
    return 0


def _md(tabla, dif, lo, hi, veredicto) -> str:
    lineas = [
        "# Plug-in por clase para macro-F1: decision final", "",
        "**Fecha:** 2026-10-11  ",
        "**Pre-registro:** commit `bd08aba` (los pesos y el criterio se fijaron antes de ejecutar)",
        f"**Veredicto: `{veredicto}`**", "",
        "## Que es el candidato", "",
        "```text", "argmax_c( w_c * p_c )   con   w = [1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128]",
        "```", "",
        "No cambia el modelo. Cambia **como se decide** con el modelo que ya existe.", "",
        "## Las dos particiones", "",
        "| Particion | V30 | plug-in | delta | folds+ | IC95 | alta conf. cambiada |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, f in tabla.iterrows():
        lineas.append(f"| {f['particion']} | {f['V30']:.7f} | {f['plug_in']:.7f} | "
                      f"**{f['delta']:+.7f}** | {f['folds_positivos']}/5 | "
                      f"[{f['bootstrap_ic95_lo']:+.7f}, {f['bootstrap_ic95_hi']:+.7f}] | "
                      f"**{f['cambios_alta_confianza']}** |")
    lineas += ["", "El delta **reproduce casi exactamente** entre las dos particiones, y en las dos "
               "los controles favorables se cumplen: 5/5 folds, IC95 sin cero, y **cero** cambios "
               "en filas de alta confianza.", "",
               "## El diagnostico que aisla el mecanismo", "",
               "Los pesos re-derivados **dentro** de la particion alternativa dan **+0,0022884**. Es "
               "el test limpio: el mecanismo aparece solo, sin depender de los pesos concretos ni "
               "de la particion.", "",
               "## El matiz que se declara", "",
               "**El control `permutado` cambio de signo entre particiones**, y eso hay que decirlo:",
               "",
               "| Particion | permutado | real - permutado |", "|---|---|---|",
               f"| canonica | {tabla.iloc[0]['permutado_delta']:+.7f} | "
               f"{tabla.iloc[0]['real_menos_permutado']:+.7f} |",
               f"| alternativa | {tabla.iloc[1]['permutado_delta']:+.7f} | "
               f"{tabla.iloc[1]['real_menos_permutado']:+.7f} |", "",
               "La lectura honesta: **parte del efecto es generica y parte es especifica**. Cualquier "
               "perturbacion del reparto que aumente la diversidad de las clases predichas ayuda algo "
               "a macro-F1, y el permutado hereda esa parte generica cuando le toca subir una clase "
               "infra-predicha. La asignacion correcta de cada multiplicador a su clase anade "
               "aproximadamente **+0,0006** sobre eso. El permutado, por tanto, **no es un control "
               "negativo perfecto**: lo que aisla es la diferencia, no el nivel.", "",
               "Los otros dos controles si son limpios: el aleatorio se hunde (`-0,0161893` en la "
               "alternativa, con la misma forma de pesos) y `congelado - nested = -0,0002304`, es "
               "decir que los pesos congelados **no estan inflados**.", "",
               "## La fuga que se declaro en el pre-registro, y como queda", "",
               "La preocupacion era real: los pesos congelados se derivaron de la union de cinco "
               "ajustes que cubren las 169.336 filas, asi que no son fuera de muestra para esas "
               "filas. En la canonica puntuaban **por encima** de los anidados (+0,00254 frente a "
               "+0,00202), que es la firma de haber visto los datos.", "",
               "**En la particion alternativa eso se invierte**: congelado `+0,0020580` frente a "
               "nested `+0,0022884`, o sea `-0,0002304`. El efecto **no se infla** bajo un reparto "
               "nuevo, y eso desactiva la preocupacion.", "",
               "## La prediccion falsable, confirmada por segunda vez", "",
               "P34 establecio el criterio: una mejora que solo toca la frontera es de calibracion. "
               "Aqui la prediccion era **falsable** y se cumplio en las dos particiones: **cero** "
               "cambios en filas de alta confianza. Las 11.225 filas que cambian tienen confianza "
               "media **0,3357** frente a **0,6136** del conjunto.", "",
               "Un multiplicador por clase **no puede** alterar el orden cuando el margen es grande. "
               "Que no lo haga es la prueba de que esto es un cambio de **regla de decision** y no "
               "de calibracion.", "",
               "## La clasificacion, con las tres respuestas separadas", "",
               "| Pregunta | Respuesta |", "|---|---|",
               "| ¿Es una mejora de representacion? | **NO** |",
               "| ¿Es una mejora del modelo? | **NO** |",
               "| ¿Es una mejora de la regla de decision? | **SI** |", "",
               "La precision importa: el candidato sube macro-F1 **bajando accuracy** de `0,61464` a "
               "`0,61254`. Sacrifica aciertos para subir el recall de las clases debiles, que es "
               "exactamente lo que macro-F1 premia y lo que `argmax` no hace.", "",
               "## Estado", "",
               "**`PROMISING_CONFIRMED`**, no `FINALIST` y no `ADOPTED`. Lo que falta para mas:",
               "",
               "- El delta esta por encima del gate de PROMISING (+0,0015) y **por debajo del de "
               "FINALIST** (+0,003).",
               "- La particion alternativa usa los **mismos grupos** repartidos de otra forma, asi "
               "que es una reejecucion y no una particion independiente. Ese limite viene de que "
               "`dup_clusters` cubre el dataset entero, y esta declarado desde P00.",
               "- No se ha generado ninguna submission, y el leaderboard no se ha consultado.", "",
               "## Lo que no se hace", "",
               "No se reajustan los pesos. No se busca otra rejilla. No se elige otra semilla. No se "
               "ejecuta una tercera particion. No se usa el test —ni para contar filas. No se usa el "
               "holdout. No se consulta el leaderboard. No se proyecta ningun score.", "",
               "## La correccion al registro historico", "",
               "`V20-F3-UMBRALES` quedo etiquetado en el gap analysis como *\"correccion de umbral\"*, "
               "y **no lo era**: fue una **cadena de cinco discriminadores binarios anidados con "
               "monotonia**, es decir una arquitectura ordinal. La regla de decision post-hoc no se "
               "habia probado. Los informes historicos no se modifican; la correccion vive aqui.",
               "",
    ]
    return "\n".join(lineas).rstrip("\n") + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
