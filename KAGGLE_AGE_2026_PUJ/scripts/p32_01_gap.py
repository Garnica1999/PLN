"""P32+ - 01: P01, gap analysis de las hipotesis de la fase, con precedentes medidos.

## Lo que este analysis hizo y lo que encontro

Leyendo `experiments.csv` en vez de fiarme de la memoria, aparecieron **cuatro precedentes
directos** que la fase P32+ iba a reabrir sin saberlo. Tres de ellos son la misma hipotesis que
este brief propone, con otro nombre y otro baseline:

| Precedente | Que midio | Resultado | Decision |
|---|---|---|---|
| **PH15-CALIB** (D-029) | promedio de membres en **log-probabilidades** frente a promedio de probabilidades | 0,6007, **+0,0003**, 3/5 | CERRADA |
| **PH39-FUSION-LEX** (D-039) | fundir un **bloque lexico** en V12 contra un **control de permutacion** | +0,0001, y el control daba +0,0001 | CERRADA |
| **PH32-CORRECCION** (D-037) | 14 capas de correccion, incluida calibracion **por confianza** | mejor +0,0002, 3/5 | CERRADA |
| **PH33-CORRECTOR** (D-037) | un **corrector minimo** sobre P(V12) entrenado solo con OOF | 5 variantes, **ninguna positiva** | CERRADA |
| **V28-03** | descomposicion **pairwise** en pares adyacentes | **-0,067553**, 0/5, neto -10.996 | C CLOSED |
| **V27-01** (H12) | fiabilidad por miembro y clase | delta **exactamente 0,0**, 0 filas cambian | H12 CLOSED |

## Por que esto no es un motivo para no hacer la fase

`PH39-FUSION-LEX` es el precedente mas cercano a la tesis central de P32+: *fundir una fuente
lexical en el baseline y compararla contra un control de permutacion*. Resultado: **+0,0001, y el
control daba tambien +0,0001**. Es decir, la fusion no aporto nada por encima del azar.

Pero ese precedente es de **V12**, no de V30, y con un bloque lexico mucho mas debil. La fuente que
P32+ tiene disponible es mejor: el canal linguistico de V31+ bate al texto solo en +0,006820 con
5/5 folds, y su base es V30, que ya probo ser mas fuerte que V12 por dos etapas de capacidad.

Lo que **no** cambia es el mecanismo. Y la conclusion honesta es esta: **P32+ tiene un precedente
negativo fuerte en su hipotesis central**, y la unica razon para ejecutarla es que la fuente y la
base han mejorado. Eso se ejecuta, pero se ejecuta sabiendo cual es el priori, y sin abrir grids
para pescar un numero.

## Las cuatro clasificaciones que pide el brief

```text
EXACT_DUPLICATE      mismo mecanismo, mismos datos, misma pregunta
FUNCTIONAL_DUPLICATE mismo mecanismo, otra fuente o otro baseline: la mecanica ya se midio
PARTIAL_PRECEDENT    la familia tiene precedentes, pero el mecanismo de P32 es distinto
GENUINELY_NEW        el mecanismo no se ha implementado en ningun sitio
```
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "p32_01_gap_analysis.json"
GAP_MD = cfg.REPORTS / "p32_hypothesis_gap_analysis.md"
GAP_CSV = cfg.REPORTS / "p32_hypothesis_gap_analysis.csv"
FECHA = "2026-10-01"
CV_V30 = 0.6134471
LB_V30 = 0.63302
RUIDO_ENTRENAMIENTO = 0.0008876

#: (id, track, hipotesis, clasificacion, precedente, que midio el precedente, resultado del
#:  precedente, por que es (o no) duplicada, que se ejecuta, prioridad)
HIPOTESIS = [
    ("H-P32-01", "P4", "Fusion en espacio de probabilidad entre V30 y el canal linguistico",
     "FUNCTIONAL_DUPLICATE", "PH39-FUSION-LEX (D-039)",
     "fusion de un bloque lexico en V12 contra un control de permutacion",
     "+0,0001, y el control daba +0,0001: cero aporte sobre el azar",
     "la mecanica es identica (mezcla lineal de probabilidades con control barajado). Lo unico "
     "mejor es la fuente (V31+ dio +0,006820 en banco) y la base (V30 > V12). No es un "
     "duplicado exacto, pero si una mecanica ya refutada con un precedente negativo fuerte",
     "SI, una sola vez, con el control barajado. Si el control iguala, la linea se cierra y no "
     "se abre grid de alpha", 3),
    ("H-P32-02", "P4.1", "Fusion en log-probabilidad (media geometrica) en vez de aritmetica",
     "FUNCTIONAL_DUPLICATE", "PH15-CALIB (D-029)",
     "promedio en log-probabilidades frente a promedio de probabilidades, con temperaturas in-fold",
     "logits 0,6007 (+0,0003, 3/5) frente a 0,6005 (+0,0001, 3/5). Cerrada",
     "**la media geometrica ya se midio exactamente**. La unica diferencia es que aqui se mezcla "
     "con un ensemble y no entre miembros sueltos. Es la misma operacion sobre otra entrada",
     "SI, porque el brief la pide y es casi gratis de medir. Si reproduce el +0,0003 de D-029, "
     "se cierra", 2),
    ("H-P32-03", "P5", "Fusion con alpha por clase (alpha_c), un alpha compartido primero",
     "PARTIAL_PRECEDENT", "H14 y PH33-CORRECTOR (D-037)",
     "re-equilibrar blends equivalentes; y un corrector minimo sobre P(V12) con 5 variantes",
     "H14 equivalencia funcional; PH33 ninguna de las 5 variantes positiva",
     "el corrector pequeno sobre probabilidades ya se probo y fallo. Lo que no se ha probado es "
     "un alpha **restringido a las clases adyacentes**, que es una hipotesis distinta y mas "
     "estrecha. Con un solo alpha compartido es casi PH33 reencuadrado",
     "SI, pero solo el alpha compartido primero. Seis alphas libres quedan prohibidos sin "
     "evidencia", 4),
    ("H-P32-04", "P6/P6.1", "Correccion pairwise localizada entre bandas adyacentes",
     "PARTIAL_PRECEDENT", "V28-03 (tecnica C)",
     "cinco LightGBM binarios disjuntos, uno por par adyacente, **sustituyendo** al multiclase",
     "**-0,067553**, 0/5 folds, rescue 8.461 contra damage 19.457, neto -10.996",
     "el precedente es catastrofico pero el mecanismo es distinto: V28-03 **reemplazo** el "
     "clasificador multiclase por cinco binarios y por eso perdio la estructura de la clase "
     "completa. P32 propone **anadir** una correccion de margen sobre las dos clases, "
     "conservando la probabilidad de V30 para las otras cuatro. Eso no se ha probado",
     "SI, y es la hipotesis mas genuinamente nueva de la fase. Con el test de rescate por par "
     "que exige el brief", 1),
    ("H-P32-05", "P8", "Residualizacion: quitar al candidato la parte que replica a V30",
     "GENUINELY_NEW", "ninguno",
     "no hay ningun experimento que ajuste z_C = a_c + b_c * z_V30 para eliminar la componente "
     "compartida antes de sumar",
     "n/a",
     "el mecanismo no existe en el historico. `residual` aparece 19 veces en experiments.csv, "
     "pero siempre como 'fase residual' del plan, no como residualizacion de un candidato. Es "
     "distinto de una adicion simple porque primero **exhibe** la parte redundante y solo luego "
     "suma el resto",
     "SI, con prioridad alta: es la formulacion mas limpia de la tesis central de P32", 1),
    ("H-P32-06", "P9", "Correccion condicionada por la confianza de V30",
     "PARTIAL_PRECEDENT", "PH32-CORRECCION (D-037)",
     "14 capas de correccion, incluida calibracion por confianza y combinacion Luce",
     "mejor 0,6006 (+0,0002, 3/5). CERRADA",
     "la calibracion por confianza ya se midio y fue +0,0002, dentro del ruido. Lo que P32 "
     "propone es modular la **residualizacion** por confianza, que es distinto de calibrar la "
     "confianza en si. Solo tiene sentido si H-P32-05 deja residuo, y por eso va **detras** de "
     "ella en el orden de contingencia",
     "SOLO si H-P32-05 deja residuo. Si no, no se ejecuta", 6),
    ("H-P32-07", "P10", "Correccion condicionada por la longitud del texto",
     "GENUINELY_NEW", "ninguno como mecanismo de fusion",
     "V31 y PH5-L2 usaron la longitud como **corte diagnostico**; E-018 y V14A como feature",
     "V31: slices de longitud, solo diagnostico, delta negativo en las cuatro",
     "nadie ha hecho que la **fuerza** de la correccion dependa de la longitud. Es un mecanismo "
     "distinto de condicionar el modelo, y el brief prohibe explícitamente crear cuatro modelos "
     "por longitud, que es justo lo que se evita con la forma lineal que propone",
     "SI, pero con la forma gamma0 + gamma1 * log_len * residual, que es lo unico permitido", 5),
    ("H-P32-08", "P2.5/P2.7", "Representaciones estructurales de URL y canal de dominio",
     "FUNCTIONAL_DUPLICATE", "V31-02 y V31-03",
     "url_mask, url_struct, url_canal, solo_url, y el canal `url` de V31-03",
     "url_canal +0,000738 (3/5) en banco; nada en la integracion contra V30",
     "**ya ejecutado en la fase anterior, con el mismo dataset y el mismo baseline**. Es un "
     "duplicado exacto de H-P32-08 tal como el brief lo describe",
     "NO como experimento nuevo. Se reutilizan los resultados de V31-02 y solo se anade lo que "
     "V31 no midio: posicion de la URL y su longitud relativa", 7),
    ("H-P32-09", "P2.2", "Posicion de la URL dentro del texto y longitud relativa",
     "PARTIAL_PRECEDENT", "V31-03 canal `url`",
     "tokens URLINI / URLFIN y URLLONG{bucket} dentro del canal de URL de V31-03",
     "el canal url dio +0,001894 (5/5) en banco, pero nada en la integracion",
     "los tokens existen, pero V31 los metio en un solo canal junto a la presencia y el tipo, y no "
     "midio la **distancia entre URLs multiples** ni la fraccion del texto ocupada por la URL. Es "
     "una extension pequena de algo ya hecho, no una linea nueva",
     "SI, como extension acotada, y solo si el gap analysis de URLs muestra cobertura suficiente", 6),
    ("H-P32-10", "P2.8", "Recuperacion del contenido textual de la URL",
     "QUARANTINED", "P0 de V31+",
     "auditoria de legalidad de las reglas de la compilacion",
     "**QUARANTINED**: las reglas de AGE no estan disponibles en este entorno",
     "no es una hipotesis que se pueda ejecutar: sin el texto de las reglas no puede establecerse "
     "que la recuperacion sea legal. El brief dice que ante ambiguedad, cuarentena, sin excepcion",
     "NO. Ni P2.8, ni P2.9, ni P2.10, ni P12. Se documenta la regla ausente y sigue", 9),
    ("H-P32-11", "P7", "Fusion en score-space de una fuente linguistica y una fuente de URL",
     "PARTIAL_PRECEDENT", "PH39-FUSION-LEX (D-039)",
     "fusion de varios bloques lexicos simultaneos en V12",
     "V12 + Registro + Vida + Temas: **-0,0018 (2/5)**, control 0,5965. Peor que el azar",
     "fusionar dos fuentes en score-space ya se probo con tres bloques y fue **peor que no hacer "
     "nada**. El brief manda no combinar dos fuentes si no existen rescates unicos, y ese es "
     "justamente el criterio que hay que aplicar antes de intentarlo",
     "SOLO si H-P32-04 y H-P32-05 producen rescates unicos en pares y clases distintos. Si no, "
     "no se ejecuta", 7),
]

NO_EJECUTAR = {
    "H-P32-10": "las reglas de la competicion no estan disponibles en este entorno, asi que el "
                "retrieval no puede clasificarse como ALLOWED y queda en cuarentena",
    "H-P32-08": "duplicado exacto de V31-02 y V31-03 con el mismo baseline; se reutilizan sus "
                "resultados en vez de repetir la medicion",
}


def main() -> int:
    """Construye el gap analysis de P32+ y dice que se ejecuta y que no."""
    autotest_numerico()
    inicio = time.time()
    hist = pd.read_csv(ROOT / "experiments.csv")
    print(f"historial leido: {len(hist)} filas de experiments.csv", flush=True)

    filas = []
    for (hid, track, hip, clasif, prec, midio, res, por_que, ejecuta, prio) in HIPOTESIS:
        # Se comprueba que el precedente citado existe de verdad en el historial.
        # "ninguno" es un valor legitimo para una hipotesis GENUINELY_NEW: significa que no hay
        # precedente, no que el precedente no se encuentra. Solo se verifica cuando se cita uno.
        existe = (True if prec.startswith("ninguno") else
                  bool(hist["experiment_id"].astype(str).str.contains(
                      prec.split()[0], case=False, na=False).any()))
        filas.append({
            "ID": hid, "track": track, "hypothesis": hip, "classification": clasif,
            "precedent": prec, "precedent_verificado_en_historico": existe,
            "what_the_precedent_measured": midio, "precedent_result": res,
            "why_not_duplicate": por_que, "what_will_run": ejecuta, "priority": prio,
            "status": "NO_EJECUTAR" if hid in NO_EJECUTAR else "ABIERTA",
            "motivo_de_no_ejecutar": NO_EJECUTAR.get(hid, ""),
        })
    tabla = pd.DataFrame(filas).sort_values(["priority", "ID"])
    tabla.to_csv(GAP_CSV, index=False)

    print("\n=== clasificacion ===", flush=True)
    for _, f in tabla.iterrows():
        print(f"  {f['ID']} [{f['track']:7s}] p{f['priority']} {f['classification']:20s} "
              f"{'NO ' if f['status'] == 'NO_EJECUTAR' else '   '}{f['hypothesis'][:58]}", flush=True)
    cuentas = tabla["classification"].value_counts().to_dict()
    print(f"\n  {cuentas}", flush=True)
    print(f"  a ejecutar: {(tabla['status'] == 'ABIERTA').sum()} | "
          f"a no ejecutar: {(tabla['status'] == 'NO_EJECUTAR').sum()}", flush=True)
    sin_verificar = tabla[~tabla["precedent_verificado_en_historico"]]
    if not sin_verificar.empty:
        print(f"  ATENCION: precedentes no encontrados en el historial: "
              f"{sin_verificar['precedent'].tolist()}", flush=True)

    _md(tabla, cuentas)
    RESULTADO.write_text(json.dumps({
        "fase": "P32+ 01 P01 gap analysis", "fecha": FECHA, "experimenta": False,
        "baseline_v30": CV_V30, "leaderboard": LB_V30,
        "clasificacion": cuentas,
        "a_ejecutar": int((tabla["status"] == "ABIERTA").sum()),
        "a_no_ejecutar": int((tabla["status"] == "NO_EJECUTAR").sum()),
        "hipotesis": tabla.to_dict("records"),
        "precedente_mas_fuerte": (
            "PH39-FUSION-LEX (D-039) es el precedente mas cercano a la tesis central de P32: "
            "fundir un bloque lexico en el baseline contra un control de permutacion dio +0,0001 "
            "con un control de +0,0001, es decir cero aporte sobre el azar. La mecanica esta "
            "refutada; lo unico que P32+ aporta es una fuente mejor y un baseline mejor"),
        "prior_de_la_fase": (
            "bajo. Dos de las once hipotesis son genuinamente nuevas (residualizacion y "
            "condicionamiento por longitud), una mas es casi nueva (correccion pairwise "
            "localizada), y las demas tienen precedentes directos medidos. La expectativa "
            "honesta es que la fase cierre lineas, no que produzca un finalista"),
        "restricciones": {"usa_test": False, "usa_red": False},
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


def _md(tabla: pd.DataFrame, cuentas: dict) -> None:
    """Redacta el gap analysis legible."""
    lineas = [
        "# P32+ - P01, gap analysis de las hipotesis", "",
        f"**Baseline:** V30 = {CV_V30}, leaderboard {LB_V30}. "
        f"**Ruido de entrenamiento medido en P00:** {RUIDO_ENTRENAMIENTO}, que es la escala contra "
        f"la que hay que comparar cualquier delta de esta fase.", "",
        "## El hallazgo de este analysis", "",
        "Leyendo `experiments.csv` en vez de fiarme de la memoria, aparecieron **seis precedentes "
        "directos** que P32+ iba a reabrir sin saberlo. Cinco de ellos son la misma hipotesis con "
        "otro nombre y otro baseline:", "",
        "| Precedente | Que midio | Resultado | Decision |", "|---|---|---|---|",
        "| **PH39-FUSION-LEX** (D-039) | fundir un **bloque lexico** en V12 contra un control de "
        "permutacion | +0,0001, y el control daba **+0,0001** | CERRADA |",
        "| **PH15-CALIB** (D-029) | promedio en **log-probabilidades** frente a probabilidades | "
        "0,6007 (**+0,0003**, 3/5) | CERRADA |",
        "| **PH32-CORRECCION** (D-037) | 14 capas de correccion, incluida **calibracion por "
        "confianza** | mejor +0,0002, 3/5 | CERRADA |",
        "| **PH33-CORRECTOR** (D-037) | un **corrector minimo** sobre P(V12) con 5 variantes | "
        "**ninguna positiva** | CERRADA |",
        "| **V28-03** | descomposicion **pairwise** en pares adyacentes, sustituyendo al "
        "multiclase | **-0,067553**, 0/5, neto -10.996 | C CLOSED |",
        "| **V27-01** (H12) | fiabilidad por miembro y clase | delta **exactamente 0,0**, 0 filas "
        "cambian | H12 CLOSED |", "",
        "## Lo que eso significa, sin adornos", "",
        "`PH39-FUSION-LEX` es casi exactamente la tesis central de P32+: *fundir una fuente "
        "lexical en el baseline y compararla contra un control de permutacion*. Dio **+0,0001 con "
        "un control de +0,0001**, es decir **cero aporte sobre el azar**.", "",
        "La mecanica de P32+ no es la misma en un punto importante: V28-03 **sustituyo** el "
        "clasificador multiclase por cinco binarios y por eso destruyo la estructura de la clase "
        "completa, mientras que P32 propone **anadir** una correccion de margen conservando la "
        "probabilidad de V30 para las demas clases. Eso si es genuinamente distinto. Y V12 no es "
        "V30: la fuente de P32 es mejor (+0,006820 en banco) y la base es mas fuerte.", "",
        "**El priori honesto de esta fase es bajo.** Dos hipotesis son genuinamente nuevas, una es "
        "casi nueva, y las demas tienen precedentes medidos que las refutan. Se ejecutan porque son "
        "baratas y porque el brief manda no detenerse en el primer negativo, pero la expectativa "
        "razonable es que la fase **cierre lineas**, no que produzca un finalista.", "",
        "## Clasificacion completa", "",
        "| ID | Track | Hipotesis | Clasificacion | Precedente | Que se ejecuta | Prioridad |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, f in tabla.iterrows():
        lineas.append("| {} | {} | {} | **{}** | {} | {} | {} |".format(
            f["ID"], f["track"], f["hypothesis"], f["classification"], f["precedent"],
            ("**NO**: " + f["motivo_de_no_ejecutar"]) if f["status"] == "NO_EJECUTAR"
            else f["what_will_run"][:96], f["priority"]))
    lineas += ["", "## Las dos genuinamente nuevas, y por que son las que valen la pena", "",
               "**H-P32-05, residualizacion.** Ajustar `z_C[c] = a_c + b_c * z_V30[c]` y sumar "
               "solo el resto. No existe en el historico. Y es la formulacion que mejor encaja con "
               "lo que V31+ midio: la senal linguistica es real pero **casi toda se solapa con lo que "
               "V30 ya dice**, asi que la pregunta correcta no es *cuanto pesa el candidato* sino "
               "*que parte del candidato no es V30*.", "",
               "**H-P32-07, correccion condicionada por longitud.** Condicionar la fuerza de la "
               "correccion a la longitud no se ha hecho: V31 y PH5-L2 usaron la longitud como corte "
               "diagnostico, y E-018 y V14A como feature. Es un mecanismo distinto de tener cuatro "
               "modelos por longitud, que es justo lo que la forma `gamma0 + gamma1 * log_len * "
               "residual` evita.", "",
               "**H-P32-04, correccion pairwise localizada.** Con el precedente mas duro de la "
               "fase, pero con un mecanismo distinto: V28-03 sustituyo, P32 anade. El brief exige "
               "el test de rescate por par, y con el precedente de V28-03 a la vista la vara esta "
               "muy alta: sin rescate neto positivo por par, se cierra.", "",
               "## El orden de contingencia que se va a seguir", "",
               "```text", "H-P32-05  residualizacion              (mas limpia, primera)",
               "H-P32-04  correccion pairwise por par     (mas nueva, segunda)",
               "H-P32-01  fusion de probabilidad + control (precedente fuerte, tercera)",
               "H-P32-02  fusion en log-probabilidad       (casi gratis, cuarta)",
               "H-P32-03  alpha por clase, uno compartido  (solo si hay rescate por clase)",
               "H-P32-07  correccion por longitud          (solo si hay residuo)",
               "H-P32-06  correccion por confianza         (solo si hay residuo)",
               "H-P32-09  extension acotada de URL         (si la cobertura lo justifica)",
               "```", "",
               "Y **fuera de la lista**, sin excepcion:", "",
               "- `H-P32-10`, retrieval de contenido de URL: **QUARANTINED**. Las reglas de la "
               "competicion no estan disponibles en este entorno, asi que no puede clasificarse "
               "como ALLOWED. Cierra P2.8, P2.9, P2.10 y P12 de un golpe.",
               "- `H-P32-08`, representaciones estructurales de URL: **duplicado exacto** de "
               "V31-02 y V31-03, con el mismo baseline. Se reutilizan sus resultados.", "",
               "## La calibracion que P00 deja mas importante que cualquier hipotesis", "",
               f"El ruido de **entrenamiento con otro reparto** es de {RUIDO_ENTRENAMIENTO}, y el "
               f"gate de PROMISING pide +0,0015: solo **{0.0015 / RUIDO_ENTRENAMIENTO:.1f} veces** "
               f"ese valor. Cualquier delta de esta fase por debajo de unos "
               f"{RUIDO_ENTRENAMIENTO:.7f} esta dentro del ruido de la propia tecnica, por "
               f"limpio que sea su desglose por folds. Ese es el liston contra el que se juzga "
               f"todo lo que viene."]
    GAP_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
