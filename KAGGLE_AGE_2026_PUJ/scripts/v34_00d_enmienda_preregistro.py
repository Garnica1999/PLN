"""V34 - E1 - Enmienda pre-ejecucion 2 del pre-registro: presupuesto del control A2.

## Defecto medido

El pre-registro congelaba el control A2 como `SGDClassifier(loss="log_loss", alpha=1e-6,
learning_rate="optimal", shuffle=False, fit_intercept=True, max_iter=P_MAX, tol=None)`. El
implementador (`ml-coder`) midio en `/tmp` que **`fit(max_iter=1, warm_start=True)` repetido 10
veces NO es un presupuesto de 10 pasadas**: `sklearn.linear_model.BaseSGDClassifier._fit` hace
`self.t_ = 1.0` al empezar cada llamada a `fit`, luego el contador `t_` del schedule
`1/(alpha*(t0+t))` se reinicia 10 veces y queda en `t_ = 401` en cada pasada (medido) en lugar de
`t_ = 4001`. Con ese esquema el control recibiria un presupuesto de un orden de magnitud menor que A1 y
la atribucion quedaria contaminada por un defecto de API, no por la hipotesis.

## Enmienda

El control se ajusta con **`partial_fit` una vez por pasada** (10 llamadas, una por orden congelado
de cada pasada), de modo que `t_` se acumula (`t_ = 4001`, identico al de `fit(max_iter=10)`), se
puede evaluar el early stopping **despues de cada pasada** (que el pre-registro exige: `P_min = 3`,
paciencia 2, restauracion del mejor `coef_`) y se restaura el mejor `coef_`. El numero de pasadas
sigue siendo 10, que es lo que el pre-registro congelo; lo que se corrige es **como** se ejecuta.

## Desvio declarado

`partial_fit` x 10 **no es identico bit a bit** a un unico `fit(max_iter=10)`: medido, difieren
4.5e-5 a 2 pasadas y 2.0e1 a 4 pasadas en un problema sintetico. La causa es el contabilizado
interno de `_fit` para la primera epoca. La implementacion elegida es la que **hace posible el ES por
pasada** que el protocolo requiere y la que da al control el mismo presupuesto de 10 pasadas que
A1; se declara en el informe y en el acta D-085 como desvio documentado, no como fallo.

## Estado

Enmienda **PRE-EJECUCION**: ningun fold de V34-E1 ha sido entrenado (el runner esta incompleto y
`--verificar` aun no pasa). No cambia ningun gate, ni la puerta G0-G6, ni la regla de alpha, ni la
criterio de exito; solo fija como se ejecuta el presupuesto del control.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREREG_JSON = ROOT / "experiments" / "v34_00_preregistro.json"
PREREG_MD = ROOT / "reports" / "v34_preregistro.md"
RUNLOG = ROOT / "RUNLOG.md"
EXPERIMENTS = ROOT / "experiments.csv"
FECHA = "2026-09-30"

ENMIENDAS = [
    {
        "fecha": FECHA,
        "objeto": "control.a2_sgd.ajuste (presupuesto de 10 pasadas)",
        "de": "fit(max_iter=P_MAX=10, tol=None, shuffle=False) en una sola llamada por fold",
        "a": "partial_fit x 10, una llamada por orden congelado de cada pasada, con t_ acumulado, "
             "ES evaluado despues de cada pasada y restauracion del mejor coef_",
        "motivo": "defecto medido de API: BaseSGDClassifier._fit hace self.t_ = 1.0 en cada "
                  "llamada, luego fit(max_iter=1, warm_start=True) repetido reinicia el schedule "
                  "10 veces y queda en t_ = 401 (medido) en vez de t_ = 4001; el control "
                  "recibiria un presupuesto de un orden de magnitud menor que A1",
        "medicion": "t_ = 401 con fit repetido; t_ = 4001 con partial_fit y con "
                    "fit(max_iter=10); diferencias partial_fit vs fit(max_iter=10): 4.5e-5 a 2 "
                    "pasadas y 2.0e1 a 4 pasadas en un problema sintetico",
        "desvio_declarado": "partial_fit x 10 no es identico bit a bit a fit(max_iter=10); es la "
                            "unica forma de evaluar el ES por pasada que el protocolo exige, y "
                            "mantiene el presupuesto de 10 pasada declarado",
        "afecta_a_gates": False,
        "estado_datos": "sin resultados; enmienda pre-ejecucion",
    },
]


def main() -> int:
    datos = json.loads(PREREG_JSON.read_text(encoding="utf-8"))
    control = datos["control"]
    antes = dict(control.get("ajuste", {})) or {"modo": "fit_unico", "max_iter": control.get("pasadas")}
    control["ajuste"] = {
        "modo": "partial_fit_por_pasada",
        "llamadas": control.get("pasadas"),
        "t_acumulado": "t_ se acumula entre llamadas (401 si se usara fit repetido, por el "
                       "reinicio de self.t_ en BaseSGDClassifier._fit)",
        "es_por_pasada": True,
        "restaura_mejor_coef": True,
        "max_iter": None,
        "nota": "el numero de pasadas (10) es el que el pre-registro congelo; lo que se corrige "
                "es como se ejecuta, porque fit(max_iter=1, warm_start=True) no acumula el "
                "schedule",
    }
    control["desvio_declarado_partial_fit"] = {
        "no_bit_identico_a_fit_max_iter_10": True,
        "medicion": "4.5e-5 a 2 pasadas y 2.0e1 a 4 pasadas en un problema sintetico",
        "motivo_de_la_eleccion": "es la unica via que permite evaluar el early stopping despues de "
                                 "cada pasada, que el pre-registro exige (P_min 3, paciencia 2, "
                                 "restaurar el mejor coef_), y da el mismo presupuesto que A1",
    }
    enmiendas = datos.setdefault("enmiendas", [])
    for e in ENMIENDAS:
        if not any(x.get("objeto") == e["objeto"] for x in enmiendas):
            enmiendas.append(e)
    PREREG_JSON.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  control.ajuste: {antes} -> partial_fit_por_pasada x {control['ajuste']['llamadas']}",
          flush=True)
    print(f"  JSON: enmiendas={len(enmiendas)}", flush=True)

    if "## Enmienda pre-ejecucion 2" not in PREREG_MD.read_text(encoding="utf-8"):
        bloque = (
            "\n## Enmienda pre-ejecucion 2 (2026-09-30)\n\n"
            "- **Control A2**: `fit(max_iter=P_MAX=10, ...)` en una sola llamada pasa a "
            "`partial_fit` x 10, una por\n  orden congelado de cada pasada. Motivo (medido en /tmp "
            "por el implementador): `BaseSGDClassifier._fit` hace\n  `self.t_ = 1.0` en cada "
            "llamada, luego `fit(max_iter=1, warm_start=True)` repetido reinicia el schedule 10 "
            "veces y queda en\n  `t_ = 401` en vez de `t_ = 4001`; el control recibiria un "
            "presupuesto de un orden de magnitud menor que A1. Con\n  `partial_fit` el `t_` se "
            "acumula y se alcanza `t_ = 4001`, identico al de `fit(max_iter=10)`.\n"
            "- **Desvio declarado**: `partial_fit` x 10 no es identico bit a bit a "
            "`fit(max_iter=10)` (4.5e-5 a 2 pasadas y 2.0e1 a 4 pasadas en un\n  problema "
            "sintetico, por el contabilizado interno de la primera epoca). Es la unica via que "
            "permite el ES por pasada que el\n  protocolo exige (`P_min = 3`, paciencia 2, "
            "restaurar el mejor `coef_`) y mantiene el presupuesto de 10 pasadas.\n"
            "- No cambia gates, puerta, regla de alpha ni criterio de exito. Sin resultados: "
            "enmienda **pre-ejecucion**.\n"
        )
        PREREG_MD.write_text(PREREG_MD.read_text(encoding="utf-8").rstrip("\n") + "\n" + bloque,
                             encoding="utf-8")
        print("  MD del pre-registro: seccion de enmienda 2 anadida", flush=True)

    if "Enmienda pre-ejecucion 2 de V34-E1" not in RUNLOG.read_text(encoding="utf-8"):
        with RUNLOG.open("a", encoding="utf-8") as f:
            f.write("\n---\n\n## Enmienda pre-ejecucion 2 de V34-E1 · " + FECHA + "\n\n"
                    "- El control A2 pasa de `fit(max_iter=10)` unico a `partial_fit` x 10 "
                    "(una por orden congelado): medido en /tmp,\n  `fit(max_iter=1, "
                    "warm_start=True)` repetido reinicia `self.t_ = 1.0` en cada llamada y queda "
                    "en `t_ = 401` en vez de\n  `t_ = 4001`, luego no era un presupuesto de 10 "
                    "pasadas. Desvio declarado: `partial_fit` x 10 no es bit-identico a "
                    "`fit(max_iter=10)`.\n- Sin resultados; no cambia gates, puerta ni regla de "
                    "alpha.\n")
        print("  RUNLOG.md: enmienda 2 anotada", flush=True)

    lineas = EXPERIMENTS.read_text(encoding="utf-8").splitlines()
    cols = [c.strip() for c in lineas[0].split(",")]
    if "V34-E1-PREREG-ENMIENDA-2" not in {lin.split(",", 1)[0] for lin in lineas[1:] if lin}:
        fila = {"experiment_id": "V34-E1-PREREG-ENMIENDA-2", "fecha": FECHA, "bloque": "34.0",
                "etapa": "Enmienda pre-ejecucion 2: presupuesto del control A2 con partial_fit",
                "pregunta": "fit(max_iter=1, warm_start=True) x 10 es un presupuesto de 10 "
                            "pasadas?",
                "hipotesis": "no: BaseSGDClassifier._fit reinicia self.t_ = 1.0 en cada llamada",
                "features": "n/a", "preprocessing": "n/a", "representacion": "n/a", "modelo": "n/a",
                "hiperparametros": "A2: partial_fit x 10 con t_ acumulado (4001) en vez de fit "
                                   "unico; desvio bit-a-bit declarado",
                "validacion": "medicion en /tmp del contador t_ y de la diferencia frente a "
                              "fit(max_iter=10)",
                "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "",
                "micro_f1_mean": "", "accuracy_mean": "",
                "resultado": "enmienda registrada; sin resultados del experimento",
                "interpretacion": "correccion de un defecto de API medido, no de conveniencia",
                "decision": "ejecutar V34-E1 con la enmienda",
                "duracion_seg": "0.0",
                "artefactos": "experiments/v34_00_preregistro.json (campo enmiendas) | "
                              "reports/v34_preregistro.md"}
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerow([fila[c] for c in cols])
        with EXPERIMENTS.open("a", encoding="utf-8") as f:
            f.write(buf.getvalue())
        print("  experiments.csv: fila V34-E1-PREREG-ENMIENDA-2", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
