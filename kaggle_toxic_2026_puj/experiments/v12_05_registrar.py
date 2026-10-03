"""Registra V12-05 en el registro de experimentos de forma trazable.

Se escribe como fichero y no como bloque en linea porque el contenido lleva comillas, parentesis y
acentos que un bloque en linea rompe, y un registro mal formado es peor que un registro ausente.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

RUTA = Path("experiments.csv")


def main() -> None:
    """Anade la fila de V12-05 y verifica la integridad del registro."""
    campos = list(csv.DictReader(RUTA.open(encoding="utf-8")).fieldnames)
    ids = {f["experiment_id"] for f in csv.DictReader(RUTA.open(encoding="utf-8"))}

    interpretacion = (
        "**La respuesta tiene dos mitades que no hay que mezclar.** En el eje del objetivo de "
        "entrenamiento, si quedaba y la v12 lo ha cerrado: la auditoria V12-00 encontro tres "
        "huecos genuinamente no probados, los tres en ese eje, y los tres se ejecutaron con gates "
        "identicos y fallaron. En el conjunto del proyecto, hay un techo medido de +0.046271 que "
        "sigue sin capturar. **La conclusion que la fase deja establecida**: el techo condicionado "
        "por la etiqueta verdadera de HS demuestra que existe senal estructural abundante, del "
        "orden de +0.046, que es 23 veces el gate exploratorio y 300 veces la mejor ganancia "
        "anidada historica de +0.000148. Y el cuello de botella NO es la ausencia de capacidad "
        "predictiva para TR y AG, porque su F1 donde la etiqueta esta definida es de 0,8941: la "
        "dificultad es **inferir la condicion**, saber para cada fila si HS es cero. Las tres vias "
        "para superarla fallan por razones distintas y documentadas: la llave dura da -0.004187 "
        "porque su llave se equivoca en 143 filas; la llave suave, que es V6-B, cubre solo las 20 "
        "filas donde HS se predice bien y no alcanza las 58 restantes; y acoplar la relacion en el "
        "objetivo empeora los falsos positivos justo en la zona que hay que corregir. "
        "**Rendimientos decrecientes**: H1 +0.002738 sostenido por un solo pliegue con el maximo "
        "de su curva sobre el neutro, H2a +0.001616 con el intervalo incluyendo el cero, y H3 "
        "-0.002469 con deteccion 0.4915, es decir peores que el azar. No es que la primera idea "
        "fuera mala y las siguientes peores: es que el eje entero tiene poco margen recuperable "
        "frente a lo que queda disponible en otro sitio"
    )

    decision = (
        "**FASE CERRADA SIN CANDIDATA.** 5 experimentos, 4 hipotesis, 0 candidatas, 0 submissions, "
        "0 evaluaciones de holdout2 gastadas, segunda particion no ejecutada. V03 permanece "
        "congelada con 0.7772114525165721 y LB 0.78306. Se detiene la exploracion por decision "
        "expresa del usuario tras H3 y la auditoria de V6-B. Queda abierto un solo problema, que "
        "no es de esta fase: conseguir que la condicion de HS se estime mejor, porque de eso "
        "depende todo lo demas"
    )

    notas = (
        "Artefactos: reports/v12_final_research.md, reports/v12_05_informe_final.json, "
        "reports/v12_master_results.csv con 5 filas y 28 columnas, y "
        "reports/v12_hypothesis_registry.csv con 4 hipotesis. **LA CORRECCION CONCEPTUAL QUE EL "
        "INFORME CONSERVA, y que es lo mas importante de la fase**: H2 y H3 NO demuestran que la "
        "funcion de perdida no sea el problema en sentido absoluto. Un unico mecanismo positivo, "
        "el GCE de H2, no autoriza a descartar la familia entera. Lo que si autorizan a decir es "
        "mas preciso: las variaciones de objetivo evaluadas hasta ahora tienen un impacto pequeno "
        "comparado con el techo que introduce el error de la condicion de HS, que es del 3.5 por "
        "ciento en el mejor caso. La distincion importa porque la primera afirmacion cerraria una "
        "familia de forma definitiva y la segunda solo establece que el margen recuperable por esa "
        "via es pequeno frente a otro problema ya medido. Sin commit hasta peticion expresa"
    )

    nueva = {
        "experiment_id": "V12-05",
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "date": "2026-09-26",
        "script": "experiments/v12_05_informe_final.py",
        "question": "Cual es la conclusion de la fase v12, y que deja establecido y que deja abierto?",
        "hypothesis": (
            "La v12 se abre con una auditoria que encuentra tres huecos genuinamente no probados, "
            "todos en el objetivo de entrenamiento, y los ejecuta. La consolidacion debe responder "
            "la pregunta cientifica final separando lo que la fase cierra de lo que deja abierto"
        ),
        "preprocessing": "no aplica: consolidacion. NO se entrena nada",
        "feature_groups": "n/a",
        "representation": "n/a",
        "model": "n/a",
        "hyperparameters": "n/a",
        "validation": (
            "las cifras de la tabla maestra y del registro de hipotesis se LEEN de los informes "
            "JSON de cada experimento, no se teclean"
        ),
        "f1": "0.7772114525165721",
        "macro_f1": "",
        "weighted_f1": "",
        "precision": "",
        "recall": "",
        "training_time": "0.0",
        "dimension": "24646",
        "result": json.dumps(
            {
                "experimentos": 5,
                "hipotesis": ["H1", "H2a", "H2b", "H3"],
                "candidatas": 0,
                "submissions_generadas": 0,
                "holdout2_gastadas": 0,
                "techo_oraculo": 0.046271,
                "postproceso_duro": -0.004187,
                "mejor_delta_por_hipotesis": {
                    "H1": 0.002738,
                    "H2a": 0.001616,
                    "H2b": -0.116581,
                    "H3": -0.002469,
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "interpretation": interpretacion,
        "decision": decision,
        "notes": notas,
    }

    if nueva["experiment_id"] in ids:
        print("ya existe")
    else:
        with RUTA.open("a", encoding="utf-8", newline="") as manejador:
            csv.DictWriter(
                manejador, fieldnames=campos, lineterminator="\n", extrasaction="ignore"
            ).writerow({**{c: "" for c in campos}, **nueva})
        print("V12-05 registrado")

    final = list(csv.DictReader(RUTA.open(encoding="utf-8")))
    identificadores = [f["experiment_id"] for f in final]
    malformadas = [
        f["experiment_id"]
        for f in final
        if len(f) != len(campos) or any(v is None for v in f.values())
    ]
    no_ascii = [
        f["experiment_id"]
        for f in final
        if any(127 < ord(c) for v in f.values() for c in (v or ""))
    ]
    print(
        f"experiments.csv: {len(final)} filas | {len(set(identificadores))} claves | "
        f"malformadas {len(malformadas)} | con no-ASCII {len(no_ascii)}"
    )
    print("v12:", [x for x in identificadores if x.startswith("V12-")])


if __name__ == "__main__":
    main()
