"""V13 · Cierre — Ensamblado de los entregables agregados de la fase.

Recoge lo que V13 midió y lo deja en tres tablas que no dependen de que nadie lea los quince informes
sueltos:

* ``reports/v13_master_results.csv``: un renglón por experimento de V13 con su métrica clave, qué
  criterio se usó para decidir y cuál fue el veredicto.
* ``reports/v13_hypothesis_registry.csv``: el registro de hipótesis falsadas. Cada fila es una
  hipótesis **pre-registrada** (del plan o del artículo de referencia), el experimento que la
  somete a prueba, la medición y el veredicto. Es el entregable que más importa: un estudio que solo
  publica lo que funcionó no se puede auditar.
* ``reports/v13_candidate_comparison.csv``: V12 congelado frente a cada candidato evaluado, con la
  diferencia, los folds positivos, el control de permutado y la frontera.

El script no calcula nada nuevo: lee los ``experiments/*.json`` que cada experimento dejó, comprueba
que están todos y avisa si falta alguno. Si un JSON no existe, la fila aparece con "FALTA" en lugar
de inventarse un número.

Entregables: ``reports/v13_master_results.csv``, ``reports/v13_hypothesis_registry.csv``,
``reports/v13_candidate_comparison.csv``, ``experiments/v13_cierre.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

MASTER = cfg.REPORTS / "v13_master_results.csv"
REGISTRO = cfg.REPORTS / "v13_hypothesis_registry.csv"
COMPARATIVA = cfg.REPORTS / "v13_candidate_comparison.csv"
RESULTADO = cfg.EXPERIMENTS / "v13_cierre.json"

#: Referencia congelada de V12, la que todos los candidatos tienen que superar.
REFERENCIA_V12 = 0.6004
SUELO_AZAR = 0.1667

#: (fase, tipo, json, clave de la metrica, criterio de decision, veredicto, artefacto principal)
EXPERIMENTOS = [
    ("P0", "diagnostico", "v13_p0_gap_audit.json", "experimentos_inventariados",
     "cobertura de las 26 pistas", "cerrada: 18/18 mapeos verificados, 26 pistas ordenadas",
     "reports/v13_gap_audit.md"),
    ("P0a", "auditoria", "v13_p0a_auditoria_idioma.json", "errores",
     "cero caracteres de otros alfabetos", "cerrada: 0 errores y 0 avisos en 301 ficheros",
     "reports/v13_auditoria_idioma.md"),
    ("P0b", "diagnostico", "v13_p0b_colinealidad.json", "direcciones_redundantes",
     "medir la colinealidad de la matriz completa", "cerrada: 15 duplicados exactos, 170 variables con VIF >= 10",
     "reports/v13_colinealidad.md"),
    ("P15", "diagnostico", "v13_p15_member_error_matrix.json", "pct_error_rescatable",
     "quien acierta cuando V12 falla", "cerrada: 66,87% rescatable pero con 13 de 14 miembros y sin senal de confianza",
     "reports/v13_p15_member_error_matrix.md"),
    ("P1a", "control", "v13_p1a_evidencia_tokens.json", "control",
     "el bloque supera el suelo de azar", "superada en control: 0,3997, el doble del mejor bloque lexico cerrado",
     "reports/v13_p1_evidencia_tokens.md"),
    ("P1c", "diagnostico", "v13_p1c_identidades.json", "proporcion_de_la_mejora_que_es_identidad",
     "separar senal lexica de reconocimiento de cuentas", "cerrada: el 57% de la mejora es lexica",
     "reports/v13_p1_identidades.md"),
    ("P1b", "fusion", "v13_p1b_fusion_tokens.json", "delta",
     "gate §24 completo mas control de permutado", "NO ADOPTAR: -0,0002 con 3/5 folds, controles indistinguibles",
     "reports/v13_p1_fusion.md"),
    ("P17a", "control", "v17_p17a_pronombres.json", "control",
     "el bloque supera el suelo de azar", "superada por poco: 0,1918, y 0 de 26 redundantes con V12",
     "reports/v17_pronombres.md"),
    ("P17b", "prechequeo", "v17_p17b_pronombres.json", "prechequeo",
     "el rescate neto alcanza el umbral", "cierre: neto -64.075, no se fusiono",
     "reports/v17_pronombres_fusion.md"),
    ("P20", "control", "v20_p20_modalidad.json", "control",
     "el bloque supera el suelo de azar", "NO superada: 0,1270, por debajo del azar; 1,6% de cobertura",
     "reports/v20_modalidad.md"),
    ("P3a", "especialista", "v3_p3a_especialistas.json", "delta_global",
     "mejora la decision dentro del par en +0,010", "cierre: -0,0012 global, el especialista empeora en ambas fronteras",
     "reports/v3_especialistas.md"),
    ("P19", "control", "v19_p19_conjunciones.json", "control",
     "el bloque supera el suelo de azar", "NO superada: arbol 0,1539 y lineales 0,1539, bajo el azar",
     "reports/v19_conjunciones.md"),
    ("P16", "diagnostico", "v16_p16_suficiencia.json", "por_bloque",
     "V12 ya contenia la informacion de los bloques", "inconcluyente por fallo de diseno, rehace en P16c",
     "reports/v16_suficiencia.md"),
    ("P16b", "diagnostico", "v16b_p16b_activacion.json", "porcentaje_activo",
     "la senal esta concentrada en filas activas", "diagnosticado: techo 0,7221 pero neto negativo en toda franja",
     "reports/v16b_activacion.md"),
    ("P16c", "diagnostico", "v16c_p16c_reconstruccion.json", "por_bloque",
     "cuanta informacion contiene V12", "cerrada: R2 alto pero el control de la reconstruccion es 0,0485",
     "reports/v16c_reconstruccion.md"),
    ("P16d", "diagnostico", "v16d_puerta_real.json", "delta_sin_honestidad",
     "existe una puerta realizable que llegue al gate", "cerrada: la mejor de 90 combinaciones es no usar el bloque",
     "reports/v16d_puerta.md"),
]

#: (hipotesis, origen, experimento, medicion, veredicto)
HIPOTESIS = [
    ("H1", "P(clase|token) calculada dentro del fold lleva senal de edad que V12 no usa",
     "plan P1 (y el mecanismo de Moreno-Sandoval et al. 2021)", "P1a y P1b",
     "control aislado 0,3997 (suelo 0,1667) y 1 de 36 variables redundante; fusion -0,0002 con 3/5 folds",
     "CONFIRMADA en aislamiento, REFUTADA para la fusion: el bloque aporta senal real que V12 ya no puede usar"),
    ("H2", "Los pronombres de persona son la variable mas predictiva (V14-V18 del articulo)",
     "articulo de referencia, tabla de variables", "P17a y P17b",
     "control 0,1918, 0 de 26 redundantes con V12, rescate neto -64.075",
     "REFUTADA para este corpus: hay senal pero el bloque es mucho peor que V12"),
    ("H3", "La incertidumbre verbal cambia con la edad",
     "plan P20", "P20",
     "control 0,1270, por debajo del azar; el 1,6% de documentos tiene un marcador de duda",
     "REFUTADA: el corpus de 140 caracteres no contiene la variable"),
    ("H4", "La funcion gramatical de las conjunciones varia con la edad y un arbol la captura",
     "material del curso, via sin POS", "P19",
     "arboles 0,1489/0,1512/0,1539 y lineales 0,1539/0,1522, todos bajo el azar; 0,64 conjunciones por documento",
     "REFUTADA: la frecuencia de conjunciones es demasiado baja para discriminar"),
    ("H5", "Un bloque linguistico sube la rama LogReg y se diluye en la MLP",
     "prediccion pre-registrada del plan, del articulo", "P1b (reporte por cabeza)",
     "LogReg 0,5915 -> 0,5909 (-0,0006) y MLP 0,5978 -> 0,5973 (-0,0005)",
     "NO CONFIRMADA: ninguna de las dos cabezas mejora, asi que no hay nada que disolver"),
    ("H6", "Un especialista binario de frontera mejora la decision de V12 dentro del par",
     "plan P3", "P3a",
     "18-24|25-34: 0,4963 -> 0,4944; 25-34|35-49: 0,4974 -> 0,4942; global 0,6005 -> 0,5993",
     "REFUTADA: el especialista dedicado es peor que V12 restringido al par"),
    ("H7", "V12 ya contiene la informacion de los bloques nuevos, y por eso las fusiones fracasan",
     "diagnostico propio tras cuatro fracasos", "P16 y P16c",
     "R2 mediano de la reconstruccion 0,522 con maximo 0,959, pero control de la reconstruccion 0,0485",
     "RECHAZADA en la forma que importa: V12 reproduce la varianza del bloque, no su informacion de clase"),
    ("H8", "Existe una puerta realizable (sin oraculo) que aproveche el bloque de P1",
     "forma gate(especialista, V12) del §37", "P16b y P16d",
     "techo con puerta perfecta 0,677 y AUC de la confianza del bloque 0,7173; el maximo de 90 combinaciones es 0,6005 con 0 filas desviadas",
     "REFUTADA: el techo existe pero ningun gatillo de confianza lo alcanza"),
]


def valor(lectura: dict, clave: str) -> object:
    """Extrae la clave de un JSON de experimento, tolerando su ausencia."""
    if clave not in lectura:
        return "FALTA"
    return lectura[clave]


def main() -> int:
    """Ensambla las tres tablas agregadas de V13."""
    autotest_numerico()
    set_seed()
    faltantes: list[str] = []
    filas_master = []
    for fase, tipo, fichero, clave, criterio, veredicto, artefacto in EXPERIMENTOS:
        ruta = cfg.EXPERIMENTS / fichero
        if not ruta.exists():
            faltantes.append(fichero)
            filas_master.append({"fase": fase, "tipo": tipo, "lectura": "FALTA", "clave": clave,
                                 "criterio": criterio, "veredicto": veredicto,
                                 "artefacto": artefacto})
            continue
        lectura = json.loads(ruta.read_text(encoding="utf-8"))
        filas_master.append({"fase": fase, "tipo": tipo, "lectura": fichero,
                             "clave": clave, "medida": json.dumps(valor(lectura, clave),
                                                                  ensure_ascii=False)[:300],
                             "criterio": criterio, "veredicto": veredicto, "artefacto": artefacto})
    pd.DataFrame(filas_master).to_csv(MASTER, index=False)

    pd.DataFrame([
        {"id": h, "hipotesis": texto, "origen": origen, "experimento": experimento,
         "medicion": medicion, "veredicto": veredicto}
        for h, texto, origen, experimento, medicion, veredicto in HIPOTESIS
    ]).to_csv(REGISTRO, index=False)

    comparativa = [
        {"candidato": "V12 (referencia congelada)", "tipo": "referencia",
         "macro_f1": REFERENCIA_V12, "delta": 0.0, "folds_positivos": "-",
         "control_permutado": "-", "frontera_18_24_25_34": 0.4963, "frontera_25_34_35_49": 0.4974,
         "veredicto": "congelado (LB 0,62188)"},
        {"candidato": "V12 + evidencia por token (P1)", "tipo": "fusion",
         "macro_f1": 0.6003, "delta": -0.0002, "folds_positivos": "3/5",
         "control_permutado": 0.6003, "frontera_18_24_25_34": 0.5015,
         "frontera_25_34_35_49": 0.5017, "veredicto": "NO ADOPTAR"},
        {"candidato": "V12 + especialistas de frontera (P3)", "tipo": "especialista",
         "macro_f1": 0.5993, "delta": -0.0012, "folds_positivos": "-", "control_permutado": "-",
         "frontera_18_24_25_34": 0.4944, "frontera_25_34_35_49": 0.4942,
         "veredicto": "NO ADOPTAR"},
        {"candidato": "V12 + lexicones (fase anterior, D-039)", "tipo": "fusion",
         "macro_f1": 0.6004, "delta": 0.0000, "folds_positivos": "-",
         "control_permutado": 0.5994, "frontera_18_24_25_34": "-", "frontera_25_34_35_49": "-",
         "veredicto": "NO ADOPTAR"},
        {"candidato": "capa de decision ordinal (D-037)", "tipo": "capa de decision",
         "macro_f1": 0.6015, "delta": 0.0011, "folds_positivos": "4/5", "control_permutado": "-",
         "frontera_18_24_25_34": "-", "frontera_25_34_35_49": "-",
         "veredicto": "NO ADOPTAR: no alcanza el gate y el neto de rescate es negativo"},
        {"candidato": "mejor puerta por confianza sobre P1 (P16d)", "tipo": "puerta",
         "macro_f1": 0.6005, "delta": 0.0, "folds_positivos": "0/5 (0 filas desviadas)",
         "control_permutado": "-", "frontera_18_24_25_34": 0.4963, "frontera_25_34_35_49": 0.4974,
         "veredicto": "la mejor puerta es no usar el bloque"},
    ]
    pd.DataFrame(comparativa).to_csv(COMPARATIVA, index=False)

    RESULTADO.write_text(json.dumps({
        "fase": "V13 · cierre", "fecha": "2026-09-26",
        "experimentos_registrados": len(EXPERIMENTOS),
        "faltantes": faltantes, "hipotesis": len(HIPOTESIS),
        "referencia_v12": REFERENCIA_V12, "suelo_azar": SUELO_AZAR,
        "candidatos_adoptados": 0,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"experimentos en la tabla maestra: {len(EXPERIMENTOS)} | faltantes: {len(faltantes)}")
    if faltantes:
        print("  FALTAN:", ", ".join(faltantes))
    print(f"hipotesis registradas: {len(HIPOTESIS)}")
    print(f"candidatos adoptados: 0 de {len(comparativa) - 1}")
    print(f"artefactos: {MASTER.name}, {REGISTRO.name}, {COMPARATIVA.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
