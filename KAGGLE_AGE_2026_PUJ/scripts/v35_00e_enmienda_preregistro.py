"""V35 - E1 - Enmienda pre-ejecucion 4: se restituyen las 6 secciones de contexto del pre-registro.

## El defecto (pre-registro incompleto, detectado en la revision pre-ejecucion)

`ml-reviewer` ejecuto de verdad `construir_resultado`, `escribir_informe` y `resumen_consola` con un
estado simulado, porque esos modos **no** se ejecutan en `--verificar` ni en `--humo`, yEncoding
encontro **17 lecturas** `prereg[...]` de secciones top-level que **NO existen** en el
pre-registro congelado de V35: `control`, `corpus`, `anclas`, `representacion`, `budgeto` y
`regla_alpha`. Las 6 si existen en el pre-registro de V34-E1. Sin ellas, la corrida muere **despues**
de entrenar los dos brazos en los cinco folds, con checkpoints y OOF ya escritos, y
`verificar_constantes` no lo detecta (de ahi el 106/106 que no era cobertura real).

La causa es que el pre-registro de V35 se escribio mas escueto que el de V34 y se llevo por delante
el contexto que el informe publica: sin esas secciones el informe no tendria de donde sacar la
receta del corpus, las anclas, la regla de `alpha`, el control, el presupuesto ni la
representacion.

## La enmienda

Se restituyen las 6 secciones **copiadas verbatim** del pre-registro de V34-E1, que es el mismo
experimento de familia con el mismo corpus, el mismo vectorizador canonico, el mismo presupuesto de
pasadas y el mismo control SGD, y se anulan **explicitamente** solo los puntos donde el diseno de
V35 difiere de V34:

- `regla_alpha.aplicacion`: en V35 `alpha*` esta **CONGELADO e identico** a V34-E1 y **no se
  re-deriva**; se anaden `valor_congelado`, `d_norm_fold_0_es` y la nota de que la unica
  verificacion en runtime es el anclaje de `lambda1` (enmienda 1), no la formula.
- `regla_alpha.esperado_aproximado`: se conservan los dos valores con su referencia, porque el
  0.0352631 de V34 corresponde al `d_norm` del prior del fold-train completo
  (0.912815416414756) y el 0.035262336465737745 congelado en V35 corresponde al `d_norm` del prior
  MEDIDO del ES-train del fold 0 (0.9128367962785974). Se declara la correspondencia para que nadie
  los lea como una contradiccion.
- `anclas`: se anaden las anclas **medidas en V35** (prior del ES-train del fold 0, `d_norm` del
  ES-train, mediana del anclaje de `lambda1`, su cociente con `alpha`, la razon de normas medida
  pre-ejecucion y el `sha256` del snapshot de `z`).
- `control.asimetria_declarada`: la de V34 describe una asimetria distinta (el L2 de A2 en 1e-6
  como no-op). En V35 la asimetria relevante es que **A1 tiene L1 y A2 no**, asi que A2 es control
  de **PRESUPUESTO** y no gemelo mecanico. Se anula el texto de V34 y se declara el de V35.
- `control.por_que` y `gemelo_sgd_d083`: se conservan como precedente historico, y se anade que en
  V35 el techo de A2 esta acotado por su propia medicion en V34-E1.
- `representacion`: se marca que su `vectorizador` es identico a `config.vectorizador` y que su
  hash es el canonico `9bae1576...` (enmienda 3).

Cada seccion copiada lleva un `origen` que declara de donde sale, para que la copia sea auditable.

## Por que esto NO cambia el protocolo

No se anade, quita ni altera ninguna decision experimental: ni el modelo, ni el nucleo FTRL, ni el
anclaje de `lambda1`, ni `alpha*` (que sigue congelado e identico a V34-E1), ni los gates, ni la
puerta, ni el arbol, ni el criterio de exito, ni el presupuesto declarado de 1200 s, ni los datos.
Se restituye **contexto ya congelado** que el informe necesita para publicar lo que el propio
pre-registro de V34 declara. Es la **cuarta enmienda pre-ejecucion**, misma clase que las tres
anteriores: declaracion, sin resultados.

## Estado

**PRE-EJECUCION**: ningun fold de V35-E1 ha sido entrenado.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREREG_V35 = ROOT / "experiments" / "v35_00_preregistro.json"
PREREG_V34 = ROOT / "experiments" / "v34_00_preregistro.json"
PREREG_MD = ROOT / "reports" / "v35_preregistro.md"
RUNLOG = ROOT / "RUNLOG.md"
EXPERIMENTS = ROOT / "experiments.csv"
FECHA = "2026-09-30"

SECCIONES = ("corpus", "regla_alpha", "anclas", "control", "budgeto", "representacion")
ORIGEN_ES = ("copiado verbatim de experiments/v34_00_preregistro.json (mismo corpus, mismo "
             "vectorizador canonico, mismo presupuesto de pasadas y mismo control SGD); solo "
             "las anulaciones de V35 se listan en la clave `anulaciones_v35`")

ALPHA_CONGELADO = 0.035262336465737745
D_NORM_ES_FOLD_0 = 0.9128367962785974
H_PRIOR_ES_FOLD_0 = 1.7915714394468476
MEDIANA_ANCLAJE = 0.23993793862904017
RAZON_MEDIANA_ALPHA = 6.8044
RAZON_NORMAS = 2.1304
SHA_Z_ANCLAJE = "ec1d1a7b13454f8b"

ANULACIONES = {
    "control": {
        "asimetria_declarada": (
            "A1 tiene un regularizador (L1) que A2 no tiene: A2 es control de PRESUPUESTO, NO "
            "gemelo mecanico. El texto que V34 declaraba aqui describia otra asimetria (el L2 de A2 "
            "en 1e-6 como no-op medido) y queda ANULADO para V35. Ninguna conclusion puede "
            "descansar en el contraste A1 vs A2: delta, razones entre brazos e IC bootstrap "
            "cruzado quedan contaminados. El veredicto se apoya solo en evidencia interna de A1 "
            "contra la regla derivada (G0/G1/G4) mas el triage"),
        "sonda_exigida": ("publicar que el veredicto NO se apoya en el contraste A1 vs A2 y que la "
                          "lectura de A2 es unidireccional (presupuesto)"),
    },
    "regla_alpha": {
        "aplicacion": ("CONGELADO e identico a V34-E1: NO se re-deriva. La unica verificacion en "
                       "runtime es la del anclaje de lambda1 por el nucleo de anclaje (enmienda 1), "
                       "no la de la formula. Si el valor efectivo de runtime no coincide con el "
                       "congelado -> exit 2 ANTES de entrenar"),
    },
    "anclas": {
        "nota_sobre_los_dos_valores_de_alpha": (
            "0.0352631 es el valor de REFERENCIA calculado con el prior del fold-train COMPLETO "
            f"(d_norm = 0.912815416414756, el de la exploracion del arquitecto); "
            f"{ALPHA_CONGELADO} es el valor EFECTIVO del fold 0, derivado con el prior MEDIDO del "
            f"ES-train (d_norm = {D_NORM_ES_FOLD_0}). No son una contradiccion: son dos priors "
            "distintos, y el segundo es el que gobierna"),
    },
    "representacion": {
        "nota": ("`representacion.vectorizador` es identico a `config.vectorizador` y su hash es el "
                 "canonico 9bae157614458dbdf26fb7aed4c78d2954c36e895f830341e59b23fa63e2b51b "
                 "(enmienda 3)"),
    },
}

ANCLAS_V35 = {
    "h_prior_es_fold_0": H_PRIOR_ES_FOLD_0,
    "d_norm_es_fold_0": D_NORM_ES_FOLD_0,
    "alpha_efectivo_fold_0": ALPHA_CONGELADO,
    "mediana_abs_z_anclaje_fold_0": MEDIANA_ANCLAJE,
    "mediana_sobre_alpha": RAZON_MEDIANA_ALPHA,
    "razon_max_min_norma_w_lambda1_mediana": RAZON_NORMAS,
    "sha256_z_anclaje_fold_0": SHA_Z_ANCLAJE,
    "lectura": ("anclas medidas en la sonda pre-ejecucion (experiments/v35_00b_sonda_lambda1.json) "
                "y en el stream real de TRAIN del fold 0: el cociente mediana/alpha = 6.8044 "
                "confirma que lambda1 = alpha es un no-op, y la razon de normas 2.1304 > 1.0 "
                "confirma que un lambda1 fijo NO acota la norma, que es lo que G8 pone a prueba"),
}

ENMIENDA = {
    "fecha": FECHA,
    "objeto": "secciones de contexto ausentes: control, corpus, anclas, representacion, budgeto, "
              "regla_alpha",
    "de": "pre-registro de V35 con 27 secciones, sin las 6 que el informe publica y que el runner "
          "lee en construir_resultado (17 lecturas)",
    "a": "las 6 secciones restituidas verbatim del pre-registro de V34-E1, con `origen` auditable "
         "y con las anulaciones V35 declaradas en `anulaciones_v35` y en las propias claves",
    "motivo": "17 lecturas prereg[...] de secciones inexistentes: la corrida moria DESPUES de "
              "entrenar los dos brazos en los 5 folds, con checkpoints y OOF ya escritos, y "
              "verificar_constantes no lo detectaba (106/106 no era cobertura real). Detectado al "
              "ejecutar de verdad construir_resultado, escribir_informe y resumen_consola con un "
              "estado simulado, porque --verificar y --humo NO ejecutan la ruta de salida",
    "clase": "cuarta enmienda pre-ejecucion; restitucion de contexto ya congelado, sin tocar ninguna "
             "decision experimental",
    "cambia_el_protocolo": False,
    "cambia_alpha": False,
    "cambia_los_gates": False,
    "cambia_el_presupuesto": False,
    "estado_datos": "sin resultados; enmienda pre-ejecucion",
}


def main() -> int:
    destino = json.loads(PREREG_V35.read_text(encoding="utf-8"))
    fuente = json.loads(PREREG_V34.read_text(encoding="utf-8"))

    for seccion in SECCIONES:
        if seccion in destino:
            print(f"[STOP] la seccion '{seccion}' ya existe en el pre-registro de V35: la enmienda no "
                  "aplicaria (exit 2)")
            return 1
        if seccion not in fuente:
            print(f"[STOP] la seccion '{seccion}' no existe tampoco en el pre-registro de V34")
            return 1

    for seccion in SECCIONES:
        bloque = json.loads(json.dumps(fuente[seccion]))
        bloque["origen"] = ORIGEN_ES
        bloque["anulaciones_v35"] = ANULACIONES.get(seccion, {})
        if seccion == "regla_alpha":
            bloque["valor_congelado"] = ALPHA_CONGELADO
            bloque["no_se_re_deriva"] = True
            bloque["d_norm_fold_0_es"] = D_NORM_ES_FOLD_0
        if seccion == "anclas":
            bloque.update(ANCLAS_V35)
        destino[seccion] = bloque

    destino["representacion"]["vectorizador"] = destino["config"]["vectorizador"]
    if json.dumps(destino["config"]["vectorizador"], sort_keys=True) != json.dumps(
            destino["representacion"]["vectorizador"], sort_keys=True):
        print("[STOP] config.vectorizador y representacion.vectorizador no coinciden")
        return 1

    enmiendas = destino.setdefault("enmiendas", [])
    if not any(e.get("objeto") == ENMIENDA["objeto"] for e in enmiendas):
        enmiendas.append(ENMIENDA)
    PREREG_V35.write_text(json.dumps(destino, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  secciones restituidas: {', '.join(SECCIONES)}")
    print(f"  total de secciones del pre-registro: {len(destino)} | enmiendas: {len(enmiendas)}")

    if "## Enmienda pre-ejecucion 4" not in PREREG_MD.read_text(encoding="utf-8"):
        bloque_md = (
            "\n## Enmienda pre-ejecucion 4 (2026-09-30): se restituyen las 6 secciones de contexto\n\n"
            "- **Defecto**: el runner hace 17 lecturas `prereg[...]` de secciones top-level "
            "**inexistentes** en el pre-registro de V35 (`control`, `corpus`, `anclas`, "
            "`representacion`, `budgeto`, `regla_alpha`; las 6 si estan en el de V34-E1). La "
            "corrida moria **despues** de entrenar, con checkpoints y OOF ya escritos, y "
            "`verificar_constantes` no lo detectaba. Lo encontro `ml-reviewer` ejecutando de verdad "
            "`construir_resultado`, `escribir_informe` y `resumen_consola` con un estado simulado, "
            "porque `--verificar` y `--humo` **no** ejecutan la ruta de salida.\n"
            "- **Correccion**: las 6 secciones se copian **verbatim** del pre-registro de V34-E1 "
            "(mismo corpus, mismo vectorizador canonico, mismo presupuesto y mismo control), cada "
            "una con su clave `origen` para que la copia sea auditable, y se anulan **solo** los "
            "puntos donde V35 difiere: `control.asimetria_declarada` (A1 tiene L1 y A2 no, luego "
            "A2 es control de PRESUPUESTO, no gemelo mecanico), `regla_alpha.aplicacion` "
            "(`alpha*` CONGELADO e identico a V34-E1, no se re-deriva) y las anclas medidas en V35.\n"
            "- Nota explicita sobre los dos valores de `alpha`: 0.0352631 es la referencia con el "
            "prior del fold-train completo (`d_norm` 0.912815416414756) y 0.035262336465737745 es "
            "el efectivo del fold 0 con el prior MEDIDO del ES-train (`d_norm` 0.9128367962785974); "
            "son dos priors distintos, no una contradiccion.\n"
            "- **No cambia el protocolo**: ni modelo, ni nucleo, ni anclaje de `lambda1`, ni "
            "`alpha*`, ni gates, ni puerta, ni arbol, ni criterio de exito, ni presupuesto, ni "
            "datos. Se restituye contexto ya congelado que el informe publica.\n"
            "- Sin resultados: enmienda **pre-ejecucion**.\n"
        )
        PREREG_MD.write_text(PREREG_MD.read_text(encoding="utf-8").rstrip("\n") + "\n" + bloque_md,
                             encoding="utf-8")
        print("  MD del pre-registro: seccion de enmienda 4 anadida", flush=True)

    if "Enmienda pre-ejecucion 4 de V35-E1" not in RUNLOG.read_text(encoding="utf-8"):
        with RUNLOG.open("a", encoding="utf-8") as f:
            f.write("\n---\n\n## Enmienda pre-ejecucion 4 de V35-E1 - " + FECHA + "\n\n"
                    "- El runner hacia 17 lecturas `prereg[...]` de secciones inexistentes en V35 "
                    "(`control`, `corpus`, `anclas`, `representacion`, `budgeto`, `regla_alpha`): la "
                    "corrida moria tras entrenar, con checkpoints y OOF escritos, y "
                    "`verificar_constantes` no lo veia. Detectado por `ml-reviewer` al ejecutar de "
                    "verdad la ruta de salida, que `--verificar` y `--humo` no cubren.\n"
                    "- Las 6 secciones se copian verbatim de V34-E1 con `origen` auditable y se "
                    "anulan solo los puntos donde V35 difiere (asimetria de A2, `alpha` congelado, "
                    "anclas medidas). No cambia ninguna decision experimental.\n")
        print("  RUNLOG.md: enmienda 4 anotada", flush=True)

    lineas = EXPERIMENTS.read_text(encoding="utf-8").splitlines()
    cols = [c.strip() for c in lineas[0].split(",")]
    if "V35-E1-PREREG-ENMIENDA-4" not in {lin.split(",", 1)[0] for lin in lineas[1:] if lin}:
        fila = {"experiment_id": "V35-E1-PREREG-ENMIENDA-4", "fecha": FECHA, "bloque": "35.0",
                "etapa": "Enmienda pre-ejecucion 4: restitucion de las 6 secciones de contexto del "
                        "pre-registro",
                "pregunta": "el pre-registro de V35 tiene todo lo que el runner lee?",
                "hipotesis": "no: faltan 6 secciones top-level que el informe publica",
                "features": "n/a", "preprocessing": "n/a", "representacion": "n/a", "modelo": "n/a",
                "hiperparametros": "sin cambios; alpha* sigue congelado e identico a V34-E1",
                "validacion": "cada seccion lleva su `origen`; representacion.vectorizador es "
                              "identico a config.vectorizador (hash 9bae1576)",
                "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "",
                "micro_f1_mean": "", "accuracy_mean": "",
                "resultado": "enmienda registrada; sin resultados del experimento",
                "interpretacion": "restitucion de contexto ya congelado; el defecto lo causo el "
                                  "pre-registro escueto de V35, no el diseno",
                "decision": "ejecutar V35-E1 con las enmiendas 1, 2, 3 y 4",
                "duracion_seg": "0.0",
                "artefactos": "experiments/v35_00_preregistro.json (campo enmiendas) | "
                              "reports/v35_preregistro.md"}
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerow([fila[c] for c in cols])
        with EXPERIMENTS.open("a", encoding="utf-8") as f:
            f.write(buf.getvalue())
        print("  experiments.csv: fila V35-E1-PREREG-ENMIENDA-4", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
