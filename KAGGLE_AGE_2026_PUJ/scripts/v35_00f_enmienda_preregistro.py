"""V35 - E1 - Enmienda pre-ejecucion 5: una sola declaracion de asimetria en el pre-registro.

## El defecto (dos declaraciones para la misma cosa)

La enmienda 4 restituyo `control` verbatim del pre-registro de V34-E1 y declaro en
`anulaciones_v35` que su `asimetria_declarada.no_equivalencia_estructural` quedaba anulada para
V35, **pero no la sustituyo**: el texto de V34 (el L2 de A2 en 1e-6 como no-op) seguia en
`control.asimetria_declarada.no_equivalencia_estructural` mientras el texto valido de V35 vivia
en `anulaciones_v35`. Es decir, el pre-registro declaraba DOS no-equivalencias estructurales
distintas para el mismo par de brazos, y solo una era la de V35. `ml-reviewer` ya habia
senalado justamente esta clase de defecto (productor y consumidor discordantes) y el runner ya
no imprime ese texto (la linea del informe lee la asimetria de la SONDA, que si es la de V35), de
modo que el riesgo era de lectura del documento de registro, no de la corrida. Un documento de
registro con dos declaraciones para lo mismo es exactamente el tipo de ambiguedad que este
proyecto ha pagado antes.

## La enmienda

1. `control.asimetria_declarada.no_equivalencia_estructural` pasa a ser **el texto de V35**: A1
   tiene un regularizador L1 que A2 no tiene, luego A2 es control de PRESUPUESTO y no gemelo
   mecanico, y el contraste entre brazos queda contaminado.
2. El texto de V34 que alli vivia se conserva integro, y con su valor, en
   `control.anulaciones_v35.no_equivalencia_estructural_texto_anulado_de_v34`, para que la
   anulacion sea **trazable** y no una perdida.
3. `control.anulaciones_v35.asimetria_declarada` pasa a ser la declaracion vigente de V35, para que
   las dos rutas (la clave directa y la de anulaciones) digan lo mismo y no haya dos textos
   vigentes que puedan leerse como contradictorios.

No queda ningun texto de V34 presentado como declaracion vigente: o se sustituye, o queda marcado
como anulado con su valor.

## Por que esto NO cambia el protocolo

Es una **desambiguacion documental** de la declarada asimetria entre A1 y A2, que ya estaba fijada
por el especialista antes de congelar (condicion 6: A2 es control de presupuesto, no gemelo
mecanico; ninguna conclusion puede descansar en el contraste A1 vs A2) y que el runner ya
respetaba. No cambia ni el modelo, ni el nucleo, ni el anclaje, ni `alpha*`, ni los gates, ni la
puerta, ni el arbol, ni el criterio de exito, ni el presupuesto, ni los datos. Quinta enmienda
pre-ejecucion, sin resultados.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREREG_JSON = ROOT / "experiments" / "v35_00_preregistro.json"
PREREG_MD = ROOT / "reports" / "v35_preregistro.md"
RUNLOG = ROOT / "RUNLOG.md"
EXPERIMENTS = ROOT / "experiments.csv"
FECHA = "2026-09-30"

NO_EQUIV_V35 = ("A1 tiene un regularizador (L1) que A2 no tiene: A2 es control de PRESUPUESTO, "
                "NO gemelo mecanico. Ninguna conclusion puede descansar en el contraste A1 vs A2: "
                "delta, razones entre brazos e IC bootstrap cruzado quedan contaminados. El "
                "veredicto se apoya solo en evidencia interna de A1 contra la regla derivada "
                "(G0/G1/G4) mas el triage")
SONDA_EXIGIDA_V35 = ("publicar que el veredicto NO se apoya en el contraste A1 vs A2 y que la "
                     "lectura de A2 es unidireccional (presupuesto)")

ENMIENDA = {
    "fecha": FECHA,
    "objeto": "control.asimetria_declarada (desambiguacion de la no-equivalencia estructural)",
    "de": "dos declaraciones para el mismo par de brazos: control.asimetria_declarada."
          "no_equivalencia_estructural con el texto de V34 (L2 de A2 en 1e-6 como no-op) y "
          "control.anulaciones_v35.asimetria_declarada con el de V35; solo el segundo es el de V35",
    "a": "una sola declaracion vigente, la de V35, en control.asimetria_declarada; el texto de V34 "
         "queda integro y con su valor en "
         "control.anulaciones_v35.no_equivalencia_estructural_texto_anulado_de_v34",
    "motivo": "un documento de registro con dos declaraciones para lo mismo es la ambiguedad que "
              "el proyecto ya pago; el runner ya no imprimia el texto de V34 (la linea del informe "
              "lee la asimetria de la sonda), luego el riesgo era de lectura, pero la ambiguedad "
              "seja",
    "clase": "quinta enmienda pre-ejecucion; desambiguacion documental, sin cambiar el protocolo",
    "cambia_el_protocolo": False,
    "cambia_los_gates": False,
    "estado_datos": "sin resultados; enmienda pre-ejecucion",
}


def main() -> int:
    datos = json.loads(PREREG_JSON.read_text(encoding="utf-8"))
    control = datos["control"]
    asimetria = control["asimetria_declarada"]

    texto_v34 = asimetria.get("no_equivalencia_estructural", "")
    if texto_v34.startswith("A1 tiene un regularizador"):
        print("[STOP] la declaracion vigente ya es la de V35: la enmienda no aplicaria (exit 2)")
        return 1
    if not texto_v34:
        print("[STOP] no hay texto de V34 que anular en "
              "control.asimetria_declarada.no_equivalencia_estructural")
        return 1

    anulaciones = control.setdefault("anulaciones_v35", {})
    anulaciones["no_equivalencia_estructural_texto_anulado_de_v34"] = texto_v34
    anulaciones["motivo_de_anulacion"] = (
        "describe una asimetria distinta (el L2 de A2 en 1e-6 como no-op medido en V34) y no la de "
        "V35, que es la presencia del L1 solo en A1. No se aplica a V35 y se conserva aqui, con su "
        "valor, para que la anulacion sea trazable")
    anulaciones["asimetria_declarada"] = NO_EQUIV_V35

    asimetria["no_equivalencia_estructural"] = NO_EQUIV_V35
    asimetria["sonda_exigida"] = SONDA_EXIGIDA_V35
    asimetria["vigente_para"] = "V35-E1 (reinicio one-shot de la familia FTRL, acta D-086)"

    enmiendas = datos.setdefault("enmiendas", [])
    if not any(e.get("objeto") == ENMIENDA["objeto"] for e in enmiendas):
        enmiendas.append(ENMIENDA)
    PREREG_JSON.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
    print("  control.asimetria_declarada.no_equivalencia_estructural -> texto de V35")
    print(f"  texto de V34 conservado en anulaciones_v35 "
          f"({len(texto_v34)} caracteres)")
    print(f"  enmiendas: {len(enmiendas)}")

    if "## Enmienda pre-ejecucion 5" not in PREREG_MD.read_text(encoding="utf-8"):
        bloque = (
            "\n## Enmienda pre-ejecucion 5 (2026-09-30): una sola declaracion de asimetria\n\n"
            "- **Defecto**: la enmienda 4 marco el texto de asimetria de V34 como anulado pero no "
            "lo sustituyo, luego el pre-registro declaraba **dos** no-equivalencias "
            "estructurales para el mismo par de brazos y solo una era la de V35. El runner ya no "
            "imprimia el texto de V34 (la linea del informe lee la asimetria de la sonda), asi que "
            "el riesgo era de lectura del documento de registro.\n"
            "- **Correccion**: `control.asimetria_declarada.no_equivalencia_estructural` pasa a "
            "ser **el texto de V35** (A1 tiene L1 y A2 no, luego A2 es control de PRESUPUESTO y no "
            "gemelo mecanico), y el texto de V34 queda integro y con su valor en "
            "`control.anulaciones_v35.no_equivalencia_estructural_texto_anulado_de_v34`. No queda "
            "ningun texto de V34 presentado como declaracion vigente.\n"
            "- **No cambia el protocolo**: la asimetria ya estaba fijada por el especialista antes "
            "de congelar y el runner ya la respetaba; es desambiguacion documental. Ni modelo, ni "
            "nucleo, ni anclaje, ni `alpha*`, ni gates, ni puerta, ni arbol, ni presupuesto.\n"
            "- Sin resultados: enmienda **pre-ejecucion**.\n"
        )
        PREREG_MD.write_text(PREREG_MD.read_text(encoding="utf-8").rstrip("\n") + "\n" + bloque,
                             encoding="utf-8")
        print("  MD del pre-registro: seccion de enmienda 5 anadida", flush=True)

    if "Enmienda pre-ejecucion 5 de V35-E1" not in RUNLOG.read_text(encoding="utf-8"):
        with RUNLOG.open("a", encoding="utf-8") as f:
            f.write("\n---\n\n## Enmienda pre-ejecucion 5 de V35-E1 - " + FECHA + "\n\n"
                    "- El pre-registro declaraba dos asimetrias para A1 vs A2 (la de V34 y la de "
                    "V35). Ahora hay una sola vigente, la de V35, y el texto de V34 queda "
                    "conservado con su valor como anulado trazable.\n")
        print("  RUNLOG.md: enmienda 5 anotada", flush=True)

    lineas = EXPERIMENTS.read_text(encoding="utf-8").splitlines()
    cols = [c.strip() for c in lineas[0].split(",")]
    if "V35-E1-PREREG-ENMIENDA-5" not in {lin.split(",", 1)[0] for lin in lineas[1:] if lin}:
        fila = {"experiment_id": "V35-E1-PREREG-ENMIENDA-5", "fecha": FECHA, "bloque": "35.0",
                "etapa": "Enmienda pre-ejecucion 5: una sola declaracion de asimetria A1 vs A2",
                "pregunta": "el pre-registro declara una o dos no-equivalencias estructurales?",
                "hipotesis": "dos tras la enmienda 4, por lo que hay que dejar una sola vigente",
                "features": "n/a", "preprocessing": "n/a", "representacion": "n/a", "modelo": "n/a",
                "hiperparametros": "sin cambios",
                "validacion": "el texto de V34 queda con su valor en anulaciones_v35",
                "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "",
                "micro_f1_mean": "", "accuracy_mean": "",
                "resultado": "enmienda registrada; sin resultados del experimento",
                "interpretacion": "desambiguacion documental de una declaracion ya fijada",
                "decision": "ejecutar V35-E1 con las enmiendas 1 a 5",
                "duracion_seg": "0.0",
                "artefactos": "experiments/v35_00_preregistro.json (campo enmiendas) | "
                              "reports/v35_preregistro.md"}
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerow([fila[c] for c in cols])
        with EXPERIMENTS.open("a", encoding="utf-8") as f:
            f.write(buf.getvalue())
        print("  experiments.csv: fila V35-E1-PREREG-ENMIENDA-5", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
