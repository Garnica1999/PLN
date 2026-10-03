"""V35 - E1 - Enmienda pre-ejecucion 1: el anclaje de lambda1 se verifica con el NUCLEO DE ANCLAJE.

## El defecto (circularidad del anclaje)

El pre-registro pedia verificar la mediana `|z_j|` **desde el `z` almacenado por A1 ya regularizado**
y exigir diferencia EXACTA 0 con la constante congelada `0.23993793862904017`. Eso es imposible:
en FTRL el estado `z` depende de `w`, y `w` depende de `lambda1`, luego el `z` de A1 **nunca puede
reproducir el umbral que lo define**. Medido en el stream real de TRAIN del fold 0 (misma matriz,
etiquetas, orden congelado y `alpha`):

| nucleo                                        | mediana(|z|>0)      | sha256(z)        |
|-----------------------------------------------|---------------------|------------------|
| anclaje, `lambda1 = 0.0`                      | **0.23993793862904017** (dif. exacta 0.0) | `ec1d1a7b...` EXACTO |
| A1 real, `lambda1 = mediana`                  | 0.2333793834039217 (dif. **-0.006558555225118473**) | `5298129a...` |

## La enmienda

La reproducibilidad del anclaje se verifica con el **NUCLEO DE ANCLAJE** (`lambda1 = 0.0`, es decir
sin regularizar) ejecutado sobre el **mismo** stream de la pasada 1 del fold 0 de A1: misma matriz,
mismas etiquetas, mismo orden congelado y mismo `alpha`. De ese `z` se toma la mediana de `|z_j| > 0`
y se exige:

- diferencia **EXACTA 0** con la constante congelada `0.23993793862904017`, y
- `sha256` del snapshot de `z` **identico** al de la sonda pre-ejecucion
  (`experiments/v35_00b_sonda_lambda1.json`), que a su vez es el artefacto **ya ejecutado** y
  versionado que produjo el numero.

Ademas se publica, como **diagnostico secundario**, la mediana del `z` del core ya regularizado
(valor esperado 0.2333793834039217), declarando que **no** es el anclaje y que su diferencia con el
anclaje (-0.006558555225118473) es la magnitud del acoplamiento `z <- w(lambda1)`.

## Por que esto NO cambia el protocolo

`lambda1` sigue siendo **el mismo numero derivado**: la mediana de `|z_j| > 0` medida sobre el stream
de TRAIN al final de la primera pasada, tal como la definio el acta D-086 y como la midio la sonda
pre-ejecucion versionada. Lo que se corrige es **como se comprueba que el numero se reproduce**, no
**cual es el numero**: un anclaje medido con un nucleo sin regularizar solo puede reproducirse con
ese mismo nucleo. No hay segundo `lambda1`, no hay fallback, no hay re-derivacion posterior a las
gates y no cambia ningun gate, ni la puerta G0-G8, ni el arbol, ni el criterio de exito.

## Estado

**PRE-EJECUCION**: ningun fold de V35-E1 ha sido entrenado (el runner esta incompleto y su
`--verificar` aun no pasa). La implementacion del runner ya sigue esta lectura, de modo que la
enmenda documenta lo que el codigo hace en vez de obligarle a cambiar.
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

MEDIANA_ANCLAJE = 0.23993793862904017
MEDIANA_CORE_REGULARIZADO = 0.2333793834039217
DIFERENCIA_REGULARIZADO = -0.006558555225118473
SHA256_Z_ANCLAJE = "ec1d1a7b13454f8b"

ENMIENDA = {
    "fecha": FECHA,
    "objeto": "anclaje_lambda1.verificacion (item 2 del diseno)",
    "de": "exigir diferencia exacta 0 y sha256 del snapshot de z leido del core YA REGULARIZADO de A1",
    "a": "verificar con el NUCLEO DE ANCLAJE (lambda1 = 0.0) sobre el mismo stream de la pasada 1 "
         "del fold 0 de A1 (misma matriz, etiquetas, orden congelado y alpha): diferencia exacta 0 "
         "con 0.23993793862904017 y sha256 identico al de la sonda pre-ejecucion versionada",
    "motivo": "circularidad: en FTRL z depende de w y w depende de lambda1, luego el z de A1 nunca "
              "puede reproducir el umbral que lo define. Medido: con lambda1 = mediana la mediana "
              f"del z es {MEDIANA_CORE_REGULARIZADO} (diferencia {DIFERENCIA_REGULARIZADO}), con "
              f"lambda1 = 0 es {MEDIANA_ANCLAJE} (diferencia exacta 0.0)",
    "diagnostico_secundario": f"se publica la mediana del core ya regularizado ({MEDIANA_CORE_REGULARIZADO}) "
                               f"declarando que NO es el anclaje y que su diferencia ({DIFERENCIA_REGULARIZADO}) "
                               "es la magnitud del acoplamiento z <- w(lambda1)",
    "sha256_del_anclaje": SHA256_Z_ANCLAJE,
    "cambia_el_protocolo": False,
    "cambia_el_numero": False,
    "afecta_a_gates": False,
    "estado_datos": "sin resultados; enmienda pre-ejecucion",
}


def main() -> int:
    datos = json.loads(PREREG_JSON.read_text(encoding="utf-8"))
    anclaje = datos["anclaje_lambda1"]
    antes = dict(anclaje.get("verificacion_runtime", {}))
    anclaje["verificacion_runtime"] = {
        "nucleo": "anclaje (lambda1 = 0.0)",
        "stream": "pasada 1 del fold 0 de A1: misma matriz, mismas etiquetas, mismo orden congelado "
                  "y mismo alpha",
        "criterio": "diferencia EXACTA 0 con el valor congelado y sha256 del snapshot de z "
                    "identico al de la sonda pre-ejecucion versionada",
        "valor_congelado": MEDIANA_ANCLAJE,
        "sha256_del_snapshot": SHA256_Z_ANCLAJE,
        "diagnostico_secundario": {
            "descripcion": "mediana del z del core YA REGULARIZADO; NO es el anclaje",
            "valor_esperado": MEDIANA_CORE_REGULARIZADO,
            "diferencia_esperada": DIFERENCIA_REGULARIZADO,
            "lectura": "la diferencia es la magnitud del acoplamiento z <- w(lambda1); por eso el "
                       "anclaje solo puede reproducirse con el core sin regularizar",
        },
    }
    anclaje["circularidad_corregida"] = (
        "el anclaje se mide con un core sin regularizar porque z depende de w y w depende "
        "de lambda1"
    )
    enmiendas = datos.setdefault("enmiendas", [])
    if not any(e.get("objeto") == ENMIENDA["objeto"] for e in enmiendas):
        enmiendas.append(ENMIENDA)
    PREREG_JSON.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  anclaje_lambda1.verificacion_runtime: {antes or '(no existia)'} -> nucleo de anclaje",
          flush=True)
    print(f"  JSON: enmiendas={len(enmiendas)}", flush=True)

    if "## Enmienda pre-ejecucion 1" not in PREREG_MD.read_text(encoding="utf-8"):
        bloque = (
            "\n## Enmienda pre-ejecucion 1 (2026-09-30): el anclaje se verifica con el NUCLEO DE ANCLAJE\n\n"
            "- **Defecto (circularidad)**: el pre-registro pedia verificar la mediana `|z_j|` desde el\n"
            "  `z` de A1 ya regularizado y exigir diferencia exacta 0. Es imposible: `z` depende de\n"
            "  `w` y `w` depende de `lambda1`, luego el `z` de A1 nunca reproduce el umbral que lo\n"
            f"  define. Medido: con `lambda1` = mediana la mediana es {MEDIANA_CORE_REGULARIZADO}\n"
            f"  (diferencia {DIFERENCIA_REGULARIZADO}); con `lambda1 = 0` es {MEDIANA_ANCLAJE}\n"
            "  (diferencia exacta 0.0).\n"
            "- **Correccion**: la reproducibilidad se verifica con el **nucleo de anclaje**\n"
            "  (`lambda1 = 0.0`) sobre el mismo stream de la pasada 1 del fold 0 de A1, exigiendo\n"
            f"  diferencia exacta 0 y `sha256` identico (`{SHA256_Z_ANCLAJE}...`) al de la sonda\n"
            "  pre-ejecucion versionada. Se publica ademas, como diagnostico secundario, la mediana\n"
            f"  del core ya regularizado ({MEDIANA_CORE_REGULARIZADO}) y su diferencia\n"
            f"  ({DIFERENCIA_REGULARIZADO}) como magnitud del acoplamiento `z <- w(lambda1)`.\n"
            "- **No cambia el protocolo ni el numero**: `lambda1` sigue siendo la mediana medida\n"
            "  sobre TRAIN al final de la primera pasada (acta D-086); lo que se corrige es COMO se\n"
            "  comprueba que se reproduce. Sin segundo `lambda1`, sin fallback, sin re-derivacion\n"
            "  posterior a las gates; ningun gate, puerta ni criterio de exito cambian.\n"
            "- Sin resultados: enmienda **pre-ejecucion**.\n"
        )
        PREREG_MD.write_text(PREREG_MD.read_text(encoding="utf-8").rstrip("\n") + "\n" + bloque,
                             encoding="utf-8")
        print("  MD del pre-registro: seccion de enmienda 1 anadida", flush=True)

    if "Enmienda pre-ejecucion 1 de V35-E1" not in RUNLOG.read_text(encoding="utf-8"):
        with RUNLOG.open("a", encoding="utf-8") as f:
            f.write("\n---\n\n## Enmienda pre-ejecucion 1 de V35-E1 - " + FECHA + "\n\n"
                    "- El anclaje de lambda1 era circular: se media con un core sin regularizar y se\n"
                    "  verificaba con el core ya regularizado, cuyo z depende de w y w de lambda1.\n"
                    f"  Medido: lambda1 = mediana -> {MEDIANA_CORE_REGULARIZADO} "
                    f"(diferencia {DIFERENCIA_REGULARIZADO}); lambda1 = 0 -> {MEDIANA_ANCLAJE} "
                    "(diferencia exacta 0.0).\n"
                    "- Correccion: la reproducibilidad se verifica con el NUCLEO DE ANCLAJE "
                    f"(lambda1 = 0.0) sobre el mismo stream, con sha256 `{SHA256_Z_ANCLAJE}`. No\n"
                    "  cambia el protocolo, ni gates, ni puerta, ni el numero del anclaje.\n")
        print("  RUNLOG.md: enmienda 1 anotada", flush=True)

    lineas = EXPERIMENTS.read_text(encoding="utf-8").splitlines()
    cols = [c.strip() for c in lineas[0].split(",")]
    if "V35-E1-PREREG-ENMIENDA-1" not in {lin.split(",", 1)[0] for lin in lineas[1:] if lin}:
        fila = {"experiment_id": "V35-E1-PREREG-ENMIENDA-1", "fecha": FECHA, "bloque": "35.0",
                "etapa": "Enmienda pre-ejecucion 1: verificacion del anclaje de lambda1 con el "
                        "nucleo de anclaje",
                "pregunta": "el z de A1 puede reproducir el umbral que lo define?",
                "hipotesis": "no: z depende de w y w depende de lambda1 (circularidad)",
                "features": "n/a", "preprocessing": "n/a", "representacion": "n/a", "modelo": "n/a",
                "hiperparametros": "verificacion con lambda1 = 0.0; valor congelado "
                                   f"{MEDIANA_ANCLAJE} sin cambios",
                "validacion": "diferencia exacta 0 y sha256 identico al de la sonda pre-ejecucion",
                "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "",
                "micro_f1_mean": "", "accuracy_mean": "",
                "resultado": "enmienda registrada; sin resultados del experimento",
                "interpretacion": "correccion de un defecto de diseno del anclaje, no de "
                                  "conveniencia",
                "decision": "ejecutar V35-E1 con la enmienda",
                "duracion_seg": "0.0",
                "artefactos": "experiments/v35_00_preregistro.json (campo enmiendas) | "
                              "reports/v35_preregistro.md"}
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerow([fila[c] for c in cols])
        with EXPERIMENTS.open("a", encoding="utf-8") as f:
            f.write(buf.getvalue())
        print("  experiments.csv: fila V35-E1-PREREG-ENMIENDA-1", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
