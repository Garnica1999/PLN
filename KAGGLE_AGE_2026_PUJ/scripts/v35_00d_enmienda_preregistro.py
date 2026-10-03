"""V35 - E1 - Enmienda pre-ejecucion 3: se corrige el `token_pattern` del pre-registro.

## El defecto (deriva del pre-registro)

`experiments/v35_00_preregistro.json` congelo el vectorizador con el `token_pattern`
`'(?u)\\\\b\\\\w+\\\\b'` (es decir, la cadena literal con **backslashes dobles**) y su hash
`33311e5eddc0b0bbfd07505b9ed3474bb824d1b41fac92d3d7bfd3fd902ba977`. Ese patron **no compila
ninguna palabra**: medido, `re.compile` del patron congelado no encuentra match en
`"hola mundo"`, luego el vocabulario seria **vacio** y la representacion **inservible**.

El vectorizador canonico del proyecto es `'(?u)\\b\\w+\\b'` (una sola barra), con hash
`9bae157614458dbdf26fb7aed4c78d2954c36e895f830341e59b23fa63e2b51b`, y es el que declaran los
pre-registros de V33 y de V34 y el que produjo las 107793 coordenadas que registran tanto el
nucleo de V34 como la sonda pre-ejecucion de esta via. El error es de **redaccion del
pre-registro** (una cadena cruda con las barras ya escapadas), no de ejecucion: ningun fold de
V35-E1 se ha entrenado y el codigo nunca uso el patron defectuoso.

## La enmienda

Se corrigen `config.vectorizador.token_pattern` y `hashes.vectorizador` a los valores canonicos,
de modo que el pre-registro vuelve a ser la fuente unica de verdad y coincide con V33, con V34 y
con el vectorizador que el runner construye. El codigo del runner ya declara la equivalencia en
lugar de igualar en silencio, y la publica.

## Por que esto NO cambia el protocolo

No cambia el modelo, ni el nucleo FTRL, ni el anclaje de `lambda1`, ni los gates, ni la puerta,
ni el arbol, ni el criterio de exito, ni el presupuesto, ni los datos. Cambia **la declaracion de
un parametro que estaba mal escrito** por el valor que el resto del proyecto ya usa y que el
propio runner necesita para ser ejecutable. Es la **tercera enmienda pre-ejecucion** y sigue el
precedente de `v34_00e` (misma clase: correccion de un literal del pre-registro antes de
ejecutar).

## Estado

**PRE-EJECUCION**: ningun fold de V35-E1 ha sido entrenado.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREREG_JSON = ROOT / "experiments" / "v35_00_preregistro.json"
PREREG_MD = ROOT / "reports" / "v35_preregistro.md"
RUNLOG = ROOT / "RUNLOG.md"
EXPERIMENTS = ROOT / "experiments.csv"
FECHA = "2026-09-30"

TOKEN_PATTERN_CONGELADO = r"(?u)\\b\\w+\\b"
HASH_CONGELADO = "33311e5eddc0b0bbfd07505b9ed3474bb824d1b41fac92d3d7bfd3fd902ba977"
TOKEN_PATTERN_CANONICO = r"(?u)\b\w+\b"
HASH_CANONICO = "9bae157614458dbdf26fb7aed4c78d2954c36e895f830341e59b23fa63e2b51b"
PRUEBA = "hola mundo"

ENMIENDA = {
    "fecha": FECHA,
    "objeto": "config.vectorizador.token_pattern y hashes.vectorizador",
    "de": {"token_pattern": TOKEN_PATTERN_CONGELADO, "hashes_vectorizador": HASH_CONGELADO},
    "a": {"token_pattern": TOKEN_PATTERN_CANONICO, "hashes_vectorizador": HASH_CANONICO},
    "motivo": "el patron congelado lleva backslashes dobles: no compila ninguna palabra (medido, "
              f"re.compile del patron congelado no encuentra match en {PRUEBA!r}), luego el "
              "vocabulario seria vacio y la representacion inservible. El canonico es el que "
              "declaran los pre-registros de V33 y V34 y el que produjo las 107793 coordenadas "
              "que registran el nucleo de V34 y la sonda pre-ejecucion de esta via",
    "clase": "tercera enmienda pre-ejecucion; mismo precedente que v34_00e (correccion de un "
             "literal del pre-registro antes de ejecutar)",
    "cambia_el_protocolo": False,
    "cambia_el_modelo": False,
    "cambia_los_gates": False,
    "estado_datos": "sin resultados; enmienda pre-ejecucion",
}


def main() -> int:
    datos = json.loads(PREREG_JSON.read_text(encoding="utf-8"))
    vec = datos["config"]["vectorizador"]

    if vec.get("token_pattern") != TOKEN_PATTERN_CONGELADO:
        print("[STOP] el token_pattern actual no es el congelado que corrige esta enmienda: "
              f"{vec.get('token_pattern')!r}", flush=True)
        return 1
    if datos["hashes"]["vectorizador"] != HASH_CONGELADO:
        print("[STOP] el hash congelado no es el que corrige esta enmienda", flush=True)
        return 1
    if re.compile(TOKEN_PATTERN_CONGELADO).search(PRUEBA) is not None:
        print("[STOP] el patron congelado SI encuentra match: la premisa de la enmienda cae", flush=True)
        return 1
    if re.compile(TOKEN_PATTERN_CANONICO).search(PRUEBA) is None:
        print("[STOP] el patron canonico NO encuentra match", flush=True)
        return 1

    antes_pat, antes_hash = vec["token_pattern"], datos["hashes"]["vectorizador"]
    vec["token_pattern"] = TOKEN_PATTERN_CANONICO
    recomputado = hashlib.sha256(
        json.dumps(vec, sort_keys=True).encode("utf-8")).hexdigest()
    if recomputado != HASH_CANONICO:
        print(f"[STOP] el hash recomputado {recomputado} no es el canonico {HASH_CANONICO}",
              flush=True)
        return 1
    datos["hashes"]["vectorizador"] = HASH_CANONICO
    vec["nota_token_pattern"] = (
        "canonical con una sola barra; el valor con barras dobles que figuraba antes en este "
        f"pre-registro (hash {HASH_CONGELADO[:16]}) deja el vocabulario vacio y no se usa. "
        f"Coherente con V33 y V34 (hash {HASH_CANONICO[:16]})"
    )
    enmiendas = datos.setdefault("enmiendas", [])
    if not any(e.get("objeto") == ENMIENDA["objeto"] for e in enmiendas):
        enmiendas.append(ENMIENDA)
    PREREG_JSON.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  token_pattern: {antes_pat!r} -> {TOKEN_PATTERN_CANONICO!r}", flush=True)
    print(f"  hash vectorizador: {antes_hash[:16]}... -> {HASH_CANONICO[:16]}... "
          f"(recomputado y verificado)", flush=True)
    print(f"  JSON: enmiendas={len(enmiendas)}", flush=True)

    if "## Enmienda pre-ejecucion 3" not in PREREG_MD.read_text(encoding="utf-8"):
        bloque = (
            "\n## Enmienda pre-ejecucion 3 (2026-09-30): `token_pattern` del vectorizador corregido\n\n"
            f"- **Defecto**: el pre-registro congelo `'(?u)\\\\b\\\\w+\\\\b'` (backslashes dobles, hash "
            f"`{HASH_CONGELADO[:16]}`). Medido: `re.compile` de ese patron **no encuentra match en "
            f"`{PRUEBA!r}`**, luego el vocabulario seria **vacio** y la representacion inservible. "
            "Es un error de redaccion del pre-registro (cadena cruda con las barras ya escapadas): "
            "ningun fold se ha entrenado y el codigo nunca uso el patron defectuoso.\n"
            f"- **Correccion**: `'(?u)\\b\\w+\\b'` (una sola barra) con hash `{HASH_CANONICO[:16]}`, "
            "que es el que declaran los pre-registros de V33 y V34 y el que produjo las 107793 "
            "coordenadas que registran el nucleo de V34 y la sonda pre-ejecucion de esta via.\n"
            "- **No cambia el protocolo**: no toca modelo, nucleo FTRL, anclaje, gates, puerta, "
            "arbol, criterio de exito, presupuesto ni datos; corrige el literal de un parametro "
            "para que el pre-registro vuelva a ser la fuente unica de verdad. Mismo precedente "
            "que `v34_00e`.\n"
            "- Sin resultados: enmienda **pre-ejecucion**.\n"
        )
        PREREG_MD.write_text(PREREG_MD.read_text(encoding="utf-8").rstrip("\n") + "\n" + bloque,
                             encoding="utf-8")
        print("  MD del pre-registro: seccion de enmienda 3 anadida", flush=True)

    if "Enmienda pre-ejecucion 3 de V35-E1" not in RUNLOG.read_text(encoding="utf-8"):
        with RUNLOG.open("a", encoding="utf-8") as f:
            f.write("\n---\n\n## Enmienda pre-ejecucion 3 de V35-E1 - " + FECHA + "\n\n"
                    "- El `token_pattern` congelado llevaba backslashes dobles y no encuentra "
                    f"match en {PRUEBA!r} (vocabulario vacio, representacion inservible); el "
                    f"canonico es `'(?u)\\b\\w+\\b'` con hash {HASH_CANONICO[:16]}, el mismo que "
                    "declaran V33 y V34.\n"
                    "- Corregidos `config.vectorizador.token_pattern` y `hashes.vectorizador`; el "
                    "hash se recomputo y se verifico contra el valor canonico. No cambia modelo, "
                    "nucleo, anclaje, gates, puerta, arbol ni presupuesto.\n")
        print("  RUNLOG.md: enmienda 3 anotada", flush=True)

    lineas = EXPERIMENTS.read_text(encoding="utf-8").splitlines()
    cols = [c.strip() for c in lineas[0].split(",")]
    if "V35-E1-PREREG-ENMIENDA-3" not in {lin.split(",", 1)[0] for lin in lineas[1:] if lin}:
        fila = {"experiment_id": "V35-E1-PREREG-ENMIENDA-3", "fecha": FECHA, "bloque": "35.0",
                "etapa": "Enmienda pre-ejecucion 3: token_pattern del vectorizador corregido al "
                        "valor canonico",
                "pregunta": "el token_pattern congelado es utilizable?",
                "hipotesis": "no: con backslashes dobles el vocabulario es vacio",
                "features": "n/a", "preprocessing": "n/a", "representacion": "n/a", "modelo": "n/a",
                "hiperparametros": f"token_pattern canonico {TOKEN_PATTERN_CANONICO!r}; hash "
                                   f"{HASH_CANONICO[:16]} recomputado y verificado",
                "validacion": "re.compile del patron congelado no encuentra match; el canonico si",
                "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "",
                "micro_f1_mean": "", "accuracy_mean": "",
                "resultado": "enmienda registrada; sin resultados del experimento",
                "interpretacion": "correccion de un literal mal escrito en el pre-registro, no de "
                                  "conveniencia; mismo precedente que v34_00e",
                "decision": "ejecutar V35-E1 con las enmiendas 1, 2 y 3",
                "duracion_seg": "0.0",
                "artefactos": "experiments/v35_00_preregistro.json (campo enmiendas) | "
                              "reports/v35_preregistro.md"}
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerow([fila[c] for c in cols])
        with EXPERIMENTS.open("a", encoding="utf-8") as f:
            f.write(buf.getvalue())
        print("  experiments.csv: fila V35-E1-PREREG-ENMIENDA-3", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
