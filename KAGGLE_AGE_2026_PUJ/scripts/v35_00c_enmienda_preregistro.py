"""V35 - E1 - Enmienda pre-ejecucion 2: las sondas (k)-(m) son de firma; la identidad va en fold 0.

## El supuesto que resultaba falso

El pre-registro declaraba que las sondas (k), (l) y (m) "reproducen aqui los tres casos del
gemelo" dentro del runner. `ml-coder` mostro que eso, tal como estaba, es imposible si corren
sobre el **problema mini** que usan las sondas de la plantilla: el mini tiene del orden de
1350 x 51 coordenadas y el gemelo pre-ejecucion se midio sobre el ES-train real del fold 0, de
121921 filas x 107793 coordenadas. Con `lambda1 = 0.2399` el mini apenas poda (escasez efectiva
1.000000 en los tres casos), luego alli no hay nada que reproducir: la identidad numerica es
imposible por dimension, no por error de implementacion.

Ademas hay una asimetria de coste: cargar el corpus y ajustar el vectorizador del fold 0 dentro
de `--verificar` volveria la verificacion pesada (~85 s) y con efectos de carga, justo en el modo
que debe ser rapido y sin efectos secundarios.

## La enmienda

Las sondas (k), (l) y (m) se declaran **NO-GATE y de FIRMA DE MECANISMO** cuando corren en el
miniproblema: comparan el **orden** de las tres normas `||w||_2` y el signo de la razon (> 1.0),
que es lo que la hipotesis afirma, y publican los deltas como diagnostico. No son identidad
numerica y no se declaran como tales.

La **identidad numerica exacta de los tres casos** se ejecuta **dentro del runner, durante la
corrida real, en el fold 0**, sobre el stream real de ES-train (misma matriz, mismas etiquetas,
mismo orden congelado y mismo `alpha` que la sonda pre-ejecucion), y se compara contra
`experiments/v35_00b_sonda_lambda1.json` con tolerancias declaradas:

- `ll_train` y `||w||_2` de cada pasada: diferencia relativa <= 1e-9 frente al artefacto;
- mediana del anclaje y `sha256` del snapshot de `z`: diferencia EXACTA 0 e `sha256` identico
  (esto es el mismo criterio de la enmienda 1, verificado en el fold 0 real).

## Por que esto NO cambia el protocolo

Las sondas son **no-gate por pre-registro**: no deciden el veredicto en ningun caso. La decision
de mecanismo se apoya en evidencia interna de A1 sobre el stream real (la trayectoria de
`||w||_2` que ya alimenta G8) mas el triage, no en evidencia importada de `/tmp`. Ademas, el caso
`lambda1 = 0` ya tiene evidencia real versionada y ejecutada del ciclo V34-E1 (mismo nucleo, mismo
stream del fold 0) y el caso `lambda1 = mediana` es el propio A1, luego la repeticion de los tres
casos es una **confirmacion**, no la fuente de la afirmacion. No cambia el anclaje, ni los gates,
ni la puerta, ni el arbol, ni el criterio de exito, ni el numero de pasadas, ni el presupuesto
declarado (los ~60 s de los tres casos caben de sobra en los 1200 s declarados).

## Estado

**PRE-EJECUCION**: ningun fold de V35-E1 ha sido entrenado. El runner esta en construccion y su
`--verificar` aun no pasa, luego esta enmienda documenta el diseno antes de que exista resultado.
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

TOLERANCIA_RELATIVA = 1e-9
FILAS_MINI = 1350
COORD_MINI = 51
FILAS_ARTIFACTO = 121921
COORD_ARTIFACTO = 107793
ESCASEZ_MINI = 1.0

ENMIENDA = {
    "fecha": FECHA,
    "objeto": "sondas_nuevas (k)-(m): dimension del problema y alcance del contraste",
    "de": "las sondas (k), (l) y (m) reproducen dentro del runner los tres casos del gemelo "
          "pre-ejecucion (identidad numerica), sin decir donde corren",
    "a": "en el miniproblema de --verificar son NO-GATE y de FIRMA DE MECANISMO (orden de las tres "
         "normas y signo de la razon, con deltas publicados como diagnostico); la IDENTIDAD "
         "NUMERICA exacta de los tres casos se ejecuta dentro del runner durante la corrida real, "
         "en el fold 0, sobre el stream real de ES-train, con diferencia relativa <= 1e-9 en "
         "ll_train y ||w||_2, y diferencia EXACTA 0 mas sha256 identico en la mediana del anclaje",
    "motivo": f"el mini tiene del orden de {FILAS_MINI} x {COORD_MINI} coordenadas frente a las "
              f"{FILAS_ARTIFACTO} x {COORD_ARTIFACTO} del gemelo pre-ejecucion, luego la identidad "
              f"numerica es imposible por dimension; ademas con lambda1 = 0.2399 el mini apenas "
              f"poda (escasez efectiva {ESCASEZ_MINI} en los tres casos), de modo que alli no hay "
              f"magnitud que reproducir. Cargar el corpus dentro de --verificar volveria la "
              f"verificacion pesada y con efectos de carga",
    "no_gate": True,
    "cambia_el_protocolo": False,
    "cambia_el_anclaje": False,
    "afecta_a_gates": False,
    "evidencia_de_mecanismo": "la decision se apoya en la trayectoria interna de A1 sobre el stream "
                              "real (que alimenta G8) mas el triage, no en evidencia importada de "
                              "/tmp; ademas lambda1 = 0 ya tiene evidencia real versionada del "
                              "ciclo V34-E1 y lambda1 = mediana es el propio A1, luego la "
                              "repeticion de los tres casos es confirmacion y no la fuente",
    "presupuesto": f"los ~60 s de los tres casos caben en los {1200.0:g} s declarados",
    "estado_datos": "sin resultados; enmienda pre-ejecucion",
}


def main() -> int:
    datos = json.loads(PREREG_JSON.read_text(encoding="utf-8"))
    sondas = datos["sondas_nuevas"]
    sondas["politica"] = (
        "son sondas NO-GATE: no deciden el veredicto, pero la decision de mecanismo NO puede "
        "descansar en evidencia importada de /tmp. En el miniproblema de --verificar son de FIRMA "
        "DE MECANISMO (orden de las tres normas ||w||_2 y signo de la razon > 1.0, con los deltas "
        "publicados como diagnostico), porque la identidad numerica es imposible por dimension "
        f"(mini {FILAS_MINI} x {COORD_MINI} frente a {FILAS_ARTIFACTO} x {COORD_ARTIFACTO} del "
        f"gemelo, y con lambda1 = 0.2399 el mini apenas poda: escasez efectiva {ESCASEZ_MINI})"
    )
    sondas["identidad_numerica"] = {
        "donde": "dentro del runner, en el FOLD 0 de la corrida real, sobre el stream real de "
                 "ES-train (misma matriz, mismas etiquetas, mismo orden congelado y mismo alpha que "
                 "la sonda pre-ejecucion)",
        "criterio": f"diferencia relativa <= {TOLERANCIA_RELATIVA:g} en ll_train y ||w||_2 de cada "
                    "pasada frente a experiments/v35_00b_sonda_lambda1.json; diferencia EXACTA 0 "
                    "y sha256 identico en la mediana del anclaje y en el snapshot de z",
        "publica": "si, en el estado, en el informe y en el resultado, como no-gate",
    }
    enmiendas = datos.setdefault("enmiendas", [])
    if not any(e.get("objeto") == ENMIENDA["objeto"] for e in enmiendas):
        enmiendas.append(ENMIENDA)
    PREREG_JSON.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  JSON: enmiendas={len(enmiendas)}; sondas k-l-m con politica y identidad declaradas",
          flush=True)

    if "## Enmienda pre-ejecucion 2" not in PREREG_MD.read_text(encoding="utf-8"):
        bloque = (
            "\n## Enmienda pre-ejecucion 2 (2026-09-30): sondas (k)-(m) de firma; la identidad va en "
            "fold 0\n\n"
            "- **Supuesto falso**: el pre-registro decia que las sondas (k)-(m) reproducen los tres "
            f"casos del gemelo, sin decir donde. Sobre el miniproblema es imposible: "
            f"{FILAS_MINI} x {COORD_MINI} frente a {FILAS_ARTIFACTO} x {COORD_ARTIFACTO} del "
            f"gemelo, y con `lambda1 = 0.2399` el mini apenas poda (escasez efectiva "
            f"{ESCASEZ_MINI} en los tres casos).\n"
            "- **Correccion**: en el miniproblema de `--verificar` son **NO-GATE y de firma de "
            "mecanismo** (orden de las tres normas `||w||_2` y signo de la razon, con deltas como "
            "diagnostico). La **identidad numerica exacta** corre **dentro del runner, en el fold 0 "
            "de la corrida real**, sobre el stream real de ES-train, con diferencia relativa "
            f"<= {TOLERANCIA_RELATIVA:g} en `ll_train` y `||w||_2`, y diferencia **exacta 0** mas "
            "`sha256` identico en la mediana del anclaje.\n"
            "- **No cambia el protocolo**: son no-gate por pre-registro, no tocan el anclaje, ni los "
            "gates, ni la puerta, ni el arbol, ni el criterio de exito. La decision de mecanismo se "
            "apoya en la trayectoria interna de A1 (la que alimenta G8) mas el triage; ademas "
            "`lambda1 = 0` ya tiene evidencia real versionada de V34-E1 y `lambda1 = mediana` es "
            "el propio A1, luego la repeticion es confirmacion.\n"
            "- Sin resultados: enmienda **pre-ejecucion**.\n"
        )
        PREREG_MD.write_text(PREREG_MD.read_text(encoding="utf-8").rstrip("\n") + "\n" + bloque,
                             encoding="utf-8")
        print("  MD del pre-registro: seccion de enmienda 2 anadida", flush=True)

    if "Enmienda pre-ejecucion 2 de V35-E1" not in RUNLOG.read_text(encoding="utf-8"):
        with RUNLOG.open("a", encoding="utf-8") as f:
            f.write("\n---\n\n## Enmienda pre-ejecucion 2 de V35-E1 - " + FECHA + "\n\n"
                    "- El pre-registro pedia identidad numerica de los tres casos del gemelo en las "
                    f"sondas (k)-(m) sin decir donde; sobre el mini es imposible ({FILAS_MINI} x "
                    f"{COORD_MINI} frente a {FILAS_ARTIFACTO} x {COORD_ARTIFACTO}).\n"
                    "- Correccion: en el mini son de FIRMA DE MECANISMO y no-gate; la identidad "
                    "exacta corre en el fold 0 de la corrida real con tolerancia relativa "
                    f"{TOLERANCIA_RELATIVA:g} y diferencia EXACTA 0 en el anclaje.\n")
        print("  RUNLOG.md: enmienda 2 anotada", flush=True)

    lineas = EXPERIMENTS.read_text(encoding="utf-8").splitlines()
    cols = [c.strip() for c in lineas[0].split(",")]
    if "V35-E1-PREREG-ENMIENDA-2" not in {lin.split(",", 1)[0] for lin in lineas[1:] if lin}:
        fila = {"experiment_id": "V35-E1-PREREG-ENMIENDA-2", "fecha": FECHA, "bloque": "35.0",
                "etapa": "Enmienda pre-ejecucion 2: las sondas (k)-(m) son de firma en el mini; "
                        "la identidad numerica va en el fold 0 real",
                "pregunta": "pueden las sondas (k)-(m) reproducir identicamente los tres casos del "
                            "gemelo en el miniproblema?",
                "hipotesis": "no: el mini tiene otra dimension y con lambda1 = 0.2399 apenas poda",
                "features": "n/a", "preprocessing": "n/a", "representacion": "n/a", "modelo": "n/a",
                "hiperparametros": "sin cambios; tolerance relativa "
                                   f"{TOLERANCIA_RELATIVA:g} declarada para la identidad",
                "validacion": "firma de mecanismo en el mini (no-gate); identidad en fold 0 real",
                "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "",
                "micro_f1_mean": "", "accuracy_mean": "",
                "resultado": "enmienda registrada; sin resultados del experimento",
                "interpretacion": "correccion de un supuesto de diseno, no de conveniencia",
                "decision": "ejecutar V35-E1 con las enmiendas 1 y 2",
                "duracion_seg": "0.0",
                "artefactos": "experiments/v35_00_preregistro.json (campo enmiendas) | "
                              "reports/v35_preregistro.md"}
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerow([fila[c] for c in cols])
        with EXPERIMENTS.open("a", encoding="utf-8") as f:
            f.write(buf.getvalue())
        print("  experiments.csv: fila V35-E1-PREREG-ENMIENDA-2", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
