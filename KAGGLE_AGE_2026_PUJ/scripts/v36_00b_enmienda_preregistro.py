"""V36 - E1 - Enmienda pre-ejecucion 1: tres defectos del pre-registro y dos controles degenerados.

## Defecto 1 - el mapa tiene 6 clases y el texto decia 7

`kaggle-specialist` y `ml-architect` describieron el alfabeto como de **7** clases, y ese numero se
copio al texto de este pre-registro. La **tabla literal** que lo implementa tiene **6** claves
(`S`, `D`, `P`, `L`, `M`, `O`), y su `sha256` `cac5b2f32699a389...` **coincide** con el congelado: el
mapa nunca estuvo mal, el **texto** si. Se corrigen las tres apariciones de "7 clases".

## Defecto 2 - el rango de n estaba mal

`rango_n_de_transiciones` decia `[1, 2]`, que son **unigramas y bigramas**. El diseno, y lo que el
runner implementa, son **bigramas y trigramas**: `n` en **{2, 3}**. Con 6 clases eso da `6^2 = 36`
bigramas y `6^3 = 216` trigramas posibles, de las cuales se observan **27** y **127** respectivamente
(**154 columnas**). Se corrige el campo y se publica la cifra **medida** en vez de la que traia el
informe de arquitectura (`18/66`), que **no reproduce**.

## Defecto 3 - el control de permutaciones del alfabeto es DEGENERADO

La condicion (ii) del ruling pedia "permutar el alfabeto de 7 clases (>= 3 permutaciones) y exigir que
el mapa real gane al control". **Medido y falso**: permutar los *nombres* de las clases es una
**renombracion biyectiva** del alfabeto, luego la matriz resultante es una **permutacion exacta de
columnas** (diferencia maxima `3.3e-16`, `idf` identico) y elespacio de hipotesis del clasificador
**lineal es invariante** bajo esa permutacion. Ese control mide **ruido del solucionador**, no
especificidad de la particion. Declararlo como prueba de especificidad seria publicar un control que
no puede fallar.

**Correccion**: el control de especificidad pasa a ser un **remapeo aleatorio de caracteres a las
mismas 6 clases**, derivado de una **semilla declarada** y **nunca de los datos ni de los
resultados**: se sortea que clase recibe cada caracter, conservando el numero de clases y los
recuentos aproximados. Ahi el mapa real **si** puede perder frente al control, porque cambia **que
caracteres** caen en cada clase, no su nombre. Se exigen **>= 3 remapeos**.

## Defecto 4 - el control de barajado dentro del documento es DEGENERADO

`C3_barajado` pedia "barajar las transiciones dentro de cada documento". **Medido y falso**: el bloque
es una **bolsa de n-gramas** (el documento es un multiconjunto de recuentos), luego barajar los
tokens dentro del documento deja la matriz **bit a bit identica**. La pregunta "el orden de las
transiciones importa" es **intestable** con esta representacion, y ese control no puede fallar.

**Correccion**: el control pasa a ser un **barajado ENTRE documentos**: se barajan las transiciones
entre documentos, lo que **conserva la distribucion marginal global** y destruye la **asociacion por
documento**. Si la ganancia sobrevive, la senal es solo la distribucion global de transiciones y no
dice nada de cada tweet; si cae, la senal es por documento. Ese si es un control con contenido.

## Por que esto NO cambia el protocolo

No cambia el mapa literal (su `sha256` no se toca), ni el modelo, ni `LinearSVC(C=1.0)`, ni los folds
(que se leen del canonico), ni los gates de la serie, ni las hipotesis nulas, ni el arbol de decision,
ni la one-shot. Se corrigen **numeros escritos** (clases, rango de n, columnas) y se **sustituyen dos
controles que no podia fallar** por otros dos que si pueden. Es la primera enmienda pre-ejecucion de
V36, y sigue el precedente de las cinco enmiendas de V35: declaracion, sin resultados.

## Estado

**PRE-EJECUCION**: ningun fold de V36-E1 ha sido entrenado.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREREG_JSON = ROOT / "experiments" / "v36_00_preregistro.json"
PREREG_MD = ROOT / "reports" / "v36_preregistro.md"
RUNLOG = ROOT / "RUNLOG.md"
EXPERIMENTS = ROOT / "experiments.csv"
FECHA = "2026-09-30"

N_CLASES_REAL = 6
NS_CORREGIDO = [2, 3]
BIGRAMAS_OBSERVADAS = 27
TRIGRAMAS_OBSERVADAS = 127
COLUMNAS_TOTALES = BIGRAMAS_OBSERVADAS + TRIGRAMAS_OBSERVADAS
SHA_MAPA = "cac5b2f32699a389"
DIF_MAX_PERMUTACION = 3.3e-16

ENMIENDA = {
    "fecha": FECHA,
    "objeto": "n_classes declaradas, rango_n_de_transiciones, columnas observadas, y los dos "
              "controles de especificidad y de barajado",
    "de": {"clases": 7, "rango_n_de_transiciones": [1, 2],
           "columnas": "18/66/27/127 (del informe de arquitectura, no reproducible)",
           "control_especificidad": "permutar los NOMBRES de las clases del alfabeto (>= 3)",
           "control_barajado": "barajar las transiciones DENTRO de cada documento"},
    "a": {"clases": N_CLASES_REAL, "rango_n_de_transiciones": NS_CORREGIDO,
          "columnas": f"{BIGRAMAS_OBSERVADAS} bigramas + {TRIGRAMAS_OBSERVADAS} trigramas = "
                      f"{COLUMNAS_TOTALES} observadas (medido en el runner)",
          "control_especificidad": "REMAPEO ALEATORIO de caracteres a las mismas 6 clases, con "
                                   "semilla declarada y nunca de los datos ni de los resultados "
                                   "(>= 3 remapeos)",
          "control_barajado": "barajado ENTRE documentos: conserva la distribucion marginal global y "
                              "destruye la asociacion por documento"},
    "motivo": {
        "clases": "la tabla literal tiene 6 claves y su sha256 cac5b2f32699a389 coincide con el "
                  "congelado: el mapa nunca estuvo mal, el texto si (3 apariciones de '7 clases')",
        "rango_n": "el diseno son bigramas y trigramas (n en {2, 3}), no unigramas y bigramas",
        "columnas": "las cifras 18/66 del informe de arquitectura no reproducen; las observadas son "
                    "27 y 127, que el runner mide y publica",
        "control_especificidad": f"permutar los nombres de las clases es una renombracion "
                                 f"biyectiva: la matriz es una permutacion exacta de columnas "
                                 f"(diferencia maxima {DIF_MAX_PERMUTACION}, idf identico) y el "
                                 f"espacio de hipotesis del clasificador lineal es invariante, luego "
                                 f"el control mide ruido del solucionador y no puede fallar",
        "control_barajado": "el bloque es una bolsa de n-gramas, luego barajar dentro del documento "
                            "deja la matriz bit a bit identica: la pregunta es intestable y el "
                            "control no puede fallar",
    },
    "cambia_el_mapa_literal": False,
    "sha256_del_mapa_literal": SHA_MAPA,
    "cambia_el_modelo": False,
    "cambia_los_gates": False,
    "cambia_el_arbol": False,
    "sigue_one_shot": True,
    "estado_datos": "sin resultados; enmienda pre-ejecucion",
}


def main() -> int:
    datos = json.loads(PREREG_JSON.read_text(encoding="utf-8"))

    mapa = datos["mapa_de_clases_literal"]
    if len(mapa) != N_CLASES_REAL:
        print(f"[STOP] el mapa literal tiene {len(mapa)} claves, no {N_CLASES_REAL}: la premisa de la "
              "enmienda cae (exit 2)")
        return 1
    if datos["hashes"]["mapa_literal_de_clases"][:16] != SHA_MAPA:
        print("[STOP] el sha256 del mapa literal no es el congelado (exit 2)")
        return 1
    if datos["rango_n_de_transiciones"] != [1, 2]:
        print(f"[STOP] rango_n actual {datos['rango_n_de_transiciones']} != [1, 2]: la enmienda no "
              "aplicaria (exit 2)")
        return 1

    datos["n_clases_del_alfabeto"] = N_CLASES_REAL
    datos["rango_n_de_transiciones"] = NS_CORREGIDO
    datos["columnas_observadas"] = {
        "bigramas": BIGRAMAS_OBSERVADAS, "trigramas": TRIGRAMAS_OBSERVADAS,
        "total": COLUMNAS_TOTALES,
        "medido": "por el runner, no copiado del informe de arquitectura",
        "lectura": f"con 6 clases hay 6^2 = 36 bigramas y 6^3 = 216 trigramas posibles; se observan "
                   f"{BIGRAMAS_OBSERVADAS} y {TRIGRAMAS_OBSERVADAS}"}
    datos["controles"]["C2_permutaciones"] = (
        "REMAPEO ALEATORIO de caracteres a las mismas 6 clases, con semilla declarada y nunca "
        "derivada de los datos ni de los resultados; >= 3 remapeos. Sustituye a la permutacion de "
        "los NOMBRES de las clases, que era degenerada: al ser una renombracion biyectiva deja la "
        f"matriz como permutacion exacta de columnas (diferencia maxima {DIF_MAX_PERMUTACION}, idf "
        "identico) y el espacio de hipotesis lineal es invariante, luego media ruido del "
        "solucionador y no podia fallar")
    datos["controles"]["C3_barajado"] = (
        "barajado ENTRE documentos de las transiciones: conserva la distribucion marginal global y "
        "destruye la asociacion por documento. Sustituye al barajado DENTRO de cada documento, que "
        "era degenerado: el bloque es una bolsa de n-gramas, luego barajar dentro del documento deja "
        "la matriz bit a bit identica y la pregunta era intestable")
    datos["sondas"]["permutaciones"] = (
        "resultado de los >= 3 REMAPEOS aleatorios de caracteres a las mismas 6 clases, exigido por la "
        "condicion (ii) corregida; el mapa real debe ganar a todos ellos, y si no gana el veredicto "
        "es 'sin senal especifica de categoria' y L4 se cierra igual")
    datos["sondas"]["barajado"] = (
        "resultado del control de barajado ENTRE documentos, exigido por C3 corregida: separa la "
        "senal por documento de la simple distribucion global de transiciones")
    datos["hipotesis_nulas"]["especificidad"]["criterio"] = (
        "el mapa real debe ganar a >= 3 REMAPEOS aleatorios de caracteres a las mismas 6 clases; si "
        "no gana, el veredicto es 'sin senal especifica de categoria' y L4 se cierra igual")

    enmiendas = datos.setdefault("enmiendas", [])
    if not any(e.get("objeto") == ENMIENDA["objeto"] for e in enmiendas):
        enmiendas.append(ENMIENDA)
    PREREG_JSON.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  n_clases_del_alfabeto: 7 -> {N_CLASES_REAL}")
    print(f"  rango_n_de_transiciones: [1, 2] -> {NS_CORREGIDO}")
    print(f"  columnas_observadas: 27 + 127 = {COLUMNAS_TOTALES}")
    print("  C2_permutaciones -> remapeo aleatorio de caracteres (>= 3)")
    print("  C3_barajado -> barajado entre documentos")
    print(f"  enmiendas: {len(enmiendas)}")

    md = PREREG_MD.read_text(encoding="utf-8")
    if "## Enmienda pre-ejecucion 1" not in md:
        bloque = (
            "\n## Enmienda pre-ejecucion 1 (2026-09-30): 3 defectos del pre-registro y 2 controles "
            "degenerados\n\n"
            "Detectado por `ml-coder` al implementar el runner, antes de ejecutar.\n\n"
            "1. **El mapa tiene 6 clases y el texto decia 7.** La tabla literal tiene 6 claves (`S`, "
            "`D`, `P`, `L`, `M`, `O`) y su `sha256` `cac5b2f32699a389` **coincide** con el congelado: el "
            "mapa nunca estuvo mal, el **texto** si (3 apariciones de \"7 clases\").\n"
            "2. **El rango de n estaba mal**: decia `[1, 2]` (unigramas y bigramas) y el diseno son "
            "**bigramas y trigramas**, `n` en **{2, 3}**.\n"
            "3. **Las columnas observadas** son **27 bigramas + 127 trigramas = 154** (medido), no las "
            "`18/66` del informe de arquitectura, que no reproducen.\n"
            "4. **El control de permutaciones del alfabeto era DEGENERADO**: permutar los *nombres* de "
            "las clases es una **renombracion biyectiva**, luego la matriz es una **permutacion exacta "
            f"de columnas** (diferencia maxima `{DIF_MAX_PERMUTACION}`, `idf` identico) y el espacio "
            "de hipotesis del clasificador lineal es **invariante**. Media ruido del solucionador y "
            "**no podia fallar**. **Sustituido** por un **remapeo aleatorio de caracteres a las mismas "
            "6 clases**, con semilla declarada y nunca de los datos: ahi el mapa real **si** puede "
            "perder, porque cambia que caracteres caen en cada clase y no su nombre. Siguen exigiendose "
            "**>= 3**.\n"
            "5. **El control de barajado dentro del documento era DEGENERADO**: el bloque es una "
            "**bolsa de n-gramas**, luego barajar dentro del documento deja la matriz **bit a bit "
            "identica** y la pregunta \"el orden importa\" es **intestable**. **Sustituido** por un "
            "barajado **entre documentos**, que conserva la distribucion marginal global y destruye "
            "la asociacion por documento.\n"
            "- **No cambia el protocolo**: ni el mapa literal (su `sha256` no se toca), ni el modelo, "
            "ni `LinearSVC(C=1.0)`, ni los folds, ni los gates, ni las hipotesis nulas, ni el arbol, "
            "ni la one-shot. Se corrigen **numeros escritos** y se sustituyen **dos controles que no "
            "podian fallar** por dos que si pueden.\n"
            "- Sin resultados: enmienda **pre-ejecucion**.\n"
        )
        PREREG_MD.write_text(md.rstrip("\n") + "\n" + bloque, encoding="utf-8")
        print("  MD del pre-registro: seccion de enmienda 1 anadida", flush=True)

    # las tres apariciones de "7 clases" del cuerpo principal
    md = PREREG_MD.read_text(encoding="utf-8")
    corte = md.find("\n## Enmienda pre-ejecucion 1")
    if corte > 0:
        cuerpo, seccion = md[:corte], md[corte:]
        antes = cuerpo.count("7 clases")
        cuerpo = cuerpo.replace("mapa de 7 clases", "mapa de 6 clases").replace(
            "7 clases", "6 clases").replace("alfabeto de 7 clases", "alfabeto de 6 clases")
        PREREG_MD.write_text(cuerpo + seccion, encoding="utf-8")
        print(f"  cuerpo del MD: {antes} apariciones de '7 clases' corregidas", flush=True)

    if "Enmienda pre-ejecucion 1 de V36-E1" not in RUNLOG.read_text(encoding="utf-8"):
        with RUNLOG.open("a", encoding="utf-8") as f:
            f.write("\n---\n\n## Enmienda pre-ejecucion 1 de V36-E1 · " + FECHA + "\n\n"
                    "- Tres defectos del pre-registro: el mapa tiene **6** clases (el texto decia 7, con "
                    "sha256 `cac5b2f32699a389` verificado), el rango de n era `[1, 2]` cuando el "
                    "diseno es bigramas y trigramas `[2, 3]`, y las columnas observadas son **27 + 127 = "
                    "154**, no `18/66`.\n"
                    "- Dos controles eran **degenerados**: permutar los NOMBRES de las clases es una "
                    f"renombracion biyectiva (permutacion exacta de columnas, idf identico, "
                    f"diferencia maxima {DIF_MAX_PERMUTACION}) y el espacio de hipotesis lineal es "
                    "invariante; y barajar dentro del documento deja la matriz bit a bit identica "
                    "porque el bloque es una bolsa de n-gramas. **Ninguno de los dos podia fallar.**\n"
                    "- Sustituidos por: **remapeo aleatorio de caracteres a las mismas 6 clases** "
                    "(>= 3, con semilla declarada) y **barajado entre documentos**.\n"
                    "- No cambia el protocolo: ni mapa, ni modelo, ni folds, ni gates, ni arbol, ni "
                    "one-shot.\n")
        print("  RUNLOG.md: enmienda 1 anotada", flush=True)

    lineas = EXPERIMENTS.read_text(encoding="utf-8").splitlines()
    cols = [c.strip() for c in lineas[0].split(",")]
    if "V36-E1-PREREG-ENMIENDA-1" not in {l.split(",", 1)[0] for l in lineas[1:] if l}:
        fila = {"experiment_id": "V36-E1-PREREG-ENMIENDA-1", "fecha": FECHA, "bloque": "36.0",
                "etapa": "Enmienda pre-ejecucion 1: 3 defectos del pre-registro y 2 controles "
                        "degenerados",
                "pregunta": "los controles de especificidad y de barajado pueden fallar?",
                "hipotesis": "no: los dos son degenerados por construccion",
                "features": "n/a", "preprocessing": "n/a",
                "representacion": "mapa de 6 clases (sha256 cac5b2f32699a389, sin cambios)",
                "modelo": "n/a",
                "hiperparametros": "n en {2, 3} corregido; 27 bigramas + 127 trigramas = 154 columnas",
                "validacion": "remapeo aleatorio de caracteres (>= 3) y barajado entre documentos",
                "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "",
                "micro_f1_mean": "", "accuracy_mean": "",
                "resultado": "enmienda registrada; sin resultados del experimento",
                "interpretacion": "correcion de defectos del pre-registro y de dos controles que no "
                                  "podian fallar; el protocolo no cambia",
                "decision": "ejecutar V36-E1 con la enmienda 1",
                "duracion_seg": "0.0",
                "artefactos": "experiments/v36_00_preregistro.json (campo enmiendas) | "
                              "reports/v36_preregistro.md"}
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerow([fila[c] for c in cols])
        with EXPERIMENTS.open("a", encoding="utf-8") as f:
            f.write(buf.getvalue())
        print("  experiments.csv: fila V36-E1-PREREG-ENMIENDA-1", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())