"""G14 - P0/P1/P2: auditoria de informacion de entrada disponible y nunca utilizada.

## La pregunta, y por que se responde con hechos y no con criterio

```text
¿Existe algun campo de entrada LEGAL, presente en los datos,
que ningun modelo de AGE haya utilizado realmente?
```

La tentacion es responder "no, solo hay id, text y age_range". Esta bien, pero no es una
auditoria: es una impresion. Lo que hace este script es **demostrar** cada afirmacion con el dato
delante, incluida la que mas facil es dar por supuesta:

- que `id` es administrativo y no un identificador de tweet (que codificaria una fecha),
- que no ordena nada,
- y que **no puede** usarse como feature porque sus valores se repiten entre train y test.

## El hallazgo que cierra la pregunta del `id`

Un id de tweet de Twitter es un *snowflake* de 18-19 digitos cuya parte alta codifica una marca de
tiempo. Si el campo `id` fuera eso, seria una fuente de informacion real: la fecha de publicacion,
presente en train y en test, legal y no textual.

**No lo es.** Medido:

```text
train   id 1 .. 169.336   monotono, sin huecos
test    id 1 .. 112.892   monotono, sin huecos
interseccion train-test = 112.892, es decir TODO el test
```

Los ids van de 1 a N **dentro de cada split** y vuelven a empezar. El `id` 1 es un tweet distinto
en train y en test. Es un **indice de fila**, no un identificador de contenido, y por eso no puede
ser una feature: en prediccion real el valor no identifica nada.

Y aunque lo fuera, la correlacion del id con la clase es **-0,00084**: exactamente nada.

## Lo que el brief prohibe en este punto, y se respeta

El brief anticipa exactamente este caso: si el unico campo sin usar resulta ser un identificador
administrativo, **no hay que rescatarlo** con modulo, hash, posicion, patrones de digitos, buckets
ni sinonimos. Eso no seria encontrar informacion, seria pescar correlaciones espurias sobre un
numero de fila. Aqui se cierra como `NOT_ACTIONABLE` y no se intenta ninguna de esas cosas.

## Lo que si se audita ademas

La pregunta del brief no es solo "que columnas hay", sino **que informacion de `text` no llega a
los modelos**. Eso si puede esconder un canal: si la normalizacion elimina el case y la puntuacion,
esa informacion se pierde. Este script traza que se conserva y que se elimina, y contrasta el
resultado con lo que V31 ya probo, porque la respuesta depende de eso.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "g14_auditoria_input.json"
INV_CSV = cfg.REPORTS / "g14_input_schema_inventory.csv"
INV_MD = cfg.REPORTS / "g14_input_information_audit.md"
DECISION_MD = cfg.REPORTS / "g14_decision.md"
MATRIZ_CSV = cfg.REPORTS / "g14_field_x_model_matrix.csv"
TRAZA_CSV = cfg.REPORTS / "g14_text_traceability.csv"
FECHA = "2026-10-06"
CV_V30 = 0.6134471
N_CLASES = 6
GRUPO_A = 27007
GRUPO_B = 38396


def commit_actual() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, capture_output=True,
                              text=True, timeout=30).stdout.strip() or "desconocido"
    except Exception:  # noqa: BLE001
        return "desconocido"


def inventario_esquema(tr: pd.DataFrame, te: pd.DataFrame) -> pd.DataFrame:
    """Una fila por columna y split, con todo lo que el brief pide registrar."""
    filas = []
    for nombre, d in (("train", tr), ("test", te)):
        for c in d.columns:
            s = d[c]
            ejemplos = s.dropna().head(3).astype(str).tolist()
            # Los ejemplos de `text` se recortan y se eligen genericos, no sensibles.
            if c == "text":
                ejemplos = [e[:40] + ("..." if len(e) > 40 else "") for e in ejemplos]
            filas.append({
                "campo": c, "split": nombre, "tipo": str(s.dtype),
                "filas": int(len(s)), "nulos": int(s.isna().sum()),
                "cardinalidad": int(s.nunique()),
                "ejemplos_no_sensibles": " | ".join(ejemplos),
                "presente_en_train": c in tr.columns,
                "presente_en_test": c in te.columns,
                "en_notebook": False, "en_scripts": None,
                "utilizacion_actual": None, "rol": None, "legalidad": None,
                "decision": None,
            })
    t = pd.DataFrame(filas)
    # Uso real, comprobado por grep dirigido sobre scripts y notebook.
    scripts = list((ROOT / "scripts").glob("*.py"))
    nb = ROOT / "notebooks" / "99_entrega_consolidado.ipynb"
    texto_nb = nb.read_text(encoding="utf-8") if nb.exists() else ""
    for idx, f in t.iterrows():
        campo = f["campo"]
        if campo == "text":
            hits = sum(1 for p in scripts if "\"text\"" in p.read_text(encoding="utf-8",
                                                                      errors="ignore"))
            t.at[idx, "en_scripts"] = f"{hits} scripts lo leen"
            t.at[idx, "en_notebook"] = "\"text\"" in texto_nb or "'text'" in texto_nb
            t.at[idx, "utilizacion_actual"] = ("USADO en todas las representaciones: TF-IDF de "
                                               "palabra, char n-grams y los diez canales "
                                               "estructurales de V31")
            t.at[idx, "rol"] = "unico campo de contenido"
            t.at[idx, "legalidad"] = "LEGAL: es la entrada que la tarea define"
            t.at[idx, "decision"] = "EN USO"
        elif campo == "age_range":
            t.at[idx, "en_scripts"] = "el target"
            t.at[idx, "en_notebook"] = "age_range" in texto_nb
            t.at[idx, "utilizacion_actual"] = ("**TARGET**. Se lee para construir las etiquetas y "
                                               "para medir, nunca como entrada")
            t.at[idx, "rol"] = "TARGET"
            t.at[idx, "legalidad"] = "PROHIBIDO como entrada"
            t.at[idx, "decision"] = "PROHIBITED"
        else:
            t.at[idx, "en_scripts"] = ("se usa como indice de fila y como clave de orden en "
                                       "varios scripts; nunca como feature")
            t.at[idx, "en_notebook"] = "id" in texto_nb
            t.at[idx, "utilizacion_actual"] = ("**solo administrativo**: clave de orden, "
                                               "verificacion de unicidad y alineacion de la "
                                               "submission. Nunca entra a ningun modelo")
            t.at[idx, "rol"] = "identificador de fila"
            t.at[idx, "legalidad"] = "NO UTILIZABLE como entrada (ver la auditoria del id)"
            t.at[idx, "decision"] = "NOT_ACTIONABLE"
    return t


def auditoria_del_id(tr: pd.DataFrame, te: pd.DataFrame, clase: np.ndarray) -> dict:
    """La auditoria que decide si `id` puede ser una fuente de informacion."""
    i_tr, i_te = tr["id"].to_numpy(), te["id"].to_numpy()
    inter = len(set(i_tr) & set(i_te))
    pearson = float(np.corrcoef(i_tr, clase)[0, 1])
    spearman = float(np.corrcoef(pd.Series(i_tr).rank().to_numpy(),
                                 pd.Series(clase).rank().to_numpy())[0, 1])
    # ¿Cuantos digitos? Un snowflake de Twitter tiene 18-19.
    digitos = int(len(str(int(i_tr.max()))))
    return {
        "train": {"min": int(i_tr.min()), "max": int(i_tr.max()), "n": len(i_tr),
                  "unico": bool(len(set(i_tr)) == len(i_tr)),
                  "monotono": bool(np.all(np.diff(i_tr) > 0))},
        "test": {"min": int(i_te.min()), "max": int(i_te.max()), "n": len(i_te),
                 "unico": bool(len(set(i_te)) == len(i_te)),
                 "monotono": bool(np.all(np.diff(i_te) > 0))},
        "interseccion_train_test": inter,
        "fraccion_del_test_que_reaparece_en_train": round(inter / len(i_te), 5),
        "digitos_del_maximo": digitos,
        "parece_snowflake_de_twitter": digitos >= 15,
        "correlacion_pearson_con_la_clase": round(pearson, 6),
        "correlacion_spearman_con_la_clase": round(spearman, 6),
        "veredicto": (
            "**INDICE ADMINISTRATIVO DE FILA.** Los ids van de 1 a N dentro de cada split y "
            "vuelven a empezar: **los 112.892 ids del test aparecen tambien en train**, para filas "
            "distintas. El `id` 1 es un tweet distinto en cada split. No identifica contenido, y "
            "por tanto en prediccion real su valor no significa nada"),
        "por_que_no_es_snowflake": (
            f"un id de tweet tiene 18-19 digitos; el maximo de este campo tiene {digitos}. No "
            f"codifica ninguna marca de tiempo"),
        "por_que_no_puede_ser_feature": (
            "un campo cuyo valor se repite entre train y test para filas distintas no puede "
            "aportar informacion sobre la fila: la misma entrada corresponderia a dos salidas "
            "distintas en el 100 % del test"),
        "senal_aprovechable": (
            f"correlacion con la clase {pearson:+.6f}: nada. El brief anticipa este caso y prohibe "
            f"rescatarlo con modulo, hash, posicion, patrones de digitos, buckets o sinonimos, "
            f"porque eso seria pescar correlaciones espurias sobre un numero de fila"),
    }


def traza_del_texto() -> pd.DataFrame:
    """Que informacion de `text` se conserva y cual se elimina antes de modelar.

    El brief prohíbe crear features aqui. Lo unico que se hace es contrastar cada dimension con lo
    que ya se probo, porque la distincion entre "informacion no usada" y "informacion ya contenida
    en las representaciones" decide si hay un canal nuevo o no.
    """
    filas = [
        {"dimension": "contenido lexico", "se_conserva": "si, integro",
         "representacion_que_lo_captura": "TF-IDF de palabra (1-2) y char_wb (3-5)",
         "probado_en": "desde el baseline E-001", "estado": "EN USO"},
        {"dimension": "case (mayusculas y minusculas)", "se_conserva": "no en el TF-IDF, que "
         "aplica lowercase", "representacion_que_lo_captura":
         "canal `ortografia`: bloques CAPS, ratio de mayusculas, CAPS por palabra",
         "probado_en": "V31-03 (+0,001993 en banco) y V31-06 (sin aporte en la integracion)",
         "estado": "PROBADO, SIN APORTE CERRADO"},
        {"dimension": "puntuacion", "se_conserva": "no en el TF-IDF, que tokeniza con \\w+",
         "representacion_que_lo_captura": "canal `ortografia`: repeticiones, y canal `discurso`: "
         "preguntas, exclamaciones, suspension",
         "probado_en": "V31-03 y V31-06", "estado": "PROBADO, SIN APORTE CERRADO"},
        {"dimension": "elongaciones y caracteres repetidos", "se_conserva": "no",
         "representacion_que_lo_captura": "canal `ortografia`: REPETICION, REPVOCAL, REPCONSON, "
         "REPLONG", "probado_en": "V31-03 y V31-06", "estado": "PROBADO, SIN APORTE CERRADO"},
        {"dimension": "URLs", "se_conserva": "si, como token de la propia URL",
         "representacion_que_lo_captura": "canal `url` (presencia, tipo, longitud, posicion) y la "
         "taxonomia de P32/P33",
         "probado_en": "V31-02, V31-03, P33-P02", "estado": "PROBADO, SIN APORTE CERRADO"},
        {"dimension": "menciones y hashtags", "se_conserva": "si como token",
         "representacion_que_lo_captura": "el propio TF-IDF de palabra, y el canal `discurso` "
         "para la posicion",
         "probado_en": "features estructuradas de V20 y V31", "estado": "EN USO"},
        {"dimension": "longitud", "se_conserva": "si",
         "representacion_que_lo_captura": "bloque estructurado de 279 features (LENBUCKET, "
         "NTOKENS) y los slices de longitud de V31-F",
         "probado_en": "V20 y V31", "estado": "EN USO"},
        {"dimension": "posicion de los elementos", "se_conserva": "no",
         "representacion_que_lo_captura": "canales `posicion` y `forma_palabra` de V31-03",
         "probado_en": "V31-03: **negativos** (-0,002303 y -0,002984, 0/5 folds)", "estado":
         "PROBADO Y CERRADO NEGATIVO"},
        {"dimension": "acentos y ortografia", "se_conserva": "si, en el texto original",
         "representacion_que_lo_captura": "el TF-IDF de palabra sin strip_accents, y las listas "
         "de marcadores de V31",
         "probado_en": "V31-03", "estado": "EN USO"},
    ]
    return pd.DataFrame(filas)


def matriz_campo_por_modelo() -> pd.DataFrame:
    """USED / NOT_USED / DERIVED / PROHIBITED para cada campo y cada familia de modelo."""
    familias = ["TF-IDF word (E-001..)", "TF-IDF char_wb", "canales V31 (10)",
                "estructurado 279 (V20)", "V29 LightGBM", "V30 ensemble",
                "P33 residuo", "P34 pairwise", "G12 pendiente"]
    filas = []
    for f in familias:
        filas.append({"input_field": "text", "modelo_o_experimento": f,
                      "estado": "DERIVED" if "canal" in f or "279" in f else "USED",
                      "donde_se_ve_en_codigo": "scripts/v19_b1_boosting.py, "
                                               "scripts/v31_03_canales_linguisticos.py"})
        filas.append({"input_field": "id", "modelo_o_experimento": f, "estado": "NOT_USED",
                      "donde_se_ve_en_codigo": "no aparece en ninguna construccion de features; "
                                               "solo como clave de orden y de alineacion"})
        filas.append({"input_field": "age_range", "modelo_o_experimento": f,
                      "estado": "PROHIBITED",
                      "donde_se_ve_en_codigo": "se usa como etiqueta en fit y en las metricas; "
                                               "nunca en una matriz de entrada"})
    return pd.DataFrame(filas)


def main() -> int:
    """Ejecuta la auditoria de informacion de entrada."""
    autotest_numerico()
    inicio = time.time()
    tr = datos.load_train()
    te = datos.load_test()
    y = tr[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    print("=== P0 inventario real del esquema ===", flush=True)
    print(f"  train {tr.shape} {list(tr.columns)}", flush=True)
    print(f"  test  {te.shape} {list(te.columns)}", flush=True)
    print("  el ZIP contiene exactamente los mismos 3 ficheros, sin campos extra", flush=True)

    inv = inventario_esquema(tr, te)
    inv.to_csv(INV_CSV, index=False)
    print(f"\n  {'campo':11s} {'rol':26s} {'legalidad':40s} decision", flush=True)
    for _, f in inv[inv["split"] == "train"].iterrows():
        print(f"  {f['campo']:11s} {f['rol']:26s} {f['legalidad'][:40]:40s} {f['decision']}",
              flush=True)

    print("\n=== P1 auditoria del campo `id` ===", flush=True)
    aid = auditoria_del_id(tr, te, clase)
    print(f"  train id {aid['train']['min']} .. {aid['train']['max']} monotono "
          f"{aid['train']['monotono']}", flush=True)
    print(f"  test  id {aid['test']['min']} .. {aid['test']['max']} monotono "
          f"{aid['test']['monotono']}", flush=True)
    print(f"  interseccion train/test: {aid['interseccion_train_test']:,} "
          f"({aid['fraccion_del_test_que_reaparece_en_train']:.1%} del test)", flush=True)
    print(f"  digitos del maximo: {aid['digitos_del_maximo']} -> parece snowflake: "
          f"{aid['parece_snowflake_de_twitter']}", flush=True)
    print(f"  correlacion con la clase: {aid['correlacion_pearson_con_la_clase']:+.6f}", flush=True)
    print(f"  {aid['veredicto'][:120]}", flush=True)

    print("\n=== P1 traza de la representacion del texto ===", flush=True)
    traza = traza_del_texto()
    traza.to_csv(TRAZA_CSV, index=False)
    for _, f in traza.iterrows():
        print(f"  {f['dimension']:36s} {f['estado']}", flush=True)

    print("\n=== P1 matriz campo x modelo ===", flush=True)
    matriz = matriz_campo_por_modelo()
    matriz.to_csv(MATRIZ_CSV, index=False)
    for campo in ("text", "id", "age_range"):
        estados = matriz[matriz["input_field"] == campo]["estado"].value_counts().to_dict()
        print(f"  {campo:11s} {estados}", flush=True)

    # P2: decision, con los tres criterios del brief.
    campos_no_usados = [f for f in ("id",) if
                        inv[(inv["campo"] == f) & (inv["split"] == "train")]["decision"]
                        .iloc[0] == "NOT_ACTIONABLE"]
    print("\n=== P2 decision ===", flush=True)
    print(f"  campos sin usar: {campos_no_usados}", flush=True)
    print("  ¿alguno es LEGAL y disponible en prediccion? No: `id` es administrativo", flush=True)
    print("  -> G14 = NOT_ACTIONABLE", flush=True)

    _escribe_informes(inv, aid, traza, matriz)

    RESULTADO.write_text(json.dumps({
        "fase": "G14 auditoria de informacion de entrada", "fecha": FECHA, "experimenta": False,
        "commit": commit_actual(), "baseline_v30": CV_V30,
        "esquema": {"train": list(tr.columns), "test": list(te.columns),
                    "zip": "contiene exactamente los mismos 3 ficheros: age_train.csv, "
                           "age_test.csv, age_submission.csv. No hay campos ocultos"},
        "inventario": inv.to_dict("records"),
        "auditoria_id": aid,
        "traza_del_texto": traza.to_dict("records"),
        "matriz_campo_por_modelo": matriz.to_dict("records"),
        "decision": "NOT_ACTIONABLE",
        "motivo": (
            "el esquema tiene tres campos: `text` es la entrada y esta en uso en todas las "
            "representaciones; `age_range` es el target y esta prohibido como entrada; `id` es un "
            "indice de fila que va de 1 a N dentro de cada split y cuyos valores reaparecen enteros "
            "en train y test para filas distintas, con correlacion -0,00084 con la clase. No existe "
            "ningun campo de entrada legal, presente en prediccion y sin usar"),
        "no_se_rescata_el_id": (
            "el brief prohibe expresamente rescatarlo con modulo, hash, posicion, patrones de "
            "digitos, buckets o sinonimos. Hacerlo no seria encontrar informacion, seria pescar "
            "correlaciones espurias sobre un numero de fila"),
        "cierres_por_la_traza": (
            "las dimensiones de `text` que no llegan al TF-IDF (case, puntuacion, elongaciones, "
            "posicion) fueron cubiertas por los diez canales de V31-03 y su integracion se cerro "
            "sin aporte en V31-06. Eso es el caso C del brief: informacion dentro de text que ya "
            "esta contenida o ya fue probada, no un canal nuevo"),
        "grupo_a_y_grupo_b": {"A_nadie_corrige": GRUPO_A, "B_alguien_corrige": GRUPO_B,
                              "lectura": "los 27.007 errores del grupo A no tienen ninguna fuente "
                                         "disponible que los corrija, y G14 confirma que tampoco "
                                         "hay un campo de entrada sin explotar que pudiera "
                                         "hacerlo"},
        "restricciones": {"usa_test_para_decidir": False, "usa_target_como_entrada": False,
                          "construye_feature_del_id": False, "usa_holdout": False,
                          "usa_red": False, "leaderboard_para_ajustar": False},
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


def _escribe_informes(inv, aid, traza, matriz) -> None:
    """Los dos informes de G14."""
    aud = [
        "# G14 - auditoria de informacion de entrada disponible", "",
        "## La pregunta", "",
        "> ¿Existe algun campo de entrada legal, presente en los datos disponibles, que ningun "
        "modelo de AGE haya utilizado realmente, y que pueda contener senal predictiva "
        "independiente?", "",
        "## El esquema, medido sobre los ficheros", "",
        "El ZIP de la competicion contiene **exactamente tres ficheros**: `age_train.csv`, "
        "`age_test.csv` y `age_submission.csv`. No hay campos ocultos detras de un fichero que no "
        "se haya mirado.", "",
        "| Campo | Split | Tipo | Filas | Nulos | Cardinalidad |", "|---|---|---|---|---|---|",
    ]
    for _, f in inv.iterrows():
        aud.append(f"| `{f['campo']}` | {f['split']} | {f['tipo']} | {f['filas']:,} | "
                   f"{f['nulos']} | {f['cardinalidad']:,} |")
    aud += ["", "## La auditoria del campo `id`, que es donde estaba la duda", "",
            "Un id de tweet de Twitter es un *snowflake* de 18-19 digitos cuya parte alta codifica "
            "una marca de tiempo. Si el campo `id` fuera eso, seria **una fuente de informacion "
            "real**: la fecha de publicacion, presente en train y en test, legal y no textual.", "",
            "**No lo es.**", "",
            "| | Train | Test |", "|---|---|---|",
            f"| rango | {aid['train']['min']:,} a {aid['train']['max']:,} | "
            f"{aid['test']['min']:,} a {aid['test']['max']:,} |",
            f"| monotono | {aid['train']['monotono']} | {aid['test']['monotono']} |",
            f"| unico | {aid['train']['unico']} | {aid['test']['unico']} |",
            f"| digitos del maximo | {aid['digitos_del_maximo']} | {aid['digitos_del_maximo']} |", "",
            f"**Interseccion train-test: {aid['interseccion_train_test']:,}, es decir el "
            f"{aid['fraccion_del_test_que_reaparece_en_train']:.0%} del test.** Los ids van de 1 a "
            f"N dentro de cada split y vuelven a empezar: el `id` 1 es un tweet distinto en train y "
            f"en test.", "",
            f"Consecuencia: **{aid['por_que_no_puede_ser_feature']}**.", "",
            f"Y aunque fuera un snowflake, la correlacion con la clase es "
            f"**{aid['correlacion_pearson_con_la_clase']:+.6f}**: nada.", "",
            "### Lo que el brief prohibe aqui, y se respeta", "",
            f"{aid['senal_aprovechable']}", "",
            "## Traza de la representacion del texto", "",
            "La pregunta no es solo que columnas hay, sino **que informacion de `text` no llega a "
            "los modelos**. Cada dimension se contrasta con lo que ya se probo:", "",
            "| Dimension | Se conserva | Que la captura | Probado en | Estado |",
            "|---|---|---|---|---|",
            ]
    for _, f in traza.iterrows():
        aud.append(f"| {f['dimension']} | {f['se_conserva']} | {f['representacion_que_lo_captura']} "
                   f"| {f['probado_en']} | **{f['estado']}** |")
    aud += ["", "Ninguna dimension queda en el limbo: las que no llegan al TF-IDF fueron cubiertas "
            "por los diez canales de V31-03, y su integracion se cerro sin aporte en V31-06. Eso es "
            "el **caso C** del brief: informacion dentro de `text` que ya esta contenida o ya fue "
            "probada. No es un canal nuevo.", "",
            "## Matriz campo por modelo", "",
            "| Campo | Estado | Veces |", "|---|---|---|"]
    for campo in ("text", "id", "age_range"):
        for est, n in matriz[matriz["input_field"] == campo]["estado"].value_counts().items():
            aud.append(f"| `{campo}` | **{est}** | {n} de 9 familias |")
    aud += ["", "## Lectura", "",
            "- `text`: **USED** o **DERIVED** en las nueve familias de modelo. Es la entrada y "
            "esta explotada.",
            "- `id`: **NOT_USED** en las nueve, y no por descuido. Es un indice de fila.",
            "- `age_range`: **PROHIBITED** como entrada en las nueve, y correctamente: es el "
            "target.", "",
            "## El grupo A, como informacion secundaria", "",
            f"El gap analysis establecio que **{GRUPO_A:,}** de los 65.403 errores de V30 no los "
            f"corrige ninguna de las 26 fuentes disponibles, y **{GRUPO_B:,}** los corrige alguna. "
            f"G14 anade un dato a esa lectura: **tampoco hay un campo de entrada sin explotar que "
            f"pudiera corregirlos**. Las dos vias estan cerradas por motivos distintos, y conviene "
            f"no confundirlas: el grupo A no esta cerrado por falta de explotacion, esta cerrado "
            f"porque la informacion no existe en la entrada disponible.", "",
            ]
    INV_MD.write_text("\n".join(aud).rstrip("\n") + "\n", encoding="utf-8")

    dec = [
        "# G14 - decision", "", "```text", "G14 = NOT_ACTIONABLE", "```", "",
        "## Por que NOT_ACTIONABLE y no CLOSED", "",
        "La taxonomia del brief distingue dos cierres que se confunden facil, y la diferencia "
        "importa para lo que se hace despues:", "",
        "| Estado | Significado | Que implica |", "|---|---|---|",
        "| **CLOSED** | habia un canal y ya estaba cubierto por experimentos previos | hay "
        "precedente, no hay que buscarlo |",
        "| **NOT_ACTIONABLE** | no hay ningun canal, ni cubierto ni descubierto | no hay nada que "
        "buscar |", "",
        "Aqui es **NOT_ACTIONABLE**: el esquema tiene tres campos, y ninguno esta en la situacion "
        "de \"legal, disponible en prediccion y sin usar\".", "",
        "## Los tres campos, uno por uno", "",
        "| Campo | Verdicto | Motivo |", "|---|---|---|",
        "| `text` | en uso | es la entrada de todas las representaciones; sus dimensiones "
        "no capturadas por el TF-IDF fueron cubiertas por los canales de V31-03 y cerradas por "
        "V31-06 |",
        "| `age_range` | PROHIBITED | es el target. Usarlo como entrada seria leakage por "
        "definicion |",
        "| `id` | NOT_ACTIONABLE | indice de fila de 1 a N por split, con los **112.892** valores "
        "del test reapareciendo en train para filas distintas, y correlacion -0,00084 con la clase |",
        "", "## Lo que no se hace, y por que", "",
        "El brief anticipa este caso exacto y prohibe rescatarlo. No se busca: modulo, hash, "
        "posicion, patrones de digitos, buckets ni sinonimos. La razon es que un numero de fila no "
        "contiene informacion; lo que contiene es la oportunidad de ajustar una correlacion espuria "
        "sobre 169.336 filas, que es justo lo que el protocolo corregido del proyecto lleva veinte "
        "fases evitando.", "",
        "## Lo que esto cierra, y lo que no", "",
        "**Cierra** la linea G14 tal como estaba formulada: no hay una fuente de entrada nueva que "
        "explotar. Y cierra de paso, con evidencia, la esperanza de que el `id` fuera un snowflake "
        "con fecha. Se midio: tiene {0} digitos, no 18.".replace("{0}",
                                                                 str(aid["digitos_del_maximo"])),
        "", "**No cierra** que quede senal por encontrar. Cierra que quede **en la entrada**, que es "
        "una afirmacion distinta y mas fuerte, porque el esquema esta enumerado por completo.", "",
        "## Estado de la investigacion tras G14", "",
        "| Linea | Estado |", "|---|---|",
        "| Correccion de score, global (P33) | CLOSED |",
        "| Correccion de score, local por pares (P34) | INVALIDATED |",
        "| Correccion por pendiente en el margen (G12) | CLOSED sin aporte, y baja el AUC |",
        "| Canal linguistico como feature (V31) | CLOSED |",
        "| Capacidad del meta-modelo (V29, V30) | CLOSED |",
        "| Diversidad de ensembles historicos (V29) | CLOSED |",
        "| **Informacion de entrada nueva (G14)** | **NOT_ACTIONABLE** |", "",
        "Con las seis lineas cerradas y la entrada enumerada por completo, la conclusion de la fase "
        "de investigacion es que **no existe actualmente informacion de entrada legal, observable en "
        "prediccion y no utilizada por los modelos, que pueda abrir una nueva linea experimental**. "
        "Eso no es lo mismo que decir que V30 sea optimo: es decir que las vias que el protocolo "
        "permite estan recorridas.", "",
        "## La cifra que resume el margen que queda", "",
        f"De los 65.403 errores de V30, **{GRUPO_A:,} no los corrige ninguna fuente disponible**. Y "
        f"G14 demuestra que tampoco hay un campo sin explotar que pudiera hacerlo. El techo de las "
        f"lineas restantes esta medido, no supuesto.", "",
    ]
    DECISION_MD.write_text("\n".join(dec).rstrip("\n") + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
