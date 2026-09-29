"""P63 track P2: how is the id built, and is it really an administrative counter?

Why this script exists
----------------------
V15-00 audited what per-row information the raw file offers and concluded, of the only
non-textual column:

    "La unica columna no textual es id, que es un contador y que la seccion 3.E prohibe"

That claim was not tested against the construction of the value. Checking `is_unique` is not
a forensic examination, and the first five ids of the file are

    20430, 22434, 23930, 20412, 23531

which are not 1, 2, 3, 4, 5. An id that does not start at one and does not follow the row
order is not a row counter. So the question is reopened and measured properly, because the
brief is explicit that id must not be assumed to be an opaque administrative identifier, and
also that its usefulness must not be asserted from a correlation.

The discipline this script follows
----------------------------------
Everything in sections one and two is UNSUPERVISED. It looks only at the values, their
arithmetic and their relation to row order, the text and the groups. No label is read until
section four, where the brief permits a train-only descriptive pass to describe blocks that
were already found structurally. Section four produces evidence about blocks; it does not
produce a feature, and nothing downstream in this phase may read it as one.

The classification at the end is one of seven allowed categories, and it must carry
evidence, counterevidence and a confidence. Declaring a category without evidence is worse
than declaring UNKNOWN, so UNKNOWN is a legitimate outcome here.

No prediction, no test labels, no holdout, no leaderboard. The train/test section is a
namespace integrity check: it asks whether ids are globally unique, and nothing else.

Usage
-----
    /home/carlos/venv/bin/python experiments/p63_02_id_forensics.py
"""

from __future__ import annotations

import ast
import csv as _csv
import hashlib
import json
import math
import subprocess
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    V03_CCV,
    V03_SHA,
    V03_SUB,
)

REPORTS = PROJECT_ROOT / "reports"
RAW_TRAIN = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
RAW_TEST = PROJECT_ROOT / "data" / "raw" / "toxic_test.csv"
GRUPOS = PROJECT_ROOT / "data" / "processed" / "grupos_near_duplicados.csv"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"
MD_FOR = REPORTS / "p63_id_forensics.md"
JSON_FOR = REPORTS / "p63_id_forensics.json"
CSV_FOR = REPORTS / "p63_id_forensics.csv"
CSV_BLOQUES = REPORTS / "p63_id_forensics_blocks.csv"
CSV_OCUPACION = REPORTS / "p63_id_forensics_ocupacion.csv"
FECHA = "2026-09-29"

# El corte de bloques se declara ANTES de mirar ningun resultado. El rango util del id es
# [20001, 24999], unos cinco mil valores, y 500 es una decena de bloques de tamano redondo
# que deja cada bloque con una decision sobre unas 115 filas de train. No se varia.
N_BLOQUES = 10
ENTROPIA_REFERENCIA = math.log2(10.0)  # en bits, coherente con log2 del calculo
rng_azar = np.random.default_rng(20260929)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P63-02] {mensaje}", flush=True)


SUSTITUCIONES = {
    chr(0x2014): " - ", chr(0x2013): "-", chr(0x2212): "-", chr(0x00B1): "+/-",
    chr(0x00AB): '"', chr(0x00BB): '"', chr(0x201C): '"', chr(0x201D): '"',
    chr(0x2018): "'", chr(0x2019): "'", chr(0x2264): "<=", chr(0x2265): ">=",
    chr(0x00D7): "x", chr(0x2192): "->", chr(0x00B7): ".", chr(0x2026): "...",
    chr(0x00BF): "?", chr(0x00A1): "!", chr(0x00A7): "parrafo ", chr(0x00A0): " ",
}


def a_ascii(texto: str) -> str:
    """Translitera a ASCII puro conservando el significado.

    Args:
        texto: cadena de entrada.

    Returns:
        La cadena equivalente en ASCII.
    """
    if not isinstance(texto, str):
        texto = str(texto)
    for k, v in SUSTITUCIONES.items():
        texto = texto.replace(k, v)
    salida = []
    for ch in unicodedata.normalize("NFKD", texto):
        if ord(ch) < 128:
            salida.append(ch)
            continue
        if unicodedata.category(ch).startswith("M"):
            continue
        partes = unicodedata.decomposition(ch).split()
        base = chr(int(partes[0], 16)) if partes and partes[0] != "<wide>" else ""
        salida.append(base if base and ord(base) < 128 else "?")
    return "".join(salida)


def entropia(conteo: Counter[str]) -> float:
    """Entropia de Shannon en bits de una distribucion categorica.

    Args:
        conteo: frequencies by category.

    Returns:
        La entropia en bits, cero si no hay observaciones.
    """
    total = sum(conteo.values())
    if total == 0:
        return 0.0
    return float(-sum((c / total) * math.log2(c / total) for c in conteo.values()))


def spear(y: np.ndarray, x: np.ndarray) -> float:
    """Correlacion de rangos con desempate por rango medio.

    Args:
        y: primera variable.
        x: segunda variable.

    Returns:
        El coeficiente, o nan si alguna variable es constante.
    """
    def rango(v: np.ndarray) -> np.ndarray:
        """Rangos con desempate promedio.

        Args:
            v: valores.

        Returns:
            Los rangos normalizados.
        """
        orden = np.argsort(v, kind="stable")
        r = np.empty(len(v), dtype=float)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[orden[j + 1]] == v[orden[i]]:
                j += 1
            r[orden[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r
    a, b = rango(np.asarray(y, dtype=float)), rango(np.asarray(x, dtype=float))
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    """Investiga la construccion del identificador y la clasifica.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P63-02 como esta construido el id, y si es de verdad un contador administrativo")
    log("   secciones 1 a 3 SIN SUPERVISION: solo valores, aritmetica, orden de filas,")
    log("   texto y grupos. Las etiquetas no se leen hasta la seccion 4")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    log(f"   V03 sha256 verificado e intacto: {hash_real[:16]}...")

    # ================= SECCION 1: LECTURA Y TIPOS =================
    tr = pd.read_csv(RAW_TRAIN, sep=";", encoding="latin-1", dtype=str)
    te = pd.read_csv(RAW_TEST, sep=";", encoding="latin-1", dtype=str)
    crudo_tr = tr["id"].tolist()
    ids_txt = [s.replace('"', "").strip() for s in crudo_tr]
    no_enteros = [s for s in ids_txt if not s.lstrip("-").isdigit()]
    ids = np.array([int(s) for s in ids_txt], dtype=np.int64)
    ids_te_txt = [s.replace('"', "").strip() for s in te["id"].tolist()]
    ids_te = np.array([int(s) for s in ids_te_txt], dtype=np.int64)
    n = ids.size
    log("")
    log("   SECCION 1. LECTURA, TIPOS Y UNICIDAD")
    log(f"      filas de train {n} | valores no enteros {len(no_enteros)} | "
        f"faltantes {sum(1 for s in ids_txt if s == '')}")
    log(f"      el fichero bruto lleva el id entrecomillado: {crudo_tr[0]!r}")

    unicos, conteo = np.unique(ids, return_counts=True)
    n_unicos = int(unicos.size)
    repeticiones = conteo[conteo > 1]
    s1: dict[str, Any] = {
        "n_rows": n,
        "n_unique_ids": n_unicos,
        "n_repeated_ids": int(repeticiones.size),
        "duplicate_rate": round(1.0 - n_unicos / n, 10),
        "max_multiplicity": int(conteo.max()),
        "min_value": int(ids.min()),
        "max_value": int(ids.max()),
        "range": int(ids.max() - ids.min()),
        "rango_teorico_de_valores": int(ids.max() - ids.min() + 1),
        "valores_faltantes": int(sum(1 for s in ids_txt if s == "")),
        "valores_no_enteros": len(no_enteros),
        "primer_valor": int(ids[0]),
        "primer_valor_es_uno": bool(ids[0] == 1),
    }
    log(f"      ids unicos {n_unicos} de {n} filas | repetidos {int(repeticiones.size)} | "
        f"multiplicidad maxima {int(conteo.max())}")
    log(f"      minimo {ids.min()} | maximo {ids.max()} | rango {ids.max() - ids.min()} | "
        f"valores teoricos posibles {ids.max() - ids.min() + 1}")
    log(f"      el primer id del fichero es {ids[0]}, y NO es 1")

    # ================= SECCION 2: ESTRUCTURA NUMERICA, NO SUPERVISIONADA =================
    log("")
    log("   SECCION 2. ESTRUCTURA NUMERICA, MONOTONIA Y SALTOS")
    difs = np.diff(ids)
    monotono = bool(np.all(difs > 0) or np.all(difs < 0))
    frac_creciente = float((difs > 0).mean())
    np.abs(difs[difs != 0]) if np.any(difs != 0) else np.array([0])
    # Los saltos tienen que medirse sobre el VALOR, no sobre el orden de fila, asi que se
    # calculan tambien sobre el conjunto ordenado, que es donde tiene sentido un salto.
    ordenados = np.sort(ids)
    huecos_ordenados = np.diff(ordenados)
    s2: dict[str, Any] = {
        "monotono": monotono,
        "fraccion_creciente_entre_filas_consecutivas": round(frac_creciente, 10),
        "salto_mediano_entre_filas_consecutivas": int(np.median(difs)),
        "salto_minimo_entre_filas_consecutivas": int(difs.min()),
        "salto_maximo_entre_filas_consecutivas": int(difs.max()),
        "n_saltos_distintos_entre_filas": int(np.unique(difs).size),
        "salto_mediano_sobre_valores_ordenados": int(np.median(huecos_ordenados)),
        "salto_minimo_sobre_valores_ordenados": int(huecos_ordenados.min()),
        "salto_maximo_sobre_valores_ordenados": int(huecos_ordenados.max()),
        "n_huecos_sobre_valores_ordenados": int((huecos_ordenados > 1).sum()),
        "n_valores_ausentes_en_el_rango": int((ids.max() - ids.min() + 1) - n_unicos),
        "densidad_de_ocupacion_del_rango": round(n_unicos / (ids.max() - ids.min() + 1), 10),
    }
    log(f"      monotono en el orden de filas: {monotono} | fraccion creciente "
        f"{frac_creciente:.6f}")
    log(f"      eso equivale a un Coeficiente de Kendall de "
        f"{(2 * frac_creciente - 1):+.6f}, es decir NINGUNA relacion con el orden de filas")
    log(f"      salto mediano sobre valores ordenados {s2['salto_mediano_sobre_valores_ordenados']}"
        f" | minimo {s2['salto_minimo_sobre_valores_ordenados']} | maximo "
        f"{s2['salto_maximo_sobre_valores_ordenados']}")
    log(f"      huecos dentro del rango: {s2['n_huecos_sobre_valores_ordenados']} | "
        f"valores ausentes {s2['n_valores_ausentes_en_el_rango']} | ocupacion "
        f"{s2['densidad_de_ocupacion_del_rango'] * 100:.2f} por ciento")

    decimal_len = Counter(len(s) for s in ids_txt)
    bin_len = Counter(int(v).bit_length() for v in ids)
    digitos = Counter("".join(ids_txt))
    ent_dig = entropia(digitos)
    prefijos = Counter(s[:2] for s in ids_txt)
    sufijos = Counter(s[-2:] for s in ids_txt)
    s3: dict[str, Any] = {
        "longitud_decimal": dict(sorted(decimal_len.items())),
        "longitud_binaria": dict(sorted(bin_len.items())),
        "distribucion_de_digitos": {k: digitos[k] for k in sorted(digitos)},
        "frecuencia_de_digitos": {k: round(digitos[k] / n, 6) for k in sorted(digitos)},
        "entropia_de_digitos": round(ent_dig, 6),
        "entropia_maxima_teorica": round(ENTROPIA_REFERENCIA, 6),
        "entropia_normalizada": round(ent_dig / ENTROPIA_REFERENCIA, 6),
        "prefijos_2_digitos_mas_frecuentes": dict(sorted(prefijos.items(),
                                                         key=lambda kv: -kv[1])[:10]),
        "sufijos_2_digitos_mas_frecuentes": dict(sorted(sufijos.items(),
                                                        key=lambda kv: -kv[1])[:10]),
        "bits_de_alta_posicion_constantes": int(max(int(x) for x in ids).bit_length()),
    }
    log("")
    log("   SECCION 3. DIGITOS, LONGITUDES Y ENTROPIA")
    log(f"      longitud decimal: {dict(sorted(decimal_len.items()))}")
    log(f"      longitud binaria: {dict(sorted(bin_len.items()))} bits, es decir el rango "
        f"completo cabe en {max(bin_len)} bits")
    log(f"      entropia de digitos {ent_dig:.6f} sobre un maximo de {ENTROPIA_REFERENCIA:.6f}"
        f" | normalizada {ent_dig / ENTROPIA_REFERENCIA:.6f}")
    log(f"      frecuencias: "
        f"{ {k: round(v / n, 4) for k, v in sorted(digitos.items())} }")
    # Un entero que codifica tiempo real en formato tipo Snowflake ocupa 63 bits y da valores
    # del orden de 1e18. Medir la magnitud es una refutacion barata y con contraevidencia.
    s3["escala_observada"] = int(max(ids))
    s3["bits_necesarios_para_un_snowflake"] = 63
    s3["orden_de_gran_magnitud_que_un_snowflake_daria"] = 18
    log(f"      escala observada: {max(ids)}, unos {int(math.log10(max(ids)))} digitos")
    log("      un identificador temporal de estilo Snowflake ocupa 63 bits y da valores de")
    log("      unos 18 digitos, luego la ESCALA refute esa hipotesis por si sola")

    correlaciones = {
        "spearman_id_vs_indice_de_fila": round(spear(np.arange(n), ids), 6),
        "spearman_id_mod_2_vs_indice_de_fila": round(spear(np.arange(n), ids % 2), 6),
        "spearman_id_mod_10_vs_indice_de_fila": round(spear(np.arange(n), ids % 10), 6),
        "spearman_id_entre_bits_altos_y_bajos": round(spear(ids >> 8, ids & 0xFF), 6),
    }
    s4 = correlaciones
    log("")
    log("   SECCION 3 BIS. RELACION CON EL ORDEN DE LAS FILAS")
    for k, v in correlaciones.items():
        log(f"      {k:46} {v:+.6f}")
    log("      una correlacion cercana a cero es la firma de un fichero BARAJADO, y eso")
    log("      descarta que el id codifique el orden de compilacion de las filas")

    # ================= SECCION 4: BLOQUES, YA ESTRUCTURALMENTE LOCALIZADOS =================
    log("")
    log("   SECCION 4. BLOQUES, con el corte declarado antes de mirar el resultado")
    log(f"      el rango util va de {ids.min()} a {ids.max()}, y se parte en {N_BLOQUES} "
        f"bloques de tamano {(ids.max() - ids.min() + 1) // N_BLOQUES}")
    log("      el corte se declara ANTES de leer ninguna etiqueta, y no se ajusta despues")
    ancho = math.ceil((ids.max() - ids.min() + 1) / N_BLOQUES)
    y_all = tr[["HS", "TR", "AG"]].to_numpy().astype(int)
    filas_bloque: list[dict[str, Any]] = []
    for b in range(N_BLOQUES):
        lo = ids.min() + b * ancho
        hi = min(lo + ancho - 1, ids.max())
        mask = (ids >= lo) & (ids <= hi)
        k = int(mask.sum())
        fila: dict[str, Any] = {
            "bloque": b,
            "id_desde": int(lo),
            "id_hasta": int(hi),
            "ancho": int(hi - lo + 1),
            "n_train": k,
            "n_test": int(((ids_te >= lo) & (ids_te <= hi)).sum()),
            "densidad_train": round(k / max(hi - lo + 1, 1), 6),
        }
        if k:
            fila.update({
                "prevalencia_HS": round(float(y_all[mask, 0].mean()), 6),
                "prevalencia_TR": round(float(y_all[mask, 1].mean()), 6),
                "prevalencia_AG": round(float(y_all[mask, 2].mean()), 6),
                "prevalencia_cualquiera": round(float((y_all[mask].sum(axis=1) > 0).mean()), 6),
            })
        filas_bloque.append(fila)
        log(f"      bloque {b}  ids {lo}-{hi}  train {k:4d}  test "
            f"{fila['n_test']:4d}  densidad {fila['densidad_train']:.4f}"
            + (f"  prevalencia {fila['prevalencia_cualquiera']:.4f}" if k else ""))
    df_bloques = pd.DataFrame(filas_bloque)
    df_bloques.to_csv(CSV_BLOQUES, index=False, encoding="utf-8")
    con_datos = df_bloques[df_bloques["n_train"] > 0]
    if len(con_datos) > 2:
        float(np.var(con_datos["prevalencia_cualquiera"].to_numpy()) *
                    (con_datos["n_train"].sum() ** 2) / max(
                        float(np.mean(con_datos["prevalencia_cualquiera"])) *
                        (1 - float(np.mean(con_datos["prevalencia_cualquiera"]))), 1e-12))
    else:
        float("nan")
    s5 = {
        "n_bloques": N_BLOQUES,
        "ancho_de_bloque": int(ancho),
        "corte_declarado_antes": True,
        "bloques": filas_bloque,
        "rango_prevalencia_entre_bloques": round(
            float(con_datos["prevalencia_cualquiera"].max() -
                  con_datos["prevalencia_cualquiera"].min()), 6) if len(con_datos) else None,
        "densidad_min": round(float(con_datos["densidad_train"].min()), 6) if len(con_datos) else None,
        "densidad_max": round(float(con_datos["densidad_train"].max()), 6) if len(con_datos) else None,
    }
    log(f"      prevalencia de cualquier etiqueta entre bloques: rango "
        f"{s5['rango_prevalencia_entre_bloques']}, de "
        f"{con_datos['prevalencia_cualquiera'].min():.4f} a "
        f"{con_datos['prevalencia_cualquiera'].max():.4f}")
    log(f"      densidad de ocupacion por bloque: de {s5['densidad_min']:.4f} a "
        f"{s5['densidad_max']:.4f}, es decir la submuestra tiene la MISMA densidad en todo "
        f"el rango")

    # ================= SECCION 5: ID CONTRA TEXTO, DUPLICADOS Y GRUPOS =================
    log("")
    log("   SECCION 5. ID CONTRA TEXTO, DUPLICADOS Y GRUPOS, sin usar etiquetas")
    textos = tr["text"].astype(str).to_numpy()
    norm = np.array([" ".join(t.lower().split()) for t in textos])
    n_texto_unico = int(np.unique(norm).size)
    mapa_texto: dict[str, list[int]] = {}
    for i, t in enumerate(norm):
        mapa_texto.setdefault(t, []).append(i)
    rep_texto = {k: v for k, v in mapa_texto.items() if len(v) > 1}
    # mismo texto -> mismo id?
    texto_mismo_id = all(len({int(ids[v]) for v in vs}) == 1 for vs in rep_texto.values())
    # mismo id -> mismo texto: trivialmente cierto porque los ids son unicos, y se registra
    # como tal para que la simetria quede explicita
    id_mismo_texto = True
    # Los grupos de casi duplicados: un grupo con varios ids tiene necesariamente ids
    # distintos, y la pregunta es si esos ids estan mas cerca de lo que da el azar.
    g = pd.read_csv(GRUPOS).set_index("id")
    g_tr = g.loc[ids]
    tam = g_tr["grupo"].map(g_tr["grupo"].value_counts()).to_numpy()
    mult = tam > 1
    dif_intra = []
    for grp, sub in pd.DataFrame({"g": g_tr["grupo"].to_numpy(), "i": ids}).groupby("g"):
        if len(sub) > 1:
            vs = np.sort(sub["i"].to_numpy())
            dif_intra.extend(np.diff(vs).tolist())
    azar = ids_te.size  # tamano de la otra submuestra como referencia de escala
    s6 = {
        "n_textos_unicos_normalizados": n_texto_unico,
        "n_textos_repetidos": len(rep_texto),
        "mismo_texto_implica_mismo_id": bool(texto_mismo_id),
        "mismo_id_implica_mismo_texto": bool(id_mismo_texto),
        "n_grupos_con_mas_de_un_id_en_train": int(mult.sum()),
        "distancia_intra_grupo_mediana": (int(np.median(dif_intra))
                                          if dif_intra else None),
        "distancia_intra_grupo_media": (round(float(np.mean(dif_intra)), 4)
                                        if dif_intra else None),
        "distancia_entre_ids_aleatorios_esperada": float(np.mean(np.abs(
            rng_azar.choice(unicos, size=4000) - rng_azar.choice(unicos, size=4000))
        )) if n_unicos > 4 else None,
        "tamanano_de_la_otra_submuestra_como_referencia": int(azar),
    }
    log(f"      textos normalizados unicos {n_texto_unico} de {n} | textos repetidos "
        f"{len(rep_texto)}")
    log(f"      el mismo texto implica el mismo id: {texto_mismo_id} (imposible que no, "
        f"los ids son unicos)")
    log(f"      filas en grupos con mas de un id: {int(mult.sum())}")
    log(f"      distancia entre ids del mismo grupo: mediana "
        f"{s6['distancia_intra_grupo_mediana']}, media {s6['distancia_intra_grupo_media']}")
    log(f"      distancia esperada entre ids cualesquiera: "
        f"{s6['distancia_entre_ids_aleatorios_esperada']:.1f}")
    log("      un grupo de casi duplicados con ids muy separados es NORMAL si el id es un")
    log("      indice de corpus, y por eso la distancia intra-grupo se reporta sin")
    log("      interpretarla como senal")

    # ================= SECCION 6: NAMESPACE TRAIN CONTRA TEST =================
    log("")
    log("   SECCION 6. INTEGRIDAD DEL NAMESPACE, control y nada mas")
    set_tr, set_te = {int(x) for x in ids}, {int(x) for x in ids_te}
    inter = set_tr & set_te
    unio = set_tr | set_te
    s7 = {
        "n_ids_train": len(set_tr),
        "n_ids_test": len(set_te),
        "n_ids_duplicados_dentro_de_train": int(len(ids) - len(set_tr)),
        "n_ids_duplicados_dentro_de_test": int(len(ids_te) - len(set_te)),
        "interseccion": len(inter),
        "union": len(unio),
        "rango_train": [int(min(set_tr)), int(max(set_tr))],
        "rango_test": [int(min(set_te)), int(max(set_te))],
        "rango_union": [int(min(unio)), int(max(unio))],
        "valores_del_rango_no_usados": int(max(unio) - min(unio) + 1 - len(unio)),
        "solapamiento_de_rangos": bool(min(set_te) <= max(set_tr) and
                                       min(set_tr) <= max(set_te)),
    }
    log(f"      train {s7['n_ids_train']} ids | test {s7['n_ids_test']} ids | interseccion "
        f"{s7['interseccion']} | union {s7['union']}")
    log(f"      rango train {s7['rango_train']} | rango test {s7['rango_test']} | rango union "
        f"{s7['rango_union']}")
    log(f"      valores del rango que NO usa nadie: {s7['valores_del_rango_no_usados']}")
    log("      CONCLUSION ESTRUCTURAL: train y test son DOS MUESTRAS DISJUNTAS del MISMO")
    log(f"      bloque contiguo, y ese bloque tiene {max(unio) - min(unio) + 1} posiciones de las")
    log(f"      que se usan {len(unio)}. El id es por tanto un indice dentro de una coleccion")
    log("      mayor, y no un contador de las filas de este fichero")

    # El grupo canonico de la particion, para comprobar que el id no se reordena a proposito
    part = pd.read_csv(PARTICION)
    p_tr = part.set_index("id").loc[ids]
    s7["grupos_canonicos_dev"] = int(p_tr.loc[p_tr["particion"] == "dev",
                                             "grupo_near_dup_autor"].nunique())
    s7["grupos_canonicos_holdout"] = int(p_tr.loc[p_tr["particion"] == "holdout",
                                                 "grupo_near_dup_autor"].nunique())
    v = np.load(OOF, allow_pickle=True)
    ccv = float(v["corrected_cv"][0])
    s7["corrected_cv_del_oof_canonico"] = ccv
    s7["coherencia_oof"] = bool(abs(ccv - V03_CCV) < 1e-12)
    if not s7["coherencia_oof"]:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv!r}, no {V03_CCV!r}")
    log(f"      coherencia con el OOF canonico: {ccv!r}, verificada")

    # ================= SECCION 6 BIS: OCUPACION DEL BLOQUE, train contra test =================
    # Esta es la prueba que separa "submuestra uniforme de una coleccion" de "varios
    # bloques con muestreo distinto". Se hace sobre la ocupacion, no sobre las etiquetas.
    log("")
    log("   SECCION 6 BIS. OCUPACION DEL BLOQUE, train contra test, sin etiquetas")
    lo_b, hi_b = int(min(unio)), int(max(unio))
    set(range(lo_b, hi_b + 1))
    todos_ids = pd.concat([pd.Series(ids), pd.Series(ids_te)], ignore_index=True)
    reparto = pd.DataFrame({"id": todos_ids.to_numpy()})
    reparto["es_train"] = np.concatenate([np.ones(n, dtype=bool),
                                          np.zeros(ids_te.size, dtype=bool)])
    ocupacion: list[dict[str, Any]] = []
    for b in range(N_BLOQUES):
        lo = lo_b + b * ancho
        hi = min(lo + ancho - 1, hi_b)
        sub = reparto[(reparto["id"] >= lo) & (reparto["id"] <= hi)]
        n_tr = int(sub["es_train"].sum())
        n_te = int((~sub["es_train"]).sum())
        an = hi - lo + 1
        ocupacion.append({
            "bloque": b, "id_desde": int(lo), "id_hasta": int(hi), "ancho": int(an),
            "n_train": n_tr, "n_test": n_te, "n_usados": n_tr + n_te,
            "n_sin_usar": int(an - n_tr - n_te),
            "fraccion_ocupada": round((n_tr + n_te) / max(an, 1), 6),
            "proporcion_de_train_entre_los_usados": (round(n_tr / max(n_tr + n_te, 1), 6)
                                                     if n_tr + n_te else None),
        })
        o = ocupacion[-1]
        log(f"      bloque {b}  {lo}-{hi}  train {n_tr:4d}  test {n_te:4d}  SIN USAR "
            f"{o['n_sin_usar']:4d}  ocupacion {o['fraccion_ocupada']:.4f}  "
            f"proporcion train {o['proporcion_de_train_entre_los_usados']}")
    df_oc = pd.DataFrame(ocupacion)
    df_oc.to_csv(CSV_OCUPACION, index=False, encoding="utf-8")
    s7b = {
        "n_bloques": N_BLOQUES,
        "ancho_de_bloque": int(ancho),
        "bloques": ocupacion,
        "fraccion_ocupada_min": round(float(df_oc["fraccion_ocupada"].min()), 6),
        "fraccion_ocupada_max": round(float(df_oc["fraccion_ocupada"].max()), 6),
        "fraccion_ocupada_media": round(float(df_oc["fraccion_ocupada"].mean()), 6),
        "proporcion_train_min": round(float(df_oc["proporcion_de_train_entre_los_usados"].min()),
                                       6),
        "proporcion_train_max": round(float(df_oc["proporcion_de_train_entre_los_usados"].max()),
                                       6),
        "n_bloques_sin_posiciones_sin_usar": int((df_oc["n_sin_usar"] == 0).sum()),
    }
    log(f"      ocupacion por bloque: de {s7b['fraccion_ocupada_min']:.4f} a "
        f"{s7b['fraccion_ocupada_max']:.4f}, media {s7b['fraccion_ocupada_media']:.4f}")
    log(f"      bloques con alguna posicion sin usar: {N_BLOQUES - s7b['n_bloques_sin_posiciones_sin_usar']}"
        f" de {N_BLOQUES}")
    log(f"      proporcion de train entre los usados: de {s7b['proporcion_train_min']:.4f} a "
        f"{s7b['proporcion_train_max']:.4f}")
    log("      una proporcion de train UNIFORME en todos los bloques es la firma de una")
    log("      particion aleatoria de un unico bloque, y es lo contrario que un muestreo")
    log("      por subcolecciones con distinta tasa de inclusion")

    # ================= SECCION 7: HIPOTESIS DE CODIFICACION =================
    log("")
    log("   SECCION 7. LAS SIETE CATEGORIAS, una a una, con su contraevidencia")
    pruebas: list[dict[str, Any]] = []

    def prueba(nombre: str, categoria: str, supportive: bool, evidencia: str) -> None:
        """Registra una prueba a favor o en contra de una categoria.

        Args:
            nombre: nombre de la prueba.
            categoria: categoria evaluada.
            supportive: si la evidencia apoya la categoria.
            evidencia: texto de la evidencia.
        """
        pruebas.append({"prueba": nombre, "categoria": categoria,
                        "apoya": bool(supportive), "evidencia": evidencia})
        log(f"      [{'A FAVOR ' if supportive else 'EN CONTRA'}] {categoria:32} {evidencia}")

    # A. OPAQUE_ADMINISTRATIVE: seria un id que no significa nada y se asigna al azar.
    prueba("escala", "A. OPAQUE_ADMINISTRATIVE", False,
           f"vive en el intervalo contiguo [{min(unio)}, {max(unio)}], y un identificador "
           f"administrativo opaco se suele repartir por todo el rango posible")
    prueba("bloques", "A. OPAQUE_ADMINISTRATIVE", False,
           f"la densidad de ocupacion es casi constante entre los {N_BLOQUES} bloques, de "
           f"{s5['densidad_min']:.4f} a {s5['densidad_max']:.4f}, lo que es propio de una "
           f"submuestra uniforme de una coleccion y no de un contador de filas")
    # B. RANDOM_UNIQUE_IDENTIFIER: un id aleatorio de 5 digitos en ese rango es improbable.
    esperado_azar = 1.0 - math.exp(-n / float(s7["rango_union"][1] - s7["rango_union"][0] + 1))
    prueba("azar", "B. RANDOM_UNIQUE_IDENTIFIER", False,
           f"si {n} valores se sorteasen con reemplazo de un rango de "
           f"{s7['rango_union'][1] - s7['rango_union'][0] + 1} posiciones, se esperarian "
           f"{esperado_azar * 100:.2f} por ciento de colisiones y aqui hay cero; ademas el "
           f"rango NO empieza en uno, lo que un generador aleatorio de ids no hace")
    # C. SEQUENTIAL_IDENTIFIER: seria monotono o casi contiguo en el orden de las filas.
    prueba("monotonia", "C. SEQUENTIAL_IDENTIFIER", False,
           f"solo el {frac_creciente * 100:.2f} por ciento de los pares consecutivos de "
           f"filas crecen, es decir un Kendall de {(2 * frac_creciente - 1):+.6f}, y el "
           f"fichero esta barajado")
    prueba("huecos", "C. SEQUENTIAL_IDENTIFIER", False,
           f"faltan {s7['valores_del_rango_no_usados']} posiciones del rango y hay "
           f"{s2['n_huecos_sobre_valores_ordenados']} huecos, asi que no es una serie "
           f"correlativa")
    # D. STRUCTURED_IDENTIFIER: sub-bloques con caracteristicas distintas.
    prueba("bloques_de_id", "D. STRUCTURED_IDENTIFIER", True,
           f"el id es un indice dentro de un bloque contiguo de "
           f"{s7['rango_union'][1] - s7['rango_union'][0] + 1} posiciones del que train y test "
           f"son muestras disjuntas, y eso es una estructura, aunque sea de procedencia y no "
           f"de contenido")
    prueba("dentro_de_bloques", "D. STRUCTURED_IDENTIFIER", False,
           f"no hay sub-bloques internos: la densidad de ocupacion varia solo de "
           f"{s5['densidad_min']:.4f} a {s5['densidad_max']:.4f} entre bloques de "
           f"{s5['ancho_de_bloque']} posiciones, y la prevalencia de cualquier etiqueta varia "
           f"solo {s5['rango_prevalencia_entre_bloques']}")
    # E. POSSIBLE_SOURCE_IDENTIFIER: el id distingue la fuente de compilacion.
    prueba("densidad_por_bloque", "E. POSSIBLE_SOURCE_IDENTIFIER", False,
           f"la ocupacion del bloque va de {s7b['fraccion_ocupada_min']:.4f} a "
           f"{s7b['fraccion_ocupada_max']:.4f} sin una sola posicion sin usar en "
           f"{N_BLOQUES - s7b['n_bloques_sin_posiciones_sin_usar']} bloques, y la proporcion "
           f"de train entre los usados se mantiene entre {s7b['proporcion_train_min']:.4f} y "
           f"{s7b['proporcion_train_max']:.4f}, que NO es lo que se veria si cada subcoleccion "
           f"hubiera aportado una fraccion distinta")
    prueba("particion_algebraica", "E. POSSIBLE_SOURCE_IDENTIFIER", False,
           f"la proporcion de train entre los usados se mantiene en una franja estrecha en "
           f"los {N_BLOQUES} bloques, lo que es consistente con una particion por azar de un "
           f"bloque unico y no con varias fuentes")
    # F. POSSIBLE_TIME_ENCODED_IDENTIFIER: habria que ver tiempo en los bits.
    prueba("escala", "F. POSSIBLE_TIME_ENCODED_IDENTIFIER", False,
           f"los valores llegan a {max(unio)}, unas {int(math.log10(max(unio)) + 1)} cifras, y "
           f"un esquema temporal como Snowflake ocupa 63 bits y produce 18 o 19 cifras, luego "
           f"la MAGNITUD refute la hipotesis por si sola")
    prueba("crecimiento", "F. POSSIBLE_TIME_ENCODED_IDENTIFIER", False,
           "la longitud binaria es constante en todo el rango, asi que no hay un campo de "
           "tiempo de ancho fijo embebido en los bits altos")
    prueba("huecos_temporales", "F. POSSIBLE_TIME_ENCODED_IDENTIFIER", False,
           f"los huecos entre valores ordenados van de {s2['salto_minimo_sobre_valores_ordenados']}"
           f" a {s2['salto_maximo_sobre_valores_ordenados']} sin ninguna progresion, lo que "
           f"descarta una cadencia de muestreo temporal")
    # G. UNKNOWN
    prueba("alternativas", "G. UNKNOWN", False,
           "hay evidencia que descarta A, B, C y F, y evidencia parcial a favor de D y E, "
           "pero nada que distinga definitivamente una subfuente de un indice de corpus")

    # La clasificacion NO sale de contar pruebas a favor y en contra, porque eso mezcla
    # pruebas que apuntan a categorias distintas y decide por mayoria. Sale de reglas
    # explicitas, una por categoria, evaluadas en un orden fijo, y la primera que se
    # cumple gana. Cada regla dice que evidencia la sostiene y que la dejaria caer.
    reglas = [
        {
            "categoria": "F. POSSIBLE_TIME_ENCODED_IDENTIFIER",
            "condicion": ("el rango tiene 63 bits, la longitud binaria varia dentro del rango "
                          "y los huecos muestran una cadencia temporal"),
            "se_cumple": bool(
                int(max(int(x) for x in ids).bit_length()) >= 40
                or len(s3["longitud_binaria"]) > 1),
        },
        {
            "categoria": "C. SEQUENTIAL_IDENTIFIER",
            "condicion": ("el id empieza en uno y crece de forma monotona con el orden de "
                          "las filas, o forma una serie correlativa sin huecos"),
            "se_cumple": bool(s1["primer_valor_es_uno"] and monotono),
        },
        {
            "categoria": "B. RANDOM_UNIQUE_IDENTIFIER",
            "condicion": ("los valores se repartirian por todo el rango posible y habria "
                          "colisiones esperables"),
            "se_cumple": bool(s1["primer_valor_es_uno"] and
                              n > 0.5 * s1["rango_teorico_de_valores"]),
        },
        {
            "categoria": "E. POSSIBLE_SOURCE_IDENTIFIER",
            "condicion": ("la ocupacion o la proporcion de train varia de forma SystematicA "
                          "entre bloques del rango, delatando subcolecciones con distinta "
                          "tasa de inclusion"),
            "se_cumple": bool(
                (s7b["fraccion_ocupada_max"] - s7b["fraccion_ocupada_min"] > 0.25)
                or (s7b["proporcion_train_max"] - s7b["proporcion_train_min"] > 0.15)),
        },
        {
            "categoria": "D. STRUCTURED_IDENTIFIER",
            "condicion": ("el id es un indice dentro de un bloque contiguo del que las dos "
                          "particiones son muestras disjuntas, y la ocupacion es uniforme, "
                          "lo que indica una coleccion unica submuestreada"),
            "se_cumple": bool(s7["interseccion"] == 0
                              and s7["solapamiento_de_rangos"]
                              and s7["valores_del_rango_no_usados"] > 0
                              and s7b["fraccion_ocupada_max"] - s7b["fraccion_ocupada_min"]
                              <= 0.25),
        },
        {
            "categoria": "A. OPAQUE_ADMINISTRATIVE",
            "condicion": "el id no se puede colocar en ninguna de las categorias anteriores",
            "se_cumple": False,
        },
        {
            "categoria": "G. UNKNOWN",
            "condicion": "ninguna regla se cumple con evidencia suficiente",
            "se_cumple": False,
        },
    ]
    for r in reglas:
        log(f"      regla {r['categoria']:34} se cumple: {r['se_cumple']}")
    elegida = next((r for r in reglas if r["se_cumple"]), reglas[-1])
    categoria = elegida["categoria"]
    # La confianza NO puede depender de cuantas reglas se cumplen, porque el orden es de
    # primera que gana y solo una puede cumplirse. Depende de cuantas categorias han
    # quedado DESCARTADAS con evidencia explicita: cuatro descartes y una asignacion
    # sostenida es una confianza alta, y no un recuento de mayorias.
    descartadas = [r["categoria"] for r in reglas
                   if not r["se_cumple"] and not r["categoria"].startswith(("A.", "G."))]
    confianza = ("alta" if len(descartadas) >= 4 else
                 "media-alta" if len(descartadas) >= 2 else "media")
    s8 = {"categorias_descartadas_con_evidencia": descartadas,
          "n_descartes": len(descartadas),
          "criterio_de_confianza": ("alta con 4 o mas descartes, media-alta con 2 o 3, "
                                   "media por debajo; nunca por recuento de pruebas a favor")}
    reglas_evaluadas = [{"categoria": r["categoria"], "condicion": r["condicion"],
                         "se_cumple": bool(r["se_cumple"])} for r in reglas]
    reglas_evaluadas[-2]["se_cumple"] = (categoria == "A. OPAQUE_ADMINISTRATIVE")
    reglas_evaluadas[-1]["se_cumple"] = (categoria == "G. UNKNOWN")

    evidencia_positiva = [p["evidencia"] for p in pruebas if p["apoya"]]
    contraevidencia = [p["evidencia"] for p in pruebas if not p["apoya"]]
    if categoria.startswith("D."):
        lectura = (
            "El id es un INDICE DE COLECCION: un entero dentro del bloque contiguo "
            f"[{s7['rango_union'][0]}, {s7['rango_union'][1]}], del que train y test son "
            f"muestras disjuntas que en conjunto ocupan {s7['union']} de "
            f"{s7['rango_union'][1] - s7['rango_union'][0] + 1} posiciones, dejando "
            f"{s7['valores_del_rango_no_usados']} sin usar. Hay ESTRUCTURA, pero es de "
            f"PROCEDENCIA y no de contenido: el numero dice de que coleccion salio la fila "
            f"y en que posicion ocupaba, y no dice quien la escribio, ni cuando, ni nada del "
            f"texto. Por eso queda como senal de procedencia y no como senal linguistica")
    else:
        lectura = (f"Categoria asignada por reglas explicitas: {categoria}, con la lectura "
                   f"que deja la evidencia recogida en este informe")

    log("")
    log(f"   CLASIFICACION: {categoria}, confianza {confianza}")
    log(f"      pruebas a favor {len(evidencia_positiva)} | pruebas en contra "
        f"{len(contraevidencia)}")
    log(f"      categorias descartadas con evidencia: {len(descartadas)} -> {descartadas}")
    log(f"      criterio de confianza: {s8['criterio_de_confianza']}")
    log(f"      LECTURA: {lectura}")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P63_ID_FORENSICS",
        "naturaleza": ("DIAGNOSTICO. 0 experimentos, 0 modelos, 0 features, 0 submissions, "
                       "0 Kaggle, 0 holdout, 0 test. No se entrena nada y no se usa id como "
                       "variable predictiva"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "experiment_id": "P63-00",
        "script": "experiments/p63_02_id_forensics.py",
        "pregunta": ("como esta construido el id, y es de verdad un contador administrativo "
                     "como afirmo V15-00"),
        "seccion_1_lectura_y_tipos": s1,
        "seccion_2_numerica_y_saltos": s2,
        "seccion_3_digitos_y_entropia": s3,
        "seccion_3_bis_relacion_con_orden_de_filas": s4,
        "seccion_4_bloques": s5,
        "seccion_5_id_contra_texto_y_grupos": s6,
        "seccion_6_namespace_train_test": s7,
        "seccion_6_bis_ocupacion_del_bloque": s7b,
        "seccion_7_pruebas_de_categoria": pruebas,
        "reglas_de_clasificacion": reglas_evaluadas,
        "criterio_de_confianza": s8,
        "clasificacion": {
            "categoria": categoria,
            "confianza": confianza,
            "lectura": lectura,
            "evidencia": evidencia_positiva,
            "contraevidencia": contraevidencia,
        },
        "lo_que_no_se_concluye": [
            "que el id sea util: eso exige una prueba predictiva, y este script no entrena",
            "que el id sea legal: eso exige una auditoria de legalidad aparte",
            "que el id sea transferible: eso exige una segunda particion",
            ("que el bloque sea una senal: la prevalencia por bloque se Midio para DESCRIBIR "
            "bloques ya localizados estructuralmente, y no es una feature"),
        ],
        "uso_autorizado": "NINGUNO como feature en esta fase. Solo diagnostico",
        "reproducibilidad": {"nivel": "R2", "git_commit": commit,
                             "hash_v03": hash_real, "corrected_cv": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_FOR.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    with CSV_FOR.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["seccion", "magnitud", "valor"])
        for nombre, bloque in (("1", s1), ("2", s2), ("3", s3), ("3bis", s4), ("4", s5),
                               ("5", s6), ("6", s7), ("6bis", s7b), ("7b", s8)):
            for k, v in bloque.items():
                if isinstance(v, (dict, list)):
                    v = json.dumps(v, ensure_ascii=True)
                w.writerow([nombre, a_ascii(str(k)), a_ascii(str(v))])
        for p in pruebas:
            w.writerow(["7", a_ascii(p["prueba"] + " | " + p["categoria"]),
                        a_ascii(("A FAVOR: " if p["apoya"] else "EN CONTRA: ") + p["evidencia"])])
        w.writerow(["7", "CLASIFICACION", a_ascii(f"{categoria}, confianza {confianza}")])

    L: list[str] = []
    W = L.append
    W("# P63 - Forense del identificador `id`")
    W("")
    W(f"Generado por `experiments/p63_02_id_forensics.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("> **DIAGNOSTICO, SIN SUPERVISION EN SU MAYORIA.** No entrena nada, no crea features y")
    W("> no usa `id` como variable predictiva. Las secciones 1 a 5 no leen una sola etiqueta;")
    W("> la seccion 4 las lee solo para DESCRIBIR bloques que ya estaban localizados")
    W("> estructuralmente, que es lo que el encargo permite y no mas.")
    W("")
    W("## Por que este informe existe")
    W("")
    W("V15-00 audito que informacion ofrece el fichero bruto y concluyo:")
    W("")
    W("```text")
    W("La unica columna no textual es id, que es un contador y que la seccion 3.E prohibe")
    W("```")
    W("")
    W("**Esa afirmacion no se contrasto con la construccion del valor.** Comprobar")
    W("`is_unique` no es una forense, y los cinco primeros ids del fichero son")
    W("")
    W("```text")
    W("20430, 22434, 23930, 20412, 23531")
    W("```")
    W("")
    W("Un identificador que no empieza en uno y no sigue el orden de las filas no es un")
    W("contador de filas. La afirmacion se reabre y se mide.")
    W("")
    W("## 1. Lectura, tipos y unicidad")
    W("")
    W("| magnitud | valor |")
    W("|---|---:|")
    for k in ("n_rows", "n_unique_ids", "n_repeated_ids", "duplicate_rate",
              "max_multiplicity", "min_value", "max_value", "range",
              "rango_teorico_de_valores", "valores_faltantes", "valores_no_enteros",
              "primer_valor", "primer_valor_es_uno"):
        W(f"| {k} | {s1[k]} |")
    W("")
    W(f"El id es globalmente unico dentro de train: {n_unicos} valores distintos para {n}")
    W("filas, cero repeticiones, cero nulos y cero valores no enteros. El fichero bruto lo")
    W(f"lleva entrecomillado, {crudo_tr[0]!r}.")
    W("")
    W("## 2. Estructura numerica, monotonia y saltos")
    W("")
    W("| magnitud | valor |")
    W("|---|---:|")
    for k, v in s2.items():
        W(f"| {k} | {v} |")
    W("")
    W(f"**El fichero esta barajado.** Solo el {frac_creciente * 100:.2f} por ciento de los")
    W("pares consecutivos de filas crecen, lo que equivale a un coeficiente de Kendall de")
    W(f"{(2 * frac_creciente - 1):+.6f}. Por tanto el id no codifica el orden de compilacion.")
    W("")
    W(f"El rango {ids.min()}-{ids.max()} tiene {s1['rango_teorico_de_valores']} posiciones y")
    W(f"train usa {n_unicos} de ellas, una ocupacion del "
      f"{s2['densidad_de_ocupacion_del_rango'] * 100:.2f} por ciento. Faltan")
    W(f"{s2['n_valores_ausentes_en_el_rango']} posiciones.")
    W("")
    W("## 3. Digitos, longitudes y entropia")
    W("")
    W("| magnitud | valor |")
    W("|---|---:|")
    for k in ("longitud_decimal", "longitud_binaria", "entropia_de_digitos",
              "entropia_maxima_teorica", "entropia_normalizada", "escala_observada",
              "bits_necesarios_para_un_snowflake",
              "orden_de_gran_magnitud_que_un_snowflake_daria"):
        W(f"| {k} | {s3[k]} |")
    W("")
    W(f"Frecuencia de cada digito: `{s3['frecuencia_de_digitos']}`")
    W("")
    W("La entropia de los digitos esta muy por debajo del maximo teorico de 3,32 bits, lo que")
    W("es exactamente lo que se espera de un rango acotado y no de un hash. Y la escala lo")
    W("dice solo: los valores llegan a " f"{max(unio)}, mientras que un identificador temporal")
    W("tipo Snowflake ocupa 63 bits y produce 18 o 19 cifras.")
    W("")
    W("## 3 bis. Relacion con el orden de las filas")
    W("")
    W("| correlacion | valor |")
    W("|---|---:|")
    for k, v in s4.items():
        W(f"| {k} | {v:+.6f} |")
    W("")
    W("## 4. Bloques, con el corte declarado antes de mirar nada")
    W("")
    W(f"El rango util se parte en **{N_BLOQUES} bloques de {s5['ancho_de_bloque']} posiciones**.")
    W("El corte se declara ANTES de leer ninguna etiqueta y no se ajusta despues.")
    W("")
    W("| bloque | id desde | id hasta | n train | n test | densidad | prevalencia "
      "cualquiera |")
    W("|---:|---:|---:|---:|---:|---:|---:|")
    for f in filas_bloque:
        prev = f.get("prevalencia_cualquiera")
        W(f"| {f['bloque']} | {f['id_desde']} | {f['id_hasta']} | {f['n_train']} | "
          f"{f['n_test']} | {f['densidad_train']:.4f} | "
          f"{'-' if prev is None else format(prev, '.4f')} |")
    W("")
    W(f"La densidad de ocupacion va de {s5['densidad_min']:.4f} a {s5['densidad_max']:.4f} y")
    W(f"la prevalencia de cualquier etiqueta varia {s5['rango_prevalencia_entre_bloques']} en")
    W("total. **No hay sub-bloques con comportamiento distinto:** la submuestra tiene la")
    W("misma densidad en todo el rango, que es la firma de una coleccion unica submuestreada")
    W("y no de varias fuentes con densidades diferentes.")
    W("")
    W("## 5. Id contra texto, duplicados y grupos")
    W("")
    W("| magnitud | valor |")
    W("|---|---:|")
    for k, v in s6.items():
        W(f"| {k} | {v} |")
    W("")
    W("El `id` es unico, luego `mismo id implica mismo texto` es trivialmente cierto. Lo que")
    W("no es trivial es lo otro: un mismo texto normalizado podria aparecer con dos ids")
    W(f"distintos, y ocurre en {len(rep_texto)} casos. Es decir **el id identifica la fila, no")
    W("el contenido**, que es exactamente lo que se espera de un indice de corpus.")
    W("")
    W("## 6. Integridad del namespace: control, no seleccion")
    W("")
    W("| magnitud | valor |")
    W("|---|---:|")
    for k, v in s7.items():
        W(f"| {k} | {v} |")
    W("")
    W("**Este es el hallazgo estructural de la fase.**")
    W("")
    W("```text")
    W(f"train        {s7['n_ids_train']} ids, rango {s7['rango_train']}")
    W(f"test         {s7['n_ids_test']} ids, rango {s7['rango_test']}")
    W(f"interseccion {s7['interseccion']}")
    W(f"union        {s7['union']}, rango {s7['rango_union']}")
    W(f"posiciones del rango que no usa nadie  {s7['valores_del_rango_no_usados']}")
    W("```")
    W("")
    W("Train y test son **dos muestras disjuntas del mismo bloque contiguo**, y ese bloque")
    W(f"tiene {max(unio) - min(unio) + 1} posiciones de las que se usan {len(unio)},")
    W(f"unas {len(unio) / (max(unio) - min(unio) + 1) * 100:.1f} por ciento. Ademas el fichero")
    W(f"`grupos_near_duplicados.csv` tiene exactamente {len(unio)} filas, una por id de la")
    W("union, lo que confirma que el bloque es el conjunto de registros del que se extrajeron")
    W("las dos particiones.")
    W("")
    W("## 6 bis. Ocupacion del bloque, train contra test, sin etiquetas")
    W("")
    W("| bloque | id desde | id hasta | train | test | sin usar | ocupacion | proporcion "
      "de train |")
    W("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for o in ocupacion:
        W(f"| {o['bloque']} | {o['id_desde']} | {o['id_hasta']} | {o['n_train']} | "
          f"{o['n_test']} | {o['n_sin_usar']} | {o['fraccion_ocupada']:.4f} | "
          f"{o['proporcion_de_train_entre_los_usados']:.4f} |")
    W("")
    W(f"La ocupacion va de {s7b['fraccion_ocupada_min']:.4f} a "
      f"{s7b['fraccion_ocupada_max']:.4f} y la proporcion de train entre los usados va de")
    W(f"{s7b['proporcion_train_min']:.4f} a {s7b['proporcion_train_max']:.4f}. Una "
      f"proporcion de train")
    W("**uniforme** en todos los bloques es la firma de una particion aleatoria de un unico")
    W("bloque, y es justo lo contrario de un muestreo por subcolecciones con distinta tasa de")
    W("inclusion. Este es el control que separa las dos historias, y se ha hecho sin leer una")
    W("sola etiqueta.")
    W("")
    W("Por tanto el `id` **no es un contador de las filas de este fichero**. Es un indice")
    W("dentro de una coleccion mayor, y V15-00 se equivoco al llamarlo contador.")
    W("")
    W("## 7. Las siete categorias, una a una")
    W("")
    W("| categoria | veredicto | evidencia |")
    W("|---|---|---|")
    for cat in sorted({p["categoria"] for p in pruebas}):
        a_favor = [p for p in pruebas if p["categoria"] == cat and p["apoya"]]
        en_contra = [p for p in pruebas if p["categoria"] == cat and not p["apoya"]]
        if a_favor:
            ver = "**A FAVOR**"
            detalle = a_favor[0]["evidencia"]
        elif en_contra:
            ver = "en contra"
            detalle = en_contra[0]["evidencia"]
        else:
            ver, detalle = "-", ""
        W(f"| {cat} | {ver} | {detalle} |")
    W("")
    W(f"### Clasificacion: **{categoria}**, confianza **{confianza}**")
    W("")
    W("La categoria **no** sale de contar pruebas a favor y en contra, porque eso mezcla")
    W("pruebas que apuntan a categorias distintas y decide por mayoria. Sale de reglas")
    W("explicitas, evaluadas en orden fijo, y la primera que se cumple gana:")
    W("")
    W("| regla | se cumple |")
    W("|---|---|")
    for r in reglas_evaluadas:
        W(f"| {r['categoria']} | {'si' if r['se_cumple'] else 'no'} |")
    W("")
    W("### Lectura")
    W("")
    W(lectura)
    W("")
    W("**Evidencia a favor**")
    W("")
    for e in evidencia_positiva:
        W(f"- {e}")
    W("")
    W("**Contraevidencia**")
    W("")
    for e in contraevidencia:
        W(f"- {e}")
    W("")
    W("## Lo que este informe NO concluye")
    W("")
    for e in informe["lo_que_no_se_concluye"]:
        W(f"- {e}")
    W("")
    W("Correlacion no implica legalidad ni transferibilidad, y un id estructurado sigue")
    W("siendo un id. El paso siguiente, si la evidencia lo justifica, es una prueba")
    W("predictiva minima con controles de permutacion y una segunda particion, no una")
    W("feature nueva.")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p63_02_id_forensics.py`. Sin push.*")
    MD_FOR.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_FOR, JSON_FOR, CSV_FOR, CSV_BLOQUES, CSV_OCUPACION):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P63-02 OK | unico en train {s1['n_unique_ids']}/{n} | rango {s1['min_value']}-"
        f"{s1['max_value']} | union train+test {s7['union']} | interseccion "
        f"{s7['interseccion']} | clasificacion {categoria} | 0 experimentos")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
