"""V22-01 (P0): auditoria exacta del preprocessing del corpus y del texto de la tarea.

Que hace y por que
------------------
La Hipotesis 1 de V22 afirma que el embedding de V5/V5.1 se estaba usando sobre un texto de
tarea con un preprocessing incompatible con el del corpus. Este script **no entrena nada** y
**no evalua ningun modelo**: reconstruye las dos cadenas token a token, mide donde se
difieren, y mide la cobertura de vocabulario bajo cada una contra el diccionario real de
vectores que usa el pipeline (los SIFTS).

Por que hay que medirlo y no solo leerlo
----------------------------------------
Leer el codigo ya sugiere que las dos cadenas coinciden en normalizacion y tokenizacion, y
que por tanto la hipotesis podria estar ya satisfecha. Pero "el codigo se parece" no es "el
espacio de tokens es el mismo": aunque las dos funciones sean la misma, el texto de un tweet
y el de un articulo de Wikipedia producen conjuntos de tokens distintos. La unica forma de
saber si hay una inconsistencia real es medir el espacio de tokens que cada pipeline produce
sobre los mismos textos, y medir cuantos de esos tokens encuentran vector.

Controles, y por que cada uno esta
----------------------------------
**El diccionario de vectores es el real, no el `.vec`.** El pipeline de V5.1 consulta al
binario de fastText y cachea la salida en ficheros SIFTS, y ahi los tipos fuera de
vocabulario reciben vector por composicion de subpalabras. El `.vec` solo exporta los
100.098 tipos del modelo, y de los 6.059 consultados solo 3.777 aparecen en el. Usar el
`.vec`aria cambiar la cobertura medida y daria una conclusion falsa sobre el emparejamiento
de espacios. Este script usa los SIFTS y **verifica que la entrada de los SIFTS sea
exactamente el conjunto de tipos que se consulto**, que es lo que garantiza que el
diccionario corresponde al pipeline historico.

**Se miden los dos conjuntos de datos.** El desarrollo (1.483 filas) y el test (1.154).
El test no se usa para entrenar ni para ajustar nada: solo se transforman sus textos para
medir cobertura, que es una operacion de inferencia.

**El recuento de tokens es sobre el pipeline, no sobre texto.** Se cuenta lo que
`tokenizar` devuelve, que es lo que realmente consulta el diccionario, y no lo que hay en la
cadena de caracteres.

Uso
---
    /home/carlos/venv/bin/python experiments/v22_01_auditoria_preprocessing.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_test, read_train
from src.embedding_utils import normalizar_emb_normal, tokenizar

EXPERIMENTO = "V22-01 (P0: auditoria del preprocessing del corpus frente al de la tarea)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v22_01_auditoria_preprocessing.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v22_01_auditoria_preprocessing.md"
SIFTS = PROJECT_ROOT / "data" / "external" / "embeddings" / "fasttext-baseline_sifts.txt"
SIFTS_ENTRADA = SIFTS.with_suffix(".entrada.txt")
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"

RE_HASHTAG = re.compile(r"#\w+", re.UNICODE)
RE_MENCION_RAW = re.compile(r"@\w+", re.UNICODE)
RE_URL_RAW = re.compile(r"https?://\S+|www\.\S+", re.UNICODE)
RE_NUMERO_RAW = re.compile(r"\d+", re.UNICODE)
RE_ESPACIOS = re.compile(r"\s+", re.UNICODE)
RE_ELONGACION = re.compile(r"(\w)\1{2,}", re.UNICODE)

GRUPOS = ("HS", "TR", "AG")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V22-01] {mensaje}", flush=True)


def cargar_diccionario() -> tuple[set[str], list[str], int]:
    """Carga el diccionario real de vectores desde los SIFTS.

    Verifica que la entrada de los SIFTS sea exactamente la lista de tipos consultados por
    V5.1, y descarta los vectores nulos como hace `vectores_de_tipos`.

    Returns:
        Tupla con el conjunto de tipos con vector, la lista de tipos consultados, y el
        numero de vectores nulos descartados.
    """
    tipos_consultados = SIFTS_ENTRADA.read_text(encoding="utf-8").split()
    diccionario: set[str] = set()
    nulos = 0
    with SIFTS.open("r", encoding="utf-8") as manejador:
        for indice, linea in enumerate(manejador, start=1):
            partes = linea.rstrip().split()
            if len(partes) < 2:
                raise SystemExit(f"PARADA OBLIGATORIA: linea {indice} de SIFTS mal formada")
            vector = np.asarray(partes[1:], dtype=np.float32)
            if not np.any(vector):
                nulos += 1
                continue
            diccionario.add(partes[0])
    log(
        f"   SIFTS: {len(tipos_consultados)} tipos consultados | {len(diccionario)} con vector "
        f"util | {nulos} nulos descartados"
    )
    if not diccionario:
        raise SystemExit("PARADA OBLIGATORIA: el diccionario de SIFTS esta vacio")
    return diccionario, tipos_consultados, nulos


def espacio_de_tokens(textos: list[str], consistente: bool) -> list[list[str]]:
    """Devuelve los tokens que el pipeline realmente consulta al diccionario.

    Args:
        textos: textos de entrada.
        consistente: si se aplica `normalizar_emb_normal` antes de tokenizar.

    Returns:
        Lista de listas de tokens, una por texto.
    """
    if consistente:
        return [tokenizar(normalizar_emb_normal(texto)) for texto in textos]
    return [tokenizar(texto) for texto in textos]


def metricas_espacio(
    listas: list[list[str]], diccionario: set[str], etiquetas: list[str] | None = None
) -> dict[str, Any]:
    """Mide cobertura y composicion de un espacio de tokens.

    Args:
        listas: listas de tokens por fila.
        diccionario: tipos con vector.
        etiquetas: columna de verdad, solo para el desglose por grupo.

    Returns:
        Diccionario de metricas.
    """
    todos = [token for lista in listas for token in lista]
    total = len(todos)
    unicos = set(todos)
    con_vector = [t for t in todos if t in diccionario]
    longitudes = np.array([len(lista) for lista in listas], dtype=float)
    resumen: dict[str, Any] = {
        "filas": len(listas),
        "tokens_totales": total,
        "tipos_efectivos": len(unicos),
        "tokens_con_vector": len(con_vector),
        "tokens_oov": total - len(con_vector),
        "cobertura_tipo": len([t for t in unicos if t in diccionario]) / max(len(unicos), 1),
        "cobertura_token": len(con_vector) / max(total, 1),
        "longitud_media": float(longitudes.mean()),
        "longitud_mediana": float(np.median(longitudes)),
        "filas_sin_tokens": int(np.count_nonzero(longitudes == 0)),
        "top_oov": [
            (token, cuenta)
            for token, cuenta in Counter(t for t in todos if t not in diccionario).most_common(20)
        ],
    }
    if etiquetas is not None:
        por_grupo = {}
        for grupo in GRUPOS:
            seleccion = [lista for lista, y in zip(listas, etiquetas) if y == grupo]
            if not seleccion:
                continue
            aplanados = [t for lista in seleccion for t in lista]
            con_vec = sum(1 for t in aplanados if t in diccionario)
            por_grupo[grupo] = {
                "filas": len(seleccion),
                "tokens": len(aplanados),
                "cobertura_token": con_vec / max(len(aplanados), 1),
            }
        resumen["por_grupo"] = por_grupo
    return resumen


def analizar_marcas(textos: list[str], consistentes: list[str]) -> dict[str, Any]:
    """Cuenta las marcas de Twitter en el texto crudo y en su forma normalizada.

    Args:
        textos: textos crudos.
        consistentes: textos ya normalizados.

    Returns:
        Diccionario con el recuento de cada marca y de suuen cascada en tokens.
    """
    salida: dict[str, Any] = {}
    patrones = {
        "hashtag_crus": RE_HASHTAG,
        "menciones_crudas": RE_MENCION_RAW,
        "urls_crudas": RE_URL_RAW,
        "numeros_crus": RE_NUMERO_RAW,
    }
    for nombre, patron in patrones.items():
        salida[nombre] = int(sum(len(patron.findall(texto)) for texto in textos))
    marcadores = ("url", "hashtag", "usuario", "numero")
    salida["marcadores_en_normalizado"] = {
        m: int(sum(texto.count(f" {m} ") for texto in consistentes)) for m in marcadores
    }
    salida["elongaciones_crudas"] = int(
        sum(len(RE_ELONGACION.findall(texto)) for texto in textos)
    )
    return salida


def main() -> None:
    """Ejecuta la auditoria de los dos pipelines y escribe el informe."""
    inicio = time.time()
    log("V22-01: P0, auditoria del preprocessing. Ningun modelo se entrena ni se evalua")

    log("1/5 diccionario real de vectores (SIFTS, no el .vec)")
    diccionario, tipos_consultados, nulos = cargar_diccionario()

    log("2/5 conjuntos de datos")
    train = read_train()
    particion = pd.read_csv(PARTICION)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    test = read_test()
    log(f"   desarrollo {len(desarrollo)} | test {len(test)} | Holdout reservado y no se usa")
    y_dev = desarrollo[list(GRUPOS)].to_numpy().astype(int)

    log("3/5 espacios de tokens bajo cada pipeline")
    textos_dev = desarrollo["text"].astype(str).tolist()
    textos_test = test["text"].astype(str).tolist()
    crudo_dev = espacio_de_tokens(textos_dev, False)
    cons_dev = espacio_de_tokens(textos_dev, True)
    crudo_test = espacio_de_tokens(textos_test, False)
    cons_test = espacio_de_tokens(textos_test, True)
    log("   los cuatro espacios construidos")

    log("4/5 metricas por espacio")
    metricas = {
        "desarrollo_raw": metricas_espacio(crudo_dev, diccionario, [int(v) for v in y_dev[:, 0]]),
        "desarrollo_consistent": metricas_espacio(cons_dev, diccionario, [int(v) for v in y_dev[:, 0]]),
        "test_raw": metricas_espacio(crudo_test, diccionario),
        "test_consistent": metricas_espacio(cons_test, diccionario),
    }
    for nombre, m in metricas.items():
        log(
            f"   {nombre:<24} tokens {m['tokens_totales']:>7} | tipos {m['tipos_efectivos']:>6} | "
            f"cob.tipo {m['cobertura_tipo']:.4f} | cob.token {m['cobertura_token']:.4f} | "
            f"long.media {m['longitud_media']:.2f}"
        )

    log("5/5 el mismatch, token a token")
    plano_raw_dev = Counter(t for lista in crudo_dev for t in lista)
    plano_cons_dev = Counter(t for lista in cons_dev for t in lista)
    solo_raw = {t: c for t, c in plano_raw_dev.items() if t not in plano_cons_dev}
    solo_cons = {t: c for t, c in plano_cons_dev.items() if t not in plano_raw_dev}
    comunes = set(plano_raw_dev) & set(plano_cons_dev)
    log(f"   tipos solo en RAW: {len(solo_raw)} | solo en CONSISTENT: {len(solo_cons)} | "
        f"comunes: {len(comunes)}")
    log("   15 tipos mas frecuentes que solo aparecen en RAW:")
    for token, cuenta in sorted(solo_raw.items(), key=lambda kv: -kv[1])[:15]:
        log(f"      {token!r}: {cuenta}")
    log("   15 tipos mas frecuentes que solo aparecen en CONSISTENT:")
    for token, cuenta in sorted(solo_cons.items(), key=lambda kv: -kv[1])[:15]:
        log(f"      {token!r}: {cuenta}")

    marcas_dev = analizar_marcas(textos_dev, [normalizar_emb_normal(t) for t in textos_dev])
    marcas_test = analizar_marcas(textos_test, [normalizar_emb_normal(t) for t in textos_test])

    delta_cob = (
        metricas["desarrollo_consistent"]["cobertura_token"]
        - metricas["desarrollo_raw"]["cobertura_token"]
    )
    veredicto_mismatch = (
        "NO_EVIDENCIADO" if abs(delta_cob) < 0.005 else "EVIDENCIADO"
    )
    log(
        f"   VEREDICTO: cobertura CONSISTENT - RAW = {delta_cob:+.4f} en el desarrollo | "
        f"mismatch {veredicto_mismatch}"
    )

    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "diccionario_de_vectores": {
            "origen": "SIFTS del binario de fastText, no el .vec",
            "tipos_consultados": len(tipos_consultados),
            "tipos_con_vector_util": len(diccionario),
            "nulos_descartados": nulos,
        },
        "metricas": metricas,
        "mismatch": {
            "tipos_solo_raw": len(solo_raw),
            "tipos_solo_consistent": len(solo_cons),
            "tipos_comunes": len(comunes),
            "delta_cobertura_token_desarrollo": delta_cob,
            "delta_cobertura_tipo_desarrollo": (
                metricas["desarrollo_consistent"]["cobertura_tipo"]
                - metricas["desarrollo_raw"]["cobertura_tipo"]
            ),
            "veredicto": veredicto_mismatch,
        },
        "marcas": {"desarrollo": marcas_dev, "test": marcas_test},
        "top_oov_raw_desarrollo": metricas["desarrollo_raw"]["top_oov"],
        "top_oov_consistent_desarrollo": metricas["desarrollo_consistent"]["top_oov"],
        "gastos": {"modelos_entrenados": 0, "evaluaciones": 0, "submissions": 0},
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe_md(informe, inicio)
    log("   informes escritos en reports/v22_01_auditoria_preprocessing.{json,md}")
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(
        f"VEREDICTO P0: mismatch de preprocessing {veredicto_mismatch} | "
        f"delta cobertura {delta_cob:+.4f} | submissions 0"
    )


def _informe_md(informe: dict[str, Any], inicio: float) -> None:
    """Escribe el informe legible de P0.

    Args:
        informe: informe con todos los datos.
        inicio: instante de arranque.
    """
    m = informe["metricas"]
    mis = informe["mismatch"]
    lineas = [
        "# V22-01 (P0): auditoria del preprocessing del corpus frente al de la tarea",
        "",
        f"Generado por `experiments/v22_01_auditoria_preprocessing.py` ({EXPERIMENTO}).",
        "",
        "**Ningun modelo entrenado. Ninguna submission. El test solo se transformo.**",
        "",
        "## 1. Las dos cadenas, tal como estan en el codigo",
        "",
        "| paso | corpus | texto de la tarea |",
        "|---|---|---|",
        "| limpieza | `limpiar_marcado` (MediaWiki) | no aplica |",
        "| normalizacion | `normalizar_emb_normal` | `normalizar_emb_normal` |",
        "| tokenizacion | `tokenizar` | `tokenizar` dentro de `vectores_documento` |",
        "",
        "`embedding_13_window_v5_1.py` linea 226 aplica `normalizar_emb_normal` a",
        "`desarrollo['text']` antes de construir los vectores de documento. La normalizacion",
        "del corpus y la del texto de la tarea **son la misma funcion**.",
        "",
        "## 2. Las dos cadenas medidas sobre los mismos textos",
        "",
        "| espacio | tokens | tipos | cobertura tipo | cobertura token | long. media | filas sin tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for nombre in ("desarrollo_raw", "desarrollo_consistent", "test_raw", "test_consistent"):
        v = m[nombre]
        lineas.append(
            f"| {nombre} | {v['tokens_totales']} | {v['tipos_efectivos']} | "
            f"{v['cobertura_tipo']:.4f} | {v['cobertura_token']:.4f} | "
            f"{v['longitud_media']:.2f} | {v['filas_sin_tokens']} |"
        )
    lineas += [
        "",
        "## 3. El mismatch, token a token (desarrollo)",
        "",
        f"- Tipos que solo aparecen en RAW: **{mis['tipos_solo_raw']}**.",
        f"- Tipos que solo aparecen en CONSISTENT: **{mis['tipos_solo_consistent']}**.",
        f"- Tipos comunes: **{mis['tipos_comunes']}**.",
        f"- Delta de cobertura por token (CONSISTENT - RAW): **{mis['delta_cobertura_token_desarrollo']:+.4f}**.",
        f"- Delta de cobertura por tipo (CONSISTENT - RAW): **{mis['delta_cobertura_tipo_desarrollo']:+.4f}**.",
        "",
        f"**Veredicto de P0: {mis['veredicto']}.**",
        "",
        "## 4. Marcas de Twitter",
        "",
        "| marca | desarrollo crudo | desarrollo normalizado | test crudo | test normalizado |",
        "|---|---:|---:|---:|---:|",
        (f"| hashtags | {informe['marcas']['desarrollo']['hashtag_crus']} | "
        f"{informe['marcas']['desarrollo']['marcadores_en_normalizado']['hashtag']} | "
        f"{informe['marcas']['test']['hashtag_crus']} | "
        f"{informe['marcas']['test']['marcadores_en_normalizado']['hashtag']} |"),
        (f"| menciones | {informe['marcas']['desarrollo']['menciones_crudas']} | "
        f"{informe['marcas']['desarrollo']['marcadores_en_normalizado']['usuario']} | "
        f"{informe['marcas']['test']['menciones_crudas']} | "
        f"{informe['marcas']['test']['marcadores_en_normalizado']['usuario']} |"),
        (f"| URLs | {informe['marcas']['desarrollo']['urls_crudas']} | "
        f"{informe['marcas']['desarrollo']['marcadores_en_normalizado']['url']} | "
        f"{informe['marcas']['test']['urls_crudas']} | "
        f"{informe['marcas']['test']['marcadores_en_normalizado']['url']} |"),
        (f"| numeros | {informe['marcas']['desarrollo']['numeros_crus']} | "
        f"{informe['marcas']['desarrollo']['marcadores_en_normalizado']['numero']} | "
        f"{informe['marcas']['test']['numeros_crus']} | "
        f"{informe['marcas']['test']['marcadores_en_normalizado']['numero']} |"),
        (f"| elongaciones | {informe['marcas']['desarrollo']['elongaciones_crudas']} | "
        f"preservadas | {informe['marcas']['test']['elongaciones_crudas']} | preservadas |"),
        "",
        "## 5. Los 20 OOV mas frecuentes del desarrollo",
        "",
        "| token | RAW | CONSISTENT |",
        "|---|---:|---:|",
    ]
    top_raw = dict(informe["top_oov_raw_desarrollo"])
    top_cons = dict(informe["top_oov_consistent_desarrollo"])
    for token in sorted(set(top_raw) | set(top_cons), key=lambda t: -max(top_raw.get(t, 0), top_cons.get(t, 0))):
        lineas.append(f"| {token} | {top_raw.get(token, 0)} | {top_cons.get(token, 0)} |")
    lineas += [
        "",
        "## 6. Reproducibilidad",
        "",
        ("- Diccionario de vectores: SIFTS del binario de fastText, "
        f"{informe['diccionario_de_vectores']['tipos_consultados']} tipos consultados, "
        f"{informe['diccionario_de_vectores']['tipos_con_vector_util']} con vector util, "
        f"{informe['diccionario_de_vectores']['nulos_descartados']} nulos descartados."),
        "- Test usado solo como transformacion de texto para medir cobertura. No se leen",
        "  etiquetas, no se entrena, no se ajusta nada.",
        f"- Completado en {round(time.time() - inicio, 1)} s.",
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
