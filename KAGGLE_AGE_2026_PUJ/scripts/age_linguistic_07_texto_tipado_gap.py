"""P7-P0 - auditoria de novedad del canal lemma_POS (texto tipado) por OPERACION.

## Que hace, antes de entrenar nada

1. Barrido por OPERACION sobre el codigo ejecutable (scripts/, src/, notebooks/): busca
   consumidores reales de `lemma_pos_seq`, construcciones de tokens tipados y joins de lema+POS.
   El nombre de la tecnica no cuenta: manda el codigo.
2. Clasifica P7: DUPLICATE si existe una implementacion equivalente ya usada; GENUINELY_NEW si no.
3. Mide la cobertura del bloque tipado sobre los artefactos congelados (parquet md/lg): tokens,
   vocabulario de unigramas lemma_POS y estimacion de bigramas.
4. Deja fijados legalidad (D-076), la especificacion de representacion y la comparacion
   explicita con P2 (POS-secuencia) y P6 (NER estructural).

Salidas: reports/age_typed_text_gap_analysis.md, reports/age_typed_text_gap_analysis.csv,
experiments/age_linguistic_07_texto_tipado_gap.json. No usa test, no usa holdout, no entrena.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

FECHA = "2026-09-30"
PARQUETS = {"md": ROOT / "features" / "linguistic" / "anotacion_md_train.parquet",
            "lg": ROOT / "features" / "linguistic" / "anotacion_lg_train.parquet"}
GAP_MD = cfg.REPORTS / "age_typed_text_gap_analysis.md"
GAP_CSV = cfg.REPORTS / "age_typed_text_gap_analysis.csv"
RESULTADO = cfg.EXPERIMENTS / "age_linguistic_07_texto_tipado_gap.json"
PROPIO = Path(__file__).name
PRODUCTOR = "age_linguistic_00_anotacion.py"  # crea la columna; no es consumidor

PATRON_CONSUMO = re.compile(r"lemma_pos_seq")
PATRON_LITERAL_TIPADO = re.compile(
    r"\w+_(?:VERB|NOUN|ADJ|ADV|PRON|DET|ADP|CCONJ|SCONJ|AUX|PROPN|NUM|INTJ)\b")
PATRON_JOIN_LEMA_POS = re.compile(r"join\([^)]*\.(?:lemma_|pos_)")
PATRON_CONSTRUCCION = re.compile(r"_\{(?:t|tok|token|tk)\.pos_|f\"\{lema\}_|\{lema\}_\{")

FILAS = [
    {"id": "P7-TEXT-TIPADO", "familia": "texto tipado (texto original + lemma_POS)",
     "operacion_probada": "modelo entrenado sobre [bloque de texto original] + [bloque de tokens "
                          "lemma_POS], unigramas y bigramas, con escala por bloque",
     "evidencia": "El barrido por operacion encuentra CERO consumidores de `lemma_pos_seq` como "
                  "representacion entrenada: la columna existe SOLO como anotacion almacenada "
                  "(creada por age_linguistic_00_anotacion.py) y mencionada en informes. Los "
                  "unicos usos de lemas en codigo entrenado son AGREGADOS (v14a, 3 variables de "
                  "diversidad dentro de las 37 cerradas). El scan P0 ya verifico 0 firmas de "
                  "construccion de secuencias en 252 ficheros.",
     "status": "GENUINELY_NEW",
     "razon": "Ninguna fase construyo ni uso tokens lemma_POS como features. Diferente de P2 "
              "(POS sin lexema: cierra la funcion sin conservar el contenido) y de P6 (tipos de "
              "entidad sin contenido lexico general)."},
    {"id": "P7-DUP-LEMA-AGG", "familia": "variantes ya cubiertas de lema", "operacion_probada":
     "diversidad de lemas / hapax por lema / forma por lema",
     "evidencia": "v14a (3 de las 37 variables; fusion -0,0015 con 1/5 folds).",
     "status": "CLOSED", "razon": "Agregados, no tokens; forma distinta y ya medida."},
    {"id": "P7-DUP-SUFIJOS", "familia": "morfologia de superficie", "operacion_probada":
     "rasgos por sufijo",
     "evidencia": "Audit de V13: redundante con char_wb(3-5).",
     "status": "CLOSED", "razon": "La parte lexica de lemma_POS NO es un rasgo de sufijo: es el "
                                  "lexema completo con su categoria; no es el duplicado que V13 cerro."},
]


def escanear() -> dict:
    """Barrido por operacion sobre el codigo ejecutable; el auditor no se cuenta a si mismo."""
    archivos = []
    for patron in ("scripts/*.py", "src/*.py", "notebooks/*.ipynb"):
        archivos += [p for p in ROOT.glob(patron) if p.name != PROPIO]
    consumidores, construcciones, literales, joins = [], [], [], []
    for p in archivos:
        texto = p.read_text(encoding="utf-8", errors="replace")
        if p.name != PRODUCTOR and PATRON_CONSUMO.search(texto):
            consumidores.append(p.name)
        if p.name != PRODUCTOR:
            hallazgos = PATRON_CONSTRUCCION.findall(texto)
            if hallazgos:
                construcciones.append(p.name)
            hallazgos = PATRON_LITERAL_TIPADO.findall(texto)
            if hallazgos:
                literales.append({"archivo": p.name, "ejemplos": sorted(set(hallazgos))[:4]})
            hallazgos = PATRON_JOIN_LEMA_POS.findall(texto)
            if hallazgos:
                joins.append({"archivo": p.name, "ejemplos": sorted(set(hallazgos))[:4]})
    # contexto: quien produce la columna y quien la menciona
    return {"archivos_barridos": len(archivos),
            "consumidores_lemma_pos_seq": sorted(consumidores),
            "constructores_tokens_tipados": sorted(construcciones),
            "literales_tipados": literales, "joins_lema_pos": joins}


def cobertura(clave: str) -> dict:
    frame = pd.read_parquet(PARQUETS[clave], columns=["lemma_pos_seq", "n_tokens"])
    secuencias = frame["lemma_pos_seq"].astype(str)
    no_vacias = int((secuencias.str.len() > 0).sum())
    muestras = secuencias[secuencias.str.len() > 0]
    unigramas = set()
    bigramas = set()
    for s in muestras.to_numpy():
        tokens = s.split()
        unigramas.update(tokens)
        if len(tokens) > 1:
            bigramas.update(zip(tokens, tokens[1:]))
    return {"documentos": int(len(frame)), "documentos_no_vacios": no_vacias,
            "pct_no_vacios": round(100.0 * no_vacias / len(frame), 2),
            "tokens_totales": int(frame["n_tokens"].sum()),
            "vocab_unigramas": len(unigramas), "vocab_bigramas": len(bigramas),
            "ejemplo": muestras.iloc[0][:120] if no_vacias else ""}


def escribe_gap_md(ops: dict, cob: dict) -> None:
    lineas = [
        "# Texto tipado (P7) - auditoria de novedad y especificacion",
        "",
        f"**Fecha**: {FECHA} | **Metodo**: auditoria POR OPERACION sobre el codigo ejecutable "
        f"({ops['archivos_barridos']} ficheros entre scripts/, src/ y notebooks/); el nombre de "
        "la tecnica no cuenta como evidencia. Se excluye al propio auditor y al productor de la "
        "anotacion.",
        "",
        "## Clasificacion",
        "",
        "| ID | Familia | Estado |",
        "|---|---|---|",
    ]
    for fila in FILAS:
        lineas.append(f"| {fila['id']} | {fila['familia']} | **{fila['status']}** |")
    lineas += [
        "",
        "## Evidencia del barrido",
        "",
        f"- Consumidores de `lemma_pos_seq` como representacion entrenada (excluido el productor): "
        f"**{len(ops['consumidores_lemma_pos_seq'])}** "
        f"{ops['consumidores_lemma_pos_seq'] or ''}",
        f"- Constructores de tokens tipados fuera del productor: "
        f"**{len(ops['constructores_tokens_tipados'])}** "
        f"{ops['constructores_tokens_tipados'] or ''}",
        f"- Literales de token tipado (tipo `palabra_VERB`) en codigo: "
        f"**{len(ops['literales_tipados'])}** {ops['literales_tipados'] or ''}",
        f"- Joins de lema/POS en concatenaciones: **{len(ops['joins_lema_pos'])}** "
        f"{ops['joins_lema_pos'] or ''}",
        "",
        "La columna `lemma_pos_seq` existe SOLO como anotacion almacenada (la crea "
        "`scripts/age_linguistic_00_anotacion.py` al construir los parquets). Ninguna fase la ha "
        "consumido como features de un modelo. La clasificacion es **GENUINELY_NEW**.",
        "",
        "## Cobertura del bloque tipado (artefactos congelados)",
        "",
        "| Anotador | Docs no vacios | Tokens | Vocab unigramas | Vocab bigramas |",
        "|---|---|---|---|---|",
    ]
    for clave in ("md", "lg"):
        c = cob[clave]
        lineas.append(f"| {clave} | {c['documentos_no_vacios']:,} ({c['pct_no_vacios']}%) | "
                      f"{c['tokens_totales']:,} | {c['vocab_unigramas']:,} | "
                      f"{c['vocab_bigramas']:,} |")
    lineas += [
        "",
        f"Ejemplo de secuencia tipada (md): `{cob['md']['ejemplo']}`",
        "",
        "## Legalidad (D-076, sin herramientas nuevas)",
        "",
        "- No se usa ningun anotador nuevo: se reutilizan las anotaciones CONGELADAS de spaCy "
        "(md/lg, no-transformer) con SHA256 registrados en "
        "`experiments/age_linguistic_01_anotacion.json`.",
        "- Solo etiquetas y lemas (forma); vectores y embeddings siguen prohibidos y no se tocan.",
        "- Sin datos externos, sin labels de edad externas, sin metadatos de usuario, sin test, "
        "sin holdout.",
        "",
        "## Especificacion de la representacion (congelada antes de ejecutar)",
        "",
        "```text",
        "B = V30 + plug-in + texto tipado",
        "texto tipado = [bloque A: texto original] + [bloque B: tokens lemma_POS]",
        "  bloque A: TF-IDF word (1,2), min_df=3, token_pattern \\b\\w+\\b   (contenido lexical)",
        "  bloque B: TF-IDF word (1,2), min_df=5, token_pattern \\S+      (lemma_ETIQUETA)",
        "  vectorizadores SEPARADOS (sin colision de vocabulario; cada uno se ajusta SOLO",
        "  en el train de cada fold); cada bloque queda L2 por fila (normalizacion por bloque)",
        "  escala por bloque: 1.0 fija en ambos (neutro; precedente V31-03), definida ANTES",
        "```",
        "",
        "Unigramas y bigramas de lemma_POS (sin trigramas). No se anaden NER, dependencias, POS "
        "puro, morfologia, longitud, URL, hashtags ni features densas.",
        "",
        "## Comparacion explicita",
        "",
        "| Fase | Representacion | Que conserva | Que anade |",
        "|---|---|---|---|",
        "| P2 (cerrado) | secuencia de POS | nada lexical | funcion abstracta de cada token |",
        "| P6 (cerrado) | tipos de entidad | nada lexical | estructura de entidades |",
        "| **P7** | texto original + lemma_POS | **el contenido lexical del texto** | "
        "**funcion linguistica por token, ligada a su lexema** |",
        "",
        "La hipotesis de P7 no es 'mas NLP': es que el componente lexical tipado puede aportar "
        "informacion complementaria al campeon que ni el TF-IDF convencional ni POS aislado "
        "capturan. P2 y P6 no se extrapolan a P7.",
    ]
    texto = "\n".join(l.rstrip() for l in lineas) + "\n"
    GAP_MD.write_text(texto, encoding="utf-8")


def escribe_csv() -> None:
    cols = ["id", "familia", "operacion_probada", "evidencia", "status", "razon"]
    with GAP_CSV.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        escritor.writeheader()
        for fila in FILAS:
            escritor.writerow({c: fila[c] for c in cols})


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    print("=== barrido por operacion ===", flush=True)
    ops = escanear()
    print(f"  archivos: {ops['archivos_barridos']} | consumidores de lemma_pos_seq (sin el "
          f"productor): {ops['consumidores_lemma_pos_seq'] or 'NINGUNO'}", flush=True)
    print(f"  constructores de tokens tipados: {ops['constructores_tokens_tipados'] or 'NINGUNO'}",
          flush=True)
    if ops["consumidores_lemma_pos_seq"] or ops["constructores_tokens_tipados"]:
        print("[STOP] hay consumidores/constructores no inventariados: revisar antes de seguir",
              flush=True)
        return 1

    print("=== cobertura del bloque tipado ===", flush=True)
    cob = {clave: cobertura(clave) for clave in ("md", "lg")}
    for clave, c in cob.items():
        print(f"  {clave}: {c['pct_no_vacios']}% docs no vacios | {c['tokens_totales']:,} tokens | "
              f"vocab unigramas {c['vocab_unigramas']:,} | bigramas {c['vocab_bigramas']:,}",
              flush=True)

    escribe_gap_md(ops, cob)
    escribe_csv()
    print(f"  {GAP_MD.name} | {GAP_CSV.name}", flush=True)

    RESULTADO.write_text(json.dumps({
        "fase": "P7-P0 auditoria de novedad del texto tipado", "fecha": FECHA,
        "clasificacion": {f["id"]: f["status"] for f in FILAS},
        "operaciones": ops, "cobertura": cob,
        "sin_test": True, "sin_holdout": True, "sin_entrenamiento": True,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[fin] {time.time() - inicio:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
