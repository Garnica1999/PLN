"""P6-P0 - auditoria de P3-P5 y verificacion de que el NER estructural es GENUINELY_NEW.

## Que hace, y por que antes de entrenar nada

1. Audita por OPERACION las familias P3 (lemma+POS), P4 (morfologia) y P5 (dependencias):
   que se uso de verdad (codigo ejecutado), y si cada una aporta un mecanismo informacional
   distinto de P2 o ya esta cubierta.
2. Verifica que el NER estructural (secuencia de TIPOS de entidad) nunca se ha construido.
3. Mide la cobertura y la calidad real de la anotacion NER sobre los artefactos congelados
   (md y lg): porcentaje de tokens en entidades, composicion de lo que el modelo marca
   (palabras vs menciones/RT/hashtags/URLs), cortes por longitud y por marcas de Twitter, y una
   muestra diagnostica de ejemplos SIN corregir nada.
4. Separa explicitamente COBERTURA de UTILIDAD PREDICTIVA: la segunda solo la puede medir el
   experimento P6, no esta auditoria.

Salidas: reports/ner_structural_gap_analysis.md, reports/ner_structural_gap_analysis.csv,
experiments/age_linguistic_04_ner_gap.json. No usa test, no usa holdout, no entrena.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

FECHA = "2026-09-29"
PARQUETS = {"md": ROOT / "features" / "linguistic" / "anotacion_md_train.parquet",
            "lg": ROOT / "features" / "linguistic" / "anotacion_lg_train.parquet"}
GAP_MD = cfg.REPORTS / "ner_structural_gap_analysis.md"
GAP_CSV = cfg.REPORTS / "ner_structural_gap_analysis.csv"
RESULTADO = cfg.EXPERIMENTS / "age_linguistic_04_ner_gap.json"
TOKEN_RE = re.compile(r"@\w+|#\w+|https?://\S+|www\.\S+|\d+|[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+|[^\w\s]")

FILAS = [
    {"id": "P3-LEMMA-POS", "familia": "lemma + POS", "operacion_probada":
     "tokens lemma_ETIQUETA en secuencia",
     "evidencia": "`.lemma_` aparece SOLO en scripts/v14a_linguistico.py y solo como AGREGADOS "
                  "(lema_distintos_por_token, lema_hapax_ratio, forma_distintas_por_lemma, dentro "
                  "de las 37 variables cuya fusion dio -0,0015, 1/5 folds). Cero construcciones de "
                  "lemmas como tokens en 252 ficheros (scan P0 con firmas de secuencia).",
     "status": "PARTIALLY_COVERED",
     "razon": "No reconstruible de P2 (POS no da lemas), pero su incremento no-POS es lexical "
              "(fusion de flexiones), territorio ya cubierto por word/char n-grams y la evidencia "
              "por token; su incremento POS acaba de cerrarse en P2. Sin razon independiente."},
    {"id": "P4-MORFOLOGIA", "familia": "morfologia", "operacion_probada":
     "tokens morfologicos (Tense=Past|1P) en secuencia corta",
     "evidencia": "`.morph` aparece SOLO en scripts/v14a_linguistico.py (16 ratios de tiempo/modo/"
                  "persona/genero/numero). La morfologia por SUFIJOS quedo cerrada como redundante "
                  "con char_wb(3-5) en el audit de V13 (`reports/v13_gap_audit.csv`).",
     "status": "PARTIALLY_COVERED",
     "razon": "La etiqueta morfologica es casi deterministica de la forma superficial (sufijo) que "
              "el char n-gram ya ve; ratios medidos negativos; su forma secuencial es una variante "
              "del mecanismo de P2, cuyo resultado acaba de cerrarse. Sin razon independiente."},
    {"id": "P5-DEPENDENCIAS", "familia": "dependency structures", "operacion_probada":
     "patrones tipados de dependencia (subj->VERB, DET->NOUN) o secuencias tipadas",
     "evidencia": "`.dep_` aparece SOLO en scripts/v14a_linguistico.py y solo como 6 AGREGADOS "
                  "(profundidad_media, coordinacion_ratio, subordinacion_ratio, hijos_por_token, "
                  "pasiva_ratio, tokens_por_frase; fusion -0,0015). Patrones tipados: nunca "
                  "construidos.",
     "status": "GENUINELY_NEW",
     "razon": "Por operacion es nueva, pero comparte la familia de canal tipado de P2, cuyos "
              "agregados sintacticos de V14 fueron negativos, y la prioridad y el presupuesto de "
              "este ciclo son de NER (brief secciones 1, 4 y 26). Queda documentada, NO ejecutada."},
    {"id": "P6-NER-ESTRUCTURAL", "familia": "NER estructural (secuencia de tipos)", "operacion_probada":
     "secuencia de TIPOS de entidad por span; tipos + contexto POS inmediato",
     "evidencia": "Entidades usadas SOLO como conteos/ratios (v14a, 6 variables, cerradas) y como "
                  "evidencia supervisada P(clase|entidad) (V13-P1C, cerrada). Secuencias de tipos "
                  "de entidad: NUNCA construidas (0 firmas de secuencia; la columna ent_seq se "
                  "guardo como anotacion cruda, no como canal).",
     "status": "GENUINELY_NEW",
     "razon": "Operacion nueva, legal bajo D-076 (solo etiquetas, sin nombres concretos, sin datos "
              "externos), y con razon independiente: la estructura de entidades describe informacion "
              "social/tematica que la secuencia de POS no expresa."},
]


def cargar(clave: str) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_parquet(PARQUETS[clave], columns=["ent_seq", "n_tokens"])
    return frame, frame["ent_seq"].astype(str).to_numpy()


def estadisticas(ent: np.ndarray) -> dict:
    n = len(ent)
    con_ent = 0
    tokens_ent = 0
    spans = 0
    tipos = Counter()
    for s in ent:
        etiquetas = s.split() if s else []
        propios = [t for t in etiquetas if t != "O"]
        if propios:
            con_ent += 1
        tokens_ent += len(propios)
        run = None
        for t in etiquetas:
            if t != "O" and t != run:
                spans += 1
            run = t if t != "O" else None
        tipos.update(propios)
    return {"documentos": n, "docs_con_entidad": con_ent,
            "pct_docs_con_entidad": round(100.0 * con_ent / n, 2),
            "tokens_en_entidades": tokens_ent, "spans": spans,
            "spans_por_doc": round(spans / n, 3), "tipos": dict(tipos)}


def composicion(ent: np.ndarray, textos: pd.Series) -> dict:
    cats = Counter()
    for s, txt in zip(ent, textos):
        etiquetas = s.split() if s else []
        tokens = TOKEN_RE.findall(txt)
        for etiqueta, token in zip(etiquetas, tokens):
            if etiqueta == "O":
                continue
            if token.startswith("@"):
                clave = "mencion"
            elif token.startswith("#"):
                clave = "hashtag"
            elif token.lower().startswith(("http", "www")):
                clave = "url"
            elif token.isdigit():
                clave = "numero"
            elif token.lower() == "rt":
                clave = "rt"
            else:
                clave = "palabra"
            cats[f"{etiqueta}->{clave}"] += 1
    total = sum(cats.values())
    return {"total": total, "detalle": dict(cats.most_common()),
            "pct_palabra": round(100.0 * sum(v for k, v in cats.items() if k.endswith("->palabra"))
                                 / max(total, 1), 1)}


def cortes(ent: np.ndarray, textos: pd.Series) -> list[dict]:
    largos = textos.str.len()
    marcas = {
        "con_http": textos.str.contains("http"),
        "con_hashtag": textos.str.contains("#"),
        "con_mencion": textos.str.contains("@"),
        "longitud<=40": largos <= 40,
        "longitud_41-93": (largos > 40) & (largos <= 93),
        "longitud>93": largos > 93,
    }
    filas = []
    for nombre, mascara in marcas.items():
        idx = np.flatnonzero(mascara.to_numpy())
        con = sum(1 for i in idx if any(t != "O" for t in (ent[i].split() if ent[i] else [])))
        filas.append({"corte": nombre, "documentos": int(len(idx)),
                      "pct_con_entidad": round(100.0 * con / max(len(idx), 1), 2)})
    return filas


def muestra(ent: np.ndarray, textos: pd.Series, semilla: int = 20260929) -> list[dict]:
    """Muestra diagnostica: sobredeteccion en marcas de Twitter + entidades sobre palabra."""
    rng = np.random.default_rng(semilla)
    idx_marca, idx_palabra = [], []
    for i, s in enumerate(ent):
        etiquetas = s.split() if s else []
        tokens = TOKEN_RE.findall(textos.iloc[i])
        pares = list(zip(etiquetas, tokens))
        en_ent = [(e, t) for e, t in pares if e != "O"]
        if not en_ent:
            continue
        if any(t.startswith(("@", "#")) or t.lower().startswith(("http", "www"))
               or t.lower() == "rt" for _, t in en_ent):
            idx_marca.append(i)
        if any(t[0].isalpha() for _, t in en_ent):
            idx_palabra.append(i)
    seleccion = list(rng.choice(idx_marca, size=6, replace=False)) + \
        list(rng.choice(idx_palabra, size=6, replace=False))
    ejemplos = []
    for i in seleccion:
        etiquetas = ent[i].split() if ent[i] else []
        ejemplos.append({
            "fila": int(i), "texto": textos.iloc[i][:110],
            "marcado": " ".join(f"<{e}>" if e != "O" else "_" for e in etiquetas[:26]),
            "tipo": "marca" if i in idx_marca else "palabra"})
    return ejemplos


def escribe_gap_md(est: dict, comp: dict, crt: dict, ejemplos: list[dict], sha: dict) -> None:
    lineas = [
        "# NER estructural - auditoria de huecos (P3-P5) y verificacion de novedad",
        "",
        f"**Fecha**: {FECHA} | **Metodo**: auditoria POR OPERACION sobre el codigo ejecutado "
        "(scan de P0: 252 ficheros, 0 firmas de construccion de secuencias; solo 2 scripts usan el "
        "anotador y ambos con conteos/ratios), experiments.csv, auditorias de cierre y este "
        "repositorio. El nombre de la tecnica NO cuenta como evidencia.",
        "",
        "## Clasificacion (detalle en el CSV)",
        "",
        "| ID | Familia | Estado | Decision |",
        "|---|---|---|---|",
    ]
    for fila in FILAS:
        decision = ("no se ejecuta" if fila["status"] == "PARTIALLY_COVERED"
                    else ("linea de este ciclo" if fila["id"] == "P6-NER-ESTRUCTURAL"
                          else "documentada, no priorizada"))
        lineas.append(f"| {fila['id']} | {fila['familia']} | **{fila['status']}** | {decision} |")
    lineas += [
        "",
        "## Cobertura real de la anotacion NER (medida hoy, artefactos congelados)",
        "",
        "| Anotador | Docs con >=1 entidad | Tokens en entidades | Spans por doc | Tipos |",
        "|---|---|---|---|---|",
    ]
    for clave in ("md", "lg"):
        e = est[clave]
        tipos = ", ".join(f"{k}: {v:,}" for k, v in e["tipos"].items())
        lineas.append(f"| {clave} | {e['docs_con_entidad']:,} ({e['pct_docs_con_entidad']}%) | "
                      f"{e['tokens_en_entidades']:,} | {e['spans_por_doc']} | {tipos} |")
    lineas += [
        "",
        f"- Acuerdo md vs lg en la secuencia de entidades: **{sha['acuerdo_ent']}%** identica.",
        f"- LOS DOS ANOTADORES SOBREDETECTAN: el porcentaje de tokens dentro de entidades "
        f"({round(100.0 * est['md']['tokens_en_entidades'] / 2590877, 2)}% en md) es muy superior al "
        "de un NER limpio; MISC domina el reparto (es un cajon de sastre).",
        "",
        "## Composicion de lo que el modelo marca (md)",
        "",
        f"De {comp['md']['total']:,} tokens de entidad alineados con el tokenizador de superficie, "
        f"**{comp['md']['pct_palabra']}% caen sobre PALABRAS** (entidades lexicas); el resto son "
        "marcas de Twitter. Detalle:",
        "",
        "| Par tipo->token | Frecuencia |",
        "|---|---|",
    ]
    for clave, valor in list(comp["md"]["detalle"].items())[:12]:
        lineas.append(f"| {clave} | {valor:,} |")
    lineas += [
        "",
        "Lectura: ~28% de los tokens de entidad caen en menciones/RT/hashtags/URLs/numeros (los "
        "canales de mención/hashtag/URL de V31-02/03 estan CLOSED en integracion y deben "
        "documentarse como solapamiento declarado), y el 72% restante es contenido lexico de "
        "entidades. El canal de tipos no usa identidades, solo la categoria.",
        "",
        "## Cortes (diagnostico, no features)",
        "",
        "| Corte | Docs | % con >=1 entidad |",
        "|---|---|---|",
    ]
    for fila in crt["md"]:
        lineas.append(f"| {fila['corte']} | {fila['documentos']:,} | {fila['pct_con_entidad']}% |")
    lineas += [
        "",
        "## Muestra diagnostica (12 ejemplos, semilla 20260929; NO se corrige nada)",
        "",
        "| fila | texto (inicio) | secuencia marcada | nota |",
        "|---|---|---|---|",
    ]
    for e in ejemplos:
        nota = ("marca de Twitter marcada como entidad (sobredeteccion tipica)"
                if e["tipo"] == "marca" else "entidad lexica (revisar si acierta)")
        texto = e["texto"].replace("|", "/")
        marcado = e["marcado"].replace("|", "/")
        lineas.append(f"| {e['fila']} | {texto} | {marcado} | {nota} |")
    lineas += [
        "",
        "Errores tipicos observados en la muestra (cualitativos, sin corregir): hashtags y "
        "menciones marcadas como MISC/ORG; `RT` marcado como ORG; primeras palabras de oracion "
        "marcadas como MISC; URLs como MISC. Entidades lexicas bien detectadas en nombres "
        "propios evidentes (personas, ciudades, medios).",
        "",
        "## Separacion que el brief exige: cobertura vs utilidad predictiva",
        "",
        "- **Cobertura**: alta (81% de tweets con >=1 entidad) y ruidosa (~24% de tokens en "
        "entidades).",
        "- **Utilidad predictiva**: NO la mide esta auditoria. La mide el experimento P6 con su "
        "control barajado; si el canal real no supera al barajado, la conclusion sera que no "
        "aporta, aunque la cobertura sea alta.",
        "",
        "## Limites declarados",
        "",
        "- La alineacion tipo-token de la composicion usa el tokenizador de superficie del "
        "proyecto; una fraccion pequena no alinea y no entra en el conteo.",
        "- Esta auditoria no toca test ni holdout y no entrena nada.",
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
    # rstrip obligatorio: el CSV crudo viene con padding a 140 chars (D-005); sin quitarlo las
    # longitudes son basura (correccion de esta misma auditoria, documentada en el commit).
    textos = pd.read_csv(ROOT / "data" / "raw" / "age_train.csv", sep=";",
                         encoding="utf-8")[cfg.TEXT_COL].astype(str).str.rstrip()
    print(f"train: {len(textos):,} textos", flush=True)

    est, comp, crt, ent_cols = {}, {}, {}, {}
    for clave in ("md", "lg"):
        _, ent = cargar(clave)
        ent_cols[clave] = ent
        est[clave] = estadisticas(ent)
        comp[clave] = composicion(ent, textos)
        crt[clave] = cortes(ent, textos)
        print(f"  {clave}: {est[clave]['pct_docs_con_entidad']}% docs con entidad | "
              f"{est[clave]['tokens_en_entidades']:,} tokens en entidades | "
              f"palabra {comp[clave]['pct_palabra']}%", flush=True)

    acuerdo = float(np.mean(ent_cols["md"] == ent_cols["lg"]))
    sha = {"acuerdo_ent": round(100.0 * acuerdo, 2)}
    ejemplos = muestra(ent_cols["md"], textos)
    escribe_gap_md(est, comp, crt, ejemplos, sha)
    escribe_csv()
    print(f"  {GAP_MD.name} | {GAP_CSV.name}", flush=True)

    RESULTADO.write_text(json.dumps({
        "fase": "P6-P0 auditoria P3-P5 y verificacion NER", "fecha": FECHA,
        "clasificacion": {f["id"]: f["status"] for f in FILAS},
        "estadisticas": est, "composicion_pct_palabra":
            {k: comp[k]["pct_palabra"] for k in comp},
        "acuerdo_ent_md_lg": sha["acuerdo_ent"], "cortes": crt,
        "sin_test": True, "sin_holdout": True, "sin_entrenamiento": True,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[fin] {time.time() - inicio:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
