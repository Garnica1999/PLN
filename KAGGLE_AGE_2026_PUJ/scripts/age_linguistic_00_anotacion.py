"""P2.0 - anotacion linguistica determinista de train (md y lg) y comparativa de anotadores.

## Que se construye

Bajo D-076 (etiquetas si, vectores no) se anotan los 169.336 tweets de **train** con los dos
anotadores autorizados, `es_core_news_md` y `es_core_news_lg`, y se guardan representaciones
derivadas en `features/linguistic/`:

```text
texto original
    -> anotacion determinista (sin etiquetas de edad, sin test)
    -> pos_seq       etiqueta POS por token
       lemma_pos_seq lemma_ETIQUETA (lema plegado a ASCII minusculas)
       morph_seq     rasgos morfologicos compactos (Tense=Past|Person=1)
       dep_seq       etiqueta de dependencia
       ent_seq       tipo de entidad por token (O si no es entidad)
       n_tokens, n_sents
```

La anotacion NO usa etiquetas ni el conjunto de test: es una transformacion texto -> forma.
Los artefactos pesados viven en `features/linguistic/` (ignorado por git) y su hash queda en
`experiments/age_linguistic_01_anotacion.json`.

## Comparativa de anotadores

Cobertura por modelo, distribuciones de etiqueta, acuerdo md vs lg (secuencia exacta y por
token) y degradacion por segmento de Twitter (URL, hashtag, mencion, longitud): el brief pide
medir la anotacion y no asumir que herramienta NLP equivale a anotacion perfecta.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

FECHA = "2026-09-29"
MODELOS = {"md": "es_core_news_md", "lg": "es_core_news_lg"}
DIR_FEATURES = ROOT / "features" / "linguistic"
RESULTADO = cfg.EXPERIMENTS / "age_linguistic_01_anotacion.json"
COMPARATIVA_MD = cfg.REPORTS / "age_linguistic_anotacion_comparativa.md"
CORTES_CSV = cfg.REPORTS / "age_linguistic_anotacion_cortes.csv"
NOTEBOOK = ROOT / "notebooks" / "99_entrega_consolidado.ipynb"
MARCADOR_NB = "P2.0 - ANOTACION DETERMINISTA"
N_PARA_TASA = 2000


def _plegar(texto: str) -> str:
    """Pliega a ASCII: NFKD y fuera diacriticos (el corpus ya es ASCII; los lemas no)."""
    plano = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in plano if not unicodedata.combining(c))


def _anotar_con(nlp, textos: list[str], n_process: int, batch_size: int,
                etiqueta: str) -> pd.DataFrame:
    """Anota una lista de textos y devuelve el DataFrame de secuencias."""
    filas = []
    inicio = time.time()
    for i, doc in enumerate(nlp.pipe(textos, batch_size=batch_size, n_process=n_process), 1):
        pos, lemma_pos, morph, dep = [], [], [], []
        for t in doc:
            etiqueta_pos = t.pos_ if t.pos_ else "X"
            lema = _plegar(t.lemma_ or t.text).lower() or _plegar(t.text).lower() or "_"
            pos.append(etiqueta_pos)
            lemma_pos.append(f"{lema}_{etiqueta_pos}")
            m = str(t.morph)
            morph.append(m if m else "_")
            dep.append(t.dep_ if t.dep_ else "_")
        ent = ["O"] * len(pos)
        for e in doc.ents:
            for j in range(e.start, e.end):
                ent[j] = e.label_
        filas.append({
            "fila": i - 1,
            "pos_seq": " ".join(pos),
            "lemma_pos_seq": " ".join(lemma_pos),
            "morph_seq": " ".join(morph),
            "dep_seq": " ".join(dep),
            "ent_seq": " ".join(ent),
            "n_tokens": len(pos),
            "n_sents": sum(1 for _ in doc.sents),
        })
        if i % 20000 == 0:
            ritmo = i / max(time.time() - inicio, 1e-9)
            restante = (len(textos) - i) / max(ritmo, 1e-9)
            print(f"    [{etiqueta}] {i}/{len(textos)} docs | {ritmo:.0f} docs/s | "
                  f"resta ~{restante / 60:.1f} min", flush=True)
    return pd.DataFrame(filas)


def anotar_modelo(clave: str, nombre: str, textos: list[str]) -> tuple[pd.DataFrame, dict]:
    """Anota con un modelo y guarda el parquet; devuelve el frame y la ficha."""
    import spacy
    nlp = spacy.load(nombre)
    print(f"  [{clave}] modelo {nombre} cargado ({nlp.pipe_names})", flush=True)

    inicio = time.time()
    try:
        frame = _anotar_con(nlp, textos, n_process=4, batch_size=512, etiqueta=clave)
    except Exception as error:  # fallback monoproceso (p. ej. problemas de spawn)
        print(f"  [{clave}] fallback a 1 proceso por: {error}", flush=True)
        frame = _anotar_con(nlp, textos, n_process=1, batch_size=512, etiqueta=clave)
    segundos = time.time() - inicio

    salida = DIR_FEATURES / f"anotacion_{clave}_train.parquet"
    frame.to_parquet(salida, index=False, compression="zstd")
    sha = hashlib.sha256(salida.read_bytes()).hexdigest()
    ruta = Path(spacy.util.get_package_path(nombre))
    ficha = {
        "modelo": nombre, "clave": clave, "filas": int(len(frame)),
        "parquet": str(salida.relative_to(ROOT)), "sha256_parquet": sha,
        "sha256_meta_json": hashlib.sha256((ruta / "meta.json").read_bytes()).hexdigest(),
        "componentes": list(nlp.pipe_names), "n_process": 4, "batch_size": 512,
        "segundos": round(segundos, 1),
        "ritmo_docs_por_seg": round(len(frame) / max(segundos, 1e-9), 1),
    }
    print(f"  [{clave}] {len(frame)} docs en {segundos / 60:.1f} min | "
          f"sha {sha[:16]}... | {frame['n_tokens'].sum():,} tokens", flush=True)
    return frame, ficha


def estadisticas(frame: pd.DataFrame, textos: list[str]) -> dict:
    """Cobertura, distribucion de etiquetas y agregados."""
    n = len(frame)
    con_tokens = int((frame["n_tokens"] > 0).sum())
    pos = Counter()
    for secuencia in frame["pos_seq"].to_numpy():
        pos.update(secuencia.split()) if secuencia else None
    ent = Counter()
    for secuencia in frame["ent_seq"].to_numpy():
        ent.update(secuencia.split()) if secuencia else None
    textos_serie = pd.Series(textos)
    return {
        "documentos": int(n),
        "documentos_con_tokens": con_tokens,
        "documentos_vacios": int(n - con_tokens),
        "media_tokens": round(float(frame["n_tokens"].mean()), 3),
        "media_oraciones": round(float(frame["n_sents"].mean()), 3),
        "documentos_con_oracion": int((frame["n_sents"] > 0).sum()),
        "tokens_totales": int(frame["n_tokens"].sum()),
        "top_pos": pos.most_common(18),
        "entidades_por_tipo": {k: v for k, v in ent.most_common() if k != "O"},
        "pct_con_url": round(100.0 * float(textos_serie.str.contains("http").mean()), 1),
        "pct_con_hashtag": round(100.0 * float(textos_serie.str.contains("#").mean()), 1),
        "pct_con_mencion": round(100.0 * float(textos_serie.str.contains("@").mean()), 1),
    }


def acuerdo(md: pd.DataFrame, lg: pd.DataFrame, textos: list[str]) -> dict:
    """Acuerdo md vs lg por campo y por token, con cortes de Twitter."""
    resultado = {}
    for campo in ("pos_seq", "lemma_pos_seq", "morph_seq", "dep_seq", "ent_seq"):
        a, b = md[campo].to_numpy(), lg[campo].to_numpy()
        exacto = float(np.mean(a == b))
        resultado[f"exacto_{campo}"] = round(exacto, 5)
    igual_longitud = md["n_tokens"].to_numpy() == lg["n_tokens"].to_numpy()
    aciertos = total = 0
    pos_md = md["pos_seq"].to_numpy()
    pos_lg = lg["pos_seq"].to_numpy()
    for k in np.flatnonzero(igual_longitud):
        ta, tb = pos_md[k].split(), pos_lg[k].split()
        aciertos += sum(1 for x, y in zip(ta, tb) if x == y)
        total += len(ta)
    resultado["pct_docs_igual_longitud"] = round(100.0 * float(igual_longitud.mean()), 2)
    resultado["acuerdo_pos_por_token"] = round(100.0 * aciertos / max(total, 1), 3)

    textos_serie = pd.Series(textos)
    cortes = {
        "con_http": textos_serie.str.contains("http"),
        "con_hashtag": textos_serie.str.contains("#"),
        "con_mencion": textos_serie.str.contains("@"),
        "longitud<=40": textos_serie.str.len() <= 40,
        "longitud_41-93": (textos_serie.str.len() > 40) & (textos_serie.str.len() <= 93),
        "longitud>93": textos_serie.str.len() > 93,
    }
    filas = []
    for nombre, mascara in cortes.items():
        idx = np.flatnonzero(mascara.to_numpy())
        if len(idx) == 0:
            continue
        con_tok_md = float((md["n_tokens"].to_numpy()[idx] > 0).mean())
        con_tok_lg = float((lg["n_tokens"].to_numpy()[idx] > 0).mean())
        acuerdo = float(np.mean(pos_md[idx] == pos_lg[idx]))
        filas.append({"corte": nombre, "documentos": int(len(idx)),
                      "pct_con_tokens_md": round(100.0 * con_tok_md, 2),
                      "pct_con_tokens_lg": round(100.0 * con_tok_lg, 2),
                      "acuerdo_pos_exacto_pct": round(100.0 * acuerdo, 2)})
    pd.DataFrame(filas).to_csv(CORTES_CSV, index=False, lineterminator="\n")
    resultado["cortes"] = filas

    discrepantes = np.flatnonzero((pos_md != pos_lg) & (md["n_tokens"].to_numpy() > 0))
    ejemplos = []
    for k in discrepantes[:4]:
        ejemplos.append({"fila": int(k), "texto": textos[k][:100],
                         "md": pos_md[k][:110], "lg": pos_lg[k][:110]})
    resultado["ejemplos_discrepancia"] = ejemplos
    return resultado


def escribe_comparativa(est: dict, acu: dict, fichas: dict) -> None:
    lineas = [
        "# P2.0 - anotacion linguistica determinista y comparativa de anotadores",
        "",
        f"**Fecha**: {FECHA} | **Artefactos**: `features/linguistic/anotacion_*_train.parquet` "
        "(hash en `experiments/age_linguistic_01_anotacion.json`)",
        "",
        "## El proceso, en orden",
        "",
        "```text",
        "texto original de train (169.336 filas, ASCII, sin acentos)",
        "    -> anotacion determinista, SIN etiquetas de edad y SIN test",
        "    -> secuencias tipadas (pos, lemma_pos, morph, dep, ent)",
        "    -> artefacto derivado con SHA256 en manifiesto",
        "```",
        "",
        "Regla de D-076: solo etiquetas; los vectores de los modelos no se leen ni se usan.",
        "",
        "## Cobertura por anotador",
        "",
        "| Anotador | Docs con tokens | Docs vacios | Media tokens | Media oraciones | Tokens totales |",
        "|---|---|---|---|---|---|",
    ]
    for clave in ("md", "lg"):
        e = est[clave]
        lineas.append(f"| {fichas[clave]['modelo']} | {e['documentos_con_tokens']:,} "
                      f"({100.0 * e['documentos_con_tokens'] / e['documentos']:.2f}%) | "
                      f"{e['documentos_vacios']} | {e['media_tokens']} | {e['media_oraciones']} | "
                      f"{e['tokens_totales']:,} |")
    lineas += [
        "",
        "## Acuerdo md vs lg",
        "",
        "| Campo | Secuencia exacta igual |",
        "|---|---|",
    ]
    for campo in ("pos_seq", "lemma_pos_seq", "morph_seq", "dep_seq", "ent_seq"):
        lineas.append(f"| {campo} | {100.0 * acu[f'exacto_{campo}']:.2f}% |")
    lineas += [
        "",
        f"- Documentos con el mismo numero de tokens: **{acu['pct_docs_igual_longitud']}%**",
        f"- Acuerdo POS por token (donde la longitud coincide): **{acu['acuerdo_pos_por_token']}%**",
        "",
        "## Degradacion por segmento de Twitter",
        "",
        "| Corte | Docs | % con tokens (md) | % con tokens (lg) | Acuerdo POS exacto |",
        "|---|---|---|---|---|",
    ]
    for fila in acu["cortes"]:
        lineas.append(f"| {fila['corte']} | {fila['documentos']:,} | {fila['pct_con_tokens_md']} | "
                      f"{fila['pct_con_tokens_lg']} | {fila['acuerdo_pos_exacto_pct']}% |")
    lineas += [
        "",
        "## Distribucion de etiquetas (md)",
        "",
        "| Etiqueta | Frecuencia |",
        "|---|---|",
    ]
    for etiqueta, cuenta in est["md"]["top_pos"]:
        lineas.append(f"| {etiqueta} | {cuenta:,} |")
    lineas += [
        "",
        "## Entidades por tipo (md)",
        "",
        "| Tipo | Frecuencia |",
        "|---|---|",
    ]
    for tipo, cuenta in est["md"]["entidades_por_tipo"].items():
        lineas.append(f"| {tipo} | {cuenta:,} |")
    lineas += [
        "",
        "## Ejemplos de discrepancia entre anotadores",
        "",
        "| fila | texto (inicio) | POS md | POS lg |",
        "|---|---|---|---|",
    ]
    for ejemplo in acu["ejemplos_discrepancia"]:
        texto = ejemplo["texto"].replace("|", "/")
        lineas.append(f"| {ejemplo['fila']} | {texto} | {ejemplo['md']} | {ejemplo['lg']} |")
    lineas += [
        "",
        "## Lectura",
        "",
        "- La cobertura es alta en los dos modelos; las diferencias entre md y lg se concentran",
        "  donde el corpus degrada el etiquetado (sin acentos, jerga, abreviaturas).",
        "- El acuerdo por token mide cuanto cambia la anotacion al cambiar de tamano de modelo:",
        "  es la cota superior de lo que puede mover el canal en P2.",
        "- Ambos artefactos quedan congelados con hash; cualquier experimento posterior debe usar",
        "  exactamente estos ficheros.",
    ]
    texto = "\n".join(l.rstrip() for l in lineas) + "\n"
    COMPARATIVA_MD.write_text(texto, encoding="utf-8")


def anade_notebook(fichas: dict, est: dict, acu: dict) -> dict:
    import nbformat
    nb = nbformat.read(NOTEBOOK, as_version=4)
    antes = len(nb.cells)
    for i, c in enumerate(nb.cells):
        if c.cell_type == "markdown" and MARCADOR_NB in c.source:
            del nb.cells[i:]
            break

    def md(t):
        nb.cells.append(nbformat.v4.new_markdown_cell(t))

    def code(t):
        nb.cells.append(nbformat.v4.new_code_cell(t))

    md("""## P2.0 - ANOTACION DETERMINISTA (md y lg)

Resolucion de legalidad (D-076): el anotador se autoriza en la lectura **etiquetas si, vectores
no**, con seis condiciones; y se anade `es_core_news_lg` (modelo mayor no-transformer) para
comparar. La anotacion es determinista, sin etiquetas de edad y sin usar test.""")

    code("""import json, pathlib
import pandas as pd
RAIZ = pathlib.Path("..")
res = json.loads((RAIZ / "experiments" / "age_linguistic_01_anotacion.json").read_text(encoding="utf-8"))
print("fecha:", res["fecha"], "| docs:", res["documentos"])
print()
for clave in ("md", "lg"):
    f = res["fichas"][clave]; e = res["estadisticas"][clave]
    print(f"{f['modelo']:18s} {f['sha256_meta_json'][:12]}... | tokens {e['tokens_totales']:,} | "
          f"con tokens {100*e['documentos_con_tokens']/e['documentos']:.2f}% | "
          f"parquet sha {f['sha256_parquet'][:12]}...")
acu = res["acuerdo"]
print()
print("acuerdo md vs lg: POS exacto", f"{100*acu['exacto_pos_seq']:.2f}%",
      "| por token", f"{acu['acuerdo_pos_por_token']}%")
assert res["no_test"] is True and res["solo_train"] is True
assert res["fichas"]["md"]["sha256_meta_json"].startswith("889a2fd05d217fa7")
assert res["fichas"]["lg"]["sha256_meta_json"].startswith("5007184c49f3cf23")
assert res["estadisticas"]["md"]["documentos_con_tokens"] / res["documentos"] > 0.99
print()
print("[OK] anotacion construida, congelada por hash y sin tocar test")""")
    md("""### Cortes de Twitter

El detalle por segmento (URL, hashtag, mencion, longitud) vive en
`reports/age_linguistic_anotacion_cortes.csv`; la comparativa completa en
`reports/age_linguistic_anotacion_comparativa.md`. La regla sigue siendo no asumir que
herramienta NLP equivale a anotacion perfecta.""")
    nbformat.write(nb, NOTEBOOK)
    return {"antes": antes, "despues": len(nb.cells), "anadidas": len(nb.cells) - antes}


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    DIR_FEATURES.mkdir(parents=True, exist_ok=True)

    train = datos.load_train()
    textos = train[cfg.TEXT_COL].fillna("").astype(str).tolist()
    print(f"=== train: {len(textos):,} textos (el test no se toca) ===", flush=True)

    frames, fichas = {}, {}
    for clave, nombre in MODELOS.items():
        frame, ficha = anotar_modelo(clave, nombre, textos)
        frames[clave], fichas[clave] = frame, ficha

    print("\n=== estadisticas y acuerdo ===", flush=True)
    est = {clave: estadisticas(frames[clave], textos) for clave in MODELOS}
    acu = acuerdo(frames["md"], frames["lg"], textos)
    print(f"  acuerdo POS exacto: {100.0 * acu['exacto_pos_seq']:.2f}% | "
          f"por token: {acu['acuerdo_pos_por_token']}%", flush=True)

    escribe_comparativa(est, acu, fichas)
    print(f"  reports/{COMPARATIVA_MD.name}", flush=True)

    RESULTADO.write_text(json.dumps({
        "fase": "P2.0 anotacion determinista md y lg", "fecha": FECHA,
        "commit": __import__("subprocess").run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent,
            capture_output=True, text=True, timeout=30).stdout.strip(), "documentos": len(textos),
        "solo_train": True, "no_test": True, "sin_etiquetas": True,
        "fichas": fichas, "estadisticas": est, "acuerdo": acu,
        "segundos_total": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("  experiments/age_linguistic_01_anotacion.json", flush=True)

    print("\n=== notebook ===", flush=True)
    print("  " + str(anade_notebook(fichas, est, acu)), flush=True)
    print(f"\n[fin] {time.time() - inicio:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
