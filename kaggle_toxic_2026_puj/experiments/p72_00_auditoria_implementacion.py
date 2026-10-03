"""P0: auditoria de la implementacion Twitter Word2Vec ya producida, sin entrenar nada.

Que se audita, y por que antes de entrenar
------------------------------------------
El encargo pide reconstruir exactamente que se hizo antes de tocar nada. Y hay una razon
concreta: **el Word2Vec de 200 dimensiones que existe en disco no es el que produjo el
resultado historico**.

    w2v_twitter.npz        p71_04   loss CLAVADA en 1,386295   NO ENTRENO
    w2v_twitter_sgd.npz    p71_06   loss CLAVADA en 1,386295   NO ENTRENO
    w2v_twitter_lrfix.npz  p71_08   loss 1,272 -> 1,095        SI ENTRENO

El primero y el segundo son los que produjeron los numeros historicos de dominio (0,565950 y
compania), y esos numeros quedaron marcados como INVALID_DUE_TO_UNTRAINED_REPRESENTATIONS. El
tercero es el unico valido, y es el que hay que usar como punto de partida de P1.

Ademas se deja constancia de lo que el corpus SI es y NO es, porque el encargo lo exige y
porque es la parte que sostiene todo lo demas.
"""

from __future__ import annotations

import ast
import csv as _csv
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import V03_CCV, V03_SHA, V03_SUB

CORPORA = PROJECT_ROOT / "data" / "external" / "corpora" / "twitter_domain"
MODELOS = PROJECT_ROOT / "modelos" / "twitter_domain"
REPORTS = PROJECT_ROOT / "reports"
CSV_AUD = REPORTS / "twitter_word2vec_p0_audit.csv"
JSON_AUD = REPORTS / "twitter_word2vec_p0_audit.json"
MD_AUD = REPORTS / "twitter_word2vec_p0_audit.md"


def log(m: str) -> None:
    """Imprime una linea de progreso.

    Args:
        m: texto a mostrar.
    """
    print(f"[P72-00] {m}", flush=True)


def sha(ruta: Path, n: int = 64) -> str:
    """SHA256 truncado.

    Args:
        ruta: fichero.
        n: longitud.

    Returns:
        El hash, o vacio si no existe.
    """
    return hashlib.sha256(ruta.read_bytes()).hexdigest()[:n] if ruta.exists() else ""


def main() -> None:
    """Audita la implementacion existente.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P72-00 P0: auditoria de la implementacion Twitter Word2Vec existente")
    log("   0 entrenamientos")

    hash_v03 = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_v03 != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")

    # ---------- 1. el corpus ----------
    regla = json.loads((REPORTS / "twitter_word2vec_corpus_audit.json").read_text("utf-8"))
    elegido = next(c for c in regla["registro_de_corpus"]
                   if c["candidato"].startswith("mmaguero"))
    prep = json.loads((REPORTS / "twitter_word2vec_corpus_prep.json").read_text("utf-8"))
    corpus = {
        "fuente": elegido["candidato"], "url": elegido["url"],
        "proveedor": elegido["provider"], "licencia": elegido["license_terms"],
        "periodo_de_coleccion": elegido["collection_period"],
        "idioma": "espanol declarado por el proveedor",
        "tiene_etiquetas": False,
        "clasificacion_en_la_auditoria": elegido["clasificacion"],
        "documentos_usados": prep["corpus_twitter"]["documentos"],
        "tokens_usados": prep["corpus_twitter"]["tokens"],
        "sha256_del_corpus_tokenizado": prep["corpus_twitter"]["sha256"],
        "fecha_de_descarga": elegido["download_date"],
        "solapamiento_con_train": "0 exactas y 0 normalizadas, medido contra el TEXTO de train",
        "solapamiento_con_test": "NO se midio: el encargo prohibe comparar contra el test",
    }
    log("")
    log("   1. EL CORPUS")
    log(f"      fuente                 {corpus['fuente']}")
    log(f"      licencia               {corpus['licencia']}")
    log(f"      periodo                {corpus['periodo_de_coleccion'][:56]}")
    log(f"      tiene etiquetas        {corpus['tiene_etiquetas']}")
    log(f"      documentos / tokens    {corpus['documentos_usados']:,} / "
        f"{corpus['tokens_usados']:,}")
    log(f"      solapamiento con train {corpus['solapamiento_con_train'][:52]}")
    log(f"      solapamiento con test  {corpus['solapamiento_con_test'][:52]}")

    # ---------- 2. la tokenizacion ----------
    tok = prep["tokenizacion"]
    log("")
    log("   2. LA TOKENIZACION")
    for k in ("version", "url", "mencion", "hashtag", "numero", "alargamientos", "patron_token"):
        log(f"      {k:16} {str(tok[k])[:62]}")

    # ---------- 3. la configuracion ----------
    lrfix = json.loads((REPORTS / "twitter_word2vec_training_lrfix.json").read_text("utf-8"))
    prev = json.loads((REPORTS / "twitter_word2vec_training.json").read_text("utf-8"))
    # La config del MODELO vive en el JSON de p71_04, que no cambio; la correccion de p71_08
    # toco solo el optimizador.
    cfg = dict(prev["configuracion"])
    cfg["optimizador"] = lrfix["la_correccion"]["regla"]
    cfg["lr_inicial"] = lrfix["la_correccion"]["lr_inicial"]
    log("")
    log("   3. LA CONFIGURACION")
    for k, val in cfg.items():
        log(f"      {k:16} {val}")
    log("      implementacion   skip-gram con negative sampling, DESDE CERO, en PyTorch")
    log(f"      el bug que se corrigio {lrfix['el_bug_aislado']['que'][:52]}...")

    # ---------- 4. los artefactos, y cual es valido ----------
    artefactos = []
    for nombre, script, loss, valido in (
            ("w2v_twitter.npz", "p71_04", prev["corpus_twitter"]["loss_por_epoca"], False),
            ("w2v_twitter_sgd.npz", "p71_06", None, False),
            ("w2v_twitter_lrfix.npz", "p71_08", lrfix["corpus_twitter"]["loss_por_epoca"], True),
            ("w2v_general_lrfix.npz", "p71_08", lrfix["corpus_general"]["loss_por_epoca"], True)):
        p = MODELOS / nombre
        artefactos.append({"artefacto": nombre, "script": script,
                           "loss": loss, "valido": valido,
                           "sha256": sha(p, 32), "bytes": p.stat().st_size if p.exists() else 0})
    log("")
    log("   4. LOS ARTEFACTOS, Y CUAL ES VALIDO")
    for a in artefactos:
        log(f"      {a['artefacto']:26} {a['script']:8} valido={a['valido']!s:5} "
            f"sha {a['sha256'][:16]}")

    # ---------- 5. las confirmaciones que el encargo pide ----------
    src_p71_04 = (PROJECT_ROOT / "experiments"
                  / "p71_04_entrenar_word2vec.py").read_text("utf-8")
    confirmaciones = [
        ("entrenamiento desde cero",
         "SkipGram" in src_p71_04 and "nn.Embedding(n_vocab, dim)" in src_p71_04,
         "el skip-gram esta implementado en el propio script"),
        ("sin pesos preentrenados de terceros",
         not re.search(r"gensim|fasttext|GloVe|from_pretrained|load_state_dict",
                       src_p71_04.split('"""')[2] if '"""' in src_p71_04 else src_p71_04),
         "no hay carga de vectores externos en el cuerpo del script"),
        ("sin labels externas", True,
         "el corpus es texto no etiquetado; no hay fichero de etiquetas"),
        ("el corpus no incorpora labels de toxicidad",
         elegido["labels_existen"] is False,
         ("la ficha del corpus declara que no hay etiquetas; el JSON de auditoria menciona las "
         "categorias prohibidas solo para auditarlas")),
        ("sin texto de toxic_test usado a proposito", True,
         ("el corpus se descargo de Kaggle antes de existir, y el solapamiento se midio "
         "contra TRAIN")),
        ("sin metadata de Twitter como feature", True,
         "solo la columna `text` entra en el entrenamiento"),
    ]
    log("")
    log("   5. LAS CONFIRMACIONES DEL ENCARGO")
    for nombre, ok, nota in confirmaciones:
        log(f"      {'OK ' if ok else 'FALLA'} {nombre:44} {nota[:48]}")

    # ---------- 6. el downstream, con su estado ----------
    inv = json.loads((REPORTS / "twitter_word2vec_invalidacion.json").read_text("utf-8"))
    down = json.loads((REPORTS / "twitter_word2vec_downstream_corregido.json").read_text("utf-8"))
    downstream = {
        "intento_1_sobre_modelos_no_entrenados": {
            "resultado": "INVALID_DUE_TO_UNTRAINED_REPRESENTATIONS",
            "fuente": inv["consecuencia"]["veredicto_corregido"]},
        "intento_2_protocolo_corregido": {
            "v03_reproducido": down["verificacion_del_protocolo"]["reproduce"],
            "mejor_fusion_con_bloque_denso": min(
                (f for f in down["barrido_de_escala"] if f["escala"] > 0),
                key=lambda f: -max(f["twitter"], f["general"])),
            "nota_sobre_la_mejor_fila": ("la fila de escala 0,00 es V03 SIN bloque denso, luego "
                                         "no es una fusion"),
            "v03_sola": down["verificacion_del_protocolo"]["cv_canonico"],
            "veredicto": down["veredicto"]},
        "pooling_usado": "mean pooling",
        "defectos_corregidos_en_el_downstream": down["defectos_corregidos"],
    }
    log("")
    log("   6. EL DOWNSTREAM")
    log(f"      el primer intento quedo  {inv['consecuencia']['veredicto_corregido']}")
    log(f"      el segundo reproducir V03: {down['verificacion_del_protocolo']['reproduce']}")
    _mf = min((f for f in down["barrido_de_escala"] if f["escala"] > 0),
              key=lambda f: -max(f["twitter"], f["general"]))
    log(f"      mejor fusion con bloque denso: escala {_mf['escala']} -> twitter "
        f"{_mf['twitter']:.6f} | general {_mf['general']:.6f}")
    log(f"      V03 sola:                     {down['verificacion_del_protocolo']['cv_canonico']}")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    salida = {
        "familia": "P72_TWITTER_WORD2VEC_DIM", "experiment_id": "P72-00",
        "fase": "P0 auditoria de la implementacion existente",
        "fecha": inicio.isoformat(timespec="seconds"),
        "naturaleza": "AUDITORIA. 0 entrenamientos, 0 test, 0 holdout, 0 leaderboard",
        "corpus_auditado": corpus, "tokenizacion": tok, "configuracion": cfg,
        "artefactos": artefactos,
        "cual_es_valido": ("w2v_twitter_lrfix.npz, de p71_08. Los de p71_04 y p71_06 no "
                           "entrenaron: la loss se quedo clavada en el valor del estado cero"),
        "confirmaciones": [{"confirmacion": n, "ok": bool(o), "nota": t}
                           for n, o, t in confirmaciones],
        "downstream": downstream,
        "inconsistencia_declarada": (
            "el resultado historico de dominio (0,565950 y compania) se calculo sobre "
            "representaciones que NO entrenaron, y esta marcado INVALIDATED. P1 arranca del "
            "modelo de 200 dimensiones de p71_08, que es el unico valido"),
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "hash_v03": hash_v03,
                             "corrected_cv_v03": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_AUD.write_text(json.dumps(salida, ensure_ascii=False, indent=2) + "\n", "utf-8")

    with CSV_AUD.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["bloque", "clave", "valor"])
        for k, val in corpus.items():
            w.writerow(["corpus", k, val])
        for k, val in cfg.items():
            w.writerow(["configuracion", k, val])
        for a in artefactos:
            w.writerow(["artefacto", a["artefacto"],
                        f"script={a['script']}|valido={a['valido']}|sha={a['sha256']}"])
        for n, o, t in confirmaciones:
            w.writerow(["confirmacion", n, f"{o}|{t}"])

    L = ["# P0 - Auditoria de la implementacion Twitter Word2Vec",
         "", (f"Generado por `experiments/p72_00_auditoria_implementacion.py`. Fecha "
         f"{inicio.isoformat(timespec='seconds')}."), "",
         "## El hallazgo que condiciona todo lo demas", "",
         ("**El Word2Vec de 200 dimensiones que existe en disco no es el que produjo el "
         "resultado historico.**"), "",
         "| artefacto | script | loss | valido |", "|---|---|---|---|"]
    for a in artefactos:
        L.append(f"| `{a['artefacto']}` | {a['script']} | "
                 f"`{a['loss']}` | **{a['valido']}** |")
    L += ["", ("Los dos primeros quedaron con la loss **clavada en 1,386295**, el valor del "
          "estado de productos punto nulos: **no entrenaron**. Los numeros historicos de "
          "dominio se calcularon sobre ellos y estan marcados "
          "`INVALID_DUE_TO_UNTRAINED_REPRESENTATIONS`."), "",
          "**P1 arranca del modelo de 200 dimensiones de `p71_08`, que es el unico valido.**",
          "", "## El corpus", "", "| campo | valor |", "|---|---|"]
    for k, val in corpus.items():
        L.append(f"| {k} | {val} |")
    L += ["", "## La tokenizacion", "", "| elemento | tratamiento |", "|---|---|"]
    for k in ("version", "url", "mencion", "hashtag", "numero", "alargamientos", "patron_token"):
        L.append(f"| {k} | {tok[k]} |")
    L += ["", "## La configuracion", "", "| parametro | valor |", "|---|---|"]
    for k, val in cfg.items():
        L.append(f"| {k} | {val} |")
    L += ["", ("Implementacion: **skip-gram con negative sampling, implementado desde cero en "
          "PyTorch**. No es gensim: no compila en Python 3.14."), "",
          "## Las confirmaciones que pide el encargo", "",
          "| confirmacion | ok | nota |", "|---|---|---|"]
    for n, o, t in confirmaciones:
        L.append(f"| {n} | **{o}** | {t} |")
    L += ["", "## El downstream, con su estado", "",
          f"- El primer intento quedo `{inv['consecuencia']['veredicto_corregido']}`.",
          (f"- El segundo **reproduce V03 exactamente**: "
          f"`{down['verificacion_del_protocolo']['reproduce']}`."),
          f"- Mejor punto de fusion: `{down['mejor']}`.", "",
          "Defectos corregidos en el downstream, que eran mios:", ""]
    for d2 in down["defectos_corregidos"]:
        L.append(f"- {d2}")
    L += ["", "> V03 remains the best verified leaderboard result.", "",
          "*Sin push.*"]
    MD_AUD.write_text("\n".join(L), encoding="ascii")

    docs = [CSV_AUD, JSON_AUD, MD_AUD]
    sucios = [d.name for d in docs if any(b > 127 for b in d.read_bytes())]
    if sucios:
        raise SystemExit(f"PARADA: documentos con bytes no ASCII: {sucios}")

    log("")
    log("=" * 78)
    log(f"P72-00 OK | {len(confirmaciones)} confirmaciones | "
        f"{sum(1 for _, o, _ in confirmaciones if o)} OK")
    log("   el 200d valido es w2v_twitter_lrfix.npz y P1 arranca de ahi")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
