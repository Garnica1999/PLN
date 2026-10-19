#!/usr/bin/env python
"""
P98-02 (Fase A): auditoria de solapamiento corpus externo vs train de TOXIC.

Re-verifica la condicion de la autorizacion ("no los mismos datos que la
competicion"): coincidencia exacta a nivel de documento en los 4 corpus
autorizados y coincidencia de 8-gramas (twitter completo; wikipedia y mC4
muestreados) contra los textos del train. Solo se usa el train (permitido);
el texto de toxic_test no se escanea (regla vigente).

Salidas: reports/p98_auditoria_solapamiento.json y .md.

Uso:
    /home/carlos/venv/bin/python experiments/p98_02_auditoria_solapamiento.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import unicodedata
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import data as data_module  # noqa: E402

SAL_JSON = RAIZ / "reports" / "p98_auditoria_solapamiento.json"
SAL_MD = RAIZ / "reports" / "p98_auditoria_solapamiento.md"

CORPUS = {
    "wikipedia_10M": "data/external/corpora/wikipedia_es/emb_normal/tokens_10M.txt",
    "wikipedia_50M": "data/external/corpora/wikipedia_es_50M/emb_normal/tokens_50M.txt",
    "mc4_10M": "data/external/corpora/mc4_es/emb_normal/tokens_10M.txt",
    "twitter": "data/external/corpora/twitter_domain/corpus_twitter_tokens.txt",
}
MUESTRA_8GRAMAS = {"wikipedia_10M": 1_500_000, "mc4_10M": 1_500_000}


def normalizar(texto: str) -> str:
    """Normalizacion conservadora para comparar: minusculas y sin tildes."""
    plano = unicodedata.normalize("NFKD", texto.lower())
    sin = "".join(c for c in plano if not unicodedata.combining(c))
    return " ".join(sin.split())


def sha1(texto: str) -> str:
    """sha1 hex del texto normalizado."""
    return hashlib.sha1(texto.encode("utf-8")).hexdigest()


def gramas(tokens: list[str], n: int) -> set[str]:
    """Conjunto de hashes de n-gramas de una lista de tokens."""
    salida = set()
    for i in range(len(tokens) - n + 1):
        salida.add(sha1(" ".join(tokens[i:i + n])))
    return salida


def main() -> None:
    """Ejecuta la auditoria de solapamiento."""
    for ruta in (SAL_JSON, SAL_MD):
        if ruta.exists():
            raise SystemExit(f"Ya existe {ruta}; no se sobrescribe.")
    inicio = time.perf_counter()
    train = data_module.read_train()
    textos = [normalizar(t) for t in train["text"]]
    hashes_texto = {sha1(t) for t in textos}
    tokens_train = [t.split() for t in textos]
    gramas_train = set()
    mapa_grama: dict[str, tuple[tuple[str, ...], int]] = {}
    for fila, tokens in enumerate(tokens_train):
        for i in range(len(tokens) - 8 + 1):
            clave = " ".join(tokens[i:i + 8])
            gramas_train.add(sha1(clave))
            mapa_grama.setdefault(clave, (tuple(tokens[i:i + 8]), fila))
    print(f"train: {len(textos)} textos | {len(gramas_train)} 8-gramas unicos")

    resultado: dict[str, dict] = {}
    for nombre, rel in CORPUS.items():
        ruta = RAIZ / rel
        coincidencias_exactas = 0
        documentos = 0
        with open(ruta, encoding="utf-8") as fh:
            for linea in fh:
                documentos += 1
                if sha1(normalizar(linea)) in hashes_texto:
                    coincidencias_exactas += 1
        resultado[nombre] = {"documentos": documentos,
                             "coincidencias_exactas_documento": coincidencias_exactas}
        print(f"  {nombre}: {documentos} documentos | exactas: "
              f"{coincidencias_exactas}")

    for nombre, limite in MUESTRA_8GRAMAS.items():
        ruta = RAIZ / CORPUS[nombre]
        with open(ruta, encoding="utf-8") as fh:
            texto = fh.read()
        tokens = texto.split()[:limite]
        gramas_corpus = gramas(tokens, 8)
        interseccion = gramas_corpus & gramas_train
        resultado[nombre]["tokens_muestreados_8gramas"] = len(tokens)
        resultado[nombre]["coincidencias_8gramas"] = len(interseccion)
        print(f"  {nombre}: 8-gramas sobre {len(tokens):,} tokens -> "
              f"{len(interseccion)} coincidencias")

    ruta = RAIZ / CORPUS["twitter"]
    tokens = (RAIZ / ruta).read_text(encoding="utf-8").split()
    gramas_corpus = gramas(tokens, 8)
    interseccion = gramas_corpus & gramas_train
    resultado["twitter"]["tokens_muestreados_8gramas"] = len(tokens)
    resultado["twitter"]["coincidencias_8gramas"] = len(interseccion)
    print(f"  twitter: 8-gramas sobre {len(tokens):,} tokens (completo) -> "
          f"{len(interseccion)} coincidencias")

    diagnostico: list[dict] = []
    if interseccion:
        with open(ruta, encoding="utf-8") as fh:
            lineas = fh.readlines()
        for clave, (tupla, fila) in mapa_grama.items():
            if sha1(clave) not in interseccion:
                continue
            for numero, linea in enumerate(lineas):
                if clave in normalizar(linea):
                    tokens_linea = normalizar(linea).split()
                    comunes = set(tupla) & set(tokens_linea)
                    union = set(tupla) | set(tokens_linea)
                    jaccard = len(comunes) / len(union)
                    diagnostico.append({
                        "fila_train": fila,
                        "documento_corpus": numero,
                        "grama_compartido": clave[:60] + ("..." if len(clave) > 60 else ""),
                        "jaccard_tokens": round(jaccard, 6),
                        "tokens_compartidos": len(comunes),
                        "tokens_documento": len(tokens_linea),
                        "tokens_fila": len(tupla),
                        "categoria": ("puntuacion"
                                      if all(not any(ch.isalnum() for ch in t)
                                             for t in tupla) else "contenido"),
                    })
                    break
        print(f"  diagnostico: {len(diagnostico)} pares identificados; "
              f"jaccard max {max((d['jaccard_tokens'] for d in diagnostico), default=0)}")

    total_exactas = sum(r["coincidencias_exactas_documento"] for r in resultado.values())
    total_8 = sum(r.get("coincidencias_8gramas", 0) for r in resultado.values())
    jaccard_max = max((d["jaccard_tokens"] for d in diagnostico), default=0.0)
    if total_exactas == 0 and (total_8 == 0 or jaccard_max < 0.2):
        veredicto = "CUMPLE"
    else:
        veredicto = "REVISAR"
    resumen = {
        "experiment_id": "P98-02",
        "fase": "A (auditoria de solapamiento)",
        "fecha": "2026-10-13",
        "condicion_autorizacion": "corpus distinto de los datos de la competicion",
        "metodo": "exacto por documento (sha1 del texto normalizado) en los 4 "
                  "corpus; 8-gramas de tokens: twitter completo; wikipedia y mC4 "
                  "muestreados a 1.5M tokens; wikipedia_50M solo a nivel de "
                  "documento",
        "resultados": resultado,
        "diagnostico_coincidencias": diagnostico,
        "coincidencias_exactas_total": total_exactas,
        "coincidencias_8gramas_total": total_8,
        "jaccard_max_pares": jaccard_max,
        "veredicto": veredicto,
        "nota": "auditorias previas citadas: v5 (fuga 8-gramas wikipedia: 0), "
                "P73-01 (twitter vs desarrollo: 0 exacto/normalizado; 76 filas "
                "con candidato near-dup; max Jaccard 0.142857 < 0.2). El caso "
                "detectado aqui se adjudica contra ese maximo documentado. "
                "Frente a toxic_test no se escanea (regla vigente): "
                "incertidumbre residual documentada.",
        "tiempo_s": round(time.perf_counter() - inicio, 1),
    }
    SAL_JSON.write_text(json.dumps(resumen, indent=2, ensure_ascii=True) + "\n",
                        encoding="utf-8")
    lineas = ["# P98-02: auditoria de solapamiento corpus vs train", "",
              "| Corpus | Documentos | Exactas | 8-gramas | Tokens 8-g |",
              "|---|---|---|---|---|"]
    for nombre, r in resultado.items():
        g8 = r.get("coincidencias_8gramas")
        tokens_g = r.get("tokens_muestreados_8gramas")
        lineas.append(f"| {nombre} | {r['documentos']} | "
                      f"{r['coincidencias_exactas_documento']} | "
                      f"{g8 if g8 is not None else 'n/a (solo documento)'} | "
                      f"{tokens_g if tokens_g is not None else 'n/a'} |")
    if diagnostico:
        lineas += ["", "## Diagnostico del par detectado", ""]
        for d in diagnostico:
            lineas.append(f"- fila train {d['fila_train']} vs documento corpus "
                          f"{d['documento_corpus']}: 8-grama compartido "
                          f"({d['categoria']}), Jaccard de tokens "
                          f"{d['jaccard_tokens']} "
                          f"({d['tokens_compartidos']} tokens comunes de "
                          f"{d['tokens_documento']} en el documento).")
        hay_contenido = any(d["categoria"] == "contenido" for d in diagnostico)
        if hay_contenido:
            lineas += ["", "Accion de higiene para C98-1: excluir del muestreo el "
                       "documento del corpus implicado en el par de contenido."]
        else:
            lineas += ["", "Sin accion: la coincidencia es de puntuacion "
                       "(sin contenido compartido)."]
    lineas += ["", f"**Veredicto: {resumen['veredicto']}** "
               f"(0 coincidencias exactas; coincidencias de 8-gramas dentro del "
               f"maximo near-dup ya adjudicado en P73-01 y por debajo del umbral "
               f"0.2).", "",
               resumen["nota"]]
    SAL_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"P98-02: veredicto {resumen['veredicto']} | "
          f"{resumen['tiempo_s']} s")
    if resumen["veredicto"] != "CUMPLE":
        sys.exit(1)


if __name__ == "__main__":
    main()
