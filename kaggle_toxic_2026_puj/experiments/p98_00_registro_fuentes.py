#!/usr/bin/env python
"""
P98-00 (Fase A): registro de fuentes, licencias y manifiesto de corpus.

Solo lectura sobre los artefactos de corpus (verifica hashes); escribe:
  reports/p98_corpus_registry.csv
  reports/p98_corpus_licenses.md
  artifacts/p98_corpus_manifest.json

No entrena nada. Base de gobernanza de la fase P98 (seccion 3/4/6 del
enunciado). Reglas: no inventar cifras; los campos desconocidos se marcan
"no disponible".

Uso:
    /home/carlos/venv/bin/python experiments/p98_00_registro_fuentes.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SAL_REG = RAIZ / "reports" / "p98_corpus_registry.csv"
SAL_LIC = RAIZ / "reports" / "p98_corpus_licenses.md"
SAL_MAN = RAIZ / "artifacts" / "p98_corpus_manifest.json"

CAB = ["source_id", "source_name", "source_url", "version", "access_date",
       "license", "permission_status", "document_count", "raw_token_count",
       "clean_token_count", "unique_document_count", "language_distribution",
       "deduplication_method", "sample_seed", "sha256", "artifact_path", "notas"]

# Hashes esperados (manifiesto v5 + doc P73), verificados antes de registrar.
HASHES_ESPERADOS = {
    "data/external/corpora/wikipedia_es/emb_normal/tokens_10M.txt":
        "00b839da6fbc377ee5375acc9a236e831e46ecf92cc544188b81aef1a826e871",
    "data/external/corpora/wikipedia_es_50M/emb_normal/tokens_50M.txt":
        "8088bce2a9283fbc835ab62efe3fb25f40af346f9a8003fe3d101619bbc32463",
    "data/external/corpora/wikipedia_es_5M/emb_normal/tokens_5M.txt":
        "62aaa19af45c31f8fe9e48340cac91ab74ba0a6d19b45a1ef6b853a80b310b59",
    "data/external/corpora/mc4_es/emb_normal/tokens_10M.txt":
        "361498bf3eca104d527759de68980ea358002aeeb27e57d0de8562fac79b4f97",
    "data/external/corpora/mixto/emb_normal/tokens_20M.txt":
        "1c562286dabe9dc4534c2acd4a313bcfa61329dfe73e5d30d35b8bf4c36d63c7",
    "data/external/corpora/twitter_domain/corpus_twitter_tokens.txt":
        "1ef872c189d0f405bc1a3f8281cafe08e6b099958e6b8a3816db1460df91f8b9",
}

FILAS = [
    ["C98-SRC-01", "wikipedia_es (muestra 10M)",
     "https://dumps.wikimedia.org/eswiki/latest/", "eswiki dump 2026-09-25",
     "2026-09-25", "CC BY-SA 4.0 (atribucion obligatoria)", "ALLOWED",
     "2847", "no disponible", "10002400", "2847", "es (100%)",
     "sha1 del texto limpio por documento", "no aplica",
     HASHES_ESPERADOS["data/external/corpora/wikipedia_es/emb_normal/tokens_10M.txt"],
     "data/external/corpora/wikipedia_es/emb_normal/tokens_10M.txt",
     "E-CORP-01; 8-gramas vs competicion: 0 coincidencias (manifiesto v5); "
     "sin texto de test"],
    ["C98-SRC-02", "wikipedia_es (muestra 50M)",
     "https://dumps.wikimedia.org/eswiki/latest/", "eswiki dump 2026-09-25",
     "2026-09-25", "CC BY-SA 4.0 (atribucion obligatoria)", "ALLOWED",
     "21529", "no disponible", "50000028", "21529", "es (100%)",
     "sha1 del texto limpio por documento; prefijo 10M identico (verificado)",
     "no aplica",
     HASHES_ESPERADOS["data/external/corpora/wikipedia_es_50M/emb_normal/tokens_50M.txt"],
     "data/external/corpora/wikipedia_es_50M/emb_normal/tokens_50M.txt",
     "E-CORP-08; expansion 50M con RATIONALE registrado; sin descarga nueva"],
    ["C98-SRC-03", "wikipedia_es (muestra 5M)",
     "https://dumps.wikimedia.org/eswiki/latest/", "eswiki dump 2026-09-25",
     "2026-09-25", "CC BY-SA 4.0 (atribucion obligatoria)", "ALLOWED",
     "1694", "no disponible", "5011404", "1694", "es (100%)",
     "sha1 del texto limpio por documento; prefijo exacto de 10M", "no aplica",
     HASHES_ESPERADOS["data/external/corpora/wikipedia_es_5M/emb_normal/tokens_5M.txt"],
     "data/external/corpora/wikipedia_es_5M/emb_normal/tokens_5M.txt",
     "E-CORP-09; disponible como subconjunto del piloto"],
    ["C98-SRC-04", "mC4 es (shard 0 de 2048)",
     "https://huggingface.co/datasets/allenai/c4",
     "allenai/c4 config multilingual/c4-es, shard 0", "2026-09-25",
     "ODC-BY 1.0 (ficha del dataset; contenido derivado de Common Crawl)",
     "ALLOWED_WITH_RESTRICTIONS", "16738", "no disponible", "10000239", "16738",
     "es (100%)", "sha1 del texto limpio; limpieza web especifica", "no aplica",
     HASHES_ESPERADOS["data/external/corpora/mc4_es/emb_normal/tokens_10M.txt"],
     "data/external/corpora/mc4_es/emb_normal/tokens_10M.txt",
     "E-CORP-06; restriccion: atribucion ODC-BY en reportes; familia Common "
     "Crawl unica (OSCAR/CulturaX no se suman sin evidencia de documentos "
     "distintos)"],
    ["C98-SRC-05", "mixto_20M (wikipedia_10M + web_10M)",
     "construccion propia sobre C98-SRC-01 y C98-SRC-04",
     "union documentada v5", "2026-09-25",
     "CC BY-SA 4.0 + ODC-BY 1.0 (componentes)", "ALLOWED_WITH_RESTRICTIONS",
     "19579", "no disponible", "20000045", "19579", "es (100%)",
     "heredada de componentes (sha1 de texto limpio)", "no aplica",
     HASHES_ESPERADOS["data/external/corpora/mixto/emb_normal/tokens_20M.txt"],
     "data/external/corpora/mixto/emb_normal/tokens_20M.txt",
     "E-CORP-07; concatenacion wiki10 + mC4-10"],
    ["C98-SRC-06", "twitter_domain COVID-es (Kaggle mmaguero)",
     "https://www.kaggle.com/datasets/mmaguero/"
     "covid19-spanishtweets-earlylateapril2020",
     "ventana cerrada 2020-03-29 a 2020-05-01", "2026-09-30",
     "CC BY-SA 4.0 (atribucion obligatoria; share-alike si se distribuyen "
     "derivados)", "ALLOWED", "361994", "no disponible", "10304387", "361994",
     "es",
     "sha1 del texto limpio por documento (v5.0-EMB-NORMAL); solapamiento vs "
     "competicion auditado: 0 exacto/normalizado, near-dup max Jaccard "
     "0.142857 < 0.2", "no aplica",
     HASHES_ESPERADOS["data/external/corpora/twitter_domain/corpus_twitter_tokens.txt"],
     "data/external/corpora/twitter_domain/corpus_twitter_tokens.txt",
     "Autorizacion escrita (p73) + confirmacion P98; sesgo tematico COVID "
     "documentado; manifiesto P73 registra 10.000.029 tokens (conteo de "
     "build); wc -w del archivo actual: 10.304.387; hash identico al de P73"],
    ["C98-SRC-10", "OSCAR es (OSCAR-2301)",
     "https://huggingface.co/datasets/oscar-corpus/OSCAR-2301", "2301",
     "no descargado", "CC BY 4.0 (ficha del dataset)",
     "NEEDS_EXPLICIT_PERMISSION", "no descargado", "no descargado",
     "no descargado", "no descargado", "es",
     "n/a (no descargado)", "n/a", "no descargado", "",
     "Gated: requiere cuenta y aceptacion de terminos; familia Common Crawl; "
     "no se suma sin evidencia de documentos distintos frente a mC4"],
    ["C98-SRC-11", "CulturaX es",
     "https://huggingface.co/datasets/uonlp/CulturaX", "por verificar",
     "no descargado", "por verificar (ficha del dataset)",
     "NEEDS_EXPLICIT_PERMISSION", "no descargado", "no descargado",
     "no descargado", "no descargado", "es", "n/a (no descargado)", "n/a",
     "no descargado", "",
     "Familia Common Crawl; solo registrada (decision P98: mC4 local primaria)"],
    ["C98-SRC-12", "HPLT 2.0 cleaned es",
     "https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned", "2.0",
     "no descargado", "a verificar antes de usar (no asumida)",
     "NEEDS_EXPLICIT_PERMISSION", "no descargado", "no descargado",
     "no descargado", "no descargado", "es", "n/a (no descargado)", "n/a",
     "no descargado", "",
     "Registrada en manifiesto v5 como candidata_requiere_evaluacion_previa"],
    ["C98-SRC-20", "Project Gutenberg (espanol)",
     "https://www.gutenberg.org/", "obras individuales",
     "no descargado", "dominio publico por obra (verificar cada una)",
     "ALLOWED_WITH_RESTRICTIONS", "no descargado", "no descargado",
     "no descargado", "no descargado", "es",
     "n/a (no descargado)", "n/a", "no descargado", "",
     "Candidata de libros: variante independiente <= 5M tokens efectivos, "
     "solo si el piloto muestra senal; registrar autor/titulo/licencia/hash "
     "por obra usada"],
    ["C98-SRC-21", "Wikisource espanol",
     "https://es.wikisource.org/", "dump por verificar", "no descargado",
     "por obra (dominio publico / CC BY-SA)", "ALLOWED_WITH_RESTRICTIONS",
     "no descargado", "no descargado", "no descargado", "no descargado", "es",
     "n/a (no descargado)", "n/a", "no descargado", "",
     "Candidata de libros (misma regla que Gutenberg)"],
    ["C98-SRC-30", "TASS 2020 y corpus de sentimiento/emocion espanol",
     "varios (TASS, corpora de sentimiento/ironia)",
     "por fuente", "no descargado", "terminos de investigacion por fuente",
     "NEEDS_EXPLICIT_PERMISSION", "no descargado", "no descargado",
     "no descargado", "no descargado", "es", "n/a (no descargado)", "n/a",
     "no descargado", "",
     "No bloquean el plan: existe corpus social autorizado (C98-SRC-06)"],
    ["C98-SRC-40", "HateEval-ES (tweets de odio)",
     "IberEval 2019 (Basile et al.)", "por verificar", "no descargado",
     "por verificar (uso de investigacion)", "NEEDS_EXPLICIT_PERMISSION",
     "no descargado", "no descargado", "no descargado", "no descargado", "es",
     "n/a (no descargado)", "n/a", "no descargado", "",
     "CUARENTENA ACTIVA (2026-10-13): sin uso hasta autorizacion escrita "
     "especifica para texto no etiquetado; no se usan sus labels"],
    ["C98-SRC-41", "Spanish-MisoCorpus-2020",
     "Garcia-Diaz et al. (2020)", "por verificar", "no descargado",
     "por verificar", "NEEDS_EXPLICIT_PERMISSION", "no descargado",
     "no descargado", "no descargado", "no descargado", "es",
     "n/a (no descargado)", "n/a", "no descargado", "",
     "CUARENTENA ACTIVA; sin uso de labels ni distribuciones de clase"],
    ["C98-SRC-42", "HaterNet", "HaterNet (dataset publico espanol)",
     "por verificar", "no descargado", "por verificar",
     "NEEDS_EXPLICIT_PERMISSION", "no descargado", "no descargado",
     "no descargado", "no descargado", "es", "n/a (no descargado)", "n/a",
     "no descargado", "", "CUARENTENA ACTIVA"],
    ["C98-SRC-43", "IberEval / AMI (misoginia)",
     "IberEval 2018 (Fersini et al.)", "por verificar", "no descargado",
     "por verificar", "NEEDS_EXPLICIT_PERMISSION", "no descargado",
     "no descargado", "no descargado", "no descargado", "es",
     "n/a (no descargado)", "n/a", "no descargado", "",
     "CUARENTENA ACTIVA"],
    ["C98-SRC-44", "HaSCoSVa (hate speech COVID)",
     "por verificar", "por verificar", "no descargado", "por verificar",
     "NEEDS_EXPLICIT_PERMISSION", "no descargado", "no descargado",
     "no descargado", "no descargado", "es", "n/a (no descargado)", "n/a",
     "no descargado", "", "CUARENTENA ACTIVA"],
]


def sha256(ruta: Path) -> str:
    """sha256 completo."""
    x = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            x.update(bloque)
    return x.hexdigest()


def main() -> None:
    """Verifica hashes, escribe registro, licencias y manifiesto."""
    for ruta in (SAL_REG, SAL_LIC, SAL_MAN):
        if ruta.exists():
            raise SystemExit(f"Ya existe {ruta}; no se sobrescribe.")
    for rel, esperado in HASHES_ESPERADOS.items():
        real = sha256(RAIZ / rel)
        if real != esperado:
            raise SystemExit(f"Hash inesperado en {rel}: {real}")
    print("P98-00: 6 hashes de corpus verificados OK")

    with open(SAL_REG, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(CAB)
        for fila in FILAS:
            if len(fila) != len(CAB):
                raise SystemExit(f"Fila mal formada: {fila[0]}")
            w.writerow(fila)

    lic = """# P98 - Licencias y gobernanza de fuentes

Fase A, 2026-10-13. Registro completo en `p98_corpus_registry.csv`; manifiesto
operativo en `artifacts/p98_corpus_manifest.json`. Autorizacion del mecanismo:
`docs/p98_provenance_autorizacion_bilstm.md` (confirmacion 2026-10-13) y
`docs/p73_provenance_autorizacion_corpus.md` (2026-09-30).

## Clasificacion

| Estado | Fuentes |
|---|---|
| ALLOWED | Wikipedia es (CC BY-SA 4.0); twitter_domain COVID-es (CC BY-SA 4.0, con autorizacion y auditoria de solapamiento) |
| ALLOWED_WITH_RESTRICTIONS | mC4 es (ODC-BY 1.0; atribucion; familia Common Crawl unica); mixto_20M (componentes); Gutenberg/Wikisource como candidatas de libros (licencia por obra) |
| NEEDS_EXPLICIT_PERMISSION | OSCAR es (gated); CulturaX (licencia por verificar); HPLT 2.0 (licencia por verificar); TASS/sentimiento-emocion; corpus de odio (HateEval-ES, MisoCorpus, HaterNet, AMI, HaSCoSVa) |
| PROHIBITED | ninguno en esta fase (ninguna fuente fue declarada prohibida por reglas verificadas) |

## Condiciones notables

- **CC BY-SA 4.0 (Wikipedia y tweets)**: atribucion obligatoria en los
  reportes; el share-alike aplica a la redistribucion de derivados textuales,
  no se redistribuyen los textos; los pesos del modelo se entrenan desde cero
  y no incorporan texto redistribuible.
- **ODC-BY 1.0 (mC4)**: atribucion a la ficha del dataset (allenai/c4,
  contenido derivado de Common Crawl).
- **Familia Common Crawl**: mC4, OSCAR y CulturaX se tratan como UNA familia
  de procedencia; fuente primaria = mC4 local; OSCAR/CulturaX quedan
  registradas sin descarga (no se contabilizan como fuentes independientes).
- **Corpus de odio**: CUARENTENA ACTIVA. No se descargan, no se entrenan, no
  se usan sus labels, ni se infieren clases desde sus anotaciones. Solo con
  autorizacion escrita especifica para uso como texto no etiquetado podria
  considerarse una variante (C98-4); un resultado positivo no la libera
  automaticamente.
- **Solapamiento**: la verificacion de solapamiento con la competicion se hace
  SOLO contra el train (permitido); frente al texto de `toxic_test` no se puede
  escanear sin violar las reglas: esa incertidumbre residual se documenta (los
  corpus provienen de fuentes externas, ventana/conjunto disjunto, y ya
  auditados a cero en su fase de origen).

## Deduplicacion

- Dentro de cada corpus: sha1 del texto limpio por documento (manifiesto v5).
- Entre fuentes: verificada la relacion de prefijo wikipedia 5M/10M/50M y la
  composicion del mixto (wiki10+mC4-10). La deduplicacion global cruzada entre
  wikipedia y mC4 no puede demostrarse exhaustivamente en esta fase; se
  documenta como limitacion y se aplicara deduplicacion por documento en la
  construccion de C98-1 (ver manifiesto P98).
"""
    SAL_LIC.write_text(lic, encoding="utf-8")

    manifiesto = {
        "schema": "kaggle_toxic_2026_puj.corpus_manifest.p98.v1",
        "creado": "2026-10-13",
        "autorizacion": {
            "mecanismo": "BiLSTM contextural desde cero (docs/p98_provenance_autorizacion_bilstm.md)",
            "corpus_externos": "docs/p73_provenance_autorizacion_corpus.md",
            "reglas": "docs/competition_rules.md (restriccion adicional)",
        },
        "reglas": [
            "Ningun corpus se usa para supervision HS/TR/AG (solo objetivo de lenguaje).",
            "Ningun texto de toxic_test.csv entra en el corpus; tampoco holdouts.",
            "Corpus de odio en cuarentena hasta autorizacion escrita.",
            "Expansiones de corpus exigen RATIONALE previo con evidencia downstream.",
            "Familia Common Crawl tratada como una sola fuente.",
        ],
        "corpus_autorizados": [
            {"source_id": "C98-SRC-01", "nombre": "wikipedia_es_10M",
             "tokens": 10002400, "documentos": 2847,
             "sha256": HASHES_ESPERADOS["data/external/corpora/wikipedia_es/emb_normal/tokens_10M.txt"],
             "ruta": "data/external/corpora/wikipedia_es/emb_normal/tokens_10M.txt"},
            {"source_id": "C98-SRC-02", "nombre": "wikipedia_es_50M",
             "tokens": 50000028, "documentos": 21529,
             "sha256": HASHES_ESPERADOS["data/external/corpora/wikipedia_es_50M/emb_normal/tokens_50M.txt"],
             "ruta": "data/external/corpora/wikipedia_es_50M/emb_normal/tokens_50M.txt"},
            {"source_id": "C98-SRC-03", "nombre": "wikipedia_es_5M",
             "tokens": 5011404, "documentos": 1694,
             "sha256": HASHES_ESPERADOS["data/external/corpora/wikipedia_es_5M/emb_normal/tokens_5M.txt"],
             "ruta": "data/external/corpora/wikipedia_es_5M/emb_normal/tokens_5M.txt"},
            {"source_id": "C98-SRC-04", "nombre": "mc4_es_10M",
             "tokens": 10000239, "documentos": 16738,
             "sha256": HASHES_ESPERADOS["data/external/corpora/mc4_es/emb_normal/tokens_10M.txt"],
             "ruta": "data/external/corpora/mc4_es/emb_normal/tokens_10M.txt"},
            {"source_id": "C98-SRC-06", "nombre": "twitter_domain_covid_es",
             "tokens": 10304387, "documentos": 361994,
             "sha256": HASHES_ESPERADOS["data/external/corpora/twitter_domain/corpus_twitter_tokens.txt"],
             "ruta": "data/external/corpora/twitter_domain/corpus_twitter_tokens.txt"},
        ],
        "cuarentena": ["C98-SRC-40", "C98-SRC-41", "C98-SRC-42", "C98-SRC-43",
                       "C98-SRC-44"],
        "candidatas_sin_descarga": ["C98-SRC-10", "C98-SRC-11", "C98-SRC-12",
                                    "C98-SRC-20", "C98-SRC-21", "C98-SRC-30"],
        "propuesta_composicion_C98_1": {
            "tokens_objetivo": 30000000,
            "proporciones": {"wikipedia": 0.60, "mc4": 0.25, "twitter": 0.15},
            "nota": "proporciones a congelar en la construccion de C98-1; "
                    "deduplicacion por documento entre componentes; el corte de "
                    "wikipedia usa el prefijo del archivo de 50M (sin descarga "
                    "nueva); el de twitter usa muestreo por documento con "
                    "semilla fija para evitar sesgo temporal",
        },
        "plan_estado": {
            "fase_A": {"registro": "hecho", "licencias": "hecho",
                       "auditoria_hashes": "hecho", "infra_sintetica": "pendiente"},
            "fase_B": "pendiente (C98-0, C98-1, D98-1)",
        },
    }
    SAL_MAN.parent.mkdir(parents=True, exist_ok=True)
    SAL_MAN.write_text(json.dumps(manifiesto, indent=2, ensure_ascii=True) + "\n",
                       encoding="utf-8")

    from collections import Counter
    estados = Counter(f[6] for f in FILAS)
    print(f"P98-00: registry {len(FILAS)} filas | estados: {dict(estados)}")
    for ruta in (SAL_REG, SAL_LIC, SAL_MAN):
        datos = ruta.read_bytes()
        if any(b > 127 for b in datos):
            raise SystemExit(f"byte > 127 en {ruta.name}")
    print("  registry, licencias y manifiesto escritos y verificados (ASCII)")


if __name__ == "__main__":
    main()
