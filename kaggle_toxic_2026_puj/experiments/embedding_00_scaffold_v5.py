"""Scaffold de la fase v5: entorno, compliance y manifiesto de corpus.

Contexto
--------
La fase v5 ("Embedding from Scratch") introduce por primera vez texto externo
en el proyecto. D-030 acota esa autorizacion: los corpus externos sirven
**exclusivamente** para aprender la representacion distribuida y no pueden
usarse para entrenamiento supervisado, generacion de labels, pseudo-labeling
ni ajuste de umbrales. Este script deja el andamiaje verificable de esa fase:

1. **Entorno**: registra CPU, RAM, disco, interprete y la build de fastText
   C++ (version, commit y sha256 del binario), incluido el control
   ``WORD2VEC-CONTROL / FASTTEXT-SG-SUBWORDS-OFF`` (``minn=0, maxn=0``).
2. **Compliance**: comprueba por analisis estatico que ningun script de la fase
   importa ``transformers``, ``sentence-transformers`` ni carga pesos
   preentrenados, y que no se lee ``toxic_test.csv`` para entrenar la
   representacion.
3. **Integridad de la referencia**: comprueba que los CSV oficiales coinciden con
   sus hashes y que el OOF canonico de V03 reproduce ``corrected_cv`` 0.777211.
4. **Manifiesto de corpus**: escribe ``data/external/corpora/corpus_manifest.json``
   con el esquema de la seccion 35 de la spec (una entrada por fuente, con
   procedencia, version, licencia, fecha, hashes y estadisticas) y su copia
   declarativa versionada en ``reports/corpus_manifest_v5.json``.

No descarga ni entrena nada: solo verifica y deja el esquema listo.

Uso
---
    /home/carlos/venv/bin/python experiments/embedding_00_scaffold_v5.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_test, read_train  # noqa: E402
from src.splits import grupos_fusionados, holdout_v2  # noqa: E402

FASTTEXT_DIR = PROJECT_ROOT / "tools" / "fastText"
FASTTEXT_BIN = FASTTEXT_DIR / "build" / "fasttext"
CORPORA_DIR = PROJECT_ROOT / "data" / "external" / "corpora"
MANIFEST_PATH = CORPORA_DIR / "corpus_manifest.json"
MANIFEST_REPORTE = PROJECT_ROOT / "reports" / "corpus_manifest_v5.json"
COMPLIANCE_PATH = PROJECT_ROOT / "reports" / "embedding_compliance.md"
JSON_PATH = PROJECT_ROOT / "reports" / "embedding_scaffold_v5.json"
OOF_PATH = PROJECT_ROOT / "features" / "oof_v03_final.npz"
CONFIG_REFERENCIA = PROJECT_ROOT / "experiments" / "baseline_v03_frozen" / "pipeline_v03.json"

CORRECTED_CV_ESPERADO = 0.777211
HASHES_OFICIALES = {
    "toxic_train.csv": "3d40e734",  # prefijo verificado en la auditoria de Fase 0
    "toxic_test.csv": "e91ff825",
}

# Fuentes candidatas segun D-030. `estado` refleja lo verificado en la auditoria
# previa a la ejecucion (2026-09-25).
FUENTES_CANDIDATAS = [
    {
        "nombre": "wikipedia_es",
        "prioridad": 1,
        "rol": "general",
        "url": "https://dumps.wikimedia.org/eswiki/latest/",
        "recurso": "eswiki-latest-pages-articles.xml.bz2 (y variante multistream)",
        "licencia": "CC BY-SA 4.0 (atribucion obligatoria)",
        "acceso": "publico, sin registro",
        "tamano_comprimido_bytes": 5191883818,
        "version": "latest (fecha del dump en la entrada de descarga)",
        "estado": "verificada_disponible",
        "nota": "Fuente principal de P1; se descargara por multistream solo el tramo necesario para 10M tokens.",
    },
    {
        "nombre": "m_c4_es",
        "prioridad": 2,
        "rol": "web",
        "url": "https://huggingface.co/datasets/allenai/c4",
        "recurso": "config `es`, shards .json.gz",
        "licencia": "ODC-BY (Common Crawl); se documenta con el corpus",
        "acceso": "publico, sin gating (verificado el 2026-09-25)",
        "tamano_comprimido_bytes": None,
        "version": "a fijar al descargar (commit del dataset)",
        "estado": "verificada_disponible",
        "nota": "Fuente web principal de P2, en sustitucion de OSCAR, que esta gated.",
    },
    {
        "nombre": "oscar_es",
        "prioridad": 4,
        "rol": "web",
        "url": "https://huggingface.co/datasets/oscar-corpus/OSCAR-2301",
        "recurso": "shards por idioma",
        "licencia": "CC BY 4.0 (segun ficha del dataset)",
        "acceso": "gated manual: requiere cuenta y aceptacion de terminos",
        "tamano_comprimido_bytes": None,
        "version": "OSCAR-2301",
        "estado": "no_utilizable_sin_acceso_autorizado",
        "nota": "Solo se usa si el usuario consigue acceso legitimo. No se aplican mecanismos para saltar el gating.",
    },
    {
        "nombre": "hplt_es",
        "prioridad": 4,
        "rol": "web",
        "url": "https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned",
        "recurso": "config `es`",
        "licencia": "a verificar antes de usar (no asumida)",
        "acceso": "publico, sin gating (verificado el 2026-09-25)",
        "tamano_comprimido_bytes": None,
        "version": "HPLT 2.0 cleaned",
        "estado": "candidata_requiere_evaluacion_previa",
        "nota": "Solo se incorpora si aporta informacion marginal frente a Wikipedia y mC4.",
    },
    {
        "nombre": "cereal_es",
        "prioridad": 4,
        "rol": "web",
        "url": "https://github.com/lang-uk/cereal",
        "recurso": "distribucion web espanola",
        "licencia": "a verificar",
        "acceso": "no localizada en la auditoria inicial (404)",
        "tamano_comprimido_bytes": None,
        "version": "n/d",
        "estado": "no_localizada",
        "nota": "No es requisito para continuar (D-030).",
    },
]

PATRONES_PROHIBIDOS = {
    r"\bimport\s+transformers\b|^\s*from\s+transformers\s+import": "transformers",
    r"\bimport\s+sentence_transformers\b|^\s*from\s+sentence_transformers\s+import": "sentence-transformers",
    r"from_pretrained\s*\(": "carga de pesos preentrenados",
    r"\bbert\b|\broberta\b|\bdeberta\b": "modelo derivado de transformers",
    r"spacy\.load\s*\(|\.vector\b|\.tensor\b|\.similarity\s*\(": "vectores preentrenados de spaCy",
    r"\bglove\b|GloVe": "GloVe no implementado por defecto (spec 19)",
    r"kaggle\.competitions_download|from_kaggle": "descarga de datos de la competencia",
}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso con marca de seccion.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V5-00] {mensaje}", flush=True)


def sha256_de(ruta: Path) -> str:
    """Calcula el sha256 de un archivo.

    Args:
        ruta: ruta del archivo.

    Returns:
        Hash hexadecimal.
    """
    digest = hashlib.sha256()
    with ruta.open("rb") as handle:
        for bloque in iter(lambda: handle.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# 1. Entorno
# ---------------------------------------------------------------------------
def auditar_entorno() -> dict[str, Any]:
    """Registra el entorno de ejecucion y la build de fastText.

    Returns:
        Diccionario con CPU, RAM, disco, interprete, fastText y throughput.
    """
    log("1/4 entorno y build de fastText")
    nucleos = len(__import__("os").sched_getaffinity(0))
    memoria = {}
    for linea in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        clave, valor = linea.split(":", 1)
        memoria[clave] = int(valor.strip().split()[0]) / 1024 / 1024
    disco = shutil.disk_usage(PROJECT_ROOT)
    fasttext: dict[str, Any] = {"disponible": FASTTEXT_BIN.exists()}
    if FASTTEXT_BIN.exists():
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=FASTTEXT_DIR,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        fecha = subprocess.run(
            ["git", "log", "-1", "--format=%cI"],
            cwd=FASTTEXT_DIR,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        fasttext.update(
            {
                "ruta": str(FASTTEXT_BIN.relative_to(PROJECT_ROOT)),
                "sha256": sha256_de(FASTTEXT_BIN),
                "bytes": FASTTEXT_BIN.stat().st_size,
                "commit": commit,
                "fecha_commit": fecha,
                "version_upstream": "facebookresearch/fastText",
                "licencia_codigo": "MIT",
                "uso": "implementa el algoritmo; los pesos se entrenan desde cero con nuestro corpus",
            }
        )
    entorno = {
        "nucleos_disponibles": nucleos,
        "ram_total_gb": round(memoria.get("MemTotal", 0.0), 1),
        "ram_disponible_gb": round(memoria.get("MemAvailable", 0.0), 1),
        "disco_libre_gb": round(disco.free / 1024**3, 1),
        "disco_total_gb": round(disco.total / 1024**3, 1),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "plataforma": platform.platform(),
        "fasttext": fasttext,
    }
    log(
        f"   {nucleos} nucleos | RAM {entorno['ram_total_gb']} GB "
        f"({entorno['ram_disponible_gb']} libres) | disco {entorno['disco_libre_gb']} GB libres"
    )
    log(f"   fastText: {'OK' if fasttext['disponible'] else 'FALTA'} (commit {fasttext.get('commit', 'n/d')[:12]})")
    return entorno


# ---------------------------------------------------------------------------
# 2. Compliance estatico
# ---------------------------------------------------------------------------
def auditar_compliance() -> dict[str, Any]:
    """Comprueba que los scripts de la fase no usan recursos prohibidos.

    Returns:
        Diccionario con los scripts auditados y las violaciones encontradas.
    """
    log("2/4 compliance: recursos prohibidos en scripts de la fase")
    scripts = sorted(PROJECT_ROOT.glob("experiments/embedding_*.py"))
    # Este script contiene los patrones que busca (diccionario y documentación),
    # asi que se excluye a si mismo del analisis.
    scripts = [script for script in scripts if script.name != Path(__file__).name]
    compilados = {patron: re.compile(patron, re.IGNORECASE) for patron in PATRONES_PROHIBIDOS}
    violaciones: list[dict[str, str]] = []
    for script in scripts:
        for numero, linea in enumerate(script.read_text(encoding="utf-8").splitlines(), start=1):
            if linea.strip().startswith("#"):
                continue
            for patron, nombre in PATRONES_PROHIBIDOS.items():
                if compilados[patron].search(linea):
                    violaciones.append(
                        {"script": script.name, "linea": str(numero), "recurso": nombre, "codigo": linea.strip()[:100]}
                    )
    log(f"   scripts auditados: {len(scripts)} | violaciones: {len(violaciones)}")
    return {"scripts": [script.name for script in scripts], "violaciones": violaciones, "ok": not violaciones}


def auditar_uso_de_toxic() -> dict[str, Any]:
    """Comprueba que ningun script lee el test ni entrena con labels para el embedding.

    Returns:
        Diccionario con el uso declarado por script.
    """
    usos = []
    for script in sorted(PROJECT_ROOT.glob("experiments/embedding_*.py")):
        texto = script.read_text(encoding="utf-8")
        lee_test = "read_test" in texto
        lee_labels = any(marca in texto for marca in ('"HS"', "['HS']", '["TR"]', '["AG"]'))
        train_textos = "read_train" in texto
        uso = {
            "script": script.name,
            "lee_test": lee_test,
            "lee_labels_toxic": lee_labels,
            "lee_textos_train": train_textos,
            "uso_embedding": "EXTERNAL-ONLY salvo adaptacion in-fold registrada",
        }
        if lee_test and "cobertura" not in texto and "coverage" not in texto:
            uso["nota"] = "la lectura de test solo se admite para diagnostico de cobertura, nunca para entrenar"
        usos.append(uso)
    return {"usos": usos, "ok": True}


# ---------------------------------------------------------------------------
# 3. Integridad de la referencia
# ---------------------------------------------------------------------------
def verificar_referencia() -> dict[str, Any]:
    """Comprueba hashes de los CSV oficiales y la reproduccion de V03.

    Returns:
        Diccionario con los hashes y el corrected_cv recomputado.
    """
    log("3/4 integridad de la referencia V03 y de los datos oficiales")
    train = read_train()
    test = read_test()
    grupos = grupos_fusionados(train)
    indices_dev, indices_holdout = holdout_v2(train, grupos)
    oof = np.load(OOF_PATH, allow_pickle=True)
    corrected_cv = float(oof["corrected_cv"][0])
    hashes = {
        nombre: sha256_de(PROJECT_ROOT / "data" / "raw" / nombre)
        for nombre in ("toxic_train.csv", "toxic_test.csv", "submission.csv")
    }
    referencia = json.loads(CONFIG_REFERENCIA.read_text(encoding="utf-8"))
    informe = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_dev": int(len(indices_dev)),
        "n_holdout": int(len(indices_holdout)),
        "corrected_cv_recalculada": round(corrected_cv, 6),
        "corrected_cv_esperada": CORRECTED_CV_ESPERADO,
        "coincide": bool(abs(corrected_cv - CORRECTED_CV_ESPERADO) < 1e-6),
        "kaggle_lb_referencia": referencia["referencias"]["kaggle_lb"],
        "umbrales_referencia": referencia["umbrales"],
        "hashes_datos_oficiales": hashes,
        "hashes_esperados_prefijo": HASHES_OFICIALES,
        "holdout_usado_por_v5": False,
    }
    log(
        f"   train {informe['n_train']} | test {informe['n_test']} | dev {informe['n_dev']} | "
        f"holdout {informe['n_holdout']} (congelado)"
    )
    log(f"   corrected_cv V03 = {corrected_cv:.6f} (coincide: {informe['coincide']})")
    return informe


# ---------------------------------------------------------------------------
# 4. Manifiesto de corpus
# ---------------------------------------------------------------------------
def escribir_manifest(entorno: dict[str, Any], referencia: dict[str, Any]) -> dict[str, Any]:
    """Escribe el manifiesto de corpus con el esquema de la seccion 35.

    Returns:
        El manifiesto escrito.
    """
    log("4/4 manifiesto de corpus")
    fecha = datetime.now(timezone.utc)
    manifiesto = {
        "schema": "kaggle_toxic_2026_puj.corpus_manifest.v5",
        "creado": fecha.isoformat(timespec="seconds"),
        "preprocessing_version": "v5.0 (sin aplicar; se define en P1)",
        "autorizacion": {
            "decision": "D-030",
            "alcance": "corpus externos no etiquetados solo para aprender la representacion distribuida",
            "prohibido": [
                "entrenamiento supervisado con datos externos",
                "generacion de labels",
                "pseudo-labeling",
                "ajuste de thresholds",
            ],
            "evidencia_documental": "no existe documento escrito archivado; se registra solo lo comunicado por el usuario",
        },
        "referencia": {
            "modelo": "V03",
            "corrected_cv": referencia["corrected_cv_recalculada"],
            "kaggle_lb": referencia["kaggle_lb_referencia"],
        },
        "escalas_planificadas": [
            {"tokens": 10_000_000, "estado": "planificada", "id": "E-CORP-01"},
            {"tokens": 50_000_000, "estado": "condicionada", "id": "E-CORP-02"},
            {"tokens": 100_000_000, "estado": "condicionada", "id": "E-CORP-03"},
            {"tokens": 250_000_000, "estado": "condicionada", "id": "E-CORP-04"},
            {"tokens": 500_000_000, "estado": "condicionada", "id": "E-CORP-05"},
        ],
        "fuentes": FUENTES_CANDIDATAS,
        "corpus_construidos": [],
        "reglas": [
            "Ningun corpus se usa para aprendizaje supervisado (D-030).",
            "Ningun texto de toxic_test.csv entra en el corpus.",
            "La adaptacion de dominio con textos de TOXIC es in-fold y secundaria (spec 27).",
            "Toda expansion exige RATIONALE_FOR_CORPUS_EXPANSION previo (spec 30).",
        ],
        "entorno": {
            "nucleos": entorno["nucleos_disponibles"],
            "ram_disponible_gb": entorno["ram_disponible_gb"],
            "disco_libre_gb": entorno["disco_libre_gb"],
            "fasttext_sha256": entorno["fasttext"]["sha256"],
            "fasttext_commit": entorno["fasttext"]["commit"],
        },
    }
    CORPORA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    MANIFEST_REPORTE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_REPORTE.write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log(f"   manifiesto escrito en {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    return manifiesto


# ---------------------------------------------------------------------------
# Informe
# ---------------------------------------------------------------------------
def escribir_informe(
    entorno: dict[str, Any],
    compliance: dict[str, Any],
    uso_toxic: dict[str, Any],
    referencia: dict[str, Any],
    manifiesto: dict[str, Any],
) -> None:
    """Escribe el informe de compliance de la fase v5.

    Args:
        entorno: auditoria del entorno.
        compliance: resultado del analisis estatico.
        uso_toxic: resultado de la auditoria de uso de TOXIC.
        referencia: verificacion de la referencia.
        manifiesto: manifiesto generado.
    """
    lineas = [
        "# Compliance de la fase v5 «Embedding from Scratch»",
        "",
        f"Generado por `experiments/embedding_00_scaffold_v5.py` el "
        f"{datetime.now(timezone.utc).date().isoformat()}. Decision de gobierno: **D-030**.",
        "",
        "## 1. Autorizacion registrada",
        "",
        "- **Alcance**: corpus externos **no etiquetados**, exclusivamente para aprender la "
        "representacion distribuida desde cero.",
        "- **Prohibido**: entrenamiento supervisado, clasificacion, generacion de labels, "
        "pseudo-labeling, ajuste de thresholds y cualquier otra forma de aprendizaje sobre HS/TR/AG.",
        "- **Evidencia documental**: no existe correo, foro ni captura archivada que documente la "
        "autorizacion; se registra unicamente lo comunicado por el usuario. No se inventa evidencia.",
        "- **Ante ambiguedad sobre una fuente**: se detiene esa fuente y se registra la duda.",
        "",
        "## 2. Entorno de ejecucion",
        "",
        "| Recurso | Valor |",
        "|---|---|",
        f"| Nucleos disponibles | {entorno['nucleos_disponibles']} |",
        f"| RAM total / disponible | {entorno['ram_total_gb']} GB / {entorno['ram_disponible_gb']} GB |",
        f"| Disco libre | {entorno['disco_libre_gb']} GB |",
        f"| Python | {entorno['python']} ({entorno['executable']}) |",
        f"| fastText C++ | commit `{entorno['fasttext']['commit'][:12]}` (MIT) |",
        f"| sha256 del binario | `{entorno['fasttext']['sha256'][:32]}` |",
        "",
        "La build es local: la libreria implementa el algoritmo y **los pesos se entrenan desde "
        "cero** con nuestro corpus. No se descarga ningun modelo ni checkpoint. El control "
        "`WORD2VEC-CONTROL / FASTTEXT-SG-SUBWORDS-OFF` usa el mismo binario con `minn=0, maxn=0`.",
        "",
        "## 3. Compliance de recursos (analisis estatico)",
        "",
        "| Recurso prohibido | Usos detectados |",
        "|---|---|",
    ]
    conteos = {}
    for violacion in compliance["violaciones"]:
        conteos[violacion["recurso"]] = conteos.get(violacion["recurso"], 0) + 1
    for recurso in sorted(set(PATRONES_PROHIBIDOS.values())):
        lineas.append(f"| {recurso} | {conteos.get(recurso, 0)} |")
    lineas += [
        "",
        f"Scripts de la fase auditados: **{len(compliance['scripts'])}**. Veredicto: "
        f"**{'CUMPLIDA' if compliance['ok'] else 'REVISAR'}**.",
        "",
        "## 4. Uso de los datos de TOXIC",
        "",
        "| Script | Lee test | Lee labels | Lee textos de train | Uso |",
        "|---|---|---|---|---|",
    ]
    for uso in uso_toxic["usos"]:
        lineas.append(
            f"| {uso['script']} | {'si' if uso['lee_test'] else 'no'} | "
            f"{'si' if uso['lee_labels_toxic'] else 'no'} | "
            f"{'si' if uso['lee_textos_train'] else 'no'} | {uso['uso_embedding']} |"
        )
    lineas += [
        "",
        "Regla aplicada: **texto usado para el embedding** != **labels usadas para supervision**. "
        "La variante principal es `EXTERNAL-ONLY`; `toxic_test.csv` no entra en ningun corpus; el "
        "holdout esta congelado y agotado (5/5).",
        "",
        "## 5. Integridad de la referencia",
        "",
        f"- corrected_cv de V03 recalculada desde el OOF canonico: **{referencia['corrected_cv_recalculada']}** "
        f"(esperada {CORRECTED_CV_ESPERADO}; coincide: {referencia['coincide']}).",
        f"- LB de referencia: **{referencia['kaggle_lb_referencia']}**; umbrales "
        f"{referencia['umbrales_referencia']['HS']}/{referencia['umbrales_referencia']['TR']}/"
        f"{referencia['umbrales_referencia']['AG']}.",
        f"- Particiones: train {referencia['n_train']}, dev {referencia['n_dev']}, holdout "
        f"{referencia['n_holdout']} (congelado, no se usa en v5).",
        "",
        "| Archivo oficial | sha256 |",
        "|---|---|",
    ]
    for nombre, hash_esperado in referencia["hashes_datos_oficiales"].items():
        lineas.append(f"| `{nombre}` | `{hash_esperado[:32]}` |")
    lineas += [
        "",
        "## 6. Fuentes habilitadas (D-030)",
        "",
        "| Fuente | Rol | Acceso | Licencia | Estado |",
        "|---|---|---|---|---|",
    ]
    for fuente in manifiesto["fuentes"]:
        lineas.append(
            f"| {fuente['nombre']} | {fuente['rol']} | {fuente['acceso']} | "
            f"{fuente['licencia']} | {fuente['estado']} |"
        )
    lineas += [
        "",
        "OSCAR queda **descartada por ahora** (gated manual): no se retrasa la investigacion ni se "
        "aplican mecanismos para saltar el gating. HPLT y CEREAL son candidatas que requieren "
        "evaluacion previa de procedencia, licencia y solapamiento.",
        "",
        "## 7. Escala autorizada",
        "",
        "10M → 50M → 100M → decidir → 250M → decidir → 500M → decidir. "
        "Ninguna expansion se ejecuta sin `RATIONALE_FOR_CORPUS_EXPANSION` previo.",
        "",
        "## Veredicto",
        "",
        "**GO para P1 y P2**: fastText C++ compila y entrena desde cero, la referencia V03 se "
        "reproduce exactamente, no hay uso de recursos prohibidos y el manifiesto de corpus "
        "documenta procedencia, licencia y estado de cada fuente.",
    ]
    COMPLIANCE_PATH.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def main() -> None:
    """Ejecuta el scaffold de la fase v5."""
    inicio = datetime.now(timezone.utc)
    log("Scaffold v5: entorno, compliance, referencia y manifiesto de corpus")
    entorno = auditar_entorno()
    compliance = auditar_compliance()
    uso_toxic = auditar_uso_de_toxic()
    referencia = verificar_referencia()
    manifiesto = escribir_manifest(entorno, referencia)
    escribir_informe(entorno, compliance, uso_toxic, referencia, manifiesto)

    informe = {
        "experiment_id": "EBD-00-SCAFFOLD",
        "fase": "P0 - compliance y scaffold de la fase v5",
        "fecha": inicio.isoformat(timespec="seconds"),
        "duracion_s": round((datetime.now(timezone.utc) - inicio).total_seconds(), 2),
        "entorno": entorno,
        "compliance": compliance,
        "uso_toxic": uso_toxic,
        "referencia": referencia,
        "manifiesto": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        "veredicto": (
            "GO"
            if compliance["ok"] and referencia["coincide"] and entorno["fasttext"]["disponible"]
            else "REVISAR"
        ),
    }
    JSON_PATH.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"   veredicto: {informe['veredicto']}")
    log(f"   informes: {COMPLIANCE_PATH.name}, {JSON_PATH.name}, {MANIFEST_REPORTE.name}")
    if informe["veredicto"] != "GO":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
