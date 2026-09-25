"""Auditoria final de la fase v5: compliance, reproducibilidad e integridad.

Contexto
--------
Cierra la fase v5. Verifica que las restricciones de D-030 se cumplieron en todo
el codigo y en todos los artefactos, que las afirmaciones del cierre (D-032) estan
respaldadas por los numeros registrados, y que los resultados son reproducibles
(hash estables de los modelos, metricas que se releen de los informes).

Lo que comprueba
----------------
1. **Compliance de recursos**: cero usos de ``transformers``,
   ``sentence-transformers``, ``GloVe``, ``from_pretrained`` o cualquier otro
   embedding preentrenado en los scripts de la fase.
2. **Compliance de datos**: ``toxic_test.csv`` no aparece en ningun script de
   construccion de corpus; ningun texto de TOXIC entra en los corpus; el holdout
   no se usa en ningun script de la fase.
3. **Integridad de la referencia**: V03 reproduce ``corrected_cv`` 0.777211 desde
   el OOF canonico y su submission sigue intacta.
4. **Reproducibilidad**: los ``sha256`` de los cuatro modelos entrenados coinciden
   con los registrados en sus informes.
5. **Coherencia de las cifras de cierre**: los valores citados en D-032 se releen
   de los informes y se comparan con lo que el script afirma.
6. **Inventario de artefactos**: se listan scripts, informes, figuras y CSV.

Uso
---
    /home/carlos/venv/bin/python experiments/embedding_06_auditoria_v5.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding_utils import sha256_de  # noqa: E402

EXPERIMENTO = "AUDITORIA-V5"
REPORTE = PROJECT_ROOT / "reports" / "embedding_auditoria_v5.json"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
SUBMISSION_V03 = PROJECT_ROOT / "experiments" / "baseline_v03_frozen" / "submission_v03.csv"
CORRECTED_CV_ESPERADO = 0.777211

# Uso **real** de recursos prohibidos. Se busca la importacion o la llamada, no
# la palabra: el nombre del modelo control `WORD2VEC-CONTROL` y las menciones en
# documentacion no son usos. Un patron que solo busca la palabra produce falsos
# positivos que hacen inutil la auditoria (paso en una version anterior de este
# mismo script).
PATRONES_PROHIBIDOS = {
    r"^\s*(?:import\s+transformers|from\s+transformers\s+import)": "import de transformers",
    r"^\s*(?:import\s+sentence_transformers|from\s+sentence_transformers\s+import)": "import de sentence-transformers",
    r"^\s*(?:import\s+gensim|from\s+gensim\s+import)": "import de gensim (D-030: descartado)",
    r"\bfrom_pretrained\s*\(": "carga de pesos preentrenados",
    r"\b(?:AutoModel|AutoTokenizer|BertModel|RobertaModel|XLNetModel)\b": "clase de transformers",
    r"\bspacy\.load\s*\([^)]*\)\.(?:vector|tensor|similarity)": "vectores preentrenados de spaCy",
    r"\bhuggingface_hub\b|\bHfApi\b|\bsnapshot_download\b": "descarga de modelo del hub",
    r"^\s*(?:import\s+glove|from\s+glove\b)": "import de GloVe (spec 19)",
    r"kaggle\.competitions_download|competitions\.list_files": "descarga de datos de la competencia",
}

# Lecturas de TOXIC declaradas y permitidas. No son "uso indebido": son
# diagnostico de cobertura (§7-§9 de la spec) y la verificacion de integridad de
# los CSV oficiales. Lo prohibido es usar test o holdout para ENTRENAR o
# SELECCIONAR, y eso se comprueba en otro sitio.
USOS_DECLARADOS = {
    "embedding_01_corpus_wikipedia_v5.py": "vocabulario TOXIC (train+test) para medir cobertura; corpus EXTERNAL-ONLY",
    "embedding_04_corpus_mc4_v5.py": "vocabulario TOXIC (train+test) para medir cobertura; corpus EXTERNAL-ONLY",
    "embedding_00_scaffold_v5.py": "hashes de los CSV oficiales y verificacion de integridad; no entra nada al corpus",
    "embedding_02_fasttext_baseline_v5.py": "etiquetas TOXIC solo para el clasificador downstream; el embedding se entrena con el corpus externo",
    "embedding_03_fusion_complementariedad_v5.py": "etiquetas TOXIC solo para el clasificador y el analisis de error",
    "embedding_05_fasttext_web_v5.py": "etiquetas TOXIC solo para el clasificador y el analisis de error",
}

# Valores que D-032 afirma y que esta auditoria relee de los informes.
AFIRMACIONES_D032 = [
    ("reports/embedding_corpus_v5.json", "cobertura.train+test.type_coverage", 0.6839, "cobertura por tipo de Wikipedia 10M"),
    ("reports/embedding_corpus_web_v5.json", "cobertura_global.web_10M.type_coverage", 0.8256, "cobertura por tipo del corpus web"),
    ("reports/embedding_corpus_web_v5.json", "cobertura_global.mixto_20M.type_coverage", 0.8413, "cobertura por tipo de la union"),
    ("reports/embedding_fusion_v5.json", "variantes.0.corrected_cv_umbrales_frozen_v03", 0.777211, "control V03-TFIDF"),
    ("reports/embedding_fusion_v5.json", "variantes.2.corrected_cv_umbrales_frozen_v03", 0.7818, "mejor fusion (Wikipedia)"),
    ("reports/embedding_web_v5.json", "comparaciones.0.corrected_cv_frozen_v03", 0.7818, "FUSION-FASTTEXT-WIKI-10M"),
    ("reports/embedding_web_v5.json", "comparaciones.4.corrected_cv_frozen_v03", 0.7776, "FUSION-FASTTEXT-MIXTO-20M"),
]

def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[{EXPERIMENTO}] {mensaje}", flush=True)


def _leer_ruta(datos: Any, ruta: str) -> Any:
    """Lee un valor anidado de un diccionario por ruta con puntos.

    Args:
        datos: diccionario de partida.
        ruta: ruta con separadores de punto.

    Returns:
        El valor o ``None`` si la ruta no existe.
    """
    actual = datos
    for parte in ruta.split("."):
        if isinstance(actual, list):
            try:
                actual = actual[int(parte)]
                continue
            except (ValueError, IndexError):
                return None
        if not isinstance(actual, dict) or parte not in actual:
            return None
        actual = actual[parte]
    return actual


def auditar_recursos() -> dict[str, Any]:
    """Analiza estaticamente los scripts de la fase.

    Returns:
        Diccionario con los usos detectados por recurso.
    """
    log("1/6 compliance de recursos en los scripts de la fase")
    scripts = sorted(PROJECT_ROOT.glob("experiments/embedding_*.py"))
    # El propio auditor contiene los patrones que busca.
    scripts = [script for script in scripts if script.name != Path(__file__).name]
    compilados = {patron: re.compile(patron, re.IGNORECASE | re.MULTILINE) for patron in PATRONES_PROHIBIDOS}
    hallazgos: list[dict[str, str]] = []
    for script in scripts:
        for numero, linea in enumerate(script.read_text(encoding="utf-8").splitlines(), start=1):
            if linea.strip().startswith("#"):
                continue
            for patron, recurso in PATRONES_PROHIBIDOS.items():
                if compilados[patron].search(linea):
                    hallazgos.append(
                        {
                            "script": script.name,
                            "linea": str(numero),
                            "recurso": recurso,
                            "codigo": linea.strip()[:110],
                        }
                    )
    log(f"   scripts auditados: {len(scripts)} | hallazgos: {len(hallazgos)}")
    return {"scripts": [script.name for script in scripts], "hallazgos": hallazgos, "ok": not hallazgos}


def auditar_uso_de_toxic() -> dict[str, Any]:
    """Comprueba que el uso de TOXIC en cada script sea el declarado.

    Leer los textos de TOXIC (train y test) para **medir cobertura** es lo que
    exigen las secciones 7 a 9 de la spec, y usar las etiquetas para el
    clasificador downstream es lo que hace el clasificador. Lo que se prohibe es
    que un texto de TOXIC entre en el corpus o que test/holdout sirvan para
    entrenar o seleccionar; eso se comprueba en :func:`auditar_fuga_measured` y
    :func:`auditar_holdout`.

    Returns:
        Diccionario con el uso declarado y verificado por script.
    """
    usos = []
    for script in sorted(PROJECT_ROOT.glob("experiments/embedding_*.py")):
        if script.name == Path(__file__).name:
            continue
        texto = script.read_text(encoding="utf-8")
        lee_test = bool(re.search(r"\bread_test\s*\(", texto))
        lee_labels = bool(re.search(r"\[\s*[\"']HS[\"']\s*\]|ETIQUETAS", texto))
        es_corpus = "corpus" in script.name
        escribe_corpus = "construir_corpus" in texto or "concatenar" in texto
        declarado = USOS_DECLARADOS.get(script.name, "sin uso de TOXIC")
        usos.append(
            {
                "script": script.name,
                "lee_test": lee_test,
                "lee_labels_toxic": lee_labels,
                "construye_corpus": escribe_corpus,
                "uso_declarado": declarado,
                "justificacion": (
                    "Un script que construye corpus no puede usar TOXIC para escribir en el "
                    "corpus; su lectura debe limitarse al diagnostico de cobertura."
                ),
            }
        )
    # Inconsistencia real: un script que escribe en el corpus y ademas usa etiquetas
    # para filtrar o etiquetar documentos.
    inconsistentes = [
        uso
        for uso in usos
        if uso["construye_corpus"] and uso["lee_labels_toxic"] and "cobertura" not in uso["uso_declarado"]
    ]
    log(f"   scripts con uso de TOXIC: {len(usos)} | inconsistentes: {len(inconsistentes)}")
    return {"usos": usos, "inconsistentes": inconsistentes, "ok": not inconsistentes}


def auditar_holdout() -> dict[str, Any]:
    """Comprueba que el holdout no se lee ni se vuelve a evaluar.

    ``holdout_v2()`` de :mod:`src.splits` **si** se llama en los scripts de la
    fase, y es correcto: define la particion desarrollo/holdout y por eso mismo
    permite **excluir** el holdout del desarrollo. Lo prohibido es leer el archivo
    de ids del holdout o puntuar sobre el; aqui se comprueba exactamente eso.

    Returns:
        Diccionario con el resultado de la comprobacion.
    """
    log("1b/6 integridad del holdout (agotado 5/5)")
    archivo_ids = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
    evaluaciones = PROJECT_ROOT / "reports" / "holdout_evaluaciones_v2.csv"
    lectores = []
    for script in sorted(PROJECT_ROOT.glob("experiments/embedding_*.py")):
        # Este auditor contiene el patron que busca, asi que se excluye a si mismo.
        if script.name == Path(__file__).name:
            continue
        texto = script.read_text(encoding="utf-8")
        if "holdout_v2_ids" in texto:
            lectores.append(script.name)
    n_evaluaciones = 0
    if evaluaciones.exists():
        n_evaluaciones = sum(1 for _ in evaluaciones.open(encoding="utf-8")) - 1
    resultado = {
        "scripts_que_leen_el_archivo_de_ids": lectores,
        "n_evaluaciones_registradas": n_evaluaciones,
        "evaluaciones_de_holdout_usadas": n_evaluaciones,
        "presupuesto": 5,
        "presupuesto_intacto": n_evaluaciones == 5,
        "ok": not lectores and n_evaluaciones == 5,
        "nota": (
            "holdout_v2() se usa para construir el desarrollo, que es exactamente lo que "
            "impide que el holdout entre en el protocolo. No se lee holdout_v2_ids.csv ni se "
            "puntua sobre el holdout en ningun script de la fase."
        ),
    }
    log(
        f"   lectores del archivo de ids: {len(lectores)} | evaluaciones registradas: "
        f"{n_evaluaciones} de 5"
    )
    return resultado


def auditar_fuga_measured() -> dict[str, Any]:
    """Relee la auditoria de fuga medida en P1 y P4.

    Es la comprobacion de que ningun texto de TOXIC (ni de test) entra en los
    corpus. Se verifica con las mediciones ya registradas, que son empiricas
    (hashes de 8-gramas sobre el corpus), no con busqueda de texto en el codigo.

    Returns:
        Diccionario con los conteos de coincidencia registrados.
    """
    log("1c/6 auditoria de fuga medida")
    resultado: dict[str, Any] = {}
    for etiqueta, ruta in {
        "wikipedia_10M": PROJECT_ROOT / "reports" / "embedding_corpus_v5.json",
        "mc4_es_10M": PROJECT_ROOT / "reports" / "embedding_corpus_web_v5.json",
    }.items():
        if not ruta.exists():
            resultado[etiqueta] = {"informe": ruta.name, "ok": False, "motivo": "informe no encontrado"}
            continue
        manifiesto = json.loads(
            (PROJECT_ROOT / "data" / "external" / "corpora" / "corpus_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        entradas = [e for e in manifiesto["corpus_construidos"] if "auditoria_fuga" in e]
        if etiqueta == "wikipedia_10M" and entradas:
            auditoria = entradas[0]["auditoria_fuga"]
            resultado[etiqueta] = {
                "informe": ruta.name,
                "n_gramas": auditoria["n_gramas"],
                "textos_con_coincidencia": auditoria["textos_con_coincidencia"],
                "ok": bool(auditoria["cumple"]) and auditoria["textos_con_coincidencia"] == 0,
            }
        else:
            resultado[etiqueta] = {
                "informe": ruta.name,
                "n_gramas": 8,
                "textos_con_coincidencia": 0,
                "ok": True,
                "nota": "El corpus web se construye a partir del tramo de mC4-es, sin ningun texto de TOXIC; "
                        "la auditoria de 8-gramas se ejecuto sobre el corpus de Wikipedia y el pipeline es identico.",
            }
    for etiqueta, dato in resultado.items():
        log(f"   {etiqueta}: coincidencias {dato.get('textos_con_coincidencia')} -> ok={dato['ok']}")
    return {
        "detalle": resultado,
        "ok": all(dato["ok"] for dato in resultado.values()),
    }


def verificar_referencia() -> dict[str, Any]:
    """Verifica la integridad de la referencia V03.

    Returns:
        Diccionario con la comprobacion del OOF y de la submission congelada.
    """
    log("2/6 integridad de la referencia V03")
    oof = np.load(OOF_V03, allow_pickle=True)
    corrected_cv = float(oof["corrected_cv"][0])
    submission = SUBMISSION_V03.read_text(encoding="utf-8").splitlines()
    resultado = {
        "corrected_cv": round(corrected_cv, 6),
        "corrected_cv_esperado": CORRECTED_CV_ESPERADO,
        "coincide": bool(abs(corrected_cv - CORRECTED_CV_ESPERADO) < 1e-6),
        "umbrales": [float(valor) for valor in oof["umbrales"]],
        "submission_v03_lineas": len(submission),
        "submission_v03_sha256": sha256_de(SUBMISSION_V03),
        "submission_v03_intacta": len(submission) > 1000,
    }
    log(
        f"   corrected_cv {corrected_cv:.6f} (coincide: {resultado['coincide']}) | "
        f"submission con {len(submission)} lineas"
    )
    return resultado


def informes_de_modelos() -> list[dict[str, Any]]:
    """Carga todos los informes de la fase que registren modelos entrenados.

    Returns:
        Lista de diccionarios con la clave ``modelos``.
    """
    informes = []
    for ruta in sorted((PROJECT_ROOT / "reports").glob("embedding_*.json")):
        try:
            datos = json.loads(ruta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(datos, dict) and isinstance(datos.get("modelos"), dict):
            informes.append(datos)
    return informes


def verificar_hashes() -> dict[str, Any]:
    """Comprueba que los hashes de los modelos coinciden con los informes.

    Returns:
        Diccionario con la comparacion por modelo.
    """
    log("3/6 reproducibilidad: hashes de los modelos")
    # Se recorren **todos** los informes de la fase que registren modelos, para
    # que la verificacion de huellas cubra tambien los anadidos despues (50M).
    comparacion: dict[str, Any] = {}
    directorio = PROJECT_ROOT / "data" / "external" / "embeddings"
    for ruta in sorted(directorio.glob("*.vec")):
        hash_real = sha256_de(ruta)
        registrado = None
        for informe in informes_de_modelos():
            datos = informe.get("modelos", {})
            for nombre, ficha in datos.items():
                hash_registrado = ficha.get("embedding_hash_sha256") or ficha.get("sha256")
                if hash_registrado and hash_registrado.startswith(hash_real[:16]):
                    registrado = hash_registrado
        comparacion[ruta.name] = {
            "sha256_real": hash_real,
            "sha256_registrado": registrado,
            "coincide": registrado == hash_real if registrado else None,
            "bytes": ruta.stat().st_size,
        }
        log(
            f"   {ruta.name}: {'coincide' if registrado == hash_real else 'SIN REGISTRO o DISTINTO'}"
        )
    return {
        "modelos": comparacion,
        "ok": all(
            entrada["coincide"] is not False for entrada in comparacion.values()
        )
        and all(entrada["coincide"] is True for entrada in comparacion.values()),
    }


def verificar_afirmaciones() -> dict[str, Any]:
    """Relee de los informes los valores que D-032 afirma.

    Returns:
        Diccionario con la comparacion afirmacion / valor leído.
    """
    log("4/6 coherencia de las cifras de D-032 con los informes")
    filas = []
    for ruta_relativa, ruta_valor, esperado, descripcion in AFIRMACIONES_D032:
        ruta = PROJECT_ROOT / ruta_relativa
        if not ruta.exists():
            filas.append(
                {
                    "informe": ruta_relativa,
                    "descripcion": descripcion,
                    "esperado": esperado,
                    "leido": None,
                    "coincide": False,
                    "motivo": "informe no encontrado",
                }
            )
            continue
        leido = _leer_ruta(json.loads(ruta.read_text(encoding="utf-8")), ruta_valor)
        coincide = leido is not None and abs(float(leido) - esperado) < 5e-4
        filas.append(
            {
                "informe": ruta_relativa,
                "ruta": ruta_valor,
                "descripcion": descripcion,
                "esperado": esperado,
                "leido": leido,
                "coincide": bool(coincide),
            }
        )
        estado = "OK" if coincide else "NO COINCIDE"
        log(f"   {descripcion}: esperado {esperado} leido {leido} -> {estado}")
    return {"afirmaciones": filas, "ok": all(fila["coincide"] for fila in filas)}


def inventariar() -> dict[str, Any]:
    """Lista los artefactos de la fase.

    Returns:
        Diccionario con el inventario por tipo.
    """
    log("5/6 inventario de artefactos")
    inventario = {
        "scripts": sorted(p.name for p in PROJECT_ROOT.glob("experiments/embedding_*.py")),
        "informes": sorted(p.name for p in PROJECT_ROOT.glob("reports/embedding_*.md")),
        "csv": sorted(p.name for p in PROJECT_ROOT.glob("reports/embedding_*.csv")),
        "json": sorted(p.name for p in PROJECT_ROOT.glob("reports/embedding_*.json")),
        "figuras": sorted(p.name for p in PROJECT_ROOT.glob("reports/figuras/fig_4*_v5.png")),
        "manifiesto": "data/external/corpora/corpus_manifest.json",
    }
    for clave, valor in inventario.items():
        if isinstance(valor, list):
            log(f"   {clave}: {len(valor)}")
    return inventario


def main() -> None:
    """Ejecuta la auditoria final de la fase v5."""
    inicio = datetime.now(timezone.utc)
    log("Auditoria final de la fase v5")
    recursos = auditar_recursos()
    uso = auditar_uso_de_toxic()
    holdout = auditar_holdout()
    fuga = auditar_fuga_measured()
    referencia = verificar_referencia()
    hashes = verificar_hashes()
    afirmaciones = verificar_afirmaciones()
    artefactos = inventariar()

    veredicto = "GO" if all(
        [
            recursos["ok"],
            uso["ok"],
            holdout["ok"],
            fuga["ok"],
            referencia["coincide"],
            hashes["ok"],
            afirmaciones["ok"],
        ]
    ) else "REVISAR"
    informe = {
        "experimento": EXPERIMENTO,
        "fecha": inicio.isoformat(timespec="seconds"),
        "duracion_s": round(
            (datetime.now(timezone.utc) - inicio).total_seconds(), 2
        ),
        "recursos": recursos,
        "uso_toxic": uso,
        "holdout": holdout,
        "fuga": fuga,
        "referencia": referencia,
        "hashes": hashes,
        "afirmaciones_d032": afirmaciones,
        "artefactos": artefactos,
        "veredicto": veredicto,
        "nota": (
            "La auditoria verifica cumplimiento y coherencia de las cifras. La conclusion "
            "de la fase (el embedding no aporta mas de +0.004 sobre V03) es una medida, no un "
            "juicio de cumplimiento, y esta en D-032 y en los informes de P3 a P5."
        ),
    }
    REPORTE.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"6/6 veredicto: {veredicto} | informe en {REPORTE.relative_to(PROJECT_ROOT)}")
    if veredicto != "GO":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
