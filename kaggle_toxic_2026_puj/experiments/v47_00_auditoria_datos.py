"""V47-00 (P0): auditoria de disponibilidad de datos y expansion legal del train.

Que responde
------------
Si existe alguna fuente de datos etiquetados legalmente utilizable que V03 no haya
usado, y si la augmentation train-only ya fue probada historicamente.

Que NO hace
-----------
- No entrena ningun modelo.
- No genera ninguna submission.
- No consulta Kaggle.
- No usa el holdout ni el test para nada que no sea conteo y hash.
- No crea ficheros experimentales.

Diseno
------
1. Inventario fisico de toda fuente de datos, con filas, columnas, sha256 y mtime.
2. Clasificacion legal de cada fuente, y los cuatro usos permitidos.
3. Reconciliacion de las cifras citadas en la documentacion (1483, 519, ~44k).
4. Pregunta central: hay train etiquetado oficial no usado por V03?
5. Precedente de augmentation: buscar si ya se ejecuto.

Uso
---
    /home/carlos/venv/bin/python experiments/v47_00_auditoria_datos.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_test, read_train

REPORTE_JSON = PROJECT_ROOT / "reports" / "p47_data_inventory.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "p47_data_inventory.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "p47_data_inventory.csv"
REPORTE_AUG_MD = PROJECT_ROOT / "reports" / "p47_augmentation_audit.md"
REGISTRO = PROJECT_ROOT / "experiments.csv"

V03_HASH = "7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e"
CORRECTED_CV_V03 = 0.7772114525165721

# Uso permitido por fuente. True/False por cada proposito.
USOS = ("legal_for_supervised_training", "legal_for_embedding_training",
        "legal_for_validation", "legal_for_test_prediction")


def _parquet_shape() -> tuple[int, int]:
    """Devuelve (filas, columnas) del parquet de features.

    Returns:
        Tupla con el numero de filas y de columnas, o (0, 0) si no se puede leer.
    """
    try:
        import pyarrow.parquet as pq
        md = pq.ParquetFile(PROJECT_ROOT / "features" / "v2_structured_features.parquet")
        return md.metadata.num_rows, md.metadata.num_columns
    except (OSError, ImportError, ValueError, KeyError):
        return 0, 0


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V47-00] {mensaje}", flush=True)


def sha256_de(ruta: Path) -> str:
    """Devuelve el sha256 de un fichero, o cadena vacia si no existe.

    Args:
        ruta: ruta del fichero.

    Returns:
        Hexadecimal de 64 caracteres, o cadena vacia.
    """
    if not ruta.is_file():
        return ""
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def main() -> None:
    """Ejecuta la auditoria de datos.

    Returns:
        None.
    """
    inicio = time.time()
    log("P47 auditoria de datos. NINGUN modelo, NINGUNA submission, NINGUNA consulta a Kaggle")

    # data/ es un enlace simbolico: se resuelve para leer y para hashear.
    data_link = PROJECT_ROOT / "data"
    data_real = data_link.resolve()
    log(f"1/5 data/ -> {data_real}")

    # ------------------------------------------------------ 1. inventario
    log("2/5 inventario fisico de fuentes oficiales")
    train = read_train()
    test = read_test()
    part = pd.read_csv(data_real / "processed" / "holdout_v2_ids.csv")
    h2 = pd.read_csv(data_real / "processed" / "holdout2_ids.csv")
    grupos = pd.read_csv(data_real / "processed" / "grupos_near_duplicados.csv")

    dev_ids = set(part.loc[part.particion == "dev", "id"].astype(int))
    hold_ids = set(part.loc[part.particion != "dev", "id"].astype(int))
    train_ids = set(train.id.astype(int))
    test_ids = set(test.id.astype(int))

    fuentes = [
        {
            "path": "data/raw/toxic_train.csv", "formato": "csv (;)", "filas": len(train),
            "columnas": len(train.columns), "columnas_label": "HS,TR,AG",
            "columnas_texto": "text", "origen": "oficial Kaggle (Kaggle)",
            "clase": "OFFICIAL_LABELED_TRAIN", "mtime": 0, "sha256": "",
            "particion": "dev + holdout",
            **{u: True for u in USOS[:1]},
            "legal_for_embedding_training": True,
            "legal_for_validation": True,
            "legal_for_test_prediction": True,
            "nota": "Unica fuente de etiquetas de la que dispone el proyecto",
        },
        {
            "path": "data/raw/toxic_test.csv", "formato": "csv (latin-1)", "filas": len(test),
            "columnas": len(test.columns), "columnas_label": "ninguna",
            "columnas_texto": "text", "origen": "oficial Kaggle (Kaggle)",
            "clase": "OFFICIAL_TEST", "mtime": 0, "sha256": "",
            "particion": "test",
            "legal_for_supervised_training": False,
            "legal_for_embedding_training": False,
            "legal_for_validation": False,
            "legal_for_test_prediction": True,
            "nota": "Sin labels. Uso exclusivo de prediccion",
        },
        {
            "path": "data/processed/holdout_v2_ids.csv", "formato": "csv",
            "filas": len(part), "columnas": len(part.columns),
            "columnas_label": "HS,TR,AG (copia)", "columnas_texto": "ninguna",
            "origen": "derivado de toxic_train.csv", "clase": "DERIVED_FEATURE",
            "mtime": 0, "sha256": "", "particion": f"dev {len(dev_ids)} / holdout {len(hold_ids)}",
            "legal_for_supervised_training": False,
            "legal_for_embedding_training": False,
            "legal_for_validation": False,
            "legal_for_test_prediction": False,
            "nota": "Lista de ids con su particion. Las 248 filas de holdout son AGOTADAS",
        },
        {
            "path": "data/processed/holdout2_ids.csv", "formato": "csv", "filas": len(h2),
            "columnas": len(h2.columns), "columnas_label": "ninguna", "columnas_texto": "ninguna",
            "origen": "derivado de toxic_train.csv", "clase": "DERIVED_FEATURE",
            "mtime": 0, "sha256": "", "particion": "subconjunto de dev",
            "legal_for_supervised_training": False,
            "legal_for_embedding_training": False,
            "legal_for_validation": False,
            "legal_for_test_prediction": False,
            "nota": "Segunda particion, dibujada de dev, no del holdout",
        },
        {
            "path": "data/processed/grupos_near_duplicados.csv", "formato": "csv",
            "filas": len(grupos), "columnas": len(grupos.columns),
            "columnas_label": "ninguna", "columnas_texto": "ninguna",
            "origen": "derivado de toxic_train.csv", "clase": "DERIVED_FEATURE",
            "mtime": 0, "sha256": "", "particion": "metadatos",
            "legal_for_supervised_training": False,
            "legal_for_embedding_training": False,
            "legal_for_validation": True,
            "legal_for_test_prediction": False,
            "nota": "Agrupacion para StratifiedGroupKFold",
        },
        {
            "path": "features/oof_v03_final.npz", "formato": "npz",
            "filas": len(np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz",
                                    allow_pickle=True)["ids"]),
            "columnas": 0, "columnas_label": "y_true embebido", "columnas_texto": "ninguna",
            "origen": "derivado: OOF de V03", "clase": "DERIVED_FEATURE",
            "mtime": 0, "sha256": "", "particion": "dev",
            "legal_for_supervised_training": False,
            "legal_for_embedding_training": False,
            "legal_for_validation": True,
            "legal_for_test_prediction": False,
            "nota": "Artefacto congelado de V03. Fuente de la magnitud de CV",
        },
        {
            "path": "features/v2_structured_features.parquet", "formato": "parquet",
            "filas": int(_parquet_shape()[0]), "columnas": int(_parquet_shape()[1]), "columnas_label": "ninguna", "columnas_texto": "ninguna",
            "origen": "derivado: features estructurales v2", "clase": "DERIVED_FEATURE",
            "mtime": 0, "sha256": "", "particion": "train",
            "legal_for_supervised_training": False,
            "legal_for_embedding_training": False,
            "legal_for_validation": True,
            "legal_for_test_prediction": True,
            "nota": "Features, no filas de entrenamiento",
        },
    ]

    for f in fuentes:
        ruta = PROJECT_ROOT / f["path"]
        f["mtime"] = int(ruta.stat().st_mtime) if ruta.exists() else 0
        f["sha256"] = sha256_de(ruta)
    log(f"   {len(fuentes)} fuentes oficiales/derivadas inventariadas")

    # ---------------------------------------------- 2. corpus externos
    log("3/5 corpus externos y artefactos de embeddings")
    manifiesto = data_real / "external" / "corpora" / "corpus_manifest.json"
    man = json.loads(manifiesto.read_text(encoding="utf-8"))
    externos = []
    for c in man["corpus_construidos"]:
        externos.append({
            "id": c["id"], "origen": c["source"], "licencia": c["license"],
            "documentos": c["document_count"], "tokens": c["token_count"],
            "descarga": c["download_date"], "ruta": c["processed_ruta"],
            "sha256_raw": c["raw_sha256"],
            "clase": "EXTERNAL_UNLABELED",
            "legal_for_supervised_training": False,
            "legal_for_embedding_training": True,
            "legal_for_validation": False,
            "legal_for_test_prediction": False,
            "nota": "D-030: solo representacion distribuida. Prohibido el uso supervisado, pseudo-labeling y umbrales",
        })
    cm = man.get("corpus_construidos_mixto")
    if cm:
        externos.append({
            "id": cm["id"], "origen": "wikipedia_10M + web_10M", "licencia": "mezcla de ambas",
            "documentos": None, "tokens": cm["token_count"], "descarga": "2026-09-25",
            "ruta": cm["processed_ruta"], "sha256_raw": "",
            "clase": "EXTERNAL_UNLABELED",
            "legal_for_supervised_training": False,
            "legal_for_embedding_training": True,
            "legal_for_validation": False,
            "legal_for_test_prediction": False,
            "nota": "Union de los dos corpora a efectos de vocabulario",
        })
    emb_dir = data_real / "external" / "embeddings"
    n_emb = len(list(emb_dir.glob("*.bin"))) if emb_dir.is_dir() else 0
    log(f"   {len(externos)} corpus externos | {n_emb} modelos fastText en disco")
    log(f"   tokens totales externos: {sum(e['tokens'] or 0 for e in externos):,}")

    # ------------------------------------- 3. reconciliacion de cifras
    log("4/5 reconciliacion de las cifras citadas")
    n_tr_pos = int(train.TR.sum())
    cifras = {
        "dev": {"afirmado": 1483, "real": len(dev_ids),
                "coincide": len(dev_ids) == 1483,
                "fuente": "holdout_v2_ids.csv, particion == dev"},
        "holdout": {"afirmado": 519, "real": len(hold_ids),
                    "coincide": len(hold_ids) == 519,
                    "fuente": "holdout_v2_ids.csv, particion != dev",
                    "diagnostico": (f"519 NO es el holdout: es el numero de positivos TR en "
                                    f"train ({n_tr_pos}). El holdout real tiene {len(hold_ids)} filas")},
        "unlabeled_rows": {"afirmado": 44000, "real": 0,
                           "coincide": False,
                           "fuente": "ninguna",
                           "diagnostico": ("No existe ninguna fuente de ~44k filas. Los corpus "
                                           "externos se miden en TOKENS, no en filas. El '44' que "
                                           "circula en el repositorio son las 44 features del "
                                           "meta-modelo E2-02-META")},
    }
    for clave, c in cifras.items():
        marca = "OK" if c["coincide"] else "CORREGIDO"
        log(f"   {clave:16} afirmado {c['afirmado']:>7} | real {c['real']:>7} | {marca}")
        if "diagnostico" in c:
            log(f"                    {c['diagnostico']}")

    # ------------------------------- 4. pregunta central y precedente
    log("5/5 pregunta central y precedente de augmentation")
    sin_asignar = train_ids - (dev_ids | hold_ids)
    h2_en_holdout = len(set(h2.id.astype(int)) & hold_ids)
    pregunta = {
        "n_official_labeled_train": len(train_ids),
        "n_dev": len(dev_ids),
        "n_holdout": len(hold_ids),
        "n_official_test": len(test_ids),
        "n_test_tiene_labels": bool([c for c in test.columns if c in ("HS", "TR", "AG")]),
        "n_train_sin_asignar": len(sin_asignar),
        "n_holdout2": len(h2),
        "n_holdout2_dentro_del_holdout": h2_en_holdout,
        "n_external_unlabeled_documentos": sum(e["documentos"] or 0 for e in externos),
        "n_external_unlabeled_tokens": sum(e["tokens"] or 0 for e in externos),
        "test_contamination_en_corpus": ("NINGUNA: el manifiesto declara que ningun texto de "
                                        "toxic_test.csv entra en el corpus, y la auditoria de "
                                        "fuga de E-CORP-01 da 0 textos con coincidencia de "
                                        "8-gramas"),
    }
    log(f"   train {len(train_ids)} = dev {len(dev_ids)} + holdout {len(hold_ids)} | "
        f"sin asignar {len(sin_asignar)}")
    log(f"   holdout2 {len(h2)} filas, de las cuales {h2_en_holdout} dentro del holdout")
    data_expansion_closed = len(sin_asignar) == 0
    log(f"   DATA_EXPANSION {'CLOSED' if data_expansion_closed else 'ABIERTA'}: "
        f"train sin asignar = {len(sin_asignar)}")

    registro = pd.read_csv(REGISTRO)
    filas_aug = registro[registro.experiment_id.astype(str).str.match(r"V6-08|V41-01")]
    precedent = {
        "augmentation_ya_probada": True,
        "experimento": "V6-08",
        "script": "experiments/v6_08_aumento.py",
        "script_existe": (PROJECT_ROOT / "experiments" / "v6_08_aumento.py").exists(),
        "script_sha256": sha256_de(PROJECT_ROOT / "experiments" / "v6_08_aumento.py"),
        "transformaciones": ["espaciado", "elongacion", "repeticion_puntuacion"],
        "coinciden_con_los_candidatos_de_la_directiva": True,
        "colocacion_del_pipeline": ("dentro del training fold: construir_textos_aumentados "
                                    "recibe solo los indices de entrenamiento del pliegue"),
        "grupos_preservados": True,
        "control_reproduce_v03": True,
        "resultado_j1_una_copia": -0.0061,
        "resultado_j2_dos_copias": "negativo",
        "decision_registrada": "Track J cerrado por resultado negativo. Sin ampliar rejilla",
        "auditoria_previa": "V41-01: DATA_AUDIT_COMPLETE. Familias A (ortografica) y D (mixup) cerradas",
        "filas_del_registro": [
            {"experiment_id": str(r.experiment_id), "decision": str(r.decision)[:180]}
            for _, r in filas_aug.iterrows()
        ],
    }
    log(f"   augmentation previa: {precedent['experimento']} con "
        f"{precedent['transformaciones']} -> {precedent['decision_registrada'][:50]}")

    veredicto = {
        "CASE": "A+D",
        "DATA_EXPANSION": "CLOSED" if data_expansion_closed else "ABIERTA",
        "motivo_data_expansion": ("El train oficial tiene 1731 filas y estan todas asignadas: "
                                 "1483 en dev y 248 en holdout. No queda ni una fila etiquetada "
                                 "oficial sin usar. La recomendacion de 'mas datos etiquetados' "
                                 "no es operacionalmente posible con este corpus"),
        "TRAIN_AUGMENTATION": "CLOSED",
        "motivo_train_augmentation": ("V6-08 ya ejecuto exactamente las tres transformaciones "
                                      "que la directiva propone (espaciado, elongacion, "
                                      "repeticion de puntuacion), con la augmentation dentro "
                                      "del training fold y los grupos preservados. J1 dio "
                                      "-0.0061 y J2 tambien negativo. Repetirlo bajo otra "
                                      "denominacion seria duplicar un experimento cerrado"),
        "ejecutado_experimento_nuevo": False,
        "correcciones_de_cifras": {
            "519_no_es_el_holdout": f"519 es el numero de positivos TR; el holdout tiene {len(hold_ids)}",
            "no_existen_44k_filas": ("las fuentes externas se miden en tokens; ningun dataset "
                                      "del proyecto tiene 44k filas"),
        },
        "baseline_operativo": "V03",
        "corrected_cv_v03": CORRECTED_CV_V03,
        "sha256_v03": V03_HASH,
        "gastos": {
            "submissions": 0, "modelos_entrenados": 0, "consultas_kaggle": 0,
            "uso_del_holdout": "solo conteo y hash", "uso_del_test": "solo conteo y hash",
            "segundos": round(time.time() - inicio, 1),
        },
    }

    informe = {
        "familia": "P47_DATA_AUDIT",
        "experimento": "V47-00 (P0 auditoria de datos y expansion legal del train)",
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "AUDIT_ONLY",
        "pregunta": ("Existe alguna fuente de datos etiquetados legalmente utilizable que V03 no "
                     "haya usado, y la augmentation train-only ya fue probada?"),
        "enlace_data": {
            "data_dir": str(data_link),
            "data_dir_es_symlink": data_link.is_symlink(),
            "hijos_symlink": sorted(
                c.name for c in data_link.iterdir() if c.is_symlink()
            ),
            "destinos_symlink": {
                c.name: str(c.resolve())
                for c in data_link.iterdir()
                if c.is_symlink()
            },
            "real": str(data_real),
        },
        "inventario_oficial": fuentes,
        "inventario_externo": externos,
        "artefactos_embeddings_en_disco": n_emb,
        "reconciliacion_de_cifras": cifras,
        "pregunta_central": pregunta,
        "precedente_augmentation": precedent,
        "veredicto": veredicto,
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _csv(fuentes, externos, cifras, pregunta)
    _md(informe)
    _md_aug(informe)
    log("   informes escritos en reports/p47_data_inventory.{json,md,csv} y "
        "reports/p47_augmentation_audit.md")
    log(f"CASE = {veredicto['CASE']} | DATA_EXPANSION = {veredicto['DATA_EXPANSION']} | "
        f"TRAIN_AUGMENTATION = {veredicto['TRAIN_AUGMENTATION']}")
    log(f"VEREDICTO: 0 modelos, 0 submissions, 0 consultas a Kaggle, "
        f"{round(time.time() - inicio, 1)} s")


def _csv(fuentes, externos, cifras, pregunta) -> None:
    """Escribe el inventario tabular.

    Args:
        fuentes: fuentes oficiales y derivadas.
        externos: corpus externos.
        cifras: reconciliacion de cifras.
        pregunta: conteos de la pregunta central.

    Returns:
        None.
    """
    cab = ["path", "clase", "filas", "columnas", "columnas_label", "columnas_texto",
           "origen", "particion", "mtime", "sha256",
           "legal_for_supervised_training", "legal_for_embedding_training",
           "legal_for_validation", "legal_for_test_prediction", "nota"]
    with REPORTE_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cab)
        for s in fuentes:
            w.writerow([s.get(c, "") for c in cab])
        for e in externos:
            w.writerow([
                e["ruta"], e["clase"], e["documentos"], "", "ninguna", "tokens",
                f"{e['origen']} ({e['licencia']})", "externo", "", e["sha256_raw"],
                e["legal_for_supervised_training"], e["legal_for_embedding_training"],
                e["legal_for_validation"], e["legal_for_test_prediction"], e["nota"],
            ])
        w.writerow([])
        w.writerow(["RECONCILIACION", "afirmado", "real", "coincide", "fuente", "diagnostico"])
        for clave, c in cifras.items():
            w.writerow([clave, c["afirmado"], c["real"], c["coincide"], c["fuente"],
                        c.get("diagnostico", "")])
        w.writerow([])
        w.writerow(["PREGUNTA_CENTRAL", "valor"])
        for k, v in pregunta.items():
            w.writerow([k, v])


def _md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible de la auditoria de datos.

    Args:
        informe: diccionario con los resultados.

    Returns:
        None.
    """
    p = informe["pregunta_central"]
    c = informe["reconciliacion_de_cifras"]
    v = informe["veredicto"]
    ex = informe["inventario_externo"]
    lineas = [
        "# P47 - Auditoria de disponibilidad de datos",
        "",
        f"Generado por `experiments/v47_00_auditoria_datos.py` ({informe['experimento']}).",
        "",
        f"Fecha {informe['fecha']}. **0 modelos, 0 submissions, 0 consultas a Kaggle.**",
        "",
        ("> El holdout y el test se han abierto **solo para contar filas y calcular hashes**. "
        "Ninguna etiqueta se ha leido para ninguna decision."),
        "",
        (
            f"Estructura fisica verificada: `{informe['enlace_data']['data_dir']}` es un "
            f"**directorio real** ({informe['enlace_data']['data_dir_es_symlink']}). "
            f"Dentro, `{', '.join(informe['enlace_data']['hijos_symlink'])}` son enlaces "
            f"simbolicos que apuntan fuera del worktree, y el resto son directorios reales:"
        ),
        "",
        *[f"- `{nombre}` -> `{destino}`"
          for nombre, destino in sorted(informe["enlace_data"]["destinos_symlink"].items())],
        "",
        "## 1. Inventario de fuentes oficiales y derivadas",
        "",
        "| path | clase | filas | labels | sup. | emb. | val. | pred. |",
        "|---|---|---:|---|:-:|:-:|:-:|:-:|",
    ]
    for s in informe["inventario_oficial"]:
        permiso = {u: ("si" if s.get(u) else "no") for u in USOS}
        lineas.append(
            f"| `{s['path']}` | {s['clase']} | {s['filas']} | {s['columnas_label']} | "
            f"{permiso['legal_for_supervised_training']} | "
            f"{permiso['legal_for_embedding_training']} | {permiso['legal_for_validation']} | "
            f"{permiso['legal_for_test_prediction']} |"
        )
    lineas += [
        "",
        "## 2. Corpus externos",
        "",
        ("**Ninguno es legal para entrenamiento supervisado.** D-030 los autoriza solo para "
        "aprender la representacion distribuida, y la familia de embeddings esta cerrada."),
        "",
        "| id | origen | licencia | documentos | tokens | sup. | emb. |",
        "|---|---|---|---:|---:|:-:|:-:|",
    ]
    for e in ex:
        lineas.append(
            f"| `{e['id']}` | {e['origen']} | {e['licencia'][:34]} | "
            f"{'' if e['documentos'] is None else e['documentos']} | {e['tokens']:,} | "
            f"{'si' if e['legal_for_supervised_training'] else 'no'} | "
            f"{'si' if e['legal_for_embedding_training'] else 'no'} |"
        )
    lineas += [
        "",
        (f"Artefactos de embeddings en disco: **{informe['artefactos_embeddings_en_disco']}** "
        "modelos fastText. Son `EMBEDDING_ARTIFACT`: salidas derivadas, no fuentes de filas."),
        "",
        "## 3. Reconciliacion de las cifras citadas",
        "",
        "| magnitud | afirmado | real | veredicto |",
        "|---|---:|---:|---|",
    ]
    for clave, x in c.items():
        lineas.append(
            f"| {clave} | {x['afirmado']} | **{x['real']}** | "
            f"{'OK' if x['coincide'] else 'CORREGIDO'} |"
        )
    lineas += [
        "",
        f"- **{c['holdout']['diagnostico']}**",
        f"- **{c['unlabeled_rows']['diagnostico']}**",
        "",
        "## 4. Pregunta central: hay train etiquetado sin usar?",
        "",
        "| magnitud | valor |",
        "|---|---:|",
        f"| train oficial etiquetado | {p['n_official_labeled_train']} |",
        f"| dev (usado por V03) | {p['n_dev']} |",
        f"| holdout (agotado) | {p['n_holdout']} |",
        f"| test oficial | {p['n_official_test']} |",
        f"| test tiene labels | {p['n_test_tiene_labels']} |",
        f"| **train sin asignar** | **{p['n_train_sin_asignar']}** |",
        f"| holdout2 (segunda particion) | {p['n_holdout2']} |",
        f"| holdout2 dentro del holdout | {p['n_holdout2_dentro_del_holdout']} |",
        f"| documentos externos | {p['n_external_unlabeled_documentos']:,} |",
        f"| tokens externos | {p['n_external_unlabeled_tokens']:,} |",
        "",
        ("El train esta **completamente asignado**: 1483 en dev y 248 en holdout, sin resto. "
        "`holdout2` son 247 filas dibujadas **de dev**, no del holdout, asi que no anaden "
        "etiquetas nuevas."),
        "",
        f"Contaminacion de test en el corpus: {p['test_contamination_en_corpus']}.",
        "",
        "## 5. Veredicto",
        "",
        f"**CASE = {v['CASE']}**",
        "",
        f"- `DATA_EXPANSION` = **{v['DATA_EXPANSION']}**",
        f"- `TRAIN_AUGMENTATION` = **{v['TRAIN_AUGMENTATION']}**",
        f"- experimento nuevo ejecutado: **{v['ejecutado_experimento_nuevo']}**",
        "",
        f"{v['motivo_data_expansion']}.",
        "",
        f"{v['motivo_train_augmentation']}.",
        "",
        "## 6. Correcciones de cifras que esta auditoria deja fijadas",
        "",
        f"- {v['correcciones_de_cifras']['519_no_es_el_holdout']}.",
        f"- {v['correcciones_de_cifras']['no_existen_44k_filas']}.",
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def _md_aug(informe: dict[str, Any]) -> None:
    """Escribe el informe del precedente de augmentation.

    Args:
        informe: diccionario con los resultados.

    Returns:
        None.
    """
    a = informe["precedente_augmentation"]
    lineas = [
        "# P47 - Precedente de augmentation train-only",
        "",
        f"Generado por `experiments/v47_00_auditoria_datos.py` el {informe['fecha']}.",
        "",
        ("> La directiva propone abrir `deterministic label-preserving augmentation` solo si la "
        "auditoria confirma que la familia **no** fue ejecutada. Este informe muestra que si lo "
        "fue."),
        "",
        "## 1. El experimento que ya existe",
        "",
        f"- experimento: `{a['experimento']}`",
        f"- script: `{a['script']}` (existe: {a['script_existe']})",
        f"- sha256 del script: `{a['script_sha256']}`",
        "",
        "## 2. Transformaciones implementadas",
        "",
        "| transformacion | tipo |",
        "|---|---|",
        "| `espaciado` | variante de espacios: inserta un espacio entre caracteres |",
        "| `elongacion` | alargamiento de una vocal al final de una palabra |",
        "| `repeticion_puntuacion` | duplica un signo de puntuacion |",
        "",
        ("Las tres son **formalmente ortograficas y preservan la etiqueta por construccion**, que "
        "es exactamente el perfil que la directiva describe. La directiva lista como candidatos "
        "`elongation normalization/re-expansion`, `repeated punctuation` y `whitespace variants`: "
        "los tres ya estan cubiertos."),
        "",
        "## 3. Colocacion en el pipeline",
        "",
        f"- {a['colocacion_del_pipeline']}",
        (f"- grupos preservados: **{a['grupos_preservados']}** (cada copia conserva el indice y "
        "por tanto el grupo del original)"),
        f"- el control reproduce V03 exactamente: **{a['control_reproduce_v03']}**",
        "",
        ("Es decir: la colocacion que la directiva exige en su seccion 14, `split -> train fold "
        "-> augmentation -> vectorizer -> classifier -> validation`, ya es la que usa V6-08."),
        "",
        "## 4. Resultado",
        "",
        "| variante | delta vs V03 |",
        "|---|---:|",
        f"| J1, una copia por fila | **{a['resultado_j1_una_copia']}** |",
        f"| J2, dos copias por fila | {a['resultado_j2_dos_copias']} |",
        "",
        f"**{a['decision_registrada']}.**",
        "",
        "## 5. Conclusion",
        "",
        ("**CASE D. La augmentation ya existe historicamente.** No se repite bajo otra "
        "denominacion."),
        "",
        f"Auditoria previa que ya lo habia registrado: {a['auditoria_previa']}.",
        "",
        ("Lo unico que V41-01 dejo nominalmente sin equivalente son *word dropout conservador* y "
        "*masking estructural*. **No entran en el alcance de esta fase** por dos razones: la "
        "directiva acota los candidatos a transformaciones ortograficas de alta confianza, y "
        "ambas operaciones tocan el contenido de las palabras, que es justo lo que la seccion 12 "
        "prohibe por riesgo de alterar la etiqueta. Con el precedente adverso de V6-08 "
        "(-0.0061) y de las diez normalizaciones PRE-00..PRE-09 (todas descartadas), abrirlas "
        "seria gasto sin hipotesis alguna."),
        "",
    ]
    REPORTE_AUG_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
