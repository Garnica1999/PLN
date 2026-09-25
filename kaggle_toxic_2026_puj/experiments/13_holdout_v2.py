"""P0.2 - Congela la particion desarrollo/holdout del protocolo v2 (D-016).

Pregunta que responde
---------------------
¿Cual es la particion desarrollo/holdout que se usara para el resto de la
investigacion v2, y queda registrada de forma verificable antes de mirar
resultados?

Metodo
------
1. Se cargan train y los grupos fusionados (near-duplicado UNION pseudo-autor).
2. Se aplica ``StratifiedGroupKFold(n_splits=7, shuffle=True, seed=42)``
   estratificando por la combinacion HS x TR x AG.
3. Se congela el pliegue 0 como holdout (~14 % del corpus) y el resto como
   desarrollo, verificando que ningun grupo cruce la frontera.
4. Se guardan la particion (ids), su hash y las reglas de uso en
   ``config/holdout_v2.json`` y ``data/processed/holdout_v2_ids.csv``.

Regla de uso: maximo 5 evaluaciones al holdout en todo el proyecto, solo para
candidatos finalistas y con registro en ``RUNLOG.md``.

Salidas
-------
``config/holdout_v2.json``, ``data/processed/holdout_v2_ids.csv`` y una fila
``HOLD-01`` en ``experiments.csv``.

Ejecucion
---------
    /home/carlos/venv/bin/python experiments/13_holdout_v2.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_module
from src.splits import (
    ETIQUETAS,
    N_SPLITS_HOLDOUT,
    RANDOM_STATE,
    combinaciones,
    grupos_fusionados,
    holdout_v2,
)

CONFIG_PATH = PROJECT_ROOT / "config" / "holdout_v2.json"
IDS_PATH = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
EXPERIMENTS_CSV = PROJECT_ROOT / "experiments.csv"
MAX_EVALUACIONES = 5


def log(message: str) -> None:
    """Imprime un mensaje de progreso con prefijo de fase.

    Args:
        message: texto a imprimir.
    """
    print(f"[P0.2] {message}", flush=True)


def sha256_of(path: Path) -> str:
    """Calcula el sha256 de un archivo.

    Args:
        path: ruta del archivo.

    Returns:
        Cadena hexadecimal con el hash.
    """
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def distribucion(frame: pd.DataFrame, columna: str) -> dict:
    """Cuenta cuantas filas (o grupos) caen en cada combinacion de etiquetas.

    Args:
        frame: subconjunto del corpus.
        columna: nombre de la columna con la combinacion.

    Returns:
        Diccionario ``combinacion -> conteo`` ordenado.
    """
    conteo = frame[columna].value_counts().sort_index()
    return {clave: int(valor) for clave, valor in conteo.items()}


def main() -> None:
    """Congela la particion desarrollo/holdout y la registra."""
    train = data_module.read_train()
    grupos = grupos_fusionados(train)
    indices_dev, indices_holdout = holdout_v2(train, grupos)
    log(f"dev {len(indices_dev)} filas | holdout {len(indices_holdout)} filas")

    frame = train.copy()
    frame["grupo_near_dup_autor"] = grupos
    frame["combinacion"] = combinaciones(train)
    frame["particion"] = "dev"
    frame.loc[indices_holdout, "particion"] = "holdout"
    # Verificacion de fuga: ningun grupo puede tener filas en ambas particiones.
    partes = frame.groupby("grupo_near_dup_autor")["particion"].nunique()
    if (partes > 1).any():
        raise RuntimeError("Hay grupos con filas en dev y holdout a la vez")

    resumen = {
        "fecha": "2026-09-25",
        "protocolo": "v2 (D-016)",
        "grupos": "near-duplicado UNION pseudo-autor (primera mencion)",
        "particion": f"StratifiedGroupKFold(n_splits={N_SPLITS_HOLDOUT}, shuffle=True, seed={RANDOM_STATE})",
        "pliegue_congelado": 0,
        "n_tweets_dev": len(indices_dev),
        "n_tweets_holdout": len(indices_holdout),
        "n_grupos_total": len(set(grupos.tolist())),
        "n_grupos_holdout": len(set(grupos[indices_holdout].tolist())),
        "distribucion_dev": distribucion(frame[frame["particion"] == "dev"], "combinacion"),
        "distribucion_holdout": distribucion(
            frame[frame["particion"] == "holdout"], "combinacion"
        ),
        "regla_de_uso": {
            "maximo_evaluaciones": MAX_EVALUACIONES,
            "solo_finalistas": True,
            "registro": "RUNLOG.md",
        },
        "hash_ids_csv": "",
    }
    IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame[["id", "particion", "grupo_near_dup_autor", "combinacion"] + list(ETIQUETAS)].to_csv(
        IDS_PATH, index=False
    )
    resumen["hash_ids_csv"] = sha256_of(IDS_PATH)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"config escrita: {CONFIG_PATH} (sha256 ids {resumen['hash_ids_csv'][:16]}...)")

    fila = {
        "experiment_id": "HOLD-01",
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": "Cual es la particion desarrollo/holdout del protocolo v2?",
        "hypothesis": "Un holdout agrupado y estratificado congelado controla la multiplicidad",
        "preprocessing": "no aplica",
        "feature_groups": "no aplica",
        "representation": "no aplica",
        "model": "no aplica (particion)",
        "hyperparameters": json.dumps(
            {"n_splits": N_SPLITS_HOLDOUT, "seed": RANDOM_STATE}, ensure_ascii=False
        ),
        "validation": "StratifiedGroupKFold(7) sobre grupos fusionados",
        "f1": "",
        "macro_f1": "",
        "weighted_f1": "",
        "precision": "",
        "recall": "",
        "training_time": "",
        "dimension": "",
        "result": json.dumps(
            {
                "n_dev": resumen["n_tweets_dev"],
                "n_holdout": resumen["n_tweets_holdout"],
                "n_grupos_holdout": resumen["n_grupos_holdout"],
                "hash_ids_csv": resumen["hash_ids_csv"],
            },
            ensure_ascii=False,
        ),
        "interpretation": "El desarrollo (6/7) se usa para experimentar; el holdout (1/7) se reserva",
        "decision": "Congelado; maximo 5 evaluaciones al holdout, solo finalistas",
        "date": "2026-09-25",
        "script": "experiments/13_holdout_v2.py",
        "notes": "Ningun grupo cruza la frontera dev/holdout (verificado)",
    }
    existente = pd.read_csv(EXPERIMENTS_CSV) if EXPERIMENTS_CSV.exists() else pd.DataFrame()
    if "HOLD-01" in existente.get("experiment_id", pd.Series(dtype=str)).tolist():
        existente = existente[existente["experiment_id"] != "HOLD-01"]
        existente.to_csv(EXPERIMENTS_CSV, index=False)
    pd.DataFrame([fila]).to_csv(
        EXPERIMENTS_CSV, mode="a", header=not EXPERIMENTS_CSV.exists(), index=False
    )
    log("fila HOLD-01 registrada en experiments.csv")
    log(f"holdout: {resumen['distribucion_holdout']}")


if __name__ == "__main__":
    main()
