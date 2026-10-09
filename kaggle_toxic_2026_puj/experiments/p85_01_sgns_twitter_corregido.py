"""P85-01: SGNS corregido sobre corpus Twitter/X (reapertura experimental unica).

Autorizacion: reapertura experimental unica y estrictamente acotada (2026-10-08).
Preregistro congelado antes de la primera ejecucion completa: `docs/p85_preregistro.md`.

Diseno:
- Brazo A (control general): embedding S de P84-04 (corpus general, semilla 20261016),
  reutilizado SIN reentrenar; su downstream se re-puntua con la normalizacion
  documentada (EMB-NORMAL-TW v1), porque el pooling publicado de P84-04 uso texto
  crudo (cobertura 0.6585) y no es comparable bajo la regla de P78.
- Brazo B (candidato Twitter/X): UN entrenamiento SGNS nuevo (semilla 20261016) con la
  implementacion exacta de P84-04 (agregacion "suma"), corpus Twitter/X autorizado.
- Normalizacion del texto de desarrollo: EMB-NORMAL-TW v1, identica en los dos brazos.
- Downstream: pooling medio + StandardScaler + LR C=2.0 balanced liblinear rs=42;
  umbrales congelados de V03 (primaria) y anidados (secundaria); pliegues D-016.
- Metricas: embedding-only y fusion (media de probabilidades) por brazo; primaria = F1
  con umbrales congelados de la fusion; bootstrap pareado 10000/20260926;
  complementariedad V03 x candidato y V03 x fusion.

Uso:
    /home/carlos/venv/bin/python experiments/p85_01_sgns_twitter_corregido.py
    /home/carlos/venv/bin/python experiments/p85_01_sgns_twitter_corregido.py --humo

Nota (2026-10-08, posterior a la corrida): los dos escritores CSV fijan
`lineterminator="\n"`; la corrida original uso el terminador por defecto del
modulo csv (CRLF) y la fila P85-01 se re-anexo manualmente con LF (Enmienda 2
del preregistro). Los resultados no cambian.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import unicodedata
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

import p84_04_sgns_corregido_anclado as base
from src.evaluation import bootstrap_diferencia_pareada, metrica_macro

# ===== SECCION 0: constantes congeladas =====

SEMILLA: Final[int] = 20261016
HASH_TWITTER: Final[str] = "1ef872c189d0f405bc1a3f8281cafe08e6b099958e6b8a3816db1460df91f8b9"
HASH_GENERAL: Final[str] = "00b839da6fbc377ee5375acc9a236e831e46ecf92cc544188b81aef1a826e871"
HASH_EMB_GENERAL: Final[str] = "0216f9f6a952937925d8dcdd1cd885db8f776efed9ed47c339348c840bdff853"
HASH_OOF_V03: Final[str] = "559265816ce4f6312c87f008f90324f0eec70a74d1e6247364c162793338feab"
HASH_TRAIN: Final[str] = "3d40e7345b588860cf4744914be88ddc094e4f03591516c8176ab40b7f76d42f"
HASH_GRUPOS: Final[str] = "afabf9ae1c295ea8f203120bf28578cb4bdb4e8f16b89178d820baeaac62e622"
NOTA_FASTTEXT: Final[str] = (
    "el binario fastText no participa en este experimento (la implementacion SGNS es PyTorch); "
    "no se verifica su hash. El binario en disco (0307cde9a9e9dc02f2b688e4645ffbb38b8be2f80c33530777d7abd8a031f167) "
    "difiere del congelado de P84-04 (59c51cc53802bcad9e7832145f6525a281d22d42e7c476bfbac545c0c392137f, "
    "del worktree v67 ya retirado) y no se usa aqui."
)
V03_CCV: Final[float] = 0.7772114525165721
BOOTSTRAP_REPS: Final[int] = 10000
BOOTSTRAP_SEMILLA: Final[int] = 20260926
UMBRALES: Final[np.ndarray] = base.UMBRALES_CONGELADOS
ETIQUETAS: Final[tuple[str, ...]] = base.ETIQUETAS

RUTA_TWITTER: Final[Path] = (
    PROJECT_ROOT / "data" / "external" / "corpora" / "twitter_domain" / "corpus_twitter_tokens.txt"
)
RUTA_GENERAL: Final[Path] = (
    PROJECT_ROOT / "data" / "external" / "corpora" / "wikipedia_es" / "emb_normal" / "tokens_10M.txt"
)
RUTA_EMB_GENERAL: Final[Path] = PROJECT_ROOT / "modelos" / "p84" / "emb_20261016_S.npz"
RUTA_PREREGISTRO: Final[Path] = PROJECT_ROOT / "docs" / "p85_preregistro.md"
RUTA_SCRIPT: Final[Path] = Path(__file__)
DIR_MODELOS: Final[Path] = PROJECT_ROOT / "modelos" / "p85"
RUTA_EMB_TWITTER: Final[Path] = DIR_MODELOS / "emb_20261016_S_twitter.npz"
RUTA_OOF: Final[Path] = PROJECT_ROOT / "features" / "p85_01_oof_emb.npz"
RUTA_JSON: Final[Path] = PROJECT_ROOT / "reports" / "p85_01_sgns_twitter.json"
RUTA_MD: Final[Path] = PROJECT_ROOT / "reports" / "toxic_twitter_sgns_corrected.md"
RUTA_CSV: Final[Path] = PROJECT_ROOT / "reports" / "toxic_twitter_sgns_corrected.csv"
RUTA_EXPERIMENTS: Final[Path] = PROJECT_ROOT / "experiments.csv"

# ===== SECCION 1: tokenizacion EMB-NORMAL-TW v1 (verbatim de p71_03_preparar_corpus.py) =====

RE_URL: Final[re.Pattern[str]] = re.compile(r"https?://\S+|www\.\S+")
RE_MENCION: Final[re.Pattern[str]] = re.compile(r"@\w+")
RE_NUMERO: Final[re.Pattern[str]] = re.compile(r"\b\d[\d.,]*\b")
RE_TOKEN: Final[re.Pattern[str]] = re.compile(r"[#@]?\w+(?:['\u2019]\w+)*", re.UNICODE)
RE_ESPACIOS: Final[re.Pattern[str]] = re.compile(r"\s+")


def normalizar_tw(texto: str) -> str:
    """Normalizacion EMB-NORMAL-TW v1, copiada verbatim de p71_03.

    Minusculas conservando tildes; URL, mencion y numero a marcador; hashtag
    conservado con su contenido; sin corrector ortografico ni colapso de
    alargamientos.

    Args:
        texto: texto crudo.

    Returns:
        Texto normalizado.
    """
    t = unicodedata.normalize("NFKC", texto).lower()
    t = RE_URL.sub(" url ", t)
    t = RE_MENCION.sub(" @usuario ", t)
    t = RE_NUMERO.sub(" numero ", t)
    return RE_ESPACIOS.sub(" ", t).strip()


# ===== SECCION 2: utilidades =====


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P85-01] {mensaje}", flush=True)


def sha256_de(ruta: Path) -> str:
    """Calcula el sha256 de un archivo.

    Args:
        ruta: ruta del archivo.

    Returns:
        Hash hexadecimal.
    """
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


def verificar_hash(ruta: Path, esperado: str, nombre: str) -> dict[str, Any]:
    """Verifica el sha256 de un archivo congelado.

    Args:
        ruta: ruta del archivo.
        esperado: hash esperado.
        nombre: nombre para el informe.

    Returns:
        Evidencia de la verificacion.

    Raises:
        SystemExit: si el hash no coincide.
    """
    obtenido = sha256_de(ruta)
    if obtenido != esperado:
        raise SystemExit(f"STOP: {nombre} sha256 {obtenido} != {esperado}")
    return {"nombre": nombre, "ruta": str(ruta.relative_to(PROJECT_ROOT)), "sha256": obtenido}


def escribir_json(ruta: Path, carga: dict[str, Any]) -> None:
    """Escribe un JSON con indentacion y sin escapes.

    Args:
        ruta: ruta de salida.
        carga: contenido.
    """
    ruta.write_text(json.dumps(carga, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def bytes_no_ascii(rutas: list[Path]) -> list[str]:
    """Lista los ficheros con algun byte > 127 (regla del proyecto).

    Args:
        rutas: ficheros a escanear.

    Returns:
        Nombres de los ficheros con bytes > 127.
    """
    return [ruta.name for ruta in rutas if any(b > 127 for b in ruta.read_bytes())]


# ===== SECCION 3: metricas =====


def lecturas(proba: np.ndarray, datos: Any) -> dict[str, Any]:
    """Calcula las lecturas de una matriz de probabilidades OOF.

    Args:
        proba: probabilidades OOF (n, 3).
        datos: datos de Gate 0.

    Returns:
        Diccionario con lecturas congelada y 0.5, globales, por pliegue y por etiqueta.
    """
    pred_f = (proba >= UMBRALES[None, :]).astype(np.int64)
    pred_05 = (proba >= 0.5).astype(np.int64)
    return {
        "f1_congelada": float(metrica_macro(datos.y, pred_f)),
        "f1_05": float(metrica_macro(datos.y, pred_05)),
        "f1_congelada_por_pliegue": [
            float(metrica_macro(datos.y[b], pred_f[b])) for _, b in datos.pliegues
        ],
        "f1_05_por_pliegue": [
            float(metrica_macro(datos.y[b], pred_05[b])) for _, b in datos.pliegues
        ],
        "f1_por_etiqueta_congelada": base.metricas_multietiqueta(datos.y, pred_f)["f1_por_etiqueta"],
        "pred_congelada": pred_f,
        "pred_05": pred_05,
    }


def net_de_matriz(matriz: dict[str, Any]) -> dict[str, Any]:
    """Extrae salvage, damage y net de una matriz de errores.

    Args:
        matriz: salida de matriz_errores.

    Returns:
        Diccionario con salvage, damage y net por etiqueta y global.
    """
    por_etiqueta = {
        etiqueta: {
            "salvage": valores["solo_referencia_error"],
            "damage": valores["solo_candidato_error"],
            "net": valores["solo_referencia_error"] - valores["solo_candidato_error"],
        }
        for etiqueta, valores in matriz["por_etiqueta"].items()
    }
    total = matriz["global"]
    return {
        "por_etiqueta": por_etiqueta,
        "global": {
            "salvage": total["solo_referencia_error"],
            "damage": total["solo_candidato_error"],
            "net": total["solo_referencia_error"] - total["solo_candidato_error"],
        },
    }


# ===== SECCION 4: corrida =====


def cargar_gate0_propio() -> Any:
    """Gate 0 propio: hashes relevantes, particiones e identidad V03.

    Replica la logica de `base.cargar_gate0` SIN la verificacion del binario
    fastText, que no participa en este experimento (ver NOTA_FASTTEXT).

    Returns:
        Datos de Gate 0 (base.DatosGate0).

    Raises:
        SystemExit: si cualquier verificacion congelada falla.
    """
    hashes = [
        verificar_hash(RUTA_GENERAL, HASH_GENERAL, "tokens_10M.txt"),
        verificar_hash(PROJECT_ROOT / "features" / "oof_v03_final.npz", HASH_OOF_V03, "oof_v03_final.npz"),
        verificar_hash(PROJECT_ROOT / "data" / "raw" / "toxic_train.csv", HASH_TRAIN, "toxic_train.csv"),
        verificar_hash(
            PROJECT_ROOT / "data" / "processed" / "grupos_near_duplicados.csv",
            HASH_GRUPOS,
            "grupos_near_duplicados.csv",
        ),
    ]
    train = base.read_train()
    if len(train) != base.FILAS_TRAIN:
        raise SystemExit(f"STOP: toxic_train.csv tiene {len(train)} filas")
    grupos = base.grupos_fusionados(train)
    indice_dev, indice_holdout = base.holdout_v2(train, grupos)
    if len(indice_dev) != base.FILAS_DEV or len(indice_holdout) != base.FILAS_HOLDOUT:
        raise SystemExit(
            f"STOP: particion inesperada dev={len(indice_dev)} holdout={len(indice_holdout)}"
        )
    des = train.iloc[indice_dev].reset_index(drop=True)
    grupos_dev = grupos[indice_dev]
    oof = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)
    y = oof["y_true"].astype(np.int64)
    p_v03 = oof["proba"].astype(np.float64)
    ids_v03 = oof["ids"].astype(np.int64)
    umbrales_v03 = oof["umbrales"].astype(np.float64)
    if not np.array_equal(des["id"].to_numpy(dtype=np.int64), ids_v03):
        raise SystemExit("STOP: los ids de desarrollo no coinciden con el OOF canonico")
    y_dev = des[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    if not np.array_equal(y_dev, y):
        raise SystemExit("STOP: las etiquetas de desarrollo no coinciden con el OOF canonico")
    if not np.allclose(umbrales_v03, UMBRALES, atol=0.0, rtol=0.0):
        raise SystemExit(f"STOP: umbrales congelados inesperados: {umbrales_v03.tolist()}")
    pred_v03 = (p_v03 >= umbrales_v03[None, :]).astype(np.int64)
    if not np.array_equal(pred_v03, oof["pred"].astype(np.int64)):
        raise SystemExit("STOP: pred del npz no reproduce proba >= umbrales congelados")
    pliegues = base.pliegues_corregidos(des, grupos_dev)
    tamanos = [len(np.asarray(b)) for _, b in pliegues]
    if tamanos != base.TAMANOS_PLIEGUES:
        raise SystemExit(f"STOP: tamanos de pliegue inesperados: {tamanos}")
    folde = np.zeros(len(des), dtype=np.int64)
    for numero, (_, validacion) in enumerate(pliegues, start=1):
        folde[np.asarray(validacion)] = numero
    if not np.array_equal(folde, oof["pliegues"].astype(np.int64)):
        raise SystemExit("STOP: el numero de pliegue no coincide con el npz canonico")
    contadores = {
        "dev_filas_leidas": len(des),
        "holdout_filas_usadas": 0,
        "test_filas_usadas": 0,
        "toxic_test_abierto": False,
        "nota": "el holdout (248) se aparta por indice y no se usa; toxic_test.csv nunca se abre",
    }
    evidencia = {
        "hashes": hashes,
        "fasttext": NOTA_FASTTEXT,
        "particiones": {
            "dev": len(des),
            "holdout": len(indice_holdout),
            "pliegues_tamanos": tamanos,
            "pliegues_sin_grupos_compartidos": True,
            "filas_cubiertas_una_vez": bool(
                (np.bincount(folde, minlength=6)[1:] == np.array(tamanos)).all()
            ),
        },
        "contadores": contadores,
        "oof_v03": {
            "corrected_cv_npz": float(oof["corrected_cv"][0]),
            "umbrales": umbrales_v03.tolist(),
            "f1_pliegues": [float(valor) for valor in np.asarray(oof["f1_pliegues"])],
            "f1_por_etiqueta": [float(valor) for valor in np.asarray(oof["f1_por_etiqueta"])],
        },
    }
    return base.DatosGate0(
        des=des,
        textos=des["text"].to_numpy(dtype=object),
        y=y,
        ids=ids_v03,
        p_v03=p_v03,
        pred_v03=pred_v03,
        umbrales_v03=umbrales_v03,
        grupos_dev=grupos_dev,
        pliegues=pliegues,
        folde=folde,
        evidencia=evidencia,
    )


def parsear_argumentos(argv: list[str] | None) -> argparse.Namespace:
    """Define y parsea los argumentos.

    Args:
        argv: argumentos (None usa sys.argv[1:]).

    Returns:
        Namespace con humo.
    """
    analizador = argparse.ArgumentParser(
        description="P85-01: SGNS corregido sobre corpus Twitter/X frente al control general."
    )
    analizador.add_argument(
        "--humo",
        action="store_true",
        help="validacion de glue: corpus diminuto, 2 epocas, sin artefactos finales",
    )
    return analizador.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Ejecuta la comparacion completa.

    Args:
        argv: argumentos de linea de comandos.

    Returns:
        Codigo de salida.
    """
    args = parsear_argumentos(argv)
    humo = bool(args.humo)
    t0 = time.perf_counter()

    hash_preregistro = sha256_de(RUTA_PREREGISTRO)
    hash_script = sha256_de(RUTA_SCRIPT)
    log(f"preregistro sha256 {hash_preregistro}")
    log(f"script sha256 {hash_script}")

    datos = cargar_gate0_propio()
    log(f"Gate 0 propio OK: dev {len(datos.y)} | holdout 0 | test 0 (sin verificacion del binario fastText)")

    hashes = [
        verificar_hash(RUTA_TWITTER, HASH_TWITTER, "corpus_twitter_tokens.txt"),
        verificar_hash(RUTA_GENERAL, HASH_GENERAL, "tokens_10M.txt"),
        verificar_hash(RUTA_EMB_GENERAL, HASH_EMB_GENERAL, "emb_20261016_S.npz (control general)"),
    ]
    log("hashes de corpus y control general OK")

    textos_norm = np.array([normalizar_tw(t) for t in datos.textos], dtype=object)

    device = base.sembrar_global(SEMILLA)
    cfg = replace(base.CONFIG_SGNS, epocas=2) if humo else base.CONFIG_SGNS
    log(f"device {device} | humo {humo}")

    # --- Brazo B: SGNS corregido sobre corpus Twitter/X (un entrenamiento) ---
    docs_tw = base.leer_tokens(RUTA_TWITTER, max_tokens=200_000 if humo else None)
    vocab_tw, p_tw, cnt_tw = base.construir_vocabulario(docs_tw, cfg.min_count, cfg.submuestreo)
    ids_tw = base.construir_ids_por_doc(docs_tw, vocab_tw)
    _, cdf_tw = base.tabla_negativos(vocab_tw, cnt_tw)
    log(f"corpus Twitter: {len(docs_tw)} documentos | vocab {len(vocab_tw)}")
    resultado_tw = base.entrenar_modelos([("S", "suma")], ids_tw, p_tw, cdf_tw, SEMILLA, cfg, device)
    res_tw = resultado_tw["S"]
    log(f"entrenamiento Twitter terminado: {res_tw.segundos} s | pasos {res_tw.pasos}")

    d_tw, cob_tw = base.pooling_medio(textos_norm, vocab_tw, res_tw.w_entrada)
    incidencias: list[str] = []
    down_tw = base.evaluar_downstream(
        d_tw, datos.y, datos.pliegues, datos.grupos_dev, SEMILLA, incidencias
    )
    log(f"downstream Twitter OK | cobertura {cob_tw['fraccion_cubierta']}")

    # --- Brazo A: control general (embedding S de P84-04, sin reentrenar) ---
    z = np.load(RUTA_EMB_GENERAL, allow_pickle=True)
    W_gen = np.asarray(z["W"], dtype=np.float32)
    palabras_gen = [str(p) for p in z["palabras"]]
    vocab_gen = {palabra: indice for indice, palabra in enumerate(palabras_gen)}
    d_gen, cob_gen = base.pooling_medio(textos_norm, vocab_gen, W_gen)
    down_gen = base.evaluar_downstream(
        d_gen, datos.y, datos.pliegues, datos.grupos_dev, SEMILLA, incidencias
    )
    log(f"downstream general OK | cobertura {cob_gen['fraccion_cubierta']}")

    # --- Lecturas y comparaciones ---
    v03 = {
        "f1_congelada": float(metrica_macro(datos.y, datos.pred_v03)),
        "f1_05": float(metrica_macro(datos.y, (datos.p_v03 >= 0.5).astype(np.int64))),
        "f1_congelada_por_pliegue": [
            float(metrica_macro(datos.y[b], datos.pred_v03[b])) for _, b in datos.pliegues
        ],
    }
    if abs(v03["f1_congelada"] - V03_CCV) > 1e-12 or abs(v03["f1_05"] - base.V03_F1_AT_05) > 1e-12:
        raise SystemExit("STOP: V03 no reproduce sus constantes")
    pliegues_canonicos = [float(v) for v in datos.evidencia["oof_v03"]["f1_pliegues"]]
    if max(abs(a - b) for a, b in zip(v03["f1_congelada_por_pliegue"], pliegues_canonicos)) > 1e-9:
        raise SystemExit("STOP: los pliegues de V03 no coinciden con el npz canonico")

    brazos: dict[str, Any] = {}
    for nombre, proba in (("twitter", down_tw["proba"]), ("general", down_gen["proba"])):
        emb = lecturas(proba, datos)
        fusion = lecturas((proba + datos.p_v03) / 2.0, datos)
        boot_fusion = bootstrap_diferencia_pareada(
            datos.y, fusion["pred_congelada"], datos.pred_v03,
            repeticiones=BOOTSTRAP_REPS, semilla=BOOTSTRAP_SEMILLA,
        )
        ensemble = base.bootstrap_ensemble_gain(
            datos.y, proba, datos.p_v03, BOOTSTRAP_REPS, BOOTSTRAP_SEMILLA
        )
        matriz_cand = base.matriz_errores(datos.y, emb["pred_congelada"], datos.pred_v03)
        matriz_fusion = base.matriz_errores(datos.y, fusion["pred_congelada"], datos.pred_v03)
        brazos[nombre] = {
            "embedding_only": {k: v for k, v in emb.items() if not k.startswith("pred")},
            "fusion": {k: v for k, v in fusion.items() if not k.startswith("pred")},
            "delta_fusion_congelada": round(fusion["f1_congelada"] - V03_CCV, 10),
            "delta_fusion_05": round(fusion["f1_05"] - base.V03_F1_AT_05, 10),
            "delta_embedding_only_congelada": round(emb["f1_congelada"] - V03_CCV, 10),
            "pliegues_positivos_fusion": int(
                sum(1 for a, b in zip(fusion["f1_congelada_por_pliegue"], v03["f1_congelada_por_pliegue"]) if a > b)
            ),
            "bootstrap_delta_fusion_congelada": boot_fusion,
            "ensemble_gain_05": ensemble,
            "matriz_candidato_vs_v03": {"matriz": matriz_cand, "net": net_de_matriz(matriz_cand)},
            "matriz_fusion_vs_v03": {"matriz": matriz_fusion, "net": net_de_matriz(matriz_fusion)},
            "_proba": proba,
            "_pred_fusion": fusion["pred_congelada"],
            "_pred_emb": emb["pred_congelada"],
        }

    tw = brazos["twitter"]
    gen = brazos["general"]
    condiciones = {
        "C1_delta_fusion_ge_0015": bool(tw["delta_fusion_congelada"] >= 0.0015),
        "C2_cuatro_de_cinco_pliegues": bool(tw["pliegues_positivos_fusion"] >= 4),
        "C3_net_fusion_positivo": bool(tw["matriz_fusion_vs_v03"]["net"]["global"]["net"] > 0),
        "C4_bootstrap_ic95_inferior_positivo": bool(
            tw["bootstrap_delta_fusion_congelada"]["ic95"][0] > 0
        ),
        "C5_dominio_twitter_supera_general": bool(
            tw["fusion"]["f1_congelada"] > gen["fusion"]["f1_congelada"]
        ),
    }
    if all(condiciones.values()):
        caso, decision = "A", "PROMISING"
    elif condiciones["C5_dominio_twitter_supera_general"]:
        caso, decision = "B", "CLOSED (la adaptacion de dominio no se transfiere al objetivo final)"
    else:
        caso, decision = "C", "CLOSED (no reabrir la familia de embeddings)"
    log(f"decision: Caso {caso} | {decision}")
    for clave, valor in condiciones.items():
        log(f"  {clave}: {valor}")

    segundos_totales = round(time.perf_counter() - t0, 1)

    informe: dict[str, Any] = {
        "experiment_id": "P85-01",
        "script": "experiments/p85_01_sgns_twitter_corregido.py",
        "fecha": "2026-10-08",
        "humo": humo,
        "preregistro_sha256": hash_preregistro,
        "script_sha256": hash_script,
        "gate0": datos.evidencia,
        "config": {
            "semilla": SEMILLA,
            "sgns": base.CONFIG_SGNS.__dict__ if not humo else cfg.__dict__,
            "umbrales_congelados": UMBRALES.tolist(),
            "bootstrap": {"repeticiones": BOOTSTRAP_REPS, "semilla": BOOTSTRAP_SEMILLA},
            "normalizacion_dev": "EMB-NORMAL-TW v1 (identica en los dos brazos)",
        },
        "hashes": hashes,
        "corpus_twitter": {
            "documentos": len(docs_tw),
            "vocabulario": len(vocab_tw),
            "hash": HASH_TWITTER,
        },
        "cobertura": {"twitter": cob_tw, "general": cob_gen},
        "entrenamiento_twitter": {
            "segundos": res_tw.segundos,
            "pasos": res_tw.pasos,
            "loss_por_epoca": res_tw.loss_por_epoca,
            "grad_media_por_epoca": res_tw.grad_media_por_epoca,
            "grad_max_por_epoca": res_tw.grad_max_por_epoca,
            "hash_pares_epoca1": res_tw.hash_pares_epoca1,
            "hash_negativos_lote1": res_tw.hash_negativos_lote1,
        },
        "v03": v03,
        "brazos": {
            nombre: {k: v for k, v in datos_brazo.items() if not k.startswith("_")}
            for nombre, datos_brazo in brazos.items()
        },
        "condiciones": condiciones,
        "caso": caso,
        "decision": decision,
        "incidencias": incidencias,
        "tiempo_total_segundos": segundos_totales,
        "versiones": base.versiones_entorno(),
        "compliance": {
            "dev": len(datos.y),
            "holdout_usado": 0,
            "test_usado": 0,
            "toxic_test_abierto": False,
            "submissions": 0,
            "leaderboard": 0,
            "push": 0,
            "rejillas": 0,
            "entrenamientos_nuevos": 0 if humo else 1,
        },
    }

    if humo:
        log("modo humo: sin artefactos finales")
        print(json.dumps({"caso": caso, "condiciones": condiciones, "segundos": segundos_totales}, ensure_ascii=False))
        return 0

    # --- Artefactos finales ---
    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    np.savez(
        RUTA_EMB_TWITTER,
        W=res_tw.w_entrada,
        palabras=np.array(sorted(vocab_tw, key=lambda w: vocab_tw[w]), dtype=object),
        dim=base.CONFIG_SGNS.dim,
        semilla=SEMILLA,
        brazo="S",
        corpus="corpus_twitter_tokens.txt",
        hash_corpus=HASH_TWITTER,
    )
    np.savez(
        RUTA_OOF,
        proba_twitter=brazos["twitter"]["_proba"],
        proba_general=brazos["general"]["_proba"],
        y=datos.y,
        ids=datos.ids,
        semilla=SEMILLA,
    )
    escribir_json(RUTA_JSON, informe)

    filas_csv: list[dict[str, Any]] = []
    for nombre in ("v03", "general", "twitter"):
        if nombre == "v03":
            filas_csv.append({
                "brazo": "v03", "lectura": "congelada", "ambito": "global", "pliegue": "",
                "f1": f"{V03_CCV:.10f}", "delta_vs_v03": "0.0000000000",
                "cobertura_fraccion": "", "filas_sin_token": "",
            })
            for numero, valor in enumerate(v03["f1_congelada_por_pliegue"], start=1):
                filas_csv.append({
                    "brazo": "v03", "lectura": "congelada", "ambito": "pliegue", "pliegue": str(numero),
                    "f1": f"{valor:.10f}", "delta_vs_v03": "",
                    "cobertura_fraccion": "", "filas_sin_token": "",
                })
            filas_csv.append({
                "brazo": "v03", "lectura": "umbral_0.5", "ambito": "global", "pliegue": "",
                "f1": f"{v03['f1_05']:.10f}", "delta_vs_v03": "",
                "cobertura_fraccion": "", "filas_sin_token": "",
            })
            continue
        b = brazos[nombre]
        cob = cob_tw if nombre == "twitter" else cob_gen
        for lectura, clave in (("embedding_only", "embedding_only"), ("fusion", "fusion")):
            datos_lectura = b[clave]
            filas_csv.append({
                "brazo": nombre, "lectura": f"{lectura}_congelada", "ambito": "global", "pliegue": "",
                "f1": f"{datos_lectura['f1_congelada']:.10f}",
                "delta_vs_v03": f"{datos_lectura['f1_congelada'] - V03_CCV:.10f}",
                "cobertura_fraccion": f"{cob['fraccion_cubierta']:.6f}",
                "filas_sin_token": str(cob["filas_sin_ningun_token"]),
            })
            for numero, valor in enumerate(datos_lectura["f1_congelada_por_pliegue"], start=1):
                filas_csv.append({
                    "brazo": nombre, "lectura": f"{lectura}_congelada", "ambito": "pliegue",
                    "pliegue": str(numero), "f1": f"{valor:.10f}",
                    "delta_vs_v03": f"{valor - v03['f1_congelada_por_pliegue'][numero - 1]:.10f}",
                    "cobertura_fraccion": "", "filas_sin_token": "",
                })
            filas_csv.append({
                "brazo": nombre, "lectura": f"{lectura}_umbral_0.5", "ambito": "global", "pliegue": "",
                "f1": f"{datos_lectura['f1_05']:.10f}",
                "delta_vs_v03": f"{datos_lectura['f1_05'] - v03['f1_05']:.10f}",
                "cobertura_fraccion": "", "filas_sin_token": "",
            })
    with RUTA_CSV.open("w", encoding="utf-8", newline="") as fh:
        escritor = csv.DictWriter(
            fh,
            fieldnames=["brazo", "lectura", "ambito", "pliegue", "f1", "delta_vs_v03",
                        "cobertura_fraccion", "filas_sin_token"],
            lineterminator="\n",
        )
        escritor.writeheader()
        escritor.writerows(filas_csv)

    filas_tabla = "\n".join(
        f"| {fila['brazo']} | {fila['lectura']} | {fila['ambito']} {fila['pliegue']} | "
        f"{fila['f1']} | {fila['delta_vs_v03']} |"
        for fila in filas_csv
    )
    net_tw = tw["matriz_fusion_vs_v03"]["net"]
    net_cand = tw["matriz_candidato_vs_v03"]["net"]
    texto_md = f"""# TOXIC - SGNS corregido sobre corpus Twitter/X (reapertura unica)

Fecha: 2026-10-08. Script: `experiments/p85_01_sgns_twitter_corregido.py`
(sha256 `{hash_script}`). Preregistro congelado: `docs/p85_preregistro.md`
(sha256 `{hash_preregistro}`). Sin entrenamiento de embeddings preentrenados, sin
test, sin holdout, sin leaderboard, sin submissions, sin push.

## 0. Resultado

```text
caso      {caso}
decision  {decision}
```

## 1. Las tres piezas, sin confundirlas

- **P78 (historico, con sus limitaciones):** control anidado limpio del embedding Word2Vec
  100d, pero la matriz procedia del entrenador `p71_04`, que promediaba el termino negativo
  por K (auditoria P79): no implementa SGNS. Senal frente al azar (+0.0249 fragil), sin
  transferencia util.
- **P84-04 (SGNS corregido sobre corpus general):** implementacion corregida (suma) y
  anclada; corpus wikipedia 10M; resultados publicados con pooling sobre texto CRUDO
  (cobertura dev 0.6585). Aqui se reutiliza su embedding y se RE-PUNTUA su downstream con
  la normalizacion documentada, sin reentrenar.
- **Nuevo (SGNS corregido sobre corpus Twitter/X):** un entrenamiento, semilla {SEMILLA},
  misma configuracion y mismo flujo; normalizacion del dev EMB-NORMAL-TW v1 identica en
  los dos brazos.

## 2. P0: hallazgos de la auditoria previa

- P78 uso `modelos/twitter_domain/w2v_twitter_100d.npz` (entrenada por `p71_04`,
  defectuosa); P84-04 usa la corregida (brazo S, ancla lograda).
- El pooling publicado de P84-04 uso texto crudo (cobertura 0.6585, 72 filas muertas):
  la discrepancia de normalizacion de P73/P74. Se corrige normalizando el dev con
  EMB-NORMAL-TW v1 en los DOS brazos; cobertura {cob_tw['fraccion_cubierta']:.4f} (Twitter)
  y {cob_gen['fraccion_cubierta']:.4f} (general).
- Corpus: Twitter `corpus_twitter_tokens.txt` sha256 `{HASH_TWITTER}` (autorizado,
  D-246, CC BY-SA 4.0, periodo cerrado); general `tokens_10M.txt` sha256 `{HASH_GENERAL}`;
  control general `emb_20261016_S.npz` sha256 `{HASH_EMB_GENERAL}` (reutilizado).
- El binario fastText no participa en este experimento (la implementacion SGNS es
  PyTorch); se registra que el binario en disco difiere del congelado de P84-04
  (worktree v67 ya retirado) y no se usa aqui.

## 3. Diseno congelado

dim 200; ventana 5; min_count 3; negativos 10; submuestreo 1e-4; 5 epocas; lote 256;
SGD lr 6.4 -> 1e-3; pooling medio; StandardScaler + LR C=2.0 balanced liblinear rs=42;
umbrales congelados de V03 (primaria) y anidados (secundaria); pliegues D-016;
bootstrap 10000/{BOOTSTRAP_SEMILLA}; semilla unica {SEMILLA} (limitacion declarada: no se
estima la varianza de reentrenamiento; no se rescata con mas semillas).

## 4. Resultados

Cobertura: Twitter {cob_tw['fraccion_cubierta']:.6f} (filas sin token {cob_tw['filas_sin_ningun_token']});
general {cob_gen['fraccion_cubierta']:.6f} (filas sin token {cob_gen['filas_sin_ningun_token']}).

| brazo | lectura | ambito | f1 | delta vs V03 |
|---|---|---|---|---|
{filas_tabla}

Deltas principales:

```text
twitter  embedding-only congelada  {tw['embedding_only']['f1_congelada']:.6f}  delta {tw['delta_embedding_only_congelada']:+.6f}
twitter  fusion congelada          {tw['fusion']['f1_congelada']:.6f}  delta {tw['delta_fusion_congelada']:+.6f}
twitter  fusion 0.5                {tw['fusion']['f1_05']:.6f}  delta {tw['delta_fusion_05']:+.6f}
general  embedding-only congelada  {gen['embedding_only']['f1_congelada']:.6f}  delta {gen['delta_embedding_only_congelada']:+.6f}
general  fusion congelada          {gen['fusion']['f1_congelada']:.6f}  delta {gen['delta_fusion_congelada']:+.6f}
pliegues positivos (fusion twitter vs V03): {tw['pliegues_positivos_fusion']}/5
bootstrap delta fusion twitter: IC95 {tw['bootstrap_delta_fusion_congelada']['ic95']}
ensemble_gain 0.5 (twitter): {tw['ensemble_gain_05']['ensemble_gain']:+.6f} IC95 {tw['ensemble_gain_05']['ic95']}
```

## 5. Complementariedad

Candidato (embedding-only congelada) frente a V03:

```text
global: salvage {net_cand['global']['salvage']} | damage {net_cand['global']['damage']} | net {net_cand['global']['net']}
HS {net_cand['por_etiqueta']['HS']['salvage']}/{net_cand['por_etiqueta']['HS']['damage']}/{net_cand['por_etiqueta']['HS']['net']} |
TR {net_cand['por_etiqueta']['TR']['salvage']}/{net_cand['por_etiqueta']['TR']['damage']}/{net_cand['por_etiqueta']['TR']['net']} |
AG {net_cand['por_etiqueta']['AG']['salvage']}/{net_cand['por_etiqueta']['AG']['damage']}/{net_cand['por_etiqueta']['AG']['net']}
```

Fusion frente a V03:

```text
global: salvage {net_tw['global']['salvage']} | damage {net_tw['global']['damage']} | net {net_tw['global']['net']}
HS {net_tw['por_etiqueta']['HS']['salvage']}/{net_tw['por_etiqueta']['HS']['damage']}/{net_tw['por_etiqueta']['HS']['net']} |
TR {net_tw['por_etiqueta']['TR']['salvage']}/{net_tw['por_etiqueta']['TR']['damage']}/{net_tw['por_etiqueta']['TR']['net']} |
AG {net_tw['por_etiqueta']['AG']['salvage']}/{net_tw['por_etiqueta']['AG']['damage']}/{net_tw['por_etiqueta']['AG']['net']}
```

## 6. Decision preregistrada

```text
C1 delta fusion >= +0.0015        {condiciones['C1_delta_fusion_ge_0015']}
C2 >= 4/5 pliegues positivos      {condiciones['C2_cuatro_de_cinco_pliegues']}
C3 net de la fusion > 0           {condiciones['C3_net_fusion_positivo']}
C4 bootstrap IC95 inferior > 0    {condiciones['C4_bootstrap_ic95_inferior_positivo']}
C5 dominio twitter > general      {condiciones['C5_dominio_twitter_supera_general']}
```

**Caso {caso}: {decision}.** No se declara ADOPTED; no se genera submission; no se abren
mas experimentos tras esta comparacion.

## 7. Compliance y verificacion

```text
dev {len(datos.y)} | holdout 0 | test 0 | leaderboard 0 | submissions 0 | push 0
entrenamientos nuevos: 1 (Twitter) | rejillas: 0
V03 FROZEN | {base.V03_CCV:.16f}
tiempo total: {segundos_totales} s (entrenamiento Twitter {res_tw.segundos} s)
incidencias: {len(incidencias)}
```

> V03 remains the best verified leaderboard result.
"""
    RUTA_MD.write_text(texto_md, encoding="utf-8")

    sucios = bytes_no_ascii([RUTA_MD, RUTA_CSV, RUTA_JSON, RUTA_SCRIPT])
    if sucios:
        raise SystemExit(f"STOP: bytes > 127 en {sucios}")

    # --- Fila en el registro (una sola vez) ---
    filas_reg = list(csv.DictReader(RUTA_EXPERIMENTS.read_text(encoding="utf-8-sig").splitlines()))
    columnas = list(filas_reg[0].keys())
    if not any(fila["experiment_id"] == "P85-01" for fila in filas_reg):
        filas_reg.append({
            "experiment_id": "P85-01",
            "competition": "KAGGLE_TOXIC_2026_PUJ",
            "question": "Al corregir el objetivo SGNS y mantener un corpus de dominio Twitter/X, aparece informacion distribuicional complementaria que no se observo en P71/P78 ni en P84-04?",
            "hypothesis": "SGNS corregido (suma) sobre corpus Twitter/X autorizado frente al control general con la misma normalizacion del texto de desarrollo",
            "preprocessing": "texto de desarrollo normalizado con EMB-NORMAL-TW v1 (identico en los dos brazos); corpus externos solo como texto no etiquetado",
            "feature_groups": "embedding denso (dim 200) por pooling medio",
            "representation": "SGNS desde cero (suma del termino negativo, implementacion anclada de P84-04)",
            "model": "embedding + StandardScaler + LogisticRegression C=2.0 balanced liblinear rs=42",
            "hyperparameters": "dim 200; ventana 5; min_count 3; negativos 10; submuestreo 1e-4; 5 epocas; lote 256; SGD lr 6.4->1e-3; semilla 20261016",
            "validation": "pliegues agrupados D-016; umbrales congelados V03 (primaria) y anidados (secundaria); bootstrap 10000/20260926",
            "f1": f"{tw['fusion']['f1_congelada']:.10f}",
            "macro_f1": "",
            "weighted_f1": "",
            "precision": "",
            "recall": "",
            "training_time": f"{res_tw.segundos} s",
            "dimension": "200",
            "result": f"twitter fusion congelada {tw['fusion']['f1_congelada']:.6f} (delta {tw['delta_fusion_congelada']:+.6f}); general fusion {gen['fusion']['f1_congelada']:.6f}; embedding-only twitter {tw['embedding_only']['f1_congelada']:.6f}",
            "interpretation": f"caso {caso}; C1..C5 = " + ", ".join(f"{k}={v}" for k, v in condiciones.items()),
            "decision": decision,
            "date": "2026-10-08",
            "script": "experiments/p85_01_sgns_twitter_corregido.py",
            "notes": "reapertura experimental unica autorizada; control general re-puntuado con la normalizacion documentada (el pooling publicado de P84-04 uso texto crudo); sin submissions; sin push",
        })
        with RUTA_EXPERIMENTS.open("w", encoding="utf-8-sig", newline="") as fh:
            escritor = csv.DictWriter(fh, fieldnames=columnas, lineterminator="\n")
            escritor.writeheader()
            escritor.writerows(filas_reg)
        log("fila P85-01 anadida a experiments.csv")
    else:
        log("fila P85-01 ya presente en experiments.csv")

    log(f"artefactos escritos | {segundos_totales} s")
    print(json.dumps({"caso": caso, "decision": decision, "condiciones": condiciones,
                      "delta_fusion_twitter": tw["delta_fusion_congelada"],
                      "delta_fusion_general": gen["delta_fusion_congelada"],
                      "segundos": segundos_totales}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
