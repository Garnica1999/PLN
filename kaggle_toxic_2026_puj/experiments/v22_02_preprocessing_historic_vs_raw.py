"""V22-02 (P1, P2, P3): RAW frente a HISTORICO, con el embedding ya entrenado.

Que hace y por que
------------------
P1 establece el control: el pipeline historico de V5.1, que ya aplica
`normalizar_emb_normal` al texto de la tarea, debe reproducir el valor publicado
(0.781766 para FUSION-B0-200). P2 introduce la variante sin normalizar. P3 compara ambas y
cierra la hipotesis de preprocessing.

**Que embedding se usa.** El de V5: FastText skipgram 200d sobre wikipedia_es 10M, el mismo
`fasttext-baseline` que produjo FUSION-B0-200. No se entrena nada. Los vectores de tipo se
toman de los SIFTS, que es lo que hizo V5.1, y no del `.vec`.

**Por que el diccionario es el mismo en los dos brazos.** El encargo prohibe cambiar dos
variables a la vez. Si cada brazo usara su propio diccionario, un cambio en cobertura
seria indistinguible de un cambio en el modelo. Se usa el mismo diccionario de 6.059 tipos
en los dos, que es ademas el que construye `vocabulario_toxic(read_train())` aplicando
`normalizar_emb_normal`. Ese detalle es el que hace el resultado interpretable: el
diccionario vive en el espacio normalizado, asi que desviarse de el solo puede perder
cobertura.

**Por que el control importa mas de lo que parece.** En V21, una reimplementacion razonable
del pipeline de V5 daba 0.7728 en vez de 0.7818 por tres detalles de construccion. Si el
control de este script no reproduce el valor publicado, las cifras de los dos brazos no
serian atribuibles a la variable que dicen medir, y el script se detiene.

Uso
---
    /home/carlos/venv/bin/python experiments/v22_02_preprocessing_historic_vs_raw.py
"""

from __future__ import annotations

import importlib.util
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

from src.data import read_train
from src.embedding_utils import normalizar_emb_normal, tokenizar
from src.splits import grupos_fusionados, pliegues_corregidos

EXPERIMENTO = "V22-02 (P1 control, P2 y P3: RAW frente al pipeline historico)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v22_02_preprocessing.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v22_02_preprocessing.md"
OOF_CANONICO = PROJECT_ROOT / "features" / "oof_v03_final.npz"
EMB_DIR = PROJECT_ROOT / "data" / "external" / "embeddings"
SIFTS = EMB_DIR / "fasttext-baseline_sifts.txt"
SIFTS_ENTRADA = SIFTS.with_suffix(".entrada.txt")
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"

ETIQUETAS = ("HS", "TR", "AG")
UMBRALES = np.array([0.44, 0.51, 0.46])
CORRECTED_CV_V03 = 0.7772114525165721
PUBLICADO_B0_200 = 0.781766
TOLERANCIA = 0.0015

ARMS = (
    ("HISTORICO", "normalizar_emb_normal, el pipeline de V5.1"),
    ("CRUDO", "sin normalizar: tokenizacion directa del texto del tweet"),
)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V22-02] {mensaje}", flush=True)


def cargar_maquinaria() -> Any:
    """Importa los modulos de V5 con la maquinaria de fusion y de vocabulario.

    Returns:
        Tupla con el modulo de fusion, el modulo de P1 y el diccionario de vectores.
    """
    specs = (
        ("p3_fusion", "embedding_03_fusion_complementariedad_v5.py"),
        ("p1_corpus", "embedding_01_corpus_wikipedia_v5.py"),
    )
    modulos = {}
    for nombre, fichero in specs:
        especificacion = importlib.util.spec_from_file_location(
            nombre, PROJECT_ROOT / "experiments" / fichero
        )
        if especificacion is None or especificacion.loader is None:
            raise SystemExit(f"PARADA OBLIGATORIA: no se puede cargar {fichero}")
        modulo = importlib.util.module_from_spec(especificacion)
        sys.modules[nombre] = modulo
        especificacion.loader.exec_module(modulo)
        modulos[nombre] = modulo
    tipos = SIFTS_ENTRADA.read_text(encoding="utf-8").split()
    vectores: dict[str, np.ndarray] = {}
    nulos = 0
    with SIFTS.open("r", encoding="utf-8") as manejador:
        for indice, linea in enumerate(manejador, start=1):
            partes = linea.rstrip().split()
            if len(partes) != 201:
                raise SystemExit(f"PARADA OBLIGATORIA: linea {indice} de SIFTS con {len(partes)-1} valores")
            vector = np.asarray(partes[1:], dtype=np.float32)
            if not np.any(vector):
                nulos += 1
                continue
            vectores[partes[0]] = vector
    log(
        f"   SIFTS: {len(tipos)} consultados | {len(vectores)} con vector | {nulos} nulos | "
        f"el diccionario es identico en los dos brazos"
    )
    return modulos["p3_fusion"], modulos["p1_corpus"], vectores


def macro_f1(verdad: np.ndarray, predicho: np.ndarray) -> float:
    """Macro-F1 de las tres clases positivas.

    Args:
        verdad: matriz verdadera binaria.
        predicho: matriz predicha binaria.

    Returns:
        La media de las tres F1.
    """
    valores = []
    for j in range(verdad.shape[1]):
        comunes = int(np.count_nonzero(verdad[:, j] & predicho[:, j]))
        denominador = int(np.count_nonzero(verdad[:, j])) + int(np.count_nonzero(predicho[:, j]))
        valores.append(2 * comunes / denominador if denominador > 0 else 0.0)
    return float(np.mean(valores))


def main() -> None:
    """Ejecuta el control, la variante y la comparacion."""
    inicio = time.time()
    modulo, _modulo_p1, vectores = cargar_maquinaria()
    log("V22-02: P1 control, P2 y P3. Ningun modelo se entrena")

    log("1/5 desarrollo, pliegues canonicos y alineacion por identificador")
    train = read_train()
    particion = pd.read_csv(PARTICION)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    grupos_full = grupos_fusionados(train)
    mapa_grupo = dict(zip(train["id"].astype(int), grupos_full.tolist()))
    grupos_dev = np.array([mapa_grupo[i] for i in desarrollo["id"].astype(int)])
    pliegues = pliegues_corregidos(desarrollo, grupos_dev)
    y = desarrollo[list(ETIQUETAS)].to_numpy().astype(int)
    textos_crudos = desarrollo["text"]

    oof = np.load(OOF_CANONICO, allow_pickle=True)
    ids_oof = np.asarray(oof["ids"]).astype(np.int64)
    y_oof = np.asarray(oof["y_true"], dtype=int)
    mapa_posicion = {v: i for i, v in enumerate(desarrollo["id"].astype(int).to_numpy())}
    posiciones = np.array([mapa_posicion[i] for i in ids_oof])
    if not np.array_equal(y[posiciones], y_oof):
        raise SystemExit("PARADA OBLIGATORIA: las etiquetas no coinciden tras alinear por id")
    pred_v03 = np.asarray(oof["pred"], dtype=int)
    log(f"   desarrollo {len(desarrollo)} | alineacion verificada | pred_v03 del OOF canonico")

    log("2/5 los dos brazos de preprocessing, con el MISMO embedding y el MISMO clasificador")
    vectores_documento: dict[str, np.ndarray] = {}
    cobertura: dict[str, dict[str, Any]] = {}
    for nombre, _ in ARMS:
        if nombre == "HISTORICO":
            lista = [normalizar_emb_normal(t) for t in textos_crudos]
        else:
            lista = list(textos_crudos)
        documento = modulo.vectores_documento(lista, vectores)
        vectores_documento[nombre] = documento
        todos = [token for t in lista for token in tokenizar(t)]
        con_vector = sum(1 for token in todos if token in vectores)
        tipos = {token for token in todos}
        cobertura[nombre] = {
            "tokens": len(todos),
            "tipos": len(tipos),
            "cobertura_token": con_vector / max(len(todos), 1),
            "cobertura_tipo": sum(1 for t in tipos if t in vectores) / max(len(tipos), 1),
            "longitud_media": float(np.mean([len(tokenizar(t)) for t in lista])),
            "filas_con_documento_nulo": int(np.count_nonzero(np.linalg.norm(documento, axis=1) == 0)),
        }
        c = cobertura[nombre]
        log(
            f"   {nombre:<11} tokens {c['tokens']:>6} | tipos {c['tipos']:>5} | cob.tipo "
            f"{c['cobertura_tipo']:.4f} | cob.token {c['cobertura_token']:.4f} | filas nulas "
            f"{c['filas_con_documento_nulo']}"
        )

    log("3/5 evaluacion: embedding solo y fusion con V03, en los dos brazos")
    resultados: dict[str, Any] = {}
    for nombre, _ in ARMS:
        documento = vectores_documento[nombre]
        fusion = modulo.evaluar_variante(
            f"V22-{nombre}", textos_crudos, y, pliegues, documento,
            usar_tfidf=True, modo_embedding="l2",
        )
        proba = fusion["proba"][posiciones]
        pred = (proba >= UMBRALES[None, :]).astype(int)
        solo = modulo.evaluar_variante(
            f"V22-{nombre}-SOLO", textos_crudos, y, pliegues, documento,
            usar_tfidf=False, modo_embedding="l2",
        )
        pred_solo = (solo["proba"][posiciones] >= UMBRALES[None, :]).astype(int)
        valor = macro_f1(y_oof, pred)
        resultados[nombre] = {
            "corrected_cv": valor,
            "delta_vs_v03": valor - CORRECTED_CV_V03,
            "emb_solo": macro_f1(y_oof, pred_solo),
            "proba": proba,
            "pred": pred,
        }
        log(
            f"   {nombre:<11} fusion {valor:.6f} (dV03 {valor - CORRECTED_CV_V03:+.6f}) | "
            f"embedding solo {resultados[nombre]['emb_solo']:.6f}"
        )

    log("4/5 el control de cordura de P1")
    control = resultados["HISTORICO"]["corrected_cv"]
    desviacion = abs(control - PUBLICADO_B0_200)
    log(
        f"   HISTORICO calculado {control:.6f} | publicado {PUBLICADO_B0_200:.6f} | desviacion "
        f"{desviacion:.6f} | dentro de tolerancia: {desviacion <= TOLERANCIA}"
    )
    if desviacion > TOLERANCIA:
        log("   PARADA: el pipeline no reproduce el valor publicado. Nada mas es atribuible")
        REPORTE_JSON.write_text(
            json.dumps(
                {
                    "experimento": EXPERIMENTO,
                    "estado": "PARADA_POR_CONTROL_DE_CORDURA",
                    "historiano": control,
                    "publicado": PUBLICADO_B0_200,
                    "desviacion": desviacion,
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        return

    log("5/5 perfil de cambios y veredicto de P3")
    filas = []
    for nombre, _ in ARMS:
        pred = resultados[nombre]["pred"]
        cambio = pred != pred_v03
        rescates = int(np.count_nonzero(cambio & (pred_v03 != y_oof) & (pred == y_oof)))
        danos = int(np.count_nonzero(cambio & (pred_v03 == y_oof) & (pred != y_oof)))
        filas.append(
            {
                "brazo": nombre,
                "corrected_cv": resultados[nombre]["corrected_cv"],
                "delta_vs_v03": resultados[nombre]["delta_vs_v03"],
                "emb_solo": resultados[nombre]["emb_solo"],
                "celdas": int(np.count_nonzero(cambio)),
                "rescates": rescates,
                "danios": danos,
                "neto": rescates - danos,
            }
        )
        log(
            f"   {nombre:<11} dF1 {filas[-1]['delta_vs_v03']:+.6f} | celdas {filas[-1]['celdas']:>4} "
            f"| rescates {rescates:>4} | danos {danos:>4} | NETO {filas[-1]['neto']:>+4}"
        )
    hist, crudo = filas[0], filas[1]
    delta_conservar = hist["corrected_cv"] - crudo["corrected_cv"]
    log(f"   HISTORICO - CRUDO = {delta_conservar:+.6f}")

    # El sentido de la comparacion importa y la primera version lo invirtio. Un valor
    # positivo de `delta_conservar` significa que el pipeline historico **supera** a la
    # variante sin normalizar, es decir, que la normalizacion ayudo. Eso **refuta** la
    # hipotesis de que hubiera un mismatch que corregir, porque V5.1 ya aplicaba
    # `normalizar_emb_normal` al texto de la tarea. Un valor negativo seria el caso en el
    # que el pipeline historico dejaba informacion en la mesa, y ese si seria una
    # oportunidad para V22.
    if delta_conservar > 0:
        veredicto = "HIPOTESIS_DE_PREPROCESSING_REFUTADA"
        razon = (
            "El pipeline historico ya era el correcto. La Hipotesis 1 de V22 afirmaba que el "
            "embedding se usaba sobre un texto de tarea incompatible con el del corpus, pero "
            "V5.1 ya aplicaba normalizar_emb_normal al texto de la tarea, y apartarse de esa "
            "normalizacion empeora la metrica en vez de mejorarla. No habia un mismatch que "
            "corregir: habia una decision de diseno ya tomada correctamente"
        )
    else:
        veredicto = "EL_PIPELINE_HISTORICO_DEJABA_PERDIDA_INFORMACION"
        razon = (
            "Apartarse de la normalizacion historica mejora, lo que si seria una oportunidad "
            "de V22: el proyecto estaba consultando el embedding sobre un espacio que no era "
            "el suyo"
        )
    log(f"   VEREDICTO P3: {veredicto} (HISTORICO - CRUDO = {delta_conservar:+.6f})")

    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding": "FastText skipgram 200d sobre wikipedia_es 10M (fasttext-baseline de V5)",
        "sin_entrenar": True,
        "diccionario": "SIFTS del binario, identico en los dos brazos",
        "control_de_cordura": {
            "historiano": control,
            "publicado": PUBLICADO_B0_200,
            "desviacion": desviacion,
            "dentro_de_tolerancia": desviacion <= TOLERANCIA,
        },
        "cobertura": cobertura,
        "resultados": filas,
        "comparacion": {
            "metricas": "HISTORICO - CRUDO",
            "valor": delta_conservar,
            "veredicto": veredicto,
            "razon": razon,
        },
        "gastos": {"modelos_entrenados": 0, "submissions": 0},
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe_md(informe, inicio)
    log("   informes escritos en reports/v22_02_preprocessing.{json,md}")
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(
        f"VEREDICTO: {veredicto} | HISTORICO - CRUDO = {delta_conservar:+.6f} | "
        f"control OK {desviacion <= TOLERANCIA} | submissions 0"
    )


def _informe_md(informe: dict[str, Any], inicio: float) -> None:
    """Escribe el informe legible de la comparacion.

    Args:
        informe: informe con los datos.
        inicio: instante de arranque.
    """
    filas = informe["resultados"]
    comp = informe["comparacion"]
    control = informe["control_de_cordura"]
    lineas = [
        "# V22-02 (P1, P2, P3): RAW frente al pipeline historico",
        "",
        f"Generado por `experiments/v22_02_preprocessing_historic_vs_raw.py` ({EXPERIMENTO}).",
        "",
        "**Ningun modelo entrenado. Ninguna submission.**",
        "",
        "## 1. Control de cordura de P1",
        "",
        (f"El pipeline historico da **{control['historiano']:.6f}** frente a los "
        f"**{control['publicado']:.6f}** publicados (desviacion "
        f"{control['desviacion']:.6f}), dentro de tolerancia: **{control['dentro_de_tolerancia']}**."),
        "",
        "## 2. Cobertura de cada brazo, con el MISMO diccionario",
        "",
        "| brazo | tokens | tipos | cob. tipo | cob. token | long. media | filas nulas |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for nombre in ("HISTORICO", "CRUDO"):
        c = informe["cobertura"][nombre]
        lineas.append(
            f"| {nombre} | {c['tokens']} | {c['tipos']} | {c['cobertura_tipo']:.4f} | "
            f"{c['cobertura_token']:.4f} | {c['longitud_media']:.2f} | "
            f"{c['filas_con_documento_nulo']} |"
        )
    lineas += [
        "",
        "## 3. La comparacion",
        "",
        "| brazo | corrected_cv | delta V03 | emb solo | celdas | rescates | danos | neto |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for f in filas:
        lineas.append(
            f"| {f['brazo']} | {f['corrected_cv']:.6f} | {f['delta_vs_v03']:+.6f} | "
            f"{f['emb_solo']:.6f} | {f['celdas']} | {f['rescates']} | {f['danios']} | "
            f"**{f['neto']:+d}** |"
        )
    lineas += [
        "",
        f"**HISTORICO - CRUDO = {comp['valor']:+.6f}**",
        "",
        f"## 4. Veredicto de P3: {comp['veredicto']}",
        "",
        comp["razon"],
        "",
        f"Completado en {round(time.time() - inicio, 1)} s.",
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
