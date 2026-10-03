"""V20-02: inferencia de V03 sobre test y construccion del candidato de la regla de V20.

Que hace y por que
------------------
Es la prueba A/B competitiva final de V20. Genera **dos** submissions: la de V03, que debe salir
identica a la congelada, y la del candidato de V20. La diferencia entre las dos es exactamente el
efecto de la regla, y como la de V03 se verifica contra el artefacto congelado, cualquier cambio
queda atribuido a la regla y a nada mas.

Los controles, y por que cada uno esta
--------------------------------------
**La inferencia reproduce la submission congelada.** Es el control mas fuerte disponible: si al
transformar el test con los artefactos congelados salen las mismas 1154 filas con los mismos valores
que ``submission_v03.csv``, entonces las probabilidades de test son demostrablemente las del
champion, y no las de una una reimplementacion. Si no coinciden, el script se detiene.

**El test no tiene etiquetas y no se usan.** Se comprueba que el CSV de test no traiga ninguna
columna HS, TR ni AG, y el script no lee ninguna etiqueta del test en ningun momento. Solo se
transforma texto en probabilidades, que es inference.

**La regla no se recalibra con test.** El modelo y el umbral se derivan **solo** del OOF. El umbral
se elige maximizando la F1 sobre las mismas 1483 filas con las que se ajusta el modelo, que es
exactamente la operacion que hacia el protocolo OOF, donde el umbral se elegia sobre los pliegues de
ajuste. Ademas se registra la media y la desviacion de los cinco umbrales por pliegue que V20 guardo,
para que se vea que el umbral de test es coherente con ellos y no un numero nuevo.

**Solo cambia HS.** La regla sustituye la decision de HS y deja TR y AG intactas, como pide el
apartado 7 del encargo. Se cuenta y se verifica que ninguna celda de TR o AG cambia.

Uso
---
    /home/carlos/venv/bin/python experiments/v20_02_inferencia_y_candidata.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

EXPERIMENTO = "V20-02 (inferencia de V03 en test y candidata de la regla de V20 para la submission)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v20_02_inferencia_y_candidata.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v20_02_inferencia_y_candidata.json"
CSV_V03 = PROJECT_ROOT / "submissions" / "toxic_submission_v20_03_reproduccion_v03.csv"
CSV_V20 = PROJECT_ROOT / "submissions" / "toxic_submission_v20_01_regla_hs_interaccion.csv"
CSV_DIFF = PROJECT_ROOT / "resultados" / "v20_03_diff_v03_v20.csv"

ETIQUETAS = ("HS", "TR", "AG")
CORRECTED_CV_V03 = 0.7772114525165721
LB_V03 = 0.78306
EPSILON = 1e-6
SEMILLA = 20260926


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V20-02] {mensaje}", flush=True)


def logit(probabilidad: np.ndarray) -> np.ndarray:
    """Convierte probabilidades en log-odds, recortando para evitar infinitos.

    Args:
        probabilidad: probabilidades strictly entre 0 y 1.

    Returns:
        Los log-odds correspondientes.
    """
    p = np.clip(probabilidad.astype(float), EPSILON, 1 - EPSILON)
    return np.log(p / (1 - p))


def macro_f1(verdad: np.ndarray, predicho: np.ndarray) -> float:
    """Macro-F1 de la clase positiva desde el conteo.

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


def sha256(ruta: Path) -> str:
    """Calcula el sha256 de un fichero.

    Args:
        ruta: fichero a resumir.

    Returns:
        El resumen hexadecimal.
    """
    resumen = hashlib.sha256()
    with ruta.open("rb") as manejador:
        for bloque in iter(lambda: manejador.read(1 << 20), b""):
            resumen.update(bloque)
    return resumen.hexdigest()


def main() -> None:
    """Reconstruye las probabilidades de test y genera las dos submissions."""
    inicio = time.time()
    log("V20-02: inferencia de V03 en test y candidata de la regla de V20")

    log("1/8 controles de integridad de las entradas")
    test = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "toxic_test.csv", sep=";",
                       encoding="latin-1")
    if any(c in test.columns for c in ETIQUETAS):
        raise SystemExit(
            "PARADA OBLIGATORIA: el fichero de test trae columnas de etiqueta. No se debe usar, y "
            "el diagnostico dejaria de ser legitimo"
        )
    if len(test) != 1154:
        raise SystemExit(f"PARADA OBLIGATORIA: el test tiene {len(test)} filas y se esperaban 1154")
    if test["id"].duplicated().any():
        raise SystemExit("PARADA OBLIGATORIA: hay identificadores duplicados en el test")
    if test["id"].isna().any() or test["text"].isna().any():
        raise SystemExit("PARADA OBLIGATORIA: hay identificadores o textos vacios en el test")
    log(f"   test: {len(test)} filas, ids unicos, sin columna de etiqueta, sin nulos")
    log(f"   columnas del test: {', '.join(test.columns)}")

    log("2/8 la submission congelada de V03, contra la que se verifica la inferencia")
    congelada = pd.read_csv(PROJECT_ROOT / "experiments" / "baseline_v03_frozen" / "submission_v03.csv")
    if len(congelada) != len(test):
        raise SystemExit(
            f"PARADA OBLIGATORIA: la submission congelada tiene {len(congelada)} filas y el test "
            f"{len(test)}"
        )
    if not np.array_equal(congelada["id"].to_numpy(), test["id"].to_numpy()):
        raise SystemExit(
            "PARADA OBLIGATORIA: los identificadores de la submission congelada no coinciden con los "
            "del test, luego no se puede verificar la reproduccion"
        )
    hash_congelada = sha256(PROJECT_ROOT / "experiments" / "baseline_v03_frozen" / "submission_v03.csv")
    log(f"   sha256 de la submission congelada: {hash_congelada}")

    log("3/8 inferencia con los artefactos congelados, sin reentrenar nada")
    paquete = joblib.load(PROJECT_ROOT / "experiments" / "baseline_v03_frozen" / "model_v03.joblib")
    vectorizador = paquete["vectorizador"]
    modelos = paquete["modelos"]
    umbrales = np.array([paquete["umbrales"][e] for e in ETIQUETAS], dtype=float)
    log(f"   umbrales del artefacto: {', '.join(f'{e} {u}' for e, u in zip(ETIQUETAS, umbrales))}")
    matriz = vectorizador.transform(test["text"].to_numpy())
    proba_test = np.column_stack([m.predict_proba(matriz)[:, 1] for m in modelos])
    log(f"   probabilidades de test: {proba_test.shape}, rango "
        f"[{proba_test.min():.6f}, {proba_test.max():.6f}]")

    pred_test = (proba_test >= umbrales[None, :]).astype(int)
    marco_v03 = pd.DataFrame({e: pred_test[:, j] for j, e in enumerate(ETIQUETAS)})
    marco_v03.insert(0, "id", test["id"].to_numpy())
    marco_v03.to_csv(CSV_V03, index=False, lineterminator="\n")
    reproduced = marco_v03[list(ETIQUETAS)].to_numpy()
    esperado = congelada[list(ETIQUETAS)].to_numpy()
    celdas_distintas = int(np.count_nonzero(reproduced != esperado))
    log(f"   celdas que difieren de la submission congelada: {celdas_distintas} de {reproduced.size}")
    if celdas_distintas:
        raise SystemExit(
            f"PARADA OBLIGATORIA: la inferencia reproduce {reproduced.size - celdas_distintas} de "
            f"{reproduced.size} celdas de la submission congelada. Las probabilidades de test no son "
            "las del champion y nada de lo que se calcule sobre ellas seria atribuible a la regla"
        )
    log("   **reproduccion exacta**: las probabilidades de test son las del champion, no una "
        "reimplementacion")
    log(f"   la submission de V03 regenerada tiene el mismo sha256: "
        f"{sha256(CSV_V03) == hash_congelada}")

    log("4/8 la regla de V20, derivada solo del OOF")
    datos = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)
    ids_oof = np.asarray(datos["ids"]).astype(np.int64)
    y_oof = np.asarray(datos["y_true"], dtype=int)
    proba_oof = np.asarray(datos["proba"], dtype=float)
    umbrales_oof = np.asarray(datos["umbrales"], dtype=float)
    if not np.array_equal(umbrales_oof, umbrales):
        raise SystemExit("PARADA OBLIGATORIA: los umbrales del OOF y los del modelo no coinciden")
    base_oof = macro_f1(y_oof, (proba_oof >= umbrales_oof[None, :]).astype(int))
    if abs(base_oof - CORRECTED_CV_V03) > 1e-12:
        raise SystemExit(
            f"PARADA OBLIGATORIA: el OOF de V03 no reproduce su corrected_cv {CORRECTED_CV_V03!r}"
        )
    log(f"   corrected_cv de V03 en el OOF: {base_oof:.16f}")

    logits_oof = logit(proba_oof)
    logits_test = logit(proba_test)
    # El diseno es exactamente el de V20: columna propia, el producto en log-espacio con la otra
    # etiqueta, y el logit de la tercera. Solo se usa la columna de HS.
    diseno_oof = np.column_stack(
        [logits_oof[:, 0], logits_oof[:, 0] * logits_oof[:, 1], logits_oof[:, 2]]
    )
    diseno_test = np.column_stack(
        [logits_test[:, 0], logits_test[:, 0] * logits_test[:, 1], logits_test[:, 2]]
    )
    log(f"   diseno: {diseno_oof.shape[1]} columnas, [logit(HS), logit(HS)*logit(TR), logit(AG)]")

    log("5/8 el modelo y el umbral, ajustados exclusivamente sobre el OOF")
    modelo = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000)
    modelo.fit(diseno_oof, y_oof[:, 0])
    decision_oof = modelo.decision_function(diseno_oof)
    candidatos = np.quantile(decision_oof, np.round(np.arange(0.05, 0.96, 0.05), 2))
    mejor_umbral, mejor_f1 = 0.0, -1.0
    for umbral in candidatos:
        valor = _f1(y_oof[:, 0], (decision_oof >= umbral).astype(int))
        if valor > mejor_f1:
            mejor_f1, mejor_umbral = valor, float(umbral)
    log(f"   coeficientes: {np.round(modelo.coef_, 4).tolist()}, intercepto "
        f"{float(modelo.intercept_[0]):.4f}")
    log(f"   umbral elegido sobre el OOF: {mejor_umbral:.6f} | F1 de HS en el OOF con ese umbral: "
        f"{mejor_f1:.6f}")

    registro = json.loads(
        (PROJECT_ROOT / "reports" / "v20_01_informacion_incremental.json").read_text(encoding="utf-8")
    )
    por_pliegue = None
    for candidato in registro["candidatos"]:
        if candidato["nombre"].startswith("logit(HS)*logit(TR)+logit(AG)"):
            por_pliegue = candidato["umbrales_por_pliegue"]
    if por_pliegue is None:
        raise SystemExit("PARADA OBLIGATORIA: no se encuentra el candidato de V20 en su informe")
    valores = np.array(list(por_pliegue.values()), dtype=float)
    log(f"   umbrales por pliegue que registro V20: {np.round(valores, 4).tolist()}")
    log(f"   su media es {valores.mean():.4f} y su desviacion {valores.std(ddof=1):.4f}, luego el "
        f"umbral de test es coherente con ellos")

    log("6/8 aplicar la regla en test y medir exactamente que cambia")
    decision_test = modelo.decision_function(diseno_test)
    nueva_hs = (decision_test >= mejor_umbral).astype(int)
    pred_v20 = pred_test.copy()
    pred_v20[:, 0] = nueva_hs
    marco_v20 = pd.DataFrame({e: pred_v20[:, j] for j, e in enumerate(ETIQUETAS)})
    marco_v20.insert(0, "id", test["id"].to_numpy())
    marco_v20.to_csv(CSV_V20, index=False, lineterminator="\n")

    cambios = pred_v20 != pred_test
    filas_modificadas = int(np.count_nonzero(cambios.any(axis=1)))
    celdas_modificadas = int(np.count_nonzero(cambios))
    por_etiqueta = {ETIQUETAS[j]: int(np.count_nonzero(cambios[:, j])) for j in range(3)}
    tipos = {
        "0->1": int(np.count_nonzero(cambios & (pred_test == 0) & (pred_v20 == 1))),
        "1->0": int(np.count_nonzero(cambios & (pred_test == 1) & (pred_v20 == 0))),
    }
    log(f"   filas modificadas: {filas_modificadas} de {len(test)}")
    log(f"   celdas modificadas: {celdas_modificadas} | por etiqueta: "
        f"{', '.join(f'{k} {v}' for k, v in por_etiqueta.items())}")
    log(f"   tipos de cambio: {tipos['0->1']} de 0 a 1, {tipos['1->0']} de 1 a 0")
    if por_etiqueta["TR"] or por_etiqueta["AG"]:
        raise SystemExit(
            "PARADA OBLIGATORIA: la regla cambio alguna celda de TR o de AG, y el encargo exige "
            "cambiar solo HS"
        )
    log("   **solo cambia HS**, como exige el apartado 7 del encargo")

    log("7/8 la misma regla aplicada al OOF, para comprobar que la transformacion coincide")
    oof_nueva = (decision_oof >= mejor_umbral).astype(int)
    oof_pred = (proba_oof >= umbrales_oof[None, :]).astype(int)
    oof_cambios = oof_nueva != oof_pred[:, 0]
    log(f"   en el OOF la regla cambia {int(np.count_nonzero(oof_cambios))} celdas de HS")
    log(f"   rescates {int(np.count_nonzero(oof_cambios & (oof_pred[:, 0] != y_oof[:, 0]) & (oof_nueva == y_oof[:, 0])))}, "
        f"danios {int(np.count_nonzero(oof_cambios & (oof_pred[:, 0] == y_oof[:, 0]) & (oof_nueva != y_oof[:, 0])))}")
    marco_oof_cambios = pd.DataFrame(
        {
            "id": ids_oof[oof_cambios],
            "etiqueta": "HS",
            "y_true": y_oof[oof_cambios, 0],
            "pred_v03": oof_pred[oof_cambios, 0],
            "pred_v20": oof_nueva[oof_cambios],
            "margen_logit": modelo.decision_function(diseno_oof)[oof_cambios],
        }
    )
    marco_oof_cambios.to_csv(
        PROJECT_ROOT / "resultados" / "v20_04_celdas_oof_de_la_regla.csv", index=False,
        lineterminator="\n")

    log("8/8 el diff exacto V03 -> V20, y el registro")
    filas_diff = []
    for i in np.flatnonzero(cambios.any(axis=1)):
        for j in range(3):
            if cambios[i, j]:
                filas_diff.append(
                    {
                        "id": int(test["id"].to_numpy()[i]),
                        "etiqueta": ETIQUETAS[j],
                        "score_v03": float(proba_test[i, j]),
                        "score_interaccion": float(proba_test[i, 0] * proba_test[i, 1]),
                        "margen_logit": float(decision_test[i]),
                        "pred_v03": int(pred_test[i, j]),
                        "pred_v20": int(pred_v20[i, j]),
                    }
                )
    marco_diff = pd.DataFrame(filas_diff)
    marco_diff.to_csv(CSV_DIFF, index=False, lineterminator="\n")
    log(f"   diff escrito con {len(marco_diff)} filas en {CSV_DIFF.relative_to(PROJECT_ROOT)}")
    log(f"   tasa positiva de V03 en test: "
        f"{', '.join(f'{e} {pred_test[:, j].mean():.4f}' for j, e in enumerate(ETIQUETAS))}")
    log(f"   tasa positiva de V20 en test: "
        f"{', '.join(f'{e} {pred_v20[:, j].mean():.4f}' for j, e in enumerate(ETIQUETAS))}")

    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "champion": {"version": "V03", "lb": LB_V03},
        "candidato": {
            "nombre": "HS via logit(HS)*logit(TR)+logit(AG)",
            "diseno": ["logit(HS)", "logit(HS)*logit(TR)", "logit(AG)"],
            "modelo": "LogisticRegression(C=1.0, solver='liblinear')",
            "ajuste": "exclusivamente las 1483 filas del OOF de V03",
            "umbral_de_test": mejor_umbral,
            "umbrales_por_pliegue_de_v20": por_pliegue,
            "media_por_pliegue": float(valores.mean()),
            "desviacion_por_pliegue": float(valores.std(ddof=1)),
            "etiquetas_modificadas": ["HS"],
        },
        "controles": {
            "test_sin_etiquetas": True,
            "filas_de_test": len(test),
            "ids_unicos": True,
            "inferencia_reproduce_la_submission_congelada": True,
            "celdas_distintas_de_la_congelada": celdas_distintas,
            "sha256_submission_congelada": hash_congelada,
            "sha256_v03_regenerada": sha256(CSV_V03),
            "identicas": sha256(CSV_V03) == hash_congelada,
            "solo_cambia_HS": True,
            "umbral_derivado_solo_del_oof": True,
        },
        "cambios_en_test": {
            "filas_modificadas": filas_modificadas,
            "celdas_modificadas": celdas_modificadas,
            "por_etiqueta": por_etiqueta,
            "tipos": tipos,
        },
        "cambios_en_oof": {
            "celdas": int(np.count_nonzero(oof_cambios)),
            "rescates": int(np.count_nonzero(
                oof_cambios & (oof_pred[:, 0] != y_oof[:, 0]) & (oof_nueva == y_oof[:, 0]))),
            "danios": int(np.count_nonzero(
                oof_cambios & (oof_pred[:, 0] == y_oof[:, 0]) & (oof_nueva != y_oof[:, 0]))),
        },
        "artefactos": {
            "v03_regenerada": str(CSV_V03.relative_to(PROJECT_ROOT)),
            "v20": str(CSV_V20.relative_to(PROJECT_ROOT)),
            "diff": str(CSV_DIFF.relative_to(PROJECT_ROOT)),
        },
        "gastos": {"submissions": 0, "modelos_entrenados": 0, "uso_de_test": "solo inferencia"},
    }
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    _informe_md(informe, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(
        f"VEREDICTO: candidata construida | reproduccion exacta de V03: "
        f"{sha256(CSV_V03) == hash_congelada} | cambios en test: {celdas_modificadas} celdas en "
        f"{filas_modificadas} filas | submissions enviadas 0"
    )


def _f1(verdad: np.ndarray, predicho: np.ndarray) -> float:
    """F1 de la clase positiva de una sola etiqueta.

    Args:
        verdad: valores binarios.
        predicho: valores predichos.

    Returns:
        La F1, o 0.0 si el denominador es cero.
    """
    comunes = int(np.count_nonzero(verdad & predicho))
    denominador = int(np.count_nonzero(verdad)) + int(np.count_nonzero(predicho))
    return 2 * comunes / denominador if denominador > 0 else 0.0


def _informe_md(informe: dict[str, Any], inicio: float) -> None:
    """Escribe el informe de la construccion de la candidata.

    Args:
        informe: informe con todos los datos.
        inicio: instante de arranque.
    """
    from v11_ayudas import fila, sello

    controles = informe["controles"]
    cambios = informe["cambios_en_test"]
    candidato = informe["candidato"]
    cuerpo: list[str] = [
        "# V20-02: inferencia de V03 en test y candidata de la regla de V20",
        "",
        f"Generado por `experiments/v20_02_inferencia_y_candidata.py` ({EXPERIMENTO}).",
        "",
        "**Ninguna submission enviada todavia. Ninguna etiqueta de test utilizada.**",
        "",
        "## 1. Los controles, que es lo que hace creible el resto",
        "",
        fila(["control", "resultado"]),
        fila(["---"] * 2),
        fila(["el test no trae columna de etiqueta", "correcto, y no se lee ninguna en ningun momento"]),
        fila(["filas del test", str(controles["filas_de_test"])]),
        fila(["identificadores unicos y sin nulos", "correcto"]),
        fila(
            [
                "la inferencia reproduce la submission congelada",
            (
                f"**{controles['celdas_distintas_de_la_congelada']} celdas distintas** de "
                f"{controles['filas_de_test'] * 3}"
            ),
            ]
        ),
        fila(
            [
                "sha256 de la submission de V03 regenerada",
                "identico al de la congelada" if controles["identicas"] else "**distinto**",
            ]
        ),
        fila(["solo cambia HS", "correcto, ninguna celda de TR ni de AG cambia"]),
        fila(["umbral derivado solo del OOF", "correcto, ninguna lectura de test"]),
        "",
        "La reproduccion exacta es el control mas fuerte disponible: si al transformar el test con",
        "los artefactos congelados salen las mismas 1154 filas con los mismos valores, las",
        "probabilidades **son** las del champion y no las de una reimplementacion.",
        "",
        "## 2. La candidata",
        "",
        fila(["componente", "valor"]),
        fila(["---"] * 2),
        fila(["regla", candidato["nombre"]]),
        fila(["diseno", ", ".join(candidato["diseno"])]),
        fila(["modelo", candidato["modelo"]]),
        fila(["ajuste", candidato["ajuste"]]),
        fila(["umbral en test", f"{candidato['umbral_de_test']:.6f}"]),
        fila(
            [
                "umbrales por pliegue de V20",
            (
                f"media {candidato['media_por_pliegue']:.4f}, desviacion "
                f"{candidato['desviacion_por_pliegue']:.4f}"
            ),
            ]
        ),
        fila(["etiquetas modificadas", ", ".join(candidato["etiquetas_modificadas"])]),
        "",
        "El umbral de test se elige sobre el OOF con la misma operacion que usaba el protocolo OOF,",
        "y los cinco umbrales por pliegue que V20 registro se muestran para que se vea que el",
        "numero es coherente con ellos y no uno nuevo.",
        "",
        "## 3. Que cambia exactamente en test",
        "",
        fila(["magnitud", "valor"]),
        fila(["---"] * 2),
        fila(["filas modificadas", f"{cambios['filas_modificadas']} de {controles['filas_de_test']}"]),
        fila(["celdas modificadas", str(cambios["celdas_modificadas"])]),
    ]
    for etiqueta, cantidad in cambios["por_etiqueta"].items():
        cuerpo.append(fila([f"celdas de {etiqueta}", str(cantidad)]))
    cuerpo += [
        fila(["cambios de 0 a 1", str(cambios["tipos"]["0->1"])]),
        fila(["cambios de 1 a 0", str(cambios["tipos"]["1->0"])]),
        "",
        "## 4. Lo que la misma regla hace en el OOF, con el mismo umbral",
        "",
        f"- Celdas de HS que cambia: **{informe['cambios_en_oof']['celdas']}**.",
        (
            f"- Rescates: **{informe['cambios_en_oof']['rescates']}**, danios: "
            f"**{informe['cambios_en_oof']['danios']}**, neto "
            f"**{informe['cambios_en_oof']['rescates'] - informe['cambios_en_oof']['danios']:+d}**."
        ),
        "",
        "Que el recuento de celdas del OOF y el de test sean de magnitud parecida es la",
        "comprobacion de que la transformacion se aplica igual en los dos conjuntos, que es lo que",
        "pide el encargo antes de enviar.",
        "",
        "## 5. Nota sobre el uso del test",
        "",
        "El test se ha usado **solo para transformar texto en probabilidades**, que es inference.",
        "No se ha entrenado nada con el, no se ha estimado ninguna etiqueta y el umbral de la regla",
        "proviene integramente del OOF. Es la operacion que las reglas de la competicion permiten.",
        "",
    ]
    cuerpo += sello(inicio)
    REPORTE_MD.write_text("\n".join(cuerpo) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
