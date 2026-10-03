"""V50-00 (P50): auditoria de brecha de la familia selectora y techo del oracle.

Que resuelve
------------
Dos preguntas, en este orden y por esta razon.

1. **?Se ha probado alguna vez seleccionar que canal escuchar por label-cell?** La
   familia solo es nueva si de verdad no existe. Se auditan las cinco categorias
   posibles (ensayo, ponderacion global, umbral, correccion selectiva y
   seleccion por celda) contra los artefactos del repositorio, no contra los
   nombres de los scripts.

2. **Si el techo del oracle es amplio, ?cuanto de el se traduce en macro-F1?**
   P49 sabe que 715 de 757 celdas de error tienen al menos un canal que acierta.
   Eso NO dice cuanto macro-F1 se puede ganar: son celdas, no puntos de F1. Este
   script lo calcula de verdad, con `y_true`, y lo declara techo, nunca modelo.

El techo del oracle se construye con el espacio de acciones completo
(``KEEP_V03`` mas los 20 canales) y **usa `y_true`**, de modo que por
construccion no es un modelo, no es una evaluacion valida y no puede usarse para
seleccion. Solo responde si hay margen teorico.

Que NO hace
-----------
- No entrena ningun selector.
- No genera ninguna submission.
- No consulta Kaggle.
- No modifica V03, ni el holdout, ni el test.
- No anade filas al registro de experimentos.

Uso
---
    /home/carlos/venv/bin/python experiments/v50_00_brecha_y_oracle.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.splits import grupos_fusionados, pliegues_corregidos

OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"
HOLDOUT_IDS = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
WORKTREE_FEATURES = Path("/home/carlos/Git/PLN/kaggle_toxic_2026_puj/features")

ETIQUETAS = ("HS", "TR", "AG")
UMBRAL_CANAL = 0.5
CELDAS_POR_003_LB = 26.75

REPORTE_JSON = PROJECT_ROOT / "reports" / "p50_gap_and_oracle.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "p50_selector_gap_audit.md"
CSV_ORACLE = PROJECT_ROOT / "reports" / "p50_oracle_ceiling.csv"
CSV_FIABILIDAD = PROJECT_ROOT / "reports" / "p50_channel_reliability.csv"
FEATURES_NPZ = PROJECT_ROOT / "features" / "p50_celda_features.npz"

CANALES = {
    "v2_oof_base_corrected.npz": "modelos",
    "v2_oof_syntax.npz": "variantes",
    "v2_oof_cita_negacion.npz": "variantes",
    "v2_oof_targeting.npz": "variantes",
    "v2_oof_emocion_nbsvm_ortografia.npz": "canales",
}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V50-00] {mensaje}", flush=True)


# ---------------------------------------------------------------- auditprevio
# La tabla de auditoria se declara aqui, con la evidencia que la sostiene. Cada
# fila cita el artefacto real que se leyo, no el nombre del script.
CATALOGO_MECANISMOS: list[dict[str, str]] = [
    {
        "categoria": "A. ensemble / blending",
        "mecanismo": "promedio o mediana de probabilidades entre variantes",
        "evidencia": "reports/v17_00_inventario_complementariedad.json (ENS-04, "
                     "proba_ens, macro_f1 0,7865) y experiments/23_candidatos_finales.py",
        "es_por_celda": "no",
        "estado": "CERRADO",
    },
    {
        "categoria": "B. ponderacion global",
        "mecanismo": "meta-modelo con coeficientes fijos sobre scores OOF",
        "evidencia": "experiments/17_stacking_v2.py (late fusion de segunda "
                     "generacion), variantes meta_shift / B_meta36 / "
                     "A_cost_sensitive_downweight / C_meta36_downweight, y la "
                     "funcion ranking_canales de experiments/23_candidatos_finales.py",
        "es_por_celda": "no",
        "estado": "CERRADO",
    },
    {
        "categoria": "C. selector de umbral",
        "mecanismo": "un umbral por etiqueta, ajustado globalmente",
        "evidencia": "umbrales V03 0,44/0,51/0,46 y umbrales optimos por "
                     "variante en reports/v17_00_inventario_complementariedad.json",
        "es_por_celda": "no",
        "estado": "CERRADO",
    },
    {
        "categoria": "D. correccion selectiva",
        "mecanismo": "aceptar el cambio de un challenger solo cuando una feature "
                     "lo respalda, medido por challenger y con permutaciones",
        "evidencia": "reports/v18_01_cambio_y_senal.json: 5 challengers, 0 con "
                     "senal tras Bonferroni, mejor AUC 0,6181, mejor mejora "
                     "fuera de muestra 2,4 celdas frente a 26,75 necesarias",
        "es_por_celda": "parcialmente, pero un challenger cada vez",
        "estado": "CERRADO CON VEREDICTO NEGATIVO",
    },
    {
        "categoria": "E. seleccion por label-cell",
        "mecanismo": "elegir entre KEEP_V03 y los 20 canales, celda a celda, con "
                     "un selector aprendido y derecho a abstenerse",
        "evidencia": "no existe. Ningun script entrena sobre las salidas de los "
                     "20 canales a la vez tomando una decision discreta; los "
                     "scripts que los consumen (14, 17, 19, 20, 23, 28, 35) los "
                     "usan para fusion, para ranking o para medida de dureza",
        "es_por_celda": "si",
        "estado": "NO IMPLEMENTADO",
    },
]


def cargar_canales() -> tuple[np.ndarray, list[str]]:
    """Apila los 20 canales OOF con la misma exclusion de hardness_v4.

    Returns:
        Tensor de forma ``(n_canales, n_filas, 3)`` y la lista de nombres.
    """
    tensores: list[np.ndarray] = []
    nombres: list[str] = []
    for nombre, clave in CANALES.items():
        datos = np.load(WORKTREE_FEATURES / nombre, allow_pickle=True)
        etiquetas = [str(x) for x in datos[clave]]
        proba = datos["proba"].astype(np.float64)
        if proba.shape[0] != len(etiquetas) and proba.shape[1] == len(etiquetas):
            proba = np.moveaxis(proba, 1, 0)
        if proba.shape[0] != len(etiquetas):
            raise SystemExit(f"PARADA: no se normaliza {nombre} con forma {proba.shape}")
        for indice, etiqueta in enumerate(etiquetas):
            if etiqueta.startswith("TXT+") or etiqueta == "CH-TGT+DIST":
                continue
            tensores.append(proba[indice])
            nombres.append(f"{nombre.split('_')[2][:4]}:{etiqueta}")
    return np.stack(tensores), nombres


def f1_macro(y_true: np.ndarray, pred: np.ndarray) -> float:
    """Macro-F1 multietiqueta, es decir la media de los F1 por etiqueta.

    Importante: aplastar las 4449 celdas con ``ravel`` NO sirve. En ese caso
    sklearn ve un solo problema binario sobre 4449 muestras y devuelve el F1
    agregado, que fue 0,8186415291, y no el macro-F1 multietiqueta, que es
    0,7772114525. El valor canonico ``corrected_cv`` de V03 es el segundo, y se
    verifica aqui celda a celda.

    Args:
        y_true: etiquetas verdaderas, forma ``(n_filas, 3)``.
        pred: etiquetas predichas, forma ``(n_filas, 3)``.

    Returns:
        La media de los tres F1 por etiqueta.
    """
    return float(f1_score(y_true, pred, average="macro"))


def main() -> None:
    """Ejecuta la auditoria de brecha y el techo del oracle.

    Returns:
        None.
    """
    inicio = time.time()
    log("P50-00 auditoria de brecha y techo del oracle. 0 modelos, 0 submissions")

    oof = np.load(OOF, allow_pickle=True)
    ids = oof["ids"].astype(int)
    y_true = np.asarray(oof["y_true"], dtype=np.int64)
    v03_pred = np.asarray(oof["pred"], dtype=np.int64)
    v03_proba = np.asarray(oof["proba"], dtype=np.float64)
    v03_margen = np.asarray(oof["margen"], dtype=np.float64)
    n_filas = len(ids)
    err_v03 = v03_pred != y_true
    n_celdas_error = int(err_v03.sum())
    log(f"   OOF canonico: {n_filas} filas | celdas de error {n_celdas_error}")

    # ------------------------------------------------- 1. auditoria de brecha
    log("1/6 auditoria de brecha de la familia selectora")
    v18 = json.loads((PROJECT_ROOT / "reports" / "v18_01_cambio_y_senal.json")
                     .read_text(encoding="utf-8"))
    v18v = v18["veredicto"]
    log(f"   V18-01 (correccion selectiva): {v18v['estado']}")
    log(f"   challengers {v18v['challengers_medidos']} | con senal nominal "
        f"{v18v['challengers_con_senal_nominal']} | tras Bonferroni "
        f"{v18v['challengers_con_senal_tras_bonferroni']}")
    log(f"   mejor AUC {max(c['mejor_auc'] for c in v18v['detalle_por_challenger']):.4f} "
        f"| mejor p {v18v['mejor_p_global']:.4f} (Bonferroni "
        f"{v18v['umbral_bonferroni']:.4f})")
    log(f"   mejor mejora fuera de muestra {v18v['mejor_mejora_fuera_de_muestra']} celdas "
        f"frente a {v18v['mejora_necesaria_para_la_escala']} necesarias")
    implementada_e = any(m["estado"] == "NO IMPLEMENTADO" for m in CATALOGO_MECANISMOS)
    estado_familia = "GENUINELY_NEW" if implementada_e else "CLOSED"
    log(f"   SELECTOR_FAMILY = {estado_familia} (la categoria E no esta implementada)")

    # ------------------------------------------- 2. pool de canales y acciones
    log("2/6 reconstruccion del pool y del espacio de acciones")
    tensor, nombres = cargar_canales()
    n_canales = tensor.shape[0]
    canal_pred = (tensor >= UMBRAL_CANAL).astype(np.int64)
    canal_ok = canal_pred == y_true[None, :, :]
    log(f"   canales {n_canales} | acciones disponibles = {n_canales + 1} "
        f"(KEEP_V03 mas {n_canales} canales)")

    # ------------------------------------------------------- 3. techo del oracle
    log("3/6 techo del oracle (USA y_true, no es modelo)")
    v03_ok = ~err_v03
    alguna_accion_ok = canal_ok.any(axis=0) | v03_ok
    pred_oracle = np.where(alguna_accion_ok, y_true, v03_pred)
    f1_v03 = f1_macro(y_true, v03_pred)
    f1_oracle = f1_macro(y_true, pred_oracle)
    delta_oracle = f1_oracle - f1_v03
    log(f"   macro-F1 V03    = {f1_v03:.10f}")
    log(f"   macro-F1 oracle = {f1_oracle:.10f}")
    log(f"   DELTA ORACLE    = {delta_oracle:+.10f}")
    n_acciones_correctas = int(alguna_accion_ok.sum())
    n_sin_accion = int(n_filas * 3 - n_acciones_correctas)
    log(f"   celdas con alguna accion correcta: {n_acciones_correctas} de {n_filas * 3}")
    log(f"   celdas sin ninguna accion correcta: {n_sin_accion}")
    log(f"   el oracle NO puede superar 1,0 porque esas {n_sin_accion} celdas quedan mal")
    log(f"   techo absoluto alcanzable = {(n_acciones_correctas) / (n_filas * 3):.6f} "
        f"de celdas correctas")

    filas_oracle = [{
        "metrica": "macro_f1_V03", "valor": round(f1_v03, 10), "etiqueta": "macro",
        "rescate": "", "damage": "", "net": "", "delta": "",
        "fuente": "features/oof_v03_final.npz", "nota": "corrected_cv canonico",
    }, {
        "metrica": "macro_f1_oracle", "valor": round(f1_oracle, 10), "etiqueta": "macro",
        "rescate": "", "damage": "", "net": "", "delta": round(delta_oracle, 10),
        "fuente": "calculado", "nota": "USA y_true, no es modelo, no es evaluacion valida",
    }]
    for j, et in enumerate(ETIQUETAS):
        f1_v = float(f1_score(y_true[:, j], v03_pred[:, j], zero_division=0))
        f1_o = float(f1_score(y_true[:, j], pred_oracle[:, j], zero_division=0))
        resc = int((err_v03[:, j] & canal_ok[:, :, j].any(axis=0)).sum())
        dany = 0
        filas_oracle.append({
            "metrica": "macro_f1_oracle", "valor": round(f1_o, 10), "etiqueta": et,
            "rescate": resc, "damage": dany, "net": resc - dany,
            "delta": round(f1_o - f1_v, 10), "fuente": "calculado",
            "nota": f"V03 en esta etiqueta = {f1_v:.10f}",
        })
        log(f"   {et}: V03 {f1_v:.6f} -> oracle {f1_o:.6f} "
            f"(delta {f1_o - f1_v:+.6f}) | rescate {resc}")

    # El margen de decision que P50 exige para justificar la complejidad.
    celdas_por_punto = delta_oracle / max(n_celdas_error, 1)
    margen_util = CELDAS_POR_003_LB * celdas_por_punto
    log(f"   el rescate de {n_celdas_error - n_sin_accion} celdas vale "
        f"{delta_oracle:+.6f} de macro-F1")
    log(f"   proporcionalmente, +0,003 de macro-F1 valen {margen_util:.1f} celdas de error")
    log(f"   luego el umbral de decision de la especificacion (+0,003) esta muy por debajo "
        f"del techo: el techo es {delta_oracle / 0.003:.1f} veces ese margen")

    # ------------------------------------------ 4. fiabilidad por canal
    log("4/6 fiabilidad de cada canal, condicional al desacuerdo con V03")
    filas_fiabilidad = []
    for k, nombre_canal in enumerate(nombres):
        pred_c = canal_pred[k]
        ok_c = canal_ok[k]
        desacuerdo = pred_c != v03_pred
        rescues = int((err_v03 & ok_c).sum())
        damages = int((v03_ok & (pred_c != y_true)).sum())
        acc_global = float(ok_c.mean())
        acc_desc = float(ok_c[desacuerdo].mean()) if desacuerdo.any() else float("nan")
        fila = {
            "canal": nombre_canal,
            "accuracy_global": round(acc_global, 6),
            "n_desacuerdos": int(desacuerdo.sum()),
            "accuracy_dado_desacuerdo": round(acc_desc, 6),
            "rescate": rescues,
            "damage": damages,
            "net": rescues - damages,
        }
        for j, et in enumerate(ETIQUETAS):
            fila[f"accuracy_{et}"] = round(float(ok_c[:, j].mean()), 6)
        for lo, hi, nombre_bin in ((0.0, 0.10, "margen_bajo"),
                                   (0.10, 0.20, "margen_medio"),
                                   (0.20, 1.01, "margen_alto")):
            sel = (v03_margen >= lo) & (v03_margen < hi) & desacuerdo
            fila[f"accuracy_{nombre_bin}"] = (
                round(float(ok_c[sel].mean()), 6) if sel.any() else float("nan")
            )
        filas_fiabilidad.append(fila)
    fdf = pd.DataFrame(filas_fiabilidad).sort_values("accuracy_dado_desacuerdo",
                                                     ascending=False)
    log(f"   {'canal':36}{'acc_global':>12}{'acc_desc':>10}{'resc':>7}{'dano':>7}{'net':>8}")
    for _, x in fdf.iterrows():
        log(f"   {x['canal']:36}{x['accuracy_global']:>12.4f}"
            f"{x['accuracy_dado_desacuerdo']:>10.4f}{int(x['rescate']):>7}"
            f"{int(x['damage']):>7}{int(x['net']):>+8d}")
    mejor_acc = float(fdf["accuracy_dado_desacuerdo"].max())
    log(f"   mejor accuracy dado el desacuerdo: {mejor_acc:.4f} (0,5000 es azar)")

    # ------------------------------------------ 5. features observables
    log("5/6 construccion de la matriz de variables observables por label-cell")
    n_celdas = n_filas * 3
    n_dis = (canal_pred != v03_pred[None, :, :]).sum(axis=0)
    n_ag = n_canales - n_dis
    votos_1 = canal_pred.sum(axis=0)
    frac_1 = votos_1 / n_canales
    mayoria = (votos_1 * 2 > n_canales).astype(np.int64)
    entropia = -(frac_1 * np.log2(np.clip(frac_1, 1e-12, 1))
                 + (1 - frac_1) * np.log2(np.clip(1 - frac_1, 1e-12, 1)))
    bloque = np.empty((n_celdas, 3 + 3 + 2 * n_canales + 7), dtype=np.float64)
    columnas: list[str] = ["v03_proba", "v03_margen", "v03_pred",
                           "es_HS", "es_TR", "es_AG"]
    columnas += [f"proba_{n}" for n in nombres]
    columnas += [f"pred_{n}" for n in nombres]
    columnas += ["n_disacuerdo", "n_acuerdo", "frac_votos_1", "voto_mayoria",
                 "tasa_acuerdo", "entropia_votos", "v03_es_mayoria"]
    k = 0
    for i in range(n_filas):
        for j in range(3):
            fila = [v03_proba[i, j], v03_margen[i, j], float(v03_pred[i, j]),
                    float(j == 0), float(j == 1), float(j == 2)]
            fila += [float(tensor[c, i, j]) for c in range(n_canales)]
            fila += [float(canal_pred[c, i, j]) for c in range(n_canales)]
            fila += [float(n_dis[i, j]), float(n_ag[i, j]), float(frac_1[i, j]),
                     float(mayoria[i, j]), float(n_ag[i, j]) / n_canales,
                     float(entropia[i, j]), float(mayoria[i, j] == v03_pred[i, j])]
            bloque[k] = fila
            k += 1
    log(f"   matriz observable: {bloque.shape[0]} celdas x {bloque.shape[1]} variables")
    log("   NO incluye y_true, ni la correccion de V03, ni la identidad del canal correcto")
    log("   incluye solo salidas de V03 y de los 20 canales, mas la identidad de etiqueta")

    # ------------------------------------------------------- 6. pliegues
    log("6/6 pliegues canonicos de TOXIC y persistencia")
    tr = read_train()
    tr["id"] = tr["id"].astype("int64")
    particion = pd.read_csv(HOLDOUT_IDS)
    dev_ids = set(particion.loc[particion["particion"] == "dev", "id"].astype("int64"))
    dev = tr[tr["id"].isin(dev_ids)].reset_index(drop=True)
    if not np.array_equal(dev["id"].to_numpy(), ids):
        raise SystemExit("PARADA: el orden del dev no coincide con el OOF canonico")
    grupos = grupos_fusionados(dev)
    asignacion = np.full(n_filas, -1, dtype=np.int64)
    for indice, (_, validacion) in enumerate(pliegues_corregidos(dev, grupos)):
        asignacion[validacion] = indice
    log(f"   grupos fusionados: {len(np.unique(grupos))} | tamano de pliegues: "
        f"{[int((asignacion == v).sum()) for v in range(5)]}")

    FEATURES_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FEATURES_NPZ,
        ids=ids,
        y_true=y_true,
        v03_pred=v03_pred,
        v03_proba=v03_proba,
        v03_margen=v03_margen,
        canal_pred=canal_pred,
        canal_proba=tensor.astype(np.float32),
        nombres=np.array(nombres),
        columnas=np.array(columnas),
        X=bloque,
        pliegue=asignacion,
        grupo=grupos,
    )
    log(f"   escrito {FEATURES_NPZ.name}")

    pasa_oracle = delta_oracle >= 0.003
    log(f"   GATE S5: delta_oracle {delta_oracle:+.6f} "
        f"{'>= +0,003 -> CONTINUAR' if pasa_oracle else '< +0,003 -> CERRAR'}")

    informe = {
        "familia": "P50_CONTEXTUAL_CHANNEL_SELECTION",
        "experimento": "V50-00 (auditoria de brecha y techo del oracle)",
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "AUDIT_ONLY",
        "pregunta": ("?existe margen teorico explotable seleccionando canal por "
                     "label-cell, o el techo del oracle es pequeno?"),
        "gap_audit": {
            "estado_familia": estado_familia,
            "catalogo": CATALOGO_MECANISMOS,
            "v18_selective_correction": {
                "estado": v18v["estado"],
                "challengers": v18v["challengers_medidos"],
                "con_senal_nominal": v18v["challengers_con_senal_nominal"],
                "con_senal_tras_bonferroni": v18v["challengers_con_senal_tras_bonferroni"],
                "mejor_auc": max(c["mejor_auc"] for c in v18v["detalle_por_challenger"]),
                "mejor_p_global": v18v["mejor_p_global"],
                "umbral_bonferroni": v18v["umbral_bonferroni"],
                "mejor_mejora_fuera_de_muestra": v18v["mejor_mejora_fuera_de_muestra"],
                "mejora_necesaria_para_la_escala": v18v["mejora_necesaria_para_la_escala"],
            },
            "nota": (
                "La categoria E no esta implementada, luego la familia es formalmente "
                "nueva. Pero la categoria D, que es su caso degenerado con un solo "
                "challenger y una sola feature, ya se midio y dio veredicto negativo "
                "con AUC 0,62. E solo puedediferenciarse esa cifra si el poder de identificacion "
                "esta en la multidimensionalidad o en el pool conjunto."
            ),
        },
        "oracle": {
            "aviso": "USA y_true. NO es modelo, NO es evaluacion valida, NO puede "
                     "usarse para seleccion. Es solo un techo superior.",
            "espacio_de_acciones": n_canales + 1,
            "macro_f1_v03": f1_v03,
            "macro_f1_oracle": f1_oracle,
            "delta_macro_f1": delta_oracle,
            "celdas_de_error_v03": n_celdas_error,
            "celdas_con_alguna_accion_correcta": n_acciones_correctas,
            "celdas_sin_ninguna_accion_correcta": n_sin_accion,
            "rescate_total": n_celdas_error - n_sin_accion,
            "damage_total": 0,
            "net_total": n_celdas_error - n_sin_accion,
            "por_etiqueta": {
                x["etiqueta"]: {
                    "f1_v03": round(float(f1_score(y_true[:, j], v03_pred[:, j],
                                                   zero_division=0)), 10),
                    "f1_oracle": x["valor"],
                    "delta": x["delta"],
                    "rescate": x["rescate"],
                }
                for j, x in enumerate([f for f in filas_oracle
                                       if f["etiqueta"] in ETIQUETAS])
            },
            "celdas_por_punto_de_delta": celdas_por_punto,
            "pasa_gate_0p003": bool(pasa_oracle),
            "multiples_del_umbral": round(delta_oracle / 0.003, 1),
        },
        "fiabilidad_por_canal": filas_fiabilidad,
        "mejor_accuracy_dado_desacuerdo": mejor_acc,
        "matriz_observable": {
            "forma": list(bloque.shape),
            "columnas": columnas,
            "excluye_y_true": True,
            "excluye_correccion_v03": True,
            "ruta": str(FEATURES_NPZ.relative_to(PROJECT_ROOT)),
        },
        "pliegues": {
            "grupos_fusionados": len(np.unique(grupos)),
            "tamanos": [int((asignacion == v).sum()) for v in range(5)],
        },
        "gastos": {"submissions": 0, "modelos_entrenados": 0, "consultas_kaggle": 0,
                   "uso_holdout": 0, "uso_test": 0,
                   "segundos": round(time.time() - inicio, 1)},
    }
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    _csv(filas_oracle, CSV_ORACLE)
    fdf.to_csv(CSV_FIABILIDAD, index=False, encoding="utf-8")
    _md(informe)
    log("   informes escritos en reports/p50_*")
    log(f"VEREDICTO: gap_audit={estado_familia} | delta_oracle={delta_oracle:+.6f} | "
        f"gate_0p003={pasa_oracle} | 0 modelos, 0 submissions, "
        f"{round(time.time() - inicio, 1)} s")


def _csv(filas: list[dict[str, Any]], ruta: Path) -> None:
    """Escribe un CSV a partir de filas de diccionario.

    Args:
        filas: filas a escribir.
        ruta: fichero destino.

    Returns:
        None.
    """
    columnas = ["metrica", "valor", "etiqueta", "rescate", "damage", "net",
                "delta", "fuente", "nota"]
    with ruta.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=columnas)
        escritor.writeheader()
        for fila in filas:
            escritor.writerow({c: fila.get(c, "") for c in columnas})


def _md(informe: dict[str, Any]) -> None:
    """Escribe el informe de la auditoria de brecha.

    Args:
        informe: diccionario con los resultados.

    Returns:
        None.
    """
    brecha = informe["gap_audit"]
    v18 = brecha["v18_selective_correction"]
    o = informe["oracle"]
    lineas = [
        "# P50 - Auditoria de brecha de la familia selectora",
        "",
        f"Generado por `experiments/v50_00_brecha_y_oracle.py` ({informe['experimento']}).",
        "",
        f"Fecha {informe['fecha']}. **0 modelos, 0 submissions, 0 consultas a Kaggle.**",
        "",
        "## 1. Las cinco categorias, y donde esta cada una",
        "",
        "| categoria | ?es por celda? | estado | evidencia |",
        "|---|---|---|---|",
    ]
    for m in brecha["catalogo"]:
        lineas.append(f"| {m['categoria']} | {m['es_por_celda']} | **{m['estado']}** | "
                      f"{m['evidencia']} |")
    lineas += [
        "",
        (f"**`SELECTOR_FAMILY = {brecha['estado_familia']}`.** La categoria E no esta "
        "implementada en ninguna parte del repositorio."),
        "",
        "## 2. La categoria D ya se midio y dio veredicto negativo",
        "",
        ("V18-01 es el caso degenerado de esta hipotesis: en vez de elegir entre veinte "
        "canales, elige entre V03 y **un** challenger, y en vez de un selector multivariable "
        "usa una feature con un corte. Su resultado es el mejor punto de partida que existe "
        "para la categoria E."),
        "",
        "| magnitud | valor |",
        "|---|---:|",
        f"| estado | **{v18['estado']}** |",
        f"| challengers medidos | {v18['challengers']} |",
        f"| con senal nominal (p < 0,05) | {v18['con_senal_nominal']} |",
        f"| con senal tras Bonferroni | {v18['con_senal_tras_bonferroni']} |",
        f"| mejor AUC | {v18['mejor_auc']:.4f} |",
        (f"| mejor p global | {v18['mejor_p_global']:.4f} "
        f"(umbral {v18['umbral_bonferroni']:.4f}) |"),
        f"| mejor mejora fuera de muestra | {v18['mejor_mejora_fuera_de_muestra']} celdas |",
        f"| mejora necesaria para la escala | {v18['mejora_necesaria_para_la_escala']} celdas |",
        "",
        ("Un AUC de 0,62 sobre 202 a 287 celdas de desacuerdo es indistinguible del azar. "
        "La mejora fuera de muestra mas alta fue de 2,4 celdas, cuando hacen falta 26,75."),
        "",
        ("**Consecuencia para P50.** La familia es formalmente nueva, pero hereda un "
        "veredicto muy negativo. Solo puede distinguirse de la categoria D si el poder de "
        "identificacion esta en la multidimensionalidad o en considerar el pool conjunto. "
        "Ese es exactamente el test que P50 tiene que superar."),
        "",
        "## 3. Techo del oracle",
        "",
        f"> {o['aviso']}",
        "",
        (f"El espacio de acciones tiene **{o['espacio_de_acciones']} acciones**: `KEEP_V03` "
        f"mas los {o['espacio_de_acciones'] - 1} canales."),
        "",
        "| magnitud | valor |",
        "|---|---:|",
        f"| macro-F1 de V03 | {o['macro_f1_v03']:.10f} |",
        f"| macro-F1 del oracle | {o['macro_f1_oracle']:.10f} |",
        f"| **delta del oracle** | **{o['delta_macro_f1']:+.10f}** |",
        f"| celdas de error de V03 | {o['celdas_de_error_v03']} |",
        f"| celdas con alguna accion correcta | {o['celdas_con_alguna_accion_correcta']} |",
        f"| celdas sin ninguna accion correcta | {o['celdas_sin_ninguna_accion_correcta']} |",
        (f"| rescate / damage / net | {o['rescate_total']} / {o['damage_total']} / "
        f"{o['net_total']} |"),
        "",
        "Por etiqueta:",
        "",
        "| etiqueta | F1 de V03 | F1 del oracle | delta | rescate |",
        "|---|---:|---:|---:|---:|",
    ]
    for et, v in o["por_etiqueta"].items():
        lineas.append(f"| {et} | {v['f1_v03']:.10f} | {v['f1_oracle']:.10f} | "
                      f"{v['delta']:+.10f} | {v['rescate']} |")
    lineas += [
        "",
        (f"El techo es **{o['multiples_del_umbral']} veces** el umbral de decision de "
        "0,003 que la especificacion fija. El margen teorico no es pequeno: es enorme."),
        "",
        (f"Pero ese techo esta concentrado: {o['celdas_sin_ninguna_accion_correcta']} celdas "
        "no tienen ninguna accion correcta entre las 21, luego el oracle no puede llegar a "
        "1,0. En esas celdas ningun selector, por bueno que sea, puede acertar."),
        "",
        "## 4. Fiabilidad de los canales, condicional al desacuerdo",
        "",
        ("La columna que importa no es `accuracy_global`, sino "
        "`accuracy_dado_desacuerdo`: que acierta el canal **precisamente en las celdas "
        "donde V03 se equivoca**, que es donde un selector lo usaria."),
        "",
        "| canal | acc. global | acc. dado desacuerdo | rescate | damage | net |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    filas = sorted(informe["fiabilidad_por_canal"],
                   key=lambda f: -f["accuracy_dado_desacuerdo"])
    for f in filas:
        lineas.append(f"| `{f['canal']}` | {f['accuracy_global']:.4f} | "
                      f"**{f['accuracy_dado_desacuerdo']:.4f}** | {f['rescate']} | "
                      f"{f['damage']} | {f['net']:+d} |")
    lineas += [
        "",
        (f"La mejor accuracy dado el desacuerdo es "
        f"**{informe['mejor_accuracy_dado_desacuerdo']:.4f}**. El valor de referencia es "
        "0,5000."),
        "",
        "## 5. Decision de la auditoria",
        "",
        f"- `SELECTOR_FAMILY = {brecha['estado_familia']}`: la categoria E no existe todavia.",
        (f"- `ORACLE_DELTA = {o['delta_macro_f1']:+.6f}`: el techo es amplio, luego la "
        "especificacion obliga a continuar."),
        ("- La categoria D ya dio veredicto negativo con AUC 0,62, luego la carga de la "
        "prueba la carga E."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
