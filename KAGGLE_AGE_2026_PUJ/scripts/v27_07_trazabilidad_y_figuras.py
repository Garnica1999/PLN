"""V27 - 07 - Trazabilidad en experiments.csv y figuras maestras de la investigacion completa.

Este script hace tres cosas, y ninguna es entrenar:

1. **Anade a `experiments.csv` las tres filas de los experimentos realmente ejecutados** en V27:
   `V27-01` (H12), `V27-02` (H11) y `V27-05` (H03). No anade filas para `V27-03` (correccion de un
   diagnostico) ni `V27-04` (analisis estructural), porque ninguno es un experimento de hipotesis, y
   tampoco para el cierre de H14, que no entreno. El prompt vigente es explicito: una hipotesis
   cerrada sin entrenamiento no consume experimento.

2. **Genera las figuras maestras** que pide el punto 31 del prompt: corrected_cv por experimento,
   delta frente a V20, consistencia por fold, rescate frente a dano, F1 por clase y evolucion
   cronologica, mas el efecto del percentil y el efecto de las restricciones monotonas.

3. **Audita la consistencia** entre la tabla maestra, los artefactos y `experiments.csv`, y falla
   con codigo distinto de cero si algo no cuadra. Es la auditoria del punto 32 aplicada a los datos
   antes de redactar el informe final.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

REGISTRO = ROOT / "experiments.csv"
MAESTRA = cfg.REPORTS / "v26_master_results.csv"
FIG_MAESTRA = cfg.FIGURES / "v27_maestra_resultados.png"
FIG_FOLDS = cfg.FIGURES / "v27_maestra_folds.png"
FIG_PERCENTIL = cfg.FIGURES / "v27_maestra_efecto_percentil.png"
CV_V20 = 0.6097782
LB_V20 = 0.62689
#: Ruta del artefacto de cada experimento de V27, para citarla en `experiments.csv`.
RUTA_ARTEFACTO = {
    "V27-01": "v27_01_h12_reliability.json",
    "V27-02": "v27_02_h11_monotone.json",
    "V27-05": "v27_05_h03_percentil.json",
}
#: Orden cronologico de los candidatos medidos, con su artefacto.
CANDIDATOS = [
    ("V26-01", "H02 afín", "experiments/v26_01_rank_representacion.json", "#4575b4"),
    ("V27-01", "H12 fiabilidad", "experiments/v27_01_h12_reliability.json", "#74add1"),
    ("V27-02", "H11 monotonas", "experiments/v27_02_h11_monotone.json", "#fdae61"),
    ("V27-05", "H03 percentil", "experiments/v27_05_h03_percentil.json", "#984ea3"),
]
#: T0 de V20, el canonico contra el que se mide todo.
FILAS_EXPERIMENTO = {
    "V27-01": {
        "bloque": "21", "fecha": "2026-09-28",
        "etapa": "V27 H12 fiabilidad por modelo y clase entregada al meta-modelo",
        "pregunta": ("si la fiabilidad de cada miembro del ensemble depende de la clase de edad, "
                     "puede el meta-modelo beneficiarse de conocer explicitamente esa fiabilidad?"),
        "hipotesis": ("el meta-modelo necesita la fiabilidad por miembro y clase, que no puede "
                      "inferir sola de las probabilidades por fila"),
        "features": ("las 279 de V20 mas 78 columnas de fiabilidad: 13 modelos con las seis clases "
                     "por 6 clases. Se excluyen E028a_pares (5 columnas) y E015_ordinal (1)"),
        "preprocessing": ("media de la columna del modelo y clase sobre las filas de esa clase del "
                          "TRAINING FOLD, replicada a todas las filas"),
        "representacion": ("cambia: anade 78 columnas constantes por fila. No altera ninguna de las "
                           "279 existentes"),
        "modelo": "V20 exacto: cinco LightGBM binarios con HIERBA congelada, cadena de umbrales y blend de cuatro cuartos",
        "hiperparametros": ('{"medida": "media por clase del training fold", "modelos": 13, '
                            '"columnas_anadidas": 78, "gate_delta": 0.003, "gate_folds": 4}'),
        "validacion": ("StratifiedGroupKFold de 5 folds con las mismas agrupaciones que V20; el "
                       "control debe reproducir 0,6097782 o la fase se detiene"),
        "pregunta_abierta": None,
    },
    "V27-02": {
        "bloque": "21", "fecha": "2026-09-28",
        "etapa": "V27 H11 restricciones monotonas nativas de LightGBM sobre la evidencia de clase",
        "pregunta": ("ciertas relaciones entre evidencia y clases pueden presentar estructura "
                     "monotona y el boosting libre esta gastando capacidad en relaciones no "
                     "plausibles?"),
        "hipotesis": ("para el umbral k, subir P(clase j) sube P(Y <= k-1) para toda j, luego log "
                      "P(clase) es monotona creciente con el acumulado"),
        "features": ("las 279 de V20 sin modificar. 42 con restriccion +1: los 7 modelos con "
                     "probabilidad genuina por clase"),
        "preprocessing": ("ninguno. El parametro nativo `monotone_constraints` se fija antes de "
                          "entrenar y no se toca despues"),
        "representacion": "identica a V20: no hay feature nueva, solo una restriccion del modelo",
        "modelo": "V20 exacto con `monotone_constraints` = +1 en 42 features y 0 en las 237 libres",
        "hiperparametros": ('{"restricciones": 42, "signo": "+1 creciente", "libres": 237, '
                            '"modelos": ["E-009", "E-010", "E-011", "E015_softmax", '
                            '"E017_probabilidades", "E021a_ordinal", "E021b_rocchio"], '
                            '"gate_delta": 0.003, "gate_folds": 4}'),
        "validacion": ("StratifiedGroupKFold de 5 folds con las mismas agrupaciones que V20; el "
                       "control debe reproducir 0,6097782 o la fase se detiene"),
        "pregunta_abierta": None,
    },
    "V27-05": {
        "bloque": "21", "fecha": "2026-09-28",
        "etapa": "V27 H03 transformacion a percentil de la evidencia de los modelos base",
        "pregunta": ("la evidencia de los modelos base tiene una estructura de rango util que una "
                     "transformacion afin no puede expresar?"),
        "hipotesis": ("el percentil conserva el orden entre filas y destruye la distancia, que es "
                      "la unica forma de aportar algo que el z-score de H02 no aportaba"),
        "features": "las 279 de V20 sin anadir ninguna; 84 columnas reescritas y 195 intactas",
        "preprocessing": ("percentil empirico contra la distribucion ordenada de cada TRAINING FOLD; "
                          "la validacion se transforma despues y nunca participa"),
        "representacion": "cambia: las 84 columnas de evidencia de los 15 modelos base pasan a percentil",
        "modelo": "V20 exacto: cinco LightGBM binarios con HIERBA congelada, cadena de umbrales y blend de cuatro cuartos",
        "hiperparametros": ('{"transformacion": "percentil empirico", "cortes": "ninguno", '
                            '"columnas": 84, "gate_delta": 0.003, "gate_folds": 4}'),
        "validacion": ("StratifiedGroupKFold de 5 folds con las mismas agrupaciones que V20; el "
                       "control debe reproducir 0,6097782 o la fase se detiene"),
        "pregunta_abierta": None,
    },
}


def _candidato(dato: dict) -> dict:
    """El segundo candidato del artefacto, que es el tratado y no el control."""
    return list(dato["candidatos"].values())[1]


def main() -> int:
    """Anade las filas, dibuja las figuras maestras y audita la consistencia."""
    autotest_numerico()
    inicio = time.time()
    datos = {cid: (etiqueta, json.loads((ROOT / ruta).read_text(encoding="utf-8")), color)
             for cid, etiqueta, ruta, color in CANDIDATOS}

    # ---------------------------------------------------------------- 1. experiments.csv
    registro = pd.read_csv(REGISTRO)
    n_registro = len(registro)
    columnas = registro.columns.tolist()
    nuevas = []
    # Solo se anaden filas de V27. `V26-01` ya esta en el registro con el identificador
    # `V26-01-REPRESENTACION`, y `V26-00-GAP-ANALYSIS` tambien: anadirlos seria duplicar.
    for cid in FILAS_EXPERIMENTO:
        etiqueta, dato, _ = datos[cid]
        if (registro["experiment_id"] == cid).any():
            print(f"[csv] {cid} ya existe, no se duplica", flush=True)
            continue
        t1 = _candidato(dato)
        base = dict(FILAS_EXPERIMENTO[cid])
        base.update({
            "experiment_id": cid, "pregunta_abierta": None,
            # `macro_f1_mean` es la **media no ponderada de los cinco folds**, que es la
            # convencion que usan V12 a V26 en este registro. El `corrected_cv` pooled es otra
            # magnitud y va en la columna `resultado`, mezclarlas daria una auditoria falsa.
            "macro_f1_mean": t1["media"], "macro_f1_std": t1["desviacion"],
            "resultado": (f"control {dato['control_t0']['corrected_cv']} reproduce "
                          f"{dato['control_t0']['reproducido']} | T1 pooled {t1['corrected_cv']} | "
                          f"delta {t1['delta_vs_V20']:+} | {t1['folds_positivos']}/5 folds | "
                          f"rescue {t1['rescue']} | damage {t1['damage']} | neto {t1['net']:+} | "
                          f"{t1['changed_rows']:,} filas cambian | "
                          f"IC95 {dato['gate']['bootstrap']['ic95']} | gate no cumplido"),
            "interpretacion": dato.get("conclusion_mecanismo")
                              or dato.get("distincion_con_H02")
                              or dato.get("novedad_funcional"),
            "decision": dato["decision"], "duracion_seg": dato.get("segundos"),
            "artefactos": (f"{etiqueta} | reports/v26_master_results.csv | "
                           f"experiments/{RUTA_ARTEFACTO[cid]} | figuras v27_*"),
        })
        base.pop("pregunta_abierta", None)
        nuevas.append({c: base.get(c, "") for c in columnas})
    if nuevas:
        registro = pd.concat([registro, pd.DataFrame(nuevas)], ignore_index=True)
    # Reparacion de la primera pasada, que escribio el pooled en `macro_f1_mean`. Se hace aqui y no
    # a mano para que quede en el historial del script y sea reproducible.
    reparadas = 0
    for cid in FILAS_EXPERIMENTO:
        mascara = registro["experiment_id"].astype(str).str.startswith(cid)
        if not mascara.any():
            continue
        t1 = _candidato(datos[cid][1])
        # Nombre distinto de `antes` a proposito: reutilizarlo pisaria el conteo de filas del
        # registro y el resumen final imprimiria una lista de valores en vez de un numero.
        valor_previo = registro.loc[mascara, "macro_f1_mean"].astype(float).round(6).tolist()
        registro.loc[mascara, "macro_f1_mean"] = t1["media"]
        if valor_previo != [round(t1["media"], 6)]:
            reparadas += 1
    registro.to_csv(REGISTRO, index=False)
    print(f"[csv] experiments.csv: {n_registro} filas antes de esta pasada, {len(registro)} ahora | "
          f"{len(nuevas)} anadidas, {reparadas} reparadas a media de folds, 0 ficticias",
          flush=True)

    # ---------------------------------------------------------------- 2. figuras maestras
    _figura_maestra(datos)
    _figura_folds(datos)
    _figura_percentil(datos)

    # ---------------------------------------------------------------- 3. auditoria
    problemas = _audita(registro, datos)
    if problemas:
        print("\n[AUDITORIA] DESAJUSTES:", flush=True)
        for problema in problemas:
            print(f"  - {problema}", flush=True)
        return 1
    print("[auditoria] experiments.csv <-> master results <-> artefactos: coherente", flush=True)
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


def _figura_maestra(datos: dict) -> None:
    """corrected_cv por experimento, delta frente a V20, rescate frente a dano y F1 por clase."""
    ids = list(datos)
    valores = [_candidato(datos[i][1]) for i in ids]
    fig, ejes = plt.subplots(2, 2, figsize=(15, 10))
    colores = [datos[i][2] for i in ids]

    eje = ejes[0, 0]
    eje.bar(range(len(ids) + 1), [CV_V20] + [v["corrected_cv"] for v in valores],
            color=["#333333"] + colores)
    eje.set_xticks(range(len(ids) + 1))
    eje.set_xticklabels(["V20"] + [f"{i}\n{datos[i][1]}" for i in ids], fontsize=7)
    eje.axhline(CV_V20, color="black", linestyle="--", linewidth=1)
    eje.set_ylabel("corrected_cv")
    eje.set_ylim(CV_V20 - 0.0008, CV_V20 + 0.0008)
    eje.set_title("corrected_cv por experimento (V20 congelado en la linea)", fontsize=10)

    eje = ejes[0, 1]
    deltas = [v["delta_vs_V20"] for v in valores]
    eje.bar(range(len(ids)), deltas, color=colores)
    eje.axhline(0, color="grey", linewidth=1)
    eje.axhline(0.003, color="black", linestyle="--", linewidth=1)
    eje.set_xticks(range(len(ids)))
    eje.set_xticklabels([f"{i}\n{datos[i][1]}" for i in ids], fontsize=7)
    eje.set_ylabel("delta frente a V20")
    eje.set_title("delta frente a V20 (la linea es el gate de +0,003)", fontsize=10)
    for i, d in enumerate(deltas):
        eje.text(i, d + (0.00008 if d >= 0 else -0.00016), f"{d:+.6f}", ha="center", fontsize=7)

    eje = ejes[1, 0]
    ancho = 0.35
    eje.bar(np.arange(len(ids)) - ancho / 2, [v["rescue"] for v in valores], ancho,
            label="rescate", color="#1a9850")
    eje.bar(np.arange(len(ids)) + ancho / 2, [v["damage"] for v in valores], ancho,
            label="dano", color="#d73027")
    eje.set_xticks(range(len(ids)))
    eje.set_xticklabels([f"{i}\n{datos[i][1]}" for i in ids], fontsize=7)
    eje.set_ylabel("filas")
    eje.set_title("rescate frente a dano (el neto decide el signo del delta)", fontsize=10)
    eje.legend(fontsize=8)

    eje = ejes[1, 1]
    clases = list(valores[0]["f1_por_clase"].keys())
    t0 = json.loads((ROOT / "experiments/v26_01_rank_representacion.json")
                    .read_text(encoding="utf-8"))["candidatos"]["T0 V20 control"]["f1_por_clase"]
    x = np.arange(len(clases))
    eje.bar(x - ancho / 2, [t0[c] for c in clases], ancho, label="V20", color="#333333")
    for i, v in enumerate(valores):
        eje.plot(x, [v["f1_por_clase"][c] for c in clases], marker="o", linewidth=1.2,
                 color=colores[i], label=f"{ids[i]} {datos[ids[i]][1]}")
    eje.set_xticks(x)
    eje.set_xticklabels(clases, fontsize=7)
    eje.set_ylabel("F1")
    eje.set_title("F1 por clase", fontsize=10)
    eje.legend(fontsize=6)
    fig.suptitle("AGE: investigacion completa desde V20. V20 sigue siendo el champion", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_MAESTRA, dpi=130)
    plt.close(fig)


def _figura_folds(datos: dict) -> None:
    """Consistencia por fold y evolucion cronologica de los candidatos."""
    ids = list(datos)
    fig, ejes = plt.subplots(1, 2, figsize=(15, 5))
    eje = ejes[0]
    for i in ids:
        v = _candidato(datos[i][1])
        eje.plot(range(5), [v[f"fold_{k}"] for k in range(1, 6)], marker="o", linewidth=1.3,
                 color=datos[i][2], label=f"{i} {datos[i][1]}")
    eje.plot(range(5), [CV_V20] * 5, linestyle="--", color="black", linewidth=1, label="V20 pool")
    eje.set_xticks(range(5))
    eje.set_xticklabels([f"fold {k}" for k in range(1, 6)])
    eje.set_ylabel("macro-F1 del fold")
    eje.set_title("consistencia por fold", fontsize=10)
    eje.legend(fontsize=7)

    eje = ejes[1]
    for i in ids:
        v = _candidato(datos[i][1])
        eje.bar(i, v["folds_positivos"], color=datos[i][2])
    eje.axhline(4, color="black", linestyle="--", linewidth=1)
    eje.set_ylabel("folds positivos")
    eje.set_ylim(0, 5)
    eje.set_title("folds positivos sobre 5 (el gate exige 4)", fontsize=10)
    for i in ids:
        v = _candidato(datos[i][1])
        eje.text(i, v["folds_positivos"] + 0.1, str(v["folds_positivos"]), ha="center", fontsize=8)
    fig.suptitle("Estabilidad de fold de los candidatos de V26 y V27", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_FOLDS, dpi=130)
    plt.close(fig)


def _figura_percentil(datos: dict) -> None:
    """Efecto del percentil: delta por fold, rescate/dano y distribucion del cambio."""
    v = _candidato(datos["V27-05"][1])
    fig, ejes = plt.subplots(1, 3, figsize=(17, 4.8))
    eje = ejes[0]
    eje.bar(range(5), v["delta_por_fold"], 0.5, color="#984ea3")
    eje.axhline(0, color="grey", linewidth=1)
    eje.axhline(0.003, color="black", linestyle="--", linewidth=1)
    eje.set_xticks(range(5))
    eje.set_xticklabels([f"fold {k}" for k in range(1, 6)])
    eje.set_ylabel("delta de macro-F1")
    eje.set_title(f"H03 percentil: delta {v['delta_vs_V20']:+.6f}, "
                  f"{v['folds_positivos']}/5 folds", fontsize=10)

    eje = ejes[1]
    clases = list(v["f1_por_clase"].keys())
    base = json.loads((ROOT / "experiments/v27_01_h12_reliability.json")
                      .read_text(encoding="utf-8"))["candidatos"]["T0 V20 control"]["f1_por_clase"]
    eje.bar(np.arange(len(clases)) - 0.2, [base[c] - v["f1_por_clase"][c] for c in clases], 0.4,
            color=["#1a9850" if base[c] > v["f1_por_clase"][c] else "#d73027"
                   for c in clases])
    eje.axhline(0, color="grey", linewidth=1)
    eje.set_xticks(range(len(clases)))
    eje.set_xticklabels(clases, fontsize=8)
    eje.set_ylabel("F1 de V20 menos F1 de H03")
    eje.set_title("mejora por clase (verde positivo, rojo negativo)", fontsize=10)

    eje = ejes[2]
    complementariedad = v["complementariedad_por_clase"]
    eje.bar(range(len(clases)), [complementariedad[c]["rescata"] for c in clases], 0.4,
            label="rescata", color="#1a9850")
    eje.bar(np.arange(len(clases)) + 0.4, [complementariedad[c]["danio"] for c in clases], 0.4,
            label="dano", color="#d73027")
    eje.set_xticks(np.arange(len(clases)) + 0.2)
    eje.set_xticklabels(clases, fontsize=8)
    eje.set_ylabel("filas")
    eje.set_title(f"complementariedad por clase (neto {v['net']:+})", fontsize=10)
    eje.legend(fontsize=8)
    fig.suptitle("H03: el unico resultado positivo, y por que no llega a finalista", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_PERCENTIL, dpi=130)
    plt.close(fig)


def _audita(registro: pd.DataFrame, datos: dict) -> list[str]:
    """Comprobaciones cruzadas entre experiments.csv, la tabla maestra y los artefactos.

    `datos` se pasa como argumento y **no** se lee del ambito de `main()`: leerlo de ahi seria un
    `F821`, y mas grave haria que la auditoria dependediera de un closure en vez de su contrato.
    """
    problemas = []
    maestra = pd.read_csv(MAESTRA)
    for cid, (_, dato, _) in datos.items():
        t1 = _candidato(dato)
        fila = maestra.loc[maestra["ID"] == cid]
        if fila.empty:
            problemas.append(f"{cid} no esta en {MAESTRA.name}")
            continue
        if abs(float(fila["corrected_cv"].iloc[0]) - t1["corrected_cv"]) > 1e-6:
            problemas.append(f"{cid}: la maestra dice "
                              f"{fila['corrected_cv'].iloc[0]} y el artefacto {t1['corrected_cv']}")
        # En `experiments.csv` el identificador lleva sufijo (`V26-01-REPRESENTACION`,
        # `V27-01-H12`), asi que se busca por prefijo y no por igualdad exacta.
        fila_csv = registro.loc[registro["experiment_id"].astype(str).str.startswith(cid)]
        if fila_csv.empty:
            problemas.append(f"{cid} no esta en experiments.csv")
        elif abs(float(fila_csv["macro_f1_mean"].iloc[0]) - t1["media"]) > 1e-6:
            problemas.append(f"{cid}: experiments.csv dice "
                              f"{fila_csv['macro_f1_mean'].iloc[0]}, el artefacto dice media "
                              f"{t1['media']} y pooled {t1['corrected_cv']}")
        if not dato["control_t0"]["reproducido"]:
            problemas.append(f"{cid}: el control de V20 no reproduce el canonico")
    n_controles = int((registro["experiment_id"].str.contains("-T0")).sum())
    if n_controles:
        problemas.append(f"experiments.csv contiene {n_controles} filas de control, que no son "
                         f"experimentos")
    return problemas


if __name__ == "__main__":
    raise SystemExit(main())
