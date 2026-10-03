"""V28 - 04 - Cierre de la fase: trazabilidad, figuras maestras y auditoria.

El gap analysis dejo **dos** tecnicas ejecutables de seis. Las dos se han medido y las dos son
negativas. Este script consolida lo que paso, sin anadir ningun experimento.

## Que hace

1. **Anade las dos filas reales** de V28 a `experiments.csv` y a `reports/v28_master_results.csv`.
   No anade filas para B, D, E, F, que se cerraron por equivalencia funcional **sin entrenar**: el
   prompt lo prohibe explicitamente ("No crear filas ficticias para hypotheses cerradas por gap
   analysis").
2. **Genera las figuras maestras** de V28: corrected_cv por experimento, delta frente a V20,
   consistencia por fold, F1 por clase, rescate frente a dano y los diagnosticos de A y C.
3. **Audita** la cadena `experiments.csv -> tabla maestra -> artefactos` y falla si algo no cuadra.

## Lo que la fase cierra

- **A (bagging del meta-modelo)**: CLOSED, delta -0,000312. La varianza **existia** (los 5 miembros
  solo acordaban en el 91,9 % de las decisiones) y promediarla empeoro. La varianza de la estimacion
  del meta-modelo **no es donde esta el error**.
- **C (descomposicion pairwise)**: CLOSED, delta -0,067553. Los pares disjuntos **si** redujeron los
  errores de su objetivo local (-6.730 en los pares adyacentes) y **si** son complementarios en la
  masa dificil (aciertan 7.788 de las 61.949 filas que V19 y V20 fallan juntas), pero el precio es
  el colapso de las clases que no son frontera.
- **B, D, E, F**: cerradas por equivalencia funcional, sin consumir experimento.

**V20 sigue siendo el champion.** No se genera ninguna submission.
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

RESULTADO = cfg.EXPERIMENTS / "v28_04_cierre.json"
REGISTRO = ROOT / "experiments.csv"
MAESTRA = cfg.REPORTS / "v28_master_results.csv"
REGISTRO_H = cfg.REPORTS / "v28_hypothesis_registry.csv"
GAP = cfg.REPORTS / "v28_hypothesis_gap_analysis.csv"
FIG_MAESTRA = cfg.FIGURES / "v28_maestra_resultados.png"
FIG_FOLDS = cfg.FIGURES / "v28_maestra_folds.png"
CV_V20 = 0.6097782
LB_V20 = 0.62689
GATE_DELTA = 0.003
GATE_FOLDS = 4
FECHA = "2026-09-28"
#: Los dos experimentos realmente ejecutados, con su artefacto.
EJECUTADOS = {
    "V28-02": ("experiments/v28_02_tecnica_A_bagging.json", "A", "Meta-model bagging"),
    "V28-03": ("experiments/v28_03_tecnica_C_pairwise.json", "C", "Pairwise class decomposition"),
}
#: Fila de `experiments.csv` por experimento. Sin filas para B, D, E ni F.
FILAS = {
    "V28-02": {
        "bloque": "22",
        "etapa": "V28 tecnica A: bagging del meta-modelo sobre las mismas 279 features",
        "pregunta": ("la estimacion de la frontera tiene varianza residual que un ensemble de "
                     "ajustes independientes del meta-modelo reduciria?"),
        "hipotesis": ("embolsar el meta-modelo sin tocar el espacio de informacion reduce la "
                      "varianza de la frontera y mejora el macro-F1"),
        "features": "las 279 de V20, sin modificar y sin anadir ninguna",
        "preprocessing": ("bootstrap de Poisson(1) como sample_weight por miembro; semilla "
                          "derivada de cfg.SEED, miembro, fold y umbral"),
        "representacion": "identica a V20: no hay feature nueva",
        "modelo": (f"V20 exacto mas N={5} ajustes independientes, cada uno con los mismos "
                   f"300 estimadores de HIERBA"),
        "hiperparametros": ('{"n_miembros": 5, "agregacion": "promedio de probabilidades", '
                           '"remuestreo": "Poisson(1) sample_weight", "gate_delta": 0.003, '
                           '"gate_folds": 4}'),
        "validacion": ("StratifiedGroupKFold de 5 folds con las mismas agrupaciones que V20; el "
                       "control debe reproducir 0,6097782 o la fase se detiene"),
        "artefactos": "experiments/v28_02_tecnica_A_bagging.json | figuras v28_A_bagging_*",
    },
    "V28-03": {
        "bloque": "22",
        "etapa": "V28 tecnica C: descomposicion pairwise en 5 pares adyacentes disjuntos",
        "pregunta": ("un clasificador binario disjoint por par adyacente puede aprender fronteras "
                     "que el clasificador multiclase no representa con suficiente precision?"),
        "hipotesis": ("las confusiones dominantes de V20 son entre pares adyacentes, que concentran "
                      "el 59,9 % de sus 66.005 errores, y una descomposicion disjunta las ataca"),
        "features": "las 279 de V20, sin modificar; cada par usa las 279 enteras",
        "preprocessing": ("por par, solo las filas de sus dos clases, unas 56.000 frente a las "
                          "169.000 de V20: el coste mecanico de ser disjunta"),
        "representacion": "identica a V20",
        "modelo": ("cinco LightGBM binarios disjuntos, uno por par adyacente seleccionado, con "
                   "los mismos 300 estimadores de HIERBA"),
        "hiperparametros": ('{"pares": ["13-17|18-24", "18-24|25-34", "25-34|35-49", '
                           '"35-49|50-64", "50-64|65-xx"], "corte_de_seleccion": 5000, '
                           '"agregacion": "cadena de probabilidad independiente", '
                           '"gate_delta": 0.003, "gate_folds": 4}'),
        "validacion": ("StratifiedGroupKFold de 5 folds con las mismas agrupaciones que V20; el "
                       "control debe reproducir 0,6097782 o la fase se detiene"),
        "artefactos": "experiments/v28_03_tecnica_C_pairwise.json | figuras v28_C_pairwise_*",
    },
}


def _t1(artefacto: dict) -> dict:
    """El candidato tratado, que es el segundo y no el control."""
    return list(artefacto["candidatos"].values())[1]


def main() -> int:
    """Consolida V28 y audita."""
    autotest_numerico()
    inicio = time.time()
    datos = {eid: json.loads((ROOT / ruta).read_text(encoding="utf-8"))
             for eid, (ruta, _, _) in EJECUTADOS.items()}

    # ---------------------------------------------------------------- experiments.csv
    registro = pd.read_csv(REGISTRO)
    columnas = registro.columns.tolist()
    antes = len(registro)
    nuevas = []
    for eid, (_, tecnica, _) in EJECUTADOS.items():
        if (registro["experiment_id"] == eid).any():
            print(f"[csv] {eid} ya existe", flush=True)
            continue
        t1 = _t1(datos[eid])
        base = dict(FILAS[eid])
        base.pop("pregunta_abierta", None)
        base.update({
            "experiment_id": eid, "fecha": FECHA,
            "macro_f1_mean": t1["media"], "macro_f1_std": t1["desviacion"],
            "resultado": (f"control {datos[eid]['control_t0']['corrected_cv']} reproduce "
                          f"{datos[eid]['control_t0']['reproducido']} | T1 {t1['corrected_cv']} | "
                          f"delta {t1['delta_vs_V20']:+} | {t1['folds_positivos']}/5 folds | "
                          f"rescue {t1['rescue']} | damage {t1['damage']} | neto {t1['net']:+} | "
                          f"{t1['changed_rows']:,} filas cambian | "
                          f"IC95 {datos[eid]['gate']['bootstrap']['ic95']} | gate no cumplido"),
            "interpretacion": datos[eid].get("novedad") or "",
            "decision": datos[eid]["decision"], "duracion_seg": datos[eid].get("segundos"),
        })
        nuevas.append({c: base.get(c, "") for c in columnas})
    if nuevas:
        registro = pd.concat([registro, pd.DataFrame(nuevas)], ignore_index=True)
        registro.to_csv(REGISTRO, index=False)
    print(f"[csv] experiments.csv: {antes} -> {len(registro)} | {len(nuevas)} filas reales anadidas, "
          f"0 ficticias (B, D, E y F no generan fila: se cerraron sin entrenar)", flush=True)

    # ---------------------------------------------------------------- registry de hipotesis
    gap = pd.read_csv(GAP)
    filas_registry = []
    for _, f in gap.iterrows():
        hid = f["ID"]
        if hid in datos or hid in ("A", "C"):
            eid = next((e for e in EJECUTADOS if EJECUTADOS[e][1] == hid), None)
            if eid:
                t1 = _t1(datos[eid])
                estado = datos[eid]["estado"]
                resultado = (f"delta {t1['delta_vs_V20']:+.6f}, {t1['folds_positivos']}/5 folds, "
                             f"neto {t1['net']:+}, {t1['changed_rows']:,} filas")
                decision = datos[eid]["decision"]
            else:
                estado, resultado, decision = f["veredicto"], "no se midio", "cerrada sin experimento"
        else:
            estado, resultado, decision = f["veredicto"], "no se midio", "cerrada sin experimento"
        filas_registry.append({
            "ID": hid, "Tecnica": f["tecnica"], "Veredicto_gap_analysis": f["veredicto"],
            "Experimento": next((e for e in EJECUTADOS if EJECUTADOS[e][1] == hid), ""),
            "Status": estado, "Result": resultado, "Decision": decision,
            "Evidencia": str(f["evidencia"])[:200],
            "Criterio_de_cierre": str(f.get("por_que_no_se_ejecuta")
                                      or f.get("por_que_no_se_reabre")
                                      or f.get("por_que_no_esta_cerrada", ""))[:200]})
    pd.DataFrame(filas_registry).to_csv(REGISTRO_H, index=False)
    print(f"[registry] {len(filas_registry)} hipoticas registradas, "
          f"{sum(1 for r in filas_registry if r['Experimento'])} con experimento real", flush=True)

    _reconstruye_maestra(datos)
    _figuras(datos)
    problemas = _audita(registro, datos)
    RESULTADO.write_text(json.dumps({
        "fase": "V28 04 cierre", "fecha": FECHA, "experimenta": False,
        "tecnicas_ejecutadas": {k: {"delta": v["delta"], "decision": v["decision"],
                                    "estado": v["estado"]} for k, v in datos.items()},
        "tecnicas_cerradas_sin_experimento": ["B", "D", "E", "F"],
        "veredictos_gap_analysis": {f["ID"]: f["veredicto"] for _, f in gap.iterrows()},
        "presupuesto_autorizado": 8, "presupuesto_gastado": len(EJECUTADOS),
        "finalistas": 0, "submissions_generadas": 0,
        "baseline_vigente": CV_V20, "leaderboard_v20": LB_V20,
        "auditoria": problemas, "coherente": not problemas,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    if problemas:
        print("\n[AUDITORIA] DESAJUSTES:", flush=True)
        for p in problemas:
            print(f"  - {p}", flush=True)
        return 1
    print("[auditoria] experiments.csv <-> tabla maestra <-> artefactos: coherente", flush=True)
    print(f"\n[cierre] V20 sigue siendo el champion en {_dec_local(CV_V20)}. "
          f"0 submissiones. 0 finalistas.")
    print(f"[fin] {time.time() - inicio:.0f} s")
    return 0


def _dec_local(n: float, casas: int = 7) -> str:
    """float con coma decimal, convencion del informe."""
    return f"{n:.{casas}f}".replace(".", ",")


#: Esquema unico y fijo de la tabla maestra. Las columnas que solo tienen sentido para una tecnica
#: viven en su artefacto, no aqui: una tabla compartida con esquemas distintos por fila es
#: ilegible para pandas, y fue exactamente lo que paso antes de fijar esto.
ESQUEMA_MAESTRA = ["ID", "modelo", "corrected_cv", "delta_vs_V20", "fold_1", "fold_2", "fold_3",
                   "fold_4", "fold_5", "media", "desviacion", "folds_positivos", "rescue",
                   "damage", "net", "changed_rows", "frontera_1"]


def _reconstruye_maestra(datos: dict) -> None:
    """Reescribe `v28_master_results.csv` desde los artefactos, con el esquema canonico.

    Se reconstruye en vez de anadir, porque los scripts de A y C ya escribieron filas y sus
    esquemas no coincidian. Los artefactos son la fuente de verdad: la tabla se deriva de ellos.
    """
    filas = []
    for eid, dato in datos.items():
        candidatos = list(dato["candidatos"].values())
        for control, nombre in ((True, f"{EJECUTADOS[eid][1]}-T0"), (False, eid)):
            item = candidatos[0] if control else candidatos[1]
            filas.append({columna: item.get(columna, "") for columna in ESQUEMA_MAESTRA}
                        | {"ID": nombre, "modelo": item["modelo"]})
    tabla = pd.DataFrame(filas)[["ID"] + [c for c in ESQUEMA_MAESTRA if c != "ID"]]
    tabla.to_csv(MAESTRA, index=False)
    print(f"[maestra] {MAESTRA.name} reconstruida desde los artefactos: {len(tabla)} filas, "
          f"{len(ESQUEMA_MAESTRA)} columnas canonicas", flush=True)


def _figuras(datos: dict) -> None:
    """Figuras maestras de V28: corrected_cv, delta, folds, F1 y rescate/dano."""
    ids = list(datos)
    t1s = [_t1(datos[i]) for i in ids]
    etiquetas = {i: EJECUTADOS[i][2] for i in ids}
    colores = ["#08519c", "#a50f15"]
    fig, ejes = plt.subplots(2, 2, figsize=(15, 10))

    eje = ejes[0, 0]
    eje.bar(range(len(ids) + 1), [CV_V20] + [t["corrected_cv"] for t in t1s],
            color=["#333333"] + colores)
    eje.set_xticks(range(len(ids) + 1))
    eje.set_xticklabels(["V20"] + [f"{i}\n{etiquetas[i]}" for i in ids], fontsize=7)
    eje.axhline(CV_V20, color="black", linestyle="--", linewidth=1)
    eje.set_ylabel("corrected_cv")
    eje.set_ylim(min(CV_V20 - 0.08, min(t["corrected_cv"] for t in t1s) - 0.01), CV_V20 + 0.005)
    eje.set_title("corrected_cv por experimento (V20 congelado en la linea)", fontsize=10)

    eje = ejes[0, 1]
    eje.bar(range(len(ids)), [t["delta_vs_V20"] for t in t1s], color=colores)
    eje.axhline(0, color="grey", linewidth=1)
    eje.axhline(GATE_DELTA, color="black", linestyle="--", linewidth=1)
    eje.set_xticks(range(len(ids)))
    eje.set_xticklabels([f"{i}\n{etiquetas[i]}" for i in ids], fontsize=7)
    eje.set_ylabel("delta frente a V20")
    eje.set_title("delta frente a V20 (linea = gate de +0,003)", fontsize=10)
    for k, t in enumerate(t1s):
        eje.text(k, t["delta_vs_V20"] - 0.006, f"{t['delta_vs_V20']:+.6f}", ha="center", fontsize=7)

    eje = ejes[1, 0]
    for k, i in enumerate(ids):
        eje.plot(range(5), [t1s[k][f"fold_{j}"] for j in range(1, 6)], marker="o",
                 linewidth=1.3, color=colores[k], label=f"{i} {etiquetas[i]}")
    eje.plot(range(5), [CV_V20] * 5, linestyle="--", color="black", linewidth=1, label="V20 pool")
    eje.set_xticks(range(5))
    eje.set_xticklabels([f"fold {j}" for j in range(1, 6)])
    eje.set_ylabel("macro-F1 del fold")
    eje.set_title("consistencia por fold", fontsize=10)
    eje.legend(fontsize=7)

    eje = ejes[1, 1]
    ancho = 0.35
    for k, i in enumerate(ids):
        t = t1s[k]
        eje.bar(np.arange(len(ids)) + (k - 0.5) * ancho, [t["rescue"], t["damage"]], ancho,
                label=f"{i} rescue/damage", color=colores[k])
    eje.set_xticks(range(len(ids)))
    eje.set_xticklabels(ids, fontsize=8)
    eje.set_ylabel("filas")
    eje.set_title("rescate frente a dano", fontsize=10)
    eje.legend(fontsize=7)
    fig.suptitle("V28: las dos tecnicas genuinamente nuevas, ambas negativas. V20 sigue siendo el "
                 "champion", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_MAESTRA, dpi=130)
    plt.close(fig)

    fig2, ejes2 = plt.subplots(1, 2, figsize=(14, 5))
    eje = ejes2[0]
    clases = list(t1s[0]["f1_por_clase"].keys())
    x = np.arange(len(clases))
    control = list(datos[ids[0]]["candidatos"].values())[0]["f1_por_clase"]
    eje.bar(x - 0.26, [control[c] for c in clases], 0.26, label="V20", color="#333333")
    for k, i in enumerate(ids):
        eje.bar(x + (k - 0.5) * 0.26, [t1s[k]["f1_por_clase"][c] for c in clases], 0.26,
                label=etiquetas[i], color=colores[k])
    eje.set_xticks(x)
    eje.set_xticklabels(clases, fontsize=8)
    eje.set_ylabel("F1")
    eje.set_title("F1 por clase", fontsize=10)
    eje.legend(fontsize=7)

    eje = ejes2[1]
    a2 = datos[ids[0]]["control_de_varianza_A2"]
    eje.bar(["acuerdo entre\nlos 5 miembros", "acuerdo con\nV20"],
            [a2["acuerdo_medio_entre_miembros"], a2["acuerdo_de_prediccion_con_V20"]],
            color=["#08519c", "#999999"])
    eje.axhline(0.9, color="black", linestyle="--", linewidth=1)
    eje.set_ylim(0.85, 1.0)
    eje.set_ylabel("proporcion de coincidencia")
    eje.set_title("A2: la varianza existia y promediarla no ayudo", fontsize=10)
    fig2.suptitle("Diagnosticos especificos de A y C", fontsize=11)
    fig2.tight_layout()
    fig2.savefig(FIG_FOLDS, dpi=130)
    plt.close(fig2)


def _audita(registro: pd.DataFrame, datos: dict) -> list[str]:
    """Comprobaciones cruzadas entre experiments.csv, la tabla maestra y los artefactos."""
    problemas = []
    if registro["experiment_id"].duplicated().any():
        problemas.append("experiments.csv tiene identificadores duplicados")
    if not MAESTRA.exists():
        problemas.append("falta reports/v28_master_results.csv")
        return problemas
    maestra = pd.read_csv(MAESTRA)
    for eid, dato in datos.items():
        if not (registro["experiment_id"] == eid).any():
            problemas.append(f"{eid} no esta en experiments.csv")
        fila = maestra.loc[maestra["ID"] == eid]
        if fila.empty:
            problemas.append(f"{eid} no esta en v28_master_results.csv")
            continue
        t1 = _t1(dato)
        if abs(float(fila["corrected_cv"].iloc[0]) - t1["corrected_cv"]) > 1e-6:
            problemas.append(f"{eid}: la maestra dice {fila['corrected_cv'].iloc[0]} y el "
                             f"artefacto {t1['corrected_cv']}")
        if not dato["control_t0"]["reproducido"]:
            problemas.append(f"{eid}: el control de V20 no reproduce el canonico")
    # Ninguna fila ficticia: los identificadores de las tecnicas cerradas no pueden existir.
    for hid in ("V28-B", "V28-D", "V28-E", "V28-F"):
        if (registro["experiment_id"] == hid).any():
            problemas.append(f"{hid} no debe tener fila: se cerro sin entrenar")
    return problemas


if __name__ == "__main__":
    raise SystemExit(main())
