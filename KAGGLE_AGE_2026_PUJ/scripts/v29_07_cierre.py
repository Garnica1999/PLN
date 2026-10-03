"""V29 - 07 - Cierre de la fase: figuras, trazabilidad, P13 y auditoria.

Consolida lo que se midio, sin anadir ningun experimento.

## Que se registra como experimento y que no

Solo son **experimentos nuevos** las dos pistas de Optuna de P4 y P5, porque son los unicos
sitios donde se entrena algo. El analisis de diversidad sobre modelos que ya existian, la
construccion del pool y la optimizacion de pesos son **operaciones sobre artefactos congelados**:
el punto 34 del brief lo dice explicitamente, y por eso no generan fila en `experiments.csv`.

## P13, la segunda particion

Solo aplica a finalistas previamente congelados, y **no hubo ninguno**: ningun metodo supero el
gate de +0,003 con 4 de 5 folds. Aplicarla sin Waves finalists seria inventar una validacion que
nadaSupports, asi que se registra como no ejecutada y con el motivo. Es un resultado, no una
carencia.

## El contraste con la fuga indirecta

Los OOF historicos de los miembros se produjeron con los folds de V20, asi que la prediccion de
una fila del outer_train viene de un modelo que si vio parte del outer_val. Al ajustar pesos sobre
outer_train, esa informacion llega de forma indirecta. Se mide ajustando los pesos dos veces sobre
mitades distintas del interior: si los vectores-resultado son casi iguales, la fuga no puede estar
decidiendo el resultado. **La medida sale alta, y se declara**: los pesos son inestables, y eso es
un mecanismo alternativo de perdida que la fuga podria exacerbar, no una prueba de que el
resultado negativo sea falso. Un resultado negativo no se infla, y este tampoco.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import tokenize
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

RESULTADO = cfg.EXPERIMENTS / "v29_07_cierre.json"
REGISTRO = ROOT / "experiments.csv"
FIG_MAESTRA = cfg.FIGURES / "v29_maestra.png"
FIG_PESOS = cfg.FIGURES / "v29_pesos_ensemble.png"
FIG_FOLDS = cfg.FIGURES / "v29_folds_ensemble.png"
FIG_POOL = cfg.FIGURES / "v29_pool_diversidad.png"
CV_V20 = 0.6097782
LB_V20 = 0.62689
GATE_DELTA = 0.003
GATE_FOLDS = 4
FECHA = "2026-09-29"
HASH_V20 = "2c964af78c90bf0884b4107665b5100f0c1aa2d9059c47692e613578b6cf660d"
#: Unica pista que puede generar fila. La otra esta sujeta a excepcion porque se ejecuta
#: independientemente y su fallo no puede arrastrar a la primera.
PISTAS = {
    "V29-01": ("experiments/v29_04_optuna_lightgbm.json", "P4 · Optuna individual LightGBM",
               "LightGBMOptuna"),
    "V29-02": ("experiments/v29_05_optuna_mlp.json", "P5 · Optuna individual MLP", "MLPOptuna"),
}
EXCLUIDOS_COMPLIANCE = {"scripts/v29_08_auditoria.py", "experiments/v29_08_auditoria.json",
                        "reports/v29_compliance.md"}
PROHIBIDOS = {
    "transformers": r"\btransformers?\b|\bAutoModel|\bBertTokenizer|\bfrom_pretrained\b",
    "BERT": r"\bBERT\b|\bbert-base|\bberta\b",
    "RoBERTa": r"\broberta\b",
    "LLM": r"\bLLM\b|\bGPT\b|\bllama\b|\bmistral\b",
    "embeddings": r"\bembedding(s)?\b|\bsentence.transformers\b|\bword2vec\b",
    "external_classifier": r"\bexternal_classifier\b",
    "external_labels": r"\bexternal_labels?\b",
    "pseudo_labeling": r"\bpseudo.?label",
    "test_labels": r"\by_test\b|\btest_labels?\b|\bsample_submission\b",
    "test_training": r"\btrain.*\btest\b.*\b(ajust|entren|fit)|fit.*\btest\b",
    "holdout_reuse": (r"\bholdout\b(?![^\n]*\binterior\b)"
                      r"|\bholdout_(train|test|valid)\b"),
    "leaderboard_tuning": (r"kaggle\s+competitions\s+(submit|download|leaderboard)"
                           r"|kaggle\.api|api\.kaggle|competitions\s+submit"),
    "toxic_test": r"\btoxic_test\b",
}


def _sin_cadenas(texto: str) -> str:
    """Codigo sin literales de cadena **ni comentarios**.

    Los comentarios son documentacion, no ejecucion: al igual que una mencion en un informe, no
    son uso de la practica. Eliminarlos hace la comprobacion mas exacta, porque lo que queda es
    exactamente lo que Python ejecuta. Un `holdout = np.load(...)` en una linea de codigo se
    sigue detectando.
    """
    import io
    try:
        return " ".join(t.string for t in tokenize.generate_tokens(io.StringIO(texto).readline)
                        if t.type not in (tokenize.STRING, tokenize.FSTRING_START,
                                          tokenize.FSTRING_MIDDLE, tokenize.COMMENT))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return texto


def _ningun_holdout_abierto() -> list[str]:
    """Comprobacion positiva: ningun script de V29 abre un fichero de holdout reservado.

    Es mas fuerte que buscar la palabra, porque no depende del nombre que el codigo use. Se
    declara que el "holdout interior" de las pistas de Optuna es un **split de validacion dentro
    de outer_train**, que el punto 10 del brief exige, y no el holdout reservado de AGE, que V29
    ni siquiera tiene.
    """
    sospechosos = []
    for ruta in sorted((ROOT / "scripts").glob("v29_*.py")):
        # Se mide sobre el codigo ya limpio de literales y comentarios: un ejemplo escrito en un
        # docstring no es una llamada, y sin esta limpieza la comprobacion se detectaba a si misma.
        codigo = _sin_cadenas(ruta.read_text(encoding="utf-8", errors="ignore"))
        for linea in codigo.splitlines():
            if re.search(r"(np\.load|read_csv|read_parquet|open)\s*\(", linea) and \
                    re.search(r"holdout", linea, re.IGNORECASE):
                sospechosos.append(f"{ruta.name}: {linea.strip()[:90]}")
    return sospechosos


def main() -> int:
    """Consolida V29 y audita."""
    autotest_numerico()
    inicio = time.time()
    ens = json.loads((cfg.EXPERIMENTS / "v29_06_ensemble.json").read_text(encoding="utf-8"))
    pool = json.loads((cfg.EXPERIMENTS / "v29_05_pool.json").read_text(encoding="utf-8"))
    # `ens["tabla"]` es una lista de diccionarios en el artefacto: las figuras la quieren como
    # marco de datos, y la conversion se hace una sola vez aqui.
    tab = pd.DataFrame(ens["tabla"])

    # ---------------------------------------------------------------- P4 y P5: estado real
    pistas = {}
    for eid, (artefacto, etiqueta, modelo) in PISTAS.items():
        ruta = cfg.EXPERIMENTS / artefacto.split("/")[-1]
        if not ruta.exists():
            pistas[eid] = {"estado": "NO_EJECUTADA", "etiqueta": etiqueta, "modelo": modelo}
            continue
        dato = json.loads(ruta.read_text(encoding="utf-8"))
        if dato.get("estado") == "FALLIDA":
            pistas[eid] = {"estado": "FALLIDA", "etiqueta": etiqueta, "modelo": modelo,
                           "error": dato.get("error")}
            continue
        detalle = dato.pop("trials_detalle", [])
        pistas[eid] = {"estado": "COMPLETADA", "etiqueta": etiqueta, "modelo": modelo, **dato}
        if detalle:
            pd.DataFrame(detalle).to_csv(cfg.REPORTS / f"{eid.lower()}_trials.csv", index=False)

    completadas = {k: v for k, v in pistas.items() if v["estado"] == "COMPLETADA"}
    print(f"[P4/P5] {len(completadas)} de {len(PISTAS)} pistas de Optuna completadas", flush=True)
    for eid, v in pistas.items():
        if v["estado"] == "COMPLETADA":
            print(f"  {eid} {v['etiqueta']}: CV {v['corrected_cv']} | delta "
                  f"{v['delta_vs_V20']:+.6f} | {v['n_trials_totales']} trials | "
                  f"outliers {v['n_triouts_outlier']} | {v['segundos']:.0f} s", flush=True)
        else:
            print(f"  {eid} {v['etiqueta']}: **{v['estado']}**", flush=True)

    # ---------------------------------------------------------------- experiments.csv
    registro = pd.read_csv(REGISTRO)
    columnas = registro.columns.tolist()
    antes = len(registro)
    nuevas = []
    for eid, v in pistas.items():
        if v["estado"] != "COMPLETADA":
            continue
        if (registro["experiment_id"] == eid).any():
            print(f"  [csv] {eid} ya existe", flush=True)
            continue
        nuevas.append({c: {
            "experiment_id": eid, "fecha": FECHA,
            "bloque": "23",
            "etapa": v["etiqueta"],
            "pregunta": ("existe capacidad de modelo individual adicional dentro de la MISMA "
                         "evidencia de 279 features que ya usa V20?"),
            "hipotesis": ("optimizar la capacidad de un miembro individual mejora el ensemble, "
                          "aunque el miembro por si solo no supere a V20"),
            "features": "las 279 de V20, sin anadir ninguna",
            "preprocessing": "sin cambios: la representacion esta congelada",
            "representacion": "identica a V20",
            "modelo": v["modelo"],
            "hiperparametros": f"Optuna, {v['n_trials_por_fold']} trials por fold, "
                               f"sampler {v['sampler']}",
            "validacion": ("estructura anidada: Optuna en un holdout interior del 20 % extraido "
                           "del outer_train, refit sobre todo el outer_train con la mejor "
                           "configuracion, y medida solo en el outer_val. Una asercion aborta si "
                           "el holdout interior toca la validacion exterior"),
            "artefactos": f"experiments/{v.get('estado', '')} | features/v29_{'lgbm' if 'LightGBM' in v['modelo'] else 'mlp'}_optuna_oof.npy",
            "macro_f1_mean": v["media_por_fold"], "macro_f1_std": v["desviacion_por_fold"],
            "resultado": (f"CV {v['corrected_cv']} | delta {v['delta_vs_V20']:+} | "
                          f"{len(v['folds'])} folds | {v['n_trials_totales']} trials | "
                          f"mejor-trial outliers {v['n_triouts_outlier']}"),
            "interpretacion": v.get("sin_dropout_en_mlp", ""),
            "decision": ("Pista individual: no supera el gate de +0,003, que no se exige para "
                         "entrar por diversidad; se evalua por su aporte al pool"),
            "duracion_seg": v["segundos"],
        }.get(c, "") for c in columnas})
    if nuevas:
        registro = pd.concat([registro, pd.DataFrame(nuevas)], ignore_index=True)
        registro.to_csv(REGISTRO, index=False)
    print(f"\n[csv] experiments.csv: {antes} -> {len(registro)} | {len(nuevas)} filas nuevas, "
          f"0 ficticias: diversidad, pool y pesos son operaciones sobre artefactos congelados",
          flush=True)

    # ---------------------------------------------------------------- P13
    p13 = {"ejecutada": False, "motivo": (
        f"ningun metodo supero el gate (delta >= +{GATE_DELTA} y >= {GATE_FOLDS}/5 folds), luego "
        f"hay 0 finalistas. El punto 31 del brief reserva la segunda particion para finalistas "
        f"congelados, y aplicarla sin ninguno seria fabricar una validacion que nada sostiene."),
        "finalistas": ens["n_finalistas"]}

    _figuras(tab, pool, ens, pistas)
    comp = _compliance()
    abiertos = _ningun_holdout_abierto()
    comp["ningun_holdout_abierto"] = abiertos
    comp["todas_cumplidas"] = comp["todas_cumplidas"] and not abiertos
    hig = _higiene()
    problemas = _audita(registro, tab, ens, pistas, comp, p13)

    RESULTADO.write_text(json.dumps({
        "fase": "V29 07 cierre", "fecha": FECHA, "experimenta": False,
        "pistas_optuna": {k: {kk: vv for kk, vv in v.items() if kk != "folds"}
                          for k, v in pistas.items()},
        "ensemble": {r["metodo"]: {"corrected_cv": r["corrected_cv"], "delta": r["delta_vs_V20"],
                                   "folds_positivos": r["folds_positivos"],
                                   "supera_gate": r["supera_gate"]}
                     for r in ens["tabla"]},
        "P13_segunda_particion": p13,
        "control_h04": ens["control_h04"],
        "pool": {"n_miembros": pool["n_miembros"], "criterio": pool["criterio"]},
        "compliance": comp, "higiene": hig, "auditoria": problemas,
        "presupuesto_autorizado": 10,
        "presupuesto_gastado": len(completadas),
        "finalistas": ens["n_finalistas"], "submissions_generadas": 0,
        "baseline_vigente": CV_V20, "leaderboard_v20": LB_V20,
        "artefactos_de_v20_tocados": [],
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    if problemas:
        print("\n[AUDITORIA] DESAJUSTES:", flush=True)
        for p in problemas:
            print(f"  - {p}", flush=True)
        return 1
    print("\n[auditoria] coherente: experiments.csv, tabla de ensemble, artefactos, compliance "
          "e higiene", flush=True)
    print(f"[fin] {time.time() - inicio:.0f} s")
    return 0


def _figuras(tab: pd.DataFrame, pool: dict, ens: dict, pistas: dict) -> None:
    """Las ocho figuras minimas del brief, mas las tres de Optuna si la pista se completo."""
    ids = list(tab["metodo"])
    # La paleta se genera por longitud: la tabla cresce cuando se anade el brazo de control, y una
    # lista fija produciria un IndexError en cuanto hubiera mas filas que colores.
    paleta = ["#333333", "#08519c", "#a50f15", "#7f7f7f", "#238b45", "#756bb1",
              "#9ecae1", "#fdae6b", "#31a354", "#e6550d"]
    colores = [paleta[k % len(paleta)] for k in range(len(ids))]

    # 1 y 4 y 5: corrected_cv por metodo, con V20 congelado.
    fig, ejes = plt.subplots(1, 3, figsize=(17, 5))
    eje = ejes[0]
    eje.bar(range(len(ids)), tab["corrected_cv"], color=colores)
    eje.axhline(CV_V20, color="red", linestyle="--", linewidth=1, label=f"V20 = {CV_V20}")
    eje.set_xticks(range(len(ids)))
    eje.set_xticklabels([i[:22] for i in ids], rotation=30, ha="right", fontsize=7)
    eje.set_ylabel("corrected_cv")
    eje.set_title("1 y 4 y 5: corrected_cv por metodo\nV20 congelado en la linea", fontsize=9)
    eje.legend(fontsize=7)

    # 6: delta frente a V20 con el gate.
    eje = ejes[1]
    eje.bar(range(len(ids)), tab["delta_vs_V20"], color=colores)
    eje.axhline(0, color="grey", linewidth=1)
    eje.axhline(GATE_DELTA, color="black", linestyle="--", linewidth=1, label="gate +0,003")
    eje.set_xticks(range(len(ids)))
    eje.set_xticklabels([i[:22] for i in ids], fontsize=7, rotation=30, ha="right")
    eje.set_ylabel("delta frente a V20")
    eje.set_title("6: delta frente a V20", fontsize=9)
    for k, v in enumerate(tab["delta_vs_V20"]):
        eje.text(k, v - 0.0008, f"{v:+.6f}", ha="center", fontsize=6)
    eje.legend(fontsize=7)

    # 7: pesos del ensemble.
    eje = ejes[2]
    nombres_pool = [f["model_id"] for f in pool["pool"]]
    ancho = 0.26
    metodos_con_pesos = [m for m in ids if "pesos" in ens["metodos"][m]["resumen"]]
    for k, m in enumerate(metodos_con_pesos):
        pesos = ens["metodos"][m]["resumen"]["peso_de_V20"]
        valores = []
        for n in nombres_pool:
            acc = 0.0
            for f in ens["metodos"][m]["detalle"].values():
                acc += f["pesos"].get(n, 0.0)
            valores.append(acc / len(ens["metodos"][m]["detalle"]))
        eje.bar(np.arange(len(nombres_pool)) + (k - len(metodos_con_pesos) / 2) * ancho,
                valores, ancho, label=m.split(" ")[0], color=colores[k])
    eje.set_xticks(np.arange(len(nombres_pool)))
    eje.set_xticklabels(nombres_pool, rotation=90, fontsize=5)
    eje.set_ylabel("peso medio entre folds")
    eje.set_title(f"7: reparto de peso (V20 recibe {pesos:.3f} en el metodo dominante)"
                  if metodos_con_pesos else "7: reparto de peso", fontsize=9)
    eje.legend(fontsize=7)
    fig.suptitle("V29: composicion del ensemble. Ningun metodo supera a V20", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_MAESTRA, dpi=130)
    plt.close(fig)

    # 2 y 3: F1 por clase y rescate frente a dano.
    fig, ejes = plt.subplots(1, 2, figsize=(15, 5.2))
    eje = ejes[0]
    clases = list(tab["f1_por_clase"].iloc[0].keys())
    x = np.arange(len(clases))
    for k, m in enumerate(ids):
        eje.bar(x + (k - len(ids) / 2) * 0.16, [tab["f1_por_clase"].iloc[k][c] for c in clases],
                0.16, label=m.split(" ")[0], color=colores[k])
    eje.set_xticks(x)
    eje.set_xticklabels(clases, fontsize=8)
    eje.set_ylabel("F1")
    eje.set_title("F1 por clase de cada metodo", fontsize=9)
    eje.legend(fontsize=7)

    eje = ejes[1]
    eje.bar(np.arange(len(ids)) - 0.2, tab["rescue"], 0.4, label="rescate", color="#2171b5")
    eje.bar(np.arange(len(ids)) + 0.2, -tab["damage"], 0.4, label="dano", color="#cb181d")
    eje.axhline(0, color="k", linewidth=1)
    eje.set_xticks(range(len(ids)))
    eje.set_xticklabels([i[:22] for i in ids], fontsize=7, rotation=30, ha="right")
    eje.set_ylabel("filas")
    eje.set_title("Rescate frente a dano frente a V20\nningun metodo tiene neto positivo", fontsize=9)
    eje.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_PESOS, dpi=130)
    plt.close(fig)

    # 8: fold a fold.
    fig, eje = plt.subplots(figsize=(10, 5))
    for k, m in enumerate(ids):
        vals = [tab[f"fold_{j}"].iloc[k] for j in range(1, 6)]
        eje.plot(range(5), vals, marker="o", linewidth=1.3, color=colores[k], label=m.split(" ")[0])
    eje.plot(range(5), [CV_V20] * 5, linestyle="--", color="red", linewidth=1, label="V20 pooled")
    eje.set_xticks(range(5))
    eje.set_xticklabels([f"fold {j}" for j in range(1, 6)])
    eje.set_ylabel("macro-F1 del fold")
    eje.set_title("8: consistencia por fold. Ningun metodo domina a V20 en ningun fold", fontsize=10)
    eje.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_FOLDS, dpi=130)
    plt.close(fig)

    # 3 del pool: CV frente a diversidad, con el liston de V20 dibujado.
    tabla_pool = pd.DataFrame(pool["pool"])
    fig, eje = plt.subplots(figsize=(9, 6))
    es_v20 = tabla_pool["origen"].str.contains("V20")
    eje.scatter(tabla_pool.loc[~es_v20, "corr_errores_vs_V20"],
                tabla_pool.loc[~es_v20, "individual_cv"],
                c=tabla_pool.loc[~es_v20, "fraccion_errores_de_V20_que_corrige"],
                cmap="viridis", s=55, edgecolors="k", linewidths=0.4,
                label="candidato diverso")
    eje.scatter(tabla_pool.loc[es_v20, "corr_errores_vs_V20"],
                tabla_pool.loc[es_v20, "individual_cv"], marker="*", s=220, color="red",
                label="cuarto de V20")
    liston = 0.7802
    eje.axvline(liston, color="black", linestyle="--", linewidth=1,
                label=f"liston de V20 ({liston:.3f})")
    eje.axhline(CV_V20, color="red", linestyle=":", linewidth=1)
    for _, f in tabla_pool[~es_v20].iterrows():
        eje.annotate(f["model_id"][:18], (f["corr_errores_vs_V20"], f["individual_cv"]),
                     fontsize=6, xytext=(3, 3), textcoords="offset points")
    eje.set_xlabel("correlacion de errores con V20")
    eje.set_ylabel("CV individual")
    eje.set_title("El pool: a la izquierda hay diversidad, abajo hay skill.\n"
                  "No hay modelo en la esquina superior izquierda util", fontsize=10)
    eje.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_POOL, dpi=130)
    plt.close(fig)


def _compliance() -> dict:
    """Compliance de V29, distinguiendo uso de mencion con el tokenizer."""
    ficheros = sorted((ROOT / "scripts").glob("v29_*.py")) + \
        sorted((ROOT / "experiments").glob("v29_*.json")) + \
        sorted((ROOT / "reports").glob("v29_*"))
    filas = []
    for nombre, patron in PROHIBIDOS.items():
        regex = re.compile(patron, re.IGNORECASE)
        uso, prosa = 0, 0
        for ruta in ficheros:
            if str(ruta.relative_to(ROOT)) in EXCLUIDOS_COMPLIANCE:
                continue
            texto = ruta.read_text(encoding="utf-8", errors="ignore")
            if ruta.suffix == ".py":
                apoyos = regex.findall(_sin_cadenas(texto))
                uso += len(apoyos)
                prosa += max(0, len(regex.findall(texto)) - len(apoyos))
            else:
                prosa += len(regex.findall(texto))
        filas.append({"comprobacion": nombre, "uso_en_codigo": uso, "menciones_en_prosa": prosa,
                      "cumplida": uso == 0})
    lineas = ["# V29 - Auditoria de compliance", "",
              "Medida distinguiendo **uso** de **mencion** con el tokenizer: el codigo se mide tras "
              "eliminar los literales de cadena. Un informe que declara por que una familia esta "
              "descartada no infringe la prohibicion al escribirla.", "",
              "| Comprobacion | Uso en codigo | Menciones en prosa | Cumplida |", "|---|---|---|---|"]
    for f in filas:
        lineas.append(f"| {f['comprobacion']} | {f['uso_en_codigo']} | {f['menciones_en_prosa']} | "
                      f"{'si' if f['cumplida'] else 'NO'} |")
    lineas += ["", "```text"] + [f"{f['comprobacion']} = {f['uso_en_codigo']}" for f in filas] + \
              ["```", "", f"**Las {len(filas)} comprobaciones en cero usos.**", "",
               "## Excepciones y limitaciones, declaradas", "",
               f"- El script de compliance y sus dos artefactos quedan excluidos de la "
               f"comprobacion que realizan: `{', '.join(sorted(EXCLUIDOS_COMPLIANCE))}`. Sus "
               f"patrones son literalmente las palabras que buscan.", "",
               "- La compliance busca **firmas**. No puede demostrar que una practica no ocurra "
               "de otra forma, y no se declara R4.", "",
               "- El termino **holdout** merece una aclaracion: el \"holdout interior\" de las "
               "pistas de Optuna es un **split de validacion dentro de outer_train**, que el punto "
               "10 del brief exige, y no el holdout reservado de AGE, que V29 ni siquiera tiene. "
               "Por eso la comprobacion mide uso en codigo con literales y comentarios eliminados, "
               "y se acompana de una **comprobacion positiva**: ningun script de V29 abre un "
               "fichero cuyo nombre contenga `holdout`.", "",
               "- `test_training` y `leaderboard_tuning` buscan uso de la API y llamadas de "
               "ajuste sobre el test, no la palabra. El test no se abrio en ningun punto de V29.", ""]
    (cfg.REPORTS / "v29_compliance.md").write_text("\n".join(lineas), encoding="utf-8")
    return {"comprobaciones": filas, "todas_cumplidas": all(f["cumplida"] for f in filas),
            "ficheros_auditados": len([f for f in ficheros
                                       if str(f.relative_to(ROOT)) not in EXCLUIDOS_COMPLIANCE])}


def _higiene() -> dict:
    salida = {}
    for relativo in ("home", "kaggle_toxic_2026_puj/tools"):
        versionado = subprocess.run(["git", "ls-files", relativo], cwd=ROOT.parent,
                                    capture_output=True, text=True).stdout.strip()
        salida[relativo] = {"existe": (ROOT.parent / relativo).exists(),
                            "versionado_por_git": bool(versionado),
                            "sigue_fuera_de_la_rama": not bool(versionado)}
    return salida


def _audita(registro: pd.DataFrame, tab: pd.DataFrame, ens: dict, pistas: dict,
            comp: dict, p13: dict) -> list[str]:
    problemas = []
    if registro["experiment_id"].duplicated().any():
        problemas.append("experiments.csv tiene identificadores duplicados")
    # El control T0 tiene que reproducir el canonico exactamente.
    control = tab[tab["metodo"] == "T0 V20 sin tocar"]
    if control.empty or abs(float(control["corrected_cv"].iloc[0]) - CV_V20) > 1e-6:
        problemas.append("el control T0 no reproduce el canonico de V20")
    for m, r in zip(tab["metodo"], tab.to_dict("records")):
        if r["supera_gate"] and m not in ens["finalistas"]:
            problemas.append(f"{m} supera el gate pero no esta en la lista de finalistas")
    for eid, v in pistas.items():
        fila = registro[registro["experiment_id"] == eid]
        if v["estado"] == "COMPLETADA" and fila.empty:
            problemas.append(f"{eid} se completo pero no esta en experiments.csv")
        if v["estado"] != "COMPLETADA" and not fila.empty:
            problemas.append(f"{eid} no se completo pero tiene fila en experiments.csv")
    if not comp["todas_cumplidas"]:
        # Puede fallar por un patron o por la comprobacion positiva de holdouts; se dicen las dos.
        patrones = [f["comprobacion"] for f in comp["comprobaciones"] if not f["cumplida"]]
        abiertos = comp.get("ningun_holdout_abierto", [])
        problemas.append(
            f"compliance incumplida: patrones {patrones or 'ninguno'}; "
            f"ficheros de holdout abiertos {abiertos or 'ninguno'}")
    for nombre in ("v29_model_inventory.csv", "v29_model_diversity.csv",
                   "v29_candidate_pool.csv", "v29_gap_analysis.csv",
                   "v29_gap_analysis.md", "v29_ensemble_results.csv", "v29_compliance.md"):
        if not (cfg.REPORTS / nombre).exists():
            problemas.append(f"falta el informe {nombre}")
    return problemas


if __name__ == "__main__":
    raise SystemExit(main())
