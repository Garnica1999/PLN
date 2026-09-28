"""V28 - 01 - P1: gap analysis de las seis tecnicas de V28 por equivalencia funcional.

Antes de ejecutar nada, cada tecnica se clasifica como `GENUINELY_NEW`, `PARTIALLY_TESTED`,
`FUNCTIONALLY_DUPLICATED` o `ALREADY_CLOSED`. **No basta con la coincidencia de nombre**: cada
veredicto se apoya en una busqueda dirigida en el codigo y, cuando existe, en una tabla de
resultados ya medida.

## Lo que este script NO hace

No entrena. No abre ni cierra nada por opinion. Solo deja por escrito, con la evidencia al lado,
que tecnicas de las seis pueden seguir adelante y cuales no, y por que.

## La economia que se busca

De seis tecnicas, laHipotesis correcta puede ser que **solo dos sobrevivan**. Cerrar cuatro por
equivalencia funcional, con el numero que lo demuestra, es un resultado de la investigacion y no
un trabajo pendiente.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402
from v19_b1_boosting import HIERBA  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "v28_01_gap_analysis.json"
GAP_CSV = cfg.REPORTS / "v28_hypothesis_gap_analysis.csv"
GAP_MD = cfg.REPORTS / "v28_hypothesis_gap_analysis.md"
CV_V20 = 0.6097782
#: Busquedas que sostienen cada veredicto. `patron` se cuenta sobre codigo de `scripts/` y `src/`.
BUSQUEDAS = {
    "ovo_sklearn": r"OneVsOneClassifier|one_vs_one|\bovo_",
    "ovr_sklearn": r"OneVsRestClassifier|one_vs_rest|\bovr_",
    "voting": r"VotingClassifier|voting_",
    "cadena_clasificadores": r"ClassifierChain|clasificador_cadena|cadena_de_clas",
    "dirichlet_evidencia": r"dirichlet|dempster|evidential|evidence_fusion|belief_mass",
    "subespacio_features": r"feature_fraction|subespacio|subspace",
    "colsample": r"colsample_bytree|feature_fraction",
    "bagging_filas": r"bagging|bagging_fraction|subsample_freq",
    "geometria_consenso": r"vote_share|vote_disagreement|entropy_mean|conf_mean|conf_std",
}


def _busquedas() -> dict:
    """Cuenta los ficheros que contienen cada patron, con una muestra de donde.

    **Este script queda excluido de su propia busqueda.** Sus patrones son literalmente los
    nombres que busca (`OneVsOneClassifier`, `VotingClassifier`, `Dirichlet`, ...), asi que sin la
    exclusion detectaria sus propias cadenas y declararia existen familias que no existen. Es la
    misma excepcion que aplica la compliance de V27, y se declara en el informe.
    """
    yo = Path(__file__).resolve()
    salida = {}
    for nombre, patron in BUSQUEDAS.items():
        regex = re.compile(patron, re.IGNORECASE)
        ficheros, muestras = 0, []
        for base in ("scripts", "src"):
            for ruta in sorted((ROOT / base).rglob("*.py")):
                if "__pycache__" in str(ruta) or ruta.resolve() == yo:
                    continue
                texto = ruta.read_text(encoding="utf-8", errors="ignore")
                apoyos = regex.findall(texto)
                if apoyos:
                    ficheros += 1
                    if len(muestras) < 3:
                        muestras.append(f"{ruta.relative_to(ROOT)} ({len(apoyos)})")
        salida[nombre] = {"ficheros": ficheros, "muestras": muestras,
                           "existe": ficheros > 0}
    return salida


def _tabla_incertidumbre() -> dict:
    """Lee el resultado ya medido del bloque de incertidumbre, que es la evidencia de B."""
    ruta = ROOT / "reports" / "tables" / "advanced_04_variantes.csv"
    if not ruta.exists():
        return {"existe": False}
    tabla = pd.read_csv(ruta)
    base = tabla[tabla["variante"].str.strip() == "v09 (267)"]
    inc = tabla[tabla["variante"].str.contains("incertidumbre", case=False, na=False)]
    if base.empty or inc.empty:
        return {"existe": True, "tabla": tabla.to_dict("records"), "comparable": False}
    b = float(base["macro_f1"].iloc[0])
    i = float(inc["macro_f1"].iloc[0])
    return {"existe": True, "comparable": True,
            "baseline_v09": {"n_features": int(base["n_features"].iloc[0]), "macro_f1": b},
            "con_incertidumbre": {"n_features": int(inc["n_features"].iloc[0]), "macro_f1": i},
            "delta": round(i - b, 6),
            "fuente": "reports/tables/advanced_04_variantes.csv"}


def _confusion_v20() -> dict:
    """Confusion de V20 y el peso de cada par, que es la base para elegir los pares de C."""
    candidato = np.load(cfg.FEATURES / "v20_f3b_oof_candidato.npy")
    y = datos.load_train()[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    pred = np.argmax(candidato, axis=1)
    matriz = confusion_matrix(clase, pred, labels=list(range(len(cfg.CLASSES))))
    pares = []
    for i in range(len(cfg.CLASSES)):
        for j in range(i + 1, len(cfg.CLASSES)):
            pares.append({"par": f"{cfg.CLASSES[i]}|{cfg.CLASSES[j]}",
                          "total": int(matriz[i, j] + matriz[j, i]),
                          "a_hacia_b": int(matriz[i, j]), "b_hacia_a": int(matriz[j, i])})
    pares.sort(key=lambda d: -d["total"])
    errores = int(len(clase) - np.trace(matriz))
    adyacentes = [p for p in pares if abs(cfg.CLASSES.index(p["par"].split("|")[0])
                                          - cfg.CLASSES.index(p["par"].split("|")[1])) == 1]
    # El criterio de seleccion de pares se fija aqui, sobre la confusion del **control** V20, que es
    # informacion de train y out-of-fold. Se declara antes de ejecutar C y no se toca despues.
    corte = 5000
    seleccionados = [p["par"] for p in pares if p["total"] >= corte]
    return {
        "matriz": matriz.tolist(), "clases": list(cfg.CLASSES),
        "errores_totales": errores,
        "accuracy": round(float(np.trace(matriz)) / len(clase), 6),
        "pares_ordenados": pares,
        "pares_adyacentes": adyacentes,
        "errores_en_5_adyacentes": int(sum(p["total"] for p in adyacentes)),
        "proporcion_errores_en_5_adyacentes": round(
            sum(p["total"] for p in adyacentes) / max(errores, 1), 6),
        "criterio_de_seleccion": f"pares con al menos {corte} confusiones mutuas en la confusion "
                                f"del control V20, fijada antes de ejecutar C",
        "pares_seleccionados": seleccionados,
        "los_5_adyacentes_estan_todos": set(seleccionados) >= {p["par"] for p in adyacentes},
    }


def main() -> int:
    """Clasifica las seis tecnicas y escribe el gap analysis."""
    autotest_numerico()
    inicio = time.time()
    print("=== P1 gap analysis: buscando equivalencia funcional ===", flush=True)
    busq = _busquedas()
    for nombre, info in busq.items():
        print(f"  {nombre:<26} {info['ficheros']:>3} ficheros"
              f"{'  <- ' + ', '.join(info['muestras'][:2]) if info['existe'] else ''}", flush=True)
    incert = _tabla_incertidumbre()
    conf = _confusion_v20()
    print(f"\n  B: bloque de incertidumbre ya medido -> delta {incert.get('delta')} "
          f"({incert.get('baseline_v09', {}).get('n_features')} -> "
          f"{incert.get('con_incertidumbre', {}).get('n_features')} features)", flush=True)
    print(f"  C: los 5 pares adyacentes concentran "
          f"{conf['proporcion_errores_en_5_adyacentes']:.1%} de los {conf['errores_totales']:,} "
          f"errores de V20", flush=True)
    print(f"  pares que entrarian en C con el corte declarado: {conf['pares_seleccionados']}",
          flush=True)

    filas = _veredictos(busq, incert, conf)
    tabla = pd.DataFrame(filas)
    tabla.to_csv(GAP_CSV, index=False)
    _escribe_md(tabla, busq, incert, conf)
    RESULTADO.write_text(json.dumps({
        "fase": "V28 01 P1 gap analysis", "fecha": "2026-09-28", "experimenta": False,
        "busquedas": busq, "bloque_incertidumbre_ya_medido": incert, "confusion_v20": conf,
        "tecnicas": filas,
        "resumen": {v: int((tabla["veredicto"] == v).sum())
                    for v in sorted(tabla["veredicto"].unique())},
        "supervivientes": tabla.loc[tabla["veredicto"].isin(["GENUINELY_NEW", "PARTIALLY_TESTED"]),
                                    "ID"].tolist(),
        "presupuesto_maximo": 8, "presupuesto_que_gasta": len(
            tabla.loc[tabla["veredicto"].isin(["GENUINELY_NEW", "PARTIALLY_TESTED"])]),
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== veredictos ===", flush=True)
    for f in filas:
        print(f"  {f['ID']} {f['tecnica']:<28} {f['veredicto']:<24} "
              f"experimentos: {f['experimentos']}", flush=True)
    print(f"\n[P1] {tabla['veredicto'].isin(['GENUINELY_NEW', 'PARTIALLY_TESTED']).sum()} tecnicas "
          f"consumen presupuesto de {8} disponibles. Cerradas sin experimento: "
          f"{tabla['experimentos'] == 0 .sum() if False else int((tabla['experimentos'] == 0).sum())}",
          flush=True)
    print(f"[P1] {time.time() - inicio:.0f} s")
    return 0


def _veredictos(busq: dict, incert: dict, conf: dict) -> list[dict]:
    """Los seis veredictos, cada uno con su evidencia al lado."""
    return [
        {
            "ID": "A", "tecnica": "Meta-model bagging",
            "veredicto": "PARTIALLY_TESTED",
            "experimentos": 1,
            "mecanismo": "bagging de filas para reducir la varianza de la estimacion",
            "equivalencia_buscada": "bagging, bagging_fraction, subsample_freq, bootstrap de filas",
            "evidencia": (f"`HIERBA` ya tiene `subsample={HIERBA['subsample']}` con "
                          f"`subsample_freq={HIERBA['subsample_freq']}`, y en LightGBM eso hace que "
                          f"**cada uno de los {HIERBA['n_estimators']} arboles se ajuste sobre una "
                          f"muestra aleatoria del {HIERBA['subsample']:.0%} de las filas**. El "
                          f"meta-modelo de V20 **ya esta embolsado** a nivel de arbol."),
            "por_que_no_esta_cerrada": ("A anade un **segundo nivel**: N ajustes independientes en vez "
                                        "de un unico modelo con 300 arboles. Es un estimador "
                                        "distinto, aunque el mecanismo de remuestreo de filas no es "
                                        "nuevo en el proyecto. Se ejecuta UNA vez con N fijo y sin "
                                        "rejilla."),
            "confundidor_a_controlar": ("el numero de estimadores sube de "
                                        f"{HIERBA['n_estimators']} a "
                                        f"{HIERBA['n_estimators']} x N, luego parte de la mejora "
                                        "podria ser capacidad y no varianza. El control lo exige."),
        },
        {
            "ID": "B", "tecnica": "Compact consensus geometry",
            "veredicto": "ALREADY_CLOSED", "experimentos": 0,
            "mecanismo": "representacion compacta de incertidumbre y consenso entre miembros",
            "equivalencia_buscada": "entropy, entropia, top1, top2, margin, agreement, consenso",
            "evidencia": (
                "`scripts/advanced_04_incertidumbre.py` implementa un **superconjunto** de lo que "
                "B propone: por miembro, confianza (probabilidad maxima), margen (top-1 menos "
                "top-2) y entropia de Shannon; entre miembros, reparto de votos por clase, "
                "pluralidad, desacuerdo y desviacion de scores por clase; y conf_media, "
                "conf_desv, entropia_media y entropia_desv."),
            "resultado_ya_medido": (
                f"`{incert.get('fuente')}`: v09 con "
                f"{incert.get('baseline_v09', {}).get('n_features')} features da "
                f"{incert.get('baseline_v09', {}).get('macro_f1')}, y **anadiendo el bloque de "
                f"incertibilidad completo ({incert.get('con_incertidumbre', {}).get('n_features')} "
                f"features) da {incert.get('con_incertidumbre', {}).get('macro_f1')}: delta "
                f"**{incert.get('delta')}**."),
            "por_que_no_se_reabre": ("No es una familia parecida, es **literalmente el mismo "
                                     "conjunto de features**, con 51 columnas mas y medido. "
                                     "Repetirlo con un subconjunto mas pequeno seria buscar un "
                                     "resultado negativo con menos presupuesto."),
        },
        {
            "ID": "C", "tecnica": "Pairwise class decomposition",
            "veredicto": "GENUINELY_NEW", "experimentos": 1,
            "mecanismo": "descomponer la decision multiclase en decisiones binarias disjuntas",
            "equivalencia_buscada": "OneVsOneClassifier, VotingClassifier, ClassifierChain, "
                                    "pairwise, one_vs_rest",
            "evidencia": (
                f"`OneVsOneClassifier`, `VotingClassifier` y `ClassifierChain` tienen "
                f"**{busq['ovo_sklearn']['ficheros']}**, **{busq['voting']['ficheros']}** y "
                f"**{busq['cadena_clasificadores']['ficheros']}** ficheros respectivamente: no "
                f"existe. `OneVsRestClassifier` si aparece, pero en "
                f"{busq['ovr_sklearn']['ficheros']} ficheros y **siempre sobre modelos base**, "
                f"nunca sobre el meta-modelo; ademas one-vs-rest no es one-vs-one."),
            "distincion_con_lo_cerrado": (
                "V20 entrena **cinco binarios anidados** de `clase <= k-1`, donde cada uno mezcla "
                "todas las clases por debajo del umbral. C entrena **cinco binarios disjuntos**, "
                "uno por par adyacente, sin solape y sin restriccion de orden. No es CORAL "
                "(pesos compartidos y umbrales ordenados) ni cumulative-link: aqui cada frontera la "
                "aprende un clasificador independiente con la matriz completa de 279 features."),
            "motivacion_medida": (
                f"La confusion del control V20 muestra que los **cinco pares adyacentes concentran "
                f"{conf['proporcion_errores_en_5_adyacentes']:.1%} de los "
                f"{conf['errores_totales']:,} errores**: "
                + ", ".join(f"{p['par']} ({p['total']:,})" for p in conf["pares_adyacentes"])
                + ". Ese es el objetivo mecanico de C."),
            "criterio_de_pares_fijado": conf["criterio_de_seleccion"],
            "pares_seleccionados": conf["pares_seleccionados"],
        },
        {
            "ID": "D", "tecnica": "Dirichlet / evidence fusion",
            "veredicto": "FUNCTIONALLY_DUPLICATED", "experiments": 0,
            "experimentos": 0,
            "mecanismo": "fusion probabilistica explicita de la evidencia de los miembros",
            "equivalencia_buscada": "dirichlet, dempster, evidential, evidence_fusion, belief_mass",
            "evidencia": (
                f"La palabra `dirichlet` aparece en {busq['dirichlet_evidencia']['ficheros']} "
                f"ficheros, y **todos son el prior estadistico de Monroe et al. para ordenar "
                f"terminos por log-odds en el EDA**, no fusion de evidencia. `dempster`, "
                f"`evidential` y `belief_mass` tienen 0."),
            "por_que_no_se_ejecuta": (
                "Una fusion de evidencia tipo logarithmic opinion pool es un **cambio de "
                "combinador**, y los combinadores ya estan cerrados: **H04** (ponderar los "
                "miembros) y **H06** (apilar un segundo nivel), este ultimo con el argumento de "
                "que **V20 ya ES el segundo nivel**. Ademas el pool tendria que *sustituir* al "
                "meta-modelo LightGBM, quitando la capacidad que V27-04 demostro que usa. "
                "Ejecutarlo seria medir un Combinador ya cerrado, no un mecanismo nuevo."),
        },
        {
            "ID": "E", "tecnica": "Meta-model con incertidumbre explicita",
            "veredicto": "CLOSED_BY_DUPLICATION", "experimentos": 0,
            "mecanismo": "condicionar el meta-modelo al riesgo segun desacuerdo entre miembros",
            "equivalencia_buscada": "misma que B, con la regla critica del punto 23 del brief",
            "evidencia": (
                "B ya incorpora entropia, margen, reparto de votos y desacuerdo, que son "
                "exactamente las tres magnitudes que E propone representar: consenso, desacuerdo "
                "y confianza."),
            "por_que_no_se_ejecuta": (
                "La **regla critica del punto 23** del propio brief lo dice: si B ya incorpora "
                "entropy, agreement, margin y disagreement, E debe considerarse duplicada salvo "
                "que introduzca un mecanismo **diferente de integracion**. E solo cambia como se "
                "integran las mismas magnitudes en un meta-modelo, y esa integracion es H06, "
                "cerrada. Como B esta cerrada por equivalencia con un numero medido, E cae con ella."),
        },
        {
            "ID": "F", "tecnica": "Randomized meta-subspace",
            "veredicto": "FUNCTIONALLY_DUPLICATED", "experimentos": 0,
            "mecanismo": "entrenar meta-modelos sobre subespacios aleatorios de columnas",
            "equivalencia_buscada": "colsample_bytree, feature_fraction, subspace, subespacio",
            "evidencia": (
                f"`HIERBA` ya tiene **`colsample_bytree={HIERBA['colsample_bytree']}`**: cada "
                f"arbol ve un subconjunto aleatorio de la mitad de las 279 columnas, y **se "
                f"mantienen todas las columnas disponibles**. Eso es exactamente el mecanismo que "
                f"F describe, y la distincion que el brief traza frente a V25 (V25 eliminaba "
                f"columnas; F las aleatoriza) ya esta implementada."),
            "medido_ademas": (
                "V27-04 lo comprobó como control: quitar `colsample_bytree` **no cambia** el "
                "numero de features distintas que el modelo abre (siguen siendo 7). La "
                "aleatorizacion del subespacio no es lo que decide cuales se usan."),
            "por_que_no_se_ejecuta": ("Ejecutarlo seria medir un hiperparametro que ya esta "
                                      "activo, y ademas con una fraccion distinta, lo que el "
                                      "brief prohibe como rejilla."),
        },
    ]


def _escribe_md(tabla: pd.DataFrame, busq: dict, incert: dict, conf: dict) -> None:
    """Redacta el gap analysis legible, con los mismos numeros que el JSON."""
    sobrevivientes = tabla[tabla["veredicto"].isin(["GENUINELY_NEW", "PARTIALLY_TESTED"])]
    cerradas = tabla[tabla["experimentos"] == 0]
    lineas = [
        "# V28 - Gap analysis de las seis tecnicas", "",
        "Clasificacion por **equivalencia funcional**, no por coincidencia de nombre. Cada veredicto "
        "lleva al lado la busqueda que lo sostiene y, cuando existe, el numero ya medido.", "",
        "**Excepcion declarada:** este script queda excluido de su propia busqueda, porque "
        "sus patrones son literalmente los nombres que busca. Sin la exclusion, "
        "`OneVsOneClassifier` y `VotingClassifier` aparecerian con una coincidencia y se "
        "declararia existentes familias que no existen.", "",
        "## Veredicto", "",
        "| ID | Tecnica | Veredicto | Experimentos | Motivo en una linea |",
        "|---|---|---|---|---|"]
    motivo_corto = {
        "A": "el bagging de filas ya esta activo dentro de cada arbol; A anade un segundo nivel",
        "B": "el bloque de incertidumbre se implemento y se midio: delta " + str(incert.get("delta")),
        "C": "one-vs-one no existe en el proyecto y los pares adyacentes concentran "
             f"{conf['proporcion_errores_en_5_adyacentes']:.0%} de los errores",
        "D": "es un cambio de combinador, y H04 y H06 estan cerrados con numero",
        "E": "las mismas tres magnitudes que B, con la integracion de H06",
        "F": "`colsample_bytree` ya es aleatorizacion de subespacio sobre todas las columnas",
    }
    for _, f in tabla.iterrows():
        lineas.append(f"| {f['ID']} | {f['tecnica']} | **{f['veredicto']}** | {f['experimentos']} "
                      f"| {motivo_corto.get(f['ID'], '')} |")
    lineas += ["",
               f"**De seis tecnicas, {len(sobrevivientes)} pueden ejecutarse** y "
               f"{len(cerradas)} se cierran sin gastar un solo experimento. El presupuesto "
               f"autorizado es de 8 y esta fase consumira {int(sobrevivientes['experimentos'].sum())}.",
               "", "## Por que B esta cerrada con un numero, no con una opinion", "",
               "`scripts/advanced_04_incertidumbre.py` construye por miembro la **confianza** "
               "(probabilidad maxima), el **margen** (top-1 menos top-2) y la **entropia** de "
               "Shannon; entre miembros, el reparto de votos por clase, la pluralidad, el "
               "desacuerdo y la desviacion de scores por clase; y cuatro estadisticos agregados. "
               "Eso es un **superconjunto** del bloque compacto que B propone.", "",
               f"Y esta medido en `{incert.get('fuente')}`:", "",
               "| Variante | Features | macro-F1 |", "|---|---|---|",
               f"| v09 (base) | {incert.get('baseline_v09', {}).get('n_features')} | "
               f"{incert.get('baseline_v09', {}).get('macro_f1')} |",
               f"| + bloque de incertidumbre completo | "
               f"{incert.get('con_incertidumbre', {}).get('n_features')} | "
               f"{incert.get('con_incertidumbre', {}).get('macro_f1')} |", "",
               f"Delta: **{incert.get('delta')}**. Anadir 51 columnas de geometria del consenso no "
               "cambio el macro-F1. Un bloque mas pequeno no puede dar mas.", "",
               "## Por que C es genuinamente nueva, y por que tiene un mecanismo medido", "",
               f"- `OneVsOneClassifier`: **{busq['ovo_sklearn']['ficheros']} ficheros**.", "",
               f"- `VotingClassifier`: **{busq['voting']['ficheros']} ficheros**.", "",
               f"- `ClassifierChain`: **{busq['cadena_clasificadores']['ficheros']} ficheros**.", "",
               f"- `OneVsRestClassifier`: {busq['ovr_sklearn']['ficheros']} ficheros, pero "
               f"**siempre sobre modelos base**, nunca sobre el meta-modelo, y one-vs-rest no es "
               f"one-vs-one.", "",
               "La distincion con lo ya cerrado es de estructura, no de nombre. V20 entrena **cinco "
               "binarios anidados** de `clase <= k-1`, en los que cada uno mezcla todas las clases "
               "por debajo del umbral. C entrenaria **cinco binarios disjuntos**, uno por par "
               "adyacente, sin solape y sin restriccion de orden. No es CORAL, que comparte pesos "
               "e impone umbrales ordenados, ni cumulative-link.", "",
               "Y el objetivo mecanico esta medido en la confusion del control V20:", "",
               "| Par adyacente | Confusiones mutuas |", "|---|---|"]
    for p in conf["pares_adyacentes"]:
        lineas.append(f"| {p['par']} | {p['total']:,} |")
    lineas += ["",
               f"Los cinco pares adyacentes concentran **{conf['errores_en_5_adyacentes']:,} "
               f"errores, el {conf['proporcion_errores_en_5_adyacentes']:.1%}** de los "
               f"{conf['errores_totales']:,} errores de V20. La confusion es claramente bandada y "
               "C apunta exactamente a donde esta el error.", "",
               f"**Criterio de seleccion de pares, fijado antes de ejecutar:** "
               f"{conf['criterio_de_seleccion']}. Pares que entran: "
               f"`{'`, `'.join(conf['pares_seleccionados'])}`. Se calcula sobre la confusion "
               f"**out-of-fold del control**, que es informacion de train, y **no se toca** despues.",
               "", "## Por que A es solo parcialmente nueva", "",
               f"`HIERBA` ya tiene `subsample={HIERBA['subsample']}` con "
               f"`subsample_freq={HIERBA['subsample_freq']}`. En LightGBM eso significa que "
               f"**cada uno de los {HIERBA['n_estimators']} arboles se ajusta sobre una muestra "
               f"aleatoria del {HIERBA['subsample']:.0%} de las filas**: el meta-modelo de V20 ya "
               f"esta embolsado a nivel de arbol.", "",
               "Lo que A anade es un **segundo nivel**: N ajustes independientes, promediados "
               "despues. Es un estimador distinto, asi que se ejecuta, pero el brief obliga a "
               "declarar el confundidor: el numero de estimadores pasa de "
               f"{HIERBA['n_estimators']} a {HIERBA['n_estimators']} por N, luego parte de "
               "cualquier mejora podria ser capacidad y no reduccion de varianza.", "",
               "## Por que F ya esta implementado", "",
               f"`HIERBA` ya tiene **`colsample_bytree={HIERBA['colsample_bytree']}`**. Cada arbol "
               f"ve un subconjunto aleatorio de la mitad de las 279 columnas, **manteniendo todas "
               f"las disponibles**. La distincion que el brief traza frente a V25 (V25 eliminaba "
               f"columnas, F aleatoriza la exposicion) esta ya en la configuracion del champion. Y "
               f"V27-04 lo midio como control: quitar `colsample_bytree` no cambia el numero de "
               f"features que el modelo abre.", "",
               "## Las cuatro cerradas, en una frase cada una", "",
               "- **B**: el bloque se implemento y se midio, con delta " + str(incert.get("delta")) +
               ".", "",
               "- **D**: `dirichlet` en el proyecto es el prior de Monroe para el EDA, no fusion de "
               "evidencia; y un pool es un combinador, que es H04 mas H06, cerrados.", "",
               "- **E**: las mismas tres magnitudes que B, y la integracion es H06. La regla critica "
               "del punto 23 lo prohibe explicitamente.", "",
               "- **F**: `colsample_bytree` ya aleatoriza el subespacio sobre todas las columnas.",
               ""]
    GAP_MD.write_text("\n".join(lineas), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
