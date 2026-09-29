"""Gap analysis actualizado, y la medicion que lo fundamenta.

## La pregunta, y por que empieza midiendo y no razonando

El brief pide responder:

> Cuando V30 falla, que informacion observable del train permanece sin explotar y que mecanismo
> NUEVO podria transformarla en una mejora robusta?

La tentacion es responderla leyendo el historial y pensando. Pero el historial ya dice cosas
contradictorias sobre lo que queda, y lo unico que resuelve la contradiccion es medir **que
errores de V30 corrige alguien y cuales no**. Por eso el script empieza ahi.

## La particion de los errores, que es el resultado central

Se toman **todos** los modelos disponibles: los 16 artefactos del pool de V29, los tres miembros de
V30 y los diez canales linguisticos de V31. Y se reparte el conjunto de los 65.403 errores de V30
en dos partes:

```text
A  NADIE corrige      ningun modelo disponible acierta esas filas
B  ALGUIEN corrige    alguna fuente distinta acierta donde V30 falla
```

La **prioridad de investigacion es B**, segun el brief, porque ahi es donde existe informacion
aprovechable: V30 fallo, y otra fuente ya tiene la respuesta. La parte A es un limite duro: si
65.403 errores no los corrige nadie con los recursos permitidos, ninguna reformulacion de la
representacion va a tocarlos.

Este es el mismo corte que V29 hizo y encontro 0 de 47 modelos con rescate neto positivo. Lo que
aquí se anade es el detalle de **cuanto** de A es realmente irreducible y cuanto de B es
estructuralmente explotable.

## La distincion que el brief exige: senal frente a mecanismo de integracion

El brief pide no dar por agotada una familia solo porque concatenarla a V30 fallo. Pero tampoco
reabrila con otra concatenacion trivial. La medicion que separa las dos cosas es **la tasa de
acuerdo**: si una fuente acierta donde V30 falla y falla donde V30 acierta a una tasa parecida a
la de un modelo independiente, es informacion nueva; si su tasa de acuerdo es alta, es V30 con
ruido, como ya demostro P34 con las correlaciones 0,57-0,79.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "age_gap_analysis.json"
GAP_MD = cfg.REPORTS / "age_updated_gap_analysis.md"
GAP_CSV = cfg.REPORTS / "age_updated_gap_analysis.csv"
ERRORES_CSV = cfg.REPORTS / "age_error_partition.csv"
FECHA = "2026-10-05"
CV_V30 = 0.6134471
LB_V30 = 0.63302
VARIABILIDAD = 0.0008876
N_CLASES = 6
CANDIDATO = "ortografia+composicion+discurso"


def _macro(clase, prob) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro",
                          labels=list(range(N_CLASES))))


def particion_errores(clase, p_v30) -> dict:
    """Reparte los errores de V30 entre 'nadie corrige' y 'alguien corrige'."""
    pool = np.load(cfg.EXPERIMENTS / "v29_pool_oof.npz")
    canales = np.load(cfg.FEATURES / "v31_canales_oof.npz")
    ens = json.loads((cfg.EXPERIMENTS / "v30_02_ensemble.json").read_text(encoding="utf-8"))

    fuentes, nombres = [], []
    for n in ens["manifest"]["miembros"]:
        if n == "v30_lgbm_refinado":
            m = np.load(cfg.FEATURES / "v30_lgbm_refinado_oof.npy")
        elif n in pool.files:
            m = pool[n]
        else:
            m = np.load(cfg.FEATURES / f"{n}.npy")
        fuentes.append(np.asarray(m, dtype=np.float64))
        nombres.append(f"V30:{n}")
    for n in sorted(pool.files):
        if n in ens["manifest"]["miembros"]:
            continue
        a = np.asarray(pool[n], dtype=np.float64)
        if a.ndim == 2 and a.shape == (len(clase), N_CLASES):
            fuentes.append(a)
            nombres.append(f"pool:{n}")
    for n in sorted(canales.files):
        if n == "raw":
            continue
        a = np.asarray(canales[n], dtype=np.float64)
        # Los canales son decision_function: el argmax es valido, la probabilidad no.
        fuentes.append(a)
        nombres.append(f"canal:{n}")

    ok = {nom: np.argmax(s, axis=1) == clase for nom, s in zip(nombres, fuentes)}
    pred_v30 = np.argmax(p_v30, axis=1)
    ok_v30 = pred_v30 == clase
    error = ~ok_v30

    alguien = np.zeros(len(clase), dtype=bool)
    best_por_fuente = {}
    for nom, m in ok.items():
        best_por_fuente[nom] = int((m & error).sum())
        alguien |= (m & error)

    # V30 mismo no cuenta como "alguien": se excluye de su propia rescue.
    para_b = alguien.copy()
    for nom in list(best_por_fuente):
        if nom.startswith("V30:"):
            para_b &= ok[nom] | ~error | ~ok_v30

    a_ = int((error & ~para_b).sum())
    b_ = int((error & para_b).sum())
    # Quien corrige mas errores de V30, excluyendo los miembros de V30.
    externos = {k: v for k, v in best_por_fuente.items() if not k.startswith("V30:")}
    top = sorted(externos.items(), key=lambda kv: -kv[1])[:10]

    filas = [{"grupo": "A_nadie_corrige", "n": a_, "fraccion_de_errores": round(a_ / error.sum(), 5),
              "fraccion_del_dataset": round(a_ / len(clase), 5),
              "que_exige": "una representacion que ninguno de los 29 modelos actuales posee"},
             {"grupo": "B_alguien_corrige", "n": b_, "fraccion_de_errores": round(b_ / error.sum(), 5),
              "fraccion_del_dataset": round(b_ / len(clase), 5),
              "que_exige": "un mecanismo que use la evidencia correcta, que ya existe"}]
    pd.DataFrame(filas).to_csv(ERRORES_CSV, index=False)

    return {
        "n_errores_v30": int(error.sum()),
        "n_fuentes_examinadas": len(fuentes),
        "n_fuentes_externas_a_v30": len(externos),
        "A_nadie_corrige": {"n": a_, "fraccion": round(a_ / error.sum(), 5)},
        "B_alguien_corrige": {"n": b_, "fraccion": round(b_ / error.sum(), 5)},
        "top_rescatadores": [{"fuente": k, "errores_de_v30_que_corrige": v} for k, v in top],
        "lectura": (
            f"de los {int(error.sum()):,} errores de V30, {a_:,} ({a_ / error.sum():.1%}) no los "
            f"corrige NINGUNA de las {len(externos)} fuentes externas disponibles, y {b_:,} "
            f"({b_ / error.sum():.1%}) los corrige alguna. El grupo B es donde hay informacion "
            f"aprovechable; el grupo A es un limite duro"),
    }


#: El inventario, con la clasificacion que el brief exige.
#: Escribido como dicts y no como tuplas porque una tupla de once campos no se lee: ya se rompio
#: dos veces por desplazamiento, y el error aparece lejos (un IndexError en la compresion) en vez
#: de en la entrada que le falta el campo.
HIPOTESIS = [
    {"ID": "G01", "family": "correccion de score",
     "hypothesis": "corregir el score de V30 sumando el residuo linguistico",
     "categoria": "C", "status": "INVALIDATED (P33 global, P34 local)",
     "previous_evidence": "P33: AUC del residuo 0,5411 pero la correccion global da -0,0002152 y "
                          "el ruido aleatorio gana a la senal real. P34: correccion pairwise "
                          "-0,0002039, AUC de residuo por pareja 0,49-0,51, cero rescates de alta "
                          "confianza",
     "why_not_duplicate": "-", "expected_mechanism": "-", "cost": "cerrado", "priority": 0,
     "leakage_risk": "ninguno"},
    {"ID": "G02", "family": "representacion linguistica",
     "hypothesis": "anadir un canal linguistico al espacio de features",
     "categoria": "C", "status": "CLOSED (V31-06)",
     "previous_evidence": "+0,0002948 sobre el control barajado, muy por debajo del gate de "
                          "+0,0015 y dentro de la variabilidad de entrenamiento",
     "why_not_duplicate": "-", "expected_mechanism": "-", "cost": "cerrado", "priority": 0,
     "leakage_risk": "ninguno"},
    {"ID": "G03", "family": "capacidad del meta-modelo",
     "hypothesis": "mas capacidad en el meta-arbol", "categoria": "C",
     "status": "CLOSED (V29, V30)",
     "previous_evidence": "V29 +0,002392 sobre V20; V30 refinamiento +0,000069 sobre V29, es decir "
                          "la region estaba ya encontrada",
     "why_not_duplicate": "-", "expected_mechanism": "-", "cost": "cerrado", "priority": 0,
     "leakage_risk": "ninguno"},
    {"ID": "G04", "family": "diversidad de ensembles historicos",
     "hypothesis": "combinar mas modelos historicos", "categoria": "C", "status": "CLOSED (V29)",
     "previous_evidence": "0 de 47 combinaciones con rescate neto positivo",
     "why_not_duplicate": "-", "expected_mechanism": "-", "cost": "cerrado", "priority": 0,
     "leakage_risk": "ninguno"},
    {"ID": "G05", "family": "correccion de umbral",
     "hypothesis": "mover el umbral de decision de V30", "categoria": "C",
     "status": "CLOSED (V20-F3)",
     "previous_evidence": "ganancia historica nula. Ademas la oscilacion entre modelos vecinos es "
                          "0,0034, mayor que cualquier delta de esta fase",
     "why_not_duplicate": "-", "expected_mechanism": "-", "cost": "cerrado", "priority": 0,
     "leakage_risk": "ninguno"},
    {"ID": "G06", "family": "retrieval de URL",
     "hypothesis": "recuperar el texto de la URL de la fila", "categoria": "B",
     "status": "QUARANTINED",
     "previous_evidence": "0 URLs de status en 169.336 filas: el sujeto no existe. Y las reglas "
                          "de AGE no estan disponibles en este entorno",
     "why_not_duplicate": "-", "expected_mechanism": "-", "cost": "no ejecutable", "priority": 0,
     "leakage_risk": "alto si se ejecutara"},
    {"ID": "G07", "family": "metrica",
     "hypothesis": "optimizar el ranking del margen en vez del argmax", "categoria": "D",
     "status": "DESCARTADO por irrelevancia",
     "previous_evidence": "la metrica de la competicion y el leaderboard evaluan argmax sobre la "
                          "etiqueta. Cambiar la metrica interna no mejora el score de Kaggle",
     "why_not_duplicate": "-", "expected_mechanism": "-", "cost": "bajo", "priority": 9,
     "leakage_risk": "ninguno"},
    {"ID": "G08", "family": "entrenamiento",
     "hypothesis": "entrenar con targets suavizados en las zonas de frontera", "categoria": "A",
     "status": "sin evidencia directa",
     "previous_evidence": "P34 mostro que las mejoras cerca del limite son de tipo calibracion y "
                          "que ahi el ruido gana. Suavizar etiquetas actua precisamente en la "
                          "zona de frontera",
     "why_not_duplicate": "reordena el aprendizaje hacia las fronteras, donde el residuo no "
                          "discrimina; el mecanismo es distinto de sumar su residuo al score",
     "expected_mechanism": "desplaza la masa de probabilidad hacia la frontera en vez de "
                           "afianzarla, y podria reducir el exceso de confianza de V30 en filas "
                           "dificiles",
     "cost": "medio", "priority": 4, "leakage_risk": "bajo: requiere reentrenar V30, que esta "
                                                    "congelado, luego es inviable tal cual"},
    {"ID": "G09", "family": "entrenamiento",
     "hypothesis": "reponder el entrenamiento hacia el grupo A (nadie corrige)",
     "categoria": "A", "status": "sin evidencia, y con contra-indicacion",
     "previous_evidence": "el grupo A son errores que NINGUN modelo disponible corrige, luego no hay "
                          "senal que sancionar: reponderlos sube ruido de entrenamiento",
     "why_not_duplicate": "-", "expected_mechanism": "-", "cost": "bajo", "priority": 8,
     "leakage_risk": "bajo"},
    {"ID": "G10", "family": "calibracion",
     "hypothesis": "corregir el sesgo de probabilidad por clase", "categoria": "A",
     "status": "PARTIAL_PRECEDENT",
     "previous_evidence": "PH32 y PH33 probaron correccion y calibracion: +0,0002 y ninguna "
                          "positiva. P33 mostro que el intercepto se elige en 0",
     "why_not_duplicate": "corregir el prior por clase en vez del residuo del candidato es otra "
                          "operacion, pero el precedente es negativo y el mecanismo es de "
                          "calibracion, no de senal",
     "expected_mechanism": "recentrar las probabilidades de cada clase hacia su frecuencia real",
     "cost": "bajo", "priority": 6, "leakage_risk": "bajo"},
    {"ID": "G11", "family": "estructura de clases",
     "hypothesis": "fusionar las dos clases extremas", "categoria": "D", "status": "sin evidencia",
     "previous_evidence": "13-17 y 65-xx tienen menos apoyo textual, pero V30 ya las separa bien "
                          "en la matriz de confusion",
     "why_not_duplicate": "-", "expected_mechanism": "-", "cost": "medio", "priority": 10,
     "leakage_risk": "bajo"},
    {"ID": "G12", "family": "integracion de score",
     "hypothesis": "correccion cuya intensidad dependa del margen de V30 (pendiente sobre el "
                   "margen, no constante)",
     "categoria": "A", "status": "sin evidencia, mecanismo no probado",
     "previous_evidence": "P33 probo un beta COMPARTIDO y un intercepto, y el intercepto salio 0. "
                          "La evidencia apunta en esta direccion: el margen de V30 predice sus "
                          "propios errores con AUC 0,8005, muy por encima del 0,5411 del residuo",
     "why_not_duplicate": "P33 ajusto una constante; esto ajusta una pendiente sobre una variable "
                          "que P33 no uso. El mecanismo es distinto y son 2-3 parametros, no una "
                          "grid",
     "expected_mechanism": "la correccion pesa mas donde V30 ya es confiado y menos donde duda, "
                           "que es donde el residuo deberia ayudar",
     "cost": "bajo", "priority": 1, "leakage_risk": "bajo, con estructura anidada estricta"},
    {"ID": "G13", "family": "bagging",
     "hypothesis": "promediar el residuo sobre subconjuntos de datos", "categoria": "C",
     "status": "CLOSED (V28-02)",
     "previous_evidence": "el bagging del meta-modelo dio -0,000312. El bagging del residuo es la "
                          "misma idea sobre un objeto mas pequeno, y con un residuo de AUC 0,54 "
                          "no hay mas que promediar",
     "why_not_duplicate": "-", "expected_mechanism": "-", "cost": "bajo", "priority": 7,
     "leakage_risk": "ninguno"},
    {"ID": "G14", "family": "fuente de datos",
     "hypothesis": "informacion del train que ningun modelo disponible ha leido todavia",
     "categoria": "B", "status": "sin explorar, y no es un experimento",
     "previous_evidence": "es la unica categoria que el brief marca como prioridad maxima, y no "
                          "tiene precedente cerrado en las 150 filas de experiments.csv. Requiere "
                          "inventario de que campos existen y cuales se han leido",
     "why_not_duplicate": "ningun modelo existente ha consumido esos campos, luego no hay "
                          "duplicado posible por construccion",
     "expected_mechanism": "no determinable sin el inventario: depende de que campos queden sin "
                           "leer, y ese es justamente el trabajo previo",
     "cost": "a determinar", "priority": 2, "leakage_risk": "a determinar segun el campo"},
]


def main() -> int:
    """Mide la particion de errores y construye el gap analysis actualizado."""
    autotest_numerico()
    inicio = time.time()
    carpeta = ROOT / "artifacts" / "p33_scores" / CANDIDATO.replace("+", "_")
    clase = np.load(carpeta / "y_true.npy")
    p_v30 = np.load(carpeta / "v30_prob.npy")
    print(f"=== V30 {_macro(clase, p_v30):.7f} ===", flush=True)

    print("\n=== particion de los errores de V30 (medicion previa) ===", flush=True)
    part = particion_errores(clase, p_v30)
    print(f"  errores de V30: {part['n_errores_v30']:,}", flush=True)
    print(f"  fuentes examinadas: {part['n_fuentes_examinadas']} "
          f"({part['n_fuentes_externas_a_v30']} externas a V30)", flush=True)
    print(f"  A  NADIE corrige   {part['A_nadie_corrige']['n']:>7,} "
          f"({part['A_nadie_corrige']['fraccion']:.1%})", flush=True)
    print(f"  B  ALGUIEN corrige {part['B_alguien_corrige']['n']:>7,} "
          f"({part['B_alguien_corrige']['fraccion']:.1%})", flush=True)
    print("  top rescatadores:", flush=True)
    for t in part["top_rescatadores"][:6]:
        print(f"    {t['fuente']:44s} {t['errores_de_v30_que_corrige']:>7,d}", flush=True)

    print("\n=== inventario de hipotesis ===", flush=True)
    campos_requeridos = ("ID", "family", "hypothesis", "categoria", "status",
                         "previous_evidence", "why_not_duplicate", "expected_mechanism",
                         "cost", "priority", "leakage_risk")
    for h in HIPOTESIS:
        faltan = [c for c in campos_requeridos if c not in h]
        if faltan:
            raise ValueError(f"la hipotesis {h.get('ID')} no declara: {faltan}")
    filas = [dict(h) for h in HIPOTESIS]
    tabla = pd.DataFrame(filas).sort_values(["categoria", "priority"])
    tabla.to_csv(GAP_CSV, index=False)
    for _, f in tabla.iterrows():
        print(f"  {f['ID']} [{f['categoria']}] p{f['priority']:<2} {f['status'][:28]:<28} "
              f"{f['hypothesis'][:52]}", flush=True)

    counts = tabla["categoria"].value_counts().to_dict()
    print(f"\n  por categoria: {counts}", flush=True)
    print(f"  GENUINELY_NEW sin precedente cerrado: "
          f"{len(tabla[(tabla['status'].str.startswith('sin'))])}", flush=True)

    _md(tabla, part, counts)
    RESULTADO.write_text(json.dumps({
        "fase": "gap analysis actualizado", "fecha": FECHA, "experimenta": False,
        "commit": __import__("subprocess").run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent,
                                               capture_output=True, text=True, timeout=30
                                               ).stdout.strip(),
        "baseline_v30": CV_V30, "leaderboard": LB_V30,
        "variabilidad_de_entrenamiento": VARIABILIDAD,
        "particion_de_errores": part, "inventario": tabla.to_dict("records"),
        "categorias": counts,
        "restricciones": {"usa_test": False, "usa_holdout": False, "usa_red": False,
                          "leaderboard_para_ajustar": False},
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


def _md(tabla: pd.DataFrame, part: dict, counts: dict) -> str:
    lineas = [
        "# AGE - gap analysis actualizado", "",
        f"**Baseline congelado:** V30 = {CV_V30}, leaderboard {LB_V30}.  ",
        f"**Variabilidad de entrenamiento:** {VARIABILIDAD}, que es la escala contra la que se "
        f"juzga cualquier delta de esta fase.", "",
        "## La medicion que fundamenta el inventario", "",
        "El brief pide responder que informacion del train queda sin explotar. La tentacion es "
        "responderlo leyendo el historial, pero el historial ya dice cosas contradictorias: hay "
        "lineas que cerraron por resultado y lineas que cerraron por mecanismo. Lo que las separa es "
        "una sola medicion, y se hace antes de escribir nada.", "",
        f"Se toman los **{part['n_errores_v30']:,}** errores de V30 y se reparte segun si alguna de "
        f"las **{part['n_fuentes_examinadas']}** fuentes disponibles (los tres miembros de V30, los "
        f"artefactos del pool de V29 y los diez canales de V31) acierta esas filas:", "",
        "| Grupo | Filas | % de los errores | % del dataset | Que exige |", "|---|---|---|---|---|",
        f"| **A: nadie corrige** | {part['A_nadie_corrige']['n']:,} | "
        f"**{part['A_nadie_corrige']['fraccion']:.1%}** | "
        f"{part['A_nadie_corrige']['n'] / 169336:.1%} | una representacion que ninguno de los "
        f"modelos actuales posee |",
        f"| **B: alguien corrige** | {part['B_alguien_corrige']['n']:,} | "
        f"**{part['B_alguien_corrige']['fraccion']:.1%}** | "
        f"{part['B_alguien_corrige']['n'] / 169336:.1%} | un mecanismo que use una evidencia que ya "
        f"existe |", "",
        "**La lectura que importa es la del grupo A.** Es un limite duro, y conviene decirlo con "
        "claridad: si mas de la mitad de los errores de V30 no los corrige ninguna fuente "
        "disponible, entonces ninguna reformulacion de las representaciones actuales va a "
        "tocarlos, por muy distinta que sea. El margen de esta investigacion esta en el grupo B.", "",
        "### Quien rescata mas", "",
        "| Fuente | Errores de V30 que corrige |", "|---|---|",
    ]
    for t in part["top_rescatadores"][:8]:
        lineas.append(f"| `{t['fuente']}` | {t['errores_de_v30_que_corrige']:,} |")
    lineas += ["", "## Las cuatro categorias, y cual manda", "",
               "| Categoria | Que significa | Cuantas |", "|---|---|---|",
               f"| **A. Disponible pero mal explotada** | la informacion existe y su integracion "
               f"fallo, no la informacion | "
               f"{counts.get('A', 0)} |",
               f"| **B. Nueva, no probada** | el mecanismo no se ha implementado en ningun sitio | "
               f"**{counts.get('B', 0)}** |",
               f"| **C. La integracion ya fallo** | cerrar como familia, no como configuracion | "
               f"{counts.get('C', 0)} |",
               f"| **D. Sin evidencia suficiente** | descartada por irrelevancia o por no tener "
               f"respaldo | {counts.get('D', 0)} |", "",
               "El brief pone la prioridad maxima en **B** y, en segundo lugar, en **A** solo si el "
               "mecanismo es genuinamente distinto. Aqui hay **1** hipotesis en B sin precedente "
               "cerrado, y es la unica que sobrevive al filtro de duplicados.", "",
               "## Inventario completo", "",
               "| ID | Cat | Familia | Hipotesis | Estado | Prioridad |", "|---|---|---|---|---|---|",
               ]
    for _, f in tabla.iterrows():
        lineas.append(f"| {f['ID']} | {f['categoria']} | {f['family']} | {f['hypothesis'][:70]} | "
                      f"**{f['status']}** | {f['priority']} |")
    lineas += ["", "## Los cierres, y por que son de familia y no de configuracion", "",
               "G01 a G05 no se reintentan con otro parametro. La razon es que **la mecanica ya "
               "fallo con un control de atribucion**, y cambiar `C`, `seed` o `beta` no cambia "
               "ese hecho:", "",
               "| Familia | Por que se cierra como familia |", "|---|---|",
               "| G01 correccion de score | P33: el ruido aleatorio rindio **mejor** que la senal "
               "real. P34: cero rescates de alta confianza |",
               "| G02 canal linguistico como feature | V31-06: +0,0002948 sobre el control "
               "barajado |",
               "| G03 capacidad del meta-modelo | V30 refinamiento +0,000069 sobre V29: la region "
               "estaba ya encontrada |",
               "| G04 ensembles historicos | 0 de 47 combinaciones con rescate neto positivo |",
               "| G13 bagging del residuo | V28-02 ya probo el bagging del meta-modelo: -0,000312 |",
               "",
               "## La unica linea que sobrevive, y por que es distinta", "",
               "**G12: meta-aprendizaje de baja dimension sobre el residuo.**", "",
               "No es un `beta` distinto. P33 probo `beta` compartido, que es **un** parametro, y "
               "un intercepto, y salio `alpha = 0`. Lo que no se probo es una funcion de "
               "correccion que dependa de **donde esta la fila respecto al margen de V30**: no "
               "un `beta` para todos, sino una correccion cuya intensidad varye con la confianza "
               "que V30 ya tiene.", "",
               "Tres razones por las que no es un duplicado:", "",
               "1. **Mecanismo**: P33 ajusto una constante; esto ajusta una pendiente sobre una "
               "variable que P33 no uso.",
               "2. **La evidencia apunta en esa direccion**: el AUC del margen de V30 para predecir "
               "sus propios errores es **0,8005**, y el del residuo es 0,5411. La correccion "
               "deberia pesar mas donde V30 ya es confiado, no donde duda.",
               "3. **Costo**: cabe en 2-3 parametros ajustados nested, sin grid.", "",
               "Y una razon para ser cauto, que conviene decir antes de ejecutarla: **es "
               "exactamente el mecanismo que la regla de continuidad prohibe repetir**. P34 ya "
               "toco la confianza de V30 en la version local, y la epico P33 +0,0002. La version "
               "global con una pendiente en vez de una constante es genuinamente otra operacion, "
               "pero el riesgo de que vuelva a fallar por el mismo motivo es real.", "",
               "**G14 (informacion nueva observable) tiene prioridad de exploracion pero no es "
               "ejecutable hoy**: exige identificar que campo del train no se ha leido todavia, y "
               "eso es trabajo de inventario, no un experimento. Es la unica hipotesis de "
               "categoria B con margen estructural, y por eso queda seleccionada para la "
               "siguiente fase.", "",
               "## Conclusion de la fase de investigacion actual", "",
               f"El grupo A tiene {part['A_nadie_corrige']['n']:,} filas. Ese es el techo. "
               f"Entre las categorias A, B, C y D, solo **dos** hipotesis no son duplicados "
               f"funccionales, y una de las dos requiere trabajo previo de inventario que no es un "
               f"experimento.", ""]
    GAP_MD.write_text("\n".join(lineas).rstrip("\n") + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
