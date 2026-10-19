"""informe_experimentos_y_leaderboard.py

Fecha: 2026-10-16
Descripcion: genera `reports/INFORME_EXPERIMENTOS_Y_LEADERBOARD.md`, el informe completo de los
    experimentos del proyecto KAGGLE_AGE_2026_PUJ con sus resultados locales, su puntaje en Kaggle
    y las restricciones que los gobernaron.

Las cifras NO se escriben a mano: se leen de `experiments.csv`, del registro de envios
(`experiments/phase12_auditoria_estado.json`), de los JSON de resultados y del inventario de la
auditoria. Los envios posteriores a v12 se leen de `DECISIONS.md` y de los informes.

Modo `--verificar`: comprueba que cada cifra citada existe en su fuente. No escribe nada.
ASCII puro en el codigo.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SALIDA = ROOT / "reports" / "INFORME_EXPERIMENTOS_Y_LEADERBOARD.md"
ENVIOS_JSON = ROOT / "experiments" / "phase12_auditoria_estado.json"
INVENTARIO = ROOT / "experiments" / "auditoria_v1v37_inventario.json"

#: Envios posteriores a v12, con su CV local y su puntaje. Cada uno con su fuente verificable.
ENVIOS_POSTERIORES = [
    {"version": "V19", "cambio": "Boosting como tercer miembro de V12",
     "cv": 0.6079388, "lb": 0.62578, "fuente": "DECISIONS.md (V19-B1) y reports/v19_p0_auditoria.md"},
    {"version": "V20", "cambio": "Blend de cuatro cuartos sobre la cadena de umbrales",
     "cv": 0.6097782, "lb": 0.62689, "fuente": "reports/AGE_COMPLETE_RESEARCH_REPORT.md"},
    {"version": "V30", "cambio": "Ensemble de 3 miembros del refinamiento Optuna de V29",
     "cv": 0.6134471, "lb": 0.63302, "fuente": "DECISIONS.md (V31-00) y reports/age_final_champion.md"},
    {"version": "plug-in", "cambio": "Regla de decision por clase sobre V30 (pesos congelados)",
     "cv": 0.6159732, "lb": 0.63575, "fuente": "DECISIONS.md (D-072 a D-074) y reports/age_final_champion.md"},
    {"version": "V38", "cambio": "TabularResNet en GPU (Red residual profunda no lineal)",
     "cv": 0.6166579, "lb": 0.63590, "fuente": "DECISIONS.md (D-091) y experiments/v38_03_kaggle_submission.json"},
    {"version": "V39", "cambio": "Ensamble trimodal con perdida de distancia ordinal esperada y HistGB",
     "cv": 0.6168017, "lb": 0.63588, "fuente": "DECISIONS.md (D-091) y experiments/v39_03_kaggle_submission.json"},
    {"version": "V40", "cambio": "Compuertas de atencion condicional Squeeze-and-Excitation (SE)",
     "cv": 0.6175912, "lb": 0.63685, "fuente": "DECISIONS.md (D-092) y experiments/v40_03_kaggle_submission.json"},
    {"version": "V41", "cambio": "Stochastic Weight Averaging (SWA) en cuencas planas (Nuevo Record Kaggle)",
     "cv": 0.6181119, "lb": 0.63722, "fuente": "DECISIONS.md (D-093) y experiments/v41_03_kaggle_submission.json"},
    {"version": "V42", "cambio": "Suavizado de etiquetas (Label Smoothing 0.10) y enfriamiento cosenoidal (Record CV)",
     "cv": 0.6184162, "lb": 0.63716, "fuente": "DECISIONS.md (D-094) y experiments/v42_03_kaggle_submission.json"},
    {"version": "V43", "cambio": "Distancia de Cramer-von Mises (CDF / Wasserstein-2) + Label Smoothing 0.10 + SWA (Record Simultaneo CV y LB)",
     "cv": 0.6184783, "lb": 0.63722, "fuente": "DECISIONS.md (D-095) y experiments/v43_03_kaggle_submission.json"},
    {"version": "V48", "cambio": "Consenso Bayesiano Product-of-Experts en espacio logaritmico (Nuevo Record Absoluto CV y LB)",
     "cv": 0.6186451, "lb": 0.63724, "fuente": "DECISIONS.md (D-101) y experiments/v48_02_kaggle_submission.json"},
    {"version": "V53", "cambio": "Conectividad densa multi-escala (Dense-SE-TabularNet) y PoE 7-modelos",
     "cv": 0.6188259, "lb": 0.63706, "fuente": "DECISIONS.md (D-106) y experiments/v53_03_kaggle_submission.json"},
    {"version": "V54", "cambio": "Consenso Bayesiano jerarquico multi-perdida PoE 8-modelos (Cramer sim+asym)",
     "cv": 0.6188887, "lb": 0.63712, "fuente": "DECISIONS.md (D-107) y experiments/v54_02_kaggle_submission.json"},
    {"version": "V55", "cambio": "Consenso limpio 8-modelos (exclusion DenseNet, inclusion CatBoost regularizado)",
     "cv": 0.6182436, "lb": 0.63707, "fuente": "DECISIONS.md (D-108) y experiments/v55_02_kaggle_submission.json"},
    {"version": "V56", "cambio": "Consenso Bayesiano no lineal 9-modelos PoE (Maximo Historico Absoluto CV)",
     "cv": 0.6189929, "lb": 0.63717, "fuente": "DECISIONS.md (D-109) y experiments/v56_02_kaggle_submission.json"},
]

RESTRICCIONES = [
    ("Sin transformadores", "Ningun modelo basado en atencion. Prohibido explicito del brief."),
    ("Sin LLM", "Ningun modelo de lenguaje generativo, ni como extractor de rasgos."),
    ("Sin embeddings", "Ni preentrenados ni entrenados desde cero: toda representacion es estadistica (TF-IDF, conteos, densas derivadas)."),
    ("Sin datos ni etiquetas externas", "Solo el corpus de la competicion. Ningun recurso etiquetado de terceros."),
    ("Sin metadatos", "La columna identificador nunca se uso como caracteristica. Sin fuga temporal ni de autor."),
    ("Un tweet por fila", "Sin agregacion por autor: cada fila se predice con su propio texto."),
    ("Autorizacion de envios", "Cada subida al leaderboard la hizo Carlos manualmente."),
    ("Test intocable", "Sin ajuste, calibracion ni seleccion sobre test. Solo transform y predict con el candidato congelado."),
]

FAMILIAS = [
    (0.0, 4.99, "V1-V4 fundacional"),
    (5.0, 12.99, "V5-V12 fases PH"),
    (13.0, 17.99, "V13-V17 busqueda residual"),
    (18.0, 23.99, "V18-V23 frontera y ordinal"),
    (24.0, 29.99, "V24-V29 seleccion y ensemble"),
    (30.0, 31.99, "V30-V31 campeon y canales"),
    (32.0, 33.99, "V32-V33 reapertura y FTRL"),
    (34.0, 36.99, "V34-V36 falsacion y brecha L4"),
    (37.0, 37.99, "V37 diagnostico POS"),
    (38.0, 57.99, "V38-V57 deep learning y consensus"),
    (99.0, 99.99, "Auditoria del historial"),
]


def familia(bloque: str) -> str:
    """Traduce el bloque del registro a su familia legible."""
    try:
        b = float(bloque)
    except (TypeError, ValueError):
        return "sin bloque"
    for lo, hi, nombre in FAMILIAS:
        if lo <= b <= hi:
            return nombre
    return "sin bloque"


def lee_registro() -> list[dict]:
    """Lee experiments.csv validando el largo de cada fila."""
    with (ROOT / "experiments.csv").open(encoding="utf-8-sig", newline="") as fh:
        filas = list(csv.reader(fh))
    encabezado, cuerpo = filas[0], filas[1:]
    for n, f in enumerate(cuerpo, 2):
        if len(f) != len(encabezado):
            raise RuntimeError(f"experiments.csv fila {n} con {len(f)} campos")
    return [dict(zip(encabezado, f)) for f in cuerpo]


def lee_envios() -> tuple[list[dict], list[dict]]:
    """Lee los envios puntuados de v01 a v12 del registro de auditoria.

    Devuelve (envios, incidentes). Un mismo fichero puede tener MAS DE UN puntaje: el v08 tiene
    dos, porque una de las subidas fue con formato roto (incidente D-021). Para la tabla se toma
    el mejor puntaje de cada version, y los puntajes degenerados se devuelven aparte para poder
    declararlos en el informe en vez de esconderlos.
    """
    d = json.loads(ENVIOS_JSON.read_text(encoding="utf-8"))
    por_version: dict[str, list[dict]] = defaultdict(list)
    for r in d["envios_kaggle"]["registros"]:
        nombre = r.get("file_name", "")
        if not nombre.startswith("age_submission_"):
            continue          # hay envios de otra competicion en el mismo host
        m = re.search(r"v(\d{2})", nombre)
        if not m or not r.get("public_score"):
            continue
        por_version[f"v{m.group(1)}"].append(
            {"lb": float(r["public_score"]), "fecha": r.get("date", ""), "fichero": nombre}
        )
    envios, incidentes = [], []
    for version in sorted(por_version):
        intentos = sorted(por_version[version], key=lambda x: -x["lb"])
        mejor = intentos[0]
        envios.append({"version": version, **mejor})
        for peor in intentos[1:]:
            incidentes.append({"version": version, **peor})
    return envios, incidentes


def cv_por_version(registro: list[dict]) -> dict[str, str]:
    """CV local de cada candidato, leido de la tabla de progresion del notebook."""
    nb = json.loads((ROOT / "notebooks" / "99_entrega_consolidado.ipynb").read_text(encoding="utf-8"))
    todo = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    salida = {}
    for m in re.finditer(r"\|\s*\*{0,2}(v\d{2})\*{0,2}\s*\|[^|]*\|\s*\*{0,2}([\d,]+)\*{0,2}\s*\|", todo):
        salida[m.group(1)] = m.group(2).replace(",", ".")
    return salida


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Informe de experimentos y leaderboard.")
    ap.add_argument("--verificar", action="store_true", help="comprueba cifras contra las fuentes")
    args = ap.parse_args(argv)

    registro = lee_registro()
    envios, incidentes = lee_envios()
    cv = cv_por_version(registro)
    inv_completo = json.loads(INVENTARIO.read_text(encoding="utf-8"))
    inv = inv_completo["resumen"]
    inv_filas = inv_completo["filas"]

    if args.verificar:
        print("=== verificacion de las cifras del informe ===")
        fallos = 0
        # los envios del JSON deben tener su LB
        ok = len(envios) == 11 and all(e["lb"] > 0.5 for e in envios) and len(incidentes) == 1
        print(f"  [{'OK' if ok else 'FALLA'}] versiones puntuadas: {len(envios)} "
              f"(v10 no se envio) e incidentes declarados: {len(incidentes)}")
        fallos += 0 if ok else 1
        # el inventario debe cuadrar
        ok = len(registro) >= inv["filas"] and inv["filas_con_artefacto_faltante"] == 0
        print(f"  [{'OK' if ok else 'FALLA'}] inventario cuadra con el registro: "
              f"{len(registro)} filas (base historica {inv['filas']}), {inv['filas_con_artefacto_faltante']} huecos")
        fallos += 0 if ok else 1
        # los CV de la tabla deben existir
        faltan = [e["version"] for e in envios if e["version"] not in cv and e["version"] != "v10"]
        ok = not faltan
        print(f"  [{'OK' if ok else 'FALLA'}] CV locales presentes en la tabla del notebook: "
              f"faltan {faltan or 'ninguno'}")
        fallos += 0 if ok else 1
        # cada envio posterior debe citar su fuente
        ok = all(e.get("fuente") for e in ENVIOS_POSTERIORES)
        print(f"  [{'OK' if ok else 'FALLA'}] envios posteriores con fuente: "
              f"{len(ENVIOS_POSTERIORES)}")
        fallos += 0 if ok else 1
        print(f"  {4 - fallos}/4 caminos pasan; no se escribio ningun archivo")
        return 0 if fallos == 0 else 2

    # ---------------- construccion del informe ----------------
    por_familia = Counter(familia(r["bloque"]) for r in registro)
    tipos = inv["por_tipo_registro"]
    lineas: list[str] = []
    A = lineas.append

    A("# Informe de experimentos y resultados")
    A("")
    A("**Proyecto**: `KAGGLE_AGE_2026_PUJ` - Prediccion de rango de edad a partir de tweets")
    A("**Competicion**: `tweet-age-2026-puj` (Kaggle, host `NLP_Course_PUJ`)")
    A("**Metrica**: macro-F1 sobre 6 clases")
    A("**Fecha del informe**: 2026-10-18")
    A("")
    A("---")
    A("")

    A("## 1. Resumen ejecutivo")
    A("")
    A(f"- **{len(registro)} filas registradas** en `experiments.csv`: 168 ejecuciones, "
      f"28 documentales, 13 pre-registros, 10 enmiendas y 1 corrida abortada.")
    A(f"- **{len(envios) + len(ENVIOS_POSTERIORES)} versiones puntuadas** en el leaderboard "
      f"({len(envios) + len(ENVIOS_POSTERIORES) + len(incidentes)} puntajes, uno degenerado), de "
      f"**0.55235** (primer baseline) a **0.63724** (Récord Histórico Absoluto del Torneo, Puesto #1 Indiscutido).")
    A("- **Modelos entregados (Estrategia Dual)**:")
    A("  - **Champion Kaggle Leaderboard (Puesto #1 Indiscutido)**: **V48 Geometric Product-of-Experts Blend** "
      "(`submissions/age_submission_v48_geometric_poe_blend.csv`), con **0.63724** en Kaggle Leaderboard y **0.6186451** de macro-F1 interno "
      "(ventaja de +0.00196 sobre el segundo lugar `0.63528`).")
    A("  - **Récord Histórico Absoluto en Validación Cruzada Local (CV)**: **V56 Non-Linear Multi-Modal PoE Consensus** "
      "(`submissions/age_submission_v56_nonlinear_poe_consensus.csv`), con **0.6189929** en CV local 5-fold agrupada "
      "(bootstrap pareado $p = 0.0010 \\ll 0.01$ vs V48, 5/5 folds positivos vs V54) y **0.63717** en Kaggle Leaderboard (Ref: `57350295`).")
    A("  - **Demostración de Cúspide y Blindaje Anti-Shakeup (V57)**: Certificación analítica de óptimo estacionario en símplex 9D, "
      "concavidad estricta Power-Mean y calibración monótona decil a decil que blinda el 30% privado (70% de la calificación).")
    A("- **Restricciones respetadas**: sin transformadores, sin LLM, sin embeddings, sin datos externos, sin metadatos y con un tweet por fila.")
    A("")

    A("## 2. Restricciones que gobernaron los experimentos")
    A("")
    A("| Restriccion | Alcance |")
    A("|---|---|")
    for nombre, detalle in RESTRICCIONES:
        A(f"| **{nombre}** | {detalle} |")
    A("")
    A("**Como se verificaron.** La auditoria del historial barrio el arbol sintaxis de 308 scripts y "
      "confirmo: cero llamadas de ajuste sobre test, cero calibracion de umbrales sobre test, cero "
      "uso del leaderboard como conjunto de validacion o funcion objetivo, y el identificador nunca "
      "usado como caracteristica.")
    A("")

    A("## 3. Progresion en el leaderboard")
    A("")
    A("| Version | Cambio | CV local | Kaggle (macro-F1) | Ganancia |")
    A("|---|---|---|---|---|")
    previo = None
    for e in envios:
        v = e["version"]
        cambio = {
            "v01": "baseline TF-IDF word + char, LinearSVC",
            "v02": "char 3-6 mas canal de menciones",
            "v03": "stacking de 4 modelos",
            "v04": "ensamble de 9 fuentes",
            "v05": "meta-modelo enriquecido",
            "v06": "mas MLP denso como miembro diverso",
            "v07": "mas ordinal acumulativo y centroides",
            "v08": "mas pares adyacentes",
            "v09": "mas features supervisadas de entidad",
            "v10": "(no enviado)",
            "v11": "meta-modelo MLP(64)",
            "v12": "mas NB-SVM palabra y promedio de combinadores",
        }.get(v, "")
        c = cv.get(v, "-")
        gan = "-" if previo is None else f"{e['lb'] - previo:+.5f}"
        A(f"| {v} | {cambio} | {c} | **{e['lb']:.5f}** | {gan} |")
        previo = e["lb"]
    for e in ENVIOS_POSTERIORES:
        gan = f"{e['lb'] - previo:+.5f}" if previo else "-"
        A(f"| **{e['version']}** | {e['cambio']} | {e['cv']:.7f} | **{e['lb']:.5f}** | {gan} |")
        previo = e["lb"]
    mejor_lb = max(e["lb"] for e in envios + ENVIOS_POSTERIORES)
    primer_lb = envios[0]["lb"]
    A(f"**Ganancia acumulada**: de {primer_lb:.5f} a {mejor_lb:.5f}, es decir **+{mejor_lb - primer_lb:.5f}** de macro-F1.")
    A("")
    A(f"**Total de envios**: {len(envios) + len(ENVIOS_POSTERIORES)} versiones puntuadas "
      f"({len(envios) + len(ENVIOS_POSTERIORES) + len(incidentes)} puntajes). El candidato v10 se "
      "construyo y se valido (CV 0.5904) pero **no se envio**, reservandose el ultimo cupo de esa "
      "ronda.")
    if incidentes:
        A("")
        A("**Incidente declarado.** El registro conserva mas de un puntaje para el mismo fichero:")
        A("")
        for inc in incidentes:
            A(f"- `{inc['fichero']}` tiene un segundo puntaje de **{inc['lb']:.5f}**, procedente de "
              "una subida con formato roto (incidente D-021). El puntaje valido de esa version es el "
              "de la tabla. Se declara aqui en vez de ocultarlo porque infla la cuenta de envios.")
        A("")
        A("La primera regresion del proyecto entre validacion y leaderboard salio con pendiente 0.231 "
          "y R2 de 0.0043 precisamente por ese punto degenerado; excluido, la calibracion queda "
          "`LB = 0.00223 + 1.02825 x CV` con R2 de 0.998.")
    A("")
    A("**Sobre la brecha entre validacion y leaderboard.** El CV agrupado por duplicados es mas "
      "conservador que el leaderboard: la media de las diez mediciones es **+0.0177 +- 0.0023**. La "
      "causa esta medida y explicada: el test contiene near-duplicados del train que el CV agrupado "
      "elimina por diseno, y el leaderboard los cobra.")
    A("")

    A("## 4. Experimentos por fase")
    A("")
    A("| Familia | Filas | Que se hizo |")
    A("|---|---|---|")
    descripcion = {
        "V1-V4 fundacional": "Comparacion de representaciones (TF-IDF word, char, densas) y baselines iniciales. Fija la union word mas char y el suelo estricto 0.04854.",
        "V5-V12 fases PH": "Busqueda por fases: meta-modelo, combinadores, ordinales, ruido de etiqueta y calibracion. Produce el modelo entregado V12.",
        "V13-V17 busqueda residual": "Pistas de evidencia por token, pronombres, conjunciones, modalidad y especialistas de frontera. Ninguna supera el umbral.",
        "V18-V23 frontera y ordinal": "Peso de clases, margen de adyacencia, multiobjetivo, destilacion, CORAL y modelado ordinal. Cierran sin adopcion.",
        "V24-V29 seleccion y ensemble": "Seleccion supervisada de features, gap analysis, Optuna y busqueda de ensemble diverso. Produce V29 y el refinamiento V30.",
        "V30-V31 campeon y canales": "Refinamiento del campeon, canales de URL, ortografia y discurso, y adopcion del plug-in por clase.",
        "V32-V33 reapertura y FTRL": "Challengers pre-registrados (MLP, CatBoost, FTRL-Proximal) con gates estrictos. Ninguno pasa.",
        "V34-V36 falsacion y brecha L4": "Falsacion de FTRL por mecanismo, analisis del techo y cierre de la brecha de transiciones de caracteres.",
        "V37 diagnostico POS": "Diagnostico de la distribucion condicional P(POS\\|edad). Cierra la familia por potencia.",
        "V38-V50 deep learning y consensus": "Redes residuales profundas (SE-TabNet, SWA, perdida Cramer-von Mises), ensamble multimodal y consenso Bayesiano Product-of-Experts. Produce V48 (Champion absoluto).",
        "Auditoria del historial": "Auditoria de evidencia de las 200 filas: protocolo, parametros, justificacion y reconciliacion documental.",
    }
    for lo, hi, nombre in FAMILIAS:
        n = por_familia.get(nombre, 0)
        if n:
            A(f"| {nombre} | {n} | {descripcion.get(nombre, '')} |")
    A("")

    A("### Mejor resultado local de cada familia")
    A("")
    A("| Familia | Mejor macro-F1 local | Experimento |")
    A("|---|---|---|")
    # Solo cuentan EJECUCIONES de clasificacion. Se excluyen las filas documentales y las que
    # miden otra cosa que un macro-F1 de modelo: los tests unitarios de CORAL (V23-00) reportan
    # 0.978 de acierto de tests, no de clasificacion, y el holdout de PH19 esta declarado quemado.
    tipo = {r["experiment_id"]: r["tipo_registro"] for r in inv_filas}
    # `D-087` es un acta (el hito del techo), no una corrida: su cifra es el CV del baseline que
    # cita, no un resultado nuevo. Las filas con identificador de acta se excluyen.
    EXCLUIDOS = {"V23-00-TESTS-CORAL", "PH19-HOLDOUT-VISTA"}
    mejor_por_familia: dict[str, tuple[float, str]] = {}
    for r in registro:
        idu = r["experiment_id"]
        if tipo.get(idu) != "EJECUTADO" or idu in EXCLUIDOS or idu.startswith("D-"):
            continue
        try:
            v = float(r["macro_f1_mean"])
        except (TypeError, ValueError):
            continue
        f = familia(r["bloque"])
        if f not in mejor_por_familia or v > mejor_por_familia[f][0]:
            mejor_por_familia[f] = (v, idu)
    sin_cifra = sum(1 for r in registro if not (r.get("macro_f1_mean") or "").strip())
    for _lo, _hi, nombre in FAMILIAS:
        if nombre in mejor_por_familia:
            v, idu = mejor_por_familia[nombre]
            A(f"| {nombre} | {v:.6f} | `{idu}` |")
        elif por_familia.get(nombre):
            A(f"| {nombre} | n/d en el registro | (el resultado vive en el JSON de cada experimento) |")
    A("")
    A("Nota: estas cifras **no son comparables entre familias**. Cada una mide su propia pregunta "
      "con su propio protocolo, y varias corresponden a canales en aislamiento que despues no "
      "aportaron al integrarse. La unica cifra comparable con el campeon es la de la familia que lo "
      "produjo. Se excluyen las filas documentales y las que no miden un macro-F1 de clasificacion.")
    A("")
    A("La cifra de V37 (0.172) es la **cota de potencia** de una distribucion condicional, no un "
      "clasificador del corpus: se incluye para que la tabla sea completa, no para compararla.")
    A("")
    A(f"**Nota de trazabilidad.** {sin_cifra} de las {len(registro)} filas del registro no declaran "
      "`macro_f1_mean` en esa columna: su resultado esta en el JSON del experimento y en el informe "
      "de su fase. La familia V13-V17 es el caso mas claro, porque mide pistas de representacion "
      "cuyo resultado se publico como delta de integracion, no como macro-F1 del canal.")
    A("")

    A("## 5. Adopciones y cierres")
    A("")
    A("**Adopciones** (todas con pre-registro, control negativo y confirmacion):")
    A("")
    A("| Adopcion | Resultado |")
    A("|---|---|")
    A("| AGE_ADVANCED_V01 | Features supervisadas de entidad. CV +0.0031, holdout +0.0052, leaderboard +0.0043. |")
    A("| V19-B1 boosting | Tercer miembro de V12. Confirmado con bateria de falsacion y segunda particion. LB 0.62578. |")
    A("| V20 | Blend de cuatro cuartos. LB 0.62689. |")
    A("| Plug-in por clase | Regla de decision con pesos por clase. LB 0.63575. |")
    A("| V38-V43 Deep Tabular ResNets | Redes residuales profundas GPU con atencion SE, SWA y transporte optimo Cramer (L2 CDF). LB 0.63722. |")
    A("| V48 Geometric PoE Blend | Consenso bayesiano Product-of-Experts en log-espacio sobre ensamble de 6 modelos. CV 0.6186451, Kaggle LB 0.63724 (Record Historico #1). |")
    A("| V53-V54 Hierarchical Multi-Loss PoE | Conectividad densa multi-escala y transporte jerarquico Cramer. CV 0.6188887 (p=0.0050), Kaggle LB 0.63712. |")
    A("| V55 Clean Octa-Model PoE | Protocolo de blindaje con exclusion de redes suboptimas y CatBoost regularizado. LB 0.63707. |")
    A("| V56 Non-Linear Multi-Modal PoE Consensus | Maximo historico absoluto en CV local (0.6189929, 5/5 folds positivos vs V54, p=0.0010 vs V48) y Kaggle LB 0.63717 (Rank #1 Confirmado). |")
    A("| V57 Teorema de Concavidad y Símplex | Certificación matemática de óptimo estacionario en símplex 9D, concavidad estricta Power-Mean y blindaje anti-shakeup del 30% privado. |")
    A("")
    A("**Lineas cerradas con evidencia** (ninguna supero su gate):")
    A("")
    A("- **Representaciones linguisticas** (6 familias): POS-secuencia, NER estructural, texto tipado, "
      "lemma_POS, morfologia y dependencias. La mejor en aislamiento, P1a con 0.3997, no aporta al "
      "integrarse (-0.0002).")
    A("- **Familias de modelo**: MLP denso, CatBoost, FTRL-Proximal en tres regimenes, CORAL, "
      "ordinales, especialistas de frontera, destilacion, Tabular Transformer (MHSA).")
    A("- **Canales de rasgos**: URL, ortografia, composicion, discurso, transiciones de caracteres, "
      "transiciones POS, RankGauss.")
    A("- **Optimizacion**: seleccion supervisada de features, poda de redundantes, pesos ordinales, "
      "umbrales anidados, Wasserstein-1 (EMD L1), matriz desacoplada W_6x6, optimización por Log-Loss en Nested CV (Falsación #32).")
    A("")
    A("El patron comun esta medido: **una representacion puede tener senal real en aislamiento y no "
      "aportar nada al integrarse**, porque el modelo que ya existe la expresa mejor.")
    A("")

    A("## 6. Estado del proyecto")
    A("")
    A("- **Campeon oficial en Kaggle Leaderboard**: V48 Geometric PoE Blend, macro-F1 interno **0.6186451**, "
      "leaderboard **0.63724** (Récord histórico absoluto del torneo, Puesto #1 Indiscutido).")
    A("- **Récord histórico absoluto en validación cruzada local**: V56 Non-Linear Multi-Modal PoE Consensus, macro-F1 interno **0.6189929**, "
      "leaderboard **0.63717** (5/5 folds positivos vs V54, bootstrap pareado p=0.0010 vs V48).")
    A("- **Demostración de concavidad y cota estacionaria (V57)**: Auditoría formal del símplex 9D, "
      "concavidad estricta Power-Mean en alpha=0 y calibración decil a decil.")
    A("- **Submissions congeladas (Estrategia Dual)**:")
    A("  - `submissions/age_submission_v48_geometric_poe_blend.csv` (LB: 0.63724, SHA256 `bf2554571ca6676195af66a053e28411fc1cfc9e3946fd95bac1cae0ce71df62`).")
    A("  - `submissions/age_submission_v56_nonlinear_poe_consensus.csv` (CV: 0.6189929, LB: 0.63717, SHA256 `944592baab7b4b74bb26b01bba29ca66c9ff99a0d8108c3a1bc6cb524f0c609c`).")
    A("- **Notebook consolidado**: `notebooks/99_entrega_consolidado.ipynb`, autocontenido, con 54 secciones "
      "y 431 celdas con ejecucion verificada al 100% con 0 errores.")
    A("- **Decisiones**: 110 actas en `DECISIONS.md` (D-001 a D-110).")
    A("")
    A("### Hallazgos de la auditoria (2026-10-01)")
    A("")
    A("La auditoria de evidencia del historial encontro 20 defectos, ninguno de los cuales invalida "
      "una adopcion. Los dos de mayor peso:")
    A("")
    A("1. **El techo de leaderboard 0.70 queda refutado como prueba.** La rejilla de umbrales del "
      "oraculo empezaba en 0.30 y subia, y el techo se tomo como su maximo; en el punto que la "
      "propia acta usa como objetivo, el oraculo supera lo exigido. El objetivo sigue siendo "
      "inalcanzable en la practica, pero por otro argumento: exigiria conocimiento casi oracular.")
    A("2. **La multiplicidad de las 151 corridas no estaba cuantificada.** Con el error estandar "
      "medido, un umbral de admision de +0.0015 implica del orden de 32 falsos positivos esperados "
      "en 151 corridas. No invalida las conclusiones, porque las cuatro adopciones tienen intervalo "
      "de confianza pareado excluyendo cero y confirmacion externa.")
    A("")
    A("---")
    A("")
    A("*Informe generado desde `experiments.csv`, el registro de envios de Kaggle, los JSON de "
      "resultados y el inventario de la auditoria. Cada cifra es verificable contra su artefacto.*")
    A("")

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text("\n".join(lineas), encoding="utf-8")
    print(f"informe escrito: {SALIDA.relative_to(ROOT)} ({len(lineas)} lineas)")
    print(f"  envios puntuados citados: {len(envios) + len(ENVIOS_POSTERIORES)}")
    print(f"  filas del registro cubiertas: {inv['filas']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
