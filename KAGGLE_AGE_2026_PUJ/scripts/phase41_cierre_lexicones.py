"""FASE 41 (P8) — Cierre documental de la fase de lexicones.

Lee los artefactos de P1 a P7 y escribe:

* ``reports/age_lexicon_audit.md`` — veredicto por familia (§28), respuesta a §35 y criterio de
  cierre aplicado (§29);
* las decisiones ``D-038`` (auditoría de paquetes embeddings/transformers) y ``D-039`` (veredicto
  de la línea de lexicones) en ``DECISIONS.md``;
* la sesión en ``RUNLOG.md``, el estado en ``PROJECT_STATE.md`` y la Parte IV en ``FINAL_REPORT.md``;
* las filas de ``experiments.csv``.

Es idempotente: si las decisiones ya existen, no las duplica.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "phase41_cierre.json"
AUDITORIA = cfg.REPORTS / "age_lexicon_audit.md"
GATE = "delta >= +0,003 y >= 4/5 folds positivos, y superar al control de permutacion"


def carga(nombre: str) -> dict | None:
    """Carga un artefacto de la fase si existe."""
    ruta = cfg.EXPERIMENTS / nombre
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


def main() -> int:
    """Escribe el cierre de la fase y actualiza la memoria del proyecto."""
    p37, p38, p39, p40 = (carga(f"phase{n}_" + nombre) for n, nombre in
                         ((37, "catalogo_lexicones.json"), (38, "bloques_lexicos.json"),
                          (39, "fusion_v12.json"), (40, "analisis_error.json")))
    if p39 is None:
        raise SystemExit("falta phase39: ejecuta la fusion antes de cerrar")
    resultados = pd.DataFrame(p39["tabla"])
    reales = resultados[resultados["tipo"] == "bloque real"]
    adoptar = p39["adoptan"]
    cerrar_todas = not adoptar

    # ------------------------------------------------------------------ auditoría
    veredictos = []
    for _, fila in reales.iterrows():
        familia = str(fila["candidato"]).replace("V12 +", "").split("+")[0].strip()
        medida = p39["candidatos"][fila["candidato"]]
        veredictos.append((familia, fila, medida))
    lineas = [
        "# Auditoría de la fase de lexicones (P8)", "",
        f"**Pregunta**: ¿aporta un pequeño número de lexicones lingüísticos generales una señal de "
        f"registro, etapa vital o dominio temático que complemente a V12?", "",
        f"**Gate pre-registado**: {GATE}. **Holdout**: agotado, no se ha reutilizado. "
        "**Leaderboard**: no se ha consultado. **Sin** embeddings, transformers, modelos externos, "
        "vectorizadores aprendidos nuevos, corpus/labels externos ni etiquetas de test.", "",
        "## 1. Qué se construyó", "",
        f"- **{p37['subcategorias']} subcategorías** en tres familias: registro (estilo), etapas de "
        "vida (contexto vital) y dominios temáticos (contenido).",
        f"- {p37['entradas_declaradas']} entradas declaradas -> **{p37['entradas_efectivas']} "
        f"efectivas** tras el cribado de aplicabilidad al corpus (frecuencia mínima 3 en el train, "
        "sin usar etiquetas).",
        f"- {p37['tokens_unicos_tras_ambiguedad']} tokens únicos; "
        f"{p37['tokens_ambiguos_descartados']} tokens ambiguos descartados (DD-5).",
        f"- Cobertura de tokens por familia: " + ", ".join(
            f"{familia} {medida['cobertura_pct']:.2f}%"
            for familia, medida in p37["cobertura_por_familia"].items()) + ".",
        "- Subcategorías retiradas por quedar sin vocabulario utilizable: "
        f"{p37.get('subcategorias_vacias') or 'ninguna'} (mínimo 5 entradas efectivas).", "",
        "## 2. Control lexicon-only: ¿existe la señal? (§14)", "",
        "Suelo de referencia: azar 0,1667 y clase prior 0,0485 (clases casi balanceadas). "
        "V12 alcanza 0,6004.", "",
        "| Familia | Columnas | Filas con alguna señal | LogReg | LinearSVC | Lectura |",
        "|---|---|---|---|---|---|",
    ]
    for familia, medida in p38["familias"].items():
        control = medida["control_lexicon_only"]
        lectura = ("supera el azar" if control["logreg"]["macro_f1"] > 0.1667 else
                   "por debajo del azar: señal muy débil o nula")
        lineas.append(f"| {familia} | {medida['n_features']} | "
                      f"{medida['filas_con_alguna_senal_pct']:.2f}% | "
                      f"{control['logreg']['macro_f1']} ± {control['logreg']['std']} | "
                      f"{control['linearsvc']['macro_f1']} ± {control['linearsvc']['std']} | {lectura} |")
    lineas += [
        "", "## 3. Redundancia (§12)", "",
        f"- {p38['features_redundantes']} de {p38['features_totales']} columnas superan "
        f"|rho| >= {p38['umbral_correlacion_redundancia']} contra alguna variable existente "
        f"({p38['redundantes_por_denso']} contra las densas, {p38['redundantes_por_v12']} contra "
        f"las 279 features de V12; correlacion maxima {p38['max_correlacion_v12']}).",
        "- El bloque **no** es una reformulación de lo que V12 ya tiene: su contenido es nuevo para "
        "el modelo, aunque sea débil.", "",
        "## 4. Fusión con V12 y control de permutación (P6, DD-6)", "",
        f"Referencia V12 reproducida en la misma sesión: **{p39['referencia_v12']['macro_f1']}** ± "
        f"{p39['referencia_v12']['std']} (per fold {p39['referencia_v12']['per_fold']}).", "",
        "| Candidato | Columnas | macro-F1 | Δ | Folds+ | Control permutado (media/max) | "
        "Ventaja sobre control | Gate |", "|---|---|---|---|---|---|---|---|",
    ]
    for nombre, medida in p39["candidatos"].items():
        controles = [c["macro_f1"] for c in medida["controles_permutados"]]
        lineas.append(
            f"| {nombre} | {medida['real']['per_fold'] and ''}"
            f"{next(f['n_columnas_anadidas'] for f in p39['tabla'] if f['candidato'] == nombre)} | "
            f"{medida['real']['macro_f1']} | {medida['delta']:+.4f} | {medida['folds_positivos']}/5 | "
            f"{sum(controles) / len(controles):.4f} / {medida['mejor_control']:.4f} | "
            f"{medida['ventaja_sobre_control_maximo']:+.4f} | "
            f"{'SI' if medida['gate_numerico'] else 'no'} |")
    lineas += ["",
               "**El control de permutación es la pieza decisiva**: un bloque real que no supere a su "
               "propia nula (mismo número de columnas, filas permutadas) no aporta señal, aunque "
               "mejore la media.", ""]
    if p40:
        lineas += ["## 5. Complementariedad y error (§19-§21)", ""]
        for nombre, medida in p40["candidatos"].items():
            complementaridad = medida["complementariedad"]
            lineas.append(f"- **{nombre}**: rescate {complementaridad['solo_candidato']}, daño "
                          f"{complementaridad['solo_v12']}, **neto {complementaridad['neto']:+d}**, "
                          f"modificadas {complementaridad['filas_modificadas_pct']:.3f}%, fronteras "
                          f"centrales {medida['fronteras_centrales_delta']}.")
        lineas.append("")
    lineas += [
        "## 6. Veredicto por familia (§28)", "",
        "| Familia | Resultado | Decisión |", "|---|---|---|",
    ]
    for familia, fila, medida in veredictos:
        if medida["gate_numerico"] and medida["supera_control"]:
            decision = "E - supera el gate: candidata (sin envío, requiere revisión manual)"
        elif medida["gate_numerico"]:
            decision = ("C - mejora nominal pero la explica el control de permutado: "
                        "se descarta")
        elif medida["supera_control"]:
            decision = ("B/D - señal por encima de la nula pero sin llegar al gate: cerrada")
        else:
            decision = "A - negativa: cerrada"
        lineas.append(f"| {familia} | {fila['macro_f1']} (Δ {fila['delta']:+.4f}, "
                      f"{fila['folds_positivos']}/5) | {decision} |")
    lineas += [
        "", "## 7. Respuesta a la pregunta científica (§35)", "",
        "> ¿Los lexicones proporcionan una dimensión lingüística de edad que V12 no captura?",
        "", "**No, de forma explotable.** El detalle por dimensión:", "",
    ]
    for familia, medida in p38["familias"].items():
        control = medida["control_lexicon_only"]["logreg"]["macro_f1"]
        cobertura = p37["cobertura_por_familia"][familia]["cobertura_pct"]
        if control > 0.1667:
            veredicto = ("señal débil pero real por sí misma: el bloque supera el azar sin V12")
        else:
            veredicto = "sin señal medible por sí misma (por debajo del azar de 0,1667)"
        fusion = [m for n, m in p39["candidatos"].items() if familia.lower() in n.lower()]
        en_fusion = ("ninguna" if not fusion else
                     f"Δ {fusion[0]['delta']:+.4f} con ventaja sobre el control "
                     f"{fusion[0]['ventaja_sobre_control_maximo']:+.4f}")
        lineas.append(f"- **{familia}**: control lexicon-only {control:.4f} (cobertura "
                      f"{cobertura:.2f}% de los tokens) -> {veredicto}; fusión con V12: {en_fusion}.")
    lineas += [
        "", "## 8. Criterio de cierre (§29)", "",
    ]
    if cerrar_todas:
        lineas.append("Se aplica **A** (ninguna familia alcanza el gate), reinforce **B/D** (el "
                      "bloque no es redundante, pero su señal no se traduce en mejora) y **C** "
                      "(donde el Δ nominal no sobrevive al control de permutación). **La línea se "
                      "cierra.** V12 permanece congelado y no se genera ninguna submission.")
    else:
        lineas.append(f"**E**: {adoptar} supera el gate. Queda como candidata sin enviar, "
                      "pendiente de revisión manual (§26).")
    lineas += [
        "", "## 9. Auditoría de recursos (§32)", "",
        "- `transformers`, `sentence-transformers` y `torch` están instalados en el venv y listados "
        "en `requirements.txt`; **el uso en el camino de modelado es 0** y el único import del "
        "proyecto es una detección de capacidad de GPU (`src/gpu.py`).",
        "- Embeddings, Word2Vec, FastText, GloVe, sentence-transformers, modelos de lenguaje: "
        "**0 usos**. Corpus o labels externos: **0**. Etiquetas de test: **0**.",
        "- Los únicos recursos externos Allowed en esta fase son listas léxicas simbólicas "
        "elaboradas en el proyecto, versionadas con SHA256 en `reports/age_lexicon_catalog.csv`.",
        "- El cribado de frecuencia mínima usa el vocabulario del train completo: es **selección no "
        "supervisada y transductiva**, sin usar etiquetas en ningún punto; se declara por "
        "transparencia.", ""]
    AUDITORIA.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ memoria
    decisiones = ROOT / "DECISIONS.md"
    texto = decisiones.read_text(encoding="utf-8")
    if "## D-038" not in texto:
        texto += """
---

## D-038 — Auditoría de paquetes: el entorno contiene transformers y torch, el proyecto no los usa
- **Observation**: `transformers`, `sentence-transformers` y `torch` están instalados en
  `/home/carlos/venv` y listados en `requirements.txt`. El barrido de imports en `src/` y
  `scripts/` encuentra un único `import torch`, dentro de una comprobación de capacidad de GPU
  (`src/gpu.py`).
- **Decision**: documentarlo explícitamente en lugar de ocultarlo. La restricción de AGE es sobre
  el **uso** (sin embeddings, transformers ni modelos de lenguaje en el camino de modelado), y ese
  uso es 0. Se registra que el entorno no está limpio de paquetes para que la auditoría sea
  auditable en ambos sentidos.
- **Consequences**: cualquier revisión futura debe distinguir "paquete instalado" de "paquete
  usado"; el informe de auditoría incluye las dos columnas.

---

## D-039 — La línea de lexicones se cierra: el bloque es nuevo pero no aporta sobre V12
- **Observation**: 24 subcategorías léxicas (registro, etapas de vida, dominios temáticos),
  1.369 entradas efectivas, 72 features agregadas. El cribado de redundancia muestra que el bloque
  **no** es una reformulación de lo que V12 ya tiene (correlación máxima contra las 279 features de
  V12 por debajo del umbral). El control *lexicon-only* da 0,152 (registro), 0,073 (vida) y 0,203
  (temas) frente a un suelo de azar de 0,1667: solo **temas** supera el azar por sí solo, y por poco.
- **Experiment**: fusión con V12 bajo corrected CV pareado, con control de permutación de varias
  semillas (DD-6): ver `experiments/phase39_fusion_v12.json` y `reports/age_lexicon_results.csv`.
- **Result**: ningún bloque alcanza el gate (Δ ≥ +0,003 con ≥4/5 folds) y ninguno supera a su
  control permutado. La complementariedad es nula o negativa (rescate ≤ daño).
- **Decision**: **cerrar la línea**. V12 permanece congelado; no se genera submission.
- **Reason**: es la segunda línea consecutiva (tras la capa de decisión) en la que la señal es real
  pero insuficiente. La explicación coherente con el diagnóstico previo: el error residual de V12
  no es de representación sino de **observabilidad de la edad** (35% del error es techo de
  información). Un léxico general añade contexto temático y de registro, que es exactamente lo que
  los n-gramas de V12 ya representan bien; lo que falta es información que el corpus no contiene.
- **Consequences**: (1) la línea de features está agotada bajo las restricciones de AGE; (2) el
  criterio que queda abierto es la **frontera** (18-24|25-34 y 25-34|35-49), no más señal léxica;
  (3) el artefacto `lexicons/` y su catálogo se conservan como recurso reutilizable.
- **Evidence**: `reports/age_lexicon_audit.md`, `reports/age_lexicon_results.csv`,
  `reports/age_lexicon_ablation.csv`, `reports/age_lexicon_redundancy.csv`,
  `reports/age_lexicon_complementarity.md`, `reports/age_lexicon_error_analysis.md`.
"""
        decisiones.write_text(texto, encoding="utf-8")

    filas = [
        {"experiment_id": "PH36-COBERTURA-LEX", "fecha": "2026-09-25", "bloque": 6,
         "etapa": "P0 auditoria de cobertura", "pregunta": "¿Que hipotesis lexicas son genuinamente nuevas?",
         "hipotesis": "Ninguno de los 67 experimentos previos construyo un canal lexico agregado.",
         "features": "-", "preprocessing": "-", "representacion": "-", "modelo": "-",
         "hiperparametros": "{}", "validacion": "revisión de artefactos (grep dirigido)",
         "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "", "micro_f1_mean": "",
         "accuracy_mean": "", "resultado": "0 experimentos y 0 lineas de codigo previas mencionan un canal lexico (nivel 1 de verificacion).",
         "interpretacion": "La linea es genuinamente nueva; los modelos ordinales y el canal supervisado de entidades ya existian pero no como canal lexico agregado.",
         "decision": "Cobertura publicada; 3 hipotesis abiertas (H-LEX-A/B/C).", "duracion_seg": 5,
         "artefactos": "reports/age_lexicon_hypothesis_coverage.md"},
        {"experiment_id": "PH37-CATALOGO-LEX", "fecha": "2026-09-25", "bloque": 6,
         "etapa": "P1 catalogo de lexicones", "pregunta": "¿Que recursos lexicos son admisibles?",
         "hipotesis": "Un conjunto pequeño de listas autoelaboradas es admisible y utilizable.",
         "features": "24 subcategorias, 1.369 entradas efectivas", "preprocessing": "transliteracion al ASCII del corpus; cribado de frecuencia minima 3; eliminacion de ambiguos",
         "representacion": "sin representaciones nuevas", "modelo": "-", "hiperparametros": '{"min_frecuencia": 3}',
         "validacion": "cribado no supervisado sobre el vocabulario del train",
         "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "", "micro_f1_mean": "",
         "accuracy_mean": "", "resultado": "1.588 declaradas -> 1.369 efectivas; cobertura registro 5,72% / vida 0,62% / temas 10,04%; 131 tokens ambiguos descartados; vida/hogar retirada por quedar con 2 entradas.",
         "interpretacion": "El cribado de aplicabilidad elimino tambien el ruido de las listas fuente; la familia vida apenas aparece en el corpus.",
         "decision": "Catalogo publicado con SHA256 por fichero.", "duracion_seg": 30,
         "artefactos": "reports/age_lexicon_catalog.csv, lexicons/**"},
        {"experiment_id": "PH38-BLOQUES-LEX", "fecha": "2026-09-25", "bloque": 6,
         "etapa": "P2-P5 bloques, control, redundancia y ablacion",
         "pregunta": "¿Existe la señal lexica y es redundante con lo que V12 ya tiene?",
         "hipotesis": "Las categorias linguisticas de alto nivel capturan dimensiones no expresadas por tokens individuales.",
         "features": "72 agregados (3 por subcategoria + 1 por familia)", "preprocessing": "conteo, ratio y presencia por subcategoria",
         "representacion": "sin representaciones nuevas", "modelo": "LogReg y LinearSVC (control lexicon-only) y ablación por subcategoria",
         "hiperparametros": '{"C_logreg": 1.0, "C_svc": 0.2}', "validacion": "corrected CV agrupado (LOFO)",
         "macro_f1_mean": 0.2033, "macro_f1_std": 0.0022, "weighted_f1_mean": "", "micro_f1_mean": "",
         "accuracy_mean": "", "resultado": "Control lexicon-only: temas 0,2033 | registro 0,1520 | vida 0,0731 (azar 0,1667). Redundancia: 0 de 72 columnas con |rho|>=0,8 contra V12. Mejores subcategorias aisladas: religion 0,1419, abreviaciones_internet 0,1247, television 0,1178.",
         "interpretacion": "El bloque aporta dimensionalidad nueva para el modelo, pero solo la familia de temas supera el azar por si misma y muy marginally.",
         "decision": "Continuar a P6 con las tres familias y su control de permutacion.", "duracion_seg": 300,
         "artefactos": "reports/age_lexicon_ablation.csv, reports/age_lexicon_redundancy.csv, features/lexicon_features_*.npy"},
        {"experiment_id": "PH39-FUSION-LEX", "fecha": "2026-09-25", "bloque": 6,
         "etapa": "P6 fusion con V12", "pregunta": "¿El bloque lexico mejora a V12 por encima de su control de permutacion?",
         "hipotesis": "La señal lexica es complementaria y se traduce en macro-F1.",
         "features": "V12 (279) + bloque lexico", "preprocessing": "permutacion de filas como control DD-6 (3 semillas si pasa el gate)",
         "representacion": "sin representaciones nuevas", "modelo": "combinador de produccion (0,5 LogReg + 0,5 MLP)",
         "hiperparametros": '{"C_logreg": 1.0, "semillas_permutacion": [1000, 1001, 1002]}',
         "validacion": "corrected CV agrupado, comparacion pareada fold a fold contra V12",
         "macro_f1_mean": p39["referencia_v12"]["macro_f1"], "macro_f1_std": p39["referencia_v12"]["std"],
         "weighted_f1_mean": "", "micro_f1_mean": "", "accuracy_mean": "",
         "resultado": "; ".join(f"{nombre}: {medida['delta']:+.4f} ({medida['folds_positivos']}/5) "
                                f"control {medida['mejor_control']:.4f}" for nombre, medida
                                in p39["candidatos"].items()),
         "interpretacion": "Ningun bloque alcanza el gate; el control de permutacion explica cualquier mejora nominal.",
         "decision": "CERRAR la linea (D-039)." if not adoptar else f"Candidatas: {adoptar}",
         "duracion_seg": int(p39.get("segundos", 0)),
         "artefactos": "experiments/phase39_fusion_v12.json, reports/age_lexicon_results.csv"},
    ]
    if p40:
        filas.append({"experiment_id": "PH40-ANALISIS-ERROR-LEX", "fecha": "2026-09-25", "bloque": 6,
                      "etapa": "P7 analisis de error", "pregunta": "¿El lexico rescata errores reales de V12?",
                      "hipotesis": "Complementariedad positiva con V12 en fronteras y segmentos.",
                      "features": "argmax OOF de V12 y de cada candidato", "preprocessing": "-",
                      "representacion": "-", "modelo": "-", "hiperparametros": "{}",
                      "validacion": "analisis sobre OOF",
                      "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "",
                      "micro_f1_mean": "", "accuracy_mean": "",
                      "resultado": "; ".join(f"{n}: rescate {m['complementariedad']['solo_candidato']} / "
                                             f"daño {m['complementariedad']['solo_v12']} / neto "
                                             f"{m['complementariedad']['neto']:+d}" for n, m
                                            in p40["candidatos"].items()),
                      "interpretacion": "Rescate menor o igual al dano: la modificacion no es constructiva.",
                      "decision": "Sin cambio en el veredicto.", "duracion_seg": 60,
                      "artefactos": "reports/age_lexicon_complementarity.md, reports/age_lexicon_error_analysis.md"})
    ya = (ROOT / "experiments.csv").read_text(encoding="utf-8")
    nuevas = [fila for fila in filas if fila["experiment_id"] not in ya]
    with (ROOT / "experiments.csv").open("a", newline="", encoding="utf-8") as archivo:
        for fila in nuevas:
            csv.DictWriter(archivo, fieldnames=list(fila)).writerow(fila)

    runlog = ROOT / "RUNLOG.md"
    texto = runlog.read_text(encoding="utf-8")
    if "Sesión 22" not in texto:
        texto += """
## 2026-09-25 — Sesión 22 (lexicones: fases P0-P8)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 130 | Entorno y housekeeping | `scripts/fase00_entorno.py` | 16 núcleos/32 hilos, 31,2 GB, RX 9070 XT; commit de la fase de decisión y rama nueva |
| 131 | Cobertura de hipótesis (P0) | `scripts/phase36_auditoria_lexicones.py` | 0 experimentos previos con canal léxico |
| 132 | Catálogo de lexicones (P1) | `scripts/phase37_catalogo_lexicones.py` | 24 subcat, 1.369 entradas, SHA256 por fichero |
| 133 | Bloques, control y redundancia (P2-P5) | `scripts/phase38_bloques_lexicos.py` | Control lexicon-only 0,203/0,152/0,073; 0 columnas redundantes con V12 |
| 134 | Revisión de código (puerta obligatoria) | `@ml-reviewer` | APROBADO CON CAMBIOS: el control de permutación no gateaba la adopción (corregido) |
| 135 | Fusión con V12 y control de permutación (P6) | `scripts/phase39_fusion_v12.py` | Ningún bloque alcanza el gate |
| 136 | Análisis de error y complementariedad (P7) | `scripts/phase40_analisis_error.py` | Rescate ≤ daño en todos los candidatos |
| 137 | Cierre documental (P8) | `scripts/phase41_cierre_lexicones.py` | D-038 y D-039; línea cerrada |

### Bugs corregidos en la sesión
- `ml-reviewer` detectó que el control de permutación se calculaba pero no gateaba la adopción
  (falso positivo posible): ahora `adopta` exige superar la nula, con 3 semillas si el gate numérico
  pasa, y se añade la comprobación de cabeza de DD-8.
- El cribado de redundancia de P5 usaba `spearmanr` con una matriz 2D, que devuelve la matriz de
  correlaciones entre variables: leía la autocorrelación (1,0) y marcaba las 72 columnas como
  redundantes. Corregido con ranks vectorizados; el resultado real es 0 columnas redundantes.
- `src/lexicons.py`: umbral mínimo de 5 entradas por subcategoría (retira `vida/hogar`), formas
  ambiguas comprobadas también en carga y tokenización con límites de palabra.
- `phase37`: fecha constante para que los SHA256 sean reproducibles.

### Resultado
La línea de lexicones queda **cerrada** (D-039). El bloque es genuinamente nuevo (0 de 72
columnas redundantes con V12) pero su señal es insuficiente: solo la familia de temas supera el
azar en control aislado, y ningún bloque alcanza el gate ni supera a su control de permutación.
"""
        runlog.write_text(texto, encoding="utf-8")

    estado = ROOT / "PROJECT_STATE.md"
    texto = estado.read_text(encoding="utf-8")
    inicio = texto.index("## Fase actual")
    fin = texto.index("## Estado según el formato del brief §83")
    texto = texto[:inicio] + f"""## Fase actual
- **FASE DE LEXICONES (2026-09-25) CERRADA — línea cerrada (D-039).** 24 subcategorías léxicas
  (registro, etapas de vida, dominios temáticos), 1.369 entradas efectivas y 72 features
  agregadas, construidas como función determinista del texto (sin etiquetas). Control *lexicon-only*:
  temas **0,2033**, registro **0,1520**, vida **0,0731** frente a un suelo de azar de 0,1667.
  Redundancia contra las 279 features de V12: **0 de 72 columnas**. Fusión con V12: **ningún bloque
  alcanza el gate** (Δ ≥ +0,003 con ≥4/5 folds) ni supera a su control de permutación; la
  complementariedad es nula o negativa (rescate ≤ daño).
- **Modelo final: V12 congelado.** Leaderboard de referencia **0,62188**. **No se generó ni se
  envió ninguna submission nueva.**
- **D-038**: `transformers`, `sentence-transformers` y `torch` están instalados en el venv, pero su
  uso en el camino de modelado es 0 (único import: detección de GPU). Se documenta sin maquillar.
- **Conclusión**: bajo las restricciones de AGE, la línea de representación léxica está agotada.
  El criterio que queda abierto es la **frontera** (18-24|25-34 y 25-34|35-49), no más señal.
- **Entregables**: `reports/age_lexicon_hypothesis_coverage.md`, `reports/age_lexicon_catalog.csv`,
  `reports/age_lexicon_ablation.csv`, `reports/age_lexicon_redundancy.csv`,
  `reports/age_lexicon_results.csv`, `reports/age_lexicon_complementarity.md`,
  `reports/age_lexicon_error_analysis.md`, `reports/age_lexicon_audit.md`, `lexicons/**`.

""" + texto[fin:]
    estado.write_text(texto, encoding="utf-8")

    informe = ROOT / "FINAL_REPORT.md"
    texto = informe.read_text(encoding="utf-8")
    if "Parte IV" not in texto:
        texto = texto.rstrip() + f"""

---

# Parte IV — Lexicones lingüísticos (2026-09-25)

## 26. Qué se construyó

24 subcategorías léxicas de conocimiento general (6 de registro e informalidad, 6 de etapas de
vida y contexto social, 12 de dominios temáticos), {p37['entradas_efectivas']} entradas efectivas
tras cribado de aplicabilidad al corpus, y 72 features agregadas (conteo, ratio y presencia por
subcategoría, más un agregado por familia). La transformación `tweet -> categorías` es
determinista y **no usa etiquetas**, a diferencia del canal supervisado de entidades ya existente.

## 27. Resultado por fase

| Fase | Resultado |
|---|---|
| P0 cobertura | 0 experimentos previos con canal léxico: la línea es nueva |
| P1 catálogo | 24 subcategorías, 1.588 entradas declaradas -> {p37['entradas_efectivas']} efectivas, SHA256 por fichero |
| P2-P5 control | lexicon-only: temas {p38['familias']['temas']['control_lexicon_only']['logreg']['macro_f1']}, registro {p38['familias']['registro']['control_lexicon_only']['logreg']['macro_f1']}, vida {p38['familias']['vida']['control_lexicon_only']['logreg']['macro_f1']} (azar 0,1667) |
| P5 redundancia | 0 de 72 columnas con \\|ρ\\| ≥ 0,8 contra las 279 features de V12: el bloque es nuevo |
| P6 fusión | ningún bloque alcanza el gate; ninguno supera a su control de permutación |
| P7 complementariedad | rescate ≤ daño en todos los candidatos |

## 28. Tres conclusiones defendibles

1. **El bloque no es redundante**: la correlación máxima contra las 279 features de V12 queda por
   debajo del umbral, así que la señal no es una reformulación de lo que el modelo ya tenía. La
   hipótesis de §12 ("¿añade una dimensión nueva?") se responde **sí** a nivel de variable.
2. **La señal nueva no es útil**: solo la familia de temas supera el azar en aislamiento, y por
   poco. Un contexto temático es real pero marginal para la edad, que es una combinación de muchos
   factores débiles.
3. **El control de permutación es imprescindible**: sin él, cualquier Δ nominal positivo del
   bloque podría atribuirse al tamaño del bloque y no a su contenido.

## 29. Conclusión

Los lexicones **no** aportan una dimensión de edad explotable sobre V12. Junto con el resultado de
la capa de decisión (33 capas, ninguna adopta), la evidencia converge: bajo las restricciones de
AGE, el límite no está en la representación sino en la **observabilidad de la edad** (35% del
error es techo de información). V12 permanece congelado (leaderboard 0,62188) y la línea queda
cerrada con D-039.

_Generado por `scripts/phase41_cierre_lexicones.py`._
"""
        informe.write_text(texto, encoding="utf-8")

    RESULTADO.write_text(json.dumps({
        "fase": "P8 — cierre de la fase de lexicones",
        "fecha": "2026-09-25",
        "adoptan": adoptar, "linea_cerrada": cerrar_todas,
        "veredictos": {familia: (medida["gate_numerico"] and medida["supera_control"])
                       for familia, _, medida in veredictos},
        "experimentos_registrados": [fila["experiment_id"] for fila in filas],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"auditoria: {AUDITORIA.name} |.linea cerrada: {cerrar_todas} | "
          f"adoptan: {adoptar if adoptar else 'ninguno'}")
    print(f"experimentos registrados: {[f['experiment_id'] for f in nuevas]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
