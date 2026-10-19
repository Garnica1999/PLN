"""P92-02: guia de anotacion v1, plan, README de anotadores y manifiestos.

Lee `reports/p92_audit_results.json` y escribe:
- reports/p92_annotation_guidelines_v1.md;
- reports/p92_human_annotation_plan.md;
- reports/p92_sampling_manifest.json;
- artifacts/human_annotation_v1/README_annotators.md;
- artifacts/human_annotation_v1/manifest.json (hashes de los 6 archivos del paquete).

Sin entrenamiento, sin etiquetas, sin test/holdout. La auditoria de preparacion
(`p92_annotation_readiness_audit.md`) se escribe despues de ejecutar el verificador.

Uso:
    /home/carlos/venv/bin/python experiments/p92_02_informe_plan.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

RAIZ: Path = Path(__file__).resolve().parents[1]
RUTA_RESULTS = RAIZ / "reports" / "p92_audit_results.json"
DIR_PAQUETE = RAIZ / "artifacts" / "human_annotation_v1"

RUTA_GUIA = RAIZ / "reports" / "p92_annotation_guidelines_v1.md"
RUTA_PLAN = RAIZ / "reports" / "p92_human_annotation_plan.md"
RUTA_SAMPLING = RAIZ / "reports" / "p92_sampling_manifest.json"
RUTA_README = DIR_PAQUETE / "README_annotators.md"
RUTA_MANIFEST = DIR_PAQUETE / "manifest.json"


def escribir(ruta: Path, texto: str) -> None:
    """Escribe texto exigiendo ASCII puro."""
    if any(ord(c) > 127 for c in texto):
        raise SystemExit(f"STOP: bytes > 127 en {ruta.name}")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(texto, encoding="utf-8")


def sha256_de(ruta: Path) -> str:
    """Calcula el sha256 de un archivo."""
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


def tabla(cabeceras: list[str], filas: list[list[object]]) -> str:
    """Construye una tabla markdown."""
    lineas = ["| " + " | ".join(cabeceras) + " |",
              "|" + "|".join("---" for _ in cabeceras) + "|"]
    for fila in filas:
        lineas.append("| " + " | ".join(str(c) for c in fila) + " |")
    return "\n".join(lineas)


def main() -> int:
    """Genera los documentos del paquete."""
    for ruta in (RUTA_GUIA, RUTA_PLAN, RUTA_SAMPLING, RUTA_README, RUTA_MANIFEST):
        if ruta.exists():
            raise SystemExit(f"STOP: ya existe {ruta}")
    d = json.loads(RUTA_RESULTS.read_text(encoding="utf-8"))
    a = d["componente_A"]
    b = d["componente_B"]
    cal = d["calibracion"]
    prin = d["principal"]

    guia = f"""# P92 - Guia de anotacion provisional v1 (2026-10-09)

Estado: **PROVISIONAL (v1)**. Aplicable a la calibracion y, si no se reemplaza por una
version v2 antes de empezar, a las muestras principales. Version congelada para las
muestras principales: v1 (a menos que la calibracion produzca un changelog y se congele v2
ANTES de iniciar la anotacion principal).

## 1. Definiciones oficiales (verbatim, docs/competition_rules.md)

- **HS**: "indica si hay discurso de odio (1 = si, 0 = no)".
- **TR**: "indica si el discurso esta dirigido contra una persona o grupo (1 = si, 0 = no)".
- **AG**: "indica si el texto es agresivo (1 = si, 0 = no)".

Estas definiciones son la UNICA fuente normativa disponible. No existen criterios
operativos oficiales ni guia de anotadores original. **No inventes criterios** y no
impongas jerarquias entre etiquetas: no asumas que TR=1 exige HS=1 ni que AG=1 exige HS=1
(la relacion observada en los datos no esta confirmada como regla formal).

## 2. Unidad de anotacion

Un **tweet completo**. Cada anotador decide las TRES etiquetas del mismo texto, de forma
independiente entre si. El texto se muestra exactamente como aparece en la fuente; no lo
normalices, no lo corrijas y no elimines menciones, URLs, hashtags ni puntuacion.

## 3. Procedimiento por etiqueta

Para cada etiqueta (HS, TR, AG), por separado:

1. Lee el texto completo.
2. Aplica unicamente la definicion oficial.
3. Si la definicion permite una decision responsable -> **POSITIVE** o **NEGATIVE**.
4. Si la definicion no permite decidir (falta contexto, ambiguedad genuina, referencia
   ininteligible) -> **UNABLE_TO_DETERMINE** y explica el motivo en `uncertainty_notes`.

`UNABLE_TO_DETERMINE` **no es una tercera clase semantica**: indica que no puedes asignar
responsablemente una etiqueta binaria con las instrucciones y el contexto disponibles. No
lo conviertas automaticamente en POSITIVE ni en NEGATIVE.

## 4. Textos ambiguos y falta de contexto

- Registra la ambiguedad en `uncertainty_notes` (breve, en tus palabras).
- Muchos textos son respuestas sin el tweet original: si la ausencia de contexto impide
  decidir, usa UNABLE_TO_DETERMINE y anotalo.
- No busques los tweets en internet ni consultes ninguna fuente externa para decidir.
- No consultes las etiquetas originales, predicciones de modelos, umbrales ni ningun
  material del proyecto: tu juicio debe ser independiente.

## 5. Reglas de cegamiento

- No consultes ni intentes deducir la etiqueta original ni por que el caso fue incluido.
- No comentes los casos con el otro anotador antes de terminar tu primera ronda completa.
- La discusion y la adjudicacion son una etapa POSTERIOR, separada y documentada.
- No uses herramientas automaticas (LLM, clasificadores) para asignar tus etiquetas.

## 6. Registro de desacuerdos

Los desacuerdos entre anotadores no se resuelven en esta ronda: se conservan ambas
respuestas y se remiten a la adjudicacion (plantilla separada). Registra tambien, si
quieres, si tu duda es semantica (que significa), normativa (que regla aplica), contextual
(falta informacion) o de aplicacion de la guia.

## 7. Limitaciones normativas pendientes (no resueltas por los anotadores)

1. La definicion de HS no distingue operativamente "odio" de "ofensa".
2. La definicion de AG no distingue "agresivo" de "vulgar".
3. La relacion formal entre HS y TR (y entre HS y AG) no esta documentada; los datos
   muestran 0 excepciones TR=1 con HS=0 y 22 excepciones AG=1 con HS=0 en desarrollo.
4. No hay instrucciones para casos limite (citas, sarcasmo, discurso reproducido).

Estas limitaciones se registran como preguntas abiertas; **no se resuelven inventando
reglas** en la anotacion.

## 8. Formato de respuesta

Columnas `HS_annotation`, `TR_annotation`, `AG_annotation`: una de
`POSITIVE` / `NEGATIVE` / `UNABLE_TO_DETERMINE` por celda. `uncertainty_notes`: texto
libre opcional. No cambies `review_id`, `text` ni `guideline_version`.
"""
    escribir(RUTA_GUIA, guia)

    filas_prov = [[p["source_path"], p["partition"], p["row_count"], p["eligible_for_review"],
                   p["test_status"], p["reason"]] for p in d["provenance"]]
    plan = f"""# P92 - Plan de revision humana de etiquetas (preparacion del piloto)

Fecha: 2026-10-09 (fecha del proyecto). Fase de PREPARACION: no se ha anotado nada; la
anotacion humana comienza cuando los archivos se entreguen a personas reales. Sin
entrenamiento, sin modificacion de etiquetas, sin test/holdout, sin submissions.

## 1. Motivacion

P91 concluyo NO_CONFIRMED_LABEL_ISSUE con ANNOTATION_POLICY_UNRESOLVED y
GOVERNANCE_REVIEW_REQUIRED, y dejo listo un protocolo y una cola de candidatos. P92
construye el paquete ciego reproducible para dos anotadores independientes. El objetivo es
estimar el desacuerdo humano con las etiquetas originales y estudiar tipos de ambiguedad,
SIN modificar el dataset ni el proceso de validacion.

## 2. Procedencia y elegibilidad (P1)

{tabla(["source_path", "partition", "row_count", "eligible", "test_status", "reason"], filas_prov)}

Elegibilidad demostrada por id: la particion dev (1483) coincide EXACTAMENTE con los ids
del OOF canonico (verificado por interseccion); el holdout (248) queda excluido por
particion; el test no se accedio. La seleccion se construyo a partir de ids elegibles
verificados (no se copio ningun archivo completo al paquete).

## 3. Muestras

- **Componente A (referencia probabilistica)**: {a['tweets']} tweets unicos ({a['celdas']} celdas),
  recuperados de P91 y verificados contra el RNG exacto (semilla {a['semilla']}, muestreo
  aleatorio simple sin reemplazo sobre 1483; probabilidad de inclusion 100/1483 conocida).
- **Componente B (enriquecido)**: {b['tweets']} tweets unicos de {b['disponibles_enriquecidos']} disponibles en la cola P91
  (semilla {d['semillas']['B']}); seleccion por tiers: PILI {b['seleccionados_por_tier']['pili']} +
  DRC {b['seleccionados_por_tier']['drc']} + HCD {b['seleccionados_por_tier']['hcd']} + LSM {b['seleccionados_por_tier']['lsm']}; cap {b['cap_grupo']} tweets por grupo
  near-dup/fusionado; A y B disjuntos. B NO es representativa de la poblacion (seleccion
  deliberada por motivos de revision): no se usa para estimar tasas sin ponderacion.
- **Calibracion**: {cal['tweets']} tweets fuera de A y B (semilla {cal['semilla']}); ronda de familiarizacion; NO
  forma parte de las estimaciones de frecuencia.
- **Principal**: {prin['tweets']} tweets unicos ({prin['celdas']} decisiones binarias por anotador).

## 4. Paquete ciego

`artifacts/human_annotation_v1/`: `annotator_A.csv`, `annotator_B.csv` (mismos {prin['tweets']} casos,
orden aleatorio independiente; semillas {d['semillas']['orden_A']} y {d['semillas']['orden_B']}), `calibration_sample.csv`,
`adjudication_template.csv` ({prin['celdas']} filas sin decisiones), `README_annotators.md`.
Los `review_id` son opacos (`P92R-####`, asignados con semilla {d['semillas']['review_id']}) y permiten alinear
respuestas sin exponer ids de origen. Oculto por completo: ids originales, etiquetas,
probabilidades OOF, predicciones, umbrales, folds, componente, motivos de seleccion y
grupos. Guia aplicable: `p92_annotation_guidelines_v1.md` (version v1).

## 5. Clave interna

`reports/p92_internal_sampling_key.csv` (FUERA del directorio del paquete): vincula cada
`review_id` con el id original, componente, motivos, estrato, etiquetas originales, grupo y
particion. Solo para auditoria y evaluacion posterior.

## 6. Acuerdo y adjudicacion (P6, al recibir respuestas)

Por etiqueta: acuerdo bruto, acuerdo positivo y negativo, Cohen kappa (2 anotadores) y
Krippendorff alpha cuando corresponda, matriz de confusion entre anotadores, tasa de
UNABLE_TO_DETERMINE y acuerdo exacto de la combinacion HS/TR/AG. Limitaciones: con
prevalencias extremas (TR positivo 30 por ciento; AG sin HS raro) kappa se degrada; nunca
se resume la calidad con una sola metrica. Adjudicacion: plantilla separada; conservar
ambas respuestas; sin reglas automaticas; sin mayoria artificial; tercer revisor
cualificado cuando sea posible; UNABLE_TO_DETERMINE se conserva si no hay decision.

## 7. Niveles de evidencia (P7)

NOT_ANNOTATED -> INDEPENDENT_ANNOTATIONS_COMPLETE -> DISAGREEMENT_REVIEW_PENDING ->
ADJUDICATED -> SILVER_REFERENCE_V1 -> HIGH_CONFIDENCE_ADJUDICATED_REFERENCE. No se
denomina "gold standard" al resultado; no se reemplazan las etiquetas originales; las
capas (original, anotador A, anotador B, adjudicado, estado) se conservan separadas.

## 8. Plan analitico posterior (P8, no ejecutado)

Componentes separados (A probabilistica; B enriquecida; C calibracion). Margenes P91-R
verificados: por etiqueta +-4.52 pp (p=0.05, DEFF=1.2); +-6.22 pp (p=0.10, DEFF=1.2);
agregado de 300 celdas +-2.61 pp. DEFF sujeto a sensibilidad; dependencia entre etiquetas
del mismo tweet considerada; numeradores y denominadores siempre; el conjunto enriquecido
no se generaliza a la poblacion; no se usa para seleccionar modelos o parametros.

## 9. Revision de etiquetas originales (P9)

Estados separados: MODEL_DISAGREEMENT, ANNOTATOR_DISAGREEMENT, POSSIBLE_LABEL_ISSUE,
ADJUDICATED_DIFFERENCE, CONFIRMED_ANNOTATION_ERROR, APPROVED_DATASET_CORRECTION. Una
correccion confirmada requiere regla aplicable + anotaciones independientes + adjudicacion
documentada + decision explicita + autorizacion de gobernanza. En P92 NO se modifica
ninguna etiqueta.

## 10. Estado del arte (P10)

Se reutiliza `reports/p91_label_state_of_art.md` (confident learning, deteccion de errores
de anotacion, acuerdo, cegamiento). No se repite la busqueda; no se instalan herramientas;
no se ejecuta confident learning.

## 11. Limitaciones del piloto

Guia provisional sin criterios operativos oficiales; B no representativa; dependencia por
grupos; dos anotadores como minimo (la adjudicacion requiere un tercero cualificado);
ninguna conclusion de "gold" ni de correccion sin autorizacion separada.
"""
    escribir(RUTA_PLAN, plan)

    sampling = {
        "pilot": "P92 human label review",
        "date": "2026-10-09",
        "guideline_version": "v1",
        "population_definition": "tweets del desarrollo autorizado (1483) con etiquetas originales",
        "eligible_partitions": ["dev"],
        "excluded_partitions": ["test (no accedido)", "holdout (no accedido)", "holdout2 (no accedido)"],
        "sampling_frame": "data/processed/holdout_v2_ids.csv (particion dev); ids verificados contra features/oof_v03_final.npz",
        "sampling_unit": "tweet (3 decisiones por tweet: HS, TR, AG)",
        "sample_size": {"A_referencia": a["tweets"], "B_enriquecido": b["tweets"],
                        "calibracion": cal["tweets"], "principal": prin["tweets"]},
        "sampling_probability": {"A": "100/1483 (SRS sin reemplazo, conocida)",
                                 "B": "no probabilistica (seleccion por evidencia; no usar sin ponderacion)",
                                 "calibracion": "SRS de 15 sobre dev fuera de A y B"},
        "random_seeds": d["semillas"],
        "stratification": {"A": "ninguna (SRS)", "B": "tiers PILI/DRC/HCD/LSM con round-robin por etiqueta en HCD",
                           "grupos": f"cap {b['cap_grupo']} tweets por grupo near-dup/fusionado"},
        "group_handling": "cap 2 por grupo; dependencia residual documentada",
        "precision_assumptions": {"por_etiqueta_p005_deff12": "+-4.52 pp (N=1483, n=100)",
                                  "por_etiqueta_p010_deff12": "+-6.22 pp",
                                  "agregado_celdas_p005_deff12": "+-2.61 pp (N=4449, n=300)",
                                  "nota": "calculos condicionados a supuestos (P91-R); no precision garantizada"},
        "limitations": ["B no representativa de la poblacion",
                        "dependencia por grupos/pseudo-autor",
                        "guia provisional sin criterios operativos oficiales",
                        "no es gold standard; sin adjudicacion no hay etiqueta final"],
        "hashes": d["hashes"],
        "package_dir": "artifacts/human_annotation_v1",
        "internal_key": "reports/p92_internal_sampling_key.csv (fuera del paquete)",
    }
    escribir(RUTA_SAMPLING, json.dumps(sampling, ensure_ascii=False, indent=1) + "\n")

    readme = f"""# Revision humana de etiquetas - instrucciones para anotadores (P92, guia v1)

Gracias por participar. Tu tarea: leer tweets y decidir, de forma independiente, si
presentan **discurso de odio (HS)**, si el discurso esta **dirigido contra una persona o
grupo (TR)** y si el texto es **agresivo (AG)**.

## Archivos

1. `calibration_sample.csv` (15 textos): ronda de familiarizacion. Anotalo completo ANTES
   de comentarlo con la otra persona anotadora.
2. Tu archivo personal (`annotator_A.csv` o `annotator_B.csv`, 250 textos): la tarea
   principal. NO lo comentes con la otra persona hasta terminar tu primera ronda.

## Como responder

En cada fila, escribe en `HS_annotation`, `TR_annotation` y `AG_annotation` una de estas
tres respuestas EXACTAS (mayusculas): `POSITIVE`, `NEGATIVE`, `UNABLE_TO_DETERMINE`.

- Decide cada etiqueta por separado.
- `UNABLE_TO_DETERMINE` = no puedes decidir con responsabilidad con lo que tienes (p. ej.
  falta el tweet al que responde). No es una etiqueta mas: es una declaracion de
  incertidumbre.
- En `uncertainty_notes` puedes anotar el motivo (opcional, breve).
- No cambies las columnas `review_id`, `text` ni `guideline_version`; no modifiques el
  texto ni el orden de las filas.

## Definiciones (las unicas disponibles)

- HS: "indica si hay discurso de odio".
- TR: "indica si el discurso esta dirigido contra una persona o grupo".
- AG: "indica si el texto es agresivo".

Son definiciones breves y no hay criterios operativos oficiales: no hay respuestas
"trampa". Si dudas, usa `UNABLE_TO_DETERMINE` y anota por que.

## Reglas

- NO consultes las etiquetas originales, predicciones de modelos ni ningun material del
  proyecto: tu juicio debe ser independiente.
- NO busques los tweets en internet.
- NO uses herramientas automaticas (LLM, clasificadores) para decidir.
- NO discutas casos con la otra persona anotadora antes de terminar tu ronda completa.
- Los desacuerdos no se resuelven aqui: se conservan ambas respuestas y se adjudican
  despues.

## Entrega

Devuelve el CSV completo con tus respuestas. No borres filas ni columnas. Si tienes dudas
sobre el formato, pregunta antes de empezar.

La guia completa esta en `p92_annotation_guidelines_v1.md` (repositorio del proyecto).
"""
    escribir(RUTA_README, readme)

    archivos = ["annotator_A.csv", "annotator_B.csv", "calibration_sample.csv",
                "adjudication_template.csv", "README_annotators.md"]
    with (DIR_PAQUETE / "annotator_A.csv").open(newline="") as fh:
        n_a = len(list(csv.DictReader(fh)))
    manifest = {
        "package": "human_annotation_v1",
        "date": "2026-10-09",
        "guideline_version": "v1",
        "counts": {"annotator_rows": n_a, "principal_tweets": prin["tweets"],
                   "cells_per_annotator": prin["celdas"], "calibration_tweets": cal["tweets"],
                   "adjudication_rows": prin["celdas"]},
        "blinding": {"hidden": ["original_example_id", "original_labels", "oof_probabilities",
                                "v03_predictions", "threshold_distances", "fold",
                                "sample_component", "selection_reasons", "dup_groups",
                                "other_annotator_responses"],
                     "visible": ["review_id", "text", "empty annotation columns",
                                 "uncertainty_notes", "guideline_version"]},
        "random_seeds": d["semillas"],
        "files": {nombre: sha256_de(DIR_PAQUETE / nombre) for nombre in archivos},
        "internal_key_separate": "reports/p92_internal_sampling_key.csv",
        "limitations": ["guia provisional v1", "B no representativa",
                        "sin adjudicacion aun; ningun gold standard"],
    }
    escribir(RUTA_MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")

    for ruta in (RUTA_GUIA, RUTA_PLAN, RUTA_SAMPLING, RUTA_README, RUTA_MANIFEST):
        if any(b > 127 for b in ruta.read_bytes()):
            raise SystemExit(f"STOP: bytes > 127 en {ruta.name}")
    print("P92 parte 2 completada: guia, plan, sampling, README y manifiesto del paquete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
