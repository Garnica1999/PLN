"""P91-02: informe de auditoria de etiquetas, protocolo de revision, estado del arte y gap.

Lee `reports/p91_audit_results.json` y produce:

- reports/p91_label_quality_audit.md (auditoria principal);
- reports/p91_label_review_protocol.md (protocolo de revision humana ciega);
- reports/p91_label_state_of_art.md (sintesis de literatura verificada);
- reports/p91_annotation_gap_analysis.csv (cruce con el historial).

Sin entrenamiento, sin modificacion de etiquetas, sin test/holdout, sin submissions.

Uso:
    /home/carlos/venv/bin/python experiments/p91_02_informe_protocolo.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

RAIZ: Path = Path(__file__).resolve().parents[1]
RUTA_RESULTS = RAIZ / "reports" / "p91_audit_results.json"
RUTA_AUDIT = RAIZ / "reports" / "p91_label_quality_audit.md"
RUTA_PROTO = RAIZ / "reports" / "p91_label_review_protocol.md"
RUTA_SOTA = RAIZ / "reports" / "p91_label_state_of_art.md"
RUTA_GAP = RAIZ / "reports" / "p91_annotation_gap_analysis.csv"


def escribir(ruta: Path, texto: str) -> None:
    """Escribe texto exigiendo ASCII puro."""
    if any(ord(c) > 127 for c in texto):
        raise SystemExit(f"STOP: bytes > 127 en {ruta.name}")
    ruta.write_text(texto, encoding="utf-8")


def main() -> int:
    """Genera los cuatro entregables de P91."""
    for ruta in (RUTA_AUDIT, RUTA_PROTO, RUTA_SOTA, RUTA_GAP):
        if ruta.exists():
            raise SystemExit(f"STOP: ya existe {ruta.name}")
    d = json.loads(RUTA_RESULTS.read_text(encoding="utf-8"))
    c = d["conteos_por_etiqueta"]
    u = d["unanimidad_4modelos"]
    v = d["vecindad"]
    dup = d["duplicados"]
    cola = d["cola"]

    # ---------------- Auditoria principal ----------------
    audit = f"""# P91 - Auditoria de calidad de etiquetas (label quality research)

Fecha: 2026-10-10. Fase diagnostica y metodologica: NO entrena, NO modifica etiquetas,
NO usa test ni holdouts, NO genera submissions. Continua desde el cierre de P90.03-R/P90.04.

## 1. Resumen ejecutivo

**Decision combinada: `ANNOTATION_POLICY_UNRESOLVED` + `GOVERNANCE_REVIEW_REQUIRED` +
`REVIEW_QUEUE_PREPARED` + `HUMAN_ANNOTATION_PENDING`, con `NO_CONFIRMED_LABEL_ISSUE` segun
la evidencia disponible.** Las reglas oficiales existen pero son definiciones de una linea
sin criterios operativos ni guia de anotadores; la relacion formal TR=>HS **no esta
documentada** (solo una regularidad empirica perfecta: 0 excepciones en 1731 filas) y AG
presenta 22 excepciones AG=1 con HS=0 en dev que contradicen la lectura condicionada. La
evidencia computacional **no confirma** un problema de etiquetas: solo 0-4 celdas por
etiqueta tienen evidencia OOF extrema contra el gold (p<=0.05 / p>=0.95), la contradiccion
en near-duplicados es 2.77-2.89 por ciento (ya medida en P87/V4-A01, sin valor de
modelado), y las intervenciones historicas sobre esas filas son +-0.0005. Se preparo una
cola de revision ciega ({cola['celdas']} celdas; {cola['muestra_referencia_tweets']} tweets
de referencia aleatoria) y un protocolo reproducible, pero **la revision humana no se
ejecuta**: falta autorizacion de gobernanza (P87-00 la declaro LOW_EXPECTED_VALUE y no la
solicito; el silencio de las reglas no es autorizacion).

## 2. Estado del repositorio y alcance

Git verificado al inicio: rama `main`, HEAD `e43c594`, 1 archivo modificado (notebook) +
52 sin seguimiento; los cambios locales de P90.02-P90.04 se conservan. No se ejecuto
entrenamiento, no se genero ninguna prediccion nueva, no se toco `toxic_test`, holdout ni
holdout2, y no se uso el leaderboard. Datos usados: solo desarrollo (1483 filas) y
artefactos OOF historicos (hashes en la seccion 7).

## 3. P1 - Reglas oficiales de anotacion

**Fuente oficial mas cercana** (unica disponible en el repositorio):
`docs/competition_rules.md` (texto verbatim de la pagina Overview de la competencia,
capturado 2026-09-22; organizador: curso PLN, Maestria en IA, PUJ).

| label | official_definition (verbatim) | criteria extraibles | source |
|---|---|---|---|
| HS | "indica si hay discurso de odio (1 = si, 0 = no)" | presencia de discurso de odio en el texto | competition_rules.md |
| TR | "indica si el discurso esta dirigido contra una persona o grupo (1 = si, 0 = no)" | dirigido a persona O grupo (ambos cuentan) | competition_rules.md |
| AG | "indica si el texto es agresivo (1 = si, 0 = no)" | agresividad del texto | competition_rules.md |

**No existen en el repositorio instrucciones originales para anotadores ni criterios
operativos** (edge cases, requisitos de contexto, unidad de anotacion): la busqueda
exhaustiva en `docs/` y en el proyecto no encontro ninguna. Clasificacion de la evidencia:

- `OFFICIAL_RULE`: las tres definiciones de una linea y el formato de los datos.
- `DOCUMENTED_INTERPRETATION`: D-001 ("TR y AG son atributos condicionados a que exista
  discurso de odio") y D-005 (TR oficial vs HatEval: "dirigido contra una persona o
  grupo" no distingue persona de grupo; HatEval si); D-005 instruye "no asumir el
  significado de TR".
- `EMPIRICAL_PATTERN`: TR=1 solo dentro de HS=1 (519/519 en train, 0 excepciones en
  1731; 445/445 en dev); AG=1 con HS=0 en 26 filas de train (22 en dev, 1.5 por ciento);
  mencion inicial (V de Cramer 0.512) y referencia colectiva (57.0 por ciento en TR=0
  frente a 10.0 por ciento en TR=1) caracterizan TR empiricamente (D-005).
- `UNRESOLVED_AMBIGUITY`: criterios operativos de HS ("discurso de odio" vs ofensa),
  de AG ("agresivo" vs vulgar) y la relacion formal entre HS y TR/AG.

## 4. P5 - Relacion formal entre HS y TR

| caso | que dicen las reglas oficiales | evidencia empirica (train 1731 / dev 1483) |
|---|---|---|
| HS=0, TR=1 | no prohibido ni permitido explicitamente; nada en el texto oficial | 0 casos (0/519 TR positivos) |
| HS=1, TR=0 | compatible (odio no dirigido a persona/grupo) | 203 celdas dev (79+124) |
| HS=1, TR=1 | caso central | 445 celdas dev |
| HS=0, TR=0 | caso mayoritario | 813 + 22 (AG solo) dev |

1. Documentada expresamente: **NO** (ningun documento oficial la enuncia).
2. Derivada necesariamente de las definiciones: **NO** (las tres definiciones son
   independientes; nada implica TR=>HS formalmente).
3. Excepciones conceptuales permitidas: **el reglamento guarda silencio** (no se puede
   afirmar que TR=1&HS=0 este prohibido; tampoco lo contrario).
4. Ejemplos suficientes: **SI** para la coherencia empirica (0 excepciones en 1731 filas;
   la regularidad es perfecta en los datos, pero sigue siendo un patron, no una regla).
5. P87 resolvio la pregunta: **NO**; la bloqueo por gobernanza (LOW_EXPECTED_VALUE) y dejo
   el cierre documental de D-005 como beneficio no ejecutado.

**Decision: `DOCUMENTATION_INSUFFICIENT`.** Pregunta exacta para los responsables del
dataset (no se contacta a nadie en esta fase): "Las etiquetas TR y AG, se anotaron
unicamente sobre textos ya positivos en HS (anotacion condicionada), o el esquema permite
TR=1 o AG=1 con HS=0? Si existe esa regla, cual es el texto normativo? Cuales son los
criterios operativos de HS (que distingue odio de ofensa) y de AG (agresividad de
vulgaridad)?" Las 22 excepciones AG=1&HS=0 sugieren que AG no estuvo estrictamente
condicionada; las 0 excepciones TR sugieren lo contrario para TR. No se modifica ninguna
etiqueta para satisfacer la jerarquia inferida.

## 5. P2 - Diagnostico computacional (evidencia, no veredictos)

Poblacion: desarrollo autorizado (1483 tweets; 4449 celdas instancia-etiqueta). Fuentes:
OOF canonico (V03), 4 modelos base historicos (`v2_oof_base_corrected.npz`),
`label_consistency_v4.npz` y grupos historicos. Todo en
`reports/p91_label_quality_diagnostics.csv` (40 filas con numerador, denominador, fuente
y limitaciones).

| diagnostico | HS | TR | AG |
|---|---|---|---|
| marginales positivos | 648 | 445 | 567 |
| FN con p<=0.05 (evidencia extrema contra gold) | {c['HS']['fn_p_le_0.05']} | {c['TR']['fn_p_le_0.05']} | {c['AG']['fn_p_le_0.05']} |
| FN con p<=0.10 | {c['HS']['fn_p_le_0.10']} | {c['TR']['fn_p_le_0.10']} | {c['AG']['fn_p_le_0.10']} |
| FP con p>=0.95 | {c['HS']['fp_p_ge_0.95']} | {c['TR']['fp_p_ge_0.95']} | {c['AG']['fp_p_ge_0.95']} |
| FP con p>=0.90 | {c['HS']['fp_p_ge_0.90']} | {c['TR']['fp_p_ge_0.90']} | {c['AG']['fp_p_ge_0.90']} |
| celdas cerca del threshold (|m|<0.025) | {c['HS']['cerca_threshold']} | {c['TR']['cerca_threshold']} | {c['AG']['cerca_threshold']} |
| desacuerdo unanime de los 4 modelos | {u['HS']} | {u['TR']} | {u['AG']} |
| discrepancia con vecinos (label_consistency_v4) | {v['discrepancia_por_etiqueta']['HS']} | {v['discrepancia_por_etiqueta']['TR']} | {v['discrepancia_por_etiqueta']['AG']} |

- Estructura: TR=1&HS=0 = {d['tr_sin_hs']}; AG=1&HS=0 = {d['ag_sin_hs']}; combinaciones
  completas en el CSV.
- Duplicados: near-dup 1 grupo divergente / 3 filas (de {dup['near_dup_grupos_no_unitarios']} grupos no
  unitarios); fusionados 13 grupos / {dup['fused_filas_div']} filas divergentes (de {dup['fused_grupos_no_unitarios']} no
  unitarios); vecindad de consistencia: {v['en_cluster_dev']} tweets dev en algun cluster.
- Histograma de desacuerdo 0-4 modelos: {d['desacuerdo_histograma']}.

**Que puede inferirse**: que la evidencia extrema contra el gold es ESCASA (4 celdas
FN<=0.05 y 4 FP>=0.95 en total), que la contradiccion textual demostrable es marginal
(2.77-2.89 por ciento del train; P87) y que el desacuerdo unanime de 4 modelos
correlacionados ({sum(u.values())} celdas, 11.1 por ciento) es en su mayoria error de
modelo, no senal de etiqueta. **Que NO puede inferirse**: la tasa de error de anotacion
(no hay adjudicacion humana), la semantica pretendida por los autores (reglas
insuficientes) ni que corregir etiquetas mejoraria el score (sin experimento valido).
La distancia al threshold no es una probabilidad de error de anotacion.

## 6. P7 - Gap analysis historico (resumen)

Tabla completa en `reports/p91_annotation_gap_analysis.csv`. Sintesis: la revision humana
de TR ya fue auditada y **bloqueada por gobernanza** (P87-00/D-292: LOW_EXPECTED_VALUE;
protocolo disenado, no ejecutado; material con 10 filas del holdout que debe filtrarse a
dev); el confident learning ya se probo **en el proyecto** (P59-C: MARGINAL, el estimador
de ruido no supera al control de dificultad, AUC 0.973019 vs 0.974143); la tolerancia a
ruido por reweighting esta cerrada (V30/V4-B03/P69); las intervenciones sobre filas
contradictorias miden +-0.0005 y el techo de cualquier intervencion de etiqueta (~+0.009,
P57) esta por debajo de la banda de particion (0.017). P91 agrega: muestreo con FPC,
separacion referencia/enriquecido, candidatos basados en evidencia exacta multi-modelo y
las metricas de acuerdo - una mejora de protocolo, no una tecnica nueva.

## 7. Hashes de los artefactos usados

{json.dumps(d['hashes'], ensure_ascii=False, indent=1)}

## 8. Estado de gobernanza y decision recomendada

- Las reglas no autorizan ni prohiben revisar/corregir etiquetas de train (silencio).
  La autorizacion para cualquier revision humana **no existe** y no se solicita aqui.
- Decisiones finales de la fase (combinables, por aspectos independientes):
  `ANNOTATION_POLICY_UNRESOLVED` + `GOVERNANCE_REVIEW_REQUIRED` (relacion HS/TR y
  criterios operativos), `REVIEW_QUEUE_PREPARED` + `HUMAN_ANNOTATION_PENDING` (cola y
  protocolo listos, revision no ejecutada), `NO_CONFIRMED_LABEL_ISSUE` (con la evidencia
  actual no hay ningun error de anotacion confirmado).
- Siguiente paso recomendado: mantener V03 congelado y la fase de modelado cerrada; si
  Carlos decide elevar la pregunta de gobernanza al organizador, el protocolo de
  `p91_label_review_protocol.md` queda listo para ejecutarse bajo autorizacion.
"""
    escribir(RUTA_AUDIT, audit)

    # ---------------- Protocolo ----------------
    proto = f"""# P91 - Protocolo de revision humana ciega de etiquetas (disenado, NO ejecutado)

Estado: **PENDIENTE DE AUTORIZACION DE GOBERNANZA**. Este documento permite ejecutar la
revision de forma independiente y reproducible cuando exista autorizacion. El agente no
sustituye la adjudicacion humana. Material: desarrollo autorizado (1483 filas); el
holdout, holdout2 y el test no se usan jamas.

## 1. Poblacion, marco y unidad

- `population_definition`: tweets del desarrollo autorizado (1483) con etiquetas
  originales; `eligible_partitions`: dev (excluye holdout/holdout2/test).
- `sampling_frame`: ids de dev en `data/processed/holdout_v2_ids.csv` (particion dev).
- `sampling_unit`: tweet (el anotador ve el texto completo y anota HS, TR y AG).
- `stratification`: ninguna obligatoria; opcional por combinacion HS/TR/AG para cubrir
  las combinaciones raras 001 (22) y 110 (24) si la gobernanza lo pide.
- `group_handling`: las unidades en grupos near-dup/fusionados son evidencia dependiente;
  para prevalencias se usa el tweet como unidad (los grupos se reportan aparte); en la
  cola enriquecida se incluyen las filas divergentes con su identificador de grupo.

## 2. Componentes de la muestra (separadas, no mezcladas para prevalencias)

**A. Muestra probabilistica de referencia.** Muestreo aleatorio simple de 100 tweets de
los 1483 (semilla fija {cola['semilla']}; probabilidad de inclusion conocida 100/1483;
las 3 etiquetas de cada tweet entran como celdas). Permite estimar la frecuencia de
problemas con ponderaciones correctas. Tamano justificado en la seccion 3.

**B. Muestra enriquecida con candidatos.** Cola de {cola['celdas']} celdas
(`reports/p91_label_review_candidates.csv`) con motivos verificables:
`HIGH_CONFIDENCE_OOF_DISAGREEMENT` ({cola['razones'].get('HIGH_CONFIDENCE_OOF_DISAGREEMENT', 0)}),
`POSSIBLE_INSTANCE_LABEL_INCONSISTENCY` ({cola['razones'].get('POSSIBLE_INSTANCE_LABEL_INCONSISTENCY', 0)}),
`DOCUMENTED_RULE_CONFLICT` ({cola['razones'].get('DOCUMENTED_RULE_CONFLICT', 0)}),
`LABEL_SPECIFIC_MARGIN` ({cola['razones'].get('LABEL_SPECIFIC_MARGIN', 0)}),
`RANDOM_REFERENCE_SAMPLE` ({cola['razones'].get('RANDOM_REFERENCE_SAMPLE', 0)}).
Topes por componente documentados (cap {cola['cap_enriquecido_por_componente']}); ninguna
celda se presenta como error confirmado (`annotation_status = pending_authorization`).

## 3. Tamano muestral (justificacion, correccion por poblacion finita)

Para una prevalencia esperada p en celdas (4449), con nivel 95 por ciento y FPC:
`n = N*z^2*p(1-p) / (z^2*p(1-p) + (N-1)*e^2)`.

- p=0.05, e=0.03, N=4449: n ~ 194 celdas; con efecto de diseno 1.2 ~ 233 celdas; los 100
  tweets de referencia aportan 300 celdas -> semi-ancho esperado +-2.6 puntos porcentuales.
- A nivel tweet (N=1483, p~0.10, n=100): semi-ancho ~ +-5.7 puntos porcentuales (FPC
  0.933); suficiente para una primera tasa, no para estimaciones finas.
- `precision_assumptions`: p desconocida; se reportan ambos escenarios; cualquier cambio
  de tamano debe fijarse ANTES de ver resultados.
- `limitations`: dependencia por grupos y pseudo-autor; desacuerdo humano no es error.

## 4. Cegamiento

Se muestra al anotador: texto original completo; definiciones oficiales verbatim
(`docs/competition_rules.md`) y ejemplos de referencia autorizados; los campos de
anotacion; la opcion `unable_to_determine` (si la gobernanza la aprueba). NO se muestra:
etiquetas originales, probabilidades OOF, predicciones, distancia al threshold, features
del logit, motivo de seleccion, identidad de otros anotadores. Orden de ejemplos
aleatorizado por anotador. Los textos NO se modifican ni se recortan.

## 5. Anotacion y adjudicacion

- Dos anotadores independientes como minimo; cada etiqueta por separado
  (positive / negative / unable_to_determine); sin discusion antes de cerrar la ronda.
- Adjudicacion separada para discrepancias, con justificacion escrita y sin acceso a las
  etiquetas originales.
- Con un solo anotador, la revision es preliminar y no se presenta como consenso.
- Estados (no equivalentes): `possible label issue` -> `confirmed annotation error`
  (adjudicado) -> `adjudicated label` -> `approved dataset correction` (requiere
  autorizacion separada).

## 6. Medicion del acuerdo

Por etiqueta: acuerdo bruto; acuerdo positivo y negativo; Cohen kappa (dos anotadores) o
Krippendorff alpha nominal (mas de dos; admite falta de datos); matrices de confusion
entre anotadores; frecuencia de `unable_to_determine`; acuerdo exacto de la combinacion
HS/TR/AG. **Limitaciones declaradas**: kappa se degrada con prevalencias extremas
(TR=1 en 30 por ciento; AG=1 con HS=0 es raro) - se reporta junto a los acuerdos
positivo/negativo, nunca solo. Desacuerdo humano != etiqueta original incorrecta
(puede indicar definicion ambigua).

## 7. Flujo y trazabilidad

1. Autorizacion de gobernanza (documentada). 2. Construccion del paquete ciego
(`review_id` por celda; textos desde `toxic_train.csv`; clave de evaluacion SEPARADA con
gold y evidencia). 3. Anotacion independiente. 4. Adjudicacion. 5. Analisis de acuerdo y
tasas. 6. Informe con estados por celda. 7. Solo con autorizacion separada: politica de
correccion versionada (solo train; nunca test/holdout).

## 8. Criterios de decision (P6)

Seis preguntas separadas: (1) hay desacuerdo entre anotadores; (2) la etiqueta original
contradice una regla oficial; (3) hay error de anotacion confirmado; (4) los errores
afectan una fraccion relevante; (5) una correccion podria cambiar el aprendizaje;
(6) podria evaluarse de forma valida. Ninguna implica la siguiente.

Gates para una eventual correccion: autorizacion; anotacion ciega; adjudicacion
independiente; politica documentada; cambios SOLO en train; validacion anidada o por
grupos; analisis por pares (celda a celda); bootstrap/incertidumbre; reproducibilidad.
Criterios numericos fijados ANTES de mirar resultados (p. ej., proceder solo si la tasa
de errores adjudicados supera el umbral que la banda de particion 0.017 exige y el analisis
por pares es favorable).

## 9. Requisitos pendientes

Autorizacion de gobernanza (no solicitada; P87-00 la declaro LOW_EXPECTED_VALUE); reglas
operativas de anotacion (no disponibles); decision sobre el uso de
`unable_to_determine`. Sin ellos, este protocolo no se ejecuta.
"""
    escribir(RUTA_PROTO, proto)

    # ---------------- Estado del arte ----------------
    sota = """# P91 - Estado del arte metodologico (verificado)

Fecha de acceso: 2026-10-10. Busqueda dirigida a UNA decision: como auditar la calidad de
etiquetas multietiqueta con procedimiento ciego y reproducible. No es una busqueda de
arquitecturas. Cada referencia se verifico contra su fuente primaria (ACL Anthology,
JAIR/MIT Press, arXiv API); no se cita ningun trabajo no verificado.

## 1. Confident learning (estimacion de ruido con probabilidades fuera de muestra)

- Northcutt, C. G.; Jiang, L.; Chuang, I. L. (2021). "Confident Learning: Estimating
  Uncertainty in Dataset Labels". Journal of Artificial Intelligence Research (JAIR),
  70:1373-1411. DOI 10.1613/jair.1.12125 (verificado). Revisado por pares.
- Metodo: estima la distribucion conjunta etiqueta dada x etiqueta verdadera (matriz
  confident joint) usando solo probabilidades fuera de muestra y las etiquetas dadas;
  identifica y ordena posibles errores.
- Supuestos: proceso de ruido condicional a la clase; requiere probabilidades OOF bien
  calibradas en el sentido de rango; no exige etiquetas limpias de referencia.
- Multietiqueta: la adaptacion natural es por par instancia-etiqueta (extension por
  etiqueta binaria); la documentacion de cleanlab (herramienta, no paper) ofrece
  `multilabel_classification` con esa lectura y advierte que NO modela etiquetas
  verdaderas omitidas (falsos negativos de anotacion). URL:
  https://docs.cleanlab.ai (acceso 2026-10-10; documentacion de herramienta).
- Limites relevantes aqui: la confianza del modelo no es certeza humana; con 4 modelos
  muy correlacionados el desacuerdo no es independiente.
- Cambio de decision que aporta: fija el criterio de ordenacion de candidatos
  (self-confidence por celda) y su limite. **Precedente interno: P59-C ya implemento
  confident learning desde cero (sin cleanlab) y el resultado fue MARGINAL** (el estimador
  de ruido no supero al control de dificultad: AUC 0.973019 vs 0.974143). No se instala ni
  ejecuta cleanlab.

## 2. Errores de etiqueta a escala (motivacion, no prueba local)

- Northcutt, C. G.; Athalye, A.; Mueller, J. (2021). "Pervasive Label Errors in Test Sets
  Destabilize Machine Learning Benchmarks". NeurIPS 2021 Track on Datasets and Benchmarks
  (arXiv 2103.14749; estado verificado). Revisado por pares.
- Aporte: los errores de etiqueta son frecuentes en datasets populares y afectan rankings.
- Uso en TOXIC: justifica auditar, NO demuestra que este dataset tenga el mismo problema.

## 3. Deteccion de errores de anotacion en NLP (metodos y evaluacion)

- Klie, J.-C.; Webber, B.; Gurevych, I. (2023). "Annotation Error Detection: Analyzing the
  Past and Present for a More Coherent Future". Computational Linguistics, 49(1):157-198.
  DOI 10.1162/coli_a_00464 (verificado en ACL Anthology, 2023.cl-1.4).
- Aporte: reimplementa 18 metodos, evaluacion uniforme y mejores practicas; ninguno
  domina; recomienda protocolos ciegos y analisis por pares.
- Cambio de decision: fija que la cola use MULTIPLES criterios (no un solo detector) y que
  la evaluacion de la revision se haga por pares con tasas, no con una metrica unica.

## 4. Consenso con multiples anotadores

- Goh, H. W.; Tkachenko, U.; Mueller, J. (2022). "CROWDLAB: Supervised learning to infer
  consensus labels and quality scores for data with multiple annotators". NeurIPS 2022
  Human-in-the-Loop Learning Workshop (arXiv 2210.06812; estado verificado: workshop).
- Aporte: combina predicciones del modelo y anotaciones para consenso y calidad por
  ejemplo; util DESPUES de la revision, cuando existan 2+ anotaciones.
- Cambio de decision: la agregacion de la revision no sera mayoria simple a ciegas; se
  reporta el acuerdo completo y el consenso ponderado como analisis secundario.

## 5. Sesgo de anotacion en discurso de odio (diseno ciego)

- Sap, M.; Card, D.; Gabriel, S.; Choi, Y.; Smith, N. A. (2019). "The Risk of Racial Bias
  in Hate Speech Detection". ACL 2019, pp. 1668-1678 (P19-1163, verificado). Revisado por
  pares.
- Aporte: el contexto del anotador (priming de dialecto/raza) cambia las etiquetas de
  toxicidad; los desacuerdos reflejan a veces sesgo del anotador, no error del texto.
- Cambio de decision: cegamiento estricto, sin priming, y prohibicion de discutir casos
  antes de cerrar la primera ronda; el desacuerdo se interpreta con cautela.

## 6. Sintesis y decision

La literatura no cambia la metodologia que este proyecto recomendaria: la revision debe
ser ciega, con 2+ anotadores, adjudicacion separada, acuerdo por etiqueta con metricas
robustas a prevalencia, y deteccion de candidatos multicriterio con probabilidades OOF.
Los aportes concretos son: (i) la formalizacion de la tarea y mejores practicas de Klie
et al. 2023; (ii) el limite de confident learning (no modela etiquetas omitidas) y su
precedente interno MARGINAL (P59-C); (iii) el diseno ciego y sin priming (Sap et al.
2019); (iv) CROWDLAB solo aplica DESPUES de obtener multiples anotaciones. No se busca mas
literatura: ninguna decision pendiente depende de ello.
"""
    escribir(RUTA_SOTA, sota)

    # ---------------- Gap analysis CSV ----------------
    filas_gap = [
        {"hypothesis": "Revision humana de ambiguedades de anotacion en TR (cierre documental de D-005)",
         "historical_decision": "P87-00 / D-292: LOW_EXPECTED_VALUE; no se solicito autorizacion",
         "historical_result": "Protocolo disenado y no ejecutado; material historico con 10 filas del holdout (filtrar a dev: 140 prioritarias); techo de intervencion ~+0.009 < banda 0.017; intervenciones medidas +-0.0005",
         "remaining_question": "Tasa real de ambiguedad en texto NO duplicado; kappa de TR; cierre documental de D-005",
         "new_evidence_needed": "Revision ciega con 2+ anotadores y adjudicacion (requiere autorizacion)",
         "methodological_novelty": "P91 refina el protocolo de P87 (muestreo con FPC, componentes referencia/enriquecido, candidatos con evidencia exacta multi-modelo, acuerdo por etiqueta); misma familia de auditoria",
         "governance_status": "BLOCKED: sin autorizacion; el silencio de las reglas no autoriza",
         "recommendation": "COMPLEMENTARY"},
        {"hypothesis": "Confident learning / estimacion de ruido con probabilidades OOF",
         "historical_decision": "P59-C: implementado desde cero (sin cleanlab)",
         "historical_result": "MARGINAL: el estimador de ruido no supera al control de dificultad (AUC 0.973019 vs 0.974143)",
         "remaining_question": "Ninguna nueva: cleanlab multietiqueta es la misma familia y no modela etiquetas omitidas",
         "new_evidence_needed": "Ninguna que cambie la decision",
         "methodological_novelty": "Nula (misma tecnica; P91 solo la usa como criterio de ordenacion)",
         "governance_status": "PERMITTED (diagnostico)",
         "recommendation": "DUPLICATE"},
        {"hypothesis": "Tolerancia a ruido por reweighting / downweight de filas sospechosas",
         "historical_decision": "Familia noise CLOSED (V30, V4-B03, P69; candidato A_cost_sensitive adoptado y cerrado)",
         "historical_result": "Resultados negativos o marginales; techo por debajo de la banda",
         "remaining_question": "Ninguna",
         "new_evidence_needed": "Ninguna",
         "methodological_novelty": "Nula",
         "governance_status": "PERMITTED pero cerrado",
         "recommendation": "CLOSED"},
        {"hypothesis": "Correccion de contradicciones en grupos de texto casi duplicado",
         "historical_decision": "V4-A01/A02: medida la contradiccion (HS/TR 2.773 por ciento, AG 2.889 por ciento del train); P90.02 documento los grupos divergentes en dev",
         "historical_result": "El error de V03 es MENOR dentro de los clusters contradictorios (0.023 vs 0.113); intervenciones +-0.0005; 43 filas dev informativas",
         "remaining_question": "Solo descriptiva: los grupos divergentes quedan en la cola de revision",
         "new_evidence_needed": "Ninguna para modelado",
         "methodological_novelty": "Nula",
         "governance_status": "PERMITTED (diagnostico)",
         "recommendation": "CLOSED"},
        {"hypothesis": "Relacion formal TR=1 => HS=1 como regla de anotacion",
         "historical_decision": "D-001 (interpretacion condicionada) y D-005 (TR oficial vs HatEval): caracterizacion empirica, sin regla formal",
         "historical_result": "519/519 en train y 445/445 en dev; 0 excepciones; la regla formal no esta en ningun documento",
         "remaining_question": "Anotacion condicionada si/no; texto normativo; criterios operativos de HS",
         "new_evidence_needed": "Aclaracion de los responsables del dataset/competencia (no se contacta en esta fase)",
         "methodological_novelty": "Nula: P91 documenta la insuficiencia y formula la pregunta exacta",
         "governance_status": "BLOCKED: decision DOCUMENTATION_INSUFFICIENT",
         "recommendation": "BLOCKED"},
        {"hypothesis": "Excepciones AG=1 con HS=0 (22 dev) frente a la lectura condicionada de AG",
         "historical_decision": "D-001: interpretacion de AG condicionada; 26 casos historicos en train",
         "historical_result": "Las excepciones existen y son estables (1.5 por ciento); ninguna decision historica las resolvio formalmente",
         "remaining_question": "Regla oficial de AG: condicionada o independiente",
         "new_evidence_needed": "Aclaracion de los responsables del dataset/competencia",
         "methodological_novelty": "Nula (misma pregunta de gobernanza)",
         "governance_status": "BLOCKED",
         "recommendation": "BLOCKED"},
        {"hypothesis": "Cola de candidatos a revision basada en atribucion exacta OOF (P90.03-R) y desacuerdo multi-modelo",
         "historical_decision": "Sin precedente directo: P87 uso material de fase 2 (80 filas, semilla 42) sin evidencia OOF",
         "historical_result": "P91 construye 882 celdas con motivos verificables y 100 tweets de referencia aleatoria; evidencia extrema contra el gold escasa (4 FN<=0.05 y 4 FP>=0.95)",
         "remaining_question": "Ejecucion de la revision (bloqueada) y estimacion de tasas con la componente de referencia",
         "new_evidence_needed": "Autorizacion de gobernanza; reglas operativas de anotacion",
         "methodological_novelty": "Media: seleccion multicriterio con evidencia exacta + muestreo con FPC; mejora de protocolo, no tecnica nueva",
         "governance_status": "BLOCKED para ejecucion; cola y protocolo listos",
         "recommendation": "COMPLEMENTARY"},
    ]
    with RUTA_GAP.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas_gap[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(filas_gap)

    for ruta in (RUTA_AUDIT, RUTA_PROTO, RUTA_SOTA, RUTA_GAP):
        if any(b > 127 for b in ruta.read_bytes()):
            raise SystemExit(f"STOP: bytes > 127 en {ruta.name}")
    print("P91 parte 2 completada: audit, protocolo, sota y gap escritos")
    print(f"  gap: {len(filas_gap)} filas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
