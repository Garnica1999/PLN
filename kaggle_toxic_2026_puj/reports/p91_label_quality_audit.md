# P91 - Auditoria de calidad de etiquetas (label quality research)

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
cola de revision ciega (882 celdas; 100 tweets
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
| FN con p<=0.05 (evidencia extrema contra gold) | 0 | 4 | 0 |
| FN con p<=0.10 | 5 | 8 | 1 |
| FP con p>=0.95 | 1 | 1 | 2 |
| FP con p>=0.90 | 5 | 9 | 6 |
| celdas cerca del threshold (|m|<0.025) | 81 | 26 | 72 |
| desacuerdo unanime de los 4 modelos | 199 | 118 | 179 |
| discrepancia con vecinos (label_consistency_v4) | 42 | 42 | 43 |

- Estructura: TR=1&HS=0 = 0; AG=1&HS=0 = 22; combinaciones
  completas en el CSV.
- Duplicados: near-dup 1 grupo divergente / 3 filas (de 2 grupos no
  unitarios); fusionados 13 grupos / 43 filas divergentes (de 37 no
  unitarios); vecindad de consistencia: 68 tweets dev en algun cluster.
- Histograma de desacuerdo 0-4 modelos: {'0': 3255, '1': 351, '2': 176, '3': 171, '4': 496}.

**Que puede inferirse**: que la evidencia extrema contra el gold es ESCASA (4 celdas
FN<=0.05 y 4 FP>=0.95 en total), que la contradiccion textual demostrable es marginal
(2.77-2.89 por ciento del train; P87) y que el desacuerdo unanime de 4 modelos
correlacionados (496 celdas, 11.1 por ciento) es en su mayoria error de
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

{
 "oof": "559265816ce4f6312c87f008f90324f0eec70a74d1e6247364c162793338feab",
 "v2base": "27dfa83736b979518a94bc93e32bb9cae07bc82eaff92234fe9e15d1f9822774",
 "consistencia": "8c90b9dd7f68fb8f3afdc9687b5adb99aed239162414575a43b87e6cf50c7587",
 "grupos": "afabf9ae1c295ea8f203120bf28578cb4bdb4e8f16b89178d820baeaac62e622",
 "particion": "e5b62e5f22c4c63a8c43df4019969504d44c0baa5b503c19ba3560032f5ee6cd"
}

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

## Nota de consistencia P91-R (aditiva, 2026-10-09)

Correccion documental aditiva; el texto anterior se conserva integro (registro original:
sha256 09f3ff4517b28e5287d9e3f00d20492d2ea6601fda27f7a27dc1d43bc96a705d). La cabecera
de este informe registra "Fecha: 2026-10-10", posterior a la fecha del proyecto
(2026-10-09): la fecha queda como **no verificada** (la fase P91 se ejecuto en la sesion de
trabajo correspondiente; ver `reports/p91r_methodology_consistency.md`, seccion 5). Ademas,
la relacion entre motivos y candidatos de la cola quedo aclarada en P91-R: los motivos no
son mutuamente excluyentes (882 celdas unicas, 953 asignaciones) y el codigo
`DOCUMENTED_RULE_CONFLICT` denota una tension estructural empirica (EMPIRICAL_STRUCTURE_ANOMALY),
no una violacion de regla oficial. La decision cientifica de P91 no cambia.
