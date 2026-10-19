# P93 - Acuerdo formal entre anotadores (250 tweets principales)

Fecha: 2026-10-13. Fase diagnostica: sin entrenamiento, sin predicciones,
sin cambios de etiquetas ni de guia. Unidad: par (review_id, etiqueta);
750 celdas por metrica conjunta. La muestra principal incluye un componente
enriquecido no representativo: no estimar prevalencia poblacional.

## 1. Procedencia de la guia (resumen)

- Clasificacion: **GUIDE_V2_1_CONFIRMED** (a nivel de paquete; sin acuse de
  recibo firmado, limitacion registrada en p93_provenance_audit.md).
- Guia v2.1 presente en ambos paquetes (hash interno ff699f54a48da779);
  el archivo v2 retirado no esta en las carpetas; la columna
  guideline_version=v2 corresponde a la POLITICA (definido por la propia v2.1).

## 2. Distribuciones marginales

| etiqueta | anotador | POSITIVE | NEGATIVE | UNABLE_TO_DETERMINE | suma |
|---|---|---|---|---|---|
| HS | A | 36 | 111 | 103 | 250 |
| HS | B | 35 | 108 | 107 | 250 |
| TR | A | 140 | 77 | 33 | 250 |
| TR | B | 168 | 69 | 13 | 250 |
| AG | A | 79 | 157 | 14 | 250 |
| AG | B | 110 | 109 | 31 | 250 |

## 3. Acuerdo por etiqueta

| etiqueta | raw 3cat | binario cond. | dice pos | dice neg | kappa 3cat | kappa bin | alpha (cat) | alpha (faltante) |
|---|---|---|---|---|---|---|---|---|
| HS | 199/250 | 119/120 | 62/63 | 176/177 | 0.667 | 0.978 | 0.667 | 0.979 |
| TR | 201/250 | 192/213 | 270/291 | 114/135 | 0.631 | 0.773 | 0.629 | 0.773 |
| AG | 172/250 | 170/207 | 138/175 | 202/239 | 0.462 | 0.639 | 0.450 | 0.635 |

Matrices 3x3 completas por etiqueta (numeros, sin porcentajes):

- HS: NN=88 NP=0 NU=23 | PN=1 PP=31 PU=4 | UN=19 UP=4 UU=80
- TR: NN=57 NP=16 NU=4 | PN=5 PP=135 PU=0 | UN=7 UP=17 UU=9
- AG: NN=101 NP=31 NU=25 | PN=6 PP=69 PU=4 | UN=2 UP=10 UU=2

Sensibilidad (separada, como exige el protocolo):

- A) UTD como tercera categoria: cifras de la tabla anterior (raw 3cat, kappa
   y alpha con UTD=categoria).
- B) Condicional al subconjunto determinable (ambos binarios): la columna
  'binario cond.' y 'kappa bin'; excluye las celdas con UTD de cualquiera de
  los dos. Celdas excluidas por etiqueta: HS=130, TR=37, AG=43. Esta metrica describe SOLO ese subconjunto; no es el acuerdo global.

## 4. Acuerdo conjunto

- Triple exacto (HS/TR/AG identicas): 131/250.
- Tweets con al menos un desacuerdo: 119/250.
- Celdas con desacuerdo: 178/750.
- Tweets con al menos un UTD: 150/250.
- Celdas con UTD (cualquiera de los dos): 210/750.

## 5. Desacuerdos humanos (resumen de P3)

- BINARY_DISAGREEMENT: 59 celdas.
- UTD_VS_BINARY: 119 celdas.
- UTD_BOTH (acuerdo en incertidumbre, no desacuerdo): 91 celdas.
- MISSING_OR_INVALID: 0 celdas.
Detalle por celda: p93_human_disagreement_cells.csv (interno).

## 6. Comparacion interna con el gold (P4)

Clasificacion por celda (interna; el gold NO se usa para decidir quien tiene
razon). Resumen por etiqueta y por componente de muestreo:

| etiqueta | componente | matches | disagree_with_gold | human_disagreement | utd_involved | total |
|---|---|---|---|---|---|---|
| HS | A | 48 | 3 | 0 | 49 | 100 |
| HS | B | 48 | 20 | 1 | 81 | 150 |
| TR | A | 55 | 28 | 7 | 10 | 100 |
| TR | B | 50 | 59 | 14 | 27 | 150 |
| AG | A | 60 | 12 | 10 | 18 | 100 |
| AG | B | 44 | 54 | 27 | 25 | 150 |

- Componente A = submuestra aleatoria (100/1236, probabilidad conocida);
  componente B = enriquecido no representativo. No combinar para estimar una
  tasa poblacional; toda cifra es descriptiva de esta muestra.
- GUIDE_PROVENANCE_UNRESOLVED: 0 celdas (procedencia confirmada a nivel de
  paquete; ver limitaciones).

## 7. Candidatos a adjudicacion (P5)

- Cola total: 176 celdas (A y B coinciden en binario distinto
  del gold). Por etiqueta: HS={conteo_cand.get('HS', 0)}, TR={conteo_cand.get('TR', 0)}, AG=66.
- Por componente: A=43, B=133.
- NO son errores confirmados: son candidatos para revision por una tercera
  persona (ver p93_adjudication_protocol.md). Cola interna:
  p93_gold_discrepancy_candidates.csv; paquete ciego:
  p93_adjudication_blind_candidates.csv (NO distribuir automaticamente).

## 8. Limitaciones

- Sin acuse de recibo firmado de la guia (procedencia a nivel de paquete).
- Secuencia desviada: la ronda principal se anoto antes de completar la
  calibracion v2.1 (documentado; no se altera la cronologia).
- Componente B no representativo; B con artefactos de export en textos
  (no afectan anotaciones).
- El acuerdo humano no estima el error del gold; los candidatos no son
  errores confirmados.
