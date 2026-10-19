# P90.04 - Revision documental de hipotesis futuras (gap review)

Fecha: 2026-10-10. Revision exclusivamente documental sobre el repositorio
`KAGGLE_TOXIC_2026_PUJ`. **No contiene experimentos nuevos ni cifras no verificadas.**
Continua a P90.03 (gap analysis auxiliar) y P90.03-R (atribucion exacta OOF recuperada).

## 1. Objetivo

Determinar, con el historial completo (`PROJECT_STATE.md`, `DECISIONS.md`, `RUNLOG.md`,
`FINAL_REPORT.md`, `experiments.csv` y los informes de familia), si las atribuciones exactas de
P90.03-R dejan alguna pregunta relevante no cerrada y si existe una linea genuinamente nueva.
No se ejecuta ningun experimento minimo.

## 2. Mecanismos evaluados (6) y clasificacion

| H-WC-1 | Ponderacion explicita del canal word/char dentro de la union (explotar la estruc | V6-04 Track I (char_weight 0.5/1.5/2.0; control I0 reproduce V03 exact | CLOSED | CLOSED |
| H-PRE-1 | Neutralizar/normalizar los tokens de funcion y URL que cargan pesos negativos ap | PRE-02 (stopwords NLTK eliminadas; descartada); PRE-03 (URLs sustituid | CLOSED | CLOSED |
| H-AF-1 | Gating de contexto afiliativo (construcciones posesivas/afectivas que cancelan e | HIB-H11 (bloques bromista/interpersonal -0.0213); HIB-UNION (-0.0316); | CLOSED | CLOSED |
| H-NEG-1 | Corregir condicionada por contexto las asociaciones anti-toxicas aprendidas de t | HIB-H14 (lexico inducido x colectiva -0.0297); reweighting V30/P69; fa | CLOSED | CLOSED |
| H-ANOT-1 | Revision humana de anotacion en TR (destinatario implicito) y gobernanza de etiq | P87 (revision de anotaciones TR: LOW_EXPECTED_VALUE); D-045 (factoriza | BLOCKED | BLOCKED |
| H-SEL-1 | Operador selectivo que use informacion distinta del score del modelo (nota Abier | P75/P77 operador selectivo (sin finalista; segunda particion no ejecut | CLOSED | CLOSED |

## 3. Lectura por linea

- **H-WC-1 (canales word/char)**: la estructura exacta de canales de P90.03-R coincide con la
  pregunta de V6-04 Track I (D-047), que midio la escala explicita dentro de la union: control I0
  reproduce V03 exactamente (0.777211, IC95 [0,0]) y las tres escalas quedan por debajo del Gate 1
  (0.5: -0.0117; 1.5: +0.0016, 3/5; 2.0: -0.0047). Cerrada por resultado negativo.
- **H-PRE-1 (stopwords/URLs)**: PRE-02 (stopwords) y PRE-03 (URLs) ya probaron la manipulacion
  directa de esos tokens; descartadas. El peso negativo aprendido es una propiedad del corpus
  (registro), no un defecto de tokenizacion.
- **H-AF-1 (afiliativo)**: el caso exacto 23839 (perra +0.5589 vs mi perra -0.5511) es la
  cuantificacion del mecanismo de HIB-H11/contextual_selection, ya cerrado con resultados
  negativos (bloques densos -0.0213/-0.0316; P0=V03).
- **H-NEG-1 (asociaciones anti-toxicas)**: la correccion condicionada de tokens como rabe/acoso
  mapea a HIB-H14 (lexico x colectiva, -0.0297) y a las familias de reweighting/noise cerradas.
- **H-ANOT-1 (anotacion TR)**: unica linea bloqueada (P87, LOW_EXPECTED_VALUE); requiere
  gobernanza, no modelado.
- **H-SEL-1 (operador selectivo)**: la nota historica "Abierto" quedo cubierta por los cierres de
  P68 (enrutado por estructura) y contextual_selection en la auditoria de oportunidades.

## 4. Resultado

**NO_NEW_HYPOTHESIS_IDENTIFIED.** Seis mecanismos evaluados: 5 CLOSED, 1 BLOCKED, 0 DUPLICATE,
0 POTENTIALLY_NEW. Ninguna linea cumple simultaneamente los cinco criterios de novedad exigidos
(observacion exacta, mecanismo distinto, no repeticion de familia cerrada, experimento minimo
refutable, evaluable sin test/holdouts/leaderboard).

## 5. Recomendacion

No abrir una fase de modelado. La evidencia exacta de P90.03-R mejora la explicacion de V03
(mecanismos de acumulacion, lexico de misoginia, evidencia negativa anti-toxica, diferencias
FP/FN) pero no constituye una mejora demostrada ni una hipotesis nueva ejecutable dentro de las
reglas vigentes. La unica accion con valor potencial seria una revision de anotacion autorizada
(P87), fuera del alcance actual.

## 6. Trazabilidad

Tabla completa con las 12 columnas del protocolo en `reports/p90_04_hypothesis_gap_review.csv`.
