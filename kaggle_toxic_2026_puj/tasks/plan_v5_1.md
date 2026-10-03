# Plan de la fase v5.1: cierre arquitectonico del embedding

Estado: **en curso**. Rama `proyecto-kaggle-toxic-2026-v5`, worktree `/home/carlos/Git/PLN`.

## 1. Que cierra esta fase

La fase v5 cerro con D-034: el embedding desde cero es un `CANDIDATE` no adoptado. La
aportacion complementaria a V03 se midio en cinco configuraciones (5M, 10M, 50M, web,
mixto) y siempre cae entre +0,0001 y +0,0046, con el intervalo de bootstrap tocando el
0. Las tres dimensiones del corpus quedan agotadas.

Lo que queda abierto es una sola variable: **la arquitectura del entrenamiento**. v5.1
responde si alguna configuracion de fastText aporta a V03 mas que la de referencia
(FastText 200d / window 5 / min_count 3 / subpalabras 3-6 / negative 10 sobre Wikipedia
10M, LB de la referencia: corrected_cv 0.777211 y mejor fusion 0.7818).

## 2. Reglas de la fase

- Sin transformers, LLM ni embeddings preentrenados. Sin etiquetas externas. Test nunca
  para entrenar. Holdout agotado (5/5) e intacto. Sin submissions ni sondas de LB.
- Corpus de trabajo fijo: **Wikipedia 10M (E-CORP-01)**, que es el de mejor fusion
  observada. No se cambia corpus y arquitectura a la vez.
- Punto de operacion comun: **umbrales congelados de V03** (HS 0.44, TR 0.51, AG 0.46).
  Nadie gana ajustando su propio umbral.
- Gate de adopcion: **delta >= +0.005** frente a V03, con estabilidad entre pliegues e
  intervalo de bootstrap que no toque el 0. Un +0.001 o +0.003 **no** se adopta aunque
  salga positivo.
- Presupuesto duro: **9 entrenamientos exploratorios + 1 de reserva**. La reserva solo
  se usa ante evidencia clara y nunca para abrir una familia nueva de hiperparametros.
- Contabilidad separada: `EMBEDDING TRAINING UNIT` != `DOWNSTREAM EVALUATION UNIT` !=
  `AUDIT / REPRODUCTION`. Solo la primera consume presupuesto.
- Un experimento = un script nuevo. No se edita ni borra un script anterior.

## 3. Fases

| fase | contenido | estado |
|---|---|---|
| P0 | auditoria de firmas, identidad de entrenamiento y tabla de presupuesto | **hecha** |
| P1 | vector_size 100 / 300, con decision de cierre de familia | pendiente |
| P2 | window 3 / 7 (condicional) | pendiente |
| P3 | min_count 2 / 5 (condicional) | pendiente |
| P4 | subwords 3-5 / 3-7 (condicional) | pendiente |
| P5 | negative 5 (condicional) | pendiente |
| P6 | comparacion final: heatmap de arquitectura, tabla maestra, curva de escala | pendiente |
| P7 | complementariedad: salvage, damage, slices, solapamiento con char_wb | pendiente |
| P8 | integracion completa en el notebook consolidado y validacion por ejecucion | pendiente |
| P9 | FINAL_REPORT, PROJECT_STATE y DECISIONS | pendiente |
| P10 | auditoria final de reglas | pendiente |

## 4. Criterio de poda

Una familia se cierra si se cumplen **las dos** condiciones:

1. ninguna variante alcanza delta de fusion >= +0.005 frente a V03;
2. la dispersion interna de la familia (maximo menos minimo de la fusion) es <= 0.005.

Cerrada una familia no se exploran mas valores suyos (ni 400d, ni window 15, ni...). El
objetivo es gastar como maximo el presupuesto, no consumirlo.

## 5. Decision de cierre

La fase se cierra con `V03 sigue siendo final` salvo que alguna configuracion supere el
gate. Se entregaran dos conclusiones separadas: si la arquitectura cambia materialmente la
calidad de la representacion, y si la mejor configuracion aporta informacion adicional a
V03. No son la misma pregunta.
