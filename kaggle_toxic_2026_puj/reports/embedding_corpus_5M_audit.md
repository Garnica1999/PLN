# Corpus de 5M tokens: tramo inferior de la curva de aprendizaje (fase v5, P9)

Generado por `experiments/embedding_09_corpus_5M_v5.py` (E-CORP-09).

## 1. Motivo de la reduccion

- Completar la curva de aprendizaje de la seccion 10 por debajo del punto de partida de 10M, a peticion del usuario, para comprobar si la curva es monotona y si existe un suelo de calidad de la representacion por debajo de 10M tokens.
- **Gobernanza**: La seccion 30 exige RATIONALE_FOR_CORPUS_EXPANSION para expandir. Una reduccion no es una expansion, de modo que no le aplica; se registra el motivo. Ademas la reduccion es un subconjunto exacto de un corpus ya construido: no hay descarga adicional, ni coste irreversible, ni riesgo de perder un corpus existente.
- **Coste**: Construccion ~10 s; entrenamiento ~1 min por modelo con 5 epocas.

## 2. Construccion

- Tokens: 5.011.404 | documentos: 1.694 | tipos: 173.942
- **Verificacion**: los 5M tokens son un **prefijo exacto** de los 10M: True. Por eso la comparacion 5M vs 10M no puede atribuirse a una diferencia de receta, solo de cantidad de datos.
- Tiempo de construccion: 7.0 s

## 3. Cobertura 5M vs 10M

| metrica | 5M | 10M | delta |
|---|---|---|---|
| cobertura por tipo | 0.6302 | 0.6839 | +0.0537 |
| cobertura por token | 0.8646 | 0.8833 | +0.0187 |
| word-OOV | 3164 | 2704 | -460 |
| tipos del corpus | 173.942 | 271.209 | +97.267

Tipos **nuevos** que aporta el salto de 5M a 10M: **97.267**. de los cuales **460** aparecen en el vocabulario TOXIC.

## 4. Cobertura por grupo

| grupo | 5M (token) | 10M (token) |
|---|---|---|
| elongaciones | 0.0236 | 0.0394 |
| coloquial | 0.6920 | 0.6920 |
| insultos | 0.7544 | 0.7777 |
| formas_raras | 0.5975 | 0.6565 |
