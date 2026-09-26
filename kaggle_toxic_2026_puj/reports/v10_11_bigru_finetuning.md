# V10-11: BiGRU con afinamiento de embeddings (P4, cierre de H1)

Generado por `experiments/v10_11_bigru_finetuning.py` (V10-11 (P4, BiGRU con afinamiento)).

## 1. Que cierra

V9-04 midio H1 con los embeddings **congelados** y dio 0.6943, frente a 0.6636 del control
de media mas regresion lineal y 0.777211 de V03. La conclusion fue que el mecanismo
secuencial si mejora al pooling, con +0.031, pero no compite con V03.

Quedaba **afinar** los embeddings en vez de congelarlos. Congelados, la red solo puede
reordenar lo que ya esta ahi; afinados, puede adaptar la representacion al dominio.

## 2. Las tres comparaciones, y por que la tercera es la importante

| variante | embebido | learning rate | que separa |
|---|---|---|---|
| congelado | sin gradientes | 1e-3 | lo que ya dio V9-04 |
| **afinado** | **con gradientes** | **1e-4 embebido, 1e-3 resto** | el efecto del afinamiento |
| afinando todo | con gradientes | 1e-3 para todo | **el control que faltaba** |

La tercera es la que hace que las otras dos signifiquen algo: **distingue afinar de
simplemente dar mas parametros a la red**. Sin ese control, un acierto del afinamiento
podria ser solo mas capacidad, que no es lo que se quiere defender.

El learning rate diferencial multiplica por diez la separacion: la capa de embebido viene
preentrenada sobre otro corpus y moverla mucho la destruye antes de que aproveche, y la
red recurrente es nueva y necesita aprender desde cero.

## 3. Decision sobre sobreentrenamiento

- decision: **early stopping estricto**
- quien la fijo: el pre-registro en tasks/plan_v10.md, antes de ejecutar nada
- motivo: El objetivo es medir el techo de la variante, y sobreentrenar no daria un techo sino un sobreajuste. Ademas la mejor epoca se elige por la perdida de validacion, que es la misma senal que se usa para las curvas de aprendizaje; la metrica de la competencia se evalua una sola vez al final y no se usa para ajustar.

## 4. Resultados

| variante | F1 | HS | TR | AG | mejor epoca por pliegue |
|---|---|---|---|---|---|
| congelado | **0.693994** | 0.6742 | 0.7515 | 0.6562 | [12, 10, 10, 8, 13] |
| afinado | **0.711639** | 0.6941 | 0.7650 | 0.6759 | [12, 11, 10, 8, 12] |
| afinando_todo | **0.699086** | 0.6698 | 0.7437 | 0.6837 | [7, 7, 7, 7, 8] |

| referencia | F1 |
|---|---|
| V03, TF-IDF union mas regresion lineal | **0.777211** |
| V9-04, media mas regresion lineal | 0.663600 |

- afinado menos congelado: **+0.017645**
- afinado menos afinando todo: **+0.012553**
- afinado menos V03: **-0.065572**
- **gana a V03: False**

## 5. Veredicto

**H1 QUEDA CERRADA con dos variantes, congelada y afinada.** Ninguna compite con V03,
que era el criterio de cierre pre-registrado para P4.

La lectura es coherente con el resto del proyecto. V03 es una combinacion de TF-IDF
word(1,2) y char_wb(3,5) con una regresion lineal: captura redundancia **de forma
explicita**, en vez de|Aprenderla a traves de una recurrencia. Con mil y pico filas, la
regresion lineal sobre 24.646 rasgos explicitos aprovecha mejor los datos que una red
que tiene que descubrir la estructura por su cuenta. No es que las redes no sirvan, es
que **este** problema, con estos datos, no las necesita.

Y la consecuencia practica: **la via de representacion secuencial queda cerrada.** No se
abre con mas capas, mas unidades o mas epocas, porque el criterio de cierre estaba escrito
antes de ejecutar y ya se cumplio.

Fecha: 2026-09-26T21:03:19+00:00. Duracion: 46.7 s.
