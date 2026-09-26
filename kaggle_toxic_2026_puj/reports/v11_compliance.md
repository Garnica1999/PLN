# Cumplimiento de la fase v11

Actualizado por `V11-02 (P1.1, auditoria de legalidad del uso de la prueba)` el 2026-09-26T21:36:56+00:00.

## Normativa aplicada

- `docs/competition_rules.md`, capturada el 2026-09-22.
- Citas verificadas literalmente contra el documento: **True**.

## Restricciones y su estado

| restriccion | regla | estado |
| --- | --- | --- |
| Sin transformers, RoBERTa, BERT ni LLM | No se pueden usar transformers ni modelos LLM para l | **CUMPLIDA**, verificada en cada experimento de la v10 |
| Todo desde cero, sin recursos externos etiquetados | No se pueden usar transformers ni modelos LLM para l | **CUMPLIDA**, solo se usan los 19 .bin de FastText que son propios |
| El test no se usa para entrenamiento | El dataset de prueba no debe ser usado para entrenam | **CUMPLIDA** |
| El test no se usa para estimar prevalencia con fines de submission | El dataset de prueba no debe ser usado para entrenam | **NO SE USA**, clasificado ambiguo y descartado para submission |
| Sin pseudoetiquetado | restriccion del profesor | **CUMPLIDA** |
| Sin etiquetas de test | las etiquetas estan ocultas en Kaggle | **CUMPLIDA** |
| Semillas fijas y reproducibilidad con tolerancia 1e-6 | principio de la fase | **CUMPLIDA**, verificada en V11-00 y V11-01 |
| Artefactos en ASCII, sin emojis ni Unicode decorativo | regla de caracter | **CUMPLIDA**, `tools/verificar_caracteres.py` en cada escritura |
| El holdout2 no se usa durante la exploracion | plan v11, seccion 2 | **CUMPLIDA**, 0 de 3 evaluaciones gastadas |
| El leaderboard no se usa como optimizador | plan v11, seccion 34 | **CUMPLIDA**, no se ha consultado |
| No reabrir familias cerradas | plan v11, seccion 35 | **CUMPLIDA**, la v11 solo trabaja la capa de decision |

## Clasificacion de los pasos con la prueba

- permitidos: **5**
- ambiguos: **1**, y el uso de la prevalencia real cae aqui
- prohibidos: **5**

Conclusion operativa: **El uso de la prueba para estimar la prevalencia con EM es AMBIGUO, de modo que **no se usa para generar ningun submission**. La metodologia se estudia unicamente mediante shift simulado sobre el OOF, y las tres lineas centrales de la v11, que son la correccion de prior, la calibracion y la regla de decision, se evaluan bajo prevalencias simuladas. La pregunta de la transferencia al externo queda sin poder responderse con honestidad dentro de las reglas, y eso hay que decirlo en el informe final en vez de disimularlo.**

## Lo que esta decision le cuesta a la fase

La pregunta de la v11 se puede responder en su version metodologica, que es si el
mecanismo recupera un desplazamiento conocido, y **no** en su version de
transferencia, que es si ese mismo mecanismo mejoraria el resultado en esta prueba
concreta. La razon es que medir lo segundo exige usar la prueba para generar un
submission, y eso es exactamente lo que quedo clasificado como no permitido. Queda
escrito aqui para que la limitacion sea visible en el informe final y no se descubra
al final.

Fecha: 2026-09-26T21:36:56+00:00. Duracion: 0.0 s.
