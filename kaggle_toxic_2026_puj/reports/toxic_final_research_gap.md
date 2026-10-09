# TOXIC - Brecha de investigacion final

Fecha: 2026-10-08. Generado por la auditoria de oportunidades legales restantes
(`reports/toxic_remaining_legal_opportunities.md` y `.csv`). Sin entrenamiento,
sin submissions, sin test, sin holdout y sin leaderboard.

## 1. La pregunta, respondida

Existe alguna intervencion permitida por las reglas, no ejecutada previamente y
con un mecanismo de mejora defendible, que pueda justificar una nueva fase
experimental?

**No. Caso B.** La investigacion de modelado permanece cerrada y la
experimentacion se detiene.

**No se identifica actualmente una via experimental nueva, legal y
suficientemente justificada dentro de las restricciones vigentes del proyecto.**

Reparto de las 20 lineas auditadas: 0 `ACTIONABLE`, 5 `BLOCKED_BY_RULES`,
1 `DUPLICATE` y 14 `CLOSED`; la via de TR queda `BLOCKED_BY_RULES` y la de
near-duplicados queda `QUARANTINED`.

## 2. Lo que queda bloqueado por reglas (y por que)

| linea | bloqueo | que haria falta |
| --- | --- | --- |
| Correccion de etiquetas de TR | Las reglas solo prohiben usar el test para entrenamiento; no dicen nada sobre revisar o corregir etiquetas de entrenamiento, luego la correccion es ambigua | Autorizacion escrita del organizador; protocolo de revision humana ya disenado (seccion 4.3 del informe) |
| Uso transductivo del test (near-duplicados; EM real) | "El dataset de prueba no debe ser usado para entrenamiento" mas la ambiguedad B de V11-02 | Autorizacion escrita mas procedimiento pre-registrado que separe evaluacion final de desarrollo |
| Datos etiquetados adicionales | No existe fuente legal adicional (P47) | Una fuente legal nueva de etiquetas del mismo esquema |
| Corpus Twitter/X de 597M (pysentimiento) | Sin licencia declarada y recoleccion por timelines sin periodo acotado | Licencia explicita y periodo demostrablemente anterior |
| Transformers, LLM, embeddings preentrenados, etiquetas externas, pseudo-labeling | Prohibicion expresa de las reglas y de D-002/D-030/D-057 | Un cambio de reglas |

## 3. Lo que queda medido y cerrado

- **Campeon y techo.** V03 `corrected_cv` 0.7772114525165721 y LB 0.78306; el
  mejor OOF honesto del proyecto (0.7904) fue rechazado en el leaderboard; el
  OOF mas alto (0.8039, Optuna) cayo -0.0070; el meta con mejor holdout
  (+0.0180) cayo en sus cuatro envios.
- **Familias.** Todas las del mapa de la seccion 3 del informe estan
  `EXECUTED_AND_CLOSED`, `INVALIDATED` o `DUPLICATE`; ninguna queda
  `GENUINELY_OPEN`.
- **Puerta 2.** H1, H2, H3, H4 y H5 estan ejecutadas y cerradas (fase v9 y
  posteriores).
- **P84.** DL-01 corregido, fastText supervisado, ROS, SGNS corregido y
  EM/threshold: los cinco cerrados; EXP-06 sin candidatos.
- **Preguntas abiertas.** Las cinco de P57 no cumplen el umbral de reapertura.
- **Post-P68.** Los ocho ejes del encargo tienen cita historica; 0 sin recorrer;
  condicion de cierre cumplida.
- **Candidato downweight.** Congelado: `PROMISING_UNCONFIRMED` / `NEGATIVE`
  (LB externo 0.77522) / `EXPERIMENTAL_RESEARCH_CLOSED` / `PENDING`.
- **Word2Vec de dominio Twitter/X.** Ejecutada con corpus legal autorizado
  (mmaguero, CC BY-SA 4.0, periodo cerrado, 10.000.029 tokens, solapamiento cero
  con train) y cerrada por medicion: P72 sin ventaja, P73/P74 invalidados, P75
  sin transferencia, P78 sin transferencia util (brecha 0.117450 frente a V03);
  P84-04 negativo sobre corpus general. El corpus de 597M del encargo original
  sigue bloqueado.

## 4. La unica pregunta que sobrevive

> Existe informacion en los modelos alternativos que ninguno de los mecanismos
> probados ha sabido explotar. El techo medido lo confirma (+0,0391 de techo
> oracle sobre las celdas delegables), y cuatro mecanismos distintos fallaron al
> intentar acercarse a el.

Eso no es una tecnica: es una pregunta sin mecanismo de alcanzabilidad. Para
reabrirla haria falta una razon por la que la informacion sea alcanzable desde el
entrenamiento y no solo desde la decision; esa razon no existe hoy. Queda
declarada como pregunta, no como linea.

## 5. Condiciones de reapertura

Una linea solo podra recomendarse si cumple simultaneamente:

1. mecanismo genuinamente nuevo;
2. evidencia concreta que justifique investigarlo;
3. compatibilidad normativa;
4. datos disponibles legalmente;
5. control capaz de atribuir el resultado al mecanismo;
6. evaluacion posible sin test, holdout ni leaderboard como instrumentos de
   tuning;
7. coste razonable respecto del beneficio esperado.

Hoy ninguna linea cumple las siete. La reapertura exigiria, como minimo, una de
estas tres cosas: autorizacion normativa explicita (TR o transduccion), una
fuente de senal legal genuinamente nueva, o un mecanismo fuera de la lista de los
ya medidos con una razon de alcanzabilidad.

## 6. Cierre

La investigacion de modelado permanece cerrada. No se propone ninguna fase nueva,
no se generan submissions y no se toca el test ni el holdout. El siguiente avance
debe venir de una fuente de senal o un mecanismo legal realmente distinto; si no
existe, el cierre es preferible a generar nuevos experimentos sin una hipotesis
defendible.

> V03 remains the best verified leaderboard result.
