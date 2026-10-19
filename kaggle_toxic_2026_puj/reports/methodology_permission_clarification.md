# Borrador de consulta academica: delimitacion del espacio metodologico

Estado: BORRADOR - NO ENVIADO. El propietario decide si lo utiliza.
Fecha: 2026-10-14.

## Contexto

- V03 sigue siendo el modelo de referencia: corrected_cv 0.7772114525165721;
  leaderboard 0.78306 (maximo real verificado del proyecto).
- Las familias clasicas (TF-IDF + lineales, ensembles, pesos, umbrales) y los
  modelos entrenados desde cero ya fueron explorados ampliamente: las rondas
  recientes de experimentos (~15) no produjeron candidato adoptable.
- Las representaciones contextuales recurrentes y los cambios de corpus
  (30M/60M y composicion) no produjeron mejoras robustas.
- El proyecto quiere continuar investigando SIN incumplir las reglas y
  necesita una delimitacion explicita del espacio metodologico permitido.

## Preguntas (se agradece respuesta explicita por escrito)

**A. Modelos de representacion preentrenados.** Esta familia esta actualmente
considerada PROHIBIDA. Se solicita aclaracion sobre si se permitiria evaluar
un encoder de lenguaje preentrenado por terceros, ajustado EXCLUSIVAMENTE con
las etiquetas oficiales de TOXIC, sin utilizar datasets externos etiquetados.
No se utilizara sin autorizacion escrita; el silencio o una respuesta ambigua
no se interpretaran como permiso.

**B. Corpus no etiquetados.** Aclarar el alcance permitido para representaciones
entrenadas desde cero, distinguiendo: corpus generales; texto social; corpus
con licencia no comercial; corpus derivados de redes sociales; corpus externos
originalmente etiquetados, aunque se pretendiera descartar sus labels.
Las fuentes restringidas permanecen en cuarentena hasta que su caso se
resuelva de forma especifica.

**C. Adjudicacion de labels oficiales.** Preguntar si se permitiria entrenar
con una copia alternativa de los objetivos DESPUES de una auditoria humana
independiente de ejemplos del train oficial, conservando intactas las labels
originales y evaluando la prediccion contra la referencia oficial.
Aclaracion: P101 esta bloqueado porque no hay revisores disponibles y porque
la gobernanza de esta operacion es ambigua; la auditoria NO ha confirmado
errores de anotacion.

**D. Otras tecnicas.** Preguntar si existen restricciones adicionales
aplicables a: aprendizaje contrastivo entrenado desde cero; modelos no lineales
entrenados exclusivamente con los datos oficiales; tecnicas de calibracion y
toma de decisiones ajustadas mediante validacion anidada.
(No se asume que estas tecnicas sean nuevas ni que deban ejecutarse; el fin es
establecer el marco permitido.)

## Compromisos vigentes mientras no haya respuesta

- Sin entrenamiento nuevo; sin submissions; sin uso de test, holdouts ni
  leaderboard para desarrollar; exploracion en pausa (PROJECT_STATE.md:
  MODEL_EXPLORATION_STATUS = PAUSED).
