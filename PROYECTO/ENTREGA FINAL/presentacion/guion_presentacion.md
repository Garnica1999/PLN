# Guion de la presentación (Entrega 3)

Duración estimada: 60 a 90 segundos por diapositiva.
Cada bloque indica la cifra que debe decirse en voz alta.

## 1. Portada

*Cifra en voz alta: 23.742 sentencias.*

Buenas tardes. Soy Carlos Andrés Garnica Salazar y presento la Entrega 3 de Juris-PLN, el proyecto de Procesamiento de Lenguaje Natural de la Maestría en Ingeniería de Sistemas y Computación de la Javeriana. El título resume el recorrido completo: partimos de 23.742 sentencias de la Corte Constitucional y llegamos a un sistema que devuelve los artículos de la Constitución que fundamentan cada providencia, con la evidencia textual que lo respalda. En los próximos quince minutos voy a presentar el problema, la forma de medirlo, los resultados, lo que no funcionó y los límites. Todos los resultados están congelados y construidos sobre datos públicos.

## 2. El problema

*Cifra en voz alta: 48 por ciento de los tokens limpios.*

El punto de partida es un corpus de 23.742 sentencias publicadas entre 1992 y 2021, con 261 millones de palabras. Cuando un jurista necesita saber en qué se apoya una decisión, no tiene un índice: la cita que sostiene la ratio vive dentro del texto. Por eso lee el documento completo y lo repite caso por caso. La tarea es repetitiva, costosa y difícil de auditar a escala. Hay un dato que explica la dificultad: el 48 por ciento de los tokens limpios son fórmulas institucionales, no conceptos jurídicos. Casi la mitad del texto no aporta señal semántica para recuperar.

## 3. La pregunta y el producto

*Cifra en voz alta: evidencia textual verificable.*

La pregunta de trabajo es concreta: dada una providencia, qué artículos de la Constitución la fundamentan y con qué evidencia textual. La respuesta no es una opinión: cada artículo llega con la frase que lo respalda, para que cualquiera pueda verificarla. Sobre esa base el producto añade dos vistas: las citas resueltas y la jurisprudencia asociada a cada artículo. El público son juristas, investigadores y litigantes. Conviene delimitar la utilidad desde el inicio: el sistema ordena la búsqueda y la hace auditable, no dictamina ni sustituye el criterio jurídico.

## 4. Los datos

*Cifra en voz alta: 10.530 palabras limpias de media.*

El corpus tiene 23.742 sentencias: 17.099 de tutela, 6.246 de constitucionalidad y 397 de sentencia unificada. Cada documento promedia 10.530 palabras limpias y una mediana de 7.158, es decir, documentos largos con una cola importante. La figura muestra dos cosas: la diversidad léxica medida con MTLD por tipo de providencia, y su evolución anual. El dato importante está en la línea: el MTLD cae de 64,88 a 56,77 entre 1992 y 2021. El vocabulario se empobrece y se institucionaliza, lo que refuerza la necesidad de un recuperador que no dependa solo de coincidencia de palabras.

## 5. Cómo se mide

*Cifra en voz alta: 90 casos y 820 pares.*

La evaluación tiene tres piezas. Primero, un benchmark interno de 90 casos con etiquetas generadas por un modelo de lenguaje, que sirve para iterar rápido. Segundo, un gold humano B3 de 50 casos y 820 pares artículo sentencia, que es la referencia más exigente. Tercero, C3, un corpus de 2022 a 2026 que no participó en el desarrollo y mide deriva temporal. Las métricas son nDCG arroba 10, F1 arroba 5 y MRR, siempre con intervalos de confianza al 95 por ciento por bootstrap pareado, porque comparar promedios sueltos enmascara diferencias pequeñas. El gate de adopción es el ADR-16: la mejora debe ser significativa y consistente entre conjuntos.

## 6. Resultado de recuperación

*Cifra en voz alta: 0,6907 frente a 0,6013.*

Aquí está el resultado central. La línea base BM25 con expansión por hubs obtiene 0,6013 de nDCG arroba 10. La versión 2.2, que combina el retriever B2 con el reranker B1, sube a 0,6907. Sobre el gold humano la misma configuración alcanza 0,7852, porque el gold es más limpio que las etiquetas automáticas. La mejora pareada frente a la línea base es de 0,067 puntos, con intervalo de confianza al 95 por ciento entre 0,042 y 0,089: el cero queda fuera, así que la mejora no es ruido. En el corpus posterior, C3, el sistema mantiene 0,7472, sin deriva significativa.

## 7. Lo que no funcionó

*Cifra en voz alta: 31,81 por ciento de ruido de etiqueta.*

La transparencia es parte del resultado. Probamos un ajuste LoRA sobre el modelo C2 y perdió 0,198 puntos en consultas: no generalizó. Una segunda época del retriever B2 sumó 0,0013, una diferencia dentro del ruido y sin significancia. Con 200 casos zero shot generados por el modelo de lenguaje tampoco superamos la línea base. Y el clasificador M4 se queda en 0,2870 de macro F1, con un 31,81 por ciento de ruido de etiqueta detectado con cleanlab y documentado en el ADR-19. Por eso el área de clasificación queda como exploratoria y no se ofrece en el producto: preferimos declararlo antes que presentar una cifra que no se sostiene.

## 8. El producto

*Cifra en voz alta: tres capas de guardrails.*

El producto corre entero en local, en tres procesos. Streamlit en el puerto 8502 sirve la interfaz. FastAPI en el 8123 expone los contratos y aplica los guardrails. El núcleo validado, con recuperación, reranking y el modelo local vía ollama, hace el trabajo pesado. Los endpoints principales son link, evidencia, jurisprudencia por artículo y ask. Los guardrails están en tres capas: un router determinista que clasifica la intención antes del modelo, un agente con tool calling que solo consulta herramientas del núcleo, y una verificación de grounding que exige cita en el texto. Sin cita, no hay respuesta.

## 9. La demo en cuatro pantallas

*Cifra en voz alta: id 2195, T-293 de 1997.*

La demo se recorre en cuatro pantallas. La primera va de la sentencia a la Constitución, con la cita textual al lado de cada artículo. La segunda va del artículo a la jurisprudencia: el artículo 86 aparece en 16.405 sentencias del corpus, un número que muestra la escala del problema. La tercera muestra la evaluación con la procedencia de cada resultado, para que el usuario sepa de dónde sale. La cuarta entrega la ficha en PDF y las sentencias similares. El ejemplo que usamos siempre es el mismo: la sentencia con identificador 2195, que corresponde a la T-293 de 1997, con 6 artículos extraídos y 10 recuperados.

## 10. El agente, medido

*Cifra en voz alta: 13,4 segundos de mediana.*

El agente también se mide. Sobre un gold set de 100 prompts, el ruteo exacto acierta el 90 por ciento. La ruta jurídica, los saludos no rechazados, el rechazo de consultas fuera de alcance, el uso correcto de herramientas y el grounding llegan al 100 por ciento. Es decir, cuando el sistema responde, la respuesta está anclada en evidencia recuperada. La latencia es el costo visible: 13,4 segundos de mediana y 60,8 en el percentil 95. Ese p95 define el presupuesto de espera de la demo y explica por qué la interfaz muestra el avance por etapas.

## 11. Límites y ética

*Cifra en voz alta: corpus hasta 2021.*

Los límites están declarados. El corpus termina en 2021; el periodo posterior solo se usa para medir deriva. El clasificador M4 no es fiable y no se ofrece como resultado. El gold humano se anotó con apoyo de un modelo de lenguaje, no con validación de expertos legales, así que hay que leerlo como proxy y no como verdad de referencia. Los datos son públicos y no contienen información personal. Queda un riesgo técnico abierto: el modelo local puede degradar su salida, y para eso existe el fallback determinista, que devuelve igualmente la evidencia recuperada.

## 12. Cierre

*Cifra en voz alta: C1 con 150 casos.*

Cierro con lo esencial: el producto funciona y es auditable. Cada respuesta se puede rastrear hasta la frase que la sostiene, y los resultados negativos están documentados igual que los positivos. El trabajo inmediato tiene tres frentes: C1 con 150 casos y anotación controlada, el despliegue externo con límites de uso declarados, y la validación con expertos legales, que es el único camino para convertir el gold proxy en referencia. La frase que resume la postura del proyecto es la que aparece al pie: recuperar la evidencia, no reemplazar al jurista. Gracias.
