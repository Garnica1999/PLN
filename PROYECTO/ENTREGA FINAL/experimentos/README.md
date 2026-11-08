# Experimentos de mejora (ciclo 2026-11-01)

Dos experimentos posteriores al cierre de la Entrega 3. Ninguno cambia el
producto adoptado (v2.2) porque ninguno pasó el gate de ADR-16.

## 1. Fusión tardía de rutas (`exp_fusion_rutas.py`)

Parte del benchmark v13 (90 casos) y de los rankings top-10 por ruta que dejó
la evaluación congelada. Mide el techo de seleccionar ruta por caso y prueba
la fusión por rango recíproco (RRF) con pesos elegidos por validación cruzada
de 5 pliegues, con delta pareado e IC 95 % bootstrap.

Resultado: v2.2 0,6926; oráculo de selección por caso 0,6996 (+0,0070, dentro
del ruido); **techo de reordenar el top-10 actual 0,7359** (+0,043); el top-10
contiene **4,8 de los 11,4 relevantes** por caso; Borda 0,5497, conteo de rutas
0,5431 y consenso de tres rutas 0,6926 (sin cambio). Ninguna regla mejora.
`cita_exacta` se excluye porque devuelve los artículos que M2 extrajo y el gold
del benchmark son esos mismos artículos.

Lectura: el margen no está en combinar rutas ni en reordenar lo ya recuperado.
Con 4,8 de 11,4 relevantes en el top-10, la mejora exige **promover candidatos
de las posiciones 11 a 50**, es decir, un mejor reranker (o más anotación para
desambiguar el gold).

Artefactos: `exp_fusion_rutas.json`, `exp_fusion_rutas.md`.

## 2. Calidad de respuesta del RAG (`evaluacion/evaluar_calidad_respuesta.py`)

Mide fidelidad, relevancia y precisión del contexto sobre 30 respuestas en
alcance del gold set, con un juez de familia distinta al generador
(`gemma4:26b-factual`) y la evidencia real de cada consulta reconstruida con
`/v1/link` (extraídos, recuperados, citadas y resumen).

Resultado (29 de 30 respuestas válidas; una pregunta por una sentencia fuera
del corpus):

| Dimensión | Media | IC 95 % |
|---|---:|---|
| Fidelidad a la evidencia | 0,881 | [0,774; 0,967] |
| Relevancia de la respuesta | 0,824 | [0,672; 0,945] |
| Precisión del contexto | 0,890 | [0,772; 0,986] |

Es gold-LLM, no anotación humana. Un primer intento con `alia-legal` se
descartó porque devolvía valores fuera de la escala 0 a 1; queda como
`evaluacion/calidad_respuesta_intento_alia.*`. Dos hallazgos de método: los
modelos de razonamiento necesitan `think=false` (si no, gastan la salida en el
bloque de pensamiento y devuelven vacío) y el contexto debe ser la evidencia
completa de las herramientas, no el resumen de la traza.

## 3. Sonda listwise sobre el top-10 (`exp_listwise_top10.py`)

Para los 90 casos se le pidió al LLM local (qwen, `think=false`) que reordenara
los diez candidatos de v2.2 con la sentencia y el texto de cada artículo.

Resultado: v2.2 0,6926; LLM 0,6764; delta pareado -0,0162 [-0,0331; +0,0004];
87 de 90 respuestas parseadas; 5 a 20 s por consulta. El LLM no explota el
techo de reordenamiento (+0,043): el cross-encoder B1 ordena mejor. Confirma
que el margen está en promover candidatos de las posiciones 11 a 50, no en
reordenar el top-10.

Artefactos: `exp_listwise_top10.json`, `exp_listwise_top10.md`.

## 4. Reentrenamiento de B1 con negativos duros (no ejecutado)

El guion de entrenamiento parte de `BAAI/bge-reranker-v2-m3` (no del B1 ya
ajustado), una época son ~17.800 pasos y el registro de la corrida anterior
muestra 2 a 3 horas por época. No cabe en el calendario de la entrega y el
resultado es incierto, así que se deja como trabajo futuro.
