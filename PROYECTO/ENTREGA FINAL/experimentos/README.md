# Experimentos de mejora (ciclo 2026-11-01)

Dos experimentos posteriores al cierre de la Entrega 3. Ninguno cambia el
producto adoptado (v2.2) porque ninguno pasó el gate de ADR-16.

## 1. Fusión tardía de rutas (`exp_fusion_rutas.py`)

Parte del benchmark v13 (90 casos) y de los rankings top-10 por ruta que dejó
la evaluación congelada. Mide el techo de seleccionar ruta por caso y prueba
la fusión por rango recíproco (RRF) con pesos elegidos por validación cruzada
de 5 pliegues, con delta pareado e IC 95 % bootstrap.

Resultado: v2.2 0,6926; oráculo de selección por caso 0,6996 (+0,0070, dentro
del ruido); unión de los top-10 de todas las rutas 0,5732; ninguna fusión
mejora (la de todas las rutas cae a 0,6350). `cita_exacta` se excluye porque
devuelve los artículos que M2 extrajo y el gold del benchmark son esos mismos
artículos.

Lectura: el margen no está en combinar rutas. El recall del candidato está
saturado (0 ítems fuera del pool en el gold humano) y la pérdida es de orden
dentro del top-10, así que la palanca es el reranker.

Artefactos: `exp_fusion_rutas.json`, `exp_fusion_rutas.md`.

## 2. Calidad de respuesta del RAG (`evaluacion/evaluar_calidad_respuesta.py`)

Evalúa fidelidad, relevancia y precisión del contexto sobre 30 respuestas en
alcance, con un juez LLM distinto del generador. Se ejecutó con `alia-legal`
(4,9 GB, el único juez que cabía con la GPU compartida) y **no es admisible**:
devolvió relevancia 9,00 (fuera de la escala 0 a 1) y dos casos sin parsear.
Los artefactos quedan como `calidad_respuesta_intento_alia.*`.

El script ya valida la escala (descarta al juez que se sale) y acepta
`JURIS_JUEZ` para usar un juez mayor. Pendiente: repetirlo con `gemma4:26b`
o `qwen3.8:docs-32k` cuando la GPU esté libre de otros proyectos.

## 3. Reentrenamiento de B1 con negativos duros (no ejecutado)

El guion de entrenamiento parte de `BAAI/bge-reranker-v2-m3` (no del B1 ya
ajustado), una época son ~17.800 pasos y el registro de la corrida anterior
muestra 2 a 3 horas por época. No cabe en el calendario de la entrega y el
resultado es incierto, así que se deja como trabajo futuro.
