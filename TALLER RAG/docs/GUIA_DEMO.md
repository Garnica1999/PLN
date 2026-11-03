# Guía de demostración

## Antes de empezar

```bash
docker compose ps
curl http://localhost:8010/health
```

Condiciones: Milvus, LLM y embeddings en verde, y al menos una colección con documentos (`GET /stats`).

Si la capa de datos está en el servidor, comprueba el túnel antes de la demo:

```bash
ss -ltn | grep 19530
curl http://localhost:8010/health
```

El túnel se abre con `ssh -N -L 19530:localhost:19530 carlos@192.168.1.100`. Si el puerto no escucha, el API no recupera aunque el frontend siga abierto.

El servicio `prewarm` del compose ya cargó el generador, el recuperador y el reranker al levantar el stack. Si hiciste `down` y `up`, espera a que el contenedor `prewarm` salga con código 0. Para repetirlo a mano: `docker compose exec -T api python -m src.scripts.prewarm`. Sin precalentamiento, la primera pregunta tarda unos 29 s; con él, la demo arranca en los 6 a 7 s normales.

## Guion (5 a 7 minutos)

1. Abrir `http://localhost:8501`.
2. Revisar el panel: `API conectada`, modelo `gemma4:e4b`, k=5, colección `rag_full_bge_m3_d1024_v1`, `Razonamiento` y `Modo estricto` activos.
3. Preguntar por el Retamo Espinoso: "¿Cuántos municipios han reportado la presencia del Retamo Espinoso en el departamento?". Mostrar la respuesta (22 municipios) con sus citas y el bloque de razonamiento.
4. Abrir las evidencias: título, enlace, puntuación y fragmento. Señalar que cada cita `[n]` corresponde a una evidencia.
5. Mostrar los tiempos: recuperación, razonamiento, generación, total y tokens por segundo. El MTP mantiene unos 217 tok/s.
6. Preguntar "¿Qué pasó con la moneda de 1.000 pesos?". Cambiar el modelo a `gemma4:e4b-mtp5` y repetir para comparar latencia: el draft 5 es más lento que el 3.
7. Caso de abstención: preguntar "¿Cuántos empleados tiene la empresa Nebulosa Andina?". El sistema debe decir que no hay evidencia suficiente.
8. Caso fuera de dominio con `Modo estricto`: preguntar "¿Cuál es la capital de Francia?". Se abstiene con `abstain_reason="fuera_dominio"` en menos de 100 ms, sin router ni generación.
9. Apagar `Modo estricto` y repetir la misma pregunta: ahora responde con una cita del corpus. Es el comportamiento esperado sin la compuerta.
10. Activar `Modo noche` y mostrar el cambio de paleta. Recargar la página para mostrar que la elección se conserva.
11. Caso de error: bajar el servicio de Ollama (`docker compose stop ollama`), preguntar y mostrar el mensaje de error de la interfaz. Levantarlo de nuevo.

## Interfaz

La UI usa Material 3 sobre Streamlit: superficies tonales, forma de 4 a 28 px, escala tipográfica de display a label y capas de estado. El panel lateral es un cajón con esquinas grandes, la entrada de chat es una barra de búsqueda con elevación tonal y las sugerencias iniciales son tarjetas, no botones centrados.

| Estado | Captura |
|---|---|
| Inicio | `reports/figuras/ui_m3_vacio.png` |
| Chat con razonamiento | `reports/figuras/ui_m3_chat.png` |
| Modo estricto fuera de dominio | `reports/figuras/ui_m3_estricto.png` |
| Modo noche | `reports/figuras/ui_m3_noche.png` |
| Panel lateral oculto | `reports/figuras/ui_m3_panel_oculto.png` |

Los colores salen del verde `#1f6b57`: `primary-container` para acentos, `secondary-container` para el mensaje del usuario, `tertiary-container` para avisos informativos y `error-container` para las abstenciones. El interruptor `Modo noche` cambia a la paleta oscura de la misma semilla y también ajusta los menús y el calendario. Los tokens están en `src/rag/ui.py`.

Las 22 combinaciones de texto y fondo de ambos esquemas pasan WCAG AA. El mínimo es 4,21:1 en el borde de campo, que exige 3:1. Se comprueba con `python src/scripts/check_contrast.py`.

Las tres preguntas de ejemplo se sortean de un banco de nueve preguntas del gold en cada sesión. `Nueva conversación` vacía el historial y vuelve a sortearlas.

`Razonamiento`, `Modo estricto` y `Modo noche` se guardan en `config/ui_prefs.json` y se recuperan al recargar la página. De fábrica, `Razonamiento` y `Modo estricto` vienen activos; `Modo noche` viene apagado. El archivo está montado en el contenedor `frontend` como `./config:/work/config`.

Si se cambia un interruptor mientras el modelo responde, Streamlit detiene el script en curso. La pregunta queda en `st.session_state.pending` y se reintenta en el siguiente rerun, de modo que la respuesta aparece igual y la conversación no se corta.

## Ambigüedad

Pregunta de ejemplo:

```text
¿Cuántos votos a favor obtuvo la iniciativa en la Cámara de Representantes?
```

La interfaz muestra la aclaración y las opciones (paz total, trazabilidad ganadera, servicio social, aguas de lastre, etc.). Al elegir "El Congreso aprueba ley para acabar con la contaminación a los mares", el sistema responde con los 95 votos y la cita. En la API de una pasada, `POST /ask` devuelve la respuesta condicional con todas las interpretaciones.

Otra prueba de alcance: "¿Cuántos municipios fueron afectados por el terremoto?" ofrece 472, 437 y 450; la respuesta condicional las lista con su cita.

## Qué mirar en cada respuesta

- ¿La respuesta usa solo la evidencia?
- ¿Cada afirmación tiene cita?
- ¿Las citas apuntan a documentos que existen?
- ¿La abstención aparece cuando la evidencia no alcanza?
- ¿El tiempo de recuperación se mantiene por debajo de 2 segundos?

## Frases para el cierre

- La evidencia recuperada es idéntica entre modelos; lo que cambia es el generador.
- El corpus completo tiene 618.286 chunks; la recuperación es más difícil que en el piloto, por eso la demo usa preguntas del gold validado.
- `gemma4:e4b` con MTP draft 3 es el generador final: 217 tok/s y 6,8 s de latencia media.
- El sistema entrega 37 de 38 preguntas del gold, se abstiene en las 5 sin respuesta y acierta 26 de 37 en match estricto.
- Con `Modo estricto` el chat da 14 de 14; sin la compuerta, 11 de 14.
- RAGAS actual: faithfulness 0,952; answer relevancy 0,840; context precision 0,881; context recall 0,921; answer correctness 0,657.
- Toda cifra del informe sale de `reports/corridas/` y `reports/tablas/`.
