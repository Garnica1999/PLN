# Anotación humana de E4

Se anotan 76 respuestas: 19 preguntas respondibles del split de prueba por 4 modelos, repetición 1. El orden está barajado y el archivo no indica qué modelo respondió. No abras `key.csv` hasta terminar.

Segundo anotador confirmado: Julián Carrillo.

## Flujo con archivo

1. Envía `eval/annotations/Julian_Carrillo.xlsx` a Julián.
2. Él lo abre en Excel o LibreOffice, califica las 76 filas en las columnas `correccion`, `fidelidad`, `citas` y `abstencion`, y guarda.
3. Cuando lo devuelva, déjalo en `eval/annotations/` con el nombre que traiga.
4. Ejecuta:

```bash
/home/carlos/venv/bin/python src/scripts/annotation_kappa.py
```

Eso escribe `reports/tablas/e4_anotacion.csv`, `e4_kappa.csv` y `e4_h1.csv`, y decide si H1 se sostiene o se refuta.

Si el archivo llega con otro nombre, el script usa ese nombre como identificador del anotador. No hace falta renombrarlo, pero conviene que sea reconocible.

## Rúbrica

Cada eje va de 0 a 3. Usa la lista desplegable; no escribas otro valor.

- **Corrección**: 3 responde lo que pide la pregunta; 2 correcta con imprecisión; 1 parcial o con error relevante; 0 incorrecta o no responde.
- **Fidelidad**: 3 todo el contenido sale de la evidencia; 2 algún añadido menor; 1 inventa o contradice parte; 0 inventa el núcleo.
- **Citas**: 3 todas válidas y apuntan al pasaje que sostiene la respuesta; 2 válidas pero no al pasaje; 1 parciales; 0 sin citas o todas inválidas.
- **Abstención**: 3 responde cuando hay evidencia; 2 responde con duda razonable; 1 se abstiene teniendo evidencia clara; 0 se abstiene sin motivo.

La pregunta, la respuesta de referencia, el span, el criterio, la respuesta del modelo y las evidencias recuperadas están en la misma fila. Juzga contra la evidencia, no contra tu conocimiento del tema.

## Reglas

- No compares respuestas entre sí ni intentes adivinar el modelo.
- No cambies el orden de las filas ni el `item_id`.
- Si dudas, usa la nota y elige el valor que mejor describa el texto.
- No dejes ejes vacíos: el script rechaza las filas incompletas.

## App local (opcional)

Quien esté en la máquina del taller puede anotar en la interfaz, que produce el mismo CSV:

```bash
/home/carlos/venv/bin/streamlit run src/rag/annotate.py --server.port 8502
```

Abre `http://localhost:8502` y escribe el nombre. El avance queda en `eval/annotations/<nombre>.csv`.

## Cierre

Con las dos anotaciones, el script calcula el kappa de Cohen ponderado por eje, las medias por modelo y el contraste de H1 con bootstrap pareado por pregunta. Si el kappa es bajo, hay que revisar la rúbrica y volver a anotar antes de usar los resultados.
