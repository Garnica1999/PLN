# Guia de anotacion - Validacion humana de etiquetas silver M4

Fecha de generacion: 2026-10-03
Script: scripts/21_pack_validacion_m4.py (semilla 42)
Muestra: muestra_200.jsonl (200 documentos)

## 1. Objetivo

Medir la precision real del etiquetado silver de M4. El silver asigna a cada
sentencia el bloque tematico de la Constitucion (CP) al que pertenecen la
mayoria de las citas detectadas (via cita_cp) o, si no hay citas, un termino
lexico de respaldo (via fallback_derecho, menos fiable). La anotacion humana de
una muestra permite estimar la exactitud del silver y decidir si es una base
confiable para entrenar y evaluar.

Esta anotacion es diagnostico: NO entrena nada. Por eso la muestra se tomo del
corpus etiquetado completo (23.654 documentos) e incluye documentos de train,
val, test y del holdout sellado.

## 2. Que es el "bloque constitucional predominante"

La Constitucion esta dividida en 16 bloques tematicos (mapa articulo de la CP
hacia bloque, ADR-1). El bloque predominante de un documento es el bloque al que
pertenecen la mayor cantidad de articulos DISTINTOS de la CP citados en el
documento. Los empates se rompen por el articulo citado de menor numero. Su
tarea es decidir cual es el bloque predominante leyendo el documento.

## 3. Como leer la metadata

- via: cita_cp (etiqueta derivada de citas a la CP, ruta principal) o
  fallback_derecho (sin citas; se uso un termino lexico; menos fiable).
- n_citas_cp: numero de citas a articulos de la CP detectadas en el documento.
- top3: los tres bloques con mas articulos distintos citados y su conteo. Es
  parte del silver: uselo al final, solo para verificar.
- particion: train/val/test del protocolo M4; en_holdout marca documentos del
  holdout sellado (aparecen porque la anotacion no entrena).

## 4. Regla de oro

1. Lea el extracto completo (si necesita mas texto, use --ver ID en otra
   terminal).
2. Decida el bloque predominante SOLO con el texto.
3. Recien entonces compare con el bloque SILVER y registre su respuesta.

No consulte el producto, ni anotaciones previas, ni el silver de otros
documentos antes de decidir. El bloque SILVER se imprime al final del documento
justamente para facilitar esta regla.

## 5. Opciones de respuesta

- Una de las 16 clases del menu (numero o alias):
-  1. derechos_colectivos_ambiente (alias: ambiente, colectivos)
-  2. derechos_fundamentales (alias: fundamentales, df)
-  3. derechos_sociales_economicos_culturales (alias: sociales, desc)
-  4. elecciones (alias: elecciones, elec)
-  5. habitantes_territorio (alias: habitantes, fronteras)
-  6. organismos_control (alias: control, organismos)
-  7. organizacion_estado (alias: estado)
-  8. organizacion_territorial (alias: territorial)
-  9. participacion_democratica (alias: participacion, democracia)
- 10. principios_fundamentales (alias: principios, pf)
- 11. proteccion_aplicacion_derechos (alias: proteccion, tutela)
- 12. rama_ejecutiva (alias: ejecutiva, gobierno)
- 13. rama_judicial (alias: judicial)
- 14. rama_legislativa (alias: legislativa, congreso)
- 15. reforma_constitucion (alias: reforma)
- 16. regimen_economico (alias: economico, regimen)
- ambiguo (0): dos o mas bloques compiten sin predominio claro, o el texto no
  alcanza para decidir.
- ninguno (-): ningun bloque constitucional predomina (documento procesal,
  administrativo o sin anclaje tematico claro en la CP).

## 6. Confianza

- alta: el bloque predominante es evidente tras leer el documento.
- media: hay un bloque probable pero con dudas razonables.
- baja: decision al limite entre dos o mas bloques.

Enter = media. La confianza no cambia la respuesta, pero permite analisis
posteriores (por ejemplo, exactitud en el subconjunto de confianza alta).

## 7. Notas

Opcional. Recomendado anotar los articulos o temas que sustentan la decision, o
la razon de "ambiguo"/"ninguno". Enter para omitir.

## 8. Tiempo estimado

1 a 3 minutos por documento (200 documentos = 3 a 10 horas). Se puede
anotar en tandas: el progreso se guarda despues de cada respuesta y --anotar
reanuda automaticamente donde quedo.

## 9. Duplicados

Si dos documentos parecen el mismo texto o muy similares, anote ambos: cada id
cuenta. Registre en la nota que es un posible duplicado. No salte documentos por
esta razon.

## 10. Flujo de comandos

    /home/carlos/venv/bin/python scripts/21_pack_validacion_m4.py --resumen
    /home/carlos/venv/bin/python scripts/21_pack_validacion_m4.py --anotar --anotador carlos
    /home/carlos/venv/bin/python scripts/21_pack_validacion_m4.py --ver 12345
    /home/carlos/venv/bin/python scripts/21_pack_validacion_m4.py --exportar-csv
    /home/carlos/venv/bin/python scripts/21_pack_validacion_m4.py --calcular-acuerdo

Durante --anotar: numero o alias para responder, 0 = ambiguo, - = ninguno,
s = saltar el documento (queda pendiente), q = guardar y salir, ? = menu.

## 11. Segundo anotador y acuerdo

Para medir acuerdo inter-anotador, un segundo anotador debe ejecutar --anotar
con otro nombre (por ejemplo --anotador ana) sobre la misma muestra. Luego
--calcular-acuerdo reporta exactitud vs silver por anotador, matriz de confusion
silver -> humano, porcentaje de ambiguo/ninguno, kappa de Cohen por pares y
Fleiss global (sobre documentos respondidos por todos).
