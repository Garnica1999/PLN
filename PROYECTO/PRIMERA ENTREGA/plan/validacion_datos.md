# Validacion de consistencia de datos (Fase 7, Tarea 0)

Fuente de verdad: `resultados/metricas.json` (notebook, corpus completo) y total sin stopwords reportado por el notebook (128.806.664).
Tolerancia en diapositivas: redondeos presentacionales con error relativo < 0,15 %.
En el informe (tex/pdf) las cifras de la tabla deben ser exactas.

| Cifra (canonica) | Deck compañero | Deck nuestro | Informe tex | Informe pdf |
|---|---|---|---|---|
| Filas CSV (23.750) | OK | OK | OK | OK |
| Sentencias (23.742) | OK | OK | OK | OK |
| Excluidas (8) | OK | OK | OK | OK |
| Tutela (17.099) | OK | OK | OK | OK |
| Constitucionalidad (6.246) | OK | OK | OK | OK |
| Unificacion (397) | OK | OK | OK | OK |
| Palabras raw (260.988.316) | OK | OK | OK | OK |
| Palabras limpias (250.012.101) | OK | OK | OK | OK |
| Sin stopwords (128.806.664) | OK | OK | OK | OK |
| Promedio raw/doc (10.992,68) | OK | OK | OK | OK |
| Promedio limpio/doc (10.530,37) | FALTA | FALTA | OK | OK |
| Anio minimo (1992) | OK | OK | OK | OK |
| Anio maximo (2021) | OK | OK | OK | OK |
| Docs 2021 (20) | OK | OK | OK | OK |
| Stopwords (48 %) | OK | OK | OK | OK |
| Parseo (99,97 %) | OK | OK | OK | OK |
| Excluidos autos (6) | OK | OK | OK | OK |
| Excluidos HTML (2) | OK | OK | OK | OK |
| Anexo (25) | OK | OK | OK | OK |
| Top-1 con (de 22.543.375) | FALTA | OK | OK | OK |
| Top-1 sin (articulo 947.616) | FALTA | OK | OK | OK |
| Stopwords formas (306) | OK | OK | OK | OK |

Cifras exactas exigidas al informe (tex) que NO aparecen: ninguna (todas presentes)

## Figuras incrustadas vs figuras del notebook (md5)
- fig_top20_con_stopwords.png: presente (md5 identico) en el deck del companero
- fig_top20_sin_stopwords.png: presente (md5 identico) en el deck del companero
- fig_sentencias_por_anio.png: presente (md5 identico) en el deck del companero
- fig_nube_global.png: presente (md5 identico) en el deck del companero

## Apendice (tras fusion del deck del companero, 2026-09-06)
- Re-ejecutada la validacion sobre PLN_PROYECTO.pptx fusionado (22 slides) y con la ruta del PDF corregida.
- Mejora: 'Stopwords formas (306)' ahora OK (anadido en paso 09 de procesamiento).
- Siguen como FALTA (ausencia, NO discrepancia): promedio limpio/doc (10.530,37) y top-1 con/sin (22.543.375 / 947.616) — el deck del companero muestra esas cifras DENTRO de las figuras (barras con etiquetas) y usa el promedio raw (~10.993) como KPI; no contradicen ningun valor canonico.
- Discrepancia real corregida en el proceso: informe decia "75 formas" de dominio juridico y el notebook usa 65 → main.tex actualizado a "65 formas" y PDF universitario recompilado (8 paginas).
- Conclusion: datos CONSISTENTES entre notebook, diapositivas (ambos decks) e informe.
