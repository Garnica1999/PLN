# Revisión de documentación — secciones del notebook final (AGE 2026 PUJ)

**Fecha:** 2026-10-18. **Alcance:** los 19 archivos de `plan/kaggle_notebook/secciones/` (se incluye `fase3_5_residual_v13v17.md`, aunque ya no se ensambla). **Método:** contraste de la prosa contra `RUNLOG.md`, `DECISIONS.md` (D-106..D-110), `experiments.csv`, `CHAMPION.md` y los paquetes 05-07; verificación puntual contra los datos crudos y los artefactos (se reprodujo el OOF del campeón V48 = 0.6186451 y su F1 por clase). No se tocaron fences, tablas ni cifras.

## Corregido directamente (prosa)

- `01_fase1.md:52-53`: «con las filas de las variables barajadas» → «con las filas o las variables barajadas» (frase rota).
- `02_fase2.md:35`: «¿cómo de largos son los textos?» → «¿qué tan largos son los textos?» (registro y consistencia con la pregunta siguiente).
- `03_fase3.md:28-29`: «re-ejecutarlas costaba ~5 minutos ... y no cambia ninguna decisión» → «no cambiaba ninguna decisión» (concordancia temporal).

## Hallazgos para decisión (no modificados)

1. `01_fase1.md:8` — «mediana de 91 caracteres»: el corpus da **93** (train y test, tras `strip`; `FINAL_REPORT.md:1217` también dice 93). Arreglo: 93.
2. `02_fase2.md:101` — «razón mayoritaria/minoritaria ronda 1.2x»: el código de la propia celda y el glosario dan **1.062** (17.04/16.04). Arreglo: 1.06x.
3. `03_fase3.md:1,5-6` + `00_portada.md:60-61` — «fase histórica V1–V17, re-ejecutada» y «Cada experimento se vuelve a ejecutar aquí desde cero»: V13–V17 no están en el notebook y el meta-modelo V12 no se reconstruye (queda documentado; ver comentario del fence en `03_fase3.md:238-239`). Arreglo: acotar a V1–V12 re-ejecutado («los experimentos re-ejecutables»).
4. `03_fase3.md:28` — sigmas «±0.0030» y «±0.0044»: `experiments.csv` da **±0.0034** (E-002b) y **±0.0050** (E-003b). Arreglo: corregir ambos.
5. `04_fase4.md:3-4` — «La respuesta llegó en tres pasos» y luego se describen cuatro (V19→V20, V29, V30 y la regla plug-in). Arreglo: «cuatro pasos» o «tres pasos y la pieza final».
6. `04b_fase4_nn.md:26` — «Los seis componentes (más el HGB de V39) alimentan al campeón V48»: V48 usa 4 de los 6 (V39-ordinal, V40, V42, V43) + HGB + V30; V38 y V41 no entran. Arreglo: reformular («cuatro de los seis, más el HGB y V30»).
7. `05_fase5.md:80-82` — «los rangos centrales (25-34, 35-49) son los mejor clasificados» y «13-17 concentra parte del error»: está invertido. OOF del campeón: 65-xx 0.7446, 13-17 0.6845, 50-64 0.6712; peores 25-34 0.5051 y 18-24 0.5306 (coherente con V44: «las clases extremas superan 0.68 y 0.74»). Arreglo: invertir la frase.
8. `05_fase5.md:53` (comentario de celda) — «la sección 48 del notebook consolidado»: V48 es la **sección 47**; la 48 es V49/V50. Arreglo: citar 47 o quitar el número.
9. `06_fase6.md:8-10` y comentario de la celda (`:15`) — «diez comprobaciones»: la celda implementa **8** y la propia prosa enumera 8; `anexo_c` distingue «V48: 8/8» del pipeline del plug-in «10/10». Arreglo: «ocho comprobaciones» en prosa y comentario.
10. `06_fase6.md:95` — decimales con coma («0,115», «5,4e-5») en un documento con decimales con punto. Arreglo: 0.115 / 5.4e-5.
11. `06_fase6.md:99-108` — duplicación: «cada celda imprime el valor exacto...» aparece dos veces y «Segunda» / «En conjunto» repiten la misma tolerancia. Arreglo: unificar en una sola frase.
12. `06_fase6.md:132` — «un ensemble de árboles y cinco redes tabulares»: el PoE son 2 componentes de árboles (V30 y V39-HGB) y 4 redes. Arreglo: «dos ensembles de árboles y cuatro redes».
13. `00_portada.md:7` y `06_fase6.md:124` — «+0.00892 sobre el segundo lugar»: cifra del momento del envío (2.º = 0.62832); `CHAMPION.md` vigente da **+0.00196** (2.º = 0.63528). Arreglo: actualizar o precisar «al momento del envío».
14. `00_portada.md:16` — «tres restricciones inviolables» frente a las **cinco** de la tabla 1.2 (faltan «sin datos externos» y «sin metadatos»). Arreglo: «cinco» o «tres restricciones nucleares».
15. `00_portada.md:62-63` — «los fracasos seleccionados ... re-ejecutados»: 4.3 es [RE/DOC] y 7 de 8 subsecciones son [DOC]; además la guía de lectura no menciona el arco V53–V57. Arreglo: «documentados (con el MLP denso re-ejecutado)» y añadir el arco 2026.
16. `anexo_a_inventario.md:11-18,22` — A.1 totaliza **215** y A.2 arranca «sobre las 215 filas», pero el registro y el total de A.2 suman **220** (a A.1 le faltan las 5 filas V53-V57). Arreglo: añadir la columna V53-V57 (5 ejecutadas) y total 220.
17. `anexo_b_referencias.md:5,39` — «el apartado 49 del notebook `99_entrega_consolidado.ipynb`»: la bibliografía es hoy la **sección 54** (la 49 es V53). Arreglo: 54. Además «51 fichas»: `references.bib` tiene **52** entradas (30 previas + 22 añadidas).
18. `fase4_6_v53_v57.md:1` — el encabezado dice «[DOC]» pero la sección re-ejecuta (su intro dice «Sección mixta»); la convención del resto es «[RE/DOC]». Arreglo: cambiar el rótulo.
19. `01_fase1.md:11-12` — «La métrica oficial del leaderboard es F1 Score. Internamente el proyecto adoptó la macro-F1»: la portada y el glosario afirman que la métrica oficial **es** macro-F1. Arreglo: «F1 Score (macro-F1)» para alinear las tres secciones.
20. `fase4_5_v44_v52.md:29` (vs `DECISIONS.md` D-106/D-108) — V47 standalone: 0.6110824 (D-099, `experiments.csv:211`, checklist) frente a 0.613465/0.61347 (D-106/D-108). El notebook usa 0.6110824; conviene fijar la cifra canónica antes de ejecutar 4.6.6 (la celda imprime el valor reconstruido).
21. Citas huérfanas: `03_fase3.md:124` (MATTR, Yule, Maas) y `04b`/`fase4_5` («RankGauss», y con el mismo criterio «winsorización» y «Power-Mean») aparecen sin autor+año. Arreglo: añadir fichas (p. ej. Covington y McEnery 2010; Yule 1944; Maas 1972; Johannemann et al. 2019) o marcarlas.

**Dudas:** (a) si la métrica del leaderboard es macro-F1 por definición de la competencia, el punto 19 se resuelve solo con la redacción; (b) los dos hallazgos de conteo (A.1 y «51 fichas») dependen de si el anexo debe incluir ya el arco V53-V57 en todas sus tablas.
