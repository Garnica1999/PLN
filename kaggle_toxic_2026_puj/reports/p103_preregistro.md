# P103 - RONDA 2 (N5 + N6): pre-registro

Fecha: 2026-10-14. Origen: RONDA 2, Fase 1 (triage N1-N8). Alcance confirmado
por el propietario: SOLO N5 y N6. N1, N2 y N8-A/B/C quedan descartados (datos
externos no permitidos por el profesor; N2 no se consulta por delta esperado
~0 y riesgo de reglas).

## 1. Reglas comunes

- V03 congelado (corrected_cv 0.7772114525165721; umbrales 0.44/0.51/0.46).
- Sin test; sin holdouts; sin leaderboard; sin submissions; sin
  transformers/LLM; sin datos externos.
- Un experimento = un script nuevo; artefactos historicos intactos; ASCII en
  artefactos generados.
- Control obligatorio en CADA experimento: M2 (word(1,2) + char_wb(3,5),
  min_df=2, max_df=1.0, sublinear, lowercase) + LR balanced C=2.0 liblinear
  seed 42, ajustado dentro del pliegue, debe reproducir el OOF congelado en
  la canonica (diff max de probabilidad <= 1e-6 y corrected_cv exacto) y en
  la particion 47 (pred identica; 0 celdas distintas).
- Folds: canonicos congelados (features/oof_v03_final.npz, verificados contra
  src.splits.pliegues_corregidos) y particion 47 (seed 43; hash verificado
  contra reports/p98_14_preregistro.json).
- Evaluacion: umbrales V03 congelados; corrected_cv; delta vs V03; pliegues
  mejorados (de 5); celdas cambiadas; IC95 de bootstrap por clusters (2000
  remuestreos, seed 20260925); delta en la particion 47.
- Gate A3 (unico criterio de aceptacion, todas obligatorias): delta canonico
  >= +0.003; >= 4/5 pliegues; IC95 del bootstrap excluye cero; delta positivo
  en la particion 47; >= 200 celdas cambiadas vs V03.
- Poder estadistico (declarado antes de correr): con este arnes el error
  estandar del bootstrap por clusters es ~0.003-0.004 (IC de P102-01/02/04);
  el delta minimo detectable con IC95 que excluya cero es ~+0.006 a +0.008, y
  con 80% de potencia ~+0.008 a +0.011. Si el delta puntual queda por debajo
  del piso de deteccion, el resultado se reportara como "sin senal
  detectable", sin forzar conclusion.
- Presupuesto: 3 h de CPU en total. Parada: si se agota, cierre con lo
  ejecutado.
- Registro: filas P103-01 y P103-02 en experiments.csv; RUNLOG; DECISIONS.

## 2. P103-01 (N5) - vectorizadores por etiqueta con seleccion anidada

Hipotesis: cada etiqueta (HS/TR/AG) puede elegir su propio espacio TF-IDF
(min_df, max_df, rango de palabra, rango char_wb) y esa libertad, seleccionada
de forma anidada, mejora a V03 mas alla del gate A3.

Malla por etiqueta (20 configuraciones, fijada antes de correr):
- 18 = min_df {1,2,3} x max_df {0.9,1.0} x rango word {(1,1),(1,2),(1,3)},
  con char_wb fijo en (3,5);
- 2 = rango char_wb {(2,5),(3,6)} en el ancla (min_df=2, max_df=1.0,
  word=(1,2)).
Ambos bloques usan la misma (min_df, max_df); sublinear_tf=True;
lowercase=True. El ancla (2, 1.0, (1,2), (3,5)) es la configuracion de V03.

Seleccion ANIDADA: por pliegue externo y etiqueta, CV interna de 3 pliegues
(StratifiedGroupKFold, seed 42, grupos fusionados, estratificacion por
combinacion HSxTRxAG) sobre el train del pliegue externo; criterio = media de
la F1 binaria de la etiqueta con umbral 0.5 en los pliegues internos; empates:
primer elemento del orden de la malla. PROHIBIDO elegir mirando el OOF global.
La configuracion elegida se aplica una sola vez al pliegue externo.

Evaluacion: OOF canonico (5 pliegues congelados) y particion 47 (mismo
protocolo anidado). Reporta la configuracion elegida en cada pliegue y su
estabilidad (conteo de elecciones por etiqueta).

Paralelizacion: joblib con 8 procesos sobre configuraciones; cada ajuste es
independiente y determinista (no cambia el resultado).

Salidas: reports/p103_01_vectorizadores_por_etiqueta.json y .md.

## 3. P103-02 (N6) - word dropout / masking conservador

Hipotesis: aumentar solo el train de cada pliegue con copias con dropout de
tokens (p 0.05-0.15; 1-2 copias) mejora a V03 mas alla del gate A3.

Aumento: tokens separados por espacio; cada token se elimina con probabilidad
p; k copias por documento; si una copia queda vacia se conserva el original.
Semillas deterministas: 20261014 + 1000*bloque (0 canonica, 1 particion 47) +
100*pliegue + 10*interno + copia. SOLO el train del pliegue; nunca validacion
ni test.

Malla corta (6 configuraciones): p {0.05,0.10,0.15} x copias {1,2}.
Seleccion anidada por pliegue (una configuracion por pliegue para las tres
etiquetas): CV interna de 3 pliegues sobre el train aumentado; criterio =
f1_media_columnas (umbral 0.5) medio en validacion interna (sin aumentar).
Modelo fijo: M2 (V03); vectorizador y LR ajustados dentro del pliegue.

Evaluacion y gate: identicos a P103-01.

Salidas: reports/p103_02_word_dropout.json y .md.

## 4. Cierre

- Si ningun item pasa el gate A3: sprint cerrado sin candidato; V03 permanece
  congelado; recomendacion de slots privados por defecto: V03 +
  v08_meta_v2. Si algun item pasara el gate completo, se propone para el
  slot 2 frente a v08_meta_v2 (V03 no se toca).
- Proyecto congelado al terminar: sin procesos pendientes; sin submissions.
