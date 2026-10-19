# P105 - RONDA 4 (S2/S7a + S4/S7b + S7c): pre-registro

Fecha: 2026-10-14. Origen: RONDA 4, Fase 1 (triage S1-S7). Alcance confirmado
por el propietario: SOLO S2-residual/S7a, S4/S7b y S7c, en ese orden.
Presupuesto: 1 h de CPU como maximo.

## 1. Reglas comunes

- V03 congelado (corrected_cv 0.7772114525165721; umbrales 0.44/0.51/0.46).
- Sin test; sin holdouts; sin leaderboard; sin submissions; sin
  transformers/LLM; sin datos externos (se reutilizan artefactos ya
  autorizados solo en lo ya pre-registrado).
- Un experimento = un script nuevo; artefactos historicos intactos; ASCII.
- Control obligatorio: M2 (word(1,2) + char_wb(3,5), min_df=2, max_df=1.0,
  sublinear, lowercase, norm='l2' por defecto) + LR balanced C=2.0 liblinear
  seed 42, dentro del pliegue, que reproduce el OOF congelado en la canonica
  (diff <= 1e-6; ccv exacto) y la particion 47 (0 celdas distintas).
- Gate A3 (unico criterio): delta canonico >= +0.003; >= 4/5 pliegues; IC95
  por clusters excluye cero; delta 47 positivo; >= 200 celdas cambiadas.
- Piso de deteccion declarado (P102-P104): se ~0.0035 mediano (rango
  0.0017-0.0059); delta minimo detectable 95% ~+0.0069 mediano (rango +0.0033
  a +0.0115). Efectos por debajo del piso se reportan "sin senal detectable".
- Registro: filas P105-01/02/03 en experiments.csv; RUNLOG; DECISIONS.

## 2. P105-01 (S2/S7a) - normalizacion por bloque vs conjunta

Bloques crudos: TF-IDF word(1,2) y char_wb(3,5), min_df=2, max_df=1.0,
sublinear, lowercase, norm=None (sin normalizar).

- **V1_l2_por_bloque**: L2 por fila a cada bloque + hstack. Nota
  pre-registrada: con los defaults de sklearn esto es IDENTICO a V03 (sus
  bloques ya son unitarios por fila); se ejecuta como verificacion de
  identidad (diff esperado 0.0) y NO compite como candidato.
- **V2_l2_conjunta**: hstack de bloques crudos + L2 por fila CONJUNTO (un
  solo factor por documento que se reparte entre bloques segun sus normas
  crudas). La lectura alternativa (L2 conjunto sobre el hstack ya unitario de
  V03) se reduce a un reescalado global ~1/sqrt(2) (equivalente a C x2,
  familia V8-06 ya medida) y por eso no se duplica.
- **V3_norma_media**: bloques crudos; cada bloque reescalado para que sus
  filas tengan norma igual a la norma cruda MEDIA del bloque en el train del
  pliegue (equivalente: fila unitaria x norma media). Fold-local.

## 3. P105-02 (S4/S7b) - folding fonetico espanol

- Reglas, en este orden exacto: lower; 'qu'->'k'; 'ch' protegido; 'ce'->'se'
  y 'ci'->'si' (seseo); 'z'->'s'; 'c'->'k'; 'll'->'y'; 'v'->'b'; 'h' muda
  eliminada salvo precedida de 'c'.
- Canal: char_wb(3,5) sobre el texto PLEGADO, anadido a la union de V03
  (word y char_wb originales intactos). Transformacion determinista, sin
  ajuste, sin test; se precomputa una vez.
- Variante unica; control; gate; ambas particiones.

## 4. P105-03 (S7c) - pesos de bloque por etiqueta (nested)

- Malla corta: char_weight en {0.5, 0.75, 1.0, 1.25, 1.5} (word = 1.0);
  union = hstack([word, w * char]).
- Seleccion ANIDADA por pliegue y etiqueta (CV interna de 3 pliegues
  StratifiedGroupKFold seed 42 sobre el train del pliegue; criterio = F1 de
  la etiqueta @0.5; empates: primer valor de la malla). Reporta los pesos
  elegidos por pliegue y su estabilidad.
- Control; gate; ambas particiones.

## 5. Cierre

- Sin candidato por defecto: V03 permanece; slots privados: V03 +
  v08_meta_v2. Proyecto congelado al terminar; sin submissions.
