# P104 - RONDA 3 (R4 + R5 + R7-1): pre-registro

Fecha: 2026-10-14. Origen: RONDA 3, Fase 1 (triage R1-R7). Alcance confirmado
por el propietario: SOLO R4, R5 y R7-1, en ese orden. Quedan fuera R1, R2, R3,
R6, R7-2 y R7-3. Presupuesto: 1.5 h de CPU como maximo.

## 1. Reglas comunes

- V03 congelado (corrected_cv 0.7772114525165721; umbrales 0.44/0.51/0.46).
- Sin test; sin holdouts; sin leaderboard; sin submissions; sin
  transformers/LLM; sin datos externos nuevos (los embeddings propios ya
  autorizados se reutilizan como artefacto congelado).
- Un experimento = un script nuevo; artefactos historicos intactos; ASCII.
- Control obligatorio en CADA experimento: M2 (word(1,2) + char_wb(3,5),
  min_df=2, max_df=1.0, sublinear, lowercase) + LR balanced C=2.0 liblinear
  seed 42, dentro del pliegue, debe reproducir el OOF congelado en la
  canonica (diff max de probabilidad <= 1e-6 y corrected_cv exacto) y en la
  particion 47 (pred identica; 0 celdas distintas).
- Folds: canonicos congelados (features/oof_v03_final.npz) y particion 47
  (seed 43; hash verificado contra reports/p98_14_preregistro.json).
- Evaluacion: umbrales V03 congelados; corrected_cv; delta vs V03; pliegues
  mejorados (de 5); celdas cambiadas; IC95 de bootstrap por clusters (2000
  remuestreos, seed 20260925); delta en la particion 47; piso de deteccion.
- Gate A3 (unico criterio de aceptacion): delta canonico >= +0.003; >= 4/5
  pliegues; IC95 excluye cero; delta 47 positivo; >= 200 celdas cambiadas.
- Piso de deteccion del arnes (declarado): se del bootstrap ~0.0030
  (mediana de las 6 mediciones de P102/P103); delta minimo detectable al 95%
  ~+0.0059 (mediana; rango +0.0033 a +0.0086). Si el efecto puntual queda por
  debajo del piso, se reporta "sin senal detectable" y NO se usa como
  argumento para cambiar de slot.
- Registro: filas P104-01/02/03 en experiments.csv; RUNLOG; DECISIONS.

## 2. P104-01 (R4) - stemming Snowball espanol

Hipotesis: el stemming Snowball sobre la rama de palabra aporta a V03.

- Stemmer: nltk.stem.SnowballStemmer('spanish') aplicado a cada token
  separado por espacio del texto (determinista, no se ajusta con datos, no
  toca test; se precomputa una vez sobre dev).
- Variantes (el resto del pipeline es identico a V03: min_df=2, sublinear,
  LR balanced C=2.0 liblinear seed 42, umbrales congelados):
  - S1 = word(1,2) sobre texto stemmed + char_wb(3,5) sobre texto crudo;
  - S2 = S1 + word(1,2) sobre texto crudo (union de stemmed y sin stemmar).
- Expectativa declarada: la lematizacion spaCy (PRE-09) quedo en 0.7425
  (-0.016 vs referencia); esperado ~0/negativo.
- Salidas: reports/p104_01_stemming_snowball.json y .md.

## 3. P104-02 (R5) - clusters de palabras como bolsa adicional

Hipotesis: una bolsa de clusters de palabra (k-means sobre los embeddings
propios) anade informacion a la union de V03.

- Embedding: modelos/twitter_domain/w2v_twitter_lrfix.npz (P72-00: 77.657
  palabras, 200d; entrenado sobre corpus externo Twitter/X de 10M tokens SIN
  etiquetas, autorizado en P71-02; no usa texto de TOXIC ni del test).
- k-means: MiniBatchKMeans(k, random_state=42, batch_size=4096, n_init=3)
  sobre las 77.657 palabras del artefacto (solo vocabulario externo).
- Justificacion de no-fuga (pre-registrada): ni el embedding ni el clustering
  usan etiquetas, folds ni texto de dev/test; el mapa token->cluster es una
  funcion fija del artefacto externo congelado; los TF-IDF (bloque de clusters
  y representacion de V03) se ajustan DENTRO del pliegue, como siempre. No se
  requiere refit fold-local.
- Variantes: k en {100, 300, 1000}; bloque nuevo = TF-IDF word(1,1), min_df=2,
  sublinear, sobre el texto de clusters (token -> 'c{id}' u 'coov'); hstack a
  la union exacta de V03.
- Mapeo de token: token.lower() en vocabulario; si no, version sin puntuacion
  de borde (incluye @ y #); si no, 'coov'.
- Salidas: reports/p104_02_clusters_palabras.json y .md.

## 4. P104-03 (R7-1) - L1 / Elastic Net por etiqueta

Hipotesis: la regularizacion L1/ElasticNet por etiqueta con seleccion anidada
supera a la L2 de V03.

- Malla (16 configuraciones): penalty l1 (l1_ratio None) y elasticnet con
  l1_ratio {0.2, 0.5, 0.8} x C {0.5, 1.0, 2.0, 4.0}; solver saga,
  class_weight balanced, random_state 42, max_iter 3000, tol 1e-3.
- Representacion: la union exacta de V03 (sin cambios).
- Seleccion ANIDADA por pliegue y etiqueta (CV interna de 3 pliegues; criterio
  F1 de la etiqueta @0.5); empates: primer elemento del orden de la malla.
- Reporta la configuracion elegida por pliegue/etiqueta y el numero de
  features no nulas (nnz) por etiqueta.
- Salidas: reports/p104_03_l1_elastic.json y .md.

## 5. Cierre

- Si ningun item pasa el gate A3: cierre de la ronda sin candidato; V03
  permanece congelado; slots privados por defecto: V03 + v08_meta_v2.
- Proyecto congelado al terminar; sin submissions.
