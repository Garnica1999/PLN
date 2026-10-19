# Dossier de redacción — Informe final de la competencia AGE (`tweet-age-2026-puj`)

Este dossier es la fuente única de verdad para redactar `informe/informe.tex`. Todas las cifras
aquí listadas están verificadas contra los artefactos del proyecto (`CHAMPION.md`, `RUNLOG.md`,
`DECISIONS.md`, `experiments.csv`, el notebook ejecutado y los JSON de experimentos). No inventar
cifras nuevas: si algo no está aquí, se puede consultar la fuente indicada en la sección 8.

## 0bis. Mapeo de decisiones (artículo ↔ registro del proyecto)

El artículo numera las decisiones cruciales como Decisión 1..11 (sin códigos internos). Equivalencia
con el registro del proyecto (`DECISIONS.md`): 1↔D-003 (validación agrupada), 2↔D-005 (padding),
3↔D-009 (ASCII), 4↔D-004 (política de envíos), 5↔D-087 (techo 0.642), 6↔D-101 (PoE geométrico),
7--11↔D-106..D-110 (arco 2026).

## 0. Especificación del entregable

- **Archivo**: `informe/informe.tex` (la clase `IEEEtai.cls` y `references.bib` ya están copiadas
  en la misma carpeta). Compilación: `pdflatex` -> `bibtex` -> `pdflatex` x2 (TinyTeX; ya probada).
- **Clase**: `\documentclass[journal]{IEEEtai}` (IEEE Transactions on Artificial Intelligence).
- **Idioma**: español (paquete `babel` con opción `spanish`), redacción humanizada, clara, técnica,
  semiformal. Sin emojis. Sin frases de relleno tipo "es importante señalar".
- **Extensión**: 12–15 páginas máximo (objetivo 13–14, dos columnas).
- **Autores** (4, mismos de los talleres del curso PLN):
  Erika Valeria Rivera Jiménez (riveraev@javeriana.edu.co), Julián Guillermo Carrillo Meneses
  (carrillojuliang@javeriana.edu.co), Carlos Andrés Garnica Salazar (carlos_garnicas@javeriana.edu.co),
  Oscar Andrés Prieto Moreno (o-prieto@javeriana.edu.co). Afiliación: Pontificia Universidad
  Javeriana, Maestría en Inteligencia Artificial, Bogotá, Colombia.
- **Título** (responde a la pregunta de investigación; no cambiar sin autorización):
  **"Predicción de rangos etarios a partir de tweets en español mediante técnicas de procesamiento del lenguaje natural y aprendizaje automático supervisado"**
- **Resumen** (~230 palabras) + **IEEEImpStatement** (3–4 frases de impacto).
- **Bibliografía**: `\bibliographystyle{ieeetr}` + `\bibliography{references}` (52 fichas en
  `references.bib`; NO citar `rosenthal2011`, está marcada como no verificada). Citación numérica
  IEEE con claves `\cite{clave}`.

## 1. Pregunta de investigación y contribuciones

**Pregunta**: ¿hasta dónde puede llegar un sistema de PLN clásico —sin transformers, sin
embeddings y sin LLM— en la clasificación del rango de edad de tweets en español, y dónde está el
límite real del dato?

**Respuesta que articula el artículo** (esto es la columna vertebral del texto):
1. Llega al **primer puesto del leaderboard** (LB 0.63724, Ref. 57283487) con un consenso
   geométrico de expertos clásicos y una regla de decisión plug-in, superando el baseline inicial
   (0.55235) en +0.08489.
2. El **récord local de validación** es 0.6189929 (macro-F1, LOFO agrupado de 5 folds) con un
   consenso de nueve modelos (V56), estadísticamente significativo frente al campeón (5/5 folds,
   bootstrap pareado p = 0.0010).
3. El **techo del dato** está medido y declarado: LB 0.642 (acta D-087); 0.70 es inalcanzable bajo
   las restricciones.
4. La búsqueda sistemática de señal (220 filas de experimentos, 110 decisiones D-001..D-110)
   delimitó qué técnicas aportan y cuáles no, con falsaciones reproducibles.
5. Todo es **reproducible**: notebook Kaggle autocontenido (178 celdas, 4 h 35 min, 13/13
   verificaciones, idempotencia exacta 12/12 incluida la SHA de la submission).

## 2. Estructura sección por sección

### I. Introducción (~1.25 págs)
- El problema: perfilar la edad del autor de un tweet (6 rangos: 13-17, 18-24, 25-34, 35-49,
  50-64, 65-xx) es un problema de estilometría con señal débil y vocabulario solapado entre
  cohortes vecinas; literatura previa (Schler et al. 2006; Nguyen et al. 2013; PAN 2013-2019).
- La competencia `tweet-age-2026-puj` (Kaggle, curso PLN PUJ): 169.336 tweets de entrenamiento,
  112.892 de test, métrica macro-F1.
- **Restricciones inviolables** (D-001): prohibidos transformers, LLM y todo modelo de
  vectorización/embedding (ni preentrenado ni entrenado). Todo el desempeño viene de TF-IDF,
  features artesanales y modelos clásicos.
- Contribuciones (lista de 5, ver sección 1 de este dossier).
- Mencionar el artículo con `\IEEEPARstart`.

### II. Trabajos relacionados (~1 pág)
Cuatro corrientes, cada una con su "qué se tomó":
- **A. Perfilado de autor y edad**: Schler et al. 2006 (esquema de 6 rangos idéntico), Argamon et
  al. 2007, Nguyen et al. 2011/2013/2014 (la edad en Twitter es difícil), Sap et al. 2014,
  Morgan-López et al. 2017, PAN (Rangel et al. 2013-2019), Verhoeven et al. 2016.
- **B. Representación estadística y NB-SVM**: Wang & Manning 2012 (NB-SVM, base del baseline
  V12), Mukherjee & Liu 2010 (log-count ratio), Sapkota et al. 2013/2015 (n-gramas de carácter),
  López-Monroy et al. 2013, Monroe et al. 2008 (log-odds).
- **C. Ordinalidad y distancias**: Frank & Hall 2001, Cao et al. 2020 (CORAL), Díaz & Marathe 2019
  (soft labels ordinales), Cramér 1928 / von Mises 1931 (distancias), Székely 2013, Frogner et
  al. 2015 (transporte).
- **D. Redes tabulares, ensambles y decisión**: Gorishniy et al. 2021 (revisión de DL tabular),
  Hu et al. 2018 (SE), Huang et al. 2017 (DenseNet), Izmailov et al. 2018 (SWA), Loshchilov &
  Hutter 2017 (cosine), Szegedy 2016 / Müller 2019 (label smoothing), Klambauer 2017 (SNN),
  Hinton 2002 (product of experts), Heskes 1998 (logarithmic pools), Ke et al. 2017 (LightGBM),
  Prokhorenkova 2018 (CatBoost), y optimización de F-medidas (Lipton et al. 2014 — base del
  plug-in; Parambath 2014; Waegeman 2014; Narasimhan 2015; Eban 2017).
- **Leakage**: Kaufman et al. 2012; paradoja de Simpson (Simpson 1951) como cierre de V52.

### III. Datos y análisis exploratorio (~1.5 págs) — Figuras 1-3
- Tabla I con: train 169.336 (`id;text;age_range`), test 112.892 (`id;text`), submission 112.892.
- Clases casi balanceadas por diseño: 13-17 17.0 %, 18-24 16.9 %, 25-34 16.7 %, 35-49 16.7 %,
  50-64 16.6 %, 65-xx 16.0 %.
- Artefactos del corpus: ancho fijo de 140 caracteres con relleno (se recorta, D-005);
  transliteración a ASCII (único no-ASCII `¿` con 88 apariciones, D-009); mediana de 93
  caracteres; 104 textos vacíos en train y 62 en test; 643 filas (0.38 %) con etiquetas
  contradictorias para el mismo texto; `id` descartado como feature (desviación máxima de 1.25 pp
  entre deciles); sin metadatos (no hay usuario, fecha ni seguidores).
- Provenance no identificada: no se usaron datos externos (consecuencia declarada).
- **Duplicados y near-duplicados** (el riesgo dominante): 1.21 % de las filas de test aparecen
  literalmente en train; 4.26 % comparten clúster de near-duplicado (Jaccard >= 0.8 sobre tokens
  con blocking por tokens raros); 92.8 % de los clústeres son de etiqueta única; 271.539 clústeres
  sobre 282.228 textos (train+test).
- Suelos: azar uniforme 1/6 = 0.1667; suelo estricto (predecir la mayoritaria) 0.04854.
- Figuras 1 (clases + longitudes), 2 (términos por clase), 3 (clústeres de near-duplicados).

### IV. Metodología y decisiones fundacionales (~1.5 págs) — Tabla II
Protocolo (D-002, D-003):
- Métrica: macro-F1 (la del leaderboard) sin ponderar sobre 6 clases.
- **StratifiedGroupKFold de 5 folds agrupado por clúster de near-duplicados** (D-003): la decisión
  que evitó el autoengaño. Evidencia: un baseline de memorización pura alcanza 0.0608 con split
  estratificado ingenuo y 0.0000 con el esquema agrupado.
- Comparación de modelos por leave-one-fold-out (LOFO) sobre predicciones out-of-fold; el test
  nunca participa del ajuste.
- **Segunda partición** (semilla 987654, solape parcial declarado): validación del linaje
  V30/campeón; reproduce 0.6144120 de base y 0.6164700 con plug-in (valores publicados).
- **Regla plug-in por clase** (Lipton et al. 2014): `argmax(p * w)` con pesos congelados
  [1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128]; se decide sobre validación, nunca sobre test.
- Política de envíos (D-004): el supuesto inicial de un límite de ~10 envíos era erróneo; hubo 15
  envíos registrados (12 puntuados al momento de la calibración). Calibración CV<->LB: brecha
  media +0.0177 ± 0.0023, explicada por la memorización de near-duplicados del test.
- Tabla II: decisiones cruciales (D-003 validación agrupada; D-005 padding; D-009 ASCII; D-004
  política de envíos; D-087 techo 0.642; D-101 PoE geométrico; D-106..D-110 arco 2026) con
  "impacto" en una frase cada una.

### V. Representación, features y línea base (~1.25 págs) — Figuras 4-5
- E-001 (baseline): TF-IDF `word(1-2)` + `char_wb(3-5)` + LinearSVC: CV 0.5344 ± 0.0050; primer
  envío LB 0.55235. Variantes documentadas: solo palabra 0.4971 ± 0.0034; solo carácter
  0.5408 ± 0.0050.
- La matriz del meta (279 columnas): 78 de 14 miembros (probabilidades OOF con transformación log
  cuando corresponde) + 35 densas (textometría: longitudes, signos, entropías, dígitos, zlib) +
  87 estructuradas (entidades: URLs, menciones, hashtags, horas, números, con medias condicionales
  por clase, márgenes y entropías) + 73 supervisadas in-fold + 6 de ADV-07 (márgenes NB-SVM de
  palabra). 14 miembros individuales: E-009..E-028a (LogReg/MLP/árboles/NB/SGD) + ADV-05 (NB-SVM
  char). El aporte de cada miembro está en la Figura 5.
- Línea V1-V20: V12 (NB-SVM + promedio LogReg/MLP) CV 0.6005 / LB 0.62182; V19 (+LightGBM)
  0.6079388 / 0.62578; V20 (cuartos + umbrales anidados) 0.6097782 / 0.62689.
- Clústeres de near-duplicados (271.539) como agrupación de la validación (Figura 3).

### VI. EL CAMPEÓN: V48 Geometric Product-of-Experts (~2 págs) — Figuras 6-9 — ÉNFASIS
Esta es la sección más importante (pedido explícito del autor del informe). Contenido:
- **Formulación**: consenso en log-espacio `log P ∝ Σ_m w_m log P_m` (producto de expertos,
  Hinton 2002; pools logarítmicos, Heskes 1998), pesos normalizados, recorte eps 1e-12 y
  renormalización por fila. Es el óptimo de la familia de operadores de consenso probada en el
  proyecto (V52: la curva Power-Mean es cóncava con vértice en el PoE geométrico).
- **Los seis expertos y sus pesos** (Tabla/figure): V30 LightGBM refinado con Optuna w=0.5628
  (Ke et al. 2017; Akiba et al. 2019 vía Optuna en el proceso); V42 TabNet suavizado + SWA
  w=0.1724; V43 TabNet Cramér + SWA w=0.0832; V40 SE-TabNet w=0.0754; V39-HGB
  HistGradientBoosting w=0.0549; V39-Ordinal TabNet w=0.0513.
- **Regla plug-in** sobre el consenso (Lipton et al. 2014) y su efecto por clase (Tabla V):
  sube 18-24 (+0.0064), 65-xx (+0.0065), 25-34 (+0.0019), baja 13-17 (-0.0045).
- **Resultado**: CV 0.6186451 (esquema publicado: el miembro V30 usa su OOF de la segunda
  partición) y **LB 0.63724, puesto #1** (Ref. 57283487). La variante con el V30 canónico es
  0.6177 (declarada). Ablación cualitativa: sin V30 el consenso pierde ~0.006; los miembros
  neuronales aportan el resto de la mejora sobre V30+plug-in (0.6159732).
- **Progresión hasta el plug-in** (Figura 9) y **matriz de confusión / F1 por clase del campeón**
  (Figura 8): los extremos se clasifican mejor (65-xx F1 0.7416 con plug-in; 13-17 0.6809) y los
  rangos centrales concentran el error (25-34 0.5011; 18-24 0.5273) por solapamiento de
  vocabulario entre cohortes vecinas.
- **Calibración CV<->LB**: brecha ~+0.0186 (0.63724 - 0.6186451), consistente con la brecha
  histórica +0.0177 ± 0.0023.

### VII. Qué funcionó y qué no (~1.5 págs) — Tabla IV
- **Funcionó**: NB-SVM + n-gramas de carácter (V12); boosting y umbrales anidados (V19/V20);
  Optuna sobre la matriz 279 (V30, +0.0037); plug-in (V30 +0.0025); redes tabulares con pérdidas
  ordinales y SWA (V38-V43: de 0.6167 a 0.6185); el PoE geométrico (V48).
- **No funcionó (falsaciones documentadas, V13-V17 y V44-V52)**: features lingüísticas por
  potencia (POS, cinco fases, ninguna adopción); poda por colinealidad (-0.0040); distancias L1
  en fronteras (V44); RankGauss (V46); CatBoost oblivious (V49); MHSA sobre la matriz (V50);
  Power-Mean con alpha != 0, SNN y la paradoja de Simpson (V52). Nota elegante: V47 (Cramér
  asimétrico con recorte) y V51 (Cramér consistente) fallaron como modelos individuales pero
  fueron **adoptados como miembros del consenso 2026** (su señal débil es complementaria).
- **El techo**: D-087 declara LB 0.642 alcanzable como máximo; 0.70 refutado; la búsqueda de
  señal residual quedó saturada (lecciones cuantificadas).
- Tabla IV: 5-7 falsaciones clave con hipótesis, resultado y lección.

### VIII. La frontera 2026 y la auditoría matemática (~2 págs) — Figuras 10-12
- **V53** Dense-SE-TabularNet (SE: Hu et al. 2018; DenseNet: Huang et al. 2017) + consenso PoE de
  7 modelos: CV 0.6188259, LB 0.63706, bootstrap p = 0.0280.
- **V54** consenso de 8 (unifica Cramér simétrico + asimétrico con recorte): CV 0.6188887,
  LB 0.63712, p = 0.0050.
- **V55** octa-modelo limpio (excluye redes subóptimas, inyecta CatBoost): LB 0.63707; protocolo
  de blindaje.
- **V56** consenso de 9 modelos: **récord CV 0.6189929**, LB 0.63717 (Ref. 57350295), 5/5 folds
  positivos, bootstrap pareado vs V48 p = 0.0010 con IC95 [+0.0001485, +0.0005590]. Pesos: V30
  0.5594, V43 0.0602, V47 0.0209, V42 0.1687, V40 0.0608, V39-ord 0.0438, V39-hgb 0.0520,
  V51 0.0104, V53 0.0238.
- **V57** auditoría matemática sobre el OOF de V56 (Figuras 11-12):
  (a) **símplex 9D**: perturbación ±20 % por peso; todos los modelos estacionarios (|ΔF1| <=
  ~4e-4); (b) **concavidad Power-Mean**: el óptimo global es alpha = 0 (el PoE), con caída suave
  hacia ambos lados (-1.4e-4 en -0.15; -2.9e-5 en +0.15); (c) **deciles de confiabilidad**:
  monótonos y bien calibrados (decil 0: confianza 0.278 / exactitud 0.268; decil 9: 0.992 /
  0.995); (d) **brecha de menciones**: 63.9 % de los tweets tienen mención y se clasifican mucho
  mejor (exactitud 0.6802, confianza 0.6863) que los que no (36.1 %, 0.4997 / 0.5004).
  La auditoría blinda el 30 % privado frente al shakeup (D-110).
- **Selección dual** V48 (campeón LB) / V56 (récord CV) ratificada.

### IX. Validación estadística y reproducibilidad (~1 pág) — Tabla VI
- Bootstrap pareado (1000 remuestreos) de las deltas vs V48: V53 p = 0.0280; V54 p = 0.0050;
  V56 p = 0.0010; deltas por fold de V56 vs V54: +0.000078, +0.000005, +0.000115, +0.000249,
  +0.000079 (5/5 positivos).
- Reproducibilidad: notebook autocontenido (178 celdas, 4 h 35 min en CPU 8 hilos + GPU);
  checklist 13/13; idempotencia exacta entre corridas (12/12 valores idénticos, incluida la SHA256
  de la submission reconstruida); tolerancias declaradas (Tabla VII): métricas crudas 1e-5 a
  8e-4; plug-in ~1e-3; deriva entre proveedores de GPU < 0.2 % de decisiones (hasta ~1.2 % en la
  red densa); columnas zlib estables (media 5.4e-5).
- 220 experimentos registrados, 110 decisiones, 0 errores de auditoría de idioma.

### X. Discusión y limitaciones (~0.75 pág)
- El techo es del dato, no del modelo: redundancia alta (58 clústeres de colinealidad cubren 171
  de 310 variables), vocabulario solapado entre cohortes, sin metadatos.
- La señal restante está en las fronteras y ya la capturan los binarios existentes.
- Riesgo de shakeup y su mitigación (auditoría V57).
- Limitaciones: corpus transliterado a ASCII (se pierde señal de emojis/acentos), sin datos
  externos por reglas, provenance desconocida.

### XI. Conclusiones (~0.5 pág)
- Respuesta a la pregunta: el PLN clásico llega al primer puesto (0.63724) con consensos de
  expertos y decisión calibrada; el techo del dato es 0.642 (LB).
- La validación agrupada por near-duplicados como práctica indispensable.
- Selección dual V48/V56 y reproducibilidad total como estándar de entrega.

## 3. Cifras verificadas (referencia rápida)

| Elemento | CV local | LB Kaggle | Nota |
|---|---|---|---|
| E-001 baseline | 0.5344 ± 0.0050 | 0.55235 | TF-IDF word+char + LinearSVC |
| V12 | 0.6005 | 0.62182 | NB-SVM + LogReg/MLP |
| V19 | 0.6079388 | 0.62578 | +LightGBM |
| V20 | 0.6097782 | 0.62689 | cuartos + umbrales |
| V30 | 0.6134471 | 0.63302 | Optuna |
| V30 + plug-in | 0.6159732 | 0.63575 | |
| V38 | 0.6166579 | 0.63590 | TabularNet |
| V39 | 0.6168017 | 0.63588 | trimodal |
| V40 | 0.6175912 | 0.63685 | SE-TabNet |
| V41 | 0.6181119 | 0.63722 | SWA |
| V42 | 0.6184162 | 0.63716 | Smooth-SWA |
| V43 | 0.6184783 | 0.63722 | Cramér-SWA |
| **V48 campeón** | **0.6186451** | **0.63724 (#1)** | PoE 6 modelos + plug-in |
| V53 | 0.6188259 | 0.63706 | PoE 7 (Dense-SE) |
| V54 | 0.6188887 | 0.63712 | PoE 8 |
| V55 | (0.6182436 doc.) | 0.63707 | octa limpio |
| **V56 récord CV** | **0.6189929** | 0.63717 | PoE 9 |
| Segunda partición | 0.6144120 | — | 0.6164700 con plug-in |

Plug-in por clase (F1 argmax -> F1 plug-in): 13-17 0.68537->0.68092; 18-24 0.52084->0.52727;
25-34 0.49916->0.50108; 35-49 0.57314->0.57379; 50-64 0.66706->0.66815; 65-xx 0.73511->0.74158.

## 4. Figuras (todas en `informe/figuras/`, regeneradas a tamano de imprenta)

| # | Archivo | Contenido | Seccion |
|---|---|---|---|
| 1 | `fig_01_eda.png` | clases + longitudes (caracteres train/test; palabras train) | III |
| 2 | `fig_02_terminos.png` | terminos por rango (ancho completo, 6 paneles) | III |
| 3 | `fig_03_clusters.png` | tamanos de clusteres + test con gemelo | III/IV |
| 4 | `fig_04_bloques.png` | tamano y varianza por familia de features | V |
| 5 | `fig_05_miembros.png` | aporte de los 14 miembros del meta | V |
| 6 | `fig_arquitectura_v48.png` | arquitectura del campeon (ancho completo) | VI |
| 7 | `fig_07_confusion.png` | confusion + F1 por rango del campeon (ancho completo) | VI |
| 8 | `fig_08_progresion.png` | estructura condicional + progresion de la validacion | VI |
| 9 | `fig_evolucion.png` | evolucion CV/LB por version (ancho completo) | VI/VII |
| 10 | `fig_10_familia_nn.png` | familia neuronal V38-V43 + HGB | VII |
| 11 | `fig_11_powermean.png` | concavidad Power-Mean (V57) | VIII |
| 12 | `fig_v57_analisis.png` | simplex + deciles (V57, ancho completo) | VIII |
| 13 | `fig_13_test.png` | distribucion de decisiones del test | IX |

Generadores: `informe/genera_figuras_articulo.py` (figuras 1-5, 7-8, 10-11, 13 desde
artefactos locales; fig 7 reconstruye el campeon con macro-F1 0.6186451) y
`informe/genera_figuras.py` (figuras 6, 9 y 12). Las figuras nb_* del notebook quedan
como respaldo de trazabilidad (ya no se incluyen).

## 5. Tablas (especificación; el redactor las construye)

- **Tabla I**: datos (train/test/submission, filas, columnas, separador) + propiedades (clases %,
  padding, ASCII, vacíos, etiquetas contradictorias).
- **Tabla II**: decisiones cruciales (6-7 filas): acta, decisión, impacto.
- **Tabla III**: hitos de la evolución (11 filas: V12, V19, V20, V30, V30+plug, V41, V43, V48,
  V53, V54, V56) con CV, LB y "mecanismo".
- **Tabla IV**: falsaciones clave (5-7 filas): hipótesis, resultado, lección.
- **Tabla V**: plug-in por clase (6 filas): F1 argmax, F1 plug-in, recall, n predicho.
- **Tabla VI**: significancia (V53/V54/V56 vs V48): ΔCV, folds positivos, p bootstrap, IC95.
- **Tabla VII**: tolerancias de reproducibilidad (5-6 filas).

## 6. Citas: claves de `references.bib` y uso

Corrientes A/B/C/D como en la sección II. Claves disponibles: schler2006, argamon2007,
mukherjee2010, nguyen2011, nguyen2013, nguyen2014, sap2014, flekova2013, hovy2018,
morganlopez2017, liu2021, rangel2013..rangel2019, sapkota2013, sapkota2015, lopezmonroy2013,
monroe2008, parambath2014, waegeman2014, narasimhan2015, lipton2014, eban2017, verhoeven2014,
verhoeven2016, izmailov2018, hu2018, huang2017, gorishniy2021, loshchilov2017, muller2019,
szegedy2016, klambauer2017, cramer1928, vonmises1931, szekely2013, frogner2015, diaz2019,
frank2001, cao2020, hinton2002, heskes1998, ke2017, prokhorenkova2018, wang2012, kaufman2012,
simpson1951. NO citar rosenthal2011 (no verificada). Citar Optuna como "Akiba et al., 2019"
solo si se menciona; no hay ficha en el bib: mencionar sin cita o citar el uso genérico.
Regla: cada técnica mencionada debe llevar su cita; ~35-45 citas en total.

## 7. Reglas de estilo y LaTeX

- Preámbulo base (mismo de los talleres del curso):
  `\documentclass[journal]{IEEEtai}` + hyperref, utf8, spanish babel, graphicx, float, caption,
  amsmath/amsthm/amssymb, tabularx, booktabs, multirow, multicol, subcaption.
  `\renewcommand{\tablename}{Tabla}`. `\markboth{Journal of IEEE Transactions on Artificial
  Intelligence, Vol. 00, No. 0, Octubre 2026}{E. Rivera et al.: ...}`.
- Autores: `\author{E. V. Rivera Jiménez, J. G. Carrillo Meneses, C. A. Garnica Salazar, y O. A.
  Prieto Moreno}` + `\thanks` con afiliación y correos (uno por autor o agrupados).
- Español correcto con tildes; cifras con punto decimal; macro-F1 con guion; "leaderboard" en
  cursiva o comillas la primera vez.
- No usar negritas excesivas; tablas con booktabs; figuras con `\includegraphics[width=...]` y
  `\caption` descriptivo (sin "Figura X:" dentro del caption).
- Código: no incluir listados de código largos (máximo 1-2 fragmentos de 3-5 líneas si aportan).
- Cierre: `\bibliographystyle{ieeetr}` + `\bibliography{references}`.

## 8. Fuentes para detalle (lectura dirigida, solo si se necesita)

- `CHAMPION.md` — tabla histórica de campeones y récords (cifras CV/LB por versión).
- `RUNLOG.md` — entradas V53-V57 con metodología y resultados finos.
- `DECISIONS.md` — D-001, D-002, D-003, D-004, D-005, D-009, D-011, D-087, D-090, D-099, D-101,
  D-104, D-106..D-110 (textos completos de las decisiones citadas).
- `INFORME_COMPLETO.md` — cuenta completa del proyecto (fases V13-V17, lecciones).
- `plan/kaggle_notebook/secciones/fase4_5_v44_v52.md` — falsaciones V44-V52 (detalle por
  experimento).
- `plan/kaggle_notebook/secciones/fase4_6_v53_v57.md` — arco 2026 documentado.
- `plan/kaggle_notebook/secciones/05_fase5.md` — el campeón y la submission (detalle).
- `reports/state_of_art.csv` — fichas SOTA por paper (una fila por artículo).
- `experiments/v57_analisis_matematico_estadistico.json` — números exactos de la auditoría V57.
