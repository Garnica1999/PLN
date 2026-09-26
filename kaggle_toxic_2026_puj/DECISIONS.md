# DECISIONS - KAGGLE_TOXIC_2026_PUJ

Registro de decisiones de arquitectura, modelado y metodologia.
Formato por decision: contexto, observacion, hipotesis, decision, razon,
evidencia, experimento, resultado, conclusion final.

---

## D-001 - Formulacion multi-etiqueta jerarquica (no multiclase)

- **Contexto**: la competencia pide predecir HS, TR y AG por texto.
- **Observacion**: en `toxic_train.csv`, TR=1 solo ocurre cuando HS=1
  (519/519, 0 violaciones); AG=1 con HS=0 solo en 26 casos (1.5 %).
- **Hipotesis**: TR y AG son atributos condicionados a que exista discurso de
  odio, no categorias excluyentes.
- **Decision**: modelar 3 clasificadores binarios acoplados (multi-etiqueta) y
  evaluar un post-proceso jerarquico `HS=0 => TR=0, AG=0` medido en validacion.
- **Razon**: TR y AG no son mutuamente excluyentes con HS; tratarlos como
  multiclase perderia informacion y violaria la estructura de anotacion.
- **Evidencia**: crosstab HS x TR x AG (Fase 0, `reports/eda/combinaciones_etiquetas.csv`).
- **Experimento**: E00b (post-proceso jerarquico) en Fase 9/10.
- **Resultado**: pendiente.
- **Conclusion final**: pendiente.

---

## D-002 - Prohibicion absoluta de etiquetas/recursos externos de HatEval

- **Contexto**: el corpus sigue el esquema de HatEval (SemEval-2019 Task 5,
  subconjunto espanol); existe una copia publica de textos y etiquetas originales.
- **Observacion**: 17 de los textos de test tienen near-duplicados en train
  (similitud 1.0 con URLs distintas), lo que demuestra que el corpus combina
  fuentes repetidas; los originales circulan en GitHub/HuggingFace.
- **Hipotesis**: usar etiquetas externas o buscar los textos originales
  inflaria artificialmente el F1 y violaria las reglas academicas.
- **Decision**: prohibido consultar, descargar, cruzar o usar cualquier etiqueta
  o corpus externo; solo `data/raw/` oficial. Los lexicones publicos se admiten
  unicamente como features opcionales documentadas (D-007).
- **Razon**: integridad academica y validez del resultado.
- **Evidencia**: `reports/leakage_audit.md` seccion 4.
- **Experimento**: no aplica (regla).
- **Resultado**: no aplica.
- **Conclusion final**: regla permanente del proyecto.

---

## D-003 - spaCy solo como anotador linguistico (sin vectores preentrenados)

- **Contexto**: se usaran features POS/sintacticas con `es_core_news_md`.
- **Observacion**: ese modelo incluye vectores preentrenados (300d), prohibidos
  por las reglas de la competencia si se usan como representacion.
- **Hipotesis**: las etiquetas POS/lemas/dependencias (anotacion linguistica) no
  son "embeddings preentrenados" y pueden usarse; los vectores no.
- **Decision**: usar `spaCy` con `exclude=["ner"]` y prohibir `doc.vector`,
  `token.vector`, `doc.tensor` y `similarity()` en cualquier feature; assert en
  el codigo de features.
- **Razon**: maxima senal linguistica sin violar la restriccion de embeddings.
- **Evidencia**: reglas del profesor (texto verbatim en `docs/competition_rules.md`).
- **Experimento**: E05 (aporte de POS).
- **Resultado**: pendiente.
- **Conclusion final**: pendiente.

---

## D-004 - Validacion: StratifiedGroupKFold con grupos de near-duplicados

- **Contexto**: 4 grupos de near-duplicados cruzan train y test; 3 pares
  train-train con similitud 1.0.
- **Observacion**: un `StratifiedKFold` aleatorio repartiria copias casi
  identicas entre entrenamiento y validacion, sobreestimando el F1 local.
- **Hipotesis**: agrupar por near-duplicado da una estimacion mas honesta.
- **Decision**: validacion primaria `StratifiedGroupKFold(k=5)` con grupos =
  componentes conexas de similitud >= 0.85 (`data/processed/grupos_near_duplicados.csv`)
  y estratificacion por combinacion HS x TR x AG; control de robustez adicional
  con pseudo-autor (primera mencion) como grupo.
- **Razon**: reducir fuga por duplicados sin dejar de estratificar por clase.
- **Evidencia**: `reports/leakage_audit.md` secciones 1 y 1b.
- **Experimento**: protocolo aplicado desde E01.
- **Resultado**: pendiente.
- **Conclusion final**: pendiente.

---

## D-005 - TR: discrepancia oficial vs HatEval, caracterizacion empirica

- **Contexto**: la definicion oficial dice "dirigido contra una persona o grupo";
  HatEval define TR como rango del target (0=grupo, 1=individuo).
- **Observacion**: los ejemplos oficiales son compatibles con ambas lecturas;
  los datos muestran TR=1 solo dentro de HS=1.
- **Hipotesis**: la etiqueta observada puede corresponder al esquema HatEval
  re-etiquetado; hay que medirlo, no asumirlo.
- **Decision**: caracterizar TR con analisis cualitativo (100 ejemplos) y
  cuantitativo (lexico/sintaxis/2a persona) en Fase 2; modelar la etiqueta tal
  como aparece en los datos.
- **Razon**: instruccion explicita del proyecto ("no asumir el significado de TR").
- **Evidencia**: `reports/samples_por_clase.csv`; `docs/competition_rules.md`.
- **Experimento**: E03 (features de targeting).
- **Resultado**: pendiente.
- **Conclusion final**: pendiente.

---

## D-006 - Parser y encoding oficiales

- **Contexto**: los CSV tienen separadores distintos y un archivo no es UTF-8.
- **Observacion**: train UTF-8 con sep `;` (1.732 lineas, todas 5 campos con
  quoting estandar; QUOTE_NONE rompe 13 filas); test cp1252 con sep `;`
  (6 bytes 0xBF); submission con `,` y 4 columnas.
- **Hipotesis**: el quoting estandar preserva el 100 % de los campos.
- **Decision**: `src/data.py` lee con quoting estandar, decode UTF-8 con caida a
  cp1252, y valida filas/columnas/ids con asserts.
- **Razon**: evitar perdida silenciosa de textos o de `;` internos.
- **Evidencia**: diagnostico en RUNLOG 12:50; asserts verdes en Fase 0.
- **Experimento**: 00_auditoria_datos.py (integridad).
- **Resultado**: 0 filas perdidas; 21 textos con `;` preservados.
- **Conclusion final**: parser congelado; todo script futuro carga con `src.data`.

---

## D-007 - Sin datos externos de entrenamiento; lexicones documentados y opcionales

- **Contexto**: se quieren features de profanity/sentimiento/emocion.
- **Observacion**: la competencia no prohibe lexicones, pero no autoriza corpus
  externos de etiquetas; el profesor indica "hacer todo desde cero".
- **Hipotesis**: lexicones publicos (HurtLex, ML-SentiCon, etc.) aportan senal
  complementaria; su uso debe ser trazable y reversible.
- **Decision**: todo lexicon se registra en `reports/lexicon_catalog.csv` con
  nombre, fuente, idioma, licencia, finalidad y riesgo; se evalua su aporte con
  ablacion (E02b/E04) y se mantiene SIEMPRE una variante 100 % autocontenida
  (lexicos construidos solo con train + WordNet/OMW) como respaldo.
- **Razon**: cumplir reglas y poder defender cada feature ante el jurado.
- **Evidencia**: instruccion del proyecto; reglas verbatim.
- **Experimento**: E02b (profanity), E04 (sentimiento/emocion).
- **Resultado**: pendiente.
- **Conclusion final**: pendiente.

---

## D-008 - Protocolo de acceso a GPU (tarjeta compartida con el escritorio)

- **Contexto**: la RX 9070 XT es unica GPU y controla el escritorio; puede estar
  ocupada por otros trabajos del usuario.
- **Observacion**: baseline medido: VRAM en reposo 2.029.953.024 B, 0 KFD PIDs,
  uso instantaneo 15 % (compositor). El % instantaneo es ruidoso.
- **Hipotesis**: un chequeo multi-muestra + KFD PIDs detecta ocupacion real.
- **Decision**: antes de cada job GPU, `rocm-smi --showuse --showmeminfo vram
  --showpids` x5 muestras; ocupada si (a) hay PID ajeno en KFD, (b) VRAM >
  baseline + 3 GB sostenido, (c) uso >= 50 % en >=3/5 muestras con VRAM elevada.
  Si esta ocupada: no matar procesos, no forzar, registrar evidencia en RUNLOG y
  `experiments/gpu_preflight.log`, avisar al usuario y **detener todo,
  devolviendo el control** hasta nueva orden. Si esta libre: registrar, lanzar y
  verificar al terminar que no quedan procesos propios.
- **Razon**: no interferir con el trabajo del usuario en su maquina.
- **Evidencia**: `config/gpu_baseline.json` (medido 2026-09-22).
- **Experimento**: aplica desde Fase 11 (DL desde cero).
- **Resultado**: pendiente.
- **Conclusion final**: pendiente.

---

## D-009 - Metrica oficial: hipotesis primaria y sonda E00

- **Contexto**: Kaggle declara "F1 Score" para una submission de 3 columnas
  binarias; la implementacion (macro column-wise, micro, samples) no se
  especifica.
- **Observacion**: con la distribucion de train, la plantilla todo-1,1,1 daria
  ~0.541 (media column-wise), ~0.543 (micro), ~0.270 (macro sobre 6 pares) o
  ~0.284 (samples/exact match).
- **Hipotesis**: la implementacion es la media column-wise de F1 binaria.
- **Decision**: adoptar la media column-wise como metrica primaria de seleccion
  local; reportar siempre macro-6, micro y exact match como secundarias; enviar
  la sonda E00 (plantilla oficial) para confirmar y dejar constancia en
  `submissions/leaderboard_log.csv`.
- **Razon**: decidir con evidencia y no con una suposicion.
- **Evidencia**: valores esperados calculados; tag de metrica de Kaggle.
- **Experimento**: E00 (sonda; ejecutada por Carlos el 2026-09-22).
- **Resultado**: score publico 0.51494 con la plantilla todo-1. Los valores
  esperados eran 0.541 (media column-wise), 0.543 (micro), 0.270 (macro-6) y
  0.284 (samples). La sonda descarta macro-6 y samples. La confirmacion llego
  con v01: local 0.75778 (media column-wise) vs LB 0.75677 (dif -0.0010),
  mientras el macro-6 local era 0.8097 (descartado). La desviacion de 0.51494
  frente a 0.541 indica que el test tiene ~7 % menos positivos que train.
- **Conclusion final**: metrica oficial = F1 macro sobre las 3 columnas
  (equivalente a la media column-wise de la F1 binaria de la clase positiva).
  Es la metrica primaria de seleccion local; macro-6, micro y samples quedan
  como diagnostico. El protocolo de validacion agrupada reproduce el LB con
  error ~0.001, por lo que las decisiones se pueden tomar en local con
  confianza.

---

## D-010 - El corpus oficial no tiene diacriticos

- **Contexto**: los textos llegan sin tildes ni enye ("organizacin", "Espaa").
- **Observacion**: train es 100 % ASCII; test tiene 6 bytes `0xBF` ("?") en cp1252.
- **Hipotesis**: los organizadores normalizaron a ASCII; no hay informacion de
  acentos que recuperar.
- **Decision**: no "reconstruir" acentos con diccionarios; tratar la ausencia de
  diacriticos como caracteristica del corpus y documentarlo como limitacion.
- **Razon**: evitar introducir informacion externa no presente en los datos.
- **Evidencia**: `reports/dataset_quality.csv`.
- **Experimento**: variantes A-L de preprocesamiento (Fase 4) compararan
  normalizaciones explicitas.
- **Resultado**: pendiente.
- **Conclusion final**: pendiente.

---

## D-012 - Ajuste de umbral por etiqueta: adoptado como apuesta, con control

- **Contexto**: con la mejor representacion (MOD-03b_BEST) el modelo subpredice
  positivos; se ajustaron umbrales por etiqueta sobre las probabilidades OOF
  (HS 0.44, TR 0.51, AG 0.46).
- **Observacion**: la ganancia medida en el mismo OOF con el que se eligieron
  los umbrales es +0.0063 (regla pre-registrada cumplida: >= +0.005 y 4/5
  pliegues). La verificacion anidada independiente del revisor (umbral ajustado
  en los pliegues no evaluados) da +0.0014 de media, con solo 2/5 pliegues
  positivos: la ganancia esta sobreestimada ~4x.
- **Hipotesis**: los umbrales pueden dar una mejora real pequena, o ninguna.
- **Decision**: generar dos submissions y decidir con el leaderboard, que actua
  como fuente secundaria de evidencia: `v02` con umbral 0.5 (control) y `v03`
  con umbrales optimos (apuesta). Prohibido re-optimizar umbrales contra el LB.
- **Razon**: honestidad metodologica; la validacion local y el razonamiento
  experimental siguen siendo la base (regla del proyecto).
- **Evidencia**: `reports/fase13a_refit_umbral_metricas.json`
  (`advertencia_sesgo_optimista`), revision de `ml-reviewer` del 2026-09-22.
- **Experimento**: TH-01 / S02 / S03 en `experiments.csv`.
- **Resultado**: LB v02 (umbral 0.5) = **0.77993** (local 0.7927, gap -0.0128);
  LB v03 (umbrales optimos) = **0.78306** (local 0.7989, gap -0.0158). Los
  umbrales dieron una mejora real de **+0.0031** en el LB, dentro del rango
  entre la estimacion anidada (+0.0014) y la optimista (+0.0063).
- **Conclusion final**: se MANTIENEN los umbrales por etiqueta (v03 es la mejor
  submission hasta ahora). Se documenta ademas un hallazgo de calibracion: el
  gap local-LB de la familia union word+char (~-0.013/-0.016) es mucho mayor
  que el de B1 (-0.001), coherente con que el test tenga ~7 % menos positivos
  (E00) y con la mayor capacidad de memorizacion de 27k features de caracter.
  Regla practica adoptada: dentro de la familia union, exigir mejoras locales
  >= +0.010 antes de dar una mejora por realista, y confirmar los cambios
  grandes con una sonda de LB.

---

## D-013 - La capacidad del modelo modula el gap local-LB (seleccion por OOF)

- **Contexto**: se probaron 5 configuraciones en el leaderboard con el mismo
  protocolo local (OOF con 5-fold agrupado por near-duplicados).
- **Observacion**: el gap local-LB crece con la capacidad del modelo:
  B1 (4.024 features, C=1) -0.001; MOD-03b (27.431, C=2) -0.0128/-0.0158;
  ENS-04 -0.0234; OPT-01 (32.060, C=3.26) -0.0279. El modelo con el MEJOR OOF
  (OPT-01, 0.8039) fue el que peor transfirio (LB 0.77602), mientras MOD-03b
  (OOF 0.7989) mantiene el mejor LB (0.78306).
- **Hipotesis**: el OOF con prevalencia de train (~43.7 % HS) premia modelos que
  explotan patrones especificos de la muestra de train; el test tiene ~7 % menos
  positivos (E00), asi que esa capacidad extra no transfiere. Reducir capacidad
  (C, min_df) deberia mejorar la transferencia sin perder nivel absoluto.
- **Decision**: pre-registrar DOS sondas de robustez (maximo 2 envios) con la
  representacion de MOD-03b y umbrales OOF: `v06` (C=1.0, min_df=2) y `v07`
  (C=1.0, min_df=5). Se adopta como modelo final solo la que supere el LB de
  v03 (0.78306); en caso contrario se mantiene MOD-03b con umbrales. Prohibido
  seguir sondeando mas alla de estas dos sin una hipotesis nueva.
- **Razon**: evitar sobreajustar el leaderboard y, a la vez, aprender si el
  control de capacidad es la palanca correcta.
- **Evidencia**: `submissions/leaderboard_log.csv` (columna diferencia),
  D-012 y este registro.
- **Experimento**: `experiments/10_sondas_robustez.py` (filas S06/S07).
- **Resultado**: v06 (C=1.0, min_df=2) LB 0.77227 y v07 (C=1.0, min_df=5) LB
  0.77034, ambos por debajo de v03 (0.78306). **Hipotesis REFUTADA**: reducir la
  capacidad no redujo el gap local-LB (v07 gap -0.0290 con la mitad del gap
  train-OOF), es decir, el gap train-OOF no es proxy del gap local-LB.
- **Conclusion final**: refutada. El control de capacidad (C, min_df) no mejora
  la transferencia al test en esta familia.

---

## D-014 - Cierre de la exploracion del leaderboard: plateau y modelo final

- **Contexto**: se enviaron 7 submissions utiles (v01..v07) con distintos
  modelos, representaciones, umbrales y ensembles.
- **Observacion**: todos los modelos de la familia union TF-IDF + LR caen en la
  banda **0.77034-0.78306** del LB, un rango de 0.0127 que equivale a **0.99
  errores estandar** (SE binomial aproximado 0.0129 con n=1154). Es decir, las
  diferencias entre esas submissions son indistinguibles del ruido de muestreo.
  El unico contraste pareado fiable es v02 vs v03 (mismo modelo, distintos
  umbrales): +0.0031 a favor de los umbrales OOF.
- **Hipotesis**: la familia ha alcanzado su plateau con los datos y las
  restricciones disponibles (sin transformers, sin embeddings preentrenados,
  1.731 ejemplos de entrenamiento).
- **Decision**: adoptar **MOD-03b_BEST con umbrales por etiqueta 0.44/0.51/0.46**
  (submission v03, LB 0.78306) como modelo final; **detener el sondeo del
  leaderboard**; redirigir el esfuerzo a los entregables academicos (notebook
  30 % de la nota, informe, auditorias) y a completar la Fase 11 (DL desde cero)
  como requisito metodologico, no como palanca de rendimiento.
- **Razon**: seguir enviando variantes dentro de la banda de ruido no aporta
  evidencia y arriesga sobreajustar el leaderboard (regla del proyecto).
- **Evidencia**: `submissions/leaderboard_log.csv` (7 filas con gap),
  `reports/fase13d_sondas_robustez.json`, D-012 y D-013.
- **Experimento**: no aplica (decision de cierre).
- **Resultado**: modelo final = MOD-03b + umbrales; LB 0.78306.
- **Conclusion final**: cerrada la fase competitiva; se documenta el plateau y
  la cota local optimista (-0.013 a -0.029) como limitacion del proyecto.

## D-015 - El control por pseudo-autor calibra la validacion con el LB

- **Contexto**: la auditoria final (QA) detecto que el control por pseudo-autor
  previsto en D-004 nunca se ejecuto; 17 pseudo-autores (77 tweets, 4.4 %)
  cruzaban pliegues con el split solo por near-duplicados.
- **Observacion**: al fusionar los grupos (near-duplicado + pseudo-autor) y
  reevaluar el modelo final con el mismo protocolo, la F1 OOF baja de 0.7927 a
  **0.7838** (diferencia -0.0088). Comparado con el LB real de esa misma
  configuracion (v02 = 0.77993), el gap local-LB pasa de **-0.0128 a -0.0039**.
- **Hipotesis**: el split solo por near-duplicados era optimista porque un mismo
  usuario tiende a producir tweets con etiquetas correlacionadas; agrupar por
  pseudo-autor elimina esa dependencia.
- **Decision**: adoptar el agrupamiento fusionado (near-duplicado + pseudo-autor)
  como protocolo de validacion recomendado para trabajo futuro y reportar ambas
  estimaciones en el informe; NO se reabre la seleccion de modelo (D-014), pero
  el hallazgo se documenta como la explicacion principal del gap local-LB.
- **Razon**: mejora la calibracion de la validacion sin cambiar el modelo final
  ni gastar envios.
- **Evidencia**: `reports/control_pseudo_autores.json`, fila `PA-01` en
  `experiments.csv`, `reports/auditoria_final.md` (brecha 1).
- **Experimento**: `experiments/12_control_pseudo_autores.py`.
- **Resultado**: OOF 0.7838 con grupos fusionados (5 pliegues, 0 grupos
  compartidos); std entre pliegues 0.0187.
- **Conclusion final**: el protocolo de validacion recomendado pasa a ser
  `StratifiedGroupKFold` con grupos fusionados; el gap local-LB del proyecto se
  explica en su mayor parte por el pseudo-autor, no por la representacion.

---

## D-011 - Entregable y estilo del codigo

- **Contexto**: 30 % de la nota depende del notebook; el flujo exige scripts por
  experimento y prohibe emojis en el codigo.
- **Observacion**: el profesor espera "un notebook claro y bien documentado".
- **Hipotesis**: un notebook final consolidado en espanol, con narrativa
  OBSERVACION -> HIPOTESIS -> METODO -> RESULTADO -> INTERPRETACION -> DECISION,
  maximiza la nota sin sacrificar rigor.
- **Decision**: desarrollar en scripts/notebooks por fase (`notebooks/01-10`) y
  cerrar con `notebooks/10_final_pipeline.ipynb` como entregable; codigo con
  docstrings completos, sin emojis, identificadores en ingles y prosa en espanol.
- **Razon**: cumplir ambos criterios (competencia y rubrica).
- **Evidencia**: reglas verbatim; skill de calidad de codigo.
- **Experimento**: no aplica.
- **Resultado**: no aplica.
- **Conclusion final**: regla permanente del proyecto.

---

## D-016 - Protocolo de validacion v2: desarrollo/holdout, OOF corregido y regla de adopcion

- **Contexto**: la fase de investigacion v2 (plan aprobado el 2026-09-25) busca
  una fuente de senal nueva para superar el LB 0.78306. Con 24 lineas de
  investigacion sobre 1.731 ejemplos, la multiplicidad de comparaciones es el
  riesgo principal: apareceran "mejoras" de +0.003 por azar.
- **Observacion**: (a) el protocolo historico (solo near-duplicado) era optimista
  (D-015); (b) el holdout debe congelarse antes de mirar resultados; (c) los OOF
  existentes (`oof_bases.npz`, `oof_mod03b.npz`) estaban calculados con el
  protocolo historico y no sirven para cadenas ni stacking sin heredar el sesgo.
- **Hipotesis**: con grupos fusionados (near-duplicado UNION pseudo-autor), un
  holdout congelado y bootstrap por tweet, las mejoras que sobrevivan seran
  reales.
- **Decision**:
  1. **Grupos** = near-duplicado UNION pseudo-autor; `src/splits.py` centraliza el
     protocolo.
  2. **Holdout congelado** por `StratifiedGroupKFold(n_splits=7, seed=42)`:
     pliegue 0 = holdout (248 tweets, los 6 combos representados); el resto =
     desarrollo (1.483). Maximo **5 evaluaciones** al holdout en todo el
     proyecto, solo finalistas y registradas en RUNLOG.
  3. **Metrica primaria** = `corrected_cv` sobre el desarrollo; `historical_cv`
     se reporta para continuidad. Referencia en el desarrollo: V03 con umbrales
     = **0.7772** (M2 a 0.5 = 0.7718).
  4. **Todo OOF**; prohibido ajustar cualquier cosa con validacion o test.
  5. **Regla de adopcion**: corrected_cv >= +0.005 y mejora en >= 3/5 pliegues, o
     complementariedad demostrada (`ensemble_gain` >= 0.0015 con IC bootstrap >
     0). Sondas de LB: maximo 3, solo finalistas.
  6. **Contingencia** (pre-registrada): si el std entre pliegues del desarrollo
     supera 0.025, pasar a CV repetida (3 semillas x 5 pliegues); si aun no
     discrimina, ampliar el holdout al 20 %.
- **Razon**: controlar la multiplicidad y alinear la validacion con el LB
  (D-015) sin gastar el presupuesto de envios.
- **Evidencia**: `config/holdout_v2.json` (con hash), `data/processed/holdout_v2_ids.csv`,
  `features/v2_oof_base_corrected.npz`, `reports/baseline_v03_dev_metrics.json`,
  `reports/advanced_gap_analysis_v2.md`, `reports/research_backlog_v2.md`.
- **Experimento**: `experiments/13_holdout_v2.py` (HOLD-01) y
  `experiments/14_oof_corregido_v2.py` (V2-BASE-01..04).
- **Resultado**: desarrollo 1.483 / holdout 248; corrected_cv de los bases:
  M2 (V03) 0.7718 (umbrales 0.7772), M3 0.7677, M1 0.7486, M4 0.7377;
  std entre pliegues 0.015-0.023 (por debajo del umbral de contingencia).
  Nota: en el desarrollo la diferencia corregido vs historico (0.7772 vs 0.7736)
  esta dentro del ruido; la evidencia de D-015 proviene del train completo.
- **Conclusion final**: protocolo v2 congelado; ninguna linea de investigacion se
  evalua fuera de el.

---

## D-017 - P1: cadenas cerradas sin mejora; meta-modelo de dependencia adoptado

- **Contexto**: P1 del plan v2 (lineas E2-01 cadenas de etiquetas y E2-02
  meta-modelo de dependencia) sobre el desarrollo (n=1.483, protocolo corregido
  D-016). Referencia: V03 con umbrales = 0.7772.
- **Observacion**: anadir el score continuo de la etiqueta previa como feature
  (con anidamiento interno de 4 pliegues para evitar fuga) da 0.7737-0.7744
  frente a 0.7718 del base sin cadena: mejora +0.002, dentro del ruido (IC
  bootstrap incluye 0; solo 1/5 pliegues mejora). El meta-modelo sobre las
  probabilidades OOF de los 4 bases + margenes + entropias + 44 features
  estructurales alcanza **0.7853** con verificacion anidada estricta (0.7850 sin
  ella), es decir **+0.0081** sobre la referencia, con 4/5 pliegues mejorados y
  IC bootstrap [+0.0021, +0.0248] (> 0). Cota optimista con umbrales OOF
  (0.48/0.56/0.44): 0.7888.
- **Hipotesis**: la dependencia entre etiquetas (phi 0.74-0.83) se captura mejor
  en el nivel meta (scores + incertidumbre + estructura) que inyectando un score
  en la representacion del texto.
- **Decision**: cerrar la linea E2-01 (cadenas; sin evidencia de mejora) y
  adoptar E2-02 (META) como **candidato v2-A** y canal base del stacking v2 (P6).
  El holdout congelado sigue intacto (0/5 evaluaciones) y se reserva para P7.
- **Razon**: cumple la regla de adopcion de D-016 (nivel >= +0.005, >= 3/5
  pliegues e IC bootstrap > 0) y ademas la via de complementariedad
  (`ensemble_gain` +0.0152).
- **Evidencia**: `reports/cadenas_etiquetas_v2.csv` y `.json`,
  `reports/figuras/fig_27_cadenas_v2.png`, filas `E2-01-*` y `E2-02-META` en
  `experiments.csv`.
- **Experimento**: `experiments/15_cadenas_etiquetas_v2.py`.
- **Resultado**: META 0.7853 (n=1.483, std entre pliegues 0.014); E2-01 cerrada
  sin mejora.
- **Conclusion final**: pendiente de (a) integrar los canales de P2-P5 y
  (b) validar el candidato final en el holdout congelado antes de gastar sonda
  de LB.

---

## D-018 - P2: targeting estructural sin mejora standalone; canales preservados para el stacking y reordenamiento del plan

- **Contexto**: P2 del plan v2 (E2-05 distancia/estructura target-insulto y canal
  abstracto de patrones; E2-06 colectivo + agresion) sobre el desarrollo.
- **Observacion**: todas las variantes quedan por debajo de la referencia
  (0.7772) y del candidato v2-A (0.7853): canal abstracto 0.6076,
  canal + distancias 0.6484, union + canal 0.7356, union + n-gramas simbolicos
  0.7487 (base M2 = 0.7718). La concatenacion reproduce la degradacion historica
  (densa y ahora simbolica). Sin embargo: (a) el recall de AG en la rebanada de
  referencia colectiva sube de 0.452 (V03) a **0.478** con el canal
  CH-TGT+DIST; (b) los OOF del canal quedan disponibles
  (`features/v2_oof_targeting.npz`).
- **Hipotesis**: la senal de targeting es real pero **no linealmente separable**
  al concatenar; su valor solo puede aparecer en una fusion tardia regularizada
  que pondere canales segun su fiabilidad.
- **Decision**: cerrar E2-05/E2-06 como variantes standalone (sin adopcion) y
  **preservar los canales como entradas del stacking de P6**, con la condicion
  de que el meta demuestre mejora con IC bootstrap > 0. Ademas, **reordenar el
  plan** (seccion 101 del enunciado): P6 (late fusion / stacking) pasa antes que
  P3-P5, porque la evidencia de P1+P2 dice que la fusion a nivel meta es la
  unica via que ha funcionado y que la concatenacion degrada en ambas formas
  (densa y simbolica).
- **Razon**: seguir la evidencia en vez del orden inicial; las lineas P3-P5 se
  justifican solo si aportan canales evaluables por el meta.
- **Evidencia**: `reports/targeting_v2.csv` y `.json`,
  `reports/figuras/fig_28_targeting_v2.png`, filas `E2-05-*` y `E2-06-*` en
  `experiments.csv`, `features/v2_oof_targeting.npz`.
- **Experimento**: `experiments/16_targeting_v2.py`.
- **Resultado**: 0.6076-0.7487 (todas < 0.7718); unicos gains negativos
  (-0.010 a -0.033); recall de AG en la rebanada colectiva +0.026.
- **Conclusion final**: linea cerrada como standalone; los canales pasan a P6.

---

## D-019 - P6: el stacking adopta el meta por complementariedad; los canales adicionales no aportan

- **Contexto**: P6 del plan v2 (late fusion / stacking de segunda generacion) con
  todos los campos OOF disponibles: 4 bases + 2 canales de targeting + margenes,
  entropias y 10 features estructurales.
- **Observacion**: el mejor meta es **META-BASES-RIDGE (Ridge alpha=3) con
  0.7874** (3/5 pliegues) frente a 0.7853 del candidato v2-A (meta de P1) y
  0.7772 de V03. Su ventaja de **nivel** frente a v2-A es ruido (IC bootstrap
  [-0.0054, +0.0093] incluye 0), pero la **complementariedad** es positiva:
  `ensemble_gain` +0.0089 con IC [+0.0019, +0.0160]. Anadir los canales de
  targeting al meta NO aporta (delta -0.0006, IC [-0.0077, +0.0062]). Efecto
  colateral valioso: el recall de AG en la rebanada de referencia colectiva
  sube de 0.452 (V03) a 0.574 y su F1 de 0.523 a 0.589 sin degradar fuera.
- **Hipotesis**: la dependencia entre etiquetas y la combinacion de
  incertidumbres aportan informacion complementaria real, mientras que las
  features de canal (targeting) no aportan nada al meta.
- **Decision**: adoptar el **meta v2 (Ridge alpha=3 sobre los 44 features de P1)**
  como candidato `v2_meta_candidato` y cerrar la linea de canales adicionales de
  targeting en el meta. Consecuencia de plan: **P3-P5 quedan en espera con
  expectativa baja** (producen canales, y los canales no aportan al meta); se
  prioriza P7 (holdout + sonda LB) y solo se retomaran P3-P5 si el holdout
  muestra que falta senal.
- **Razon**: la evidencia de P1+P6 dice que el valor esta en el nivel meta, no en
  canales nuevos; insistir en canales seria repetir la linea que ya fallo dos
  veces (concatenacion densa y simbolica).
- **Evidencia**: `reports/stacking_v2.csv` y `.json`,
  `reports/figuras/fig_29_stacking_v2.png`, `models/v2_meta_candidato.joblib`,
  `features/v2_meta_oof.npz`, filas `E2-18-*` en `experiments.csv`.
- **Experimento**: `experiments/17_stacking_v2.py`.
- **Resultado**: META-BASES-RIDGE 0.7874 (+0.0020 vs v2-A, +0.0102 vs V03);
  canales no aportan; AG en colectiva +0.12 de recall.
- **Conclusion final**: candidato para P7: evaluar en el holdout congelado
  (1-2 evaluaciones de las 5) y, si confirma, reajustar sobre todo el train y
  gastar una sonda de LB.

---

## D-020 - P7: el holdout confirma el candidato (delta pareado +0.018) y se autoriza una sonda de LB

- **Contexto**: P7 del plan v2: validacion del candidato (meta Ridge alpha=3)
  frente a V03 en el **holdout congelado** (248 tweets, 2 de las 5 evaluaciones
  permitidas) y generacion de la submission.
- **Observacion**: en el holdout, V03 = 0.8285 (HS 0.857 / TR 0.849 / AG 0.779)
  y el meta = **0.8465** (HS 0.862 / TR 0.857 / AG 0.820), delta pareado
  **+0.0180** con IC95 [+0.0003, +0.0372] (2.3 % de los remuestreos <= 0).
  El holdout resulto un pliegue mas facil que cualquier pliegue del desarrollo
  (0.83/0.85 vs 0.774/0.787), y la auditoria (`ml-reviewer`) reprodujo los
  numeros exactos, verifico 0 grupos/textos compartidos con el desarrollo y
  confirmo que **no hay fuga en el stacking** y que la submission v08 coincide
  con el pipeline declarado.
- **Hipotesis**: la ganancia del meta sobre V03 es real pero de magnitud incierta:
  aplicando el gap de la familia union (D-012: -0.013 a -0.016) a la referencia
  del desarrollo (0.7874), el LB esperado del candidato cae en **~0.771-0.774**,
  es decir podria quedar **por debajo** de v03 (0.78306).
- **Decision**: autorizar **una** sonda de LB con
  `submissions/toxic_submission_v08_meta_v2.csv` (Carlos la envia), con la
  advertencia honesta del rango esperado; no se reajusta nada mas hasta conocer
  el resultado. Si el LB no supera 0.78306, se mantiene v03 como modelo final y
  la linea v2 se documenta como "senal complementaria no transferida".
- **Razon**: la sonda es informativa en ambos sentidos (1 de 10 envios diarios) y
  cierra la pregunta sobre la transferencia del meta.
- **Evidencia**: `reports/p7_holdout_resultados.json`,
  `reports/holdout_evaluaciones_v2.csv` (2/5), `models/v2_meta_final.joblib`,
  auditoria de `ml-reviewer` (sesion `ses_f26eaf579ffexCAJBQPJrty64v`),
  fila `S08` en `experiments.csv`.
- **Experimento**: `experiments/18_p7_holdout_y_submission_v08.py`.
- **Resultado**: GO en el holdout; submission v08 validada (1154 filas; positivos
  predichos HS 44.8 / TR 32.5 / AG 38.6 %); **LB v08 = 0.78163**, por debajo de
  v03 (0.78306) en -0.0014 (dentro del error estandar ~0.013). El meta mejoro
  en dev (+0.0102) y en el holdout (+0.0180) pero **no transfirio al LB**.
- **Conclusion final**: v03 se mantiene como modelo final; la linea v2 se
  documenta como "senal complementaria no transferida al test". Lectura
  metodologica para el informe: con 8 sondas, el maximo observado (v03) esta
  sesgado hacia arriba por seleccion (winner's curse); el nivel esperado de la
  familia es ~0.776-0.780 y todo el rango de sondas (0.77034-0.78306) cabe en
  ~1 error estandar (n=1154). El gap dev-LB del meta (-0.0058) frente al de
  v03 (+0.0059) muestra que las estimaciones locales optimistas no predicen la
  transferencia, incluso cuando el holdout congelado las respalda.

---

## D-021 - P8: backlog v2 cerrado al 100 % (24/24) sin mejora adicional

- **Contexto**: el paquete P8 evaluo las cinco lineas que quedaban del backlog
  v2: calibracion (E2-14), encoder de entidades (E2-13), umbrales por rebanada
  (E2-16), especialistas por rebanada (E2-20) y espacios de features por
  etiqueta (E2-21).
- **Observacion**: ninguna supera la regla de adopcion (D-016):
  - E2-14 (calibracion Platt/isotónica anidadas): mejora la calidad de las
    probabilidades (Brier 0.1583 -> 0.1199; log-loss 0.5002 -> 0.4035) pero la
    F1 con umbrales reajustados es invariante; **cerrada** (mecanismo sin
    mejora).
  - E2-13 (encoder de entidades): cobertura informativa **0.5 %** (menciones
    0.56 %, hashtags 0.39 %, dominio trivial t.co); **cerrada por sparsity**.
  - E2-16 (umbrales por rebanada con shrinkage): curva monotona, m=0 -0.0074,
    m=100 -0.0017 frente al global anidado; el umbral local sobreajusta;
    **cerrada**.
  - E2-20 (especialistas): 11 de 21 celdas significativas tras BH, pero los
    especialistas rinden de media **-0.045** dentro de su rebanada; **cerrada**.
  - E2-21 (espacios por etiqueta): **-0.0253** (0/5 pliegues; HS -0.016,
    TR -0.024, AG -0.020); **cerrada**.
- **Hipotesis**: ninguna fuente de senal razonable del backlog aporta mejora
  transferible sobre V03 con este tamano de datos y estas restricciones.
- **Decision**: cerrar el backlog v2 al 100 %; mantener **V03** como modelo
  final; no gastar mas sondas de leaderboard (condicion de terminacion B del
  plan cumplida: todas las fuentes de senal razonables fueron evaluadas y
  documentadas).
- **Razon**: los 24 resultados son consistentes entre si (los deltas locales son
  del orden del error estandar del test); insistir seria multiplicidad sin
  hipotesis nueva.
- **Evidencia**: `reports/backlog_restante_v2.csv` y `.json`,
  `reports/figuras/fig_32_backlog_restante_v2.png`, filas `E2-13/14/16/20/21-*`
  en `experiments.csv` (115 filas), `tasks/todo.md` (Gate P8).
- **Experimento**: `experiments/21_backlog_restante_v2.py`.
- **Resultado**: 24/24 lineas cerradas; holdout intacto (2/5 evaluaciones
  usadas); notebook y `FINAL_REPORT.md` actualizados con el cierre.
- **Conclusion final**: proyecto cerrado con V03 (LB 0.78306) como resultado
  final; la investigacion v2 queda documentada como evidencia negativa
  sistematica y como aportacion metodologica (protocolo de validacion
  corregido).

---

## D-022 - Cierre documental: la spec v2 queda archivada, auditada y con brechas acotadas

- **Contexto**: la especificacion maestra de la fase v2 (113 secciones, FASES 0-99)
  fue el contrato de investigacion; el proyecto la ejecuto con los paquetes P0-P8
  pero no la tenia archivada ni auditada contra evidencia.
- **Observacion**: (a) la spec no estaba en el repositorio; (b) seis artefactos
  exigidos por las FASES 88/98/55/89 no existian; (c) `research_backlog_v2.md`
  conservaba los estados "pendiente"; (d) el informe v2 decia "0 RT" cuando son 0
  los retuits con prefijo canonico `RT @` y 8 los tweets con el token RT;
  (e) las cifras de `experiments.csv` (118) y scripts (23) estaban desactualizadas
  en notebook e informe.
- **Hipotesis**: cerrar el ciclo spec -> evidencia -> auditoria deja la entrega
  academica defendible, reproducible y sin discrepancias de trazabilidad.
- **Decision**: (a) archivar la spec en `docs/advanced_research_v2_spec.md`;
  (b) generar los artefactos faltantes con
  `experiments/22_artefactos_auditoria_v2.py`; (c) escribir la matriz
  `reports/spec_compliance_v2.md` y las auditorias `reports/rule_audit_v2.md` y
  `reports/target_encoding_audit_v2.md`; (d) cerrar los estados del backlog y
  corregir las cifras; (e) registrar las brechas acotadas (E2-12, E2-22, E2-24;
  FASES 13/54/70/73/74/75/77) como evidencia negativa documentada, sin fabricar
  resultados.
- **Razon**: la condicion de finalizacion B (FASE 99) exige evaluar y documentar
  las fuentes de senal razonables; la trazabilidad completa forma parte de la
  entrega.
- **Evidencia**: `docs/advanced_research_v2_spec.md`,
  `reports/spec_compliance_v2.md`, `reports/rule_audit_v2.md`,
  `reports/target_encoding_audit_v2.md`, `reports/error_intersection_v2.csv`,
  `reports/error_by_label_combination_v2.csv`, `reports/model_complementarity_v2.csv`,
  `reports/feature_catalog_v2.csv`, `features/v2_structured_features.parquet`,
  filas AUD-01..AUD-03 de `experiments.csv`.
- **Resultado**: cumplimiento auditado de FASES 0-99 con observaciones acotadas;
  sin cambios en el modelo final (sigue V03, LB 0.78306).

---

## D-023 - Fase final de candidatos: metas restringidos, submissions experimentales y decision de no sondear

- **Contexto**: la fase "FINAL CANDIDATE" pidio convertir la evidencia v2 en
  candidatos concretos y submissions reproducibles (maximo 3), sin gastar sondas
  por defecto y conservando V03 como referencia.
- **Observacion**: (a) screening de 125 experimentos (A=5, B=9, C=3, D=18, E=67,
  F=8, G=15); (b) OOF canonico de V03 reconstruido y verificado
  (`features/oof_v03_final.npz`, corrected_cv 0.777211); (c) metas restringidos
  evaluados con protocolo anidado: C-EMO 0.7809 (+0.0037, 3/5), C-EMO-SYN 0.7797
  (+0.0025, 3/5), C-SYN 0.7793 (+0.0021, 3/5), C-EMO-SYN-TGT-CYT 0.7777 (+0.0005,
  2/5); (d) holdout: C-EMO 0.8364 (+0.0081, IC95 [-0.0075, +0.0228]) y C-EMO-SYN
  0.8336 (+0.0052, IC95 [-0.0115, +0.0222]) - ambos IC cruzan 0; (e) el unico
  candidato PROMISING sigue siendo el meta v08 (0.7874), ya no transferido al LB
  (D-020).
- **Hipotesis**: una fusion restringida (V03 + emocion/sintaxis) reduce la varianza
  del meta de 44 features y podria transferir mejor.
- **Experimento**: `experiments/23_candidatos_finales.py` (evaluacion) y
  `experiments/25_submissions_candidatos.py` (refit y submissions v09/v10/v11).
- **Resultado**: ningun candidato nuevo cumple el criterio PROMISING (delta
  >= +0.005 o IC del holdout > 0); las tres submissions se generan como artefactos
  experimentales validados y reproducibles (hashes estables en re-ejecucion).
- **Decision**: mantener **V03 como referencia y submission principal**; no
  recomendar sonda de leaderboard; conservar los candidatos documentados. Si se
  decidiera sondear por decision explicita, el mejor justificado es C-EMO.
- **Razon**: los deltas locales son del orden del error estandar del test
  (n=1154, EE ~0.013) y los IC del holdout cruzan 0; el presupuesto D-016 deja 2
  sondas y no hay evidencia que las justifique.
- **Evidencia**: `reports/final_candidate_screening.md`,
  `reports/final_candidate_comparison.md`, `reports/final_candidates.csv`,
  `reports/canal_ranking_final.csv`, `reports/submission_validation_v2.md`,
  `reports/submission_hashes_v2.json`, `submissions/submission_manifest.csv`,
  `submissions/toxic_submission_v09..v11*.csv`, filas FIN-01..FIN-10 de
  `experiments.csv`; holdout 4/5.

---

## D-024 - Sondeo LB de los candidatos finales: los tres por debajo de V03; cierre de la fase

- **Contexto**: por decision explicita de Carlos se enviaron a Kaggle las tres
  submissions experimentales de la fase final (v09, v10, v11; D-023).
- **Observacion**: resultados del LB: v09 (C-EMO) **0.77895**; v10 (C-SYN)
  **0.77978**; v11 (C-EMO-SYN) **0.78150**. Los tres por debajo de V03 (0.78306);
  dif del mejor (v11) = -0.00156, muy dentro del error estandar del test
  (n=1154, EE ~0.013). Gaps local-LB: v09 -0.00199, v10 +0.00047, v11 +0.00184.
- **Hipotesis**: los metas restringidos (V03 + canales) podrian transferir mejor
  que el meta de 44 features.
- **Experimento**: sondas v09/v10/v11 (`submissions/toxic_submission_v09..v11*.csv`).
- **Resultado**: ninguna supera a V03; el ranking local no predijo el ranking del
  LB (C-EMO era el mejor local y obtuvo el peor LB; C-EMO-SYN, segundo local,
  obtuvo el mejor). El par complementario emocion+sintaxis fue el que mejor
  transfirio, sin alcanzar la referencia.
- **Decision**: **V03 se mantiene como modelo y submission finales**; la fase
  final queda cerrada con 12 sondas registradas; no se generan mas sondas.
- **Razon**: los tres resultados caen dentro de la banda del LB documentada
  (0.770-0.783) y son indistinguibles entre si y de V03; persistir seria
  multiplicidad sin hipotesis nueva (condicion de cierre FASE 99-B).
- **Evidencia**: `submissions/leaderboard_log.csv` (12 sondas),
  `reports/final_candidate_comparison.md` (resultados LB),
  `submissions/submission_manifest.csv` (estados actualizados), filas
  FIN-08..FIN-10 de `experiments.csv`.

## D-025 - Apertura de la fase v4: protocolo, gates y presupuesto de control

- **Contexto**: tras el cierre de la fase v2 (D-021) y de la fase final de
  candidatos (D-024), Carlos abre una nueva fase de investigacion
  (`ADVANCED RESEARCH v4`, 108 secciones) con el objetivo de encontrar
  alternativas genuinamente nuevas a V03 (LB 0.78306) y no mas variantes de
  modelo. La especificacion prohibe repetir lineas ya probadas (§84, §85-§89).
- **Observacion**: (a) V03 esta congelada y verificada byte a byte; (b) el
  protocolo v2 (D-016) sigue vigente; (c) el presupuesto de control esta
  agotado en su mayor parte: 4/5 evaluaciones de holdout usadas y 12 sondas de
  LB; (d) con n=1154 y EE ~0.013, 0.78306 es el maximo observado de una familia
  cuyo nivel esperado es ~0.776-0.780 (winner's curse, D-014/D-020).
- **Hipotesis**: las lineas nunca exploradas (consistencia de etiquetas, dureza
  de ejemplos y reweighting, weighting supervisado, modelo conjunto de
  etiquetas, calibracion de las entradas del meta) pueden o bien producir una
  mejora transferible, o bien cerrar con evidencia la pregunta de cuanto del
  error es ruido de etiqueta, informacion ausente o limitacion del modelo.
- **Decision**:
  1. **Protocolo identico a D-016** (grupos near-duplicado UNION pseudo-autor,
     desarrollo 1.483, `corrected_cv` primaria, bootstrap pareado por tweet,
     regla de adopcion >= +0.005 y >= 3/5 pliegues o complementariedad con
     IC > 0). No se modifica ni V03 ni el holdout.
  2. **Presupuesto**: la quinta (ultima) evaluacion de holdout se reserva para
     el unico mejor finalista de v4. El envio al leaderboard es manual y solo
     para candidatos que pasen todos los gates; se marca `READY_FOR_KAGGLE_PROBE`.
  3. **Multiplicidad**: `reports/research_backlog_v4.md` se pre-registra antes de
     experimentar; `reports/v4_gap_analysis.md` clasifica cada idea de la spec
     como NEW / NOT_TESTED / TESTED_NEGATIVE / TESTED_BUT_BADLY_INTEGRATED /
     TESTED_BUT_UNCERTAIN / TESTED_AND_CLOSED / TRANSFER_FAILED.
  4. **Orden por valor esperado**: A/B (ruido y dureza) -> C/E (geometria y
     modelo conjunto) -> H (calibracion de las entradas del meta) -> F/G
     (diagnostico y modelos estadisticos) -> D (solo subtemas no cubiertos).
  5. **Rama** `proyecto-kaggle-toxic-2026-v4` desde `main`, con PR al cierre.
  6. **Entregable**: la fase v4 se documenta en `reports/advanced_research_v4.md`
     y, al cerrar, se anade una seccion nueva a `FINAL_REPORT.md` y se actualiza
     el notebook preservando las conclusiones previas.
- **Razon**: con 128 experimentos y un plateau documentado, la fase debe
  producir evidencia de generalizacion, no un maximo local; el presupuesto de
  control se protege para que un unico finalista pueda verificarse.
- **Evidencia**: `docs/advanced_research_v4_spec.md` (spec archivada),
  `tasks/plan_v4.md`, `reports/v4_gap_analysis.md`,
  `reports/research_backlog_v4.md`, `reports/signal_coverage_v4.csv`,
  `experiments/26_scaffold_v4.py` y `reports/baseline_v03_v4_check.json`
  (veredicto OK: 5/5 hashes, `corrected_cv` 0.777211 reproducida, 0 grupos
  compartidos entre desarrollo y holdout).

## D-026 - P1 (TRACK A y B): el ruido de etiqueta por duplicacion es escaso y no explica el error; la dureza util es la distancia a la frontera

- **Contexto**: primera linea de la fase v4 (P1). Objetivo: medir la
  consistencia de etiquetas en clusters de texto repetido y construir un mapa de
  dureza **out-of-fold** (spec §7-§13), antes de intentar reponderar ejemplos.
- **Observacion**: (V4-A01) con cuatro criterios de agrupamiento (exacto,
  normalizado, Jaccard 0.8, cosimilitud TF-IDF 0.85) hay **15 clusters con 2+
  miembros** que cubren 84 tweets (4.85 %) y solo **2 son contradictorios**. No
  hay duplicados exactos en train (los 5 exactos estaban en test). La
  contradiccion afecta a HS 48 (2.77 %), TR 48 (2.77 %) y AG 50 (2.89 %)
  tweets, y se concentra en **un solo cluster de 48 tweets** con el patron
  "Cllate puta" (38 anotados 000 frente a 10 con 110/111). (V4-A02) la tasa de
  error OOF de V03 es **menor** dentro de los clusters contradictorios
  (HS 0.046 vs 0.218; TR 0.023 vs 0.113; AG 0.093 vs 0.190). (V4-B01) la
  dificultad por etiqueta si ordena bien (`p_true` vs error: r = -0.795, -0.810,
  -0.801) pero **agregar las tres etiquetas rompe el orden**: la BCE media
  correlaciona -0.136 con el numero de errores, mientras la cercania a la
  frontera correlaciona +0.345 y la violacion del umbral +0.811. (V4-B02) las
  poblaciones quedan en 1.144 faciles, 296 "dificiles coherentes" (error medio
  0.351 por etiqueta) y 43 "posible ruido" (error medio 0.054).
- **Hipotesis**: (a) el ruido de anotacion es material y explica parte del
  error; (b) la BCE OOF media es una buena medida de dureza para ponderar.
  Ambas quedan **refutadas**.
- **Metodo**: `experiments/27_label_consistency_v4.py` (4 criterios de
  clustering, union-find, vecindario top-3 persistido) y
  `experiments/28_hardness_v4.py` (dureza OOF, vecindad, desacuerdo de 20
  canales OOF no redundantes, bootstrap pareado de 2.000 remuestreos).
- **Resultado**: los numeros de las cuatro filas `V4-A01`, `V4-A02`, `V4-B01` y
  `V4-B02` de `experiments.csv`; artefactos `reports/label_consistency_v4.csv`,
  `reports/hardness_vs_noise_v4.md`, `features/hardness_v4.npz`.
- **Interpretacion**: el error de V03 **no** es ruido de anotacion por
  duplicacion, sino dificultad del modelo ante ejemplos coherentes. La ambiguedad
  de anotacion existe (el cluster de insultos tiene 21 % de desacuerdo) pero el
  modelo la resuelve bien porque sigue la anotacion mayoritaria. Ademas, la via
  que la spec propone en §11 (ponderar por perdida OOF) es tecnicamente
  defectuosa en este corpus: la BCE agregada no preserva el orden de dificultad
  por compensacion entre etiquetas.
- **Decision**: (1) V4-A01 y V4-A02 quedan cerradas con evidencia; (2) V4-B03
  se reorienta a ponderar por **distancia a la frontera de decision** y por
  `error_score`, nunca por BCE agregada; (3) V4-B05 (downweight/remove de
  contradicciones) se mantiene con **expectativa baja** (actua sobre < 3 % de los
  ejemplos) y se ejecuta solo como control negativo, no como candidata; (4) la
  ambiguedad local del vecino mas cercano se conserva como variable de
  analisis para V4-F03.
- **Razon**: evita gastar las tres estrategias de la spec §8 en una via cuyo
  techo de impacto ya esta cuantificado, y evita un reweighting mal definido.
- **Evidencia**: `reports/label_consistency_v4.json`,
  `reports/label_consistency_v4_ejemplos.md` (revision manual de los clusters),
  `reports/hardness_v4.json`, `reports/hardness_v4.csv`,
  `reports/figuras/fig_32_hardness_v4.png`.

## D-027 - P2 (TRACK B): el hard example mining clasico empeora; solo funciona bajar el peso de los ejemplos que mas violan el umbral

- **Contexto**: V4-B03, V4-B04 y V4-B05 del backlog v4, con el diseño sin fuga
  exigido por la spec (§11): para cada pliegue externo se calcula la dureza con
  un CV **anidado** de 4 pliegues sobre el entrenamiento, de modo que ningun peso
  procede de un modelo que vio ese ejemplo. El control `lr:none` reproduce
  exactamente la referencia (0.7772), lo que valida el pipeline.
- **Observacion**: (a) subir el peso a los ejemplos difficiles **empeora** de
  forma consistente: `mild_up` 0.7751, `medium_up` 0.7676, `clipped_extremos`
  0.7736 frente a 0.7772, con IC bootstrap negativo para `medium_up`
  ([-0.0166, -0.0038]); (b) `medium_down` 0.7590, es decir, bajar el peso a los
  ejemplos mas cercanos a la frontera tambien empeora; (c) el unico esquema que
  mejora es `down_violacion`, que baja el peso de los ejemplos que **mas violan**
  el umbral: LR 0.7784 (+0.0036, IC95 [-0.0016, +0.0092]) y SVC 0.7801, que es
  +0.0086 respecto de su propio control (0.7715); (d) la ponderacion
  independiente por etiqueta empeora (0.7686); (e) las tres estrategias de la
  spec §8 no mueven la metrica: downweight 0.7777 (+0.0005) y remove 0.7768
  (-0.0004).
- **Hipotesis**: dar mas peso a los ejemplos que el modelo falla (hard example
  mining) y penalizar los que mas violan el umbral (descartar ruido) son la misma
  familia de mecanismos con signo opuesto, y solo uno puede ayudar.
- **Metodo**: `experiments/29_reweighting_v4.py` (6 esquemas x 2 modelos +
  ponderacion por etiqueta + 2 estrategias de contradiccion; 5 pliegues; OOF
  anidado; bootstrap pareado de 2.000 remuestreos).
- **Resultado**: filas `V4-B03`, `V4-B04` y `V4-B05` de `experiments.csv`;
  `reports/reweighting_v4.csv` (14 variantes) y `features/oof_reweight_v4.npz`.
- **Interpretacion**: en este corpus los ejemplos que el modelo falla son, desde
  el punto de vista del modelo, frontera irreductible con la representacion
  disponible (insulto en registro de afiliacion, odio implicito, referencia
  colectiva): subir su peso empuja la frontera y degrada el resto, mientras que
  excluirlos del entrenamiento permite un limite de decision mas limpio. El
  patron es coherente con D-026 (el ruido textual no explica el error) y con la
  revision bibliografica (DiscrimLoss, `reports/estado_arte_v4.md` §4).
- **Decision**: (1) V4-B03, V4-B04 y V4-B05 quedan **cerradas sin adopcion**
  (ninguna variante cumple el criterio D-016 de +0.005 y 3/5 pliegues);
  (2) el TRACK B se da por cerrado en su parte de reweighting; (3) el unico
  mecanismo con señal positiva (`down_violacion`) se arrastra al TRACK C como
  V4-C05 para comprobar si es **complementario** del barrido de class weights;
  (4) no se borra ninguna etiqueta ni ejemplo (§8, §90).
- **Razon**: se evita gastar mas LINEAs en una familia con evidencia negativa
  consistente y se concentra el esfuerzo en el mecanismo que mostro senal.
- **Evidencia**: `reports/reweighting_v4.json`,
  `reports/figuras/fig_33_reweighting_v4.png`.

## D-028 - P4 (TRACK H): el shift de calibracion del meta se mide y se refuta; el meta de 36 features queda como candidato

- **Contexto**: `advanced_research_v2.md` §4.3 dejo abierta la hipotesis de que
  el meta v2 no transfirio al leaderboard porque sus entradas (scores OOF de
  bases ajustadas sobre 4/5 del desarrollo) se undistribuyen al predecir con
  bases ajustadas sobre el 100 % del train. V4-H01 mide ese efecto dentro del
  desarrollo, sin usar el holdout ni el test, reproduciendo el mecanismo exacto.
- **Observacion**: (a) el desplazamiento es **pequeno**: KS medio 0.0624 entre las
  dos distribuciones; (b) `meta_shift` (bases al 100 %, sin alinear) = 0.7828
  frente a `meta_control` (mismo regimen que el entrenamiento del meta) = 0.7791;
  es decir, el shift **no causa dano**, el dano medido es de -0.0036 (a favor);
  (c) la alineacion por cuantiles recupera +0.00005 (nula) y Platt empeora
  -0.0171; (d) el meta de 36 features (4 bases x 3 etiquetas x [p, margen,
  entropia], Ridge alpha=3) alcanza **0.7828 = +0.0056 sobre V03 con 4/5 pliegues
  mejorados e IC95 del delta [+0.0036, +0.0169]**.
- **Hipotesis**: el shift de calibracion es la causa de la no transferencia del
  meta v2 (LB 0.78163 frente a 0.78306 de V03).
- **Metodo**: `experiments/32_meta_shift_v4.py`; por pliegue externo se calculan
  las features del meta en dos regimenes (CV interno de 4 pliegues sobre el
  entrenamiento externo frente a bases al 100 %), y se comparan cuatro
  situaciones (control, shift, cuantil y Platt) con la misma arquitectura.
- **Resultado**: filas y artefactos de `V4-H01`; `reports/meta_shift_v4.csv` y
  `.json`.
- **Interpretacion**: la hipotesis queda **refutada**: el mecanismo medido es
  demasiado pequeno para explicar la diferencia de LB, que ademas esta dentro
  del error estandar del test. La lectura correcta es la de D-020: el meta y V03
  son indistinguibles en el test, y la "mejora local" del meta era sobreestimada
  por el mismo mecanismo de seleccion que la produjo.
- **Decision**: (1) V4-H01 cierra la hipotesis abierta de v2 con evidencia
  (queda registrada como cerrada, no como candidata a transferir); (2) el meta de
  36 features **supera el criterio D-016** y pasa a la lista de candidatos de
  v4, pero se marca con **riesgo de transferencia alto** por el precedente
  D-020 (mismo mecanismo de stacking, LB 0.78163); (3) no se consume la ultima
  evaluacion de holdout hasta cerrar la comparativa de P8.
- **Razon**: separar el mecanismo (cerrado) del candidato (abierto) evita gastar
  el presupuesto de control en una explicacion que ya se demostro insuficiente.
- **Evidencia**: `reports/meta_shift_v4.json`,
  `reports/figuras/fig_36_meta_shift_v4.png`, `features/oof_meta_shift_v4.npz`.

## D-029 - P8 (cierre de v4): el holdout rechaza al unico candidato que pasaba los gates; V03 sigue como modelo final

- **Contexto**: cuatro candidatos evaluados con evidencia multiple
  (`experiments/36_candidatos_v4.py`, `reports/v4_candidates.csv`). El unico que
  superaba los tres criterios de D-016 a la vez era **C**
  (``meta36_downweight``): corrected_cv 0.7882 (+0.0110), 4/5 pliegues mejorados,
  IC95 del delta pareado [+0.0054, +0.0200] y la mayor correccion neta frente a
  V03 (+34 ejemplos). Los otros tres fallaban el criterio de IC: A
  (0.7854, 5/5 pliegues, IC [-0.0002, +0.0117]), D (0.7904, 5/5 pliegues, IC
  [-0.0011, +0.0148]) y B (0.7828, IC [+0.0031, +0.0166], pero nivel menor).
- **Observacion**: el holdout congelado de 248 tweets, que no se habia usado en
  toda la fase y cuya quinta evaluacion estaba reservada para el mejor finalista
  (D-025), da **0.8163 para el candidato C frente a 0.8285 de V03 en el mismo
  holdout: delta -0.0122**. El candidato C empeora en la unica particion que no
  participo en ninguna decision de la fase.
- **Hipotesis**: la combinacion de downweight por violacion y class weights por
  etiqueta, dentro de un meta Ridge, mejora la generalizacion y no solo el
  corrected_cv.
- **Metodo**: `experiments/38_refit_final_v4.py`: refit sobre todo el desarrollo
  con los multiplicadores de clase (mediana de los 5 pliegues: 1.2/1.5/1.2) y los
  pesos de ejemplo derivados del `error_score` OOF; prediccion sobre holdout y
  test; submission generada y validada (sha256 `ec25354c...`).
- **Resultado**: holdout 0.8163 (HS 0.8624, TR 0.8058, AG 0.7807) frente a 0.8285
  de V03; submission v12 generada y validada pero **marcada como descartada** en
  `submissions/submission_manifest.csv`.
- **Interpretacion**: es el **cuarto caso consecutivo** en que una mejora clara en
  corrected_cv no transfiere (Optuna, meta v2, candidatos v09-v11 y ahora C).
  La evidencia acumulada es ya suficiente para afirmar que, con 1.731 ejemplos de
  train, 1.154 de test y sin modelos preentrenados, el corrected_cv de este
  protocolo **no es un selector fiable de candidatos**: su diferencia con el
  error estandar del test (~0.013) es del mismo orden que las mejoras que
  selecciona.
- **Decision**: (1) **ningun candidato de v4 se adopta**; V03 se mantiene como
  modelo y submission finales; (2) la submission v12 se conserva como artefacto
  experimental reproducible y marcada `DESCARTADO` (no se envia); (3) el holdout
  queda agotado (5/5) y no se gastan mas sondas de LB en v4; (4) la fase v4 se
  cierra con la condicion de terminacion B de la spec (§102): hipotesis nuevas
  investigadas y cerradas con evidencia, sin fabricar mejora.
- **Razon**: el holdout es la unica evidencia externa disponible y contradicts
  con el corrected_cv; con multiplicidad ya alta, descartar es la decision
  metodologicamente correcta.
- **Evidencia**: `reports/refit_final_v4.json`, `reports/v4_candidates.csv`,
  `reports/v4_candidate_comparison.md`, `reports/holdout_evaluaciones_v2.csv`
  (5/5), `submissions/submission_manifest.csv`.

## D-030 - Scope of external unlabeled data for from-scratch embeddings

- **Contexto**: la fase v5 («Embedding from Scratch») introduce por primera vez en
  el proyecto texto externo. Las fases anteriores trabajaron bajo D-002, que
  prohibía usar corpus o etiquetas externos. La especificación v5 abre esa
  posibilidad **con un alcance acotado**, y el usuario **(Carlos)** cerró el
  alcance exacto antes de autorizar la ejecución.
- **Autorización (transcripción literal de lo comunicado por el usuario)**:

  > El profesor autorizó para esta competencia el uso de datos externos no
  > etiquetados exclusivamente para entrenar desde cero una representación de
  > embeddings/vectorización. Los datos externos no pueden utilizarse para
  > entrenamiento supervisado, clasificación, generación de labels,
  > pseudo-labeling, ajuste de thresholds ni ninguna otra forma de aprendizaje
  > sobre HS/TR/AG.

- **Alcance de la autorización** (límite explícito):
  1. Los corpus externos se usan **exclusivamente** para aprender la
     representación distribuida (pesos de embeddings/vectorización).
  2. Queda **prohibido** cualquier uso de datos externos para: entrenamiento
     supervisado del clasificador final, clasificación, generación de labels,
     pseudo-labeling, ajuste de umbrales o cualquier otra forma de aprendizaje
     sobre HS/TR/AG.
  3. Toda supervisión del clasificador procede **solo** del train oficial de
     TOXIC.
  4. El embedding se entrena **completamente desde cero**; el vocabulario o
     tokenizer, cuando corresponda, se construye desde cero.
  5. Cada corpus externo debe documentar `source`, `url`, `version`, `license`,
     `download_date` y `sha256` (§29 y §35 de la spec v5).
  6. No se incorpora ninguna fuente nueva de forma silenciosa durante una
     ejecución.
- **Evidencia**: **NO existe** correo, foro ni captura formal archivada en el
  repositorio que documente literalmente la autorización. Esta decisión registra
  **únicamente** la información comunicada directamente por el usuario en la
  sesión del 2026-09-25. No se inventa, reconstruye ni infiere ninguna evidencia
  escrita. Queda como acción opcional del usuario incorporar el documento
  original si dispone de él; no bloquea la fase.
- **Regla ante ambigüedad**: si durante el desarrollo aparece una
  interpretación ambigua sobre si una fuente concreta entra dentro de esta
  autorización, **no se asume permitida**: se registra la duda en
  `DECISIONS.md` y se detiene **únicamente esa fuente** hasta resolverla.
- **Fuentes habilitadas por el usuario** (con sus límites):
  - **Wikipedia ES**: fuente principal (P1); disponible, con procedencia clara,
    poco ruido y cobertura general.
  - **mC4 (`allenai/c4`, config `es`)**: fuente web principal (P2); accesible
    sin el bloqueo manual que presenta OSCAR.
  - **OSCAR**: opcional; solo con acceso legítimo y directo. No se retrasa la
    investigación por OSCAR ni se usan mecanismos para saltar restricciones de
    acceso.
  - **HPLT / CEREAL**: opcionales y no obligatorios. Antes de incorporarse se
    documenta procedencia, licencia, cobertura en español, tamaño, diversidad,
    solapamiento con Wikipedia/mC4 y coste de procesamiento. No se dedica tiempo
    desproporcionado a CEREAL.
  - Prioridad de composición: `Wikipedia ES` → `mC4-es` → `mezcla` →
    `HPLT/CEREAL/OSCAR` solo si aportan información marginal real. **No** se
    construye una mezcla enorme antes de medir el aporte individual.
- **Escala autorizada**: hasta **500M tokens** como techo de esta fase inicial,
  con la secuencia obligatoria `10M → 50M → 100M → decidir → 250M → decidir →
  500M → decidir`. No se supera 500M sin una nueva decisión explícita del
  usuario. Antes de cada expansión se registra `RATIONALE_FOR_CORPUS_EXPANSION`
  con al menos una condición de la §30 de la spec.
- **Implementación autorizada**:
  - **fastText C++ oficial** compilado localmente (g++, cmake, make): la
    librería implementa el algoritmo, pero **los pesos se entrenan desde cero**
    y no se descarga ningún modelo.
  - **Word2Vec-control** mediante el mismo binario con `minn=0, maxn=0`,
    registrado como `WORD2VEC-CONTROL / FASTTEXT-SG-SUBWORDS-OFF`.
  - **PyTorch** solo como respaldo o control metodológico; no se reimplementa
    fastText en PyTorch si la versión C++ resuelve el experimento.
  - **gensim**: no se instala ni se depende de él (no hay wheel para Python
    3.14 y la compilación desde source tiene riesgo alto).
  - **GloVe**: no implementado por defecto; solo si FastText y Word2Vec muestran
    una señal que justifique una tercera geometría.
- **Anti-fuga y auditoría**: la auditoría debe comprobar 0 usos de
  `transformers`, BERT, RoBERTa, `from_pretrained`, embeddings preentrenados,
  labels externas, labels de test, inclusión de `toxic_test.csv` y de datasets
  externos de hate speech; y debe distinguir **texto usado para el embedding** de
  **labels usadas para supervisión**. La variante principal es
  **EXTERNAL-ONLY**: no se añaden los 1.731 tweets de TOXIC al corpus. La
  adaptación de dominio es secundaria y **in-fold** (`embedding_train_k` = corpus
  externo + textos de train del fold `k`), nunca con validación, holdout ni test.
- **Normalizaciones**: `EMB-NORMAL` y `EMB-ASCII` (lowercase, transliteración
  reproducible, URLs, menciones, hashtags, formas informales y preservación de
  elongaciones). **Sin** correctores ortográficos, **sin** reconstrucción de
  tildes, **sin** información semántica externa y **sin** etiquetas.
- **Protocolo de evaluación**: `corrected_cv` con grupos near-duplicado ∪
  pseudo-autor (`src/splits.py`) y referencia V03 0.7772 (nunca 0.7989 como
  referencia primaria). El **holdout está agotado (5/5)** y no se usa ni se crea
  otro. Jerarquía de evidencia: `corrected_cv` → leaderboard.
- **Criterios de adopción** (§29): **A** mejora ≥ +0.005 con estabilidad
  razonable; **B** complementariedad robusta; **C** evidencia representacional
  fuerte aunque la F1 sea marginal. Se distingue `adopted` / `candidate` /
  `exploratory` y **no se presenta una mejora exploratoria como mejora final**.
- **Entrega**: rama independiente `proyecto-kaggle-toxic-2026-v5`; V03 intacta;
  no se borra evidencia histórica; no se modifica retrospectivamente D-024 ni
  decisiones anteriores; **sin** submissions automáticas; **sin** push
  automático sin autorización explícita en la sesión; PR al finalizar y **sin**
  merge automático.
- **Alcance autorizado en esta sesión**: **P0, P1 y P2**. No se descargan 500M
  tokens antes de completar el diagnóstico de cobertura (P1) y el primer
  resultado downstream (P2).
- **Evidencia en disco**: `docs/embedding_from_scratch_v5_spec.md` (spec maestra
  transcrita), `tasks/plan_v5.md`, `reports/embedding_compliance.md`,
  `data/external/corpora/corpus_manifest.json` y scripts `embedding_*.py`.
- **Conclusión**: la autorización queda acotada a la representación
  distribuida; cualquier uso que exceda ese alcance se considera no autorizado y
  detiene el experimento correspondiente.

## D-031 - Estado de la representacion distribuida y palanca de mejora

- **Contexto**: P0 a P3 de la fase v5 (embedding desde cero sobre corpus espanol
  externo no etiquetado, D-030) ya tienen veredicto. Esta decision fija el estado
  de cada linea, lo que queda abierto y **que evidencia seria necesaria** para
  cambiar de estado. No sustituye a D-030 y no altera ninguna decision anterior.
- **Estado por linea**:
  1. **Embedding como sustituto de V03: REFUTADO.** 0.6530 frente a 0.777211
     (delta -0.1242, IC95 [-0.1481, -0.1006]) con el pooling A, el mejor de los
     tres poolings. El suelo aleatorio de la misma dimension da 0.5404, de modo
     que hay senal real (+0.113) pero la mitad de la que aporta V03.
  2. **Subpalabras: CONFIRMADAS como geometria, no como mejora downstream.**
     Los vecinos pasan de distribucionales a morfologicos (`violencia` ->
     `violentas`, `violento` frente a `discriminacion`, `delincuencia`) y aportan
     +0.0138 frente al control Word2Vec. La identidad del vecino de una variante
     ortografica **no** es concluyente con 10M tokens y no se afirma.
  3. **Fusion hstack con bloque L2: CANDIDATE.** 0.7818 (delta +0.0046,
     IC95 [-0.0032, +0.0124]). Aporta, pero no alcanza el +0.005 pre-registrado ni
     el IC95 que excluye el 0, y su mejora se concentra en elongaciones (+0.0368),
     urls (+0.0204), hashtags (+0.0151) y colectivos (+0.0150): las rebanadas que
     P1 identifico como peor cubiertas. Complementariedad 2x2: 8 celdas solo de la
     fusion frente a 5 solo de V03.
  4. **Fusion con bloque denso estandarizado: DESCARTADA.** 0.6901. La escala
     relativa de los bloques decide que senal sobrevive en un LR regularizado con
     L2; se fija la normalizacion L2 por documento como convencion de la fase.
  5. **Apilamiento (meta-modelo anidado): CERRADO.** 0.7678, delta -0.0094 con
     IC95 completamente negativo; rompe 53 celdas por cada 5 que arregla. Quinta
     vez en el proyecto que el stacking generico no aporta.
  6. **Pooling**: A (media simple) gana a B (TF-IDF-weighted, 0.6348) y a C
     (media+max, 0.6388). Ponderar por IDF concentra el vector en pocas palabras,
     que es justo lo que el averaging debia distribuir.
  7. **LR gana a LinearSVC** por 0.0019, irrelevante frente al error del
     protocolo (std de pliegues 0.0161).
- **Por que NO se escala el corpus de Wikipedia a 50M/100M**: la seccion 30
  exige que una expansion aporte cobertura, ganancia marginal de vocabulario,
  mejora downstream o diversidad. El diagnostico de P1 senala que la deficiencia
  es de **registro, no de tamano**: el 66.6 % de los tipos ausentes son
  deformaciones ortograficas (que los subpalabras ya cubren) y el resto son formas
  coloquiales y elongadas que un corpus enciclopedico no contiene. Mas paginas de
  Wikipedia no las aportan. Ademas, la parte 1 del dump ya descargada contiene
  material de sobra para 100M tokens, de modo que la decision no es de descarga
  sino de/utilidad.
- **Palanca que la evidencia si identifica**: un corpus de **registro web**
  (mC4-es; OSCAR sigue gated) atacaria directamente `coloquial` (0.214) y
  `elongaciones` (0.035), que son las rebanadas donde la fusion mejora mas.
  Incorporar una fuente nueva **no** se hace de forma silenciosa (D-030, punto 6):
  requiere registrar procedencia, licencia, version y hashes en
  `corpus_manifest.json` y la autorizacion explicita del usuario.
- **Evidencia que haria cambiar el estado de la fusion a `adopted`**: (a) repetir
  la fusion sobre un corpus de registro web y comprobar que la mejora crece y se
  concentra en las mismas rebanadas, o (b) mejora local que se mantenga tras
  corregir el ruido del protocolo. **No** se considera evidencia suficiente: una
  mejora del `corrected_cv` aislada, por cinco reasons acumulados (Optuna, meta v2,
  v09-v11, candidato C de v4 y ahora esta).
- **Sin submissions**: no se genera ninguna. El envio, si alguna vez, lo decide el
  usuario con evidencia externa, no con este numero.

## D-032 - Cierre de la fase v5: la cobertura improves, la complementariedad no

- **Contexto**: D-031 dejo una sola palanca abierta, el **registro** del corpus,
  y fijo la condicion para cerrarla: repetir la fusion con un corpus de registro
  web y comprobar si la mejora crece. P4 construyo el corpus de mC4-es (10M tokens,
  tramo de 40 MB del shard 0, sha256 `96ed39f4...`, licencia ODC-BY) y P5 entreno
  FastText sobre el y evaluo las fusiones. Esta decision registra el resultado.
- **Lo que se confirma**: la cobertura del vocabulario TOXIC sube de forma amplia.
  `web 10M` alcanza 0.8256 por tipo y 0.9512 por token frente a 0.6839 y 0.8833 de
  Wikipedia; la `union 20M` llega a 0.8413 y 0.9553. Los grupos que P1 marco debiles
  se Specifically: elongaciones 0.039 -> 0.221, coloquial 0.692 -> 0.941, insultos
  0.778 -> 0.983. El corpus web anade 1.346 tipos TOXIC que la fuente enciclopedica
  no tiene, y varias variantes ortograficas (`callate`, `arabe`, `estn`, `espaa`)
  pasan a estar en el propio vocabulario.
- **Lo que se refuta**: la mecanica causal que se hipotetizo. La mejora de
  cobertura se ve con claridad en el embedding **aislado** (0.6530 -> 0.6638 web ->
  0.6658 mixto, +0.0128), pero la **aportacion complementaria sobre V03 no crece**:
  `FUSION-WIKI` 0.7818 (+0.0046), `FUSION-WEB` 0.7812 (+0.0040), `FUSION-MIXTO`
  0.7776 (+0.0004). La contribution marginal del embedding es practicamente
  constante (~+0.004) con cualquiera de los tres corpora, con IC95 que toca el 0 en
  todos los casos. Las rebanadas tampoco se separan: elongaciones +0.037 (wiki),
  +0.021 (web), +0.037 (mixto); el patron es el mismo.
- **Lectura**: el ~+0.004 no proviene de la cobertura sino de una senal debil y
  distinta que TF-IDF char(3,5) ya captura en su mayor parte. Un corpus mejor
  afina esa misma senal y no anade una nueva. Por eso ni 2,5x mas de cobertura ni
  un registro muy distinto cambian el resultado.
- **Estado final de la fase v5**:
  1. Sustituto de V03: **refutado** (0.6658 en el mejor caso frente a 0.777211).
  2. Complementariedad: **CANDIDATE no adoptado**, senal constante de +0.004 con
     IC95 que incluye el 0 y muy por debajo del +0.005 pre-registrado.
  3. Palanca de corpus: **agotada**. Ni el tamano (D-031) ni el registro (esta
     decision) mueven la aportacion complementaria.
  4. Subpalabras: **si** aportan frente al control sin subpalabras (+0.0138 en P2,
     +0.0050 en la fusion web) y cambian la geometria de forma medible.
  5. Stacking: **cerrado** (cinquena vez que no aporta).
  6. Escala de corpus: **no se baja de 10M a 50M/100M ni a 500M**. El techo de 500M
     queda sin usar y no se usara sin una razon nueva.
- **Condiciones de cierre de la seccion 43**: se cumple la condicion A (el
  embedding es consistentemente inferior como sustituto), la B (hay senal pero no
  alcanza el alcance exigido) y la E (la complementariedad no crece ni con mas
  cobertura ni con otro registro). La fase se cierra **sin adoptar** el candidato.
- **Lo que queda vivo**: `FUSION-FASTTEXT-WIKI-10M` (0.7818) queda registrado como
  artefacto reproducible, con su matriz de complementariedad y sus rebanadas, por
  si en el futuro se quisiera repetir la medida con evidencia externa. **No** se
  genera submission: no hay holdout (agotado 5/5) y gastar una sonda del
  leaderboard para +0.004 con IC95 tocando el 0 no esta justificado. Si el usuario
  decide enviarla, se hace de forma expresa y se registra como tal.
- **Leccion general para el proyecto**: la cobertura de vocabulario era el
  parametro que toda la fase daba por relevante (secciones 7 a 9) y resulto ser
  una condicion necesaria pero **no suficiente** para que una representacion
  distribuida aporte sobre una representacion dispersa de char n-grams. Medir la
  cobertura fue correcto; suponer que su mejora se traduria en mejora downstream
  fue el error, y lo detecto un experimento de control (comparar el embedding solo
  frente a su fusion) en lugar de la metrica.

## D-033 - Curva 10M -> 50M: el techo no es la representacion

- **Contexto**: D-032 cerro la fase con la palanca de **registro** agotada. Quedaba
  por medir la otra dimension de la §31, el **tamano** del corpus con arquitectura
  fija (seccion 10 de la spec). El usuario lo solicito explicitamente, con lo que
  esta decision no cambia el criterio de adopcion: mide una prediccion declarada.
- **Razon de la expansion** (`RATIONALE_FOR_CORPUS_EXPANSION-50M`, registrada en
  `corpus_manifest.json` antes de construir): condiciones 1, 2 y 3 de la §30, **sin
  descarga adicional** (la parte 1 del dump, ya bajada en P1, contiene el material)
  y con el pipeline de P1 **importado**, de modo que los 10M anteriores estan
  contenido **identicamente** en los 50M (verificado: `True`). Costes: construccion
  76 s, entrenamiento 577 s por modelo.
- **Lo que se midio** (arquitectura, semilla, protocolo, umbrales y bootstrap
  identicos; lo unico que cambia es el corpus):

  | corpus | cobertura por tipo | embedding solo | fusion con V03 | delta |
  |---|---|---|---|---|
  | Wikipedia 10M | 0.6839 | 0.6530 | 0.7818 | +0.0046 |
  | Wikipedia 50M | 0.7703 | **0.6687** | 0.7803 | +0.0031 |
  | web mC4-es 10M | 0.8256 | 0.6638 | 0.7812 | +0.0040 |
  | mixto 20M | 0.8413 | 0.6658 | 0.7776 | +0.0004 |

- **Conclusion 1 (H-50M refutada)**: quintuplicar los tokens **no** aumenta la
  aportacion complementaria; la fusion baja 0.0015. Sumado a D-032, las dos
  dimensiones del corpus quedan medidas y ninguna mueve el resultado.
- **Conclusion 2 (H-contra confirmada)**: el embedding **aislado** si mejora de
  forma clara (+0.0157) y la cobertura sube (+0.0864 por tipo). La representacion
  es genuinamente mejor con 5x datos; simplemente **no translates** en
  complementariedad. Por tanto el techo **no es la calidad de la representacion**.
- **Conclusion 3 (diagnostico del limite)**: el limite es estructural. Con 1.483
  ejemplos de entrenamiento y un bloque de 27.431 char n-grams, un vector denso de
  200 numeros no tiene espacio para aportar: mejora de 0.003 a 0.005 sobre V03 con
  cualquier corpus, y el IC95 toca el 0 siempre. Un corpus de 500M tokens, mas
  representacion o mejor registro solo se moverian dentro de ese margen.
- **Conclusion 4 (subpalabras)**: su ventaja **se reduce** con la escala (+0.0138 a
  10M, +0.0030 a 50M), de forma coherente con la mecanica de fastText: a mas
  tokens, mas palabras tienen evidencia propia en el vocabulario y hay menos que
  reconstruir con subpalabras.
- **Decision**: **no se escala a 100M ni a 500M**. Se cumple el criterio de parada
  declarado en la justificacion de la expansion. La fusion sigue siendo
  `CANDIDATE` no adoptado (mejor resultado de toda la fase: +0.0046 a 10M de
  Wikipedia, IC95 que toca el 0). **V03 se mantiene.**
- **Lo que esta decision deja escrito para el futuro**: si algun dia se quisiera
  volver a intentar esta linea, la variable que queda por explorar **no es el
  corpus ni la representacion**, sino el techo del clasificador y el tamaño de los
  datos etiquetados. Con 1.731 ejemplos, la mejora esperable de cualquier segunda
  fuente de senal es de milesimas, y el protocolo disponible no tiene evidencia
  externa para distinguirla del ruido.

## D-034 - Tramo inferior de la curva (5M): existe un suelo, y la curva del embedding es plana

- **Contexto**: D-033 midio la curva hacia arriba (10M -> 50M) y concluyo que quintuplicar
  los tokens no aumenta la aportacion complementaria. Quedaba el tramo inferior de la §10,
  que se midio a peticion del usuario. Bajar de escala no es una expansion, asi que la §30
  no le aplica: se registro `REDUCCION_A_5M` en el manifiesto con el motivo y con la nota de
  que los 5M tokens son un **prefijo exacto** de los 10M (`es_prefijo_exacto_de_10M: True`),
  sin descarga adicional y sin tocar los corpus existentes.
- **Curva completa** (arquitectura, semilla, protocolo, umbrales y bootstrap identicos; lo
  unico que cambia es el corpus):

  | corpus | cobertura por tipo | embedding solo | fusion con V03 | delta |
  |---|---|---|---|---|
  | Wikipedia 5M | 0.6302 | 0.6670 | 0.7757 | **-0.0015** |
  | Wikipedia 10M | 0.6839 | 0.6530 | 0.7818 | +0.0046 |
  | Wikipedia 50M | 0.7703 | 0.6687 | 0.7803 | +0.0031 |
  | web mC4-es 10M | 0.8256 | 0.6638 | 0.7812 | +0.0040 |
  | mixto 20M | 0.8413 | 0.6658 | 0.7776 | +0.0004 |

- **Hallazgo 1 (suelo operativo)**: a 5M la fusion **deja de aportar** y cae por debajo de
  V03 (delta -0.0015, IC95 [-0.0091, +0.0060]). 10M es el minimo que produce el efecto
  marginal y, por encima, el efecto **no crece**. Queda un dato accionable: cualquier
  corpus de menos de ~10M tokens se descarta por principio en esta linea.
- **Hallazgo 2 (la curva del embedding aislado es plana)**: 0.6670 (5M), 0.6530 (10M),
  0.6687 (50M). La no monotonia aparente del punto de 10M **no** significa que 10M sea
  peor: los intervalos de bootstrap de 5M y 10M se solapan casi por completo, asi que la
  diferencia de 0.014 no es distinguible del ruido. Leido bien, el embedding aislado esta
  en la misma banda (0.653-0.669) en todo el rango 5M-50M: la representacion no mejora de
  forma apreciable al multiplicar por diez los datos.
- **Hallazgo 3 (correccion de D-033)**: D-033 afirmo que la ventaja de las subpalabras "se
  reduce con la escala". El tramo inferior muestra que **no** es cierto de forma monotona:
  +0.0060 a 5M, +0.0138 a 10M, +0.0030 a 50M, o sea una joroba con maximo en 10M, y los tres
  valores dentro del ruido. La caida a 50M sigue siendo coherente con la mecanica de
  fastText, pero la caida a 5M se explica por lo contrario (faltan ejemplos por palabra).
  **Se corrige la afirmacion**: la ventaja de las subpalabras es pequena en todo el rango y
  no depende de forma monotona de la escala.
- **Estado final**: el embedding sigue siendo `CANDIDATE` no adoptado. Las tres
  dimensiones explorables del corpus (tamano 5M/10M/50M, registro y composicion) estan
  medidas y **ninguna** produce una aportacion complementaria por encima de +0.005 con IC95
  que excluya el 0. V03 se mantiene.
- **Nota de coste**: los tres puntos de la curva costaron 43 s, 577 s y (esta vez) 43 s de
  entrenamiento respectivamente, porque el tiempo es lineal en tokens. Un corpus de 500M
  habria costado del orden de 1,5 h por modelo. El techo autorizado queda sin usar, y la
  decision no escalate se sostiene por evidencia y no por coste.

## D-035 - Baseline arquitectonico de v5.1, identidad de entrenamiento y presupuesto

- **Contexto**: v5 cerro con D-034. La lectura de esa fase es que la representacion no es
  el techo: el embedding aislado esta plano entre 5M y 50M tokens (0.653-0.669, dentro del
  ruido) y la aportacion complementaria a V03 se queda entre +0.0001 y +0.0046 en las cinco
  configuraciones medidas, con el intervalo de bootstrap tocando el 0. Las tres dimensiones
  del corpus quedan agotadas. La unica variable que la spec de v5.1 deja abierta es la
  **arquitectura** del entrenamiento, con un presupuesto duro de 9()+1 unidades.
- **Baseline arquitectonico (B0)**: Wikipedia 10M (E-CORP-01) / 200d / window 5 /
  min_count 3 / subpalabras 3-6 / negative 10 / sample 1e-4 / epochs 5 / sg 1 / seed 42 /
  bucket 200.000. Ya esta entrenado (sha256 `e54a363d`) y **no se reentrena**: se carga como
  artefacto. Todas las propuestas de v5.1 son esta misma configuracion con **un** parametro
  cambiado sobre el **mismo** corpus y la **misma** semilla, para que la comparacion sea
  celda a celda.
- **Identidad de entrenamiento** (seccion 34 de la spec, con dos desviaciones razonadas):
  `training_signature = sha256(corpus_procesado | corpus_version | vector_size | window |
  min_count | min_n | max_n | negative | sample | epochs | sg | seed | bucket)`.
  1. Se usa el sha256 del **corpus procesado**, no el del dump. El dump es identico para
     5M, 10M y 50M de Wikipedia, asi que con la huella del dump tres corpora distintos
     compartiran identidad y la comprobacion de la seccion 34 seria falsa.
  2. Se incluye `bucket`, que la spec no lista: determina parte de la matriz de subpalabras
     y por tanto los pesos. Excluirlo permitiria declarar nueva una configuracion que en
     realidad es otra. `hilos` queda fuera porque no cambia la configuracion, solo el orden
     no determinista de actualizacion de pesos, y esa no determinismo ya esta medida.
- **Resultado de la auditoria (EBD-17)**: los diez entrenamientos de v5 tienen **firmas
  distintas** y las **nueve** propuestas de v5.1 **no colisionan** con ninguna. Presupuesto
  abierto: 0 de 9 exploratorios y 0 de 1 reserva. Reglas de la seccion 36 limpias: 0
  recursos prohibidos, 0 textos de test en los corpus, holdout intacto 5/5, V03 verificado
  en 0.777211.
- **Gap de trazabilidad detectado**: P5 no guardaba `minn`/`maxn` por modelo (solo dentro de
  la etiqueta de texto), de modo que la firma de los tres modelos de esa corrida se declara
  en el inventario de EBD-17 y se verifica contra el artefacto. Sin esa correccion, la firma
  del control Word2Vec del corpus web coincidia con la de FastText. Los scripts de v5 no se
  modifican: son evidencia historica.
- **Contabilidad separada**: `EMBEDDING TRAINING UNIT` (cambia los pesos) !=
  `DOWNSTREAM EVALUATION UNIT` (pooling, clasificador, umbral, analisis) !=
  `AUDIT / REPRODUCTION`. Las 33 evaluaciones downstream historicas quedan en la Tabla B y
  no consumen presupuesto.
- **Nota de ubicacion**: el worktree `proyecto-kaggle-toxic-2026-v5` se migro de
  `/home/carlos/Git/PLN-v5` a `/home/carlos/Git/PLN` a peticion del usuario. Se verificaron
  los sha256 de los cinco corpus y de los diez `.vec`, y el control V03 reproduce
  0.777211 exacto desde la nueva ubicacion. El trabajo sin commitear de KAGGLE_AGE_2026_PUJ
  quedo aparcado en `git stash` (no es un commit) y sus datos no se tocaron.

## D-036 - vector_size se cierra como inerte y el gate de adopcion pasa a ser de dos niveles

- **Contexto**: P1 es la primera familia arquitectonica de v5.1. Entrena 100d y 300d
  manteniendo **todo lo demas fijo** (corpus E-CORP-01, ventana 5, min_count 3, subpalabras
  3-6, negative 10, 5 epocas, sg skipgram, seed 42, bucket 200.000). El 200d no se
  reentrena: se **reevalua** en la misma corrida, que es una `DOWNSTREAM EVALUATION UNIT` y
  no consume presupuesto. Esa reevaluacion es ademas el control de comparabilidad, y
  **reproduce v5 exacto** (FUSION 0.7818, embedding solo 0.6523 con umbrales congelados).
- **Resultado** (9 evaluaciones downstream, 2 unidades de entrenamiento):

  | variante | corrected_cv (frozen) | delta vs V03 | IC95 pareado |
  |---|---|---|---|
  | FUSION-VS-100 | 0.7823 | **+0.0050** | [-0.0028, +0.0130] |
  | FUSION-B0-200 | 0.7818 | +0.0046 | [-0.0032, +0.0124] |
  | FUSION-VS-300 | 0.7805 | +0.0033 | [-0.0043, +0.0109] |

  Embedding aislado: 0.6420 (100d), 0.6523 (200d), 0.6516 (300d). Dispersion interna de la
  familia: **0.0017**.
- **Decision**: la familia `vector_size` se **CIERRA por inercia**. No se exploran 400d,
  500d ni 768d.
- **Gate de dos niveles (correccion de criterio)**: el mejor delta toca el umbral de
  +0.005 por **cinco diezmilésimas** y su intervalo de confianza **incluye el 0**. La
  seccion 20 pedia el gate "junto con estabilidad razonable entre pliegues", y la
  complementariedad se exige con bootstrap. A partir de aqui una variante es **adoptable**
  solo si `delta >= +0.005` **y** `IC95 no incluye 0`. Tratar el umbral como un test de
  significacion y no como un criterio de magnitud fue el error que D-033 y D-034
  corrigieron al final de v5: cinco configuraciones distintas dar un +0.004 con el
  intervalo tocando el 0 es la senal de que la cifra no significa nada.
- **Hallazgo**: la capacidad dimensional **no es la palanca**. Con 1.483 ejemplos
  etiquetados, comprimir cada documento en 100, 200 o 300 coordenadas da lo mismo. Los
  slices conservan la señal de v5 (elongaciones +0.0344, urls +0.0181, hashtags +0.0151),
  es decir, es la misma aportacion de siempre y no una nueva.
- **Correccion de metodo que hay que registrar**: los tipos que se consultan al modelo son
  los **6.059 del train**, no los de train+test (8.855). Con el conjunto equivocado, 100d
  salia con delta **-0.0065** y la familia aparecia "abierta por dispersion". La conclusion
  (inerte) era la misma, pero los numeros no, y un error de universo de tipos es
  exactamente el tipo de fallo que produce una conclusion falsa con apariencia solida.
  La cobertura si se mide sobre train+test, como en v5 (0.6839 por tipo, 0.8833 por token).
- **Presupuesto**: 2 de 9 unidades exploratorias gastadas, 0 de 1 de reserva.

## D-037 - La familia window se cierra por inercia y el orden de ventanas es contraintuitivo

- **Contexto**: P2, segunda familia arquitectonica. Entrena ventana 3 y ventana 7 sobre
  E-CORP-01 manteniendo el resto de la arquitectura fija. El baseline B0 (ventana 5) se
  reevalua como control y **reproduce 0.7818 exacto**.
- **Resultado**:

  | variante | corrected_cv (frozen) | delta vs V03 | IC95 pareado | embedding solo |
  |---|---|---|---|---|
  | FUSION-WS-3 | **0.7830** | +0.0057 | [-0.0021, +0.0136] | **0.6694** |
  | FUSION-B0-200 (ws 5) | 0.7818 | +0.0046 | [-0.0032, +0.0124] | 0.6523 |
  | FUSION-WS-7 | 0.7806 | +0.0034 | [-0.0040, +0.0110] | 0.6498 |

  Dispersion interna: **0.0024**.
- **Decision**: familia `window` **CERRADA por inercia**. No se exploran ventanas 1, 2, 9
  ni 15.
- **Hallazgo que conviene no pasar por alto**: el orden es **3 > 5 > 7**, es decir, cuanto
  **menos** contexto de vecindario, mejor. Es contraintuitivo y encaja con la lectura que
  v5 dejo, con esta lectura: el limite no esta en la funcion objetivo sino en la cantidad de datos. Con
  10M tokens, ampliar la ventana reparte la misma evidencia sobre mas pares y diluye la
  senal de cada palabra. La ventana 3 produce ademas el **mejor embedding aislado de toda
  la fase v5** (0.6694, por encima del 0.6687 de Wikipedia 50M) y la mejor fusion en valor
  absoluto (0.7830).
- **Por que no se adopta pese a ser el mejor numero absoluto**: el IC95 incluye el 0 y la
  dispersion interna (0.0024) es tres veces menor que el umbral de poda. Un +0.0057 cuyo
  intervalo es [-0.0021, +0.0136] no es evidencia de nada; es la octava configuracion de
  v5 y v5.1 que cae en la banda +0.003..+0.006 con el intervalo tocando el 0. Adoptar la
  octava seria exactamente el fallo que D-031 a D-034 evitaron.
- **Advertencia operativa que se anade para P3-P5**: los dos mejores deltas de v5.1
  (100d con +0.0050 y ventana 3 con +0.0057) son **variantes distintas del baseline**, y
  en ambos casos el numero sube por la misma via que en los diez puntos de v5: dentro del
  ruido. Si P3, P4 o P5 producen un delta mayor que +0.0057, hay que mirar su IC95 antes
  que su punto estimate.
- **Presupuesto**: 4 de 9 unidades exploratorias gastadas, 0 de 1 de reserva.

## D-038 - min_count se cierra por inercia, y se descubre por que su mecanismo no puede funcionar con subpalabras

- **Contexto**: P3, tercera familia arquitectonica. Entrena min_count 2 y min_count 5
  manteniendo el resto fijo. El baseline (min_count 3) se reevalua y reproduce 0.7818.
- **Resultado**:

  | min_count | vocabulario del modelo | corrected_cv (frozen) | delta vs V03 | IC95 |
  |---|---|---|---|---|
  | 2 | 139.479 | 0.7771 | **-0.0001** | [-0.0080, +0.0078] |
  | 3 (baseline) | 100.098 | **0.7818** | +0.0046 | [-0.0032, +0.0124] |
  | 5 | 69.310 | 0.7792 | +0.0020 | [-0.0056, +0.0097] |

  Dispersion interna: **0.0047**.
- **Decision**: familia `min_count` **CERRADA por inercia**. No se exploran min_count 1 ni 8.
- **Hallazgo mecanistico, que vale mas que el numero**: los **6.059** tipos del train
  reciben vector en los **tres** puntos de la familia. Bajar `min_count` a 2 agranda el
  vocabulario general de 100.098 a 139.479 tipos, pero **no anade ni un solo tipo de TOXIC
  con vector**, porque con subpalabras 3-6 fastText ya compone vector para cualquier tipo
  consultation. Es decir, la via por la que `min_count` se suponia util (dar vector propio
  a mas palabras de TOXIC deformadas) **esta ya ocupada por las subpalabras**, y por eso la
  familia no puede mover nada. `min_count` solo seria una palanca en un modelo **sin**
  subpalabras, que es exactamente el papel del control Word2Vec.
- **Consecuencia para la lectura de toda la fase**: la configuracion actual ya esta en el
  maximo alcanzable de cobertura *de modelo* dado el corpus, y las tres familias
  arquitectonicas medidas hasta ahora (dimension, ventana, umbral de frecuencia) no mueven
  ni la cobertura ni la aportacion. El limite no es la representacion.
- **Presupuesto**: 6 de 9 unidades exploratorias gastadas, 0 de 1 de reserva.

## D-039 - El intervalo 3-6 ya es el optimo, y la regla de dispersion se corrige para el caso en que el baseline es el maximo

- **Contexto**: P4, cuarta familia arquitectonica y la que toca el **mecanismo** (subpalabras)
  en vez de un parametro de ajuste. Entrena los intervalos 3-5 y 3-7 con `min_n` fijo en 3 y
  bucket fijo en 200.000. El baseline 3-6 se reevalua y reproduce 0.7818.
- **Resultado**:

  | intervalo | corrected_cv (frozen) | delta vs V03 | IC95 | embedding solo |
  |---|---|---|---|---|
  | 3-5 | 0.7791 | +0.0018 | [-0.0053, +0.0092] | 0.6397 |
  | **3-6 (baseline)** | **0.7818** | +0.0046 | [-0.0032, +0.0124] | **0.6523** |
  | 3-7 | 0.7767 | -0.0005 | [-0.0084, +0.0075] | 0.6465 |

  Dispersion interna: **0.0051**.
- **Decision**: familia `subwords` **CERRADA**. No se exploran 2-6, 3-8 ni 4-6.
- **Lectura**: el intervalo 3-6 **ya era el optimo** y la prediccion se cumplio en su parte
  fuerte. Acortar a 3-5 **empeora** (no es "al menos igual de bueno"), porque reconstruye
  menos formas ortograficas, que es justo lo que un corpus de tuits deformados necesita.
  Alargar a 3-7 empeora mas, coherente con que mete mas n-gramas en el mismo bucket de
  200.000 y sube la tasa de colision. El efecto es **monotono**, no ruido.
- **Correccion del criterio de dispersion (regla revisada)**: el umbral de 0.005 esta
  disenado para detectar **indiferencia**, que es lo que ocurria en P1, P2 y P3. Aqui la
  dispersion es 0.0051, apenas por encima, y el baseline es el **maximo** de la familia. En
  ese caso el rango mide cuanto empeora al alejarse del baseline, no cuanto mejora, y un
  rango alto con el baseline arriba es evidencia de que el baseline ya es optimo. Regla a
  partir de aqui: la familia se cierra si ninguna variante es adoptable **y** (el baseline
  es el maximo **o** la dispersion es <= 0.005); se mantiene abierta solo si alguna variante
  supera al baseline sin ser adoptable, que es el unico caso en el que la dispersion mide
  ruido alrededor de una mejora potencial. La maquinaria compartida **no se modifica**
  (es evidencia historica de P1 a P3 con la regla literal); la correccion se aplica en el
  script de P4 sobre el veredicto que devuelve.
- **Presupuesto**: 8 de 9 unidades exploratorias gastadas, 0 de 1 de reserva.

## D-040 - negative sampling se cierra inerte y el presupuesto de v5.1 queda agotado sin tocar la reserva

- **Contexto**: P5, quinta y ultima familia arquitectonica exploratoria. Entrena 5 negativos
  manteniendo todo lo demas fijo. El baseline (10 negativos) se reevalua y reproduce 0.7818.
- **Resultado**:

  | negative | corrected_cv (frozen) | delta vs V03 | IC95 | embedding solo |
  |---|---|---|---|---|
  | 5 | 0.7811 | +0.0039 | [-0.0044, +0.0123] | 0.6664 |
  | **10 (baseline)** | **0.7818** | +0.0046 | [-0.0032, +0.0124] | 0.6523 |

  Dispersion interna: **0.0006**, la mas baja de las cinco familias medidas.
- **Decision**: familia `negative` **CERRADA por inercia**. No se prueban 15 ni 20.
- **Presupuesto cerrado**: **9 de 9** unidades exploratorias consumidas y **0 de 1** de
  reserva. La reserva **no se gasta** porque su funcion era confirmar con una repeticion una
  mejora ya observada, y no hay ninguna: ninguna de las nueve configuraciones nuevas es
  adoptable. Gastar la reserva para abrir una familia nueva seria gastar por obligacion.
- **Cuadro de las cinco familias**:

  | familia | mejor fusion | delta | IC95 | dispersion | veredicto |
  |---|---|---|---|---|---|
  | vector_size | 0.7823 (100d) | +0.0050 | [-0.0028, +0.0130] | 0.0017 | cerrada |
  | window | 0.7830 (ws 3) | +0.0057 | [-0.0021, +0.0136] | 0.0024 | cerrada |
  | min_count | 0.7818 (baseline) | +0.0046 | [-0.0032, +0.0124] | 0.0047 | cerrada |
  | subwords | 0.7818 (baseline) | +0.0046 | [-0.0032, +0.0124] | 0.0051 | cerrada |
  | negative | 0.7818 (baseline) | +0.0046 | [-0.0032, +0.0124] | 0.0006 | cerrada |

- **Conclusion de la fase, en sus dos mitades**:
  1. **Calidad de la representacion**: la arquitectura **si** cambia la calidad del embedding
     aislado (0.6397 a 0.6694 segun el intervalo de subpalabras, +0.030) pero **no** la
     aportacion a V03. Ninguna de las 14 configuraciones medidas supera el gate con un
     intervalo de confianza que lo respalde.
  2. **Aporte a la tarea**: ninguna. V03 se mantiene.

## D-041 - Conclusion de la evaluacion arquitectonica: la representacion mejora, la tarea no

- **Contexto**: P6 consolida las cinco familias medidas en v5.1 (D-036 a D-040). 14
  configuraciones: 9 nuevas y el baseline reevaluado en cada familia como control. El
  control **reproduce v5 exacto en las cinco corridas**, que es lo que hace comparables las
  celdas entre familias.
- **Resultado global**:

  | magnitud | minimo | maximo | rango |
  |---|---|---|---|
  | embedding aislado (media+LR) | 0.6397 | 0.6694 | **0.0297** |
  | fusion con V03 | 0.7767 | 0.7830 | **0.0063** |

  El rango del embedding aislado es **4,7 veces** el de la fusion. Configuraciones adoptables:
  **0**.
- **Conclusion 1, de representacion**: la arquitectura **si** cambia la calidad de la
  representacion, y la forma del optimo es reveladora. La mejor celda es **WS-3**
  (ventana 3) con 0.6694: gana **reduciendo la ventana de contexto**, no acortando el
  intervalo de subpalabra, que se queda en el punto medio 3-6 (3-5 da 0.6397 y 3-7 da
  0.6465, ambos peores). Es decir, la senal que el embedding aporta util es **local y
  ortografica**, no una ventana de distribucion amplia. Los slices lo confirman: el efecto se
  concentra en elongaciones, URLs y hashtags.
- **Conclusion 2, de tarea**: ninguna configuracion aporta informacion adicional a V03 por
  encima del gate. El mejor delta de toda la tabla es **+0.0057** (WS-3) con IC95
  [-0.0021, +0.0136], que incluye el 0. Las 14 celdas se mueven en 0.0063, del orden del
  ruido del protocolo. **V03 se mantiene** con `corrected_cv` 0.777211.
- **Conclusion 3, sobre el corpus**: **no se escala**. La seccion 12 exige cinco condiciones a
  la vez; fallan la 1 (ninguna arquitectura nueva mejora claramente), la 2 (no hay razon para
  esperar que mas escala potencie una aportacion, no una cobertura) y la 3 (la curva de 5M a
  50M no muestra tendencia: +0.0046 y +0.0031, es decir, multiplicar por diez los tokens
  **reduce** el delta). Solo se cumple la 4 (queda la reserva). La decision no escala se
  sostiene en evidencia y no en presupuesto: el techo autorizado de 500M queda sin usar.
- **Variable que queda abierta**: el **tamano de los datos etiquetados** (1.483). No es de
  esta fase y no se explora aqui; se documenta como la unica palanca que la evidencia deja
  en pie.
- **Nota de trazabilidad**: el informe de P0 pasa a llamarse
  `reports/embedding_architecture_auditoria_v5_1.md` para que
  `reports/embedding_architecture_v5_1.md` sea el informe de resultados que pide la spec.
- **Presupuesto**: 9 de 9 unidades exploratorias y 0 de 1 de reserva. El costo real de la
  fase son unos 15 minutos de entrenamiento repartidos en nueve modelos.

## D-042 - Que aporta el embedding, con su poder real: mismo mecanismo, menos ruido, y una correccion de vocabulario

- **Contexto**: P7 responde la pregunta que queda tras D-041: si nada es adoptable, **que**
  hace el embedding y por que no hace mas. Reevalua cuatro configuraciones sin entrenar
  nada (son `DOWNSTREAM EVALUATION UNIT`).
- **salvage / damage** (unidad: filas, 1.483):

  | variante | corrected_cv | salvage | damage | correccion neta |
  |---|---|---|---|---|
  | B0-200 (baseline) | 0.7818 | 8 | 5 | +3 |
  | **WS-3** | **0.7830** | 8 | **2** | **+6** |
  | VS-100 | 0.7823 | 7 | 4 | +3 |
  | NEG-5 | 0.7811 | 8 | 6 | +2 |

- **Lectura correcta**: WS-3 no arregla mas que el baseline (8 frente a 8). Arregla lo mismo y
  **rompe menos** (2 frente a 5). Toda la diferencia de +0,0057 esta en el damage, no en el
  salvage. Eso es lo que distingue una mejora real de un lucky draw: una configuracion que
  solo sube porque acierta mas filas nuevas suele ser la que mas ruido mete, y aqui se ve al
  reves.
- **Interaccion ortografica**: el delta del embedding es **+0,0083** en las 1.289 filas que
  contienen alguna forma de los grupos `elongaciones` o `formas_raras`, y **-0,0045** en las
  194 restantes. La diferencia es +0,0128, la direccion es la esperada (el embedding aporta
  donde la palabra esta deformada y resta donde no lo esta), pero el IC95 [-0,0024, +0,0292]
  **incluye el 0**. No se puede concluir que no haya interaccion: **el test tiene poco
  poder**, porque la mascara cubre el 87 % de las filas (el grupo `formas_raras` del
  vocabulario TOXIC incluye tipos frecuentes del propio lexico de odio, asi que casi cualquier
  fila contiene alguno). Un grupo mas estrecho (`elongaciones`, 57 filas) seria demasiado
  ruidoso. Se registra como direccion consistente en las cuatro variantes, no como resultado.
- **Correccion de vocabulario que hay que hacer**: ``src.evaluation._f1_por_celda`` se llama
  asi y su docstring habla de celdas (fila, etiqueta), pero devuelve la media de las tres
  etiquetas de **cada fila**. La unidad de analisis de toda la fase v5 y v5.1 son las **1.483
  filas**, no las 4.449 celdas. Los conteos publicados en v5 no cambian, porque siempre se
  contaron filas, pero la etiqueta estaba mal. Los scripts de v5 no se editan (son evidencia
  historica); la correccion se aplica en P7 y en la documentacion de v5.1.
- **Conclusion de este P7**: el embedding aporta **la misma informacion que V03, con mas
  ruido**, y la unica palanca para reducir ese ruido (una ventana de contexto menor) reduces
  el damage sin mover el salvage. Nada de eso cambia la conclusion de adopcion de D-041.

## D-043 - Cierre de la fase v5.1: auditoria GO, V03 se mantiene y el limite es el tamano de los datos etiquetados

- **Contexto**: P11 es la auditoria final de la fase. Por la regla de contabilidad de la
  seccion 2 de la spec es una ``AUDIT / REPRODUCTION UNIT`` y **no** cuenta como experimento
  nuevo: por eso no añade fila a `experiments.csv` y su resultado vive en
  `reports/embedding_auditoria_v5_1.json` y en esta decision.
- **Resultado de la auditoria (11/11 comprobaciones OK, veredicto GO)**:

  | comprobacion | resultado |
  |---|---|
  | recursos prohibidos (transformers, BERT, RoBERTa, LLM, preentrenados) | 0 usos |
  | textos de test en los corpus | 0 coincidencias (8-gramas, 9,7 millones) |
  | uso del holdout | 0 (agotado 5/5, intacto) |
  | modelos con hash verificado | 19 de 19, firmas unicas |
  | presupuesto | 9 de 9 exploratorios, 0 de 1 de reserva |
  | V03 | 0.777211, submission de 1.155 lineas intacta |
  | submissions nuevas en la fase | 0 |
  | notebook validado por ejecucion | GO, 32/32 celdas, 0 errores |
  | artefactos y figuras | completos |
  | rama | `proyecto-kaggle-toxic-2026-v5`, **sin push** |

- **Decision de cierre**: la fase v5.1 se **cierra**. El embedding sigue siendo un
  `CANDIDATE` no adoptado y **V03 se mantiene** como modelo final (`corrected_cv` 0.777211,
  LB 0.78306). No se genera ninguna submission nueva ni ninguna sonda de leaderboard.
- **Conclusion de la fase, en sus dos mitades, que es lo que la fase pretendia separar**:
  1. **Representacion**: la arquitectura si cambia la calidad del embedding (rango 0,0297)
     y su forma revela que la señal util es local y ortografica (gana reduciendo la ventana,
     con el intervalo de subpalabras en el punto medio 3-6). Resultado positivo.
  2. **Tarea**: ninguna de las 14 configuraciones aporta a V03 por encima del ruido (mejor
     delta +0,0057 con IC95 que incluye el 0). Resultado negativo.
  Ninguno de los dos anula al otro, y confundirlos fue el error que la fase v5 cometio una
  vez y que v5.1 deja corregido.
- **El limite que la evidencia deja en pie**: el **tamano de los datos etiquetados**
  (1.483). Ni la representacion ni su arquitectura son el techo. Es la unica variable que
  queda abierta y **no** es de esta linea.
- **Integridad del proceso**: 13 commits locales, ninguna rama pusheada, ninguna fusion
  automatica, ningun PR. El trabajo sin commitear de KAGGLE_AGE_2026_PUJ sigue aparcado en
  `git stash` (no es un commit) y sus datos no se tocaron; se restaura al terminar esta
  linea de trabajo, como quedo acordado.

## D-044 - Apertura de la fase v6: gap analysis con verificacion programatica, gate de dos niveles y cierre preventivo de D y E

- **Contexto**: la fase v6 se abre con un objetivo concreto: buscar mecanismos que no se hayan
  evaluado de verdad, sin repetir familias cerradas ni gastar presupuesto en lineas cuya
  evidencia previa ya las refute. El proyecto acumula 168 experimentos bien formados (174
  filas, 6 de ellas con las columnas desplazadas) y 43 decisiones.
- **Decision 1, identificacion de duplicaciones con evidencia programatica**: en vez de
  escribir el gap analysis de memoria, `experiments/v6_00_gap_analysis.py` **busca cada
  mecanismo en el codigo real** con 15 patrones que exigen implementacion (una asignacion o
  una llamada, no una palabra en un comentario) y excluyendo los scripts de la propia fase,
  que se detectarian a si mismos. Resultado: **C2, C3 y C4 estan cerradas** por V4-D
  (interacciones 0.7426, ventanas 0.7734) y por E2-10 (linea ortografica entre -0.028 y
  -0.039). **C queda reducida a C1 (skip-grams)**, que si es genuinamente nuevo.
- **Decision 2, matiz sobre D**: la arquitectura convolucional 1D **ya existe** (DL-02,
  CharCNN), pero sobre **embeddings aleatorios**. La propuesta de v6 es la misma familia de
  convulsion sobre una representacion aprendida, que es un input distinto. D se conserva
  como linea, marcada "arquitectura si, input no".
- **Decision 3, J no es duplicado de PRE**: PRE-00..09 fueron diez transformaciones
  deterministas de superficie, y todas fueron **descartadas**. Eso es *normalizacion* (quitar
  ruido), no *aumento* (crear instancias). El precedente de superficie es adverso y J queda
  para el final, pero no se considera cerrada.
- **Decision 4, criterio de adopcion en dos niveles** (por indicacion del usuario): la
  especificacion proponia `delta >= +0.003` y "4/5 folds **o** complementariedad robusta", que
  es mas laxo que el criterio vigente desde D-036, y D-029 ya registro que un candidato que
  paso los gates fue rechazado por el holdout. Queda: **Gate 1** = `delta >= +0.003` **y**
  `IC95 que excluya el 0` **y** `>= 4/5 pliegues positivos`; **Gate 2** = segunda particion
  agrupada con otra semilla, solo para finalistas; **adopcion** = superar ambos. El holdout no
  se usa: esta agotado (5/5).
- **Decision 5, cierre preventivo de D y E por techo teorico** (P0-bis): la interseccion de
  errores muestra que el mejor candidato existente acierta **9 a 12 filas de 1.483** donde V03
  falla, frente a 1.133 donde nadie acierta. Si esas 12 filas cambiasen una sola etiqueta
  cada una y ninguna se rompiera, la metrica subiria como mucho **+0.0027**, por debajo del
  gate de +0.003. Como combinar dos clasificadores **elige** entre sus aciertos y no los
  suma, **ninguna linea cuyo techo sea recuperar filas ajenas puede superar el gate**. D
  (TextCNN) y E (kNN) se cierran **por presupuesto y no por resultado negativo**, que es una
  distincion importante: no se ejecutan porque su techo esta por debajo del umbral.
- **Decision 6, el track B tiene soporte empirico y no solo teorico**: **TR=1 implica HS=1 de
  forma determinista** en el desarrollo (0 casos de TR=1 con HS=0 frente a 569 con HS=1). No
  es una tendencia, es una invariante del corpus, y un clasificador TR independiente no puede
  expresarla. Ademas P(AG=HS=1) = 0.841 frente a P(AG|HS=0) = 0.0263 con solo 22 ejemplos, lo
  que justifica la forma regularizada B2r: la rama HS=0 no tiene datos para aprenderla.
- **Decision 7, orden de ejecucion**: se sigue el orden de la especificacion (B, A, C, D) con
  C reducida a C1, y las lineas que quedan (E, F, G, H, I, J) se priorizan por lo que la
  evidencia muestre. **I** (ponderacion word/char) sigue ausente en el codigo y es la mas
  barata del proyecto, asi que entra en cuanto B y A esten medidos.
- **Presupuesto**: 24 experimentos principales y 6 de reserva, sin obligacion de consumirlos.

## D-045 - Track B cerrado: la invariante TR implica HS es cierta pero no accionable, y por que el control B0 es obligatorio

- **Contexto**: el diagnostico V6-01 encontro que **TR=1 implica HS=1 de forma determinista**
  (0 casos de TR=1 con HS=0 frente a 569 casos de TR=1). Eso es una invariante del corpus que
  un clasificador TR independiente no puede expresar, y justificaba factorizar
  `p(TR|X) = p(HS|X) * p(TR|HS=1, X)` con el condicional entrenado solo en su dominio natural,
  que es genuinamente distinto de las cadenas (E2-01) y del label powerset (V4-E), ya cerrados.
- **Decision 1, el grupo de control B0 es obligatorio en toda linea de factorizacion**: antes
  de comparar, `v6_02_label_factorization.py` reentrena los tres marginales de V03 con su propio
  codigo. B0 da **0.777211 exacto, delta +0.0000, IC95 [0, 0], 0 salvage y 0 damage**: el
  pipeline reproduce V03 bit a bit. Sin ese control, cualquier delta posterior seria
  inatribuible (podria ser un simple efecto de reentrenar con otra instancia) y la conclusion
  sobre la factorizacion no seria valida. Es la misma razon por la que D-042 exigio corregir la
  granularidad antes de comparar: **primero se demuestra que la vara mide lo que dice medir**.
- **Decision 2, resultado negativo con Gate 1 incumplido en las cinco variantes**: B1 (TR
  condicional) **-0.0107** con IC95 [-0.0184, -0.0033] y 1/5 pliegues; B1c y B3 -0.0115; B2
  (AG en dos ramas, con la rama HS=0 regularizada por tener 22 positivos de 835) -0.0008 con
  IC95 que incluye el 0 y 2/5 pliegues. **Ninguna pasa el Gate 1.** Por la seccion 54 el track
  se **cierra** y no se abre rejilla nueva.
- **Decision 3, el resultado negativo tiene causa mecanica medida, no es un "no se sabe"**: la
  invariante es cierta, pero su magnitud accionable es minuscula. V03 solo predice TR=1 con
  HS=0 en **20 filas de 1.483 (1.35 %)**. La restriccion **si funciona**: la variante
  factorizada predice TR=1 con HS=0 en **0 filas**, que era el objetivo declarado. El problema
  es que el producto no solo prohibe TR=1 con HS=0: tambien **atenua** p(TR) en toda fila con
  p(HS) intermedia, y ahi es donde V03 acertaba. De ahi el saldo de **1 fila rescatada frente a
  35 rotas**. Se registraban 78 filas predichas TR=1 sobre HS=0 real, que son falsos positivos
  garantizados por la invariante, y corregirlos no compensa el coste de reescalar TR en todas
  las filas. **La conclusion es cuantitativa: 1.35 % de incoherencia no pagan ese precio.**
- **Decision 4, la lectura correcta no es "la factorizacion no sirve"**: es "no puede servir
  para esta referencia". Contra un clasificador que violara la invariante en muchas mas filas,
  o con un umbral de TR mas alto que hiciera subir la probabilidad de las filas coherentes, la
  misma linea tendria otro resultado. Queda escrito para que nadie lo lea como refutacion
  universal de la factorizacion probabilistica.
- **Decision 5, los umbrales ajustados no son comparables**: reoptimizar un umbral por etiqueta
  con barrido greedy sobre el mismo OOF que se evalua esta sesgado al alza (B0 sube a 0.7808
  solo por reajustar umbrales, sin cambiar el modelo). **El gate se evalua siempre con los
  umbrales congelados de V03**, y la columna de umbrales ajustados queda solo como
  dato de diagnostico.

## D-046 - Track A cerrado: la ponderacion por grupo no se adopta, y la repeticion es señal, no ruido

- **Contexto**: el gap analysis (V6-00) confirmo por busqueda en el codigo que la ponderacion
  por tamano de grupo nunca se habia probado. Lo probado antes era ponderar por **dureza**
  (V4-B03, -0.0101) y un downweight de frontera (D-027), que son fuentes de varianza
  distintas: la dureza es una propiedad del texto, el tamano de grupo lo es de como se
  construyo el corpus.
- **Decision 1, los pesos se calculan dentro del pliegue**: el grupo es una propiedad de la
  particion, no del texto. Contar los grupos una sola vez sobre todo el desarrollo haria que
  cada pliegue arrastrase informacion de las filas con las que despues se valida, que es
  leakage aunque el peso sea un entero. En `v6_03_ponderacion_grupo.py` los grupos se cuentan
  solo sobre el tren de cada pliegue.
- **Decision 2, el control A0 es obligatorio y se cumple**: con `w = 1` y el mismo codigo, el
  resultado es **0.777211 exacto, delta +0.0000, IC95 [0, 0], 0 salvage y 0 damage**. Es el
  mismo principio que el control B0 de D-045: primero se demuestra que la vara mide lo que
  dice medir.
- **Decision 3, resultado negativo**: A1 `1/sqrt(g)` +0.0006 (IC95 [-0.0029, +0.0040], 3/5
  pliegues, 0 salvage y 2 damage), A2 `1/g` +0.0002 (IC95 que incluye el 0, 2/5), A3
  `min(1/g, 0.25)` **-0.0149** (IC95 que excluye el 0 por el lado negativo, 1/5). **Ninguna
  pasa el Gate 1** y por la seccion 54 el track se cierra sin ampliar rejilla.
- **Decision 4, la estructura del corpus acota el track y conviene dejarlo escrito antes que
  el resultado**: el desarrollo tiene **1415 grupos distintos con maximo 7 filas**, y
  **1378 de 1483 filas (92.9 %) estan en grupos de tamano 1**, que ya tienen peso 1 en los
  cuatro esquemas. Solo el **7.1 %** de las filas puede ver cambiado su peso. La hipotesis era
  plausible, pero su alcance maximo lo fijaba el corpus y no el modelo.
- **Decision 5, el patron monotono descarta la lectura de que la repeticion sea ruido**: los
  tres esquemas se ordenan exactamente segun cuanto penalizan a los grupos repetidos (A1
  +0.0006, A2 +0.0002, A3 -0.0149) y **empeoran al hacerlo**. Como el tope de A3 es mas
  agresivo que `1/g` para todo `g >= 2`, la progresion no es un accidente de los parametros
  sino una relacion monotona. La lectura es que **un tuit repetido es señal y no ruido de
  recoleccion**: que el mismo texto ofensivo aparezca varias veces lo hace mas representativo
  de la distribucion, no menos. Bajar su peso hasta la unidad basica confunde repeticion con
  azar, y por eso A3 se lleva por delante las tres F1 a la vez. Esta conclusion es mas util que
  el propio resultado negativo, porque dice que **no conviene abrir de nuevo esta linea** ni
  en variantes mas suaves.
- **Decision 6, A3 empeora con IC95 que excluye el 0**: no es ruido estadistico, es un
  empeoramiento medible. Se descarta sin ambiguedad.

## D-047 - Track I cerrado: el optimo de la escala word/char es interior pero su techo es la mitad del gate

- **Contexto**: el gap analysis (V6-00) confirmo por busqueda en el codigo que **no existe
  ningun `char_weight` ni `word_weight` en el proyecto**. Lo medido antes fue el valor de cada
  bloque por separado y su union (V2-BASE-M1, M4, M2/M3), es decir, **dos bloques y su union
  sin escalar**, nunca con una escala explicita dentro de la union. Era la linea mas barata que
  quedaba en v6.
- **Decision 1, por que el peso heredado no es una decision del proyecto**: `hstack` da a los
  dos bloques el mismo peso, pero ese peso no lo fija el criterio del proyecto sino la
  normalizacion L2 de cada `TfidfVectorizer`: cada bloque se normaliza por su propia norma, de
  modo que la escala relativa depende del numero de caracteristicas activas por documento. La
  pregunta era si el optimo estaba en otro punto.
- **Decision 2, el control I0 es obligatorio y se cumple**: con `char_weight = 1.0` la
  multiplicacion es la identidad y el resultado es **0.777211 exacto, delta +0.0000, IC95
  [0, 0], 0 salvage y 0 damage**. Tercer control de la fase (tras B0 de D-045 y A0 de D-046)
  con la misma funcion: sin el, los otros tres puntos no serian comparables.
- **Decision 3, resultado negativo con optimo interior**: `char_weight` 0.5 da -0.0117, 1.5 da
  **+0.0016** (IC95 [-0.0048, +0.0078], 3/5 pliegues) y 2.0 da -0.0047. **Ninguna pasa el Gate
  1.** El perfil es creciente de 0.5 a 1.5 y decreciente en 2.0, luego **el optimo es
  interior y el peso heredado de `hstack` no era optimo**: el balance entre evidencia lexica y
  sublexica deberia inclinarse algo hacia el bloque de caracteres. Esa es la respuesta a la
  pregunta del track.
- **Decision 4, no se afina alrededor de 1.5 pese al optimo interior**: el mejor delta es
  +0.0016, **menos de la mitad del gate (+0.003)**, su IC95 **incluye el 0** y solo mejora
  3/5 pliegues. Afinar costaria unos 25 segundos, pero con el techo estimado la conclusion
  no cambia: **la escala no era el cuello de botella**. Se deja el numero escrito para que la
  decision sea revisable, y no se gasta presupuesto en una linea cuyo techo es la mitad del
  umbral de adopcion.

## D-048 - Track C1 cerrado: los skip-grams son neutros, y el control destino un bug de tokenizacion

- **Contexto**: el gap analysis (V6-00) confirmo que no hay ninguna implementacion de
  skip-grams en el proyecto: el `ngram_range=(1, 2)` de V03 solo produce unigramas y bigramas
  **contiguos**. V4-D habia cerrado "interacciones como tokens" y E2-10 "forma x token", pero
  como **canales anadidos** sobre una representacion ya construida, con sus propios pesos.
- **Decision 1, por que no es repeticion de V4-D**: aqui se modifica el **analizador del
  bloque de palabras**, de modo que las relaciones con hueco **compiten con las palabras por
  el mismo presupuesto de regularizacion**. No se puede "anadir sin dao": el termino tiene que
  merecer su peso. Esa es la diferencia que hace la prueba informativa en vez de redundante.
- **Decision 2, el control destino un bug real y por eso estaba puesto**: la primera version
  del tokenizador propio emitia **solo unigramas**, y el control dio 0.7554 en lugar de
  0.777211, con 1407 caracteristicas en lugar de las miles de V03. Los deltas de esa version
  comparaban "V03" contra "V03 sin bigramas" y no eran atribuibles, asi que se descartaron
  enteros. Corregido para emitir tambien los bigramas contiguos, unidos con espacio igual que
  `TfidfVectorizer` por defecto: el control da **0.777211 exacto, delta +0.0000, IC95 [0, 0],
  0 salvage, 0 damage, 2744 caracteristicas**. Cuarto control de la fase con la misma funcion
  que B0, A0 e I0. **Ninguno de los cuatro ha fallado desde que existen**, y este habria
  pasado desapercibido sin el.
- **Decision 3, resultado: los skip-grams son NEUTROS, no dañinos**: hueco 1 da -0.0022
  (IC95 [-0.0084, +0.0039], 2/5) y huecos 1 y 2 da -0.0001 (IC95 [-0.0085, +0.0082], 2/5),
  con el bloque de palabras pasando de 2744 a 5025 caracteristicas, un **+83 %**. Anadir
  dimensionalidad no cuesta nada y no aporta nada. Es un resultado **distinto del de V4-D**,
  donde las interacciones como tokens costaron -0.0346.
- **Decision 4, lo que neutraliza la hipotesis es la premisa, no la intuicion**: que la
  relacion con hueco ni anade senal aprovechable ni resta capacidad **confirma que la
  intuicion de partida era razonable** (insulto, luego risa, luego marcador, con palabras en
  medio) y que no era una repeticion de V4-D. Lo que falla es la premisa de que quede senal
  por representar en 1.483 filas. El intercambio por etiqueta lo confirma: con un hueco HS
  sube de 0.7617 a 0.7648 mientras TR baja de 0.8157 a 0.8103. No es una ganancia, es un
  reparto.
- **Decision 5, C queda cerrada por completo**: C2, C3 y C4 estaban cerradas por
  duplicacion (V4-D y E2-10) y C1 se cierra aqui por resultado negativo. **La sub-linea C
  queda sin nada abierto** tras haber sido la propuesta central de la especificacion de v6.

## D-049 - Track H cerrado: el bagging no se refuta, el protocolo de evaluacion no lo deja medir

- **Contexto**: el gap analysis (V6-00) confirmo que no existe ningun bagging en el proyecto.
  Los ensembles previos (ENS-01 a ENS-04) mezclaban representaciones distintas, que es otra
  pregunta; aqui se promedian **V03-equivalentes** sobre remuestreos de grupos. El track A
  (D-046) habia avisado de un techo: con 92.9 % de las filas en grupos de tamano 1,
  **remuestrar por grupos y por filas son practicamente lo mismo**.
- **Decision 1, el techo se mide en vez de suponerlo**: por eso el script no solo informa un
  delta, sino que **mide la diversidad de los cinco modelos que promedia**: correlacion media
  entre pares, correlacion minima y diferencia media absoluta de probabilidad. Sin esa
  medicion un delta negativo seria un misterio; con ella, es una consecuencia medida.
- **Decision 2, dos bugs corregidos durante la ejecucion, ambos reales**: (a) el
  remuestreo usaba `np.isin`, que incluye las filas de un grupo **una sola vez** aunque el
  grupo saliera dos veces, de modo que lo que se llamaba bootstrap era en realidad una
  submuestra sin reemplazo; ahora la multiplicidad se respeta con `np.repeat` sobre los
  bloques de cada grupo. (b) La agrupacion de los modelos por modulo del indice emparejaba
  modelos de **pliegues distintos**, que se validan sobre filas distintas, y producia un
  error de forma; ahora se indexan por el indice real del pliegue.
- **Decision 3, resultado**: H0 (control, un modelo con el tren completo) reproduce V03
  **exacto: 0.777211, delta +0.0000, IC95 [0, 0]**. Quinto control de la fase con la misma
  funcion. H1 (promedio de 5) da **-0.0105** con IC95 [-0.0183, -0.0030] que **excluye el 0**,
  **0/5 pliegues mejoran**, 1 fila rescatada frente a 11 rotas.
- **Decision 4, el resultado NO es por falta de diversidad, y esto es lo importante**: la
  correlacion media entre pares es **0.9573** (minima 0.9379) y la diferencia media absoluta
  es 0.0606. Hay **diversidad real**: los cinco modelos no son el mismo. La hipotesis que el
  track se probo a si mismo (que el techo de la estructura de grupos haria que el bagging no
  tuviera nada que promediar) queda **refutada por la medicion**, y a pesar de eso el
  promedio empeora. Esa distincion importa: el track no fallo por la razon que anticipaba.
- **Decision 5, el cierre es de protocolo, no de mecanismo**: la explicacion coherente con
  los numeros es un **efecto de escala**: promediar cinco modelos contrae las probabilidades
  hacia el centro, y los umbrales congelados de V03 estan calibrados para **un** modelo
  individual. Para evaluar el bagging con justicia habria que reoptimizar los umbrales para
  el promedio, pero eso **rompe la comparabilidad con V03** y reproduce el sesgo ya
  documentado en D-045 (reoptimizar sobre el mismo OOF que se evalua dio a B0 un 0.7808
  frente a 0.7772 sin cambiar el modelo). **No hay forma limpia de medir esta linea con el
  protocolo actual.** Se cierra con esa precision y se deja constancia de que podria
  reabrirse si se dispusiera de una particion de calibracion separada del OOF de
  evaluacion.

## D-050 - Track F cerrado: la deteccion de sospecha encuentra dificultad, no ruido

- **Contexto**: el gap analysis (V6-00) confirmo que el ruido nunca se midio a nivel de par.
  V4-A lo habia medido por **duplicado exacto** y concluyo que era escaso: 2 clusters, 84
  tuits, 4.85 % del corpus, con V03 acertando bien dentro (error 0.046 frente a 0.218 fuera).
  Ese 0.046 se media por duplicado exacto, y la pregunta abierta era si hay ruido que **no**
  es duplicado, es decir pares similares pero no identicos con etiqueta opuesta.
- **Decision 1, detectar y ponderar van en rondas separadas por una razon concreta**: si el
  entrenamiento se ponderara por una sospecha calculada con el mismo modelo que despues se
  corrige, la sospecha se definiria a si misma y el resultado no significaria nada. La
  deteccion usa el OOF **congelado** de V03, cuya prediccion de cada fila se hizo sin esa
  fila, y solo despues se reentrena.
- **Decision 2, control F0 y resultado**: F0 reproduce V03 **exacto: 0.777211, delta +0.0000,
  IC95 [0, 0]**. Sexto control de la fase. F1 (cuatro senales, lambda 0.5) da **+0.0001**
  con IC95 que incluye el 0 y 2/5 pliegues; F2 (solo la senal estructural, lambda 1.0) da
  -0.0010; F3 (las cuatro, lambda 1.0) da **-0.0041**. **Ninguna pasa el Gate 1**.
- **Decision 3, la hipotesis del informe se refuto y se corrige**: la version escrita antes
  de ejecutar afirmaba que V03 ya clasificaba bien las filas marcadas, de modo que bajar su
  peso no podria aportar nada. **Los datos dicen lo contrario**: V03 acierta **10.5 %** en las
  449 filas sospechosas frente a **28.3 %** en las 1034 no sospechosas, mas del doble. La
  deteccion **si** identifica filas donde el modelo falla.
- **Decision 4, pero identificar el fallo no la convierte en senal de ruido, y esa distincion
  es la que decide el track**: caben dos lecturas, y el comportamiento de `lambda` las
  separa. Si la deteccion encontrara etiquetado erroneo, subir `lambda` deberia mejorar; como
  empeora, encuentra **dificultad genuina en la frontera**. Bajar el peso de esas filas
  **aleja al modelo de donde estan los errores**. Concluido: **lo que la deteccion marca es
  donde hay que estar, no donde hay que limpiar.**
- **Decision 5, la senal tiene valor como marcador de dificultad aunque no como correccion**:
  es un producto secundario aprovechable para el analisis de errores, y no para limpiar el
  entrenamiento. Se deja escrito para que el trabajo de deteccion no se descarte entero: sirve
  para otra cosa.
- **Decision 6, el ruido por par es mas escaso que el ruido por duplicado**: la senal
  `estructural`, que generaliza el conflicto de V4-A a la similitud, marca **12 pares de
  4449 (0.27 %)**, frente a los 84 tuits del duplicado exacto. **Al ampliar la definicion de
  conflicto el ruido se hace mas escaso, no mas abundante.** El precedente de V4-A era ya
  adverso y este resultado lo refuerza.

## D-051 - Track J cerrado: el aumento empeora y doblar las copias no empeora mas, luego el problema es el sesgo y no la cantidad

- **Contexto**: el gap analysis (V6-00) distinguio que **J no es una repeticion de PRE-00 a
  PRE-09**: esas diez variantes eran *normalizacion* (sustituyen el texto) y el aumento crea
  filas nuevas aplicando la variacion **solo al tren**. El precedente de superficie era
  adverso, con un mejor de 0.7623 frente a 0.7587.
- **Decision 1, las transformaciones preservan la etiqueta por construccion**: `espaciado`
  (separa las letras de una palabra), `elongacion` (alarga la ultima letra) y `puntuacion`
  (duplica un signo). **No** se sustituyen palabras por sinonimos, **no** se reordena, **no**
  se toca negacion ni objetivo. Cualquier transformacion que toque el significado es una
  fuente de etiquetas falsas, y el proyecto tiene una restriccion de seguridad al respecto
  desde v2.
- **Decision 2, resultado**: J0 (control) reproduce V03 **exacto: 0.777211, delta +0.0000,
  IC95 [0, 0]**. Septimo control de la fase. J1 (una copia, 1.186 -> 2.372 filas de train)
  da **-0.0061** con IC95 [-0.0141, +0.0018] y 1/5 pliegues. J2 (dos copias, hasta 3.558
  filas) da **-0.0063** con IC95 [-0.0145, +0.0016] y 1/5. **Ninguna pasa el Gate 1.**
- **Decision 3, el dato que va mas alla del signo**: doblar las copias de una a dos **casi no
  cambia nada** (-0.0061 frente a -0.0063, con el IC95 estrechandose). Si el problema fuera la
  cantidad de datos sinteticos, duplicarla deberia empeorar de forma apreciable. Un efecto
  praticamente constante al cambiar el tamano apunta a un **sesguase de las copias y no a
  un exceso de ellas**: las variantes deformadas forman un patron coherente que el modelo
  aprende, y da igual que haya una o dos copias de ese patron.
- **Decision 4, este cierre abarca mas que el track**: cierra a la vez el aumento y la
  superficie, porque el mecanismo de superficie era el unico que quedaba sin explorar tras
  las diez normalizaciones. Un resultado negativo aqui es un cierre mas solido, no mas debil.
- **Decision 5, v6 se queda sin lineas ejecutables**: **B, A, C, I, H, F y J** estan cerradas
  por resultado negativo y **D y E** lo estaban por techo teorico (D-044 decision 5). **G** se
  cera "muy baja: no ejecutar" desde el gap analysis, con el score y su margen ya presentes
  en chains y en el meta desde E2-01 y E2-18. **No queda ninguna linea de la especificacion
  de v6 sin ejecutar**, lo que convierte el cierre de la fase en un cierre de verdad y no en
  una pausa.

## D-052 - Cierre de la fase v6: ninguna adopcion, y por que el presupuesto sin gastar es un resultado

- **Contexto**: la fase v6 se abrio (D-044) para buscar mecanismos que el proyecto no hubiera
  evaluado de verdad, con un presupuesto autorizado de 24 experimentos principales y 6 de
  reserva, y un criterio de adopcion en dos niveles.
- **Decision 1, la fase se cierra sin ninguna adopcion**: de los 14 tracks y sub-tracks de
  la especificacion, **7 se midieron (B, A, I, C1, H, F, J) y ninguno supero el Gate 1**; 2
  quedaron cerrados **sin gastar una unidad** (D y E, por techo teorico); 1 se marco como no
  ejecutable (G); y C2, C3 y C4 ya estaban cerrados por duplicacion. **No queda ninguna
  linea ejecutable.**
- **Decision 2, se gastaron 7 de 24 experimentos y eso no es un incumplimiento**: el
  presupuesto es un techo autorizado, no una obligacion. Dos decisiones de D-044
  redujeron el gasto antes de empezar (cerrar D y E por techo, y cerrar C2, C3 y C4 por
  duplicacion), y los tracks ejecutados se cerraron uno a uno por resultado negativo. **Gastar
  por gastar seria exactamente el error que D-044 intenta evitar**, porque las
  multinacionales que la fase compro (tracks D, E, C2, C3, C4) ya eran 0 unidades.
- **Decision 3, los 7 controles de reproduccion son el resultado mas importante**: B0, A0, I0,
  C1-0, H0, F0 y J0 dieron **0.777211 exacto** cada uno. La fase se ejecuto con una red de
  seguridad que hacia imposible publicar un delta sin atribuible, y **la red funciono**: el
  control C1-0 detecto que su tokenizador emitia solo unigramas y daba 0.7554, lo que
  invalido los deltas de esa primera version antes de publicarlos. Sin ese control, un
  resultado falso se habria registrado como evidencia. Es la razon por la que todo track
  nuevo de este proyecto debe llevar control.
- **Decision 4, los entregables se generan, no se escriben**: `v6_09_entregables.py` lee los
  JSON y CSV que los experimentos ya escribieron y **falla si falta alguno**, en vez de
  rellenar huecos con ceros. Un entregable que rellena lo que no encuentra es peor que uno
  que no existe, porque parece completo.
- **Decision 5, cinco conclusiones sobreviven al cierre**, y son el resultado que hace que
  la fase valiera la pena: (a) **un tuit repetido es senal, no ruido de recoleccion**, y no
  conviene reabrir el track A ni en variantes mas suaves; (b) **la interseccion de errores
  da un techo de +0.0027** para cualquier linea cuya unica via sea recuperar filas ajenas a
  V03, y combinar modelos **elige** entre sus aciertos en vez de sumarlos; (c) **la deteccion
  de sospecha sirve para medir dificultad, no para limpiar entrenamiento**; (d) **la escala
  entre word y char no era el cuello de botella**, aunque su optimo este en 1.5; (e) **no hay
  forma limpia de medir un promedio de modelos con umbrales calibrados para uno**, y queda
  planteada la condicion que lo resolveria: una particion de calibracion separada del OOF de
  evaluacion.
- **Decision 6, V03 sigue congelada y la rama sigue sin push**: `corrected_cv` 0.777211, LB
  0.78306, **0 submissions y 0 sondas de LB en toda la fase**. El push sigue pendiente de
  autorizacion expresa de Carlos.

## D-053 - Apertura de la fase v7 y cierre del track P9: aproximar mejor el objetivo no es mejor

### Apertura de la fase

- **Contexto**: la fase v6 se cerro sin adopciones (D-052) y dejo trece restricciones
  asumidas. La especificacion de v7 propone once tracks. El usuario fija el objetivo de superar
  0.80 tanto en leaderboard como en metrica local, y delega la decision sobre el gate.
- **Decision 1, la fase v7 se abre en un worktree aparte**, en `/home/carlos/Git/PLN-v7`, con
  la rama `proyecto-kaggle-toxic-2026-v7` creada desde `proyecto-kaggle-toxic-2026-v6`. Motivo:
  el arbol principal estaba en otra rama con trabajo sin versionar del proyecto
  KAGGLE_AGE, y tocarlo habria arriesgado ese trabajo. `data/` (11 GB) se enlaza a la copia
  unica y `features/` queda en el propio worktree. v6 no se borra ni se reescribe.
- **Decision 2, se repara un defecto real de v6**: `.gitignore` ignoraba `tools/` entero, de
  modo que `verificar_caracteres.py` se anunciaba en el commit de v6 pero **nunca llego al
  repositorio**. El patron pasa a ser `tools/*` con la excepcion `!tools/verificar_*.py`,
  porque git no vuelve a incluir un fichero cuyo directorio padre esta excluido. La puerta de
  verificacion de caracteres queda versionada y da 0 errores.
- **Decision 3, la auditoria de brecha cierra ocho tracks y deja dos abiertos**. Con
  verificacion programatica y lectura de cifras: P0.2 y P0.5 (0.774954), P0.4 (+0.001831),
  P2.1 (10 filas de interseccion), P2.3 (0.761203), P3 (0.741036), P4 (-0.023836), P5
  (-0.250519), P6 y P7 (-0.002292). **Quedan P8 y P9 como mecanismos genuinamente nuevos**,
  mas P0.1 y P0.3 que se ejecutan con reservas expresas. El orden deduced es **P9, P8,
  P0.1, P0.3**, no el orden literal de la especificacion, porque los cuatro primeros tracks de
  aquel orden ya estaban medidos.
- **Decision 4, el freno de transferencia** que el usuario delego decidir. Se adopta:
  `delta >= +0.003` con IC95 que excluye el 0 y 4/5 pliegues, **mas** la condicion de que por
  encima de `+0.005` un candidato solo llega a la mesa de finales si no aumenta la dimension
  ni la brecha entre ajuste y validacion. La justificacion no es teorica: R1-C1, R2-MINDF5 y
  OPT-01 tuvieron los tres mejor OOF del proyecto (entre +0.015 y +0.025) y los tres
  perdieron entre 0.007 y 0.013 de leaderboard, y los tres aumentaban capacidad. V03 no la
  aumenta. La regla es la que incumplieron los tres fracasos y que V03 cumple.

### Cierre del track P9

- **Decision 5, el control es BCE con el mismo codigo y no los 0.777211 de V03**, porque un
  modelo lineal ajustado de forma iterativa no es identico a la solucion dual exacta de
  liblinear. Lo que se exige es que el control se acerque a la referencia, y ademas se
  informa del hueco: **todo delta se mide contra el control**, que es la unica manera de que
  sea atribuible a la perdida y no al cambio de optimizador.
- **Decision 6, resultado**: el control BCE da **0.777165 frente a 0.777211**, un hueco de
  **-0.000046**, dentro de la tolerancia declarada de 0.006. Focal da **-0.0024** e IC95 que
  incluye el 0 con 2/5 pliegues; F1 suave da **-0.0033** con 3/5. **Ninguna pasa el gate** y
  el track se cierra.
- **Decision 7, el resultado negativo tiene lectura mecanica y no es un "no se sabe"**: las
  dos perdidas orientadas quedan ligeramente por debajo y **la mas directa de las dos, la F1
  suave, es la peor**. Optimizar el objetivo literalmente con 1.186 filas acopla el descenso a
  la composicion de esa muestra y sobreajusta, que es exactamente lo que le paso a OPT-01,
  donde buscar la metrica fuera de pliegue de forma directa dio el mejor OOF del proyecto y el
  peor leaderboard. La conclusion que sobrevive al cierre es **que en este problema aproximar
  mejor el objetivo no es mejor**: la entropia binaria ponderada es un campo de Kraft suave y
  bien condicionado, y con esta cantidad de filas esa propiedad vale mas que la fidelidad al
  objetivo.
- **Decision 8, dos bugs reales detectados antes de que contaminaran el resultado**:
  (a) el script predicia sobre la matriz de **entrenamiento** y guardaba esas probabilidades
  como si fueran de validacion, lo que habria hecho el experimento entero sin sentido; lo
  detecto el verificador de codigo variable no usada. (b) Los pesos se guardaban como vector
  cuando el producto matricial disperso de torch exige una matriz. Ninguno llego a ningun
  resultado publicado.

## D-054 - Track P8 cerrado: la repeticion es senal, pero no es senal nueva, y esa distincion no la podia hacer v6

- **Contexto**: la auditoria (V7-00) dejo P8 como la ultima senal nueva. La distincion con
  la fase v6 era mecanica: alli se **reponderon** filas por tamano de grupo, aqui el grupo se
  convierte en una **caracteristica** que aporta la distribucion de etiquetas de las otras
  observaciones del mismo grupo, informacion que el texto no puede dar porque requiere
  etiquetas de vecinos.
- **Decision 1, el techo se mide antes de entrenar, y esa decision es la que hace el track
  barato**: el diagnostico se ejecuta antes de ajustar ningun modelo y ya acota la linea.
  Resultado: **solo 105 de 1483 filas (7.08 %) tienen grupo repetido**, y de los pares de
  grupo y etiqueta, **80 son uniformes frente a 31 contradictorios**.
- **Decision 2, diseno de la caracteristica para que no tenga valores ausentes**: la variable
  es la **desviacion de la prior del grupo respecto a la prior global** del pliegue. Si el
  grupo tiene una sola fila, la desviacion es exactamente cero: neutral, sin imputacion y
  sin columna muerta. Para una fila de entrenamiento el prior se calcula **dejando fuera la
  propia fila**; para una de validacion se usan las filas de su grupo que esten en el
  entrenamiento, que por construccion no incluyen su etiqueta. Una cuarta columna con el
  soporte distingue "sin evidencia" de "evidencia nula".
- **Decision 3, control**: P8-0 reproduce V03 **exacto: 0.777211, hueco +0.000000**, con
  tolerancia declarada de 0.0005. Octavo control de las fases v6 y v7.
- **Decision 4, resultado**: P8-1 **-0.0011** e IC95 que incluye el 0 con 1/5 pliegues;
  P8-2 **-0.0017** con 1/5. **Ninguna pasa el gate.**
- **Decision 5, la conclusion que sobrevive al cierre es una distincion que v6 no podia
  hacer**. La fase v6 concluyo que "un tuit repetido es senal, no ruido de recoleccion", y es
  cierto. Aqui se separa un paso mas: **ser senal y ser senal nueva son dos cosas
  distintas**. La repeticion es senal que **V03 ya captura**, porque dos filas del mismo grupo
  son casi el mismo texto, el modelo lineal ya les da la misma representacion y ya las
  predice igual. Por eso el prior, que es casi determinista (80 de 111 pares son
  uniformes), **no aporta nada que el texto no tenga ya**.
- **Decision 6, tres hechos que acotan la linea y que se pueden medir sin entrenar**:
  (a) solo el 7.08 % de las filas tiene la informacion que la caracteristica puede aportar;
  (b) donde existe, es casi siempre redundante; (c) **el error de V03 no se concentra en esas
  filas**, porque su acierto es identico con y sin evidencia (22.9 % en ambos casos). Y el
  dato que remata el diagnostico: **la caracteristica no arregla ni una sola fila de las que
  tiene evidencia**. No estorba: es inerte exactamente donde se aplica.
- **Decision 7, una afirmacion del informe fue refutada por los datos y se corrigio**: la
  primera version de `v7_02_prior_grupo.py` afirmaba que V03 ya acertaba mejor en las filas
  con evidencia de grupo. Al ejecutarse, el acierto resulto **identico** (22.9 % frente a
  22.9 %). Se corrigio el texto. Es la segunda vez en la fase que una afirmacion previa
  escrita sin datos queda desmentida al medirlos, y por eso los informes de este proyecto se
  generan desde los scripts y no al reves.
- **Decision 8, dos bugs corregidos antes de contaminar ningun resultado**: un desajuste de
  indice global al pasar el array de grupos a la funcion de columnas, que rompia el
  emparejamiento, y una division entre cero cuando el grupo de una fila de entrenamiento
  tenia una sola fila en el entrenamiento, que habria metido un valor no finito en el modelo.

## D-055 - P0 cerrado: los umbrales estan debilmente identificados, y por eso no hay nada que recalibrar

- **Contexto**: la fase v6 dejo abierta la medicion del sesgo de umbrales, porque reajustar
  umbrales sobre el mismo conjunto fuera de pliegue que se evalua infla la metrica en
  +0.0036 sin cambiar el modelo. E2-16 ya habia medido un umbral global anidado en
  0.776613, pero con un protocolo mas laxo: reutilizaba las probabilidades fuera de pliegue
  de los otros cuatro pliegues como conjunto de calibracion.
- **Decision 1, el protocolo tiene tres niveles que no se mezclan**: el modelo se ajusta
  solo en el ajuste interior, el umbral se ajusta solo en la calibracion interior, y se
  aplica una sola vez sobre la validacion exterior. La particion interna respeta los grupos y
  estratifica por la **combinacion** de las tres etiquetas, que es mas informativa que
  cualquier etiqueta suelta en un problema multilabel.
- **Decision 2, hay que separar si el codigo esta bien de cuanto cuesta el protocolo, y son
  dos efectos distintos**. Por eso hay tres brazos:
  - **P0-0** ajusta sobre todo el entrenamiento exterior con umbrales congelados. Es el
    brazo que valida el codigo y da **0.777211 exacto, delta +0.000000**.
  - **P0-1-control** usa el pipeline de tres niveles con umbrales congelados, de modo que
    ajusta sobre cinco sextos de las filas. Da 0.769123, es decir **-0.0081**, y ese delta
    **no es un error: es el coste del protocolo**.
  - **P0-1-anidado** comparte conjunto de ajuste con el control, de modo que el coste se
    cancela y el delta **-0.0019** con IC95 [-0.0092, +0.0057] y 2/5 pliegues mide el
    **efecto del umbral**. Comparar contra V03 seria mezclar dos cosas.
- **Decision 3, el propio gate detecto un error de diseno, y la correccion tambien importa**.
  La primera version fijaba una tolerancia de 0.002 sobre el control de tres niveles y se
  detuvo al superarla. **La deteccion era correcta y la tolerancia estaba mal planteada**: el
  error no era del codigo sino del protocolo, que necesariamente reduce el conjunto de
  ajuste. Anadir el brazo P0-0 separo las dos cosas. Es exactamente el mismo error de
  interpretacion que la fase v6 cometio al tratar la reoptimizacion de umbrales como una
  mejora, y por eso el gate de §32 debe ir **acompañado de una reflexion sobre que mide**.
- **Decision 4, el resultado principal es el informe de dispersion, que era lo que v6 dejo
  pendiente**. Quince umbrales, cinco pliegues por tres etiquetas:
  - HS: media 0.4120, mediana 0.4100, desviacion 0.0402, rango intercuartil 0.0500, rango 0.1000.
  - TR: media 0.4900, mediana 0.4500, desviacion 0.0652, rango intercuartil 0.0500, rango 0.1500.
  - AG: media 0.4560, mediana 0.4400, desviacion 0.0586, rango intercuartil 0.0800, rango 0.1400.
- **Decision 5, conclusion: los umbrales estan DEBILMENTE IDENTIFICADOS**. Con desviaciones
  de 0.04 a 0.065 sobre una rejilla de 0.20 a 0.80, el umbral de TR puede moverse de 0.37 a
  0.60 sin que el dato lo impida. Los tres umbrales congelados de V03 **caen dentro del rango
  por pliegue**, asi que **no son claramente sobreajuste**, pero las medianas calibradas son
  sistematicamente **mas bajas** que ellos, y el anidado estricto termina **peor** que los
  congelados. Las dos cosas juntas dicen que la eleccion de umbral es casi arbitraria y que
  V03 eligio un punto razonable.
- **Decision 6, eso explica los dos hechos medidos antes**: por que E2-16 dio -0.0006 y por que
  reajustar umbrales sobre el mismo conjunto infla +0.0036. **No hay informacion en los datos
  que identifique el umbral.** Cualquier diferencia que se le atribuya a "afinar umbrales" es
  en buena parte ruido de una decision que los datos no determinan.
- **Decision 7, P0.3 y P10 se cierran sin ejecutar, y por evidencia y no por presupuesto**.
  Se ejecutaban solo si P0.1 mostraba margen, y la dispersion es alta. Pero la dispersion alta
  **no implica upside**: un estimador robusto daria un valor mas cerca de la mediana, que
  esta **mas lejos** de los umbrales congelados que la media, y el anidado, que es la
  estimacion mas informada, ya es peor. Cerrarlas es una conclusion, no un recorte.
- **Decision 8, con esto se cierra la capa de decision**, que era la unica via por la que el
  proyecto habia movido alguna vez el leaderboard. V03 mantiene sus tres umbrales congelados
  **por evidencia medida y no por inercia**: se ha medido cuanta dispersion tienen, se ha
  medido que recalibrarlos no mejora, y se ha medido que el protocolo que permite medirlo
  cuesta -0.0081 en exactitud por reducir el conjunto de ajuste.

## D-056 - Cierre de la fase v7: sin adopciones, con dos mecanismos nuevos refutados y una medicion que v6 dejo pendiente

- **Contexto**: el objetivo declarado es superar 0.80 tanto en leaderboard como en metrica
  local. La fase ejecuto tres tracks (P9, P8 y P0.1) mas dos scripts de analisis, con un
  presupuesto autorizado de 25 mas 10 experimentos.
- **Decision 1, la auditoria de brecha cerro ocho de los once tracks candidatos con su
  cifra**, y eso es el resultado que mas presupuesto ahorro: P0.2 y P0.5 (0.774954), P0.4
  (+0.001831), P2.1 (10 filas de interseccion), P2.3 (0.761203), P3 (0.741036), P4
  (-0.023836), P5 (-0.250519), P6 y P7 (-0.002292). **El orden que se ejecuto no fue el de
  la especificacion sino el que la auditoria impuso**, y fue correcto porque los cuatro
  primeros tracks de aquel orden ya estaban medidos.
- **Decision 2, dos mecanismos genuinamente nuevos se refutan con su causa medida**:
  - **P9, perdida orientada a la F1**: -0.0022 frente a un control BCE que da 0.777165
    contra los 0.777211 de V03, o sea un hueco de 0.000046 que hace atribuible el negativo a
    la perdida. La **F1 suave, que es el sustituto literal del objetivo, fue la peor de las
    tres**. Con 1.186 filas, optimizar el objetivo directamente acopla el descenso a la
    composicion de esa muestra. Conclusion que sobrevive: **aproximar mejor el objetivo no es
    mejor**.
  - **P8, prior de etiqueta del grupo**: -0.0011 frente a un control exacto. El techo medido
    antes de entrenar ya lo acotaba: solo el 7.08 % de las filas tiene evidencia de grupo, de
    los pares grupo-etiqueta 80 son uniformes frente a 31 contradictorios, y el acierto de V03
    es **identico** con y sin evidencia. Conclusion que sobrevive: **ser senal y ser senal
    nueva son dos cosas distintas**, y la repeticion es senal que V03 ya captura.
- **Decision 3, P0.1 resuelve la medicion que v6 dejo pendiente y cierra la capa de
  decision**. El producto es el informe de dispersion de los quince umbrales: desviaciones
  de 0.0402 (HS), 0.0652 (TR) y 0.0586 (AG), con rangos de 0.10 a 0.15 sobre una rejilla de
  0.20 a 0.80. **Los umbrales estan debilmente identificados.** Los congelados de V03 caen
  dentro del rango por pliegue, asi que no son sobreajuste, pero las medianas calibradas son
  sistematicamente mas bajas y el anidado estricto termina **peor** (-0.0019 frente a su
  control). Eso explica los dos hechos medidos antes: por que E2-16 dio -0.0006 y por que
  reajustar umbrales sobre el mismo conjunto infla +0.0036. **No hay informacion en los datos
  que identifique el umbral.** V03 mantiene los suyos **por evidencia medida y no por
  inercia**, que es la diferencia entre un congelado justificado y uno que nadie revisó.
- **Decision 4, hay que separar si el codigo esta bien de cuanto cuesta el protocolo, y son
  dos efectos distintos**. Por eso P0.1 tiene tres brazos: P0-0 valida el codigo y da
  0.777211 exacto; P0-1-control tiene -0.0081, que **no es un error** sino el coste de
  ajustar sobre cinco sextos de las filas; y P0-1-anidado comparte conjunto de ajuste con el
  control, de modo que el coste se cancela y su -0.0019 mide el efecto del umbral. El propio
  gate de la seccion 32 detecto que la tolerancia estaba mal planteada, y esa es exactamente
  la clase de error que la fase v6 cometio al tratar la reoptimizacion de umbrales como una
  mejora: **por eso un gate debe ir acompanado de una reflexion sobre que mide**.
- **Decision 5, la correccion de expectativas sobre 0.80 queda con tres medidas, no con
  opiniones**: (a) los tres mejores modelos locales de la historia del proyecto
  (R1-C1, R2-MINDF5 y OPT-01, con entre +0.015 y +0.025) perdieron entre 0.007 y 0.013 de
  leaderboard, de modo que una mejora local de +0.02 **no es evidencia de mejora externa**;
  (b) los tres aumentaban capacidad, de donde sale el freno de transferencia adoptado en
  D-053; (c) el unico movimiento positivo del leaderboard vino de la capa de decision, y esta
  fase acaba de medir cuanto cabe ahi, que es poco y con alta dispersion.
- **Decision 6, el cumplimiento se comprueba por recuento, no por declaracion**. Los nueve
  recuentos dan cero infracciones. Dos fallos de la propia auditoria se corrigieron en el
  proceso: (a) se contaba a si misma, porque contiene los nombres de las prohibiciones que
  busca, el mismo problema de auto-referencia que resolvio la auditoria de brecha con el
  filtro por prefijo; (b) el patron del holdout marcaba como infraccion el uso legitimo de
  `holdout_v2`, que devuelve el desarrollo en el primer valor y el holdout en el segundo, y la
  fase solo usa el primero. El patron se redujo a la infraccion real y **se verifico con tres
  casos de prueba, uno de ellos el uso legitimo**, porque un patron de auditoria que no se
  prueba no es una garantia.
- **Decision 7, que queda abierto**: **P1** en su formulacion de solo prediccion, que promedia
  los cinco modelos al predecir el test y **no se puede medir en local**, y la verificacion de
  transferencia de un finalista en el leaderboard, que es el unico lugar donde queda por
  medir si una mejora local se transfiere. Todo lo demas esta cerrado con cifra.

## D-057 - Apertura de la fase v8: rama y worktree aislados, prohibicion de modelos preentrenados ratificada y freno de transferencia heredado

- **Contexto**: la fase v7 cerro todas sus lineas salvo una, el promedio de los cinco modelos
  de pliegue en el momento de la prediccion. Su hipotesis es genuina pero su comportamiento
  externo no lo captura el protocolo local, de modo que la fase v8 nace con una sola sonda
  externa autorizada. A la vez, el material de clase sobre clasificacion de texto y
  atribucion de autoria aporta tecnicas que el proyecto nunca ha probado.
- **Decision 1, aislamiento del trabajo**. La fase v8 vive en el worktree
  `/home/carlos/Git/PLN-v8`, en la rama `proyecto-kaggle-toxic-2026-v8`, creada desde el
  cierre de v7 (`2f8fac8`). El arbol principal ya estaba en `kaggle-age-2026-puj-v13`, de
  modo que el worktree aparte protege tambien el trabajo del otro proyecto. El directorio
  `data/` queda enlazado al del arbol principal y `tools/` copiado, con el verificador de
  caracteres dando 0 errores. **v7 no se borra ni se modifica.**
- **Decision 2, se mantiene la prohibicion de modelos preentrenados aunque las reglas de la
  competencia los permitan**. Las reglas de la competencia del material de clase admiten
  transformers, BERT, RoBERTa, spacy, nltk y modelos preentrenados. El proyecto **no los
  usa** desde la fase v5. Es una decision consciente y no un olvido: el marco de trabajo es
  el clasico, y por eso las tecnicas del material se explotan **dentro** de ese marco. Queda
  escrito para que nadie lo lea despues como una restriccion pendiente de levantar.
- **Decision 3, el freno de transferencia de la fase v7 se hereda sin cambios**. Por encima
  de +0.005 de mejora local, un candidato solo llega a la mesa de finales si **no** aumenta
  la dimension ni la brecha entre ajuste y validacion. El fundamento son tres mediciones
  independientes: R1-C1 gano +0.0157 local y perdio 0.0108 externo, R2-MINDF5 gano +0.0221 y
  perdio 0.0127, y OPT-01 gano +0.0240 y perdio 0.0070. Los tres perdiaron en el externo y los
  tres aumentaban capacidad. El unico movimiento positivo de leaderboard vino de la capa de
  decision, que no aumenta capacidad.
- **Decision 4, la auditoria de brecha pasa a ser programatica y con evidencia**. El script
  `v8_00_gap_audit.py` busca en el codigo real, exige **implementacion** y no mencion, y
  excluye los scripts de la propia fase para no autoinferirse. Resultado: de las diez pistas
  experimentales, cinco no se han probado y cinco tienen precedente; del material de clase,
  veintidos tecnicas de las cuales **nueve son genuinamente nuevas** y trece cerradas. Tres
  falsos positivos del propio inventario quedaron corregidos: la asimetria casaba con la
  prosa de un informe anterior, la traduccion con el metodo de conversion de jerga, y las
  idiosincrasias con un comentario sobre cobertura de lexico. El conteo simple resulto
  genuinamente usado en la linea ortografica de E2-10 y su cierre se corrigio para citarlo.
- **Decision 5, dos lineas de la especificacion llegan con precedente que hay que decir antes
  de ejecutarlas**. **P2, la ponderacion por razon logaritmica de recuentos, es el estimador
  de Mosteller y Wallace de 1964**, el metodo fundacional de la atribucion de autoria, y el
  proyecto ya la midio en cuatro variantes dentro de E2-15, cerradas entre -0.016 y -0.039. No
  es una idea nueva con otro nombre: es una reejecucion. **P3, la estabilidad de features, si
  tiene respaldo doble**: es el capitulo 8 de ese mismo libro, que el material de clase resume
  como palabras inestables y que se mide traduciendo documentos de ida y vuelta a cinco
  idiomas. La traduccion no esta disponible por las restricciones, pero la pregunta es la
  misma y la operacionalizacion por submuestreo de la especificacion es legitima.
- **Decision 6, orden de ejecucion que impone la auditoria, no la especificacion**. Primero P1
  con su unica sonda externa, porque de su resultado depende toda la bifurcacion. Si no mejora,
  se cierra sin variantes y se entra por P3, que es la forma barata de reducir la dimension
  sin perder senal, y despues P2 con el precedente ya medido. Si mejora, se congela la
  arquitectura y se ejecutan como maximo dos lineas mas, P9 y P10.

## D-058 - P1: identidad de pipeline verificada celda a celda, alcance del cambio pequeno y parada obligatoria antes del envio

- **Contexto**: P1 promedia las cinco probabilidades de los cinco modelos de pliegue para
  predecir la prueba, con la representacion, los hiperparametros y los umbrales congelados de
  V03. Es la unica linea de la fase cuyo resultado no se puede medir en local.
- **Decision 1, el control se hace contra el artefacto, no solo contra la cifra**. El
  promedio por pliegue reconstruido coincide con el OOF canonico de V03 con una diferencia
  maxima por celda de **0.000e+00**, y la metrica es 0.777211 exacto. Eso es una
  comprobacion de identidad de entrenamiento mas fuerte que cualquier tolerancia: si las
  probabilidades coinciden celda a celda, el pipeline **es** el de V03 y ninguna cifra
  posterior puede deberse al codigo.
- **Decision 2, se documenta el error de control que casi cuela y su causa**. La primera
  version del script tomaba como control el modelo completo ajustado sobre todo el desarrollo
  y medido sobre si mismo, y devolvio 0.969085 en vez de 0.777211. La causa es que esa cifra
  es **resubstitucion**, no validacion: un modelo evaluado sobre las mismas filas con las que
  se ajusto tiene el numero inflado por construccion. La referencia de V03 es el OOF, es decir
  el promedio por pliegue. El script ahora lo dice en el informe y etiqueta esa cifra como no
  comparable, porque el error no era del codigo sino de la lectura de que valida el codigo.
- **Decision 3, la metrica local de esta linea no es informativa y el script lo dice**. La
  columna de desarrollo es el promedio por pliegue, que **es** la definicion del OOF canonico,
  asi que tiene que coincidir con V03 y coincide. Intentar medir la hipotesis en local seria
  medir el OOF contra si mismo. Por eso la parte local se usa solo para validar el codigo y la
  unica cifra nueva es la del promedio sobre la prueba.
- **Decision 4, el alcance del cambio es pequeno, y por tanto tambien su techo**. Tres
  medidas se alinean: la contraccion media de las probabilidades al promediar es de
  **+0.001334**, las tasas de positivos sobre la prueba se mueven en la tercera cifra decimal
  (HS de 0.4428 a 0.4376, TR de 0.2842 a 0.2834, AG de 0.3605 a 0.3553), y la submission
  difiere de la de V03 en **92 filas de 1154 y 100 celdas de 3462, un 2.9 %**. Como
  referencia, el unico movimiento positivo de leaderboard del proyecto, de v02 a v03, cambio
  126 celdas y dio +0.0031. **La prediccion razonable es que el ensemble se mueva en el orden
  de milésimas, y ese es el techo que hay que tener presente al leer el numero.** Se escribe
  antes del envio y no despues, porque un techo declarado despues del resultado no sirve.
- **Decision 5, la dispersion entre pliegues es apreciable y conviene registrarla**. Las
  metricas en su propio pliegue van de 0.7386 (pliegue 4) a 0.7960 (pliegue 1), un rango de
  0.057. El promedio da el mismo peso a un pliegue mas débil, lo que es correcto en media pero
  significa que la reduccion de varianza no es uniforme entre pliegues.
- **Decision 6, parada obligatoria antes del envio**. El fichero
  `submissions/toxic_submission_v08_p1_fold_ensemble.csv` esta escrito y pasa las nueve
  comprobaciones de integridad, y **no se envia automaticamente**. **No se crean variantes de
  umbral, de pesos ni de semilla antes de conocer el numero externo.** El resultado decide la
  bifurcacion: si mejora de forma material, se congela la arquitectura y se ejecutan como
  maximo P9 y P10; si no mejora, P1 se cierra sin variantes y se entra por P3.

## D-059 - Resultado de P1 y adopcion de un filtro de detectabilidad: el limite esta en el leaderboard, no en el modelo

- **Contexto**: P1 era la unica sonda externa autorizada en la fase. El techo de la linea
  estaba declarado antes de enviarla, en el orden de milesimas, y el control de codigo habia
  dado identidad de pipeline con una diferencia maxima por celda de 0.000e+00. La pregunta que
  queda es que hacer con el numero, y si ese numero significa algo.
- **Decision 1, P1 se cierra como linea negativa, sin variantes**. El externo dio **0.77470**
  frente a los 0.78306 de V03, una diferencia de **-0.00836**. No se reajusta umbral, no se
  alteran pesos y no se cambia la semilla. La causa mecanica estaba medida y era esperable: el
  promedio contrae las probabilidades en +0.001334 y los umbrales congelados de V03 estan
  calibrados para un modelo individual, de modo que el efecto de escala movio 100 celdas de
  3462.
- **Decision 2, el numero no es evidencia de nada, y eso hay que decirlo con claridad**. El
  error estandar del leaderboard con 1154 filas es 0.01524 para la F1, y la sensibilidad de la
  metrica primaria es de 7.93e-04 por acierto neto, de modo que **el ruido equivale a 19.2
  aciertos netos**. La diferencia de P1 son -10.5 netos, es decir un doceavo del ruido, y
  -10.5 por cada cien celdas que cambian: indistinguible de una moneda al aire. **No hay
  evidencia de que el ensemble sea peor, hay evidencia de que no es mejor.** V03 sigue como
  referencia **por no tener evidencia en contra, no por tener evidencia a favor**, y esa
  distincion queda escrita porque cambia lo que cabe afirmar.
- **Decision 3, se revisa una premisa propia, y es lo mas importante de esta decision**. El
  proyecto venia sosteniendo, con el freno de transferencia heredado de la fase v7, que **el
  unico movimiento positivo del leaderboard vino de la capa de decision**, el paso de v02 a
  v03, que dio +0.0031. Al pasar esa cifra a la misma unidad que el ruido: **son +3.9
  aciertos netos frente a un ruido de 19.2, o sea una senal cinco veces mas pequena que el
  ruido que la rodea. Ese movimiento positivo tambien fue ruido.** No es evidencia de que la
  capa de decision funcione. El freno de transferencia sigue siendo valido, pero su
  justificacion se corrige: no era que la capa de decision tuviera un historial positivo, era
  que el historial entero era indistinguible.
- **Decision 4, ninguna de las tres maniobras exploradas produce senal por encima del
  ruido**. De once sondas reales, **diez caen dentro de la banda y cero estan por encima de
  V03**. La unica distinguible es v01, con -33.2 netos, que es el modelo de solo palabra con
  constante 1.0, o sea el mas pobre de la familia, y se aleja para peor. La afirmacion
  defendible es estrecha y se escribe asi: **de once maniobras, la unica con suficiente senal
  es la que mas se aleja del mejor, y en la direccion equivocada**. Ni la capa de decision
  (apilamiento, fusion tardia, Optuna, promedio de pliegues), ni los hiperparametros
  (constante, minimo de apariciones), ni la representacion producen una diferencia que este
  leaderboard pueda resolver.
- **Decision 5, se adopta un filtro de detectabilidad previo, y es lo que gobierna la fase a
  partir de ahora**. El limite esta en el instrumento y es cuantificable. Si una candidata
  cambia C celdas de la referencia, su senal en el externo es `C por (2p - 1)`, donde p es la
  fraccion de esas celdas que ella acierta y la referencia falla, y el externo solo la resuelve
  si `2p - 1` supera 19.2 dividido por C. En tabla: con 50 celdas hace falta 0.692, con 100
  hace falta 0.596, con 200 hace falta 0.548, con 400 hace falta 0.524 y con 800 hace falta
  0.512. **Ninguna candidata cuya tasa de acierto entre las celdas modificadas alcance el
  minimo de la tabla se envia.** No es que sea mala: es que el instrumento no puede
  distinguirla, y gastar una sonda en eso es gastar presupuesto en ruido. La ventaja de este
  filtro es que **es estimable en el desarrollo antes de enviar nada**, de modo que es previo
  y gratis.
- **Decision 6, la reordenacion que se deriva**. P5, P6 y P7 bajan de prioridad, porque las
  tres mueven la capa de decision en la banda de cien a doscientas celdas, donde el minimo de
  acierto exigido supera el 60 %. **P3, la estabilidad de features, sube a la primera
  posicion**, porque reducir la dimension puede cambiar muchas celdas de golpe, que es la
  unica forma de acercarse al minimo exigido. P2, la ponderacion por razon logaritmica de
  recuentos, sigue con su precedente adverso ya medido en E2-15.
- **Decision 7, el objetivo declarado queda anotado con su probabilidad, no con su deseo**.
  Superar 0.80 esta a +0.01694 de la referencia, o sea **1.1 veces el ruido y 21 aciertos
  netos de ventaja limpia, sobre once sondas de las que diez son indistinguibles entre si**.
  No es imposible, pero no es algo que se alcance empujando la capa de decision, que es donde
  se ha estado buscando durante varias fases. La honestidad con el objetivo importa mas que
  el objetivo.

## D-060 - P3 cerrado: la estabilidad de features es el mecanismo correcto para el problema del que viene y el equivocado para este

- **Contexto**: P3 era la primera posicion de la reordenacion que imponen el filtro de
  detectabilidad de D-059, porque reducir la dimension puede cambiar muchas celdas de golpe.
  La idea es el capitulo 8 de Mosteller y Wallace, que el material de clase resume como
  palabras inestables y que el profesor midio traduciendo documentos de ida y vuelta a cinco
  idiomas. La traduccion no esta disponible, asi que la operacionalizacion es por submuestreo
  de las filas de entrenamiento.
- **Decision 1, el protocolo queda sin fuga y eso se construye asi, no por descuido**.
  El vectorizador se ajusta **una vez por pliegue sobre su tren completo**, de
  modo que el espacio de caracteristicas es fijo y solo varian las filas de cada submuestra.
  Las filas de validacion se **transforman** con ese mismo vectorizador y nunca se vuelven a
  ajustar. La funcion que mide la estabilidad recibe la matriz del tren y las etiquetas del
  tren, y no admite ningun indice exterior, de modo que no puede ver la validacion ni por
  error. En los cinco pliegues el protocolo cubre las 1483 filas exactamente una vez, sin
  solape y sin indices fuera de rango.
- **Decision 2, tres umbrales de estabilidad, los tres negativos y monotonos**.
  - Umbral 0,5: 1636 caracteristicas de 20.441, metrica 0,773904, diferencia **-0,003308**,
    IC95 [-0,01218, +0,00558], 1 de 5 pliegues.
  - Umbral 0,7: 927 caracteristicas, 0,772583, diferencia **-0,004628**, IC95 [-0,01533,
    +0,00631], 1 de 5 pliegues.
  - Umbral 0,9: 455 caracteristicas, 0,758321, diferencia **-0,018890**, IC95 [-0,03154,
    -0,00610], 0 de 5 pliegues.
  Cuanto mas agresiva es la seleccion, peor. La tendencia no es ruido: es monotona.
- **Decision 3, el filtro de detectabilidad falla en los tres, y con holgura**. La tasa de
  acierto entre las celdas modificadas es de 0,4402, 0,4436 y 0,3930, frente a minimos
  exigidos de 0,5522, 0,5361 y 0,5257. **Ninguna se envia**, y por tanto **no se gasta ninguna
  sonda**. El filtro creado en V8-02 acaba de demostrar su valor en su primer uso: sin el,
  esta linea habria ocupado una de las dos o tres sondas de la fase para devolver un numero
  sin informacion.
- **Decision 4, el hallazgo importante es como pierde, no que pierde**. En las celdas que
  toca, la linea acierta del 39 % al 44 %, es decir **por debajo del 50 % que daria una moneda
  al aire**. La seleccion por estabilidad mueve la decision de forma sistematica hacia peor.
  La explicacion es coherente con lo que hace el metodo: un rasgo que mantiene peso alto bajo
  siete submuestras es un rasgo de magnitud grande y muy repetida, no necesariamente uno que
  separa las clases, y es mas probable que sea frecuente y neutro que discriminante y
  especifico. Al conservar solo esos rasgos se pierde la senal minoritaria que separa el
  insulto de la toxicidad.
- **Decision 5, la conclusion que sobrevive a la fase, y que reencuadra la idea de origen**.
  El capitulo 8 de Mosteller y Wallace mide estabilidad **para atribuir autoria**, donde un
  rasgo frecuente y neutro si es informativo, porque el objetivo es la **forma** del texto y no
  su pertenencia a una clase. **Traducir ese criterio a un problema de clasificacion no
  conserva su sentido.** El mecanismo es el correcto para el problema del que viene y el
  equivocado para este. La idea era buena, la fuente es buena, y la traduccion al problema es
  lo que falla: conviene separar las tres cosas, porque descartar la idea sin el matiz
  habria cerrado una puerta que en otro problema si funciona.
- **Decision 6, dos afirmaciones que no se contradicen y que hay que no mezclar**. El IC95
  del umbral 0,9 **excluye cero**: la perdida es detectable en el desarrollo. Y a la vez el
  filtro dice que no se veria en el externo. No es contradiccion: la primera afirmacion es
  sobre la metrica de validacion agrupada, con mil veces mas filas; la segunda es sobre si el
  leaderboard de 1154 filas puede distinguir un movimiento de esa magnitud.
- **Decision 7, dos errores de implementacion se detectaron antes de obtener cifras, y ambos
  los detecto el propio control del script**. El primero: un mapa de indices locales
  construido sobre el entrenamiento que se consultaba con las filas de validacion, que por
  definicion no estan ahi; ademas la prediccion de la validacion no puede salir de una matriz
  ajustada solo en el entrenamiento, hay que **transformar** esas filas. El segundo: un conteo
  de caracteristicas que recibia indices globales del entrenamiento completo donde esperaba
  las filas del desarrollo. Ninguno llego a producir una cifra, y ambos los atrapo el RuntimeError
  de seguridad que el propio script escribe, lo que confirma que un control de fuga explicito
  vale mas que un comentario que lo diga.
- **Decision 8, la ordenacion que fijaba P3 como primera posicion queda agotada**. P3 se
  cierra sin variantes y **sin reintento con mas agresividad ni con otra constante de
  submuestreo**, porque la tendencia es monotona y va en la direccion contraria. La fase pasa
  a **P2, la ponderacion por razon logaritmica de recuentos**, que conserva su precedente
  adverso ya medido dentro de E2-15, cerrado entre -0.016 y -0.039, y que es el estimador de
  Mosteller y Wallace de 1964. P5, P6 y P7 siguen con la prioridad baja que les asigno el
  filtro de detectabilidad.
