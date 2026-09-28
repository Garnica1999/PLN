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
- **Conclusion final**: formulacion multi-etiqueta adoptada; el post-proceso jerarquico se midio en las Fases 4-5 y se descarto (empeora -0.0056); el modelo final (V03) no lo aplica.

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
- **Conclusion final**: se mantuvo sin excepciones: spaCy se uso solo como anotador simbolico (POS/dependencias) y las features no incorporaron vectores preentrenados; la auditoria forense lo verifico (P4).

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
- **Conclusion final**: superada por D-015/D-016: agrupar solo por near-duplicado daba una estimacion optimista (0.7927 frente a 0.7838 al anadir pseudo-autor); el protocolo vigente usa grupos fusionados.

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
- **Conclusion final**: caracterizacion confirmada con los datos finales (519/519 TR=>HS; 26 casos AG=1 con HS=0) y usada como base de la jerarquia y de los canales TR/AG.

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
- **Conclusion final**: cumplido: no se uso ningun recurso externo etiquetado ni lexicones externos (pysentimiento/VADER-es descartados por prohibicion); el unico corpus externo fue no etiquetado y solo para embeddings (D-030).

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
- **Conclusion final**: protocolo aplicado en la fase 11 (preflight rocm-smi y limite de uso); el resto del proyecto es CPU; sin incidentes.

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
- **Conclusion final**: verificado (train sin bytes no ASCII; test con 6 bytes cp1252); las variantes de normalizacion se documentaron en v2 y no se intento reconstruir diacriticos.

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
- **Conclusion final**: cadenas cerradas sin mejora (0/5); el meta de dependencia no se adopto como modelo final (D-019 superada por D-024); los canales derivados quedan disponibles para metas restringidos. de (a) integrar los canales de P2-P5 y
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

## D-061 - P2 cerrado: una premisa propia era falsa, y el suavizado sostiene el metodo entero

- **Contexto**: P2 era la razon logaritmica de recuentos, el estimador de Mosteller y Wallace
  de 1964, la siguiente linea de la ordenacion. La auditoria de V8-00 la senalaba con
  precedente adverso ya medido: E2-15 evaluo cuatro variantes cerradas entre -0.016 y -0.039.
- **Decision 1, una premisa propia era falsa y se corrige antes de ejecutar nada**. La
  auditoria de V8-00 afirmaba que el componente que el proyecto no habia probado era **la razon
  logaritmica por etiqueta**. Al leer el codigo de E2-15 se comprueba que su bucle itera sobre
  las tres etiquetas y calcula un ratio independiente para cada una. **Ya estaba hecho.** Lo
  genuinamente no probado era otro: multiplicar el ratio sobre la **representacion TF-IDF de
  V03** y con la **constante 2.0**, porque E2-15 uso conteos y la constante 1.0. La correccion
  importa porque cambiar la pregunta es la unica forma de que la linea aporte algo.
- **Decision 2, el suavizado sostiene el metodo entero, y se mide por que**. Con suavizado
  1e-6 el ratio sobre la TF-IDF da **-0,058496**; con 1e-3 la misma linea da **+0,004745**. Un
  salto de **+0,063241 por un factor de mil en un solo parametro**. El diagnostico lo explica:
  con suavizado pequeno el ratio alcanza un rango de mas de veinte unidades y una desviacion
  de mas de tres, porque en la TF-IDF las caracteristicas raras ya tienen valores pequenos y
  dividir por una probabilidad de clase casi nula **explota**. Multiplicar la matriz por eso
  destruye la escala relativa de las caracteristicas. El suavizado no es un detalle numerico:
  es la parte que hace que el metodo no destruya la representacion.
- **Decision 3, ninguna variante es admisible, y el filtro falla por poco pero falla**.
  - `nb_tfidf` con suavizado 1e-6: -0,058496, IC95 [-0,07611, -0,04061], 0 de 5 pliegues,
    tasa de acierto entre celdas 0,3266, muy por debajo del azar.
  - `nb_tfidf_suavizado` con 1e-3: +0,004745, IC95 [-0,00708, +0,01714], 3 de 5 pliegues,
    tasa 0,5281 frente a un minimo de 0,5317. **Falla por tres milesimas.**
  - `nb_counts_c2`, el control: -0,075027, IC95 [-0,09423, -0,05561], 0 de 5 pliegues,
    tasa 0,3217.
  Que la variante suavizada quede a tres milesimas de superar el filtro **no cambia nada**: el
  filtro se supera o no se supera. Y ademas su IC95 incluye cero y solo gana 3 de 5 pliegues.
- **Decision 4, no se barre el suavizado, y el motivo es el que importa**. Que una sola linea
  pase de -0,0585 a +0,0047 cambiando un parametro que nadie ha caracterizado significa que
  **el resultado positivo local no es una senal: es un punto de una superficie que no se ha
  explorado**. Barrer el suavizado seria el rescate por hiperparametro que las reglas del
  proyecto prohiben desde la fase v6, y V8-02 acaba de medir que casi nada de lo que se ajusta
  llega al externo. **No se gasta sonda.**
- **Decision 5, el control aporta algo que las cuatro variantes de E2-15 no permitian
  separar**. El control canonico con la constante 2.0 da -0,075027, frente a los -0,0160 que
  E2-15 midio con la constante 1.0. **Subir la constante de 1.0 a 2.0 con el ratio empeora la
  linea casi cinco veces.** La conclusion es que la constante y el ratio **no se pueden subir
  a la vez**: el ratio ya introduce mucho ruido en la escala y subir la constante lo amplifica.
  Ese control no existia en E2-15, y sin el ninguna diferencia observada podria atribuirse a la
  representacion y no a la constante.
- **Decision 6, el mecanismo actua, y eso tambien es un dato**. El ratio cambia 303 celdas
  en el primer pliegue, un 8 %, al compararlo con el mismo modelo sin ponderar. No es un
  multiplicador inerte. La pregunta no era si el mecanismo hace algo, sino si lo hace para
  bien, y la respuesta es que no.
- **Decision 7, como queda la ordenacion**. P2 cerrado. Quedan P4, P5, P6, P7 y las lineas de
  reserva. **P5, P6 y P7 conservan la prioridad baja** que les asigno el filtro de
  detectabilidad, porque las tres mueven la capa de decision en la banda de cien a doscientas
  celdas. P4, la regularizacion conjunta entre etiquetas, es la unica que queda con mecanismo
  propio, pero su techo lo acotan tres predecesores fallidos: cadenas, powerset y
  apilamiento. Con el filtro de detectabilidad delante, ninguna de ellas pasa por defecto, y
  por tanto la fase se acerca al punto en que hay que decidir si seguir buscando o cerrar la fase.

## D-062 - P4 cerrado: el mecanismo de compartir estructura funciona y empeora, y eso unifica P3 y P4 en una sola conclusion

- **Contexto**: P4 era la unica de las lineas restantes que toca el modelo y no la capa de
  decision, y por eso la unica que podia cambiar muchas celdas de golpe. Anade a las tres
  cabeceras de V03 un termino que penaliza su dispersion alrededor de su media, que es
  regularizacionconjunta blanda: no dice que sean iguales, solo que alejarse cuesta.
- **Decision 1, la comparacion de la linea es interna y no contra el OOF canonico**. Con el
  peso del termino nuevo en cero, esta implementacion es **una reimplementacion de V03**,
  porque usa un optimizador propio: la biblioteca no admite varias salidas binarias con una
  penalizacion compartida. Medir contra el OOF canonico mezclaria el efecto de la
  regularizacion conjunta con el de haber cambiado de optimizador, y no se podrian separar.
  El OOF canonico sirve para una sola cosa, que es comprobar que la reimplementacion reproduce
  la cifra conocida.
- **Decision 2, el control interno reproduce V03 y la identidad se confirma por el metodo
  correcto**. El control da 0,775652 frente a los 0,777211 de V03, diferencia -0,001559, con
  tolerancia declarada de 0,004, mas ancha que en otras lineas porque aqui se compara una
  reimplementacion contra el original. La identidad se comprueba con el **valor del objetivo**:
  el propio es 2,928125 y el de la biblioteca 3,359072, y con la regularizacion cuadrada la
  superficie es estrictamente convexa, de modo que la solucion propia es la optima. Los cosenos
  con la biblioteca son 0,9994, 0,9870 y 0,9944, y la diferencia se explica por la convergencia
  laxa del resolvedor, no por un error. **El criterio de identidad cambio por esto**: exigir el
  coseno es pedir que las dos implementaciones convergieran igual de bien, y el valor del
  objetivo no lo pide.
- **Decision 3, el resultado es negativo y monotono, y el mecanismo actua**. Peso 0,5:
  -0,001471, IC95 [-0,00503, +0,00234], 0 de 5 pliegues. Peso 2,0: -0,006781, IC95
  [-0,01242, -0,00096], 0 de 5 pliegues. Al tiempo, el coseno medio entre las tres cabeceras
  sube de **0,7380 a 0,8265 a 0,9216**. **El mecanismo hace exactamente lo que dice y por eso
  empeora.** Parejar las tres cabeceras es lo que haria un modelo mas simple, y V03 ya ocupa
  ese sitio con la constante 2,0 sobre la representacion completa. Compartir estructura solo
  tendria sentido si el techo no estuviera en la representacion, y V8-02 lo situa ahi.
- **Decision 4, el hallazgo que unifica P3 y P4, y que vale mas que las dos cifras**. La tasa
  de acierto entre las celdas modificadas es de **0,4186 y 0,3925**, por debajo del 0,50 que
  daria una moneda al aire. En P3 fue de 0,39 a 0,44. En P1 fue de 0,46 sobre cien celdas.
  **Tres lineas independientes, tres mecanismos distintos, el mismo sintoma: en las filas que
  tocan, todas mueven la decision hacia peor de forma sistematica.** Reducir la dimension,
  compartir estructura y promediar modelos degradan las mismas filas. Eso apunta a que el
  limite no es de mecanismo sino de representacion, y a que cualquier cambio, por pequeno que
  sea, actua sobre las filas de decision limite como ruido con sesgo negativo.
- **Decision 5, cinco errores de escala, todos antes de obtener cifra, y como se detectaron
  cada uno**. Un factor dos sobrante en el gradiente de la penalizacion, detectado comparando
  contra diferencias finitas con un error maximo de 1,6 frente a 2,99e-9 sin el factor. Un
  fallo de dimensiones en la transposicion del gradiente, detectado por el propio error. Una
  normalizacion de la perdida por el numero de filas efectivas de las tres etiquetas en vez de
  por el numero de filas. Un factor en la escala de la penalizacion que sobre-regularizaba
  **tres veces y media**. Y un recorte de probabilidad en la perdida. Los dos ultimos se
  detectaron comparando la **norma de los pesos** con la biblioteca, que era 0,06 frente a 23,3:
  una comparacion de norma detecta un error de escala cuando la metrica sola no lo hacia, y
  por eso se anade al repertorio.
- **Decision 6, P4 cerrado sin variantes y sin submission**. No se reintenta con mas pesos
  porque la tendencia es monotona y en la direccion contraria, y porque el filtro de
  detectabilidad habria bloqueado el envio de todos modos.
- **Decision 7, lo que queda y por que la decision no es tecnica**. Quedan P5, P6 y P7, las
  tres solo de capa de decision y con la prioridad baja que les asigno el filtro de
  detectabilidad de D-059. **A la luz del patron de la Decision 4 no hay razon para esperar
  que ninguna pase**, porque las tres se mueven en la banda de cien a doscientas celdas y
  todas las lineas medidas en esa banda han dado tasa de acierto por debajo del azar. Cerrar
  la fase o ejecutarlas es una decision del usuario y no tecnica, y conviene tomarla con el
  dato encima: tres lineas medidas, tres con el mismo sintoma.

## D-063 - Cambio de presupuesto autorizado: se gastan sondas para experimentally llegar a 0,80 e incluso a 0,81

- **Contexto**: al cierre de P4 quedaban P5, P6 y P7 sin ejecutar, y la recomendacion tecnica
  era no ejecutarlas porque el filtro de detectabilidad las bloquea por construccion: las tres
  mueven la capa de decision en la banda de cien a doscientas celdas, donde hace falta mas
  del 60 % de acierto, y las tres lineas ya medidas en esa banda dieron tasa por debajo del
  azar. El objetivo declarado es superar 0.80 en el externo, y el analisis de V8-02 lo situa
  a 1,1 veces el ruido del leaderboard y a 21 aciertos netos de ventaja limpia.
- **Decision 1, el usuario autoriza expresamente gastar el envio o los envios adicionales que
  hagan falta, con el objetivo de llegar a 0.80 a toda costa e incluso superar 0.81**. Eso
  cambia la regla de presupuesto que la especificacion de la fase fijaba en dos o tres
  submissions para toda la fase. **A partir de aqui el filtro de detectabilidad deja de ser
  un bloqueante y pasa a ser un criterio de priorizacion y una prediccion que se puede
  contrastar**: sigue midiendo lo que mide, pero una linea que no lo supere ya no queda
  descartada por defecto, sino que es una apuesta informada.
- **Decision 2, el objetivo sube a 0,80 e incluye la posibilidad de 0,81, y se anota su
  probabilidad con el numero a la vista**. Con la banda observada de 0.770 a 0.783 en once
  sondas, llegar a 0.80 exige +0.017, que es 1,1 veces el ruido; llegar a 0.81 exige +0.027,
  que son 1,8 veces el ruido. **Ninguna de las dos es imposible, y ninguna es esperable por
  el camino que se ha seguido hasta ahora.** Se ejecuta porque el usuario lo decide y porque
  un objetivo alto que se persigue produce lineas que un objetivo bajo no produce. Queda
  escrito el numero para que, si llega, se sepa cuanto de lo que funciona se debe al cambio de
  representacion y cuanto a haber gastado sondas.
- **Decision 3, se ejecutan P5, P6 y P7, y despues el resto de la especificacion y las
  tecnicas nuevas del material de clase**. Quedan P5, P6, P7, P8, P9 y P10 de las diez pistas
  de la especificacion, mas seis de las nueve tecnicas nuevas del material de clase que no
  se han ejecutado: asimetria y kurtosis, palabras inestables por traduccion, idiosincrasias y lexico
  regional, seleccion de rasgos por grupo con criterio de informacion, correlacion entre
  rasgos, analisis discriminante lineal, bosques aleatorios y arboles de decision, y
  validacion cruzada repetida. **Cada linea se ejecuta con el filtro de detectabilidad
  medido y reportado, y genera su submission si tiene merito local aunque no lo supere.**
- **Decision 4, el protocolo de no entregar nada automaticamente se mantiene, y ahora es
  todavia mas necesario**. Como se van a generar varias submissions, la entrega sigue siendo
  manual y cada envio lleva su numero de version y su registro en el manifiesto. Enviar sin
  saber que se esta midiendo es como se gasto el presupuesto en las fases anteriores.
- **Decision 5, los controles no se relajan**. Que se gaste mas presupuesto no cambia nada de
  lo que hace fiable una medicion: el control de codigo, la identidad de pipeline, la
  verificacion de integridad del fichero y el registro de la cifra en
  `experiments.csv` siguen siendo obligatorios en cada linea. Lo que se relaja es
  unicamente la disposicion a pagar por una sonda, no la exigencia de que la cifra sea
  atribuible.

## D-064 - P5, P6 y P7 cerradas con un descubrimiento que reencuadra la fase: la estimacion es inestable entre particiones

- **Contexto**: por D-063 el usuario autorizo gastar envios adicionales y ejecuto las tres
  lineas de capa de decision que la recomendacion tecnica proponia no ejecutar.
- **Decision 1, P5 es la primera linea de la fase que mueve la decision en la direccion
  correcta**. Elegir la constante de regularizacion por etiqueta con validacion anidada estricta
  da **+0,000306**, frente a **-0,002043** del brazo de control que usa una sola constante con
  el mismo procedimiento interno. **La idea aporta +0,002349 sobre el protocolo**, y esa
  diferencia si es atribuible a elegir por etiqueta. Lo relevante no es la magnitud, que es de
  milésimas, sino la **tasa de acierto entre las celdas modificadas: 0,5185**, frente a 0,46 de
  P1, de 0,39 a 0,44 de P3 y de 0,39 a 0,42 de P4. Es la primera vez por encima del azar.
- **Decision 2, P6 vuelve al patron y ademas cuantifica cuanto depende el proyecto del
  balanceo**. La rama por etiqueta da -0,005298 con tasa 0,3919, y el control -0,006857 con
  0,3508. **Lo de P5 era una excepcion y no una tendencia.** El dato duro de la linea es el
  tercer brazo: **sin ponderacion de clase la metrica es exactamente 0,000000**, porque el
  modelo predice todo negativo. Es el reverso del hallazgo de P1, P3 y P4, donde los cambios
  movian la decision hacia peor: aqui quitar el balanceo la destruye. Todo el proyecto depende
  de una sola decision de balanceo, y conviene tenerlo escrito.
- **Decision 3, P7 confirma el patron por cuarta vez con el corrector**. El corrector sobre el
  margen de su etiqueta da **-0,002063** con tasa 0,3611, y anadir el desacuerdo no cambia
  nada. Usar solo el desacuerdo es desastroso, -0,359271. La geometria de los margenes no lleva
  informacion que la decision de V03 no haya usado.
- **Decision 4, un error de fuga que el propio diseno exigia detectar**. Con una sola
  particion, **cuatro de los cinco modelos habrian entrenado con cada fila**, de modo que el
  desacuerdo entre ellos estaba contaminado por resubstitucion y habria sido una medida de
  memoria, no de incertidumbre. Se resolvio con **cinco repeticiones de la particion con
  semillas distintas**, de modo que cada fila recibe en cada repeticion la prediccion de un
  modelo que no la entreno. Es exactamente la validacion cruzada repetida, que ademas era una
  de las nueve tecnicas nuevas del material de clase, de modo que la linea la midio de paso.
- **Decision 5, el descubrimiento que reencuadra toda la fase, y es el mas importante**. El
  desacuerdo medio entre las cinco particiones es de **0,9175 en HS, 0,9671 en TR y 0,9256 en
  AG**, y solo **2 de 1483 filas**, un 0,1 %, tienen las tres etiquetas con desacuerdo por
  debajo de 0,4. **La estimacion es extremadamente inestable entre particiones.** Eso explica
  por que casi todo falla: casi cualquier diferencia entre dos modelos es ruido de particion,
  no senal. Y explica tambien por que el gate de +0,003 de la F1 local no ha detectado nada
  desde la fase v5.
- **Decision 6, y de ese descubrimiento sale la mejor linea de la fase**. Promediar las cinco
  repeticiones da **+0,005066**, con IC95 [-0,00371, +0,01414], **4 de 5 pliegues a favor**,
  182 celdas modificadas y **tasa de acierto 0,5495 frente a un minimo de 0,5527: falla el
  filtro por tres milésimas.** Es la mayor mejora local de la fase, la unica con cuatro pliegues
  a favor y la unica con tasa de acierto por encima de 0,55. El mecanismo es el mismo que P1,
  que promedia cinco modelos, pero **aqui los modelos son de particiones distintas y no de la
  misma particion**, y esa es la diferencia: la reduccion de varianza es real y no marginal.
- **Decision 7, el riesgo que hay que pesar antes de enviar, y se escribe antes**. **P1
  promediaba cinco modelos de la misma particion y perdio 0,00836 en el externo.** El
  mecanismo de promediar tiene un precedente negativo directo, y esta linea es su version
  mas diversiificada. Que la mejora local sea la mayor de la fase no garantiza nada en el
  externo, y el filtro de V8-02 dice que la tasa de acierto no llega por poco. **Se generan las
  dos submissions y la del ensemble de repeticiones es la que tiene merits propio para
  enviar.** El envio es manual, por D-063, y el usuario decide.

## D-065 - Las dos sondas de la fase v8 pierden, y el analisis de la curva de transferencia cierra la calibracion por tasa

- **Contexto**: P5 dio 0,77781 y el promedio de cinco repeticiones dio 0,77122, frente a los
  0,78306 de V03. La segunda era **la mejor linea local de la fase**, con +0,005066 y 4 de 5
  pliegues a favor. Con ello son cinco los casos documentados de mejora local con perdida
  externa: R1-C1, R2-MINDF5, OPT-01, P1 y este.
- **Decision 1, ninguna de las dos sondas se distingue de V03**. Convertidas a aciertos netos
  con la sensibilidad de V8-02, son **-6,6 y -14,9**, frente a una banda de ruido de 19,2. Es
  decir, las dos caen dentro de la banda. **No hay evidencia de que sean peores: hay ausencia de
  efecto**, y eso es un resultado distinto del que el orden local sugeria.
- **Decision 2, se revisa un error de diseno mio en la sonda del ensemble, y no se esconde**. El
  OOF que dio +0,005066 promedia **cinco** modelos, uno por repeticion, porque cada fila solo
  puede recibir en cada repeticion la prediccion de un modelo que no la entreno. La submission
  promedia el promedio de los cinco pliegues de cada repeticion, o sea **veinticinco** modelos.
  **Se valido un ensemble de cinco y se envio uno de veinticinco.** Promediar mas modelos
  contrae mas las probabilidades, y con los umbrales congelados de V03 ese efecto de escala
  desplaza la decision: es el mismo mecanismo que hacia perder a P1, alli medido en +0,001334 de
  contraccion, y aqui la contraccion observada es de -0,0037 en la tasa media. La correccion que
  exige es que la submission promedia cinco modelos y no veinticinco.
- **Decision 3, el analisis que de ahi sale cierra la calibracion por tasa, y es la conclusion
  mas solida de quince sondas**. Ordenando las catorce sondas comparables por su tasa de
  positivos predicha y agrupandolas en cuartiles, la relacion con el leaderboard es una
  **curva con maximo**, no una recta:

  | cuartil de tasa | sondas | tasa media | LB medio |
  |---|---|---|---|
  | baja | 4 | 0,3426 | 0,77355 |
  | media-baja | 3 | 0,3583 | 0,77715 |
  | media-alta | 3 | 0,3614 | 0,77802 |
  | alta | 4 | 0,3785 | 0,77484 |

  **El maximo esta en el cuartil de V03, y moverse en cualquier direccion cuesta.** La
  correlacion lineal de +0,469 es espuria: resume con un coeficiente una campana.
- **Decision 4, eso responde a la pregunta que llevaba P8 de la especificacion**. La
  calibracion condicionada al resultado externo era la unica linea de decision que quedaba por
  medir, y **no hay margen en la dimension de la tasa de positivos**. La F1 como funcion de esa
  tasa es unimodal, y V03 se ajusto a su maximo. Se cierra la familia entera de
  calibraciones, umbrales y pesos por etiqueta, que es donde el proyecto lleva varias fases
  buscando.
- **Decision 5, lo que el analisis no cierra, y conviene no cerrarlo**. Trece de catorce sondas
  estan dentro de la banda de ruido y **V03 es el maximo de las quince**, sin que nada lo
  supere. Hay dos explicaciones posibles y **con este tamano de conjunto no se pueden
  distinguir**: que V03 sea genuinamente el mejor modelo, o que sea el maximo de ruido de una
  serie de comparaciones. Elegir la hipotesis que mas gusta seria exactamente lo que el proyecto
  lleva haciendo desde la fase v5, asi que se deja escrita la ambiguedad.
- **Decision 6, sobre el objetivo de 0,80, con el numero delante**. Esta a **+0,01694** de la
  referencia, o sea **1,1 veces el ruido**, y ninguna de las quince sondas se acerca: la
  segunda mejor esta a -0,00143. Alcanzarlo exigiria un cambio de una magnitud que ninguna de
  las lineas medidas del proyecto ha producido. **El objetivo sigue siendo legitimo y el
  camino medido no lo alcanza**, y conviene que ambas cosas esten escritas a la vez.

## D-066 - La hipotesis del error de diseno queda refutada por medicion directa

- **Contexto**: D-065 afirmaba que la perdida de 0,01184 de la sonda del ensemble se debia al
  desacople entre un OOF de cinco modelos y una submission de veinticinco, porque promediar mas
  modelos contrae mas las probabilidades.
- **Decision 1, se construye la version coherente y se mide, y la hipotesis es falsa**. Un
  modelo por repeticion, que es lo que el OOF promedia, da una submission **identica** a la de
  veinticinco: **cero celdas de diferencia**, y la misma tasa de positivos, 0,3608. La
  contraccion de probabilidades al promediar cinco modelos es de **+0,000591**, de milésimas.
  Promediar los cinco pliegues de cada repeticion no produce una decision distinta de promediar
  uno por repeticion, porque los modelos de pliegues de una misma repeticion coinciden entre si.
- **Decision 2, el desacople existia pero era inocuo, y eso hay que decirlo con precision**. El
  error de diseno de D-065 era real en el procedimiento y falso en su consecuencia: se valido
  un ensemble de cinco y se envio uno de veinticinco, pero la prediccion resultante es la misma.
  **La perdida de 0,01184 no queda explicada por el efecto de escala**, y sigue siendo lo que
  era: -14,9 aciertos netos dentro de una banda de ruido de 19,2.
- **Decision 3, la recomendacion de no enviar se mantiene, pero por otra razon**. La version
  corregida predice una tasa de positivos de 0,3608 frente a los 0,3645 de V03, es decir **por
  debajo**, y la curva de transferencia de D-065 situa el maximo exactamente en la tasa de V03.
  Enviarla seria gastar una sonda en la apuesta que la curva dice que pierde. **El envio es
  manual por D-063.**
- **Decision 4, que se sepa lo que se ha gastado**. Tres sondas en la fase v8: P1 dio 0,77470, P5
  dio 0,77781 y el ensemble dio 0,77122, frente a los 0,78306 de V03. Quince sondas en total a lo
  largo del proyecto, **trece de las catorce comparables dentro de la banda de ruido**, y V03
  como maximo de las quince.

## D-067 - Las seis tecnicas del material de clase, y el hallazgo que explica la fase entera: la mejora local era el promediado

- **Contexto**: quedaban seis de las nueve tecnicas nuevas que el inventario del material de clase
  identifico. Tres no se ejecutan: la estabilidad de features se midio en P3, la validacion cruzada
  repetida se midio dentro de P7, y las palabras inestables por traduccion no son operables porque
  la traduccion automatica no esta disponible. Las otras abarcan seis lineas, agrupadas en tres scripts porque comparten la infraestructura de evaluacion, y cada linea conserva su hipotesis, su cifra y su fila propia en el registro.
- **Decision 1, un criterio de lectura nuevo que resulta mas informativo que el intervalo de
  confianza**. Al medir cada rama en **cinco repeticiones de la particion** se cuenta **cuantas
  repeticiones mejoran**, no solo la cifra final. El motivo lo dio la propia linea: la rama de
  rasgos de complejidad dio **+0,003493**, que supera el gate de +0,003, y **solo mejoro en una de
  las cinco repeticiones**. Una mejora que aparece en una particion de cinco es ruido de
  particion, y es exactamente lo que ocurrio con la sonda del ensemble, que gano +0,005066 local y
  perdio 0,01184 en el externo.
- **Decision 2, el hallazgo que explica la fase entera, y sale de un control que hacia falta**.
  V8-12 incluyo un brazo que es **V03 sin cambiar nada**: misma representacion, mismos
  hiperparametros, mismos umbrales, medido con el protocolo de cinco repeticiones. Ese control da
  **+0,004309 con 0 de 5 repeticiones que mejoran**, y V8-13 lo confirma con **+0,002824 con 1
  de 3**. Es decir, **la diferencia de +0,004 a +0,005 que se atribuia a la tecnica de varias
  lineas de la fase es del promediado, no de la tecnica**. Eso explica la sonda del ensemble, la
  seleccion por grupo al 25 % y los rasgos de complejidad, y es el mismo sintoma que la
  estimacion inestable entre particiones de P7.
- **Decision 3, por tanto el criterio de decision de la fase cambia de la cifra a la
  consistencia**. Ninguna de las seis lineas supera al control en numero de repeticiones, y por
  tanto **ninguna se envia**. No es una decision sobre el presupuesto: es que la evidencia dice que
  la mejora no existe mas alla del promediado, y el promediado ya esta medido en el externo con
  -0,01184.
- **Decision 4, lo que si se aprende de las seis, linea por linea**.
  - **Asimetria y kurtosis**: +0,003493 pero solo 1 de 5 repeticiones. El rasgo mide algo que la
    media no mide, y aun asi no se sostiene. Las **idiosincrasias internas** dan +0,000378 con
    1 de 5 y tasa 0,4484: la proporcion de tokens raros correlaciona con el tamano del documento
    mas que con la toxicidad.
  - **Seleccion por correlacion**: **-0,183116**, porque reducir 20.870 rasgos a 251 por magnitud y
    redundancia deja la metrica en 0,594. El criterio de magnitud no distingue un rasgo
    informativo de uno frecuente, y **queda descartada de raiz**.
  - **Seleccion por grupo con informacion**: al 25 % da +0,005324 con 2 de 5, y al 10 % da
    -0,008538 con 0 de 5. Aplicar el criterio por grupo evita que uno grande aniquile a uno
    pequeno, pero no produce senal.
  - **Discriminante lineal**: -0,030054 con 0 de 3. Sus supuestos de normalidad y covarianza
    independiente no se cumplen sobre veinte mil caracteristicas dispersas, como se anticipaba.
  - **Arbol de decision y bosque aleatorio**: -0,073168 y -0,063937, con 0 de 3 y tasas de acierto
    de 0,3529 y 0,3393. Los arboles sobre un subespacio de 2.000 caracteristicas pierden la
    estructura que hace util la union de palabra y caracter.
- **Decision 5, la conclusion de la linea mas alla de estas seis**. **Las tres familias de
  clasificador alternativas, estructuralmente distintas de la regresion logistica, pierden las
  tres.** El problema no es el clasificador lineal sino la tarea con esta representacion, y eso
  cierra la via de cambiar de modelo dentro del marco clasico.
- **Decision 6, el balance de la fase**. P1, P3, P4, P2, P5, P6, P7 y las seis tecnicas del
  material de clase: **ninguna aporta senal por encima del ruido**, y tres sondas externas
  confirmaron que las mejoras locales no trasladan. El objetivo de 0,80 sigue a +0,01694 de la
  referencia, que es 1,1 veces el ruido. **Se registry todo y no hay ninguna submission
  recomendada.**

## D-068 — Apertura de la fase v10 en worktree propio

Se abre `/home/carlos/Git/PLN-v10` en la rama `proyecto-kaggle-toxic-2026-v10` desde `23aac93`,
con `data/` enlazado a la copia compartida y `features/` y `models/` copiados, porque son
artefactos derivados que los scripts necesitan y la auditoria demostro que sin ellos fallan.

**Se elige worktree propio y no trabajar en el v8** por dos razones: el v8 ya acumula catorce
scripts de la fase v8 mas los de la auditoria, y mezclar ahi haria imposible rastrear que cambio
pertenece a que fase; y el plan pide commits con prefijo de fase, que en una rama compartida se
mezclarian.

**Sin commit hasta peticion expresa del usuario.** El arbol queda limpio pero sin commitear.

Verificacion de partida, `V10-00`: `corrected_cv` reconstruido 0.7772114525165721 con
coincidencia exacta, `sha256` de la submission v03 identico, 207 filas de registro, 15 sondas.

## D-069 — Holdout2 congelado, su gate NO se supero, y por que se creo antes de la via A

Se crea `holdout2` con `StratifiedGroupKFold(n_splits=6, semilla=20260926)` sobre las 1483 filas
del desarrollo: 247 filas en 235 grupos para holdout2, 1236 en 1180 para dev2, **cero grupos
compartidos**, particion reproducible y cero solape con el holdout v2. Se congelan
`config/holdout2.json` y `data/processed/holdout2_ids.csv` con `sha256`.

**El gate NO se supero**: la referencia de V03 recalculada en dev2 con cinco pliegues propios
quedo en **0.763436**, una desviacion de **-0.013776** frente a la tolerancia declarada de 0.005.

**Se creo antes de la via A, y no despues, por una razon concreta**: si se creara despues, las
candidatas de la via A se elegirian sobre las mismas filas que luego se usarian para juzgarlas, y
el juez dejaria de ser independiente de la seleccion.

**Advertencia escrita antes de ver resultados y que queda vigente**: holdout2 **no es virginal**.
Sus filas salen del desarrollo actual, que ya participo en las decisiones de las fases v1 a v9.
Su valor es **detectar que un candidato esta sobreajustado a su gate**, no probar generalizacion
sobre datos nunca vistos.

Por la regla pre-registrada, **dev2 no se usa para ninguna decision de adopcion**. Holdout2 si
sigue pudiendo servir como juez, pero solo por comparacion pareada dentro de sus 247 filas.

## D-070 — V10-02: el ruido de la metrica corregida es de mas de 0.017, y eso obliga a cambiar el protocolo

La desviacion de D-069 admite tres explicaciones separables. El **control de equivalencia da
diferencia exactamente 0.0**: el codigo de P0b reproduce los 0.7772114525165721 al ultimo
digito, asi que **no es un fallo de implementacion**.

- Con **ocho semillas de particion** y sin cambiar nada mas, la metrica recorre de 0.772021 a
  0.789205: recorrido **0.017184**, desviacion tipica **0.005302**.
- Con **ocho subconjuntos de 1236 filas**, recorre de 0.762169 a 0.782417: recorrido **0.020248**,
  desviacion tipica **0.007134**. El 0.763436 de dev2 cae dentro de esa banda.

**La causa es ruido de tamano y de particion, no un defecto del conjunto.** Y aparece un
resultado de fondo: **la metrica corregida tiene una banda de mas de 0.017 entre particiones
equivalentes**, lo que confirma por una via independiente y sin gastar una sola sonda el ruido de
0.01524 que V8-02 medico en el externo. Dos instrumentos, misma magnitud.

**Consecuencia, y es una correccion de protocolo, no un relaxation**: el gate de la via A estaba
escrito como "mejora anidada de al menos +0.002 en un solo calculo", y eso es **inoperante** con
esta banda. Se escribe un **addendum fechado antes de ejecutar la via A** que exige tres cifras
para toda candidata: delta medio sobre `K = 5` particiones, intervalo pareado al 95 por ciento, y
**recuento de particiones que mejoran**, con adopcion solo si se cumplen las tres: delta de
+0.002, al menos 4 de 5 particiones, e intervalo que no incluye el cero. Los gates por sub-linea
siguen vigentes sin relajar.

El criterio mas informativo sigue siendo **cuantas repeticiones mejoran**, como en D-067.

## D-071 — Arnes de evaluacion validado en los dos sentidos, y los cuatro gates de la via A

**D-071a, el arnes (V10-03) se valida antes de usarse para juzgar.** Con el control nulo, la
candidata identidad, da delta exactamente 0.0 en las cinco particiones e intervalo [0.0, 0.0]. Con
el control positivo, una candidata oraculo que peina el 2 por ciento de celdas de menor margen
usando la etiqueta verdadera, da **+0.011781, 5 de 5 particiones** e intervalo
[+0.007530, +0.014556] que no incluye el cero. Un gate que solo sabe rechazar es tan inutil como
uno que solo sabe aprobar.

**D-071b, V10-04 calibracion fuerte: NO ACCIONABLE.** La isotonica recortada reduce el ECE medio
un **29.94 por ciento** frente a Platt, muy por encima del 20 exigido, asi que la calibracion
funciona. Pero **no se traslada a la metrica**: mejor delta medio anidado **+0.000148** con
**3 de 5** particiones, y los tres intervalos pareados incluyen el cero. Es una mejora real de
calibracion, del ECE de 0.0583 a 0.0174, que mueve la F1 treinta veces menos que la banda de
ruido. Se cierra sin sonda. Se midio ademas, en vez de suponerlo, que la calibracion beta sale
peor porque su rejilla **no contiene a Platt**: el desplazamiento de Platt es libre y la rejilla
limita la asimetria a [0.7, 1.3], y se comprobo con verosimilitud en las tres etiquetas.

**D-071c, V10-05 regla F1 optima: CERRADA, y la hipotesis era incorrecta.** El mecanismo se
comporto **exactamente** como se derivo: con el shift de prevalencia la regla bajo sus umbrales
entre 0.0029 y 0.0073, que es la adaptacion automatica que V03 no puede hacer. Pero los umbrales
congelados de V03, 0.44, 0.51 y 0.46, son ya **mas altos** que el optimo plug-in con prior
natural, 0.3841, 0.4093 y 0.3789. V03 ya esta por encima del optimo esperado y la regla se aleja
de el al bajar mas. Gate A fallido con **-0.003921** en prior simulado, gate B fallido porque el
shift empeora en vez de ayudar, y 1 de 5 particiones. La leccion: los umbrales elegidos empiricamente
ganan a la regla optima **en esperada** bajo el modelo, porque la metrica de la competencia premia
un punto empirico de la superficie y no el maximo de la expectativa del clasificador.

**D-071d, V10-06 umbrales estables: la holgura es irreducible.** La parametrizacion mas restringida
no mantiene la ganancia de la de tres umbrales libres: la de desplazamiento y escala es casi igual
de mala y algo mas dispersa, y la de encogimiento es la mas estable pero con media mas baja. Gate
1 no superado, y los umbrales congelados se quedan porque no existe version a la vez igual de
buena y mas robusta.

## D-072 — CORRECCION: la cifra de +0.0036 no es el efecto de optimizar umbrales

**La fase v6 atribuyo a la reoptimizacion de umbrales un inflado de +0.0036, y esa cita era
incorrecta.** El informe `reports/v7_umbral_anidado.md` ya lo habia demostrado y asi lo explica:
el control de tres niveles con umbrales congelados da **-0.0081** solo por ajustar sobre cinco
sextos de las filas, y al comparar contra un brazo que comparte conjunto de ajuste, el efecto
aislado del umbral queda en **-0.0019**. El +0.0036 era un error de comparacion entre brazos con
distinto conjunto de ajuste, que es el mismo error de interpretacion que la propia fase v7 senalo.

**Medicion propia de V10-06**: reestimar los tres umbrales por pliegue da **+0.000131** frente a
V03. Las tres cifras miden cosas distintas y **las tres coinciden en lo substancial: el efecto del
umbral es pequeno**. Lo que no cabe es citar +0.0036 como si fuera el efecto del umbral, y esa
correccion se aplica a la documentacion de V10-04 y V10-06, que la habian repetido.

Ademas, las dispersiones de las tres parametrizaciones, de 0.0052 a 0.0054, son casi iguales a la
banda de particion de 0.0053 de D-070, lo que indica que la dispersion observada es **ruido y no
una propiedad de la parametrizacion**.

## D-073 — Shift de features: la via se CIERRA porque el discriminador esta por debajo del azar

Un discriminador que separa entrenamiento de prueba con las 24.646 caracteristicas de V03 da un
**AUC global fuera de muestra de 0.487717**, con los cinco pliegues entre 0.462749 y 0.525974, y
una exactitud de 0.512305. El AUC esta **por debajo de 0.5**: el discriminador es incapaz de
distinguir las dos poblaciones y en promedio va algo por detras del azar.

El gate pre-registrado de 0.55 se cumple con holgura y **la via de shift de caracteristicas se
cierra**. Las caracteristicas mas desplazadas tienen coeficientes de signo inesperado, sintoma de
colinealidad y no de desplazamiento real. La lectura de conjunto es coherente con V9-02: **el
unico desplazamiento medido entre las dos poblaciones es el de prevalencia, de factor 0.94**, y
D-071c ya mostro que adaptarle la capa de decision empeora. Diagnostico solo: no se entrena nada
con la prueba.

## D-074 — Integridad de holdout2 y filtro de detectabilidad: nada enviable, y una correccion de criterio

La integridad se confirma: los identificadores conservan su `sha256`, se localizan las 247 filas,
cero solape y cero grupos compartidos. **Las tres evaluaciones de holdout2 siguen sin gastar**,
porque las tres lineas de la via A se cerraron en sus propios gates y ninguna llega a la mesa de
finales.

El filtro de detectabilidad de V8-02 aporta un dato que no es un tecnicismo: con 124 celdas
cambiadas una candidata necesitaria acertar mas del **0.6548** de ellas para que el leaderboard la
separara del azar, y con 134 celdas mas del **0.6433**. Ninguna llega a ese punto.

**Correccion de un criterio propio**: la prueba de mejora era "delta mayor que cero", lo que daba
por buena una candidata con **+0.000131** y contradecia el propio veredicto del guion. Se sustituye
por los dos criterios del addendum, delta minimo y particiones que mejoran. Con la banda de
D-070, un delta positivo por si solo no significa nada.

## D-075 — Correlacion entre caracteristicas: la conclusion de la v8 era correcta pero su prueba no la sostenia

**D-075a, por que V10-09 no repite la v8.** La v8 cerro la seleccion por correlacion asi: "reducir
20.870 rasgos a 251 por magnitud y redundancia deja la metrica en 0,594, el criterio de magnitud
no distingue un rasgo informativo de uno frecuente, y **queda descartada de raiz**". Eso tiene dos
problemas: **colapsaba el 99 por ciento de las caracteristicas**, lo que no permite distinguir quitar
una de cada par de quitar casi toda la senal; y el criterio era **magnitud**, no redundancia, como
la propia conclusion reconocia. El veredicto era correcto y la prueba no lo sostenia.

**D-075b, bloque A, estructura de correlacion (lo que nunca se habia medido).** Las 20.441
caracteristicas ajustadas en un pliegue estan **poco correlacionadas**: la correlacion maxima por
caracteristica tiene **mediana 0.1704**, percentil 90 de 0.5608, y solo unos 1.203 pares de 0.90,
973 de 0.95 y 835 de 0.99. **La redundancia alta es rara y muy localizada, no un problema
general.**

**D-075c, bloque B, correlacion con la etiqueta.** Medida con el AUC de la caracteristica sola, que
es medida de rango. El AUC mediano es **0.5133, 0.5198 y 0.5147** para HS, TR y AG, y solo el
**0.29, 0.80 y 0.35 por ciento** de las caracteristicas se sale de la banda [0.45, 0.55]. Las
mejores llegan a **0.6717, 0.7569 y 0.6844**. La conclusion es que **el modelo lo construye
combinando miles de rasgos debiles, no apoyandose en unos pocos fuertes**, y eso explica por que
quitar unos pocos duela mas de lo que parece.

**D-075d, bloque C, eliminar una de cada par: EMPEORA, y el gate C no se supera.** Los tres
umbrales pre-registrados empeoran con **0 de 5 particiones** a favor: **-0.030626** a 0.90 con 283
caracteristicas eliminadas, **-0.027661** a 0.95 con 215, y **-0.027129** a 0.99 con 192. El dano
es desproporcionado para lo que se quita: el 1.38 por ciento de las caracteristicas cuesta 0.0306
de F1. Ese desproporcion abrio la duda que resolvio D-076.

**D-075e, lo que se cierra y lo que no.** Se cierra que **elegir que descartar por nivel de
correlacion no mejora a V03**, con tres umbrales y sin reintentar mas. **NO** se cierra que exista
mejor regla de cual conservar, porque esa regla no estaba pre-registrada, y afirmar lo contrario
seria hacer exactamente lo que hizo la v8.

## D-076 — El control refute la hipotesis de la complementariedad: la REGLA es irrelevante

V10-09 empeoraba tanto que la explicacion obvia era la **regla de conservacion**: se conservaba de
cada par la caracteristica mas informativa **en solitario**, y para un modelo lineal lo que vale
es lo que cada una aporta de lo que la otra no aporta. La contraprueba era conservar la mas
**debil**, y el control decisivo era conservar **al azar**.

**El control desconfirma la hipotesis.** Las cuatro reglas dan, con umbral 0.90:

| regla | eliminadas | media | delta | mejoran |
|---|---|---|---|---|
| sin eliminar | 0 | 0.775910 | 0 | - |
| fuerte en solitario | 317 | 0.744657 | **-0.031252** | 0/5 |
| debil en solitario | 265 | 0.744031 | **-0.031879** | 0/5 |
| menos redundante | 275 | 0.742025 | **-0.033885** | 0/5 |
| **aleatoria** | 273 | **0.747023** | **-0.028887** | 0/5 |

**La regla aleatoria es la mejor de las cuatro**, y la brecha entre la mejor y la peor es de
**0.005**, menos que la banda de ruido de 0.0053 de D-070. Es decir: **la regla de conservacion es
irrelevante**, y lo que cuesta 0.03 es **cuantas** caracteristicas se quitan, no cuales. Quitar
unas 270 de 20.441, de cualquier forma razonable, cuesta lo mismo.

**La eliminacion de caracteristicas correlacionadas queda descartada de forma solida**, y esta vez
con el control que faltaba en la v8: cuatro reglas, tres informadas y una ciega, todas con el mismo
resultado. Ademas significa que esas caracteristicas correlacionadas **llevan senal real y no
sustituible**, y que en la estructura de correlacion **no hay nada que explotar**.

## D-077 — P4, BiGRU con afinamiento: afinar funciona, y la tasa diferencial es lo que lo hace funcionar. H1 cerrada

La fase v9 dejo H1 con una sola variante: V9-04, con los embeddings **congelados**, dio **0.6943**,
frente a 0.6636 del control de media mas regresion lineal y 0.777211 de V03. Quedaba probar si
**afinar** los embeddings cambia el panorama.

**Afinar si funciona, y es la mejora mas grande entre dos variantes de la misma arquitectura que
ha producido el proyecto**: la variante afinada da **0.711639**, es decir **+0.017645** sobre la
congelada.

**Pero el control es lo que da valor al resultado, y es la variante que el plan no habia pedido.**
Se añadio una tercera variante, la misma red afinada pero con **una sola tasa de aprendizaje para
todo** en vez de la diferencial. Da **0.699086**, o sea solo **+0.005092** sobre la congelada. De los
+0.017645 del afinamiento, **+0.012553 los aporta la tasa de aprendizaje diferencial y no el
hecho de mover el embebido**. Sin ese control, la conclusion would've sido que afinar aporta
+0.0176, cuando la version ingenua aporta +0.0051. **El mecanismo esta identificado.**

**Las mejores epocas confirman el mecanismo**: la variante de tasa unica para siempre entre 7 y 8,
muy por debajo del tope de 14, porque una tasa alta destruye la representacion preentrenada antes
de que la red pueda aprovecharla. Congelado y afinado llegan a 8-13 y aprovechan el entrenamiento
completo.

**Control interno del protocolo**: el congelado reproduce V9-04 en **0.693994** frente a su 0.6943,
con tope de epocas y semillas ligeramente distintos. El protocolo es reproducible.

**Decision de sobreentrenamiento: early stopping estricto, y la fijo el pre-registrado** de
`tasks/plan_v10.md` antes de ejecutar nada, con paciencia 3 sobre la perdida de validacion y tope
de 14 epocas. Se anota un limite honesto: **la mejor epoca se elige por la perdida de validacion,
que es la misma senal que se usa para las curvas de aprendizaje**. Para estimar un techo eso no es
fuga, y la metrica de la competencia se evalua una sola vez al final y no se usa para ajustar; pero
si alguien quisiera usar este mismo protocolo para elegir una variante, si lo seria.

**H1 QUEDA CERRADA con dos variantes, congelada y afinada.** La mejor queda a **-0.065572** de V03,
que era el criterio de cierre pre-registrado para P4, sin gate de envio. **Sin submission y sin
sonda.**

**La via de representacion secuencial queda cerrada** y no se abre con mas capas, mas unidades o
mas epocas, porque el criterio de cierre estaba escrito antes de ejecutar y ya se cumplio.

La lectura de fondo: V03 es una union **explicita** de TF-IDF word(1,2) y char_wb(3,5) con una
regresion lineal, es decir captura la redundancia de forma directa, mientras que una red tiene que
**descubrirla** por su cuenta. Con mil y pico filas y 24.646 rasgos explicitos, la regresion
lineal aprovecha mejor los datos. No es que las redes no sirvan: es que **este** problema, con
**estos** datos, no las necesita.

## D-078 - Apertura de la v11 en worktree propio, y por que antes que un commit de v10

La v11 se abre en `/home/carlos/Git/PLN-v11`, rama `proyecto-kaggle-toxic-2026-v11`, trayendo el
estado commiteado de la v10. **v10 no se modifica.** Se verificaron las 40 copias de ficheros de
v10 byte a byte antes de eliminar las de v11 y traerlas por merge, y las 5 modificadas por
`git status --short` tambien, para no perder trabajo propio por un descuido de rutas.

**P0, V11-00, 25 de 25 comprobaciones superadas, pero la primera ejecucion dio 21 de 25 y las tres
paradas son el resultado real.** Dos eran errores de mi propia comprobacion: buscaba la misma clave
en el informe y en el registro, y en V10-05 y V10-10 no se llaman igual, ademas de que en V10-10 el
informe anida el veredicto. La tercera era un problema de verdad, tratada en D-079.

## D-079 - El campo factor_vs_train de V9-02 no significa lo que dice

El campo ``factor_vs_train`` de `reports/v9_02_prior_shift_em.json` **no** es la prevalencia de
prueba dividida por la del train completo de 1731 filas, sino por la del **desarrollo** de 1483. Se
comprobo en el codigo: `prevalencia_train = y_dev.mean(axis=0)`.

Con el train completo el factor queda 0.000196 mas alto en HS y 0.000249 en TR; contra el
desarrollo coincide a 1.5e-5, 8e-6 y 2e-6. **V9-02 es internamente consistente y no tiene
defecto; lo que no significa lo que dice es el nombre del campo.**

Se decide que la v11 use la **prevalencia absoluta** de la prueba, que es la magnitud sin
ambiguedad, y no el factor. La reproduccion, V11-01, da nivel **R3** con diferencias de -4.51e-07 a
+3.29e-07, que es el redondeo a seis decimales del JSON, e iteraciones del EM identicas: 26, 16 y 24.

## D-080 - El uso de la prueba para estimar prevalencia queda PROHIBIDO, y con el se cae P10

La auditoria de legalidad, **V11-02**, clasifico los once pasos de la metodologia contra el texto
verbatim de `docs/competition_rules.md`, con las cuatro citas verificadas literalmente. Reparto:
**5 permitidos, 1 ambiguo, 5 prohibidos**.

El paso ambiguo es **estimar la prevalencia de la prueba con EM sobre probabilidades no
etiquetadas**, y lo es de verdad: la regla prohibe entrenamiento y aqui no hay entrenamiento,
porque el modelo esta congelado y no se usa ninguna etiqueta; pero la intencion de la regla es que
el test no participe del desarrollo del sistema. **Lo decisivo es que las reglas no mencionan la
transduccion ni para permitirla ni para prohibirla**, y ese silencio es lo que hace el caso ambiguo.

La consecuencia es la que fija el plan: **el uso no se emplea para generar submission**. Y con la
decision explicita de NIVEL 2, "no usar test real para prior correction", el paso pasa de ambiguo a
**prohibido**, y **v17 transductiva queda en cuarentena**.

**Consecuencia de alcance, escrita sin adornos**: la pregunta de si el mecanismo transfiere a esta
prueba concreta **no tiene respuesta verificable dentro de las reglas**. Se puede y se respondio la
version metodologica, que es si el mecanismo recupera un desplazamiento conocido. **La v11 no
termina en una submission, y no por falta de calculo sino por una limitacion de las reglas.**

## D-081 - Los estimadores de prevalencia funcionan, y por que uno da error exactamente cero

**V11-06, con shift simulado y conocida la verdad.** Dos estimadores conceptualmente distintos:
EM de Saerens ponderado, y clasificar-y-contar ajustado de Forman y Crawford. La comparacion es
justa porque si los dos fallaran el problema seria la idea, y si fallara solo uno seria el
estimador.

Resultados: EM con error relativo medio **0.0069**; clasificar-y-contar con error **0.0000**.

El cero de clasificar-y-contar **no es casualidad y se explica antes de aceptarlo**: la simulacion
repondera filas por clase y no toca probabilidades ni predicciones, luego P(pred|y) es identica en
origen y destino, luego **la sensibilidad y la especificidad son invariantes** y la identidad que
el estimador despeja se cumple de forma **exacta**. Bajo un shift **puro** de prevalencia,
clasificar-y-contar esta exactamente identificado. Es un teorema, no una propiedad empirica
favorable.

**Se declara el punto ciego**: los dos estimadores asumen que el shift es puro, y la simulacion lo
cumple por construccion, luego el resultado demuestra que son correctos **dentro** de ese supuesto
y nada mas. Hay un apoyo inesperado: **V10-07** midio un AUC de **0.487717** entre entrenamiento y
prueba, por debajo del azar, luego no hay shift de covariables medido dentro de este proyecto. Eso
apoya el supuesto pero **no lo prueba con datos reales, y no se presenta como si lo probara**.

## D-082 - La correccion de prior NO mejora la metrica, y la causa esta medida

**V11-07, con el factor conocido por construccion, que es el caso en el que el mecanismo se puede
comprobar.** La correccion desplaza el logit de forma constante en
`logit(pi_destino) - logit(pi_origen)`, y se mide con los **umbrales congelados de V03**, sin tocar
la capa de decision, para que la ganancia se atribuya a una causa y no a la otra. V9-02 mezclaba
correccion de probabilidades con eleccion de umbral.

**Gates A y B no superados.** En el factor verdadero de cada etiqueta: HS a **-0.001032** con
**1 de 5** particiones, TR a **+0.000000** con **0 de 5**, AG a **+0.000154** con **1 de 5**. Macro
medio **-0.000293**. Los tres intervalos pareados incluyen el cero.

**La direccion del efecto es sistematica y explica el fallo**: la correccion **empeora cuando la
prevalencia baja** y **ayuda cuando sube**. A factor 1.05 da +0.001095 en TR y +0.001194 en AG; a
0.85 da -0.001997 en HS. El desplazamiento medido por V9-02 es hacia **abajo**, de factor 0.94,
luego el mecanismo actua en la direccion contraria a la que se necesita.

**La causa de fondo, medida en V10-05**: los umbrales congelados de V03, 0.44, 0.51 y 0.46, estan ya
**por encima** de los umbrales optimos de F1, que V10-05 midio en 0.3841, 0.4093 y 0.3789. La capa
de decision **ya es conservadora**, como si estuviera pre-compensando una caida de prevalencia.
Aplicar la correccion de prior encima **corrige dos veces** y se pierde exhaustividad. Es el mismo
mecanismo que hizo perder a la regla plugin de V10-05, medido por otra via, y ambos convergen en
la misma causa.

**Cierre: la via de la correccion de prior se cierra con protocolo valido y no se reintenta**, porque
el mecanismo es el unico posible. **Con ella caen P5**, calibracion mas shift, y **P7**, decision
condicionada por la prevalencia: ambas presuponen que el shift es explotable, y se ha medido que no
lo es. No es una decision de gesto, es consecuencia logica.

## D-083 - Cierre de la v11: nueve lineas, ningun candidato, y el limite escrito

**Nueve experimentos de la v11, V11-00 a V11-08**, con el registro en 227 filas y 227 claves
unicas. **Ninguna submission generada, ninguna enviada, ninguna recomendada.** **V03 se mantiene**
con 0.7772114525165721 local y 0.78306 en el externo. **holdout2 intacto, con 0 de 3
evaluaciones gastadas.**

Las doce preguntas de la revision final: **10 respondidas, 2 con respuesta UNKNOWN**, que no se
rellenan con una estimacion. La mayor laguna es si la calibracion mas el shift supera al shift
solo, y no es que falte medirla: **el shift solo ya es negativo**, luego la combinacion no tiene
una base desde la que mejorar, y medirla exigiria ademas usar la prueba.

Las tres lineas que no dependen del shift, **P4, P6 y P8**, ya estan medidas y las tres cerraron
negativas en v10: P4 NO ACCIONABLE porque la calibracion mejora el ECE de 0.0583 a 0.0174 y no la
F1; P6 empeoro -0.003921; P8 no supero su gate. **La v11 no deja ninguna linea abierta con
expectativa razonable.**

**Sobre la reproduccion**: se declara **R3** para V03, V9-02 y la mejor candidata de calibracion, y
**R2** con limitaciones para el holdout2. **No se declara R4 en ningun caso**, porque no hay segundo
ejecutor ni segundo entorno, y el enunciado dice que un output cacheado no cuenta como
reproduccion independiente. Declarar R4 seria el tipo de afirmacion que la seccion 22 prohibe.

**Dos hallazgos metodologicos de la fase, que valen mas que sus resultados**:

1. **La politica de caracteres de la seccion 17 es MAS ESTRICTA que la herramienta del proyecto.**
   `tools/verificar_caracteres.py` busca alfabetos no latinos y da veredicto correcto sobre un texto
   con acentos; la seccion 17 pide **ASCII puro**. La diferencia eran 4 caracteres en la v11 y 3
   ficheros heredados de v10.
2. **Un escaner de cumplimiento que busca palabras prohibidas en texto es inservible.** Daba doce
   falsos positivos, porque las coincidencias eran las citas de la prohibicion, su propia lista de
   patrones y la tabla que dice que no se usa. Se sustituyo por **analisis estructural de
   imports**, donde un nombre en un `import` es un uso y en un comentario no lo es. Ademas el
   informe del escaner imprimia los caracteres que violaba, violando el mismo la politica que
   auditaba, y ahora se citan solo por codigo.

**Conclusion de la fase**: el desplazamiento de prevalencia **no es una fuente real de mejora** bajo
las restricciones actuales, y se ha podido demostrar con un factor conocido por construccion. Lo
que **no** se ha podido demostrar, y queda escrito para que no se lea mas de lo que vale, es si el
mismo mecanismo mejoraria en esta prueba concreta. El objetivo de superar 0.80 sigue a la distancia
que midio la v10.

## D-084 - Un entregable vacio con el motivo escrito es mas honesto que uno lleno de datos ajenos

Al cerrar la fase con V11-08, **siete de los once entregables** pedidos tenian contenido real y
cuatro no se produjeron. **V11-09 los completa documentando el motivo de cada vacio**, y la
decision que importa es **lo que no se hizo**.

**No se copiaron cifras de v10 a tablas de v11.** Habria creado trazabilidad falsa: el lector
abriria un fichero `v11_calibration.csv` con resultados de V10-04 etiquetados como de la v11, sin
forma de distinguirlo. Un entregable ausente se confunde con uno olvidado, pero un entregable
**lleno de datos que no son suyos es peor que uno ausente**, porque no se detecta por la forma del
fichero sino por leerlo con cuidado.

Los tres CSV vacios, y la razon de cada uno:

| Entregable | Linea | Por que no hay cifras |
| --- | --- | --- |
| `v11_calibration.csv` | P4 | La calibracion **ya estaba medida y cerrada en v10** con V10-04, NO ACCIONABLE: mejora el ECE de 0.0583 a 0.0174 y no la F1. En v11 no hubo experimento propio. |
| `v11_threshold_stability.csv` | P8 | La parametrizacion estable **ya estaba cerrada en v10** con V10-06, que no supero su gate. Sin experimento propio. |
| `v11_error_intersection.csv` | transversal | Es una medida **entre dos o mas candidatos**, y solo hay uno vivo, V03. La interseccion de un conjunto consigo mismo es el propio conjunto. |

**`v11_gap_audit.md` es un PUNTERO, no una copia.** La auditoria de brechas **si se ejecuto**: es
**V11-00**, con 25 de 25 superadas. Su informe canonico es `reports/v11_00_reconciliacion.md` y ahi
estan las cinco secciones de detalle. Copiar el contenido crearia **dos fuentes de verdad para la
misma observacion**, y dos fuentes de verdad divergen: cuando divergen, ya no se sabe cual es la
verdadera.

**Dos correcciones posteriores a la ejecucion, documentadas para que no se pasen por alto:**

1. **V11-08 dejo su propia fila de registro sin escribir**, igual que le paso a V11-05. Lo
   detecta el propio script de cierre, que avisa cuando un identificador de su tabla no esta en el
   registro. El aviso funciona.
2. **Un `F401` en `v11_01_reproduccion_v9_02.py` que yo mismo introduje**: al documentar como
   comentario una variable muerta, deje su `import` huerfano. El import retirado **nunca fue
   invocado**, luego es **inerte por construccion**; y la reproduccion **se re-ejecuto despues del
   cambio** devolviendo **nivel R3 exacto**, con verificacion exacta True y denominador resuelto
   True. **No se modifico ninguna cifra de ningun informe.**

Los **14 avisos de ruff** que restan en los guiones de la v11 son todos `ISC004`, concatenacion
implicita dentro de colecciones: es estetico y no afecta a la ejecucion.

## D-085 - Un escaneo que cuenta no basta: tiene que separar lo heredado de lo propio

El segundo escaneo de caracteres, el estricto de ASCII puro, reporto **173 hallazgos globales** en
`DECISIONS.md`, `RUNLOG.md` y `PROJECT_STATE.md`. La cifra, por si sola, era inutil: invita a dos
errores opuestos a la vez, **reformatear 147 caracteres heredados** de las fases previas o
**dar la v11 por limpia** porque el total parece pequeno.

**La distincion util es por bloque, y la hace el marcador de inicio de la v11.** Resultado:

| Bloque | Caracteres no ASCII | Decision |
| --- | --- | --- |
| Antes de `## D-078`, heredado de v10 y previas | 147 | **No se toca**, es la desviacion ya documentada; modificarlo rompe la correspondencia entre codigo y resultado historico. |
| Desde `## D-078`, propio de la v11 en `DECISIONS.md` | 7, todos em-dashes U+2014 en titulos | **Corregidos** a guion. |
| Bloque propio en `RUNLOG.md` y `PROJECT_STATE.md` | 0 | Sin accion. |

**Por que importa el numero pequeno tambien es un error.** Un recuento global habria hecho pasar
por limpio un bloque mio que no lo estaba, porque 7 sobre 173 parece poco. El criterio correcto no
es cuanto, sino **de quien**: los caracteres heredados son una desviacion conocida y aceptada con
motivo, y los propios son un fallo mio corregible. Tratarlos igual habria exigido tocar v10, o
habria dejado pasar los mios.

Se mantiene entonces el verificador propio de ASCII puro **ademas** de
`tools/verificar_caracteres.py`, y se documenta que los dos son necesarios: el primero busca
alfabetos no latinos y el segundo busca ASCII puro. Ninguno de los dos, por si solo, distingue el
bloque propio del heredado, y esa distincion hay que hacerla por bloque de forma explicita.

## D-086 - Escalar un bloque de caracteristicas ES regularizar ese bloque, y por tanto H1 no es un hiperparametro

La implementacion de H1 es escalar las columnas del bloque de caracter antes de ajustar, y esa
operacion tiene un efecto **exacto** sobre la penalizacion, no aproximado. Para una regresion
logistica con penalizacion L2, sustituir `X` por `X * diag(s)` produce, en terminos de los
coeficientes originales, una penalizacion

    (1 / 2C) * sum_j s_j^2 * w_j^2

Es decir, **multiplicar el bloque j por `s_j` multiplica su peso de penalizacion efectivo por
`s_j^2`**. No cambia la clase de modelo, no cambia la dimension, no anade variables, y se hace con
el `LogisticRegression` de la biblioteca sin tocar su optimizador. Por eso H1 es un mecanismo y no
otro valor de un parametro, que es el criterio que la seccion 8 del enunciado exigia.

**Y tiene exactamente UN grado de libertad, no dos.** Escalar el bloque de palabra por `a` y el de
caracter por `b` es lo mismo, salvo una constante global, que fijar `a=1` y `b=s`. Y como la matriz
se renormaliza para conservar su norma de Frobenius, esa constante global queda fijada. El
resultado es un espacio de busqueda de **tres cocientes y cuatro constantes**, con el cociente
neutro en 1.0, que no es una rejilla.

**La renormalizacion resuelve un problema de comparacion que conviene registrar.** Sin ella,
escalar el bloque desplaza la escala de salida y los umbrales de V03 estan congelados, de modo que
un resultado negativo significaria dos cosas indistinguibles: que el mecanismo no ayuda, o que el
punto de operacion se movio. Conservando la norma de Frobenius se mantiene la escala global. La
tasa de positivos predicha de cada brazo se registro igual como diagnostico.

Los cuatro brazos hacen falta porque **un solo brazo no permite atribuir nada**: si la variante
gana, la pregunta inmediata es si gana por el ajuste por etiqueta o por la estructura de grupo. B
responde a la primera, D a la tercera, y D con un unico cociente compartido permitio medir que la
parte por etiqueta del cociente no aporta.

## D-087 - H1 CERRADA: el optimo del mecanismo es el punto de no hacer nada

**V12-01.** El brazo C llega a **+0.002738**, que supera el minimo exploratorio de +0.002, y sin
embargo **no se adopta**, y el motivo no es un fallo de gate cualquiera.

**El resultado lo sostenia un solo pliegue de cinco.** Sin el pliegue 4 el delta cae a
**+0.000134**. Y en ese pliegue los brazos C y D tienen constantes identicas y difieren **solo** en
el cociente de AG: +0.012705 contra -0.002670. Un volteo de 0.015 producido por una eleccion
discreta. Ademas la deteccion de C es **0.5196** frente al umbral de 0.5166: de 102 celdas
modificadas acertaria el 52 por ciento, es decir casi azar.

**La prueba decisiva es la curva del cociente, y separa dos explicaciones que el gate por si solo
no distingue.** Con la constante fija de V03, el maximo de la curva cae en el cociente **1.25** con
**+0.000006** sobre el valor neutro, y solo **1 de 6** cocientes no neutros bate al neutro. El
recorrido de la curva es de 0.012375, luego el mecanismo **si** tiene efecto real y grande sobre la
metrica, pero ese efecto esta **enterrado en el punto de no hacer nada**, y alejarse del neutro
empeora en cualquier direccion.

La distincion importa porque las dos explicaciones conducen a cierres opuestos: si la curva
tuviera un maximo separado del neutro, la conclusion seria que mi seleccion anidada fue ruidosa y
que la idea podria funcionar; como el maximo cae **sobre** el neutro, la conclusion es que **no hay
nada que seleccionar**. La estructura de penalizacion de V03 ya esta en su optimo.

**Lo que el experimento deja, mas alla del cierre**: la comparacion entre brazos separa el
reparto por primera vez. El ajuste por etiqueta aporta **+0.000417** y la estructura de grupo
**+0.002738**, y D con un unico cociente compartido da **+0.000636**, luego la parte por etiqueta
del cociente no aporta. Y queda contestada la parte de la hipotesis que decia que la dificultad
estaba en la estructura de regularizacion entre bloques: para el reparto palabra/caracter, la
estructura no era el cuello de botella.

**Se cierra H1 sin segunda particion**, porque no hay candidata a la que aplicarla.

## D-088 - El maximo de una curva diagnostica no es un resultado, y hay que decirlo en el informe

La curva del cociente se calcula mirando los pliegues exteriores, asi que **su maximo no es
adoptable**: proponerlo seria una busqueda sobre el conjunto de evaluacion, que es exactamente lo
que el gate exploratorio existe para impedir. Se reporta **solo** como diagnostico del motivo del
cierre, y el informe lo dice de forma expresa en la seccion que lo contiene.

## D-089 - Un diagnostico con bug puede invertir la conclusion, y por eso hay que contrastarlo

Un diagnostico intermedio mio concluyo que la constante de regularizacion **no tenia ningun efecto**
sobre el criterio interno, porque las cuatro constantes darian un valor identico. Si ese
diagnostico hubiera sido el que se reportara, la conclusion habria sido que el mecanismo entero es
inerte, lo cual es falso.

Se detecto porque el resultado era sospechoso: no puede dar predicciones identicas si la norma del peso pasa
de 2,8 a 43,8 con la constante de 0,1 a 16,0. La comprobacion directa
mostro que **C si cambia el modelo**, con 39 celdas que cambian entre C=0.1 y C=0.5. El bug estaba
en el diagnostico, que imprimia el mismo maximo cuatro veces por un bucle mal construido, y no en
el experimento.

La leccion es general y ya se habia aprendido en otra forma en este proyecto: **un numero que sale
demasiado redondo, o demasiado plano, es mas probablemente un bug que un hallazgo**. Por eso el
informe de V12-01 registra el fallo del diagnostico y su comprobacion, y no solo el resultado.

## D-090 - La estructura de etiquetas de TOXIC se midio por primera vez, y pesa mas de lo que sePensaba

**V12-02.** En TOXIC, **TR y AG solo estan definidas cuando HS=1**. La convencion de anotacion es
conocida, pero **nunca se habia medido cuanto pesa**:

- **835 de 1483 filas tienen HS=0**, el 56,30 por ciento.
- En esas 835 filas, **TR=0 en el 100 por ciento** de los casos.
- Por tanto, **el 80,44 por ciento de todos los negativos que el modelo de TR ve en entrenamiento
  son etiquetas que no estan definidas**, y el 88,76 por ciento para AG.

Lo que el modelo aprende con eso **no es una etiqueta ruidosa: es un concepto equivocado**. Se le
ensena que un tweet no toxico es "no insultante", que es un hecho trivial, y que un tweet no toxico
es "no agresivo", que es una afirmacion sin contenido. Esa es exactamente la situacion para la que
existe una perdida tolerante al ruido: **no se descarta el ejemplo, se le reduce la influencia
mediante la forma de la funcion**, que es lo que el proyecto nunca habia probado.

Ademenos se detectaron **22 filas con AG=1 y HS=0**, que son inconsistentes con la convencion y
que conviene tener presentes al interpretar cualquier resultado de AG.

## D-091 - El techo con oraculo es +0.046271, y localiza el problema con precision

Con la mascara de HS **verdadero**, la metrica sube **+0.046271**, con 196 celdas y deteccion
0,9286. Es **el techo mas alto medido en toda la historia del proyecto**, frente a una mejor ganancia
anidada historica de **+0.000148**. Ese techo **no es adoptable**, porque usa la etiqueta verdadera
de HS, y se reporta solo como limite de la via.

**El contraste con el post-proceso duro es lo que mas informa.** La mascara con HS verdadero da
+0.046271; con HS **predicho** da **-0.004187**. La diferencia entre ambas es exactamente lo que
cuesta usar la prediccion como llave, porque el HS predicho falla **143 filas** de 1483. Es decir:
**el post-proceso no falla por ser una regla dura, falla porque su llave se equivoca**. Eso explica
el -0.0056 historico con una mecanica que el cierre anterior no habia identificado.

Y explica tambien por que **no vale la pena reintentar la regla dura con otro umbral**: el problema
no es donde se pone el corte, es que la llave es una prediccion con 316 errores.

## D-092 - H2 MARGINAL: la perdida propia hace algo real, pero no llega al gate

**El brazo G, entropia cruzada generalizada, da +0.001616 en q=0,2**, con 4 de 5 pliegues a favor y
deteccion 0,6667, holgadamente por encima del umbral de 0,5166. **No pasa el gate** porque el delta
queda por debajo de +0,002 y el intervalo pareado incluye el cero.

**La curva diagnostica demuestra que es una meseta y no un pico**, que es la distincion que separa
"ruido no reproducible" de "efecto real pero pequeno": q en 0,15, 0,20 y 0,25 dan positivo, dos de
ellos superan +0,001, y la region es negativa a ambos lados.

**El contraste con H1 es el dato central de la fase.** En H1 el maximo de la curva caia **sobre** el
valor neutro con +0,000006, es decir el mecanismo no tenia nada que aportar. Aqui el maximo esta
**+0,001616 por encima** del neutro, un efecto **256 veces mayor**. La perdida propia **si hace
algo**; lo que no hace es llegar al gate. Y captura el **3,5 por ciento** del techo disponible.

**El brazo M, restringir el entrenamiento a las filas donde la etiqueta esta definida, es un
negativo limpio: -0.116581.** El motivo es concreto y no un fallo de implementacion: descartar el
56 por ciento de los datos **desplaza la escala de salida del modelo**, de modo que los umbrales
congelados de V03 dejan de tener significado para la subpoblacion que ahora se ha entrenado. No es
que la idea sea falsa: es que descartar datos no es un mecanismo de decision.

**Se clasifica MARGINAL, no adoptada.** El enunciado dice que una mejora pequena pero reproducible y
complementaria es evidencia cientifica, y esta lo es; pero el gate es el gate y no se adopta. No se
genera submission y no se ejecuta segunda particion, porque no hay candidata.

## D-093 - Dos correcciones propias durante V12-02, y por que la segunda importa mas

**La primera es un fallo de interfaz**: la funcion de perdida tomaba el vector de coeficientes al
final de la firma, de modo que la verificacion del gradiente pasaba mientras el optimizador recibia
los argumentos desordenados y fallaba. Se corrigio poniendo el vector **primero**, que es como la
invoca el optimizador, para que la verificacion y el ajuste **recorran el mismo camino de llamada**.
Un control que ejercita un camino distinto del que se usa en produccion no es un control.

**La segunda es un error de razonamiento, y es la importante**: yo afirmaba que en
``L_q = (1 - p_y^q) / q`` el valor ``q=1`` era exactamente la entropia cruzada. **Es invertido.** El
limite ``q -> 0`` es ``-log(p_y)``, la entropia cruzada, y ``q=1`` es ``1 - p_y``, el valor
absoluto. La familia interpola entre las dos.

**Lo detecto el control interno**, y de la manera mas util posible: ``q=1,0`` era el **peor** del
barrido, que es exactamente lo que corresponde a la perdida menos informativa. Un resultado que no
puede coincidir con su propia definicion es un aviso, no un dato. **El codigo estaba bien; mi
lectura de la familia estaba al reves**, y eso habria llevado a concluir que la perdida propia no hace
nada cuando en realidad hacia +0,0016.

La identidad se verifico numericamente: la diferencia entre mi perdida y el objetivo de sklearn
dividido por C cae de **3,21e-01 a 3,21e-05** al bajar q de 1e-2 a 1e-6. Y el control interno
definitivo es que **q=0,01 reproduce V03 con 3 celdas de diferencia**, lo que valida a la vez el
gradiente, el optimizador, el pipeline y los pesos de clase.

## D-094 - La perdida a medida exige verificar la derivada, y el orden de los argumentos es parte de la verificacion

El residuo maximo del gradiente analitico frente a diferencias finitas centrales es de **1,07e-07**
frente a una tolerancia de 1e-06, en los cuatro valores de q. Sin esa comprobacion, un gradiente
equivocado habria producido cifras plausibles y falsas, que es el peor fallo posible en un
experimento de este tipo: no un crash, sino un resultado que parece bueno.

**La leccion concreta es que la verificacion debe ejercitar el mismo camino de llamada que el
ajuste.** En V12-02 la verificacion pasaba mientras el optimizador fallaba, porque cada uno llamaba a
la funcion con un orden de argumentos distinto. Arreglarlo no fue cosmetico: fue lo que permitio
descubrir el segundo problema, el de la orientacion de la familia.

## D-095 - H3 CERRADA, y su modo de fallo coincide con el de V6-02 medido por otra via

**V12-03.** El control interno `lambda=0` da **+0.000053** con 3 celdas, lo que valida a la vez el
objetivo, el optimizador, el pipeline y los pesos de clase. El mejor valor de lambda da
**-0.002469** con deteccion **0,4915**, y el peor da **-0.007938** con deteccion **0,4523**: en ambos
casos **peores que el azar**, y muy por debajo del umbral de 0,5166. La curva es **monotona
decreciente** y su maximo cae en `lambda=0`, con un recorrido de 0,010311.

**El dato instructivo es el efecto sobre la dependencia HS hacia TR y AG**, que es lo que hace
especifica a esta hipotesis:

| brazo | F1 TR donde esta definida | FP de TR en zona no definida | FP de AG en zona no definida |
| --- | --- | --- | --- |
| V03 | 0,8941 | 78 | 104 |
| lambda=0,25 | 0,9024 | 84 | 121 |
| lambda=1,0 | 0,9091 | 97 | 130 |

El mecanismo **mejora la F1 de TR donde la etiqueta esta definida** y **empeora los falsos
positivos en la zona no definida**, que son exactamente los que el oraculo de H2 corrigio. Es decir:
hace bien lo que fue disenado para hacer **en local**, ordenar mejor las tres etiquetas dentro de
la subpoblacion toxica, y mal lo que la fase necesita.

**La causa es concreta y es una leccion sobre objetivos**: un objetivo de **orden** solo restringe
diferencias entre logits y **no dice nada sobre su nivel absoluto**. Al desplazar la presion directa
que la entropia cruzada ejerce sobre el nivel de `z_TR` y `z_AG` en las filas donde HS vale cero,
empeora justo ahi. Y como la metrica se evalua con **umbrales fijos**, el nivel absoluto si importa.
Un objetivo relativo no puede aportar la condicion que falta, porque la condicion que falta es
justamente de nivel.

**Y el modo de fallo coincide con el de V6-02**, que lo diagnostico como efecto de escala: el producto
`p(HS) p(TR|HS=1)` atenua `p(TR)` en toda fila con `p(HS)` intermedia, y V03 acertaba justo ahi.
Dos vias independientes, mismo mecanismo. Se cierra H3.

## D-096 - Un objetivo a medida verifico su derivada, y eso encontro cuatro fallos propios

En H3, **cuatro fallos** los detectaron las comprobaciones internas **antes de mirar un solo
resultado**, y los cuatro habrian producido cifras plausibles y falsas:

1. **La penalizacion L2 estaba en el gradiente y no en el valor del objetivo.** El residuo de la
   verificacion fue de 5,7e-02, que es exactamente del orden de `w / C`, es decir la contribucion de
   la penalizacion. Sin ese aviso, el ajuste habria optimizado una funcion cuyo valor no era el que
   su gradiente decia.
2. **El recorte del margen con `np.clip` rompia la regla de la cadena.** La funcion queda plana en la
   region recortada mientras el gradiente analitico sigue asumiendo que el margen varia. Se
   resolvio con dos formas estables **y exactas**: la probabilidad como
   `0,5 * (1 + tanh(m/2))` y la perdida con `logaddexp`. Ninguna recorta nada.
3. **La perdida por pares con `max(0, .)` tiene vertice**, y el optimizador abortaba con ABNORMAL en
   la iteracion 19 con independencia de las tolerancias, porque una busqueda lineal no puede
   funcionar sobre una funcion no derivable. Se sustituyo por el **surrogate logistico**, que es el
   estandar para objectives de orden por pares, es derivable en todo el dominio y no descarta del
   todo un par que viola, que es mejor aqui porque la estructura es suave.
4. **La busqueda lineal por defecto de 20 pasos era demasiado corta.** Con `maxls=100` converge en
   20 iteraciones y 48 evaluaciones, frente a 27 iteraciones y 165 evaluaciones del intento fallido.

Ademas un **fallo de diseno**, que es el mas instructive: el termino por pares era una **media** sobre
filas mientras la entropia cruzada era una **suma**, luego la razon entre ambos era de 1 a 3500 y
**lambda era irreconciliable**. El barrido daba el mismo resultado para lambda de 0 a 0,5, lo que no
es un resultado sino un termino despreciable. Igualada la escala, lambda vuelve a tener significado.
**La leccion: un hiperparametro solo es interpretable si los terminos que pondera estan en la misma
escala, y eso hay que comprobarlo antes de barrerlo, no despues.**

**El gradiente verifica con residuo maximo de 3,6e-07** frente a 1e-06, en los tres valores de
lambda y en dos puntos: uno aleatorio y otro **cerca de la solucion**, porque un gradiente puede
coincidir con las diferencias finitas lejos del optimo y fallar en el.

## D-097 - La hipotesis candidata NO es nueva: es V6-B, y V6-B esta cerrada con evidencia

**V12-04**, auditoria previa a cualquier ejecucion, leyendo el codigo y no el nombre. El encabezado
de `experiments/v6_02_label_factorization.py` dice literalmente **sin routing duro**, que no se
impone TR=0 cuando HS=0 en la prediccion y que **la restriccion viene del producto, no de una
regla**. El analisis estructural confirma **5 operaciones de producto y 0 condicionales cuya
condicion menciona HS**: no hay llave dura en ninguna parte.

- **B1** es `p(TR|X) = p(HS|X) * p(TR|HS=1, X)`, que es exactamente la formulacion candidata.
- **B2** es una **mezcla** con la probabilidad de HS, `p(HS|X) * p(AG|HS=1, X) + (1 - p(HS|X)) * p(AG|HS=0, X)`.
- Ademas V6-02 **tambien restringio el dominio de entrenamiento** del condicional a HS=1, que es
  la misma restriccion que probo el brazo M de V12-02.

Los cuatro resultados son negativos: B1 **-0.0107** con IC95 que excluye cero y 1 de 5 pliegues,
**1 fila rescatada frente a 35 rotas**; B1 con AG **-0.0115**; B2 **-0.0008** con IC95 que incluye
cero, 2 de 5, **0 rescatadas frente a 12 rotas**; B3 **-0.0115**. Ninguno supera ningun gate.

**La reconciliacion que cierra el diagnostico de la fase**: V6-02 mide **20 filas** en las que V03
predice TR=1 con HS **predicho** igual a 0, que es todo lo que una formulacion suave puede alcanzar,
frente a **78 filas** en las que V03 predice TR=1 y en realidad son HS=0, que es el techo del
oraculo. **Las 58 filas de diferencia son justamente las que `p(HS)` no puede identificar**, porque
HS se predice mal en 143 filas del desarrollo.

**La hipotesis candidata no se abre, porque no es nueva.** Abrirla seria repetir un experimento
cerrado, que es lo que la seccion 43 del enunciado prohibe.

## D-098 - La correccion conceptual que la fase conserva, y por que importa

**H2 y H3 no demuestran que la funcion de perdida no sea el problema en sentido absoluto.** Un unico
mecanismo positivo, el GCE de H2, **no autoriza a descartar la familia entera**, y descartar una
familia entera con un solo resultado positivo seria exactamente el sobreajuste por conclusion que la
fase ha querido evitar.

Lo que si autorizan a decir es mas preciso y mas util: **las variaciones de objetivo evaluadas hasta
ahora tienen un impacto pequeno comparado con el techo que introduce el error de la condicion de
HS**. El mejor caso, el GCE con q=0,2, da +0.001616 frente a un techo de +0.046271, es decir captura
el **3,5 por ciento**.

**La distincion importa porque las dos afirmaciones llevan a decisiones distintas.** La primera
cerraria una familia de forma definitiva y haria que nadie volviera a mirarla. La segunda establece
que el margen recuperable por esa via es pequeno **frente a otro problema ya medido**, y deja la
familia abierta para quien quiera explorarla con un mecanismo distinto. La segunda es la que se
escribe en el informe final, y es la unica honesta.

## D-099 - Cierre de la v12: 6 experimentos, 4 hipotesis, 0 candidatas, y un techo sin capturar

| ID | via | mejor delta | decision |
| --- | --- | --- | --- |
| V12-00 | auditoria de brecha | no entrena | CLOSED |
| V12-01 | H1 regularizacion por etiqueta y grupo | +0.002738 | CLOSED |
| V12-02 | H2a perdida propia | +0.001616 | MARGINAL |
| V12-02 | H2b restringir el dominio de entrenamiento | -0.116581 | CLOSED |
| V12-03 | H3 entrenamiento por pares | -0.002469 | CLOSED |
| V12-04 | auditoria de V6-B | no entrena | CLOSED |
| V12-05 | informe final | no entrena | CLOSED |

**El techo condicionado por la etiqueta verdadera de HS es de +0.046271**, y sigue sin capturar. Es
**23 veces** el gate exploratorio y **300 veces** la mejor ganancia anidada de toda la historia del
proyecto, que era de +0.000148.

**Y el cuello de botella NO es la ausencia de capacidad predictiva para TR y AG**: su F1 donde la
etiqueta esta definida es de 0,8941, muy alto. La dificultad es **inferir la condicion**, saber para
cada fila si HS es cero. Las tres vias para superarla fallan por razones distintas y documentadas: la
llave dura porque su llave se equivoca en 143 filas, la llave suave porque solo alcanza 20 de las 78,
y el acoplamiento en el objetivo porque atenua el nivel absoluto que los umbrales fijos necesitan.

**Los tres mecanismos del objetivo de entrenamiento son evidencia de rendimientos decrecientes**: no
es que la primera idea fuera mala y las siguientes peores, es que el eje entero tiene poco margen
recuperable frente a lo que queda disponible en otro sitio.

**Estado**: 6 experimentos, 0 candidatas, 0 submissions, 0 de 3 evaluaciones de holdout2, segunda
particion no ejecutada, V03 congelada e intacta. Registro en 234 filas con 234 claves y 23 campos.

## D-100 - La auditoria historica precede a la fase: 8 de los 11 mecanismos ya estaban cerrados

La fase de estimacion de HS se abrio con cuatro mecanismos propuestos, P1 a P4, y con la instruccion
de ejecutar la auditoria historica **antes de gastar presupuesto experimental**. La auditoria V13-00
es el primer experimento de la fase y no entrena nada, y su resultado es que **tres de los cuatro
mecanismos no eran nuevos**.

| mecanismo propuesto | historico | resultado historico |
| --- | --- | --- |
| P1 clasificacion conjunta de estados | V4-E, 7 variantes | 0.771289 frente a 0.777211, NO_SIGNAL |
| P2 meta de HS con scores OOF de TR y AG | E2-02 y E2-18 | 0.7853 y 0.7874, ambos ADOPTADOS y ambos rechazados por transferencia |
| P3 inferencia conjunta continua por scores | el mismo meta y V8-08 | 192 de 3462 celdas, sin superar el techo |
| P4 perdida de coherencia | H2 para la version dura, H3 para la de ranking | -0.004187 y -0.002469 |

**El hallazgo que mas condiciona la fase es que P2 ya se habia ejecutado y adoptado**: E2-02 dio
0.7853 y E2-18 dio 0.7874, ambos por encima del gate de candidato de +0.003, y ninguno se envio
porque el leaderboard fue **0.78163 frente a 0.78306** de V03. Reejecutar ese mecanismo habria sido
una violacion de la clausula de no reabrir familias cerradas, con el riesgo de transferencia ya
medido.

**Decision**: no se ejecutan P1 a P4. Se ejecutan los tres diagnosticos que la auditoria declara
genuinamente nuevos, que no son mecanismos sino mediciones ausentes del registro: el aislamiento de
la contribucion de TR y AG a HS, la descomposicion de errores por regimen, y la fraccion del techo
por mecanismo.

## D-101 - La invariante de TR es dura y la de AG no lo es, y el proyecto lo tenia al reves

La auditoria media por primera vez los estados conjuntos observados en el desarrollo, que es lo que
el enunciado exige antes de imponer ninguna regla logica a mano. Hay **6 estados** en 1483 filas:
000 con 813, 001 con 22, 100 con 79, 101 con 124, 110 con 24 y 111 con 421.

| invariante | violaciones | veredicto |
| --- | --- | --- |
| TR=1 implica HS=1 | **0** | DURA |
| AG=1 implica HS=0, es decir AG=1 con HS=0 | **22** | NO dura |

V6-01 reporto la invariante de TR como determinista con cero casos, y el registro de hoy lo confirma.
Lo que **no** era cierto era la lectura implicita de que ambas invariantes se comportaban igual. La
de AG tiene 22 filas reales que la rompen, y son estados del problema, no ruido.

**Consecuencia para P4**: una perdida de coherencia podria imponer legitimamente la invariante de
TR y **no** la de AG, porque hacerlo con la de AG seria tratar estados reales como si fueran ruido,
que es exactamente el tipo de decision que este proyecto ha rechazado siempre.

**Correccion propia, documentada**: mi primer codigo conto `violaciones_tr = estados["011"] +
estados["001"] = 0 + 22 = 22`, es decir sumo los estados 011 y 001 como si ambos violaran TR=1 con
HS=0, y eso contaba las violaciones de AG como si fueran de TR. Se verifico el conteo correcto con un
guion independiente **antes** de tocar el codigo de la auditoria. El error no era de los datos sino
mio, y la asimetria que revelo es la que hace util la medicion.

## D-102 - TR y AG SI llevan informacion sobre los errores de HS, y NO es utilizable

La medicion nueva de la fase. E2-02 y E2-18 entrenaron el meta por etiqueta sobre 12 probabilidades
OOF de 4 bases, que son 4 bases por 3 etiquetas, luego **el score propio de HS estaba siempre entre
las features** cuando se predicia HS. Eso hacia inobservable cuanto aportan TR y AG: el meta podia
predecir HS con su propio score y el efecto de las otras dos etiquetas quedaba mezclado.

Con el score propio excluido deliberadamente:

| sistema de puntuacion de HS | AUC | desviacion entre pliegues |
| --- | --- | --- |
| V03 con su propio score | 0.8677 | - |
| meta con su propio score mas TR y AG | 0.8710 | 0.0150 |
| meta **solo** con TR y AG | **0.8433** | 0.0096 |

**El aporte del score propio es de +0.0244 de AUC**, y el meta sin el score propio sigue muy lejos
del azar. Es decir: **TR y AG llevan informacion sobre los errores de HS**, y esa cifra no existia
en el registro historico.

**Pero no es utilizable.** El mismo brazo que alcanza 0.8433 de AUC **pierde 0.007541** en el
macro-F1, con la F1 de HS cayendo 0.022624. El brazo con el score propio rescata 20 filas de HS y
dana 1, y aun asi solo mueve +0.001603 en la columna HS y +0.000534 en el macro, por debajo del gate
exploratorio de +0.002 y muy lejos del gate de candidato de +0.003.

**La leccion es sobre la metrica, no sobre el mecanismo**: un AUC alto no es una buena decision en
el punto de corte, y en esta fase el umbral esta congelado en 0.44. La separacion entre capacidad de
discriminar y calidad de la decision es justo lo que el AUC oculta y lo que el gate mide.

## D-103 - El techo de +0.046271 NO esta en la estimacion de HS, y eso reencuadra la fase

El techo se ha leido durante dos fases como evidencia de que a HS le falta informacion. **Es falso**, y
lo demuestra el propio oraculo que lo produce:

| donde actua el oraculo | celdas | nota |
| --- | --- | --- |
| columna HS | **0** | no cambia ni una sola celda |
| columna TR | 78 | falsos positivos borrados |
| columna AG | 118 | falsos positivos borrados |
| filas afectadas | 134 | todas con HS verdadero igual a cero |

El techo proviene de **poner TR y AG a cero en las 835 filas donde el HS verdadero es cero**. Es un
efecto **aguas abajo** de la condicion de HS, y **ningun metodo que mejore la columna HS puede
alcanzarlo**, por mas bien calibrado que este.

**El corolario medido no admite matiz**: corregir la etiqueta de HS deja TR y AG con delta
**exactamente cero** en los dos brazos. El meta produce una probabilidad para HS y no altera lo que
el sistema decide sobre las otras dos etiquetas. Estimar mejor HS y mejorar el sistema son cosas
distintas en este problema.

**Y el regimen de los falsos negativos de HS, que es donde una correccion de HS pareceria funcionar bien, esta
fuera del techo por completo**: 143 filas con HS real uno y HS predicho cero, de las cuales 107 tienen
TR o AG verdadera a uno, y el oraculo **no toca ninguna de esas celdas**. Su regla dice algo de los
negativos de HS y nada de los positivos mal clasificados.

**Decision adoptada**: henceforth, ninguna hipotesis de mejora de HS se justifica por el techo de
+0.046271, porque ese techo no es alcanzable por ese eje. El techo solo se cita como diagnostico del
coste de la condicion, que es lo que es.

## D-104 - Cierre de la v13: 2 experimentos, 0 candidatas, y una linea cerrada formalmente

| ID | via | mejor delta | decision |
| --- | --- | --- | --- |
| V13-00 | auditoria historica de HS | no entrena | CLOSED |
| V13-01 | aislamiento de la evidencia de TR y AG sobre HS | +0.000534 | NO ADOPTADA, marginal |
| V13-02 | informe final y cierre de linea | no entrena | CLOSED |

**La linea de inferencia estructurada de etiquetas queda CERRADA FORMALMENTE**, con la formulacion
que el propio enunciado preve para este caso. No se afirma que TOXIC no pueda mejorar. Se afirma,
mas estrecho: que la inferencia estructurada condicionada por HS no es una via rentable con los
mecanismos permitidos, porque el techo no esta en la estimacion de HS sino en la aplicacion de la
condicion de HS a las decisiones de TR y AG, y esa via ya estaba cerrada por medicion desde v12, con
el post-proceso duro en -0.004187 porque su llave se equivoca en 143 filas.

**Lo que la fase aporta y no era mecanica son tres diagnosticos que no estaban en el registro**: la
separacion del aporte del score propio de HS, que es +0.0244 de AUC; la descomposicion de los
errores en los cuatro regimenes de verdad y predicha, con el conteo de falsos positivos de TR y AG en
cada uno; y la procedencia del techo, que reencuadra dos fases de trabajo previo.

**Estado**: 2 experimentos de 8 principales y 0 de 4 de reserva, 0 adoptables, 1 marginal, 1
rechazada, 0 submissions, 0 de 3 evaluaciones de holdout2, segunda particion no ejecutada, test no
utilizado, V03 congelada e intacta con identidad verificada en cada ejecucion. Registro en 237 filas
con 237 claves y 23 campos.

## D-105 - La perdida enmascarada de P1 ES H2-M, con diferencia exactamente cero

El enunciado de V14 advierte que la perdida con TR y AG enmascaradas donde HS=0 **no debe
confundirse con H2-M**, que eliminaba filas, y propone P1 precisamente para no repetirlo. La
auditoria V14-00 demuestra lo contrario.

| comprobacion | valor |
| --- | --- |
| peor diferencia de coeficientes entre H2-M y la enmascarada | **0.000e+00** |
| peor diferencia de probabilidades sobre la validacion | **0.000e+00** |
| delta del macro-F1 de la enmascarada | **-0.116581** |
| delta del macro-F1 de H2-M reconstruido | -0.116581 |
| delta de H2-M registrado en v12 | -0.116581 |

**La razon es una sola observacion de optimizacion**: una fila con peso cero en la perdida no aporta
gradiente, luego el vector de coeficientes lo determinan unicamente las filas con peso uno, que son
exactamente las que H2-M conserva. Enmascarar y eliminar dejan el mismo subconjunto de filas
aportando senal, y por eso producen el mismo estimador.

**Lo unico que puede distinguirlas es como se reescala el termino de perdida**, es decir la
normalizacion y los pesos de clase. Eso no es un mecanismo de aprendizaje: es una constante de
escala, y pertenece a la familia de calibracion y umbrales, que ya estaba cerrada.

La prueba que cierra la fase no es la identidad de estimador, que convence, sino que **la
formulacion nueva reproduce un resultado negativo ya registrado, hasta el ultimo decimal**.

**Decision**: se cumple la condicion de parada numero 1 del enunciado. No se ejecutan P1 a P4, y la
fase queda cerrada con 0 de 4 experimentos principales y 0 de 2 de reserva.

## D-106 - Mi primera verificacion de la equivalencia fue erronea, y la segunda no

La verificacion inicial uso **una sola pliegue** y concluyo que la identidad era exacta, con
diferencia 0.000e+00. Al extenderla a los cinco pliegues aparecieron diferencias de orden 4.5e-01.

**La causa era mia**: en la reconstruccion de H2-M pasaba `sample_weight` nulo, es decir **sin pesos
de clase**, cuando el brazo M real usa `class_weight="balanced"` sobre la submuestra de HS=1. Estaba
comparando H2-M contra un ajuste sin ponderar, que no es H2-M, y por eso mi H2-M reconstruido daba
-0.132999 en lugar de -0.116581.

**Lo que lo detecto** fue comprobar que el delta reconstruido coincidiera con el registrado en v12.
Sin esa comprobacion la conclusion erronea habria pasado, porque un numero pequeno no delata un
modelo mal montado.

**Leccion, que se conserva**: un veredicto de equivalencia no se sostiene en una sola pliegue, y
**reproducir un resultado historico conocido es la comprobacion que detecta un error de montaje**.

## D-107 - La conclusion comoda esta prohibida: la alternativa no fallo, es el mismo estimador

Seria comodo concluir que las etiquetas de TR y AG en el regimen HS=0 deben seguir modelandose como
negativos ordinarios, porque la alternativa fallo con -0.116581. **Eso no puede afirmarse**, y la
razon es tecnica: la alternativa **no ha fallado, ha resultado ser el mismo estimador** que la
formulacion que si fallo. No hay ninguna prueba en contra de la semantica de observacion
parcialmente definida, del mismo modo que no hay ninguna a favor.

Lo que si queda establecido es una **restriccion de lo construible**: cualquier mecanismo que se
implemente enmascarando la perdida de clasificacion sobre las filas HS=0 va a reproducir H2-M, con
su cambio de escala y su -0.116581. Para apartarse de ahi hace falta una senal distinta de cero y sin
etiqueta en esas filas, y obtenerla exigiria datos externos, una representacion nueva o informacion
transductiva, **los tres prohibidos por la jerarquia de la fase**.

Las preguntas 1, 2 y 5 de la seccion 12 del enunciado quedan por eso **sin respuesta por evidencia**,
y asi se declaran.

## D-108 - Los cuatro mecanismos "nuevos" no existen en el registro, y eso tambien se cuantifica

El barrido de los doce mecanismos que el enunciado exige buscar da un resultado que conviene hacer
visible en vez de dejar como nota al pie: **la sonda textual se marca a si misma**, porque el guion
de auditoria nombra literalmente los doce mecanismos que busca.

| mecanismo | guiones historicos | conteo bruto |
| --- | --- | --- |
| aprendizaje con etiquetas parciales | **0** | 1 |
| positive-unlabeled learning | **0** | 1 |
| selective loss | **0** | 1 |
| masked BCE | **0** | 1 |

Los cuatro tienen **cero menciones historicas** una vez descontada la autorreferencia de la propia
consulta. Es el patron que v12 documento por primera vez, aqui con su cuantificacion: cada conteo se
reporta en dos columnas, con autorreferencia y sin ella, y la segunda es la que vale como evidencia.

La sonda estructural confirma el lado positivo: **13 llamadas con `sample_weight`** en todo el codigo,
y ninguna define los pesos por la indefinicion de la etiqueta. `calcular_pesos` de V4-B05 pondera por
discrepancia entre prediccion y verdad, es decir por el error del modelo. `pesos_dureza` de V4-C
viene del mapa de dureza. `pesos(tamanos, esquema)` de V6-03 pondera por tamano de grupo. V6-07 actua
sobre el ruido del par. `pesos_por_clase` de V8-07 reescala por clase, que es lo que P5 ya midio en
+0.0023.

**La unica operacion del historial que condiciona la senal de TR y AG a HS=1 es H2-M**, y es
exactamente la que P1 reproduce.

## D-109 - Cierre de la v14: 1 experimento, 0 ejecuciones, y una linea cerrada

| ID | via | resultado | decision |
| --- | --- | --- | --- |
| V14-00 | auditoria de novedad partially-label | P1 ES H2-M, diferencia 0.000e+00 | FASE CERRADA |
| V14-01 | informe final y siete respuestas | 0 ejecuciones | CLOSED |

**La linea de entrenamiento partially-label de TR y AG en el regimen HS=0 queda CERRADA.** Y se cierra
en la forma que el propio enunciado establecia para este caso, con la formulacion que v13 ya habia
dejado establecida y que V14 confirma: el techo del oraculo procede de conocer la condicion semantica
HS al evaluar TR y AG, pero mejorar HS por si sola no produce beneficio, y las familias historicas de
routing, factorizacion, powerset, ranking y meta-combinacion estan cerradas.

**Lo que queda abierto, y no es una hipotesis de esta linea**: una formulacion que aportase una senal
distinta de cero y sin etiqueta a las filas HS=0. Caeria fuera de la jerarquia actual, y por eso
requiere una decision de gobierno distinta, no un experimento mas.

**Estado**: 1 experimento, 0 de 4 principales y 0 de 2 de reserva, 0 ejecuciones, 0 adoptables, 0
marginales, 1 rechazada, fraccion del techo del oraculo recuperada -2.52 de signo negativo, 0
submissions, 0 de 3 evaluaciones de holdout2, segunda particion no ejecutada, test no utilizado, V03
congelada e intacta. Registro en 239 filas con 239 claves unicas y 23 campos.

## D-110 - La categoria E no existe: no hay user ID, ni fecha, ni perfil

La primera comprobacion de V15, y la que cierra de raiz la via de identidad y metadata. El fichero
bruto `data/raw/toxic_train.csv` tiene exactamente la cabecera **`id;text;HS;TR;AG`**, y el
cargador `read_train` no descarta ninguna columna. No hay user ID, ni autor, ni marca temporal, ni
perfil, ni ningun campo de identificacion.

La unica columna no textual es `id`, con valores 20002, 20003, 20005 y huecos, es decir **un contador
de fila**. Y ademas la seccion 3.E del enunciado prohibe introducir user ID, fecha, perfil o
metadata externa, luego esta doblemente descartada: por no existir como senal y por estar
prohibida.

**Lo que si hay son datos dentro del texto**: correos, fechas como `16/03/2017` y telefonos como
`V-13.899.989`. Eso no es metadata, es texto, y por tanto ya llega a V03 a traves de char_wb(3,5).

**Decision**: la categoria E se cierra por inexistencia y no por resultado. No queda nada que
auditar en ella.

## D-111 - V03 tiene dos propiedades demostrablemente incompletas, y de la primera sale la unica clase no cubierta

De la representacion de V03 se siguen dos propiedades, ambas demostrables y no conjeturas:

1. **La posicion absoluta en el documento no esta representada.** `TfidfVectorizer` extrae n-gramas
   de cada palabra sin registrar en que posicion del documento esta esa palabra, y `char_wb` hace
   lo mismo dentro de cada palabra. Dos textos con las mismas palabras en distinto orden producen
   el mismo vector.
2. **Las relaciones entre palabras distantes no estan representadas.** word(1,2) solo observa una
   ventana de dos tokens, y un modelo lineal sobre TF-IDF no tiene terminos de interaccion.

De la primera se sigue una clase de senal que V03 no puede expresar: **las senales que dependen de
donde aparece algo, no de si aparece**.

**Y hay que ser preciso sobre el estado de esa clase**, porque no es nueva: `pos_mencion_inicial` fue
feature de modelo en `16_targeting_v2.py` linea 1204, y `mencion_inicial` aparece en 8 guiones mas
entre los que estan `20_emocion_nbsvm_ortografia_v2.py` y `08_features_densas_h11_h12.py`. La
sonda de V15-00 se excluye a si misma, por el motivo que v12 documento y v14 cuantifico.

Lo que **no** se ha probado es la posicion como **columna de V03**, que es donde V03 toma su
decision. Y esa es la unica intervencion que la fase autoriza, porque es la unica que no cambia de
modelo, ni umbrales, ni representacion, ni regularizacion.

## D-112 - La medicion que decide es el AUC estratificado, y dos mediciones habrian sido invalidas

Antes de medir, hay que descartar dos mediciones que parecen razonables y no lo son.

**El AUC global de un rasgo no decide nada.** Un AUC de 0.5598 contra HS es perfectamente compatible
con que V03 ya tenga ese 0.5598 dentro de su prediccion. Un rasgo puede ser informativo y no aportar
nada.

**El R2 de predecir el rasgo desde los margenes de V03 tampoco decide nada.** Tres numeros no pueden
reconstruir cuatro rasgos, luego el R2 sale bajo **por construccion** y parece una noticia buena
que no lo es. Se descarto antes de escribir el experimento definitivo.

**La medicion valida es el AUC estratificado por el margen de V03**: si el rasgo discrimina dentro de
los estratos donde V03 ya ha decidido, aporta informacion que V03 no tiene. Y como referencia, el
null se construye **permutando el rasgo dentro de cada estrato 500 veces**, conservando su
distribucion y destruyendo su relacion con la etiqueta, con valor p empirico.

**Correccion propia**: la primera implementacion hacia **una sola permutacion**, que es un punto y
no un null, y su varianza es la del estimador. Se sustituyo por las 500 permutaciones antes de
interpretar nada. Ademas solo se probo en P4 la primera de las tres candidatas que pasaron P3, y se
probaron las tres.

## D-113 - La senal residual existe, es complementaria, y es inutilizable: acierto CERO en las celdas modificadas

Este es el hallazgo de la fase, y tiene tres partes que no hay que mezclar.

| rasgo anadido a V03 | AUC en estratos | p empirico | delta | celdas | acierto entre las modificadas |
| --- | --- | --- | --- | --- | --- |
| primer_token_es_mencion | 0.5547 | 0.0000 | -0.002868 | 50 | **0.0000** |
| posicion_primera_mencion | 0.5491 | 0.0000 | -0.001352 | 14 | **0.0000** |
| mencion_en_primeros_dos | 0.5547 | 0.0000 | -0.003037 | 57 | **0.0000** |

**La senal existe y ve algo que V03 no ve.** Con un valor p empirico por debajo de 1 entre 500, la
complementariedad es real y estadisticamente solida. Eso responde literalmente a la pregunta de la
seccion 12.

**Y no es utilizable.** El filtro de detectabilidad de V8-02 exige un acierto entre las celdas
modificadas superior a 0.5166. El observado es **0.0000 en las tres candidatas**: de 50, 14 y 57
celdas modificadas, **ninguna** coincide con la verdad. No es una mejora que no llega al gate, es un
cambio que danan sin ambiguedad. La informacion ordena bien a los ejemplos y en el punto de corte
del umbral congelado empuja en el sentido contrario.

**Es la tercera estimacion independiente que coincide**: los rasgos estructurales de E2-18 costaron
-0.0052 y -0.0051, y los rasgos de posicion cuestan entre -0.0014 y -0.0030 con acierto cero.

**La leccion general, que es mas util que el resultado**: una fuente de senal puede ser real,
complementaria y estadisticamente significativa, y a la vez ser inutilizable en el punto de
decision. El AUC mide capacidad de discriminar; la metrica mide calidad de la decision; y no se
puede pasar de la una a la otra.

## D-114 - TOXIC se cierra en V03: estado CLOSED

**CLOSED. V03 continua siendo la mejor solucion reproducible bajo las restricciones actuales.**

De doce clases de senal inventariadas, **tres no existen** como campo, **cuatro ya estaban medidas**
y cinco estan en familias cerradas, incluida la unica que V03 no puede representar por construccion.
No queda una fuente de senal permitida y genuinamente nueva identificada en el input disponible.

**El dato que sostiene la decision y que conviene no olvidar**: el mejor resultado local de toda la
historia, 0.8465 en S08, **no se adopto** porque el externo fue 0.78163 frente a 0.78306 de V03. V03
sigue siendo la mejor solucion no por falta de intentos, sino por **evidencia en contra medida**, y
en un proyecto de 242 experimentos eso es una conclusion fuerte.

**No se abre una V16 para buscar microganancias.** La seccion 14 del enunciado prohibe la dinamica
de experimento negativo seguido de reformulacion, y las tres adiciones de V15 son negativas con
acierto cero, que es el caso mas limpio de ese patron.

**Estado**: 3 experimentos, 0 adoptables, 0 marginales, 3 rechazadas, 0 submissions, 0 de 3
evaluaciones de holdout2, segunda particion no ejecutada, test no utilizado, V03 congelada e intacta
con identidad verificada. Registro en 242 filas con 242 claves unicas y 23 campos.


> **Nota de correccion (P46, 2026-09-28).** Estas dos decisiones se escribieron inicialmente como D-030 y D-031, que ya estaban ocupados por decisiones de la fase v5 (alcance de datos externos y estado de la representacion). Se renumeran a D-115 y D-116. El contenido no cambio.

### D-115. La metrica local es la macro de 3 etiquetas, y `samples` queda excluido

**Contexto.** P45 abria con la pregunta de si `corrected_cv` coincide con la metrica del
host. Sin resolverlo, todo delta comparativo posterior carecia de base.

**Evidencia.** `f1_media_columnas` es identico a `f1_score(average="macro")` sobre entrada
multilabel: diferencia `0.000e+00`, 0 discrepancias en 600 casos aleatorios. `corrected_cv`
se reproduce con diferencia `+0.000e+00`. La media de pliegues y el global OOF difieren en
`+0.0001002686` y no son intercambiables; el proyecto reporta el global.

**Decision.** `METRIC_MATCH = TRUE` para `macro`. Se declara `samples` **excluido** por el
discriminante de escala: da 0.334 sobre el OOF de V03, mientras el host reporto 0.78306 para
la submission de V03, y una brecha de 0.449 no se explica por desplazamiento de prevalencia.

**Consecuencias.** Se mantiene `macro_3` como metrica de seleccion. La distincion residual
entre `macro`, `micro` y `weighted` (menos de 0.0036, spearman >= 0.98 en el ranking de
umbrales) se registra como pendiente menor, no como bloqueante.

**Lo que no se decide.** No se afirma que el host use `macro` de forma directa. El codigo del
scorer no esta en el repositorio.

### D-116. La estructura conjunta es real pero no explotable: CLOSED

**Contexto.** P45-B preguntaba si existe una formulacion de prediccion conjunto genuinamente
nueva, no equivalente a las ya probadas.

**Evidencia.** Los estados `010` y `011` nunca se observan (`TR=1` implica `HS=1` en los 265
casos de train con `TR=1`). Los errores de las tres etiquetas **no** son independientes dado
el estado verdadero: chi2 343.156, gl 20, p < 1e-6, con desviaciones de hasta 116 casos en una
celda.

**Decision.** `JOINT_LABEL_STRUCTURE = CLOSED`.

**Consecuencias.** La estructura conjunta existe pero no se traduce en ganancia. La extraccion
mas directa, la correccion estructural dura, es negativa en las dos formas (-0.000646 y
-0.003542). Label Powerset, classifier chains en ambos ordenes, espacio por etiqueta,
acoplamiento suave y meta-modelos estan todos registrados con resultado negativo o sin
transferencia. **No se abre ninguna familia conjunta nueva** y no se implementa ningun decoder
estructurado.

**Alternativas descartadas.** Decoder con funcion de energia explicita sobre estados validos:
se descarta porque su unica diferencia respecto a la correccion dura es el peso, y la version
de peso infinito ya es negativa. Ajustar el decoder a un subconjunto de estados sin
entrenamiento adicional es la misma regla.

**Nota de honestidad.** El pearson(`local_f1`, `kaggle_f1`) es 0.4876, no alto. La
transferencia local->leaderboard es ruidosa, coherente con que 0 de 22 submissions superaran a
V03. El discriminante de escala no depende de esa correlacion: se apoya en el valor absoluto
que el host reporto.


### D-117. El campeon verificado es V03 con LB 0.78306; los 0.78535 y 0.78650 eran predicciones

**Contexto.** `CHAMPION.md` atribuia a `v16_ch1` un LB de 0.78535 y a `v16_ch5` un LB de
0.78650, mientras `submissions/leaderboard_log.csv` registraba como maximo 0.78306. Si la
afirmacion de CHAMPION.md fuera cierta, V03 no seria el campeon, y todas las decisiones del
proyecto estarian apoyadas en una referencia equivocada.

**Jerarquia de evidencia aplicada.** 1. archivo de submission, 2. registro externo asociado,
3. leaderboard log estructurado, 4. metadata de experimento, 5. reports, 6. notebook,
7. CHAMPION.md, 8. texto narrativo.

**Evidencia.** La tabla de CHAMPION.md de la seccion 4 se titula, en su propio encabezado,
**`LB combinado`**, y el fichero declara la formula que produce esos numeros:
`0.78306 + 0.12011 x delta_tasa + aciertos_netos_oof x 2.804e-05`. Recalculada con los
parametros que el propio texto da, reproduce 0.785342 y 0.786495, con desviaciones de
7.9e-06 y 5.2e-06 respecto a los valores afirmados. **Son predicciones, no mediciones.**

Las mediciones reales de esas mismas submissions son 0.77828 y 0.77495, y estan en la seccion
*Estado al cierre de V16* del mismo CHAMPION.md, marcadas REJECTED. Los 22 registros del log
tienen artefacto en disco, sin duplicados, sin nulos y sin valores fuera de rango, y ninguno
supera 0.78306.

**Decision.** `CASE B`.
- `leaderboard_max_verified = 0.78306` (`v03_lr_union_thr`).
- `baseline_operativo = V03`, sha256 `7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e`.
- `CHAMPION_STATUS = INCONSISTENTE_AUTOCORREGIDO`: no hay dos fuentes en conflicto, hay una
  prediccion y una medicion en el mismo fichero, y la medicion es la que vale.

**Accion sobre CHAMPION.md.** No se borra historia. Se anade un aviso que marca la columna como
prediccion, se renombra el encabezado a `LB combinado (PREDICTO, no medido)`, se etiqueta la
formula como prediccion, y se ancla el estado verificado en la cabecera del fichero.

**Hallazgo secundario, relevante para el registro.** La columna `macro_f1` de
`experiments.csv` **mezcla tres bases de medida**: 4 filas sobre el holdout2 de 248 filas y 46
sobre el split D-004, supersedido por D-015/D-016. Las filas historicas de V03 registran
0.8322 y 0.8356 en esa base, no 0.7772114525. Ordenar esa columna sin clasificar la base
produce rankings invalidos: un maximo ingenuo devuelve S08 con 0.8743, que es una medida de
holdout. La magnitud de CV utilizable es la del OOF canonico, y se lee del artefacto congelado
`features/oof_v03_final.npz`, no de esa columna.

**Lo que no se decide.** No se reescriben las filas historicas del registro para "corregir" su
base. Cambiar 46 valores historicos seria reescribir historia sin evidencia de que el valor
nuevo sea el que se midio en su momento.


### D-118. No existe expansion legal de datos, y la augmentation ya estaba cerrada

**Contexto.** P46 dejo como recomendacion unica plausible "mas datos etiquetados", admitiendo
que no se habia demostrado que esa via fuera legal y disponible. P47 audita si lo es.

**Evidencia, recountada sobre los ficheros.**

| magnitud | valor | fuente |
|---|---:|---|
| train oficial etiquetado | 1731 | `data/raw/toxic_train.csv`, sha256 `3d40e7345b588860...` |
| dev, usado por V03 | 1483 | `holdout_v2_ids.csv`, `particion == dev` |
| holdout, agotado | 248 | `holdout_v2_ids.csv`, `particion != dev` |
| train sin asignar | **0** | 1731 = 1483 + 248 |
| test oficial | 1154 | `data/raw/toxic_test.csv`, sin labels |
| holdout2 | 247 | dibujado **de dev**, 0 dentro del holdout |
| corpus externos | 42808 docs / 95.014.116 tokens | legales solo para embeddings (D-030) |

**Correcciones de cifras que esta auditoria deja fijadas.**

1. **519 no es el tamano del holdout.** 519 es el numero de positivos TR en train. El holdout
   tiene 248 filas. La cifra 519 circulaba en la documentacion reciente como si fuera el
   holdout.
2. **No existen ~44k filas sin etiquetar.** Los corpus externos se miden en **tokens**, no en
   filas, y van de 5M a 50M. Ningun dataset del proyecto tiene 44k filas. El "44" que circula en
   el repositorio son las **44 features** del meta-modelo `E2-02-META`.

**Decision.** `DATA_EXPANSION = CLOSED`. El train esta completamente asignado. La recomendacion
de "mas datos etiquetados" **no es operacionalmente posible** con este corpus, y se retira como
via.

**Sobre la augmentation.** La directiva la abria solo si se confirmaba que la familia no se
habia ejecutado. Se ejecuto: **V6-08** (`experiments/v6_08_aumento.py`, sha256
`7cc796cbc18ac70ee4614b7f329f524ecc1b9283e80ef49f933bf7575af02acb`, existe) implemento `espaciado`, `elongacion` y `repeticion_puntuacion`, que son
exactamente los candidatos ortograficos de la directiva, con la augmentation aplicada solo a
los indices de entrenamiento del pliegue y con los grupos preservados. El control J0 reproduce
V03 exactamente. J1 dio **-0.0061** y J2 tambien negativo. `TRAIN_AUGMENTATION = CLOSED`.

**Lo unico nominalmente sin equivalente** segun la auditoria previa V41-01 son *word dropout
conservador* y *masking estructural*. No se abren: la directiva acota los candidatos a
transformaciones ortograficas de alta confianza, y ambas tocan el contenido de las palabras, que
es lo que su propia seccion 12 prohibe por riesgo de alterar la etiqueta. Con el precedente
adverso de V6-08 y de las diez normalizaciones PRE-00..PRE-09, abrirlas seria gasto sin
hipotesis.

**Consecuencia.** No queda ninguna via de datos. La recomendacion de P46 queda cerrada por
evidencia, no por prudencia.


### D-119. El EDA se convertirio completo en hipotesis, y el hueco de investigacion es nulo

**Contexto.** P48 pone a prueba la hypothesis de que el proyecto no mejora porque el EDA
original no se convirtio en mecanismos experimentales, en vez de porque no quede senal.

**Metodo.** Dos piezas, ambas sobre artefactos existentes, sin entrenar nada.

1. Trazabilidad: 15 hallazgos del EDA original, cada uno con el mecanismo que deriva de el,
   buscado en las 318 filas del registro de experimentos.
2. Atlas de residuos: las 757 celdas de error de V03 sobre el OOF canonico, cruzadas con la
   fraccion de fallo por celda de los 20 canales OOF historicos, congelada en
   `reports/hardness_v4.csv`.

**Resultado 1: la trazabilidad es casi total.** **14 de 15 hallazgos del EDA tienen mecanismo
probado y cerrado**, y **ninguno se quedo sin un solo experimento**. El unico abierto es la
rareza de etiqueta, con un unico experimento. Ademas, el informe EDA de origen solo declara
hipotesis explicita en 3 de sus 6 hallazgos clave; los otros 3 se convirtieron en experimentos
por otras vias.

Criterio de cierre aplicado: predominio de veredictos **sustantivos** de cierre sobre los de
apertura, excluyendo filas `AUDIT_ONLY` y `DIAGNOSTIC_ONLY`, que no se refieren al mecanismo. Un
primer criterio que usaba la ultima fila coincidente daba 7 hallazgos abiertos, y era erroneo:
para "grupos de near-duplicate" la ultima fila era la auditoria del campeon de P46.

**Resultado 2: no hay poblacion rescatable escondida.** De las 757 celdas de error de V03,
**715 (94,45 por ciento)** son acertadas por al menos un canal historico, y **42 (5,55 por
ciento)** no lo son por ninguno. La tasa de error global es 0,2974 y la de rescate 0,9445.

**El punto decisivo: ninguna slice se enriquece en rescate.** Se evaluaron 24 slices y
combinaciones de las ya existentes en el EDA. Varias tienen tasa de error claramente superior a
la media, con lift de 1,21 a 1,60 y p por debajo de 0,01: cita, colectivo, negacion, longitud
larga, elongacion, hashtags. **Pero su tasa de rescate es igual o inferior a la global.** Ninguna
supera 1,0 en el cociente de enriquecimiento, y varias quedan significativamente por debajo: colectivo en falso negativo de HS 0,882, negacion 0,929, longitud larga 0,955.

Eso invalida la lectura intuitiva de "hay subpoblaciones con senal oculta". La prueba que las separa
es precisamente el enriquecimiento en rescate, y ninguna lo supera.

**Sobre el margen.** La tasa de rescate cae de forma monotona con el margen de V03: 0,9929 con
margen por debajo de 0,10 y 0,8657 con margen de 0,20 o mas. Y **el 85,7 por ciento de las
celdas irrecuperables estan en margen alto** (36 de 42). Esa es la firma de un hueco de
representacion: V03 se equivoca con conviccion justo donde ningun canal acierta. Pero son **42
celdas de 757**, un 5,55 por ciento. Ninguna hipotesis se sostiene sobre esa base.

**Decision.** `EDA_RESEARCH_GAP = NONE`. No se propone ninguna hipotesis y no se ejecuta ningun
experimento. Se consume 0 de los 3 experimentos de presupuesto.

**Lo que esto responde a la pregunta de la directiva.** Seguimos sin mejorar **porque ya no
existe senal utilizable**, no porque nunca convertimos el EDA en mecanismos. Las dos ramas de la
pregunta se resuelven: la conversion fue completa, y la senal tampoco esta.


### D-120. La contradiccion 147/757 frente a 715/757 son dos magnitudes distintas, y ninguna es azar

**Contexto.** P48 presento 715 de 757 celdas de error de V03 como "rescatables" y, en la
misma fase, cito 147 de 757 como `cobertura_de_errores_v03`,inionando que los tensores OOF por
canal no estaban en disco y que 715 era una cota superior. P49 exige resolverlo antes de
cerrar nada.

**Los tensores eran recuperables.** Estan en `/home/carlos/Git/PLN/kaggle_toxic_2026_puj/features/`
y en `/home/carlos/Git/PLN-v8/kaggle_toxic_2026_puj/features/`. Reconstruido el pool con la
misma regla de exclusion que `experiments/28_hardness_v4.py` (se descartan `TXT+*` y
`CH-TGT+DIST`), salen **20 canales**, y la fraccion de fallo por celda coincide con
`reports/hardness_v4.csv` en **4449 de 4449 celdas, con diferencia 0.00e+00**.

**Que significa cada cifra.**

| cifra | unidad | definicion |
|---|---|---|
| **757** | label-cell | celdas (tweet, etiqueta) con `pred != y_true` sobre 1483 x 3. **No** son tweets con error: son 441 filas |
| **147** | label-cell | celdas de error de V03 que acierta **una sola** variante, `C_meta36_downweight` con umbral 0.5. Es la `cobertura_de_errores_v03` de V17-00 |
| **715** | label-cell | celdas de error de V03 que acierta **al menos uno de los 20 canales** de la fase v2, binarizados a 0.5 |

Las dos son observacion directa y **no son comparables**: 147 es el rescate de un modelo, 715
es el de la union de veinte. El 147 **se reproduce exactamente**: 147 rescates, 90 danos,
cobertura 0.194187582562748.

**No es azar.** Si los 20 canales fallaran de forma independiente, con la probabilidad marginal
observada, el numero de celdas de error de V03 en las que ninguno acierta seria **1,20**. El
observado es **42**, es decir 35 veces mas. Un canal acierta una celda de error de V03 con
probabilidad 0,2756 (5,51 de 20). Los canales fallan **juntos** con V03, que es exactamente la
firma de un hueco de representacion compartido.

**Pero el rescate no se puede cobrar.** De los 20 canales, **19 tienen net negativo**. El
unico positivo es `base:M2_union_lr_c2` con +11, que es el propio V03 con umbral 0.5 en lugar
de 0,44/0,51/0,46, luego no es un modelo alternativo sino una perturbacion de umbral. El
canal con mas rescates, `cita:NEG`, rescata 337 y dania 2043: **net -1706**.

**Decision.** `CASE A`. `OBSERVED_RESCUE = 715`, `UPPER_BOUND` no aplica a 715. Y una
conclusion nueva y mas fuerte que la de P48: **el rescate existe, es real y esObservable, pero
es difuso y no identificable**, porque ningun canal individual tiene net positivo y no hay
informacion por celda que diga cual de los veinte conviene escuchar.

**Correccion de P48.** Dos elementos quedan **superados**: la limitacion que declaraba los
tensores no recuperables, y el criterio de enrichment que solo miraba la tasa de rescate. El
resto, incluidas las cifras 757, 715 y 42, queda **verificado**.

**Lo que P48 dio por bueno y P49 corrige en el otro sentido.** El margin sigue siendo valido
porque se puede repetir sobre el rescate observado, y coincide: la tasa de rescate cae de
1,0000 con margen bajo a 0,8472 y 0,8182 con margen alto en TR y AG. Y las slices cambian de
diagnostico al anadir el enriquecimiento en irrecuperables: `negacion` tiene lift 1,921, o sea
se concentra en errores **irrecuperables**, mientras `hashtags` tiene lift 0,245 y se
concentra en errores **rescatables**. P48 las declaro a todas indistinguibles porque solo
midio una de las dos direcciones.


### D-121. La senal existe y es enorme, pero no es identificable: el techo es +0,2098 y el selector pierde 0,0115

**Contexto.** P49 cerro con una pregunta abierta: 715 de 757 celdas de error de V03 las acierta
al menos uno de los 20 canales historicos, pero ningun canal tiene net positivo. P50?es que es que el
problema no era falta de senal sino incapacidad de identificacion, y exige medir el techo del
oracle antes de entrenar nada.

**Paso 1, auditoria de brecha.** Se clasificaron las cinco categorias contra los artefactos, no
contra los nombres de los scripts. A (ensayo) en V17 y V23, B (ponderacion global) en V17, V23 y
`17_stacking_v2.py`, C (umbral) en los umbrales de V03, D (correccion selectiva) en **V18-01**. E
(seleccion por label-cell con abstencion) **no esta implementada en ninguna parte**, luego
`SELECTOR_FAMILY = GENUINELY_NEW`. Pero D, que es su caso degenerado, ya dio veredicto negativo:
5 challengers, 0 con senal tras Bonferroni, **mejor AUC 0,6181**, mejor mejora fuera de muestra
**2,4 celdas frente a 26,75 necesarias**.

**Paso 2, techo del oracle.** Con `y_true` y el espacio completo de 21 acciones:

```text
macro-F1 de V03     = 0,7772114525
macro-F1 del oracle  = 0,9870178614
DELTA               = +0,2098064089
```

Por etiqueta: HS +0,231382, TR +0,170787, AG +0,227250. Son **69,9 veces** el margen de 0,003 que la
especificacion fija. La hipotesis suspicion de que macro-F1 ofreciera poca ganancia practica queda
**refutada**: la ganancia seria enorme. De las 4449 celdas, 4407 tienen alguna accion correcta y
**42 no la tienen**, luego el oracle no puede llegar a 1,0.

**La fiabilidad de los canales lo anticipaba y es el dato clave.** Medida sobre el **conjunto de
desacuerdo**, es decir exactamente donde un selector usaria cada canal, la accuracy es:

| canal | accuracy global | accuracy dado desacuerdo | net |
|---|---:|---:|---:|
| `base:M2_union_lr_c2` | 0,8323 | **0,5420** | +11 |
| `emoc:NB-UNION-LR@C=1.0` | 0,8274 | 0,4860 | -11 |
| `base:M3_union_svc` | 0,8283 | 0,4828 | -7 |
| `cita:NEG` | 0,4464 | 0,1416 | **-1706** |

**19 de los 20 canales estan por debajo de 0,5** sobre las celdas donde discrepan de V03, es decir
son peores que una moneda. El unico que supera 0,5 es el propio V03 con umbral 0.5, que no es un
canal alternativo sino la accion de abstenerse.

**Paso 3, el selector.** 21 `LogisticRegression` binarios, uno por accion, con `KEEP_V03` siempre
disponible, 53 variables observables, cross-fitting con los pliegues canonicos de TOXIC agrupados.
Sin `y_true`, sin correccion de V03, sin el canal correcto.

```text
corrected_cv   = 0,7657442265
delta vs V03   = -0,0114672260
pliegues       = 0 de 5 positivos
rescate        = 48
damage         = 67
net            = -19
abstencion     = 93,46 pct
```

**La cifra que decide el caso**: la accion que elige el selector acierta **82,56 %** de las celdas.
No cambiar nunca acierta **82,98 %**. El selector es **19 celdas PEOR que abstenerse**.

**Por que falla, de forma mecanica.** La regla es argmax de P(acierta), luego premia al canal mas
**predecible**, no al mas util. Y las acciones que S1 eligio de verdad fueron:

```text
KEEP_V03              4158 veces   (net 0, correcto)
cita:NEG               109 veces   net -1706   <- el peor canal del pool
synt:SYN-POS            81 veces   net  -705   <- el segundo peor
emoc:NB-WORD-LR@C=1.0   30 veces   net   -46
emoc:ORT-AE@C=2.0       11 veces   net   -84
emoc:NB-UNION-SVC       10 veces   net   -57
```

**Todos los canales que eligio tienen net negativo**, y los dos mas elegidos son los dos peores. La
correlacion de Spearman entre el AUC de un canal y su net real es **+0,5559 (p 0,0109)**, o sea en
promedio el canal mas predecible dana menos, pero **el canal de maximo AUC (0,7237) es precisamente
el de peor net**. Ese es el fallo sistematico de un argmax sobre P(acierta).

**El danio esta todo en la frontera.** Por margen de V03: `[0, 0,10)` cambia 112 celdas, rescata 45,
dana 67, **net -22**; `[0,10, 0,20)` cambia 3, net +3; `[0,20, 1]` no cambia ninguna. El selector
solo actua donde V03 ya dudaba, y ahi la informacion para elegir no existe.

**S2 no se ejecuta.** La especificacion lo prohibe cuando S1 ya falla de forma clara, y S1 tiene
delta negativo y cero pliegues positivos. El gate de 0,003 no se pasa, luego **no hay bootstrap ni
segunda particion**.

**Decision.** `CASE B`. `CONTEXTUAL_SELECTION = CLOSED`. La conclusion no es que falte senal: es que
la senal existe, es enorme, y **no es direccionable**. Una explicacion del techo actual de V03
sustancialmente mejor que "probamos 15 familias y ninguna funciono".


### D-122. El objetivo de utilidad es mucho mas aprendible que el de correccion, y aun asi no supera a V03

**Contexto.** P50 cerro `CONTEXTUAL_SELECTION = CLOSED` probando un unico surrogado,
`argmax P(canal acierta)`, y obteniendo -0,0115. P51 reabre la linea porque ese
surrogado premia la **predecibilidad** de un canal y no su **utilidad**: por eso P50
eligio `cita:NEG` 109 veces con net -1706. La pregunta nueva es si un selector que
aprende el beneficio de sustituir V03 mejora la decision.

**La utilidad no es una invencion de P51.** `experiments/v19_01_valor_de_correccion.py`
ya derivo la forma cerrada del delta de F1 para los cuatro estados de celda, y su
propio texto dice que en V18 se contaron rescates y danos como si valieran uno cada
uno, lo cual es falso en el margen. Ademas `reports/v19_01_valor_de_correccion.json`
**no existe**, solo el `.md`. V19-01 y V19-03 tienen **cero ajustes de modelo**: la
utilidad se midio, nunca se aprendio. Luego
`SELECTOR_UTILITY_ALREADY_TESTED = FALSE` y P51 continua.

**Un hecho estructural que condiciona todo.** En celdas binarias, si el canal y V03
discrepan, la etiqueta verdadera coincide con **exactamente una** de las dos
predicciones. Luego **todo cambio de discrepancia es rescate o dano, y nunca ambos
mal**, y por tanto `delta_F1 > 0` equivale a `rescate`. El target binario de P51
coincide por construccion con el de P50 y con el de V18-01. Verificado sobre las
4449 celdas por 20 canales: `True`. **Lo unico nuevo es la magnitud.**

**La asimetria de direccion y el equilibrio que no es 0,5.** De la forma cerrada:
`FN -> TP` escala con `TP + FP + FN` y `FP -> TN` solo con `TP`. Medido sobre los
conteos de cada pliegue de entrenamiento, la razon es **1,57x**. De ahi sale que el
punto de equilibrio, expresado sobre `P(y=1)`, es **0,3892 para las dos
direcciones**, con coincidencia a 1,7e-16. No es 0,5, y no es un grid: sale de la
forma cerrada.

**La senal es mucho mas aprendible que en P50.** Este es el hallazgo de P51 con
independencia de como termine la decision:

| target | AUC out-of-fold |
|---|---|
| `channel_correct` (P50) | 0,55 a 0,77 |
| `delta_F1 > 0` (P51) | **0,5507 a 0,9249, media 0,8091** |

**Lo que hay que?? superar, y no es igual en las dos direcciones.** Tasa base de
acierto al subir de 0 a 1: **17,86 pct**, frente a un equilibrio de 38,92 pct, o sea
hace falta una elevacion de **2,18x**. Al bajar de 1 a 0: **37,61 pct**, o sea solo
**1,03x**. Las dos direcciones no son intercambiables.

**S1-U, target binario con umbral 0,5.** `corrected_cv` 0,0294622851, delta
**-0,7477491670**, 0 de 5 pliegues, rescate 414, damage 1330, net **-916**. Catastrofico,
y el motivo es preciso: un target binario trata igual una celda que gana 1,0e-3 que
otra que gana 1,6e-3, de modo que **no puede distinguir que bajar de 1 a 0 arriesga
`TP -> FN`, que vale 1,57 veces mas de lo que gana**. Por eso destroza las celdas de
margen alto: 1018 cambios, net **-780**.

Con los umbrales analiticos de equilibrio sobre la confianza maxima, el mismo modelo
baja a delta **-0,0050258278** con net -7. Es una mejora de 0,74 de macro-F1 hacer
bien la puerta, y dice que el fallo era de decision y no de objetivo.

**S2-U,/utilidad esperada, autorizada por la directiva** porque S1-U mostro senal
observable con delta negativo. Mismos veinte clasificadores; lo unico que cambia es la
agregacion, y por eso no es un modelo nuevo ni una familia nueva:

```text
E[u(i,j)] = P_j(x) * g_direccion - (1 - P_j(x)) * h_direccion
se elige argmax y se cambia solo si es mayor que cero
```

Sin umbral, luego sin grid. Resultado: `corrected_cv` **0,7670528094**, delta
**-0,0101586431**, 1 de 5 pliegues, 304 celdas cambiadas, rescate 121, damage 183,
net **-62**.

**El diagnostico que explica el resultado, y que contradice a P50.** Una permutacion
que reasigna los mismos 304 cambios a celdas elegidas al azar da una nula de net
**-115**, con IC95 **[-136, -93]**. El observado es **-62**. El selector es **53
celdas mejor que el azar**, con p de una cola **0,0000**.

```text
P50 concluyo: la informacion no es identificable
P51 demuestra: la informacion SI es identificable (AUC 0,81 y 53 celdas sobre el azar)
               pero el conjunto de cambios tiene expectativa negativa,
               y recuperar esa ganancia entera no basta para superar a no hacer nada
```

Hay ademas un **gap de calibracion** enorme: el modelo esperaba que sus cambios
valieran **+0,056618** de macro-F1 y salieron a **-0,010159**. Sus probabilidades
estan sobreestimadas.

**La asimetria se confirma con el signo inesperado.** Bajar de 1 a 0 sale **neta
positiva** (net **+5**, 50 rescates y 45 danos) y subir de 0 a 1 sale muy negativa
(net **-67**), al contrario de lo que predicen las magnitudes, porque el selector
discrimina mucho mejor al bajar: 50 de 95 aciertos, un 52,6 pct frente al
equilibrio de 38,92 pct. La variante restringida a la direccion 1->0 da delta
**-0,0050042857** con net +5. Positiva en celdas, insuficiente en macro-F1, porque 5
celdas valen alrededor de 0,0006 y hacen falta 26,75 para 0,003.

**Decision.** `CASE A`. `CONTEXTUAL_SELECTION = CLOSED`, ahora de forma definitiva,
tras haber probado gating por correccion (P50) y gating por utilidad (P51), con dos
formulaciones de selector cada uno. El techo de +0,2098 que dio P50 sigue en pie, y
sigue sin ser alcanzable: el problema no es de modelo sino de que **el conjunto de
cambios disponibles tiene expectativa negativa** y la mejor regla conocida solo
recupera parte de esa perdida.


### D-123. El techo de la mejor politica observable es NEGATIVO: la seleccion contextual se cierra por una razon matematica

**Contexto.** P51 cerro `CONTEXTUAL_SELECTION = CLOSED` con dos implementaciones
negativas, pero dejo una pregunta: ?existe una politica mejor que V03 dentro de la
informacion observable, aunque todavia no sepamos aprenderla? P52 responde con un
techo, sin entrenar nada.

**Primero, una correccion de P51.** P51 registro que la elevacion necesaria al bajar
de 1 a 0 era **1,03x**. Es un error. P51 dividio `(1 - eq_down) = 0,3892` por la tasa
base de 0,3761, pero el rescate al bajar es `P(y=0)` y su equilibrio es el propio
`eq_down = 0,6108`. La elevacion real es **1,62x**. El equilibrio (0,3892) y la elevacion
al subir (2,18x) si se reproducen. Consecuencia: la direccion 1->0 era **mas** dificil de
lo que P51 reporto, no mas facil.

**Segunda correccion, de lectura del codigo.** P51 aplico el umbral **0,5** donde la
teoria da **0,3892** al subir y **0,6108** al bajar. Luego fue mas estricto de lo
necesario al subir y mas laxo al bajar, y por eso sobre-cambio justo en la direccion
que mas dana. Ademas P51 usaba **un solo clasificador por canal mezclando las dos
direcciones**, que es lo que la directiva de P52 prohibe.

**La correccion conceptual que hace el trabajo.** `policy oracle` sin `y_true` **no es un
tercer objeto**: una politica que estima `P(rescue | observable)` **es** una politica
aprendida. Por eso P52 separa una cadena de cuatro:

```text
V03                                    0,7772114525   (inaccion, net 0)
estrato_out_of_fold                    0,7549322355   (-0,0222792170)
STRATUM_ORACLE                         0,7753804469   (-0,0018310056)
TRUE_ORACLE                            0,9870178614   (+0,2098064089)
```

El `STRATUM_ORACLE` es el **techo rigido**: la mejor politica que sea constante dentro de
estratos observables, con la accion de cada estrato elegida usando `y_true`. Usa 18
estratos definidos solo con la prediccion de V03, el bin de margen y el numero de
canales que discrepan. El `TRUE_ORACLE` reconstruido coincide con el de P50 con
diferencia 0.

**El techo es NEGATIVO.** La mejor politica estratificada posible, con el permiso de mirar
`y_true`, pierde frente a no hacer nada. De ahi se sigue por fuerza logica que **ninguna**
politica condicional sobre variables observables puede superar a V03. No es un resultado
empirico de un modelo que salio mal: es un cota.

**La descomposicion explica por que el oracle es tan grande y a la vez inalcanzable.**

```text
TRUE_ORACLE -> STRATUM_ORACLE   +0,211637   coste de coarsar la politica a estratos
STRATUM_ORACLE -> V03          -0,001831   lo que una politica estratificada gana (nada)
```

El +0,2098 del `TRUE_ORACLE` esta **casi enteramente** en la granularidad de celda. Exige
una accion distinta en cada celda, y los estratos observables no dicen cual.

**Rescate exclusivo: la hipotesis del usuario, medida y refutada.** P49 dijo que 19 de 20
canales tienen net negativo, y era razonable sospechar que cada uno aporta regiones
exclusivas cuya combinacion supere a todos. Medido:

| magnitud | valor |
|---|---:|
| celdas con rescate unico (solo un canal las rescata) | **78 de 715** |
| HHI del reparto de los 715 rescates | **0,0020** |
| canales con `net_exclusivo` positivo | **2 de 20** |
| pares con exclusividad bilateral | **0 de 190** |
| pares donde los dos canales se contradicen | **0 de 190** |

El HHI de 0,0020 con 20 canales significa reparto casi uniforme (el minimo teorico es
1/20 = 0,05, y sale **por debajo** porque los canales se solapan muchisimo). Y el dato
duro: **donde dos canales difieren de V03, nunca difieren entre si**. No hay nada que
repartir, luego ninguna combinacion puede aportar de forma exclusiva. La hipotesis de
complementariedad queda cerrada con evidencia, no con opinion.

**Correlacion de error en el regimen donde V03 falla.** Media 0,1859, maximo 1,0000,
minimo -0,1598, 14 de 190 pares por encima de 0,5, y 6 grupos a distancia 0,65. La
diversidad global de los canales no es la misma que la diversidad **precisamente donde V03
falla**, que es la unica que un selector necesitaria.

**Una lectura que se descarta a si misma.** La curva de valor de la politica sale con net
+225, y **no es valor de politica**: la accion de cada estrato se elige con labels del
training fold, de modo que en las celdas de error de V03 la mejor accion es un rescate
por construccion. Se reporta marcada como contaminada y la conclusion se apoya
unicamente en el `STRATUM_ORACLE`.

**Decision.** `CASE C`. `UTILITY_POLICY = CLOSED` y `CONTEXTUAL_SELECTION = CLOSED`, con
un motivo que ya no es "no lo probamos" sino una cota negativa. **P52-01 no se ejecuta**,
porque la directiva solo lo justifica si el techo supera V03 + 0,003.


### D-124. El espacio de acciones se colapsa a 2 y ningun patron de mascara supera V03

**Contexto.** P52 cerro la linea contextual afirmando que ninguna politica observable
supera V03, pero lo hizo sobre 18 estratos definidos por **cuantos** canales discrepan.
P53_existia para comprobar si la **identidad exacta** de esos canales aporta algo. Antes de
eso, una comprobacion estructural resulta decisiva.

**El colapso.** En las 3896 celdas con discrepancia, **todos** los canales que discrepan
proponen exactamente el valor `1 - v03_pred`.

```text
celdas con discrepancia                          3896
celdas con valor propuesto distinto de 1-v03        0
celdas con dos canales contradiciendose             0
espacio de acciones real                             2
```

Cambiar al canal 3 o al canal 17 produce **la misma prediccion**. Por tanto la mascara
**no puede portar informacion sobre el valor propuesto**, porque el valor no depende de
quien discrepa. Solo puede portar informacion a traves de las **probabilidades** de esos
canales, es decir de cuan fuerte empujan. Y eso convierte el problema entero en **una
decision binaria por celda: voltear o no**.

El `TRUE_ORACLE` reconstruido con el colapso de dos acciones da **0,9870178614**, con
diferencia 0 frente a P50. El colapso no pierde nada.

**Correcciones a P52.**

1. **El `STRATUM_ORACLE` NO estaba contaminado; la pregunta estaba mal planteada.** P52
   elige la accion de cada estrato usando `y_true` de todas las celdas del estrato,
   incluidas las de validacion. Eso es exactamente lo que hace que sea un **techo**: un
   techo holgado por construccion. Si un techo holgado cae por debajo de V03, todo lo mas
   cenido tambien cae, luego la conclusion se sostiene. Lo que P52 **no** establece es una
   cota sobre todas las particiones observables.
2. **La exclusividad bilateral de P52 estaba mal calculada.** P52|reporto `0 pares` con
   exclusividad bilateral. Ese cero era un **artefacto**: la medicion restringia `i_only` a
   las celdas donde **ambos** canales discrepan de V03, y por el colapso esos dos canales
   proposing alli el mismo valor, luego `i_only` era 0 por construccion. P53 lo mide sin
   esa restriccion y obtiene **189 de 190 pares con exclusividad bilateral de rescate**.
3. **El HHI de P52 estaba mal definido.** Se computo como `sum((exclusivo_i / 715)^2)`,
   con proporciones que **no suman 1**, porque el denominador 715 es el total de rescates
   y no la suma de los exclusivos. P53 reporta las dos metricas por separado:

```text
HHI estandar de los 78 exclusivos     0,360289  (minimo teorico con 11 no nulos: 0,090909)
cuota de exclusivos sobre los 715     10,91 pct
```

4. **`0 pares contradictorios` era ambiguo pero su numero era correcto.** Descompuesto en
   A, B, C y D sobre los 190 pares: **B = 0**. Y no es casualidad: es la identidad
   estructural del colapso.

**Lo que se mantiene de P52.** Las 78 celdas con rescate unico (P53 las recalcula y
obtiene 78), el `TRUE_ORACLE` de 0,9870178614, y la construccion de los estratos.

**Distribucion de mascaras.** 1126 patrones distintos sobre 3896 celdas con discrepancia,
frecuencia mediana 1, maximo 658, **843 patrones singleton**. La mascara esta muy
fragmentada, y esa fragmentacion es la causa de que las policies fine-grained no
generalicen.

**Rescate y dano por patron.** Sobre las 3896 celdas con discrepancia, no sobre las de
error de V03, que es donde el canje existe de verdad:

| tipo | n | rescate | dano | net |
|---|---:|---:|---:|---:|
| un solo canal | 959 | 8,13 pct | 91,87 pct | **-803** |
| dos o mas | 2937 | 21,69 pct | 78,31 pct | **-1663** |

El dano domina por cuatro a uno en las dos. Voltear a ciegas pierde siempre.

**Techos cross-fitted de P0 a P4**, todos con la accion aprendida en el training fold,
aplicada al validation fold, y con repliegue `P4 -> P3 -> P2 -> P1 -> V03`:

| nivel | corrected_cv | delta vs V03 | cambiadas | net | estados | cobertura |
|---|---:|---:|---:|---:|---:|---:|
| P0 solo V03 | 0,7772114525 | +0,0000000000 | 0 | 0 | 1 | 100,0 pct |
| P1 + margen y conteo | 0,7623696498 | -0,0148418027 | 146 | -16 | 18 | 100,0 pct |
| P2 + mascara exacta | 0,7566874656 | -0,0205239869 | 113 | -61 | 1469 | 70,8 pct |
| P3 + identidad de etiqueta | 0,7588669459 | -0,0183445066 | 97 | -57 | 1877 | 64,7 pct |
| P4 + probabilidades | 0,7712584603 | -0,0059529922 | 23 | -19 | 2940 | 36,4 pct |

**El mejor techo observable es P0, es decir no hacer nada.** Toda particion informativa
empeora a V03. Y el diagnostico de soporte explica por que: al exigir soporte minimo 20 en
training, los cambios caen a **cero** en P2, P3 y P4. Los patrones con expectativa positiva
en retrospectiva son los pequenos, luego no generalizan.

**Interaccionalidad: existe, y no sirve.** Diez canales tienen soporte suficiente en ambos
regimenes, con diferencias entre -0,2584 y +0,1069 en tasa de rescate segun actue solo o
con otros. Es decir hay estructura de tipo "canal A util cuando actua solo, canal B util
cuando A tambien discrepa". Pero no llega a producir una politica positiva.

**Decision.** `CASE C`. `MASK_BASED_SELECTION = CLOSED`. **No se entrena modelo**, porque el
techo no supera V03 + 0,003.

**Alcance de la afirmacion, que es lo que P53 pretendia corregir.** La evidencia sostiene
que **ninguna politica basada en la mascara de desacuerdo, el margen, el numero de
discrepancias, la identidad de la etiqueta y las probabilidades de los canales supera V03
en evaluacion cross-fitted**. Y como el colapso demuestra que el espacio de acciones real
tiene solo dos elementos, la conclusion es mucho mas fuerte que la de P52. Lo que **no**
se afirma, y no debe afirmarse, es que ninguna politica imaginable supere V03.


### D-125. El residuo no rescatable es frontera de etiqueta, no informacion ausente, y no tiene valor material

**Contexto.** P53 demostro que el espacio de acciones real tiene dos elementos, porque
todos los canales que discrepan proponen el mismo complemento binario. Cerrada la
seleccion contextual, P54 vuelve al EDA y pregunta que son los errores que ningun canal
puede corregir.

**Poblaciones, recalculadas desde cero y verificadas contra P49.**

```text
A = V03 correcto                              3692
B = V03 mal y algun canal acierta               715
C = V03 mal y ningun canal acierta              42
errores totales                                757
verificacion 757 / 715 / 42                    True
```

**Advertencia de soporte, que condiciona todo lo demas.** Las 42 celdas de C proceden de
**33 tweets distintos**, y 9 de las 42 celdas vienen de solo 3 tweets con las tres
etiquetas mal. El soporte efectivo es **33 observaciones, no 42**. Cualquier
enrichecimiento de este residuo debe leerse con esa cifra delante.

**Composicion.** FP (V03=1, y=0) **22** celdas, FN (V03=0, y=1) **20** celdas. Por
etiqueta, el porcentaje de errores que son no rescatables es **HS 2,85 pct**, **TR 7,32
pct**, **AG 7,58 pct**: el residuo se concentra en AG y TR, no en HS.

**Margen y confianza.** El **85,7 por ciento** de C cae en el bin de margen alto
`[0,20, 1]`, y **36 de 42** son errores de alta confianza. P49 lo sugirio y P54 lo
confirma. Es coherente con P53: en la frontera de decision, la propia confianza de V03
bloquea cualquier correccion.

**Una impresion mia que los datos corrigen.** Leer las 42 celdas sugiere con fuerza que el
residuo es `insulto lexico sin escalada`. Al medirlo:

```text
lexico de insulto en C   66,67 pct
lexico de insulto en A   53,36 pct
enrichment C/A            1,25x    p = 0,0566
```

No es discriminante. La impresion cualitativa era correcta sobre las celdas de AG y
equivocada sobre el conjunto. **El insulto esta presente en la mitad de todas las celdas**,
correctas e incorrectas por igual.

**El mayor candidato, y por que no es hipotesis.** La negacion:

| atributo | n en C | pct C | pct A | enrichment C/A | p |
|---|---:|---:|---:|---:|---:|
| negacion | 17 | 40,48 pct | 22,35 pct | **1,81** | **0,0064** |
| marcador de contraste | 5 | 11,90 pct | 5,63 pct | 2,11 | 0,0858 |
| negacion **y** contraste | **1** | 2,38 pct | 3,11 pct | **0,76** | 0,7353 |

**El mecanismo que la lectura cualitativa sugeria queda REFUTADO.** Proponia que la
negacion era *concedente*, es decir que licenciaba lo que venia despues, como en "no soy
racista pero si persona". Medido: **una sola celda de 42** cumple negacion y contraste, y
el enrichment es de **0,76x**, o sea nulo. Ademas la negacion tiene tambien enrichment de
**rescate** sobre B de 1,55x, luego **marca celdas???? UAS difficultes en general y no un modo de fallo concreto**. El mecanismo queda **sin explicar**, y se registra asi en lugar de proponer uno.

**Los otros dos candidatos, y el argumento que los cierra.**

| candidato | n en C | enrichment C/A | p | valor absoluto |
|---|---:|---:|---:|---:|
| `mi + insulto` (posesivo) | **4** | 7,33 | 0,0022 | ~0,0005 de macro-F1 |
| grupo marginal (`sin papeles`, `inmigrante`) | **9** | 2,68 | 0,0052 | ~0,0011 de macro-F1 |

El lexico de grupo marginal **no aparece en ningun script del proyecto** como feature:
`sin_papeles` e `indocumentad` dan 0 ficheros. Luego el mecanismo no se ha probado. Pero
**9 celdas no alcanzan el margen de adopcion**: 26,75 celdas valen aproximadamente 0,003
de macro-F1, luego 9 valen 0,0011. Y por el resultado ya establecido en P51 a P53, una
regla que acierte esas 9 celdas tambien dana otras, con neto negativo.

**Taxonomia retrospectiva de las 42 celdas.** Marcada como retrospectiva, porque el modelo
vio texto y etiqueta a la vez:

| categoria | n | pct de C |
|---|---:|---:|
| FP: insulto lexico sin escalada | 21 | 50,0 pct |
| FN: narrativa sobre grupo marginal | 9 | 21,4 pct |
| FN: insulto con agresividad no activada | 6 | 14,3 pct |
| FN: agresividad o target no lexicalizada | 4 | 9,5 pct |
| FP: error no lexicalizado | 1 | 2,4 pct |
| FN: discurso citado o atribuido | 1 | 2,4 pct |

**Reconciliacion con los 15 hallazgos del EDA original.** Se reportan **dos** enrichments,
porque `error enrichment` solo dice donde V03 falla: `rescue enrichment` dice donde hay
alternativa y `non-rescue enrichment` dice donde no la hay. Ninguno de los quince
hallazgos alcanza un enrichment de no-rescate superior a 2,1 con soporte util.

**Decision.** `P54_RESEARCH_GAP = NONE`, con **0 hipotesis nuevas**. La conclusion es
especifica y no un cierre generico:

1. **No es falta de informacion en el texto.** El 66,67 por ciento de C ya contiene el
   lexico de insulto, y el reparto por etiqueta es desigual de una manera que el texto si
   explica. La informacion esta presente y el modelo la ve.
2. **Es, en su mayor parte, desacuerdo de frontera de etiqueta.** Insulto lexico con
   etiqueta 0, y narrativa sobre grupo con etiqueta 1. En ambos casos el texto es
   inequivoco y la lectura del anotador es una convencion que no se deriva del texto.
3. **No queda fenomeno semantico con soporte suficiente y valor material.** El mayor
   candidato tiene 17 celdas, su mecanismo queda sin explicar, y 17 celdas valen unos
   0,002 de macro-F1, por debajo del margen de 0,003.

**Lo que no se afirma.** No se afirma que sea imposible mejorar. Se afirma que en este
residuo no hay mecanismo semantico con soporte y valor material, y que la via de datos
observables quedo cerrada por P52 y P53 con una **cota**, no por falta de intento.


### D-126. Los umbrales de V03 vienen de un protocolo mas debil que el de la metica reportada

**Contexto.** P55 audita el procedimiento de V03 sin buscar modelos. La pregunta es si
existe un defecto que explique parte de la brecha OOF a leaderboard. No es una busqueda:
es una lectura del codigo y una medicion con umbrales anidados.

**El pipeline real, leido del codigo y no de los docstrings.** El diagrama que hace el
codigo tiene dos ramas, y ahi esta el problema:

```text
raw train 1731 -> dev 1483 / holdout 248

A) 04_refit_submissions_v02_v03.py
     grupos = near-duplicado (D-004), 5 pliegues
     vectorizador y modelo DENTRO de cada pliegue  (verificado en el codigo:
       construir_matrices -> vectorizador.fit_transform(textos_entrenamiento))
     -> oof_mod03b.npz
     -> umbrales 0,44 / 0,51 / 0,46 maximizando F1 sobre ESE OOF
     -> refit sobre el 100 por ciento del train con ESOS umbrales
     -> submission V03

B) 14_oof_corregido_v2.py
     grupos = near-duplicate UNION pseudo-autor (D-016), 5 pliegues
     -> v2_oof_base_corrected.npz
     -> corrected_cv = 0,7772114525 con LOS UMBRALES DE A)
```

**La discrepancia.** Los **dos numeros de la referencia de V03 vienen de protocolos de
agrupacion distintos**, y D-004 es el mas debil porque deja que un pseudo-autor cruce
pliegues. Cuando D-016 corrigio el protocolo, el `corrected_cv` se recalculo **conservando
los umbrales viejos**, sin reoptimizarlos bajo el protocolo nuevo.

**La direccion del sesgo es sistematica, no ruido.** Con umbrales **anidados**, es decir
optimizados en cuatro pliegues y aplicados al quinto:

| etiqueta | umbrales anidados por pliegue | mediana | V03 | V03 dentro del rango |
|---|---|---:|---:|---|
| HS | 0,40 0,41 0,40 0,41 0,41 | 0,41 | **0,44** | **NO, en los 5** |
| TR | 0,50 0,48 0,50 0,48 0,48 | 0,48 | **0,51** | **NO, en los 5** |
| AG | 0,47 0,45 0,45 0,45 0,45 | 0,45 | 0,46 | si |

Los umbrales de V03 estan **0,03 por encima** de los que pide el protocolo corregido, en
HS y TR, **en los cinco pliegues sin excepcion**.

**La magnitud honesta esta por debajo del margen.** El delta in-sample, optimizando sobre
el mismo OOF que se puntua, da **+0,003629**. Ese numero **no** es una mejora esperable.
El delta **anidado**, que si es una estimacion honesta, da:

```text
macro-F1 con umbrales V03     0,7772114525
macro-F1 con umbrales anidados 0,7781367700
delta                         +0,000925
```

**+0,000925 esta por debajo del margen de adopcion de +0,003.** Por eso la correccion es
legal y el mecanismo es concreto, pero **no justifica gastar una submission**.

**La consecuencia operativa es una tasa positiva infraestimada.** Sobre el OOF:

| umbrales | HS | TR | AG |
|---|---:|---:|---:|
| de V03 | 0,4572 | 0,3001 | 0,3776 |
| anidados | 0,4970 | 0,3075 | 0,3810 |

Los umbrales de V03 son demasiado altos y el OOF produce unos 4 puntos por debajo de
positivos en HS que los umbrales del propio protocolo piden. Eso es **consistente en
direccion** con que el LB haya salido 0,0058 **por encima** del OOF, aunque no lo
demuestra.

**El mecanismo no es el tamano del entrenamiento.** La trayectoria con 80, 90 y 100 por
ciento de los grupos del training fold da macro-F1 0,7756814919, 0,7775151230 y
0,7781367700, con umbrales practicamente identicos. Luego el desplazamiento no viene de
que el modelo final se ajuste con mas datos, sino de que los umbrales se elegieron bajo
otro protocolo.

**Integridad de artefactos.** `features/oof_mod03b.npz`, que produjo los umbrales adoptados,
**no existe en disco**. Los umbrales de V03 **no son auditables**. El sha256 de la
submission V03 si coincide con el esperado, y `corrected_cv` se reproduce exactamente desde
el OOF D-016.

**Lo que la auditoria NO encuentra.** El vectorizador se ajusta **dentro** de cada pliegue,
verificado en el codigo y no en el docstring, luego no hay leakage de representacion. Las
prevalencias por pliegue son homogeneas, luego ningun pliegue domina. La dispersion del
macro-F1 entre pliegues es 0,0574, con un pliegue claramente mas bajo (0,7386) frente a
0,7960 del primero, luego la media de pliegues (0,7773117212) difiere del global
(0,7772114525) en 0,0001.

**Decision.** `P55_PROCEDURAL_GAP = B`, clasificado como **`BUG_FOUND`** y **no** como
mejora, con `ORIGINAL_RESULT` y `CORRECTED_RESULT` separados:

```text
ORIGINAL_RESULT   corrected_cv = 0,7772114525   umbrales 0,44/0,51/0,46 de D-004
CORRECTED_RESULT  corrected_cv = 0,7781367700   umbrales anidados de D-016
delta             +0,000925                     por debajo del margen de 0,003
```

**0 experimentos nuevos y 0 submissions.** La correccion es legal y podria ejecutarse, pero
su magnitud medida no alcanza el margen, y V03 permanece congelado e intacto.


### D-127. El sesgo de transferencia de umbrales de P55 no sobrevive al anidamiento completo

**Contexto.** P55 reporto que los umbrales de V03 (0,44 / 0,51 / 0,46) estan **0,03 por
encima** de los umbrales anidados del protocolo corregido D-016, **fuera del rango en los
cinco pliegues** en HS y TR, y que corregirlo daba **+0,000925**. P56 hace el anidamiento
que P55 no hacia, con vectorizador y modelo **re-ajustados** dentro de un split interior
agrupado, y el resultado corrige a P55.

**Reproduccion.** `P55_CORRECTED_RESULT_REPRODUCED = TRUE`:
0,7781367700 reproducido exactamente. La correccion procedimental de P55 es un hecho real.

**El anidamiento completo cambia el diagnostico.** Con 4 pliegues interiores agrupados y
refit completo dentro de cada outer train:

| etiqueta | estimaciones interiores | mediana | std | min | max | V03 | V03 en el rango |
|---|---|---:|---:|---:|---:|---:|---|
| HS | 0,40 0,41 0,42 0,41 0,45 | 0,41 | 0,0192 | 0,40 | 0,45 | 0,44 | **SI** |
| TR | 0,47 0,46 0,46 0,51 0,46 | 0,46 | 0,0217 | 0,46 | 0,51 | 0,51 | **SI** |
| AG | 0,40 0,48 0,43 0,51 0,45 | 0,45 | 0,0428 | 0,40 | 0,51 | 0,46 | **SI** |

**Los umbrales de V03 caen DENTRO del rango interior en las tres etiquetas.** El supuesto
sesgo sistematico de P55 era un **artefacto de su procedimiento mas simple**, que producia
rangos artificialmente estrechos: el procedimiento de P55 tomaba el umbral de las
probabilidades out-of-fold de cuatro pliegues, ya producidas por modelos entrenados, y de ahi
una varianza mucho menor.

**La cifra honesta es negativa, y la apparentemente positiva es optimismo.**

```text
T0 historico (0,44 / 0,51 / 0,46)          0,7772114525   +0,0000000000
T1 mediana aplicada GLOBALMENTE            0,7798514633   +0,0026400107   4/5 pliegues
umbral por pliegue exterior (HONESTO)       0,7745478730   -0,0026635795
optimismo de definir el umbral con todos los pliegues       +0,0053035902
```

**El optimismo es el doble de la mejora aparente.** Aplicar una sola mediana a los cinco
pliegues usa datos de los cinco para definirla, y eso vale **+0,0053**, mas que el propio
+0,0026. La cifra que decide es la del umbral de cada pliegue exterior aplicado a ese
mismo pliegue, y es **negativa**.

**Tres senales independientes apuntan en la misma direccion.**

```text
net a nivel de celda, T1 contra T0    -11   (43 rescates, 54 danos)
   por etiqueta                        HS  -5 | TR  -6 | AG   0
bootstrap del delta de T1             mediana +0,002553
                                       IC95 [-0,003456, +0,008989]  INCLUYE 0
pliegues con delta positivo            4 de 5, con el pliegue 3 en -0,0310
```

El F1 sube mientras el recuento de celdas baja. No es contradictorio: macro-F1 promedia
tres etiquetas y un cambio de umbral reordaza las positivas, y el F1 de la etiqueta afectada
puede subir sin que el recuento neto de celdas mejore. Pero **el recuento de celdas es la
medida honesta del efecto**, y es negativo en las tres etiquetas.

**Tasa positiva.** Con T1, HS pasa de 0,4572 a 0,4943 y TR de 0,3001 a 0,3216. Es decir T1
produce muchas mas positivas, y aun asi el F1 no mejora de forma robusta.

**Decision.** `CASE B`.
`THRESHOLD_TRANSFER_BIAS = NOT_CONFIRMED`. `STATUS = CLOSED`. **No se crea candidato local
y no se genera submission.** La correccion procedimental de P55 **sigue siendo un hecho
historico real** (los umbrales si se adoptaron bajo D-004), pero su impacto medido con el
procedimiento correcto es menor de lo que P55 sugeria, y en el anidamiento completo resulta
contraproducente.

**Lo que P56 corrige de P55, explicitamente.**

| afirmacion de P55 | estado tras P56 |
|---|---|
| los umbrales de V03 se adoptaron bajo D-004 | **se mantiene**, es un hecho |
| los umbrales estan fuera del rango en los cinco pliegues | **CORREGIDO**: caen dentro |
| el sesgo es sistematico, de +0,03 en HS y TR | **CORREGIDO**: no hay sesgo sistematico |
| la correccion da +0,000925 | **CORREGIDO**: la cifra honesta es -0,002664 |
| `BUG_FOUND` con impacto bajo el margen | **se mantiene como hecho**, el impacto es nulo o negativo |

**Ninguna submission.** El gate de +0,003 no se alcanza por la cifra honesta, y la cifra
global que lo alcanzaba no es adoptable por su optimismo de mas de 0,005.


### D-128. El registro mezcla metricas, un envio real no estaba en el log, y no hay candidato

**Contexto.** P57 no entrena nada. Su trabajo es cruzar `experiments.csv`,
`leaderboard_log.csv`, `submission_manifest.csv`, `CHAMPION.md` y los informes `p45` a `p56`,
y hacer legible el estado del proyecto. Al hacer el cruce aparecieron tres cosas que ninguna
fase anterior habia formulado.

**1. Las columnas `f1` y `macro_f1` de `experiments.csv` no son una metrica cada una.**

Mezclan al menos tres metricas y tres protocolos:

| magnitud | valor | metrica | protocolo | poblacion |
|---|---:|---|---|---|
| maximo absoluto de la columna `macro_f1` | 0.8743 (S08) | **macro-6**, no la de la competencia | holdout | 248 filas |
| metrica del proyecto sobre el OOF de V03 | 0.7772114525 | la de la competencia | D-016 | dev 1483 |
| macro-6 sobre el mismo OOF de V03 | 0.8182213363 | **macro-6** | D-016 | dev 1483 |
| `local_f1` que registra el log para V03 | 0.7989 | la de la competencia | **D-004** | dev |
| `corrected_cv` canonico de V03 | 0.7772114525 | la de la competencia | **D-016** | dev 1483 |

P45 declaro macro-6 diagnostica. La diferencia medida sobre el mismo OOF es **+0.04101**, y la
diferencia entre el `local_f1` del log y el `corrected_cv` canonico es **+0.021689**, que es
exactamente el coste de medir con D-004 en vez de D-016.

**Consecuencia practice:** cualquier tabla de "mejor CV" construida tomando el maximo de esas
columnas es falsa por mas de 0,04, y mezclar protocolos anade hasta 0,024 mas. La respuesta NO
es reescribir el registro, sino **anadirle la clasificacion**: `metrica`, `protocolo`,
`poblacion` y `anidado` van ahora en `reports/p57_final_experiment_inventory.csv`, una fila por
experimento, y solo se compara dentro de un mismo protocolo.

**2. Un envio real a Kaggle no estaba en el log del leaderboard.**

`experiments.csv` fila V33-03 documenta: *Submission ENVIADA y ACEPTADA. ID 56608788. Fichero
toxic_submission_v32_05_url_length.csv. Score LB 0.78306. delta_LB = 0.00000. 2 submissions
restantes hoy.* Esa fila no existe en `leaderboard_log.csv`, que tiene 22 entradas.

| magnitud | valor |
|---|---:|
| envios con LB verificado en el log | 22 |
| envio real no registrado | 1 |
| **envios historicos totales** | **23** |
| ficheros de submission en disco | 37 |
| generados y nunca enviados | 14 |

`PROJECT_STATE.md` y `CHAMPION.md` afirman ambos "0 de 22 submissions superaron V03". La
afirmacion **sigue siendo cierta en su sustancia** porque 0 de 23 superaron 0.78306, pero el
**recuento es erroneo**. Se corrige aqui y en el informe de P57; el log no se modifica, porque
es un registro de mediciones y sus 22 filas son fiables.

Este envio es ademas el control mas limpio del proyecto: la interaccion con longitud de URL dio
**+0.0045 de corrected_cv** y el LB **no se movio ni en la sexta cifra decimal**, porque la
submission resulto identica a V03.

**3. No hay ningun artefacto que cumpla los cinco criterios de candidato a la vez.**

| artefacto | que le falta |
|---|---|
| v23_01 svd300_rbf, corrected_cv 0.7904 | ya rechazado: enviado, LB 0.77938 |
| v08_meta_v2, corrected_cv 0.7874 | ya rechazado: enviado, LB 0.78163 |
| v20_01 meta de 36 features, corrected_cv 0.7882 | ya rechazado dos veces: por holdout y por LB 0.77663 |
| E2-13-META-ENT-RIDGE, 0.7885 anidado | legal y sin leakage, pero cerrado por sparsity y nunca verificado fuera |
| umbrales anidados de P56 | veredicto NOT_CONFIRMED, delta honesto negativo |
| v16_ch2 prior shift EM | transductivo: consume informacion del conjunto de test |

**El patron de fondo, que es el hecho mas firme del proyecto:**

| familia | envios | LB | veredicto |
|---|---:|---|---|
| kernels | 1 | 0.77938 | RECHAZADO |
| label_specialization | 1 | 0.77781 | RECHAZADO |
| ensembles | 3 | 0.7751, 0.7747, 0.77122 | los tres a la baja |
| stacking / meta-modelos | 5 | 0.78163, 0.77895, 0.77978, 0.78150, 0.77663 | los cinco a la baja |
| structural_features | 1 | 0.78306, delta 0 | sin efecto |

**Diez envios de candidatos con mecanismo nuevo. Los diez transfirieron a la baja.** Y el mejor
CV comparable del proyecto **no es el campeon**: es svd300_rbf con 0.7904, que se envio y perdio
0.00368.

**4. PLS es un cierre de tipo B, no A, y dos lineas no llegaron a probarse.**

- **PLS: invalidado, no negativo.** V42-01 y V43-01 se invalidaron por implementacion (rango
  invalido, API de PLS), luego sus numeros **no son evidencia**. El resultado verdadero lo
  produjo V43-02 como verificacion y es **-0.0137423671**. La hipotesis de embedding
  supervisado no se rechaza; el experimento si.
- **VW: si se probo y se cerro.** V43-01 intento FTRL y quedo invalidado por la API; V44-01 lo
  repito bien sobre la representacion exacta de V03 y cerro
  `SPARSE_ONLINE_OPTIMIZATION = CLOSED`.
- **Estructura de etiquetas: dos lineas cerradas por condicion de parada.** V13-02
  (`CERRADA FORMALMENTE`) y V14-00 (`0 de 4 experimentos`) no llegaron a ejecutarse. Ahi **no
  cabe decir "no funciono"**, porque no llego a probarse. Se registra como pregunta abierta, no
  como cierre. Y V13-01, que si se ejecuto, dio +0.000534, y V15-01, tercera estimacion
  independiente, dio -0.0014.

**5. La jerarquia de magnitudes, con la incompatibilidad explicita.**

| magnitud | valor | experimento | comparable con V03 |
|---|---:|---|---|
| best_architectural_CV (macro-6, diagnostica) | 0.8743 | S08 | **NO** |
| best_holdout_CV | 0.8465 | S08 | **NO** |
| best_CV protocolo D-004 | 0.8039 | S05 / v05_optuna | **NO** |
| best comparable D-016 | 0.7904 | V4-G01-G03 (svd300_rbf) | SI, **con seleccion** |
| best anidado D-016 | 0.7885 | E2-13-META-ENT-RIDGE | SI, sin seleccion |
| **V03 corrected_cv** | **0.7772114525165721** | V03 | SI |
| **best verified LB** | **0.78306** | v03_lr_union_thr | SI |
| current champion | V03 | sha256 `7d7478f2...` | SI |
| remaining submission budget | limite diario exacto desconocido | 0 envios hoy, reserva propia 3 | n/a |

El 0.7904 es el maximo de **cuatro** candidatos comparados sobre el mismo OOF, luego lleva
seleccion y su IC95 tocaba cero. La cifra que un argumento de "hemos probado mas cosas y
ninguna supera a V03" tiene que citar es la **anidada**, 0.7885.

**Presupuesto de submissions.** La unica cifra local es la anotacion de V33-03: *2 submissions
restantes hoy* el 2026-09-27, dia en el que se cuentan 7 envios. Eso exige un limite diario de
al menos 9, pero la cifra **exacta no consta en ningun registro local** y **no se consulta
Kaggle**. Hoy 2026-09-28 hay **0** envios. La reserva de 3 es autolimitacion del proyecto, no un
limite de la competencia.

**Metrica.** Se mantiene, sin cambio: `LOCAL_METRIC_REPRODUCED = TRUE`,
`KAGGLE_SCORER_DIRECTLY_VERIFIED = FALSE`.

**Formulacion final, y la distincion que no hay que perder.**

> V03 remains the best verified leaderboard result and the frozen architectural baseline. The
> investigated legal mechanisms did not produce a robust transferable improvement, and no
> additional mechanism with sufficient evidence has currently been identified.

```text
best verified result  !=  proof of global optimum
```

Verificado: 0.78306 es el maximo del leaderboard en 23 envios, y 0.7772114525165721 es el CV
corregido de la configuracion congelada.

**NO afirmado:** que V03 sea globalmente optimo. La brecha OOF-LB de 0.00584855 sigue sin
mecanismo identificado, y el techo del oracel de +0.2098064089 no se ha alcanzado **ni se ha
descartado como alcanzable**. El hecho de que ningun candidato con mejor CV local haya
transferido es evidencia **en contra de la optimizacion local** y a favor de V03 como eleccion.
**No es evidencia de que no exista una mejor.**

**Lo que P57 NO hace, por diseno.** No entrena, no genera submission, no consulta Kaggle, no usa
holdout ni test, no modifica V03, y **no registra ninguna fila en `experiments.csv`** porque no
hubo ninguna ejecucion experimental. La decision competitiva queda pendiente: cerrar, autorizar
una ultima investigacion dirigida, o preparar una unica submission de un candidato validado.
Ninguna se ejecuta automaticamente.


### D-129. Ningun candidato cumple los siete criterios, y la lectura de P58 sobre E2-13 era incorrecta

**Contexto.** P57 concluyo que no existia candidato y, al hacerlo, escribio que
`E2-13-META-ENT-RIDGE` era *el unico artefacto con delta honesto anidado positivo y sin
veredicto externo*, y que solo le faltaba una verificacion externa. **P58 audita esa
afirmacion y la corrige.** La decision de no enviar nada no cambia; el motivo cambia.

**1. Que significa exactamente el cierre "por sparsity" de E2-13.**

La columna `decision` de `experiments.csv` dice `CERRADA por sparsity` para las tres variantes
de E2-13. El `result` estructurado dice otra cosa, y es la que manda.

```text
cobertura_informativa   0,005247376311844078   =  0,52 por ciento
umbral de cierre        0,20                   = 20 por ciento
menciones               1079 instancias, 6 sobreviven   = 0,556 por ciento
hashtags                255 instancias, 1 sobrevive    = 0,392 por ciento
dominio                 458 instancias, 458 sobreviven, pero 1 unico en desarrollo
                        o sea el unico dominio que pasa el filtro es t.co
```

**Sparsity no invalida una mejora por si solo, y por eso se cuantifica el aporte real del
canal sobre su propio comparador**, que es la pregunta correcta:

```text
comparador                                META-BASES-RIDGE (0,7874)
E2-13-META-ENT-RIDGE                      0,7884666035195752
delta sobre el comparador                 +0,0011091426646537839
pliegues mejorados sobre el comparador    3 de 5
complementariedad (ensemble_gain)         -0,00025592446516731204
aporta con IC positivo                    False
delta sobre V03                           +0,011266603519575202   <-- este NO es suyo
```

**El cierre es metodologico y esta bien justificado.** El sparsity es la CAUSA y el incremento
medido de +0,0011 con 3 de 5 pliegues y complementariedad negativa es la PRUEBA de que la causa
es real. No se concluyo "sparsity invalida automaticamente", se concluyo "sparsity mas el
incremento medido invalida", que es otra cosa.

**2. La correccion: el delta de +0,0113 no es de E2-13, es de la familia META.**

`v2_meta_oof.npz` declara `variante = META-BASES-RIDGE`, `modelo = ridge`, `parametro = 3.0`, con
**44 features** cuyas ultimas ocho son los rasgos estructurales. Ese es exactamente el modelo
que se envio como `v08_meta_v2`, y su `corrected_cv` recomputado desde el OOF en disco da
**0,7873574608549214**, identico al del registro.

Luego **E2-13-META-ENT-RIDGE es `v08_meta_v2` mas un canal que cubre el 0,52 por ciento de las
instancias.** No es un mecanismo nuevo: es una perturbacion de un candidato ya rechazado
externamente, con LB 0,78163. El +0,0113 que P57 reporto como su_delta sobre V03 es la ganancia
de la **familia META**, refutada 5 de 5 veces en el leaderboard.

| afirmacion de P57 | estado tras P58 |
|---|---|
| E2-13 tiene el mayor delta honesto anidado | **se mantiene**: 0,7884666, el mayor anidado del registro |
| E2-13 es un candidato sin veredicto externo | **CORREGIDO**: su arquitectura ya tiene veredicto, LB 0,78163 |
| E2-13 solo necesita una verificacion externa | **CORREGIDO**: su unico delta propio es +0,0011 con 3 de 5 |
| el cierre por sparsity lo invalida | **CORREGIDO**: lo que lo invalida es el incremento medido, y esta justificado |

**3. La descomposicion celular, recomputada SIN reentrenar.**

`features/oof_candidates_v4.npz` contiene los cuatro candidatos del gate de V4-X01, con ids y
etiquetas identicos al OOF canonico. Las cinco descomposiciones **reproducen
`reports/v4_candidates.json` al sexto decimal**, lo que prueba que estos tensores son los que
alimentaron el gate pre-registrado.

```text
candidato                              ccv       delta  plieg  cambian    r    d  neto  acierto  LB
META-BASES-RIDGE (= v08_meta_v2)  0.787357  +0.010146   4/5     246  132  114   +18   0.5366  0.78163
A_cost_sensitive_downweight       0.785416  +0.008205   5/5     173   99   74   +25   0.5723  sin veredicto
B_meta36                          0.782774  +0.005562   4/5     186  115   71   +44   0.6183  sin veredicto
C_meta36_downweight               0.788173  +0.010962   4/5     212  134   78   +56   0.6321  holdout
D_svd300_rbf                      0.790410  +0.013198   5/5     240  136  104   +32   0.5667  0.77938
```

Sobre 4449 celdas. El `acierto` es la proporcion de celdas modificadas en las que el candidato
acierta donde V03 falla, y es la medida honesta del efecto.

**4. Los siete criterios de la directiva, aplicados uno por uno.**

| candidato | 1 CV | 2 repro | 3 plieg | 4 leak | 5 mec | 6 artefacto | 7 no LB | FINALISTA |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| META-BASES-RIDGE | si | si | si | si | si | si | **NO** | **NO** |
| A_cost_sensitive_downweight | si | si | si | si | si | **NO** | si | **NO** |
| B_meta36 | si | si | si | si | si | **NO** | si | **NO** |
| C_meta36_downweight | si | si | si | si | si | si | **NO** | **NO** |
| D_svd300_rbf | si | si | si | si | si | si | **NO** | **NO** |

**El criterio 6, `artefacto de prediccion completo`, es el que decide.** Solo tres de los cinco
tienen CSV de submission, y los tres ya fueron enviados o ya fueron rechazados por el holdout.

**5. El caso limite, y el unico quasi-finalista del proyecto: B_meta36.**

```text
corrected_cv   0,782774   (+0,005563 sobre V03)
pliegues       4 de 5
IC95 pareado   [+0,003147, +0,016633]   NO incluye el cero
celdas         186 cambian, 115 rescates, 71 danos, neto +44
acierto        0,6183, el mas alto de los cinco
LB             ninguno. Nunca se envio
cumple_D016    True, el unico de A y B que lo cumple
```

Es el unico candidato que pasa el gate D-016 pre-registrado y no tiene veredicto externo. Aun
asi no es finalista, por tres razones que son independientes entre si:

1. **No existe artefacto de prediccion.** `experiments/36_candidatos_v4.py` no escribe ninguna
   submission; solo `C` recibio una, via `38_refit_final_v4.py`, y quedo marcada DESCARTADO.
   El criterio 6 falla, y la directiva 12 prohibe crear una variante nueva.
2. **El unico instrumento de adjudicacion esta agotado.** El holdout lleva 5 de 5 evaluaciones.
3. **Su hermano con veredicto fue rechazado, y es casi indistinguible de un rechazo ya
   medido.** `FIN-01-C-REF-44` registra `delta_vs_meta_v08 = -4,25e-05`: `C` es
   estadisticamente el mismo modelo que `v08_meta_v2`. `B` es `C` sin el downweight, luego no
   hay evidencia de que se comporte de otro modo.

**B pasa cinco de siete y falla en el sexto por ausencia de artefacto, no por defecto del
modelo.** Es el `quasi-finalista` del proyecto y conviene nombrarlo como tal.

**6. Panorama de los 31 candidatos con corrected_cv D-016 superior a V03.**

| categoria | n | por que no son finalistas |
|---|---:|---|
| con LB verificado y **positivo** | **0** | ninguno supera 0,78306 |
| con LB verificado y negativo | 5 | v08_meta_v2 -0,00143, v23 -0,00368, v09 -0,00411, v11 -0,00156, v10 -0,00328 |
| rechazados por holdout | 1 | C_meta36_downweight, 0,8163 contra 0,8285, delta -0,0122 |
| **sin ningun veredicto externo** | **2** | A y B, y ninguno tiene artefacto |
| familias cerradas por inercia | 10 | embeddings EBD, con 9 de 9 unidades exploratorias agotadas |
| mecanismos que no alcanzan su gate | 3 | V4-B03, V4-B05 y otros, por debajo de +0,005 y de 3 de 5 |
| mediciones, no modelos | 1 | PA-01 mide la brecha D-004 contra D-016 |

**7. D_svd300_rbf: por que el CV mas alto no transfirio, y por que queda CLOSED.**

El mejor `corrected_cv` del proyecto, 5 de 5 pliegues, 136 rescates contra 104 danos y
McNemar p=0,0452, perdio 0,00368 en el externo. La firma mecanica esta en los vectores de
soporte: 1321 de 1731 para HS, 1317 para AG y 1044 para TR, es decir el 60 a 76 por ciento de
las muestras, la marca de un kernel RBF con `gamma=scale` sobre 300 dimensiones que memoriza en
vez de generalizar. El gap local-LB pasa de +0,005849 a -0,011030. **El LB ya aporto la
evidencia negativa suficiente, luego `CLOSED`.**

**8. Integridad de los artefactos.**

Los 37 CSV de `submissions/` pasan la validacion formal: 1154 filas, columnas `id,HS,TR,AG`, ids
y orden identicos a `toxic_test.csv`, etiquetas binarias, sin nulos, sha256 calculado.
**Ninguno se modifica.** Se confirma ademas que
`toxic_submission_v20_03_reproduccion_v03.csv` es **bit a bit identico** a la submission de V03,
que es la razon de que su envio diera delta exactamente 0.

**9. Presupuesto.**

```text
local_reservation   3      autolimitacion del proyecto
competition_limit   UNKNOWN   no se consulta Kaggle
envios historicos   23
envios hoy          0
```

**Decision: `CASE A`. `FINALIST = NONE`.**

> V03 remains the best verified leaderboard result.

```text
best verified result  !=  proof of global optimum
```

Lo que SOSTIENE la auditoria: que **el conjunto de artefactos que hemos producido no contiene**
un candidato que justifique una sonda final. Lo que NO SOSTIENE: que no exista una mejora
defendible. Son afirmaciones sobre estados distintos, y confundirlas seria el error de
razonamiento mas caro disponible ahora mismo.
