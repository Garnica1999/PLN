# P84: especificaciones ejecutables por fase (EXP-01 a EXP-06)

Fecha: 2026-10-02.
Proyecto: /home/carlos/Git/PLN-v67/kaggle_toxic_2026_puj.
Plan maestro: /home/carlos/.opencode/plan/kaggle_toxic_exploracion_modelos_f1_080.md.
Preregistro: docs/p84_preregistro.md (Gate 0 completado, hashes congelados).
Auditoria adversarial de premisas: reports/p84_auditoria_premisas.md (leida).
Estado: SOLO ESPECIFICACION. No autoriza ejecutar, no contiene codigo de experimentos,
no se ha tocado test, holdout, submissions ni experimentos cerrados.

Mapa de identificadores: EXP-01 = P84-01, EXP-02 = P84-02, EXP-03 = P84-03,
EXP-04 = P84-04, EXP-05 = P84-05, EXP-06 = P84-06. Los nombres de script son los
del preregistro (experiments/p84_0X_*.py).

---

## 0. Proposito y orden de lectura

Este documento convierte el plan y el preregistro en especificaciones ejecutables por
@ml-coder y verificables por @ml-qa-engineer y @ml-reviewer. Cada spec define Objective,
Inputs, Process, Boundaries (Always / Ask first / Never), Success Criteria medibles,
Outputs y Open Questions. La seccion 1 fija las convenciones comunes: ningun experimento
puede redefinirlas. La seccion 8 lista las correcciones aplicadas al plan y a la auditoria;
esas correcciones son parte normativa de estas specs.

Orden de ejecucion obligatorio: EXP-01, EXP-02, EXP-03 (micro-verificacion), EXP-04,
EXP-05. EXP-06 es una PUERTA: solo se ejecuta si existe un candidato que la cruce
(seccion 7). Gate 2 se aplica despues, solo a candidatos que pasen Gate 1.

---

## 1. Convenciones comunes (gobiernan las seis specs)

### 1.1 Referencia congelada y control de identidad

- V03: `corrected_cv` = 0.7772114525165721 sobre las 1483 filas de desarrollo, con los
  grupos fusionados, umbrales congelados HS 0.44 / TR 0.51 / AG 0.46 y LB 0.78306.
- Artefacto de referencia: `features/oof_v03_final.npz`, sha256
  559265816ce4f6312c87f008f90324f0eec70a74d1e6247364c162793338feab.
- V03 a umbral 0.5 (referencia de `ensemble_gain`): F1 = 0.771778476256122. Se RECALCULA
  desde el npz en cada corrida (`proba >= 0.5`); no se escribe a mano. Verificado:
  el recalculo reproduce el valor exacto.
- Control de identidad obligatorio en cada harness nuevo: reconstruir el OOF canonico de
  V03 con el pipeline por pliegue (TF-IDF word (1,2) + char_wb (3,5), min_df=2,
  sublinear_tf, LR C=2.0 balanced liblinear, rs=42, vectorizador ajustado dentro de
  cada pliegue). Tolerancia: `max |proba_reconstruida - proba_npz| <= 1e-12` y
  `|F1_reconstruida - 0.7772114525165721| <= 1e-12`. La reconstruccion exacta ya fue
  demostrada por p71_10. Si el control no reproduce, STOP: resolver instrumentacion
  antes de leer cualquier brazo candidato. Los resultados de un harness con control
  fallido no se interpretan ni se publican como medida.

### 1.2 Datos, particiones y contadores

- Datos crudos: `data/raw/toxic_train.csv` (sha256
  3d40e7345b588860cf4744914be88ddc094e4f03591516c8176ab40b7f76d42f, 1731 filas) y
  `data/raw/toxic_test.csv` (1154 filas, NUNCA se abre en P84).
- Grupos: `data/processed/grupos_near_duplicados.csv` (sha256
  afabf9ae1c295ea8f203120bf28578cb4bdb4e8f16b89178d820baeaac62e622), fusionados con
  pseudo-autor por `src/splits.py::grupos_fusionados`.
- Desarrollo: 1483 filas; holdout v2: 248 filas, AGOTADO (6 usos registrados frente a
  5 declarados), no se evalua, no se entrena con el, no se crea otro holdout.
- Pliegues canonicos: `pliegues_corregidos` de 5 pliegues sobre desarrollo, tamanos
  [297, 297, 296, 296, 297], sin grupos compartidos (verificado en Gate 0).
- Contador de acceso obligatorio por corrida: filas de desarrollo leidas = 1483,
  filas de holdout = 0, filas de test = 0. Si el contador no cuadra, STOP.
- Segunda particion (solo Gate 2): `StratifiedGroupKFold(5, shuffle=True,
  random_state=20261020)` sobre el mismo desarrollo y los mismos grupos. No es un
  segundo holdout: es una particion de confirmacion sobre desarrollo. No se reajustan
  hiperparametros mirandola.

### 1.3 Metricas y comparaciones

- Metrica primaria: `f1_media_columnas` (media de la F1 binaria de la clase positiva de
  HS, TR y AG; `src/evaluation.py::metricas_multietiqueta`, D-009). Diagnostico:
  macro-6, micro, samples, exact match; nunca como metrica de seleccion.
- Delta por pliegue, delta medio, numero de pliegues mejorados (>= 4/5 para promover).
- IC95 bootstrap pareado por filas, 10.000 remuestreos, semilla 20260926. OJO:
  `src/evaluation.py::bootstrap_diferencia_pareada` usa por defecto semilla 20260925;
  cada llamada DEBE pasar `repeticiones=10000, semilla=20260926` de forma explicita.
- `ensemble_gain` (convencion P6/P26, fijada en el preregistro): F1 de la media
  `(proba_candidato + proba_v03)/2` a umbral 0.5 menos 0.771778476256122, con IC
  bootstrap pareado. El IC se calcula remuestreando filas y recalculando la F1 de la
  media; la referencia 0.771778476256122 se recalcula desde el npz.
- Rescate/dano: celdas (fila, etiqueta) que pasan de 0 a 1 (rescate) y de 1 a 0 (dano)
  frente a la prediccion de V03 con umbrales congelados; neto. Se reporta siempre.
- Tasa de positivos predicha por etiqueta y precision/recall por etiqueta. Brecha
  train-validacion cuando haya etapa de ajuste.
- Toda corrida reporta tiempo real, device, semillas y versiones de librerias.

### 1.4 Gates y freno D-057

- Gate 0 (antes de cada corrida): hashes congelados verificados; particiones
  reconstruidas; contadores de acceso en 0 para holdout/test; control de identidad V03
  reproducido; criterio de cierre e hipotesis escritos antes de ejecutar.
- Gate 1 (candidato local): delta >= +0.005 sobre V03, >= 4/5 pliegues mejorados e IC95
  pareado que excluya cero. Para combinaciones se admite ademas la via de
  complementariedad: `ensemble_gain >= 0.0015` con IC pareado > 0.
- Gate 2 (confirmacion): la segunda particion (1.2) con la misma receta y SIN reajuste;
  criterio D-070/D-071: delta >= +0.002, >= 4/5 particiones mejoradas e IC95 que excluya
  cero. Un candidato que no pase Gate 2 no se promueve; no se abre una rejilla para
  rescatarlo.
- Gate 3 (leaderboard): solo con autorizacion expresa de Carlos, pipeline, submission,
  ID y sha256 verificados; como maximo una sonda confirmatoria por finalista; el score
  no se usa para reajustar nada.
- Freno D-057 operacionalizado (para EXP-01, EXP-04 y EXP-06, que aumentan capacidad
  o cambian la representacion): medir en el MISMO harness la brecha train-validacion del
  control V03 y la del candidato. La brecha se define como la media por pliegue (y por
  semilla cuando aplique) de (F1 train a 0.5 - F1 validacion a 0.5); la capacidad se
  publica como parametros del modelo mas la dimension densa de entrada. No promover si la
  brecha del candidato supera a la del control en mas de +0.05 absoluto. Para candidatos
  que aumentan capacidad materialmente, el umbral efectivo de promocion es +0.010: con
  delta en [+0.005, +0.010) el candidato NO se promueve (D-057). Dato medido para
  calibrar la regla: en v38_01 la brecha de V03 fue 0.193653 y la de DL-01 0.185431, es
  decir la brecha sola NO incrimina al MLP; la regla se aplica con las dos cifras del
  mismo harness y se publica.

### 1.5 Ruido del leaderboard: correccion vigente (sustituye las cifras del plan)

- El test tiene 1154 filas. El leaderboard publico usa aproximadamente el 70 por ciento
  (~808 filas) y el privado el 30 por ciento (~346 filas). El plan usaba 19.2 aciertos
  netos como vara de detectabilidad.
- Correccion (auditoria, riesgo R1): la cota binomial de 0.01219 corresponde a las 1154
  filas completas. Para el publico (~808) la cota correcta es ~0.0146 y para el privado
  (~346) ~0.0223. Si se escala la banda empirica de 0.01524 por la raiz del cociente de
  tamanos se obtiene ~0.0182 en publico y ~0.0278 en privado, es decir ~23 y ~35
  aciertos netos respectivamente (sensibilidad 7.93e-04 por acierto neto).
- Consecuencia normativa: (a) una ganancia local de +0.005 (unos 6.3 aciertos netos)
  esta POR DEBAJO de la detectabilidad de una sonda publica; (b) una sonda unica no
  confirma ni refuta un candidato de esa magnitud; (c) la decision real descansa en
  Gate 2, no en el LB publico; (d) el ranking final depende del privado, con ruido aun
  mayor. Toda spec que mencione una sonda debe citar estas cifras corregidas, no 19.2.
- Riesgo R6 de la auditoria: el reparto 70/30 se leyo de la pagina renderizada; debe
  reconfirmarse en la sesion autenticada de Carlos antes de cualquier decision que
  dependa de ello.

### 1.6 Familias cerradas (no repeticion) y encaje de P84

Estas familias estan cerradas y ninguna spec las reabre:
perdidas (focal -0.0024, F1 suave -0.0033, ASL -0.1529, PolyLoss -0.0017, Tukey +0.0013
marginal); ComplementNB/NB-SVM/BM25/term weighting; cadenas/powerset/stacking;
calibracion/umbrales (salvo el sliver de EXP-05); SGD/hinge/modified-Huber, hashing,
FTRL/Coin Betting/PiSTOL; rejillas de corpus/ventana/dimension/pooling de embeddings no
supervisados y sus fusiones; RNN/CNN desde cero; reweighting por dureza y rejillas de
class_weight por etiqueta (V4-B03, V4-C04).

Encaje exacto de P84 en esa lista (ver seccion 9):
- EXP-01 no abre familia: es la RE-MEDICION CORREGIDA de DL-01 (mismo mecanismo, protocolo
  limpio). Su valor es cerrar o validar una cifra historica no comparable.
- EXP-02 es nuevo como IMPLEMENTACION (no existe `train_supervised` ejecutado en el
  proyecto) pero NO es una familia funcional nueva: fastText supervisado con pooling medio
  es, algebraicamente, un clasificador lineal sobre bolsa de n-gramas hasheados con
  restriccion de bajo rango, cercano a la familia lineal/SGD/hashing cerrada y al objeto
  que V39-01 declaro redundante. Se ejecuta con expectativa pre-registrada de resultado
  nulo; solo la complementariedad con IC positivo podria abrir EXP-06.
- EXP-03, tal como estaba escrito, es matematicamente degenerado (seccion 4). Se convierte
  en verificacion de equivalencia no promovible.
- EXP-04 no es una rejilla de embedding: fija corpus/vocabulario/dimension/ventana/pooling
  y compara SOLO la agregacion del termino negativo en brazos pareados. Su entregable
  primario es cerrar el defecto de objetivo con evidencia verificable.
- EXP-05 es el unico fragmento permitido de la familia calibracion/umbral.

### 1.7 Archivos, registro y reglas de trabajo

- Un experimento = un script nuevo en `experiments/` con el nombre del preregistro. No se
  edita ni se borra ningun experimento cerrado. Codigo ASCII. Docstrings completos.
  Ejecucion con `/home/carlos/venv/bin/python`.
- Articulos: informes legibles en `reports/p84_0X_*.md`, datos completos en
  `reports/p84_0X_*.json` (ASCII, `ensure_ascii=False` permitido en el JSON solo si el
  contenido no aporta caracteres no ASCII; el codigo y los informes van en ASCII).
  Matrices OOF en `features/p84_0X_*.npz`; embeddings y modelos en `modelos/p84/`.
- Cada experimento anade UNA fila en `experiments.csv` (ids P84-01..P84-06) con las 23
  columnas existentes y `script` apuntando al script nuevo. Idempotencia por
  `experiment_id`: se reescribe la fila, nunca se duplica.
- Cada corrida anade una entrada en `RUNLOG.md` y, si hay decision de cierre o apertura,
  una decision en `DECISIONS.md` (convencion del proyecto; no se usa plan/decisiones/).
- Preregistro de cada experimento: antes de escribir codigo, el plan concreto de ese
  experimento se hashea y se registra como enmienda o addendum de `docs/p84_preregistro.md`
  (mismo patron P80/P82). Sin ese hash no hay experimento.
- Prohibido generar o enviar submissions; la decision de envio es de Carlos.
- Prohibido usar test u holdout para cualquier ajuste, seleccion, umbral o embedding.
  El corpus externo no etiquetado solo puede usarse para entrenar desde cero una
  representacion (D-030); nunca etiquetas, pseudoetiquetado ni clasificador supervisado.

---

## 2. EXP-01 (P84-01): DL-01 corregido

### Objective

Responder si el MLP sobre TF-IDF union -> SVD(300) -> capas 256-128 supera a V03 cuando
se mide sin contaminacion de pool ni de early stopping. Cifra historica a reemplazar:
OOF 0.794854 sobre 1731 filas con epoca elegida sobre el pliegue puntuado; no comparable
con 0.777211 (auditoria, premisa 3, CONFIRMADO). Criterio de exito medible: producir una
medicion valida (positiva o negativa) con control de identidad exacto y decision registrada.
Si falla el Gate 1: cerrar DL-01 como artefacto de protocolo y NO citar 0.794854 como base.

### Inputs

- `data/raw/toxic_train.csv` y `src/splits.py` (solo lectura; no se editan).
- `features/oof_v03_final.npz` (identidad y referencia).
- `experiments/11_dl_desde_cero.py`: SOLO lectura, como fuente de la arquitectura
  historica: MLP 300->256->128->3, ReLU, dropout 0.3; BCEWithLogitsLoss con
  `pos_weight = n_negativos / max(n_positivos, 1.0)` en float32 y en el device, calculado
  con las etiquetas del pliegue (definicion `peso_positivos`, lineas 612-627); AdamW
  lr 1e-3, weight_decay 1e-4; batch 32; max 30 epocas; paciencia 5; clip 1.0; AMP False
  con evidencia medida en esta GPU (BiLSTM 3.42 vs 7.48 s; CharCNN 0.41 vs 1.73 s) y sin
  autocast ni GradScaler; semillas 42/43/44; 110.339 parametros; vectorizador
  REP-05 = word (1,2) min_df=2 sublinear + char_wb (3,5) min_df=2 sublinear, lowercase,
  sin strip_accents; TruncatedSVD(300, random_state=42); StandardScaler; F1 a 0.5.
  El script historico NO se importa para entrenar ni se modifica; el nuevo script
  reimplementa la arquitectura con estos valores y cita la procedencia.
- Mecanismo de particion interna verificado: `partir_interna` de
  `experiments/p73_02_textcnn_vs_random.py` (holdout anidado 80/20 por grupos con
  tolerancia de prevalencia 0.07, 20 intentos), reutilizado por importacion o
  reimplementado con la misma regla y semillas; en ambos casos se verifica equivalencia
  sobre una semilla. P78 valido este mecanismo con 180 llamadas reales via p75_01.

### Process

1. Gate 0: verificar hashes; reconstruir dev/holdout/pliegues; contadores de acceso a 0
   para holdout y test; reconstruir el OOF canonico de V03 con el harness nuevo y exigir
   la tolerancia de 1.1; verificar V03@0.5 = 0.771778476256122 recalculado.
2. Para cada pliegue externo k = 1..5 (patron identico a p75_01):
   - Reparto interno: 3 splits agrupados por grupos fusionados con semillas
     `20261101 + 100*k + 7*j`, j = 0,1,2; usar `partir_interna` (80/20, tolerancia de
     prevalencia marginal por etiqueta 0.07). OJO: `partir_interna` controla prevalencia
     marginal por etiqueta, no la combinacion HSxTRxAG; registrar el numero de intentos de
     cada split y AVISAR si la tolerancia no se cumple tras 20 intentos, sin silenciar el
     fallback.
   - Por cada split j y cada semilla s en {42, 43, 44}: ajustar vectorizador, SVD(300)
     y StandardScaler SOLO con `inner_train`; transformar `inner_val`; entrenar el MLP
     con early stopping sobre F1@0.5 de `inner_val` (paciencia 5, mejores pesos); registrar
     la epoca del MEJOR peso restaurado (`mejor_epoca`, no las epocas ejecutadas) como E_j.
     El fold externo NO interviene en este paso.
   - E* = mediana de (E_0, E_1, E_2) por semilla.
   - Reajuste: reajustar vectorizador, SVD y scaler SOLO con `outer_train`; entrenar un
     MLP nuevo sobre `outer_train` durante exactamente E* epocas y predecir `outer_val`
     UNA sola vez por (semilla, pliegue); el contador cuenta pasadas forward de `outer_val`
     (exactamente 1 por combinacion) y las tres lecturas de umbral reutilizan la misma
     matriz de probabilidades. Abortar si se supera el contador.
3. Variables y donde se ajustan (anti-leakage): vectorizador/IDF, SVD, scaler, pos_weight,
   epoca E* y (en la lectura secundaria) umbrales se ajustan SOLO dentro de
   `outer_train` o sus particiones internas. `outer_val` se transforma con objetos
   ajustados en `outer_train`. El test no se carga en el script.
4. Lecturas por pliegue y semilla: primaria a umbral 0.5 (control de arquitectura);
   secundaria con umbral por etiqueta elegido sobre las probabilidades internas
   agrupadas de los 3 splits (rejilla declarada 0.20 a 0.80 paso 0.01, criterio F1
   maxima por etiqueta); control de transferencia con los umbrales congelados de V03
   (0.44/0.51/0.46). Se publican las tres; la primaria decide el gate.
5. Comparacion pareada vs V03 (OOF de 1483 filas): delta por pliegue, delta medio,
   pliegues mejorados, IC bootstrap 10.000/semilla 20260926, rescate/dano, tasa de
   positivos, F1 por etiqueta, precision/recall, brecha train-validacion, parametros y
   tiempos.
6. Estabilidad entre semillas: recalcular delta medio para las semillas 43 y 44. Condicion
   adicional pre-declarada: las tres semillas deben tener delta del mismo signo y la
   desviacion tipica entre semillas del delta debe ser menor que la mitad del delta medio
   de la semilla 42; si no, el resultado se publica como NO ESTABLE y no se promueve.
7. Freno D-057 (1.4): medir la brecha del control V03 en este harness y aplicar la regla
   declarada (aumento de brecha <= +0.05; capacidad publicada).
8. Decision: si Gate 1 pasa con estabilidad y freno, el candidato va a Gate 2 (no a
   submission). Si no, registrar cierre de DL-01 como artefacto de protocolo.

### Boundaries

- Always: control de identidad exacto antes de leer candidatos; contadores de acceso;
  E* elegido solo en particiones internas; reajuste sobre `outer_train` completo;
  prediccion unica de `outer_val`; bootstrap con semilla 20260926; documentar device,
  semillas y versiones; codigo ASCII y docstrings.
- Ask first: usar GPU con `torch.use_deterministic_algorithms(True)` (evidencia historica
  de cuelgue HSA en ROCm 7.2); cambiar el numero de semillas; cualquier variante de
  arquitectura; reutilizar el checkpoint historico.
- Never: tocar holdout o test; seleccionar epoca, umbral o modelo con `outer_val`; usar
  embeddings preentrenados; editar `experiments/11_dl_desde_cero.py`; citar 0.794854
  como resultado valido o comparable; generar submissions; abrir una rejilla de
  arquitecturas si el resultado es negativo.

### Success Criteria

- Gate 0: identidad V03 reproducida (tolerancia 1e-12) y contadores en 0/0.
- Gate 1: delta >= +0.005, >= 4/5 pliegues, IC95 pareado > 0 en la lectura primaria
  (umbral 0.5) de la semilla 42. Para candidatos que aumentan capacidad, el umbral
  efectivo de promocion es +0.010 (D-057): con delta en [+0.005, +0.010) el candidato
  NO se promueve.
- Estabilidad: mismo signo en las 3 semillas y sd entre semillas < delta/2.
- Freno D-057: brecha = media por pliegue y semilla de (F1 train a 0.5 - F1 validacion
  a 0.5) en el mismo harness para V03 y para el MLP; no promover si la brecha del
  candidato supera a la del control en mas de +0.05 absoluto. Capacidad publicada
  (110.339 parametros + dimension densa efectiva del pliegue).
- Resultado negativo valido: cierre registrado de DL-01 con la medicion corregida; la
  cifra historica queda marcada como no comparable y no se arrastra.

### Outputs

- Script: `experiments/p84_01_dl01_corregido.py`.
- Informes: `reports/p84_01_dl01.json` y `reports/p84_01_dl01.md`.
- OOF y curvas: `features/p84_01_oof_mlp.npz` (proba por semilla, 1483x3, pliegues,
  y_true, ids), `reports/figuras/p84_01_curvas.png` (opcional).
- Checkpoints: `modelos/p84/` con el `state_dict` final por (semilla, pliegue) y la E*
  usada, mas la dispersion de las E_j internas publicada en el informe.
- `experiments.csv`: fila P84-01 con f1 = resultado primario (umbral 0.5), resultado
  completo (lecturas 0.5/anidada/congelada, deltas, IC, pliegues, semillas, brecha),
  decision (promover a Gate 2 o cerrar DL-01).
- `RUNLOG.md` y `DECISIONS.md` (decision de cierre/apertura).
- Enmienda hasheada en `docs/p84_preregistro.md` antes de codificar.

### Open Questions

- La semilla 42 debe ser la primaria o conviene declarar la semilla mediana de las tres?
  (Afecta la comparabilidad con DL-01-SEEDS, que uso 42/43/44; la spec fija 42.)
- El coste de 3 semillas x 5 pliegues x 4 entrenamientos es de decenas de minutos; si el
  presupuesto se comprime, se permite bajar a 2 semillas solo con enmienda previa.
- Determinismo (protocolo fijado antes de ver resultados): sembrar por split con
  `fijar_semillas(semilla + j)` y en el refit con `fijar_semillas(semilla)`, patron de
  `11_dl_desde_cero.py:267-284` (`use_deterministic_algorithms(False)` y flags de cudnn).
  Pre-vuelo de doble pasada en (semilla 42, pliegue 1) con tolerancia `max|delta p| <= 1e-6`;
  si falla, reiniciar en CPU y anotarlo. Guardia adicional: evaluar `outer_val` dos veces y
  exigir `torch.equal` sobre las probabilidades.

---

## 3. EXP-02 (P84-02): clasificador fastText supervisado desde cero

### Objective

Medir si un clasificador fastText supervisado entrenado SOLO con textos y etiquetas
oficiales del `outer_train`, con subwords y word n-grams, aporta errores complementarios a
V03. Novedad verificada: no existe `train_supervised` en el proyecto (solo el embedding no
supervisado y la factorizacion supervisada V38-01, que no es el clasificador de Joulin et
al.). Expectativa pre-registrada: alta correlacion con los char n-grams de V03 y
posibilidad baja de ganancia standalone; la via realista es la complementariedad
(`ensemble_gain >= 0.0015` con IC > 0). Si no pasa ningun gate, cerrar subwords
supervisados sin abrir rejilla.

### Inputs

- Binario oficial: `tools/fastText/build/fasttext`, sha256
  59c51cc53802bcad9e7832145f6525a281d22d42e7c476bfbac545c0c392137f.
  (El build tiene `-seed` y `-thread` verificados en `tools/fastText/src/args.cc:190,177`.)
- Datos: desarrollo 1483 filas; pliegues canonicos; V03 npz; `src/splits.py`.
- Enmienda hasheada del preregistro con la rejilla declarada.
- Nada externo: sin vectores preentrenados, sin corpus externo, sin test, sin holdout.

### Process

1. Gate 0: hashes, particiones, contadores 0/0, identidad V03 con el harness (aunque
   fastText no use el harness de V03, la comparacion es contra el OOF congelado).
2. Construccion del archivo de entrenamiento por pliegue y etiqueta: una linea por fila
   de `outer_train`, texto normalizado con `lower()` y nada mas (NO se reutiliza
   `normalizar_emb_normal` de `src/embedding_utils.py`: ese normalizador es del pipeline de
   embeddings y el envoltorio de fastText alli solo cubre skipgram, no `supervised`);
   etiqueta explicita
   `__label__POS` o `__label__NEG` segun la etiqueta binaria. El texto de `outer_val`
   nunca entra al archivo de entrenamiento.
3. Hiperparametros fijados (un solo mecanismo; sin rejilla de corpus/dimension/ventana):
   `-dim 100 -lr 0.1 -minCount 2 -bucket 2000000 -loss softmax -wordNgrams 2 -minn 3
   -maxn 6 -thread 1 -seed 20260926 -verbose 2`; epocas E* elegidas en particiones
   internas; `minCount 2` se justifica por el tamano del pliegue (~1.000 filas) y queda
   fijo, no se barre. Brazo principal: subwords 3-6. Control unico pre-declarado: mismo
   todo con `-minn 0 -maxn 0` (sin subwords).
4. Particiones internas: 3 splits agrupados con el mecanismo `partir_interna` y semillas
   `20261101 + 100*k + 7*j` (las mismas que EXP-01, para que los pliegues internos sean
   identicos entre experimentos del mismo pliegue externo).
5. Seleccion de epocas: por brazo, entrenar en `inner_train` con epocas en {5, 10, 20},
   evaluar F1@0.5 por etiqueta en `inner_val`, elegir por split la epoca de F1 media
   maxima (empate: menor epoca); E* = mediana de los 3 splits. Solo con E* se reentrena
   sobre `outer_train` y se predice `outer_val` una vez por (brazo, pliegue).
6. Seleccion de umbral (capa de decision): con las probabilidades de `inner_val` de los
   3 splits (modelos ya entrenados en `inner_train`), agruparlas y elegir por etiqueta el
   umbral de F1 maxima en la rejilla 0.20 a 0.80 paso 0.01. Se aplica a `outer_val`.
   Lecturas publicadas: anidada (primaria), 0.5 (control de arquitectura) y umbrales
   congelados de V03 (control de transferencia).
7. Extraccion de probabilidad positiva: `fasttext predict-prob` sobre el texto de
   `outer_val` con `k=2` (por defecto emite k=1), parseando la probabilidad del token
   `__label__POS` por su nombre y NUNCA por posicion. Nunca se lee probabilidad de test.
8. Comparaciones: (a) principal vs V03 (OOF congelado) con la lectura anidada; (b) main
   vs control sin subwords (mismo procedimiento) para cuantificar el aporte sublexico;
   (c) complementariedad: `ensemble_gain` (1.3) + correlacion Pearson/Spearman entre
   probabilidades y matriz de solapamiento de errores.
9. Determinismo: `-thread 1`; dos corridas identicas sobre un pliegue deben producir
   probabilidades identicas (tolerancia 0). Publicar el resultado.
10. Decision: si Gate 1 standalone (delta >= +0.005, 4/5, IC > 0) o complementariedad
    (`ensemble_gain >= 0.0015`, IC > 0) se cumple, el candidato queda elegible para
    EXP-06. Si no, cerrar fastText supervisado sin buscar otra rejilla.

### Boundaries

- Always: usar solo textos y etiquetas oficiales del `outer_train`; archivo de
  entrenamiento separado por pliegue y etiqueta; `-thread 1`; fijar semilla y
  documentarla; control sin subwords con todo lo demas igual; reportar la correlacion con
  V03 y la matriz de errores aunque el resultado sea malo.
- Ask first: cualquier cambio de `-lr`, `-minCount`, `-loss`, `-dim`, `-bucket` o del
  conjunto {5,10,20}; ampliar el control a una rejilla; usar la API Python en lugar del
  binario (motivo: trazabilidad del binario congelado).
- Never: usar vectores preentrenados (`-pretrainedVectors`); usar corpus externo (esto
  violaria D-030 para clasificacion supervisada); tocar holdout/test; usar el texto de
  validacion para vocabulario o epocas; reabrir la familia lineal con otro nombre si el
  resultado es negativo; generar submissions.

### Success Criteria

- Gate 0 completo; determinismo verificado en un pliegue (variacion exactamente cero).
- Gate 1 standalone: +0.005, 4/5, IC95 > 0; o gate de complementariedad: ensemble_gain
  >= 0.0015 con IC95 > 0.
- El control sin subwords se publica completo, gane o pierda; y el resultado principal se
  publica completo aunque sea negativo.
- Si falla: cierre documentado de fastText supervisado con la medicion, sin rejilla nueva.

### Outputs

- Script: `experiments/p84_02_fasttext_supervisado.py`.
- Informes: `reports/p84_02_fasttext.json`, `reports/p84_02_fasttext.md`.
- OOF: `features/p84_02_oof_fasttext.npz` (proba main y control, 1483x3).
- Modelos temporales por pliegue/etiqueta en `experimentos/p84_02_tmp/` (se pueden borrar;
  no se consideran artefactos).
- `experiments.csv`: fila P84-02 (f1 = lectura primaria anidada; resultado con las tres
  lecturas, deltas, IC, ensemble_gain, correlaciones); `RUNLOG.md`; `DECISIONS.md`.
- Enmienda hasheada previa en `docs/p84_preregistro.md`.

### Open Questions

- `-epoch` con valores altos puede sobreajustar con ~1.000 filas; si E* resultara siempre
  20 en los 5 pliegues, se reportara como senal de que el rango es demasiado bajo, sin
  ampliarlo en esta fase.
- El control sin subwords con `-wordNgrams 2` es practicamente un modelo lineal de
  n-gramas de palabra; si gana al brazo con subwords, la conclusion es sobre ESTE
  conjunto y se publica como tal.
- La escala de las probabilidades softmax de fastText puede hacer fracasar `ensemble_gain`
  por calibracion aunque la informacion sea complementaria; la spec NO cambia la
  convencion, pero exige publicar la correlacion de rangos y el solapamiento de errores
  como diagnosticos.

---

## 4. EXP-03 (P84-03): sobremuestreo aleatorio, redisenado a verificacion de equivalencia

### Objective

El plan pedia medir si ROS aporta algo mas que `class_weight=balanced` con Gate 1 normal.
Ese diseno tiene un fallo cientifico: para un modelo lineal convexo con L2 (liblinear) la
duplicacion de filas es EXACTAMENTE una reasignacion de pesos por instancia, y la
instruccion del plan de "igualar la escala media de la perdida" convierte el brazo ROS en
una copia del control. Ademas, en cualquiera de sus variantes, ROS cae dentro de familias
ya cerradas (rejilla de class_weight V4-C04 y reweighting V4-B03, ambas negativas).
Objetivo corregido: ejecutar una micro-verificacion de equivalencia (horas, CPU) que deje
un artefacto verificable, y cerrar la familia ROS como no promovible. No existe gate de
superioridad aplicable; un delta positivo del brazo no normalizado seria un efecto de
capacidad (C efectivo) ya cubierto por familias cerradas, no un efecto de ROS.

Fundamento algebraico (se publica y se verifica). Notacion: razon de remuestreo r =
n_pos'/n_neg (positivos tras ROS entre negativos); duplicacion de una fila =
`sample_weight` multiplicado por el numero de copias; liblinear resuelve
`0.5*||w||^2 + C * suma_i s_i * loss_i`:
- ROS con razon r y masa total normalizada a n por pliegue es exactamente un
  `class_weight` con razon de pesos `w_pos/w_neg = r * (n_neg/n_pos)` y C=2. Para r=1
  la razon es `n_neg/n_pos`, es decir EXACTAMENTE `class_weight=balanced` (salvo redondeo
  entero y numerica del solver). Para r<1 es una reweighting menos agresiva que balanced,
  que sigue dentro de la familia de `class_weight` cerrada por V4-C04.
- ROS sin normalizar a C=2 nominal es el mismo `class_weight` de razon
  `r * (n_neg/n_pos)` con `C_efectivo = C * n_neg * (1 + r) / n` (con r=1 y C=2 ronda 2.25 en un pliegue tipico), es decir mezcla ratio de
  clase y capacidad (ambas familias cerradas: class_weight V4-C04 y barridos de C).
Ambas cosas estan cubiertas por familias cerradas; por eso ROS no puede promover.

### Inputs

- Desarrollo, pliegues canonicos, V03 npz, particiones internas (mecanismo compartido).
- Enmienda hasheada que registra la correccion del diseno y el caracter no promovible.
- Nada externo; ni SMOTE/ADASYN (cerrados) ni undersampling (fuera de alcance).

### Process

1. Gate 0 e identidad V03 exactos.
2. Brazos por pliegue, vectorizador/SVD/IDF ajustados SOLO con `outer_train` y ANTES de
   cualquier duplicacion (la representacion no cambia; solo cambia el ajuste del
   clasificador):
   - A: control V03 exacto (LR C=2, balanced, liblinear, rs=42).
   - B1: ROS r=1.0 con masa total normalizada a n (media 1) y C=2. Equivalencia esperada
     con A (razon de pesos n_neg/n_pos).
   - B2: ROS r=0.5 con masa total normalizada a n y C=2. Es class_weight con razon
     0.5*(n_neg/n_pos); se espera sin mejora.
   - B3: ROS r=1.0 SIN normalizar (C=2 nominal); se reporta C_efectivo por pliegue
     (`C * n_neg * (1 + r) / n`, la masa total ROS entre la masa balanceada n).
3. Duplicacion determinista: `n_pos' = round(r * n_neg)`; las copias se eligen con un
   generador sembrado 20260926 sobre una permutacion de los indices positivos de
   `outer_train`; se reportan los recuentos enteros por pliegue y etiqueta. La validacion
   nunca se duplica ni se usa para elegir copias.
4. Comparaciones: delta de cada brazo vs A y vs V03 (mismo protocolo de 1.3: por pliegue,
   IC bootstrap 10.000/20260926, rescate/dano, tasa predicha). Verificacion de equivalencia:
   `max |proba_B1 - proba_A| <= 1e-3` y `|delta_F1| <= 0.001`; si no se cumple, STOP y
   reporte de REFUTACION PARCIAL de la equivalencia con su causa (redondeo entero o
   trayectoria numerica del solver), nunca promocion.
5. Decision pre-declarada: registrar "ROS cerrado por equivalencia con reweighting y
   class_weight; no promovible" con los numeros. Ningun brazo va a Gate 2 ni a EXP-06.

### Boundaries

- Always: publicar la derivacion y las cifras de equivalencia; ajustar el vectorizador en
  `outer_train` antes de duplicar; duplicar solo en el ajuste del clasificador; usar
  semilla declarada para las copias; reportar C_efectivo.
- Ask first: ejecutar la micro-verificacion o cerrar por demostracion sin ejecutarla
  (ambas opciones son validas; la segunda debe registrar el cierre por algebra);
  cualquier cambio de r o de normalizacion.
- Never: presentar ROS como mecanismo nuevo; promover un brazo ROS; usar sintesis
  (SMOTE/ADASYN) ni undersampling; aplicar ROS antes de dividir; tocar holdout/test;
  abrir una rejilla de ratios o de C para rescatar la linea.

### Success Criteria

- A reproduce la identidad V03 (1e-12); B1 cumple la equivalencia declarada
  (|delta| <= 0.001 y max diff de probabilidades <= 1e-3) o se reporta la refutacion con
  su causa.
- B2 y B3 quedan medidos y etiquetados como class_weight/C, no como ROS.
- Decision registrada de cierre de ROS; `experiments.csv` con la fila P84-03 marcada
  "CLOSED BY EQUIVALENCE / no promovible".

### Outputs

- Script: `experiments/p84_03_ros_equivalencia.py`.
- Informes: `reports/p84_03_ros.json`, `reports/p84_03_ros.md` (con la derivacion, las
  cifras de equivalencia y la tabla de brazos).
- OOF: `features/p84_03_oof_ros.npz` (opcional; proba por brazo).
- `experiments.csv`: fila P84-03; `RUNLOG.md`; `DECISIONS.md` (cierre de la familia ROS).

### Open Questions

- Si B1 no reproduce a A dentro de tolerancia, cuanto del desvio es redondeo entero y
  cuanto es trayectoria del solver? Se reporta, pero no se convierte en gate.
- Tiene algun valor informativo ejecutar la micro-verificacion (horas) frente a cerrar
  por demostracion? La spec recomienda ejecutarla por el estandar de evidencia del
  proyecto, pero la decision de presupuesto es de Carlos (Ask first).

---

## 5. EXP-04 (P84-04): SGNS corregido y anclado a implementacion externa

### Objective

Cerrar el defecto de objetivo confirmado del entrenador Word2Vec del proyecto: el termino
negativo promedia sobre K (peso 1/K) cuando SGNS publicado suma las K contribuciones
(P79/P80 H1, P81 bloque 1). Pregunta: cambia el downstream cuando se corrige el objetivo,
con todo lo demas fijo y pareado? El entregable primario es cerrar la pregunta con
evidencia verificable (ancla + brazos pareados); NO se promete mejora ni se afirma que el
artefacto congelado sea peor que una libreria (P81: anclaje externo pendiente). Si el
brazo corregido no pasa el gate, cerrar la linea con resultado verificable.

### Inputs

- Corpus externo no etiquetado autorizado bajo D-030 y variante EXTERNAL-ONLY:
  `data/external/corpora/wikipedia_es/emb_normal/tokens_10M.txt`, sha256
  00b839da6fbc377ee5375acc9a236e831e46ecf92cc544188b81aef1a826e871.
  No se anade texto de desarrollo, holdout ni test al corpus (EXTERNAL-ONLY estricto).
- Binario oficial para el ancla: `tools/fastText/build/fasttext` (sha256 del preregistro).
- Configuracion del entrenador que si aprende (p71_08, corregida por learning rate):
  tokenizador `[#@]?\w+(?:['\u2019]\w+)*`, min_count 3, ventana 5 dinamica, negativos 10,
  subsampling 1e-4, 5 epocas, lote 256, SGD con lr = 0.025*lote con decaimiento lineal
  hasta 1e-3, entrada uniforme +-0.5/dim, salida uniforme [-0.05, 0.05].
- Dimension fija: 200. Razon declarada: la unica configuracion corregida demostrada que
  aprende es la de p71_08 (dim 200); la dimension NO es la pregunta y no se barre; el
  artefacto 100d congelado no se usa como control (distinto objetivo y distinta
  trayectoria; P80 A13 lo declara no evaluable como par E/S).
- Referencias read-only: `experiments/p71_04_entrenar_word2vec.py`,
  `experiments/p71_08_word2vec_corregido.py`, `experiments/p71_10_downstream_corregido.py`.
  No se editan; el nuevo script reimplementa el bucle con el objetivo corregido.

### Process

1. Gate 0 + identidad V03. Verificar hash del corpus y del binario.
2. Ancla de implementacion ANTES de entrenar el embedding final:
   - Prueba algebraica: con los mismos tensores y pesos, el termino negativo cumple
     `N_suma = K * N_media` exactamente y la perdida completa cumple
     `loss_suma = K * loss_media + (K - 1) * P`, donde P es la media del log-sigma
     positivo (tolerancia relativa 1e-9, K=10).
   - Chequeo de gradiente por diferencias finitas del objetivo SUMA respecto de las
     matrices de entrada y salida sobre un lote sintetico diminuto (error relativo <= 1e-5).
   - Ancla ejecutable: corpus plantado de dos familias disjuntas de palabras; entrenar
     (a) el binario oficial con el subcomando `skipgram` (no existe `-model sg` en el CLI):
     `skipgram -minn 0 -maxn 0 -dim 32 -neg 10 -epoch 5 -thread 1 -seed 20260926
     -minCount 1` (el default de minCount es 5 y el corpus plantado puede quedar sin
     vocabulario) y (b) nuestra implementacion corregida con la configuracion equivalente
     (epochs 5, neg 10, dim 32, minn/maxn 0, minCount 1); medir pureza de vecinos por
     familia en ambas y exigir
     `pureza_nuestra >= pureza_oficial - 0.05` (tolerancia declarada). Si falla el ancla,
     se reporta ANCLA NO LOGRADA y los resultados downstream NO se atribuyen a la
     semantica SGNS publicada (persiste el limite de P81).
   - Advertencia fijada por P81: el ancla verifica SEMANTICA de implementacion, no
     direccion de rendimiento. P81 A6 refuto que el test plantado favorezca a la SUMA;
     por eso el ancla no exige que la SUMA gane, solo que la implementacion implemente
     lo publicado y que nuestros numeros sean interpretables.
3. Brazos pareados (mismo corpus, mismo vocabulario, misma semilla, mismo orden de lotes
   y mismos negativos por paso; solo cambia la agregacion del termino negativo):
   - S: SUMA de las K contribuciones (objetivo corregido).
   - M: media 1/K (objetivo antiguo) como control.
   - R: matriz aleatoria congelada con la misma forma y escala (control negativo),
     sembrada por semilla de brazo.
   Semillas: {20261016, 20261017, 20261018}; en cada semilla el flujo aleatorio de datos
   (pares, permutacion, negativos) es identico entre S y M; R no entrena.
   Nota de lectura: `loss_suma` y `loss_media` NO son comparables entre si por
   construccion (difieren en K en el termino negativo) y con lr fijo la suma tambien
   escala el gradiente negativo por K: la comparacion S vs M confunde la correccion del
   objetivo con la magnitud del gradiente. Se declara y se publican las normas de
   gradiente; solo se comparan metricas downstream. Vigilar el descenso de loss de los
   tres brazos: el corpus wiki 10M difiere del twitter 10M donde p71_08 valido que el
   entrenador aprende.
4. Downstream primario (fijo, sin rejilla): mean pooling de los tokens dentro de
   vocabulario (los OOV se omiten, como en p71_10), StandardScaler ajustado en
   `outer_train`, LR C=2.0 balanced liblinear rs=42 por pliegue; umbrales congelados de
   V03 (lectura de transferencia) y umbral anidado (lectura secundaria, rejilla 0.20 a
   0.80 paso 0.01 elegida SOLO en particiones internas). Se compara S vs M y S vs R
   (pareado por semilla y pliegue) y S vs V03.
5. Complementariedad: `ensemble_gain` de S con V03 (convencion 1.3) y matriz de errores.
6. Downstream secundario (solo si S muestra senal sobre M en el paso 4): fusion del bloque
   denso con el TF-IDF de V03 a UNA escala declarada (convencion de norma de fila igual,
   escala 1.0) y/o media de probabilidades. Se etiqueta exploratorio: la familia de
   fusiones de embeddings no supervisados esta cerrada (mejor fusion historica 0.782959
   con IC [-0.002084, 0.013625]).
7. Gate: S debe superar a M y a R (delta > 0, >= 4/5 pliegues, IC > 0) y ademas superar a
   V03 con +0.005/4/5/IC > 0, o bien cumplir `ensemble_gain >= 0.0015` con IC > 0. En la
   practica el unico camino plausible es la complementariedad; se declara por adelantado.
   Aunque S no supere a M, se publica `ensemble_gain` de S con V03 como diagnostico (no
   como gate).
8. Decision: si pasa, elegible para EXP-06 y Gate 2. Si no, cerrar la familia con el
   resultado verificado (objetivo corregido medido, ancla documentada).

### Boundaries

- Always: EXTERNAL-ONLY (el corpus no incluye texto de la competencia); misma semilla y
  mismo flujo de datos entre S y M; ancla antes de entrenar; publicar la formula del
  objetivo y las normas de gradiente; reportar tiempos y device; 3 semillas.
- Ask first: bajar de 3 semillas; cambiar dim/ventana/min_count/negativos/epocas/lote
  (cualquiera de esos cambios convierte el experimento en una rejilla cerrada); anadir
  texto de la competencia al corpus (prohibido por D-030 y por el plan); usar el
  artefacto 100d congelado como control.
- Never: comparar loss entre brazos como si fueran el mismo funcional; presentar el
  resultado como mejora de Kaggle; usar validacion/holdout/test en el embedding; fusionar
  con la rejilla de escalas cerrada; abrir cabeza secuencial antes de que S muestre senal
  sobre M; editar p71_04/p71_08/p71_10; generar submissions.

### Success Criteria

- Ancla: identidad algebraica y gradiente correctos; ancla ejecutable dentro de la
  tolerancia declarada (o reporte explicito de ANCLA NO LOGRADA).
- Downstream: las tres semillas x 5 pliegues completos para S y M (y R); comparaciones
  pareadas con IC95; sin fuga (contador de acceso).
- Gate: S > M y S > R con IC > 0 y >= 4/5, y (S vs V03 +0.005/4/5/IC > 0 o ensemble_gain
  >= 0.0015 con IC > 0). Si no, cierre documentado de la linea con el defecto resuelto.
- Tiempo real publicado con hito de decision a los 40 minutos por brazo-semilla
  (presupuesto estimado: ~23 min por corrida de 10M tokens/5 epocas/dim 200, medido en
  p71_08: 1402.9 s; 3 brazos x 3 semillas mas downstream, menos de 1 dia).

### Outputs

- Script: `experiments/p84_04_sgns_corregido_anclado.py`.
- Informes: `reports/p84_04_sgns.json`, `reports/p84_04_sgns.md`,
  `reports/p84_04_ancla.json` (detalle del ancla y del chequeo de gradiente).
- Embeddings por brazo/semilla en `modelos/p84/` (npz con W y vocabulario, sha256).
- OOF: `features/p84_04_oof_emb.npz` (proba por brazo/semilla, 1483x3).
- `experiments.csv`: fila P84-04; `RUNLOG.md`; `DECISIONS.md` (cierre de la linea si no
  pasa gate).
- Enmienda hasheada previa (fija dim 200, corpus wikipedia_es 10M y la decision de no
  rejilla).

### Open Questions

- Si el ancla ejecutable no alcanza la tolerancia por determinismo multihilo del binario
  oficial, se permite `-thread 1` en ambos lados y se documenta; no se permite cambiar el
  criterio de pureza despues de ver el resultado.
- La eleccion dim 200 puede hacer incomparables los numeros con la linea historica 100d;
  es intencional y se declara (no se comparan).
- Sobre el corpus: el preregistro congela wikipedia_es 10M; si se prefiere el corpus
  twitter_domain historico, requiere enmienda previa con su hash (Ask first).

---

## 6. EXP-05 (P84-05): correccion de prevalencia EM con shrinkage, simulacion OOF

### Objective

Medir si la correccion de odds por cambio de prevalencia (EM) mejora cuando se encoge
hacia cero y se decide de forma robusta sobre un intervalo, en simulacion sobre el OOF de
desarrollo. No se lee test. Expectativa muy baja. Cierre de la familia EM/umbral si falla.

Correccion metodologica al plan: la seleccion conjunta de shrinkage m y umbral t no puede
hacerse y evaluarse sobre las mismas filas. Se sustituye por un esquema dejar-un-pliegue-
fuera (LOFO): para cada pliegue k, seleccionar (m, t) con minimax sobre los OTROS cuatro
pliegues y evaluar la regla seleccionada en el pliegue k. Las 5 evaluaciones LOFO son las
"5 repeticiones/folds" del gate. Asi la seleccion no ve las filas que puntua.

### Inputs

- `features/oof_v03_final.npz`: proba OOF (1483x3), y_true, pliegues, umbrales congelados.
- Prevalencias de desarrollo: HS 0.436952, TR 0.300067, AG 0.382333.
- Nada mas: sin test, sin holdout, sin reentrenar modelos.

### Process

1. Gate 0 e identidad (aqui el npz es la fuente; verificar hash y recalcular
   V03@0.5 = 0.771778476256122 y corrected_cv = 0.7772114525165721).
2. Simulacion de shift por etiqueta c y razon rho en {0.85, 0.90, 0.94, 1.00, 1.05}:
   ponderacion por importancia de las filas del OOF, w_i = rho si y_i = 1 y w_i = 1 si
   y_i = 0. El estimando es la F1 ponderada
   `2*suma(w*y*pred) / (suma(w*y) + suma(w*pred))` por etiqueta y su media; es una
   simulacion analitica determinista que asume p(x|y) invariable. Se declara el supuesto.
3. Estimador EM con shrinkage: para la etiqueta c, iterar hasta 1e-10
   `p'_i = sigmoid(logit(p_i) + log rho_t)`, `pi_t = suma(w_i p'_i)/suma(w_i)`,
   `rho_{t+1} = (pi_t/(1-pi_t)) / (pi_train/(1-pi_train))`; shrinkage
   `rho_usado = exp(s * log rho_hat)`, `s = n_eff/(n_eff + m)`, m en {0, 25, 50, 100},
   `n_eff` = tamano efectivo de Kish de los pesos (se publica). Correccion de scores:
   `p''_i = sigmoid(logit(p_i) + log rho_usado)`; prediccion `p''_i >= t`.
4. Rejilla declarada: m en {0, 25, 50, 100}; t en 0.20 a 0.80 paso 0.01. Criterio
   minimax: maximizar el minimo de la F1 ponderada sobre las 5 razones rho; empate:
   menor m y despues t mas cercano al umbral congelado de V03.
5. Seleccion LOFO: para cada pliegue k, seleccionar (m, t) por etiqueta con las filas de
   los otros 4 pliegues; aplicar al pliegue k; comparar contra V03 (umbrales congelados
   0.44/0.51/0.46, sin correccion) bajo los mismos pesos y la misma rho.
6. Controles obligatorios:
   - Descomposicion del efecto: comparar la regla seleccionada contra una regla solo
     umbral (mismo t, rho = 1, sin correccion EM) en las mismas condiciones. Si la
     ganancia viene solo del cambio de umbral, la familia EM se cierra (el umbral es
     familia cerrada); se publican ambos numeros.
   - Sanidad sin shift: delta de la regla completa a rho = 1.00 debe ser >= 0.
   - Guardarrail de tasa: a rho = 1.00, la tasa predicha por etiqueta debe quedar a
     +-0.005 de la de V03; es requisito de la regla final (la curva F1 vs tasa del LB
     tiene maximo en V03). Se publican tambien las tasas a todas las rho y el optimo
     sin restriccion, como diagnostico.
7. Gate (unico): delta medio (promedio de las 5 rho) >= +0.002; >= 4/5 pliegues LOFO
   positivos; IC95 bootstrap pareado por filas (10.000, 20260926, con pesos) que excluya
   cero; guardarrail de tasa cumplido; componente EM (paso 6) positiva. Si falla, cerrar
   toda la familia EM/threshold; ninguna lectura de test.
8. Decision: si pasa, elegible para Gate 2 (segunda particion) y para la matriz de
   cierre; nunca submission automatica.

### Boundaries

- Always: reportar la superficie completa (m, t, rho) y los numeros LOFO; declarar el
  estimando ponderado y el supuesto p(x|y) invariable; recalcular las referencias desde
  el npz; publicar n_eff y las tasas predichas; usar bootstrap con pesos y semilla
  20260926.
- Ask first: cambiar la rejilla, el rango de rho o el criterio minimax; aplicar la regla
  a probabilidades de test (requiere autorizacion explicita y no forma parte de P84).
- Never: leer o almacenar probabilidades de test; seleccionar (m, t) con las mismas
  filas que se puntuan; presentar un cambio de umbral disfrazado de EM; reabrir
  calibracion/threshold fuera de este sliver; usar el holdout; generar submissions.

### Success Criteria

- Gate completo segun 6.7, con las cinco evaluaciones LOFO publicadas y el desglose por
  rho y por etiqueta.
- Si falla: cierre documentado de EM/threshold con la superficie publicada; ninguna
  variante nueva.

### Outputs

- Script: `experiments/p84_05_em_shrinkage_simulado.py`.
- Informes: `reports/p84_05_em.json`, `reports/p84_05_em.md` (incluye la tabla de
  seleccion LOFO y la descomposicion EM vs umbral).
- `experiments.csv`: fila P84-05; `RUNLOG.md`; `DECISIONS.md` (cierre o apertura).
- Enmienda hasheada previa (fija LOFO, rejilla, guardarrail y semilla bootstrap).

### Open Questions

- `n_eff` de Kish reduce el shrinkage cuando los pesos son muy desiguales; se publica el
  valor por etiqueta y rho, sin cambiarlo a mitad de camino.
- El guardarrail de tasa puede hacer imposible cualquier ganancia si el unico mecanismo
  fuera mover el volumen; eso es exactamente lo que se quiere medir. Si el optimo
  restringido es identico a V03 (delta 0), el cierre es informativo.

---

## 7. EXP-06 (P84-06): puerta de combinacion de candidatos

### Objective

Combinar candidatos SOLO si han pasado su gate, con una regla pre-registrada, manteniendo
V03 como control y sin stacking generico. Si ningun candidato la cruza, EXP-06 no se
ejecuta y se documenta la puerta cerrada.

### Inputs

- Candidatos elegibles: los que hayan pasado Gate 1 standalone (delta >= +0.005, 4/5,
  IC > 0) o el gate de complementariedad (`ensemble_gain >= 0.0015` con IC > 0);
  provenientes de P84-01, P84-02 o P84-04 (P84-03 no es elegible por diseno).
- OOF congelados del candidato y de V03; pliegues canonicos; segunda particion (1.2).

### Process

1. Listar candidatos elegibles y congelar sus OOF con sha256. Si la lista esta vacia,
   escribir `reports/p84_06_puerta_cerrada.md`, anadir la fila P84-06 con resultado
   "NO EJECUTADO" y terminar.
2. Regla primaria R1 (convencion pre-registrada): media simple de probabilidades
   `(p_cand + p_v03)/2` a umbral 0.5. Se reporta F1 a 0.5 y el `ensemble_gain` con IC
   pareado 10.000/20260926.
3. Regla secundaria R2 (permitida, sin meta-modelo): media ponderada
   `w*p_cand + (1-w)*p_v03`, con w elegido por pliegue externo SOLO en particiones
   internas de `outer_train` (rejilla declarada w en {0.1,...,0.5} paso 0.1; empate:
   menor w) y umbrales de V03 congelados. Sin features de meta, sin entropias, sin
   margenes, sin stacking (cerrado).
4. Comparacion contra V03 (control) con el protocolo 1.3: delta por pliegue, IC, rescate/
   dano, tasa predicha, correlacion entre candidatos y solapamiento de errores.
5. Gate: R1 (o R2 si R1 falla y R2 se declaro antes) debe cumplir Gate 1 (+0.005, 4/5,
   IC > 0). Si pasa, va a Gate 2 en la segunda particion con la receta congelada
   (sin reajustar w, umbrales ni mezcla). Freno D-057.
6. Nada de seleccionar mezcla, pesos ni umbrales mirando el leaderboard. Una submission
   solo con autorizacion expresa (Gate 3).

### Boundaries

- Always: congelar la lista de candidatos y sus OOF antes de combinar; mantener V03 como
  brazo control; declarar la regla y la rejilla antes de ejecutar; estimar w/t solo
  dentro de `outer_train`; reportar la matriz de errores.
- Ask first: ejecutar EXP-06 si la lista de elegibles es dudosa; cualquier regla que no
  sea media (R1) o media ponderada con w interno (R2); cualquier decision de submission.
- Never: stacking o meta-modelo aprendido sobre scores OOF; calibracion (cerrada);
  elegir pesos/umbrales con el leaderboard; tocar test/holdout; ejecutar EXP-06 sin
  candidatos que hayan cruzado la puerta; presentar la combinacion como candidata si no
  pasa Gate 2.

### Success Criteria

- Puerta operativa: si no hay elegibles, cierre documentado en una pagina con la lista de
  candidatos y su gate.
- Si se ejecuta: Gate 1 superado, Gate 2 superado con la receta congelada, y decision
  GO/NO-GO documentada sin envio automatico.

### Outputs

- Script (solo si se ejecuta): `experiments/p84_06_combinacion.py`.
- Informes: `reports/p84_06_combinacion.json` / `.md`, o
  `reports/p84_06_puerta_cerrada.md`.
- `experiments.csv`: fila P84-06 (resultado "NO EJECUTADO" con razon, o el resultado
  completo); `RUNLOG.md`; `DECISIONS.md`.

### Open Questions

- La regla R2 con w seleccionado por pliegue es una forma leve de estimacion; si el
  equipo prefiere pureza, R1 sola basta y R2 se marca exploratoria.
- La combinacion de probabilidades de distinta escala (fastText vs LR) puede fallar por
  calibracion; la convencion no se cambia, pero se publican los diagnosticos de rango.

---

## 8. Correcciones aplicadas al plan y a la auditoria (normativas)

1. Ruido del leaderboard (auditoria R1): se sustituyen 0.01219/0.01524 y 19.2 aciertos
   netos por las cifras correctas segun el reparto 70/30: publico ~808 filas, cota
   binomial ~0.0146, banda escalada ~0.0182 (~23 aciertos netos); privado ~346 filas,
   ~0.0223 y ~0.0278 (~35 aciertos netos). Una ganancia local de +0.005 (unos 6.3
   aciertos netos) no es detectable con una sonda publica. Gate 2 pasa a ser el filtro
   real. El reparto 70/30 debe reconfirmarse en sesion autenticada antes de decidir una
   sonda (R6).
2. Cita erronea V4-B06 (auditoria R2): donde el plan citaba una fila V4-B06 ejecutada
   debe citarse V7-01 (focal, cerrado negativo) y `reports/estado_arte_v4.md` /
   `tasks/todo_v4.md` como plan no ejecutado. La justificacion de EXP-03 no depende de
   esa cita.
3. EXP-03 degenerado (fallo cientifico del plan): con modelo lineal convexo y L2,
   duplicar filas es reasignar pesos; "igualar la escala media" convierte ROS r=1 en
   class_weight=balanced exactamente. En sus variantes restantes, ROS cae en las
   familias cerradas de class_weight (V4-C04) y reweighting (V4-B03). Se rediseno a
   verificacion de equivalencia NO PROMOVIBLE. Ningun brazo ROS puede ir a Gate 2 ni a
   EXP-06.
4. EXP-02 reencuadrado: es nuevo como implementacion (no hay `train_supervised` en el
   proyecto) pero funcionalmente es un clasificador lineal sobre bolsa de n-gramas
   hasheados con subwords y bajo rango, cercano a la familia lineal/SGD/hashing cerrada
   y al objeto declarado redundante en V39-01. Se mantiene por su valor de cierre, con
   expectativa nula pre-registrada; el gate standalone se conserva pero la via realista
   es la complementariedad.
5. EXP-04 acotado: se fija dim 200 (configuracion de p71_08, la unica corregida que
   aprende), corpus wikipedia_es 10M congelado en el preregistro y brazo EXTERNAL-ONLY;
   no se reabren rejillas de corpus/dimension/ventana/pooling. Se declara que
   `loss_suma` y `loss_media` no son magnitudes comparables. El ancla verifica semantica
   SGNS publicada (identidad y gradiente) y el ancla ejecutable contra el binario
   oficial; NO exige que la SUMA gane (P81 A6 la refuto en su test plantado).
6. EXP-05 sin fuga de seleccion: LOFO para elegir (m, t); descomposicion EM vs umbral;
   guardarrail de tasa solo a rho=1.00; estimando de F1 ponderada declarado.
7. EXP-01 sin fuga de epoca: 3 particiones internas con el mecanismo verificado
   `partir_interna` (p73_02/p75_01), E* = mediana y reajuste fijo sobre `outer_train`;
   lectura primaria a 0.5 (como pedia el plan) y umbrales anidados solo como lectura
   secundaria; control de estabilidad entre 3 semillas; freno D-057 operacionalizado
   (+0.05 de brecha respecto del control del mismo harness).
8. Bootstrap: semilla 20260926 explicita en todas las llamadas (el default de
   `src/evaluation.py` es 20260925).
9. EXP-06 sin stacking: solo media (R1) o media ponderada con w interno (R2); prohibido
   cualquier meta-modelo o feature de segundo nivel (familia cerrada).
10. Se conserva V03 como unico campeon verificado y la frase de control
    "V03 remains the best verified leaderboard result." en todos los informes.

## 9. Matriz de no-repeticion (resumen)

| Familia cerrada | Evidencia | Tratamiento en P84 |
|---|---|---|
| Perdidas (focal/F1 suave/ASL/PolyLoss/Tukey) | V7-01 y cierres | No se toca; EXP-01 usa BCE con pos_weight (historico, no una perdida nueva) |
| ComplementNB/NB-SVM/BM25/term weighting | cierres v4/v5 | No se toca |
| Cadenas/powerset/stacking | cierres v2/v4 | EXP-06 prohibe meta-modelos |
| Calibracion/umbrales | D-071, V10 | Solo EXP-05 (LOFO, con descomposicion y guardarrail) |
| SGD/hashing/VW/FTRL | cierres v4/v5 | EXP-02 se declara cercano funcionalmente; se ejecuta solo como cierre de implementacion |
| Embeddings no supervisados (rejillas y fusiones) | v5/v5.1, mejor fusion 0.782959 con IC que cruza cero | EXP-04 no reabre rejillas; fusion solo secundaria y exploratoria |
| RNN/CNN desde cero | P82 | No se toca |
| Reweighting/class_weight por etiqueta | V4-B03/V4-C04 | EXP-03 se rediseno a verificacion de equivalencia no promovible |
| DL-01 historico (pool 1731, epoca filtrada) | P80 A17-A19 | EXP-01 es la re-medicion corregida; la cifra 0.794854 no se cita como valida |

## 10. Presupuesto y calendario (referencia: cierre 2026-10-19)

| Ventana | Trabajo | Entregable |
|---|---|---|
| Dias 1-2 | Gate 0 revalidado, enmiendas hasheadas | Gate 0 aprobado |
| Dias 3-5 | EXP-01 | Medicion valida o cierre de DL-01 |
| Dias 6-7 | EXP-02 | Baseline fastText y complementariedad |
| Dia 8 | EXP-03 (micro-verificacion) | Cierre por equivalencia |
| Dias 9-12 | EXP-04 | Objetivo cerrado y downstream medido (estimado < 1 dia; margen para el ancla) |
| Dia 13 | EXP-05 | Cierre EM/umbral |
| Dias 14-15 | Gate 2 para candidatos que pasen | Confirmacion independiente |
| Dias 16-17 | Puerta EXP-06 y decision final | Matriz de cierres y GO/NO-GO |

## 11. Trazabilidad

- Plan maestro: /home/carlos/.opencode/plan/kaggle_toxic_exploracion_modelos_f1_080.md
- Preregistro Gate 0: docs/p84_preregistro.md
- Auditoria de premisas: reports/p84_auditoria_premisas.md
- Lineas historicas citadas: experiments/11_dl_desde_cero.py, reports/p80_auditoria_w2v.md,
  reports/p81_cierre_huecos.md, reports/p79_auditoria_sgns.json, reports/v8_resolucion_liderboard.md,
  reports/v38_01_embedding_supervisado.json, reports/embedding_complementarity_v5_1.json,
  reports/twitter_word2vec_training_lrfix.json, submissions/leaderboard_log.csv, CHAMPION.md.
- Reglas de datos: D-030 (corpus externo solo para representacion), D-057 (freno de
  transferencia), D-063/D-070 (ruido y protocolo), D-009 (metrica), D-014/D-020
  (winner's curse), D-016 (criterios de adopcion).
