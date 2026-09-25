# Research Backlog v2 - KAGGLE_TOXIC_2026_PUJ

Fecha: 2026-09-25. Referencia: V03 (LB 0.78306; corrected_cv v2 = 0.7772 sobre
desarrollo, n=1.483). Reglas: `tasks/plan.md` y D-016. Estado por linea:
pendiente / en curso / adoptada / cerrada (sin mejora) / descartada.

Justificacion de orden: primero las lineas que explotan la estructura de
etiquetas y las brechas con evidencia cuantitativa fuerte (phi, FN de AG,
rebanadas debiles); despues las representaciones nuevas; la optimizacion y el
stacking van al final porque dependen de tener canales construidos.

| ID | Linea | Hipotesis | Senal | Implementacion | Validacion | Informacion esperada | Estado |
|---|---|---|---|---|---|---|---|
| E2-01 | Cadena de etiquetas HS->TR->AG | TR mejora si conoce P(HS); AG mejora con P(HS) y P(TR) | phi 0.743 / 0.831 | P(otra etiqueta) OOF como columna extra al vectorizador | corrected_cv + 3/5 pliegues | Alta: la dependencia es la estructura mas fuerte no usada | pendiente |
| E2-02 | Meta-modelo de dependencia (stacking de etiquetas) | Un meta-modelo regularizado sobre scores OOF captura la dependencia mejor que features crudas | idem | LR/Ridge sobre probabilidades, margenes y entropias OOF, anidado | corrected_cv + bootstrap | Media-alta | pendiente |
| E2-03 | POS sparse | POS n-gramas como TF-IDF aportan estructura gramatical que word/char no capturan | H: "no evaluada" | spaCy simbólico -> secuencias POS -> TF-IDF -> LR | corrected_cv (standalone y hstack) | Media | pendiente |
| E2-04 | Dependencias y morfologia sparse | Patrones de dependencia (NEG_VERB, PRON_VERB, TARGET_ADJ) son señal complementaria | H12 | spaCy deps -> tokens simbolicos -> TF-IDF | corrected_cv | Media | pendiente |
| E2-05 | Distancia target-insulto | La distancia/orden entre insulto y target discrimina insulto dirigido vs general vs citado | TR 0.82 y confusiones de targeting | lexico inducido fold-local + targets detectados -> min/media/token/char distancia, orden, densidad; y canal `<INSULT>` abstracto | corrected_cv + efecto en FP/FN | Alta | pendiente |
| E2-06 | Colectivo + agresion (FN de AG) | La agresion contra colectivos sin insulto explica los FN de AG | 45.5 % FN con colectiva vs 13.9 % TP | patrones colectivos estructurales + predicado negativo + generalizacion | corrected_cv de AG + recall de la rebanada colectiva | Alta | pendiente |
| E2-07 | Cita / atribucion / RT (H13) | Separar autor de citado mejora HS/AG en regimen noticia | FN en textos largos con URL; H13 pendiente | canales in-quote/out-quote, marcadores de atribucion, RT puro vs con comentario | corrected_cv + rebanadas noticia/RT | Alta | pendiente |
| E2-08 | Negacion | "no es X" vs "es X" cambia la etiqueta | neg disponible; no modelado | negacion por dependencias + insulto/target bajo negacion | corrected_cv + error analysis | Media | pendiente |
| E2-09 | Emocion y sentimiento local (H5/E04) | Emocion del autor y polaridad local al target complementan word/char | H5 nunca ejecutada; risas/emoticones detectados | lexicones documentados (licencia/cobertura) + ventanas ±3/5/10 al target | corrected_cv + ablacion por bloque | Media | pendiente |
| E2-10 | Multicanal ortografico | Un canal normalizado recupera lexemas ofuscados y el raw conserva estilo | elongaciones; char_wb ya absorbe parte | canales raw / colapsado / sin puntuacion interna / leet / lEXICO enmascarado | corrected_cv de raw+normalizado vs raw | Media | pendiente |
| E2-11 | Perfiles lexicos condicionales (log-odds in-fold) | Scores lexicos por clase aportan señal calibrada que los n-gramas no aislan | log-odds del EDA (in-sample) | inducir P(token|clase) fold-local + smoothing -> scores/margenes | corrected_cv + complementariedad | Media | pendiente |
| E2-12 | Canal distributional (rareza, entropia, concentracion) | La forma de la distribucion lexica distingue regimenes | EDA parcial | features de rareza/entropia/Gini fold-local | late fusion (no concatenacion) | Media-baja (densas ya fallaron) | pendiente |
| E2-13 | Perfilado de entidades (menciones/hashtags/dominios) | P(label|entidad) con smoothing aporta para TR/AG | hashtag F1 0.586; 740 menciones unicas | OOF target encoding con min frecuencia y bucket desconocido; medir cobertura antes | corrected_cv solo si cobertura > 20 % | Media-baja (sparsity) | pendiente |
| E2-14 | Calibracion | Probabilidades calibradas mejoran cadenas, umbrales y stacking | Los decision scores entran en cadenas | Platt/isotonic en OOF anidado | Brier + corrected_cv | Media | pendiente |
| E2-15 | NB-SVM | El peso log-count ratio da un modelo complementario barato | No probado | NB-SVM sobre word/char/union | corrected_cv + ensemble_gain | Media | pendiente |
| E2-16 | Umbrales por rebanada con shrinkage (H15) | Rebanadas con distribucion distinta se benefician de umbral propio | rebanadas 0.586-0.697 | umbral = mezcla(global, local) ponderada por n; solo rebanadas n>=80 | corrected_cv + bootstrap | Media-baja | pendiente |
| E2-17 | Late fusion de canales | Fusionar scores heterogeneos supera a concatenar features | concatenacion densa -0.009 a -0.032 | canales: texto, sintaxis, lexico/emocion, targeting, dependencia | corrected_cv + ensemble_gain con IC | Alta | pendiente |
| E2-18 | Stacking v2 con OOF anidado | El stacking correcto (base OOF, meta OOF) captura complementariedad | v04 fallo por base correlacionada | meta-modelo regularizado + bootstrap por tweet | IC 95 % del delta > 0 | Alta | pendiente |
| E2-19 | Holdout congelado y bootstrap | Control de multiplicidad para 24 lineas | n=1.483 y muchas comparaciones | holdout v2 + bootstrap a nivel de tweet | 5 evaluaciones maximo | Control | hecho (P0) |
| E2-20 | Especialistas por rebanada | Solo si la rebanada difiere significativamente del global | rebanadas debiles identificadas | entrenar especialista y comparar con shrinkage | corrected_cv de la rebanada | Baja (riesgo de sobreajuste) | pendiente |
| E2-21 | Label-specific feature spaces | Cada etiqueta puede necesitar representacion distinta | TR 0.82 vs HS/AG ~0.75 | espacios por etiqueta (texto+targeting para TR; texto+insulto/emocion para AG) | corrected_cv por etiqueta | Media | pendiente |
| E2-22 | Calibracion por rebanada / condicionada a incertidumbre | Umbral condicionado a la entropia del modelo | - | dos umbrales segun confianza | corrected_cv | Baja | pendiente |
| E2-23 | Self-training con pseudo-etiquetas del propio train | No aplica: prohibido usar test; el train ya esta etiquetado | - | - | - | Nula | cerrada por reglas |
| E2-24 | Aumento de datos | Solo si una transformacion conserva la etiqueta (p. ej. colapsar repeticiones) | elongaciones | augmentation determinista dentro del fold | corrected_cv | Baja (riesgo de ruido) | pendiente |

## Criterios de cierre de linea

Una linea se cierra cuando: (a) la hipotesis fue probada con la variante
razonable mas fuerte; (b) no hay mejora en corrected_cv ni complementariedad con
IC positivo; (c) el resultado es estable entre pliegues. Se documenta en
`experiments.csv`, `RUNLOG.md` y `DECISIONS.md`.

## Progreso

- P0 completado: V03 congelada, holdout v2, OOF corregido, gap analysis, backlog,
  D-016.
- Siguiente: P1 (E2-01/E2-02), luego P2 (E2-05/E2-06), P3 (E2-03/E2-04).
