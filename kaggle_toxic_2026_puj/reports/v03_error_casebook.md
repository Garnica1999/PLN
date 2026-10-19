# TOXIC - P90-00: casebook de errores de V03 sobre el desarrollo

Fecha: 2026-10-10. Script: `experiments/p90_00_casebook_errores_v03.py`.
**Analisis descriptivo. No es un experimento nuevo**: no entrena, no ajusta umbrales,
no toca test ni holdout y no genera submissions.

## Resumen ejecutivo

- Desarrollo autorizado: **1483 filas** (OOF canonico, 5 pliegues, umbrales 0.44/0.51/0.46).
- Errores a nivel tweet-etiqueta: **757**; tweets con al menos un error: **441**
  (29.74 por ciento del desarrollo).
- Por etiqueta: HS 316 (173 FP / 143 FN);
  TR 164 (82 FP / 82 FN);
  AG 277 (135 FP / 142 FN).
- Reparto por numero de etiquetas erradas: 223 con una,
  120 con dos y 98 con las tres.
- La taxonomia v4 historica se reproduce exactamente (FP_HS 173, FN_HS 143, FP_TR 82,
  FN_TR 82, FP_AG 135, FN_AG 142, ALL_THREE 98, IMPLICIT 143, QUOTED 16, COLLECTIVE 142,
  DIRECTED 82, AFFILIATIVE 83, NEWS 156).

## 1. Validacion del baseline

```text
OOF            features/oof_v03_final.npz sha256 559265816ce4f631...
corrected_cv   0.7772114525165721 (reconstruido desde proba/pred/umbrales: coincide)
etiquetas      ['HS', 'TR', 'AG'] | umbrales [0.44, 0.51, 0.46]
F1 por etiqueta  HS 0.761689 | TR 0.815730 | AG 0.754215
fuente OOF     features/v2_oof_base_corrected.npz, modelo 'M2_union_lr_c2' (fila 2 de 4; mismos ids, grupos y pliegues corregidos v2)
mapeo          proba = v2_oof_base_corrected.npz['proba'][:, posicion_M2, :] reordenada por su clave 'ids'; pred = proba >= [0.44, 0.51, 0.46] en el orden HS, TR, AG
```

Cada fila procede de un modelo out-of-fold por construccion del artefacto (mismos ids,
grupos y pliegues corregidos v2); `pred` reproduce `proba >= umbrales` celda a celda.

## 2. Resultados por etiqueta

| etiqueta | TP | FP | FN | TN | precision | recall | F1 | prev. real | prev. predicha | errores |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HS | 505 | 173 | 143 | 662 | 0.7448 | 0.7793 | 0.7617 | 0.4370 | 0.4572 | 316 |
| TR | 363 | 82 | 82 | 956 | 0.8157 | 0.8157 | 0.8157 | 0.3001 | 0.3001 | 164 |
| AG | 425 | 135 | 142 | 781 | 0.7589 | 0.7496 | 0.7542 | 0.3823 | 0.3776 | 277 |

## 3. Resultados por fold

Resumen completo en `v03_error_fold_summary.csv` (15 filas fold x etiqueta con TP/FP/FN/TN,
precision, recall, F1, error_count y error_rate).

## 4. Distancia al umbral

Intervalos predefinidos y frecuencias por etiqueta/tipo/fold en
`v03_error_confidence_summary.csv`. Alta confianza historica (delta >= 0.10 / 0.20 / 0.30
sobre la distancia al umbral, definicion de p61_01):

| etiqueta | delta>=0.10 | delta>=0.20 | delta>=0.30 | errores |
|---|---:|---:|---:|---:|
| HS | 189 | 90 | 42 | 316 |
| TR | 111 | 71 | 40 | 164 |
| AG | 166 | 96 | 48 | 277 |

La distancia al umbral NO se interpreta como probabilidad de error ni como medida
calibrada de confianza.

## 5. Errores conjuntos

```text
sin error 1042 | una 223 | dos 120 | tres 98
```

| combinacion (orden HS, TR, AG) | n |
|---|---:|
| HS_FP | 72 |
| HS_FP|TR_FP|AG_FP | 56 |
| HS_FN | 48 |
| HS_FN|AG_FN | 47 |
| AG_FN | 41 |
| HS_FN|TR_FN|AG_FN | 40 |
| AG_FP | 35 |
| HS_FP|AG_FP | 34 |
| TR_FN | 18 |
| TR_FN|AG_FN | 14 |
| HS_FP|TR_FP | 11 |
| TR_FP | 9 |
| HS_FN|TR_FN | 6 |
| TR_FP|AG_FP | 6 |
| HS_FN|TR_FN|AG_FP | 2 |
| TR_FN|AG_FP | 2 |

Relacion documentada TR=1 => HS=1: en el desarrollo hay 0 violaciones en la
etiqueta verdadera y 20 violaciones en la prediccion (TR_pred=1 con HS_pred=0).
Se registra como observacion del dataset, no como regla universal, y no se aplica
ninguna correccion automatica.

## 6. Slices historicas

Definiciones conservadas (patrones del 06, lexico inducido de la Fase 2 sobre lemas,
categorias v4 y grupo coloquial de embedding_01). Resumen completo dentro/fuera por
etiqueta en `v03_error_slice_summary.csv` (114 filas).

## 7. Errores cerca y lejos de la frontera

```text
A (distancia < 0.025)          errores fronterizos
B (0.025 <= distancia < 0.100) errores de distancia intermedia
C (distancia >= 0.100)         errores alejados del umbral
```

| etiqueta | tipo | A | B | C |
|---|---|---:|---:|---:|
| HS | FP | 19 | 42 | 112 |
| HS | FN | 21 | 45 | 77 |
| TR | FP | 7 | 21 | 54 |
| TR | FN | 6 | 19 | 57 |
| AG | FP | 18 | 40 | 77 |
| AG | FN | 13 | 40 | 89 |

## 8. Muestra de control y muestra de revision

- Controles emparejados 1:1 por etiqueta+fold+y_true con distancia comparable:
  **757** en `v03_error_controls.csv`; sin control disponible: 0.
- Muestra de revision determinista y estratificada: **100** tweets con error
  (tope 100) mas sus controles en `v03_error_review_sample.csv`, con todos los motivos de
  seleccion por tweet.

## 9. Patrones cualitativos tentativos

- **OBSERVED_PATTERN** una parte de los FP de HS/TR/AG son insultos con marcadores de broma o afiliacion (casos: 22628 (AG_FP, 'T eres bien perra'); 23076 (HS_FP, 'Cllate todo te queda bien zorra te amo'); 22524 (AG_FP, 'Jeje callate perra'); 20278 (HS_FP|TR_FP|AG_FP, 'jajajajajaj soy imbecil... yo + <3'); contraejemplo: 21892 (HS_FP|TR_FP|AG_FP, 'Callate puta tu eres el traidor.', sin marcadores y con p>=0.93)): el patron de broma/afiliacion (documentado en el 06) cubre 27/173 FP_HS, 26/135 FP_AG y 21/82 FP_TR: es una parte, no la explicacion completa
- **OBSERVED_PATTERN** los FN de AG y HS incluyen hostilidad contra grupos expresada sin lexico de insulto directo (casos: 20331 (AG_FN, 'Os acordais... inmigrantes subsaharianos...'); 20337 (AG_FN, '#Sudacas'); 20856 (AG_FN, 'El problema... los inmigrantes que llegan masivamente...'); 20678 (HS_FN|AG_FN, 'Mierda para aquellos que saltan la valla de Melilla y Ceuta. Cabrones...'); contraejemplo: 24505 (control, 'No mereces ms comunistas tampoco. Son una escoria para el pais.' clasificado correctamente)): consistente con el hallazgo historico del 06 (los FN de AG se concentran en agresion implicita contra grupos); no se asigna causa
- **TENTATIVE_PATTERN** FN de las TRES etiquetas en agresion directa de registro sexual o vulgar que el modelo puntua muy bajo (casos: 20932 (p=0.28/0.32/0.31, 'Gorda chaconda.. tu eres por mi polla la mujer perfecta...'); 22065 (p=0.32/0.28/0.32, 'lo nico que dominas es mi polla en tu cara.'); 20693 (p baja, '...te mereces por puta y caliente...'); 20956 (p=0.31/0.45/0.30, '...vo te mereces unas chuchas'); contraejemplo: 20609 y 21892: insultos del tipo 'callate puta' reciben p>0.9; la deteccion parece depender del fraseo, no solo del lexico): hipotesis a contrastar sobre dependencia de registro/fraseo; no se convierte en feature ni se afirma como causa
- **OBSERVED_PATTERN** el error de TR vive en textos sin destinatario explicito (noticia o mencion a terceros) (casos: 24269 (TR_FN, p=0.023, 'El agressor de una mujer de 25 ha sido detenido...'); 24241 (TR_FN, p=0.031, 'quisiera saber por qu tienen a este antisemita dando clase...'); contraejemplo: controles 20609 y 20347: respuestas directas con @ obtienen TR correcto con p alta): reproduce la lectura historica del 06 sobre TR; los FP de TR (82) incluyen 21 con marcadores de broma
- **OBSERVED_PATTERN** citas y meta-discurso del insulto entre los errores (casos: 21362 (HS_FP|AG_FP, "'Cllate Zorra' el 'cllate perro' de este semestre"); 20707 (AG_FN, texto entre comillas); contraejemplo: la categoria QUOTED v4 tiene solo 16 errores: el meta-discurso existe pero es minoritario): observado, sin cuantificar como causa
- **TENTATIVE_PATTERN** una parte de los errores esta pegada al umbral y no debe interpretarse como anotacion (casos: 23839 (d=0.000046); 23290 (d=0.00078); 23734 (d=0.00103); 20278 (d=0.00119); contraejemplo: 112 errores de HS, 54 de TR y 77 de AG estan a distancia >= 0.10 (grupo C), no fronterizos): separar frontera (A/B) de lejania (C) es imprescindible antes de proponer hipotesis; la distancia no es una probabilidad calibrada
- **UNSUPPORTED_HYPOTHESIS** afirmar que los FP/FN alejados son errores de anotacion, o que el sarcasmo explica los FN (casos: sin evidencia textual suficiente recolectada en este ciclo; contraejemplo: los mismos casos admiten lecturas de representacion y de anotacion; hace falta revision humana y un mecanismo legal no medido): se registra como no apoyada; no se usa para abrir ninguna via

Cada observacion referencia casos del catalogo; ninguna se convierte en feature ni se
usa para entrenar. Las categorias son tentativas y requieren contraste.

## 10. Ejemplos representativos (los mas alejados del umbral)

| tweet_id | label | tipo | prob | umbral | distancia | texto (truncado) |
|---|---|---:|---:|---:|---:|---|
| 24829 | HS | FP | 0.9582 | 0.44 | 0.5182 | @defays_mikael @rickcenturion cllate zorra ac los leprosos no opinan |
| 22628 | AG | FP | 0.9651 | 0.46 | 0.5051 | @kleopatreus T eres bien perra |
| 22524 | AG | FP | 0.9632 | 0.46 | 0.5032 | @DeenisseMb Jeje callate perra |
| 21362 | HS | FP | 0.9389 | 0.44 | 0.4989 | Cllate Zorra' el 'cllate perro' de este semestre |
| 24402 | HS | FP | 0.9350 | 0.44 | 0.4950 | zorra eres tu que no te sientes hombre y no entiendes que toda mujer e |
| 21892 | HS | FP | 0.9335 | 0.44 | 0.4935 | @saancisco Callate puta tu eres el traidor. |
| 21362 | AG | FP | 0.9479 | 0.46 | 0.4879 | Cllate Zorra' el 'cllate perro' de este semestre |
| 24269 | TR | FN | 0.0234 | 0.51 | 0.4866 | El agressor de una mujer de 25 ha sido detenido. La agresin se produjo |
| 24829 | AG | FP | 0.9447 | 0.46 | 0.4847 | @defays_mikael @rickcenturion cllate zorra ac los leprosos no opinan |
| 24241 | TR | FN | 0.0305 | 0.51 | 0.4795 | Estimados @FcpysUNCuyo quisiera saber por qu tienen a este antisemita  |

## 11. Preguntas que este analisis puede y no puede responder

**Puede responder:** que tweets falla V03 y en que etiquetas; con que probabilidad y
distancia al umbral; cuantos errores son fronterizos o alejados; que errores son
simultaneos; como se reparten por fold y por slices historicas; que casos conviene
revisar a mano.

**No puede responder (sin trabajo posterior):** si un error concreto es un error de
anotacion (requiere revision humana); si un patron cualitativo es una causa (requiere
contraste); si alguna de estas observaciones es una oportunidad accionable (requiere
un mecanismo legal no medido y su gate).

## 12. Limitaciones

- El analisis es descriptivo sobre el OOF canonico; no reestima generalizacion.
- El lexico de insulto es in-sample (inducido con etiquetas de train): sus rebanadas
  son diagnosticas, no confirmatorias.
- Las rebanadas v4 son subconjuntos de error por construccion (no particiones del texto).
- La distancia al umbral no es una probabilidad calibrada.
- Los controles son 1:1 por celda; cuando no existe candidato comparable se registra.

> V03 remains the best verified leaderboard result.
