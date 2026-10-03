# P1 - Auditoria de legalidad de herramientas linguisticas externas

**Clasificacion del anotador `es_core_news_md`: AMBIGUOUS. Estado: P2 gateado.**

**Actualizacion (D-076, 2026-09-29): RESUELTO. El anotador queda ALLOWED (condicional). Ver la seccion Resolucion al final.**

## Objeto

Antes de usar spaCy, Stanza, UDPipe, Flair u otra herramienta linguistica externa hay que
determinar si esta permitido usar modelos preentrenados de etiquetado POS, NER y sintaxis,
y clasificar cada recurso como ALLOWED / AMBIGUOUS / PROHIBITED. Regla del brief: si existe
ambiguedad normativa, el recurso no puede producir una submission y la razon se documenta.

## Fuentes normativas disponibles (2026-09-29)

| Fuente | Contenido | Fecha |
| --- | --- | --- |
| Pagina Rules de la competicion (captura en `references/age_rules_capture_2026-09-29.md`) | Reglas especificas del organizador: librerias comunes de NLP y ML permitidas (NLTK, Scikit-learn, TensorFlow, PyTorch, Pandas, Numpy); test no para entrenamiento; formato de envio. NO menciona transformers, LLMs ni modelos preentrenados. | 2026-09-29 |
| Indicacion del profesor (confirmada por Carlos) | Transformers y LLMs: prohibidos. Embeddings: prohibidos expresamente, preentrenados o entrenados desde cero. | 2026-09-29 |
| D-001 (interpretacion aprobada por Carlos, riesgo documentado) | Prohibido: transformers/LLMs y todo modelo de vectorizacion/embedding. Permitido explicitamente: 'POS tagger no-transformer de spaCy', TF-IDF, features artesanales, clasificadores clasicos. | 2026-09-22 |
| D-026 y D-043 (autorizaciones y usos reales) | PH10 midio 26 features de POS con es_core_news_md (autorizado); V14 construyo 37 variables morfo-sintacticas y de entidades con el mismo modelo (autorizado, con SHA256 registrado). | commits 3a518ac y 92fff77 |
| V31+ (compliance) | Dejo el 'POS tagger como predictor' en QUARANTINE por ser un modelo externo preentrenado y no ser las reglas explicitas sobre eso. | commit fc77f1c, 2026-09-29 |

## Prohibiciones firmes (no discutidas)

| Familia | Estado | Evidencia |
|---|---|---|
| Transformers (BERT, RoBERTa, ...) | **PROHIBITED** | indicacion del profesor, confirmada por Carlos |
| LLMs | **PROHIBITED** | indicacion del profesor, confirmada por Carlos |
| Embeddings / vectorizacion (preentrenados o entrenados) | **PROHIBITED** | prohibicion expresa; D-001 |
| Corpus o etiquetas externas | **PROHIBITED** | brief, integridad academica |

## El anotador: ficha tecnica verificada hoy

| Campo | Valor |
| --- | --- |
| Modelo | core_news_md 3.8.0 (spaCy 3.8.13) |
| Idioma / licencia | es / GNU GPL 3.0 |
| Componentes | tok2vec, morphologizer, parser, attribute_ruler, lemmatizer, ner |
| SHA256 de meta.json | 889a2fd05d217fa7c1bcffa867af7fc3... |
| Coincide con el registrado en V14 | True |
| Es transformer (config) | False |
| Usa vectores estaticos preentrenados internamente | **True** (include_static_vectors=true) |

## Por que AMBIGUOUS y no ALLOWED ni PROHIBITED

- **No esta PROHIBITED por ningun texto disponible**: no es transformer, no es LLM, no es un
  modelo de vectorizacion usado como representacion. Las tres prohibiciones firmes no lo cubren.
- **Tampoco es ALLOWED sin reservas**: es un modelo externo PRETRENETRADO (entrenado por
  Explosion sobre corpus externos, y su tok2vec consulta vectores estaticos preentrenados
  internamente, `include_static_vectors=true`); el profesor no se ha pronunciado sobre
  anotadores preentrenados como categoria; y la prohibicion expresa de embeddings, leida de
  forma estricta ('en cualquier punto del pipeline'), lo alcanzaria.
- **El proyecto tiene dos posiciones documentadas y opuestas**: D-001 lo permitia por nombre
  y D-026/D-043 lo autorizaron y usaron (con procedencia); V31 lo dejo en cuarentena.
  Esa contradiccion no la resuelve este informe: la resuelve Carlos (o el profesor).

## Interpretaciones en juego

| Lectura | Alcance de la prohibicion de embeddings | Consecuencia para el anotador |
|---|---|---|
| A: sola representacion | Prohibido USAR embeddings como representacion o feature; no alcanza a un anotador cuyas ETIQUETAS se usan | ALLOWED bajo D-001 (etiquetas si, vectores no) |
| B: estricta | Prohibido que un modelo preentrenado con vectores participe en cualquier paso | El anotador queda fuera; la fase solo podria usar anotacion construida por nosotros |

## Consecuencia mientras siga AMBIGUOUS

1. Ningun resultado de esta fase puede alimentar una submission (regla del brief).
2. P2 **no se ejecuta** hasta que la clasificacion se resuelva.
3. El candidato adoptado (V30 + plug-in) es FINAL_REFERENCE y no depende de nada de esto.

## Resolucion pendiente (elevada a Carlos)

- **Opcion A**: extender expresamente a esta fase la lectura de D-001 (etiquetas si, vectores
  no): habilita P2 con procedencia registrada, control de barajado y ablacion sin canal;
  la submission (si algun dia supera todos los gates) seguira exigiendo autorizacion explicita.
- **Opcion B**: mantener la cuarentena de V31: la fase cierra con P0/P1 documentados y sin
  experimentos con anotadores preentrenados.
- **Opcion C**: consultar al profesor y pausar P2 hasta su respuesta.

## Registro de procedencia (requisito del brief, cumplido)

Ficha completa en `reports/linguistic_annotation_registry.csv`: recurso, tipo, version,
idioma, licencia, fecha de alta, SHA256 y clasificacion. El modelo esta instalado en el venv
desde 2026-08-19; el hash de `meta.json` coincide con el registrado en V14.

## Verificaciones negativas de esta fase (hasta hoy)

| Restriccion | Estado |
|---|---|
| uso de test para decidir algo | 0 |
| uso del holdout | 0 |
| tuning con leaderboard | 0 |
| submissions generadas | 0 |
| llamadas de red en la fase | 0 (la consulta a la pagina publica de reglas es lectura de reglamento, no de datos) |
| uso de transformers/LLMs/embeddings | 0 |
| scripts con firmas de secuencias tipadas | 0 |

## Resolucion (D-076, 2026-09-29)

Carlos resolvio la ambiguedad por la **lectura A** y mando anadir un modelo mayor para
comparaciones. Clasificacion final de los anotadores:

| Recurso | Clasificacion final | Condiciones |
|---|---|---|
| es_core_news_md | **ALLOWED (condicional)** | solo etiquetas; procedencia; control de barajado; ablacion sin canal; submission gateada |
| es_core_news_lg | **ALLOWED (condicional)** | identicas; ademas es el modelo de comparacion pedido |

Las seis condiciones de D-076 son obligatorias. La lectura B (estricta, "ningun modelo
preentrenado con vectores en ningun paso") queda documentada arriba como alternativa no
adoptada. La ambiguedad queda resuelta por decision del participante, con el mismo mecanismo
que D-001/D-026/D-043.

**Habilitacion**: P2 (POS estructurado) queda habilitado con el pre-requisito de pre-registrar
su diseno. La anotacion de train con ambos modelos se construye como artefacto derivado
(`features/linguistic/`), determinista y sin etiquetas, con hash en el manifiesto.
