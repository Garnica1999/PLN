# Provenance: autorizacion de uso de corpus externo

Nota de trazabilidad. **No reproduce ni inventa el contenido de ningun correo.** Registra el
alcance de la autorizacion y donde consta.

## 1. Que autoriza

El uso de **datos externos, no pertenecientes a la competencia**, para entrenar el modelo de
vectorizacion/embedding del proyecto, con la condicion explicita de que **no sean los mismos datos
que se usan en la competencia**.

## 2. Texto de la autorizacion, verbatim

Proporcionada por Carlos Garnica en la sesion de trabajo del **2026-09-30**:

```text
Cumple. Se pueden usar datos para entrenar el modelo de vectorizacion/embedding, obviamente no
los mismos datos que se estan usando para la competencia, no tiene sentido.
```

Resuelve la ambiguedad registrada en `DECISIONS.md` (D-246) entre las lecturas (A) y (B) de la
regla "los modelos de vectorizacion deben entrenarse desde cero":

```text
(A) sin artefactos preentrenados, el modelo se entrena desde cero   <- AUTORIZADA
(B) solo con datos de la competencia                                <- NO exigida
```

## 3. Alcance efectivo en este proyecto

```text
QUE se usa        corpus publico de tuits en espanol, solo como TEXTO NO ETIQUETADO
PARA QUE          entrenar Word2Vec DESDE CERO (skip-gram con negative sampling, implementado
                  a mano en PyTorch, sin gensim y sin ningun artefacto preentrenado)
QUE NO se usa     ningun embedding preentrenado de terceros: ni GloVe, ni fastText, ni
                  Word2Vec preentrenado, ni BERT, ni RoBERTa, ni sentence-transformers, ni LLM
QUE NO se usa     ninguna etiqueta externa, ningun pseudo-labeling, ningun dato del test
CONDICION         el corpus no son los datos de la competencia. Verificado:
                  cero solapamiento exacto y normalizado (auditoria P0), y cero solapamiento por
                  near-duplicados con Jaccard >= 0.2, maximo global 0.142857
                  (reports/p73_solapamiento_corpus.json, P73-01)
```

## 4. El corpus concreto

```text
origen      mmaguero/covid19-spanishtweets-earlylateapril2020 (Kaggle)
licencia    CC BY-SA 4.0, con atribucion obligatoria y share-alike si se distribuyen derivados
periodo     29 de marzo a 1 de mayo de 2020, ventana CERRADA
volumen     361.994 documentos, 10.000.029 tokens
ruta        data/external/corpora/twitter_domain/corpus_twitter_tokens.txt
sha256      1ef872c189d0f405bc1a3f82... (completo en reports/p73_solapamiento_corpus.json)
```

Atribucion obligatoria: se cita el origen en los reportes que usan el embedding.

## 5. Estado del archivo del correo original

**No archivado en este repositorio.** El correo original no esta accesible desde los mecanismos
usados en esta sesion, y no se transcribe nada que no se haya recibido literalmente. Lo que consta
es el texto del apartado 2, con su fecha y su autor.

Accion pendiente, si Carlos lo considera necesario: adjuntar el correo original (o una captura)
en `docs/` y enlazarlo desde esta nota. Mientras tanto, la trazabilidad se apoya en el texto del
apartado 2 y en los commits que lo citan.
