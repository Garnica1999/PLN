# Provenance: autorizacion del modelo recurrente de lenguaje P98

Nota de trazabilidad. **No reproduce ni inventa el contenido de ningun correo.**
Registra el alcance de la autorizacion y donde consta.

## 1. Que autoriza

El entrenamiento, desde cero, de un **modelo de lenguaje recurrente tipo
BiLSTM** (arquitectura contextual, sin transformers ni pesos preentrenados de
terceros) sobre **corpus externos no etiquetados y no pertenecientes a la
competicion**, para aprender una representacion contextual que despues se
evalue en HS/TR/AG contra V03.

## 2. Texto de la confirmacion, verbatim

Proporcionada por Carlos Garnica en la sesion de trabajo del **2026-10-13**:

```text
No hay problema, este biLSTM es entrenado desde cero, no pasa nada
```

Alcance declarado posteriormente por el mismo en la misma fecha (decisiones
operativas P98): modelo recurrente BiLSTM desde cero; objetivo de prediccion
secuencial bidireccional; corpus externo no etiquetado y no de la competicion;
fuentes generales y sociales propuestas con auditoria de licencias,
deduplicacion y solapamiento; ausencia de transformers, pesos preentrenados,
labels externas y pseudo-labeling; exclusion total del texto de `toxic_test` y
de los holdouts; entrenamiento supervisado posterior exclusivamente con el
train oficial.

## 3. Contexto normativo que resuelve

Reglas de la competencia, `docs/competition_rules.md`, seccion "Restriccion
adicional (indicada por el profesor)", verbatim:

```text
No se pueden usar transformers ni modelos LLM para la clasificacion del texto;
no se pueden usar modelos como RoBERTa o BERT. Se pueden usar modelos de
vectorizacion, pero deben entrenarse desde cero y hacer todo desde cero.
```

Lectura aplicada: la prohibicion cubre transformers y LLM con pesos
preentrenados (BERT/RoBERTa, etc.); un BiLSTM entrenado desde cero con pesos
aleatorios y datos no etiquetados de la competicion es un modelo de
representacion desde cero, permitido. Precedentes del proyecto: D-057
(prohibicion de preentrenados ratificada) y D-246/`docs/p73_provenance_autorizacion_corpus.md`
(lectura autorizada: desde cero con datos externos no-competencia).

## 4. Condiciones operativas vigentes

```text
QUE se usa        corpus externo NO ETIQUETADO (Wikipedia CC BY-SA, mC4 ODC-BY,
                  tweets COVID CC BY-SA), solo como texto para el objetivo de
                  lenguaje; atribucion obligatoria en reportes
QUE NO se usa     transformers, BERT, RoBERTa, LLM, sentence-transformers,
                  embeddings/tokenizers preentrenados de terceros, clasificadores
                  externos, labels externas, pseudo-labeling, texto de toxic_test,
                  holdouts, leaderboard para seleccion
SUPERVISION       HS/TR/AG solo con las etiquetas del train oficial
CUARENTENA        corpus de hate speech/odio espanol: sin uso hasta autorizacion
                  escrita especifica (decision del 2026-10-13; un resultado
                  positivo no la libera automaticamente)
```

## 5. Estado del archivo del correo original

No aplica: la confirmacion fue dada en la sesion de trabajo (no por correo);
se cita verbatim arriba, con fecha y autor, y queda ademas en el registro de
decisiones del proyecto (ver DECISIONS.md, P98).

## Referencias

- `docs/competition_rules.md` (texto verbatim del organizador)
- `docs/p73_provenance_autorizacion_corpus.md` (autorizacion 2026-09-30,
  corpus externos no etiquetados para representaciones desde cero)
- `reports/p98_corpus_registry.csv`, `reports/p98_corpus_licenses.md`
- `artifacts/p98_corpus_manifest.json`
