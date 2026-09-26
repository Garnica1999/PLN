# P9 - Auditoria de caracteres e idioma

## 1. Puerta oficial del proyecto

`tools/verificar_caracteres.py` (version final, trazada en el commit e63d702 y la
excepcion de `.gitignore` del commit 32ab2f2):

```text
OK: 0 caracteres no permitidos en los ficheros propios del proyecto.
excepciones permitidas como dato: reports/embedding_vocabulary_coverage.md (hebreo y griego)
```

Resultado: PASS. La puerta cubre alfabetos no latinos y emoji/dingbats.

## 2. Escaner independiente (audit/p09_charscan.py)

487 archivos escaneados (extensiones .md .py .csv .json .txt .ipynb; excluye data,
features, models, tools/fastText y los outputs temporales de la propia auditoria).

| categoria | generado | dataset | clasificacion |
|---|---|---|---|
| emoji / dingbats | 0 | 0 | limpio |
| zero-width / selectores | 0 | 0 | limpio |
| alfabetos no latinos | 6 (solo vocabulary_coverage, permitido) | 0 | limpio |
| flechas tipograficas (->) | 171 | 0 | desviacion LOW (estilo de prosa) |
| box drawing | 5 (FINAL_REPORT.md) | 0 | desviacion LOW |
| comillas/dashes/simbolos tipograficos | 274 | 0 | desviacion LOW |
| otros no ASCII (incluye section sign, apices) | 550 | 0 | desviacion LOW |
| acentos latinos | 4148 | 0 | desviacion LOW (espanol, no contaminacion) |

Notas:
- `data/raw/toxic_test.csv` tiene 6 bytes cp1252 no ASCII (documentado en P1); el
  escaner lo omite por encoding y no es un artefacto generado.
- Las desviaciones LOW se concentran en `notebooks/10_final_pipeline.ipynb` (1201),
  `FINAL_REPORT.md` (1072+), `docs/*_spec.md`, `DECISIONS.md` (98) y `experiments.csv`
  (75): son tipografia espanola (acentos, comillas, flechas de flujo, section sign),
  no caracteres decorativos accidentales ni contaminacion de otro idioma.
- La puerta propia del proyecto NO considera esto un error; el encargo de auditoria
  pide ASCII estricto. Se documenta la brecha y NO se aplica una correccion masiva:
  reescribir 30+ documentos invalidaria la validacion EBD-26 del notebook y las
  huellas de artefactos, sin valor cientifico. Correccion propuesta (opcional):
  normalizador ASCII en una fase dedicada, con re-validacion posterior.

## 3. Emoticonos (ASCII)

15 archivos marcados por el escaner; revision de contextos: **0 violaciones reales
en contenido generado**.

- `T_T` en 7 archivos: subcadena de `NOT_TESTED` (vocabulario de la spec).
- `XD`/`xD` en RUNLOG (id de sesion), notebook (base64) y citas de tweets del dataset
  (`Comemela XDDD...` en `error_samples_para_revision.csv` y `error_analysis.md`).
- `u_u` en `gpu_baseline.json` y DL: subcadena de `gpu_use`.
- `data/raw/toxic_train.csv` contiene emoticonos reales (dato fuente; NO se toca).
- `audit/p09_charscan.py` se autonombra en la lista de patrones (esperado).

## 4. Veredicto

- project_generated_emoji = 0; zero_width = 0; decorative_accidental = 0;
  foreign_contamination = 0.
- Desviacion LOW de estilo ASCII estricto: 5148 caracteres de tipografia espanola
  en documentos generados (acentos, comillas, flechas, section sign), sin efecto
  cientifico.
- La politica del dataset se respeta: los tweets fuente no se modifican.
