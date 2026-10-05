# Nota de ontologias para Juris-PLN (E2)

Fecha: 2026-09-08. Base: seccion E2 de SEGUNDA ENTREGA/estado_arte.md (MultiEURLEX,
COLIEE, NER, MTLD, diacronia, PageRank, BETO/E5).

## Candidatas evaluadas

1. EuroVoc / MultiEURLEX (E2-A1, Chalkidis et al. 2021, EMNLP, 65.000 leyes UE en 23
lenguas con etiquetas EUROVOC). Multilingue con espanol, multietiqueta, con deriva
temporal documentada. Dominio: derecho de la UE (reglamentos, directivas), no
constitucional colombiano. Aplicabilidad: parcial. Sirve como semilla de comparacion
para M1 y como evidencia de que se exige particion cronologica, pero sus descriptores
no mapean los titulos de la CP de 1991 ni el amparo de tutela.

2. LKIF (Legal Knowledge Interchange Format, OWL). Ontologia formal de conceptos
juridicos generales (norma, rol, accion, tiempo) del proyecto ESTRELLA, en ingles con
partes en holandes. Aplicabilidad: baja para clasificacion tematica colombiana. Util
solo como patron de modelado (separar norma, caso y rol), no como vocabulario.

3. ELI (European Legislation Identifier) y ECLI (European Case Law Identifier).
Esquemas de identificacion y metadatos, no vocabularios tematicos. Aplicabilidad:
nula como ontologia de contenido; utiles como idea para IDs estables de sentencias.

4. LegalRuleML / LegalDocML (Akoma Ntoso). Lenguajes de marcado de reglas y
estructura documental. Aplicabilidad: baja a corto plazo; darian interoperabilidad
pero exigen anotacion manual costosa fuera del alcance E2.

## Decision documental (no es ADR-8)

No se adopta ninguna ontologia externa tal cual para el espanol juridico colombiano.
Se deriva una taxonomia ligera de la propia Constitucion Politica (titulos y capitulos
como bloques tematicos, articulos 1-380 como etiquetas finas, Fase G) con fallback
"derecho a X" para tutelas. EuroVoc en espanol se conserva solo como referencia de
comparacion para M1. LKIF se cita como patron de modelado futuro.

## Extracto base E2 (primeros caracteres)


```
## E2 — Estado del arte ampliado (enfoque Juris-PLN)

**Fecha de investigación:** 2026-09-08
**Métodos de verificación usados:** webfetch sobre arXiv API y páginas abs de arXiv, ACL Anthology, API de Crossref, portal UCM/Dialnet (intento a Semantic Scholar API devolvió 429 por límite de tasa). Las 10 referencias nuevas fueron comprobadas una por una (DOI o URL funcional) el día de la fecha. Ninguna es inventada y ninguna duplica las 10 de E1.
**Mapa de módulos usado abajo:** M0 validación de sentencia, M1 clasificación de área temática, M2 retrieval de artículos de la CP, M3 NER jurídico, M4 preprocesamiento y análisis morfológico, M5 textometría diacrónica, M6 red de precedentes y jurimetría, M7 embeddings y modelado neuronal. **Errata (2026-10-03):** este mapa es el de las referencias SOTA y difiere del mapa operativo real del proyecto (`00_intent.md`/`PLAN.md`: M0 validación, M1 lematización, M2 NER, M3 etiquetado de área, M4 clasificación, M5 red de citas, M6 diacronía, M7 vínculo CP). Se conserva por trazabilidad de las referencias; el mapa real está en `PROYECTO/arquitectura.md`.

### (a) Legal NLP en español (corpus judiciales hispanohablantes, clasificación de sentencias)

**E2-A1. Chalkidis, I., Fergadiotis, M., & Androutsopoulos, I. (2021).** MultiEURLEX — A multi-lingual and multi-label legal document classification dataset for zero-shot cross-lingual transfer. *Proceedings of EMNLP 2021*, pp. 6974–6996. DOI: <https://doi.org/10.18653/v1/2021.emnlp-main.559> — arXiv:2109.00904. **Verificado** (ACL Anthology + arXiv abs).
- **Aporte:** corpus de 65.000 leyes de la UE en 23 lenguas (incluido el español) con etiquetas multietiqueta EUROVOC; demuestra deriva conceptual temporal (exige particiones cronológicas, no aleatorias) y que el fine-tuning monolingüe destruye el conocimiento multilingüe (olvido catastrófico), mitigable con fine-tuning parcial, adaptadores o BitFit.
- **Implicación para este proyecto (M1, M7):** justifica clasificación multietiqueta por áreas con XLM-RoBERTa/BETO, partición cronológica train/val/test (1992–2021) y adaptadores si el corpus es pequeño.

### (b) Statute/constitution retrieval (COLIEE, EFRA, statute law retrieval, linking casos-legislación)

**E2-B1. Althammer, S., Askari, A., Verberne, S., & Hanbury, A. (2021).** DoSSIER@COLIEE 2021: Leveraging dense retrieval and summarization-based re-ranking for case law retrieval. *COLIEE 2021*. arXiv:2108.03937 — URL: <https://arxiv.org/abs/2108.03937>. **Verificado** (arXiv abs + API).
- **Aporte:** en COLIEE 2021 el retrieval a nivel de pasaje supera al de documento completo; la densa supera a la léxica y la mejor configuración combina BM25 con retrieval denso de pasajes (embeddings de dominio) más re-ranking BERT sobre resúmenes, porque los casos largos exceden la ventana de BERT.
- **Implicación para este proyecto (M2, M7):** justifica arquitectura en dos etapas para vincular sentencias con artículos de la CP: BM25 a nivel de pasaje como primera etapa y re-ranki
```
