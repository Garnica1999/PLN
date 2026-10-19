# P98 - Licencias y gobernanza de fuentes

Fase A, 2026-10-13. Registro completo en `p98_corpus_registry.csv`; manifiesto
operativo en `artifacts/p98_corpus_manifest.json`. Autorizacion del mecanismo:
`docs/p98_provenance_autorizacion_bilstm.md` (confirmacion 2026-10-13) y
`docs/p73_provenance_autorizacion_corpus.md` (2026-09-30).

## Clasificacion

| Estado | Fuentes |
|---|---|
| ALLOWED | Wikipedia es (CC BY-SA 4.0); twitter_domain COVID-es (CC BY-SA 4.0, con autorizacion y auditoria de solapamiento) |
| ALLOWED_WITH_RESTRICTIONS | mC4 es (ODC-BY 1.0; atribucion; familia Common Crawl unica); mixto_20M (componentes); Gutenberg/Wikisource como candidatas de libros (licencia por obra) |
| NEEDS_EXPLICIT_PERMISSION | OSCAR es (gated); CulturaX (licencia por verificar); HPLT 2.0 (licencia por verificar); TASS/sentimiento-emocion; corpus de odio (HateEval-ES, MisoCorpus, HaterNet, AMI, HaSCoSVa) |
| PROHIBITED | ninguno en esta fase (ninguna fuente fue declarada prohibida por reglas verificadas) |

## Condiciones notables

- **CC BY-SA 4.0 (Wikipedia y tweets)**: atribucion obligatoria en los
  reportes; el share-alike aplica a la redistribucion de derivados textuales,
  no se redistribuyen los textos; los pesos del modelo se entrenan desde cero
  y no incorporan texto redistribuible.
- **ODC-BY 1.0 (mC4)**: atribucion a la ficha del dataset (allenai/c4,
  contenido derivado de Common Crawl).
- **Familia Common Crawl**: mC4, OSCAR y CulturaX se tratan como UNA familia
  de procedencia; fuente primaria = mC4 local; OSCAR/CulturaX quedan
  registradas sin descarga (no se contabilizan como fuentes independientes).
- **Corpus de odio**: CUARENTENA ACTIVA. No se descargan, no se entrenan, no
  se usan sus labels, ni se infieren clases desde sus anotaciones. Solo con
  autorizacion escrita especifica para uso como texto no etiquetado podria
  considerarse una variante (C98-4); un resultado positivo no la libera
  automaticamente.
- **Solapamiento**: la verificacion de solapamiento con la competicion se hace
  SOLO contra el train (permitido); frente al texto de `toxic_test` no se puede
  escanear sin violar las reglas: esa incertidumbre residual se documenta (los
  corpus provienen de fuentes externas, ventana/conjunto disjunto, y ya
  auditados a cero en su fase de origen).

## Deduplicacion

- Dentro de cada corpus: sha1 del texto limpio por documento (manifiesto v5).
- Entre fuentes: verificada la relacion de prefijo wikipedia 5M/10M/50M y la
  composicion del mixto (wiki10+mC4-10). La deduplicacion global cruzada entre
  wikipedia y mC4 no puede demostrarse exhaustivamente en esta fase; se
  documenta como limitacion y se aplicara deduplicacion por documento en la
  construccion de C98-1 (ver manifiesto P98).
