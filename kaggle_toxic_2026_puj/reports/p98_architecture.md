# P98 - Arquitectura v1: LM recurrente contextual desde cero (pre-registro)

Fecha: 2026-10-13. Fase A completada (gobernanza + infraestructura). Este
documento fija la arquitectura, la configuracion, la composicion de C98-1 y el
protocolo downstream ANTES de entrenar con corpus reales.

## 1. Objetivo

Aprender una representacion contextual (dependiente de secuencia) desde cero,
sin transformers ni pesos preentrenados, y medir si aporta informacion
complementaria a TF-IDF word+char para HS/TR/AG frente a V03
(corrected_cv 0.7772114525165721; LB 0.78306). No es una fase de FastText: la
novedad es el mecanismo de contexto secuencial.

## 2. Arquitectura (estilo ELMo, sin atencion)

```text
tokens (EMB-NORMAL, reutilizado de v5)
   |
[ canal de palabra ]  embedding 200d SOLO para el vocabulario top-50k
[ canal de caracteres ] char-emb 32d -> conv1d k=3 y k=5, 128 filtros c/u
                         -> max-pool -> 256d
   |
concat(200 + 256) -> Linear(456 -> 512) + ReLU -> dropout 0.2
   |
LSTM 2 capas, hidden 1024 (una direccion por modelo: LTR y RTL separados)
   |
Linear(1024 -> 200) -> softmax ATADO a la matriz de embeddings (bottleneck)
```

- Parametros estimados: emb de palabra 50k x 200 = 10.0M; canal de caracteres
  ~0.1M; proyeccion ~0.23M; LSTM ~14.7M por direccion; dec ~0.2M;
  **total ~40M** (dos direcciones incluidas).
- Razon de la ruta de caracteres: robustez a OOV, elongaciones, errores
  ortograficos y variantes morfologicas (requisito P98 seccion 7). El softmax
  atado con cuello de botella de 200d reduce el coste del vocabulario grande.
- Prohibiciones verificadas por diseno: sin attention, sin transformers, sin
  pesos de terceros; todo inicializado aleatoriamente.

## 3. Tokenizacion y vocabulario

- Flujo de tokens: los corpus ya estan normalizados con la variante
  v5.0-EMB-NORMAL (minusculas, tildes conservadas, marcadores `url`, `usuario`,
  `hashtag`, `numero`); los documentos son lineas; al hacer streaming se
  inserta un separador de documento.
- Vocabulario de palabras: top-50,000 por frecuencia, construido UNICAMENTE
  con el corpus externo autorizado (nunca con datos de la competicion ni del
  test); el archivo de vocabulario se guardara con hash.
- Vocabulario de caracteres: todos los caracteres de los tokens (<=256).
- La misma tokenizacion se reutilizara en la cabeza supervisada y en la
  inferencia (regla P98: una sola transformacion consistente).

## 4. Entrenamiento (pre-registrado)

| Parametro | C98-0 (piloto) | C98-1 (objetivo) |
|---|---|---|
| Corpus | wikipedia 5M (local) | C98-1 30M (seccion 5) |
| Direccion | LTR | LTR + RTL (dos checkpoints) |
| LSTM | 2 capas, hidden 512 | 2 capas, hidden 1024 |
| seq_len | 96 tokens | 96 tokens |
| batch | 8192 tokens | 8192 tokens (ampliable si VRAM lo permite) |
| Optimizador | Adam lr 1e-3, clip 5 | Adam lr 1e-3, clip 5 |
| Epocas | <=3 (medir convergencia) | <=6, lr x0.5 en epocas 4 y 5, early stop por perplexity de validacion |
| Validacion | 2% documentos reservados | 2% documentos por fuente |
| Seed | 20260925 | 20260925 |

**Regla de dimensionado (no es una grid)**: el piloto mide tokens/s reales del
perfil elegido; C98-1 usara la configuracion mas grande de la escalera
pre-declarada que quepa en <=4 GPU-h por direccion para 30M tokens x 4 epocas:

```text
escalera: [hidden 1024, 2 capas] -> [768, 2] -> [1024, 1] -> [512, 2]
```

La eleccion se documenta con la medicion que la justifica. Presupuesto P98:
<=6 GPU-h por etapa; <=24 GPU-h iniciales de fase.

## 5. Corpus C98-1 (pre-registrado, 30M tokens)

| Componente | Tokens | Fuente | Seleccion |
|---|---|---|---|
| Wikipedia | 18.0M | C98-SRC-02 (50M local) | prefijo por documentos completos |
| mC4 | 7.5M | C98-SRC-04 (10M local) | prefijo por documentos completos |
| Twitter COVID-es | 4.5M | C98-SRC-06 (10.3M local) | muestreo uniforme POR DOCUMENTO, seed 20260925 (evita sesgo temporal) |

- Proporciones objetivo 60/25/15. Deduplicacion cruzada exacta por documento
  (sha1 del texto del documento) entre los tres componentes al construir.
- Los tres corpus pasaron la auditoria de solapamiento P98-02 (CUMPLE: 0
  exactas; unica coincidencia de 8-gramas = puntuacion).
- El corpus resultante se versiona en
  `data/external/corpora/p98_c98_1/tokens_30m.txt` con manifiesto y hash.
- Sin descargas nuevas en C98-1.

## 6. Downstream D98-1 (pre-registrado)

- Representacion de un tweet: concatenacion de la 2a capa LTR + 2a capa RTL en
  cada posicion de token; **pooling primario: media sobre posiciones** (2048d).
  Sin busqueda de pooling (regla P98).
- Clasificador: LogisticRegression(class_weight=balanced, C=1.0, liblinear)
  entrenado SOLO en el train de cada pliegue canonico (Grupos = near-duplicate
  UNION pseudo-autor); prediccion sobre el pliegue externo.
- Dos lecturas (ambas reportadas):
  1. embedding-only: corrected_cv de la cabeza sola;
  2. fusion: LR sobre la union del bloque disperso de V03 con el denso (2048)
     (misma forma que la fusion de v5.1), umbrales de V03 congelados.
- Comparaciones: control estatico v5.1 (referencia reutilizada, no
  reentrenada); V03; salvage/damage/neto por slices ya existentes; bootstrap
  por clusters (unidad = grupo).
- Gate de la fase: delta corrected_cv >= +0.003, >=4/5 pliegues positivos, IC
  sin cero, estabilidad. La perplexity NO es criterio de adopcion.
- D98-2 (cabeza secuencial pequena) solo si D98-1 muestra senal; D98-3
  (fine-tuning por pliegue) solo si la evidencia justifica su coste.

## 7. Trazabilidad y compliance

- Preentrenamiento SIN etiquetas de la competicion; supervision solo con el
  train oficial; sin texto de test ni holdouts en ningun corpus; leaderboard
  fuera de toda decision.
- Checkpoints en `models/p98_*` con hash; resultados de preentrenamiento en
  `reports/p98_pretraining_results.csv`; downstream en
  `reports/p98_downstream_results.csv`; complementariedad en
  `reports/p98_complementarity.csv`.
- Auditoria de cumplimiento P98 seccion 18 antes de aceptar cualquier
  resultado; `tools/verificar_caracteres.py`; ASCII; sin commits.
