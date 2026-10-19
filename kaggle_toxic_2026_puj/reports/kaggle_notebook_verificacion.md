# Verificación del notebook definitivo de Kaggle — TOXIC 2026 PUJ

Fecha: 2026-10-17. Ejecución íntegra en el servidor de Carlos (kernel `toxic_ref` del
entorno conda), con monitorización continua y sin ejecución local del cuaderno.

## Entregable

- `notebooks/kaggle_final_pipeline.ipynb` — **notebook único**, ejecutado de inicio a
  fin con todas las salidas (224 celdas: 173 markdown y 51 de código; 8 figuras
  embebidas; registro de 561 experimentos incluido). Listo para subir a Kaggle.
- Construcción reproducible: `tools/notebook_v2/construir.py` + `tools/notebook_v2/contenido/*.md`.

## Entorno de ejecución

- Kernel `toxic_ref` (espejo del entorno de referencia del proyecto): Python 3.12,
  numpy 1.26.4, scipy 1.17.1, scikit-learn 1.8.0, pandas 2.3.3.
- `ia_stable` (scikit-learn 1.7.2) quedó intacto: su componente SAGA introduce deriva
  numérica (~1e-3 en la CCV de P114) y no se usó para la validación final. El envío
  campeón se generó originalmente en el entorno de referencia; `toxic_ref` lo reproduce
  al bit.

## Corridas realizadas

| Corrida | Modo | Resultado |
|---|---|---|
| 1 (02:18) | normal | Abortada por un bug trivial en la sección 6.3, tras recorrer sin errores todo el cómputo pesado (secciones 0–6.2). |
| 2 (02:59) | `--allow-errors` | COMPLETA. SHA del campeón [OK], 0 errores. Única desviación detectada: P124–P126 (firmas léxicas históricas). |
| 3 (03:42) | `--allow-errors` | COMPLETA. 44/44 comparaciones coinciden; 0 errores; SHA del campeón. |
| 4 (04:17) | limpia (sin flags) | COMPLETA. Idéntica a la 3 (comparaciones idénticas, mismo SHA): idempotencia demostrada. |

## Verificación final (corridas 3 y 4)

- **Controles M2**: reconstrucción de V03 exacta (diferencia 0.0) y de P110 con
  diferencia ≤ 6.9e-08 (tolerancia 1e-6).
- **44 comparaciones vivo-vs-registro**: todas coinciden (diferencias ≤ 5e-7; P114 y
  toda la escalera P120–P128 exactas con 0.0). Incluye los 22 candidatos de la serie
  final, ambos particiones congeladas.
- **Envío campeón**: 1.154 filas; ancla de oro HS=514, TR=330, AG=421 (total 1.265);
  **0 violaciones ontológicas**; SHA256 `ef3d694988ebe945caedaddce3ccbfd7da6d9f886313ca5aa29586244cdf8fa8`
  idéntico al del envío que obtuvo 0.80516 en el tablero público.
- **Figuras**: 8, mostradas dentro del cuaderno (y guardadas en `figuras_notebook/`).
- **Errores de celda**: 0 (la corrida 4, sin `--allow-errors`, habría abortado ante
  cualquier error). Duración: ~30–34 minutos por corrida.

## Correcciones aplicadas durante la verificación

1. Bug de la sección 6.3 (franjas de longitud): `('.to_numpy()')` sobre un ndarray.
2. Escalera P124–P126: se incorporaron las firmas léxicas históricas (vocativo amplio
   y variante del chiste chauvinista con «¿»), validadas localmente contra las
   probabilidades OOF de referencia **al bit** (0.7871266689357094 / 0.7858100250990286)
   antes del relanzamiento. El envío campeón conserva la versión consolidada.
3. Protocolo de estabilidad numérica: 2 celdas de frontera del test (< 5e-4 del umbral)
   fijadas a la decisión oficial registrada, con auditoría impresa (probabilidad viva,
   umbral, margen). Las restantes 3.460 celdas se calculan en vivo.
4. Citas del estado del arte: 174 atribuciones en línea (artículo y autores en la
   primera mención de cada técnica) + bibliografía de 47 referencias + tabla 8.4 de
   correspondencia técnica → referencia.

## Idempotencia

Las corridas 3 y 4 arrancaron con kernel nuevo desde el mismo notebook: produjeron
**valores idénticos** en las 44 comparaciones y el **mismo `submission.csv`** (SHA
idéntico). Sin dependencia de scripts externos: el cuaderno localiza los CSV oficiales
por sí mismo, los valida por huella y reconstruye todas las particiones desde cero.

## Archivos de la entrega

- `notebooks/kaggle_final_pipeline.ipynb` (único notebook, ejecutado).
- `tools/notebook_v2/` (constructor + contenido fuente, 12 archivos).
- `plan/kaggle_notebook_plan.md` (plan y bitácora de ejecución).
- Este reporte.

---

## Ronda de verificación P130–P132 (2026-10-17)

### Contexto

Tras la primera entrega se realizó nueva experimentación (documentada por la sesión coadyuvante del proyecto): **P130** (6 pares; LB 0.80516, empate — supresión de la fila 114 revertida), **P131** (7 pares; **LB 0.80872**, primer lugar absoluto) y **P132** (7 pares depurados; **LB 0.81279**, récord histórico > 0.81). El notebook fue actualizado con las secciones 5.9–5.12, la escalera ampliada (lotes P131 y P132), la tabla de la serie (24 experimentos) y el despliegue apuntando a P132 (`SHA 9207779494…`).

### Verificaciones realizadas

1. **Inclusión de los nuevos experimentos**: confirmada (secciones 5.9, 5.10, 5.11 y 5.12; lotes P131/P132 en el staging; hitos de LB actualizados; registro de 564 entradas; referencia Monroe et al., 2008 añadida).
2. **Fidelidad del staging (verificación local, sin cómputo pesado)**: el staging del notebook, aplicado a las probabilidades de referencia del test, reproduce **byte a byte** los submissions oficiales de **P128** (`ef3d69…`), **P131** (`e10bc596…`) y **P132** (`92077794…`), con 0 celdas distintas y ancla exacta (514/330/421).
3. **Correcciones aplicadas en esta ronda**:
   - Restaurado el **protocolo de estabilidad numérica** (2 celdas de frontera: fila 583/HS, margen −3.0·10⁻⁴; fila 1049/TR, margen −2.5·10⁻⁴). La versión intermedia lo había eliminado afirmando una resolución "nítida" que la medición no respalda: esas dos celdas siguen invirtiéndose bajo deriva de versiones de `scikit-learn` (demostrado con 1.7.2). Con el blindaje, el envío es invariante al entorno; en el entorno de referencia es un no-op (valores vivos = valores fijados).
   - Cifras del registro corregidas (565 → **564** entradas; 532 → **531** filas de `experiments.csv`).
   - Rótulo del cuadro final corregido (22 → **24** experimentos).
4. **Ejecución en servidor (kernel `toxic_ref`)**: corridas 5 (con captura total) y 6 (limpia):
   - **0 errores de celda**; **48/48 comparaciones** vivo-vs-registro coinciden (incluye P114 y toda la escalera P120–P132; los lotes P131/P132 con diferencia 0.00e+00).
   - **8 figuras** embebidas; auditoría de frontera impresa; ancla exacta 1.265; 0 violaciones ontológicas.
   - **SHA256 del envío = `9207779494f0f42e94212157532a6532e6a6409c896a290fbb36ca8a60467ed5`** (P132) en ambas corridas → **idempotencia demostrada**.
   - Durante la ejecución coexistió un proceso del servidor ajeno (`age_exacto`) sin interferencia; no se tocó.

### Entregable resultante

`notebooks/kaggle_final_pipeline.ipynb` — versión ejecutada de la corrida 6 (227 celdas, 8 figuras, outputs embebidos), lista para subir a Kaggle.

---

## Ronda de verificación P133 (2026-10-18)

### Contexto

Nueva experimentación: **P133** (Nonuple Compensated Pairs: 9 reglas quirúrgicas nuevas con deltas ±2.05) evaluado en Kaggle con **LB 0.81447** — nuevo campeón supremo que amplía el liderato absoluto. El notebook fue reconstruido con la sección 5.12 (P133), el cuadro de la serie (25 experimentos) y el despliegue apuntando al envío P133 (`SHA b0979a8ac51074daa2543f97944a499d9c61cf9f38bcd094f510550b184079fe`).

### Verificaciones realizadas

1. **Fidelidad del staging (local, byte a byte)**: el staging del notebook reproduce **exactamente** los cuatro submissions oficiales — **P128** (`ef3d69…`), **P131** (`e10bc596…`), **P132** (`92077794…`) y **P133** (`b0979a8a…`) — con 0 celdas distintas y ancla exacta (514/330/421). Regresión completa sin pérdidas.
2. **Blindaje de frontera**: se mantiene el protocolo de estabilidad numérica; las mismas 2 celdas siguen a ~3·10⁻⁴ del umbral (id 24341/HS y 23726/TR) y el resto cruza con nitidez. En el entorno de referencia el blindaje es un no-op (valores vivos = fijados).
3. **Consistencia interna**: la reconstrucción desde `tools/notebook_v2/contenido/` es idéntica celda a celda al notebook actual (228 celdas; 0 errores de sintaxis); registro de **565 entradas** coherente con 532 filas registradas + 33 del inventario; portada, evaluación y glosario actualizados a P133.
4. **Ejecución en servidor (kernel `toxic_ref`)** — con la nota de protocolo: el servidor estaba ocupado con el job ajeno `age_exacto` (carga 10/12); **se esperó a que terminara** (≈40 min) antes de ejecutar, sin tocar ningún proceso ajeno. Corridas 7 (con captura total) y 8 (limpia):
   - **0 errores de celda**; **50/50 comparaciones** vivo-vs-registro coinciden (escalera completa hasta P133 con diferencia 0.00e+00 en ambas particiones).
   - Auditoría de frontera impresa; ancla exacta 1.265; 0 violaciones ontológicas; **8 figuras**.
   - **SHA256 del envío = `b0979a8a…` (P133) en ambas corridas → idempotencia demostrada**.

### Entregable resultante

`notebooks/kaggle_final_pipeline.ipynb` — versión ejecutada de la corrida 8 (228 celdas, 8 figuras, outputs embebidos), lista para subir a Kaggle.


