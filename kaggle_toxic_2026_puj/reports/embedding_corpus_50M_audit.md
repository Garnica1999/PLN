# Expansion del corpus de Wikipedia a 50M tokens (fase v5, P6)

Generado por `experiments/embedding_07_corpus_50M_v5.py` (E-CORP-08).

## 1. RATIONALE_FOR_CORPUS_EXPANSION (seccion 30)

**RATIONALE_FOR_CORPUS_EXPANSION-50M**: 10M -> 50M tokens de la misma fuente.

| Condicion de la §30 | Evidencia |
|---|---|
| 1. Cobertura | A 10M tokens se consumieron 2.847 de los 159.400 articulos disponibles en la parte 1 del dump (1,8 %). Los 50M tokens usan del orden de 12.000 articulos, todos dentro del archivo ya descargado: no hay descarga nueva. |
| 2. Ganancia marginal de vocabulario | El corpus de 10M tiene 271.209 tipos crudos y 100.098 tras min_count=3. Los 50M amplian el mismo rango de articulos del dump, que incluye articulos de dominios distintos (biografias, municipios, musica, deporte) que el corte anterior no alcanzo. |
| 3. Mejora downstream (mecanismo plausible, no demostrado) | En P4-P5 se observo que una mejor cobertura **si** mejora el embedding aislado (0.6530 -> 0.6658). Ese mecanismo justifica medir si mas tokens, con arquitectura fija, mejora tambien el modelo aislado y, con el, la aportacion marginal. |

- **Lo que no se afirma**: No se afirma que 50M tokens mejoren la aportacion complementaria. D-032 predice que no, y este experimento se lanza precisamente para medir esa prediccion en vez de suponerla.
- **Coste**: Construccion ~1 min; entrenamiento 5 epocas x 50M tokens a 641.584 palabras/s = ~6,5 min por modelo. Descarga adicional: 0 bytes.
- **Criterio de parada**: Si la aportacion complementaria no crece (IC95 tocando el 0) y el embedding aislado mejora solo marginalmente, no se escala a 100M: la conclusion de D-032 se confirma en la dimension de tamano y la fase se cierra.

## 2. Construccion (sin descarga adicional)

- **Tokens**: 50.000.000 (frente a 10.002.400 del corpus de 10M)
- **Documentos**: 21.529
- **Articulos leidos**: 21.561
- **Tipos**: 646.957 frente a 271.209
- **Tiempo**: 72.0 s
- **Verificacion de inclusion**: los primeros 10.002.400 tokens del corpus de 50M son **identicos** a los del corpus de 10M: True

El pipeline es el de P1 **importado**, no reimplementado: se redirigio su directorio de
salida y el tamano de shard. Por eso la comparacion 10M vs 50M es exacta y no una
aproximacion entre dos recetas distintas. Los shards de 10M no se sobrescribieron.

## 3. Cobertura 10M vs 50M

| metrica | 10M | 50M | delta |
|---|---|---|---|
| cobertura por tipo | 0.6839 | 0.7703 | +0.0864 |
| cobertura por token | 0.8833 | 0.9238 | +0.0405 |
| word-OOV | 2704 | 1965 | -739 |
| tipos del corpus | 271.209 | 646.957 | +375.748 |

Tipos nuevos con 40M tokens mas: **375.748**. de los cuales **739** aparecen en el vocabulario TOXIC.

## 4. Cobertura por grupo

| grupo | 10M (token) | 50M (token) |
|---|---|---|
| elongaciones | 0.0394 | 0.1417 |
| coloquial | 0.6920 | 0.7223 |
| insultos | 0.7777 | 0.9617 |
| formas_raras | 0.6565 | 0.7460 |

## 5. Interpretacion previa

La cobertura es la condicion previa y no la conclusion. Lo que decide la fase es lo que
mida el experimento siguiente (entrenar con 50M y repetir la fusion):

- si la fusion se queda en ~+0.004, D-032 queda confirmado tambien en la dimension de
  tamano y no se escala a 100M;
- si el embedding **aislado** mejora con claridad, el limite es el techo de V03 y no la
  calidad de la representacion, lo cual es un resultado distinto e igualmente util.

- Figura: `reports/figuras/fig_47_curva_cobertura_v5.png`
