# Corpus web mC4-es y cobertura de registro (fase v5, P4)

Generado por `experiments/embedding_04_corpus_mc4_v5.py` (E-CORP-06).
Este es el experimento que decide si la palanca identificada en D-031 (el **registro**
del corpus, no su tamano) existe de verdad.

## 1. Procedencia

| Campo | Valor |
|---|---|
| Fuente | Common Crawl en espanol via mC4 (`allenai/c4`, config `multilingual/c4-es`) |
| URL | https://huggingface.co/datasets/allenai/c4 |
| Recurso | multilingual/c4-es.tfrecord-00000-of-02048.json.gz (rango 0-41943039) |
| Licencia | ODC-BY 1.0 (declarada en la ficha del dataset; el contenido deriva de Common Crawl) |
| Acceso | publico, sin gating (verificado el 2026-09-25) |
| Tramo | 40 MB (13.9% del shard de 303 MB) |
| sha256 del tramo | `96ed39f4c18e17ade9f7e2f2b07e6b7da3138408ced77d45385873578195053f` |

Se descarga un **tramo** y no el shard completo porque la densidad medida es de 0.42M
tokens por MB y 10M tokens son suficientes para la comparacion a escala 1. El shard
completo (303 MB) queda disponible si la evidencia justificara mas.

## 2. Pipeline y limpieza de web

| Paso | Detalle |
|---|---|
| Lectura | streaming linea a linea del gzipped; el rango parcial termina sin marca de fin de stream y se tolera |
| Documentos leidos | 17.092 |
| Documentos utiles | 16.738 |
| Tokens | 10.000.239 |
| Tipos | 309.131 |
| Sin cuerpo util (filtrados) | 87 |
| Duplicados | 1 |

Limpieza especifica de Common Crawl: se eliminan las lineas que se repiten dentro del
mismo documento (navegacion, listados, cookies), las lineas de menos de 40 caracteres sin
digitos, las lineas con mas de una URL y los documentos que son en su mayoria URL o
digitos. Es una limpieza **estructural**: no usa labels ni informacion semantica externa.

## 3. Cobertura global comparada

| corpus | tipos TOXIC cubiertos | cobertura por tipo | cobertura por token | word-OOV |
|---|---|---|---|---|
| wikipedia_10M | 5851 / 8555 | 0.6839 | 0.8833 | 2704 |
| web_10M | 7063 / 8555 | 0.8256 | 0.9512 | 1492 |
| mixto_20M | 7197 / 8555 | 0.8413 | 0.9553 | 1358 |

El corpus web **aislado** es mejor que el enciclopedico por tipo (0.8256 frente a 0.6839), lo que tiene sentido: el texto de noticias es mas limpio y mas repetitivo que la web. Lo que interesa es la **complementariedad**: la union sube la cobertura por tipo en **+0.1574** y por token en **+0.0720** respecto a solo Wikipedia.

## 4. Cobertura por grupo (los grupos decisivos)

| grupo | wiki 10M (tipo) | web 10M (tipo) | mixto 20M (tipo) | wiki (token) | web (token) | mixto (token) |
|---|---|---|---|---|---|---|
| elongaciones  0.0354 | 0.1770 | 0.1858 | 0.0394 | 0.2205 | 0.2362 |
| coloquial  0.2143 | 0.6071 | 0.6071 | 0.6920 | 0.9414 | 0.9414 |
| insultos  0.2222 | 0.7083 | 0.7222 | 0.7777 | 0.9825 | 0.9838 |
| colectivos  0.9474 | 1.0000 | 1.0000 | 0.9944 | 1.0000 | 1.0000 |
| formas_raras  0.6359 | 0.7878 | 0.8055 | 0.6565 | 0.8061 | 0.8228 |
| menciones  1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hashtags  1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| targeting  0.9091 | 0.9091 | 0.9091 | 0.9849 | 0.9849 | 0.9849 |
| negacion  1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| noticia  1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| urls  n/d | n/d | n/d | n/d | n/d | n/d |

## 5. Contadores de la seccion 31

| Contador | Valor |
|---|---|
| `tokens_added` (web 10M) | 10.000.239 |
| `new_types_added` (web) | 199.189 |
| `new_relevant_types_added` (web, presentes en TOXIC) | 1346 |
| Tipos TOXIC totales | 8555 |
| Tipos TOXIC cubiertos por Wikipedia | 5851 |
| Tipos TOXIC cubiertos por web | 7063 |
| Tipos TOXIC cubiertos por la union | 7197 |

## 6. Decision

El corpus queda **registrado y disponible** para el experimento de la fase siguiente
(entrenar FastText sobre web y sobre la union y volver a medir la fusion). La cobertura
es la condicion previa: si la union sube la cobertura en `coloquial` y `elongaciones`, la
hipotesis de D-031 tiene respaldo; si no la sube, la palanca no es el registro y la fase
se cierra.

## 7. Limitaciones

1. El tramo es el 13,9 % del shard 0 y C4 no reordena dentro del shard: el tema de los
   primeros documentos puede sesgar la muestra.
2. Common Crawl incluye paginas comerciales y de spam; la cobertura mejora pero la
   calidad semantica no esta garantizada.
3. La cobertura del mixto se calcula como **union de vocabularios**, que es la lectura
   correcta para la pregunta de si el tipo esta disponible, pero no reconstruye los
   conteos del mixto.

- Figura: `reports/figuras/fig_45_cobertura_registro_v5.png`
- Tabla completa: `reports/embedding_coverage_registro_v5.csv`
- Corpus mixto de 20.000.045 tokens: `data/external/corpora/mixto/emb_normal/tokens_20M.txt`
