# Verificacion manual de colocaciones (20 casos)

Metodo: BigramCollocationFinder y TrigramCollocationFinder de NLTK con filtro de frecuencia minima 5. Top-20 por frecuencia cruda, PMI y razon de verosimilitud. La tabla juzga si cada n-grama es formula juridica valida.

| n-grama | tipo | metodo(s) que lo rankeo | formula valida? | justificacion |
|---|---|---|---|---|
| corte constitucional | bigrama | freq,llr | si | Combinacion de terminos del dominio (organo, norma o principio). |
| derechos fundamentales | bigrama | freq,llr | si | Nucleo juridico con modificador especifico del dominio. |
| seguridad social | bigrama | freq,llr | si | Nucleo juridico con modificador especifico del dominio. |
| derecho fundamental | bigrama | freq,llr | si | Nucleo juridico con modificador especifico del dominio. |
| constitucion politica | bigrama | freq,llr | si | Combinacion de terminos del dominio (organo, norma o principio). |
| debido proceso | bigrama | freq,llr | si | Combinacion de terminos del dominio (organo, norma o principio). |
| primera instancia | bigrama | freq,llr | no | Combinacion frecuente sin especificidad del dominio juridico. |
| minimo vital | bigrama | freq,llr | no | Combinacion frecuente sin especificidad del dominio juridico. |
| caso concreto | bigrama | freq,llr | no | Combinacion gramatical frecuente sin valor terminologico juridico. |
| debe ser | bigrama | freq | no | Combinacion gramatical frecuente sin valor terminologico juridico. |
| accion de tutela | trigrama | freq,llr | si | Sintagma juridico estable con nucleo normativo o institucional. |
| articulo del decreto | trigrama | freq | si | Sintagma juridico estable con nucleo normativo o institucional. |
| suprema de justicia | trigrama | freq | no | Combinacion frecuente sin especificidad del dominio juridico. |
| proyecto de ley | trigrama | freq | si | Formula juridica con stop intermedia (p. ej. accion_de_tutela). |
| juez de tutela | trigrama | freq,llr | si | Formula juridica con stop intermedia (p. ej. accion_de_tutela). |
| corte constitucional sentencia | trigrama | freq | si | Combinacion de terminos del dominio (organo, norma o principio). |
| articulo del codigo | trigrama | freq | si | Sintagma juridico estable con nucleo normativo o institucional. |
| administracion de justicia | trigrama | freq | no | Combinacion frecuente sin especificidad del dominio juridico. |
| derecho de peticion | trigrama | freq | si | Formula juridica con stop intermedia (p. ej. accion_de_tutela). |
| manuel jose cepeda | trigrama | freq | no | Nombre completo de magistrado o actor: entidad nombrada, no formula tematica. |

## Porcentaje de formulas validas por medida

- bi_freq: 55 %
- bi_pmi: 0 %
- bi_llr: 50 %
- tri_freq: 55 %
- tri_pmi: 0 %
- tri_llr: 100 %
- promedio freq: 55 %, PMI: 0 %, LLR: 75 %

## Conclusion para M2/M4

Se recomienda usar la razon de verosimilitud (likelihood-ratio) como medida principal en M2 (expansion y ranking de pasajes) y M4 (rasgos), porque equilibra frecuencia y asociacion: alcanza 75 % de formulas validas frente a 55 % (frecuencia) y 0 % (PMI). La frecuencia cruda rescata plantillas y nombres propios frecuentes pero arrastra combinaciones genericas; el PMI destaca rarezas poco utiles; la verosimilitud conserva formulas de dominio con soporte suficiente.
