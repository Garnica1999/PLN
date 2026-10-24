# P59-A - Voronoi y geometria local sobre la representacion de V03

Generado por `experiments/p59_01_voronoi_local.py`. Fecha 2026-10-20T02:49:01+00:00.

**Diagnostico. 0 modelos nuevos, 0 submissions, 0 consultas a Kaggle, 0 holdout, 0 test. V03 no se modifica.** No se registro ninguna fila en `experiments.csv`: el parrafo 50 no registra gap analysis, oracle ni visualizaciones, y este paso no entrena clasificador ni evalua `corrected_cv`.

## 1. Auditoria de novedad: que estaba ya hecho

Este track entra con `PARTIALLY_DUPLICATE`, y conviene decirlo antes que cualquier cifra.

| ya evaluado | resultado |
|---|---|
| V29-03 | 48 configuraciones de kNN, ninguna con net positivo |
| V4-F01-F04 | la vecindad explica el error, IC95 [-0.013,+0.049], no significativo |
| V6-07 | la sospecha es un buen marcador de dificultad, no un cleaner |

La pregunta 'la geometria local explica los errores' **ya esta contestada**, y en dos sentidos: cuando se midio directamente el efecto sobre la metrica no fue significativo, y cuando se busco una senal de sospecha se concluyo que sirve para medir dificultad, no para limpiar entrenamiento. Por eso este paso **no vuelve a preguntar eso**.

Lo que si es nuevo, y es lo unico que se mide:

- teselacion de prototipos de clase, celdas, margen y soporte de celda
- la comparacion contra las 42 celdas no rescatables de P54

## 2. La metrica, declarada y no supuesta

> cada bloque TF-IDF viene normalizado a norma 1, luego la norma de la union es sqrt(2) para toda fila y distancia_coseno_union = (cos_palabra + cos_caracter) / 2 exactamente

El parrafo 8 prohibe hacer `TF-IDF -> SVD 2D -> Voronoi`. Aqui no hay ninguna proyeccion: la geometria vive en las 27431 dimensiones originales y la distancia es un coseno explicito.

## 3. Configuracion, fijada antes de ejecutar

```text
prototipos por clase y etiqueta   8
vecinos para la densidad local    15
metodo de medoides                farthest-first (BUILD) mas actualizaciones de Lloyd, sin azar
pesos del indice de inconsistencia {'d_same': 1.0, 'desacuerdo': 1.0, 'soporte_celda': 1.0, 'cuota_opuesta': 1.0}
```

> ningun parametro se ajusto por rejilla; todos se fijaron antes de ejecutar

## 4. Controles de integridad y de fuga

| control | resultado |
|---|---|
| sha256 de V03 | verificado, intacto |
| OOF canonico | reproduce `0.7772114525165721` exacto |
| umbrales de V03 | `pred == (proba >= [0.44, 0.51, 0.46])` exacto |
| grupos canonicos partidos entre pliegues | 0 |
| etiquetas del CSV crudo contra el OOF | identicas celda a celda |
| ids de test en el conjunto de trabajo | 0 |

La geometria de cada fila se construye con los prototipos de los **cuatro pliegues de entrenamiento** y se aplica a la quinta. Los vectorizadores se ajustan tambien solo en esos cuatro. Ninguna estadistica supervisada ve el pliegue de consulta.

## 5. Tests unitarios del mecanismo (parrafo 52)

| test | veredicto | detalle |
|---|---|---|
| distancia coseno invariante a escala | OK | max |d - d_escalada| = 0.000e+00 |
| distancia coseno igual a 1 - cos(angulo) | OK | max |d - (1-cos)| = 2.220e-16 |
| sitio mas proximo coincide con la fuerza bruta (q=0) | OK | bruto=0 vector=0 |
| sitio mas proximo coincide con la fuerza bruta (q=12) | OK | bruto=0 vector=0 |
| sitio mas proximo coincide con la fuerza bruta (q=25) | OK | bruto=33 vector=33 |
| sitio mas proximo coincide con la fuerza bruta (q=39) | OK | bruto=0 vector=0 |
| distancia de un sitio consigo mismo es 0 | OK | max |diag| = 0.000e+00 |
| los medoides son miembros del conjunto y son distintos | OK | n=8 distintos=8 iteraciones=1 conv=True |
| los medoides son deterministas | OK | iguales=True |
| si n <= k se devuelven todos los miembros | OK | n_devueltos=5 |
| la particion en grupos tiene coste minimo dentro de sus medoides | OK | coste_medoides=0.0000 coste_punto_mas_cercano=0.0000 |

## 6. Resultado: el score contra los objetivos

| objetivo | n | AUC | IC95 | media del score en el objetivo | en el resto |
|---|---:|---:|---|---:|---:|
| error de V03 sobre las 4449 celdas | 757 | **0.7252** | [0.7061, 0.7446] | 0.4864 | 0.3804 |
| celda no rescatable dentro de los 757 errores | 42 | **0.4718** | [0.3954, 0.5469] | 0.3839 | 0.3986 |
| celda rescatable dentro de los 757 errores | 715 | **0.7377** | - | 0.4924 | 0.3805 |
| error de V03 en HS | 316 | **0.7358** | - | 0.5015 | 0.3906 |
| no rescatable en HS | 9 | **0.5055** | - | 0.4019 | 0.3985 |
| error de V03 en TR | 164 | **0.6645** | - | 0.4714 | 0.3957 |
| no rescatable en TR | 12 | **0.4886** | - | 0.3918 | 0.3985 |
| error de V03 en AG | 277 | **0.6781** | - | 0.4781 | 0.3932 |
| no rescatable en AG | 21 | **0.4480** | - | 0.3716 | 0.3986 |

Puerta exploratoria del parrafo 15, `0.6`: **superada** en el objetivo primario, y **NO superada** en el objetivo fuerte del parrafo 60.

## 7. Las 42 celdas no rescatables contra las 715 rescatables

Esta es la comparacion que el parrafo 44 y el parrafo 60 exigen, y la que P54 dejo abierta.

| magnitud | valor |
|---|---:|
| celdas no rescatables (C) | 42 |
| celdas rescatables (B) | 715 |
| rango percentil medio del score en C | 0.4721 |
| rango percentil medio del score en B | 0.6995 |
| separacion | -0.2275 |
| fraccion de C en el tercio superior | 0.2857 |
| fraccion de B en el tercio superior | 0.6308 |

## 8. Decision

```text
decision  MARGINAL. La geometria local predice los errores de V03, pero no distingue las celdas no rescatables, que es el objetivo fuerte del parrafo 60, y su IC95 [0.3954, 0.5469] incluye 0.5, luego no hay senal en ninguna direccion. El parrafo 14 solo tendria sentido para reordenar errores ya identificados, no para recuperar los que P54 declaro no recuperables
```

## 9. Formulacion

> V03 remains the best verified leaderboard result.

Lo que el diagnostico SOSTIENE y lo que NO SOSTIENE estan separados en el JSON maestro y en la conclusion de la fase. Ningun resultado de este paso se ha registrado como experimento, porque no lo es.

*Seccion generada por `experiments/p59_01_voronoi_local.py`. Sin push.*
