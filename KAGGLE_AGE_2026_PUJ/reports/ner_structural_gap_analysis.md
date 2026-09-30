# NER estructural - auditoria de huecos (P3-P5) y verificacion de novedad

**Fecha**: 2026-09-29 | **Metodo**: auditoria POR OPERACION sobre el codigo ejecutado (scan de P0: 252 ficheros, 0 firmas de construccion de secuencias; solo 2 scripts usan el anotador y ambos con conteos/ratios), experiments.csv, auditorias de cierre y este repositorio. El nombre de la tecnica NO cuenta como evidencia.

## Clasificacion (detalle en el CSV)

| ID | Familia | Estado | Decision |
|---|---|---|---|
| P3-LEMMA-POS | lemma + POS | **PARTIALLY_COVERED** | no se ejecuta |
| P4-MORFOLOGIA | morfologia | **PARTIALLY_COVERED** | no se ejecuta |
| P5-DEPENDENCIAS | dependency structures | **GENUINELY_NEW** | documentada, no priorizada |
| P6-NER-ESTRUCTURAL | NER estructural (secuencia de tipos) | **GENUINELY_NEW** | linea de este ciclo |

## Cobertura real de la anotacion NER (medida hoy, artefactos congelados)

| Anotador | Docs con >=1 entidad | Tokens en entidades | Spans por doc | Tipos |
|---|---|---|---|---|
| md | 137,455 (81.17%) | 612,937 | 1.749 | PER: 84,352, LOC: 92,681, ORG: 86,279, MISC: 349,625 |
| lg | 140,603 (83.03%) | 680,230 | 1.827 | PER: 95,255, LOC: 90,948, ORG: 93,144, MISC: 400,883 |

- Acuerdo md vs lg en la secuencia de entidades: **28.91%** identica.
- LOS DOS ANOTADORES SOBREDETECTAN: el porcentaje de tokens dentro de entidades (23.66% en md) es muy superior al de un NER limpio; MISC domina el reparto (es un cajon de sastre).

## Composicion de lo que el modelo marca (md)

De 591,064 tokens de entidad alineados con el tokenizador de superficie, **71.9% caen sobre PALABRAS** (entidades lexicas); el resto son marcas de Twitter. Detalle:

| Par tipo->token | Frecuencia |
|---|---|
| MISC->palabra | 244,368 |
| PER->palabra | 68,610 |
| LOC->palabra | 65,218 |
| ORG->palabra | 46,990 |
| MISC->mencion | 43,867 |
| ORG->rt | 24,687 |
| MISC->rt | 23,937 |
| LOC->mencion | 11,700 |
| MISC->url | 10,748 |
| ORG->mencion | 9,089 |
| MISC->hashtag | 8,867 |
| PER->mencion | 7,011 |

Lectura: ~28% de los tokens de entidad caen en menciones/RT/hashtags/URLs/numeros (los canales de mención/hashtag/URL de V31-02/03 estan CLOSED en integracion y deben documentarse como solapamiento declarado), y el 72% restante es contenido lexico de entidades. El canal de tipos no usa identidades, solo la categoria.

## Cortes (diagnostico, no features)

| Corte | Docs | % con >=1 entidad |
|---|---|---|
| con_http | 70,475 | 86.77% |
| con_hashtag | 34,283 | 91.7% |
| con_mencion | 108,186 | 91.13% |
| longitud<=40 | 181 | 2.21% |
| longitud_41-93 | 5 | 80.0% |
| longitud>93 | 169,150 | 81.26% |

## Muestra diagnostica (12 ejemplos, semilla 20260929; NO se corrige nada)

| fila | texto (inicio) | secuencia marcada | nota |
|---|---|---|---|
| 59349 | RT @GegeRpz: La lucha se hace con los medios que se tengan, ya lo dije no me gustan las regionales, pero nadie | <MISC> <MISC> _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ | marca de Twitter marcada como entidad (sobredeteccion tipica) |
| 90106 | RT @yourpimentel: R E A L https://t.co/H41Rw4FKZ9                                                              | _ <MISC> <MISC> <MISC> <MISC> <MISC> <MISC> _ | marca de Twitter marcada como entidad (sobredeteccion tipica) |
| 87405 | RT @thexfiles: Going to our happy place. #TheXFiles https://t.co/u26LM3kZBO                                    | <ORG> <ORG> <ORG> <ORG> <ORG> <ORG> <ORG> <ORG> _ <MISC> <MISC> <MISC> | marca de Twitter marcada como entidad (sobredeteccion tipica) |
| 132894 | RT @soylunaworldd: @alwaysjaralioff  #CarolinaKopelioff #ValentinaZenere #MichaelRonda #AgustinBernasconiTrend | <ORG> _ _ _ _ _ <MISC> <MISC> <MISC> _ <MISC> <MISC> <MISC> <MISC> <MISC> _ | marca de Twitter marcada como entidad (sobredeteccion tipica) |
| 36500 | RT @Miguel__Lorente: @_guerrapedro @tjoventut @RossellMarina @DoloBeltran @MARTAGOMEZMUSIC La magia siempre sa | <ORG> _ _ _ _ <ORG> _ _ _ _ _ _ _ _ _ _ _ _ _ | marca de Twitter marcada como entidad (sobredeteccion tipica) |
| 147120 | Centrales obreras, marcharan hoy en Bogota, apoyando el paro de pilotos de @Avianca .!                         | _ _ _ _ _ _ <LOC> _ _ _ _ _ _ _ <PER> _ _ | marca de Twitter marcada como entidad (sobredeteccion tipica) |
| 61587 | @Radioacktiva_ @GreenDay #RadioacktivaRevolutionRadio Denme una boleta!! Porfi, ayudenme a cumplir el sueno de | _ _ _ _ <ORG> <ORG> _ _ _ _ <PER> _ _ _ _ _ _ _ _ _ _ _ _ _ _ | entidad lexica (revisar si acierta) |
| 91525 | Thank you @bungie, was a great campaign #PS4share https://t.co/cfDGMdsKZK                                      | <MISC> <MISC> <MISC> _ _ _ _ _ _ <PER> _ | marca de Twitter marcada como entidad (sobredeteccion tipica) |
| 46472 | RT @Caferecords: #Repost @lisandromeza   Entrevista en la emisora Fiesta 107.2 FM desde Espana. Tour Europa 20 | <ORG> <ORG> _ _ _ _ _ _ _ _ _ <MISC> <MISC> <MISC> _ <LOC> _ <MISC> <MISC> <MISC> _ | marca de Twitter marcada como entidad (sobredeteccion tipica) |
| 37476 | Han jugado a la ruleta rusa.El engano de Volkswagen se extiende a las pruebas de dioxido de carbono https://t. | _ _ _ _ _ _ _ <MISC> <MISC> <MISC> <MISC> _ _ _ _ _ _ _ _ _ <LOC> <LOC> <LOC> | marca de Twitter marcada como entidad (sobredeteccion tipica) |
| 87777 | RT @PinchesNovios: Tu y Yo de compras deberias de pensarlo.   http://t.co/eKRB8ZXhFq                           | <MISC> <MISC> _ _ _ <MISC> <MISC> <MISC> <MISC> <MISC> <MISC> _ _ _ | marca de Twitter marcada como entidad (sobredeteccion tipica) |
| 111287 | RT si en tu barrio pasa esto  https://t.co/CArd1WiXsG                                                          | <ORG> _ _ _ _ _ _ _ <ORG> | marca de Twitter marcada como entidad (sobredeteccion tipica) |

Errores tipicos observados en la muestra (cualitativos, sin corregir): hashtags y menciones marcadas como MISC/ORG; `RT` marcado como ORG; primeras palabras de oracion marcadas como MISC; URLs como MISC. Entidades lexicas bien detectadas en nombres propios evidentes (personas, ciudades, medios).

## Separacion que el brief exige: cobertura vs utilidad predictiva

- **Cobertura**: alta (81% de tweets con >=1 entidad) y ruidosa (~24% de tokens en entidades).
- **Utilidad predictiva**: NO la mide esta auditoria. La mide el experimento P6 con su control barajado; si el canal real no supera al barajado, la conclusion sera que no aporta, aunque la cobertura sea alta.

## Limites declarados

- La alineacion tipo-token de la composicion usa el tokenizador de superficie del proyecto; una fraccion pequena no alinea y no entra en el conteo.
- Esta auditoria no toca test ni holdout y no entrena nada.
