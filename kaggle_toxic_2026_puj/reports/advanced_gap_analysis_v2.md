# Advanced Gap Analysis v2 - KAGGLE_TOXIC_2026_PUJ

Fecha: 2026-09-25. Autor: orquestador ML (revision de artefactos, no de memoria).
Referencia: V03 (LB 0.78306; corrected_cv v2 sobre desarrollo = 0.7772).

## Proposito

Identificar que fenomenos del corpus estan **descritos pero no modelados** (o
modelados solo de forma indirecta) y convertir cada brecha en una hipotesis con
experimento. La pregunta no es "que modelo es mejor" sino "que senal falta".

## Protocolo de medicion v2 (recordatorio)

- Grupos = near-duplicado UNION pseudo-autor; holdout congelado (248 tweets);
  desarrollo = 1.483 tweets.
- `corrected_cv` (primaria) sobre el desarrollo; `historical_cv` como
  continuidad.
- Referencias sobre el desarrollo: V03 con umbrales = 0.7772; M2 a 0.5 = 0.7718;
  M3 (SVC) = 0.7677; M1 (word LR) = 0.7486; M4 (char LR) = 0.7377.

## Matriz de brechas

| # | Fenomeno | Detectado en EDA | Representado en el modelo | Representacion actual | Feature explicita | Testado | Resultado | Brecha restante | Hipotesis nueva |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Targeting / mencion | Si (TR: 64.9 % TR=1 vs 9.7 % TR=0) | Si, indirecto | n-gramas word/char | No | Parcial (densas) | Densas concatenadas -0.009 a -0.032 | Distancias, orden, tipo de target | E2-05/E2-06 |
| 2 | Insulto / profanity | Si (V=0.355) | Indirecto | n-gramas; lexico inducido solo en EDA | No en modelos | Parcial | Lexicos inducidos in-sample no entraron a modelos | Canal abstracto `<INSULT>`, scores in-fold | E2-05 |
| 3 | Atribucion / cita (H13) | Si (patron de FN con URL/atribucion) | **No** | - | No | No | - | Autor vs citado; insulto dentro/fuera de comillas | E2-07 |
| 4 | Negacion | Si (existe `neg` en spaCy) | **No** | - | No | No | - | Negacion sintactica + insulto/target | E2-08 |
| 5 | Referencia colectiva | Si (TR: 57.0 % TR=0 vs 10.0 %; AG: 45.5 % de FN) | Densas probadas | n-gramas | No | Parcial | Concatenacion degrada | Canal sparse y late fusion | E2-06 |
| 6 | Referencia a persona | Si (heuristica PROPN) | **No** | - | No | No | - | Persona vs grupo vs organizacion | E2-06 |
| 7 | Interaccion social (RT) | Si (rt, mencion) | Indirecto | n-gramas | No | No | - | RT puro vs RT con comentario; autor vs citado | E2-07 |
| 8 | Lenguaje emocional (H5) | Si (risas, emoticones) | **No** | - | No | **No ejecutado** | - | Lexicones de emocion documentados + sentimiento local al target | E2-09 |
| 9 | Estructura de oracion / sintaxis | Si (disponible con spaCy) | **No** | - | No | No | - | POS/dependencias como representacion SPARSE | E2-03 |
| 10 | Morfologia | Si (genero/numero/tiempo en spaCy) | **No** | - | No | No | - | Morfologia sparse (numero/genero/modo) | E2-04 |
| 11 | Puntuacion | Si (densidades) | Indirecto | char n-gramas | No | Si (PRE-07) | Neutro (0.0000) | Interaccion puntuacion x insulto | E2-05 |
| 12 | Ortografia / evasion | Si (elongaciones, 21 textos con `;`) | Indirecto | char_wb(3,5) | No | No | - | Multicanal raw + normalizado + patrones | E2-10 |
| 13 | Variacion lexica | Si (TTR, hapax) | Indirecto | word n-gramas | No | No | - | Rareza lexica OOF | E2-11 |
| 14 | Contexto de URL / noticia | Si (TR=0 concentra URLs; FN en textos largos con URL) | Indirecto | word n-gramas | No | Si (PRE-03) | +0.0035 (< umbral) | Regimen noticia: cita + atribucion + tercera persona | E2-07 |
| 15 | Contexto de hashtag | Si (rebanada debil F1 0.586) | Indirecto | word n-gramas | No | No | - | Perfiles P(label|hashtag) OOF | E2-13 |
| 16 | Dependencia entre etiquetas | Si (phi 0.743/0.831/0.759; TR=1 => HS=1) | Post-proceso duro | - | No | Si (duro) | -0.0056 | Cadenas con P(otra etiqueta) OOF continua | E2-01/E2-02 |
| 17 | Perfiles lexicos condicionales por clase | Si (log-odds del EDA) | Solo EDA (in-sample) | - | No | No | - | Scores lexicos in-fold por etiqueta | E2-11 |
| 18 | Perfilado de entidades (menciones/hashtags/dominios) | Si | **No** | - | No | No | - | P(label|entidad) con smoothing y min frecuencia | E2-13 |
| 19 | Calibracion de probabilidades | No evaluado | No | - | No | No | - | Platt/isotonic + efecto en cadenas y umbrales | E2-14 |
| 20 | Umbrales por rebanada (H15) | Si (rebanadas con F1 bajo) | No | - | No | **No ejecutado** | - | Threshold condicionado con shrinkage | E2-16 |
| 21 | Distribucion / entropia / concentracion | Si (parcial) | No | - | No | No (densas si) | Densas degradan al concatenar | Late fusion con canal distributional | E2-12 |
| 22 | NB-SVM (log-count ratio) | No probado | No | - | No | No | - | Canal NB-SVM como modelo complementario | E2-15 |
| 23 | Fusión tardia (late fusion) | Concepto del plan v2 | No (solo concatenacion) | - | No | Si (concatenacion) | Degrada | Meta-modelo sobre scores de canales | E2-17/E2-18 |
| 24 | Multiplicidad de comparaciones | 24 lineas nuevas | No | - | No | No | - | Holdout congelado + bootstrap por tweet | E2-19 |

## Las cinco brechas de mayor valor esperado

1. **Dependencia entre etiquetas (#16)**: la señal mas fuerte del corpus
   (phi 0.83 entre HS y AG) se explota de la peor forma posible (post-proceso
   duro, -0.0056). Las cadenas con scores continuos OOF nunca se probaron.
2. **Targeting estructural (#1, #5, #6)**: TR es la etiqueta con mejor F1
   (0.816-0.820), pero las confusiones de targeting y los FN de AG con
   referencia colectiva (45.5 % vs 13.9 %) apuntan a que la relacion
   target-insulto (distancia, orden, tipo) no esta representada.
3. **Autor vs citado (#3, #7, #14)**: el corpus contiene noticias y citas; el
   insulto dentro de una cita no es del autor. Nunca se modeló.
4. **Sintaxis como representacion sparse (#9, #10)**: lo que fallo fue
   concatenar 46 features densas, no la sintaxis. POS/dependencias como
   n-gramas simbolicos es una representacion nueva.
5. **Late fusion (#23)**: la concatenacion densa fallo; la fusion tardia de
   canales heterogeneos (texto, sintaxis, lexico, targeting, dependencia) es la
   alternativa correcta y no fue probada.

## Nota metodologica sobre el corrected_cv en el desarrollo

En el desarrollo, el protocolo historico (solo near-duplicado) da 0.7736 con
umbrales frente a 0.7772 del corregido: la diferencia (0.0036) esta **dentro del
ruido** (std entre pliegues 0.016; el corpus de desarrollo es mas pequeno). La
evidencia de D-015 proviene del train completo (0.7927 vs 0.7838). El protocolo
corregido se adopta por ser el conservador y por estar alineado con el LB.
