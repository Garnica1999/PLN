# P92 - Guia de anotacion provisional v1 (2026-10-09)

Estado: **PROVISIONAL (v1)**. Aplicable a la calibracion y, si no se reemplaza por una
version v2 antes de empezar, a las muestras principales. Version congelada para las
muestras principales: v1 (a menos que la calibracion produzca un changelog y se congele v2
ANTES de iniciar la anotacion principal).

## 1. Definiciones oficiales (verbatim, docs/competition_rules.md)

- **HS**: "indica si hay discurso de odio (1 = si, 0 = no)".
- **TR**: "indica si el discurso esta dirigido contra una persona o grupo (1 = si, 0 = no)".
- **AG**: "indica si el texto es agresivo (1 = si, 0 = no)".

Estas definiciones son la UNICA fuente normativa disponible. No existen criterios
operativos oficiales ni guia de anotadores original. **No inventes criterios** y no
impongas jerarquias entre etiquetas: no asumas que TR=1 exige HS=1 ni que AG=1 exige HS=1
(la relacion observada en los datos no esta confirmada como regla formal).

## 2. Unidad de anotacion

Un **tweet completo**. Cada anotador decide las TRES etiquetas del mismo texto, de forma
independiente entre si. El texto se muestra exactamente como aparece en la fuente; no lo
normalices, no lo corrijas y no elimines menciones, URLs, hashtags ni puntuacion.

## 3. Procedimiento por etiqueta

Para cada etiqueta (HS, TR, AG), por separado:

1. Lee el texto completo.
2. Aplica unicamente la definicion oficial.
3. Si la definicion permite una decision responsable -> **POSITIVE** o **NEGATIVE**.
4. Si la definicion no permite decidir (falta contexto, ambiguedad genuina, referencia
   ininteligible) -> **UNABLE_TO_DETERMINE** y explica el motivo en `uncertainty_notes`.

`UNABLE_TO_DETERMINE` **no es una tercera clase semantica**: indica que no puedes asignar
responsablemente POSITIVE o NEGATIVE con las instrucciones y el contexto disponibles. No
lo conviertas automaticamente en POSITIVE ni en NEGATIVE.

## 4. Textos ambiguos y falta de contexto

- Registra la ambiguedad en `uncertainty_notes` (breve, en tus palabras).
- Muchos textos son respuestas sin el tweet original: si la ausencia de contexto impide
  decidir, usa UNABLE_TO_DETERMINE y anotalo.
- No busques los tweets en internet ni consultes ninguna fuente externa para decidir.
- No consultes las etiquetas originales, predicciones de modelos, umbrales ni ningun
  material del proyecto: tu juicio debe ser independiente.

## 5. Reglas de cegamiento

- No consultes ni intentes deducir la etiqueta original ni por que el caso fue incluido.
- No comentes los casos con el otro anotador antes de terminar tu primera ronda completa.
- La discusion y la adjudicacion son una etapa POSTERIOR, separada y documentada.
- No uses herramientas automaticas (LLM, clasificadores) para asignar tus etiquetas.

## 6. Registro de desacuerdos

Los desacuerdos entre anotadores no se resuelven en esta ronda: se conservan ambas
respuestas y se remiten a la adjudicacion (plantilla separada). Registra tambien, si
quieres, si tu duda es semantica (que significa), normativa (que regla aplica), contextual
(falta informacion) o de aplicacion de la guia.

## 7. Limitaciones normativas pendientes (no resueltas por los anotadores)

1. La definicion de HS no distingue operativamente "odio" de "ofensa".
2. La definicion de AG no distingue "agresivo" de "vulgar".
3. La relacion formal entre HS y TR (y entre HS y AG) no esta documentada; los datos
   muestran 0 excepciones TR=1 con HS=0 y 22 excepciones AG=1 con HS=0 en desarrollo.
4. No hay instrucciones para casos limite (citas, sarcasmo, discurso reproducido).

Estas limitaciones se registran como preguntas abiertas; **no se resuelven inventando
reglas** en la anotacion.

## 8. Formato de respuesta

Columnas `HS_annotation`, `TR_annotation`, `AG_annotation`: una de
`POSITIVE` / `NEGATIVE` / `UNABLE_TO_DETERMINE` por celda. `uncertainty_notes`: texto
libre opcional. No cambies `review_id`, `text` ni `guideline_version`.
