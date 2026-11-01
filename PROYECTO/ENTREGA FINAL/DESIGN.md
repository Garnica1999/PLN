# DESIGN.md — Entrega Final Juris-PLN

Dirección de diseño para la app y las diapositivas. Deriva de la identidad ya usada en el EDA
(paleta del generador de la presentación) y del informe (plantilla institucional).

## Identidad
- Herramienta de investigación jurídica para académicos y profesionales del derecho.
- Registro: institucional, documental, sobrio. No es una landing de producto.
- Paleta (tomada del deck del EDA):
  - Azul institucional #2F75B5 (acento único: acción principal y estado seleccionado).
  - Tinta #162536 (texto principal y superficies oscuras puntuales).
  - Gris texto #425466 (texto secundario).
  - Fondo #FFFFFF y superficie #F5F7FA.
- Sin degradados, sin glass, sin sombras generalizadas. Elevación solo en modales.
- Logo Javeriana en portada y pie del informe, solo si el archivo existe.

## Tipografía
- Títulos: serif de informe (Source Serif 4 o la del template del informe), peso 600.
- Interfaz: sans humanista (Source Sans 3 o la del sistema), 16 px base.
- Cifras y puntajes: monoespaciada tabular, solo para datos.
- Razón: el serif da peso documental; la sans mantiene legibilidad en pantalla.

## Dials
- ENERGY 1 (calma, herramienta de trabajo).
- RHYTHM 2 (variación moderada entre pantallas).
- MOTION 1 (solo estados hover y transiciones de carga).

## Reglas de la interfaz
- Un solo acento; el azul no se reparte por toda la pantalla.
- Jerarquía por tamaño y peso, no por color.
- Estados vacío, carga y error explícitos: qué pasó y qué hacer.
- Cifras reales con fuente visible (n, dataset, fecha). Nada inventado.
- El área temática se rotula "exploratoria" y remite a ADR-19.
- Selector de modelo LLM en la barra lateral, con el modelo por defecto indicado.

## Reglas de las figuras (aplican también a las diapositivas)
- Misma paleta que la diapositiva; un color de acento por figura.
- Título = hallazgo, no el tipo de gráfico.
- Línea de fuente al pie: n, dataset y fecha.
- Sin emoji, sin 3D, sin degradados decorativos, sin iconos genéricos.
- Etiquetas directas; sin leyendas cuando la etiqueta cabe en la serie.
- Contraste AA; tipografía legible en proyección (mínimo 14 pt).
- No usar estilos por defecto de matplotlib sin razón.

## Diapositivas
- 10 a 12 slides, storytelling, con el diseño del deck del EDA (misma retícula y tarjetas).
- Responden: problema real, público, quién lo usa y cómo se beneficia, diferenciación, impacto.
- Métricas clave con fuente (nDCG@10 con gold humano, C3, latencia).
