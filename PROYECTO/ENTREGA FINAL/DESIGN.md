# DESIGN.md — Entrega Final Juris-PLN

Dos superficies con una misma familia de azul:

- **Producto digital (app Streamlit):** Material Design 3, el sistema de Android 16 y 17.
- **Documentos (informe y deck):** identidad institucional serif heredada del EDA y del informe.

Las razones: la app se usa en pantalla, con interacción y estados; el papel y el deck se leen
en proyección. Comparten el azul como acento y la disciplina de una sola voz visual.

## Identidad

- Herramienta de investigación jurídica para académicos y profesionales del derecho.
- Registro: institucional, documental, sobrio. No es una landing de producto.
- Sin degradados, sin glass, sin sombras generalizadas. La elevación en la app es tonal.
- Logo Javeriana en portada y pie del informe, solo si el archivo existe.

## Producto digital: Material 3

### Color (tema claro, semilla azul institucional)

| Rol | Valor | Uso |
|---|---|---|
| primary / on-primary | `#3B5BA9` / `#FFFFFF` | acción principal, foco, selección |
| primary-container / on-primary-container | `#DBE1FF` / `#00174B` | estados informativos y selección suave |
| secondary-container / on-secondary-container | `#DAE2F9` / `#131C2B` | botones tonales, item activo del cajón |
| tertiary-container / on-tertiary-container | `#BEEAFE` / `#001F29` | avisos |
| error-container / on-error-container | `#FFDAD6` / `#410002` | errores |
| surface / on-surface | `#F9F9FF` / `#1A1B21` | fondo y texto |
| surface-container-low … highest | `#F3F3FA` … `#E2E2E9` | tarjetas, campos, tablas |
| outline / outline-variant | `#74777F` / `#C4C6D0` | bordes y divisores |

Un solo acento: el azul no se reparte por la pantalla. La jerarquía se construye con tamaño,
peso y superficie.

### Tipografía

- Roboto (400, 500, 700), con Noto Sans y la sans del sistema como respaldo.
- Escala usada: display 36/44, headline 24/32, title 22/28 y 16/24, body 16/24 y 14/20,
  label 14/20 y 11/16 en mayúsculas con tracking.
- Datos, rutas y claves: Roboto Mono con cifras tabulares.
- Los decimales y porcentajes van con coma; los miles con punto (`13.444 ms`, `100,0 %`).

### Forma y elevación

- Radios: 4 (campos), 8 (chips), 12 (tarjetas, tablas, avisos), 16 (estado vacío), 28 y full
  (botones y navegación).
- Elevación tonal: las superficies cambian de tono, no se cubren de sombras. Sombra solo en
  hover de botón y en contenedores elevados.

### Componentes

- **Botones:** filled (primario) y tonal (secundario), altura 40, radio full, etiqueta 14/500.
- **Campos:** variante filled, fondo `surface-container-highest`, radio 4 arriba, línea inferior
  de 1 px que pasa a 2 px primary con el foco.
- **Cajón de navegación:** 336 px, ítems de 48 px con indicador activo full en
  `secondary-container`; sin círculos de radio.
- **Tarjetas:** borde 1 px `outline-variant`, radio 12, fondo blanco o `surface-container-low`.
- **Tablas:** contenedor tipo tarjeta, cabecera en mayúsculas sobre `surface-container`, filas
  separadas por `outline-variant`, hover tonal; columnas técnicas en mono con corte de palabra.
- **Avisos:** banners tonales por tipo (info, éxito, aviso, error), radio 12.
- **Selección:** radios y casillas M3 (círculo de 20 px, casilla de 18 px, relleno primary).
- **Estados:** vacío en tarjeta con título y qué hacer; carga con spinner primary; error con
  banner y una acción.

### Accesibilidad y movimiento

- Contraste AA; foco visible de 2 px primary con offset.
- Objetivos táctiles de 48 px en navegación y botones.
- `prefers-reduced-motion`: sin transiciones.
- Estados nunca solo por color: siempre hay texto o forma.

### Retícula y responsive

- Ancho de contenido 1180 px, padding 2,5 rem; 1,5 rem por debajo de 1024; 1 rem por debajo
  de 720.
- En móvil el cajón se colapsa, el título baja a 28 px y las filas de metadatos se apilan.
- Capturas de referencia en `presentacion/capturas/` (P1, P2, P3, chat y móvil).

## Documentos: informe y deck

- Paleta del EDA: azul `#2F75B5`, tinta `#162536`, gris `#425466`, blanco y `#F5F7FA`.
- Títulos en serif (Source Serif 4 o la del template), peso 600; cuerpo serif 11 pt en el
  informe y sans en el deck.
- Un solo acento por figura; título = hallazgo; fuente al pie (n, dataset, fecha).
- Sin emoji, sin 3D, sin degradados decorativos, sin iconos genéricos.
- Diapositivas de 16:9 con franja azul, tarjetas y pie con numeración.

## Dials

- ENERGY 1 (calma, herramienta de trabajo).
- RHYTHM 2 (variación moderada entre pantallas).
- MOTION 1 (hover y transiciones de carga, nada más).

## Reglas transversales

- Cifras reales con fuente visible (n, dataset, fecha). Nada inventado.
- El área temática se rotula "exploratoria" y remite a ADR-19.
- Selector de modelo LLM en la barra lateral, con el modelo por defecto indicado.
- Interruptor de razonamiento en la consulta asistida, con aviso de costo.
- Sin emojis en código, documentos ni interfaz.
