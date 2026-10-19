# Plan y estado del informe final TOXIC 2026 PUJ

## Objetivo

Producir el informe final de la competencia como artículo científico en
formato IEEE Transactions, en español, con extensión de 12 a 15 páginas, que
documente el problema, los datos, el EDA, la metodología, la experimentación
completa (P108--P133), los análisis estadísticos, las decisiones y su
justificación, y la reproducibilidad del envío campeón.

## Decisiones cerradas

- **Título**: «Clasificación multietiqueta de discurso de odio, carácter
  dirigido y agresividad en tuits en español mediante TF-IDF y ensambles
  lineales».
- **Idioma**: español; sin *abstract* en inglés.
- **Plantilla**: `IEEEtran` en modo *journal* (Transactions), dos columnas,
  10 pt.
- **Autores**: Erika Valeria Rivera Jiménez, Julián Guillermo Carrillo
  Meneses, Carlos Andrés Garnica Salazar y Oscar Andrés Prieto Moreno, con
  los correos de los talleres.
- **Ubicación**: `kaggle_toxic_2026_puj/informe/`.
- **Compilador**: TinyTeX (`~/.TinyTeX/bin/x86_64-linux`), `pdflatex` +
  `bibtex` + `pdflatex` x2.
- **Énfasis científico**: separar el modelo base (validado con CV agrupada,
  bootstrap y McNemar) de la capa de decisión de última milla (adaptada al
  test y auditada con el leaderboard), y discutir el riesgo de sobreajuste.

## Estructura

1. Introducción.
2. Trabajos relacionados.
3. Datos y análisis exploratorio.
4. Metodología (validación, representación, modelos, decisión, estadística).
5. Experimentación y resultados (P108--P133, qué funcionó y qué no).
6. Análisis estadístico y discusión (bootstrap, McNemar, umbrales, error,
   validez externa, límites).
7. Reproducibilidad y despliegue (SHA, idempotencia, protocolo de frontera).
8. Conclusiones y trabajo futuro.
9. Referencias (50 entradas IEEE).

## Figuras y tablas

- 8 figuras: EDA, $\phi$, arquitectura, escalera CCV/LB, descomposición de la
  ganancia, curvas umbral--F1, matrices de confusión y bootstrap/longitud.
- 6 tablas: corpus y prevalencias, decisiones de diseño, serie completa
  P108--P133, inferencia estadística, análisis de error y reproducibilidad.

## Estado

- [x] E0. Recopilación de fuentes y cifras (cuaderno ejecutado, reportes,
  `experiments.csv`, `leaderboard_log.csv`, `DECISIONS.md`).
- [x] E1. Cálculo reproducible de estadísticas
  (`herramientas/calcular_estadisticas.py` -> `datos/estadisticas.json`).
- [x] E2. Generación de figuras a tamaño de columna y 300 dpi
  (`herramientas/generar_figuras.py` -> `figuras/`).
- [x] E3. Redacción de las ocho secciones y la bibliografía.
- [x] E4. Compilación con TinyTeX: 15 páginas, 0 errores, 53 citas
  resueltas, 0 referencias indefinidas, 0 desbordes de caja.
- [x] E5. Verificación de cifras contra los artefactos del proyecto.
- [x] E6. Revisión editorial (scientific-writer) y QA de evidencia
  (ml-qa-engineer); correcciones aplicadas: P124--P126 con la escalera
  histórica (0.787127), chi-cuadrado de deciles 12.20 (p=0.202), error
  estándar bootstrap 0.0117 (escala test 0.0133), 47 envíos de modelos
  (más la sonda de plantilla), celdas de frontera identificadas por id,
  citas reubicadas y lenguaje de sobreafirmación corregido.
- [x] E6b. Ampliación de carácter de informe: marco CRISP-DM con tabla de fases, gobernanza experimental, tabla de candidatos rechazados (P109–P130), tabla de familias previas a P108, análisis de error por rebanadas y frontera de anotación.
- [x] E6c. Auditoría editorial externa aplicada: oración ternaria reformulada, prefijos RAE unificados (subléxica, multiescala), anglicismo features sustituido por rasgos lingüísticos, nota de Wilson en la Tabla I, notación χ²(9), aclaración de tf en la ecuación (1), caveat de optimalidad en la Proposición 1, nota de P123 en la Tabla IV, conexión de la Figura 7 y rutas largas divididas en LaTeX.
- [x] E6d. Segunda auditoría editorial aplicada: corregida la duplicación «El cuaderno el cuaderno», añadido el párrafo de embeddings desde cero (FastText 200d y LM recurrente sobre 29.7M de tokens; OOF 0.652, fusión +0.0046 con IC95 que cruza el cero) y las filas correspondientes en la tabla de familias (embeddings y complejidad no lineal), más el ajuste de espaciado de flotantes en main.tex.
- [x] E6e. Tercera auditoría (maquetación final): aplicado \IEEEtriggeratref{48} para balancear la bibliografía de la última página (columna izquierda [41]-[47], derecha [48]-[54]); verificado que el informe sigue en 15 páginas. No se aplicó el ootnotesize de la bibliografía porque IEEEtran ya la compone en ootnotesize.
- [x] E7. Cierre: PDF final
  `Informe_Final_TOXIC_2026_PUJ.pdf`, 15 páginas, SHA256
  `eb2790036514cbdd5cb4d171cfe2c722291a0ae461c2b312c9f548351f01df0a`.

## Verificación rápida

```bash
cd informe
export PATH=$HOME/.TinyTeX/bin/x86_64-linux:$PATH
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
pdfinfo main.pdf | grep Pages
```

Resultado esperado: 13 páginas, sin errores, sin `??`, sin `Cuadro`.
