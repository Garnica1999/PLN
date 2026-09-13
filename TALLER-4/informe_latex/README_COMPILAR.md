# Compilación del informe LaTeX (TALLER-4)

Este directorio contiene el informe final en LaTeX del **Taller 4 — Recuperación
de Información aplicada a entrevistas y libros de la Comisión de la Verdad**
(curso *Procesamiento de Lenguaje Natural*, Pontificia Universidad Javeriana).

## Archivos
- `main.tex` — informe completo (10 secciones, 11 tablas, 11 figuras, 6 ecuaciones, citas).
- `references.bib` — 16 referencias verificables (estado del arte + clásicos de RI).
- `figuras/` — figuras usadas por el informe (copia de `../imagenes/`).
- `tg_usta.cls` — clase de la plantilla (Universidad Santo Tomás/Javeriana) usada en los talleres anteriores.
- `logo_javeriana.png` — logo institucional.

## Compilar en Overleaf (recomendado)
1. Comprimir esta carpeta (`informe_latex/`) y subirla a Overleaf (*New Project → Upload Project*).
2. Compilador: **pdfLaTeX** (menú *Menu → Compiler*). La bibliografía se resuelve automáticamente.
3. El PDF resultante incluye portada, abstract, secciones y referencias.

## Compilar en local
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex   # segunda pasada para referencias cruzadas
```
o bien:
```bash
latexmk -pdf main.tex
```

## Estado de compilación (verificado)
- No hay TeX Live instalado y `sudo` requiere contraseña en el entorno del autor.
- Se intentará compilar con **tectonic** (binario de usuario, sin sudo). Si la
  compilación falla por paquetes, Overleaf es la vía garantizada.

## Notas de contenido
- Todas las cifras del informe provienen de los artefactos reales del proyecto
  (`resultados/metricas.json`, rankings, heatmaps y salidas del notebook
  `../notebooks/Taller_4_IR_CEV.ipynb`, 125 celdas, ejecutado sin errores).
- El notebook es independiente: para reproducir todo solo se requieren los 3
  insumos (dataset de entrevistas, libros CEV y tesauro) en `data/raw/`; el resto
  lo genera él mismo (sección "Regeneración").

---
**Compilado y verificado el 2026-09-12**: `main.pdf` = **21 páginas, 2,1 MB**,
0 errores (único warning benigno: `inputenc` ignorado en motores utf8).
Comando usado: `tectonic -X compile main.tex` (tectonic 0.15.0). Copia del PDF:
`Taller_4_IR_CEV.pdf`.
