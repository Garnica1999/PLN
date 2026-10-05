# Compilación del informe — Entrega 2

**Archivo:** `main.tex` (clase `tg_usta.cls`; bibliografía en `references.bib`; figuras en `../figuras/`).
**Salida:** `main.pdf`.

## Requisitos

- TeX Live con `pdflatex` y `bibtex`. En esta máquina: TinyTeX en
  `/home/carlos/.TinyTeX/bin/x86_64-linux/` (TeX Live 2026).
- Paquetes de la clase y del preámbulo: `babel-spanish`, `harvard`, `anysize`,
  `graphicx`, `subcaption`, `booktabs`, `url`, `amsmath` (los carga `tg_usta.cls`
  o el preámbulo). Si falta alguno: `tlmgr install <paquete>`.
- La figura `logo_javeriana.png` debe permanecer junto a `main.tex` (la clase la
  usa en el encabezado). Las figuras del análisis están en `../figuras/`
  (el preámbulo define `\graphicspath{{../figuras/}{./figuras/}}`).

## Comandos (desde esta carpeta `informe/`)

```bash
export PATH="/home/carlos/.TinyTeX/bin/x86_64-linux:$PATH"

pdflatex -interaction=nonstopmode main.tex   # 1ª pasada (genera .aux)
bibtex main                                   # resuelve references.bib (apalike)
pdflatex -interaction=nonstopmode main.tex   # 2ª pasada (citas)
pdflatex -interaction=nonstopmode main.tex   # 3ª pasada (referencias cruzadas)
```

Al finalizar debe existir `main.pdf` sin errores. Se recomienda revisar el log
(`main.log`) por si quedan advertencias de referencias cruzadas o de figuras.

## Notas

- El estilo bibliográfico es `apalike` (igual que la Entrega 1). El preámbulo
  restaura el `\cite` estándar del kernel de LaTeX porque `tg_usta.cls` carga
  el paquete `harvard`, que lo redefine; no se deben modificar las citas.
- Versión F4b: los marcadores `% TODO(F4b)` del borrador v1 fueron resueltos con las cifras del notebook ejecutado (secciones 3.3, 4.4, 5.3 y 5.4) y las figuras nuevas (`fig_representacion_tfidf.png`, `fig_logodds_periodos.png`, `fig_embeddings_ejemplo.png`) están integradas y referenciadas.
- Para limpiar auxiliares: `rm -f main.aux main.bbl main.blg main.log main.out main.lof main.lot`.
- Compilación rápida de verificación (sin bibliografía actualizada):
  `pdflatex -interaction=nonstopmode main.tex`.
