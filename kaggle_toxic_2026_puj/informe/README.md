# Informe final TOXIC 2026 PUJ

Este directorio contiene el informe final de la competencia, redactado como
artículo científico en formato IEEE Transactions (dos columnas, 10 pt).

## Archivos

- `main.tex`: documento principal (preámbulo, título, resumen, palabras clave
  e inclusión de secciones).
- `secciones/`: una carpeta por sección (introducción, trabajos relacionados,
  datos, metodología, resultados, discusión, reproducibilidad y conclusiones).
- `referencias.bib`: bibliografía en formato BibTeX (50 referencias citadas).
- `figuras/`: figuras en PDF vectorial y PNG a 300 dpi.
- `datos/estadisticas.json`: cifras y pruebas estadísticas calculadas desde
  los artefactos congelados del proyecto.
- `herramientas/calcular_estadisticas.py`: reproduce el JSON anterior.
- `herramientas/generar_figuras.py`: reproduce las figuras.
- `IEEEtran.cls`, `IEEEtran.bst`: clase y estilo IEEE usados.
- `main.pdf` y `Informe_Final_TOXIC_2026_PUJ.pdf`: PDF compilado (13
  páginas; SHA256
  `eb2790036514cbdd5cb4d171cfe2c722291a0ae461c2b312c9f548351f01df0a`).

## Compilación

El compilador es TinyTeX, instalado en el directorio del usuario:

```bash
export PATH=$HOME/.TinyTeX/bin/x86_64-linux:$PATH
cd informe
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

No se requieren paquetes adicionales: el documento usa solo paquetes
presentes en la instalación de TinyTeX.

## Reproducción de cifras y figuras

```bash
cd ..
/home/carlos/venv/bin/python informe/herramientas/calcular_estadisticas.py
/home/carlos/venv/bin/python informe/herramientas/generar_figuras.py
```

El primer script lee `features/oof_v03_final.npz`,
`scratch/p114_oofs.npz`, `features/v2_oof_base_corrected.npz` y
`data/raw/toxic_train.csv`. El segundo lee `informe/datos/estadisticas.json`.
No se usa el test para estimar nada.

## Nota sobre datos

Los archivos `toxic_train.csv`, `toxic_test.csv` y `submission.csv` son de la
competencia y no se redistribuyen aquí. El cuaderno
`notebooks/kaggle_final_pipeline.ipynb` los localiza automáticamente en
`/kaggle/input`, `data/raw` o la ruta indicada por `TOXIC_DATA_DIR`.
