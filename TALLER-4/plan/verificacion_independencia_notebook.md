# Verificación de independencia total del notebook (v5)

Fecha: 2026-09-05. Alcance: `notebooks/Taller_4_IR_CEV.ipynb` (ensamblado
desde `notebook_cells/`) y su ejecución en cualquier entorno (Colab, local o
un árbol mínimo). Evidencia: `tests/test_notebook_dependencias.py`.

## Matriz de verificación v5

1. **Código 100 % en celdas normales, sin strings/exec** (v4, re-verificado):
   el ensamblador expande `scripts/01-03` como celdas de código normales
   (bloques `[auto]` + alias); el ipynb no contiene `CODIGO_EMBED_*` ni
   `exec(` (checks e1 del test y del ensamblador).
2. **Dependencias = SOLO los 3 insumos** (dataset de entrevistas, libros y
   tesauro; descarga manual del Drive del taller a `data/raw/`). La config
   curada del pipeline (51 muletillas, 62 correcciones, 42 ruido y los
   parámetros TF-IDF) viaja **embebida** como constantes literales en la
   Sec 00 (`STOPWORDS_HABLA_EMBEDIDAS`, `CORRECCIONES_EMBEDIDAS`,
   `RUIDO_EMBEDIDO`, `TFIDF_PARAMS_EMBEDIDOS`); `verificar_consistencia_config()`
   compara con `config/` solo si existe (OK/warning/[info], nunca bloquea).
   Las 30 etiquetas humanas de la evaluación y los agregados de
   normalización también viajan embebidos (Secs 01 y 03).
3. **config/, resultados/ e imagenes/ OPCIONALES o embebidos**: los únicos
   usos permitidos son contextos opcionales — guard `.exists()`, `try:`, o
   las funciones con fallback documentado `verificar_consistencia_config` y
   `mostrar_png` (nota `[aviso]` si la figura no existe). `resultados/` se
   usa solo para verificar paridad cuando existe; el ranking, el EDA y las
   métricas se calculan inline. Política codificada en el check (c) del
   test (lista blanca = solo data/: insumos + caches de `data/corpus`).
4. **Caches derivados autogenerados**: `segmentos_libros.jsonl`,
   `entrevistas_normalizadas.jsonl`, `{libros,entrevistas}_{lemas,tokens}.txt`,
   `*_docs.json` y los CSVs/figuras del EDA se generan en la sección
   "Sec 0b — Regeneración" (`REGEN_AUTOMATICO = True` por defecto) desde
   los 3 insumos, con el código canónico de `scripts/01-03` expandido como
   celdas normales. Si un insumo falta, el driver imprime el mensaje de
   descarga manual SIN excepción; los guards suaves de las secciones 01-03
   (`aviso_archivo_faltante`) muestran `[aviso]` cuando un archivo
   requerido no existe. Diagnóstico de la Sec 00: caches REQUERIDOS
   (autogenerables) vs OPCIONALES (config embebida, resultados/imagenes).
5. **Evidencia del test sandbox** (g): se ejecuta el notebook con nbclient
   (timeout 600 s/celda, cwd = árbol temporal) en un árbol con SOLO el
   .ipynb + symlinks a los caches de `data/corpus` y un `data/raw` VACÍO
   (sin config/, resultados/, imagenes/, scripts/ ni notebook_cells/):
   0 errores; salida con avisos `[aviso]/[info]` y cálculos inline
   completos (muestra determinista de 30 pares, P@1/P@3 embebidos y
   figuras PNG inline).

## Escaneo de independencia
- Imports permitidos: solo paquetes (pandas, sklearn, spacy, wordcloud,
  matplotlib, gdown, pymupdf...). Ninguna celda code importa `scripts/`,
  `experimentos/`, `importlib` (uso real: import_module/spec_from_file) ni
  muta `sys.path` hacia esas carpetas; los docstrings solo las MENCIONAN.
- Referencias de archivo en celdas code: política v5 — lista blanca SIN
  guard = archivos de `data/` (3 insumos: `entrevistas_all_2023-03-21_14-24_05.json`,
  `books-*.zip`/PDFs, `Resource-tesauro.rdf`; caches de `data/corpus`:
  `segmentos_libros.jsonl`, `segmentos/<TOMO>.jsonl`,
  `{libros,entrevistas}_{lemas,tokens}.txt`, `*_docs.json`,
  `entrevistas_normalizadas.jsonl`). Cualquier literal de config/,
  resultados/ o imagenes/ exige contexto opcional en su celda o bloque
  expandido `[auto]` (ver check (c) del test).
- Sin `../imagenes/` en celdas markdown (no renderiza en GitHub/Colab).
- Resolución de raíz: celdas `buscar_raiz()`/`_raiz_standalone()` lanzan
  `FileNotFoundError` claro con el marcador `segmentos_libros.jsonl`; no
  existe la asignación silenciosa del último candidato (nunca cae a `/`).

## Checks (a)-(g) con pytest
`/home/carlos/venv/bin/python -m pytest tests/ -q`
→ suite completa en verde (ver README del proyecto para el conteo).
`/home/carlos/venv/bin/python -m pytest tests/test_notebook_dependencias.py`
→ (a) sin imports de scripts/experimentos/importlib; (b) sin `../imagenes/`;
(c) política v5 de lecturas (lista blanca data/ + contexto opcional para
config/resultados/imagenes, por celda o bloque `[auto]`); (d) réplicas con
error claro y raise; (e) expansión completa: (e1) sin `CODIGO_EMBED_` ni
`exec(`, (e2) celdas code con el código de scripts/01-03, (e3) aliases
`main_extraccion`/`main_preprocesar`/`main_eda`, (e4) `REGEN_AUTOMATICO =
True` (v5), (e5) sin marcadores `[expand:`/`[embed:` pendientes; (f) celda
de diagnóstico de prerrequisitos con semántica v5 (REQUERIDOS/OPCIONALES);
(g) sandbox de independencia: nbclient 0 errores sin config/resultados/
imagenes/scripts y con los avisos y cálculos inline esperados.

## Nota de la expansión (fuente única en build)
`scripts/ensamblar_notebook_t4.py` convierte en el ensamblado la celda
marcador `# %% [expand: scripts/0X_*.py]` (una celda code de la Sec 0b por
script; 3 marcadores resueltos, error si la ruta no existe o queda un
marcador pendiente) en **celdas de código normales**: elimina el bloque
`if __name__ == "__main__":`, antepone en la primera celda un comentario
`[auto]` con `__file__` sintético (para que `parents[1]` resuelva a la raíz
detectada por sec00/sec0b), parte el script por sus banners de sección
(nunca dentro de docstrings, vía tokenize) y cierra el bloque con la celda
de alias `[auto]`. En Run all las celdas expandidas solo DEFINEN funciones y
constantes; la regeneración real la dispara el driver de la sección con
`REGEN_AUTOMATICO=True` solo si faltan caches y los insumos están
presentes. El notebook NUNCA importa ni ejecuta dinámicamente los scripts.
