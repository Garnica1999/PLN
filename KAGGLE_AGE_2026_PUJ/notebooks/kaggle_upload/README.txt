Subida del cuaderno consolidado a Kaggle (evita el limite del editor web)

El editor web falla al hacer "Commit" con cuadernos grandes:
  "An error occurred while committing kernel: JSON.parse: unexpected end of data
   at line 1 column 1 of the JSON data"
Ese mensaje significa que el navegador recibio una respuesta vacia del servidor
(peticion demasiado grande o agotada), no que el .ipynb este corrupto.

Procedimiento verificado (CLI autenticada, usuario carlosgarnica):

  cd KAGGLE_AGE_2026_PUJ
  cp notebooks/99_entrega_consolidado.ipynb notebooks/kaggle_upload/
  /home/carlos/venv/bin/kaggle kernels push -p notebooks/kaggle_upload

Resultado: https://www.kaggle.com/code/carlosgarnica/age-entrega-consolidado
El kernel queda privado (is_private: true); se puede hacer publico desde la web.

Alternativas:
- Subir sin salidas por la web (581 KB en vez de 4,8 MB):
    jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to notebook \
      --output 99_entrega_consolidado_sin_salidas.ipynb notebooks/99_entrega_consolidado.ipynb
- Ejecutar en Kaggle: el cuaderno consolidado depende de los artefactos del repo
  y clona https://github.com/Garnica1999/PLN; para una corrida completa en la
  plataforma debe usarse notebooks/kaggle_final_pipeline.ipynb (adjuntando la
  competicion tweet-age-2026-puj).
