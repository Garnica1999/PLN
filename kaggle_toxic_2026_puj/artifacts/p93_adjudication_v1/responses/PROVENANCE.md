# Procedencia de las revisiones C1 y C2 (recepcion 2026-10-13)

Dato humano; se conserva SIN CAMBIOS (las notas pueden contener caracteres
no ASCII del texto original de las personas revisoras; no se convierten).

- response_C1.csv / response_C2.csv: copias byte a byte de los archivos
  completados EN EL SITIO por C1 y C2 en la carpeta del paquete
  (independent_review/adjudicator_C1.csv y adjudicator_C2.csv).
- Formato de C1: 7 columnas SIN la columna text (desviacion de esquema;
  anotaciones y notas intactas). Formato de C2: 8 columnas CON text.
- El template adjudicator_C.csv permanece intacto (hash verificado).
- Derivados (no sustituyen a los originales): response_C1_completado.csv y
  response_C2_completado.csv (esquema del paquete, orden del template,
  textos canonicos, anotaciones unidas por adjudication_review_id).
- Sin comparacion con el gold en esta fase (fase posterior y separada).

Hashes (sha256 completo):

- response_C1.csv: f89f74cdabd9649d9894faef7064f9c73413d1d06129733cf829b7e3338c3fe1
- response_C2.csv: e0449a811fa8b64b6e6e1eb2cbbfbcc00bca36685f53c00eaf2de46e67830daf
- response_C1_completado.csv: 7fd2e2a5a1cd8af1e9d5903628ef9489d6343f8b69bff26690e0370179529b70
- response_C2_completado.csv: 22e33186b8efe73d587e7490d4ec1a1afa2d7ca15c542d56bf6302e3276b981b
