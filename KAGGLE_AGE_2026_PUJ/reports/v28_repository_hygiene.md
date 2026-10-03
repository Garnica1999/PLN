# V28 - Higiene del repositorio

Al cierre de V27 quedaron dos directorios sin seguimiento que no pertenecen a AGE. Este informe determina su origen, su contenido y su relacion con el proyecto. **No se borra nada, no se commitea nada y no se modifica nada.**

## Resumen

| Directorio | Ficheros | Tamano | Pertenece a AGE | Versionado | Decision |
|---|---|---|---|---|---|
| `home` | 1 | 0.0 MB | **no** | no | queda fuera de la rama |
| `kaggle_toxic_2026_puj/tools` | 702 | 17.0 MB | **no** | no | queda fuera de la rama |

**Ninguno pertenece a AGE: True.** Los dos quedan fuera de la rama `proyecto-kaggle-age-2026-v28`, y no se modifican.

## Que es cada uno

### `home`

Es un **artefacto de una ruta relativa mal resuelta**. Contiene un solo fichero, un script de la fase V44 del proyecto `kaggle_toxic_2026_puj` (clasificacion de toxicidad), que deberia haber ido a `experiments/` y acabo en `home/carlos/Git/PLN-v25/kaggle_toxic_2026_puj/experiments/`. El nombre `PLN-v25` indica ademas que procede de un worktree distinto. La causa es un script que uso una ruta relativa `home/carlos/...` en vez de una absoluta o de una relativa al proyecto. **No pertenece a AGE, no tiene resultados de AGE y no se toca.**

| Comprobacion | Valor |
|---|---|
| Ficheros | 1 |
| Tamano | 0.0 MB |
| Extensions dominantes | .py (1) |
| Scripts de Python | 1 |
| Tiene `.git` anidado | False |
| Rutas que mencionan AGE | False |
| Versionado por git | False |

### `kaggle_toxic_2026_puj/tools`

Es una **dependencia externa del proyecto `kaggle_toxic_2026_puj`**, no de AGE. Contiene `fastText`, una biblioteca C++ de texto (32 ficheros en la raiz, 701 en total con `docs/`, `python/`, `alignment/`) y `verificar_caracteres.py`, un verificador de caracteres no latinos. El proyecto `kaggle_toxic_2026_puj` si esta versionado, con 204 ficheros, pero **`tools/` no lo esta**: es una dependencia que se trae a mano y nunca se commiteo, probablemente por el tamano.

| Comprobacion | Valor |
|---|---|
| Ficheros | 702 |
| Tamano | 17.0 MB |
| Extensions dominantes | .html (198), .js (145), .d (45), .png (44), .o (43) |
| Scripts de Python | 25 |
| Tiene `.git` anidado | True |
| Rutas que mencionan AGE | False |
| Versionado por git | False |

## Datos sensibles

Se paso un escaneo de credenciales (`api_key`, `secret`, `password`, claves privadas, tokens con formato `sk-`) sobre los 702 ficheros. **Los unicos aciertos son la palabra `token` en la documentacion de fastText, referida a tokenizadores**, y ningun fichero contiene una credencial real. No hay nada queExclude del repositorio por este motivo.

## Un hallazgo lateral que corrige una belief anterior

En el cierre de V26 se dio por hecho que `tools/verificar_caracteres.py` **no existia**. Si existe, pero pertenece al proyecto `kaggle_toxic_2026_puj` y no a AGE. Para AGE el equivalente es `scripts/v13_p0a_auditoria_idioma.py`, que es el que se ha estado usando y el unico valido para este proyecto. La distincion importa: un verificador de otro proyecto puede tener reglas de caracteres distintas a las de AGE.

## Decision

1. `home/` **no se borra y no se commitea**. Es un residuo de una ruta mal resuelta; borrarlo es decision de Carlos, porque podria querer recuperarlo.

2. `kaggle_toxic_2026_puj/tools/` **no se commitea**. Es una dependencia de otro proyecto y son 18 MB.

3. Ninguno de los dos se incluye en ningun commit de V28, y ninguno se modifica.

4. **No se modifica ningun otro proyecto.** `kaggle_toxic_2026_puj` se deja exactamente como estaba.

## Control absoluto de V20, en el mismo paso

| Comprobacion | Valor |
|---|---|
| corrected_cv recomputado | 0.6097782 |
| Canonico | 0.6097782 |
| Reproduce el macro-F1 | **True** |
| OOF identico bit a bit al artefacto | **True** (max dif 0.00e+00) |
| Folds identicos a los de V20 | **True** [33868, 33868, 33867, 33866, 33867] |
| Features de la matriz | 279 |
| Artefactos de V20 tocados | 0 |
| Hash de la submission | `2c964af78c90bf08...`, intacto |
| Data hash | `32393abca28db4c9` |

La comprobacion del OOF **bit a bit** es mas fuerte que comparar el macro-F1: dos reconstrucciones podrian dar el mismo score con probabilidades distintas. Aqui dan exactamente las mismas.
