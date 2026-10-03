# V30 - Auditoria de la submission

Lo que se puede comprobar **antes** de gastar la unica sonda, y lo que se sabe **despues** de intentarla.

## El fichero

| Comprobacion | Valor |
|---|---|
| Existe | True |
| SHA256 en disco | `f543a715a1881c6128a860301b84a7273c3db8f8a709545be1e6267822e713be` |
| SHA256 en la construccion | `f543a715a1881c6128a860301b84a7273c3db8f8a709545be1e6267822e713be` |
| **Hash coincide** | **True** |
| Filas | 112,892 |
| Checks superados | **10/10** |
| Candidato | E1 Optuna macro-F1 |

## Cuanto se aparta del envio de V20

- Fichero de referencia: `age_submission_v20_candidato_umbrales.csv`
- Decisiones distintas: **7,253** (6.42%)

## Como se uso el test

solo transform y predict. Ninguna etiqueta, ninguna estadistica, ningun ajuste de umbral. Los pesos y los hiperparametros estaban congelados antes de entrar aqui.

## La subida

- Estado: **STOP_SIN_CREDENCIAL**
- Numero de submissions: **0**
- Subido: False

- Causa del STOP: el fichero ~/.kaggle/kaggle.json existe pero solo contiene la clave `token`, que es el formato antiguo; el CLI 2.2.4 exige `username` y `key`, o bien autenticacion OAuth. Sin eso la API rechaza la llamada.
