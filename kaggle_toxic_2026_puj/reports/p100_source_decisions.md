# P100 - S1: decisiones de fuentes y autorizacion (2026-10-13)

## 1. Marco

Autorizaciones del propietario (2026-10-13) que gobiernan esta fase:

- ChatSubs: evaluar para un experimento academico no comercial de
  preentrenamiento desde cero bajo condiciones: verificar ficha y artefacto;
  verificar condiciones del texto de origen; atribucion documentada; no
  redistribuir el corpus original ni publicar checkpoints, pesos o derivados
  sin revision especifica; solo subconjunto en espanol; sin usar anotaciones
  ni labels; cuarentena de corpus de odio intacta; detener la variante si la
  auditoria detecta una restriccion material; sin garantia juridica sobre el
  contenido subyacente.
- Descargas S1: A = Digital Narratives unicamente tras verificar procedencia,
  licencia y disponibilidad real; B = artefacto ChatSubs si la ficha confirma
  la licencia; C = investigar Wikimedia talk pages como reserva sin descargar
  corpus completos.
- Si solo existe COVID adicional y ChatSubs no puede utilizarse: cerrar con
  gap analysis. Si ChatSubs se autoriza: evaluar la viabilidad de ambas
  variantes pre-registradas; cerrar todo brazo sin documentos unicos
  suficientes, sin relleno por duplicacion.
- Presupuesto: maximo 2 configuraciones nuevas; maximo 10 GPU-h; no
  obligatorio agotarlo. Arquitectura, tokens totales, procedimientos
  downstream y criterios de parada congelados. No reabrir P99.
- Secuencia S1..S7; sin toxic_test/holdouts/leaderboard; sin modificar V03 ni
  los resultados historicos de P98/P99; sin submissions; sin commits/pushes
  automaticos.

## 2. Decision por fuente

| Fuente | Estado | Decision |
|---|---|---|
| S100-A DHCOVID | PROHIBITED (sin texto utilizable) | No usable: artefactos publicos = IDs; servicio /get0/ roto verificado; rehidratacion excluida. Licencia OK, disponibilidad NO. |
| S100-B ChatSubs | ALLOWED_WITH_RESTRICTIONS | Unica fuente conversacional usada (variante B) bajo condiciones del propietario. |
| S100-C esCorpiusDialog | NEEDS_EXPLICIT_PERMISSION | Registrada; no usada. |
| S100-D Reddit dialogues | NEEDS_EXPLICIT_PERMISSION | Sin licencia; no usada. |
| S100-E OPUS OpenSubtitles | NEEDS_EXPLICIT_PERMISSION | Redundante con S100-B; no usada. |
| S100-F Wikimedia talk pages | ALLOWED_WITH_RESTRICTIONS | Reserva; no descargada (verificar volumen/justificacion antes). |
| S100-G/H/I IDs Twitter | PROHIBITED | Sin texto; rehidratacion excluida. |
| S100-J TASS/sentimiento | NEEDS_EXPLICIT_PERMISSION | No usada. |

## 3. Consecuencias para las variantes P100

- Variante A (social reforzado, +4.5M social nuevo): NO VIABLE. No existe
  fuente autorizada y disponible de texto social nuevo (S100-A sin texto;
  S100-G/H/I solo IDs). Se cierra el brazo por material, sin relleno por
  duplicacion (regla del propietario).
- Variante B (social 15% + conversacional 30% + general 55%, 30M): VIABLE.
  Unica configuracion nueva a entrenar (dentro del maximo de 2).
- Sin sustitucion de fuentes dudosas; cuarentenas intactas; sin labels.
- La auditoria de S100-B no detecto restriccion material que impida el uso
  academico no comercial interno; obligaciones registradas (seccion 4).

## 4. Obligaciones operativas (ChatSubs)

- Solo subconjunto es (open_subtitles_es); sin anotaciones ni labels.
- Atribucion en reportes (ChatSubs: Kharitonova et al.; fuente original:
  OpenSubtitles).
- No redistribuir el corpus; no publicar checkpoints, pesos ni derivados sin
  revision especifica de condiciones. El proyecto no publica artefactos
  (sin commits/push).
- Uso interno academico no comercial; detener ante restriccion material.

## 5. Verificaciones tecnicas y pendientes

- Verificado: LICENSE.txt y README.txt (md5 oficiales); registro Zenodo;
  error del servicio /get0/; muestras de IDs; conteos del repositorio
  (555 carpetas); checksum oficial del tar.gz (md5 se re-verifica localmente
  al completar la descarga).
- Pendiente S1->S3: md5 local del tar.gz; extraccion; medicion del
  subconjunto es; construccion del corpus B1 y manifiesto (S3).

## 6. Lo que NO se hizo

- No se leyo, escaneo ni uso toxic_test ni holdouts; sin leaderboard.
- No se descargo ninguna fuente fuera de las autorizadas.
- No se tocaron artefactos historicos P98/P99 ni V03; sin commits.
