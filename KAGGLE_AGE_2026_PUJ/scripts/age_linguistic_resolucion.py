"""D-076 - resolucion de la legalidad del anotador y registro de los dos modelos (md y lg).

## Que hace

Registra la decision de Carlos (lectura A: el anotador se usa SOLO por sus etiquetas; los
vectores siguen prohibidos), actualiza el registro de anotadores con los DOS modelos
(`es_core_news_md` y `es_core_news_lg`, ambos no-transformer), y anade la seccion de resolucion
a la auditoria de legalidad. Habilita P2 con condiciones; no ejecuta ningun experimento, no
toca test ni holdout, no genera submission y no modifica el candidato adoptado.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

FECHA = "2026-09-29"
MODELOS = ("es_core_news_md", "es_core_news_lg")
REGISTRO_CSV = cfg.REPORTS / "linguistic_annotation_registry.csv"
LEGAL_MD = cfg.REPORTS / "age_linguistic_legality_audit.md"
RESULTADO = cfg.EXPERIMENTS / "age_linguistic_resolucion.json"
HASH_MD = "889a2fd05d217fa7c1bcffa867af7fc30911bc7b48523f87b5b2525846553ef5"

CONDICIONES = ("solo etiquetas (nunca doc.vector/similitud); procedencia registrada; control de "
               "barajado; ablacion sin canal; submission gateada a autorizacion explicita (D-076)")


def ficha_modelo(nombre: str) -> dict:
    """Ficha tecnica de un modelo instalado, verificada en runtime."""
    import spacy
    ruta = Path(spacy.util.get_package_path(nombre))
    meta_json = ruta / "meta.json"
    if not meta_json.exists():
        meta_json = ruta / nombre / "meta.json"
    sha = hashlib.sha256(meta_json.read_bytes()).hexdigest()
    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    nlp = spacy.load(nombre)
    fecha_alta = time.strftime("%Y-%m-%d", time.localtime(ruta.stat().st_mtime))
    return {
        "recurso": nombre, "version": str(meta.get("version")), "idioma": "es",
        "licencia": meta.get("license"), "fecha_alta": fecha_alta,
        "sha256": sha, "componentes": list(nlp.pipe_names),
        "tamano_mb": round(sum(f.stat().st_size for f in ruta.rglob("*") if f.is_file())
                           / 1e6, 1),
    }


def filas_registro(fichas: dict) -> list[dict]:
    """Registro completo de anotadores con la resolucion de D-076 aplicada."""
    md, lg = fichas["es_core_news_md"], fichas["es_core_news_lg"]
    return [
        {"recurso": "spacy", "tipo": "libreria", "version": "3.8.13", "idioma": "n/a",
         "licencia": "MIT", "fecha_alta": "2026-08-19", "hash_sha256": "n/a",
         "clasificacion": "ALLOWED",
         "evidencia": "reglas especificas: librerias comunes de NLP permitidas; uso historico autorizado (V14)",
         "condiciones": "registro en requirements.txt"},
        {"recurso": md["recurso"], "tipo": "anotador preentrenado no-transformer",
         "version": md["version"], "idioma": md["idioma"], "licencia": md["licencia"],
         "fecha_alta": md["fecha_alta"], "hash_sha256": md["sha256"],
         "clasificacion": "ALLOWED (condicional, D-076)",
         "evidencia": "lectura A autorizada por Carlos (2026-09-29): etiquetas si, vectores no; "
                      "no es transformer/LLM/embedding; D-026/D-043 lo usaron con procedencia",
         "condiciones": CONDICIONES},
        {"recurso": lg["recurso"], "tipo": "anotador preentrenado no-transformer (modelo mayor)",
         "version": lg["version"], "idioma": lg["idioma"], "licencia": lg["licencia"],
         "fecha_alta": lg["fecha_alta"], "hash_sha256": lg["sha256"],
         "clasificacion": "ALLOWED (condicional, D-076)",
         "evidencia": "mandato de comparacion de Carlos (2026-09-29): 'usa un modelo mas grande "
                      "en espanol para ver comparaciones y resultados'",
         "condiciones": CONDICIONES},
        {"recurso": "NLTK", "tipo": "libreria", "version": "3.9.4", "idioma": "n/a",
         "licencia": "Apache-2.0", "fecha_alta": "2026-08-19", "hash_sha256": "n/a",
         "clasificacion": "ALLOWED (libreria)",
         "evidencia": "las reglas la listan por nombre",
         "condiciones": "si se usara un tagger NLTK preentrenado regiria la misma resolucion (D-076)"},
        {"recurso": "transformers", "tipo": "libreria instalada, sin uso",
         "version": "5.16.1", "idioma": "n/a", "licencia": "Apache-2.0", "fecha_alta": "2026-08-19",
         "hash_sha256": "n/a", "clasificacion": "PROHIBITED (tecnica)",
         "evidencia": "prohibido por el profesor; D-038: instalado no es usado; uso en el proyecto = 0",
         "condiciones": "no usar"},
        {"recurso": "sentence-transformers", "tipo": "libreria instalada, sin uso",
         "version": "6.0.1", "idioma": "n/a", "licencia": "Apache-2.0", "fecha_alta": "2026-08-19",
         "hash_sha256": "n/a", "clasificacion": "PROHIBITED (tecnica)",
         "evidencia": "embeddings prohibidos expresamente",
         "condiciones": "no usar"},
        {"recurso": "torch", "tipo": "libreria", "version": "2.14.0", "idioma": "n/a",
         "licencia": "BSD-3-Clause", "fecha_alta": "2026-08-19", "hash_sha256": "n/a",
         "clasificacion": "ALLOWED (libreria)",
         "evidencia": "las reglas listan PyTorch entre las librerias permitidas",
         "condiciones": "no usar para embeddings; el proyecto lo usa solo en la comprobacion de GPU"},
        {"recurso": "stanza", "tipo": "anotador preentrenado (no instalado)",
         "version": "n/a", "idioma": "es", "licencia": "n/a", "fecha_alta": "n/a",
         "hash_sha256": "n/a",
         "clasificacion": "PROHIBITED (sus modelos dependen de embeddings preentrenados)",
         "evidencia": "embeddings prohibidos expresamente", "condiciones": "no usar"},
        {"recurso": "udpipe", "tipo": "anotador preentrenado (no instalado)",
         "version": "n/a", "idioma": "es", "licencia": "n/a", "fecha_alta": "n/a",
         "hash_sha256": "n/a", "clasificacion": "NO USAR",
         "evidencia": "misma zona gris que es_core_news_md, sin la autorizacion de D-076 y sin estar instalado",
         "condiciones": "no usar: no aporta nada que no cubran los modelos autorizados"},
        {"recurso": "flair", "tipo": "anotador preentrenado (no instalado)",
         "version": "n/a", "idioma": "es", "licencia": "n/a", "fecha_alta": "n/a",
         "hash_sha256": "n/a",
         "clasificacion": "PROHIBITED (embeddings contextuales y transformers)",
         "evidencia": "embeddings y transformers prohibidos", "condiciones": "no usar"},
    ]


def escribe_registro(filas: list[dict]) -> None:
    cols = ["recurso", "tipo", "version", "idioma", "licencia", "fecha_alta", "hash_sha256",
            "clasificacion", "evidencia", "condiciones"]
    with REGISTRO_CSV.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        escritor.writeheader()
        for fila in filas:
            escritor.writerow({c: fila[c] for c in cols})


RESOLUCION_MD = """
## Resolucion (D-076, 2026-09-29)

Carlos resolvio la ambiguedad por la **lectura A** y mando anadir un modelo mayor para
comparaciones. Clasificacion final de los anotadores:

| Recurso | Clasificacion final | Condiciones |
|---|---|---|
| es_core_news_md | **ALLOWED (condicional)** | solo etiquetas; procedencia; control de barajado; ablacion sin canal; submission gateada |
| es_core_news_lg | **ALLOWED (condicional)** | identicas; ademas es el modelo de comparacion pedido |

Las seis condiciones de D-076 son obligatorias. La lectura B (estricta, "ningun modelo
preentrenado con vectores en ningun paso") queda documentada arriba como alternativa no
adoptada. La ambiguedad queda resuelta por decision del participante, con el mismo mecanismo
que D-001/D-026/D-043.

**Habilitacion**: P2 (POS estructurado) queda habilitado con el pre-requisito de pre-registrar
su diseno. La anotacion de train con ambos modelos se construye como artefacto derivado
(`features/linguistic/`), determinista y sin etiquetas, con hash en el manifiesto.
"""

D_076 = """
---

## D-076 - Resolucion de legalidad: el anotador se AUTORIZA (etiquetas si, vectores no) y se compara con es_core_news_lg

- **Observation**: la auditoria P1 dejo el anotador `es_core_news_md` en AMBIGUOUS (D-075):
  conflicto interno entre D-001/D-026/D-043 (autorizado) y V31 (cuarentena), mas el hecho de que
  su tok2vec consulta vectores estaticos internamente. Carlos resuelve el 2026-09-29 eligiendo la
  lectura A, y manda ademas comparar con un modelo mayor en espanol.

- **Decision** (extiende D-001/D-026/D-043 a esta fase):
  - El anotador se usa **SOLO por sus etiquetas** (POS, morfologia, dependencias, entidades):
    son forma, no vectores. `doc.vector`, similitudes y cualquier embedding como representacion
    siguen PROHIBITED.
  - **Condiciones obligatorias**: (1) procedencia registrada (modelo, version, idioma, licencia,
    fecha, SHA256); (2) solo etiquetas; (3) control de barajado en todo experimento que aspire a
    gate; (4) ablacion sin canal en cualquier resultado que aspire a gate; (5) ninguna submission
    sale de esta via sin gate completo, segunda particion y autorizacion explicita de Carlos;
    (6) el candidato adoptado (V30 + plug-in, FINAL_REFERENCE) permanece intacto.
  - **Segundo anotador**: `es_core_news_lg` (misma familia no-transformer, modelo mayor) para
    comparar cobertura, acuerdo y efecto del canal. Ambos quedan en el registro de anotadores.

- **Consequences**: P2 queda HABILITADO con el pre-requisito de pre-registrar su diseno. La
  anotacion de train (determinista, sin etiquetas) se guarda en `features/linguistic/` con hash
  en `experiments/age_linguistic_01_anotacion.json`. La lectura B queda documentada como
  alternativa no adoptada. Ningun resultado puede desplazar al candidato adoptado sin gates
  completos.

- **Evidence**: `reports/linguistic_annotation_registry.csv` (actualizado),
  `reports/age_linguistic_legality_audit.md` (seccion Resolucion),
  `experiments/age_linguistic_resolucion.json`.
"""

RUNLOG = """
---

## Resolucion de legalidad del anotador (D-076) · 2026-09-29

- Carlos resuelve la ambiguedad por la lectura A: **etiquetas si, vectores no**, con seis
  condiciones (procedencia, solo etiquetas, control de barajado, ablacion sin canal, submission
  gateada, candidato adoptado intacto).
- Se anade `es_core_news_lg` (modelo mayor, no-transformer) como segundo anotador de comparacion.
- Registro de anotadores actualizado con ambos modelos y su procedencia.
- **P2 habilitado** con pre-registro previo. Sin test, sin holdout, sin submission.
"""

ESTADO = """
---

## Resolucion de legalidad (D-076, 2026-09-29)

El anotador queda **AUTORIZADO condicionalmente** (etiquetas si, vectores no) con las seis
condiciones de D-076, y se anade `es_core_news_lg` como modelo de comparacion. P2 (POS
estructurado) queda habilitado con pre-registro previo. El candidato adoptado (V30 + plug-in,
LB 0,63575) permanece como FINAL_REFERENCE.
"""


def anade(ruta: Path, marca: str, bloque: str) -> str:
    if not ruta.exists():
        return f"{ruta.name}: no existe"
    texto = ruta.read_text(encoding="utf-8")
    if marca in texto:
        return f"{ruta.name}: ya estaba"
    cuerpo = (texto if texto.endswith("\n") else texto + "\n") + bloque
    cuerpo = "\n".join(l.rstrip() for l in cuerpo.splitlines()) + "\n"
    ruta.write_text(cuerpo, encoding="utf-8")
    return f"{ruta.name}: +{len(bloque.splitlines())} lineas"


def anade_experiments(fichas: dict) -> dict:
    ruta = ROOT / "experiments.csv"
    lineas = ruta.read_text(encoding="utf-8").splitlines()
    cols = [c.strip() for c in lineas[0].split(",")]
    if "AGE-LING-RESOLUTION" in {l.split(",", 1)[0] for l in lineas[1:] if l}:
        return {"ya_estaba": True, "filas": len(lineas) - 1}
    reg = {"experiment_id": "AGE-LING-RESOLUTION", "fecha": FECHA, "bloque": "30.1",
           "etapa": "resolucion de legalidad del anotador (D-076) y registro de modelos",
           "pregunta": "se autoriza el anotador y con que condiciones?",
           "hipotesis": "lectura A: etiquetas si, vectores no",
           "features": "ninguna: decision y registro",
           "preprocessing": "ninguno", "representacion": "no aplica",
           "modelo": "es_core_news_md y es_core_news_lg (anotadores)",
           "hiperparametros": "no aplica",
           "validacion": "procedencia verificada en runtime (version, licencia, SHA256)",
           "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "", "micro_f1_mean": "",
           "accuracy_mean": "",
           "resultado": "anotador ALLOWED (condicional): etiquetas si, vectores no; 6 condiciones; "
                        f"md sha {fichas['es_core_news_md']['sha256'][:16]}..., "
                        f"lg sha {fichas['es_core_news_lg']['sha256'][:16]}...",
           "interpretacion": "P2 habilitado con pre-registro previo; lectura B documentada como no adoptada",
           "decision": "contrastar POS-secuencia (md y lg) contra el campeon adoptado en P2",
           "duracion_seg": "0.0",
           "artefactos": "reports/linguistic_annotation_registry.csv | "
                         "reports/age_linguistic_legality_audit.md | "
                         "experiments/age_linguistic_resolucion.json"}
    with ruta.open("a", encoding="utf-8") as f:
        f.write(",".join(f'"{reg[c]}"' if ("," in str(reg[c]) or '"' in str(reg[c])) else str(reg[c])
                         for c in cols) + "\n")
    return {"anadida": True, "filas": len(ruta.read_text(encoding="utf-8").splitlines()) - 1}


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    print("=== fichas de los dos modelos ===", flush=True)
    fichas = {}
    for nombre in MODELOS:
        ficha = ficha_modelo(nombre)
        fichas[nombre] = ficha
        print(f"  {nombre}: {ficha['version']} | {ficha['licencia']} | {ficha['tamano_mb']} MB | "
              f"sha {ficha['sha256'][:16]}...", flush=True)
    if fichas["es_core_news_md"]["sha256"] != HASH_MD:
        print("[STOP] md cambio de hash respecto de V14", flush=True)
        return 1

    escribe_registro(filas_registro(fichas))
    print(f"\n  registro: {len(filas_registro(fichas))} filas", flush=True)

    # nota de actualizacion en la cabecera de la auditoria (marca de vigencia)
    if LEGAL_MD.exists():
        texto = LEGAL_MD.read_text(encoding="utf-8")
        marca_cab = "**Actualizacion (D-076, 2026-09-29)"
        if marca_cab not in texto:
            viejo = "**Clasificacion del anotador `es_core_news_md`: AMBIGUOUS. Estado: P2 gateado.**"
            nuevo = viejo + ("\n\n**Actualizacion (D-076, 2026-09-29): RESUELTO. El anotador queda "
                             "ALLOWED (condicional). Ver la seccion Resolucion al final.**")
            texto = texto.replace(viejo, nuevo)
            LEGAL_MD.write_text("\n".join(l.rstrip() for l in texto.splitlines()) + "\n",
                                encoding="utf-8")
            print("  legalidad: nota de actualizacion insertada", flush=True)

    print("  " + anade(LEGAL_MD, "## Resolucion (D-076", RESOLUCION_MD), flush=True)
    print("  " + anade(ROOT / "DECISIONS.md", "## D-076", D_076), flush=True)
    print("  " + anade(ROOT / "RUNLOG.md", "Resolucion de legalidad del anotador", RUNLOG),
          flush=True)
    print("  " + anade(ROOT / "PROJECT_STATE.md", "## Resolucion de legalidad (D-076", ESTADO),
          flush=True)
    print("  " + str(anade_experiments(fichas)), flush=True)

    RESULTADO.write_text(json.dumps({
        "fase": "D-076 resolucion de legalidad", "fecha": FECHA,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent,
                                 capture_output=True, text=True, timeout=30).stdout.strip(),
        "decision": "lectura A: etiquetas si, vectores no; condiciones D-076",
        "modelos": fichas,
        "p2_habilitado": True,
        "pre_requisito": "pre-registro del diseno de P2 antes de ejecutar",
        "sin_test": True, "sin_holdout": True, "sin_submission": True,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("  experiments/age_linguistic_resolucion.json", flush=True)
    print(f"\n[fin] {time.time() - inicio:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
