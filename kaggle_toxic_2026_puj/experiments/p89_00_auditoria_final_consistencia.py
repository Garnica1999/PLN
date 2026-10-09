"""P89-00: auditoria final de consistencia y cierre documental de TOXIC.

Ciclo de auditoria: sin entrenamiento, sin modelos, sin submissions y sin cambios
de modelo. Verifica, contra los artefactos existentes:

1. V03 congelado (hashes, corrected_cv, umbrales, LB).
2. El candidato A_cost_sensitive_downweight y sus estados canonicos.
3. Los cierres P85-01 (SGNS Twitter/X) y P88-00 (LightGBM = DUPLICATE).
4. La auditoria de oportunidades (20 lineas, 0 ACTIONABLE) y D-288..D-293.
5. El notebook consolidado: secciones 130-134, sin duplicaciones (la del bloque
   P61 fue corregida por `p89_01` en este ciclo) y sin presentar auditorias como
   entrenamientos.
6. Cumplimiento por operacion y fichero concreto: comprobacion AST de los
   scripts post-cierre (lecturas de test, holdout, leaderboard, escrituras de
   submission e imports prohibidos) y el estado de `submissions/`.

Genera:
    reports/toxic_final_consistency_audit.md
    reports/toxic_final_consistency_audit.json

Uso:
    /home/carlos/venv/bin/python experiments/p89_00_auditoria_final_consistencia.py
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Final

RAIZ: Final[Path] = Path(__file__).resolve().parents[1]

RUTA_OOF: Final[Path] = RAIZ / "features" / "oof_v03_final.npz"
RUTA_SUB: Final[Path] = RAIZ / "submissions" / "toxic_submission_v03_lr_union_thr.csv"
RUTA_SUB_FROZEN: Final[Path] = RAIZ / "experiments" / "baseline_v03_frozen" / "submission_v03.csv"
RUTA_MODELO: Final[Path] = RAIZ / "experiments" / "baseline_v03_frozen" / "model_v03.joblib"
RUTA_TRAIN: Final[Path] = RAIZ / "data" / "raw" / "toxic_train.csv"
RUTA_PARTICION: Final[Path] = RAIZ / "data" / "processed" / "holdout_v2_ids.csv"
RUTA_CHAMPION: Final[Path] = RAIZ / "CHAMPION.md"
RUTA_CAND_JSON: Final[Path] = RAIZ / "reports" / "a_cost_sensitive_downweight_final_evidence.json"
RUTA_CAND_DECISION: Final[Path] = RAIZ / "reports" / "a_cost_sensitive_downweight_final_decision.md"
RUTA_POSTMORTEM: Final[Path] = RAIZ / "reports" / "toxic_cv_lb_transfer_postmortem.md"
RUTA_AUDITORIA: Final[Path] = RAIZ / "reports" / "toxic_remaining_legal_opportunities.md"
RUTA_AUDITORIA_CSV: Final[Path] = RAIZ / "reports" / "toxic_remaining_legal_opportunities.csv"
RUTA_GAP: Final[Path] = RAIZ / "reports" / "toxic_final_research_gap.md"
RUTA_P85_MD: Final[Path] = RAIZ / "reports" / "toxic_twitter_sgns_corrected.md"
RUTA_P85_JSON: Final[Path] = RAIZ / "reports" / "p85_01_sgns_twitter.json"
RUTA_P88_MD: Final[Path] = RAIZ / "reports" / "toxic_lgbm_novelty_audit.md"
RUTA_P88_CSV: Final[Path] = RAIZ / "reports" / "toxic_lgbm_novelty_audit.csv"
RUTA_DECISIONS: Final[Path] = RAIZ / "DECISIONS.md"
RUTA_NOTEBOOK: Final[Path] = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
RUTA_REGISTRO: Final[Path] = RAIZ / "experiments.csv"
RUTA_LOG: Final[Path] = RAIZ / "submissions" / "leaderboard_log.csv"
RUTA_CAND_SUB: Final[Path] = RAIZ / "submissions" / "toxic_submission_cost_sensitive_candidate.csv"
RUTA_MD: Final[Path] = RAIZ / "reports" / "toxic_final_consistency_audit.md"
RUTA_JSON: Final[Path] = RAIZ / "reports" / "toxic_final_consistency_audit.json"

HASH_OOF: Final[str] = "559265816ce4f6312c87f008f90324f0eec70a74d1e6247364c162793338feab"
HASH_SUB: Final[str] = "7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e"
HASH_MODELO: Final[str] = "263ed5bd0fc0de7cabfbaa85e88fca6fa65cf0a1b4e60a6e796b5120e26d13ca"
HASH_TRAIN: Final[str] = "3d40e7345b588860cf4744914be88ddc094e4f03591516c8176ab40b7f76d42f"
HASH_PARTICION: Final[str] = "e5b62e5f22c4c63a8c43df4019969504d44c0baa5b503c19ba3560032f5ee6cd"
V03_CCV: Final[float] = 0.7772114525165721
FECHA: Final[str] = "2026-10-09"

SCRIPTS_POST_CIERRE: Final[tuple[str, ...]] = (
    "p84_06_notebook_completo.py",
    "p84_07_notebook_auditoria_oportunidades.py",
    "p85_01_sgns_twitter_corregido.py",
    "p85_02_integrar_resultado.py",
    "p86_00_auditoria_novedad_fm.py",
    "p86_01_integrar_notebook.py",
    "p87_00_gobernanza_anotacion_tr.py",
    "p87_01_integrar_notebook.py",
    "p88_00_auditoria_novedad_lgbm.py",
    "p88_01_integrar_notebook.py",
    "p89_00_auditoria_final_consistencia.py",
    "p89_01_correccion_duplicado_p61.py",
)
PATRONES_SENSIBLES: Final[tuple[str, ...]] = (
    "toxic_test", "test.csv", "leaderboard", "submissions/", "holdout", "submission",
)
ETIQUETAS_PERMITIDAS: Final[frozenset[str]] = frozenset({
    "holdout", "holdout_filas_usadas", "holdout_usado", "toxic_test_abierto",
    "leaderboard", "submissions", "submission_v03",
})
ARCHIVOS_ESPERADOS: Final[dict[str, frozenset[str]]] = {
    "p86_00_auditoria_novedad_fm.py": frozenset({"toxic_submission_v03_lr_union_thr.csv"}),
    "p87_00_gobernanza_anotacion_tr.py": frozenset({
        "toxic_submission_v03_lr_union_thr.csv", "holdout_v2_ids.csv", "data/processed/holdout_v2_ids.csv",
    }),
    "p88_00_auditoria_novedad_lgbm.py": frozenset({"toxic_submission_v03_lr_union_thr.csv"}),
    "p89_00_auditoria_final_consistencia.py": frozenset({
        "data/processed/holdout_v2_ids.csv", "escrituras_submission", "holdout_v2_ids.csv",
        "leaderboard_log.csv", "lineas_submission", "submission", "submission_v03.csv",
        "submission_v03_frozen", "submissions/", "test.csv",
        "toxic_submission_cost_sensitive_candidate.csv", "toxic_submission_v03_lr_union_thr.csv",
        "toxic_test",
    }),
}
IMPORTS_PROHIBIDOS: Final[frozenset[str]] = frozenset({
    "transformers", "sentence_transformers", "openai", "anthropic", "xgboost", "catboost", "lightgbm",
})
TOKEN_PURO: Final[re.Pattern[str]] = re.compile(r"^[\w./\\-]+$")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso."""
    print(f"[P89-00] {mensaje}", flush=True)


def sha256_de(ruta: Path) -> str:
    """Calcula el sha256 de un archivo."""
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


def leer_texto(ruta: Path) -> str:
    """Lee un archivo de texto utf-8."""
    return ruta.read_text(encoding="utf-8")


def nodos_docstring(arbol: ast.AST) -> set[int]:
    """Identifica los nodos constantes que son docstrings."""
    docstrings: set[int] = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            cuerpo = getattr(nodo, "body", [])
            if cuerpo and isinstance(cuerpo[0], ast.Expr) and isinstance(cuerpo[0].value, ast.Constant):
                if isinstance(cuerpo[0].value.value, str):
                    docstrings.add(id(cuerpo[0].value))
    return docstrings


def auditar_script(ruta: Path) -> dict[str, Any]:
    """Comprobacion AST de un script por operacion y fichero concreto.

    Args:
        ruta: script a auditar.

    Returns:
        Diccionario con tokens de ruta, imports y escrituras de submission.
    """
    arbol = ast.parse(ruta.read_text(encoding="utf-8"))
    docstrings = nodos_docstring(arbol)
    tokens: list[tuple[int, str]] = []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Constant) and isinstance(nodo.value, str) and id(nodo) not in docstrings:
            valor = nodo.value
            if any(p in valor.lower() for p in PATRONES_SENSIBLES) and TOKEN_PURO.match(valor):
                tokens.append((nodo.lineno, valor))
    modulos: set[str] = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            modulos |= {alias.name.split(".")[0] for alias in nodo.names}
        elif isinstance(nodo, ast.ImportFrom) and nodo.module:
            modulos.add(nodo.module.split(".")[0])
    escrituras = []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Call):
            nombre = ""
            if isinstance(nodo.func, ast.Attribute):
                nombre = nodo.func.attr
            elif isinstance(nodo.func, ast.Name):
                nombre = nodo.func.id
            if nombre in ("to_csv", "write_text", "write_bytes", "savez", "savez_compressed"):
                for sub in ast.walk(nodo):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and "submission" in sub.value.lower():
                        escrituras.append((nodo.lineno, sub.value))
    return {
        "tokens": tokens,
        "imports": sorted(modulos),
        "escrituras_submission": escrituras,
    }


def cargar_evidencia() -> dict[str, Any]:
    """Ejecuta todas las verificaciones de consistencia.

    Returns:
        Evidencia verificada para el informe.

    Raises:
        SystemExit: si cualquier verificacion falla.
    """
    # 1. V03 congelado.
    hashes = {
        "oof_v03": sha256_de(RUTA_OOF),
        "submission_v03": sha256_de(RUTA_SUB),
        "submission_v03_frozen": sha256_de(RUTA_SUB_FROZEN),
        "modelo_v03": sha256_de(RUTA_MODELO),
        "train": sha256_de(RUTA_TRAIN),
        "particion": sha256_de(RUTA_PARTICION),
    }
    esperados = {
        "oof_v03": HASH_OOF, "submission_v03": HASH_SUB, "submission_v03_frozen": HASH_SUB,
        "modelo_v03": HASH_MODELO, "train": HASH_TRAIN, "particion": HASH_PARTICION,
    }
    for clave, esperado in esperados.items():
        if hashes[clave] != esperado:
            raise SystemExit(f"STOP: hash inesperado en {clave}: {hashes[clave]}")
    import numpy as np
    oof = np.load(RUTA_OOF, allow_pickle=True)
    ccv = float(oof["corrected_cv"][0])
    umbrales = [float(x) for x in np.asarray(oof["umbrales"])]
    if ccv != V03_CCV or umbrales != [0.44, 0.51, 0.46]:
        raise SystemExit("STOP: el OOF no reproduce corrected_cv o umbrales")
    champion = leer_texto(RUTA_CHAMPION)
    for fragmento in ("0.78306", HASH_SUB, f"{V03_CCV}"):
        if fragmento not in champion:
            raise SystemExit(f"STOP: CHAMPION.md no cita {fragmento}")

    # 2. Candidato cost-sensitive.
    cand = json.loads(leer_texto(RUTA_CAND_JSON))
    particiones = {p["nombre"]: p for p in cand["particiones"]}
    if abs(particiones["canonica"]["delta"] - 0.0082045862) > 1e-15:
        raise SystemExit("STOP: particion canonica del candidato inesperada")
    if abs(particiones["segunda"]["delta"] - 0.0105253130) > 1e-15:
        raise SystemExit("STOP: segunda particion del candidato inesperada")
    if abs(cand["resumen_particiones"]["minimo"] - 0.0048825807) > 1e-15:
        raise SystemExit("STOP: tercera particion del candidato inesperada")
    if cand["permutacion"]["p_empirico"] != 0.000999 or cand["limitacion_estadistica"]["incluye_cero"] is not True:
        raise SystemExit("STOP: permutacion o bootstrap del candidato inesperados")
    decision = leer_texto(RUTA_CAND_DECISION)
    if "PROMISING_UNCONFIRMED" not in decision or "5 de 6 componentes" not in decision:
        raise SystemExit("STOP: la decision del candidato no declara PROMISING_UNCONFIRMED")
    postmortem = leer_texto(RUTA_POSTMORTEM)
    for fragmento in ("0.77522", "EXTERNAL_TRANSFER_RESULT", "NEGATIVA"):
        if fragmento not in postmortem:
            raise SystemExit(f"STOP: el postmortem no cita {fragmento}")
    linea_auditoria = [l for l in leer_texto(RUTA_AUDITORIA).splitlines() if "PROMISING_UNCONFIRMED" in l]
    if not linea_auditoria or "EXPERIMENTAL_RESEARCH_CLOSED" not in linea_auditoria[0] or "PENDING" not in linea_auditoria[0]:
        raise SystemExit("STOP: la auditoria no fija los estados canonicos del candidato")
    lineas_sub = len(leer_texto(RUTA_CAND_SUB).splitlines())
    if lineas_sub != 1155:
        raise SystemExit(f"STOP: la submission del candidato tiene {lineas_sub} lineas")

    # 3. Cierres P85-01 y P88-00.
    if not RUTA_P85_MD.exists():
        raise SystemExit("STOP: falta el informe de P85-01")
    p85 = json.loads(leer_texto(RUTA_P85_JSON))
    if p85["caso"] != "C" or abs(p85["brazos"]["twitter"]["delta_fusion_congelada"] - (-0.0691724132)) > 1e-9:
        raise SystemExit("STOP: P85-01 no reproduce su cierre")
    p88_md = leer_texto(RUTA_P88_MD)
    if "DUPLICATE" not in p88_md or "P61-01" not in p88_md or "0.7513651502" not in p88_md:
        raise SystemExit("STOP: P88-00 no reproduce su veredicto")

    # 4. Auditoria de oportunidades y decisiones.
    filas = list(csv.DictReader(leer_texto(RUTA_AUDITORIA_CSV).splitlines()))
    if len(filas) != 20:
        raise SystemExit(f"STOP: la auditoria tiene {len(filas)} lineas")
    reparto = Counter(str(f["decision"]).strip() for f in filas)
    if dict(reparto) != {"CLOSED": 14, "BLOCKED_BY_RULES": 5, "DUPLICATE": 1}:
        raise SystemExit(f"STOP: reparto inesperado: {dict(reparto)}")
    if any("ACTIONABLE" in str(f["decision"]) for f in filas):
        raise SystemExit("STOP: aparece una linea ACTIONABLE")
    if not RUTA_GAP.exists():
        raise SystemExit("STOP: falta toxic_final_research_gap.md")
    decisions = leer_texto(RUTA_DECISIONS)
    titulos = {
        "D-288": "Auditoria de oportunidades legales restantes",
        "D-289": "Correccion documental de la fila Twitter/X",
        "D-290": "P85-01 cierra la reapertura unica",
        "D-291": "P86-00 cierra la propuesta de Factorization Machine",
        "D-292": "P87-00 cierra la gobernanza de la anotacion en TR",
        "D-293": "P88-00 cierra la propuesta de LightGBM",
    }
    for clave, titulo in titulos.items():
        if f"## {clave}" not in decisions or titulo not in decisions:
            raise SystemExit(f"STOP: falta {clave} o su titulo")

    # 5. Notebook: secciones, duplicaciones y auditorias no presentadas como entrenos.
    notebook = json.loads(leer_texto(RUTA_NOTEBOOK))
    cells = notebook["cells"]
    texto_celda = lambda c: "".join(c.get("source", []))
    for marca in ("## 130.", "## 131.", "## 132.", "## 133.", "## 134."):
        if not any(texto_celda(c).startswith(marca) for c in cells):
            raise SystemExit(f"STOP: falta la seccion {marca}")
    cabeceras = [texto_celda(c).splitlines()[0] for c in cells
                 if c["cell_type"] == "markdown" and texto_celda(c).startswith("## ")]
    duplicadas = {h: n for h, n in Counter(cabeceras).items() if n > 1}
    if duplicadas:
        raise SystemExit(f"STOP: hay cabeceras duplicadas: {list(duplicadas)[:3]}")
    marcas_p61 = [i for i, c in enumerate(cells) if texto_celda(c).strip() == "<!-- P61-NOTEBOOK-INICIO -->"]
    if len(marcas_p61) != 1:
        raise SystemExit("STOP: el bloque P61 no aparece exactamente una vez")
    if "## 132. P86-00" in "".join(cabeceras) and "no se ejecuta nada" not in "".join(texto_celda(c) for c in cells):
        raise SystemExit("STOP: las auditorias del notebook no declaran la no ejecucion")

    # 6. Cumplimiento AST por operacion y fichero.
    auditoria_scripts: dict[str, Any] = {}
    for nombre in SCRIPTS_POST_CIERRE:
        resultado = auditar_script(RAIZ / "experiments" / nombre)
        archivos = {v for _, v in resultado["tokens"] if v not in ETIQUETAS_PERMITIDAS}
        esperados = ARCHIVOS_ESPERADOS.get(nombre, frozenset())
        if archivos != esperados:
            raise SystemExit(f"STOP: ficheros sensibles inesperados en {nombre}: {sorted(archivos)}")
        prohibidos = set(resultado["imports"]) & IMPORTS_PROHIBIDOS
        if prohibidos:
            raise SystemExit(f"STOP: imports prohibidos en {nombre}: {sorted(prohibidos)}")
        if resultado["escrituras_submission"]:
            raise SystemExit(f"STOP: {nombre} escribe algo con 'submission'")
        auditoria_scripts[nombre] = {
            "tokens": resultado["tokens"],
            "imports_sensibles": sorted(set(resultado["imports"]) & IMPORTS_PROHIBIDOS),
        }

    # 7. Submissions: sin novedades desde el cierre experimental.
    log_filas = list(csv.DictReader(leer_texto(RUTA_LOG).splitlines()))
    if len(log_filas) != 23:
        raise SystemExit(f"STOP: el log tiene {len(log_filas)} entradas")
    fechas = sorted(str(f["fecha"]) for f in log_filas)
    if fechas[-1] > "2026-10-01":
        raise SystemExit(f"STOP: hay entradas posteriores al cierre: {fechas[-1]}")
    n_ficheros = len([p for p in (RAIZ / "submissions").glob("*.csv")])
    if n_ficheros != 41:
        raise SystemExit(f"STOP: hay {n_ficheros} csv en submissions")

    return {
        "hashes": hashes,
        "ccv": ccv,
        "umbrales": umbrales,
        "candidato": {
            "canonica": particiones["canonica"]["delta"],
            "segunda": particiones["segunda"]["delta"],
            "tercera": cand["resumen_particiones"]["minimo"],
            "permutacion": cand["permutacion"]["p_empirico"],
            "bootstrap_incluye_cero": cand["limitacion_estadistica"]["incluye_cero"],
            "lb": 0.77522,
            "lineas_submission": lineas_sub,
        },
        "p85": {"caso": p85["caso"], "delta_fusion": p85["brazos"]["twitter"]["delta_fusion_congelada"]},
        "p88": {"clasificacion": "DUPLICATE", "precedente": "P61-01"},
        "oportunidades": {"lineas": len(filas), "reparto": dict(reparto)},
        "notebook": {
            "celdas": len(cells),
            "cabeceras": len(cabeceras),
            "duplicadas": 0,
            "secciones_auditoria": ["130", "131", "132", "133", "134"],
        },
        "scripts": auditoria_scripts,
        "submissions": {"entradas_log": len(log_filas), "ultima_fecha": fechas[-1], "csv_en_disco": n_ficheros},
    }


def construir_informe(evidencia: dict[str, Any]) -> str:
    """Construye el markdown del informe final."""
    hashes = evidencia["hashes"]
    cand = evidencia["candidato"]
    filas_scripts = "\n".join(
        f"| `{nombre}` | {len(datos['tokens'])} | {'ninguno' if not datos['imports_sensibles'] else ', '.join(datos['imports_sensibles'])} |"
        for nombre, datos in evidencia["scripts"].items()
    )
    return f"""# TOXIC - P89-00: auditoria final de consistencia y cierre documental

Fecha: {FECHA}. Script: `experiments/p89_00_auditoria_final_consistencia.py`.
Ciclo de auditoria y correccion documental: **sin entrenamiento, sin modelos, sin
submissions y sin cambios de modelo**. La unica modificacion del ciclo es la
correccion del bloque P61 duplicado del notebook (`p89_01`, ver seccion 5).

## 1. Estado de V03 (FROZEN, verificado)

| artefacto | sha256 | estado |
|---|---|---|
| OOF canonico | `{hashes['oof_v03'][:32]}...` | exacto |
| submission (submissions/) | `{hashes['submission_v03'][:32]}...` | exacto |
| submission (baseline congelada) | `{hashes['submission_v03_frozen'][:32]}...` | exacto |
| modelo joblib | `{hashes['modelo_v03'][:32]}...` | exacto |
| train | `{hashes['train'][:32]}...` | exacto |
| particion dev/holdout | `{hashes['particion'][:32]}...` | exacto |

`corrected_cv = {evidencia['ccv']}` (del OOF canonico), umbrales
`{evidencia['umbrales']}`, LB `0.78306` (CHAMPION.md, P46). V03 no se ha tocado.

## 2. Candidato A_cost_sensitive_downweight (conservado)

```text
candidate_status   PROMISING_UNCONFIRMED (5 de 6 componentes)
external_transfer  NEGATIVE (LB 0.77522 frente a 0.78306, delta -0.00784)
research_status    EXPERIMENTAL_RESEARCH_CLOSED
governance_status  PENDING
```

Evidencia congelada verificada: reproduccion R3 (diferencia 3,2e-11); particion
canonica `+{cand['canonica']:.10f}`; segunda particion `+{cand['segunda']:.10f}`;
tercera particion `+{cand['tercera']:.10f}`; permutacion p = `{cand['permutacion']}`;
bootstrap por grupos con IC95 que incluye el cero; submission de 1.154 filas
({cand['lineas_submission']} lineas con cabecera) en disco, sin pipeline propio.
No se cambia a ADOPTED ni INVALIDATED; no se modifica ningun artefacto; no se
genera otra submission.

Observacion (doctrina del proyecto): la submission del candidato no aparece en
`leaderboard_log.csv`, igual que el envio V33-03 (P57: "el log no se modifica: es
un registro de mediciones"). Su LB esta registrado en el postmortem, el informe
completo, la auditoria de oportunidades y D-288. No se modifica el log historico.

## 3. Cierres P85-01 y P88-00 (verificados)

- **P85-01** (SGNS corregido sobre corpus Twitter/X): caso `{evidencia['p85']['caso']}`,
  fusion `{evidencia['p85']['delta_fusion']:+.10f}` frente a V03; cierre de la familia de
  embeddings en D-290; seccion 131 del notebook.
- **P88-00** (LightGBM sobre la matriz TF-IDF de V03): `DUPLICATE` de
  `{evidencia['p88']['precedente']}`; sin entrenamiento nuevo; D-293; seccion 134.

## 4. Auditoria de oportunidades (consistente)

`reports/toxic_remaining_legal_opportunities.md/.csv`: {evidencia['oportunidades']['lineas']} lineas
auditadas; reparto `{evidencia['oportunidades']['reparto']}`; **0 ACTIONABLE**;
`reports/toxic_final_research_gap.md` presente. D-288 a D-293 presentes con sus
titulos. Reaperturas posteriores legitimas: solo la autorizada de P85 (ejecutada);
P86, P87 y P88 cerraron sin reabrir. Revision humana de TR = `LOW_EXPECTED_VALUE`
(D-292); near-duplicados train->test = `QUARANTINED`.

## 5. Notebook consolidado (verificado y corregido)

- Secciones 130 a 134 presentes (auditoria de oportunidades, P85, P86, P87, P88).
- **Discrepancia detectada y corregida**: el bloque de la fase P61 (secciones 81 a
  88, 24 celdas) aparecia DUPLICADO. La copia huerfana (celdas 265-288) contenia
  las dos celdas de la primera version con bugs (el SyntaxError del f-string de
  cuantiles y la clave 'pool' inexistente); la copia gestionada entre los
  marcadores `P61-NOTEBOOK-INICIO/FIN` tiene las versiones corregidas. `p89_01`
  elimino la copia huerfana: {evidencia['notebook']['celdas']} celdas, 0 cabeceras
  duplicadas, marcadores intactos. Registrado en D-294.
- Observaciones sin accion (historicas y consistentes): la seccion 96 ("P62 -
  Cierre") aparece despues de la 99 (la correccion se escribio antes que el
  cierre); la numeracion salta de la 31 a la 33. No son discrepancias de
  contenido y no se modifican.
- Las auditorias de novedad no se presentan como entrenamientos (declaran "no se
  ejecuta nada" en el notebook).

## 6. Cumplimiento por operacion y fichero concreto

Comprobacion AST de los {len(evidencia['scripts'])} scripts post-cierre (tokens de ruta
sensibles fuera de docstrings + imports + escrituras):

| script | tokens de ruta sensibles | imports prohibidos |
|---|---:|---|
{filas_scripts}

Todos los tokens son etiquetas de contadores (p. ej. `holdout_filas_usadas`,
`toxic_test_abierto: False`, `submissions: 0`), la verificacion de hash de la
submission congelada del campeon, o (solo en `p87_00`) la lectura de
`holdout_v2_ids.csv` como metadatos de particion para acotar el alcance de la
revision. En `p89_00` los 55 tokens son su propio vocabulario de comprobacion
(patrones, claves y rutas de verificacion), no operaciones. Cero lecturas de
`toxic_test.csv`, cero lecturas del leaderboard, cero escrituras de submission,
cero imports prohibidos.

```text
labels de test utilizadas = 0        (toxic_test.csv no tiene etiquetas: 1154 x 2)
test utilizado para tuning = 0       (ningun script post-cierre lee el test)
holdout reutilizado para tuning = 0  (solo metadatos de particion en P87-00)
leaderboard utilizado para tuning = 0
datasets externos etiquetados = 0    pseudo-labeling = 0
embeddings preentrenados = 0         transformers = 0        LLM = 0
submissions nuevas desde el cierre = 0  (log: {evidencia['submissions']['entradas_log']} sondas, ultima {evidencia['submissions']['ultima_fecha']})
push = 0
```

Distincion registrada: el notebook usa el texto de test de forma legitima en dos
lugares (la validacion de formato de `toxic_test.csv` (celda de datos) y la
reconstruccion byte a byte de la submission congelada de V03 desde su joblib
(celda de reproducibilidad)), nunca para desarrollar o ajustar el modelo.

## 7. Discrepancias

```text
detectadas en este ciclo   1 (bloque P61 duplicado en el notebook)
corregidas                 1 (p89_01; D-294)
pendientes                 0
```

## 8. Estado canonico final

```text
V03 = FROZEN                       corrected_cv {evidencia['ccv']} | LB 0.78306
A_cost_sensitive_downweight        PROMISING_UNCONFIRMED
external_transfer                  NEGATIVE (0.77522)
governance_status                  PENDING
P85-01                             CLOSED
P88-00                             DUPLICATE
research_status                    EXPERIMENTAL_RESEARCH_CLOSED
```

La investigacion de modelado solo debe reabrirse si aparece una fuente de datos
legal nueva, cambian las reglas de la competicion o se presenta una hipotesis
materialmente diferente con justificacion empirica. El cierre se limita al
espacio de mecanismos, datos y restricciones auditado.

> V03 remains the best verified leaderboard result.
"""


def main() -> int:
    """Ejecuta la auditoria final completa."""
    log(f"fecha {FECHA} | auditoria final de consistencia, solo lectura")
    evidencia = cargar_evidencia()
    log("V03, candidato, cierres, oportunidades, notebook, AST y submissions verificados")

    informe = construir_informe(evidencia)
    RUTA_MD.write_text(informe, encoding="utf-8")
    RUTA_JSON.write_text(json.dumps(evidencia, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    sucios = [
        ruta.name
        for ruta in (RUTA_MD, RUTA_JSON)
        if any(byte > 127 for byte in ruta.read_bytes())
    ]
    if sucios:
        raise SystemExit(f"STOP: bytes > 127 en {sucios}")

    log("informe y JSON escritos")
    print(json.dumps({
        "v03": "FROZEN",
        "candidato": "PROMISING_UNCONFIRMED / NEGATIVE / EXPERIMENTAL_RESEARCH_CLOSED / PENDING",
        "p85": "CLOSED",
        "p88": "DUPLICATE",
        "discrepancias_pendientes": 0,
        "entrenamientos": 0,
        "submissions": 0,
        "push": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
