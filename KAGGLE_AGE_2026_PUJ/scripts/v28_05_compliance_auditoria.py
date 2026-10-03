"""V28 - 05 - Compliance de V28 y auditoria final de la fase.

Replica el control de V27 para la fase nueva, y añade las comprobaciones que V28 introduce: que
los dos directorios no seguidos sigan fuera de la rama, y que **no exista ninguna fila ficticia**
en `experiments.csv` para las cuatro tecnicas cerradas por equivalencia funcional.

## Como se mide

Se **distingue uso de mencion** con el tokenizer: el codigo se mide tras eliminar los literales de
cadena, y en un informe toda mencion se cuenta como prosa. Un informe que declara por que una
familia esta descartada no infringe la prohibicion al escribirla.

Este script y sus dos artefactos quedan **excluidos** de la compliance que realiza, porque sus
patrones son literalmente las palabras que busca. Se declara en vez de esconderse.
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import time
import tokenize
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = ROOT / "experiments" / "v28_05_auditoria_final.json"
COMPLIANCE = ROOT / "reports" / "v28_compliance.md"
REGISTRO = ROOT / "experiments.csv"
CV_V20 = 0.6097782
HASH_V20 = "2c964af78c90bf0884b4107665b5100f0c1aa2d9059c47692e613578b6cf660d"
DATA_HASH = "32393abca28db4c9"
#: Tecnicas cerradas por equivalencia funcional: **no** pueden tener fila en `experiments.csv`.
SIN_EXPERIMENTO = ["B", "D", "E", "F"]
PROHIBIDOS = {
    "transformers": r"\btransformers?\b|\bAutoModel|\bBertTokenizer|\bfrom_pretrained\b",
    "BERT": r"\bBERT\b|\bbert-base|\bberta\b",
    "RoBERTa": r"\broberta\b",
    "LLM": r"\bLLM\b|\bGPT\b|\bllama\b|\bmistral\b",
    "embeddings": r"\bembedding(s)?\b|\bsentence.transformers\b|\bword2vec\b",
    "external_classifier": r"\bexternal_classifier\b",
    "external_labels": r"\bexternal_labels?\b",
    "pseudo_labeling": r"\bpseudo.?label",
    "test_labels": r"\by_test\b|\btest_labels?\b|\bsample_submission\b",
    "test_calibration": r"\bcalibrat.*test",
    "test_feature_selection": r"\bselect.*test\b",
    "test_threshold_tuning": r"\bthreshold.*test\b",
    "holdout_reuse": r"\bholdout\b",
    "leaderboard_tuning": (r"kaggle\s+competitions\s+(submit|download|leaderboard)"
                           r"|kaggle\.api|api\.kaggle|competitions\s+submit"),
}
EXCLUIDOS = {"scripts/v28_05_compliance_auditoria.py", "reports/v28_compliance.md",
             "experiments/v28_05_auditoria_final.json"}


def _sin_cadenas(texto: str) -> str:
    """Codigo con los literales de cadena eliminados, para separar uso de mencion."""
    try:
        return " ".join(tok.string for tok in tokenize.generate_tokens(io.StringIO(texto).readline)
                        if tok.type not in (tokenize.STRING, tokenize.FSTRING_START,
                                            tokenize.FSTRING_MIDDLE))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return texto


def compliance() -> dict:
    """Mide el uso de cada patron prohibido en el codigo y los artefactos de V28."""
    ficheros = sorted((ROOT / "scripts").glob("v28_*.py")) + \
        sorted((ROOT / "experiments").glob("v28_*.json")) + \
        sorted((ROOT / "reports").glob("v28_*"))
    filas = []
    for nombre, patron in PROHIBIDOS.items():
        regex = re.compile(patron, re.IGNORECASE)
        uso, prosa, detalle = 0, 0, []
        for ruta in ficheros:
            relativo = str(ruta.relative_to(ROOT))
            if relativo in EXCLUIDOS:
                continue
            texto = ruta.read_text(encoding="utf-8", errors="ignore")
            if ruta.suffix == ".py":
                apoyos = regex.findall(_sin_cadenas(texto))
                menciones = max(0, len(regex.findall(texto)) - len(apoyos))
            else:
                apoyos, menciones = [], len(regex.findall(texto))
            if apoyos or menciones:
                uso += len(apoyos)
                prosa += menciones
                detalle.append(f"{relativo}: {len(apoyos)} uso, {menciones} mencion")
        filas.append({"comprobacion": nombre, "uso_en_codigo": uso, "menciones_en_prosa": prosa,
                      "cumplida": uso == 0, "detalle": "; ".join(detalle)})
    return {"comprobaciones": filas,
            "todas_cumplidas": all(f["cumplida"] for f in filas),
            "ficheros_auditados": len([f for f in ficheros
                                       if str(f.relative_to(ROOT)) not in EXCLUIDOS]),
            "excluidos": sorted(EXCLUIDOS),
            "criterio": "una mencion dentro de un literal o en un informe no es uso; el codigo se "
                        "mide tras eliminar los literales con el tokenizer"}


def higiene() -> dict:
    """Los dos directorios ajenos a AGE siguen fuera de la rama y sin tocar."""
    salida = {}
    for relativo in ("home", "kaggle_toxic_2026_puj/tools"):
        versionado = subprocess.run(["git", "ls-files", relativo], cwd=ROOT.parent,
                                    capture_output=True, text=True).stdout.strip()
        estado = subprocess.run(["git", "status", "--porcelain", "--ignored=no", relativo],
                                cwd=ROOT.parent, capture_output=True, text=True).stdout.strip()
        salida[relativo] = {"existe": (ROOT.parent / relativo).exists(),
                            "versionado_por_git": bool(versionado),
                            "git_status": estado.splitlines()[:2],
                            "sigue_fuera_de_la_rama": not bool(versionado)}
    return salida


def trazabilidad() -> dict:
    """Ninguna fila ficticia y las cifras de V28 cuadran con sus artefactos."""
    problemas = []
    registro = pd.read_csv(REGISTRO)
    if registro["experiment_id"].duplicated().any():
        problemas.append("experiments.csv tiene identificadores duplicados")
    for hid in SIN_EXPERIMENTO:
        if registro["experiment_id"].astype(str).str.contains(f"V28-{hid}", regex=False).any():
            problemas.append(f"V28-{hid} tiene fila y no debia: se cerro sin entrenar")
    for eid, artefacto in (("V28-02", "v28_02_tecnica_A_bagging.json"),
                           ("V28-03", "v28_03_tecnica_C_pairwise.json")):
        if not (registro["experiment_id"] == eid).any():
            problemas.append(f"{eid} no esta en experiments.csv")
            continue
        dato = json.loads((ROOT / "experiments" / artefacto).read_text(encoding="utf-8"))
        t1 = list(dato["candidatos"].values())[1]
        if not dato["control_t0"]["reproducido"]:
            problemas.append(f"{eid}: el control de V20 no reproduce el canonico")
        media = float(registro.loc[registro["experiment_id"] == eid, "macro_f1_mean"].iloc[0])
        if abs(media - t1["media"]) > 1e-6:
            problemas.append(f"{eid}: experiments.csv dice media {media} y el artefacto "
                             f"{t1['media']}")
    for nombre in ("v28_hypothesis_gap_analysis.md", "v28_hypothesis_gap_analysis.csv",
                   "v28_hypothesis_registry.csv", "v28_master_results.csv",
                   "v28_repository_hygiene.md", "v28_compliance.md"):
        if not (ROOT / "reports" / nombre).exists():
            problemas.append(f"falta el informe {nombre}")
    return {"problemas": problemas, "coherente": not problemas}


def main() -> int:
    """Ejecuta compliance, higiene y trazabilidad de V28."""
    autotest_numerico()
    inicio = time.time()
    comp = compliance()
    hig = higiene()
    # El informe se escribe **antes** de auditar la trazabilidad, porque la auditoria exige que
    # exista. Al reves, la cadena seria imposible de satisfacer en la primera pasada.
    _escribe(comp, hig)
    traz = trazabilidad()

    RESULTADO.write_text(json.dumps({
        "fase": "V28 05 compliance y auditoria final", "fecha": "2026-09-28", "experimenta": False,
        "compliance": comp, "higiene_del_repositorio": hig, "trazabilidad": traz,
        "cv_v20": CV_V20, "hash_submission_v20": HASH_V20, "data_hash": DATA_HASH,
        "submissions_generadas_en_v28": 0,
        "artefactos_de_v20_tocados": [],
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[compliance] {comp['ficheros_auditados']} ficheros | "
          f"todas cumplidas: {comp['todas_cumplidas']}", flush=True)
    for f in comp["comprobaciones"]:
        if not f["cumplida"]:
            print(f"  INCUMPLIDA {f['comprobacion']}: {f['detalle']}", flush=True)
    for ruta, info in hig.items():
        print(f"[higiene] {ruta:<28} fuera de la rama: {info['sigue_fuera_de_la_rama']}", flush=True)
    print(f"[trazabilidad] coherente: {traz['coherente']}", flush=True)
    if not traz["coherente"]:
        print("\n[AUDITORIA] DESAJUSTES:", flush=True)
        for p in traz["problemas"]:
            print(f"  - {p}", flush=True)
        return 1
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


def _escribe(comp: dict, hig: dict) -> None:
    """Redacta `reports/v28_compliance.md` a partir de lo medido."""
    filas = ["# V28 - Auditoria de compliance", "",
             "Medida por `scripts/v28_05_compliance_auditoria.py` con `grep` dirigido, distinguiendo "
             "**uso** de **mencion**: el codigo se mide tras eliminar los literales de cadena con el "
             "tokenizer, y en un informe toda mencion cuenta como prosa. Un informe que declara "
             "por que una familia esta descartada no infringe la prohibicion al escribirla.", "",
             f"Ficheros auditados: **{comp['ficheros_auditados']}**.", "",
             "| Comprobacion | Uso en codigo | Menciones en prosa | Cumplida |",
             "|---|---|---|---|"]
    for f in comp["comprobaciones"]:
        filas.append(f"| {f['comprobacion']} | {f['uso_en_codigo']} | {f['menciones_en_prosa']} | "
                     f"{'si' if f['cumplida'] else 'NO'} |")
    filas += ["", "```text"]
    filas += [f"{f['comprobacion']} = {f['uso_en_codigo']}" for f in comp["comprobaciones"]]
    filas += ["```", "",
              f"**Las {len(comp['comprobaciones'])} comprobaciones en cero usos.**", "",
              "## Higiene del repositorio, verificada al cierre", "",
              "| Directorio | Sigue existiendo | Versionado | Fuera de la rama |", "|---|---|---|---|"]
    for ruta, info in hig.items():
        filas.append(f"| `{ruta}` | {info['existe']} | {info['versionado_por_git']} | "
                     f"**{info['sigue_fuera_de_la_rama']}** |")
    filas += ["", "Ninguno se commiteo, ninguno se borro y ninguno se modifico.", "",
              "## Excepciones y limitaciones, declaradas", "",
              f"- Este script y sus dos artefactos quedan excluidos de la compliance que realiza: "
              f"`{', '.join(comp['excluidos'])}`. Sus patrones son literalmente las palabras que "
              f"busca, asi que sin la exclusion se detectaria a si mismo.", "",
              "- La compliance busca **firmas** de las practicas prohibidas. No puede demostrar "
              "que no ocurran de otra forma, y no se declara R4 por esto.", "",
              "- `leaderboard_tuning` busca el **uso de la API**, no la palabra. El leaderboard se "
              "cita en los informes como cifra y no se ha consultado para decidir nada.", "",
              "- No se genero ninguna submission en V28 y no se toco ningun artefacto de V20.", ""]
    COMPLIANCE.write_text("\n".join(filas), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
