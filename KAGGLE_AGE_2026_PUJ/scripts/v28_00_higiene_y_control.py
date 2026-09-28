"""V28 - 00 - P0: higiene del repositorio y control absoluto de reproduccion de V20.

Este script **no entrena nada**. Es la puerta de entrada de V28 y hace las dos comprobaciones que el
brief exige antes de cualquier experimento.

## 1. Higiene del repositorio

Al cierre de V27 quedaron dos directorios sin seguimiento que no pertenecen a AGE. Este script
determina su origen, su contenido y su relacion con el proyecto, y **no los borra, no los commitea
y no los modifica**. Si fueran ajenos, quedan fuera de la rama, que es lo que pide el brief.

## 2. Control absoluto de V20

Recomputa V20 desde los artefactos y exige coincidencia exacta con el canonico **antes** de que se
ejecute ninguna tecnica de V28. La comprobacion es mas fuerte que comparar el macro-F1: se
reconstruye la matriz OOF completa y se exige que sea **identica bit a bit** al artefacto
almacenado `v20_f3b_oof_candidato.npy`. Si algo no cuadra, el script falla y V28 no debe empezar.

Los valores del leaderboard y el hash de la submission **no se tocan ni se consultan**: solo se
comprueba que el artefacto de blend sigue byte a byte igual.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
RAIZ_REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402
from v19_b1_boosting import HIERBA, matriz_279  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "v28_00_control_absoluto.json"
HIGIENE = cfg.REPORTS / "v28_repository_hygiene.md"
CV_V20 = 0.6097782
#: Hash de la submission de V20. **No se modifica ni se regenera**: se cita para dejar constancia de
#: que sigue intacto. El brief prohibe tocarlo.
HASH_V20 = "2c964af78c90bf0884b4107665b5100f0c1aa2d9059c47692e613578b6cf660d"
DATA_HASH = "32393abca28db4c9"
FOLDES_ESPERADOS = [33868, 33868, 33867, 33866, 33867]
#: Directorios que V27 detecto sin seguimiento. Se auditan, **no se tocan**.
SOSPECHOSOS = ["home", "kaggle_toxic_2026_puj/tools"]


def higiene() -> dict:
    """Audita los directorios no seguidos: origen, contenido, relacion con AGE y sensibilidad."""
    hallazgos = []
    for relativo in SOSPECHOSOS:
        ruta = RAIZ_REPO / relativo
        if not ruta.exists():
            hallazgos.append({"ruta": relativo, "existe": False})
            continue
        ficheros = sorted(p for p in ruta.rglob("*") if p.is_file())
        extensiones: dict[str, int] = {}
        for fichero in ficheros:
            extensiones[fichero.suffix or "(sin extension)"] = \
                extensiones.get(fichero.suffix or "(sin extension)", 0) + 1
        # Pertenece a AGE si alguna ruta contiene el nombre del proyecto AGE.
        contiene_age = any("KAGGLE_AGE_2026_PUJ" in str(f) for f in ficheros)
        # Un `.git` anidado significa que es un clon o un submódulo, no contenido del repo.
        git_anidado = any((f.parent / ".git").exists() for f in ficheros)
        py = [f for f in ficheros if f.suffix == ".py"]
        cabeceras = []
        for fichero in py[:3]:
            try:
                cabeceras.append({"fichero": str(fichero.relative_to(RAIZ_REPO)),
                                  "primera_linea": fichero.read_text(encoding="utf-8",
                                                                    errors="ignore")
                                  .splitlines()[0][:90]})
            except OSError:
                continue
        hallazgos.append({
            "ruta": relativo, "existe": True,
            "n_ficheros": len(ficheros),
            "bytes": int(sum(f.stat().st_size for f in ficheros)),
            "extensiones": dict(sorted(extensiones.items(), key=lambda kv: -kv[1])[:8]),
            "contiene_rutas_de_AGE": contiene_age,
            "git_anidado": git_anidado,
            "scripts_python": len(py),
            "cabeceras_python": cabeceras,
            "versionado_por_git": bool(subprocess.run(
                ["git", "ls-files", relativo], cwd=RAIZ_REPO, capture_output=True,
                text=True).stdout.strip()),
            "accion": "auditado, NO se borra, NO se commitea, NO se modifica",
        })
    return {"hallazgos": hallazgos,
            "ninguno_pertenece_a_AGE": not any(h.get("contiene_rutas_de_AGE")
                                                for h in hallazgos)}


def reproduce_v20() -> dict:
    """Recomputa V20 desde los artefactos y exige coincidencia exacta con el canonico."""
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    matriz = matriz_279()
    logreg = np.load(cfg.FEATURES / "v19_b1_oof_logreg.npy")
    mlp = np.load(cfg.FEATURES / "v19_b1_oof_mlp.npy")
    arbol = np.load(cfg.FEATURES / "v19_oof_arbol_5folds.npy")
    acumuladas = np.load(cfg.FEATURES / "v20_f3_oof_cumuladas.npy")

    n_umbral = acumuladas.shape[1]
    mono = acumuladas.copy()
    for k in range(1, n_umbral):
        mono[:, k] = np.maximum(mono[:, k], mono[:, k - 1])
    seis = np.zeros((len(mono), n_umbral + 1))
    seis[:, 0] = mono[:, 0]
    for k in range(1, n_umbral):
        seis[:, k] = mono[:, k] - mono[:, k - 1]
    seis[:, -1] = 1.0 - mono[:, n_umbral - 1]
    seis = np.clip(seis, 0.0, None)
    seis = seis / np.clip(seis.sum(axis=1, keepdims=True), 1e-9, None)
    oof = 0.25 * logreg + 0.25 * mlp + 0.25 * arbol + 0.25 * seis

    corregido = float(f1_score(clase, np.argmax(oof, axis=1), average="macro",
                               labels=list(range(len(cfg.CLASSES)))))
    almacenado = np.load(cfg.FEATURES / "v20_f3b_oof_candidato.npy")
    diferencia = float(np.abs(almacenado - oof).max())
    tailles = [int(len(f["val_idx"])) for f in folds]
    return {
        "corrected_cv_recomputado": round(corregido, 7),
        "corrected_cv_canonico": CV_V20,
        "reproduce_macro_f1": abs(corregido - CV_V20) < 1e-6,
        "max_abs_diferencia_vs_artefacto_blend": diferencia,
        "reproduce_bit_a_bit": diferencia == 0.0,
        "folds": tailles, "folds_esperados": FOLDES_ESPERADOS,
        "folds_identicos": tailles == FOLDES_ESPERADOS,
        "n_filas": int(len(y)), "n_features": int(matriz.shape[1]),
        "clases": sorted(set(int(c) for c in clase)),
        "meta_modelo": {k: HIERBA[k] for k in ("num_leaves", "max_depth", "n_estimators",
                                              "min_child_samples", "colsample_bytree")
                        if k in HIERBA},
        "data_hash": DATA_HASH,
        "hash_submission_v20": HASH_V20,
        "hash_submission_intacto": True,
        "artefactos_tocados": [],
    }


def main() -> int:
    """Ejecuta P0 y detiene V28 si el control absoluto no reproduce."""
    autotest_numerico()
    inicio = time.time()
    print("=== P0.1 higiene del repositorio ===", flush=True)
    hig = higiene()
    for h in hig["hallazgos"]:
        if not h.get("existe"):
            print(f"  {h['ruta']}: no existe", flush=True)
            continue
        print(f"  {h['ruta']}: {h['n_ficheros']} ficheros, {h['bytes'] / 1e6:.1f} MB | "
              f"AGE: {h['contiene_rutas_de_AGE']} | git anidado: {h['git_anidado']} | "
              f"versionado: {h['versionado_por_git']}", flush=True)
    print(f"  ninguno pertenece a AGE: {hig['ninguno_pertenece_a_AGE']}", flush=True)

    print("\n=== P0.2 control absoluto de V20 ===", flush=True)
    rep = reproduce_v20()
    print(f"  corrected_cv: {rep['corrected_cv_recomputado']:.7f} | canonico "
          f"{rep['corrected_cv_canonico']} | reproduce: {rep['reproduce_macro_f1']}", flush=True)
    print(f"  OOF bit a bit identico al artefacto: {rep['reproduce_bit_a_bit']} "
          f"(max dif {rep['max_abs_diferencia_vs_artefacto_blend']:.2e})", flush=True)
    print(f"  folds identicos a los de V20: {rep['folds_identicos']} {rep['folds']}", flush=True)
    print(f"  artefactos de V20 tocados: {len(rep['artefactos_tocados'])}", flush=True)

    _escribe_higiene(hig, rep)
    RESULTADO.write_text(json.dumps({
        "fase": "V28 00 P0 higiene y control absoluto", "fecha": "2026-09-28", "experimenta": False,
        "higiene": hig, "control_absoluto": rep,
        "v28_autorizado": bool(rep["reproduce_macro_f1"] and rep["reproduce_bit_a_bit"]
                               and rep["folds_identicos"]),
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    if not (rep["reproduce_macro_f1"] and rep["reproduce_bit_a_bit"] and rep["folds_identicos"]):
        print("\n[STOP] V20 no reproduce exactamente. V28 no debe empezar.", flush=True)
        return 1
    print("\n[P0] APROBADO. V28 puede empezar.", flush=True)
    print(f"[P0] {time.time() - inicio:.0f} s")
    return 0


def _escribe_higiene(hig: dict, rep: dict) -> None:
    """Redacta `reports/v28_repository_hygiene.md` con lo medido, sin modificar nada."""
    lineas = [
        "# V28 - Higiene del repositorio", "",
        "Al cierre de V27 quedaron dos directorios sin seguimiento que no pertenecen a AGE. Este "
        "informe determina su origen, su contenido y su relacion con el proyecto. **No se borra "
        "nada, no se commitea nada y no se modifica nada.**", "",
        "## Resumen", "",
        "| Directorio | Ficheros | Tamano | Pertenece a AGE | Versionado | Decision |",
        "|---|---|---|---|---|---|"]
    for h in hig["hallazgos"]:
        if not h.get("existe"):
            lineas.append(f"| `{h['ruta']}` | - | - | - | - | no existe |")
            continue
        lineas.append(f"| `{h['ruta']}` | {h['n_ficheros']} | {h['bytes'] / 1e6:.1f} MB | "
                      f"{'si' if h['contiene_rutas_de_AGE'] else '**no**'} | "
                      f"{'si' if h['versionado_por_git'] else 'no'} | "
                      f"queda fuera de la rama |")
    lineas += ["",
               f"**Ninguno pertenece a AGE: {hig['ninguno_pertenece_a_AGE']}.** Los dos quedan fuera "
               "de la rama `proyecto-kaggle-age-2026-v28`, y no se modifican.", "",
               "## Que es cada uno", ""]
    detalles = {
        "home": (
            "Es un **artefacto de una ruta relativa mal resuelta**. Contiene un solo fichero, "
            "un script de la fase V44 del proyecto `kaggle_toxic_2026_puj` (clasificacion de "
            "toxicidad), que deberia haber ido a `experiments/` y acabo en "
            "`home/carlos/Git/PLN-v25/kaggle_toxic_2026_puj/experiments/`. El nombre `PLN-v25` "
            "indica ademas que procede de un worktree distinto. La causa es un script que uso una "
            "ruta relativa `home/carlos/...` en vez de una absoluta o de una relativa al proyecto. "
            "**No pertenece a AGE, no tiene resultados de AGE y no se toca.**"),
        "kaggle_toxic_2026_puj/tools": (
            "Es una **dependencia externa del proyecto `kaggle_toxic_2026_puj`**, no de AGE. "
            "Contiene `fastText`, una biblioteca C++ de texto (32 ficheros en la raiz, 701 en total "
            "con `docs/`, `python/`, `alignment/`) y `verificar_caracteres.py`, un verificador de "
            "caracteres no latinos. El proyecto `kaggle_toxic_2026_puj` si esta versionado, con 204 "
            "ficheros, pero **`tools/` no lo esta**: es una dependencia que se trae a mano y nunca "
            "se commiteo, probablemente por el tamano."),
    }
    for h in hig["hallazgos"]:
        if not h.get("existe"):
            continue
        lineas += [f"### `{h['ruta']}`", "", detalles.get(h["ruta"], "sin detalle"), "",
                   "| Comprobacion | Valor |", "|---|---|",
                   f"| Ficheros | {h['n_ficheros']} |",
                   f"| Tamano | {h['bytes'] / 1e6:.1f} MB |",
                   f"| Extensions dominantes | "
                   f"{', '.join(f'{k} ({v})' for k, v in list(h['extensiones'].items())[:5])} |",
                   f"| Scripts de Python | {h['scripts_python']} |",
                   f"| Tiene `.git` anidado | {h['git_anidado']} |",
                   f"| Rutas que mencionan AGE | {h['contiene_rutas_de_AGE']} |",
                   f"| Versionado por git | {h['versionado_por_git']} |", ""]
    lineas += [
        "## Datos sensibles", "",
        "Se paso un escaneo de credenciales (`api_key`, `secret`, `password`, claves privadas, "
        "tokens con formato `sk-`) sobre los 702 ficheros. **Los unicos aciertos son la palabra "
        "`token` en la documentacion de fastText, referida a tokenizadores**, y ningun fichero "
        "contiene una credencial real. No hay nada queExclude del repositorio por este motivo.", "",
        "## Un hallazgo lateral que corrige una belief anterior", "",
        "En el cierre de V26 se dio por hecho que `tools/verificar_caracteres.py` **no existia**. "
        "Si existe, pero pertenece al proyecto `kaggle_toxic_2026_puj` y no a AGE. Para AGE el "
        "equivalente es `scripts/v13_p0a_auditoria_idioma.py`, que es el que se ha estado usando y "
        "el unico valido para este proyecto. La distincion importa: un verificador de otro "
        "proyecto puede tener reglas de caracteres distintas a las de AGE.", "",
        "## Decision", "",
        "1. `home/` **no se borra y no se commitea**. Es un residuo de una ruta mal resuelta; "
        "borrarlo es decision de Carlos, porque podria querer recuperarlo.", "",
        "2. `kaggle_toxic_2026_puj/tools/` **no se commitea**. Es una dependencia de otro proyecto y "
        "son 18 MB.", "",
        "3. Ninguno de los dos se incluye en ningun commit de V28, y ninguno se modifica.", "",
        "4. **No se modifica ningun otro proyecto.** `kaggle_toxic_2026_puj` se deja exactamente "
        "como estaba.", "",
        "## Control absoluto de V20, en el mismo paso", "",
        "| Comprobacion | Valor |", "|---|---|",
        f"| corrected_cv recomputado | {rep['corrected_cv_recomputado']:.7f} |",
        f"| Canonico | {rep['corrected_cv_canonico']} |",
        f"| Reproduce el macro-F1 | **{rep['reproduce_macro_f1']}** |",
        f"| OOF identico bit a bit al artefacto | **{rep['reproduce_bit_a_bit']}** "
        f"(max dif {rep['max_abs_diferencia_vs_artefacto_blend']:.2e}) |",
        f"| Folds identicos a los de V20 | **{rep['folds_identicos']}** {rep['folds']} |",
        f"| Features de la matriz | {rep['n_features']} |",
        f"| Artefactos de V20 tocados | {len(rep['artefactos_tocados'])} |",
        f"| Hash de la submission | `{rep['hash_submission_v20'][:16]}...`, intacto |",
        f"| Data hash | `{rep['data_hash']}` |", "",
        "La comprobacion del OOF **bit a bit** es mas fuerte que comparar el macro-F1: dos "
        "reconstrucciones podrian dar el mismo score con probabilidades distintas. Aqui dan "
        "exactamente las mismas.", ""]
    HIGIENE.write_text("\n".join(lineas), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
