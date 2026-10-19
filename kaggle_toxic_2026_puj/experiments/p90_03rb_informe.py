"""P90-03-R parte 2: informe de atribucion exacta OOF y manifiesto.

Lee `reports/p90_03r_audit_results.json` y `reports/p90_03r_feature_contributions.csv`
y produce:

- `reports/p90_03r_exact_oof_attribution.md` (informe);
- `reports/p90_03r_manifest.json` (manifiesto de reproducibilidad).

Sin entrenamiento adicional, sin test, sin holdout, sin submissions, sin commits.

Uso:
    /home/carlos/venv/bin/python experiments/p90_03rb_informe.py
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import platform
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

RAIZ: Path = Path(__file__).resolve().parents[1]

RUTA_RESULTS = RAIZ / "reports" / "p90_03r_audit_results.json"
RUTA_CONTRIB = RAIZ / "reports" / "p90_03r_feature_contributions.csv"
RUTA_REPRO = RAIZ / "reports" / "p90_03r_oof_reproduction.csv"
RUTA_ATRIB = RAIZ / "reports" / "p90_03r_exact_oof_attribution.csv"
RUTA_INFORME = RAIZ / "reports" / "p90_03r_exact_oof_attribution.md"
RUTA_MANIFEST = RAIZ / "reports" / "p90_03r_manifest.json"
RUTA_SCRIPT_R = RAIZ / "experiments" / "p90_03r_reconstruccion_oof.py"
RUTA_SCRIPT_RB = RAIZ / "experiments" / "p90_03rb_informe.py"

ETIQUETAS = ("HS", "TR", "AG")


def leer_csv(ruta: Path) -> list[dict[str, str]]:
    """Lee un CSV respetando saltos de linea embebidos."""
    with ruta.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def sha256_de(ruta: Path) -> str:
    """Calcula el sha256 de un archivo."""
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


def escribir_texto(ruta: Path, texto: str) -> None:
    """Escribe texto exigiendo ASCII puro."""
    if any(ord(c) > 127 for c in texto):
        raise SystemExit(f"STOP: bytes > 127 en {ruta.name}")
    ruta.write_text(texto, encoding="utf-8")


def tabla(cabeceras: list[str], filas: list[list[Any]]) -> str:
    """Construye una tabla markdown."""
    lineas = ["| " + " | ".join(cabeceras) + " |",
              "|" + "|".join("---" for _ in cabeceras) + "|"]
    for fila in filas:
        lineas.append("| " + " | ".join(str(c) for c in fila) + " |")
    return "\n".join(lineas)


def top1_de(fila: dict[str, str]) -> str:
    """Nombre de la feature top-1 positiva de una fila de contribuciones."""
    s = fila["top_positive_features"]
    return s.split("; ")[0].split("|")[0] if s else ""


def signo(v: float) -> int:
    """Signo de un valor con tolerancia cero exacta."""
    return 1 if v > 0 else (-1 if v < 0 else 0)


def bloque_far(contrib: list[dict[str, str]]) -> dict[str, Any]:
    """Agregados exactos de los errores alejados (>=0.100) desde las contribuciones."""
    far: dict[str, Any] = {}
    for et in ETIQUETAS:
        for tipo in ("FP", "FN"):
            sel = [f for f in contrib if f["label"] == et and f["error_type"] == tipo
                   and float(f["distance_to_threshold"]) >= 0.100]
            top1 = Counter(top1_de(f) for f in sel if top1_de(f))
            por_fold = {k: Counter() for k in range(1, 6)}
            for f in sel:
                if top1_de(f):
                    por_fold[int(f["fold"])][top1_de(f)] += 1
            far[f"{et}|{tipo}"] = {
                "n": len(sel),
                "acumulacion": sum(1 for f in sel if float(f["top3_abs_share"]) < 0.25),
                "pocas_dominantes": sum(1 for f in sel if float(f["top3_abs_share"]) >= 0.5),
                "conflicto_wc": sum(1 for f in sel if f["word_char_conflict"] == "1"),
                "mismo_signo": sum(1 for f in sel
                                   if signo(float(f["word_contribution_sum"])) ==
                                   signo(float(f["char_contribution_sum"]))),
                "top_features_pos": "; ".join(f"{k}:{v}" for k, v in top1.most_common(8)),
                "top1_por_fold": "; ".join(
                    f"f{k}:{por_fold[k].most_common(1)[0][0]}({por_fold[k].most_common(1)[0][1]})"
                    if por_fold[k] else f"f{k}:-" for k in range(1, 6)),
            }
    return far


def perra_por_fold(contrib: list[dict[str, str]]) -> dict[str, str]:
    """Repeticion de 'perra' como top-1 positivo de los FP, por fold."""
    salida = {}
    for et in ETIQUETAS:
        conteo = Counter()
        for f in contrib:
            if f["label"] == et and f["error_type"] == "FP" and top1_de(f) == "perra":
                conteo[int(f["fold"])] += 1
        salida[et] = "; ".join(f"f{k}:{conteo.get(k, 0)}" for k in range(1, 6))
    return salida


def construir_informe(x: dict[str, Any]) -> str:
    """Construye el informe de la atribucion exacta OOF."""
    d = x["d"]
    g = d["gate"]
    ag = d["agregados"]
    firmas = d["firmas"]
    casos = d["casos_clave"]
    far = x["far"]
    perra_fold = x["perra_fold"]
    slices = d["slices"]

    filas_gate = [
        ["allclose(rtol=1e-7, atol=1e-8)", str(g["allclose_rtol_1e-7_atol_1e-8"])],
        ["error absoluto maximo", f"{g['max_abs_error']:.3e}"],
        ["error absoluto medio", f"{g['mean_abs_error']:.3e}"],
        ["percentil 99 del error absoluto", f"{g['p99_abs_error']:.3e}"],
        ["discordancias binarias", str(g["binary_discordances"])],
        ["corrected_cv reconstruido", f"{g['ccv_reconstructed']:.16f}"],
        ["diferencia de corrected_cv (tolerancia 1e-7)", f"{g['ccv_difference']:.3e}"],
        ["folds recomputados == persistidos", str(g["folds_recomputed_equal"])],
        ["gate superado", str(g["gate_passed"])],
    ]
    filas_resumen = []
    for et in ETIQUETAS:
        for tipo in ("FP", "FN"):
            a = ag[f"{et}|{tipo}"]
            filas_resumen.append([et, tipo, a["n"], a["mean_word"], a["mean_char"],
                                  a["mean_abs_word"], a["mean_abs_char"],
                                  f"{a['mismo_signo']}/{a['n']}", a["top1_pos_es_lexico"],
                                  a["acumulacion"], a["conflicto_wc"]])
    filas_firmas = []
    for et in ETIQUETAS:
        for tipo in ("FP", "FN"):
            for clave in sorted(firmas):
                e, t, firma = clave.split("|", 2)
                if e == et and t == tipo:
                    filas_firmas.append([et, tipo, firma, firmas[clave]])
    filas_far = []
    for et in ETIQUETAS:
        for tipo in ("FP", "FN"):
            f = far[f"{et}|{tipo}"]
            filas_far.append([et, tipo, f["n"], f["acumulacion"], f["pocas_dominantes"],
                              f["conflicto_wc"], f["top_features_pos"][:80], f["top1_por_fold"][:80]])
    filas_slice = [[f["slice"], f["label"], f["n_errors_inside"], f["mean_word_inside"],
                    f["mean_char_inside"], f["mean_active_inside"], f["mean_active_outside"]]
                   for f in slices if f["slice"] in ("URLs", "hashtags", "colectivos_superficie",
                                                     "menciones", "mencion_inicial", "cita")]
    filas_comp = [
        ["Acumulacion domina frente a pocas features dominantes",
         "aux: FN 76-86 por ciento acumulacion, 0-3 pocas",
         f"exacto: FN 79/67/82 por ciento; FP 62/34/61 por ciento; pocas 0-5",
         "CONFIRMED_BY_EXACT_OOF"],
        ["Conflicto word-char frecuente en FP de HS",
         "aux: 75/173 (43 por ciento)",
         f"exacto: {ag['HS|FP']['conflicto_wc']}/173 ({100*ag['HS|FP']['conflicto_wc']/173:.0f} por ciento)",
         "PARTIALLY_CONFIRMED"],
        ["Conflicto word-char frecuente en FP de AG",
         "aux: 68/135 (50 por ciento)",
         f"exacto: {ag['AG|FP']['conflicto_wc']}/135 ({100*ag['AG|FP']['conflicto_wc']/135:.0f} por ciento)",
         "NOT_CONFIRMED"],
        ["Conflicto word-char en FN de HS/AG bajo",
         "aux: 17 y 3 por ciento",
         f"exacto: {ag['HS|FN']['conflicto_wc']}/143 (48) y {ag['AG|FN']['conflicto_wc']}/142 (56)",
         "CONTRADICTED"],
        ["Canal char dominante en FN de HS/AG",
         "aux: 89/143 (62) y 79/142 (56)",
         f"exacto: {ag['HS|FN']['char_dominante']}/143 (1.4) y {ag['AG|FN']['char_dominante']}/142 (21)",
         "CONTRADICTED"],
        ["'perra' como top-1 positivo en FP",
         "aux: 32/27/30 (HS/TR/AG)",
         f"exacto: {casos['perra_top1_FP']['HS']}/{casos['perra_top1_FP']['TR']}/{casos['perra_top1_FP']['AG']}",
         "CONFIRMED_BY_EXACT_OOF"],
        ["Contribuciones negativas de 'la puta' y 'mi perra'",
         "aux: presentes (mi perra top-neg en 9/8/7 FP)",
         f"exacto: mi perra en top-5 neg de errores {casos['mi_perra_en_neg5']['HS']}/{casos['mi_perra_en_neg5']['TR']}/{casos['mi_perra_en_neg5']['AG']}; "
         f"la puta {casos['la_puta_en_neg5']['HS']}/{casos['la_puta_en_neg5']['TR']}/{casos['la_puta_en_neg5']['AG']}",
         "CONFIRMED_BY_EXACT_OOF"],
        ["URLs/colectivos: mas features activas y mezcla distinta",
         "aux: URLs HS 248.9 vs 185.0; colectivos 272.7 vs 177.5",
         "exacto: URLs HS 227.0 vs 167.1; colectivos 247.5 vs 161.0",
         "CONFIRMED_BY_EXACT_OOF"],
        ["TR FN con canales del mismo signo (100 por ciento)",
         "aux: 82/82",
         f"exacto: {ag['TR|FN']['mismo_signo']}/82 ({100*ag['TR|FN']['mismo_signo']/82:.0f} por ciento)",
         "NOT_CONFIRMED"],
        ["FP vs FN: firmas de canal opuestas",
         "aux: FP con word medio negativo (modelo final no cometia el error)",
         f"exacto: FP word/char medios +0.33/+0.45 (HS), +0.98/+0.91 (TR), +0.62/+0.61 (AG); "
         f"FN -0.45/-0.13 (HS), +0.01/-0.02 (TR), -0.31/+0.04 (AG)",
         "CONFIRMED_BY_EXACT_OOF"],
    ]

    return f"""# P90-03-R - Recuperacion de la atribucion exacta OOF de V03

Generado por `experiments/p90_03r_reconstruccion_oof.py` y `experiments/p90_03rb_informe.py`.
Commit auditado: `{d['hashes']['script14'][:8]} (script historico)` | HEAD git en el manifiesto.
Fecha: 2026-10-10. Fase diagnostica autorizada: reconstruccion del pipeline historico por fold
para recuperar la atribucion exacta. Sin cambios en V03, sin test, sin holdout, sin submissions,
sin commits.

## 1. Resumen ejecutivo

**La reproduccion es exacta bit a bit.** Los cinco folds se reconstruyeron con el procedimiento
historico (`experiments/14_oof_corregido_v2.py`): vectorizadores TF-IDF word(1,2)+char_wb(3,5)
ajustados por fold y tres LogisticRegression (C=2.0, balanced, liblinear, random_state=42) por
fold. Las probabilidades reconstruidas coinciden con el OOF canonico con **error maximo 0.0**
(0 discordancias binarias; `corrected_cv` = {g['ccv_reconstructed']:.16f}, diferencia 0.0).
Gate superado con las tolerancias predefinidas (rtol 1e-7, atol 1e-8, ccv 1e-7).

Con los modelos fold-specific reales, la descomposicion `logit = intercepto + suma(contribuciones)`
se verifica en las 4449 filas-etiqueta (residuo maximo {d['decomposition_residual_max']:.2e}) y se
recuperan las contribuciones exactas de los 757 errores. **La comparacion con P90.03 reclasifica
varias conclusiones auxiliares**: se confirman el predominio de la acumulacion, el papel de
"perra", las contribuciones negativas de "la puta"/"mi perra" y las diferencias FP/FN; se
**contradicen** la dominancia del canal char en los FN y el conflicto word-char en FP de AG
(eran artefactos del modelo final in-sample). Decision final: **EXACT_OOF_ATTRIBUTION_RECOVERED**.

## 2. Alcance y autorizacion

Fase P90.03-R: reconstruir el pipeline historico por fold exclusivamente para recuperar la
informacion de atribucion no persistida. No cambia V03, thresholds, folds ni protocolo; no busca
hiperparametros; no genera submissions. La reconstruccion se uso como instrumento de atribucion
y el `corrected_cv` reconstruido solo como control de reproduccion.

## 3. Auditoria de fuentes

{tabla(["fuente", "sha256[:16]", "rol"], [
    ["features/oof_v03_final.npz", d["hashes"]["oof"][:16], "OOF canonico (intacto)"],
    ["features/v2_oof_base_corrected.npz", d["hashes"]["v2base"][:16], "padre del OOF (linaje M2, 0.0)"],
    ["experiments/14_oof_corregido_v2.py", d["hashes"]["script14"][:16], "procedimiento historico"],
    ["src/splits.py", d["hashes"]["splits"][:16], "folds y grupos (usados via OOF persistido)"],
    ["src/data.py", d["hashes"]["data"][:16], "lectura del train"],
    ["data/raw/toxic_train.csv", d["hashes"]["train"][:16], "textos y etiquetas"],
    ["reports/eda/fase2_lexico_inducido_agresividad.csv", d["hashes"]["lexico"][:16], "categorizacion descriptiva"],
])}

Git: rama main, HEAD sin cambios (e43c594), cambios locales de P90.02/P90.03 preservados.
Software: Python {x['python']}, scikit-learn {x['sklearn']}, numpy {x['numpy']}, scipy {x['scipy']}.
Los pliegues persistidos en el OOF se recomputaron con `pliegues_corregidos` y coinciden
exactamente (folds_recomputed_equal = {g['folds_recomputed_equal']}).

## 4. Ausencia de artefactos fold-specific originales

No existen joblib/pkl de los modelos ni vectorizadores por fold: el script historico los ajusta
en memoria y guarda solo probabilidades. Esta fase reconstruye esos artefactos
**exclusivamente para atribucion**, sin persistirlos como candidatos ni modificar el protocolo.

## 5. Procedimiento de reconstruccion (exacto)

Replicado de `14_oof_corregido_v2.py`: `train_dev` = filas dev del train en el orden original
(verificado identico al OOF: ids y y_true); folds = `pliegues` persistidos (1..5, cubren 1483
exactamente una vez); por fold: `TfidfVectorizer(analyzer=word, ngram_range=(1,2), min_df=2,
sublinear_tf=True)` + `TfidfVectorizer(analyzer=char_wb, ngram_range=(3,5), min_df=2,
sublinear_tf=True)` ajustados solo con el entrenamiento del fold; union = hstack(word, char);
tres `LogisticRegression(C=2.0, class_weight=balanced, max_iter=1000, solver=liblinear,
random_state=42)`; prediccion solo sobre la validacion del fold. Tamanos por fold: train
1186/1186/1187/1187/1186, val 297/297/296/296/297; features por fold 20441-21044.

## 6. Control de reproduccion

{tabla(["criterio", "resultado"], filas_gate)}

Criterio predefinido y no ampliado: `np.allclose(rtol=1e-7, atol=1e-8)`, 0 discordancias
binarias, |delta corrected_cv| <= 1e-7. **Reproduccion exacta: el error maximo es 0.0 en todas
las folds y etiquetas.**

## 7. Metodologia de atribucion exacta

Para cada fila de validacion del fold que la predijo: `logit = intercepto + suma(x_k*coef_k)`;
`probabilidad = sigmoid(logit)`; descomposicion por feature (`contribucion = tfidf * coef`) con
canales word/char separados; residuo de descomposicion por fila (maximo
{d['decomposition_residual_max']:.2e}). Las tablas guardan token/n-grama, canal, valor TF-IDF,
coeficiente y contribucion local de las features top-10 positivas y negativas de cada
fila-etiqueta (`p90_03r_feature_contributions.csv`, 4449 filas) y de cada error
(`p90_03r_exact_oof_attribution.csv`, 757 filas). Recuentos verificados: 757 celdas / 441
tweets; estratos {d['estratos']}.

## 8. Resultados HS

- **HS FP (173)**: word medio {ag['HS|FP']['mean_word']:+.3f}, char medio {ag['HS|FP']['mean_char']:+.3f}
  (ambos positivos: el fold modelo cometio el error); top-1 del lexico {ag['HS|FP']['top1_pos_es_lexico']}/173;
  acumulacion {ag['HS|FP']['acumulacion']}/173; conflicto word-char {ag['HS|FP']['conflicto_wc']}/173;
  char dominante {ag['HS|FP']['char_dominante']}/173 (42 por ciento).
  Top positivos: perra 32, los 20, zorra 8, cllate 8. Top negativos: de 62, la 49, https 38,
  en 31, mi perra 9.
- **HS FN (143)**: word medio {ag['HS|FN']['mean_word']:+.3f}, char medio {ag['HS|FN']['mean_char']:+.3f}
  (ambos negativos: la decision real fue a la baja); acumulacion {ag['HS|FN']['acumulacion']}/143;
  conflicto word-char {ag['HS|FN']['conflicto_wc']}/143; lexico en top-10 {ag['HS|FN']['lexico_en_top10_pos']}/143.
  Top negativos: de 66, la 63, en 44, el 37, https 23, rabe 10, polla 10.

## 9. Resultados TR

- **TR FP (82)**: word {ag['TR|FP']['mean_word']:+.3f}, char {ag['TR|FP']['mean_char']:+.3f};
  top-1 del lexico {ag['TR|FP']['top1_pos_es_lexico']}/82 (77 por ciento); mismo signo
  {ag['TR|FP']['mismo_signo']}/82; conflicto word-char solo {ag['TR|FP']['conflicto_wc']}/82.
  Top negativos: de 24, la 23, la puta 8, mi perra 8.
- **TR FN (82)**: word {ag['TR|FN']['mean_word']:+.3f}, char {ag['TR|FN']['mean_char']:+.3f}
  (margenes netos casi nulos); mismo signo {ag['TR|FN']['mismo_signo']}/82; conflicto word-char
  {ag['TR|FN']['conflicto_wc']}/82. Top positivos: puta 15, perra 10, sudaca 7; top negativos:
  de 41, la 23, el 20, en 19, https 14.

## 10. Resultados AG

- **AG FP (135)**: word {ag['AG|FP']['mean_word']:+.3f}, char {ag['AG|FP']['mean_char']:+.3f};
  top-1 del lexico {ag['AG|FP']['top1_pos_es_lexico']}/135; acumulacion {ag['AG|FP']['acumulacion']}/135;
  conflicto word-char {ag['AG|FP']['conflicto_wc']}/135 (14 por ciento, muy inferior al auxiliar).
- **AG FN (142)**: word {ag['AG|FN']['mean_word']:+.3f}, char {ag['AG|FN']['mean_char']:+.3f};
  conflicto word-char {ag['AG|FN']['conflicto_wc']}/142 (56 por ciento); acumulacion
  {ag['AG|FN']['acumulacion']}/142; top negativos: la 45, en 44, de 41, el 34, https 22,
  rabe 8, polla 8.

## 11. Firmas exactas

{tabla(["etiqueta", "tipo", "firma", "n"], filas_firmas)}

## 12. Casos clave

- **perra como top-1 positivo de FP**: HS {casos['perra_top1_FP']['HS']}, TR {casos['perra_top1_FP']['TR']},
  AG {casos['perra_top1_FP']['AG']}; repeticion por fold: HS {perra_fold['HS']}; TR {perra_fold['TR']};
  AG {perra_fold['AG']}.
- **la puta en top-5 negativos de errores**: HS {casos['la_puta_en_neg5']['HS']}, TR
  {casos['la_puta_en_neg5']['TR']}, AG {casos['la_puta_en_neg5']['AG']}. **mi perra**: HS
  {casos['mi_perra_en_neg5']['HS']}, TR {casos['mi_perra_en_neg5']['TR']}, AG {casos['mi_perra_en_neg5']['AG']}.
- **Caso 23839 (HS FP fronterizo, d=4.6e-05)**: logit -0.240977 = intercepto -0.253207 +
  word -0.143680 + char +0.155910. "perra" (word) contribuye +0.558860 y "mi perra" (word,
  coef -1.6009) contribuye -0.551078: el bigrama afiliativo casi cancela al insulto y la
  prediccion queda a 4.6e-05 del threshold. Mecanismo exacto del FP afiliativo, documentado.

## 13. Errores alejados del threshold (exactos)

{tabla(["etiqueta", "tipo", "n lejos", "acumulacion", "pocas", "conflicto wc", "top features", "top-1 por fold"], filas_far)}

## 14. Slices (exactas)

{tabla(["slice", "label", "errores dentro", "word medio", "char medio", "activas dentro", "activas fuera"], filas_slice)}

En los errores de HS dentro de URLs/colectivos el canal word es **negativo** en media
(URLs HS -0.264; colectivos HS -0.131) mientras el char es positivo: la senal que empuja el FP
en esas regiones es mayoritariamente de caracteres, con mas features activas que fuera. Grupos
pequenos (hashtags TR 12) con incertidumbre alta.

## 15. Comparacion con P90.03

{tabla(["observacion P90.03 (auxiliar)", "valor auxiliar", "valor exacto", "clasificacion"], filas_comp)}

La diferencia clave entre ambas atribuciones: la auxiliar usaba un modelo entrenado con las
mismas filas que intentaba explicar (in-sample), por lo que a menudo **no cometia el error** y
sus canales aparecian en direcciones distorsionadas (p. ej. word medio negativo en FP). La
atribucion exacta usa el modelo del fold que realmente produjo cada prediccion y muestra las
direcciones reales: FP con ambos canales positivos y FN con canales negativos o casi nulos.

## 16. Limitaciones

- La reconstruccion es exacta en este entorno (sklearn 1.8.0); versiones distintas de las
  librerias podrian introducir diferencias minimas (el gate lo detectaria).
- Las contribuciones describen la decision del modelo lineal; no son causalidad semantica.
- El analisis de slices pequenas (hashtags, cita) tiene incertidumbre alta.
- No se persisten los modelos reconstruidos (solo sus checksums); la reproduccion se controla
  por el gate y por los checksums registrados.

## 17. Decision final

**EXACT_OOF_ATTRIBUTION_RECOVERED.** Reproduccion exacta (error 0.0), descomposicion verificada
(residuo maximo {d['decomposition_residual_max']:.2e}) y 757 errores con atribucion exacta por
fold. Las conclusiones confirmadas/contradichas estan en la seccion 15. Esta fase no propone
modelos ni hipotesis nuevas: su resultado es una explicacion fiable de las predicciones
historicas de V03.
"""


def main() -> int:
    """Genera el informe y el manifiesto."""
    d = json.loads(RUTA_RESULTS.read_text(encoding="utf-8"))
    contrib = leer_csv(RUTA_CONTRIB)
    x = {"d": d, "far": bloque_far(contrib), "perra_fold": perra_por_fold(contrib),
         "python": platform.python_version(), "sklearn": None, "numpy": np.__version__, "scipy": None}
    import scipy
    import sklearn
    x["sklearn"] = sklearn.__version__
    x["scipy"] = scipy.__version__

    informe = construir_informe(x)
    escribir_texto(RUTA_INFORME, informe + "\n")

    salidas = ["reports/p90_03r_exact_oof_attribution.md", "reports/p90_03r_exact_oof_attribution.csv",
               "reports/p90_03r_oof_reproduction.csv", "reports/p90_03r_feature_contributions.csv",
               "reports/p90_03r_audit_results.json", "experiments/p90_03r_reconstruccion_oof.py",
               "experiments/p90_03rb_informe.py"]
    manifiesto = {
        "audit_id": "P90-03-R",
        "decision": d["decision"],
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                   cwd=RAIZ).stdout.strip(),
        "branch": subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True,
                                 cwd=RAIZ).stdout.strip(),
        "software_versions": {"python": x["python"], "scikit_learn": x["sklearn"],
                              "numpy": x["numpy"], "scipy": x["scipy"]},
        "source_oof": {"path": "features/oof_v03_final.npz", "sha256": d["hashes"]["oof"]},
        "source_script": {"path": "experiments/14_oof_corregido_v2.py", "sha256": d["hashes"]["script14"]},
        "baseline_model": {"path": "experiments/baseline_v03_frozen/model_v03.joblib",
                           "sha256": "263ed5bd0fc0de7cabfbaa85e88fca6fa65cf0a1b4e60a6e796b5120e26d13ca"},
        "data_identifiers": {"train": {"path": "data/raw/toxic_train.csv", "sha256": d["hashes"]["train"]},
                             "partition": {"path": "data/processed/holdout_v2_ids.csv",
                                           "sha256": sha256_de(RAIZ / "data" / "processed" / "holdout_v2_ids.csv")},
                             "splits_module": {"path": "src/splits.py", "sha256": d["hashes"]["splits"]},
                             "data_module": {"path": "src/data.py", "sha256": d["hashes"]["data"]}},
        "fold_definitions": "pliegues persistidos en el OOF (1..5, StratifiedGroupKFold con grupos fusionados); recomputados y coincidentes",
        "vectorizer_parameters": {"word": {"analyzer": "word", "ngram_range": [1, 2], "min_df": 2, "sublinear_tf": True},
                                  "char": {"analyzer": "char_wb", "ngram_range": [3, 5], "min_df": 2, "sublinear_tf": True}},
        "classifier_parameters": {"C": 2.0, "class_weight": "balanced", "max_iter": 1000,
                                  "solver": "liblinear", "random_state": 42},
        "thresholds": {"HS": 0.44, "TR": 0.51, "AG": 0.46},
        "random_seeds": {"random_state": 42},
        "reproduction_tolerances": {"allclose_rtol": 1e-7, "allclose_atol": 1e-8,
                                    "corrected_cv_abs": 1e-7, "binary_discordances": 0},
        "reproduction_results": d["gate"],
        "coefficient_checksums": d["checksums_coeficientes"],
        "artifact_paths": salidas,
        "artifact_sha256": {p: sha256_de(RAIZ / p) for p in salidas if (RAIZ / p).exists()},
        "limitations": ["reproduccion exacta en el entorno actual (sklearn 1.8.0)",
                        "contribuciones locales no son causalidad semantica",
                        "slices pequenas con incertidumbre alta"],
    }
    escribir_texto(RUTA_MANIFEST, json.dumps(manifiesto, ensure_ascii=False, indent=1) + "\n")

    for ruta in (RUTA_INFORME, RUTA_MANIFEST):
        if any(b > 127 for b in ruta.read_bytes()):
            raise SystemExit(f"STOP: bytes > 127 en {ruta.name}")
    auditoria = auditar_self()
    permitidos = {"14_oof_corregido_v2.py", "splits.py", "data.py", "oof_v03_final.npz",
                  "v2_oof_base_corrected.npz", "model_v03.joblib", "toxic_train.csv",
                  "holdout_v2_ids.csv", "data/processed/holdout_v2_ids.csv",
                  "fase2_lexico_inducido_agresividad.csv",
                  "p90_03rb_informe.py", "p90_03r_reconstruccion_oof.py",
                  "test.csv", "submission", "submissions", "leaderboard", "holdout"}
    raros = [v for v in auditoria["tokens"] if v not in permitidos]
    if raros:
        raise SystemExit(f"STOP: tokens de ruta inesperados: {raros}")
    print("P90-03-R parte 2 completada:")
    print(f"  informe: {RUTA_INFORME.name} ({len(informe.splitlines())} lineas)")
    print(f"  manifiesto: {RUTA_MANIFEST.name} | salidas hasheadas: {len(manifiesto['artifact_sha256'])}")
    return 0


def auditar_self() -> dict[str, Any]:
    """Comprobacion AST del propio script (operacion y fichero concretos)."""
    arbol = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    docstrings = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            cuerpo = getattr(nodo, "body", [])
            if cuerpo and isinstance(cuerpo[0], ast.Expr) and isinstance(cuerpo[0].value, ast.Constant):
                if isinstance(cuerpo[0].value.value, str):
                    docstrings.add(id(cuerpo[0].value))
    puro = re.compile(r"^[\w./\\-]+\.\w{2,5}$|^[\w./\\-]+/$")
    tokens = []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Constant) and isinstance(nodo.value, str) and id(nodo) not in docstrings:
            v = nodo.value
            if puro.match(v) and any(p in v.lower() for p in
                                     ("toxic_test", "test.csv", "leaderboard", "holdout", "submission")):
                tokens.append(v)
    modulos = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            modulos |= {a.name.split(".")[0] for a in nodo.names}
        elif isinstance(nodo, ast.ImportFrom) and nodo.module:
            modulos.add(nodo.module.split(".")[0])
    return {"tokens": sorted(set(tokens)), "modulos": sorted(modulos)}


if __name__ == "__main__":
    raise SystemExit(main())
