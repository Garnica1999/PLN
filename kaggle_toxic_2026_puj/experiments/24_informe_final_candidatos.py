"""FASE FINAL - Informes de screening y comparacion de candidatos finales.

Genera los dos informes de cierre de la fase de candidatos finales a partir de
evidencia en disco (sin fabricar cifras y sin generar submissions):

- ``reports/final_candidate_screening.md``: clasifica los 125 experimentos del
  registro en las categorias A-G de la spec (mejora standalone robusta, mejora
  pequena consistente, complementariedad fuerte, local no estable, negativo, no
  transferido y trazabilidad), con las filas relevantes una a una y el resto
  agregado por decision; incluye la tabla de canales con su evidencia real.
- ``reports/final_candidate_comparison.md``: tabla comparativa de los
  candidatos finales (corrected CV, CV historico, holdout, F1 por etiqueta,
  delta frente a V03, complementariedad, complejidad y evidencia).

Los numeros provienen de ``experiments.csv``, ``reports/final_candidates.csv``,
``reports/canal_ranking_final.csv``, ``reports/stacking_v2.csv``,
``reports/model_complementarity_v2.csv`` y
``reports/holdout_evaluaciones_v2.csv``.

Ejecucion
---------
    /home/carlos/venv/bin/python experiments/24_informe_final_candidatos.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Final

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENTS_CSV = PROJECT_ROOT / "experiments.csv"
FINAL_CANDIDATES_CSV = PROJECT_ROOT / "reports" / "final_candidates.csv"
RANKING_CANALES_CSV = PROJECT_ROOT / "reports" / "canal_ranking_final.csv"
STACKING_CSV = PROJECT_ROOT / "reports" / "stacking_v2.csv"
COMPLEMENTARIEDAD_CSV = PROJECT_ROOT / "reports" / "model_complementarity_v2.csv"
HOLDOUT_CSV = PROJECT_ROOT / "reports" / "holdout_evaluaciones_v2.csv"
SCREENING_MD = PROJECT_ROOT / "reports" / "final_candidate_screening.md"
COMPARISON_MD = PROJECT_ROOT / "reports" / "final_candidate_comparison.md"

CATEGORIAS: Final[dict[str, str]] = {
    "A": "A - Mejora standalone robusta",
    "B": "B - Mejora pequena consistente",
    "C": "C - Complementariedad fuerte",
    "D": "D - Local no estable",
    "E": "E - Negativo",
    "F": "F - No transferido al leaderboard",
    "G": "G - Trazabilidad / sin modelo nuevo",
}

# Clasificacion curada de las filas relevantes (evidencia en DECISIONS.md,
# reports/advanced_research_v2.md, reports/ensembles_resultados.csv,
# reports/features_densas_resultados.csv y reports/backlog_restante_v2.csv).
CLASIFICACION: Final[dict[str, list[str]]] = {
    "A": ["E2-02-META", "E2-18-META-BASES-RIDGE", "FIN-01-C-REF-44", "TH-01", "S03"],
    "B": [
        "MOD-03b_BEST",
        "MOD-03b_W12",
        "MOD-01_BEST",
        "MOD-03a_BEST",
        "REP-05",
        "REP-06",
        "FIN-02-C-EMO",
        "FIN-04-C-EMO-SYN",
        "FIN-05-C-SYN",
    ],
    "C": ["E2-18-META-BASES", "FIN-06-C-REF-44-CH", "ENS-02"],
    "D": [
        "E2-01-HS-TR-AG",
        "E2-01-HS-TR-AG-STD",
        "E2-01-HS-AG-TR",
        "E2-01-HS-AG-TR-STD",
        "E2-01-AG-HS-TR",
        "E2-01-AG-HS-TR-STD",
        "E2-13-META-ENT-RIDGE",
        "E2-13-META-ENT-LR",
        "E2-14-MEJOR-META",
        "E2-18-META+CH",
        "E2-18-META+CH+ESTR",
        "E2-18-META-BASES-ANID",
        "E2-21-TR",
        "DL-01",
        "PRE-03",
        "PRE-05",
        "PRE-06",
        "FIN-03-C-EMO-margenes",
    ],
    "E": [
        "PRE-01",
        "PRE-02",
        "PRE-04",
        "PRE-07",
        "PRE-08",
        "PRE-09",
        "REP-01",
        "REP-02",
        "REP-03",
        "REP-04",
        "MOD-01_W12",
        "MOD-02_BEST",
        "MOD-02_W12",
        "MOD-03a_W12",
        "MOD-04",
        "ENS-01",
        "ENS-03",
        "ENS-04",
        "HIB-H12",
        "HIB-H11",
        "HIB-H14",
        "HIB-UNION",
        "B5-DENSE",
        "DL-02",
        "DL-03",
        "E2-03-POS",
        "E2-03-POS-TXT",
        "E2-04-DEPS",
        "E2-04-MORPH",
        "E2-04-SYN3",
        "E2-04-DEPS-TXT",
        "E2-04-MORPH-TXT",
        "E2-04-SYN3-TXT",
        "E2-05-CH-TGT",
        "E2-05-CH-TGT-DIST",
        "E2-06-TXT-CH",
        "E2-06-TXT-TGT-SPARSE",
        "E2-07-CITA",
        "E2-07-CITA-TXT",
        "E2-07-CITA-NEG-TXT",
        "E2-08-NEG",
        "E2-08-NEG-TXT",
        "E2-09-EMO-EST",
        "E2-09-EMO-LEX",
        "E2-09-EMO",
        "E2-09-TXT-EMO",
        "E2-15-NB-WORD",
        "E2-15-NB-CHAR",
        "E2-15-NBSVM",
        "E2-15-NBSVM-SVC",
        "E2-10-ORT-A",
        "E2-10-ORT-ABCD",
        "E2-10-ORT-AE",
        "E2-13-ENT-SOLO-LR",
        "E2-16-LOCAL-SIN-SHRINKAGE",
        "E2-16-SHRINK-M25",
        "E2-16-SHRINK-M50",
        "E2-16-SHRINK-M100",
        "E2-16-MEJOR",
        "E2-20-ESPECIALISTAS",
        "E2-21-HS",
        "E2-21-AG",
        "E2-21-ESPACIOS",
        "E2-18-META+CH+ESTR-RIDGE",
        "E2-18-META+CH+ESTR-SVC",
        "E2-18-META+CH+ESTR-ANID",
        "FIN-07-C-EMO-SYN-TGT-CYT",
    ],
    "F": ["S04", "S05", "S06", "S07", "S08", "R1-C1", "R2-MINDF5", "OPT-01"],
    "G": [
        "E00",
        "B1",
        "PRE-00",
        "S02",
        "HOLD-01",
        "V2-BASE-M1_word_lr",
        "V2-BASE-M2_union_lr_c2",
        "V2-BASE-M3_union_svc",
        "V2-BASE-M4_char_lr",
        "AUD-01",
        "AUD-02",
        "AUD-03",
    ],
}

RAZONES: Final[dict[str, str]] = {
    "A": "Adoptado con la regla vigente (nivel +0.005 y >= 3/5 pliegues o complementariedad con IC > 0).",
    "B": "Delta positivo con 3/5 pliegues; por debajo del umbral de adopcion de nivel.",
    "C": "Sin mejora standalone suficiente pero con ganancia de fusion e IC bootstrap > 0.",
    "D": "Delta positivo o ambiguo sin consistencia entre pliegues ni IC bootstrap favorable.",
    "E": "Delta negativo claro frente a su referencia; linea cerrada.",
    "F": "Mejora local o sonda pre-registrada que no transfirio al leaderboard.",
    "G": "Referencia, control, artefacto de auditoria o protocolo; sin modelo nuevo.",
}

# Notas especificas por experimento (evidencia concreta para el informe).
NOTAS: Final[dict[str, str]] = {
    "E2-02-META": "0.7853 (+0.0081 vs V03), 4/5 pliegues, IC [+0.0021,+0.0248]; D-017",
    "E2-18-META-BASES-RIDGE": "0.7874, 4/5 pliegues; ensemble_gain +0.0089 IC [+0.0019,+0.0160]; D-019",
    "FIN-01-C-REF-44": "Reproduce 0.7874 exacto con el OOF congelado (control de comparabilidad)",
    "TH-01": "Umbrales por etiqueta sobre el OOF (0.7989 historico): base de la decision D-012",
    "S03": "Unica mejora confirmada en el LB de forma pareada (+0.0031: v02 0.77993 -> v03 0.78306)",
    "MOD-03b_BEST": "Mejor de la familia union historica (0.7927); D-014: la banda 0.7703-0.7831 es 1 error estandar",
    "MOD-03b_W12": "Formulacion W12 (0.5) del adoptado; superada por TH-01",
    "MOD-01_BEST": "Mejor variante de MOD-01 (0.7849); paso previo a la union",
    "MOD-03a_BEST": "0.7723; paso intermedio de la familia union",
    "REP-05": "Union word+char adoptada (0.7865) en el protocolo historico",
    "REP-06": "Variante de union adoptada (0.7855); diferencia con REP-05 dentro del ruido",
    "FIN-02-C-EMO": "0.7809 (+0.0037, 3/5 pliegues); holdout 0.8364 (+0.0081 pareado, IC incluye 0)",
    "FIN-04-C-EMO-SYN": "0.7797 (+0.0025, 3/5); holdout 0.8336 (+0.0052 pareado, IC incluye 0)",
    "FIN-05-C-SYN": "0.7793 (+0.0021, 3/5 pliegues)",
    "E2-18-META-BASES": "0.7850; ensemble_gain +0.0154 IC > 0 (candidato v2-A por complementariedad, D-017)",
    "FIN-06-C-REF-44-CH": "0.7782 (-0.0092 vs el meta de 44 columnas); fusion con V03 +0.0096 IC [+0.0022,+0.0172]",
    "ENS-02": "Promedio M2+M3: 0.7949 anidado, ensemble_gain +0.0023; no adoptado por nivel",
    "E2-01-HS-TR-AG": "Cadenas de etiquetas: +0.0022, 1/5 pliegues; cerrada (E2-01)",
    "E2-13-META-ENT-RIDGE": "0.7885 por encima del meta pero el encoder de entidades cubre 0.5 % del corpus (E2-13)",
    "E2-14-MEJOR-META": "Calibracion: Brier 0.1583 -> 0.1199 y log-loss 0.5002 -> 0.4035, F1 invariante (E2-14)",
    "E2-18-META+CH": "0.7807; anadir los canales de targeting al meta no aporta (delta -0.0046 vs v2-A, D-019)",
    "E2-18-META+CH+ESTR": "0.7844; 2/5 pliegues vs v2-A",
    "E2-18-META-BASES-ANID": "0.7853: reproduce el meta anidado de P1 con 0/5 pliegues mejorados (empate)",
    "E2-21-TR": "0.7934 con espacio propio de TR, pero el conjunto por etiqueta cae a 0.7519 (E2-21)",
    "DL-01": "PCA de la union a 300 componentes: 0.7949 en OOF; nunca se probo en el LB",
    "FIN-03-C-EMO-margenes": "C-EMO + margen/entropia de V03: 0.7798 (+0.0026 pero 2/5 pliegues)",
    "E2-16-MEJOR": "Umbrales por rebanada con shrinkage: 0.7750, por debajo del umbral global anidado (E2-16)",
    "E2-20-ESPECIALISTAS": "11 de 21 celdas significativas pero los especialistas rinden -0.045 dentro de su rebanada",
    "E2-21-ESPACIOS": "Espacios por etiqueta: -0.0253 (0/5 pliegues)",
    "E2-15-NBSVM": "0.7612 standalone; su correccion mutua frente a V03 es alta pero no se adopto como canal",
    "S08": "Meta v08: holdout 0.8465 (delta +0.0180) pero LB 0.78163 < 0.78306; no transfiere (D-020)",
    "S04": "Ensemble ENS-04 con umbrales medianos: LB 0.77510; el umbral fue el sospechoso confirmado",
    "S05": "OPT-01 (Optuna): LB 0.77602",
    "S06": "Sonda de robustez C=1.0: LB 0.77227",
    "S07": "Sonda de robustez min_df=5: LB 0.77034",
    "R1-C1": "Sonda pre-registrada D-013 (C=1.0): OOF 0.7929, LB 0.77227",
    "R2-MINDF5": "Sonda pre-registrada D-013 (min_df=5): OOF 0.7993, LB 0.77034",
    "OPT-01": "Mejor trial de Optuna (0.8012 historico) adoptado localmente; su sonda S05 no transfirio",
    "E00": "Sonda de metrica: LB 0.51494 con la plantilla todo-1 (identifica F1 medio por columna)",
    "B1": "Baseline TF-IDF word(1,2)+LR: 0.7578 local / 0.75677 LB",
    "PRE-00": "Referencia de la familia de preprocesamiento (texto crudo, 0.7587)",
    "S02": "Refit con todo el train: reproduce el OOF agrupado (control)",
    "HOLD-01": "Holdout congelado de 248 tweets; maximo 5 evaluaciones (4 usadas)",
    "AUD-01": "Artefacto de auditoria: 101/1731 tweets fallan las 3 etiquetas (5.83 %)",
    "AUD-02": "Artefacto de auditoria por combinacion verdadera (FASE 47)",
    "AUD-03": "Tabla de complementariedad entre 29 canales OOF (FASE 50/51)",
}


# Notas generadas para las filas que no tienen curacion manual.
NOTAS_AUTO: dict[str, str] = {}


def log(mensaje: str) -> None:
    """Imprime un mensaje de progreso.

    Args:
        mensaje: texto a imprimir.
    """
    print(f"[FIN-INF] {mensaje}", flush=True)


def clasificar(experimentos: pd.DataFrame) -> dict[str, list[str]]:
    """Asigna categoria a cada experimento (curada + regla de respaldo).

    Args:
        experimentos: registro de experimentos.

    Returns:
        Mapa categoria -> ids asignados.
    """
    asignados: dict[str, str] = {}
    for categoria, identificadores in CLASIFICACION.items():
        for identificador in identificadores:
            if identificador in asignados:
                raise RuntimeError(f"{identificador} clasificado dos veces")
            asignados[identificador] = categoria
    pendientes = [
        str(identificador)
        for identificador in experimentos["experiment_id"]
        if str(identificador) not in asignados
    ]
    for identificador in pendientes:
        fila = experimentos.loc[experimentos["experiment_id"] == identificador].iloc[0]
        decision = str(fila["decision"]).lower()
        if "adoptad" in decision and "no adoptad" not in decision:
            asignados[identificador] = "B"
        elif "descartad" in decision or "no adoptad" in decision:
            asignados[identificador] = "E"
        else:
            asignados[identificador] = "G"
        NOTAS_AUTO[identificador] = "Clasificacion automatica por el texto de la decision (fila secundaria)"
    resultado: dict[str, list[str]] = {categoria: [] for categoria in CATEGORIAS}
    for identificador, categoria in asignados.items():
        resultado[categoria].append(identificador)
    for categoria, ids in resultado.items():
        ids.sort()
    return resultado


def tabla_experimentos(
    experimentos: pd.DataFrame, ids: list[str], categoria: str
) -> str:
    """Construye la tabla markdown de un grupo de experimentos.

    Args:
        experimentos: registro completo.
        ids: identificadores a listar.
        categoria: categoria A-G (para la nota por defecto).

    Returns:
        Fragmento markdown con la tabla.
    """
    lineas = [
        "| experiment_id | f1 | decision | hipotesis (resumen) | nota |",
        "|---|---|---|---|---|",
    ]
    for identificador in ids:
        fila = experimentos.loc[experimentos["experiment_id"] == identificador]
        if fila.empty:
            lineas.append(f"| {identificador} | - | (no esta en experiments.csv) | - | - |")
            continue
        registro = fila.iloc[0]
        f1 = f"{registro['f1']:.4f}" if pd.notna(registro["f1"]) else "-"
        nota = NOTAS.get(str(identificador), NOTAS_AUTO.get(str(identificador), RAZONES[categoria]))
        hipotesis = str(registro["hypothesis"]).replace("|", "/")[:110]
        decision = str(registro["decision"]).replace("|", "/")[:70]
        lineas.append(f"| {identificador} | {f1} | {decision} | {hipotesis} | {nota} |")
    return "\n".join(lineas)


def tabla_canales() -> str:
    """Construye la tabla de canales con evidencia real.

    Returns:
        Fragmento markdown con la tabla de canales puros y su complementariedad.
    """
    ranking = pd.read_csv(RANKING_CANALES_CSV)
    pureza = ranking[ranking["puro"]]
    complementariedad = pd.read_csv(COMPLEMENTARIEDAD_CSV)
    filas = [
        "| canal | familia | F1 standalone (0.5) | F1 HS | F1 TR | F1 AG | desacuerdo vs V03 | correlacion de errores | correccion mutua (V03 correcto si canal falla) | decision |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    seleccionados = {
        "EMO-ALL@C=1.0": "emocion",
        "SYN-3": "sintaxis",
        "CH-TGT+DIST": "targeting",
        "CITA": "cita_negacion",
    }
    for variante, familia in seleccionados.items():
        fila = pureza[(pureza["familia"] == familia) & (pureza["variante"] == variante)]
        if fila.empty:
            continue
        registro = fila.iloc[0]
        par = complementariedad[
            (complementariedad["canal_a"].astype(str).str.endswith(variante, na=False))
            & (complementariedad["canal_b"] == "v03_base_M2_union_lr_c2")
        ]
        if par.empty:
            par = complementariedad[
                (complementariedad["canal_b"].astype(str).str.endswith(variante, na=False))
                & (complementariedad["canal_a"] == "v03_base_M2_union_lr_c2")
            ]
        if par.empty:
            desacuerdo = correlacion = mutua = "-"
        else:
            desacuerdo = f"{float(par.iloc[0]['tasa_desacuerdo']):.4f}"
            correlacion = f"{float(par.iloc[0]['correlacion_errores']):.4f}"
            columna = (
                "tasa_correccion_mutua_a_sobre_b"
                if str(par.iloc[0]["canal_b"]) == "v03_base_M2_union_lr_c2"
                else "tasa_correccion_mutua_b_sobre_a"
            )
            mutua = f"{float(par.iloc[0][columna]):.4f}"
        filas.append(
            
                f"| {variante} | {familia} | {registro['f1_standalone_05']:.4f} | "
                f"{registro['f1_HS']:.4f} | {registro['f1_TR']:.4f} | {registro['f1_AG']:.4f} | "
                f"{desacuerdo} | {correlacion} | {mutua} | canal puro usado por los candidatos FIN |"
            
        )
    return "\n".join(filas)


def escribir_screening(experimentos: pd.DataFrame, categorias: dict[str, list[str]]) -> None:
    """Escribe ``reports/final_candidate_screening.md``.

    Args:
        experimentos: registro completo de experimentos.
        categorias: mapa categoria -> ids.
    """
    total = len(experimentos)
    nuevas = int(sum(1 for identificador in experimentos["experiment_id"] if str(identificador).startswith("FIN-")))
    partes = [
        "# Screening de candidatos finales (FASE FINAL)",
        "",
        (
            f"Fecha: 2026-09-25. Registro: `experiments.csv` con {total} filas "
            f"({total - nuevas} previas + {nuevas} filas `FIN-*` de esta fase)."
        ),
        "",
        "Objetivo: decidir si alguna linea de la investigacion v2 justifica cambiar",
        "el modelo final V03 (LB 0.78306) o gastar sondas de leaderboard. Las",
        "categorias son: A mejora standalone robusta; B mejora pequena consistente;",
        "C complementariedad fuerte; D local no estable; E negativo; F no",
        "transferido al leaderboard; G trazabilidad (referencia, control u",
        "artefacto de auditoria).",
        "",
        "## Conteo por categoria",
        "",
        "| categoria | n |",
        "|---|---|",
    ]
    for categoria, etiqueta in CATEGORIAS.items():
        partes.append(f"| {etiqueta} | {len(categorias[categoria])} |")
    partes.append(f"| **Total** | **{total}** |")

    for categoria, etiqueta in CATEGORIAS.items():
        ids = categorias[categoria]
        partes.extend(["", f"## {etiqueta} ({len(ids)})", "", RAZONES[categoria], ""])
        partes.append(tabla_experimentos(experimentos, ids, categoria))

    partes.extend(
        [
            "",
            "## Tabla de canales con evidencia real",
            "",
            "Canales puros (sin la representacion de V03) del store OOF corregido;",
            "F1 standalone a umbral 0.5 sobre el desarrollo (n=1483) y metricas de",
            "complementariedad frente a V03 (M2) de `reports/model_complementarity_v2.csv`.",
            "",
            tabla_canales(),
            "",
            "## Lectura del screening",
            "",
            "1. La unica mejora adoptada con evidencia del protocolo corregido sigue",
            "   siendo el meta de dependencia (E2-02-META, D-017) y su version Ridge",
            "   de 44 columnas (E2-18-META-BASES-RIDGE, D-019). El control `FIN-01-`",
            "   reproduce 0.7874 exacto, lo que valida la comparabilidad de esta fase.",
            "2. Todos los canales linguisticos nuevos (emocion, sintaxis, cita/negacion,",
            "   ortografia, NB-SVM) son negativos como modelo standalone (categoria E):",
            "   su F1 aislado va de 0.527 (NEG) a 0.766 (TXT+NEG) frente a 0.7772 de V03.",
            "3. Los canales aportan diversidad (desacuerdo 10-36 %, correlacion de errores",
            "   0.15-0.65) pero su fusion directa con V03 por media de probabilidades es",
            "   negativa; solo la fusion aprendida (meta) recupera senal, y anadir canales",
            "   al meta de 44 columnas DEGRADA el nivel (-0.0092, `FIN-06-`).",
            "4. Las sondas de leaderboard (categoria F) muestran un plateau: 0.77034-0.78306",
            "   en 8 envios, un rango de ~1 error estandar (n=1154). El holdout congelado",
            "   favorecio al meta v08 (+0.0180) y aun asi el LB no lo confirmo (D-020).",
            "",
            "## Honestidad sobre los limites",
            "",
            "- Las filas `FIN-0*` comparten el mismo protocolo corregido; su diferencia",
            "  con la referencia V03 (maximo +0.0037 de `FIN-02-C-EMO`) esta por debajo",
            "  del umbral de adopcion (+0.005 y >= 3/5 pliegues).",
            "- Los contrastes de esta fase son 6 candidatos x 6 configuraciones; el",
            "  maximo observado esta sesgado al alza por seleccion (winner's curse).",
            "- Ninguna cifra de este informe es una estimacion del LB: el unico contraste",
            "  pareado fiable con el LB es S03 (+0.0031) y el fracaso de las sondas v04-v08.",
        ]
    )
    SCREENING_MD.write_text("\n".join(partes) + "\n", encoding="utf-8")
    log(f"escrito {SCREENING_MD.name} ({len(partes)} lineas)")


def fila_holdout(candidatos: pd.DataFrame, holdout: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Extrae la evidencia del holdout por experimento registrado.

    Args:
        candidatos: tabla de candidatos finales.
        holdout: registro de evaluaciones del holdout.

    Returns:
        Mapa experiment_id -> datos del holdout.
    """
    resultado: dict[str, dict[str, Any]] = {}
    for _, fila in holdout.iterrows():
        experimento = str(fila["experimento"])
        for identificador in candidatos["experiment_id"].astype(str):
            if experimento.startswith(identificador):
                resultado[identificador] = json.loads(str(fila["resultado"]))
    return resultado


def escribir_comparison(candidatos: pd.DataFrame, holdout: pd.DataFrame) -> None:
    """Escribe ``reports/final_candidate_comparison.md`` (borrador, sin submissions).

    Args:
        candidatos: tabla de candidatos finales.
        holdout: registro de evaluaciones del holdout.
    """
    evidencia = fila_holdout(candidatos, holdout)
    experimentos = pd.read_csv(EXPERIMENTS_CSV)
    dimensiones = {
        str(fila["experiment_id"]): fila["dimension"] for _, fila in experimentos.iterrows()
    }
    partes = [
        "# Comparacion de candidatos finales (borrador, sin submissions)",
        "",
        "Fecha: 2026-09-25. Fuente: `reports/final_candidates.csv` y",
        "`reports/holdout_evaluaciones_v2.csv` (4/5 evaluaciones usadas: 2 de P7 y",
        "2 de esta fase). No se genera ninguna submission en esta tarea.",
        "",
        "## Tabla comparativa",
        "",
        "| candidato | corrected CV | CV historico | holdout | F1 HS/TR/AG (holdout) | delta vs V03 (dev) | complementariedad | complejidad | evidencia |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, fila in candidatos.iterrows():
        identificador = str(fila["experiment_id"])
        datos = evidencia.get(identificador)
        holdout_texto = str(fila["holdout"])
        etiquetas = "-"
        if datos:
            etiquetas = "/".join(
                f"{clave} {valor:.3f}" for clave, valor in datos["holdout_f1_por_etiqueta"].items()
            )
        delta = float(fila["delta_vs_v03"])
        complementariedad = str(fila["complementarity"]).replace("|", "/")
        dimension = dimensiones.get(identificador)
        if dimension is not None and pd.notna(dimension):
            complejidad = f"{int(dimension)} features + 3 modelos binarios"
        elif identificador == "V03":
            complejidad = "3 modelos binarios (M2 union, 24.646 features TF-IDF)"
        elif identificador == "META-BASES-RIDGE":
            complejidad = "44 features + 3 modelos binarios"
        else:
            complejidad = "3 modelos binarios por etiqueta (grid Ridge/LR)"
        partes.append(
            f"| {identificador} | {float(fila['corrected_cv']):.4f} | {fila['historical_cv']} | "
            f"{holdout_texto} | {etiquetas} | {delta:+.4f} | {complementariedad} | {complejidad} | "
            f"{fila['decision']!s} |"
        )
    partes.extend(
        [
            "",
            "## Criterios de seleccion",
            "",
            "- PROMISING: corrected CV >= 0.7772 + 0.005 = 0.7822 y mejora en >= 3/5",
            "  pliegues frente a V03. Solo lo cumple `FIN-01-C-REF-44`, que es la",
            "  reproduccion del meta v08 adoptado en P6 (ya evaluado en P7-02).",
            "- COMPLEMENTARY: nivel <= referencia pero fusion con V03 con",
            "  `ensemble_gain >= 0.0015` e IC bootstrap > 0. Lo cumple",
            "  `FIN-06-C-REF-44-CH`, pero su fusion alcanza 0.7814 en el desarrollo,",
            "  por debajo del meta v08 (0.7874), y el meta ya no transfirio al LB.",
            "- El resto queda como evidencia negativa documentada.",
            "",
            "## Decision del screening (sin submissions)",
            "",
            "- **No hay candidato nuevo PROMISING**: el mejor candidato nuevo sobre",
            "  probabilidades de V03 + canal es `FIN-02-C-EMO` (0.7809 en el desarrollo",
            "  y 0.8364 en el holdout) frente a 0.7874 / 0.8465 del meta v08.",
            "- Ninguno de los candidatos nuevos alcanza el nivel del meta adoptado y el",
            "  meta adoptado ya fallo en el LB (D-020): se mantiene **V03** como modelo",
            "  final y se recomienda **no gastar sondas** con los candidatos de canales.",
            "- Las dos evaluaciones nuevas del holdout (`FIN-02-C-EMO`, `FIN-04-C-EMO-SYN`)",
            "  quedan registradas; el holdout conserva 1 evaluacion libre (4/5 usadas).",
            "  Regla de seleccion de finalistas: se eligio el mejor candidato de cada",
            "  conjunto de canales distinto para no gastar dos evaluaciones en variantes",
            "  del mismo candidato (por eso `FIN-03-C-EMO-margenes`, variante de `FIN-02`",
            "  con +6 features, no se evaluo en el holdout).",
        ]
    )
    COMPARISON_MD.write_text("\n".join(partes) + "\n", encoding="utf-8")
    log(f"escrito {COMPARISON_MD.name} ({len(partes)} lineas)")


def main() -> None:
    """Genera los informes de screening y comparacion."""
    experimentos = pd.read_csv(EXPERIMENTS_CSV)
    candidatos = pd.read_csv(FINAL_CANDIDATES_CSV)
    holdout = pd.read_csv(HOLDOUT_CSV)
    categorias = clasificar(experimentos)
    escribir_screening(experimentos, categorias)
    escribir_comparison(candidatos, holdout)
    conteo = ", ".join(f"{categoria}={len(categorias[categoria])}" for categoria in CATEGORIAS)
    log(f"conteo del screening: {conteo}; total {sum(len(ids) for ids in categorias.values())}")


if __name__ == "__main__":
    main()
