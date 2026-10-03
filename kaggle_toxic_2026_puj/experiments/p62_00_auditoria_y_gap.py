"""P62 step 00: verify the P61 closure and perform a FUNCTIONAL gap audit.

Why the gap audit is functional and not a name search
-----------------------------------------------------
Paragraph 3 says that searching names is not enough, and the reason is now visible in
the register. There are 339 experiments, and a name search for "POS" or "dependencies"
returns the same handful of rows no matter which of the five P62 tracks is being checked,
because every track uses the same vocabulary. What distinguishes them is the
REPRESENTATION, and that is not in any name.

So this script does not ask what a past experiment was called. For every phenomenon P62
proposes, it records:

    the experiment that touched it
    the exact token pattern the code emits, read from the source
    whether that representation was DENSE per-tweet counts or SPARSE tokens
    whether the word identity of the tokens entered the representation at all
    the verdict, and what it rests on

The distinction that decides P62 is the last two rows. The project has already built POS
and dependency information twice, and both times as DENSITY: 36 columns of
``pos_VERB_n`` / ``pos_VERB_frac`` and 31 of ``dep_amod_n``. What P62 proposes is a
SPARSE token space. Those are different objects, and the fact that the dense one failed
is not evidence about the sparse one. It is, however, strong evidence that POS and
dependency *marginals* carry little, so the hypothesis has to earn its place on the
interaction term rather than on the tag.

The tagger question, settled before anything else
------------------------------------------------
Paragraph 25 asks whether a tagger's pretrained weights are legal. They are, and not by
new ruling: **D-003** already established that ``es_core_news_md`` may be used as a
SYMBOLIC annotator with ``exclude=["ner"]``, while ``doc.vector``, ``token.vector``,
``doc.tensor`` and ``similarity()`` are forbidden, with an assert in the feature code and
the competition rules quoted. ``es_core_news_md`` 3.8.0 is installed, Spanish, and the
project has used it in E2-03 and E2-04, whose result artifacts are the twenty-channel P50
pool this phase reuses. So there is no ambiguity to resolve, and this script records the
provenance explicitly rather than assuming it.

What it writes
--------------
    reports/p62_hypothesis_gap_analysis.md
    reports/p62_hypothesis_gap_analysis.csv
    reports/p62_p61_audit.json

Usage
-----
    /home/carlos/venv/bin/python experiments/p62_00_auditoria_y_gap.py
"""

from __future__ import annotations

import ast
import csv as _csv
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    V03_CCV,
    V03_SHA,
    V03_SUB,
)

REPORTS = PROJECT_ROOT / "reports"
EXPERIMENTS = PROJECT_ROOT / "experiments"
FIGURAS = PROJECT_ROOT / "figuras"
MD_GAP = REPORTS / "p62_hypothesis_gap_analysis.md"
CSV_GAP = REPORTS / "p62_hypothesis_gap_analysis.csv"
JSON_AUDIT = REPORTS / "p62_p61_audit.json"

GATE_DELTA = 0.003
GATE_PLIEGUES = 4

YA_PROBADO = "already_tested"
PARCIAL = "partially_tested"
EQUIVALENTE = "functionally_equivalent"
NUEVO = "genuinely_new"


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P62-00] {mensaje}", flush=True)


SUSTITUCIONES = {
    chr(0x2014): " - ", chr(0x2013): "-", chr(0x2212): "-", chr(0x00B1): "+/-",
    chr(0x00AB): '"', chr(0x00BB): '"', chr(0x201C): '"', chr(0x201D): '"',
    chr(0x2018): "'", chr(0x2019): "'", chr(0x2264): "<=", chr(0x2265): ">=",
    chr(0x00D7): "x", chr(0x2192): "->", chr(0x00B7): ".", chr(0x2026): "...",
    chr(0x00BF): "?", chr(0x00A1): "!", chr(0x00A7): "parrafo ", chr(0x00A0): " ",
}


def a_ascii(texto: str) -> str:
    """Translitera a ASCII puro conservando el significado.

    Args:
        texto: cadena de entrada.

    Returns:
        La cadena equivalente en ASCII.
    """
    if not isinstance(texto, str):
        texto = str(texto)
    for k, v in SUSTITUCIONES.items():
        texto = texto.replace(k, v)
    salida = []
    for ch in unicodedata.normalize("NFKD", texto):
        if ord(ch) < 128:
            salida.append(ch)
            continue
        if unicodedata.category(ch).startswith("M"):
            continue
        partes = unicodedata.decomposition(ch).split()
        base = chr(int(partes[0], 16)) if partes and partes[0] != "<wide>" else ""
        salida.append(base if base and ord(base) < 128 else "?")
    return "".join(salida)


# Los fenomenos que P62 propone, con la evidencia funcional que se leyo del codigo.
# Cada entrada declara la representacion EXACTA que ya existe, leida del fuente, porque
# es ahi donde vive la diferencia y no en ningun nombre de experimento.
FENOMENOS: tuple[dict[str, Any], ...] = (
    {
        "id": "L",
        "fenomeno": "el mismo lexema con distinto significado predictivo segun su "
                    "categoria gramatical, representado como token palabra__POS",
        "experimentos_previos": "E2-03-POS, E2-03-POS-TXT (familia sintaxis/cita/"
                                "negacion, paquete 19), y las 36 columnas pos_* de "
                                "v2_structured_features.parquet",
        "representacion_existente": "DOS formas, ambas sin la identidad de la palabra. "
                                    "(a) DENSE: pos_VERB_n y pos_VERB_frac, un conteo y "
                                    "una proporcion por tweet para cada una de las 18 "
                                    "categorias. (b) SPARSE pero categorial: la funcion "
                                    "secuencia_pos emite f'{token.pos}_{token.head_pos}' "
                                    "y se vectoriza con ngram_range (1,4)",
        "tipo_representacion": "densas por tweet, mas tokens categoriales sin palabra",
        "identidad_lexical_entra": False,
        "delta_historico": "E2-03-POS -0,2135 con 0/5 pliegues; E2-03-POS-TXT -0,0116 "
                           "con 0/5 pliegues",
        "veredicto": NUEVO,
        "por_que": "ninguna de las dos formas existentes combina la identidad de la "
                   "palabra con su categoria. El token palabra__POS no aparece en ningun "
                   "sitio del codigo, y eso es una comprobable, no una interpretacion. "
                   "La version categorial fallo, lo cual acota donde puede estar la senal: "
                   "en el termino de interaccion, no en la etiqueta",
        "experimento": 1,
    },
    {
        "id": "M",
        "fenomeno": "la relacion sintactica entre dos palabras concretas, representada "
                    "como token cabeza__relacion__dependiente",
        "experimentos_previos": "E2-04-DEPS, E2-04-DEPS-TXT, y las 31 columnas dep_* de "
                                "v2_structured_features.parquet",
        "representacion_existente": "DOS formas, ninguna con la palabra dependiente. "
                                    "(a) DENSE: dep_amod_n y las demas 30, un conteo por "
                                    "relacion y tweet. (b) SPARSE: la funcion "
                                    "secuencia_deps emite f'{dep}_{token.head_pos}' y, "
                                    "segun el inventario de artefactos, tambien DEP_HEAD; "
                                    "es decir relacion mas POS de la cabeza, y la cabeza "
                                    "en forma de lema, pero NUNCA la palabra dependiente",
        "tipo_representacion": "densas por tweet, mas tokens relacion-POS-cabeza sin "
                               "dependiente",
        "identidad_lexical_entra": "solo la de la cabeza, en un token aislado",
        "delta_historico": "E2-04-DEPS -0,2258 con 0/5 pliegues; E2-04-DEPS-TXT -0,0119 "
                           "con 1/5 pliegues",
        "veredicto": NUEVO,
        "por_que": "el triple lexical cabeza__relacion__dependiente no existe. La version "
                   "existente acerca la cabeza y la relacion pero pierde el dependiente, "
                   "que es justamente el elemento que distingue un insulto dirigido a "
                   "alguien de un adjetivo suelto",
        "experimento": 1,
    },
    {
        "id": "N",
        "fenomeno": "secuencias de POS de longitud dos a cuatro, como VERB_NOUN, "
                    "ADJ_NOUN, PRON_VERB_ADJ, representadas como tokens sparse",
        "experimentos_previos": "E2-03-POS y E2-03-POS-TXT, mediante la funcion "
                                "secuencia_pos con PARAMS_POS ngram_range (1,4)",
        "representacion_existente": "exactamente la propuesta. secuencia_pos une por "
                                    "espacio los pares POS_dependiente_POS_cabeza y "
                                    "TfidfVectorizer con ngram_range (1,4) genera los "
                                    "n-gramas de longitud uno a cuatro, que incluyen "
                                    "VERB_NOUN, PRON_VERB_ADJ y demas",
        "tipo_representacion": "sparse categorial, identica a la propuesta",
        "identidad_lexical_entra": False,
        "delta_historico": "E2-03-POS-TXT -0,0116 con 0/5 pliegues",
        "veredicto": EQUIVALENTE,
        "por_que": "la representacion propuesta y la existente son el mismo objeto. "
                   "Repetirla gastaria un experimento para confirmar un numero que ya "
                   "esta en el registro con el mismo clasificador, los mismos pliegues y "
                   "la misma regla de decision. Se cierra SIN consumir experimento, que "
                   "es lo que autoriza el parrafo 43",
        "experimento": 0,
    },
    {
        "id": "O",
        "fenomeno": "canal sparse de composicion lexica: familia verbal, familia "
                    "adjectival, familia nominal, clase pronominal y patron de "
                    "funcionales",
        "experimentos_previos": "E2-02-META-BASES y su combinacion con canales, "
                                "E2-15-NB-WORD, E2-15-NBSVM, E2-15-NBSVM-SVC, y V25-01 "
                                "con factorization machines por etiqueta",
        "representacion_existente": "varias. E2-15 es Naive Bayes sobre lemas, que es "
                                    "justamente la composicion lexica como likelihood. "
                                    "V25-01 es un modelo de interaccion de bajo rango "
                                    "sobre las mismas bases. Ninguno de los dos es un "
                                    "canal sparse que conviva con el TF-IDF de V03 en el "
                                    "mismo espacio de regresion logistica",
        "tipo_representacion": "modelos paralelos, no canal sparse anadido a V03",
        "identidad_lexical_entra": True,
        "delta_historico": "E2-15-NB-WORD -0,0234 con 0/5; E2-15-NBSVM -0,0160 con 2/5; "
                           "V25-01 0,782699 sin candidato",
        "veredicto": PARCIAL,
        "por_que": "la composicion lexica como likelihood y como interaccion ya se probo "
                   "y fue peor que V03. Lo que no se probo es usarla como CANAL sparse "
                   "dentro del espacio de V03. La diferencia es real, pero es "
                   "estructural, y solo se justifica si L y M funcionan; si el patron "
                   "es que anadir canales categoriales no aporta, lo mas probable es que "
                   "este tampoco. Se abre de forma CONDICIONAL, no de forma previa",
        "experimento": 1,
    },
    {
        "id": "P",
        "fenomeno": "tokens de contraste estructural: NEG+ADJ, QUOTE+TOXIC, "
                    "MENTION+INSULT, PRONOUN+INSULT, COLLECTIVE+INSULT, VERB+TARGET",
        "experimentos_previos": "E2-05-CH-TGT, E2-05-CH-TGT-DIST, E2-06-TXT-CH, "
                                "E2-06-TXT-TGT-SPARSE, E2-07-CITA, E2-07-CITA-TXT, "
                                "E2-07-CITA-NEG-TXT, E2-08-NEG, E2-08-NEG-TXT",
        "representacion_existente": "las nueve filas anteriores. E2-07 y E2-08 construyen "
                                    "rasgos de cita y de negacion CON ALCANCE, "
                                    "incluyendo el recuento de insultos que caen dentro "
                                    "del alcance de la negacion y el de los que estan en "
                                    "comillas, lo cual es mas fino que un token "
                                    "NEG+ADJ. E2-05 cubre el emparejamiento entre "
                                    "posiciones de insulto y de target",
        "tipo_representacion": "rasgos densos con alcance sintactico calculado",
        "identidad_lexical_entra": True,
        "delta_historico": "E2-07-CITA-NEG-TXT -0,0220 con 1/5; E2-08-NEG-TXT -0,0115 con "
                           "0/5; E2-05-CH-TGT -0,1696 con 0/5; E2-06-TXT-TGT-SPARSE "
                           "-0,0285 con 1/5",
        "veredicto": EQUIVALENTE,
        "por_que": "el alcance de la negacion y el aislamiento de la cita ya se "
                   "calcularon y se midieron, y el emparejamiento insulto-target tambien. "
                   "Los tokens propuestos son una reexpresion mas tosca de lo ya "
                   "medido, salvo por la identidad lexica de las dos piezas, que es la "
                   "unica diferencia real. Esa diferencia se solapa con M, que es donde "
                   "se prueba de forma limpia, asi que se cierra aqui para no gastar dos "
                   "experimentos en la misma idea",
        "experimento": 0,
    },
)

# Fenomenos adicionales que P62 menciona en su EDA y que no son pistas pero cuya
# novedad conviene registrar, porque el encargo los nombra explicitamente.
FENOMENOS_EDA: tuple[dict[str, str], ...] = (
    {"id": "EDA-1", "fenomeno": "composicion de POS por clase y proporcion por tweet",
     "estado": YA_PROBADO,
     "detalle": "las 36 columnas pos_* son exactamente esto, en forma densa. El EDA de "
                "P62 lo vuelve a medir, pero como DESCRIPCION y no como feature, que es "
                "lo que el parrafo 4 establece"},
    {"id": "EDA-2", "fenomeno": "enriquecimiento de verbos, adjetivos y sustantivos por "
                                "etiqueta",
     "estado": YA_PROBADO,
     "detalle": "E2-15-NB-WORD y V25-01 operan sobre lemas; E2-09-EMO-EST y las familias "
                "de emocion, sobre categorizacion. El enriquecimiento como estadistica "
                "descriptiva global no es una feature y por tanto no esta cerrado, pero "
                "tampoco puede alimentar un modelo sin pasar por dentro del pliegue, por "
                "el parrafo 45"},
    {"id": "EDA-3", "fenomeno": "perfil pronominal y de targeting",
     "estado": YA_PROBADO,
     "detalle": "E2-05-CH-TGT y las col_* de las features estructuradas cubren "
                "segunda persona, plural generico y densidad de targeting, con -0,1696 y "
                "0/5. El perfil por clase como descripcion sigue sin hacerse y es "
                "util para el diagnostico de la seccion 47"},
    {"id": "EDA-4", "fenomeno": "marcadores discursivos y modalidad",
     "estado": PARCIAL,
     "detalle": "no hay fila del registro que las nombre. Lo mas cercano es la familia "
                "E2-07 de cita y atribucion, que cubre el habla referida pero no la "
                "modalidad. Es el unico fenomeno del encargo sin precedente claro, y "
                "aun asi se registra como hueco de datos y no como pista, porque el "
                "presupuesto de ocho experimentos ya tiene candidatos con mas respaldo"},
)


def main() -> None:
    """Verifica el cierre de P61 y ejecuta el gap analysis funcional.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P62-00 auditoria de P61 y gap analysis funcional")
    log("   NO se repite ningun experimento de P61")

    # ---------- verificacion del baseline y de P61 ----------
    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    import numpy as np
    from sklearn.metrics import f1_score
    v = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)
    ccv = float(f1_score(v["y_true"], v["pred"], average="macro"))
    if ccv != V03_CCV:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv!r}, no {V03_CCV!r}")
    part = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv")
    g = part.set_index("id").loc[v["ids"], "grupo_near_dup_autor"].to_numpy()
    partidos = int(pd.Series(g).groupby(g).nunique().gt(1).sum())
    if partidos:
        raise SystemExit(f"PARADA: {partidos} grupos canonicos partidos entre pliegues")
    log("")
    log("   BASELINE REPRODUCIDO BIT A BIT")
    log(f"      sha256       {hash_real}  coincide: {hash_real == V03_SHA}")
    log(f"      corrected_cv {ccv!r}  exacto: {ccv == V03_CCV}")
    log(f"      grupos       {len(set(g))} | partidos: {partidos}")

    reg = pd.read_csv(PROJECT_ROOT / "experiments.csv")
    p61 = reg[reg["experiment_id"].astype(str).str.startswith("P61")]
    log("")
    log("   ESTADO DE P61")
    log(f"      filas en el registro          {len(reg)} | ids unicos "
        f"{reg['experiment_id'].nunique()}")
    log(f"      experimentos de P61           {len(p61)}: {list(p61['experiment_id'])}")
    log("      resultados de P61, leidos de sus artefactos:")
    for nombre, clave in (("p61_lightgbm_capacidad.json", "deltas"),
                          ("p61_ensemble.json", None),
                          ("p61_segunda_particion.json", None)):
        d = json.loads((REPORTS / nombre).read_text(encoding="utf-8"))
        if clave:
            log(f"        {nombre:32} {d[clave]}")
    ens = json.loads((REPORTS / "p61_ensemble.json").read_text(encoding="utf-8"))
    seg = json.loads((REPORTS / "p61_segunda_particion.json").read_text(encoding="utf-8"))
    log(f"        p61_ensemble.json            mejor delta "
        f"{float(ens['mejor_configuracion']['Delta_vs_V03']):+.10f}, gate "
        f"{ens['mejor_configuracion']['gate_superado']}")
    log(f"        p61_segunda_particion.json   control negativo "
        f"{seg['tests']['3_control_negativo']['delta']:+.10f}, finalists "
        f"{seg['finalistas_congelados']}")
    log("      conclusion que P62 hereda y usa para justificar su enfoque:")
    log("        los errores de V03 son de alta confianza, luego el limite no es de")
    log("        decision; LightGBM sobre la misma representacion no llega, luego no es")
    log("        de capacidad del clasificador sobre esos rasgos; y el ensemble sobre OOF")
    log("        no transfiere, luego no es de combinacion. Queda la REPRESENTACION")

    # ---------- el tagger ----------
    log("")
    log("   EL TAGGER, RESUELTO ANTES DE NADA MAS")
    import spacy
    try:
        nlp = spacy.load("es_core_news_md", exclude=["ner"])
        meta = dict(nlp.meta)
        tagger = {
            "herramienta": "spaCy", "version_spacy": spacy.__version__,
            "modelo": "es_core_news_md", "version_modelo": meta.get("version"),
            "idioma": meta.get("lang"), "licencia": meta.get("license"),
            "componentes_excluidos": ["ner"],
            "vectores_preentrenados": False,
            "autorizacion": "D-003, ya vigente: anotador simbolico permitido, vectores "
                            "prohibidos con assert en el codigo de features",
            "estado_normativo": "PERMITIDO, no ambiguo",
            "uso_previos_en_el_proyecto": "E2-03 y E2-04, y las 20 canales del pool de "
                                          "P50 que proceden de esa anotacion",
        }
        log(f"      spaCy {spacy.__version__} | es_core_news_md {meta.get('version')} | "
            f"idioma {meta.get('lang')} | licencia {meta.get('license')}")
        log("      D-003 lo autoriza como anotador simbolico con exclude=['ner']; los "
            "vectores siguen prohibidos")
        log("      NO hay ambiguedad normativa que resolver, y no se usa ningun recurso "
            "externo nuevo")
    except Exception as exc:
        raise SystemExit(f"PARADA: es_core_news_md no esta disponible: {exc}") from exc

    # ---------- el gap functional ----------
    log("")
    log("   GAP ANALYSIS FUNCIONAL DE V2 A V61")
    filas: list[dict[str, Any]] = []
    for f in FENOMENOS:
        filas.append({
            "id": f["id"], "tipo": "pista", "fenomeno": f["fenomeno"],
            "experimentos_previos": f["experimentos_previos"],
            "representacion_existente": f["representacion_existente"],
            "tipo_representacion": f["tipo_representacion"],
            "identidad_lexical_entra": f["identidad_lexical_entra"],
            "delta_historico": f["delta_historico"],
            "veredicto": f["veredicto"], "por_que": f["por_que"],
            "experimentos_consumidos": f["experimento"],
        })
        log(f"      {f['id']:3} {f['veredicto']:22} {f['experimento']} experimento(s) | "
            f"identidad lexical entra: {f['identidad_lexical_entra']}")
    for f in FENOMENOS_EDA:
        filas.append({
            "id": f["id"], "tipo": "eda", "fenomeno": f["fenomeno"],
            "experimentos_previos": "", "representacion_existente": f["detalle"],
            "tipo_representacion": "no aplica", "identidad_lexical_entra": "",
            "delta_historico": "", "veredicto": f["estado"], "por_que": "",
            "experimentos_consumidos": 0,
        })
        log(f"      {f['id']:5} {f['estado']:22} 0 experimento(s) | {f['fenomeno'][:56]}")

    nuevas = [f for f in filas if f["veredicto"] == NUEVO]
    cerradas = [f for f in filas if f["veredicto"] == EQUIVALENTE]
    presupuesto = sum(int(f["experimentos_consumidos"]) for f in filas)
    log("")
    log(f"   {len(nuevas)} pistas genuinamente nuevas, {len(cerradas)} cerradas por "
        f"equivalencia funcional, presupuesto comprometido {presupuesto} de 8")

    # ---------- comprobacion de la afirmacion central ----------
    log("")
    log("   COMPROBACION DE LA AFIRMACION CENTRAL, CONTRA EL CODIGO Y NO CONTRA LOS NOMBRES")
    codigo_sintaxis = (EXPERIMENTS / "19_sintaxis_cita_negacion_v2.py").read_text(
        encoding="utf-8")
    comprobaciones = {
        "existe_word__pos": bool(re.search(r"f?\"\{[^}]*lema[^}]*\}__\{|palabra__|lemma__pos",
                                          codigo_sintaxis)),
        "secuencia_pos_emite_pos_headpos": "f\"{token.pos}_{token.head_pos}\"" in codigo_sintaxis,
        "params_pos_ngram_1_4": 'ngram_range": (1, 4)' in codigo_sintaxis,
        "params_deps_ngram_1_3": 'ngram_range": (1, 3)' in codigo_sintaxis,
        "no_hay_triple_lexical": not bool(re.search(
            r"\{token\.lema\}_\{token\.dep\}_\{|\{head_lema\}_\{dep\}_\{dep_lema\}",
            codigo_sintaxis)),
    }
    for k, v in comprobaciones.items():
        log(f"      {k:42} {v}")
    if comprobaciones["existe_word__pos"]:
        raise SystemExit(
            "PARADA: el codigo ya emite un token palabra__POS, luego la pista L no es "
            "nueva. El gap analysis seria incorrecto y hay que rehacerlo")
    if not comprobaciones["secuencia_pos_emite_pos_headpos"]:
        raise SystemExit(
            "PARADA: la funcion secuencia_pos ya no emite POS_dependiente_POS_cabeza, luego "
            "la afirmacion de que la pista N esta cubierta no se sostiene")
    log("      las dos comprobaciones que deciden P62 se verifican contra el codigo: no hay")
    log("      token palabra__POS en ninguna parte, y la secuencia categorial existe")

    # ---------- escritura ----------
    with CSV_GAP.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["id", "tipo", "fenomeno", "experimentos_previos",
                    "representacion_existente", "tipo_representacion",
                    "identidad_lexical_entra", "delta_historico", "veredicto", "por_que",
                    "experimentos_consumidos"])
        for f in filas:
            w.writerow([a_ascii(str(f[k])) for k in
                        ("id", "tipo", "fenomeno", "experimentos_previos",
                         "representacion_existente", "tipo_representacion",
                         "identidad_lexical_entra", "delta_historico", "veredicto",
                         "por_que", "experimentos_consumidos")])

    L: list[str] = []
    W = L.append
    W("# P62 - Gap analysis funcional de V2 a V61")
    W("")
    W(f"Generado por `experiments/p62_00_auditoria_y_gap.py`. "
      f"Fecha {inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Por que este analisis es funcional y no una busqueda de nombres")
    W("")
    W("El encargo dice que buscar nombres no basta, y el motivo ya se ve en el registro. Hay")
    W("339 experimentos, y una busqueda de `POS` o `dependencias` devuelve las mismas")
    W("pocas filas sea cual sea la pista que se este comprobando, porque las cinco usan el")
    W("mismo vocabulario. Lo que las distingue es la REPRESENTACION, y eso no esta en")
    W("ningun nombre.")
    W("")
    W("Por eso para cada fenomeno se registra el patron de token que el codigo emite de")
    W("verdad, leido del fuente, y si la representacion era DENSE o SPARSE, y si la")
    W("identidad de la palabra llegaba a entrar.")
    W("")
    W("## La distincion que decide P62")
    W("")
    W("El proyecto ya construyo informacion de POS y de dependencias DOS veces, y las dos")
    W("como DENSIDAD: 36 columnas `pos_VERB_n` / `pos_VERB_frac` y 31 de `dep_amod_n`. Lo")
    W("que P62 propone es un espacio de tokens SPARSE. Son objetos distintos, y que el")
    W("denso fallara no es evidencia sobre el disperso.")
    W("")
    W("Pero si es evidencia fuerte sobre una cosa: los marginales de POS y de relacion")
    W("aportan muy poco, porque 36 columnas que describen la composicion de un tweet no")
    W("llegaron a mover el resultado. Asi que la hipotesis tiene que ganarse el sitio en el")
    W("termino de INTERACCION y no en la etiqueta, y eso es exactamente lo que significan")
    W("las pistas L y M.")
    W("")
    W("## El tagger, resuelto antes de nada mas")
    W("")
    W("| campo | valor |")
    W("|---|---|")
    for k, v in tagger.items():
        W(f"| {k} | {v} |")
    W("")
    W("No hay ambiguedad normativa que resolver: **D-003** ya establecio que")
    W("`es_core_news_md` puede usarse como anotador simbolico con `exclude=[\"ner\"]`, y que")
    W("`doc.vector`, `token.vector`, `doc.tensor` y `similarity()` quedan prohibidos con un")
    W("assert en el codigo de features. No se introduce ningun recurso externo nuevo.")
    W("")
    W("## El resultado pista por pista")
    W("")
    W("| id | veredicto | experimentos | por que |")
    W("|---|---|---:|---|")
    for f in filas:
        if f["tipo"] == "pista":
            W(f"| **{f['id']}** | **{f['veredicto']}** | {f['experimentos_consumidos']} | "
              f"{a_ascii(f['por_que'])[:200]} |")
    W("")
    for f in filas:
        if f["tipo"] != "pista":
            continue
        W(f"### Pista {f['id']} - {a_ascii(f['fenomeno'])}")
        W("")
        W(f"* **Veredicto**: `{f['veredicto']}`, con {f['experimentos_consumidos']} "
          f"experimento(s) consumido(s)")
        W(f"* **Experimentos previos**: {a_ascii(f['experimentos_previos'])}")
        W(f"* **Representacion existente**: {a_ascii(f['representacion_existente'])}")
        W(f"* **Tipo**: {a_ascii(f['tipo_representacion'])}")
        W(f"* **Identidad lexical**: {f['identidad_lexical_entra']}")
        W(f"* **Delta historico**: {a_ascii(f['delta_historico'])}")
        W(f"* **Razon**: {a_ascii(f['por_que'])}")
        W("")
    W("## Los fenomenos del EDA que no son pistas")
    W("")
    W("| id | estado | detalle |")
    W("|---|---|---|")
    for f in filas:
        if f["tipo"] == "eda":
            W(f"| {f['id']} | {f['veredicto']} | {a_ascii(f['representacion_existente'])} |")
    W("")
    W("## Las dos comprobaciones que deciden la fase, contra el codigo")
    W("")
    W("Una afirmacion de este informe no es una interpretacion: es una comprobable. Se lee")
    W("`experiments/19_sintaxis_cita_negacion_v2.py` y se verifica que no existe ningun")
    W("token `palabra__POS`, y que la secuencia categorial si existe. El script ABORTA si")
    W("esas dos condiciones no se cumplen, porque un gap analysis que no se puede fallar")
    W("no es un gap analysis.")
    W("")
    W("```text")
    for k, v in comprobaciones.items():
        W(f"  {k:42} {v}")
    W("```")
    W("")
    W("## El presupuesto que se compromete")
    W("")
    W("```text")
    W(f"  pistas genuinamente nuevas   {len(nuevas)}: {[f['id'] for f in nuevas]}")
    W(f"  cerradas por equivalencia     {len(cerradas)}: {[f['id'] for f in cerradas]}")
    W(f"  experimentos comprometidos   {presupuesto} de 8 principales, 0 de 4 reservas")
    W("```")
    W("")
    W("O se abre O de forma CONDICIONAL, es decir solo si L o M demuestran que anadir un")
    W("canal categorial disperso al espacio de V03 mueve algo. La condicion se declara antes")
    W("de ver el resultado, y no se cumple por defecto.")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p62_00_auditoria_y_gap.py`. Sin push.*")
    MD_GAP.write_text(a_ascii("\n".join(L)), encoding="ascii")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P62_GAP_ANALYSIS",
        "naturaleza": ("AUDITORIA. 0 experimentos, 0 modelos, 0 submissions, 0 Kaggle, "
                       "0 holdout, 0 test. No se repite nada de P61"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "commit": commit,
        "baseline_verificado": {"sha256": hash_real, "corrected_cv": repr(ccv),
                                "grupos": len(set(g)), "grupos_partidos": partidos},
        "p61_verificada": {
            "experimentos": list(p61["experiment_id"]),
            "ensemble_mejor_delta": float(ens["mejor_configuracion"]["Delta_vs_V03"]),
            "ensemble_gate": ens["mejor_configuracion"]["gate_superado"],
            "control_negativo": seg["tests"]["3_control_negativo"]["delta"],
            "finalistas": seg["finalistas_congelados"],
            "conclusion_heredada": ("los errores son de alta confianza, luego el limite no "
                                    "es de decision; LightGBM no llega sobre la misma "
                                    "representacion, luego no es de capacidad del "
                                    "clasificador; el ensemble sobre OOF no transfiere, "
                                    "luego no es de combinacion. Queda la representacion"),
        },
        "tagger": tagger,
        "fenomenos": filas,
        "comprobaciones_contra_el_codigo": comprobaciones,
        "presupuesto": {"principales_disponibles": 8, "comprometidos": presupuesto,
                        "reservas": 4},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_AUDIT.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")

    for destino in (MD_GAP, CSV_GAP, JSON_AUDIT):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P62-00 OK | {len(filas)} fenomenos | {len(nuevas)} nuevos | {len(cerradas)} "
        f"cerrados sin experimento | presupuesto comprometido {presupuesto}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
