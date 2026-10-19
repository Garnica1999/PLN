#!/usr/bin/env python
"""
P94-00: inventario experimental verificable (P0 de P94).

Escribe reports/p94_experiment_inventory.csv a partir de filas curadas, y
VERIFICA contra experiments.csv que los identificadores historicos citados
existen realmente en el registro (sin re-ejecutar ningun experimento).

Uso:
    /home/carlos/venv/bin/python experiments/p94_00_inventario.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
REGISTRO = RAIZ / "experiments.csv"
SALIDA = RAIZ / "reports" / "p94_experiment_inventory.csv"

CABECERA = ["family", "class_status", "historical_experiment_ids", "method",
            "best_valid_result", "historical_conclusion", "reason_for_closure",
            "known_implementation_limitations"]

# Anclas verificables por familia: identificadores que DEBEN existir en el registro.
ANCLAS = {
    "F01_base_ngrams_lineal_umbrales": ["B1", "REP-05", "MOD-03b_BEST", "TH-01"],
    "F02_protocolo_validacion_y_gobernanza": ["PA-01", "V9-01", "HOLD-01", "V10-00"],
    "F03_ensembles_remuestreo": ["ENS-02", "S04", "V6-06", "V8-01"],
    "F04_meta_stacking_fusion": ["E2-02-META", "E2-18-META-BASES-RIDGE", "S08", "V4-H01", "P66-03"],
    "F05_umbrales_calibracion_decision": ["TH-01", "S03", "V10-04", "V56-01", "P84-05", "E2-16-MEJOR"],
    "F06_busqueda_oof_hiperparametros": ["OPT-01", "S05", "P61-01", "P88-00"],
    "F07_lineales_alternativos_online": ["E2-15-NBSVM", "V26-03", "V35-01", "V36-01", "V37-01",
                                          "V44-01", "V4-G01-G03"],
    "F08_ponderacion_de_terminos": ["V37-01", "V41-01"],
    "F09_kernels_geometria_no_lineal": ["V26-01", "V27-01", "P8-REFIT-C", "V4-G01-G03"],
    "F10_embeddings_desde_cero": ["EBD-05", "EBD-13", "P84-04", "P85-01"],
    "F11_embeddings_supervisados_pls": ["V40-01", "V43-01", "V44-02"],
    "F12_deep_learning_desde_cero": ["DL-01", "DL-03", "V9-04", "V10-11", "P82", "P84-01", "P84-02"],
    "F13_estructura_de_etiquetas": ["E2-01-HS-TR-AG", "V6-02", "V12-02", "V13-02", "V14-01",
                                     "V28-00", "V45-01"],
    "F14_ruido_etiquetas_ponderacion": ["V30-00", "V4-B03", "FIN-01-C-REF-44", "P8-REFIT-C"],
    "F15_augmentation_y_expansion": ["V6-08", "V47-00", "V41-01"],
    "F16_ingenieria_de_features_linguistica": ["E2-03-POS", "E2-04-MORPH", "E2-07-CITA", "P62-01"],
    "F17_lexicones_y_features_densas": ["E2-09-EMO", "HIB-H12", "B5-DENSE"],
    "F18_ortografia_estructura_estilometria": ["E2-10-ORT-A", "V15-01", "P63-02"],
    "F19_seleccion_contextual_operadores": ["V50-01", "V51-02", "V31-04", "V32-05", "V33-01", "V33-03"],
    "F20_prevalencia_y_priors": ["V10-07", "V11-06", "P84-05"],
    "F21_modelos_bayesianos": ["P83", "E2-15-NB-WORD"],
    "F22_objetivos_y_perdidas": ["V7-01", "P60-01", "V12-03", "P59-10"],
    "F23_correccion_humana_de_etiquetas": ["P87-00"],
    "F24_propagacion_near_duplicados": ["V4-F01-F04"],
}

# Filas curadas (contenido del inventario; los hechos provienen del registro,
# PROJECT_STATE.md, DECISIONS.md, RUNLOG.md y FINAL_REPORT.md).
FILAS = [
    ["F01_base_ngrams_lineal_umbrales", "CLOSED",
     "E00; B1; PRE-00..PRE-09; REP-01..REP-06; MOD-01..MOD-04; TH-01; S02; S03; V2-BASE-M1..M4; PA-01",
     "TF-IDF union word(1,2)+char_wb(3,5) min_df=2 sublinear + LR balanced C=2.0 por etiqueta + umbrales por etiqueta",
     "corrected_cv 0.7772114525165721 (V03); LB 0.78306 (v03); F1 local historico 0.7927/0.7989 con el protocolo previo",
     "Familia campeona; unica via adoptada; el resto de variantes quedo descartada; el LB forma un plateau de ~1 error estandar",
     "Integrada como V03; sin variante superior transferible (D-012/D-013/D-014)",
     "Los umbrales de V03 vienen de un protocolo mas debil que el de la metrica reportada (D-126); el control por pseudo-autor se descubrio despues (D-015)"],
    ["F02_protocolo_validacion_y_gobernanza", "CLOSED",
     "PA-01; V9-01..V9-05; HOLD-01; V10-00..V10-03; V11-01..V11-05; P89-00",
     "StratifiedGroupKFold con grupos near-duplicado UNION pseudo-autor; reproducibilidad nivel R3; gobernanza del registro",
     "0 grupos compartidos entre pliegues; el gap local-LB del modelo final baja de -0.0128 a -0.0039 (D-015)",
     "Protocolo vigente adoptado; el holdout historico quedo agotado (6/5); la fase v11 se cerro sin candidato",
     "Infraestructura; holdout agotado; registro gobernado (D-229/D-294)",
     "Holdout agotado; ruido de particion ~0.017 (D-070); holdout2 con 0/3 evaluaciones gastadas"],
    ["F03_ensembles_remuestreo", "CLOSED",
     "ENS-01..ENS-04; S04; V6-06; V8-01; V8-06..V8-08; V8-12; V8-13; D-067",
     "Nested ensembles; bagging por grupos; fold-ensembles; promediado de repeticiones",
     "ENS-02 0.7919 anidada; S04 LB 0.77510; V6-06 bagging -0.0105 (0/5 pliegues)",
     "Ninguna combinacion transfiere al LB; la mejora local del promediado es un artefacto (D-067); el bagging no se midio limpiamente",
     "Cierres por LB y por limitacion de protocolo (seccion 54 de v6)",
     "Evaluar bagging exigiria una particion de calibracion separada del OOF (no disponible)"],
    ["F04_meta_stacking_fusion", "CLOSED",
     "E2-02; E2-05; E2-06; E2-13..E2-18; S08; FIN-06; V4-H01; V8-05; P65-01; P65-03; P66-01..P66-03; P67; P68",
     "Meta-modelos sobre scores base y canales; ridge meta; fusion en el espacio de scores; target encoding de entidades",
     "E2-18-RIDGE 0.7874; S08 LB 0.78163; FIN-01/C holdout 0.8163 vs 0.8285 de V03",
     "El meta no transfiere; la fusion de scores se re-identifico como mas calibracion que representacion (D-169/170) y se cerro en P67-P68",
     "Rechazo en LB/holdout y re-identificacion del mecanismo",
     "Las probabilidades base son OOF compartidas (limitacion declarada en su momento)"],
    ["F05_umbrales_calibracion_decision", "CLOSED",
     "TH-01; S03; V7-P0 (D-055); V8-04; V8-09; V10-04; V10-09; P55 (D-126/D-127); V56-01; P76 (D-259); P84-05; E2-16",
     "Umbrales por etiqueta; re-estimacion anidada; suavizado; correccion de prior/EM; calibracion de probabilidades",
     "TH-01 adoptada (+0.0031 LB pareado); ninguna otra variante mejora",
     "Umbrales debilmente identificados (D-055); no hay nada que recalibrar; la calibracion no era el cuello de botella (D-259)",
     "Techos medidos y sondas; el punto de operacion de V03 es inseparable de su miscalibracion",
     "El protocolo de los umbrales de V03 es mas debil que el de la metrica reportada (D-126)"],
    ["F06_busqueda_oof_hiperparametros", "CLOSED",
     "OPT-01; S05; P61-01..P61-03; P88-00; V35-00",
     "Optuna y busqueda de pesos maximizando el OOF; patron de busqueda directa de la metrica",
     "OPT-01 0.8012 local -> LB 0.77602; P61-02 rechazado por la segunda particion",
     "La busqueda por OOF no transfiere (freno de transferencia); el patron quedo cerrado por duplicacion",
     "Rechazo en LB y duplicacion (P88-00); regla dOOF > +0.0376 con neto positivo",
     "Ninguna documentada; el precedente es solido"],
    ["F07_lineales_alternativos_online", "CLOSED",
     "E2-15 (NB/NBSVM); MOD-01/03a/03b; V4-G01-G03 (Ridge/PA); V26-03 (Tsetlin); V35-01 (SGD); V36-01 (hashing); V37-01 (BM25); V44-01 (VW FTRL/CB/PiSTOL); V12-02; V12-03",
     "Modelos lineales y online alternativos sobre la representacion dispersa",
     "Ninguno supera a V03 (PA 0.7626; NB-SVM 0.7612; VW mejor PiSTOL 0.6369; Tsetlin invalidado)",
     "Todos los candidatos quedan por debajo de V03; algunos experimentos fueron invalidados por implementacion",
     "Cierres por medicion e invalidacion",
     "VW no es bit-reproducible en hilos; Tsetlin cortado en K=200; PLS con defectos de API documentados"],
    ["F08_ponderacion_de_terminos", "CLOSED",
     "v4-C (log-odds/chi2/MI/BM25/binario; V41-01); V37-01",
     "Ponderacion supervisada o alternativa de la representacion dispersa",
     "Las ponderaciones alternativas degradan; BM25 cerrado",
     "La ponderacion supervisada no aporta; familia cerrada",
     "Cierres por medicion",
     "BM25 sobre char nunca se ejecuto (la linea se cerro sin el, declarado)"],
    ["F09_kernels_geometria_no_lineal", "CLOSED",
     "V26-01; V27-01; V4-G01-G03; V8-05; P8-REFIT-C; V4-X01; D-190",
     "RBF exacto; RBF sobre SVD300; kernel de espectro de caracteres; aproximaciones RFF/Nystrom",
     "svd300_rbf 0.7904 fragil -> rechazado por holdout; V27-01 0.7621",
     "El RBF memoriza la representacion (V26); el eje de geometria de representacion quedo cerrado (D-190)",
     "Cierres por medicion y holdout",
     "Las aproximaciones RFF/Nystrom quedaron como reserva sin ejecutar (aproximan el mismo kernel memorizador)"],
    ["F10_embeddings_desde_cero", "CLOSED",
     "EBD-00..EBD-27; E-CORP-01..E-CORP-09; P71-P82; P84-04; P85-01",
     "Word2Vec/SGNS/fastText entrenados desde cero (corpus propio y corpus no etiquetados autorizados)",
     "Embedding-only ~0.65; mejor fusion 0.7095; SGNS corregido -0.151120 vs V03",
     "La representacion mejora (cobertura), la tarea no; corpus y arquitectura agotados como palancas",
     "Mediciones + directiva explicita de no reabrir la familia (P85-01)",
     "Sin ancla a gensim (no instalable en el entorno); corpus de dominio cerrado"],
    ["F11_embeddings_supervisados_pls", "CLOSED",
     "V38-01; V39-01/V39-02; V40-01; V41-00; V42-00/V42-01; V43-00..V43-02; V44-00..V44-02",
     "Proyeccion supervisada document x term (PLS) con control SVD pareado; historia de experimentos invalidados",
     "PLS correcto -0.013742 vs V03; control SVD-8 0.663295",
     "PLS queda cerrado y negativo; dos intentos previos fueron invalidos por implementacion",
     "Medicion final valida mas invalidaciones documentadas",
     "Rank<=3 de la factorizacion; APIs sin soporte; defectos corregidos y registrados"],
    ["F12_deep_learning_desde_cero", "CLOSED",
     "DL-01..DL-03; V9-04; V9-05; V10-11; P73; P74; P75; P76; P78; P82; P82-02; P84-01; P84-02",
     "MLP/CharCNN/BiLSTM/TextCNN/BiGRU con atencion/fastText supervisado, todos desde cero",
     "DL-01 0.7949 historico (artefacto de protocolo; corregido 0.7583); RNN mejor brazo ~0.6397 vs V03",
     "Lo mejor de DL queda por debajo de V03; la atencion es indistinguible de cero; DL-01 era artefacto de protocolo",
     "Mediciones corregidas con protocolo anidado y auditorias P79-P82",
     "Contaminaciones de pool/early-stopping corregidas y documentadas; coste GPU"],
    ["F13_estructura_de_etiquetas", "CLOSED",
     "E2-01; E2-02; E2-21; V6-01; V6-02; V12-02; V12-03; V13-01; V13-02; V14-01; V28-00; V28-01; V45-01; P65-02",
     "Cadenas; meta de dependencia; inferencia conjunta; especializacion por etiqueta; perdida por pares; partially-labeled",
     "Techo con oraculo +0.046271 medido; ningun mecanismo lo captura; V12-02 MARGINAL +0.0016",
     "La estructura conjunta es real pero no explotable (D-116); TR/AG llevan informacion sobre HS no utilizable (D-102)",
     "Mediciones de techo y rechazos (D-090..D-104)",
     "Los techos se miden sobre los propios errores de V03 (declarado)"],
    ["F14_ruido_etiquetas_ponderacion", "MARGINAL",
     "V4-A01/V4-A02; V4-B01..V4-B05; V30-00..V30-02; D-191..D-217; FIN-01-C-REF-44; P8-REFIT-C",
     "Deteccion de ruido; dureza; downweighting cost-sensitive; minado de ejemplos duros",
     "Candidato 3/3 particiones positivas local; LB 0.77522; holdout C rechazado (0.8163 vs 0.8285)",
     "Candidato PROMISING_UNCONFIRMED/PENDING; la causa que faltaba es la seleccion del baseline (D-216); nunca se adopto",
     "Direccion rechazada en LB/holdout; gobernanza pendiente; estado congelado",
     "Candidato congelado; N pequeno; intervenciones medidas +-0.0005"],
    ["F15_augmentation_y_expansion", "CLOSED",
     "V6-08; V47-00; V47-01; V41-01; V42-00; PRE-00..PRE-09",
     "Aumento de datos en corpus (espaciado/elongacion/puntuacion); expansion; perturbaciones de representacion",
     "El aumento empeora; no existe expansion legal de datos (D-118)",
     "La augmentation esta cerrada con evidencia; sin datos externos etiquetados permitidos",
     "Cierres por medicion y por reglas",
     "Mixup quedo como duplicado de familias cerradas (V42-00)"],
    ["F16_ingenieria_de_features_linguistica", "CLOSED",
     "E2-03-POS; E2-04-DEPS/MORPH/SYN3; E2-07-CITA; P62-01..P62-05; D-143",
     "Features simbolicas spaCy (POS, dependencias, morfologia, cita, negacion como patrones)",
     "Ninguna mejora sobre V03; la synergy palabra-POS es real (17.41%) pero no usable",
     "La linea de sintaxis/cita/negacion quedo cerrada sin mejora",
     "Cierres por medicion",
     "Solo anotacion simbolica spaCy (D-003: sin vectores preentrenados)"],
    ["F17_lexicones_y_features_densas", "CLOSED",
     "E2-09-EMO-EST/LEX/EMO/TXT-EMO; HIB-H11; HIB-H12; HIB-H14; HIB-UNION; B5-DENSE",
     "Lexicones de emocion/sentimiento; features densas de odio individual/colectivo; union con densas",
     "-0.0213 a -0.1319 vs B1; todas descartadas",
     "Las features densas per se pierden contra la representacion de n-gramas; los lexicones no aportan",
     "Cierres por medicion (el control a cero reproduce MOD-03b exacto)",
     "Los lexicones externos requieren documentacion y licencia (D-007)"],
    ["F18_ortografia_estructura_estilometria", "CLOSED",
     "E2-10-ORT-AE/ABCD; V6-08; V15-01; P63-02..P63-07; P64-02..P64-04; P64-08; D-151; D-158; D-159",
     "Variantes ortograficas; tipografia; posicion; estilometria; estructura interna; ratios",
     "Senales especificas MARGINAL/COMPLEMENTARY; ninguna adopcion; el perfil segmental es longitud con disfraz (D-158)",
     "Cerradas o marginales; V03 ya lleva la estructura en su confianza (D-159)",
     "Cierres por medicion",
     "La clasificacion del id se hace por reglas (D-148)"],
    ["F19_seleccion_contextual_operadores", "CLOSED",
     "V50-01; V50-02; V51-01; V51-02; P77 (D-260); P68-H2 (D-187/188); V31-04; V32-05; V33-01..V33-04",
     "Operadores de transferencia selectiva; reglas contextuales; interaccion URL x longitud; seleccion a nivel de celda",
     "Operadores sin finalista (monotonia negativa); V32-05 local +0.0045 -> test identico a V03 (delta LB exactamente 0)",
     "Techos negativos medidos (D-121..D-124); la seleccion contextual se cierra por razon matematica; la senal URLxL no transfiere",
     "Cierres por medicion",
     "La segunda particion del selector nunca se ejecuto (limitacion declarada, D-258.9)"],
    ["F20_prevalencia_y_priors", "CLOSED",
     "V10-07; V11-06; V11-07; V11-08; P60-05; P84-05; D-080..D-082",
     "Estimacion de prevalencia; correccion de prior; EM con shrinkage y decision robusta",
     "Los estimadores funcionan pero la correccion no mejora la metrica; EM destruye mas valor del que aporta",
     "Correccion de prior no mejora (causa medida, D-082); el uso de la prueba para estimar prevalencia quedo prohibido (D-080)",
     "Mediciones mas gobernanza",
     "Las estimaciones indirectas de prevalencia estan acotadas por gobernanza"],
    ["F21_modelos_bayesianos", "CLOSED",
     "P83; E2-15-NB-WORD",
     "Logistica bayesiana dispersa con hiper-prior Gamma (inferencia variacional) + Bayes multinomial Dirichlet; NB clasico",
     "P83: pregunta de calibracion NEGATIVA; complementariedad medida con recuentos reales; nunca buscaba superar a V03",
     "El modelo bayesiano no es competidor; el estudio de doble objetivo quedo cerrado",
     "Diseno de doble objetivo (no compite por F1) mas lectura negativa",
     "Ciclo de correccion de codigo muerto documentado; corrida GPU de ~2 minutos"],
    ["F22_objetivos_y_perdidas", "CLOSED",
     "V7-01 (F1-suave/focal); P60-01; P60-02; P60-04; P60-06; P60-07; P60-08; V12-01..V12-05; P59-10",
     "Perdidas sustitutas de la metrica (F1 suave, focal), asimetricas, numero efectivo, transporte, alineacion de gradiente; regularizacion por vecindad; perdida propia verificada por derivada",
     "F1-suave -0.0033 (la peor); mejor marginal V12-02 +0.0016",
     "Aproximar mejor el objetivo no es mejor en este problema (D-053); familia cerrada",
     "Cierres por medicion",
     "El ajuste propio usa 1186 filas; la derivada de la perdida propia se verifico (D-096)"],
    ["F23_correccion_humana_de_etiquetas", "BLOCKED",
     "P87-00; P91..P93 (programa A/B/C1/C2); P93-R5; D-005",
     "Gobernanza de anotacion; revision humana ciega; adjudicacion; candidatos de correccion",
     "14 candidatas documentadas no confirmadas; 0 errores confirmados; techo historico ~+0.009 < banda de particion 0.017",
     "Pausada con 14 candidatos preservados; las reglas son silenciosas; valor esperado bajo (P87); requiere autorizacion separada",
     "Reglas mas gobernanza mas valor esperado medido",
     "Las decisiones finales no son independientes de A/B (visibles en ronda 2); muestreo enriquecido"],
    ["F24_propagacion_near_duplicados", "BLOCKED",
     "V4-F01-F04; sonda v18; D-288",
     "Propagacion de etiquetas entre near-duplicados",
     "Techo medido nulo (0 celdas en V4-F04 y sonda v18)",
     "QUARANTINED: requiere autorizacion del organizador; beneficio esperado inferior",
     "Bloqueo por autorizacion mas techo nulo",
     "Ninguna conocida"],
]


def main() -> None:
    """Verifica anclas contra el registro y escribe el inventario."""
    with open(REGISTRO, encoding="utf-8-sig", newline="") as fh:
        filas_reg = list(csv.DictReader(fh))
    ids_reg = {f["experiment_id"].strip() for f in filas_reg}

    fallos: list[str] = []
    for fam, anclas in ANCLAS.items():
        faltan = [a for a in anclas if a not in ids_reg]
        if faltan:
            fallos.append(f"{fam}: anclas ausentes en el registro: {faltan}")
    familias_en_filas = [f[0] for f in FILAS]
    if set(familias_en_filas) != set(ANCLAS):
        fallos.append("desajuste entre filas curadas y anclas")
    if len(FILAS) != len(set(familias_en_filas)):
        fallos.append("familias duplicadas")

    with open(SALIDA, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(CABECERA)
        w.writerows(FILAS)

    # releer para validar estructura
    with open(SALIDA, encoding="utf-8", newline="") as fh:
        leidas = list(csv.reader(fh))
    if len(leidas) != len(FILAS) + 1 or any(len(f) != len(CABECERA) for f in leidas):
        fallos.append("estructura CSV invalida tras escribir")

    por_estado: dict[str, int] = {}
    for f in FILAS:
        por_estado[f[1]] = por_estado.get(f[1], 0) + 1

    print("P94-00 inventario: ejecutado")
    print(f"  familias: {len(FILAS)} | estados: {por_estado}")
    print(f"  anclas verificadas contra el registro ({len(ids_reg)} filas): "
          f"{sum(len(v) for v in ANCLAS.values())} referencias, "
          f"{'todas OK' if not fallos else 'FALLOS'}")
    if fallos:
        for f in fallos:
            print("  -", f)
        sys.exit(1)
    print("  reports/p94_experiment_inventory.csv escrito y validado")


if __name__ == "__main__":
    main()
