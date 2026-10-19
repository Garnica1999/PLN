#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path
import re
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta

train_full = data_module.read_train()
test_df = data_module.read_test()

lexico_comercial = ["sorteo", "entradas", "premio", "concurso", "fiesta exclusiva", "ganadoras del primer"]

print("--- REVISANDO TRAIN FULL PARA LEXICO COMERCIAL ---")
for i, row in train_full.iterrows():
    t_low = row.text.lower()
    if any(k in t_low for k in lexico_comercial):
        print(f"Train row {i} (id {row.id}): HS={row.HS}, TR={row.TR}, AG={row.AG} | {row.text[:80]}")

print("\n--- REVISANDO TEST PARA LEXICO COMERCIAL ---")
for i, row in test_df.iterrows():
    t_low = row.text.lower()
    if any(k in t_low for k in lexico_comercial):
        print(f"Test row {i} (id {row.id}) | {row.text[:80]}")
