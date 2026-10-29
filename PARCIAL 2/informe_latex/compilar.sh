#!/usr/bin/env bash
# Compila el informe con TinyTeX y reporta errores, referencias y páginas.
set -euo pipefail

cd "$(dirname "$0")"
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"

pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex

echo "--- Advertencias y referencias ---"
grep -nE "Warning|Undefined|undefined|Overfull|Underfull" main.log || true
echo "--- Páginas ---"
pdfinfo main.pdf | grep -E "^Pages|^Page size"
