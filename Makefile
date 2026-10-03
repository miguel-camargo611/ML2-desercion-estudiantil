# Atajos del flujo. Sin `make` (p. ej. en Windows), use los comandos de la derecha directamente.
PY ?= python

.PHONY: install all eda test lint pdf

install:                 ## pip install -e ".[dev]"
	$(PY) -m pip install -e ".[dev]"

all:                     ## desercion all  (flujo completo, ~2 h)
	desercion all

eda:                     ## desercion prepare && desercion eda  (minutos)
	desercion prepare
	desercion eda

test:                    ## pytest  (datos sintéticos, no requiere el dataset)
	$(PY) -m pytest

lint:                    ## ruff check src tests notebooks
	$(PY) -m ruff check src tests notebooks

pdf:                     ## compila los informes (requiere tectonic)
	cd reports/segunda_entrega && tectonic segunda_entrega.tex
	cd reports/primera_entrega && tectonic primera_entrega.tex
