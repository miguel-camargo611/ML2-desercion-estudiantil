# Atajos del flujo. En Windows sin `make`, use directamente: python -m desercion.cli <paso>
PY ?= python

.PHONY: install test lint prepare train compare experiments interpret figures report all

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests

prepare:
	$(PY) -m desercion.cli prepare

train:
	$(PY) -m desercion.cli train

compare:
	$(PY) -m desercion.cli compare

experiments:
	$(PY) -m desercion.cli experiments

interpret:
	$(PY) -m desercion.cli interpret

figures:
	$(PY) -m desercion.cli figures

report:
	$(PY) -m desercion.cli report

all:
	$(PY) -m desercion.cli all
