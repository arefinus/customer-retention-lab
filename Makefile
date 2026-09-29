PY ?= python

.PHONY: setup smoke demo test evaluate train build-features rank figures fixtures

setup:
	pip install -e . && pip install pytest

smoke:
	$(PY) -m customer_retention_lab smoke

demo:
	$(PY) -m customer_retention_lab demo

test:
	$(PY) -m pytest -q

build-features:
	$(PY) -m customer_retention_lab build-features

# Fits the two baselines and the calibrator on the fixture (seconds, CPU) and writes predictions.
train:
	$(PY) -m customer_retention_lab train

evaluate:
	$(PY) -m customer_retention_lab evaluate --predictions examples/output/predictions_test.csv

rank:
	$(PY) -m customer_retention_lab rank

figures:
	$(PY) -m customer_retention_lab figures

fixtures:
	$(PY) tools/make_fixture.py --seed 42
