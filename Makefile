.PHONY: validate test preflight compile

PYTHON ?= python3

validate:
	$(PYTHON) scripts/vto.py validate --config configs/experiments/model_1_default.yaml
	$(PYTHON) scripts/vto.py validate --config configs/experiments/model_2_default.yaml
	$(PYTHON) scripts/vto.py validate --config configs/experiments/model_3_default.yaml

test:
	$(PYTHON) -m pytest -q

preflight:
	$(PYTHON) scripts/vto.py preflight --config configs/experiments/model_3_default.yaml

compile:
	$(PYTHON) -m compileall -q src scripts evaluation
