SHELL := /bin/bash

PY ?= python
PIP ?= pip

.PHONY: help install dev test lint figures tables clean precache train eval all-experiments

help:
	@echo "Targets:"
	@echo "  install        Install runtime dependencies (requirements.txt)"
	@echo "  dev            Editable install with dev extras (pytest, ruff)"
	@echo "  test           Run the test suite"
	@echo "  lint           Run ruff"
	@echo "  figures        Re-render every PNG under results/figures/"
	@echo "  tables         Re-emit canonical CSVs under results/tables/"
	@echo "  precache       Pre-cache features (DATA_ROOT, CACHE_ROOT required)"
	@echo "  train          Train the full V+A+T model (DATA_ROOT, CACHE_ROOT required)"
	@echo "  eval           Evaluate a checkpoint (CKPT=...)"
	@echo "  all-experiments  Run scripts/run_all_experiments.sh"
	@echo "  clean          Remove generated tables/figures and __pycache__"

install:
	$(PIP) install -r requirements.txt

dev:
	$(PIP) install -e .[dev]

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests scripts

figures tables:
	$(PY) scripts/generate_canonical_results.py

precache:
	$(PY) scripts/precache_features.py --data-root $(DATA_ROOT) --cache-root $(CACHE_ROOT)

train:
	$(PY) scripts/train_full.py --data-root $(DATA_ROOT) --cache-root $(CACHE_ROOT)

eval:
	$(PY) scripts/eval_full.py --checkpoint $(CKPT) --data-root $(DATA_ROOT) --cache-root $(CACHE_ROOT)

all-experiments:
	bash scripts/run_all_experiments.sh

clean:
	rm -rf results/tables/*.csv results/figures/*.png
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
