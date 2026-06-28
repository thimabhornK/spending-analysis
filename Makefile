.PHONY: help install data demo eval sweep test api clean

export PYTHONPATH := src:.

help:
	@echo "Spending Analysis - make targets"
	@echo "  install   install package + dev deps (pip install -e \".[dev]\")"
	@echo "  data      regenerate synthetic taxonomy, corpus, and PO items"
	@echo "  demo      run the CLI demo over representative items"
	@echo "  eval      run the evaluation harness (writes eval/results.json)"
	@echo "  sweep     sweep the auto-classify threshold (writes eval/sweep.json)"
	@echo "  test      run the test suite"
	@echo "  api       serve the FastAPI app on :8000"
	@echo "  clean     remove caches and build artifacts"

install:
	pip install -e ".[dev]"

data:
	python scripts/generate_synthetic_data.py

demo:
	python scripts/demo.py

eval:
	python -m eval.harness

sweep:
	python -m eval.sweep

test:
	python -m pytest tests/ -q

api:
	uvicorn spend_classifier.api.app:app --reload --port 8000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf src/*.egg-info *.egg-info build dist
