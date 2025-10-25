.PHONY: test lint run build-snapshot brief

PYTHON := python3

setup:
	$(PYTHON) -m venv .venv && . .venv/bin/activate && pip install -U pip
	pip install -r requirements.txt || true

lint:
	flake8 src || echo "flake8 not installed; skipping"
	black --check src || echo "black not installed; skipping"

# Run unit + integration tests
test:
	PYTHONPATH=. pytest -q

# Run a quick walk-forward (symbol from SETTINGS)
run:
	$(PYTHON) scripts/run_walkforward.py --symbol CL

# Build a curve snapshot
build-snapshot:
	$(PYTHON) scripts/build_curve_snapshot.py --symbol CL

# Generate PM brief tables from latest outputs
brief:
	$(PYTHON) scripts/gen_pm_brief.py --symbol CL
