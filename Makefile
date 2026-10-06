# Reproduce every table and figure in docs/results.md from raw PhysioNet data.
# usage: make reproduce DATA_DIR=/path/to/physionet   (local EDF files S###R##.edf)
PY ?= python
DATA_DIR ?=
CONFIG ?= configs/benchmark.yaml
CFG_DATA = $(if $(DATA_DIR),--data-dir $(DATA_DIR),)

.PHONY: reproduce benchmark simulate report test env
reproduce: benchmark simulate report

benchmark:
	$(PY) scripts/run_benchmark.py --config $(CONFIG) --out results/main $(CFG_DATA)

simulate:
	$(PY) scripts/run_simulations.py --config $(CONFIG) --out results/sim --trials results/main/per_trial.csv $(CFG_DATA)

report:
	$(PY) scripts/make_report.py --main results/main --sim results/sim --out docs/results.md

test:
	$(PY) -m pytest -q

env:  # record the exact environment next to the results
	$(PY) -m pip freeze > results/environment.txt
