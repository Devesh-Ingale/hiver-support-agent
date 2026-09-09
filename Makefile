# Thin aliases for *nix reviewers. Windows users: run the `python -m support_agent.cli ...` forms directly.
PY ?= python

.PHONY: install eval figures test data threads profile docx

install:
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e . --no-deps

eval:            ## recompute every headline number from committed outputs (no API keys, no Kaggle)
	$(PY) -m support_agent.cli eval

figures:
	$(PY) -m support_agent.cli figures

test:
	$(PY) -m pytest

data:            ## needs Kaggle credentials
	$(PY) -m support_agent.cli data

threads:
	$(PY) -m support_agent.cli threads

profile:
	$(PY) -m support_agent.cli profile

docx:
	$(PY) -m support_agent.cli docx
