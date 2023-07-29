.PHONY: clean clean-test clean-pyc clean-build lint lint-staged lint-diff test regen-jupytext

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint: # runs on all files (slow)
	pre-commit run --all-files --show-diff-on-failure

lint-staged: # runs on staged files only
	pre-commit run --show-diff-on-failure

lint-diff: # runs on files that have changed since upstream only
	pre-commit run --from-ref origin/HEAD --to-ref HEAD

test:
	pytest

regen-jupytext: # recreate all jupytext generated scripts
	rm -r notebooks_src
	mkdir notebooks_src
	jupytext --sync --pipe black notebooks/**/*.ipynb
