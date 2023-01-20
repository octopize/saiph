SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

DOCS_REQUIREMENTS := docs/requirements.txt

install:  ## Install the stack
	pre-commit install --hook-type commit-msg
	if ! poetry lock --check 2> /dev/null; then poetry lock --no-update && poetry install --extras "matplotlib" --sync; fi;

.PHONY: install

notebook:  ## Run the notebook
	poetry run jupyter notebook
.PHONY: notebook

docs:  ## Build docs
	poetry export -f requirements.txt --output $(DOCS_REQUIREMENTS) --with dev --extras matplotlib --without-hashes 
	cat $(DOCS_REQUIREMENTS) | grep 'matplotlib\|sphinx-gallery'  > docs/tmp.txt
	mv docs/tmp.txt $(DOCS_REQUIREMENTS)

	poetry run sphinx-build -b html docs build/docs
.PHONY: docs

docs-open:  ## Open docs
	poetry run python -m webbrowser -t "file://$(abspath build/docs)/index.html"
.PHONY: docs-open

##@ Tests

ci: typecheck lint docs test  ## Run all checks
.PHONY: ci

lci: lint-fix ci ## Autofix then run CI
.PHONY: lci

lint:  ## Run linting
	poetry run black --check saiph
	poetry run flake8 saiph
	poetry run bandit -c bandit.yaml -r saiph bin
.PHONY: lint

lint-fix:  ## Run autoformatters
	poetry run black .
	poetry run isort .
.PHONY: lint-fix

typecheck:  ## Run typechecking
	poetry run mypy --show-error-codes --pretty saiph
.PHONY: typecheck

test:  ## Run tests
	poetry run pytest --benchmark-skip saiph
.PHONY: test

test-benchmark: ./tmp/fake_1k.csv ./tmp/fake_10k.csv ## Run benchmark with smaller files, often.
	@echo "Run manually with --benchmark-cox@x@gainst last run test."
	poetry run pytest --benchmark-only -m "not slow_benchmark" --benchmark-autosave --benchmark-warmup=on --benchmark-warmup-iterations=5 --benchmark-min-rounds=10 --benchmark-max-time=10
.PHONY: test-benchmark

test-benchmark-big: ./tmp/fake_1000000.csv  ## Run benchmark with a big CSV file
	@echo "Run manually with --benchmark-compare to check against last run test."
	poetry run pytest --benchmark-only -m "slow_benchmark" --benchmark-autosave
.PHONY: test-benchmark-

compare-benchmarks: ## Compare benchmarks with your previous ones
	poetry run py.test-benchmark compare --group-by=func --sort=name --columns=min,median,mean,stddev,rounds
.PHONY: compare-benchmarks

date := file_$(shell date +%FT%T%Z)
profile-cpu: ./tmp/fake_1000000.csv  ## Profile CPU usage
	sudo poetry run py-spy record -f speedscope -o "tmp/profile_${date}" -- python saiph/tests/profile_cpu.py False
.PHONY: profile-cpu

profile-memory: ./tmp/fake_1000000.csv  ## Profile memory usage
	poetry run fil-profile python saiph/tests/profile_memory.py True
.PHONY: profile-memory

./tmp/fake_1000000.csv: 
	poetry run python ./bin/create_csv.py --row-count 1000000 $@

./tmp/fake_10k.csv:
	poetry run python ./bin/create_csv.py --row-count 10000 $@

./tmp/fake_1k.csv:
	poetry run python ./bin/create_csv.py --row-count 1000 $@

.DEFAULT_GOAL := help
help: Makefile
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make \034[36m<target>\033[0m\n"} /^[\/\.a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)


