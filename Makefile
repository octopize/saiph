SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

install:  ## Install the stack
	pre-commit install --hook-type commit-msg
	poetry install
.PHONY: install

notebook:  ## Run the notebook
	poetry run jupyter notebook
.PHONY: notebook

docs:  ## Build docs
	poetry run sphinx-build -b html docs build/docs
.PHONY: docs

docs-open:  ## Open docs
	python -m webbrowser -t "file://$(abspath build/docs)/index.html"
.PHONY: docs-open

##@ Tests

ci: lint typecheck docs test  ## Run all checks
.PHONY: ci

lci: lint-fix ci ## Autofix then run CI
.PHONY: lci

lint:  ## Run linting
	poetry run black --check .
	poetry run flake8 .
.PHONY: lint

lint-fix:  ## Run autoformatters
	poetry run black .
	poetry run isort .
.PHONY: lint-fix

typecheck:  ## Run typechecking
	poetry run mypy --show-error-codes --pretty .
.PHONY: typecheck

test:  ## Run tests
	poetry run pytest saiph
.PHONY: test

.DEFAULT_GOAL := help
help: Makefile
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make \034[36m<target>\033[0m\n"} /^[\/\.a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
