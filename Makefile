SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

install:  ## Install the stack
	poetry install
.PHONY: install

notebook:  ## Run the notebook
	poetry run jupyter notebook
.PHONY: notebook

docs:  ## Build docs
	poetry run sphinx-build -b html docs/source docs/build
.PHONY: docs

docs-open:  ## Open docs
	python -m webbrowser -t "file://$(abspath docs/build)/index.html"
.PHONY: docs-open

##@ Tests

ci: lint test  ## Run all checks
.PHONY: ci

lci: lint-fix ci  ## Autofix then run CI
.PHONY: lci

test:  ## Run tests
	poetry run pytest .
.PHONY: test

lint:  ## Run linting
	poetry run black --check .
	poetry run isort -c .
	poetry run flake8 .
	poetry run pydocstyle .
.PHONY: lint

lint-fix:  ## Run autoformatters
	poetry run black .
	poetry run isort .
.PHONY: lint-fix

.DEFAULT_GOAL := help
help: Makefile
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make \034[36m<target>\033[0m\n"} /^[\/\.a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
