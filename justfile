# justfile for saiph project

# List all available recipes
default:
    @just --list

# Install the stack
install:
    pre-commit install --hook-type commit-msg
    uv sync --extra matplotlib --group dev --group doc

# Prepare a new release of saiph
prepare-release:
    uv run python release.py --bump-type patch

# Run the notebook
notebook:
    uv run jupyter notebook

# Build docs
docs:
    #!/usr/bin/env bash
    set -euo pipefail
    DOCS_REQUIREMENTS="docs/requirements.txt"
    uv pip compile pyproject.toml --group doc --extra matplotlib --output-file "$DOCS_REQUIREMENTS"
    grep -E 'matplotlib|sphinx-gallery' "$DOCS_REQUIREMENTS" > docs/tmp.txt
    mv docs/tmp.txt "$DOCS_REQUIREMENTS"
    uv run sphinx-build -b html docs build/docs

# Open docs
docs-open:
    uv run python -m webbrowser -t "file://{{justfile_directory()}}/build/docs/index.html"

# Run all checks
ci: typecheck lint docs test

# Autofix then run CI
lci: lint-fix ci

# Run linting
lint:
    uv run ruff check saiph bin
    uv run ruff format --check saiph bin

# Run autoformatters
lint-fix:
    uv run ruff check --fix saiph bin
    uv run ruff format saiph bin

# Run typechecking
typecheck:
    uv run mypy --show-error-codes --pretty saiph

# Run tests
test:
    uv run pytest --benchmark-skip saiph

# Run benchmark with smaller files, often
test-benchmark: _ensure-fake-1k _ensure-fake-10k
    @echo "Run manually with --benchmark-compare to check against last run test."
    uv run pytest --benchmark-only -m "not slow_benchmark" --benchmark-autosave --benchmark-warmup=on --benchmark-warmup-iterations=5 --benchmark-min-rounds=10 --benchmark-max-time=10

# Run benchmark with a big CSV file
test-benchmark-big: _ensure-fake-1000000
    @echo "Run manually with --benchmark-compare to check against last run test."
    uv run pytest --benchmark-only -m "slow_benchmark" --benchmark-autosave

# Compare benchmarks with your previous ones
compare-benchmarks:
    uv run py.test-benchmark compare --group-by=func --sort=name --columns=min,median,mean,stddev,rounds

# Profile CPU usage
profile-cpu: _ensure-fake-1000000
    #!/usr/bin/env bash
    set -euo pipefail
    date="file_$(date +%FT%T%Z)"
    sudo uv run py-spy record -f speedscope -o "tmp/profile_${date}" -- python saiph/tests/profile_cpu.py False

# Profile memory usage
profile-memory: _ensure-fake-1000000
    uv run fil-profile python saiph/tests/profile_memory.py True

# Helper recipes (private)
_ensure-fake-1000000:
    #!/usr/bin/env bash
    if [ ! -f ./tmp/fake_1000000.csv ]; then
        uv run python ./bin/create_csv.py --row-count 1000000 ./tmp/fake_1000000.csv
    fi

_ensure-fake-10k:
    #!/usr/bin/env bash
    if [ ! -f ./tmp/fake_10k.csv ]; then
        uv run python ./bin/create_csv.py --row-count 10000 ./tmp/fake_10k.csv
    fi

_ensure-fake-1k:
    #!/usr/bin/env bash
    if [ ! -f ./tmp/fake_1k.csv ]; then
        uv run python ./bin/create_csv.py --row-count 1000 ./tmp/fake_1k.csv
    fi
