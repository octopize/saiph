# Contributing to Saiph

## Releasing a new version

Depending on the version bump you want to do, you'll run

```bash
uv run python release.py --bump-type {patch, minor, major} # choose one
```

This will bump the version, commit it, and generate a tag and push it.

A Github action will then build that version and push it on `pypi`.