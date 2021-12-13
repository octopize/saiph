# Contributing to Saiph

## Releasing a new version

```bash
VERSION="0.2.0"

# 1. Edit version in `pyproject.toml` and `saiph/__init__.py`

# 2. Commit
git commit -am "chore: releasing version $VERSION"

# 3. Tag
git tag $VERSION

# 4. Push
git push --tags
```
