# Contributing to Saiph

## Releasing a new version

```bash
VERSION="0.2.0"

# 1. Edit version in `pyproject.toml` and `saiph/__init__.py`

# 2. Add to next commit
git add pyproject.toml saiph/__init__.py

# 3. Commit
git commit -am "chore: release version $VERSION"

# 4. Tag
git tag $VERSION

# 5. Push
git push --tags
```
