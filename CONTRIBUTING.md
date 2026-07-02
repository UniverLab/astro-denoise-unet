# Contributing to astro-denoise-unet

Thank you for your interest in contributing!

## Development workflow

```bash
git clone https://github.com/UniverLab/astro-denoise-unet.git
cd astro-denoise-unet
uv sync --all-extras
uv run pytest
```

Branch off `develop` for all changes. PRs go to `develop`, not `main`.

## CI/CD and required secrets

This project uses GitHub Actions for CI (`.github/workflows/ci.yml`), delegating
to the centralized [UniverLab/workflows](https://github.com/UniverLab/workflows)
reusable workflow.

### Required repository secrets

| Secret | Description | Where to get it |
|---|---|---|
| `PYPI_API_TOKEN` | API token to publish packages to [PyPI](https://pypi.org) | [pypi.org/manage/account/token](https://pypi.org/manage/account/token/) → API Tokens → New Token |

Configure secrets at:
`https://github.com/UniverLab/astro-denoise-unet/settings/secrets/actions`

> **Tip:** If you use [ghscaff](https://github.com/UniverLab/ghscaff), you can run `ghscaff apply` to configure missing secrets interactively.

## Release process

Releases are automated via the `publish.yml` workflow. To cut a release:

1. Bump the version in `pyproject.toml`
2. Commit: `chore: release vX.Y.Z`
3. Open PR to `main` — on merge, the workflow publishes to PyPI

## Code style

```bash
uv run ruff format .
uv run ruff check .
uv run mypy src/
uv run pytest
```

All four must pass before merging.
