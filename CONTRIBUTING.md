# Contributing to sparkfit

Thanks for your interest in improving sparkfit. Contributions of all kinds are
welcome: bug reports, new models, better memory and throughput modeling, and docs.

By participating in this project you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

```bash
git clone https://github.com/engineering87/sparkfit.git
cd sparkfit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running the checks

```bash
ruff check .      # lint
mypy src          # type check
pytest -q         # tests
```

All three run in CI on Python 3.8 to 3.12, so please make sure they pass locally before
opening a pull request.

## Adding a model to the built-in database

Models live in the `MODELS` dict in `src/sparkfit.py`. Each entry needs the fields
used by the memory and KV-cache math:

```python
"my-model-7b": dict(total_b=7.0, active_b=7.0, layers=32, hidden=4096,
                    kv_heads=8, head_dim=128),
```

For Mixture-of-Experts models, set `active_b` to the number of active parameters
per token and add `moe=True`. You can read most of these values from a model's
`config.json` on Hugging Face (`num_hidden_layers`, `hidden_size`,
`num_key_value_heads`, and so on), the same fields sparkfit's auto-fetch uses.

If you add a model, please also add or update a test in `tests/test_sparkfit.py`.

## Modeling changes

sparkfit's value is in being a transparent planner: the assumptions
(quantization bit-widths, OS reserve, bandwidth efficiency, KV-cache formula) are
all explicit constants near the top of `src/sparkfit.py`. If you change the math,
please update both the README "Methodology" section and the tests.

## Pull requests

- Keep changes focused and described clearly.
- Use [Conventional Commits](https://www.conventionalcommits.org/) where you can
  (`feat:`, `fix:`, `docs:`, and so on).
- Add an entry under "Unreleased" in `CHANGELOG.md`.

## Releases

Every notable change is recorded under "Unreleased" in `CHANGELOG.md`. To cut a
release, move that section under a dated `X.Y.Z` heading, bump the version in
`pyproject.toml` and `src/sparkfit.py`, and add longer-form notes as
`docs/releases/vX.Y.Z.md` (follow the structure of the existing files). Publish
the GitHub release with `gh release create vX.Y.Z --notes-file docs/releases/vX.Y.Z.md`.

Pushing a `vX.Y.Z` tag triggers `.github/workflows/release.yml`, which builds the
sdist and wheel and publishes them to PyPI via
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC, no API
token). This requires a one-time setup on PyPI: register this repository and the
`release.yml` workflow as a trusted publisher for the `sparkfit` project, under a
release environment named `pypi`.
