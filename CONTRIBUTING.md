# Contributing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
export PYTHONPATH=.
pytest                                   # must pass on a fresh clone with no local data
black src scripts tests && flake8 src scripts tests
```

* The test suite must not depend on ComBase-derived files. Use the `real_bundle` fixture (a synthetic stand-in) or mark a
  test `needs_combase` / `needs_expanded` so it is skipped when those local files are absent. See `tests/conftest.py`.
* Never commit `data/analysis_ledger.sqlite3`, trained or pretrained models, secrets, or ComBase-derived files. They are
  git-ignored; keep it that way.
* New external data needs a recorded licence that permits redistribution (`src/external_data.py` enforces this) and
  full provenance in `data/external/external_sources.csv`.
* Editing `src/models/*.py` or `src/model_cache.py` invalidates saved models by design; re-run
  `python scripts/pretrain_models.py` afterwards.
* Report security issues privately as described in [`SECURITY.md`](SECURITY.md).
