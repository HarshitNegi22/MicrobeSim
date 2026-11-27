# Accounts, deployment and security notes

Operational detail that used to live in the README. For the production-hardening checklist see
[`OPERATIONS.md`](OPERATIONS.md) and [`PRODUCTION_ROADMAP.md`](PRODUCTION_ROADMAP.md).

## Interface, accounts and the saved-runs ledger

The console has an overview and three analysis workspaces (fit growth curves, predict growth
rate, and simulate a cold-chain failure), plus a saved-runs ledger. Each analysis page has an
inputs panel and live results; the simulator leads with a plain verdict and the key numbers.
Users create a local account or sign in with a password before opening the workspaces.
Passwords are stored as salted PBKDF2-HMAC-SHA256 hashes; analyses are associated with the
signed-in account, and the history page only reads or deletes that account's runs. Fits,
predictions, and simulations are saved locally only when you choose **Save to history** in the
workspace. Saved records are stored in `data/analysis_ledger.sqlite3`, and identical reruns are
deduplicated per account. The database is created on first use and is ignored by Git. No account
service or database is hosted remotely. Existing pre-account history remains unassigned and is
not exposed to new accounts.
On POSIX hosts, the ledger database and SQLite sidecars are restricted to the application
account, and shared-writable database directories are rejected. On Windows, restrict the
ledger and state-directory ACLs to the account running MicrobeSim.

Sign-in is throttled after five failed attempts for a username, with a five-minute temporary
lock, and browser sessions expire after 30 minutes idle or 12 hours total. Because this is a
local-only account system, operators must protect the host and database file. A trusted host
operator can reset a password offline with `python -m src.admin USERNAME`; the new password
is prompted without echoing it to the terminal.

This local SQLite login is suitable for separating users of a trusted, single-host installation,
not a hardened public multi-tenant deployment: anyone with filesystem access to the SQLite file
can access or modify its data. Username-based throttling can also temporarily lock an account if
someone deliberately submits repeated failures. Use appropriate host access controls and HTTPS if
exposing the app over a network; for public access, use a managed identity provider and an
application server designed for multi-tenant authorization.

Back up the active SQLite ledger with SQLite's backup API so committed WAL data is included; do
not copy only the `.sqlite3` file while the app is running:

```powershell
.\.venv\Scripts\python.exe -c "import os,sqlite3; os.umask(0o077); source=sqlite3.connect('data/analysis_ledger.sqlite3'); target=sqlite3.connect('analysis_ledger.backup.sqlite3'); source.backup(target); target.close(); source.close()"
```

To restore, stop MicrobeSim and use the same API with the backup as `source` and the configured
ledger path as `target`. Keep backups encrypted and access-controlled like the live database.

The dark console theme lives in `src/ui/theme.py` and `.streamlit/config.toml`; chart styling is
centralized in `apply_theme` in `src/ui/charts.py`.

The simulator includes a ready-made scenario gallery, timeline scrubber, reference-scenario
comparison, and reusable account-private product and environmental-history profiles. Saved runs
can be assigned to account-private project folders and annotated with searchable notes. Assay
results can be compared side by side, and analysis tables/trajectories can be exported to CSV,
Excel, or PDF with run identifiers and provenance details. The interface flags conditions outside
the selected training data and offers a first-run guide. These enhancements improve exploration
and traceability; they do not turn screening estimates into calibrated safety decisions.

## Model cache and upload limits

Trained growth-rate models and their benchmark results are persisted under
`trained_models/` in the configured state directory (by default, `data/trained_models/`).
The cache key includes the dataset contents, grouping labels, model selection, library
versions, and model-source code, so changed data or code triggers a fresh benchmark.
Keep this directory in trusted, writable storage; cached model artifacts must not be
loaded from an untrusted source. The application retains at most 20 persistent model-cache
artifacts and limits its in-memory training cache to eight entries with a one-hour expiry.
On POSIX hosts, the cache directory is restricted to the application account; on Windows,
restrict the state directory's ACL to the account running MicrobeSim.
Uploads are limited to 25 MB, 100,000 rows, and 100 columns; compressed Excel workbooks are
also limited to 100 MB expanded size. Growth-rate datasets are checked for finite numeric
inputs and documented pH / water-activity bounds before fitting. Kinetic uploads are capped
at 500 curves and 200,000 cleaned points per curve, including inserted gap-fill points.

## Docker

```bash
docker build -t microbesim .
docker volume create microbesim-data
docker run --rm -p 8501:8501 -v microbesim-data:/var/lib/microbesim microbesim
# http://localhost:8501
```

The container runs as an unprivileged user and stores the SQLite database in
`/var/lib/microbesim`. Mount that directory as a persistent volume to retain accounts and saved
analyses and trained-model caches across container replacement. `MICROBESIM_STATE_DIR` can
point to another writable directory; `MICROBESIM_LEDGER_PATH` can override the database filename
directly. Keep this mutable state separate from `/app/data`, which contains the bundled
datasets.
