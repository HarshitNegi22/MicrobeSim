# Operations guide

What a deployment needs beyond `streamlit run`: sign-in, logging, health checks, metrics, uploads, audit,
model review, background jobs and CI. Everything here is configuration through environment variables;
nothing needs extra Python packages.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MICROBESIM_STATE_DIR` | `data/` | Ledger, accounts, audit trail, registry, local model cache. Must not be group/world-writable. |
| `MICROBESIM_AUTH_MODE` | `local` | `local` = username/password; `oidc` = single sign-on (below). |
| `MICROBESIM_OIDC_ALLOWED_DOMAINS` / `_EMAILS` | empty | Comma-separated allow-lists. With neither set, SSO rejects everyone. |
| `MICROBESIM_OIDC_ALLOW_ANY` | `0` | `1` lets any verified identity in (not recommended). |
| `MICROBESIM_OIDC_TRUST_UNVERIFIED` | `0` | `1` accepts e-mails the provider did not verify (not recommended). |
| `MICROBESIM_LOG_LEVEL` | `INFO` | JSON logs to stderr (`WARNING` for the CLIs). |
| `MICROBESIM_OPS_PORT` | unset | Enables `/healthz`, `/readyz`, `/metrics` on this port. |
| `MICROBESIM_OPS_HOST` | `127.0.0.1` | Bind address for the ops endpoints. Use `0.0.0.0` only inside a private network. |
| `MICROBESIM_OPS_TOKEN` | unset | If set, `/readyz` and `/metrics` need `Authorization: Bearer <token>`. |
| `SENTRY_DSN` | unset | Enables Sentry if `sentry_sdk` is installed (PII is not sent). |
| `MICROBESIM_UPLOAD_SCAN_CMD` | unset | Virus-scanner command, e.g. `clamscan --no-summary {path}`. Fails closed. |
| `MICROBESIM_UPLOAD_SCAN_TIMEOUT` | `60` | Seconds before a scan counts as failed. |
| `MICROBESIM_FOUR_EYES` | `1` | Reviewer must differ from the model author; `0` for single-person installs. |

## Single sign-on (OpenID Connect)
1. Register the app with your identity provider (Entra ID, Okta, Google, Keycloak...). Redirect URI:
   `https://<your-host>/oauth2callback`.
2. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill it in.
3. Set `MICROBESIM_AUTH_MODE=oidc` and an allow-list.
4. A user's internal id is a hash of issuer + subject, so identities cannot be chosen or collided. SSO accounts
   have no password and can never sign in through the password form; `oidc-` usernames are reserved.
5. The idle (30 min) and maximum (12 h) session limits still apply, and expiry also signs out of the provider.

Status: the identity-mapping rules are unit-tested, but I could not test the flow against a real identity
provider. Try it in staging with one test user before relying on it.

## Health and metrics
Start with `MICROBESIM_OPS_PORT=9100`. Never expose this port to the internet. Streamlit only runs the app
script when a user session begins, so the endpoints come up with the **first session**, not at process launch;
use Streamlit's `/_stcore/health` for container liveness before that.

* `GET /healthz`: process is alive (no secret needed).
* `GET /readyz`: 200 when the state directory is writable and the ledger opens; 503 otherwise. It also
  reports whether pretrained models match (`ok`/`stale`/`missing`); that does not fail readiness because the app
  trains on demand.
* `GET /metrics`: Prometheus text format: `microbesim_operation_seconds` (histogram by operation),
  `microbesim_operation_errors_total`, `microbesim_model_loads_total{source=pretrained|cache|trained}`,
  `microbesim_uploads_total{outcome}`, `microbesim_audit_write_failures_total`.

Streamlit's own `/_stcore/health` (used by the Docker `HEALTHCHECK`) is unchanged.
Metrics are per process and reset on restart; that is fine for Prometheus counters.

## Audit trail
`audit_log` in the ledger database records sign-ins (including denied attempts), account creation, password
resets, session expiry, saved/changed/deleted runs, projects and profiles, model registration and review.
Rows are hash-chained and SQLite triggers refuse UPDATE/DELETE.

    python scripts/model_registry.py audit --limit 50
    python scripts/model_registry.py audit --verify      # exit 2 if the chain is broken

Limits: someone with write access to the database file can rebuild the whole chain. For stronger assurance ship
the JSON log lines (`"event": "audit"`) to storage the application cannot modify.

## Model registry and review
Every model artifact has a registry entry: dataset hash, row count, training settings, validation metrics,
code digest, library versions, author, status. Re-training with changed data or code creates a new entry.

    python scripts/model_registry.py sync
    python scripts/model_registry.py list --status draft
    python scripts/model_registry.py card <signature>        # model card (markdown)
    python scripts/model_registry.py approve <signature> --reviewer alice --note "checked parity plots"

Approval needs a note and a reviewer other than the author. The Predict page labels each result
*Exploratory*, *Validated* or *Out of scope*: "Validated" requires an approved model, validation R2 >= 0.5,
at least 30 training rows, conditions well inside the data and a reasonably narrow interval. Even then the page
states that a qualified person must review the output.

## Uploads
Files are checked before any parser runs: extension allow-list, 25 MB cap, content must match the extension,
no macros/ActiveX/embedded objects/external links, no path traversal, entry-count limit, and an optional virus
scanner (fail closed). The accepted file's SHA-256 can be recorded. Parsing still happens in the app process;
for untrusted users run the app in a container with no network access to internal systems.

## Background jobs
"Prepare pretrained models now" runs in a background worker with live progress, so the page stays responsive.
Jobs are held in memory of one process. Several replicas would need an external queue (RQ/Celery).

## CI/CD
`.github/workflows/ci.yml`: pinned install, environment validation, blocking lint for real errors, style report,
format check on new modules, tests, a gate that fails if the pretrained models do not match the code and
libraries in the commit, then a Docker build + health-check smoke test. `release.yml` publishes a tagged image to
GHCR. Deployment to staging/production is intentionally not scripted because it depends on your platform.

Before deploying anywhere: `python scripts/validate_environment.py`.
