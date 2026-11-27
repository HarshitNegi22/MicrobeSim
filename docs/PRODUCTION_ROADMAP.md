# Production roadmap and status of the review findings

An external review listed ten shortcomings. This table says what was **verified against the code**, what was
**done**, what is **partial**, and what is **not done and why**. Several claims were based on the README only.

| # | Finding | Reality in the code | Status |
|---|---|---|---|
| 1 | Single trusted host; local auth | Per-user row isolation already existed (every run/project/profile is scoped by `user_id`; PBKDF2-600k, throttling, session limits). **Added:** OIDC single sign-on mode with allow-lists, tamper-evident audit trail. | **Partial.** Not done: PostgreSQL/MySQL, shared team/project access (roles). OIDC is unit-tested but untested against a real provider. |
| 2 | Monolith | True. New logic lives in Streamlit-independent modules (`audit`, `registry`, `guardrails`, `jobs`, `upload_security`, `observability`, `oidc`) so a service layer can reuse them. | **Not done.** An API/service split is a rewrite that needs decisions (framework, deployment, auth) and a staging environment; doing it blind would risk the working app. |
| 3 | Weak model governance | A signature-keyed cache and manifest existed, no registry. **Added:** registry with dataset hash, settings, metrics, code digest, libraries, author, status; model cards; approval with four-eyes; lineage. | **Done** (CLI-based). No MLflow; no in-app approval screen. |
| 4 | Research-grade, not decision-grade | Outputs were already labelled nominal, with intervals and an out-of-range warning. **Added:** exploratory/validated/out-of-scope tiers, approval and validation requirements, mandatory human-review notice. | **Partial.** Intervals remain nominal, not calibrated; no alerting; this does not make the tool suitable for regulated decisions. |
| 5 | Security only for local use | Size/row/zip-bomb limits already existed. **Added:** content sniffing, macro/embedded/external-link rejection, path-traversal check, scanner hook (fail closed). | **Partial.** Not done: encryption at rest (use volume/disk encryption), process-level sandboxing of parsers. |
| 6 | No monitoring | The Dockerfile already had a health check. Logging/metrics were absent. **Added:** JSON logs, Prometheus metrics, `/healthz` `/readyz` `/metrics`, optional Sentry. | **Done**, except OpenTelemetry tracing and dashboards/alert rules. |
| 7 | Performance | Models are now pretrained (no training on visits); long jobs run in a background worker with progress. | **Partial.** Uploaded-data training still runs inline; no external queue; pagination not reviewed. |
| 8 | Deployment process | No CI existed. **Added:** GitHub Actions CI (blocking lint, tests, environment validation, stale-model gate, Docker smoke test), release workflow, dependabot. | **Done** except staging/production deployment (platform-specific) and type checking. |
| 9 | Shallow provenance | PDFs already carried timestamp, version and an input SHA-256. **Added:** per-model lineage in the registry, suspect-record list with reasons, build scripts with fixed seeds. | **Partial.** No notebook versioning or one-click provenance export. |
| 10 | Tool, not platform | Roadmap below. | **Plan only.** |

## Staged plan
1. **MVP / internal tool (now).** One container, SSO or local accounts, pretrained models, audit trail, CI. Gate: CI green, `validate_environment` clean, first reviewer approves models for the organisms you actually use.
2. **Pilot.** Add staging deployment with required reviewers, backups of the state directory, log shipping (audit lines to storage the app cannot modify), alert rules on `/metrics`, a real-IdP test of SSO. Gate: two weeks without data loss or sign-in problems; reviewers use the model cards.
3. **Controlled deployment.** Move the ledger/registry/audit to PostgreSQL behind a small storage interface, add roles and team projects, move training to an external queue, add calibration of prediction intervals against held-out data. Gate: documented validation per organism, penetration test.
4. **Production platform.** Extract an API/service layer, retire direct SQLite, formal change management for models and data (the registry and audit trail are the seed), data contracts for uploads. Required before any regulated use.
