# Security policy

MicrobeSim's built-in sign-in is a local, single-host account system. It is suitable for separating users of a trusted
installation, **not** for a hardened public multi-tenant deployment; use the optional OIDC single sign-on and an
appropriate reverse proxy with HTTPS for network exposure. See `docs/ACCOUNTS_AND_DEPLOYMENT.md` and `docs/OPERATIONS.md`.

* Pretrained and cached models are pickled (joblib) files. **Never load model files from an untrusted source.**
  This repository does not ship any.
* Please report vulnerabilities privately (GitHub "Report a vulnerability" under the Security tab) rather than in a
  public issue.
