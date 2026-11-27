"""Map OpenID Connect identity claims to a MicroBeSim user (optional single sign-on).

Enable with ``MICROBESIM_AUTH_MODE=oidc`` and Streamlit's native OIDC settings (``[auth]`` in
``.streamlit/secrets.toml``, see docs/OPERATIONS.md). Sign-in is then delegated to your identity provider
(Entra ID, Okta, Google, Keycloak ...), so passwords never reach this application. This module only decides
whether a verified identity may enter and derives a stable internal user id:

* the e-mail must be present and verified (``email_verified`` true) unless ``MICROBESIM_OIDC_TRUST_UNVERIFIED=1``;
* the e-mail domain must be in ``MICROBESIM_OIDC_ALLOWED_DOMAINS`` (comma separated) or
  ``MICROBESIM_OIDC_ALLOWED_EMAILS``;
  with neither set everyone is rejected, unless ``MICROBESIM_OIDC_ALLOW_ANY=1`` (not recommended);
* the user id is ``oidc-`` + a hash of issuer and subject, so it cannot be chosen or collided by a user.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Mapping, Set

EXTERNAL_PASSWORD_MARKER = "!external"
USER_ID_PREFIX = "oidc-"


class OidcRejected(ValueError):
    """The identity was refused; the message is safe to show."""


@dataclass(frozen=True)
class OidcIdentity:
    user_id: str
    username: str
    email: str
    name: str


def auth_mode() -> str:
    return (
        "oidc"
        if os.environ.get("MICROBESIM_AUTH_MODE", "local").strip().lower() == "oidc"
        else "local"
    )


def _csv_env(name: str) -> Set[str]:
    return {item.strip().lower() for item in os.environ.get(name, "").split(",") if item.strip()}


def identity_from_claims(claims: Mapping[str, object]) -> OidcIdentity:
    email = str(claims.get("email") or "").strip().lower()
    subject = str(claims.get("sub") or "").strip()
    issuer = str(claims.get("iss") or "").strip()
    if not email or "@" not in email or not subject:
        raise OidcRejected(
            "Your identity provider did not supply a usable e-mail address and subject."
        )
    verified = claims.get("email_verified")
    if verified is not True and os.environ.get("MICROBESIM_OIDC_TRUST_UNVERIFIED") != "1":
        raise OidcRejected("Your e-mail address is not verified by the identity provider.")
    domains, emails = _csv_env("MICROBESIM_OIDC_ALLOWED_DOMAINS"), _csv_env(
        "MICROBESIM_OIDC_ALLOWED_EMAILS"
    )
    if os.environ.get("MICROBESIM_OIDC_ALLOW_ANY") != "1":
        if email not in emails and email.rsplit("@", 1)[1] not in domains:
            raise OidcRejected(
                "Your account is not on the list of users allowed to use this application."
            )
    digest = hashlib.sha256(f"{issuer}|{subject}".encode("utf-8")).hexdigest()
    name = str(claims.get("name") or email.split("@", 1)[0])[:80]
    return OidcIdentity(
        user_id=USER_ID_PREFIX + digest[:32],
        username=USER_ID_PREFIX + digest[:12],
        email=email,
        name=name,
    )
