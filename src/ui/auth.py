"""Sign-in and account-creation screen for the Streamlit app."""

from __future__ import annotations

import time

import streamlit as st

from src import audit, oidc
from src.accounts import (
    AccountError,
    authenticate,
    create_account,
    ensure_external_account,
    initialize_accounts,
)

_SESSION_IDLE_SECONDS = 30 * 60
_SESSION_MAX_SECONDS = 12 * 60 * 60


def _session_is_valid(authenticated_at: object, last_activity_at: object, now: float) -> bool:
    return (
        isinstance(authenticated_at, (int, float))
        and isinstance(last_activity_at, (int, float))
        and authenticated_at <= last_activity_at <= now
        and now - last_activity_at < _SESSION_IDLE_SECONDS
        and now - authenticated_at < _SESSION_MAX_SECONDS
    )


def _render_oidc_login(now: float, expired: bool) -> bool:
    """Single sign-on through Streamlit's native OIDC support; passwords never reach this app."""
    user = getattr(st, "user", None)
    if getattr(user, "is_logged_in", False):
        claims = dict(user.to_dict())
        try:
            identity = oidc.identity_from_claims(claims)
        except oidc.OidcRejected as exc:
            audit.record(
                None, "auth.login", str(claims.get("email", ""))[:64], "denied", {"mode": "oidc"}
            )
            st.error(str(exc))
            st.button("Sign out", on_click=st.logout)
            return False
        account = ensure_external_account(identity.user_id, identity.username)
        st.session_state.update(
            user_id=account.user_id,
            username=identity.name,
            authenticated_at=now,
            last_activity_at=now,
        )
        audit.record(account.user_id, "auth.login", account.username, detail={"mode": "oidc"})
        st.rerun()
    if expired:
        st.warning("Your session expired. Sign in again to continue.")
    st.title("Sign in")
    st.write("Sign in with your organisation account to open your private analysis workspace.")
    st.button("Sign in with your organisation account", type="primary", on_click=st.login)
    return False


def render_login() -> bool:
    """Render account access and return whether the current session is authenticated."""
    sso = oidc.auth_mode() == "oidc"
    if not sso:
        initialize_accounts()
    now = time.time()
    expired = False
    if st.session_state.get("user_id") and st.session_state.get("username"):
        authenticated_at = st.session_state.get("authenticated_at")
        last_activity_at = st.session_state.get("last_activity_at")
        if _session_is_valid(authenticated_at, last_activity_at, now):
            st.session_state["last_activity_at"] = now
            return True
        audit.record(st.session_state.get("user_id"), "auth.session_expired", "")
        st.session_state.clear()
        expired = True
        if sso:
            st.logout()  # also end the identity-provider session so it cannot silently sign back in

    if sso:
        return _render_oidc_login(now, expired)

    st.markdown(
        """
        <style>
        .st-key-ms-auth-card { max-width:650px; margin:8vh auto 2rem; padding:2rem;
          border:1px solid #202A3D; border-radius:22px; background:#0E1524; }
        .ms-auth-brand { color:#00D6C2; font-weight:750; letter-spacing:.08em; }
        .ms-auth-copy { margin:.4rem 0 1.4rem; color:#91A4C0; line-height:1.6; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container(key="ms-auth-card"):
        if expired:
            st.warning("Your session expired. Sign in again to continue.")
        st.markdown('<div class="ms-auth-brand">⚗ &nbsp; MICROBESIM</div>', unsafe_allow_html=True)
        st.title("Your private analysis workspace")
        st.markdown(
            '<p class="ms-auth-copy">Sign in to see your saved analyses, or create an account '
            "to keep your history separate from other users on this installation.</p>",
            unsafe_allow_html=True,
        )
        sign_in, register = st.tabs(["Sign in", "Create account"])
        with sign_in:
            with st.form("account_sign_in"):
                username = st.text_input("Username", autocomplete="username")
                password = st.text_input(
                    "Password", type="password", autocomplete="current-password"
                )
                submitted = st.form_submit_button("Sign in", type="primary", width="stretch")
            if submitted:
                account = authenticate(username, password)
                if account is None:
                    audit.record(None, "auth.login", username.strip().casefold()[:32], "denied")
                    st.error("Invalid username or password, or sign-in is temporarily locked.")
                else:
                    audit.record(account.user_id, "auth.login", account.username)
                    st.session_state["user_id"] = account.user_id
                    st.session_state["username"] = account.username
                    st.session_state["authenticated_at"] = now
                    st.session_state["last_activity_at"] = now
                    st.rerun()

        with register:
            st.caption("Usernames are 3–32 letters, numbers, underscores or hyphens.")
            st.caption("Passwords must contain at least 12 characters.")
            with st.form("account_create"):
                username = st.text_input("Choose a username", autocomplete="username")
                password = st.text_input(
                    "Choose a password", type="password", autocomplete="new-password"
                )
                password_confirm = st.text_input(
                    "Confirm password", type="password", autocomplete="new-password"
                )
                submitted = st.form_submit_button("Create account", type="primary", width="stretch")
            if submitted:
                if password != password_confirm:
                    st.error("The passwords do not match.")
                else:
                    try:
                        account = create_account(username, password)
                    except AccountError as exc:
                        st.error(str(exc))
                    else:
                        audit.record(account.user_id, "account.create", account.username)
                        st.session_state["user_id"] = account.user_id
                        st.session_state["username"] = account.username
                        st.session_state["authenticated_at"] = now
                        st.session_state["last_activity_at"] = now
                        st.rerun()
    return False


def logout() -> None:
    """Discard this browser session's authentication and workflow state."""
    audit.record(st.session_state.get("user_id"), "auth.logout", "")
    st.session_state.clear()
    if oidc.auth_mode() == "oidc":
        st.logout()
