"""Strict checks for uploaded files, before any parser sees them.

Layers (each can reject; nothing is executed or evaluated):
  1. extension allow-list and size cap;
  2. content sniffing: the bytes must match the extension (a ``.xlsx`` must be a ZIP, a ``.csv`` must be
     text without NUL bytes, a ``.xls`` must have the OLE header);
  3. OOXML inspection: no macros (vbaProject.bin), ActiveX, embedded objects, external links, path
     traversal names, or implausible entry counts; legacy ``.xls`` containing VBA markers is rejected;
  4. optional virus-scanner hook: set ``MICROBESIM_UPLOAD_SCAN_CMD`` (e.g. ``clamscan --no-summary {path}``).
     The file is written to a private temp directory and the command must exit 0. If a scanner is
     configured but missing, slow or failing the upload is REJECTED (fail closed).
The SHA-256 of accepted content is returned for the audit trail.
"""

from __future__ import annotations

import hashlib
import io
import os
import shlex
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Optional

MAX_BYTES = 25 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 2000
ALLOWED_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xlsm", ".xls"}
_OLE_HEADER = bytes.fromhex("D0CF11E0A1B11AE1")
_FORBIDDEN_PARTS = (
    "vbaproject.bin",
    "/activex/",
    "/embeddings/",
    "/externallinks/",
    "oleobject",
)
_LEGACY_VBA_MARKERS = (b"_VBA_PROJECT", b"VBA/", b"Macros/")


class UploadRejected(ValueError):
    """The file failed a security check; the message is safe to show to the user."""


def _sniff_text(data: bytes) -> None:
    head = data[: 64 * 1024]
    if b"\x00" in head:
        raise UploadRejected("The file contains binary data but has a text extension.")
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            head.decode(encoding)
            return
        except UnicodeDecodeError:
            continue
    raise UploadRejected(
        "The file is not readable text."
    )  # pragma: no cover (latin-1 decodes anything)


def _inspect_ooxml(data: bytes) -> None:
    if not data.startswith(b"PK\x03\x04"):
        raise UploadRejected("The file is not a valid Excel workbook (not a ZIP container).")
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise UploadRejected("The Excel workbook is corrupt.") from exc
    with archive:
        entries = archive.infolist()
        if len(entries) > MAX_ARCHIVE_ENTRIES:
            raise UploadRejected("The Excel workbook contains an implausible number of parts.")
        for entry in entries:
            name = entry.filename.replace("\\", "/")
            lowered = "/" + name.lower().lstrip("/")
            if name.startswith("/") or ".." in name.split("/") or ":" in name.split("/")[0]:
                raise UploadRejected("The Excel workbook contains an unsafe path.")
            if any(part in lowered for part in _FORBIDDEN_PARTS):
                raise UploadRejected(
                    "Workbooks with macros, embedded objects or external links are not accepted. "
                    "Save a plain .xlsx or export a CSV."
                )


def _inspect_legacy_xls(data: bytes) -> None:
    if not data.startswith(_OLE_HEADER):
        raise UploadRejected("The file is not a valid legacy Excel (.xls) workbook.")
    if any(marker in data for marker in _LEGACY_VBA_MARKERS):
        raise UploadRejected("Legacy workbooks containing macros are not accepted.")


def _scan(data: bytes, filename: str) -> None:
    command = os.environ.get("MICROBESIM_UPLOAD_SCAN_CMD", "").strip()
    if not command:
        return
    timeout = float(os.environ.get("MICROBESIM_UPLOAD_SCAN_TIMEOUT", "60"))
    with tempfile.TemporaryDirectory(prefix="microbesim-scan-") as directory:
        os.chmod(directory, 0o700)
        suffix = (
            Path(filename).suffix.lower()
            if Path(filename).suffix.lower() in ALLOWED_EXTENSIONS
            else ".bin"
        )
        target = Path(directory) / f"upload{suffix}"
        target.write_bytes(data)
        try:
            argv = [part.replace("{path}", str(target)) for part in shlex.split(command)]
            completed = subprocess.run(  # noqa: S603 (argv list, no shell)
                argv, capture_output=True, timeout=timeout, check=False, cwd=directory
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise UploadRejected("The upload could not be scanned, so it was rejected.") from exc
        if completed.returncode != 0:
            raise UploadRejected("The upload was flagged by the virus scanner and rejected.")


def inspect_upload(filename: str, data: bytes) -> Dict[str, str]:
    """Validate an upload; return ``{"sha256", "extension"}`` or raise :class:`UploadRejected`."""
    extension = Path(str(filename or "")).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise UploadRejected(f"Files of type '{extension or 'unknown'}' are not accepted.")
    if len(data) == 0:
        raise UploadRejected("The file is empty.")
    if len(data) > MAX_BYTES:
        raise UploadRejected("The file exceeds the 25 MB limit.")
    if extension in (".xlsx", ".xlsm"):
        _inspect_ooxml(data)
    elif extension == ".xls":
        _inspect_legacy_xls(data)
    else:
        _sniff_text(data)
    _scan(data, filename)
    return {"sha256": hashlib.sha256(data).hexdigest(), "extension": extension}


def _as_bytes(value) -> bytes:
    return value.encode("utf-8") if isinstance(value, str) else bytes(value)


def source_bytes(source) -> Optional[bytes]:
    """Bytes of a path or file-like upload (text streams are UTF-8 encoded), restoring the stream position."""
    if isinstance(source, (str, os.PathLike)):
        return Path(source).read_bytes()
    if hasattr(source, "getvalue"):
        return _as_bytes(source.getvalue())
    if hasattr(source, "seek") and hasattr(source, "tell") and hasattr(source, "read"):
        position = source.tell()
        source.seek(0)
        try:
            return _as_bytes(source.read())
        finally:
            source.seek(position)
    return None
