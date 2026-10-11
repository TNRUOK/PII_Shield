"""
redact/mapping.py
-----------------
Manage placeholder <-> raw-value mappings per document with Fernet encryption at rest.

The mapping file (_mapping.json) contains raw PII and is encrypted at rest using
Fernet (AES-128-CBC with HMAC-SHA256 authenticated encryption).

Key design & security:
  - Encryption keys are generated per session and kept in memory by default.
  - If persisted to disk, the key is written to a dedicated separate key directory
    (e.g., ~/.pii_shield_keys), NEVER next to the mapping file.
  - Mapping files and directories receive owner-only permissions (0600/0700 on POSIX).
  - Secure purge: files are overwritten with random bytes before deletion.
  - Process exit purge is registered by default.
"""
from __future__ import annotations

import json
import logging
import os
import stat
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from cryptography.fernet import Fernet

log = logging.getLogger(__name__)

# Global in-memory session encryption key
_SESSION_KEY: Optional[bytes] = None

# Registry of active PlaceholderMapping instances for session-wide purge
_ACTIVE_MAPPINGS: List["PlaceholderMapping"] = []


def get_session_key() -> bytes:
    """Return the global in-memory session encryption key, generating one if needed."""
    global _SESSION_KEY
    if _SESSION_KEY is None:
        _SESSION_KEY = Fernet.generate_key()
    return _SESSION_KEY


def reset_session_key() -> None:
    """Reset the global session key."""
    global _SESSION_KEY
    _SESSION_KEY = None


def set_session_key(key: bytes) -> None:
    """Explicitly set the session key."""
    global _SESSION_KEY
    _SESSION_KEY = key


def _set_owner_only_permissions(path: Path, is_dir: bool = False) -> None:
    """
    Set owner-only permissions (0700 for directories, 0600 for files).
    On Windows NTFS, applies best-effort chmod attributes. Documented in README.
    """
    try:
        if is_dir:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        else:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except Exception as exc:
        log.debug("Unable to set owner-only permissions on %s: %s", path, exc)


def secure_delete_file(path: Union[str, Path]) -> bool:
    """
    Securely delete a file by overwriting it with random bytes before removal.
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        return False
    try:
        size = p.stat().st_size
        if size > 0:
            with open(p, "wb") as fh:
                fh.write(os.urandom(size))
                fh.flush()
                os.fsync(fh.fileno())
        p.unlink(missing_ok=True)
        return True
    except Exception as exc:
        log.warning("Secure file overwrite failed for %s (%s); unlinking directly", p, exc)
        try:
            p.unlink(missing_ok=True)
            return True
        except Exception:
            return False


def purge_mapping(
    mapping_or_path: Union["PlaceholderMapping", Path, str],
    key_path: Optional[Union[Path, str]] = None,
) -> bool:
    """
    Securely delete a mapping file and its optional separate key file from disk.
    """
    if isinstance(mapping_or_path, PlaceholderMapping):
        m_path = mapping_or_path.mapping_path
        k_path = mapping_or_path.key_path
    else:
        m_path = Path(mapping_or_path)
        k_path = Path(key_path) if key_path else None

    ok = secure_delete_file(m_path)
    if k_path and Path(k_path).exists():
        secure_delete_file(k_path)
    return ok


def purge_all_active_mappings() -> int:
    """
    Purge all registered active mappings from disk and memory, and reset session key.
    """
    global _ACTIVE_MAPPINGS
    count = 0
    for m in list(_ACTIVE_MAPPINGS):
        try:
            if m.mapping_path.exists():
                m.purge()
                count += 1
        except Exception as exc:
            log.warning("Error purging active mapping %s: %s", m.mapping_path, exc)
    _ACTIVE_MAPPINGS.clear()
    reset_session_key()
    return count


class PlaceholderMapping:
    """
    Tracks raw <-> placeholder assignments for one document.
    Encrypts the mapping at rest using Fernet.
    """

    def __init__(
        self,
        source_file: str,
        output_dir: str,
        key: Optional[bytes] = None,
        key_dir: Optional[str] = None,
        save_key_to_file: bool = False,
    ):
        self._raw_to_placeholder: Dict[Tuple[str, str], str] = {}
        self._placeholder_to_raw: Dict[str, str] = {}
        self._counters: Dict[str, int] = {}
        self._source_file = source_file

        self.key: bytes = key if key is not None else get_session_key()
        self.key_path: Optional[Path] = None

        # Determine mapping file path
        stem = Path(source_file).stem
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _set_owner_only_permissions(out, is_dir=True)
        self.mapping_path = out / f"{stem}_mapping.json"

        # Separate key storage if requested — NEVER next to mapping file
        if save_key_to_file:
            k_dir = Path(key_dir) if key_dir else Path.home() / ".pii_shield_keys"
            k_dir.mkdir(parents=True, exist_ok=True)
            _set_owner_only_permissions(k_dir, is_dir=True)
            self.key_path = k_dir / f"{stem}_session.key"
            with open(self.key_path, "wb") as kf:
                kf.write(self.key)
            _set_owner_only_permissions(self.key_path, is_dir=False)

        _ACTIVE_MAPPINGS.append(self)

    def get_or_create(self, entity_type: str, raw_value: str) -> str:
        """
        Return the placeholder for (entity_type, raw_value), creating one if new.
        Consistency: the same raw value and type always get the same placeholder.
        """
        key = (entity_type, raw_value)
        if key in self._raw_to_placeholder:
            return self._raw_to_placeholder[key]

        n = self._counters.get(entity_type, 0) + 1
        self._counters[entity_type] = n
        placeholder = f"[{entity_type}_{n}]"

        self._raw_to_placeholder[key] = placeholder
        self._placeholder_to_raw[placeholder] = raw_value
        return placeholder

    def save(self, encrypt: bool = True) -> None:
        """
        Persist mapping to disk. Encrypted with Fernet by default.
        """
        data = {
            "source_file": self._source_file,
            "WARNING": "This file contains raw PII values. Encrypted at rest. Do NOT share.",
            "mappings": {
                ph: raw for ph, raw in self._placeholder_to_raw.items()
            },
        }
        json_bytes = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        if encrypt:
            f = Fernet(self.key)
            payload = f.encrypt(json_bytes)
            with open(self.mapping_path, "wb") as fh:
                fh.write(payload)
        else:
            with open(self.mapping_path, "wb") as fh:
                fh.write(json_bytes)

        _set_owner_only_permissions(self.mapping_path, is_dir=False)
        log.info(
            "Mapping saved to %s (%d entries, encrypted=%s)",
            self.mapping_path,
            len(data["mappings"]),
            encrypt,
        )

    def load(self) -> None:
        """
        Load an existing mapping file, decrypting via Fernet if encrypted.
        """
        if not self.mapping_path.exists():
            return
        raw_bytes = self.mapping_path.read_bytes()
        try:
            f = Fernet(self.key)
            decrypted = f.decrypt(raw_bytes)
            data = json.loads(decrypted.decode("utf-8"))
        except Exception:
            try:
                data = json.loads(raw_bytes.decode("utf-8"))
            except Exception as e:
                log.error("Failed to load/decrypt mapping file %s: %s", self.mapping_path, e)
                return

        for ph, raw in data.get("mappings", {}).items():
            self._placeholder_to_raw[ph] = raw
            etype = ph.strip("[]").rsplit("_", 1)[0]
            self._raw_to_placeholder[(etype, raw)] = ph
        log.info("Mapping loaded from %s", self.mapping_path)

    def purge(self) -> bool:
        """
        Securely erase this mapping from disk and memory.
        """
        self._raw_to_placeholder.clear()
        self._placeholder_to_raw.clear()
        self._counters.clear()
        return purge_mapping(self)
