"""Ed25519 signatures for evidence manifests.

Each workspace has its own signing key (`signing.key`, never exported). A pack carries the public key and a signature of
`manifest.json`, so anyone can check offline that the manifest (and through it every file hash) is unchanged since export.
A pack that carries its own key proves only that: to know the pack came from this installation, compare the fingerprint
with the one the operator published (`GET /api/evidence/signer`).
"""
from __future__ import annotations

import base64
import binascii
import contextlib
import hashlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

ALGORITHM = "ed25519"
KEY_NAME, PUB_NAME = "signing.key", "signing.pub"
_lock = threading.Lock()


class SigningError(Exception):
    pass


def _fingerprint(public: Ed25519PublicKey) -> str:
    raw = public.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()[:16]


def _pem(public: Ed25519PublicKey) -> bytes:
    return public.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)


def fingerprint_of(public_pem: bytes) -> str:
    """Fingerprint of a PEM public key; raises SigningError when it is not an Ed25519 key."""
    try:
        key = serialization.load_pem_public_key(public_pem)
    except ValueError as exc:
        raise SigningError("not a readable public key") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise SigningError("not an Ed25519 public key")
    return _fingerprint(key)


@dataclass(frozen=True)
class Signer:
    private: Ed25519PrivateKey

    @property
    def public_pem(self) -> bytes:
        return _pem(self.private.public_key())

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.private.public_key())

    def sign(self, message: bytes) -> bytes:
        """Base64 text of the signature (Ed25519 is deterministic: the same key and message give the same bytes)."""
        return base64.b64encode(self.private.sign(message)) + b"\n"


def load_or_create(workspace_root: Path) -> Signer:
    """The workspace's signing key, created on first use. The private key never leaves this folder."""
    path = workspace_root / KEY_NAME
    with _lock:
        if path.is_file():
            try:
                key = serialization.load_pem_private_key(path.read_bytes(), password=None)
            except ValueError as exc:
                raise SigningError(f"{KEY_NAME} is damaged; move it away to create a new key (old packs keep theirs)") from exc
            if not isinstance(key, Ed25519PrivateKey):
                raise SigningError(f"{KEY_NAME} is not an Ed25519 key")
            return Signer(key)
        key = Ed25519PrivateKey.generate()
        pem = key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption(),
        )
        workspace_root.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(pem)
        with contextlib.suppress(OSError):  # not every file system keeps permissions
            os.chmod(tmp, 0o600)
        os.replace(tmp, path)
        signer = Signer(key)
        (workspace_root / PUB_NAME).write_bytes(signer.public_pem)
        return signer


def verify(public_pem: bytes, message: bytes, signature_b64: bytes) -> bool:
    try:
        key = serialization.load_pem_public_key(public_pem)
        if not isinstance(key, Ed25519PublicKey):
            return False
        key.verify(base64.b64decode(signature_b64.strip(), validate=True), message)
    except (ValueError, binascii.Error, InvalidSignature):
        return False
    return True
