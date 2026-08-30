from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .paths import PATHS


class SecretVault:
    def __init__(self, key_path: Path | None = None) -> None:
        self.key_path = key_path or PATHS.secrets / "master.key"
        self.key = self._load_or_create_key()

    def _load_or_create_key(self) -> bytes:
        if self.key_path.exists():
            return base64.urlsafe_b64decode(self.key_path.read_bytes())
        key = AESGCM.generate_key(bit_length=256)
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        self.key_path.write_bytes(base64.urlsafe_b64encode(key))
        try:
            self.key_path.chmod(0o600)
        except OSError:
            pass
        return key

    def encrypt(self, value: str) -> str:
        if not value:
            return ""
        nonce = os.urandom(12)
        ciphertext = AESGCM(self.key).encrypt(nonce, value.encode(), b"sandevistan-read")
        return base64.urlsafe_b64encode(nonce + ciphertext).decode()

    def decrypt(self, value: str) -> str:
        if not value:
            return ""
        raw = base64.urlsafe_b64decode(value.encode())
        return AESGCM(self.key).decrypt(raw[:12], raw[12:], b"sandevistan-read").decode()

    def session_token(self, access_key: str, ttl_seconds: int = 86400) -> str:
        expires = str(int(time.time()) + ttl_seconds)
        digest = hmac.new(self.key, f"{access_key}:{expires}".encode(), hashlib.sha256).hexdigest()
        return f"{expires}.{digest}"

    def verify_session(self, token: str, access_key: str) -> bool:
        try:
            expires, digest = token.split(".", 1)
            if int(expires) < time.time():
                return False
            expected = hmac.new(self.key, f"{access_key}:{expires}".encode(), hashlib.sha256).hexdigest()
            return hmac.compare_digest(digest, expected)
        except (ValueError, TypeError):
            return False


VAULT = SecretVault()
