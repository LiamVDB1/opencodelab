from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import keyring
from keyring.errors import KeyringError

SERVICE = "opencodelab"


class SecretStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.fallback_path = root / "secrets.json"

    @staticmethod
    def _username(profile_name: str) -> str:
        return f"profile:{profile_name}:api-key"

    def get_api_key(
        self,
        profile_name: str,
        *,
        allow_environment: bool = True,
    ) -> str | None:
        if allow_environment:
            env_value = os.getenv("OPENCODELAB_API_KEY")
            if env_value:
                return env_value.strip()
        try:
            value = keyring.get_password(SERVICE, self._username(profile_name))
        except KeyringError:
            value = None
        if value:
            return value.strip()
        return self._fallback_get(profile_name)

    def set_api_key(self, profile_name: str, value: str) -> str:
        try:
            keyring.set_password(SERVICE, self._username(profile_name), value)
            return "system keyring"
        except KeyringError:
            self._fallback_set(profile_name, value)
            return str(self.fallback_path)

    def delete_api_key(self, profile_name: str) -> None:
        try:
            keyring.delete_password(SERVICE, self._username(profile_name))
        except KeyringError:
            pass
        data = self._fallback_load()
        if profile_name in data:
            del data[profile_name]
            self._fallback_write(data)

    def _fallback_load(self) -> dict[str, str]:
        if not self.fallback_path.exists():
            return {}
        try:
            raw = json.loads(self.fallback_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        return {str(k): str(v) for k, v in raw.items()}

    def _fallback_get(self, profile_name: str) -> str | None:
        return self._fallback_load().get(profile_name)

    def _fallback_set(self, profile_name: str, value: str) -> None:
        data = self._fallback_load()
        data[profile_name] = value
        self._fallback_write(data)

    def _fallback_write(self, data: dict[str, str]) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            self.root.chmod(0o700)
        except OSError:
            pass
        fd, temp_name = tempfile.mkstemp(prefix=".secrets-", dir=self.root)
        try:
            try:
                os.fchmod(fd, 0o600)
            except OSError:
                pass
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.fallback_path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
