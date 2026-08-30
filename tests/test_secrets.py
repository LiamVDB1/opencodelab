import json
import os
from pathlib import Path

from keyring.errors import KeyringError

from opencodelab.secrets import SecretStore


def test_keyring_failure_falls_back_to_private_atomic_file(monkeypatch, tmp_path: Path):
    store = SecretStore(tmp_path)
    monkeypatch.setattr("opencodelab.secrets.keyring.set_password", lambda *a: (_ for _ in ()).throw(KeyringError("no backend")))
    monkeypatch.setattr("opencodelab.secrets.keyring.get_password", lambda *a: (_ for _ in ()).throw(KeyringError("no backend")))

    location = store.set_api_key("profile", "secret-value")

    assert location == str(store.fallback_path)
    assert store.get_api_key("profile") == "secret-value"
    assert os.stat(store.fallback_path).st_mode & 0o777 == 0o600
    assert os.stat(tmp_path).st_mode & 0o777 == 0o700
    assert json.loads(store.fallback_path.read_text(encoding="utf-8")) == {"profile": "secret-value"}


def test_environment_secret_takes_precedence(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("OPENCODELAB_API_KEY", "  environment-secret  ")
    monkeypatch.setattr("opencodelab.secrets.keyring.get_password", lambda *a: "keyring-secret")
    assert SecretStore(tmp_path).get_api_key("profile") == "environment-secret"
    assert (
        SecretStore(tmp_path).get_api_key("profile", allow_environment=False)
        == "keyring-secret"
    )


def test_delete_removes_fallback_even_when_keyring_fails(monkeypatch, tmp_path: Path):
    store = SecretStore(tmp_path)
    store._fallback_write({"profile": "secret-value", "other": "keep"})
    monkeypatch.setattr("opencodelab.secrets.keyring.delete_password", lambda *a: (_ for _ in ()).throw(KeyringError("no backend")))

    store.delete_api_key("profile")

    assert json.loads(store.fallback_path.read_text(encoding="utf-8")) == {"other": "keep"}


def test_corrupt_fallback_is_treated_as_missing(tmp_path: Path):
    store = SecretStore(tmp_path)
    tmp_path.mkdir(exist_ok=True)
    store.fallback_path.write_text("not json", encoding="utf-8")
    assert store._fallback_get("profile") is None
