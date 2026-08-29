from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import webbrowser
from pathlib import Path

_MODAL_URL = re.compile(r"https://[A-Za-z0-9.-]+\.modal\.run")
DEFAULT_MODAL_SIGNUP_URL = "https://modal.com"


def modal_command(*args: str) -> list[str]:
    return [sys.executable, "-m", "modal", *args]


def modal_is_authenticated() -> bool:
    result = subprocess.run(
        modal_command("token", "info"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def signup_url() -> str:
    return os.getenv("OPENCODELAB_MODAL_SIGNUP_URL", DEFAULT_MODAL_SIGNUP_URL).strip()


def print_signup_message() -> None:
    url = signup_url()
    print(f"Modal account/signup: {url}")
    if url != DEFAULT_MODAL_SIGNUP_URL:
        print("Disclosure: this is an OpenCodeLab partner/referral link and may support the project at no extra cost to you.")


def open_signup() -> None:
    print_signup_message()
    webbrowser.open(signup_url())


def ensure_modal_auth(*, interactive: bool = True) -> bool:
    if modal_is_authenticated():
        return True
    if not interactive:
        return False
    print("No authenticated Modal profile found.")
    print_signup_message()
    print("Opening Modal's login/setup flow. Complete the browser approval, then return here.")
    return subprocess.run(modal_command("setup"), check=False).returncode == 0


def create_modal_secret(
    secret_name: str,
    api_key: str,
    *,
    hf_token: str | None = None,
    force: bool = False,
) -> None:
    payload = {"OPENCODELAB_API_KEY": api_key}
    if hf_token:
        payload["HF_TOKEN"] = hf_token
        payload["HUGGING_FACE_HUB_TOKEN"] = hf_token

    fd, temp_name = tempfile.mkstemp(prefix="opencodelab-secret-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        try:
            os.chmod(temp_name, 0o600)
        except OSError:
            pass
        cmd = modal_command("secret", "create", secret_name, "--from-json", temp_name)
        if force:
            cmd.append("--force")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Modal secret creation failed for {secret_name}")
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def parse_modal_endpoint(output: str) -> str | None:
    matches = _MODAL_URL.findall(output)
    if not matches:
        return None
    serve_matches = [url for url in matches if "-serve" in url]
    return (serve_matches or matches)[-1].rstrip("/")


def deploy_modal_app(deployment_path: Path, app_name: str) -> str:
    result = subprocess.run(
        modal_command("deploy", str(deployment_path), "--name", app_name),
        text=True,
        capture_output=True,
        check=False,
    )
    combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if combined:
        print(combined.rstrip())
    if result.returncode != 0:
        raise RuntimeError("Modal deployment failed")
    endpoint = parse_modal_endpoint(combined)
    if not endpoint:
        raise RuntimeError("Modal deployment succeeded but OpenCodeLab could not find the endpoint URL in Modal output")
    return endpoint
