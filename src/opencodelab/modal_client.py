from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sysconfig
import tempfile
import webbrowser
from collections.abc import Callable
from pathlib import Path

_MODAL_URL = re.compile(
    r"https://[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?\.(?:modal\.run|modal\.direct)"
    r"(?=$|[^A-Za-z0-9.-])"
)
DEFAULT_MODAL_SIGNUP_URL = "https://modal.com"
_PROXY_TOKEN_ID = re.compile(r"^wk-[A-Za-z0-9_-]+$")
_PROXY_TOKEN_SECRET = re.compile(r"^ws-[A-Za-z0-9_-]+$")
_WORKSPACE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,127}$")
_WORKSPACE_LINE = re.compile(
    r"^Workspace:\s+.*\((?P<workspace_id>[A-Za-z0-9][A-Za-z0-9_-]{1,127})\)\s*$",
    re.MULTILINE,
)
_DNS_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_MODAL_DIRECT_REGION = re.compile(r"^[a-z]{2}-[a-z0-9]+$")
_MODAL_SOURCE = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")


def _strict_modal_host_labels(host: str) -> list[str] | None:
    """Return the labels in front of an official Modal endpoint suffix, else None."""
    for suffix in (".modal.run", ".modal.direct"):
        if host.endswith(suffix):
            return host[: -len(suffix)].split(".")
    return None


def strict_modal_host_shape(host: str) -> bool:
    """Require the exact official Modal hostname shape with no spoofable labels."""
    labels = _strict_modal_host_labels(host)
    if labels is None:
        return False
    if host.endswith(".modal.run"):
        # Exactly one endpoint label directly under modal.run.
        return len(labels) == 1 and _DNS_LABEL.fullmatch(labels[0]) is not None
    # modal.direct: exactly one endpoint label plus one documented region label.
    return (
        len(labels) == 2
        and _DNS_LABEL.fullmatch(labels[0]) is not None
        and _MODAL_DIRECT_REGION.fullmatch(labels[1]) is not None
    )


def modal_command(*args: str) -> list[str]:
    scripts_dir = Path(sysconfig.get_path("scripts")).resolve()
    executable = scripts_dir / ("modal.exe" if os.name == "nt" else "modal")
    return [str(executable), *args]


def modal_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return env


def _run_modal(command: list[str], **kwargs):
    kwargs.setdefault("env", modal_subprocess_env())
    check = kwargs.pop("check", False)
    return subprocess.run(command, check=check, **kwargs)


def modal_is_authenticated() -> bool:
    result = _run_modal(
        modal_command("token", "info"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def signup_url() -> str:
    return DEFAULT_MODAL_SIGNUP_URL


def print_signup_message() -> None:
    print(f"Modal account/signup: {signup_url()}")


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
    return _run_modal(modal_command("setup"), check=False).returncode == 0


def get_modal_workspace_id() -> str:
    """Return the current non-secret Modal workspace identifier."""
    result = _run_modal(
        modal_command("token", "info"),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("Could not identify the active Modal workspace")
    match = _WORKSPACE_LINE.search(result.stdout)
    if not match:
        raise RuntimeError("Modal token info returned an unrecognized workspace identity")
    workspace_id = match.group("workspace_id")
    if not _WORKSPACE_ID.fullmatch(workspace_id):
        raise RuntimeError("Modal token info returned an invalid workspace identity")
    return workspace_id


def require_modal_workspace(expected_workspace_id: str) -> None:
    active_workspace_id = get_modal_workspace_id()
    if active_workspace_id != expected_workspace_id:
        raise RuntimeError(
            "The active Modal workspace does not own this OpenCodeLab profile. "
            "Activate the original Modal profile before changing or removing it."
        )


def create_modal_secret(
    secret_name: str,
    api_key: str,
    *,
    environment_name: str,
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
        cmd = modal_command(
            "secret",
            "create",
            secret_name,
            "--from-json",
            temp_name,
            "--env",
            environment_name,
        )
        if force:
            cmd.append("--force")
        result = _run_modal(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Modal secret creation failed for {secret_name}")
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def create_gateway_secret(
    secret_name: str,
    api_key: str,
    *,
    environment_name: str,
    proxy_token_id: str,
    proxy_token_secret: str,
) -> None:
    payload = {
        "OPENCODELAB_API_KEY": api_key,
        "OPENCODELAB_PROXY_TOKEN_ID": proxy_token_id,
        "OPENCODELAB_PROXY_TOKEN_SECRET": proxy_token_secret,
    }
    fd, temp_name = tempfile.mkstemp(prefix="opencodelab-gateway-secret-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        try:
            os.chmod(temp_name, 0o600)
        except OSError:
            pass
        result = _run_modal(
            modal_command(
                "secret",
                "create",
                secret_name,
                "--from-json",
                temp_name,
                "--env",
                environment_name,
                "--force",
            ),
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Modal gateway secret creation failed for {secret_name}")
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def create_modal_environment(environment_name: str) -> None:
    result = _run_modal(
        modal_command("environment", "create", environment_name),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not create dedicated Modal environment {environment_name!r}; "
            "choose another profile name"
        )


def _list_modal_proxy_token_ids() -> set[str]:
    listed = _run_modal(
        modal_command("workspace", "proxy-tokens", "list", "--json"),
        text=True,
        capture_output=True,
        check=False,
    )
    if listed.returncode != 0:
        raise RuntimeError("Modal proxy token listing failed during cleanup")
    try:
        tokens = json.loads(listed.stdout)
        if not isinstance(tokens, list):
            raise TypeError("token list must be an array")
        token_ids = set()
        for item in tokens:
            if not isinstance(item, dict) or not isinstance(item.get("token_id"), str):
                raise TypeError("token list contains an invalid item")
            token_id = item["token_id"]
            if not _PROXY_TOKEN_ID.fullmatch(token_id):
                raise ValueError("token list contains an invalid ID")
            token_ids.add(token_id)
        return token_ids
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError("Modal proxy token listing returned malformed output") from exc


def create_modal_proxy_token(
    *, on_token_id: Callable[[str], None] | None = None
) -> tuple[str, str]:
    before = _list_modal_proxy_token_ids()
    known_ids: set[str] = set()

    def remember(token_id: str) -> None:
        if token_id in known_ids:
            return
        known_ids.add(token_id)
        if on_token_id is not None:
            on_token_id(token_id)

    try:
        result = _run_modal(
            modal_command("workspace", "proxy-tokens", "create", "--json"),
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("Modal proxy token creation failed")

        payload = None
        parse_error: Exception | None = None
        try:
            payload = json.loads(result.stdout)
            if not isinstance(payload, dict):
                raise TypeError("token output must be an object")
        except (json.JSONDecodeError, TypeError) as exc:
            parse_error = exc

        token_id = payload.get("Modal-Key") if payload is not None else None
        if isinstance(token_id, str) and _PROXY_TOKEN_ID.fullmatch(token_id):
            remember(token_id)
        if parse_error is not None:
            raise RuntimeError("Modal proxy token creation returned malformed output") from parse_error

        token_secret = payload.get("Modal-Secret")
        authorization = payload.get("Authorization")
        expected_authorization = f"Bearer {token_id}.{token_secret}"
        if (
            not isinstance(token_id, str)
            or not isinstance(token_secret, str)
            or not _PROXY_TOKEN_ID.fullmatch(token_id)
            or not _PROXY_TOKEN_SECRET.fullmatch(token_secret)
            or (authorization is not None and authorization != expected_authorization)
        ):
            raise RuntimeError("Modal proxy token creation returned malformed output")
        return token_id, token_secret
    except (OSError, RuntimeError, ValueError, TypeError, AttributeError, KeyError) as exc:
        cleanup_error: Exception | None = None
        try:
            after = _list_modal_proxy_token_ids()
            for token_id in sorted((after - before) | known_ids):
                try:
                    remember(token_id)
                except (OSError, RuntimeError, ValueError) as callback_error:
                    cleanup_error = callback_error
                try:
                    delete_modal_proxy_token(token_id)
                except (OSError, RuntimeError, ValueError) as revoke_error:
                    cleanup_error = revoke_error
        except (OSError, RuntimeError, ValueError) as reconcile_error:
            cleanup_error = reconcile_error
        if cleanup_error is not None:
            raise RuntimeError(
                "MANUAL CLEANUP REQUIRED: could not reconcile or revoke newly created Modal "
                "proxy token(s); revoke them in Modal."
            ) from cleanup_error
        raise RuntimeError(f"{exc}; newly created proxy token(s) were revoked") from exc


def allow_modal_proxy_token(token_id: str, environment_name: str) -> None:
    result = _run_modal(
        modal_command("workspace", "proxy-tokens", "allow", token_id, environment_name),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not authorize the Modal proxy token for environment {environment_name!r}"
        )


def delete_modal_proxy_token(token_id: str) -> None:
    if not _PROXY_TOKEN_ID.fullmatch(token_id):
        raise ValueError("Invalid Modal proxy token ID")
    token_ids = _list_modal_proxy_token_ids()
    if token_id not in token_ids:
        return
    deleted = _run_modal(
        modal_command("workspace", "proxy-tokens", "delete", token_id, "--yes"),
        text=True,
        capture_output=True,
        check=False,
    )
    if deleted.returncode != 0:
        raise RuntimeError(f"Modal proxy token deletion failed for {token_id}")


def parse_modal_endpoint(output: str, *, endpoint_label: str | None = None) -> str | None:
    matches = _MODAL_URL.findall(output)
    if not matches:
        return None
    for url in reversed(matches):
        try:
            return normalize_modal_endpoint(url, endpoint_label=endpoint_label)
        except ValueError:
            continue
    return None


def _expected_gateway_subdomain(source: str, endpoint_label: str) -> str:
    full = f"{source}--{endpoint_label}"
    if len(full) > 63:
        digest = hashlib.sha256(full.encode()).hexdigest()[:6]
        return f"{full[:56]}-{digest}"
    return full


def _subdomain_matches_label(subdomain: str, endpoint_label: str) -> bool:
    """Accept any Modal source (including names containing '--') before our exact label."""
    marker = f"--{endpoint_label}"
    if subdomain.endswith(marker):
        source = subdomain[: -len(marker)]
    else:
        source = None
        # Modal's documented truncation replaces the label tail with a 7-character
        # hash ("-xxxxxx"); reconstruct the expected form for every source prefix.
        for cut in range(1, len(subdomain) - 6):
            if _expected_gateway_subdomain(subdomain[:cut], endpoint_label) == subdomain:
                source = subdomain[:cut]
                break
    return bool(source) and _MODAL_SOURCE.fullmatch(source) is not None


def normalize_modal_endpoint(value: str, *, endpoint_label: str | None = None) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(value.strip())
    host = parsed.hostname or ""
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or not strict_modal_host_shape(host)
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Modal endpoint must be an HTTPS origin on modal.run or modal.direct")
    if endpoint_label is not None and not _subdomain_matches_label(
        host.split(".", 1)[0].lower(), endpoint_label
    ):
        raise ValueError("Modal endpoint is not bound to the configured OpenCodeLab gateway")
    return f"https://{host.lower()}"


def deploy_modal_app(
    deployment_path: Path,
    app_name: str,
    *,
    environment_name: str,
    endpoint_label: str,
) -> str:
    result = _run_modal(
        modal_command(
            "deploy",
            str(deployment_path),
            "--name",
            app_name,
            "--env",
            environment_name,
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if combined:
        print(combined.rstrip())
    if result.returncode != 0:
        raise RuntimeError("Modal deployment failed")
    endpoint = parse_modal_endpoint(combined, endpoint_label=endpoint_label)
    if not endpoint:
        raise RuntimeError("Modal deployment succeeded but OpenCodeLab could not find the endpoint URL in Modal output")
    return endpoint


def stop_modal_app(app_name: str, *, environment_name: str) -> None:
    result = _run_modal(
        modal_command("app", "stop", app_name, "--yes", "--env", environment_name),
        text=True,
        capture_output=True,
        check=False,
    )
    combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
    idempotent_messages = ("App is already stopped.", "No App with name")
    if result.returncode != 0 and not any(message in combined for message in idempotent_messages):
        raise RuntimeError(f"Modal app stop failed for {app_name}")


def delete_modal_secret(secret_name: str, *, environment_name: str) -> None:
    result = _run_modal(
        modal_command(
            "secret",
            "delete",
            secret_name,
            "--allow-missing",
            "--yes",
            "--env",
            environment_name,
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Modal secret deletion failed for {secret_name}")


def modal_environment_exists(environment_name: str) -> bool:
    listed = _run_modal(
        modal_command("environment", "list", "--json"),
        text=True,
        capture_output=True,
        check=False,
    )
    if listed.returncode != 0:
        raise RuntimeError("Modal environment listing failed during cleanup")
    try:
        environments = json.loads(listed.stdout)
        names = {item["name"] for item in environments if isinstance(item, dict)}
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError("Modal environment listing returned malformed output") from exc
    return environment_name in names


def delete_modal_environment(environment_name: str) -> None:
    if not modal_environment_exists(environment_name):
        return
    deleted = _run_modal(
        modal_command("environment", "delete", environment_name, "--yes"),
        text=True,
        capture_output=True,
        check=False,
    )
    if deleted.returncode != 0:
        raise RuntimeError(f"Modal environment deletion failed for {environment_name}")
