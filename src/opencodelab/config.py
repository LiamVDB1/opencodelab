from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from platformdirs import user_config_dir

from opencodelab.models import normalize_huggingface_model

_PROFILE_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
_MODAL_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}[a-z0-9]$|^[a-z0-9]$")
_PROXY_TOKEN_ID = re.compile(r"^wk-[A-Za-z0-9_-]+$")
_PARSER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+_-]{0,31}$")
_ENDPOINT_BINDING = re.compile(r"^hmac-sha256:[0-9a-f]{64}$")


SUPPORTED_GPUS = frozenset({"L40S", "A100-80GB", "H100", "H200"})


class ConfigError(ValueError):
    """Raised when OpenCodeLab's local configuration cannot be used safely."""


def modal_resource_names(profile_name: str) -> tuple[str, str, str, str]:
    if not _PROFILE_NAME.fullmatch(profile_name):
        raise ValueError(f"Unsafe OpenCodeLab profile name: {profile_name!r}")
    environment_name = f"opencodelab-{profile_name}"
    app_name = environment_name
    return (
        environment_name,
        app_name,
        f"ocl-{profile_name}-vllm",
        f"ocl-{profile_name}-gateway",
    )


def modal_endpoint_label(profile_name: str) -> str:
    if not _PROFILE_NAME.fullmatch(profile_name):
        raise ValueError(f"Unsafe OpenCodeLab profile name: {profile_name!r}")
    digest = hashlib.sha256(profile_name.encode("utf-8")).hexdigest()[:16]
    return f"ocl-{digest}"


@dataclass(frozen=True)
class Profile:
    name: str
    model_id: str
    gpu: str
    max_model_len: int
    idle_timeout_seconds: int
    endpoint: str
    environment_name: str
    app_name: str
    secret_name: str
    gateway_secret_name: str
    endpoint_binding: str = ""
    workspace_id: str = ""
    proxy_token_id: str = ""
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    trust_remote_code: bool = False
    speculative_mtp_tokens: int = 0
    language_model_only: bool = False
    vllm_version: str = "0.24.0"


@dataclass(frozen=True)
class LabConfig:
    version: int = 2
    default_profile: str | None = None
    profiles: dict[str, Profile] = field(default_factory=dict)


def _endpoint_binding_message(profile: Profile) -> bytes:
    return (
        "opencodelab:endpoint:v1\0"
        f"{profile.name}\0{profile.workspace_id}\0{profile.endpoint}"
    ).encode()


def bind_profile_endpoint(profile: Profile, client_key: str) -> Profile:
    if not profile.endpoint:
        raise ValueError("Cannot bind an empty endpoint")
    if not isinstance(client_key, str) or not client_key:
        raise ValueError("Cannot bind an endpoint without the client credential")
    digest = hmac.new(
        client_key.encode("utf-8"),
        _endpoint_binding_message(profile),
        hashlib.sha256,
    ).hexdigest()
    return replace(profile, endpoint_binding=f"hmac-sha256:{digest}")


def verify_profile_endpoint(profile: Profile, client_key: str) -> None:
    if not profile.endpoint or not _ENDPOINT_BINDING.fullmatch(profile.endpoint_binding):
        raise ValueError(f"Profile {profile.name!r} has no valid endpoint binding")
    if not isinstance(client_key, str) or not client_key:
        raise ValueError(f"Profile {profile.name!r} has no usable endpoint credential")
    expected = hmac.new(
        client_key.encode("utf-8"),
        _endpoint_binding_message(profile),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(profile.endpoint_binding, f"hmac-sha256:{expected}"):
        raise ValueError(f"Profile {profile.name!r} has a tampered endpoint binding")


class ConfigStore:
    def __init__(self, root: Path | None = None) -> None:
        configured = os.getenv("OPENCODELAB_HOME")
        if root is not None:
            self.root = root
        elif configured:
            self.root = Path(configured).expanduser()
        else:
            self.root = Path(user_config_dir("opencodelab"))
        self.path = self.root / "config.json"

    def load(self) -> LabConfig:
        if not self.path.exists():
            return LabConfig()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise TypeError("top level must be an object")
            raw_profiles = raw.get("profiles", {})
            if not isinstance(raw_profiles, dict):
                raise TypeError("profiles must be an object")
            profiles = {
                str(name): Profile(**profile)
                for name, profile in raw_profiles.items()
                if isinstance(profile, dict)
            }
            if len(profiles) != len(raw_profiles):
                raise TypeError("each profile must be an object")
            for name, profile in profiles.items():
                if not _PROFILE_NAME.fullmatch(name) or profile.name != name:
                    raise ValueError(f"unsafe or mismatched profile name {name!r}")
                string_fields = {
                    "model_id": profile.model_id,
                    "gpu": profile.gpu,
                    "endpoint": profile.endpoint,
                    "environment_name": profile.environment_name,
                    "app_name": profile.app_name,
                    "secret_name": profile.secret_name,
                    "gateway_secret_name": profile.gateway_secret_name,
                    "workspace_id": profile.workspace_id,
                    "proxy_token_id": profile.proxy_token_id,
                    "vllm_version": profile.vllm_version,
                }
                if any(not isinstance(value, str) for value in string_fields.values()):
                    raise TypeError(f"profile {name!r} contains a non-string text field")
                normalized_model_id = normalize_huggingface_model(profile.model_id)
                if normalized_model_id != profile.model_id:
                    raise ValueError(f"profile {name!r} has a non-canonical model id")
                if profile.gpu not in SUPPORTED_GPUS:
                    raise ValueError(f"profile {name!r} uses unsupported GPU {profile.gpu!r}")
                if (
                    type(profile.max_model_len) is not int
                    or not 1024 <= profile.max_model_len <= 1_000_000
                ):
                    raise TypeError(f"profile {name!r} has an invalid context length")
                if (
                    type(profile.idle_timeout_seconds) is not int
                    or not 2 <= profile.idle_timeout_seconds <= 1200
                ):
                    raise TypeError(f"profile {name!r} has an invalid idle timeout")
                if (
                    type(profile.speculative_mtp_tokens) is not int
                    or not 0 <= profile.speculative_mtp_tokens <= 16
                ):
                    raise TypeError(f"profile {name!r} has invalid MTP settings")
                if type(profile.trust_remote_code) is not bool or type(profile.language_model_only) is not bool:
                    raise TypeError(f"profile {name!r} has invalid boolean settings")
                if any(
                    value is not None
                    and (not isinstance(value, str) or not _PARSER.fullmatch(value))
                    for value in (profile.tool_call_parser, profile.reasoning_parser)
                ):
                    raise TypeError(f"profile {name!r} has invalid parser settings")
                if not _VERSION.fullmatch(profile.vllm_version):
                    raise ValueError(f"profile {name!r} has an invalid vLLM version")
                expected_environment, expected_app, expected_secret, expected_gateway_secret = (
                    modal_resource_names(name)
                )
                if (
                    profile.environment_name != expected_environment
                    or not _MODAL_NAME.fullmatch(profile.environment_name)
                ):
                    raise ValueError(f"profile {name!r} is not bound to its OpenCodeLab Modal environment")
                if profile.app_name != expected_app or not _MODAL_NAME.fullmatch(profile.app_name):
                    raise ValueError(f"profile {name!r} is not bound to its OpenCodeLab Modal app")
                if profile.secret_name != expected_secret or not _MODAL_NAME.fullmatch(profile.secret_name):
                    raise ValueError(f"profile {name!r} is not bound to its OpenCodeLab Modal secret")
                if (
                    profile.gateway_secret_name != expected_gateway_secret
                    or not _MODAL_NAME.fullmatch(profile.gateway_secret_name)
                ):
                    raise ValueError(
                        f"profile {name!r} is not bound to its OpenCodeLab gateway secret"
                    )
                if not isinstance(profile.endpoint_binding, str):
                    raise TypeError(f"profile {name!r} has a non-string endpoint binding")
                if profile.proxy_token_id and not _PROXY_TOKEN_ID.fullmatch(profile.proxy_token_id):
                    raise ValueError(f"profile {name!r} has an invalid Modal proxy token binding")
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,127}", profile.workspace_id):
                    raise ValueError(f"profile {name!r} has no valid Modal workspace binding")
                if profile.endpoint and not profile.proxy_token_id:
                    raise ValueError(f"profile {name!r} has no valid Modal proxy token binding")
                if profile.endpoint:
                    from opencodelab.modal_client import normalize_modal_endpoint

                    normalized_endpoint = normalize_modal_endpoint(
                        profile.endpoint,
                        endpoint_label=modal_endpoint_label(profile.name),
                    )
                    if normalized_endpoint != profile.endpoint:
                        raise ValueError(f"profile {name!r} has a non-canonical endpoint")
                    if not _ENDPOINT_BINDING.fullmatch(profile.endpoint_binding):
                        raise ValueError(f"profile {name!r} has no valid endpoint binding")
                elif profile.endpoint_binding:
                    raise ValueError(f"profile {name!r} has an endpoint binding without an endpoint")
            version = int(raw.get("version", 1))
            if version != 2:
                raise ValueError(f"unsupported config version {version}")
            default_profile = raw.get("default_profile")
            if default_profile is not None and not isinstance(default_profile, str):
                raise TypeError("default_profile must be a string or null")
            if default_profile is not None and default_profile not in profiles:
                raise ValueError(f"default profile {default_profile!r} does not exist")
            return LabConfig(
                version=version,
                default_profile=default_profile,
                profiles=profiles,
            )
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            raise ConfigError(
                f"Could not read OpenCodeLab config {self.path}: {exc}. "
                "Fix the file or move it aside and run `opencodelab setup` again."
            ) from exc

    def save(self, config: LabConfig) -> None:
        payload = {
            "version": config.version,
            "default_profile": config.default_profile,
            "profiles": {name: asdict(profile) for name, profile in config.profiles.items()},
        }
        serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        self._atomic_write(self.path, serialized)

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            path.parent.chmod(0o700)
        except OSError:
            pass
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
        try:
            try:
                os.fchmod(fd, 0o600)
            except OSError:
                pass
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    def deployment_path(self, profile_name: str) -> Path:
        return self.root / "deployments" / f"{profile_name}.py"

    def overlay_path(self, profile_name: str) -> Path:
        return self.root / "opencode" / f"{profile_name}.json"

    def write_generated(self, path: Path, content: str) -> None:
        self._atomic_write(path, content)
