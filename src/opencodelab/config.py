from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from platformdirs import user_config_dir


@dataclass(frozen=True)
class Profile:
    name: str
    model_id: str
    gpu: str
    max_model_len: int
    idle_timeout_seconds: int
    endpoint: str
    app_name: str
    secret_name: str
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    trust_remote_code: bool = False
    speculative_mtp_tokens: int = 0
    language_model_only: bool = False
    vllm_version: str = "0.24.0"


@dataclass(frozen=True)
class LabConfig:
    version: int = 1
    default_profile: str | None = None
    profiles: dict[str, Profile] = field(default_factory=dict)


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
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        profiles = {
            name: Profile(**profile)
            for name, profile in raw.get("profiles", {}).items()
        }
        return LabConfig(
            version=int(raw.get("version", 1)),
            default_profile=raw.get("default_profile"),
            profiles=profiles,
        )

    def save(self, config: LabConfig) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": config.version,
            "default_profile": config.default_profile,
            "profiles": {name: asdict(profile) for name, profile in config.profiles.items()},
        }
        serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        fd, tmp_name = tempfile.mkstemp(prefix="config-", suffix=".json", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(serialized)
            try:
                os.chmod(tmp_name, 0o600)
            except OSError:
                pass
            os.replace(tmp_name, self.path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    def deployment_path(self, profile_name: str) -> Path:
        return self.root / "deployments" / f"{profile_name}.py"

    def overlay_path(self, profile_name: str) -> Path:
        return self.root / "opencode" / f"{profile_name}.json"

    def write_generated(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
