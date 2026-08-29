from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from opencodelab.config import ConfigStore, Profile

PASSTHROUGH_COMMANDS = {
    "models",
    "providers",
    "auth",
    "debug",
    "agent",
    "mcp",
    "stats",
    "session",
    "completion",
    "upgrade",
    "uninstall",
    "serve",
    "web",
    "attach",
    "github",
    "pr",
    "plugin",
    "db",
    "acp",
    "export",
    "import",
}


def render_overlay(profile: Profile) -> str:
    if not profile.endpoint:
        raise ValueError(f"Profile {profile.name!r} has not been deployed yet")
    model: dict[str, object] = {
        "name": f"{profile.model_id} ({profile.gpu})",
        "limit": {
            "context": profile.max_model_len,
            "output": min(16384, profile.max_model_len),
        },
    }
    if profile.reasoning_parser:
        model["reasoning"] = True
        model["interleaved"] = {"field": "reasoning_content"}

    payload = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "opencodelab": {
                "npm": "@ai-sdk/openai-compatible",
                "name": f"OpenCodeLab ({profile.gpu})",
                "options": {
                    "baseURL": f"{profile.endpoint.rstrip('/')}/v1",
                    "apiKey": "{env:OPENCODELAB_API_KEY}",
                    "timeout": 1200000,
                    "chunkTimeout": 1200000,
                },
                "models": {profile.model_id: model},
            }
        },
    }
    return json.dumps(payload, indent=2) + "\n"


def build_opencode_args(profile: Profile, forwarded: list[str]) -> list[str]:
    model_id = f"opencodelab/{profile.model_id}"
    if forwarded and forwarded[0] == "run":
        return ["opencode", "run", "--model", model_id, *forwarded[1:]]
    if forwarded and forwarded[0] in PASSTHROUGH_COMMANDS:
        return ["opencode", *forwarded]
    if forwarded and forwarded[0] in {"--help", "-h", "--version", "-v"}:
        return ["opencode", *forwarded]
    return ["opencode", *forwarded, "--model", model_id]


def ensure_opencode() -> str:
    executable = shutil.which("opencode")
    if not executable:
        raise RuntimeError(
            "OpenCode is not installed or not on PATH. Install it from https://opencode.ai/docs/"
        )
    return executable


def launch_opencode(
    profile: Profile,
    api_key: str,
    store: ConfigStore,
    forwarded: list[str],
) -> int:
    executable = ensure_opencode()
    overlay_path: Path = store.overlay_path(profile.name)
    store.write_generated(overlay_path, render_overlay(profile))
    env = os.environ.copy()
    env["OPENCODELAB_API_KEY"] = api_key
    env["OPENCODE_CONFIG"] = str(overlay_path)
    args = build_opencode_args(profile, forwarded)
    args[0] = executable
    return subprocess.run(args, env=env, check=False).returncode
