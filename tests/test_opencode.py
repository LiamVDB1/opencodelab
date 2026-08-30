import json
import os
import subprocess
from pathlib import Path

import pytest

from opencodelab.config import ConfigStore, Profile, bind_profile_endpoint, modal_endpoint_label
from opencodelab.opencode import (
    build_opencode_args,
    launch_opencode,
    render_overlay,
    scrub_child_environment,
)


def _profile():
    return bind_profile_endpoint(Profile(
        name="qwen38",
        model_id="gubernac/qwen3.8-27b-unc-fp8",
        gpu="L40S",
        max_model_len=65536,
        idle_timeout_seconds=90,
        endpoint=f"https://workspace--{modal_endpoint_label('qwen38')}.modal.run",
        environment_name="opencodelab-qwen38",
        app_name="opencodelab-qwen38",
        secret_name="ocl-qwen38-vllm",
        gateway_secret_name="ocl-qwen38-gateway",
        workspace_id="ws-test-workspace",
        proxy_token_id="wk-test-token",
        tool_call_parser="qwen3_coder",
        reasoning_parser="qwen3",
    ), "test-key")


def test_overlay_uses_env_secret_not_literal_key():
    overlay = json.loads(render_overlay(_profile(), "test-key"))
    provider = overlay["provider"]["opencodelab"]
    assert provider["options"]["baseURL"] == (
        f"https://workspace--{modal_endpoint_label('qwen38')}.modal.run/v1"
    )
    assert provider["options"]["apiKey"] == "{env:OPENCODELAB_API_KEY}"
    assert provider["models"]["gubernac/qwen3.8-27b-unc-fp8"]["limit"]["context"] == 65536


def test_run_command_injects_model_after_run():
    assert build_opencode_args(_profile(), ["run", "inspect this repo"]) == [
        "opencode",
        "run",
        "--model",
        "opencodelab/gubernac/qwen3.8-27b-unc-fp8",
        "inspect this repo",
    ]


def test_tui_launch_appends_model():
    assert build_opencode_args(_profile(), []) == [
        "opencode",
        "--model",
        "opencodelab/gubernac/qwen3.8-27b-unc-fp8",
    ]


def test_overlay_rejects_non_modal_endpoint():
    profile = _profile()
    profile = type(profile)(**{**profile.__dict__, "endpoint": "https://attacker.example"})
    with pytest.raises(ValueError, match="Modal endpoint"):
        render_overlay(profile, "test-key")


def test_launch_uses_overlay_without_overwriting_normal_config(monkeypatch, tmp_path: Path):
    normal_config = tmp_path / "normal-opencode.json"
    normal_config.write_text('{"theme":"existing"}', encoding="utf-8")
    monkeypatch.setenv("OPENCODE_CONFIG", str(normal_config))
    monkeypatch.setattr("opencodelab.opencode.ensure_opencode", lambda: "/usr/bin/opencode")
    observed: dict[str, object] = {}

    def fake_run(args, env, check):
        observed.update(args=args, env=env, check=check)
        return subprocess.CompletedProcess(args, 7)

    monkeypatch.setattr("opencodelab.opencode.subprocess.run", fake_run)
    profile = bind_profile_endpoint(
        _profile(), "secret-value"
    )
    result = launch_opencode(profile, "secret-value", ConfigStore(tmp_path / "lab"), ["run", "task"])

    assert result == 7
    assert normal_config.read_text(encoding="utf-8") == '{"theme":"existing"}'
    assert observed["env"]["OPENCODE_CONFIG"] != str(normal_config)
    assert observed["env"]["OPENCODELAB_API_KEY"] == "secret-value"
    assert "secret-value" not in Path(observed["env"]["OPENCODE_CONFIG"]).read_text(encoding="utf-8")


def test_scrub_child_environment_removes_modal_and_hf_secret_material():
    dirty = dict(
        os.environ,
        MODAL_TOKEN_ID="ak-m-id",
        MODAL_TOKEN_SECRET="as-m-secret",
        MODAL_PROFILE="other-workspace",
        HF_TOKEN="hf-secret",
        HUGGING_FACE_HUB_TOKEN="hf-secret",
        OPENCODELAB_API_KEY="parent-key",
        OPENCODELAB_PROXY_TOKEN_ID="wk-proxy",
        OPENCODELAB_PROXY_TOKEN_SECRET="ws-proxy",
        OPENCODE_CONFIG="/tmp/parent-overlay.json",
        PATH="/usr/bin",
    )

    cleaned = scrub_child_environment(dirty)

    for name in (
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "MODAL_PROFILE",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "OPENCODELAB_API_KEY",
        "OPENCODELAB_PROXY_TOKEN_ID",
        "OPENCODELAB_PROXY_TOKEN_SECRET",
    ):
        assert name not in cleaned
    assert cleaned["PATH"] == "/usr/bin"


def test_launch_env_excludes_parent_secrets_and_injects_profile_key_only(
    monkeypatch, tmp_path: Path
):
    for name, value in (
        ("MODAL_TOKEN_ID", "ak-m-id"),
        ("MODAL_TOKEN_SECRET", "as-m-secret"),
        ("MODAL_PROFILE", "other-workspace"),
        ("HF_TOKEN", "hf-secret"),
        ("HUGGING_FACE_HUB_TOKEN", "hf-secret"),
        ("OPENCODELAB_API_KEY", "parent-key"),
        ("OPENCODELAB_PROXY_TOKEN_ID", "wk-proxy"),
        ("OPENCODELAB_PROXY_TOKEN_SECRET", "ws-proxy"),
    ):
        monkeypatch.setenv(name, value)
    monkeypatch.setattr("opencodelab.opencode.ensure_opencode", lambda: "/usr/bin/opencode")
    observed: dict[str, object] = {}

    def fake_run(args, env, check):
        observed.update(env=dict(env))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("opencodelab.opencode.subprocess.run", fake_run)
    profile = bind_profile_endpoint(_profile(), "profile-key")
    launch_opencode(profile, "profile-key", ConfigStore(tmp_path / "lab"), [])

    env = observed["env"]
    for name in (
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "MODAL_PROFILE",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "OPENCODELAB_PROXY_TOKEN_ID",
        "OPENCODELAB_PROXY_TOKEN_SECRET",
    ):
        assert name not in env
    assert env["OPENCODELAB_API_KEY"] == "profile-key"
