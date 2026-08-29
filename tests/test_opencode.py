import json

from opencodelab.config import Profile
from opencodelab.opencode import build_opencode_args, render_overlay


def _profile():
    return Profile(
        name="qwen38",
        model_id="gubernac/qwen3.8-27b-unc-fp8",
        gpu="L40S",
        max_model_len=65536,
        idle_timeout_seconds=90,
        endpoint="https://example.modal.run",
        app_name="opencodelab-qwen38",
        secret_name="opencodelab-qwen38-auth",
        tool_call_parser="qwen3_coder",
        reasoning_parser="qwen3",
    )


def test_overlay_uses_env_secret_not_literal_key():
    overlay = json.loads(render_overlay(_profile()))
    provider = overlay["provider"]["opencodelab"]
    assert provider["options"]["baseURL"] == "https://example.modal.run/v1"
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
