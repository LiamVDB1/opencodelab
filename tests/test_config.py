import json
from pathlib import Path

from opencodelab.config import ConfigStore, LabConfig, Profile


def test_config_round_trip_without_secrets(tmp_path: Path):
    store = ConfigStore(tmp_path)
    profile = Profile(
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
    config = LabConfig(default_profile="qwen38", profiles={"qwen38": profile})

    store.save(config)
    loaded = store.load()

    assert loaded == config
    raw = json.loads((tmp_path / "config.json").read_text())
    assert "api_key" not in json.dumps(raw).lower()
    assert "hf_token" not in json.dumps(raw).lower()


def test_generated_files_live_under_config_home(tmp_path: Path):
    store = ConfigStore(tmp_path)
    assert store.deployment_path("abc") == tmp_path / "deployments" / "abc.py"
    assert store.overlay_path("abc") == tmp_path / "opencode" / "abc.json"
