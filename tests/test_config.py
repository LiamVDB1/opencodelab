import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from opencodelab.config import (
    ConfigError,
    ConfigStore,
    LabConfig,
    Profile,
    bind_profile_endpoint,
    modal_endpoint_label,
    verify_profile_endpoint,
)


def test_config_round_trip_without_secrets(tmp_path: Path):
    store = ConfigStore(tmp_path)
    profile = Profile(
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
    )
    profile = bind_profile_endpoint(profile, "client-key")
    config = LabConfig(default_profile="qwen38", profiles={"qwen38": profile})

    store.save(config)
    loaded = store.load()

    assert loaded == config
    raw = json.loads((tmp_path / "config.json").read_text())
    assert "api_key" not in json.dumps(raw).lower()
    assert "hf_token" not in json.dumps(raw).lower()
    assert "client-key" not in json.dumps(raw)
    assert raw["profiles"]["qwen38"]["endpoint_binding"].startswith("hmac-sha256:")


@pytest.mark.parametrize("gpu", ["A100-80GB", "H200"])
def test_config_round_trips_newer_single_gpu_profiles(tmp_path: Path, gpu: str):
    store = ConfigStore(tmp_path)
    profile = bind_profile_endpoint(Profile(
        name="qwen38",
        model_id="gubernac/qwen3.8-27b-unc-fp8",
        gpu=gpu,
        max_model_len=131072,
        idle_timeout_seconds=90,
        endpoint=f"https://workspace--{modal_endpoint_label('qwen38')}.modal.run",
        environment_name="opencodelab-qwen38",
        app_name="opencodelab-qwen38",
        secret_name="ocl-qwen38-vllm",
        gateway_secret_name="ocl-qwen38-gateway",
        workspace_id="ws-test-workspace",
        proxy_token_id="wk-test-token",
    ), "client-key")
    config = LabConfig(default_profile="qwen38", profiles={"qwen38": profile})

    store.save(config)

    assert store.load() == config


def test_generated_files_live_under_config_home(tmp_path: Path):
    store = ConfigStore(tmp_path)
    assert store.deployment_path("abc") == tmp_path / "deployments" / "abc.py"
    assert store.overlay_path("abc") == tmp_path / "opencode" / "abc.json"


def test_config_and_generated_files_are_private(tmp_path: Path):
    store = ConfigStore(tmp_path)
    store.save(LabConfig())
    generated = store.overlay_path("abc")
    store.write_generated(generated, "{}\n")

    assert os.stat(store.path).st_mode & 0o777 == 0o600
    assert os.stat(generated).st_mode & 0o777 == 0o600
    assert os.stat(tmp_path).st_mode & 0o777 == 0o700
    assert os.stat(generated.parent).st_mode & 0o777 == 0o700


@pytest.mark.parametrize(
    "payload",
    [
        "{not-json",
        json.dumps({"version": 1, "profiles": []}),
        json.dumps({"version": 1, "profiles": {"bad": {"model_id": "owner/model"}}}),
        json.dumps(
            {
                "version": 1,
                "default_profile": "../escape",
                "profiles": {
                    "../escape": {
                        "name": "../escape",
                        "model_id": "owner/model",
                        "gpu": "L40S",
                        "max_model_len": 4096,
                        "idle_timeout_seconds": 90,
                        "endpoint": "",
                        "app_name": "--bad-option",
                        "secret_name": "secret",
                    }
                },
            }
        ),
        json.dumps(
            {
                "version": 1,
                "default_profile": "bad-types",
                "profiles": {
                    "bad-types": {
                        "name": "bad-types",
                        "model_id": "owner/model",
                        "gpu": "L40S",
                        "max_model_len": "lots",
                        "idle_timeout_seconds": 90,
                        "endpoint": [],
                        "app_name": "opencodelab-bad-types",
                        "secret_name": "opencodelab-bad-types-auth",
                        "proxy_token_id": "wk-test-token",
                    }
                },
            }
        ),
        json.dumps(
            {
                "version": 1,
                "default_profile": "victim",
                "profiles": {
                    "victim": {
                        "name": "victim",
                        "model_id": "owner/model",
                        "gpu": "L40S",
                        "max_model_len": 4096,
                        "idle_timeout_seconds": 90,
                        "endpoint": "https://workspace--unrelated-production-app-serve.modal.run",
                        "environment_name": "unrelated-production-environment",
                        "app_name": "unrelated-production-app",
                        "secret_name": "unrelated-production-secret",
                        "gateway_secret_name": "unrelated-production-gateway",
                        "proxy_token_id": "wk-test-token",
                    }
                },
            }
        ),
        json.dumps(
            {
                "version": 1,
                "default_profile": "victim",
                "profiles": {
                    "victim": {
                        "name": "victim",
                        "model_id": "owner/model",
                        "gpu": "L40S",
                        "max_model_len": 4096,
                        "idle_timeout_seconds": 90,
                        "endpoint": "https://workspace--unrelated-app-serve.modal.run",
                        "environment_name": "opencodelab-victim",
                        "app_name": "opencodelab-victim",
                        "secret_name": "ocl-victim-vllm",
                        "gateway_secret_name": "ocl-victim-gateway",
                        "proxy_token_id": "wk-test-token",
                    }
                },
            }
        ),
    ],
)
def test_invalid_config_has_actionable_error(tmp_path: Path, payload: str):
    path = tmp_path / "config.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ConfigError, match="config.json"):
        ConfigStore(tmp_path).load()


def test_long_profile_names_have_distinct_modal_resources():
    from opencodelab.config import modal_resource_names

    first = "a" * 39 + "b"
    second = "a" * 39 + "c"
    assert modal_resource_names(first) != modal_resource_names(second)


@pytest.mark.parametrize(
    "tampered_model_id",
    [
        "https://huggingface.co/owner/model",
        "https://hf.co/owner/model/",
        "../owner/model",
        "owner/model/path",
        "owner/model\nsecond/value",
        "owner/model\t",
        "owner//model",
        "/owner/model",
        "owner/model?revision=main",
    ],
)
def test_config_rejects_noncanonical_or_malformed_persisted_model_ids(
    tmp_path: Path, tampered_model_id: str
):
    profile = Profile(
        name="model",
        model_id="owner/model",
        gpu="L40S",
        max_model_len=4096,
        idle_timeout_seconds=90,
        endpoint=f"https://workspace--{modal_endpoint_label('model')}.modal.run",
        environment_name="opencodelab-model",
        app_name="opencodelab-model",
        secret_name="ocl-model-vllm",
        gateway_secret_name="ocl-model-gateway",
        workspace_id="ws-test-workspace",
        proxy_token_id="wk-test-token",
    )
    profile = bind_profile_endpoint(profile, "client-key")
    store = ConfigStore(tmp_path)
    store.save(LabConfig(default_profile="model", profiles={"model": profile}))
    raw = json.loads(store.path.read_text(encoding="utf-8"))
    raw["profiles"]["model"]["model_id"] = tampered_model_id
    store.path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="model"):
        store.load()


def test_config_fails_closed_when_endpoint_binding_is_missing_or_malformed(tmp_path: Path):
    profile = Profile(
        name="model",
        model_id="owner/model",
        gpu="L40S",
        max_model_len=4096,
        idle_timeout_seconds=90,
        endpoint=f"https://workspace--{modal_endpoint_label('model')}.modal.run",
        environment_name="opencodelab-model",
        app_name="opencodelab-model",
        secret_name="ocl-model-vllm",
        gateway_secret_name="ocl-model-gateway",
        workspace_id="ws-test-workspace",
        proxy_token_id="wk-test-token",
    )
    store = ConfigStore(tmp_path)

    for binding in ("", "hmac-sha256:not-a-digest"):
        store.save(
            LabConfig(
                default_profile="model",
                profiles={"model": replace(profile, endpoint_binding=binding)},
            )
        )
        with pytest.raises(ConfigError, match="endpoint binding"):
            store.load()


def test_endpoint_binding_rejects_source_substitution_with_same_label(tmp_path: Path):
    profile = Profile(
        name="model",
        model_id="owner/model",
        gpu="L40S",
        max_model_len=4096,
        idle_timeout_seconds=90,
        endpoint=f"https://workspace--{modal_endpoint_label('model')}.modal.run",
        environment_name="opencodelab-model",
        app_name="opencodelab-model",
        secret_name="ocl-model-vllm",
        gateway_secret_name="ocl-model-gateway",
        workspace_id="ws-test-workspace",
        proxy_token_id="wk-test-token",
    )
    bound = bind_profile_endpoint(profile, "client-key")
    substituted = replace(
        bound,
        endpoint=f"https://another-workspace--{modal_endpoint_label('model')}.modal.run",
    )

    with pytest.raises(ValueError, match="tampered"):
        verify_profile_endpoint(substituted, "client-key")
