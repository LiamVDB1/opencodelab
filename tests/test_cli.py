import sys
from pathlib import Path

import pytest

from opencodelab.cli import (
    GPU_PRICING,
    _dispatch,
    _extract_profile,
    _validate_env_name,
    _validate_gpu,
    _validate_parser,
    cmd_config,
    cmd_doctor,
    cmd_gpu,
    cmd_list,
    cmd_login,
    cmd_remove,
    cmd_setup,
    cmd_signup,
    cmd_use,
    launch,
    main,
)
from opencodelab.config import (
    SUPPORTED_GPUS,
    ConfigStore,
    LabConfig,
    Profile,
    bind_profile_endpoint,
    modal_endpoint_label,
)


@pytest.fixture(autouse=True)
def _stable_modal_workspace(monkeypatch):
    monkeypatch.setattr(
        "opencodelab.cli.get_modal_workspace_id", lambda: "ws-test-workspace"
    )
    monkeypatch.setattr("opencodelab.cli.require_modal_workspace", lambda expected: None)


def _profile(name: str = "model", *, gpu: str = "L40S", endpoint: str | None = None) -> Profile:
    endpoint = endpoint or f"https://workspace--{modal_endpoint_label(name)}.modal.run"
    return bind_profile_endpoint(Profile(
        name=name,
        model_id=f"owner/{name}",
        gpu=gpu,
        max_model_len=32768,
        idle_timeout_seconds=90,
        endpoint=endpoint,
        environment_name=f"opencodelab-{name}",
        app_name=f"opencodelab-{name}",
        secret_name=f"ocl-{name}-vllm",
        gateway_secret_name=f"ocl-{name}-gateway",
        workspace_id="ws-test-workspace",
        proxy_token_id=f"wk-{name}-token",
    ), "test-key")


def test_single_gpu_selectors_are_allowed():
    assert _validate_gpu("L40S") == "L40S"
    assert _validate_gpu("a100-80gb") == "A100-80GB"
    assert _validate_gpu("H100") == "H100"
    assert _validate_gpu("h200") == "H200"


def test_every_priced_gpu_is_accepted_by_persisted_config():
    assert set(GPU_PRICING) == SUPPORTED_GPUS


def test_multi_gpu_selector_is_rejected_until_tensor_parallel_is_supported():
    with pytest.raises(ValueError):
        _validate_gpu("H100:2")


def test_only_documented_gpu_choices_are_allowed():
    with pytest.raises(ValueError, match="GPU must be one of L40S, A100-80GB, H100, H200"):
        _validate_gpu("A100")


@pytest.mark.parametrize("value", ["bad parser", "a/b", "--flag", "x\nvalue"])
def test_parser_names_reject_vllm_argument_injection(value: str):
    with pytest.raises(ValueError):
        _validate_parser(value)


@pytest.mark.parametrize("value", ["HF-TOKEN", "1TOKEN", "TOKEN=value", "TOKEN\nOTHER"])
def test_environment_variable_names_are_validated(value: str):
    with pytest.raises(ValueError):
        _validate_env_name(value)


def test_qwen38_setup_defaults_to_128k_context(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setenv("OPENCODELAB_HOME", str(tmp_path / "home"))

    assert cmd_setup(["--dry-run"]) == 0

    plan = capsys.readouterr().out
    assert '"model": "gubernac/qwen3.8-27b-unc-fp8"' in plan
    assert '"context": 131072' in plan


def test_setup_dry_run_is_side_effect_free_and_discloses_cost_snapshot(monkeypatch, tmp_path: Path, capsys):
    home = tmp_path / "home"
    monkeypatch.setenv("OPENCODELAB_HOME", str(home))

    assert cmd_setup(["owner/model", "--gpu", "H100", "--dry-run"]) == 0

    plan = capsys.readouterr().out
    assert '"model": "owner/model"' in plan
    assert '"gpu": "H100"' in plan
    assert '"max_cpu_gateway_containers": 1' in plan
    assert "profile key at CPU gateway" in plan
    assert '"gpu_price_per_second_usd": 0.001097' in plan
    assert "No Modal state or local config changed" in plan
    assert not home.exists()


def test_setup_success_persists_only_after_deployment(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("OPENCODELAB_HOME", str(tmp_path))
    calls: dict[str, object] = {}

    class FakeSecrets:
        def __init__(self, root):
            calls["secret_root"] = root

        def get_api_key(self, profile_name, **kwargs):
            return None

        def set_api_key(self, profile_name, value):
            calls["stored_key"] = value
            return "test keyring"

    monkeypatch.setattr("opencodelab.cli.SecretStore", FakeSecrets)
    monkeypatch.setattr("opencodelab.cli.ensure_modal_auth", lambda interactive: True)
    monkeypatch.setattr("opencodelab.cli.create_modal_environment", lambda name: None)
    monkeypatch.setattr(
        "opencodelab.cli.create_modal_proxy_token",
        lambda **kwargs: (
            kwargs["on_token_id"]("wk-model-token"),
            ("wk-model-token", "ws-model-secret"),
        )[1],
    )
    monkeypatch.setattr("opencodelab.cli.allow_modal_proxy_token", lambda *a: True)
    monkeypatch.setattr(
        "opencodelab.cli.create_gateway_secret",
        lambda name, key, **kwargs: calls.update(gateway_name=name, gateway_key=key),
    )
    monkeypatch.setattr(
        "opencodelab.cli.create_modal_secret",
        lambda name, key, **kwargs: calls.update(secret_name=name, remote_key=key, secret_kwargs=kwargs),
    )
    monkeypatch.setattr(
        "opencodelab.cli.deploy_modal_app",
        lambda path, name, **kwargs: (
            f"https://workspace--{modal_endpoint_label('model')}.modal.run"
        ),
    )

    assert cmd_setup(["owner/model", "--yes"]) == 0

    config = ConfigStore(tmp_path).load()
    profile = config.profiles["model"]
    assert profile.endpoint == f"https://workspace--{modal_endpoint_label('model')}.modal.run"
    assert profile.proxy_token_id == "wk-model-token"
    assert profile.workspace_id == "ws-test-workspace"
    assert calls["stored_key"] == calls["remote_key"] == calls["gateway_key"]
    assert "wk-model-token" not in str(calls["stored_key"])
    assert "OPENCODELAB_API_KEY" not in (tmp_path / "config.json").read_text(encoding="utf-8")
    assert "OPENCODELAB_API_KEY" in (tmp_path / "opencode" / "model.json").read_text(encoding="utf-8")
    assert str(calls["stored_key"]) not in (tmp_path / "opencode" / "model.json").read_text(encoding="utf-8")


def test_setup_completes_in_workspace_without_rbac(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setenv("OPENCODELAB_HOME", str(tmp_path))

    class FakeSecrets:
        def __init__(self, root):
            pass

        def get_api_key(self, profile_name, **kwargs):
            return None

        def set_api_key(self, profile_name, value):
            return "test keyring"

    monkeypatch.setattr("opencodelab.cli.SecretStore", FakeSecrets)
    monkeypatch.setattr("opencodelab.cli.ensure_modal_auth", lambda interactive: True)
    monkeypatch.setattr("opencodelab.cli.create_modal_environment", lambda name: None)
    monkeypatch.setattr(
        "opencodelab.cli.create_modal_proxy_token",
        lambda **kwargs: ("wk-model-token", "ws-model-secret"),
    )
    monkeypatch.setattr("opencodelab.cli.allow_modal_proxy_token", lambda *a: False)
    monkeypatch.setattr("opencodelab.cli.create_gateway_secret", lambda *a, **k: None)
    monkeypatch.setattr("opencodelab.cli.create_modal_secret", lambda *a, **k: None)
    monkeypatch.setattr(
        "opencodelab.cli.deploy_modal_app",
        lambda *a, **k: f"https://workspace--{modal_endpoint_label('model')}.modal.run",
    )

    assert cmd_setup(["owner/model", "--yes"]) == 0

    assert "proxy token is workspace-wide" in capsys.readouterr().out
    assert ConfigStore(tmp_path).load().profiles["model"].endpoint


def test_setup_deploy_failure_retains_recovery_profile(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("OPENCODELAB_HOME", str(tmp_path))

    class FakeSecrets:
        def __init__(self, root):
            pass

        def get_api_key(self, profile_name, **kwargs):
            return None

        def set_api_key(self, profile_name, value):
            return "test keyring"

    monkeypatch.setattr("opencodelab.cli.SecretStore", FakeSecrets)
    monkeypatch.setattr("opencodelab.cli.ensure_modal_auth", lambda interactive: True)
    monkeypatch.setattr("opencodelab.cli.create_modal_environment", lambda name: None)
    monkeypatch.setattr(
        "opencodelab.cli.create_modal_proxy_token",
        lambda **kwargs: (
            kwargs["on_token_id"]("wk-model-token"),
            ("wk-model-token", "ws-model-secret"),
        )[1],
    )
    monkeypatch.setattr("opencodelab.cli.allow_modal_proxy_token", lambda *a: True)
    monkeypatch.setattr("opencodelab.cli.create_gateway_secret", lambda *a, **k: None)
    monkeypatch.setattr("opencodelab.cli.create_modal_secret", lambda *a, **k: None)
    monkeypatch.setattr(
        "opencodelab.cli.deploy_modal_app",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("deployment failed")),
    )

    with pytest.raises(RuntimeError, match="Recovery profile"):
        cmd_setup(["owner/model", "--yes"])

    recovery = ConfigStore(tmp_path).load().profiles["model"]
    assert recovery.endpoint == ""
    assert recovery.proxy_token_id == "wk-model-token"


def test_setup_token_failure_retains_environment_recovery_profile(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("OPENCODELAB_HOME", str(tmp_path))
    monkeypatch.setattr("opencodelab.cli.ensure_modal_auth", lambda interactive: True)
    monkeypatch.setattr("opencodelab.cli.create_modal_environment", lambda name: None)
    monkeypatch.setattr(
        "opencodelab.cli.create_modal_proxy_token",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("token failed")),
    )

    with pytest.raises(RuntimeError, match="Recovery profile"):
        cmd_setup(["owner/model", "--yes"])

    recovery = ConfigStore(tmp_path).load().profiles["model"]
    assert recovery.environment_name == "opencodelab-model"
    assert recovery.proxy_token_id == ""
    assert recovery.endpoint == ""


def test_force_setup_replaces_vllm_secret_without_stale_hf_token(
    monkeypatch, tmp_path: Path
):
    profile = _profile()
    ConfigStore(tmp_path).save(LabConfig(default_profile="model", profiles={"model": profile}))
    monkeypatch.setenv("OPENCODELAB_HOME", str(tmp_path))
    monkeypatch.delenv("HF_TOKEN", raising=False)

    class FakeSecrets:
        def __init__(self, root):
            pass

        def get_api_key(self, profile_name, **kwargs):
            return "existing-profile-key"

    secret_calls: list[dict[str, object]] = []
    monkeypatch.setattr("opencodelab.cli.SecretStore", FakeSecrets)
    monkeypatch.setattr("opencodelab.cli.ensure_modal_auth", lambda interactive: True)
    monkeypatch.setattr(
        "opencodelab.cli.create_modal_secret",
        lambda name, key, **kwargs: secret_calls.append({"name": name, "key": key, **kwargs}),
    )
    monkeypatch.setattr(
        "opencodelab.cli.deploy_modal_app",
        lambda *args, **kwargs: profile.endpoint,
    )

    assert cmd_setup(["owner/model", "--force", "--yes"]) == 0
    assert secret_calls == [
        {
            "name": "ocl-model-vllm",
            "key": "existing-profile-key",
            "environment_name": "opencodelab-model",
            "hf_token": None,
            "force": True,
        }
    ]


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["owner/model", "--context", "100", "--dry-run"], "Context"),
        (["owner/model", "--context", "1000001", "--dry-run"], "Context"),
        (["owner/model", "--idle-timeout", "1", "--dry-run"], "Idle timeout"),
        (["owner/model", "--mtp-tokens", "17", "--dry-run"], "MTP"),
        (["owner/model", "--vllm-version", "bad/version", "--dry-run"], "vLLM version"),
    ],
)
def test_setup_validation_errors_are_clean(monkeypatch, tmp_path: Path, arguments, message):
    monkeypatch.setenv("OPENCODELAB_HOME", str(tmp_path))
    with pytest.raises(SystemExit, match=message):
        cmd_setup(arguments)


def test_setup_rejects_existing_profile_without_force(monkeypatch, tmp_path: Path):
    profile = _profile()
    ConfigStore(tmp_path).save(LabConfig(default_profile="model", profiles={"model": profile}))
    monkeypatch.setenv("OPENCODELAB_HOME", str(tmp_path))
    with pytest.raises(SystemExit, match="already exists"):
        cmd_setup(["owner/model", "--dry-run"])


def test_force_setup_rejects_different_active_workspace(monkeypatch, tmp_path: Path):
    profile = _profile()
    ConfigStore(tmp_path).save(LabConfig(default_profile="model", profiles={"model": profile}))
    monkeypatch.setenv("OPENCODELAB_HOME", str(tmp_path))
    monkeypatch.setattr("opencodelab.cli.ensure_modal_auth", lambda interactive: True)
    monkeypatch.setattr("opencodelab.cli.get_modal_workspace_id", lambda: "ws-other-workspace")
    monkeypatch.setattr(
        "opencodelab.cli.create_modal_secret",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not mutate")),
    )

    with pytest.raises(SystemExit, match="active Modal workspace"):
        cmd_setup(["owner/model", "--force", "--yes"])


def test_setup_auth_failure_stops_before_secret_creation(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("OPENCODELAB_HOME", str(tmp_path))
    monkeypatch.setattr("opencodelab.cli.ensure_modal_auth", lambda interactive: False)
    monkeypatch.setattr(
        "opencodelab.cli.create_modal_secret",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not create secret")),
    )
    with pytest.raises(SystemExit, match="authentication"):
        cmd_setup(["owner/model", "--yes"])


def test_gpu_noop_never_authenticates_or_deploys(monkeypatch, tmp_path: Path, capsys):
    profile = bind_profile_endpoint(Profile(
        name="model",
        model_id="owner/model",
        gpu="L40S",
        max_model_len=32768,
        idle_timeout_seconds=90,
        endpoint=f"https://workspace--{modal_endpoint_label('model')}.modal.run",
        environment_name="opencodelab-model",
        app_name="opencodelab-model",
        secret_name="ocl-model-vllm",
        gateway_secret_name="ocl-model-gateway",
        workspace_id="ws-test-workspace",
        proxy_token_id="wk-model-token",
    ), "test-key")
    ConfigStore(tmp_path).save(LabConfig(default_profile="model", profiles={"model": profile}))
    monkeypatch.setenv("OPENCODELAB_HOME", str(tmp_path))
    monkeypatch.setattr(
        "opencodelab.cli.SecretStore.get_api_key",
        lambda self, name, **kwargs: "test-key",
    )
    monkeypatch.setattr(
        "opencodelab.cli.ensure_modal_auth",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not authenticate")),
    )

    assert cmd_gpu(["L40S"]) == 0
    assert "already uses L40S" in capsys.readouterr().out


def test_gpu_change_redeploys_and_updates_profile(monkeypatch, tmp_path: Path, capsys):
    profile = _profile()
    ConfigStore(tmp_path).save(LabConfig(default_profile="model", profiles={"model": profile}))
    monkeypatch.setenv("OPENCODELAB_HOME", str(tmp_path))
    monkeypatch.setattr("opencodelab.cli.ensure_modal_auth", lambda interactive: True)
    monkeypatch.setattr(
        "opencodelab.cli.SecretStore.get_api_key",
        lambda self, name, **kwargs: "test-key",
    )
    monkeypatch.setattr(
        "opencodelab.cli.deploy_modal_app",
        lambda path, name, **kwargs: (
            f"https://workspace--{modal_endpoint_label('model')}.us-east.modal.direct"
        ),
    )

    assert cmd_gpu(["H100", "--yes"]) == 0

    updated = ConfigStore(tmp_path).load().profiles["model"]
    assert updated.gpu == "H100"
    assert updated.endpoint.endswith("modal.direct")
    output = capsys.readouterr().out
    assert "Model: owner/model" in output
    assert "0.001097/sec" in output


def test_gpu_change_rejects_different_active_workspace(monkeypatch, tmp_path: Path):
    profile = _profile()
    ConfigStore(tmp_path).save(LabConfig(default_profile="model", profiles={"model": profile}))
    monkeypatch.setenv("OPENCODELAB_HOME", str(tmp_path))
    monkeypatch.setattr("opencodelab.cli.ensure_modal_auth", lambda interactive: True)
    monkeypatch.setattr(
        "opencodelab.cli.SecretStore.get_api_key",
        lambda self, name, **kwargs: "test-key",
    )
    monkeypatch.setattr(
        "opencodelab.cli.require_modal_workspace",
        lambda expected: (_ for _ in ()).throw(RuntimeError("wrong workspace")),
    )
    monkeypatch.setattr(
        "opencodelab.cli.deploy_modal_app",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not deploy")),
    )

    with pytest.raises(RuntimeError, match="wrong workspace"):
        cmd_gpu(["H100", "--yes"])


def test_doctor_reports_failures_without_inference(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setenv("OPENCODELAB_HOME", str(tmp_path))
    monkeypatch.setattr("opencodelab.cli.ensure_opencode", lambda: (_ for _ in ()).throw(RuntimeError("missing")))
    monkeypatch.setattr("opencodelab.cli.ensure_modal_auth", lambda interactive: False)

    assert cmd_doctor() == 1
    output = capsys.readouterr().out
    assert "FAIL OpenCode" in output
    assert "FAIL Modal authentication" in output
    assert "No profile configured" in output
    assert "inference" not in output.lower()


def test_doctor_success_checks_local_state_only(monkeypatch, tmp_path: Path, capsys):
    profile = _profile()
    ConfigStore(tmp_path).save(LabConfig(default_profile="model", profiles={"model": profile}))
    monkeypatch.setenv("OPENCODELAB_HOME", str(tmp_path))
    monkeypatch.setattr("opencodelab.cli.ensure_opencode", lambda: "/usr/bin/opencode")
    monkeypatch.setattr("opencodelab.cli.ensure_modal_auth", lambda interactive: True)
    monkeypatch.setattr(
        "opencodelab.cli.SecretStore.get_api_key",
        lambda self, name, **kwargs: "test-key",
    )

    assert cmd_doctor() == 0
    assert "did not send an inference request" in capsys.readouterr().out


def test_main_turns_corrupt_config_into_clean_error(monkeypatch, tmp_path: Path, capsys):
    (tmp_path / "config.json").write_text("not json", encoding="utf-8")
    monkeypatch.setenv("OPENCODELAB_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["opencodelab", "list"])

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    error = capsys.readouterr().err
    assert "error:" in error
    assert "config.json" in error
    assert "Traceback" not in error


def test_remove_stops_remote_resources_before_local_cleanup(monkeypatch, tmp_path: Path):
    profile = bind_profile_endpoint(Profile(
        name="model",
        model_id="owner/model",
        gpu="L40S",
        max_model_len=32768,
        idle_timeout_seconds=90,
        endpoint=f"https://workspace--{modal_endpoint_label('model')}.modal.run",
        environment_name="opencodelab-model",
        app_name="opencodelab-model",
        secret_name="ocl-model-vllm",
        gateway_secret_name="ocl-model-gateway",
        workspace_id="ws-test-workspace",
        proxy_token_id="wk-model-token",
    ), "test-key")
    store = ConfigStore(tmp_path)
    store.save(LabConfig(default_profile="model", profiles={"model": profile}))
    store.write_generated(store.deployment_path("model"), "deployment")
    store.write_generated(store.overlay_path("model"), "overlay")
    monkeypatch.setenv("OPENCODELAB_HOME", str(tmp_path))
    events: list[str] = []

    class FakeSecrets:
        def __init__(self, root):
            pass

        def delete_api_key(self, name):
            events.append(f"local-secret:{name}")

    monkeypatch.setattr("opencodelab.cli.SecretStore", FakeSecrets)
    monkeypatch.setattr("opencodelab.cli.ensure_modal_auth", lambda interactive: True)
    monkeypatch.setattr("opencodelab.cli.modal_environment_exists", lambda name: True)
    monkeypatch.setattr(
        "opencodelab.cli.stop_modal_app",
        lambda name, **kwargs: events.append(f"stop:{name}"),
    )
    monkeypatch.setattr(
        "opencodelab.cli.delete_modal_proxy_token",
        lambda name: events.append(f"proxy-token:{name}"),
    )
    monkeypatch.setattr(
        "opencodelab.cli.delete_modal_secret",
        lambda name, **kwargs: events.append(f"remote-secret:{name}"),
    )
    monkeypatch.setattr(
        "opencodelab.cli.delete_modal_environment",
        lambda name: events.append(f"environment:{name}"),
    )

    assert cmd_remove(["model", "--yes"]) == 0

    assert events == [
        "stop:opencodelab-model",
        "remote-secret:ocl-model-gateway",
        "remote-secret:ocl-model-vllm",
        "proxy-token:wk-model-token",
        "environment:opencodelab-model",
        "local-secret:model",
    ]
    assert ConfigStore(tmp_path).load().profiles == {}
    assert not store.deployment_path("model").exists()
    assert not store.overlay_path("model").exists()


def test_remove_remote_failure_preserves_local_recovery_state(monkeypatch, tmp_path: Path):
    profile = bind_profile_endpoint(Profile(
        name="model",
        model_id="owner/model",
        gpu="L40S",
        max_model_len=32768,
        idle_timeout_seconds=90,
        endpoint=f"https://workspace--{modal_endpoint_label('model')}.modal.run",
        environment_name="opencodelab-model",
        app_name="opencodelab-model",
        secret_name="ocl-model-vllm",
        gateway_secret_name="ocl-model-gateway",
        workspace_id="ws-test-workspace",
        proxy_token_id="wk-model-token",
    ), "test-key")
    ConfigStore(tmp_path).save(LabConfig(default_profile="model", profiles={"model": profile}))
    monkeypatch.setenv("OPENCODELAB_HOME", str(tmp_path))
    monkeypatch.setattr("opencodelab.cli.ensure_modal_auth", lambda interactive: True)
    monkeypatch.setattr("opencodelab.cli.modal_environment_exists", lambda name: True)
    events: list[str] = []
    monkeypatch.setattr(
        "opencodelab.cli.stop_modal_app",
        lambda name, **kwargs: (
            events.append("stop"),
            (_ for _ in ()).throw(RuntimeError("stop failed")),
        )[1],
    )
    monkeypatch.setattr(
        "opencodelab.cli.delete_modal_secret",
        lambda name, **kwargs: events.append(f"secret:{name}"),
    )
    monkeypatch.setattr(
        "opencodelab.cli.delete_modal_proxy_token",
        lambda name: events.append(f"token:{name}"),
    )
    monkeypatch.setattr(
        "opencodelab.cli.delete_modal_environment",
        lambda name: events.append("environment"),
    )

    with pytest.raises(RuntimeError, match="stop failed"):
        cmd_remove(["model", "--yes"])

    assert "model" in ConfigStore(tmp_path).load().profiles
    assert events == [
        "stop",
        "secret:ocl-model-gateway",
        "secret:ocl-model-vllm",
        "token:wk-model-token",
    ]


def test_remove_recovers_when_environment_was_already_deleted(monkeypatch, tmp_path: Path):
    profile = _profile()
    ConfigStore(tmp_path).save(LabConfig(default_profile="model", profiles={"model": profile}))
    monkeypatch.setenv("OPENCODELAB_HOME", str(tmp_path))
    monkeypatch.setattr("opencodelab.cli.ensure_modal_auth", lambda interactive: True)
    monkeypatch.setattr("opencodelab.cli.modal_environment_exists", lambda name: False)
    revoked: list[str] = []
    monkeypatch.setattr(
        "opencodelab.cli.delete_modal_proxy_token",
        lambda token_id: revoked.append(token_id),
    )
    monkeypatch.setattr(
        "opencodelab.cli.stop_modal_app",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("environment is absent")),
    )
    monkeypatch.setattr("opencodelab.cli.SecretStore.delete_api_key", lambda self, name: None)

    assert cmd_remove(["model", "--yes"]) == 0
    assert revoked == ["wk-model-token"]
    assert ConfigStore(tmp_path).load().profiles == {}


def test_profile_commands_list_select_and_show_non_secret_config(monkeypatch, tmp_path: Path, capsys):
    first = _profile("first")
    second = _profile("second", gpu="H100")
    config = LabConfig(default_profile="first", profiles={"first": first, "second": second})
    ConfigStore(tmp_path).save(config)
    monkeypatch.setenv("OPENCODELAB_HOME", str(tmp_path))

    assert cmd_list() == 0
    assert "* first" in capsys.readouterr().out
    assert cmd_use(["second"]) == 0
    assert ConfigStore(tmp_path).load().default_profile == "second"
    assert cmd_config(["--profile", "second"]) == 0
    output = capsys.readouterr().out
    assert '"gpu": "H100"' in output
    assert "api_key" not in output.lower()


def test_empty_list_and_unknown_profile_errors(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setenv("OPENCODELAB_HOME", str(tmp_path))
    assert cmd_list() == 0
    assert "No OpenCodeLab profiles" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="not configured"):
        launch([])


def test_signup_and_dispatch_help_do_not_mutate_state(monkeypatch, capsys):
    opened: list[bool] = []
    monkeypatch.setattr("opencodelab.cli.open_signup", lambda: opened.append(True))
    assert cmd_signup(["--open"]) == 0
    assert opened == [True]
    assert cmd_signup([]) == 0
    assert "Modal account/signup" in capsys.readouterr().out
    assert _dispatch(["help"]) == 0
    assert "OpenCodeLab" in capsys.readouterr().out
    assert _dispatch(["--help"]) == 0
    assert "OpenCodeLab" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="does not accept"):
        _dispatch(["doctor", "unexpected"])


def test_login_uses_bundled_modal_and_reports_result(monkeypatch, capsys):
    monkeypatch.setattr("opencodelab.cli.ensure_modal_auth", lambda interactive: True)
    assert cmd_login([]) == 0
    assert "authenticated" in capsys.readouterr().out.lower()


def test_launch_selects_profile_and_requires_credential(monkeypatch, tmp_path: Path):
    first = _profile("first")
    second = _profile("second")
    ConfigStore(tmp_path).save(
        LabConfig(default_profile="first", profiles={"first": first, "second": second})
    )
    monkeypatch.setenv("OPENCODELAB_HOME", str(tmp_path))
    monkeypatch.setattr("opencodelab.cli.SecretStore.get_api_key", lambda self, name: None)
    with pytest.raises(SystemExit, match="credential"):
        launch(["--profile", "second", "run", "task"])

    monkeypatch.setattr(
        "opencodelab.cli.SecretStore.get_api_key",
        lambda self, name, **kwargs: "test-key",
    )
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        "opencodelab.cli.launch_opencode",
        lambda profile, key, store, args: observed.update(profile=profile.name, key=key, args=args) or 9,
    )
    assert launch(["--profile", "second", "run", "task"]) == 9
    assert observed == {"profile": "second", "key": "test-key", "args": ["run", "task"]}
    assert _extract_profile(["run", "task"]) == (None, ["run", "task"])


def test_setup_save_failure_before_modal_creates_no_remote_state(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("OPENCODELAB_HOME", str(tmp_path))
    monkeypatch.setattr("opencodelab.cli.ensure_modal_auth", lambda interactive: True)

    def fail_save(self, config):
        raise OSError("disk full")

    monkeypatch.setattr(ConfigStore, "save", fail_save)
    monkeypatch.setattr(
        "opencodelab.cli.create_modal_environment",
        lambda name: (_ for _ in ()).throw(AssertionError("no Modal resources allowed")),
    )
    monkeypatch.setattr(
        "opencodelab.cli.create_modal_proxy_token",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("no tokens allowed")),
    )

    with pytest.raises(RuntimeError, match="no Modal resources"):
        cmd_setup(["owner/model", "--yes"])

    assert not (tmp_path / "config.json").exists()


def test_setup_environment_failure_keeps_local_state_clean(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("OPENCODELAB_HOME", str(tmp_path))
    monkeypatch.setattr("opencodelab.cli.ensure_modal_auth", lambda interactive: True)
    monkeypatch.setattr(
        "opencodelab.cli.create_modal_environment",
        lambda name: (_ for _ in ()).throw(RuntimeError("collision")),
    )

    with pytest.raises(RuntimeError, match="choose another profile name"):
        cmd_setup(["owner/model", "--yes"])

    assert not ConfigStore(tmp_path).load().profiles


def test_config_accepts_bare_and_optional_show_forms(monkeypatch, tmp_path: Path, capsys):
    profile = _profile()
    ConfigStore(tmp_path).save(LabConfig(default_profile="model", profiles={"model": profile}))
    monkeypatch.setenv("OPENCODELAB_HOME", str(tmp_path))

    assert cmd_config([]) == 0
    assert '"name": "model"' in capsys.readouterr().out
    assert cmd_config(["show"]) == 0
    assert '"name": "model"' in capsys.readouterr().out
    assert cmd_config(["show", "--profile", "model"]) == 0
    assert '"model_id": "owner/model"' in capsys.readouterr().out
    with pytest.raises(SystemExit, match="does not accept"):
        cmd_config(["bogus"])


def test_setup_post_deploy_save_failure_reports_live_deployment(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("OPENCODELAB_HOME", str(tmp_path))

    class FakeSecrets:
        def __init__(self, root):
            pass

        def get_api_key(self, profile_name, **kwargs):
            return None

        def set_api_key(self, profile_name, value):
            return "test keyring"

    monkeypatch.setattr("opencodelab.cli.SecretStore", FakeSecrets)
    monkeypatch.setattr("opencodelab.cli.ensure_modal_auth", lambda interactive: True)
    monkeypatch.setattr("opencodelab.cli.create_modal_environment", lambda name: None)
    monkeypatch.setattr(
        "opencodelab.cli.create_modal_proxy_token",
        lambda **kwargs: (
            kwargs["on_token_id"]("wk-model-token"),
            ("wk-model-token", "ws-model-secret"),
        )[1],
    )
    monkeypatch.setattr("opencodelab.cli.allow_modal_proxy_token", lambda *a: True)
    monkeypatch.setattr("opencodelab.cli.create_gateway_secret", lambda *a, **k: None)
    monkeypatch.setattr("opencodelab.cli.create_modal_secret", lambda *a, **k: None)
    monkeypatch.setattr(
        "opencodelab.cli.deploy_modal_app",
        lambda path, name, **kwargs: (
            f"https://workspace--{modal_endpoint_label('model')}.modal.run"
        ),
    )
    original_save = ConfigStore.save

    def fail_final_save(self, config):
        for saved in config.profiles.values():
            if saved.endpoint:
                raise OSError("disk full")
        return original_save(self, config)

    monkeypatch.setattr(ConfigStore, "save", fail_final_save)

    with pytest.raises(RuntimeError, match="succeeded and is live"):
        cmd_setup(["owner/model", "--yes"])

    recovery = ConfigStore(tmp_path).load().profiles["model"]
    assert recovery.endpoint == ""
