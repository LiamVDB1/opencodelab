import hashlib
import json
import os
import subprocess
import sysconfig
from pathlib import Path

import pytest

from opencodelab.modal_client import (
    DEFAULT_MODAL_SIGNUP_URL,
    allow_modal_proxy_token,
    create_gateway_secret,
    create_modal_environment,
    create_modal_proxy_token,
    create_modal_secret,
    delete_modal_environment,
    delete_modal_proxy_token,
    delete_modal_secret,
    deploy_modal_app,
    ensure_modal_auth,
    get_modal_workspace_id,
    modal_command,
    modal_is_authenticated,
    modal_subprocess_env,
    normalize_modal_endpoint,
    parse_modal_endpoint,
    print_signup_message,
    require_modal_workspace,
    stop_modal_app,
)


def test_modal_command_uses_absolute_active_environment_console_script():
    cmd = modal_command("token", "info")
    assert Path(cmd[0]).is_absolute()
    assert Path(cmd[0]).parent == Path(sysconfig.get_path("scripts")).resolve()
    assert Path(cmd[0]).name in {"modal", "modal.exe"}
    assert cmd[-2:] == ["token", "info"]


@pytest.mark.parametrize("fake_kind", ["module", "package"])
def test_modal_console_script_never_imports_modal_from_cwd_or_pythonpath(
    monkeypatch, tmp_path: Path, fake_kind: str
):
    sentinel = tmp_path / "IMPORTED"
    source = f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('imported')\n"
    if fake_kind == "module":
        (tmp_path / "modal.py").write_text(source, encoding="utf-8")
    else:
        package = tmp_path / "modal"
        package.mkdir()
        (package / "__init__.py").write_text(source, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "fake-home"))

    result = subprocess.run(
        modal_command("--version"),
        cwd=tmp_path,
        env=modal_subprocess_env(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not sentinel.exists()
    assert "PYTHONPATH" not in modal_subprocess_env()
    assert "PYTHONHOME" not in modal_subprocess_env()


def test_modal_workspace_id_comes_from_token_info_without_exposing_token(monkeypatch):
    output = "Token: ak-test\nWorkspace: Example Team (ws-example_123)\nUser: person (u-test)\n"
    result = subprocess.CompletedProcess(["modal"], 0, stdout=output, stderr="")
    monkeypatch.setattr("opencodelab.modal_client.subprocess.run", lambda *a, **k: result)

    assert get_modal_workspace_id() == "ws-example_123"


@pytest.mark.parametrize(
    "result",
    [
        subprocess.CompletedProcess(["modal"], 1, stdout="", stderr="not authenticated"),
        subprocess.CompletedProcess(["modal"], 0, stdout="Token: ak-test\n", stderr=""),
        subprocess.CompletedProcess(
            ["modal"], 0, stdout="Workspace: Example (../../bad)\n", stderr=""
        ),
    ],
)
def test_modal_workspace_id_rejects_failed_or_malformed_identity(monkeypatch, result):
    monkeypatch.setattr("opencodelab.modal_client.subprocess.run", lambda *a, **k: result)
    with pytest.raises(RuntimeError, match="workspace"):
        get_modal_workspace_id()


def test_workspace_mismatch_stops_lifecycle_mutation(monkeypatch):
    monkeypatch.setattr(
        "opencodelab.modal_client.get_modal_workspace_id", lambda: "ws-other-workspace"
    )
    with pytest.raises(RuntimeError, match="does not own"):
        require_modal_workspace("ws-original-workspace")


def test_parse_modal_endpoint_prefers_serve_url():
    output = "Created web function serve => https://workspace--opencodelab-demo-serve.modal.run"
    assert parse_modal_endpoint(output) == "https://workspace--opencodelab-demo-serve.modal.run"


def test_parse_modal_endpoint_handles_missing_url():
    assert parse_modal_endpoint("deployment succeeded") is None
    assert parse_modal_endpoint("https://workspace--app.modal.run.evil.example") is None


def test_parse_modal_endpoint_accepts_current_modal_direct_url():
    output = "Created server => https://workspace--app-serve.us-east.modal.direct"
    assert parse_modal_endpoint(output) == "https://workspace--app-serve.us-east.modal.direct"


def test_endpoint_label_binding_accepts_modal_hash_truncation():
    source = "workspace-with-a-deliberately-long-environment-suffix"
    label = "ocl-1234567890abcdef"
    full = f"{source}--{label}"
    truncated = f"{full[:56]}-{hashlib.sha256(full.encode()).hexdigest()[:6]}"
    endpoint = f"https://{truncated}.modal.run"

    assert normalize_modal_endpoint(endpoint, endpoint_label=label) == endpoint
    with pytest.raises(ValueError, match="gateway"):
        normalize_modal_endpoint(endpoint, endpoint_label="ocl-wrong")


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://workspace--app.modal.run",
        "https://attacker.example",
        "https://workspace--app.modal.run.evil.example",
        "https://user@workspace--app.modal.run",
        "https://workspace--app.modal.run/path",
        "https://workspace--app.modal.run?key=value",
    ],
)
def test_modal_endpoint_rejects_credential_exfiltration_origins(endpoint: str):
    with pytest.raises(ValueError):
        normalize_modal_endpoint(endpoint)


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://workspace--ocl-abc.evil.modal.run",
        "https://workspace--ocl-abc.evil.modal.direct",
        "https://workspace--ocl-abc.modal.direct",
        "https://workspace--ocl-abc.a.b.modal.direct",
        "https://workspace--ocl-abc.not-a-region.modal.direct",
        "https://workspace--ocl-abc.not-a-region.modal.run",
        "https://workspace--ocl-abc.a.us-east.modal.run",
    ],
)
def test_modal_endpoint_rejects_unofficial_intermediate_host_labels(endpoint: str):
    with pytest.raises(ValueError, match="HTTPS origin"):
        normalize_modal_endpoint(endpoint)


def test_modal_endpoint_accepts_official_shapes():
    assert (
        normalize_modal_endpoint("https://workspace--ocl-abc.modal.run")
        == "https://workspace--ocl-abc.modal.run"
    )
    assert (
        normalize_modal_endpoint("https://workspace--ocl-abc-serve.us-east.modal.direct")
        == "https://workspace--ocl-abc-serve.us-east.modal.direct"
    )
    assert (
        normalize_modal_endpoint("https://workspace--ocl-abc.us-east.modal.run")
        == "https://workspace--ocl-abc.us-east.modal.run"
    )


def test_deploy_parses_regional_modal_run_endpoint_with_wide_console(monkeypatch, tmp_path: Path):
    # Regression: Modal 1.6 issues <source>--<label>.<region>.modal.run endpoints,
    # and at the default 80 columns its tree output wrapped them mid-hostname.
    observed: dict[str, str] = {}

    def fake_run(command, **kwargs):
        observed["columns"] = kwargs["env"]["COLUMNS"]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="\u2514\u2500\u2500 https://workspace--ocl-test.us-east.modal.run (unauthenticated)",
            stderr="",
        )

    monkeypatch.setattr("opencodelab.modal_client.subprocess.run", fake_run)

    assert deploy_modal_app(
        tmp_path / "app.py",
        "app-name",
        environment_name="opencodelab-test",
        endpoint_label="ocl-test",
    ) == "https://workspace--ocl-test.us-east.modal.run"
    assert observed["columns"] == "1000"


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://workspace--ocl-test.evil.modal.run",
        "https://workspace--ocl-test.evil.modal.direct",
        "https://workspace--ocl-test.a.b.modal.direct",
        "https://workspace--ocl-test.bad_region.modal.direct",
        "https://workspace--ocl-test.modal.run.extra.modal.direct",
    ],
)
def test_modal_endpoint_rejects_unofficial_host_shapes(endpoint: str):
    with pytest.raises(ValueError, match="HTTPS origin"):
        normalize_modal_endpoint(endpoint, endpoint_label="ocl-test")


def test_modal_endpoint_accepts_official_modal_direct_region_shape():
    endpoint = "https://workspace--ocl-test.us-east.modal.direct"
    assert normalize_modal_endpoint(endpoint, endpoint_label="ocl-test") == endpoint


def test_signup_is_always_the_ordinary_modal_url(monkeypatch, capsys):
    monkeypatch.setenv("OPENCODELAB_MODAL_SIGNUP_URL", "https://unsupported.example/ref")
    print_signup_message()

    output = capsys.readouterr().out
    assert DEFAULT_MODAL_SIGNUP_URL in output
    assert "unsupported.example" not in output
    assert "partner" not in output.lower()
    assert "referral" not in output.lower()


def test_create_modal_secret_uses_private_temp_file_and_cleans_it(monkeypatch, tmp_path: Path):
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        secret_path = Path(command[command.index("--from-json") + 1])
        observed["path"] = secret_path
        observed["mode"] = os.stat(secret_path).st_mode & 0o777
        observed["payload"] = json.loads(secret_path.read_text(encoding="utf-8"))
        observed["command"] = command
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("opencodelab.modal_client.tempfile.tempdir", str(tmp_path))
    monkeypatch.setattr("opencodelab.modal_client.subprocess.run", fake_run)
    create_modal_secret(
        "safe-name",
        "endpoint-secret",
        environment_name="opencodelab-test",
        hf_token="hf-secret",
        force=True,
    )

    assert observed["mode"] == 0o600
    assert observed["payload"] == {
        "OPENCODELAB_API_KEY": "endpoint-secret",
        "HF_TOKEN": "hf-secret",
        "HUGGING_FACE_HUB_TOKEN": "hf-secret",
    }
    assert "endpoint-secret" not in observed["command"]
    assert "hf-secret" not in observed["command"]
    assert not Path(observed["path"]).exists()


def test_gateway_secret_keeps_workspace_proxy_credential_off_command_line(monkeypatch, tmp_path: Path):
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        secret_path = Path(command[command.index("--from-json") + 1])
        observed["payload"] = json.loads(secret_path.read_text(encoding="utf-8"))
        observed["command"] = command
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("opencodelab.modal_client.tempfile.tempdir", str(tmp_path))
    monkeypatch.setattr("opencodelab.modal_client.subprocess.run", fake_run)
    create_gateway_secret(
        "ocl-model-gateway",
        "client-key",
        environment_name="opencodelab-model",
        proxy_token_id="wk-proxy",
        proxy_token_secret="ws-proxy",
    )

    assert observed["payload"] == {
        "OPENCODELAB_API_KEY": "client-key",
        "OPENCODELAB_PROXY_TOKEN_ID": "wk-proxy",
        "OPENCODELAB_PROXY_TOKEN_SECRET": "ws-proxy",
    }
    assert "client-key" not in observed["command"]
    assert "ws-proxy" not in observed["command"]


def test_environment_creation_and_proxy_grant_are_explicit(monkeypatch):
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("opencodelab.modal_client.subprocess.run", fake_run)
    create_modal_environment("opencodelab-model")
    assert allow_modal_proxy_token("wk-model", "opencodelab-model") is True

    assert commands[0][-3:] == ["environment", "create", "opencodelab-model"]
    assert commands[1][-5:] == [
        "workspace",
        "proxy-tokens",
        "allow",
        "wk-model",
        "opencodelab-model",
    ]


def test_proxy_grant_is_skipped_for_workspace_wide_tokens(monkeypatch):
    # Regression: Modal 1.6 rejects grants in workspaces without RBAC, where proxy
    # tokens are workspace-wide; setup used to abort and leave a recovery profile.
    result = subprocess.CompletedProcess(
        ["modal"], 1, stdout="", stderr="Error: Token is not environment-scoped"
    )
    monkeypatch.setattr("opencodelab.modal_client.subprocess.run", lambda *a, **k: result)

    assert allow_modal_proxy_token("wk-model", "opencodelab-model") is False


def test_other_proxy_grant_failures_still_fail_closed(monkeypatch):
    result = subprocess.CompletedProcess(["modal"], 1, stdout="", stderr="Environment not found")
    monkeypatch.setattr("opencodelab.modal_client.subprocess.run", lambda *a, **k: result)

    with pytest.raises(RuntimeError, match="Could not authorize the Modal proxy token"):
        allow_modal_proxy_token("wk-model", "opencodelab-model")


def test_deploy_failure_is_actionable_and_does_not_parse_endpoint(monkeypatch, tmp_path: Path):
    result = subprocess.CompletedProcess(["modal"], 2, stdout="", stderr="build failed")
    monkeypatch.setattr("opencodelab.modal_client.subprocess.run", lambda *a, **k: result)

    with pytest.raises(RuntimeError, match="deployment failed"):
        deploy_modal_app(
            tmp_path / "app.py",
            "app-name",
            environment_name="opencodelab-test",
            endpoint_label="ocl-test",
        )


def test_modal_teardown_commands_are_noninteractive(monkeypatch):
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("opencodelab.modal_client.subprocess.run", fake_run)
    stop_modal_app("opencodelab-model", environment_name="opencodelab-model")
    delete_modal_secret("ocl-model-vllm", environment_name="opencodelab-model")

    assert commands[0][-6:] == [
        "app",
        "stop",
        "opencodelab-model",
        "--yes",
        "--env",
        "opencodelab-model",
    ]
    assert commands[1][-7:] == [
        "secret",
        "delete",
        "ocl-model-vllm",
        "--allow-missing",
        "--yes",
        "--env",
        "opencodelab-model",
    ]


def test_proxy_token_creation_parses_secret_without_printing_it(monkeypatch):
    payload = {
        "Modal-Key": "wk-token-id",
        "Modal-Secret": "ws-token-secret",
        "Authorization": "Bearer wk-token-id.ws-token-secret",
    }
    results = iter(
        [
            subprocess.CompletedProcess(["modal"], 0, stdout="[]", stderr=""),
            subprocess.CompletedProcess(["modal"], 0, stdout=json.dumps(payload), stderr=""),
        ]
    )
    monkeypatch.setattr(
        "opencodelab.modal_client.subprocess.run", lambda *a, **k: next(results)
    )

    token_id, token_secret = create_modal_proxy_token()

    assert token_id == "wk-token-id"
    assert token_secret == "ws-token-secret"


def test_proxy_token_creation_rejects_malformed_output(monkeypatch):
    results = iter(
        [
            subprocess.CompletedProcess(["modal"], 0, stdout="[]", stderr=""),
            subprocess.CompletedProcess(
                ["modal"], 0, stdout='{"Modal-Key":"wk-new-token"}', stderr=""
            ),
            subprocess.CompletedProcess(
                ["modal"], 0, stdout='[{"token_id":"wk-new-token"}]', stderr=""
            ),
            subprocess.CompletedProcess(
                ["modal"], 0, stdout='[{"token_id":"wk-new-token"}]', stderr=""
            ),
            subprocess.CompletedProcess(["modal"], 0, stdout="", stderr=""),
        ]
    )
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return next(results)

    monkeypatch.setattr("opencodelab.modal_client.subprocess.run", fake_run)
    recorded: list[str] = []
    with pytest.raises(RuntimeError, match="malformed.*revoked"):
        create_modal_proxy_token(on_token_id=recorded.append)

    assert recorded == ["wk-new-token"]
    assert any(command[-2:] == ["wk-new-token", "--yes"] for command in commands)


def test_proxy_token_parse_failure_requires_manual_cleanup_when_reconciliation_fails(
    monkeypatch,
):
    results = iter(
        [
            subprocess.CompletedProcess(["modal"], 0, stdout="[]", stderr=""),
            subprocess.CompletedProcess(["modal"], 0, stdout="not-json", stderr=""),
            subprocess.CompletedProcess(["modal"], 1, stdout="", stderr="list failed"),
        ]
    )
    monkeypatch.setattr(
        "opencodelab.modal_client.subprocess.run", lambda *args, **kwargs: next(results)
    )

    with pytest.raises(RuntimeError, match="MANUAL CLEANUP REQUIRED"):
        create_modal_proxy_token()


def test_proxy_token_creation_snapshots_before_mutation_and_records_id_before_return(
    monkeypatch,
):
    payload = {"Modal-Key": "wk-token-id", "Modal-Secret": "ws-token-secret"}
    results = iter(
        [
            subprocess.CompletedProcess(
                ["modal"], 0, stdout='[{"token_id":"wk-existing"}]', stderr=""
            ),
            subprocess.CompletedProcess(["modal"], 0, stdout=json.dumps(payload), stderr=""),
        ]
    )
    events: list[str] = []

    def fake_run(command, **kwargs):
        events.append("list" if "list" in command else "create")
        return next(results)

    monkeypatch.setattr("opencodelab.modal_client.subprocess.run", fake_run)
    result = create_modal_proxy_token(on_token_id=lambda token_id: events.append(token_id))

    assert result == ("wk-token-id", "ws-token-secret")
    assert events == ["list", "create", "wk-token-id"]


def test_proxy_token_creation_supports_modal_1_5_1_json(monkeypatch):
    payload = {"Modal-Key": "wk-token-id", "Modal-Secret": "ws-token-secret"}
    results = iter(
        [
            subprocess.CompletedProcess(["modal"], 0, stdout="[]", stderr=""),
            subprocess.CompletedProcess(["modal"], 0, stdout=json.dumps(payload), stderr=""),
        ]
    )
    monkeypatch.setattr(
        "opencodelab.modal_client.subprocess.run", lambda *a, **k: next(results)
    )
    assert create_modal_proxy_token() == ("wk-token-id", "ws-token-secret")


def test_proxy_token_deletion_is_idempotent(monkeypatch):
    commands: list[list[str]] = []
    results = iter(
        [
            subprocess.CompletedProcess(
                ["modal"], 0, stdout='[{"token_id":"wk-other"}]', stderr=""
            ),
            subprocess.CompletedProcess(
                ["modal"], 0, stdout='[{"token_id":"wk-target"}]', stderr=""
            ),
            subprocess.CompletedProcess(["modal"], 0, stdout="", stderr=""),
        ]
    )

    def fake_run(command, **kwargs):
        commands.append(command)
        return next(results)

    monkeypatch.setattr("opencodelab.modal_client.subprocess.run", fake_run)
    delete_modal_proxy_token("wk-target")
    delete_modal_proxy_token("wk-target")

    assert sum(command[-1] == "--yes" for command in commands) == 1


def test_environment_deletion_is_idempotent(monkeypatch):
    commands: list[list[str]] = []
    results = iter(
        [
            subprocess.CompletedProcess(["modal"], 0, stdout="[]", stderr=""),
            subprocess.CompletedProcess(
                ["modal"], 0, stdout='[{"name":"opencodelab-model"}]', stderr=""
            ),
            subprocess.CompletedProcess(["modal"], 0, stdout="", stderr=""),
        ]
    )

    def fake_run(command, **kwargs):
        commands.append(command)
        return next(results)

    monkeypatch.setattr("opencodelab.modal_client.subprocess.run", fake_run)
    delete_modal_environment("opencodelab-model")
    delete_modal_environment("opencodelab-model")
    assert sum(command[-1] == "--yes" for command in commands) == 1


@pytest.mark.parametrize("message", ["App is already stopped.", "No App with name 'missing' found"])
def test_stopping_an_absent_app_is_success(monkeypatch, message):
    result = subprocess.CompletedProcess(["modal"], 1, stdout="", stderr=message)
    monkeypatch.setattr("opencodelab.modal_client.subprocess.run", lambda *a, **k: result)
    stop_modal_app("opencodelab-model", environment_name="opencodelab-model")


def test_modal_auth_checks_are_quiet_and_interactive_fallback_is_explicit(monkeypatch, capsys):
    results = iter(
        [
            subprocess.CompletedProcess(["modal"], 1),
            subprocess.CompletedProcess(["modal"], 1),
            subprocess.CompletedProcess(["modal"], 0),
        ]
    )
    monkeypatch.setattr("opencodelab.modal_client.subprocess.run", lambda *a, **k: next(results))

    assert not modal_is_authenticated()
    assert ensure_modal_auth(interactive=True)
    assert "login/setup flow" in capsys.readouterr().out


def test_noninteractive_auth_failure_does_not_start_setup(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr("opencodelab.modal_client.subprocess.run", fake_run)
    assert not ensure_modal_auth(interactive=False)
    assert len(calls) == 1


def test_deploy_success_requires_a_reported_endpoint(monkeypatch, tmp_path: Path):
    missing = subprocess.CompletedProcess(["modal"], 0, stdout="deployed", stderr="")
    monkeypatch.setattr("opencodelab.modal_client.subprocess.run", lambda *a, **k: missing)
    with pytest.raises(RuntimeError, match="could not find"):
        deploy_modal_app(
            tmp_path / "app.py",
            "app-name",
            environment_name="opencodelab-test",
            endpoint_label="ocl-test",
        )

    success = subprocess.CompletedProcess(
        ["modal"],
        0,
        stdout="Created https://workspace--ocl-test.us-east.modal.direct",
        stderr="",
    )
    monkeypatch.setattr("opencodelab.modal_client.subprocess.run", lambda *a, **k: success)
    assert deploy_modal_app(
        tmp_path / "app.py",
        "app-name",
        environment_name="opencodelab-test",
        endpoint_label="ocl-test",
    ).endswith("modal.direct")


def test_modal_endpoint_label_accepts_workspace_source_with_double_hyphen():
    label = "ocl-1234567890abcdef"
    endpoint = f"https://ws--team--{label}.modal.run"

    assert normalize_modal_endpoint(endpoint, endpoint_label=label) == endpoint
    direct = f"https://ws--team--{label}.us-east.modal.direct"
    assert normalize_modal_endpoint(direct, endpoint_label=label) == direct


def test_modal_endpoint_label_accepts_truncated_workspace_source_with_double_hyphen():
    label = "ocl-1234567890abcdef"
    source = "ws--team-with-a-deliberately-long-workspace-name-xyz"
    full = f"{source}--{label}"
    truncated = f"{full[:56]}-{hashlib.sha256(full.encode()).hexdigest()[:6]}"
    endpoint = f"https://{truncated}.modal.run"

    assert normalize_modal_endpoint(endpoint, endpoint_label=label) == endpoint


def test_modal_endpoint_label_rejects_wrong_or_broken_host_forms():
    label = "ocl-1234567890abcdef"
    for endpoint in (
        f"https://ws--team--{label}.modal.run.extra.modal.direct",
        f"https://ws--team-{label}.modal.run",
        f"https://--{label}.modal.run",
        f"https://ws--team--{label}.modal.run/path",
    ):
        with pytest.raises(ValueError, match="gateway|HTTPS origin"):
            normalize_modal_endpoint(endpoint, endpoint_label=label)
