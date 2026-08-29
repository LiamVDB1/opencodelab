from opencodelab.modal_client import modal_command, parse_modal_endpoint


def test_modal_command_uses_current_python_module():
    cmd = modal_command("token", "info")
    assert cmd[1:3] == ["-m", "modal"]
    assert cmd[-2:] == ["token", "info"]


def test_parse_modal_endpoint_prefers_serve_url():
    output = "Created web function serve => https://workspace--opencodelab-demo-serve.modal.run"
    assert parse_modal_endpoint(output) == "https://workspace--opencodelab-demo-serve.modal.run"


def test_parse_modal_endpoint_handles_missing_url():
    assert parse_modal_endpoint("deployment succeeded") is None
