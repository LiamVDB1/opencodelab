from opencodelab.config import Profile
from opencodelab.modal_template import render_modal_app


def _profile(**overrides):
    values = {
        "name": "qwen38",
        "model_id": "gubernac/qwen3.8-27b-unc-fp8",
        "gpu": "L40S",
        "max_model_len": 65536,
        "idle_timeout_seconds": 90,
        "endpoint": "",
        "app_name": "opencodelab-qwen38",
        "secret_name": "opencodelab-qwen38-auth",
        "tool_call_parser": "qwen3_coder",
        "reasoning_parser": "qwen3",
        "trust_remote_code": False,
        "speculative_mtp_tokens": 3,
        "language_model_only": True,
        "vllm_version": "0.24.0",
    }
    values.update(overrides)
    return Profile(**values)


def test_rendered_modal_app_is_generic_and_cost_bounded():
    source = render_modal_app(_profile())
    assert 'MODEL_ID = "gubernac/qwen3.8-27b-unc-fp8"' in source
    assert 'GPU = "L40S"' in source
    assert "max_containers=1" in source
    assert "IDLE_TIMEOUT_SECONDS = 90" in source
    assert "scaledown_window=IDLE_TIMEOUT_SECONDS" in source
    assert '"--max-model-len", str(MAX_MODEL_LEN)' in source
    assert '"--tool-call-parser"' in source
    assert '"qwen3_coder"' in source
    assert '"--reasoning-parser"' in source
    assert '"qwen3"' in source
    assert '"--trust-remote-code"' not in source


def test_remote_code_is_opt_in():
    source = render_modal_app(_profile(trust_remote_code=True))
    assert '"--trust-remote-code"' in source


def test_unknown_model_can_render_without_tool_parser():
    source = render_modal_app(
        _profile(
            model_id="example/custom-model",
            tool_call_parser=None,
            reasoning_parser=None,
            speculative_mtp_tokens=0,
            language_model_only=False,
        )
    )
    assert "--tool-call-parser" not in source
    assert "--reasoning-parser" not in source
    assert "--speculative-config" not in source
    assert "--language-model-only" not in source
