from opencodelab.models import (
    ModelSettings,
    infer_model_settings,
    normalize_huggingface_model,
    safe_slug,
)


def test_normalize_huggingface_url_and_id():
    assert normalize_huggingface_model("gubernac/qwen3.8-27b-unc-fp8") == "gubernac/qwen3.8-27b-unc-fp8"
    assert normalize_huggingface_model("https://huggingface.co/gubernac/qwen3.8-27b-unc-fp8") == "gubernac/qwen3.8-27b-unc-fp8"
    assert normalize_huggingface_model("https://hf.co/gubernac/qwen3.8-27b-unc-fp8/") == "gubernac/qwen3.8-27b-unc-fp8"


def test_invalid_huggingface_model_rejected():
    for value in (
        "qwen",
        "https://example.com/a/b",
        "https://huggingface.co/owner/model?revision=main",
        "https://huggingface.co/owner/model#readme",
        "owner/model\nBAD=1",
        "../model",
    ):
        try:
            normalize_huggingface_model(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {value!r}")


def test_qwen3_infers_tool_and_reasoning_parsers():
    settings = infer_model_settings("gubernac/qwen3.8-27b-unc-fp8")
    assert settings.tool_call_parser == "qwen3_coder"
    assert settings.reasoning_parser == "qwen3"


def test_unknown_model_does_not_invent_tool_parser():
    settings = infer_model_settings("example/custom-model")
    assert settings == ModelSettings()


def test_safe_slug_is_stable_and_bounded():
    assert safe_slug("My Qwen 3.8 / Test") == "my-qwen-3-8-test"
    assert len(safe_slug("x" * 200)) <= 40
