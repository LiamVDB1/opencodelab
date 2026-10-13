"""Executable behavior tests for the generated gateway, using stub Modal/FastAPI/httpx."""

import asyncio
import sys
import types

import pytest

from opencodelab.config import Profile, modal_endpoint_label
from opencodelab.modal_template import render_modal_app

ENDPOINT_LABEL = modal_endpoint_label("qwen38")
MODEL_MODAL_RUN = f"https://workspace--{ENDPOINT_LABEL}-model.modal.run"
MODEL_MODAL_DIRECT = f"https://workspace--{ENDPOINT_LABEL}-model.us-east.modal.direct"
MODEL_MODAL_RUN_REGIONAL = f"https://workspace--{ENDPOINT_LABEL}-model.us-east.modal.run"
AUTH = {"authorization": "Bearer gateway-key"}


class _Chain:
    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class _WebFn:
    def __init__(self, fn, state):
        self.raw = fn
        self.state = state

    def get_web_url(self):
        self.state["web_url_calls"] += 1
        return self.state["model_url"]


class _FakeFastAPI:
    def __init__(self, **kwargs):
        self.routes = {}

    def api_route(self, path, methods=None):
        def deco(fn):
            self.routes[path] = fn
            return fn

        return deco


class _JSONResponse:
    def __init__(self, content=None, status_code=200):
        self.body = content
        self.status_code = status_code


class _StreamingResponse:
    def __init__(self, content, status_code=200, headers=None):
        self.body = content
        self.status_code = status_code
        self.headers = dict(headers or {})


def _fastapi_stub():
    stub = types.ModuleType("fastapi")
    stub.FastAPI = _FakeFastAPI
    stub.Request = object
    responses = types.ModuleType("fastapi.responses")
    responses.JSONResponse = _JSONResponse
    responses.StreamingResponse = _StreamingResponse
    stub.responses = responses
    return stub


def _httpx_stub(state):
    stub = types.ModuleType("httpx")

    class RequestError(Exception):
        pass

    class Request:
        def __init__(self, method, url, headers=None, content=b""):
            self.method = method
            self.url = url
            self.headers = headers or {}
            self.content = content

    class Response:
        def __init__(self, status_code=200, headers=None, content=b"", next_request=None):
            self.status_code = status_code
            self.headers = headers or {}
            self.next_request = next_request
            self._content = content
            self.closed = False

        async def aiter_raw(self):
            if self._content:
                yield self._content

        async def aclose(self):
            self.closed = True
            state["upstream_closed"] = True

    class AsyncClient:
        def __init__(self, **kwargs):
            state["client_kwargs"] = kwargs
            state["client_closed"] = False
            state["upstream_closed"] = False

        def build_request(self, method, url, params=None, headers=None, content=b"", **kwargs):
            request = Request(method, url, headers=headers, content=content)
            state["built"].append(request)
            return request

        async def send(self, request, stream=False):
            state["sent"].append(request)
            if state.get("send_error"):
                raise RequestError("upstream unreachable")
            return state["responses"].pop(0)

        async def aclose(self):
            state["client_closed"] = True

    stub.AsyncClient = AsyncClient
    stub.Request = Request
    stub.Response = Response
    stub.RequestError = RequestError
    return stub


def _modal_stub(state):
    stub = types.ModuleType("modal")
    stub.Image = types.SimpleNamespace(
        from_registry=lambda *a, **k: _Chain(),
        debian_slim=lambda *a, **k: _Chain(),
    )
    stub.Volume = types.SimpleNamespace(from_name=lambda *a, **k: _Chain())
    stub.Secret = types.SimpleNamespace(from_name=lambda *a, **k: _Chain())
    stub.App = lambda name: types.SimpleNamespace(function=lambda **kw: (lambda fn: fn))
    stub.concurrent = lambda **kw: (lambda fn: fn)
    stub.web_server = lambda **kw: (lambda fn: _WebFn(fn, state))
    stub.asgi_app = lambda **kw: (lambda fn: fn)
    return stub


class _Headers:
    def __init__(self, headers):
        self._headers = {key.lower(): value for key, value in headers.items()}

    def get(self, key, default=None):
        return self._headers.get(key.lower(), default)

    def items(self):
        return list(self._headers.items())


class _IncomingRequest:
    def __init__(self, headers=None, body=b"", method="POST"):
        self.headers = _Headers(headers or {})
        self._body = body
        self.method = method
        self.query_params = {}

    async def stream(self):
        if self._body:
            yield self._body


def _profile():
    return Profile(
        name="qwen38",
        model_id="gubernac/qwen3.8-27b-unc-fp8",
        gpu="L40S",
        max_model_len=65536,
        idle_timeout_seconds=90,
        endpoint="",
        environment_name="opencodelab-qwen38",
        app_name="opencodelab-qwen38",
        secret_name="ocl-qwen38-vllm",
        gateway_secret_name="ocl-qwen38-gateway",
        tool_call_parser="qwen3_coder",
        reasoning_parser="qwen3",
        trust_remote_code=False,
        speculative_mtp_tokens=3,
        language_model_only=True,
        vllm_version="0.24.0",
    )


@pytest.fixture()
def gateway(monkeypatch):
    state = {
        "model_url": MODEL_MODAL_RUN,
        "web_url_calls": 0,
        "built": [],
        "sent": [],
        "responses": [],
        "client_closed": False,
        "upstream_closed": False,
        "send_error": False,
    }
    monkeypatch.setitem(sys.modules, "modal", _modal_stub(state))
    monkeypatch.setitem(sys.modules, "fastapi", _fastapi_stub())
    monkeypatch.setitem(sys.modules, "fastapi.responses", _fastapi_stub().responses)
    monkeypatch.setitem(sys.modules, "httpx", _httpx_stub(state))
    monkeypatch.setenv("OPENCODELAB_API_KEY", "gateway-key")
    monkeypatch.setenv("OPENCODELAB_PROXY_TOKEN_ID", "wk-proxy")
    monkeypatch.setenv("OPENCODELAB_PROXY_TOKEN_SECRET", "ws-proxy")
    module = types.ModuleType("opencodelab_generated_gateway")
    exec(compile(render_modal_app(_profile()), "<generated-app>", "exec"), module.__dict__)  # noqa: S102 - generated Modal app under test
    web = module.gateway()
    handler = web.routes["/v1/{path:path}"]

    def call(path, request):
        return asyncio.run(handler(path, request))

    return state, handler, call


def _consume(response):
    async def run():
        chunks = []
        async for chunk in response.body:
            chunks.append(chunk)
        return b"".join(chunks)

    return asyncio.run(run())


def test_gateway_rejects_missing_and_wrong_auth_before_gpu_resolution(gateway):
    state, _, call = gateway
    for headers in ({}, {"authorization": "Bearer wrong-key"}):
        result = call("chat/completions", _IncomingRequest(headers=headers, body=b"{}"))
        assert result.status_code == 401
    assert state["web_url_calls"] == 0
    assert state["sent"] == []


def test_gateway_rejects_oversized_body_before_gpu_resolution(gateway):
    state, _, call = gateway
    body = b"\0" * (16 * 1024 * 1024 + 1)
    result = call("chat/completions", _IncomingRequest(headers=AUTH, body=body))
    assert result.status_code == 413
    assert state["web_url_calls"] == 0
    assert state["sent"] == []


def test_gateway_refuses_untrusted_model_target_before_proxy_credentials(gateway):
    state, _, call = gateway
    for bad in ("https://evil.example", f"https://workspace--{ENDPOINT_LABEL}-impostor.modal.run"):
        state["model_url"] = bad
        result = call("chat/completions", _IncomingRequest(headers=AUTH, body=b"{}"))
        assert result.status_code == 502
        assert state["sent"] == []


def test_gateway_rejects_injected_source_segment_before_model_label(gateway):
    state, _, call = gateway
    for bad in (
        f"https://workspace--evil--{ENDPOINT_LABEL}-model.modal.run",
        f"https://workspace--evil--{ENDPOINT_LABEL}-model.us-east.modal.direct",
        f"https://{ENDPOINT_LABEL}-model.modal.run",
        f"https://workspace--{ENDPOINT_LABEL}-model.evil.modal.run",
        f"https://workspace--{ENDPOINT_LABEL}-model.a.us-east.modal.run",
    ):
        state["model_url"] = bad
        result = call("chat/completions", _IncomingRequest(headers=AUTH, body=b"{}"))
        assert result.status_code == 502
        assert state["sent"] == []


def test_gateway_forwards_proxy_credentials_on_trusted_modal_target(gateway):
    state, _, call = gateway
    state["model_url"] = MODEL_MODAL_RUN
    state["responses"] = [_upstream_response()]
    result = call(
        "chat/completions",
        _IncomingRequest(
            headers={**AUTH, "host": "gateway.example", "content-length": "2"},
            body=b"{}",
        ),
    )
    assert result.status_code == 200
    assert len(state["sent"]) == 1
    request = state["sent"][0]
    assert request.url == f"{MODEL_MODAL_RUN}/v1/chat/completions"
    assert request.headers["Modal-Key"] == "wk-proxy"
    assert request.headers["Modal-Secret"] == "ws-proxy"
    assert request.headers["Authorization"] == "Bearer gateway-key"
    assert "host" not in {key.lower() for key in request.headers}
    assert "content-length" not in {key.lower() for key in request.headers}
    assert _consume(result) == b"UPSTREAM"
    assert state["upstream_closed"]
    assert state["client_closed"]


def _upstream_response():
    return sys.modules["httpx"].Response(
        200, {"content-type": "application/json"}, content=b"UPSTREAM"
    )


def test_gateway_trusts_official_modal_direct_region_target(gateway):
    state, _, call = gateway
    state["model_url"] = MODEL_MODAL_DIRECT
    state["responses"] = [_upstream_response()]
    result = call("models", _IncomingRequest(headers=AUTH))
    assert result.status_code == 200
    assert state["sent"][0].url == f"{MODEL_MODAL_DIRECT}/v1/models"


def test_gateway_trusts_regional_modal_run_target(gateway):
    state, _, call = gateway
    state["model_url"] = MODEL_MODAL_RUN_REGIONAL
    state["responses"] = [_upstream_response()]
    result = call("models", _IncomingRequest(headers=AUTH))
    assert result.status_code == 200
    assert state["sent"][0].url == f"{MODEL_MODAL_RUN_REGIONAL}/v1/models"


def test_gateway_blocks_cross_origin_redirect(gateway):
    state, _, call = gateway
    redirect = sys.modules["httpx"].Request(
        "POST", "https://evil.example/v1/chat/completions"
    )
    state["responses"] = [
        sys.modules["httpx"].Response(307, {"location": "https://evil.example"}, next_request=redirect)
    ]
    result = call("chat/completions", _IncomingRequest(headers=AUTH, body=b"{}"))
    assert result.status_code == 502
    assert len(state["sent"]) == 1
    assert state["upstream_closed"]
    assert state["client_closed"]


def test_gateway_follows_same_origin_redirect(gateway):
    state, _, call = gateway
    next_request = sys.modules["httpx"].Request(
        "POST", f"{MODEL_MODAL_RUN}/v1/chat/completions"
    )
    state["responses"] = [
        sys.modules["httpx"].Response(307, {"location": MODEL_MODAL_RUN}, next_request=next_request),
        _upstream_response(),
    ]
    result = call("chat/completions", _IncomingRequest(headers=AUTH, body=b"{}"))
    assert result.status_code == 200
    assert len(state["sent"]) == 2
    assert state["sent"][1].url == f"{MODEL_MODAL_RUN}/v1/chat/completions"
    assert _consume(result) == b"UPSTREAM"


def test_gateway_upstream_failure_returns_502_and_closes_client(gateway):
    state, _, call = gateway
    state["send_error"] = True
    result = call("chat/completions", _IncomingRequest(headers=AUTH, body=b"{}"))
    assert result.status_code == 502
    assert state["client_closed"]
