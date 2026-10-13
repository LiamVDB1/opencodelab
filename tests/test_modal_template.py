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
        "environment_name": "opencodelab-qwen38",
        "app_name": "opencodelab-qwen38",
        "secret_name": "ocl-qwen38-vllm",
        "gateway_secret_name": "ocl-qwen38-gateway",
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
    compile(source, "generated_modal_app.py", "exec")
    assert 'MODEL_ID = "gubernac/qwen3.8-27b-unc-fp8"' in source
    assert 'GPU = "L40S"' in source
    assert "max_containers=1" in source
    assert "min_containers=0" in source
    assert "buffer_containers=0" in source
    assert "requires_proxy_auth=True" in source
    assert "@modal.asgi_app(label=ENDPOINT_LABEL)" in source
    assert "hmac.compare_digest" in source
    assert 'headers["Modal-Key"]' in source
    assert 'headers["Modal-Secret"]' in source
    assert source.index("hmac.compare_digest") < source.index("serve_model.get_web_url")
    assert "trusted_model_host" in source
    assert source.index("trusted_model_host(target_host)") < source.index('headers["Modal-Key"]')
    assert "follow_redirects=False" in source
    assert "next_origin != target_origin" in source
    assert source.index("next_origin != target_origin") < source.index("outbound = next_request")
    assert "async for chunk in request.stream()" in source
    assert source.index("body_size > 16 * 1024 * 1024") < source.index("serve_model.get_web_url")
    assert "excluded | connection_headers" in source
    assert "excluded | response_connection_headers" in source
    assert "async def stream_upstream()" in source
    assert "finally:" in source
    assert source.count("gpu=GPU") == 1
    assert "secrets=[gateway_auth]" in source
    assert "IDLE_TIMEOUT_SECONDS = 90" in source
    assert "scaledown_window=IDLE_TIMEOUT_SECONDS" in source
    assert '"--max-model-len", str(MAX_MODEL_LEN)' in source
    assert '"--max-num-batched-tokens",\n        "8192",' in source
    assert '"--tool-call-parser"' in source
    assert '"qwen3_coder"' in source
    assert '"--reasoning-parser"' in source
    assert '"qwen3"' in source
    assert '"--trust-remote-code"' not in source
    assert '"--api-key"' in source
    assert 'os.environ["OPENCODELAB_API_KEY"]' in source


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


# --- Executable generated-gateway behavior tests (stubbed Modal/FastAPI/httpx) ---

import asyncio
import sys
import types

from opencodelab.config import modal_endpoint_label


class _Chain:
    def __getattr__(self, name):
        return lambda *a, **k: self


class _StubResponse:
    def __init__(self, status_code=200, headers=None, content=b"", next_request=None):
        self.status_code = status_code
        self.headers = headers or {}
        self.next_request = next_request
        self._content = content
        self.closed = False

    async def aiter_raw(self):
        yield self._content

    async def aclose(self):
        self.closed = True


class _StubRequest:
    def __init__(self, method, url, headers=None):
        self.method = method
        self.url = url
        self.headers = headers or {}


class _RequestError(Exception):
    pass


class _JSONResponse:
    def __init__(self, content=None, status_code=200):
        self.content = content
        self.status_code = status_code


class _StreamingResponse:
    def __init__(self, content, status_code=200, headers=None):
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}


class _StubFastAPI:
    def __init__(self, **kwargs):
        self.routes = {}

    def api_route(self, path, methods=None):
        def deco(fn):
            self.routes[path] = fn
            return fn

        return deco


class _StubHeaders:
    def __init__(self, headers):
        self._headers = {key.lower(): value for key, value in headers.items()}

    def get(self, key, default=None):
        return self._headers.get(key.lower(), default)

    def items(self):
        return self._headers.items()


class _StubIncoming:
    def __init__(self, method="POST", headers=None, body=b"", query=None):
        self.method = method
        self.headers = _StubHeaders(headers or {})
        self._body = body
        self.query_params = query or {}
        self.streamed = False

    async def stream(self):
        self.streamed = True
        if self._body:
            yield self._body


class _StubAsgiApp:
    def __init__(self, fn, label):
        self.raw = fn
        self.label = label


class _StubWebServer:
    def __init__(self, fn, state):
        self.raw = fn
        self._state = state

    def get_web_url(self):
        self._state["web_url_calls"] += 1
        return self._state["model_url"]


def _install_stub_modules(monkeypatch, state):
    modal_stub = types.ModuleType("modal")
    modal_stub.App = lambda name: types.SimpleNamespace(
        function=lambda **kw: (lambda fn: fn)
    )
    modal_stub.Image = types.SimpleNamespace(
        from_registry=lambda *a, **k: _Chain(),
        debian_slim=lambda **k: _Chain(),
    )
    modal_stub.Volume = types.SimpleNamespace(from_name=lambda *a, **k: _Chain())
    modal_stub.Secret = types.SimpleNamespace(from_name=lambda *a, **k: _Chain())
    modal_stub.concurrent = lambda **kw: (lambda fn: fn)
    modal_stub.asgi_app = lambda **kw: (lambda fn: _StubAsgiApp(fn, kw.get("label")))
    modal_stub.web_server = lambda **kw: (lambda fn: _StubWebServer(fn, state))
    monkeypatch.setitem(sys.modules, "modal", modal_stub)

    fastapi_stub = types.ModuleType("fastapi")
    fastapi_stub.FastAPI = _StubFastAPI
    fastapi_stub.Request = _StubIncoming
    responses_stub = types.ModuleType("fastapi.responses")
    responses_stub.JSONResponse = _JSONResponse
    responses_stub.StreamingResponse = _StreamingResponse
    fastapi_stub.responses = responses_stub
    monkeypatch.setitem(sys.modules, "fastapi", fastapi_stub)
    monkeypatch.setitem(sys.modules, "fastapi.responses", responses_stub)

    httpx_stub = types.ModuleType("httpx")
    httpx_stub.RequestError = _RequestError

    class _StubAsyncClient:
        def __init__(self, **kwargs):
            state["client_kwargs"] = kwargs
            state["client_closed"] = False

        def build_request(self, method, url, params=None, headers=None, content=b""):
            request = _StubRequest(method, url, headers=headers)
            state["requests"].append(request)
            return request

        async def send(self, request, stream=False):
            state["sends"].append(request)
            if state.get("raise_request_error"):
                raise _RequestError("boom")
            return state["responses"].pop(0)

        async def aclose(self):
            state["client_closed"] = True

    httpx_stub.AsyncClient = _StubAsyncClient
    monkeypatch.setitem(sys.modules, "httpx", httpx_stub)


def _load_gateway(monkeypatch, state, profile=None):
    profile = profile or _profile()
    _install_stub_modules(monkeypatch, state)
    module = types.ModuleType("generated_gateway_app")
    exec(compile(render_modal_app(profile), "<generated>", "exec"), module.__dict__)  # noqa: S102 - generated Modal app under test
    return module


def _gateway_handler(module):
    web = module.gateway.raw()
    return web.routes["/v1/{path:path}"]


def _authed_request(body=b'{"messages": []}'):
    return _StubIncoming(
        headers={"authorization": "Bearer client-key", "host": "gateway.example"},
        body=body,
    )


def test_gateway_rejects_missing_or_wrong_auth_before_gpu_resolution(monkeypatch):
    state = {"model_url": "https://ws--model.modal.run", "web_url_calls": 0,
             "requests": [], "sends": [], "responses": []}
    monkeypatch.setenv("OPENCODELAB_API_KEY", "client-key")
    monkeypatch.setenv("OPENCODELAB_PROXY_TOKEN_ID", "wk-proxy")
    monkeypatch.setenv("OPENCODELAB_PROXY_TOKEN_SECRET", "ws-proxy")
    module = _load_gateway(monkeypatch, state)
    handler = _gateway_handler(module)

    no_auth = asyncio.run(handler("chat/completions", _StubIncoming()))
    wrong_auth = asyncio.run(
        handler("chat/completions", _StubIncoming(headers={"authorization": "Bearer nope"}))
    )

    assert no_auth.status_code == 401
    assert wrong_auth.status_code == 401
    assert state["web_url_calls"] == 0
    assert state["sends"] == []


def test_gateway_rejects_oversized_body_before_gpu_resolution(monkeypatch):
    state = {"model_url": "https://ws--model.modal.run", "web_url_calls": 0,
             "requests": [], "sends": [], "responses": []}
    monkeypatch.setenv("OPENCODELAB_API_KEY", "client-key")
    monkeypatch.setenv("OPENCODELAB_PROXY_TOKEN_ID", "wk-proxy")
    monkeypatch.setenv("OPENCODELAB_PROXY_TOKEN_SECRET", "ws-proxy")
    module = _load_gateway(monkeypatch, state)
    handler = _gateway_handler(module)

    result = asyncio.run(handler("chat/completions", _authed_request(b"x" * (16 * 1024 * 1024 + 1))))

    assert result.status_code == 413
    assert state["web_url_calls"] == 0
    assert state["sends"] == []


def test_gateway_never_sends_proxy_credentials_to_untrusted_model_url(monkeypatch):
    state = {"model_url": "https://evil.example/steal", "web_url_calls": 0,
             "requests": [], "sends": [], "responses": []}
    monkeypatch.setenv("OPENCODELAB_API_KEY", "client-key")
    monkeypatch.setenv("OPENCODELAB_PROXY_TOKEN_ID", "wk-proxy")
    monkeypatch.setenv("OPENCODELAB_PROXY_TOKEN_SECRET", "ws-proxy")
    module = _load_gateway(monkeypatch, state)
    handler = _gateway_handler(module)

    result = asyncio.run(handler("chat/completions", _authed_request()))

    assert result.status_code == 502
    assert state["sends"] == []


def test_gateway_forwards_proxy_headers_only_to_trusted_model_url(monkeypatch):
    label = modal_endpoint_label("qwen38")
    state = {
        "model_url": f"https://workspace--{label}-model.modal.run",
        "web_url_calls": 0,
        "requests": [],
        "sends": [],
        "responses": [_StubResponse(200, {"content-type": "application/json"}, b'{"ok": true}')],
    }
    monkeypatch.setenv("OPENCODELAB_API_KEY", "client-key")
    monkeypatch.setenv("OPENCODELAB_PROXY_TOKEN_ID", "wk-proxy")
    monkeypatch.setenv("OPENCODELAB_PROXY_TOKEN_SECRET", "ws-proxy")
    module = _load_gateway(monkeypatch, state)
    handler = _gateway_handler(module)

    result = asyncio.run(handler("chat/completions", _authed_request()))

    assert result.status_code == 200
    sent = state["sends"][0]
    assert sent.headers["Modal-Key"] == "wk-proxy"
    assert sent.headers["Modal-Secret"] == "ws-proxy"
    assert sent.headers["Authorization"] == "Bearer client-key"
    assert "host" not in {key.lower() for key in sent.headers}
    assert state["web_url_calls"] == 1


def test_gateway_closes_upstream_on_cross_origin_redirect(monkeypatch):
    label = modal_endpoint_label("qwen38")
    state = {
        "model_url": f"https://workspace--{label}-model.modal.run",
        "web_url_calls": 0,
        "requests": [],
        "sends": [],
        "responses": [
            _StubResponse(
                302,
                {"location": "https://evil.example/v1/chat"},
                next_request=_StubRequest("GET", "https://evil.example/v1/chat"),
            )
        ],
    }
    monkeypatch.setenv("OPENCODELAB_API_KEY", "client-key")
    monkeypatch.setenv("OPENCODELAB_PROXY_TOKEN_ID", "wk-proxy")
    monkeypatch.setenv("OPENCODELAB_PROXY_TOKEN_SECRET", "ws-proxy")
    module = _load_gateway(monkeypatch, state)
    handler = _gateway_handler(module)

    result = asyncio.run(handler("chat/completions", _authed_request()))

    assert result.status_code == 502
    assert len(state["sends"]) == 1
    assert state["client_closed"] is True


def test_gateway_follows_same_origin_redirect_and_streams_response(monkeypatch):
    label = modal_endpoint_label("qwen38")
    model_host = f"workspace--{label}-model.modal.run"
    state = {
        "model_url": f"https://{model_host}",
        "web_url_calls": 0,
        "requests": [],
        "sends": [],
        "responses": [
            _StubResponse(
                307,
                {"location": f"https://{model_host}/v1/chat"},
                next_request=_StubRequest("POST", f"https://{model_host}/v1/chat"),
            ),
            _StubResponse(200, {"content-type": "application/json"}, b"chunk-achunk-b"),
        ],
    }
    monkeypatch.setenv("OPENCODELAB_API_KEY", "client-key")
    monkeypatch.setenv("OPENCODELAB_PROXY_TOKEN_ID", "wk-proxy")
    monkeypatch.setenv("OPENCODELAB_PROXY_TOKEN_SECRET", "ws-proxy")
    module = _load_gateway(monkeypatch, state)
    handler = _gateway_handler(module)

    result = asyncio.run(handler("chat/completions", _authed_request()))

    assert result.status_code == 200
    assert len(state["sends"]) == 2
    body = asyncio.run(_drain(result.content))
    assert body == b"chunk-achunk-b"
    assert state["responses"] == []
    assert state["client_closed"] is True


async def _drain(stream):
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    return b"".join(chunks)


def test_gateway_returns_502_on_upstream_transport_error(monkeypatch):
    label = modal_endpoint_label("qwen38")
    state = {
        "model_url": f"https://workspace--{label}-model.modal.run",
        "web_url_calls": 0,
        "requests": [],
        "sends": [],
        "responses": [],
        "raise_request_error": True,
    }
    monkeypatch.setenv("OPENCODELAB_API_KEY", "client-key")
    monkeypatch.setenv("OPENCODELAB_PROXY_TOKEN_ID", "wk-proxy")
    monkeypatch.setenv("OPENCODELAB_PROXY_TOKEN_SECRET", "ws-proxy")
    module = _load_gateway(monkeypatch, state)
    handler = _gateway_handler(module)

    result = asyncio.run(handler("chat/completions", _authed_request()))

    assert result.status_code == 502
    assert state["client_closed"] is True
