from __future__ import annotations

import argparse
import json
import os
import re
import secrets as token_secrets
import sys
from dataclasses import asdict, replace

from opencodelab.config import ConfigStore, LabConfig, Profile
from opencodelab.modal_client import (
    create_modal_secret,
    deploy_modal_app,
    ensure_modal_auth,
    open_signup,
    print_signup_message,
)
from opencodelab.modal_template import render_modal_app
from opencodelab.models import (
    infer_model_settings,
    normalize_huggingface_model,
    safe_slug,
)
from opencodelab.opencode import ensure_opencode, launch_opencode, render_overlay
from opencodelab.secrets import SecretStore

DEFAULT_MODEL = "gubernac/qwen3.8-27b-unc-fp8"
DEFAULT_GPU = "L40S"
COMMANDS = {"setup", "gpu", "use", "list", "config", "doctor", "signup", "help"}
_GPU = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+_-]{0,31}$")


def _confirm(message: str, *, yes: bool) -> None:
    if yes:
        return
    if not sys.stdin.isatty():
        raise SystemExit(f"{message}\nRe-run with --yes after the user has approved the Modal deployment.")
    answer = input(f"{message} [y/N] ").strip().lower()
    if answer not in {"y", "yes"}:
        raise SystemExit("Cancelled.")


def _parser_value(value: str, inferred: str | None) -> str | None:
    if value == "auto":
        return inferred
    if value == "none":
        return None
    return value


def _validate_gpu(value: str) -> str:
    if not _GPU.fullmatch(value):
        raise ValueError("GPU must be a single Modal GPU selector such as L40S or H100; multi-GPU selectors are not supported yet")
    return value


def _validate_version(value: str) -> str:
    if not _VERSION.fullmatch(value):
        raise ValueError("vLLM version must be a simple version string such as 0.24.0")
    return value


def _build_profile(args: argparse.Namespace) -> Profile:
    model_id = normalize_huggingface_model(args.model)
    inferred = infer_model_settings(model_id)
    profile_name = safe_slug(args.name or model_id.split("/", 1)[1])
    gpu = _validate_gpu(args.gpu)
    context = args.context or (65536 if "qwen3.8" in model_id.lower() else 32768)
    if context < 1024:
        raise ValueError("Context must be at least 1024 tokens")
    if not 2 <= args.idle_timeout <= 1200:
        raise ValueError("Idle timeout must be between 2 and 1200 seconds")
    tool_parser = _parser_value(args.tool_call_parser, inferred.tool_call_parser)
    reasoning_parser = _parser_value(args.reasoning_parser, inferred.reasoning_parser)
    mtp_tokens = inferred.speculative_mtp_tokens if args.mtp_tokens is None else args.mtp_tokens
    language_model_only = (
        inferred.language_model_only if args.language_model_only is None else args.language_model_only
    )
    vllm_version = _validate_version(args.vllm_version)
    app_slug = safe_slug(profile_name, max_length=28)
    return Profile(
        name=profile_name,
        model_id=model_id,
        gpu=gpu,
        max_model_len=context,
        idle_timeout_seconds=args.idle_timeout,
        endpoint="",
        app_name=f"opencodelab-{app_slug}",
        secret_name=f"opencodelab-{app_slug}-auth",
        tool_call_parser=tool_parser,
        reasoning_parser=reasoning_parser,
        trust_remote_code=args.trust_remote_code,
        speculative_mtp_tokens=max(0, mtp_tokens),
        language_model_only=language_model_only,
        vllm_version=vllm_version,
    )


def _setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="opencodelab setup",
        description="Deploy a Hugging Face model to your own Modal account and configure it for OpenCode.",
    )
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL, help="Hugging Face owner/model or model URL")
    parser.add_argument("--name", help="Local profile name; defaults to the model name")
    parser.add_argument("--gpu", default=DEFAULT_GPU, help="Modal GPU selector (default: L40S; try H100 for lower latency)")
    parser.add_argument("--context", type=int, help="Maximum model context; defaults to 65K for Qwen3.8, 32K otherwise")
    parser.add_argument("--idle-timeout", type=int, default=90, help="Scale-to-zero window in seconds (default: 90)")
    parser.add_argument("--tool-call-parser", default="auto", help="vLLM parser: auto, none, or an explicit parser name")
    parser.add_argument("--reasoning-parser", default="auto", help="vLLM parser: auto, none, or an explicit parser name")
    parser.add_argument("--mtp-tokens", type=int, default=None, help="MTP speculative tokens; inferred for the Qwen3.8 preset")
    parser.add_argument(
        "--language-model-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pass vLLM --language-model-only; inferred for the Qwen3.8 preset",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Opt in to Hugging Face remote model code")
    parser.add_argument("--vllm-version", default="0.24.0", help="Pinned vLLM version used in the Modal image")
    parser.add_argument("--hf-token-env", default="HF_TOKEN", help="Environment variable containing a token for gated models")
    parser.add_argument("--force", action="store_true", help="Replace an existing local profile and its Modal auth secret")
    parser.add_argument("--dry-run", action="store_true", help="Print the deployment plan without changing local or Modal state")
    parser.add_argument("--yes", action="store_true", help="Skip the deployment confirmation after cost approval")
    return parser


def cmd_setup(argv: list[str]) -> int:
    args = _setup_parser().parse_args(argv)
    try:
        profile = _build_profile(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    store = ConfigStore()
    config = store.load()
    if profile.name in config.profiles and not args.force:
        raise SystemExit(
            f"Profile {profile.name!r} already exists. Use a different --name or pass --force to replace it."
        )

    plan = {
        "profile": profile.name,
        "model": profile.model_id,
        "gpu": profile.gpu,
        "context": profile.max_model_len,
        "idle_timeout_seconds": profile.idle_timeout_seconds,
        "tool_call_parser": profile.tool_call_parser,
        "reasoning_parser": profile.reasoning_parser,
        "trust_remote_code": profile.trust_remote_code,
        "max_gpu_containers": 1,
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        print("Dry run only. No Modal state or local config changed.")
        return 0

    _confirm(
        "OpenCodeLab will deploy this configuration into your Modal account. Modal compute/credits may be consumed by builds or later inference.",
        yes=args.yes,
    )
    if not ensure_modal_auth(interactive=True):
        raise SystemExit("Modal authentication did not complete successfully.")

    secret_store = SecretStore(store.root)
    existing_key = secret_store.get_api_key(profile.name) if profile.name in config.profiles else None
    api_key = existing_key or token_secrets.token_urlsafe(32)
    hf_token = os.getenv(args.hf_token_env) if args.hf_token_env else None
    if existing_key and not hf_token:
        print("Reusing the existing Modal endpoint-auth secret for this profile.")
    else:
        create_modal_secret(profile.secret_name, api_key, hf_token=hf_token, force=True)

    deployment_path = store.deployment_path(profile.name)
    store.write_generated(deployment_path, render_modal_app(profile))
    endpoint = deploy_modal_app(deployment_path, profile.app_name)
    profile = replace(profile, endpoint=endpoint)

    secret_location = secret_store.set_api_key(profile.name, api_key)
    profiles = dict(config.profiles)
    profiles[profile.name] = profile
    updated = LabConfig(
        version=config.version,
        default_profile=config.default_profile or profile.name,
        profiles=profiles,
    )
    store.save(updated)
    store.write_generated(store.overlay_path(profile.name), render_overlay(profile))

    print(f"\nConfigured {profile.name}: {profile.model_id} on {profile.gpu}")
    print(f"Endpoint: {profile.endpoint}")
    print(f"Local API credential stored in: {secret_location}")
    if not profile.tool_call_parser:
        print("Warning: no tool-call parser was inferred. Chat may work, but OpenCode tools depend on model/vLLM parser support.")
    if not hf_token:
        print("No Hugging Face token was supplied. Ungated models work directly; gated models need HF_TOKEN during setup.")
    print("Run `opencodelab` from any project to start OpenCode with this profile.")
    return 0


def _resolve_profile(config: LabConfig, name: str | None = None) -> Profile:
    profile_name = name or os.getenv("OPENCODELAB_PROFILE") or config.default_profile
    if not profile_name:
        raise SystemExit("OpenCodeLab is not configured yet. Run `opencodelab setup` first.")
    profile = config.profiles.get(profile_name)
    if not profile:
        raise SystemExit(f"Unknown OpenCodeLab profile: {profile_name}")
    return profile


def cmd_gpu(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="opencodelab gpu", description="Switch the GPU for an existing profile and redeploy it.")
    parser.add_argument("gpu", help="Modal GPU selector, e.g. L40S or H100")
    parser.add_argument("--profile")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    try:
        gpu = _validate_gpu(args.gpu)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    store = ConfigStore()
    config = store.load()
    current = _resolve_profile(config, args.profile)
    if current.gpu == gpu:
        print(f"{current.name} already uses {gpu}.")
        return 0
    updated_profile = replace(current, gpu=gpu)
    print(f"{current.name}: {current.gpu} -> {gpu}")
    _confirm(
        "OpenCodeLab will redeploy this Modal app. The GPU is selected when inference runs and can consume Modal credits.",
        yes=args.yes,
    )
    if not ensure_modal_auth(interactive=True):
        raise SystemExit("Modal authentication did not complete successfully.")
    deployment_path = store.deployment_path(updated_profile.name)
    store.write_generated(deployment_path, render_modal_app(updated_profile))
    endpoint = deploy_modal_app(deployment_path, updated_profile.app_name)
    updated_profile = replace(updated_profile, endpoint=endpoint)
    profiles = dict(config.profiles)
    profiles[updated_profile.name] = updated_profile
    store.save(replace(config, profiles=profiles))
    store.write_generated(store.overlay_path(updated_profile.name), render_overlay(updated_profile))
    print(f"Switched {updated_profile.name} to {gpu}. Run `opencodelab` normally.")
    return 0


def cmd_use(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="opencodelab use")
    parser.add_argument("profile")
    args = parser.parse_args(argv)
    store = ConfigStore()
    config = store.load()
    _resolve_profile(config, args.profile)
    store.save(replace(config, default_profile=args.profile))
    print(f"Default OpenCodeLab profile: {args.profile}")
    return 0


def cmd_list() -> int:
    config = ConfigStore().load()
    if not config.profiles:
        print("No OpenCodeLab profiles. Run `opencodelab setup`.")
        return 0
    for name, profile in config.profiles.items():
        marker = "*" if name == config.default_profile else " "
        tool_status = profile.tool_call_parser or "not configured"
        print(f"{marker} {name}: {profile.model_id} | {profile.gpu} | {profile.max_model_len} ctx | tools: {tool_status}")
    return 0


def cmd_config(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="opencodelab config")
    parser.add_argument("--profile")
    args = parser.parse_args(argv)
    store = ConfigStore()
    profile = _resolve_profile(store.load(), args.profile)
    print(json.dumps(asdict(profile), indent=2))
    return 0


def cmd_doctor() -> int:
    store = ConfigStore()
    config = store.load()
    failures: list[str] = []
    try:
        opencode = ensure_opencode()
        print(f"PASS OpenCode: {opencode}")
    except RuntimeError as exc:
        failures.append(str(exc))
        print(f"FAIL OpenCode: {exc}")

    modal_ok = ensure_modal_auth(interactive=False)
    print(f"{'PASS' if modal_ok else 'FAIL'} Modal authentication")
    if not modal_ok:
        failures.append("Modal is not authenticated")

    if not config.profiles:
        print("FAIL OpenCodeLab profile: none configured")
        failures.append("No profile configured")
    else:
        profile = _resolve_profile(config)
        print(f"PASS Profile: {profile.name} -> {profile.model_id} on {profile.gpu}")
        if profile.endpoint:
            print(f"PASS Endpoint configured: {profile.endpoint}")
        else:
            print("FAIL Endpoint: profile has not been deployed")
            failures.append("Profile endpoint missing")
        api_key = SecretStore(store.root).get_api_key(profile.name)
        print(f"{'PASS' if api_key else 'FAIL'} Local endpoint credential")
        if not api_key:
            failures.append("Local endpoint credential missing")

    if failures:
        print("\nDoctor found problems:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nOpenCodeLab is ready. This check did not send an inference request or wake a GPU.")
    return 0


def cmd_signup(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="opencodelab signup")
    parser.add_argument("--open", action="store_true", help="Open the configured Modal signup/partner URL in your browser")
    args = parser.parse_args(argv)
    if args.open:
        open_signup()
    else:
        print_signup_message()
    return 0


def _print_help() -> None:
    print(
        "OpenCodeLab\n\n"
        "Run compatible Hugging Face models inside your normal OpenCode on a scale-to-zero Modal GPU.\n\n"
        "Commands:\n"
        "  opencodelab setup [HF_MODEL]   configure and deploy a model (default: Qwen3.8 27B abliterated FP8)\n"
        "  opencodelab gpu H100           switch the current profile to another Modal GPU\n"
        "  opencodelab list               list configured model profiles\n"
        "  opencodelab use PROFILE        choose the default profile\n"
        "  opencodelab config             show non-secret profile configuration\n"
        "  opencodelab doctor             verify local setup without waking a GPU\n"
        "  opencodelab signup [--open]   show the Modal signup/partner URL\n"
        "  opencodelab                    launch OpenCode with the default profile\n"
        "  opencodelab run 'task'         run a one-shot OpenCode task with the default profile\n\n"
        "Use `opencodelab setup --help` for model/GPU/parser options."
    )


def _extract_profile(forwarded: list[str]) -> tuple[str | None, list[str]]:
    if len(forwarded) >= 2 and forwarded[0] == "--profile":
        return forwarded[1], forwarded[2:]
    return None, forwarded


def launch(forwarded: list[str]) -> int:
    profile_name, forwarded = _extract_profile(forwarded)
    store = ConfigStore()
    profile = _resolve_profile(store.load(), profile_name)
    api_key = SecretStore(store.root).get_api_key(profile.name)
    if not api_key:
        raise SystemExit(
            f"Endpoint credential for profile {profile.name!r} is missing. Re-run `opencodelab setup ... --name {profile.name} --force`."
        )
    return launch_opencode(profile, api_key, store, forwarded)


def main() -> None:
    argv = sys.argv[1:]
    if not argv:
        raise SystemExit(launch([]))
    command = argv[0]
    rest = argv[1:]
    if command == "setup":
        raise SystemExit(cmd_setup(rest))
    if command == "gpu":
        raise SystemExit(cmd_gpu(rest))
    if command == "use":
        raise SystemExit(cmd_use(rest))
    if command == "list":
        raise SystemExit(cmd_list())
    if command == "config":
        raise SystemExit(cmd_config(rest))
    if command == "doctor":
        raise SystemExit(cmd_doctor())
    if command == "signup":
        raise SystemExit(cmd_signup(rest))
    if command == "help":
        _print_help()
        return
    raise SystemExit(launch(argv))


if __name__ == "__main__":
    main()
