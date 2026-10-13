from __future__ import annotations

import argparse
import json
import os
import re
import secrets as token_secrets
import sys
from dataclasses import asdict, replace

from opencodelab.config import (
    ConfigError,
    ConfigStore,
    LabConfig,
    Profile,
    bind_profile_endpoint,
    modal_endpoint_label,
    modal_resource_names,
    verify_profile_endpoint,
)
from opencodelab.modal_client import (
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
    modal_environment_exists,
    open_signup,
    print_signup_message,
    require_modal_workspace,
    stop_modal_app,
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
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+_-]{0,31}$")
_PARSER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
GPU_PRICING = {
    "L40S": {"per_second": 0.000542, "per_hour": 1.9512},
    "A100-80GB": {"per_second": 0.000694, "per_hour": 2.4984},
    "H100": {"per_second": 0.001097, "per_hour": 3.9492},
    "H200": {"per_second": 0.001261, "per_hour": 4.5396},
}
PRICING_AS_OF = "2026-10-13"
PRICING_URL = "https://modal.com/pricing"


class _PostDeploySaveError(RuntimeError):
    """The Modal deployment is live but the local profile could not be saved."""


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
    return _validate_parser(value)


def _validate_gpu(value: str) -> str:
    normalized = value.upper()
    if normalized not in GPU_PRICING:
        choices = ", ".join(GPU_PRICING)
        raise ValueError(
            f"GPU must be one of {choices}; other and multi-GPU selectors are not supported"
        )
    return normalized


def _validate_parser(value: str) -> str:
    if not _PARSER.fullmatch(value):
        raise ValueError("Parser names may contain only letters, numbers, dots, underscores, and hyphens")
    return value


def _validate_env_name(value: str) -> str:
    if not _ENV_NAME.fullmatch(value):
        raise ValueError("Hugging Face token environment variable must be a valid environment name")
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
    context = args.context or (131072 if "qwen3.8" in model_id.lower() else 32768)
    if context < 1024:
        raise ValueError("Context must be at least 1024 tokens")
    if context > 1_000_000:
        raise ValueError("Context cannot exceed 1,000,000 tokens")
    if not 2 <= args.idle_timeout <= 1200:
        raise ValueError("Idle timeout must be between 2 and 1200 seconds")
    tool_parser = _parser_value(args.tool_call_parser, inferred.tool_call_parser)
    reasoning_parser = _parser_value(args.reasoning_parser, inferred.reasoning_parser)
    mtp_tokens = inferred.speculative_mtp_tokens if args.mtp_tokens is None else args.mtp_tokens
    if not 0 <= mtp_tokens <= 16:
        raise ValueError("MTP speculative tokens must be between 0 and 16")
    language_model_only = (
        inferred.language_model_only if args.language_model_only is None else args.language_model_only
    )
    vllm_version = _validate_version(args.vllm_version)
    _validate_env_name(args.hf_token_env)
    environment_name, app_name, secret_name, gateway_secret_name = modal_resource_names(
        profile_name
    )
    return Profile(
        name=profile_name,
        model_id=model_id,
        gpu=gpu,
        max_model_len=context,
        idle_timeout_seconds=args.idle_timeout,
        endpoint="",
        environment_name=environment_name,
        app_name=app_name,
        secret_name=secret_name,
        gateway_secret_name=gateway_secret_name,
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
    parser.add_argument("--gpu", default=DEFAULT_GPU, help="Modal GPU selector: L40S (default), A100-80GB, H100, or H200")
    parser.add_argument("--context", type=int, help="Maximum model context; defaults to 128K for Qwen3.8, 32K otherwise")
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
        "modal_environment": profile.environment_name,
        "model": profile.model_id,
        "gpu": profile.gpu,
        "context": profile.max_model_len,
        "idle_timeout_seconds": profile.idle_timeout_seconds,
        "tool_call_parser": profile.tool_call_parser,
        "reasoning_parser": profile.reasoning_parser,
        "trust_remote_code": profile.trust_remote_code,
        "max_gpu_containers": 1,
        "max_cpu_gateway_containers": 1,
        "authentication": "profile key at CPU gateway; server-side Modal proxy token at GPU",
        "gpu_price_per_second_usd": GPU_PRICING[profile.gpu]["per_second"],
        "gpu_price_per_hour_usd": GPU_PRICING[profile.gpu]["per_hour"],
        "pricing_as_of": PRICING_AS_OF,
        "pricing_url": PRICING_URL,
        "inference_requests": 0,
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
    workspace_id = get_modal_workspace_id()

    secret_store = SecretStore(store.root)
    existing_profile = config.profiles.get(profile.name)
    if existing_profile and existing_profile.workspace_id != workspace_id:
        raise SystemExit(
            "The active Modal workspace does not own this OpenCodeLab profile. "
            "Activate the original Modal profile before reconfiguring it."
        )
    profile = replace(profile, workspace_id=workspace_id)
    existing_key = (
        secret_store.get_api_key(profile.name, allow_environment=False)
        if existing_profile
        else None
    )
    if existing_profile and (
        not existing_key or not existing_profile.proxy_token_id or not existing_profile.endpoint
    ):
        raise SystemExit(
            f"Profile {profile.name!r} is an incomplete recovery profile. "
            "Run `opencodelab remove --yes` and then setup again."
        )

    deployment_path = store.deployment_path(profile.name)
    store.write_generated(deployment_path, render_modal_app(profile))

    if existing_profile:
        api_key = existing_key
        profile = replace(profile, proxy_token_id=existing_profile.proxy_token_id)
        secret_location = "existing secure storage"
    else:
        api_key = token_secrets.token_urlsafe(32)
        recovery_profiles = dict(config.profiles)
        recovery_profiles[profile.name] = profile
        recovery = replace(
            config,
            default_profile=config.default_profile or profile.name,
            profiles=recovery_profiles,
        )
        try:
            store.save(recovery)
        except OSError as exc:
            raise RuntimeError(
                "Could not persist OpenCodeLab recovery state; no Modal resources "
                "were created. Resolve the local storage problem and run setup again."
            ) from exc
        try:
            create_modal_environment(profile.environment_name)
        except RuntimeError as exc:
            try:
                store.save(config)
            except OSError:
                pass
            raise RuntimeError(
                f"Could not create dedicated Modal environment {profile.environment_name!r}; "
                "choose another profile name"
            ) from exc

        try:
            def record_proxy_token(token_id: str) -> None:
                nonlocal profile
                profile = replace(profile, proxy_token_id=token_id)
                recovery_profiles[profile.name] = profile
                store.save(replace(recovery, profiles=recovery_profiles))

            proxy_token_id, proxy_token_secret = create_modal_proxy_token(
                on_token_id=record_proxy_token
            )
            profile = replace(profile, proxy_token_id=proxy_token_id)
            secret_location = secret_store.set_api_key(profile.name, api_key)
            allow_modal_proxy_token(proxy_token_id, profile.environment_name)
            create_gateway_secret(
                profile.gateway_secret_name,
                api_key,
                environment_name=profile.environment_name,
                proxy_token_id=proxy_token_id,
                proxy_token_secret=proxy_token_secret,
            )
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(
                f"Setup did not complete. Recovery profile {profile.name!r} was retained; "
                "run `opencodelab remove --yes` before trying again."
            ) from exc

    hf_token = os.getenv(args.hf_token_env) if args.hf_token_env else None
    try:
        create_modal_secret(
            profile.secret_name,
            api_key,
            environment_name=profile.environment_name,
            hf_token=hf_token,
            force=True,
        )
        endpoint = deploy_modal_app(
            deployment_path,
            profile.app_name,
            environment_name=profile.environment_name,
            endpoint_label=modal_endpoint_label(profile.name),
        )
        profile = bind_profile_endpoint(replace(profile, endpoint=endpoint), api_key)
        store.write_generated(store.overlay_path(profile.name), render_overlay(profile, api_key))
        profiles = dict(config.profiles)
        profiles[profile.name] = profile
        updated = LabConfig(
            version=config.version,
            default_profile=config.default_profile or profile.name,
            profiles=profiles,
        )
        try:
            store.save(updated)
        except OSError as exc:
            raise _PostDeploySaveError(
                f"The Modal deployment for profile {profile.name!r} succeeded and is live at "
                f"{profile.endpoint}, but OpenCodeLab could not save the local profile to "
                f"{store.path}. The credential and OpenCode overlay are already in place; "
                f"re-run `opencodelab setup --name {profile.name} --force` to refresh local "
                "state, or run `opencodelab remove --yes` to tear the deployment down."
            ) from exc
    except _PostDeploySaveError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"Setup did not complete. Recovery profile {profile.name!r} was retained; "
            "run `opencodelab remove --yes` before trying again."
        ) from exc

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
    parser.add_argument("gpu", help="Modal GPU selector: L40S, A100-80GB, H100, or H200")
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
    if not current.endpoint or not current.proxy_token_id:
        raise SystemExit(
            f"Profile {current.name!r} is incomplete. Run `opencodelab remove --yes` "
            "and then setup again."
        )
    if current.gpu == gpu:
        print(f"{current.name} already uses {gpu}.")
        return 0
    api_key = SecretStore(store.root).get_api_key(current.name, allow_environment=False)
    if not api_key:
        raise SystemExit(f"Endpoint credential for profile {current.name!r} is missing")
    verify_profile_endpoint(current, api_key)
    updated_profile = replace(current, gpu=gpu)
    print(f"Model: {current.model_id}")
    print(f"GPU: {current.gpu} -> {gpu}")
    price = GPU_PRICING[gpu]
    print(
        f"Modal GPU price snapshot ({PRICING_AS_OF}): "
        f"${price['per_second']:.6f}/sec (~${price['per_hour']:.2f}/hr). {PRICING_URL}"
    )
    _confirm(
        "OpenCodeLab will redeploy this Modal app. The GPU is selected when inference runs and can consume Modal credits.",
        yes=args.yes,
    )
    if not ensure_modal_auth(interactive=True):
        raise SystemExit("Modal authentication did not complete successfully.")
    require_modal_workspace(current.workspace_id)
    deployment_path = store.deployment_path(updated_profile.name)
    store.write_generated(deployment_path, render_modal_app(updated_profile))
    endpoint = deploy_modal_app(
        deployment_path,
        updated_profile.app_name,
        environment_name=updated_profile.environment_name,
        endpoint_label=modal_endpoint_label(updated_profile.name),
    )
    updated_profile = bind_profile_endpoint(replace(updated_profile, endpoint=endpoint), api_key)
    profiles = dict(config.profiles)
    profiles[updated_profile.name] = updated_profile
    store.save(replace(config, profiles=profiles))
    store.write_generated(
        store.overlay_path(updated_profile.name), render_overlay(updated_profile, api_key)
    )
    print(f"Switched {updated_profile.name} to {gpu}. Run `opencodelab` normally.")
    return 0


def cmd_remove(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="opencodelab remove",
        description=(
            "Stop a profile's Modal app, delete its secrets, revoke its proxy token and "
            "dedicated Environment, and remove local configuration."
        ),
    )
    parser.add_argument("profile", nargs="?", help="Profile to remove; defaults to the active profile")
    parser.add_argument(
        "--local-only",
        action="store_true",
        help=(
            "Only remove local state; the Modal Environment, app, proxy token, and secrets "
            "remain in your account"
        ),
    )
    parser.add_argument("--yes", action="store_true", help="Skip the destructive removal confirmation")
    args = parser.parse_args(argv)

    store = ConfigStore()
    config = store.load()
    profile = _resolve_profile(config, args.profile)
    if args.local_only:
        message = (
            f"OpenCodeLab will remove local profile {profile.name!r}. Its Modal Environment, "
            "app, proxy token, and secrets will remain and must be managed in Modal."
        )
    else:
        message = (
            f"OpenCodeLab will permanently remove dedicated Modal Environment "
            f"{profile.environment_name!r}, app {profile.app_name!r}, its proxy token and "
            f"secrets, then remove local profile {profile.name!r}."
        )
    _confirm(message, yes=args.yes)

    if not args.local_only:
        if not ensure_modal_auth(interactive=True):
            raise SystemExit("Modal authentication did not complete successfully.")
        require_modal_workspace(profile.workspace_id)
        cleanup_errors: list[str] = []
        try:
            environment_exists = modal_environment_exists(profile.environment_name)
        except RuntimeError as exc:
            environment_exists = False
            cleanup_errors.append(str(exc))
        cleanup_steps = []
        if environment_exists:
            cleanup_steps.extend([
                lambda: stop_modal_app(
                    profile.app_name,
                    environment_name=profile.environment_name,
                ),
                lambda: delete_modal_secret(
                    profile.gateway_secret_name,
                    environment_name=profile.environment_name,
                ),
                lambda: delete_modal_secret(
                    profile.secret_name,
                    environment_name=profile.environment_name,
                ),
            ])
        if profile.proxy_token_id:
            cleanup_steps.append(lambda: delete_modal_proxy_token(profile.proxy_token_id))
        for cleanup in cleanup_steps:
            try:
                cleanup()
            except (RuntimeError, ValueError) as exc:
                cleanup_errors.append(str(exc))
        if not cleanup_errors and environment_exists:
            try:
                delete_modal_environment(profile.environment_name)
            except RuntimeError as exc:
                cleanup_errors.append(str(exc))
        if cleanup_errors:
            raise RuntimeError(
                "Remote cleanup was incomplete; local recovery state was retained. "
                + "; ".join(cleanup_errors)
            )

    SecretStore(store.root).delete_api_key(profile.name)
    store.deployment_path(profile.name).unlink(missing_ok=True)
    store.overlay_path(profile.name).unlink(missing_ok=True)
    profiles = dict(config.profiles)
    del profiles[profile.name]
    default_profile = config.default_profile
    if default_profile == profile.name:
        default_profile = min(profiles) if profiles else None
    store.save(replace(config, default_profile=default_profile, profiles=profiles))
    print(f"Removed OpenCodeLab profile {profile.name}.")
    return 0


def cmd_login(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="opencodelab login",
        description="Authenticate the Modal client bundled with OpenCodeLab.",
    )
    parser.parse_args(argv)
    if not ensure_modal_auth(interactive=True):
        raise SystemExit("Modal authentication did not complete successfully.")
    print("Modal is authenticated for OpenCodeLab's bundled client.")
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
    parser.add_argument("show", nargs="?", help="optional literal 'show'")
    args = parser.parse_args(argv)
    if args.show is not None and args.show != "show":
        raise SystemExit(
            f"opencodelab config does not accept {args.show!r}; "
            "use 'opencodelab config' or 'opencodelab config show'"
        )
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
        if modal_ok:
            try:
                require_modal_workspace(profile.workspace_id)
                print("PASS Modal workspace matches profile")
            except RuntimeError as exc:
                failures.append(str(exc))
                print(f"FAIL Modal workspace: {exc}")
        if profile.endpoint:
            try:
                api_key = SecretStore(store.root).get_api_key(profile.name)
                verify_profile_endpoint(profile, api_key or "")
                print(f"PASS Endpoint configured: {profile.endpoint}")
            except ValueError as exc:
                failures.append(str(exc))
                print(f"FAIL Endpoint: {exc}")
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
    parser.add_argument("--open", action="store_true", help="Open Modal's ordinary site in your browser")
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
        "  opencodelab login              authenticate the bundled Modal client\n"
        "  opencodelab gpu H100           switch the current profile to another Modal GPU\n"
        "  opencodelab remove [PROFILE]   stop and remove a deployment/profile\n"
        "  opencodelab list               list configured model profiles\n"
        "  opencodelab use PROFILE        choose the default profile\n"
        "  opencodelab config [show]     show non-secret profile configuration\n"
        "  opencodelab doctor             verify local setup without waking a GPU\n"
        "  opencodelab signup [--open]   show Modal's ordinary site\n"
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
    if not profile.endpoint or not profile.proxy_token_id:
        raise SystemExit(
            f"Profile {profile.name!r} is incomplete. Run `opencodelab remove --yes` "
            "and then setup again."
        )
    api_key = SecretStore(store.root).get_api_key(profile.name)
    if not api_key:
        raise SystemExit(
            f"Endpoint credential for profile {profile.name!r} is missing. Re-run `opencodelab setup ... --name {profile.name} --force`."
        )
    return launch_opencode(profile, api_key, store, forwarded)


def _dispatch(argv: list[str]) -> int:
    if not argv:
        return launch([])
    command = argv[0]
    rest = argv[1:]
    if command in {"--help", "-h"}:
        _print_help()
        return 0
    if command == "setup":
        return cmd_setup(rest)
    if command == "login":
        return cmd_login(rest)
    if command == "gpu":
        return cmd_gpu(rest)
    if command == "remove":
        return cmd_remove(rest)
    if command == "use":
        return cmd_use(rest)
    if command == "list":
        if rest:
            raise SystemExit("opencodelab list does not accept arguments")
        return cmd_list()
    if command == "config":
        return cmd_config(rest)
    if command == "doctor":
        if rest:
            raise SystemExit("opencodelab doctor does not accept arguments")
        return cmd_doctor()
    if command == "signup":
        return cmd_signup(rest)
    if command == "help":
        _print_help()
        return 0
    return launch(argv)


def main() -> None:
    try:
        code = _dispatch(sys.argv[1:])
    except (ConfigError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
