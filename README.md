# OpenCodeLab

Run compatible Hugging Face models inside your normal [OpenCode](https://opencode.ai/) on a scale-to-zero [Modal](https://modal.com/) GPU.

```text
OpenCode -> authenticated CPU gateway -> protected Modal GPU -> OpenAI-compatible vLLM
```

The tested launch setup is an **abliterated Qwen3.8 27B FP8** model on an **L40S 48 GB**. In one private deployment it ran with 65,536-token context, OpenCode tool calling, roughly 44-49 decode tokens/sec once warm, and a 90-second idle scale-down window. The measured cached cold start for that exact setup was about 106 seconds. These are observations, not a live public-wrapper verification or a performance guarantee.

OpenCodeLab does **not** host a shared inference service. It deploys the model into a dedicated Environment in **your own Modal account**, creates a profile-specific API credential, and launches your existing OpenCode installation with a temporary provider overlay. Your normal OpenCode settings stay intact.

> OpenCodeLab is an independent project. It is not built by, affiliated with, or endorsed by OpenCode, Modal, Hugging Face, Qwen, or the authors of any model you choose to run.

## INSTALL WITH YOUR AGENT

Give this repository to Codex, Claude Code, OpenCode, or another coding agent and paste this prompt:

```text
Install and configure OpenCodeLab from https://github.com/LiamVDB1/opencodelab on this machine. Use the repository's shipped CLI and AGENTS.md; do not reimplement the integration from scratch.

Target model: gubernac/qwen3.8-27b-unc-fp8
GPU: L40S

1. Verify Python 3.11+ and OpenCode. If OpenCode is missing, install it using its current official instructions.
2. Install OpenCodeLab from the repository, preferably with `uv tool install git+https://github.com/LiamVDB1/opencodelab` or with pipx if uv is unavailable.
3. Run `opencodelab doctor`. A non-zero result is expected until OpenCode, Modal authentication, a profile, and its local endpoint credential are all present. If I need a Modal account, run `opencodelab signup`; it prints Modal's ordinary site and is not a referral link. Then run `opencodelab login` and keep it alive while I approve the browser login.
4. Before any command that may consume Modal credits, show me the exact Hugging Face model, selected GPU, and current Modal pricing, then ask me to approve the spend boundary. Do not assume free credits equal approval.
5. After I approve, run `opencodelab setup gubernac/qwen3.8-27b-unc-fp8 --gpu L40S --context 65536 --yes`.
6. Do not send an inference request or wake the GPU just to test installation unless I explicitly approve that spend. Validate with `opencodelab doctor` and `opencodelab list` instead.
7. Never print, hardcode, commit, or paste any Modal token, Hugging Face token, or OpenCodeLab endpoint API key.
8. Finish by telling me that from any project I can now run `opencodelab` to start OpenCode with this model.
```

Want another model? Change only the `Target model` line and the model argument in step 5. OpenCodeLab accepts a Hugging Face `owner/model` id or a normal `https://huggingface.co/owner/model` URL.

## Manual install

Requirements:

- Python 3.11+
- OpenCode installed and available as `opencode`
- a Modal account

With `uv`:

```bash
uv tool install git+https://github.com/LiamVDB1/opencodelab
```

Or with `pipx`:

```bash
pipx install git+https://github.com/LiamVDB1/opencodelab
```

Check the local environment without waking a GPU:

```bash
opencodelab doctor
```

`doctor` returns non-zero until all required pieces are configured. It does not call the model endpoint. If Modal authentication is the missing item, authenticate with:

```bash
opencodelab login
```

Then configure the tested Qwen setup:

```bash
opencodelab setup gubernac/qwen3.8-27b-unc-fp8 \
  --gpu L40S \
  --context 65536
```

OpenCodeLab shows the model, GPU, dated Modal price snapshot, scale-down settings, and deployment plan before asking for confirmation. Setup can consume Modal credits for image builds or later GPU work; confirmation is the cost boundary. After setup:

```bash
cd ~/your-project
opencodelab
```

For a one-shot task:

```bash
opencodelab run "inspect this repo and explain the architecture"
```

## GPU choice

L40S is the default because it is a strong price/performance option for inference and the tested 27B FP8 model fits comfortably in 48 GB VRAM. OpenCodeLab also accepts `A100-80GB` and `H200` for single-GPU models that need more memory; those two have not been measured with the Qwen3.8 preset.

If you care more about latency than credit life, switch the same profile to H100:

```bash
opencodelab gpu H100
```

Switch back the same way:

```bash
opencodelab gpu L40S
```

Changing the configured GPU redeploys the Modal app. The expensive GPU is used when the model actually runs; OpenCodeLab still keeps `max_containers=1` and the configured scale-down window.

As of **2026-10-13**, Modal's public pricing lists:

| GPU | Price | $30 raw GPU time* |
| --- | ---: | ---: |
| L40S | $0.000542/sec (~$1.95/hr) | ~15.4 hours/month |
| A100-80GB | $0.000694/sec (~$2.50/hr) | ~12.0 hours/month |
| H100 | $0.001097/sec (~$3.95/hr) | ~7.6 hours/month |
| H200 | $0.001261/sec (~$4.54/hr) | ~6.6 hours/month |

Modal's Starter plan currently advertises **$30/month in free compute credits**. *These are raw GPU-hour divisions, not a promise of usable session time:* CPU, memory, image builds, cold-start runtime, storage beyond included allowances, and other Modal resources can also consume credits. Check [Modal pricing](https://modal.com/pricing) before relying on a number in this README.

## Other Hugging Face models

OpenCodeLab is not tied to Qwen. The generic path is:

```bash
opencodelab setup owner/model --gpu L40S
```

Or:

```bash
opencodelab setup https://huggingface.co/owner/model --gpu H100
```

What "compatible" means:

1. vLLM must be able to serve the model as an OpenAI-compatible endpoint.
2. The model must fit on the GPU/configuration you choose.
3. For OpenCode tool use, the model and your vLLM version need a compatible tool-call parser.
4. Reasoning models may need a reasoning parser.

OpenCodeLab currently auto-configures the parser path we have verified for Qwen3/Qwen3.8 and the Hermes tool parser for Qwen2.5/QwQ-style models. This inference is a **best-effort model-name convention**, not a compatibility check: it only inspects the model id. For an unknown architecture it deliberately **does not invent a parser**. Chat may work while agent tool calls do not. The safest choice for any model you have not verified yourself is an explicit `--tool-call-parser none --reasoning-parser none` (or a parser you have confirmed against your chosen vLLM version).

Override parsers explicitly when you know the correct vLLM configuration:

```bash
opencodelab setup owner/model \
  --tool-call-parser YOUR_PARSER \
  --reasoning-parser YOUR_REASONING_PARSER
```

Disable inference completely for a parser with `none`:

```bash
opencodelab setup owner/model --tool-call-parser none --reasoning-parser none
```

### Gated Hugging Face models

For a gated model, first make sure your Hugging Face account has actually been granted access to the model, then put your token in an environment variable before setup:

```bash
export HF_TOKEN=hf_...
opencodelab setup owner/gated-model
```

OpenCodeLab passes it into a Modal Secret. It does not write the Hugging Face token into the repository or normal profile config.

### Models that require remote code

Arbitrary Hugging Face model repositories are untrusted input. OpenCodeLab does **not** enable `trust_remote_code` by default.

If you have reviewed the repository and the model requires it:

```bash
opencodelab setup owner/model --trust-remote-code
```

That flag is an explicit security decision, not a compatibility checkbox.

## Multiple model profiles

Every setup gets a local profile. Add another model with a distinct name:

```bash
opencodelab setup owner/another-model --name another --gpu H100
```

List them:

```bash
opencodelab list
```

Choose the default:

```bash
opencodelab use another
```

Or launch a profile once without changing the default:

```bash
opencodelab --profile another
```

Prototype v0.1 profiles predate the dedicated gateway and Environment lifecycle and cannot be upgraded in place safely. Remove their old resources manually in the Modal dashboard, move aside the local config file named in OpenCodeLab's schema error, then run setup again. OpenCodeLab rejects the old config instead of silently launching the less-protected deployment.

## What OpenCodeLab changes

OpenCodeLab creates a random client API key for each profile. A small CPU gateway validates that key before it forwards a request to the GPU-backed vLLM server, which validates the same key again. The GPU endpoint also requires a dedicated Modal proxy token. That workspace credential is stored only in the gateway's Modal Secret; it is never passed to OpenCode or stored as the local client key. Unauthorized internet traffic can start at most the one-container CPU gateway, not the GPU function.

The local client key is stored in the system keyring when available. If no usable keyring exists, OpenCodeLab falls back to a local `secrets.json` under its user-config directory and restricts the file to the current user where the platform permits it. Non-secret profile metadata records the originating Modal workspace ID plus the dedicated Environment, app, secrets, and proxy-token ID. Remote lifecycle commands refuse to run after the active Modal profile switches to another workspace, even if that workspace happens to contain resources with the same names.

It generates two local runtime artifacts outside the repository:

- a Modal deployment file for the selected model/GPU;
- an OpenCode provider overlay.

When OpenCode starts, OpenCodeLab sets `OPENCODE_CONFIG` to that overlay and `OPENCODELAB_API_KEY` only for the child OpenCode process. The overlay references the key through an environment variable rather than hardcoding it.

OpenCodeLab does **not** replace your global OpenCode config.

## Modal signup

Run:

```bash
opencodelab signup
```

This points to Modal's ordinary site. OpenCodeLab has no Modal partnership or referral link.

## Tested configuration

The configuration used to build OpenCodeLab itself:

```text
Model: gubernac/qwen3.8-27b-unc-fp8
Upstream family: OrcaRouter Qwen3.8 27B abliterated/uncensored FP8
GPU: Nvidia L40S 48 GB
Context: 65,536
Weight format: block-FP8 E4M3
KV cache in tested private setup: BF16
Tool parser: qwen3_coder
Reasoning parser: qwen3
MTP speculative tokens: 3
Idle scale-down: 90 seconds
Maximum GPU containers: 1
Measured steady decode: ~44-49 tok/s
Measured cached cold start: ~106 seconds
```

Those speed/startup numbers are observations from one setup, not guarantees for Modal capacity, future vLLM versions, other models, or H100.

## Commands

```text
opencodelab setup [HF_MODEL]    deploy/configure a model
opencodelab login               authenticate the bundled Modal client
opencodelab                     launch normal OpenCode with the default profile
opencodelab run "task"          one-shot OpenCode task
opencodelab gpu H100            redeploy the current profile on H100
opencodelab remove [PROFILE]    remove the dedicated Modal resources and local profile
opencodelab list                list configured model profiles
opencodelab use PROFILE         choose the default profile
opencodelab config [show]       show non-secret profile config
opencodelab doctor              validate setup without sending inference
opencodelab signup [--open]     show/open the Modal signup URL
```

`opencodelab setup --help` exposes context, idle timeout, tool/reasoning parser, MTP, vLLM version, gated-model token, remote-code, and dry-run controls.

`opencodelab remove` is destructive and confirms before stopping the Modal app. It also deletes both Modal Secrets, revokes the dedicated proxy token, and deletes the dedicated Environment. Cleanup is retry-safe and preserves local recovery state if any remote step remains incomplete. `--local-only` removes only local state and deliberately leaves the Modal resources for manual management.

## Security and cost boundaries

- A profile-specific key is checked in a CPU gateway before any GPU invocation, then checked again by vLLM; the GPU endpoint independently requires a Modal proxy token that is never injected into OpenCode.
- Secrets are never intentionally stored in the Git repository.
- The Modal deployment is capped at one GPU container.
- `min_containers=0` and `buffer_containers=0` preserve scale-to-zero; the default idle scale-down window is 90 seconds.
- `trust_remote_code` is opt-in.
- `opencodelab doctor` never calls the model endpoint.
- Setup requires explicit confirmation unless `--yes` is passed.
- `--yes` means the human or supervising agent has already approved the Modal cost boundary; it is not permission to spend without approval.

See [SECURITY.md](https://github.com/LiamVDB1/opencodelab/blob/main/SECURITY.md) for the trust-boundary notes.

## What this project does not promise

- It does not make every Hugging Face model compatible with vLLM.
- It does not guarantee that every compatible model can use OpenCode tools.
- It does not guarantee a specific token speed or cold-start time.
- v0.1 intentionally supports one GPU per deployment. Multi-GPU/tensor-parallel serving is not exposed yet, to avoid silently paying for extra GPUs without matching vLLM configuration.
- It does not remove model licenses, usage terms, or legal obligations.
- It does not redistribute model weights.
- It is not a hosted inference business or a way to share Liam's Modal account.

You are responsible for the model you select, its license, the workloads you run, and the cloud costs you authorize.

## Development

See [docs/DEVELOPMENT.md](https://github.com/LiamVDB1/opencodelab/blob/main/docs/DEVELOPMENT.md) for the full local workflow and [docs/RELEASING.md](https://github.com/LiamVDB1/opencodelab/blob/main/docs/RELEASING.md) for the release checklist. The main test command does not touch a real Modal deployment:

```bash
PYTHONPATH=src uv run --no-project \
  --with pytest --with pytest-cov --with 'modal>=1.5.4,<2' --with keyring --with platformdirs \
  python -m pytest -q
```

The tests exercise command behavior and failure paths, model and endpoint validation, parser inference, config/secrets separation, generated Modal code, teardown, and OpenCode overlay/argument construction.

## License

OpenCodeLab's own code is MIT licensed. Models, OpenCode, Modal, vLLM, Hugging Face, CUDA images, and other dependencies remain under their own licenses and terms.
