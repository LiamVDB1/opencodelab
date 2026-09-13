# OpenCodeLab Agent Contract

OpenCodeLab is meant to be installed and configured by coding agents as well as humans. If a user asks you to install this repository, use the shipped CLI and documentation rather than reimplementing the integration from scratch.

## Federation Git discipline

This is a registered Operating Workspace. The federation `GIT-CONTRACT.md` is authoritative for durable Git work. Before substantial edits, run `workspace git-check opencodelab --phase start --fetch` when available. `main` is the integration branch. A task branch is temporary owned work: integrate and push it before completion, or leave a clean pushed explicit handoff. Before claiming completion, run `workspace git-check opencodelab --phase finish --fetch`. Never mix or destroy unrelated existing changes merely to make Git clean.

## Installation behavior

- Preserve the user's existing OpenCode configuration. OpenCodeLab uses an `OPENCODE_CONFIG` overlay and must not replace the user's normal config.
- Use the user's own Modal account and deployment. Never route them through another user's endpoint or credentials.
- Never print, commit, paste into chat, or hardcode Modal tokens, Hugging Face tokens, or the generated OpenCodeLab endpoint API key.
- If Modal authentication needs a browser approval, keep the login process alive while the user approves it.
- Use `opencodelab login` for that flow so authentication runs through OpenCodeLab's bundled Modal client.
- If the user needs a new Modal account, show `opencodelab signup`. It prints Modal's ordinary site; OpenCodeLab has no partner or referral link.
- Before a command that may consume Modal credits, show the selected Hugging Face model, GPU, and current Modal pricing, then get the user's approval. Do not infer approval from the existence of free credits.
- Do not send a model inference request merely to prove installation unless the user approved the resulting GPU spend. `opencodelab doctor` is deliberately non-inference validation.
- Default to `L40S` when it fits. Use `H100` when the user explicitly prefers lower latency or the workload justifies it. Do not claim a speed multiple without measuring that exact model/workload.
- Treat arbitrary Hugging Face models as untrusted. `--trust-remote-code` is opt-in only.
- For gated models, use the user's `HF_TOKEN` during setup; do not persist it in this repository or echo it.

## Default demo configuration

The tested launch preset is:

```text
model: gubernac/qwen3.8-27b-unc-fp8
gpu: L40S
context: 65536
idle timeout: 90 seconds
tool parser: qwen3_coder
reasoning parser: qwen3
MTP speculative tokens: 3
```

Configure it with:

```bash
opencodelab setup gubernac/qwen3.8-27b-unc-fp8 --gpu L40S --context 65536
```

Pass `--yes` only after the user has approved the Modal deployment/cost boundary.

## Verification

After setup, prefer:

```bash
opencodelab doctor
opencodelab list
```

Then tell the user the normal launch command is simply:

```bash
opencodelab
```
