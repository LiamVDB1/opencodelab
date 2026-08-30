# Security

OpenCodeLab crosses three meaningful trust boundaries: a local coding agent, a Modal cloud deployment, and arbitrary Hugging Face model repositories.

## Secret handling

OpenCodeLab creates two different credentials for each profile: a random client API key and a dedicated Modal proxy token. The client key is accepted only by that profile's CPU gateway and vLLM server. The workspace-wide proxy token is stored only in the gateway's Modal Secret and is used for the gateway-to-GPU hop; it is never injected into OpenCode or used as the local client credential.

Locally, OpenCodeLab prefers the operating system keyring for the profile-specific client key. If no usable keyring backend exists, it falls back to a user-config `secrets.json` and restricts the file to user-only permissions where the platform permits it.

The normal profile config contains only the non-secret proxy-token ID, and the generated OpenCode overlay contains no credential. The client key is injected into the child OpenCode process through `OPENCODELAB_API_KEY`. Before that happens, the configured endpoint must be an HTTPS Modal origin bound to the profile's deterministic short gateway label, including Modal's documented hostname-truncation rule. Local config loading also rejects Environment, app, or secret names that are not derived from the profile name. These checks reduce credential-exfiltration and unrelated-resource risks from a tampered local profile.

Hugging Face tokens for gated models are read from an environment variable during setup and sent to the user's Modal Secret. They are not intentionally written to the repository or profile config.

Never commit `.env`, token files, generated user config, or copied Modal/Hugging Face credentials.

## Arbitrary model repositories

Treat model repositories as untrusted input. OpenCodeLab validates Hugging Face model identifiers and does not enable `trust_remote_code` by default.

`--trust-remote-code` should only be used after the user has reviewed and accepted the model repository's code and trust implications.

## Endpoint exposure

The tested vLLM 0.24.0 Qwen3 parser configuration has a known whitespace-risk report in [upstream issue #48753](https://github.com/vllm-project/vllm/issues/48753). This is a compatibility risk for affected outputs, not a claim that every Qwen3 deployment breaks.

The public CPU gateway performs a constant-time bearer-key check and applies a 16 MiB request-body limit before forwarding. It is capped at one CPU container. Invalid requests never invoke the GPU function. The GPU web server requires the dedicated Modal proxy token at Modal's boundary and independently requires the profile-specific vLLM key. Do not remove any of these layers unless you deliberately intend to operate and pay for a public service.

In a Modal workspace without RBAC, proxy tokens are workspace-wide credentials. OpenCodeLab therefore keeps the generated token server-side in the gateway instead of exposing it to the coding-agent process. In an RBAC workspace, setup explicitly grants that token to the profile's dedicated Environment. Setup also records the non-secret workspace ID. Every secret, deploy, GPU-change, and teardown operation passes the persisted Environment explicitly, and later remote mutations refuse to run unless the active Modal credentials identify the original workspace.

The coding-agent child process receives a scrubbed environment: Modal tokens, Hugging Face tokens, OpenCodeLab proxy-token material, and any parent client key are removed before only the selected profile's client key is injected. The gateway also validates the model web URL against the exact official Modal hostname shape and this app's endpoint label before it attaches proxy credentials, and never follows redirects to another origin.

OpenCodeLab is designed for a user's own Modal account and caps the generated deployment at one GPU container. The generated function explicitly sets zero minimum and buffer containers, so it can scale to zero after the configured idle window.

## Cloud-cost safety

The default scale-down window is 90 seconds. Setup requires confirmation unless `--yes` is supplied. `opencodelab doctor` never calls the model endpoint, so it can validate configuration without intentionally waking a GPU.

Agents should treat `--yes` as evidence that a human already approved the relevant Modal cost boundary, not as a way to bypass approval.

Setup records a recovery profile before creating credentials or deploying the app. If a later remote or local-persistence step fails, that state remains available for teardown. `opencodelab remove` attempts every app, secret, and token cleanup step, deletes the dedicated Environment only when those steps succeed, and removes local recovery state last. Missing apps/resources are treated as successful cleanup so interrupted removal can be retried. `--local-only` intentionally skips remote cleanup and should be used only when the user will manage the remaining Modal resources separately.

`--trust-remote-code` lets code from the selected model repository run inside the GPU container. Such code can read the profile's vLLM key and any `HF_TOKEN` supplied for that deployment. The Modal workspace proxy credential remains isolated in the separate CPU gateway container, but remote model code should still be enabled only after review.

## Reporting a vulnerability

Please use GitHub's private security-advisory flow for the repository rather than posting credentials, exploit details, or live endpoint keys in a public issue.
