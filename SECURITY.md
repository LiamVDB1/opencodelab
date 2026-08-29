# Security

OpenCodeLab crosses three meaningful trust boundaries: a local coding agent, a Modal cloud deployment, and arbitrary Hugging Face model repositories.

## Secret handling

OpenCodeLab creates a random API key for the vLLM endpoint and stores the corresponding value in a Modal Secret. Locally, it prefers the operating system keyring. If no usable keyring backend exists, it falls back to a user-config `secrets.json` and attempts to set user-only permissions.

The normal profile config and generated OpenCode overlay do not contain the endpoint API key. The key is injected into the child OpenCode process through `OPENCODELAB_API_KEY`.

Hugging Face tokens for gated models are read from an environment variable during setup and sent to the user's Modal Secret. They are not intentionally written to the repository or profile config.

Never commit `.env`, token files, generated user config, or copied Modal/Hugging Face credentials.

## Arbitrary model repositories

Treat model repositories as untrusted input. OpenCodeLab validates Hugging Face model identifiers and does not enable `trust_remote_code` by default.

`--trust-remote-code` should only be used after the user has reviewed and accepted the model repository's code and trust implications.

## Endpoint exposure

The generated vLLM server requires an API key. Do not remove endpoint authentication from a public deployment unless you deliberately intend to operate and pay for a public service.

OpenCodeLab is designed for a user's own Modal account and caps the generated deployment at one GPU container.

## Cloud-cost safety

The default scale-down window is 90 seconds. Setup requires confirmation unless `--yes` is supplied. `opencodelab doctor` never calls the model endpoint, so it can validate configuration without intentionally waking a GPU.

Agents should treat `--yes` as evidence that a human already approved the relevant Modal cost boundary, not as a way to bypass approval.

## Reporting a vulnerability

Please use GitHub's private security-advisory flow for the repository rather than posting credentials, exploit details, or live endpoint keys in a public issue.
