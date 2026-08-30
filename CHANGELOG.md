# Changelog

All notable changes to OpenCodeLab are documented here.

## Unreleased

- Accept an optional literal `show` for `opencodelab config`, matching documented usage in the help text and README.

## 0.2.0 - 2026-08-30

- Invoke Modal through the absolute console script of the active environment with a scrubbed subprocess environment, so a hostile `modal.py`/`modal` package in the user's project can never be imported.
- Cryptographically bind each profile's deployed endpoint origin to its client credential with an HMAC stored in local config, fail closed on missing/legacy/tampered bindings, and verify before launch, config generation, GPU redeploys, and doctor.
- Reconcile and revoke freshly created Modal proxy tokens when output parsing or state persistence fails, persist the token ID as soon as it is known, and raise an explicit manual-cleanup error when automatic revocation cannot be proven.
- Enforce `normalize_huggingface_model` and exact normalized equality for persisted model ids and endpoints when loading local config.
- Require the exact official Modal hostname shape (no spoofable intermediate labels) for both `modal.run` and regional `modal.direct` endpoints, locally and inside the generated gateway.
- Scrub Modal, Hugging Face, and OpenCodeLab secret material from the OpenCode child environment and inject only the selected profile's client key.
- Persist recovery state before creating any Modal resource; a local persistence failure now performs no remote mutation.
- The generated gateway validates `serve_model.get_web_url()` against the strict Modal shape and this app's endpoint label before attaching proxy credentials, requiring exactly one separator between the Modal source and the expected model label.
- Endpoint-label validation accepts legitimate Modal workspace/source names containing double hyphens (including Modal's truncated/hash host form) while strict official host-shape validation and cryptographic endpoint binding stay unchanged.
- Report a failed post-deployment local save accurately: the message states the deployment is live and gives safe re-run/teardown guidance.
- Harden configuration, secret-file, endpoint-origin, parser, environment-name, and GPU validation.
- Add an authenticated CPU gateway so untrusted traffic cannot start a GPU container; keep the workspace-wide Modal proxy token server-side and make vLLM enforce a separate profile-specific key.
- Put each profile in a persisted dedicated Modal Environment, bind the original workspace ID, resource names, and endpoint labels, and verify that workspace before every later remote mutation.
- Add durable incomplete-setup recovery state and best-effort, retry-safe environment/app/token/secret teardown.
- Move local profiles to config schema 2; prototype-era profiles must be removed and set up again because they lack the dedicated Environment and gateway credentials.
- Add `opencodelab login` so isolated tool installs authenticate the bundled Modal client.
- Support current `modal.direct` and legacy `modal.run` deployment URLs.
- Make one-container and scale-to-zero limits explicit in generated Modal apps.
- Add confirmed teardown through `opencodelab remove`, preserving local recovery state on remote failure.
- Add dated GPU price disclosure to deployment and GPU-change plans.
- Remove the environment-controlled referral URL and use Modal's ordinary site only.
- Add command-level failure-path tests, coverage enforcement, CI, and development/release documentation.

## 0.1.0 - 2026-08-30

- Initial public package architecture for deploying compatible Hugging Face models to a user's own Modal account and launching them through an OpenCode config overlay.
