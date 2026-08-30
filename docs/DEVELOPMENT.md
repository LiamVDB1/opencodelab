# Development

OpenCodeLab supports Python 3.11 and newer. The test suite mocks Modal mutations and does not deploy an app, send inference, or require credentials.

## Local checks

Run the same deterministic checks used for acceptance:

```bash
PYTHONPATH=src uv run --no-project \
  --with pytest --with pytest-cov --with 'modal>=1.5.4,<2' --with keyring --with platformdirs \
  python -m pytest

uv run --no-project --with ruff ruff check .
uv run --no-project --with build python -m build
uv run --no-project --with twine twine check dist/*
```

CI runs the test suite on Python 3.11 through 3.14, repeats it against the declared Modal 1.5.4 lower bound, and enforces at least 80% coverage.

## Safe command smoke tests

Use an isolated config directory so local profiles are not read or changed:

```bash
export OPENCODELAB_HOME="$(mktemp -d)"
opencodelab help
opencodelab signup
opencodelab setup owner/model --gpu L40S --dry-run
opencodelab doctor
```

`doctor` may call `modal token info`, with output suppressed, to determine whether authentication exists. It never calls the model endpoint. Do not run setup without `--dry-run` during deterministic testing.

## Design boundaries

- Generated deployment files and OpenCode overlays live in the user's platform config directory, not the repository.
- The OpenCode overlay is passed through `OPENCODE_CONFIG`; it does not overwrite the normal global config.
- Endpoint credentials are stored in the system keyring when possible, with a mode-0600 local fallback.
- Generated Modal apps use one GPU container, zero minimum/buffer containers, and a one-container CPU gateway that authenticates before GPU invocation.
- The GPU endpoint requires a Modal proxy token held only by the gateway and vLLM separately checks the profile-specific client key.
- A profile is bound to the originating non-secret Modal workspace ID, deterministic OpenCodeLab-only Environment/app/secret names, and a short endpoint label; lifecycle commands verify that workspace and pass the persisted Environment explicitly.
- Setup persists recovery state before credential/deployment mutations, and teardown keeps that state until all remote cleanup succeeds.
- Hugging Face repositories are untrusted; remote code stays opt-in.

Do not add tests that deploy Modal or wake a GPU. Live validation requires a separate, explicitly approved cost boundary.
