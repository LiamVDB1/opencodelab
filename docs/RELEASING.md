# Releasing

OpenCodeLab releases are built from a clean Git revision. A release does not require deploying Modal or sending inference.

## Checklist

1. Update `CHANGELOG.md` and the version in `pyproject.toml` and `src/opencodelab/__init__.py`.
2. Verify pricing text against <https://modal.com/pricing>; record the checked date and avoid promising usable hours or free-credit eligibility.
3. Verify Modal CLI/API usage against current official documentation.
4. Run every command in [DEVELOPMENT.md](DEVELOPMENT.md).
5. Install the built wheel into an isolated tool environment and exercise `help`, `doctor`, `signup`, and `setup --dry-run` with an isolated `OPENCODELAB_HOME`.
6. Scan tracked files for credentials, live Modal endpoints, personal filesystem paths, and unsupported affiliation/referral claims.
7. Inspect `git diff` and `git status`; tag only the intended clean revision.
8. Publish through the repository's approved release workflow. Never include local generated deployment/config files or credentials.

Live wrapper verification, if desired, is a separate manual test requiring the operator's Modal account and explicit approval for possible credit use. Do not describe deterministic CI or a dry run as live deployment proof.
