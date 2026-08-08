# Contributing to Tunnel

Thanks for your interest in improving Tunnel.

## Development setup

Tunnel requires macOS and Python 3.11+.

```bash
git clone https://github.com/Mayan10/tunnel.git
cd tunnel
pip install -e ".[dev]"
```

## Before opening a pull request

Run the full check suite locally:

```bash
ruff check .
mypy tunnel
python3 -m unittest -v
```

All three must pass. CI runs the same checks on every pull request.

## Guidelines

- Keep changes focused. Prefer several small pull requests over one large one.
- Add or update tests for any behavior change.
- Tunnel must never mutate a source Apple Music playlist. Any change that touches
  `tunnel/apple_music.py` should preserve the existing safety checks (track-count
  verification, snapshotting before writes, and refusing partial or duplicate orders).
- Match the existing code style: no required external runtime dependencies (the `ml` extra is
  optional and must degrade gracefully when it is not installed), type-annotated functions, and
  `dataclass`-based models.
- Changes to `tunnel/embedding.py`'s input feature layout require retraining and re-committing
  `tunnel/models/audio_embedding_v1.mlmodel` via `python3 tools/train_embedding_model.py`
  (requires `pip install ".[ml]"`).

## Reporting bugs

Open an issue with your macOS version, Python version, and the exact `tunnel` command
you ran. Include the printed diagnostics line (missing BPM/year counts, local files,
audio-analyzed counts) when the issue is about ordering quality.

## License

By contributing, you agree that your contributions will be licensed under the project's
[MIT License](LICENSE).
