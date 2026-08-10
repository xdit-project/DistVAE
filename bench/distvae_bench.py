"""Compatibility launcher for the importable DistVAE benchmark harness."""

if __package__:
    from .harness.cli import main
else:
    from harness.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
