"""main.py only starts uvicorn when run as a script; importing it must not start a server."""

from unittest.mock import patch


def test_import_does_not_start_server() -> None:
    with patch("uvicorn.run") as run:
        import main  # noqa: F401
    run.assert_not_called()
