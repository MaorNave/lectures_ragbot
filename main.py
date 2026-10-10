"""One-click start: models → index → health check → web app.

    python main.py                  # everything, then http://localhost:8000/ opens in the browser
    python main.py --skip-ingest    # the index is already built: just check and serve
    python main.py --recreate       # rebuild the index from scratch first
    python main.py --no-serve       # prepare and check only

In PyCharm: right-click this file → Run 'main' (or Debug 'main' — breakpoints in smartup_rag/ work).
Each step is the matching `smartup` CLI command, so `smartup models|ingest|check|serve` still work one by one.
Ollama must be running (the Ollama app, or `ollama serve`).
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser

from smartup_rag import cli


def _step(title: str, argv: list[str]) -> bool:
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)), flush=True)
    return cli.main(argv) == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare and start the SmartUp assistant (models, index, check, server).")
    parser.add_argument("--skip-models", action="store_true", help="do not check or pull the models")
    parser.add_argument("--skip-ingest", action="store_true", help="do not update the index")
    parser.add_argument("--recreate", action="store_true", help="rebuild the index from scratch")
    parser.add_argument("--skip-check", action="store_true", help="do not run the health check")
    parser.add_argument("--no-serve", action="store_true", help="stop after the check; do not start the server")
    parser.add_argument("--no-browser", action="store_true", help="do not open the chat page in the browser")
    parser.add_argument("--host", help="bind address (default: API_HOST setting)")
    parser.add_argument("--port", type=int, help="port (default: API_PORT setting)")
    args = parser.parse_args(argv)

    # pull only what is missing (the first run downloads ~10 GB; later runs only confirm the models are there)
    if not args.skip_models and not _step("1/4 models", ["models", "--pull"]):
        return 1
    # unchanged lectures are skipped, so this takes seconds when nothing changed
    if not args.skip_ingest and not _step("2/4 index", ["ingest"] + (["--recreate"] if args.recreate else [])):
        return 1
    if not args.skip_check and not _step("3/4 health check", ["check"]):
        return 1
    if args.no_serve:
        return 0

    from smartup_rag.config import get_settings

    port = args.port or get_settings().api_port
    if not args.no_browser:  # the server needs a moment to start listening
        threading.Timer(3.0, webbrowser.open, [f"http://localhost:{port}/"]).start()
    serve = ["serve", "--port", str(port)] + (["--host", args.host] if args.host else [])
    return 0 if _step("4/4 server (Ctrl+C to stop)", serve) else 1


if __name__ == "__main__":
    sys.exit(main())
