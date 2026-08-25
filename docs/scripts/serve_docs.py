"""Serve the generated documentation with directory-style URL support."""

from __future__ import annotations

import argparse
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("target/book"),
        help="Generated site directory. Default: target/book",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the documentation in the default browser.",
    )
    args = parser.parse_args()

    directory = args.directory.resolve()
    if not (directory / "index.html").is_file():
        parser.error(
            f"{directory} does not contain index.html; build it with `just doc-all`"
        )

    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    url = f"http://{args.host}:{args.port}/"
    with ThreadingHTTPServer((args.host, args.port), handler) as server:
        print(f"Serving documentation at {url} (press Ctrl-C to stop)")
        if not args.no_open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
