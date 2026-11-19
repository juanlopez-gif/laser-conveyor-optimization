"""Bundle the player + one trace into a single self-contained HTML file.

    python visualization/build_standalone.py                       # uses visualization/data/trace.js
    python visualization/build_standalone.py --trace results/<run>/trace.json --out demo.html

Useful to share one run (e-mail, Artifact, Teams) without the repository.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build(trace_js: str, out: Path) -> Path:
    html = (HERE / "index.html").read_text(encoding="utf-8")
    head = re.search(r"<!-- HEAD START -->(.*?)<!-- HEAD END -->", html, re.S).group(1)
    body = re.search(r"<!-- BODY START -->(.*?)<!-- BODY END -->", html, re.S).group(1)
    css = (HERE / "style.css").read_text(encoding="utf-8")
    app = (HERE / "app.js").read_text(encoding="utf-8")
    page = (f"{head.strip()}\n<style>\n{css}\n</style>\n{body.strip()}\n"
            f"<script>\n{trace_js}\n</script>\n<script>\n{app}\n</script>\n")
    out.write_text(page, encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", help="trace.json of a run (default: visualization/data/trace.js)")
    ap.add_argument("--out", default=str(HERE / "standalone.html"))
    args = ap.parse_args()
    if args.trace:
        data = json.loads(Path(args.trace).read_text(encoding="utf-8"))
        trace_js = "window.SIM_TRACE = " + json.dumps(data, separators=(",", ":")) + ";"
    else:
        trace_js = (HERE / "data" / "trace.js").read_text(encoding="utf-8")
    out = build(trace_js, Path(args.out))
    print(f"Standalone page written to {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
