#!/usr/bin/env python3
"""
Refresh the copy of the default payload that is inlined in ``index.html``.

``index.html`` carries the RPA/QZ payload as a ``const BASELINE = {...}`` literal
so the page works opened straight from disk, where it is not allowed to fetch
``data/``.  ``refCache`` is seeded from that literal, so the default view reads
the inline copy and never touches ``data/scores_RPA-QZ.json``.

That is the trap this script exists to close: re-running ``export_web_data.py``
rewrites ``data/`` but leaves the inline copy alone, and the page then opens
showing the old numbers while every other reference shows the new ones.  Run
this straight after the export.

    python export_web_data.py --refs all --from-benchmark ../05_website_scr/benchmark_all_references
    python sync_inline_baseline.py

``--check`` reports whether the two are in step without writing, which is what
to run in CI or before publishing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "index.html")
HEAD = "const BASELINE = "

# The inline copy is the payload minus the sub-metric breakdowns: the page does
# not read them, and they are most of the file's weight.
DROP_KEYS = ("submetrics",)
INLINE_NOTE = ("Sub-metric breakdowns are dropped from the inline copy; "
               "data/scores_{ref}.json carries them.")


def read_page(path: str) -> tuple[str, int, int, dict]:
    src = open(path).read()
    try:
        i = src.index(HEAD)
        j = src.index("};\n", i)
    except ValueError:
        raise SystemExit(f"{path}: no `const BASELINE = {{...}};` literal found")
    return src, i, j, json.loads(src[i + len(HEAD):j + 1])


def inline_form(payload: dict, ref: str) -> dict:
    out = json.loads(json.dumps(payload))          # don't mutate the caller's
    out.setdefault("provenance", {})["inline_note"] = INLINE_NOTE.format(ref=ref)
    for f in out.get("functionals", []):
        for k in DROP_KEYS:
            f.pop(k, None)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"),
                    help="where export_web_data.py wrote the payloads")
    ap.add_argument("--page", default=PAGE, help="the html to rewrite")
    ap.add_argument("--check", action="store_true",
                    help="report drift and exit non-zero, without writing")
    a = ap.parse_args(argv)

    src, i, j, current = read_page(a.page)
    ref = current.get("reference", {}).get("id")
    if not ref:
        raise SystemExit(f"{a.page}: the inline payload names no reference")

    src_path = os.path.join(a.data_dir, f"scores_{ref}.json")
    if not os.path.exists(src_path):
        raise SystemExit(f"{src_path} not found; run export_web_data.py first")

    wanted = inline_form(json.load(open(src_path)), ref)

    if wanted == current:
        print(f"in step: inline {ref} payload matches {os.path.relpath(src_path, HERE)}")
        return 0

    moved = []
    by_id = {f["id"]: f for f in current.get("functionals", [])}
    for f in wanted.get("functionals", []):
        old = by_id.get(f["id"])
        if not old:
            moved.append(f"{f['id']}: new")
            continue
        for m, v in f.get("d", {}).items():
            ov = old.get("d", {}).get(m)
            if ov is None or abs(v - ov) > 5e-7:
                moved.append(f"{f['id']}/{m}: "
                             f"{'--' if ov is None else f'{ov*100:.1f}'} -> {v*100:.1f}")

    if a.check:
        print(f"OUT OF STEP: inline {ref} payload differs from "
              f"{os.path.relpath(src_path, HERE)}")
        for line in moved[:12]:
            print("  " + line)
        if len(moved) > 12:
            print(f"  ... and {len(moved)-12} more")
        if not moved:
            print("  (scores identical; metadata differs)")
        print("Run: python sync_inline_baseline.py")
        return 1

    blob = json.dumps(wanted, separators=(",", ":"), ensure_ascii=False)
    open(a.page, "w").write(src[:i] + HEAD + blob + src[j + 1:])
    print(f"inline {ref} payload refreshed from "
          f"{os.path.relpath(src_path, HERE)} ({len(moved)} scores changed)")
    for line in moved[:12]:
        print("  " + line)
    if len(moved) > 12:
        print(f"  ... and {len(moved)-12} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
