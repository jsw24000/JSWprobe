#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from probe_utils import IMAGE_EXTS, make_contact_sheet, natural_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a contact sheet from images.")
    parser.add_argument("--input", required=True, help="Input directory.")
    parser.add_argument("--output", required=True, help="Output PNG/JPG path.")
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--thumb-width", type=int, default=320)
    parser.add_argument("--max-images", type=int, default=0, help="0 means all images.")
    parser.add_argument("--no-label", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    paths = [
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    paths = sorted(paths, key=lambda p: natural_key(p.name))
    if args.max_images > 0:
        paths = paths[: args.max_images]
    out = make_contact_sheet(
        paths,
        args.output,
        cols=args.cols,
        thumb_width=args.thumb_width,
        label=not args.no_label,
    )
    if out is None:
        raise FileNotFoundError(f"No images found under {input_dir}")
    print(f"[probe] wrote {out}")


if __name__ == "__main__":
    main()
