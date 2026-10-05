#!/usr/bin/env python3
"""Retrieve and verify the official UCI diabetes ID mapping.

Source: UCI Diabetes 130-US Hospitals for Years 1999-2008, dataset 296,
DOI 10.24432/C5230J, CC BY 4.0. The project keeps downloaded CSV files
local; they are ignored by Git. This script downloads no model responses.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path


URL = ("https://archive.ics.uci.edu/static/public/296/"
       "diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008.zip")
EXPECTED_SHA256 = "f1bb82b471cb34649352597572c9b1fb00bd27f77b9f5a22a03dc3eb1039749e"
DEFAULT_OUT = Path(__file__).resolve().parent / "real_anchors" / "IDS_mapping.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path,
                        help="verify a previously downloaded official archive")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    raw = (args.source_zip.read_bytes() if args.source_zip else
           urllib.request.urlopen(URL, timeout=60).read())
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        mapping = archive.read("IDS_mapping.csv")
    actual_sha = hashlib.sha256(mapping).hexdigest()
    if actual_sha != EXPECTED_SHA256:
        raise SystemExit(f"Official mapping checksum changed: {actual_sha}. "
                         "Inspect the source before using new labels.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        if args.out.read_bytes() != mapping:
            raise SystemExit(f"Existing mapping differs: {args.out}; not overwritten")
        print(f"verified existing {args.out} ({actual_sha})")
    else:
        args.out.write_bytes(mapping)
        print(f"saved {args.out} ({actual_sha})")


if __name__ == "__main__":
    main()
