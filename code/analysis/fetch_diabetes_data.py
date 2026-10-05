#!/usr/bin/env python3
"""Retrieve exact public UCI diabetes inputs, with source hashes and attribution."""
import argparse
import hashlib
import io
from pathlib import Path
import urllib.request
import zipfile

URL = "https://archive.ics.uci.edu/static/public/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008.zip"
EXPECTED = {"diabetes_data.csv": "f792c388d9b470aac4fbf21c39b176534d55a116e7626881c30bbe7ab0341422",
            "IDS_mapping.csv": "f1bb82b471cb34649352597572c9b1fb00bd27f77b9f5a22a03dc3eb1039749e"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-zip", type=Path)
    ap.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent / "real_anchors")
    args = ap.parse_args()
    content = args.source_zip.read_bytes() if args.source_zip else urllib.request.urlopen(URL, timeout=60).read()
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for name, expected in EXPECTED.items():
            raw = archive.read(name)
            if hashlib.sha256(raw).hexdigest() != expected:
                raise SystemExit(f"Source checksum changed: {name}")
            target = args.outdir / ("uci_diabetes_130hosp.csv" if name == "diabetes_data.csv" else name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != raw:
                raise SystemExit(f"Existing source differs, not overwritten: {target}")
            target.write_bytes(raw)
            print(f"Verified {target}")


if __name__ == "__main__":
    main()
