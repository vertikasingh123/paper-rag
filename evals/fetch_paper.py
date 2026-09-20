"""Download the sample paper the bundled gold set is written against."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = "https://arxiv.org/pdf/1706.03762v7"
DEST = Path(__file__).resolve().parent.parent / "data" / "attention.pdf"


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"Already have {DEST}")
        return
    print(f"Downloading {URL} …")
    try:
        urllib.request.urlretrieve(URL, DEST)
    except Exception as exc:
        sys.exit(f"Download failed ({exc}). Save the PDF to {DEST} by hand.")
    print(f"Saved {DEST} ({DEST.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()
