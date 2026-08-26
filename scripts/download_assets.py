"""Reproducibly download sample assets for the Construction Helmet Detection capstone.

Images come from Wikimedia Commons (CC-licensed, no API key required) via the
MediaWiki API. Source URLs of every downloaded file are written to
logs/01_assets_download.log so attribution is preserved.
"""

import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "data" / "images"
LOG_PATH = ROOT / "logs" / "01_assets_download.log"

API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "construction-helmet-detection-capstone/1.0 (SDAIA Academy)"}

# (search term, number of images wanted)
IMAGE_QUERIES = [
    ("construction workers hard hats site", 3),
    ("building construction site crane workers", 2),
    ("workers pouring concrete boom pump", 1),
    ("FEMA roofer working on a home in Oklahoma", 1),
    ("Seabee welds structure construction", 1),
]

MIN_WIDTH = 1200


def api_search(term: str, limit: int = 10) -> list[dict]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"{term} filetype:bitmap",
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": 1600,
    }
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    pages = data.get("query", {}).get("pages", {})
    results = []
    for page in sorted(pages.values(), key=lambda p: p.get("index", 999)):
        info = (page.get("imageinfo") or [None])[0]
        if not info:
            continue
        if info.get("width", 0) < MIN_WIDTH:
            continue
        meta = info.get("extmetadata", {})
        results.append(
            {
                "title": page.get("title", ""),
                "thumburl": info.get("thumburl") or info.get("url"),
                "artist": meta.get("Artist", {}).get("value", ""),
                "license": meta.get("LicenseShortName", {}).get("value", ""),
            }
        )
    return results


def download(url: str, dest: Path, retries: int = 3) -> int:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as fh:
                fh.write(resp.read())
            return dest.stat().st_size
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            raise
    raise RuntimeError("unreachable")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines = [f"=== ASSET DOWNLOAD {datetime.now().isoformat()} ==="]
    idx = 1
    for term, want in IMAGE_QUERIES:
        got = 0
        lines.append(f"\n[search] '{term}' -> want {want}")
        try:
            candidates = api_search(term)
        except Exception as exc:  # noqa: BLE001
            lines.append(f"  ERROR during search: {exc}")
            continue
        for cand in candidates:
            if got >= want:
                break
            ext = ".jpg"
            dest = IMG_DIR / f"site_{idx:02d}{ext}"
            if dest.exists():
                lines.append(f"  SKIP (exists): {dest.name}")
                idx += 1
                got += 1
                continue
            try:
                size = download(cand["thumburl"], dest)
                lines.append(f"  OK {dest.name} <- {cand['title']} ({size} bytes)")
                lines.append(f"      license: {cand['license']} | artist: {cand['artist'][:120]}")
                idx += 1
                got += 1
                time.sleep(4)
            except Exception as exc:  # noqa: BLE001
                lines.append(f"  FAIL {cand['title']}: {exc}")
                time.sleep(2)
    lines.append(f"\nTotal images on disk: {len(list(IMG_DIR.glob('*.jpg')))}")
    report = "\n".join(lines)
    print(report)
    LOG_PATH.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
