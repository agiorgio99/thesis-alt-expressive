#!/usr/bin/env python3
"""
extract_gtsinger_zips.py
─────────────────────────
Extract downloaded GTSinger per-language zip archives into

    data/GTSinger/<Language>/<Singer>/<Technique>/<Song>/<Group>/*.{wav,json,TextGrid}

matching the layout already used by data/GTSinger/English/ and expected by
GTSingerAdapter (src/alt/dataset.py).

Why this exists: GTSinger's zips aren't guaranteed to unpack with the singer
folders directly at the top level — some add an extra wrapper folder (e.g.
the zip's own name, or a generic "GTSinger" folder). This script extracts to
a scratch dir, walks down through any single-child wrapper directories until
it finds the real root (the folder that directly contains singer-ID-like
subfolders, e.g. "ZH-Alto-1"), then moves *that* into place — so the final
layout is consistent regardless of how the archive nested things.

Language is detected from the zip's filename by matching against the known
9 GTSinger language names; if a zip's name doesn't contain one of them,
you'll be asked to pass --language explicitly for that file.

Usage
─────
  python scripts/extract_gtsinger_zips.py --zip-dir ~/Downloads
  python scripts/extract_gtsinger_zips.py --zip-dir ~/Downloads --dry-run
  python scripts/extract_gtsinger_zips.py --zip-dir ~/Downloads --force   # overwrite existing dest
  python scripts/extract_gtsinger_zips.py --zip ~/Downloads/mystery.zip --language Chinese
"""

import argparse
import logging
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent

# GTSinger's 9 languages (per the NeurIPS'24 paper / HF dataset listing).
KNOWN_LANGUAGES = [
    "Chinese", "English", "Japanese", "Korean", "Russian",
    "Spanish", "French", "German", "Italian",
]

# A singer-ID folder looks like "ZH-Alto-1", "EN-Tenor-1", "FR-Soprano-2"...
_SINGER_DIR_RE = re.compile(r"^[A-Za-z]{2,3}-[A-Za-z]+-\d+$")


def detect_language(zip_path: Path) -> str | None:
    """Guess the language from the zip's filename.

    Args:
        zip_path: Path to the .zip archive.

    Returns:
        The matched language name (e.g. "Chinese"), or None if no known
        language name appears in the filename.
    """
    stem_lower = zip_path.stem.lower()
    for lang in KNOWN_LANGUAGES:
        if lang.lower() in stem_lower:
            return lang
    return None


def find_real_root(extracted: Path) -> Path:
    """Walk down through single-child wrapper folders to the real corpus root.

    The real root is the first directory (breadth-first from ``extracted``)
    that directly contains at least one singer-ID-like subfolder.

    Args:
        extracted: The directory the zip was extracted into.

    Returns:
        The folder that directly contains per-singer directories. Falls back
        to ``extracted`` itself if no singer-ID folder is found anywhere.
    """
    frontier = [extracted]
    seen = set()
    while frontier:
        current = frontier.pop(0)
        if current in seen:
            continue
        seen.add(current)
        children = [c for c in current.iterdir() if c.is_dir() and not c.name.startswith(".")]
        if any(_SINGER_DIR_RE.match(c.name) for c in children):
            return current
        frontier.extend(children)
    return extracted


def extract_one(zip_path: Path, language: str, dst_root: Path,
                force: bool, dry_run: bool) -> None:
    """Extract one language zip into ``dst_root/<language>``.

    Args:
        zip_path: Path to the source .zip archive.
        language: Destination language folder name (e.g. "Chinese").
        dst_root: Root folder under which language folders live
            (normally ``data/GTSinger``).
        force: Overwrite an existing destination folder if True; skip if False.
        dry_run: Preview only — no extraction or filesystem changes.
    """
    dest = dst_root / language
    if dest.exists() and not force:
        log.warning("SKIP %-10s — %s already exists (use --force to overwrite)",
                    language, dest)
        return

    log.info("%-10s <- %s", language, zip_path.name)
    if dry_run:
        return

    with tempfile.TemporaryDirectory(prefix=f"gtsinger_{language}_") as tmp:
        tmp_path = Path(tmp)
        # GTSinger's zips don't set the UTF-8 filename flag, so zipfile's
        # default CP437 decoding mangles any non-ASCII song-name folder (all
        # non-English languages have these). metadata_encoding="utf-8" forces
        # correct decoding instead.
        with zipfile.ZipFile(zip_path, metadata_encoding="utf-8") as zf:
            zf.extractall(tmp_path)

        real_root = find_real_root(tmp_path)
        if real_root == tmp_path and not any(
            _SINGER_DIR_RE.match(c.name) for c in tmp_path.iterdir() if c.is_dir()
        ):
            log.warning("  could not find singer-ID folders inside %s — "
                        "extracted as-is, check the layout manually", zip_path.name)

        if force and dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(real_root), str(dest))

    log.info("  -> %s", dest)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip-dir", type=Path, default=None,
                   help="Directory containing one or more GTSinger language .zip files")
    p.add_argument("--zip", type=Path, default=None,
                   help="A single zip file to extract (use with --language)")
    p.add_argument("--language", default=None,
                   help="Force the destination language name for --zip "
                        "(auto-detected from the filename if omitted)")
    p.add_argument("--dst", type=Path, default=REPO / "data" / "GTSinger",
                   help="Destination root (default: data/GTSinger)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing destination language folder")
    p.add_argument("--dry-run", action="store_true",
                   help="Preview what would be extracted, without changing anything")
    args = p.parse_args()

    if not args.zip_dir and not args.zip:
        p.error("pass either --zip-dir or --zip")

    jobs: list[tuple[Path, str]] = []

    if args.zip:
        lang = args.language or detect_language(args.zip)
        if not lang:
            p.error(f"could not detect language from {args.zip.name!r} — pass --language explicitly")
        jobs.append((args.zip, lang))

    if args.zip_dir:
        zips = sorted(args.zip_dir.glob("*.zip"))
        if not zips:
            log.warning("No .zip files found in %s", args.zip_dir)
        for z in zips:
            lang = detect_language(z)
            if not lang:
                log.warning("SKIP %-30s — filename doesn't match any known language "
                            "(%s); extract it individually with --zip --language",
                            z.name, ", ".join(KNOWN_LANGUAGES))
                continue
            jobs.append((z, lang))

    if not jobs:
        log.error("Nothing to extract.")
        sys.exit(1)

    args.dst.mkdir(parents=True, exist_ok=True)
    log.info("Destination: %s", args.dst)
    log.info("Jobs: %d", len(jobs))
    for zip_path, lang in jobs:
        extract_one(zip_path, lang, args.dst, args.force, args.dry_run)

    if not args.dry_run:
        log.info("Done. Verify with: python scripts/count_all_utterances.py --root %s/<Language>", args.dst)


if __name__ == "__main__":
    main()
