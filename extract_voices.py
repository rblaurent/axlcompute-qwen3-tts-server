"""
Extract French voice lines from Borderlands: The Pre-Sequel PCK files.

Pipeline: PCK -> WEM -> WAV -> filtered voice-only WAVs

Usage:
    python extract_voices.py --input "G:\\SteamLibrary\\...\\French(France)" --output ./bl_tps_french_voices
"""

import argparse
import io
import json
import struct
import subprocess
import sys
import wave
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlopen


VGMSTREAM_RELEASE_URL = "https://github.com/vgmstream/vgmstream/releases/latest"
VGMSTREAM_ZIP_NAME = "vgmstream-win64.zip"

# Voice filtering thresholds
MIN_VOICE_DURATION = 0.5   # seconds
MAX_VOICE_DURATION = 60.0  # seconds
MUSIC_THRESHOLD = 60.0     # seconds


def download_vgmstream(tools_dir: Path) -> Path:
    """Download vgmstream-cli if not present. Returns path to exe."""
    exe_path = tools_dir / "vgmstream-cli.exe"
    if exe_path.exists():
        print(f"  vgmstream-cli already cached at {exe_path}")
        return exe_path

    tools_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the latest release redirect to find the actual download URL
    print("  Resolving latest vgmstream release...")
    # GitHub latest redirects to /releases/tag/rXXXX — we need the actual tag
    import urllib.request
    req = urllib.request.Request(VGMSTREAM_RELEASE_URL, method="HEAD")
    with urlopen(req) as resp:
        final_url = resp.url  # e.g. https://github.com/vgmstream/vgmstream/releases/tag/r1234
    tag = final_url.rstrip("/").split("/")[-1]
    download_url = f"https://github.com/vgmstream/vgmstream/releases/download/{tag}/{VGMSTREAM_ZIP_NAME}"

    print(f"  Downloading {download_url} ...")
    with urlopen(download_url) as resp:
        zip_data = resp.read()

    print(f"  Extracting to {tools_dir} ...")
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        zf.extractall(tools_dir)

    if not exe_path.exists():
        # Might be nested in a subfolder
        for p in tools_dir.rglob("vgmstream-cli.exe"):
            exe_path = p
            break

    if not exe_path.exists():
        raise FileNotFoundError(
            f"vgmstream-cli.exe not found in extracted archive at {tools_dir}"
        )

    print(f"  vgmstream-cli ready at {exe_path}")
    return exe_path


# ---------------------------------------------------------------------------
# Step 1: Parse Wwise PCK binary format
# ---------------------------------------------------------------------------

def parse_pck(pck_path: Path) -> list[dict]:
    """
    Parse a Wwise AKPK file and return a list of embedded WEM entries.

    Each entry dict has: id, offset, file_size, language_id
    """
    with open(pck_path, "rb") as f:
        data = f.read()

    pos = 0

    # --- Header ---
    magic = data[pos:pos + 4]
    if magic != b"AKPK":
        raise ValueError(f"Not an AKPK file: {pck_path} (magic={magic!r})")
    pos += 4

    # Header size (LE uint32) — total bytes of header region after this field
    header_size = struct.unpack_from("<I", data, pos)[0]
    pos += 4

    # Version field (uint32)
    _version = struct.unpack_from("<I", data, pos)[0]
    pos += 4

    # Language map length (uint32)
    lang_map_size = struct.unpack_from("<I", data, pos)[0]
    pos += 4

    # Bank table length, stream table length, external table length (3 x uint32)
    bank_table_size = struct.unpack_from("<I", data, pos)[0]
    pos += 4
    stream_table_size = struct.unpack_from("<I", data, pos)[0]
    pos += 4
    # Some versions have a 4th section; read if present
    if pos + 4 <= 4 + 4 + header_size:
        _extern_table_size = struct.unpack_from("<I", data, pos)[0]
        pos += 4

    # --- Language map ---
    lang_map_start = pos
    languages = {}
    num_languages = struct.unpack_from("<I", data, pos)[0]
    pos += 4
    for _ in range(num_languages):
        # offset into language string table, language id
        str_offset = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        lang_id = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        # Read null-terminated UTF-16LE string at lang_map_start + str_offset
        # (strings are relative to the language map section)
        # Actually, the string offset is from the start of the string block
        # which is right after the language entries
        languages[lang_id] = lang_id  # We'll just use the ID
    # Skip to end of language map section
    pos = lang_map_start + lang_map_size

    # --- Bank table (parse BNK entries for embedded WEM) ---
    bank_table_start = pos
    bank_entries = _parse_entry_table(data, bank_table_start, bank_table_size)
    pos = bank_table_start + bank_table_size

    # --- Stream/WEM table ---
    stream_table_start = pos
    stream_entries = _parse_entry_table(data, stream_table_start, stream_table_size)
    pos = stream_table_start + stream_table_size

    # Extract WEM files embedded inside BNK soundbanks
    bnk_wem_entries = []
    for bank in bank_entries:
        offset = bank["offset"]
        size = bank["file_size"]
        if offset + size > len(data):
            continue
        bnk_wems = _parse_bnk_embedded_wem(data, offset, size)
        bnk_wem_entries.extend(bnk_wems)

    entries = stream_entries + bnk_wem_entries

    print(f"  Parsed {len(stream_entries)} stream WEM + {len(bnk_wem_entries)} BNK-embedded WEM from {pck_path.name}")
    if languages:
        print(f"  Language IDs found: {list(languages.keys())}")

    # Attach raw data reference for extraction
    for e in entries:
        e["_data"] = data

    return entries


def _parse_entry_table(data: bytes, table_start: int, table_size: int) -> list[dict]:
    """Parse a Wwise entry table (used for both BNK and WEM/stream sections)."""
    entries = []
    pos = table_start
    table_end = table_start + table_size

    if table_size == 0:
        return entries

    # Number of entries
    num_entries = struct.unpack_from("<I", data, pos)[0]
    pos += 4

    for _ in range(num_entries):
        if pos + 20 > table_end:
            break
        wem_id = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        block_size = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        file_size = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        start_block = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        lang_id = struct.unpack_from("<I", data, pos)[0]
        pos += 4

        if block_size > 0:
            byte_offset = start_block * block_size
        else:
            byte_offset = start_block

        entries.append({
            "id": wem_id,
            "offset": byte_offset,
            "file_size": file_size,
            "language_id": lang_id,
        })

    return entries


def _parse_bnk_embedded_wem(data: bytes, bnk_offset: int, bnk_size: int) -> list[dict]:
    """Parse a BNK soundbank and extract embedded WEM entries via DIDX+DATA sections."""
    entries = []
    bnk_end = bnk_offset + bnk_size
    pos = bnk_offset

    didx_entries = []
    data_offset = None

    while pos + 8 <= bnk_end:
        tag = data[pos:pos + 4]
        section_size = struct.unpack_from("<I", data, pos + 4)[0]
        section_data_start = pos + 8

        if tag == b"DIDX":
            # Each DIDX entry: wem_id (4), offset_in_data (4), size (4) = 12 bytes
            num = section_size // 12
            for i in range(num):
                ep = section_data_start + i * 12
                wem_id = struct.unpack_from("<I", data, ep)[0]
                wem_off = struct.unpack_from("<I", data, ep + 4)[0]
                wem_sz = struct.unpack_from("<I", data, ep + 8)[0]
                didx_entries.append((wem_id, wem_off, wem_sz))

        elif tag == b"DATA":
            data_offset = section_data_start

        pos = section_data_start + section_size
        if pos > bnk_end:
            break

    if data_offset is not None:
        for wem_id, wem_off, wem_sz in didx_entries:
            abs_offset = data_offset + wem_off
            if abs_offset + wem_sz <= len(data):
                entries.append({
                    "id": wem_id,
                    "offset": abs_offset,
                    "file_size": wem_sz,
                    "language_id": -1,  # from BNK, no separate lang field
                })

    return entries


def extract_wem_files(entries: list[dict], output_dir: Path, source_name: str) -> list[Path]:
    """Extract WEM bytes from PCK data to individual files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for entry in entries:
        data = entry["_data"]
        offset = entry["offset"]
        size = entry["file_size"]

        if offset + size > len(data):
            print(f"  WARNING: Entry {entry['id']} offset+size exceeds file, skipping")
            continue

        wem_bytes = data[offset:offset + size]

        # Quick sanity check: WEM files typically start with RIFF or RIFX
        if len(wem_bytes) < 4:
            continue
        header = wem_bytes[:4]
        if header not in (b"RIFF", b"RIFX"):
            # Not a valid WEM — skip silently
            continue

        wem_path = output_dir / f"{source_name}_{entry['id']}.wem"
        wem_path.write_bytes(wem_bytes)
        paths.append(wem_path)

    return paths


# ---------------------------------------------------------------------------
# Step 2: Convert WEM -> WAV via vgmstream-cli
# ---------------------------------------------------------------------------

def convert_wem_to_wav(
    wem_path: Path, wav_dir: Path, vgmstream_exe: Path
) -> Path | None:
    """Convert a single WEM file to WAV. Returns WAV path or None on failure."""
    wav_path = wav_dir / (wem_path.stem + ".wav")
    if wav_path.exists():
        return wav_path

    result = subprocess.run(
        [str(vgmstream_exe), "-o", str(wav_path), str(wem_path)],
        capture_output=True,
        timeout=60,
    )
    if result.returncode != 0:
        return None
    if wav_path.exists():
        return wav_path
    return None


def batch_convert(
    wem_paths: list[Path], wav_dir: Path, vgmstream_exe: Path, workers: int
) -> list[Path]:
    """Convert WEM files to WAV in parallel."""
    wav_dir.mkdir(parents=True, exist_ok=True)
    wav_paths = []
    failed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(convert_wem_to_wav, p, wav_dir, vgmstream_exe): p
            for p in wem_paths
        }
        total = len(futures)
        for i, future in enumerate(as_completed(futures), 1):
            if i % 500 == 0 or i == total:
                print(f"  Converted {i}/{total} files...")
            result = future.result()
            if result is not None:
                wav_paths.append(result)
            else:
                failed += 1

    print(f"  Conversion done: {len(wav_paths)} succeeded, {failed} failed")
    return wav_paths


# ---------------------------------------------------------------------------
# Step 3: Filter and classify
# ---------------------------------------------------------------------------

def get_wav_info(wav_path: Path) -> dict | None:
    """Read WAV metadata. Returns dict with duration, sample_rate, channels."""
    try:
        with wave.open(str(wav_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            channels = wf.getnchannels()
            duration = frames / rate if rate > 0 else 0
            return {
                "duration": round(duration, 3),
                "sample_rate": rate,
                "channels": channels,
            }
    except Exception:
        return None


def classify_file(info: dict) -> str:
    """Classify a WAV file as voice, sfx, or music based on duration."""
    dur = info["duration"]
    if dur < MIN_VOICE_DURATION:
        return "sfx"
    elif dur > MUSIC_THRESHOLD:
        return "music"
    else:
        return "voice"


def filter_and_organize(wav_paths: list[Path], output_base: Path) -> dict:
    """Classify WAV files and copy/link to category folders. Returns metadata."""
    voices_dir = output_base / "voices"
    sfx_dir = output_base / "sfx"
    music_dir = output_base / "music"
    for d in (voices_dir, sfx_dir, music_dir):
        d.mkdir(parents=True, exist_ok=True)

    metadata = {}
    counts = {"voice": 0, "sfx": 0, "music": 0, "error": 0}

    for i, wav_path in enumerate(wav_paths):
        if (i + 1) % 1000 == 0:
            print(f"  Classifying {i + 1}/{len(wav_paths)}...")

        info = get_wav_info(wav_path)
        if info is None:
            counts["error"] += 1
            continue

        category = classify_file(info)
        counts[category] += 1

        dest_map = {"voice": voices_dir, "sfx": sfx_dir, "music": music_dir}
        dest = dest_map[category] / wav_path.name

        # Hard link to avoid copying (same filesystem)
        try:
            if not dest.exists():
                dest.hardlink_to(wav_path)
        except OSError:
            # Fallback: copy
            import shutil
            shutil.copy2(wav_path, dest)

        metadata[wav_path.stem] = {
            **info,
            "category": category,
            "file": wav_path.name,
        }

    print(f"  Classification: {counts['voice']} voices, {counts['sfx']} SFX, "
          f"{counts['music']} music, {counts['error']} errors")
    return metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract French voice lines from Borderlands TPS PCK files"
    )
    parser.add_argument(
        "--input", required=True,
        help="Directory containing Audio_Banks.pck and/or Audio_Streaming.pck"
    )
    parser.add_argument(
        "--output", default="./bl_tps_french_voices",
        help="Output directory (default: ./bl_tps_french_voices)"
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Parallel workers for WEM->WAV conversion (default: 8)"
    )
    parser.add_argument(
        "--skip-convert", action="store_true",
        help="Skip WEM->WAV conversion (reuse existing WAVs)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    wem_dir = output_dir / "wem"
    wav_dir = output_dir / "wav_all"

    # Find PCK files
    pck_files = sorted(input_dir.glob("*.pck"))
    if not pck_files:
        print(f"ERROR: No .pck files found in {input_dir}")
        sys.exit(1)
    print(f"Found {len(pck_files)} PCK file(s): {[p.name for p in pck_files]}")

    # Download vgmstream if needed
    tools_dir = Path(__file__).parent / "tools"
    if not args.skip_convert:
        print("\n[1/4] Checking vgmstream-cli...")
        vgmstream_exe = download_vgmstream(tools_dir)
    else:
        vgmstream_exe = None

    # Parse and extract WEM files
    print("\n[2/4] Parsing PCK files and extracting WEM audio...")
    all_wem_paths = []
    for pck_path in pck_files:
        print(f"\n  Processing {pck_path.name} ({pck_path.stat().st_size / 1024 / 1024:.1f} MB)...")
        entries = parse_pck(pck_path)
        source_name = pck_path.stem
        wem_paths = extract_wem_files(entries, wem_dir, source_name)
        print(f"  Extracted {len(wem_paths)} valid WEM files from {pck_path.name}")
        all_wem_paths.extend(wem_paths)

        # Free the data reference
        for e in entries:
            del e["_data"]

    print(f"\n  Total WEM files: {len(all_wem_paths)}")

    # Convert WEM -> WAV
    if args.skip_convert:
        print("\n[3/4] Skipping conversion (--skip-convert), loading existing WAVs...")
        wav_paths = sorted(wav_dir.glob("*.wav"))
        print(f"  Found {len(wav_paths)} existing WAV files")
    else:
        print(f"\n[3/4] Converting WEM -> WAV ({args.workers} workers)...")
        wav_paths = batch_convert(all_wem_paths, wav_dir, vgmstream_exe, args.workers)

    # Filter and organize
    print("\n[4/4] Classifying and organizing files...")
    metadata = filter_and_organize(wav_paths, output_dir)

    # Write metadata
    meta_path = output_dir / "metadata.json"
    # Summary stats
    durations = [m["duration"] for m in metadata.values()]
    voice_durations = [m["duration"] for m in metadata.values() if m["category"] == "voice"]
    summary = {
        "total_files": len(metadata),
        "voice_count": sum(1 for m in metadata.values() if m["category"] == "voice"),
        "sfx_count": sum(1 for m in metadata.values() if m["category"] == "sfx"),
        "music_count": sum(1 for m in metadata.values() if m["category"] == "music"),
        "total_duration_hours": round(sum(durations) / 3600, 2) if durations else 0,
        "voice_duration_hours": round(sum(voice_durations) / 3600, 2) if voice_durations else 0,
        "voice_avg_duration": round(sum(voice_durations) / len(voice_durations), 2) if voice_durations else 0,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "files": metadata}, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Done! Output: {output_dir}")
    print(f"  Total files:    {summary['total_files']}")
    print(f"  Voice files:    {summary['voice_count']} ({summary['voice_duration_hours']}h)")
    print(f"  SFX files:      {summary['sfx_count']}")
    print(f"  Music files:    {summary['music_count']}")
    print(f"  Avg voice dur:  {summary['voice_avg_duration']}s")
    print(f"  Metadata:       {meta_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
