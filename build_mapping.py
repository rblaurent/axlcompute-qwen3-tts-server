"""Build Wwise event -> character name mapping for Borderlands TPS.

Parses HIRC hierarchy from BNK files inside PCK containers, then
cross-references Wwise event IDs (FNV-1 hashes) with VO section names
from .INT localization files to map each WEM audio file to a character
and subtitle text. Also maps media IDs to source BNK bank names for
100% coverage of at least the bank/level context.
"""

import struct
import re
import json
from pathlib import Path
from collections import Counter


def fnv1_32(s):
    """Wwise FNV-1 hash (32-bit, lowercase input)."""
    h = 2166136261
    for c in s.lower().encode("ascii"):
        h = ((h * 16777619) ^ c) & 0xFFFFFFFF
    return h


# Character name extraction from bank names
BANK_CHARACTER_MAP = {
    "GD_Enforcer_Streaming_SF_Voice": "Enforcer",
    "GD_Gladiator_Streaming_SF_Voice": "Gladiator",
    "GD_Lawbringer_Streaming_SF_Voice": "Lawbringer",
    "GD_Prototype_Streaming_SF_Voice": "Prototype",
}


def parse_pck_banks(pck_path):
    """Parse a PCK file and return per-bank data.

    Returns list of dicts with keys: name, sounds, sound_parent, actions,
    events, container_parent.
    """
    data = pck_path.read_bytes()

    pos = 8
    _ver = struct.unpack_from("<I", data, pos)[0]; pos += 4
    lang_size = struct.unpack_from("<I", data, pos)[0]; pos += 4
    _bank_size = struct.unpack_from("<I", data, pos)[0]; pos += 4
    _stream_size = struct.unpack_from("<I", data, pos)[0]; pos += 4
    _extern_size = struct.unpack_from("<I", data, pos)[0]; pos += 4
    pos += lang_size

    num_banks = struct.unpack_from("<I", data, pos)[0]; pos += 4
    banks = []

    for _ in range(num_banks):
        vals = struct.unpack_from("<5I", data, pos); pos += 20
        offset = vals[3] * vals[1] if vals[1] > 0 else vals[3]
        bpos = offset
        bnk_end = offset + vals[2]

        bank = {
            "name": None,
            "sounds": {},
            "sound_parent": {},
            "actions": {},
            "events": {},
            "container_parent": {},
        }

        while bpos + 8 <= bnk_end:
            tag = data[bpos:bpos + 4]
            sec_size = struct.unpack_from("<I", data, bpos + 4)[0]
            sec_start = bpos + 8

            if tag == b"STID":
                enc = struct.unpack_from("<I", data, sec_start)[0]
                num = struct.unpack_from("<I", data, sec_start + 4)[0]
                p2 = sec_start + 8
                for _ in range(num):
                    _sid = struct.unpack_from("<I", data, p2)[0]; p2 += 4
                    slen = data[p2]; p2 += 1
                    name = data[p2:p2 + slen].decode("ascii", errors="replace")
                    p2 += slen
                    bank["name"] = name

            elif tag == b"HIRC":
                num_obj = struct.unpack_from("<I", data, sec_start)[0]
                opos = sec_start + 4
                for _ in range(num_obj):
                    if opos + 5 > sec_start + sec_size:
                        break
                    obj_type = data[opos]
                    obj_size = struct.unpack_from("<I", data, opos + 1)[0]
                    obj_bytes = data[opos + 5:opos + 5 + obj_size]
                    obj_id = struct.unpack_from("<I", obj_bytes, 0)[0]

                    if obj_type == 2 and obj_size >= 31:  # Sound
                        source_id = struct.unpack_from("<I", obj_bytes, 12)[0]
                        parent_id = struct.unpack_from("<I", obj_bytes, 27)[0]
                        bank["sounds"][obj_id] = source_id
                        bank["sound_parent"][obj_id] = parent_id

                    elif obj_type == 3 and obj_size >= 10:  # Action
                        bank["actions"][obj_id] = struct.unpack_from("<I", obj_bytes, 6)[0]

                    elif obj_type == 4 and obj_size >= 8:  # Event
                        num_act = struct.unpack_from("<I", obj_bytes, 4)[0]
                        if num_act <= 100 and 8 + num_act * 4 <= obj_size:
                            bank["events"][obj_id] = [
                                struct.unpack_from("<I", obj_bytes, 8 + i * 4)[0]
                                for i in range(num_act)
                            ]

                    elif obj_type in (5, 6, 7) and obj_size >= 31:  # Containers
                        parent_id = struct.unpack_from("<I", obj_bytes, 27)[0]
                        bank["container_parent"][obj_id] = parent_id

                    opos += 5 + obj_size

            bpos = sec_start + sec_size
            if bpos > bnk_end:
                break

        banks.append(bank)

    return banks


def merge_banks(bank_list):
    """Merge multiple bank dicts into flat dicts."""
    sounds, sound_parent, actions, events, container_parent = {}, {}, {}, {}, {}
    for bank in bank_list:
        sounds.update(bank["sounds"])
        sound_parent.update(bank["sound_parent"])
        actions.update(bank["actions"])
        events.update(bank["events"])
        container_parent.update(bank["container_parent"])
    return sounds, sound_parent, actions, events, container_parent


def build_event_to_media(sounds, sound_parent, actions, events, container_parent):
    """Walk event -> action -> (sound|container) -> media with nested hierarchy."""
    parent_to_media = {}
    for sid, mid in sounds.items():
        pid = sound_parent.get(sid)
        if pid:
            parent_to_media.setdefault(pid, set()).add(mid)

    parent_to_containers = {}
    for cid, pid in container_parent.items():
        parent_to_containers.setdefault(pid, set()).add(cid)

    def get_all_media(obj_id, depth=0, visited=None):
        if visited is None:
            visited = set()
        if obj_id in visited or depth > 10:
            return set()
        visited.add(obj_id)
        result = set()
        if obj_id in sounds:
            result.add(sounds[obj_id])
        if obj_id in parent_to_media:
            result.update(parent_to_media[obj_id])
        if obj_id in parent_to_containers:
            for child_cid in parent_to_containers[obj_id]:
                result.update(get_all_media(child_cid, depth + 1, visited))
        return result

    event_to_media = {}
    for eid, aids in events.items():
        media = set()
        for aid in aids:
            target = actions.get(aid)
            if target is not None:
                media.update(get_all_media(target))
        if media:
            event_to_media[eid] = media

    return event_to_media


def build_media_to_bank(banks):
    """Map media_id -> bank_name using per-bank sound data."""
    media_to_bank = {}
    for bank in banks:
        name = bank["name"]
        if not name:
            continue
        for _sid, mid in bank["sounds"].items():
            # Character-specific banks take priority
            if mid not in media_to_bank or name in BANK_CHARACTER_MAP:
                media_to_bank[mid] = name
    return media_to_bank


def collect_vo_names(game_dir):
    """Collect all VO section names and character from .INT files."""
    vo_sections = []
    seen = set()
    for int_file in list(game_dir.rglob("*.int")) + list(game_dir.rglob("*.INT")):
        try:
            raw = int_file.read_bytes()
            if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
                text = raw.decode("utf-16", errors="replace")
            else:
                text = raw.decode("utf-8", errors="replace")
        except Exception:
            continue

        for match in re.finditer(r"\[\s*(\w+)\s*\]", text):
            section = match.group(1)
            if section in seen:
                continue
            if section.startswith(("VO_", "VOBD_", "VOCT_", "VOSQ_")):
                parts = section.split("_")
                character = parts[-1]
                vo_sections.append((section, character))
                seen.add(section)

    return vo_sections


def collect_subtitles(game_dir, lang="FRA"):
    """Collect subtitle text from localization files."""
    subtitles = {}
    for loc_file in list(game_dir.rglob(f"*.{lang}")) + list(game_dir.rglob(f"*.{lang.lower()}")):
        try:
            raw = loc_file.read_bytes()
            if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
                text = raw.decode("utf-16", errors="replace")
            else:
                text = raw.decode("utf-8", errors="replace")
        except Exception:
            continue

        current_section = None
        for line in text.split("\n"):
            line = line.strip()
            sec_match = re.match(r"\[\s*(\w+)\s*\]", line)
            if sec_match:
                current_section = sec_match.group(1)
                continue
            if current_section and "Subtitles[0]" in line:
                text_match = re.search(r'Text="(.*?)"', line)
                if text_match:
                    subtitles[current_section] = text_match.group(1)

    return subtitles


def main():
    fr_pck = Path(
        "G:/SteamLibrary/steamapps/common/BorderlandsPreSequel/"
        "WillowGame/CookedPCConsole/Audio/French(France)/Audio_Banks.pck"
    )
    base_pck = Path(
        "G:/SteamLibrary/steamapps/common/BorderlandsPreSequel/"
        "WillowGame/CookedPCConsole/Audio/Audio_Banks.pck"
    )
    game_dir = Path("G:/SteamLibrary/steamapps/common/BorderlandsPreSequel")
    output_dir = Path("bl_tps_french_voices")

    # ── Step 1: Parse HIRC ─────────────────────────────────────────────
    print("Parsing HIRC hierarchy...")
    fr_banks = parse_pck_banks(fr_pck)
    base_banks = parse_pck_banks(base_pck)
    print(f"  French: {len(fr_banks)} banks, Base: {len(base_banks)} banks")

    # Build media -> bank name (for 100% bank-level mapping)
    media_to_bank = build_media_to_bank(fr_banks)

    # Merge all HIRC data
    sounds, sound_parent, actions, events, container_parent = merge_banks(fr_banks)
    s2, sp2, a2, e2, cp2 = merge_banks(base_banks)
    for k, v in e2.items():
        events.setdefault(k, v)
    for k, v in a2.items():
        actions.setdefault(k, v)
    for k, v in cp2.items():
        container_parent.setdefault(k, v)

    print(f"  Total: {len(events)} events, {len(actions)} actions, {len(sounds)} sounds")

    # ── Step 2: Event -> media mapping ─────────────────────────────────
    print("Building event -> media mapping...")
    event_to_media = build_event_to_media(
        sounds, sound_parent, actions, events, container_parent
    )
    all_media = set(m for ms in event_to_media.values() for m in ms)
    print(f"  {len(event_to_media)} events -> {len(all_media)} unique media IDs")

    # ── Step 3: VO names + subtitles ───────────────────────────────────
    print("Collecting VO names and subtitles...")
    vo_sections = collect_vo_names(game_dir)
    subtitles_fr = collect_subtitles(game_dir, "FRA")
    subtitles_en = collect_subtitles(game_dir, "INT")
    print(f"  {len(vo_sections)} VO sections, {len(subtitles_fr)} FR subs, {len(subtitles_en)} EN subs")

    # ── Step 4: Match events to VO names ───────────────────────────────
    print("Matching VO names to Wwise events via FNV-1 hash...")
    media_to_info = {}
    for section_name, character in vo_sections:
        for prefix in ("Play_", ""):
            candidate = prefix + section_name
            h = fnv1_32(candidate)
            if h in event_to_media:
                for mid in event_to_media[h]:
                    media_to_info[mid] = {
                        "character": character,
                        "event": candidate,
                        "vo_section": section_name,
                        "subtitle_fr": subtitles_fr.get(section_name, ""),
                        "subtitle_en": subtitles_en.get(section_name, ""),
                    }
                break

    # ── Step 5: Bank-based character for character-specific banks ──────
    for mid, bank_name in media_to_bank.items():
        if mid not in media_to_info and bank_name in BANK_CHARACTER_MAP:
            media_to_info[mid] = {
                "character": BANK_CHARACTER_MAP[bank_name],
                "event": "",
                "vo_section": "",
                "subtitle_fr": "",
                "subtitle_en": "",
            }

    # ── Step 6: Propagate through containers (conservative) ───────────
    print("Propagating character info through containers...")
    # Build container -> media_ids
    container_to_media = {}
    for sid, mid in sounds.items():
        pid = sound_parent.get(sid)
        if pid:
            container_to_media.setdefault(pid, []).append(mid)

    for _pass in range(3):
        prev = len(media_to_info)
        for _cid, mids in container_to_media.items():
            known_chars = [media_to_info[m]["character"] for m in mids if m in media_to_info]
            unknown = [m for m in mids if m not in media_to_info]
            if len(known_chars) < 3 or not unknown:
                continue
            char_counts = Counter(known_chars)
            top_char, top_count = char_counts.most_common(1)[0]
            # Only propagate if >=90% agree AND there's a meaningful number of knowns
            if top_count / len(known_chars) >= 0.9:
                for m in unknown:
                    media_to_info[m] = {
                        "character": top_char,
                        "event": "",
                        "vo_section": "",
                        "subtitle_fr": "",
                        "subtitle_en": "",
                    }
        gained = len(media_to_info) - prev
        if gained == 0:
            break
        print(f"  Pass {_pass + 1}: +{gained} propagated")

    # ── Step 7: Build output ──────────────────────────────────────────
    wem_dir = output_dir / "wem"
    our_ids = {}
    for f in wem_dir.glob("*.wem"):
        parts = f.stem.split("_")
        wem_id = int(parts[-1])
        our_ids[wem_id] = f.stem

    # Assemble mapping
    mapping = {}
    for wem_id, stem in our_ids.items():
        info = media_to_info.get(wem_id)
        bank_name = media_to_bank.get(wem_id, "")
        if info:
            mapping[stem] = {
                "wem_id": wem_id,
                "character": info["character"],
                "bank": bank_name,
                "event": info["event"],
                "vo_section": info["vo_section"],
                "subtitle_fr": info["subtitle_fr"],
                "subtitle_en": info["subtitle_en"],
            }
        else:
            mapping[stem] = {
                "wem_id": wem_id,
                "character": "unknown",
                "bank": bank_name,
                "event": "",
                "vo_section": "",
                "subtitle_fr": "",
                "subtitle_en": "",
            }

    # ── Step 8: Merge into metadata.json ──────────────────────────────
    meta_path = output_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        for stem, file_meta in metadata.get("files", {}).items():
            if stem in mapping:
                m = mapping[stem]
                file_meta["character"] = m["character"]
                file_meta["bank"] = m["bank"]
                file_meta["event"] = m["event"]
                file_meta["vo_section"] = m["vo_section"]
                file_meta["subtitle_fr"] = m["subtitle_fr"]
                file_meta["subtitle_en"] = m["subtitle_en"]

        # Update summary
        voice_files = {k: v for k, v in metadata["files"].items() if v.get("category") == "voice"}
        char_counts = Counter(v.get("character", "unknown") for v in voice_files.values())
        metadata["summary"]["characters"] = dict(char_counts.most_common())
        mapped_voice = sum(1 for v in voice_files.values() if v.get("character", "unknown") != "unknown")
        metadata["summary"]["voices_with_character"] = mapped_voice
        metadata["summary"]["voices_with_subtitle"] = sum(
            1 for v in voice_files.values() if v.get("subtitle_fr")
        )

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"\nUpdated {meta_path}")

    # Save standalone mapping
    map_path = output_dir / "wem_mapping.json"
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"Saved {map_path}")

    # ── Stats ─────────────────────────────────────────────────────────
    known = sum(1 for v in mapping.values() if v["character"] != "unknown")
    with_sub = sum(1 for v in mapping.values() if v["subtitle_fr"])
    print(f"\n{'=' * 60}")
    print(f"  Total files:          {len(mapping)}")
    print(f"  With character:       {known} ({100 * known / len(mapping):.1f}%)")
    print(f"  With French subtitle: {with_sub}")
    print()

    chars = Counter(v["character"] for v in mapping.values() if v["character"] != "unknown")
    print(f"  Characters ({len(chars)} unique):")
    for char, count in chars.most_common(30):
        sub_count = sum(1 for v in mapping.values() if v["character"] == char and v["subtitle_fr"])
        print(f"    {char:25s} {count:5d} lines  ({sub_count} with subtitle)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
