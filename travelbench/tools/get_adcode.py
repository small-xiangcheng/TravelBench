from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

DATA_FILE = Path(__file__).with_name("adcode.txt")
PLACEHOLDER_SUFFIXES = ("市辖区",)


def _load_entries(path: Path) -> List[Tuple[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{path} is empty")

    entries: List[Tuple[str, str]] = []
    for line_no, line in enumerate(lines[1:], start=2):  # skip header
        if not line.strip():
            continue

        parts = line.split("\t")
        if len(parts) != 3:
            raise ValueError(f"Unexpected column count on line {line_no}: {line!r}")

        name, code, _city_code = parts
        if len(code) != 6 or not code.isdigit():
            raise ValueError(f"Invalid adcode {code!r} on line {line_no}")

        entries.append((name, code))

    return entries


def _is_placeholder(name: str) -> bool:
    return name.endswith(PLACEHOLDER_SUFFIXES)


def build_fullname2adcode(path: Path = DATA_FILE) -> Dict[str, int]:
    """
    Parse the adcode file and build a mapping from full administrative names to adcodes.
    Placeholder nodes such as \"市辖区\" are skipped when composing full names.
    """

    entries = _load_entries(path)
    code_to_name = {code: name for name, code in entries}

    def parent_code(code: str) -> str | None:
        if code.endswith("0000"):
            return None
        if code.endswith("00"):
            parent = f"{code[:2]}0000"
            return parent if parent in code_to_name else None

        candidate = f"{code[:4]}00"
        if candidate in code_to_name:
            return candidate

        fallback = f"{code[:2]}0000"
        return fallback if fallback in code_to_name else None

    @lru_cache(None)
    def fullname(code: str, include_placeholder: bool) -> str:
        name = code_to_name[code]
        parent = parent_code(code)

        parent_full = fullname(parent, False) if parent else ""
        if _is_placeholder(name):
            if not include_placeholder:
                return parent_full

            # Drop duplicated parent name prefix for cleaner full names.
            if parent:
                parent_name = code_to_name[parent]
                if name.startswith(parent_name):
                    name = name[len(parent_name) :]

        return f"{parent_full}{name}" if parent_full else name

    fullname2adcode: Dict[str, int] = {}
    for name, code in entries:
        full_name = fullname(code, True)
        if not full_name:
            continue

        code_value = int(code)
        existing = fullname2adcode.get(full_name)
        if existing is not None and existing != code_value:
            raise ValueError(f"Conflicting adcode for {full_name}: {existing} vs {code_value}")

        fullname2adcode[full_name] = code_value

    return fullname2adcode


fullname2adcode = build_fullname2adcode()


def get_adcode(name: str) -> Dict[str, object]:
    """
    Tool name: get_adcode
    Description: Resolve a Chinese行政区划名称 (省、市、县/区) to its 6-digit ADCODE using the loaded dataset.
    Parameters:
        name (str): 行政区划名称。支持完整路径名（如“北京市”、“北京市朝阳区”）或仅末级名称（如“朝阳区”）。
    Returns:
        dict: {
            'matches': list[{'full_name': str, 'adcode': int}],    # 所有候选项，按名称长度和字典序排序
            'note': str                                            # 说明是否精确命中、模糊多解或未找到
        }
    """

    name = name.strip()

    # Exact match first.
    if name in fullname2adcode:
        code = fullname2adcode[name]
        match = {"full_name": name, "adcode": code}
        return {"resolved": match, "matches": [match], "note": "exact match"}

    # Fallback: suffix matches (e.g., name == '朝阳区' → multiple hits).
    matches = [
        {"full_name": full_name, "adcode": code}
        for full_name, code in fullname2adcode.items()
        if full_name.endswith(name)
    ]
    matches.sort(key=lambda item: (len(item["full_name"]), item["full_name"]))

    resolved = matches[0] if len(matches) == 1 else None
    note = "no match" if not matches else ("ambiguous matches" if not resolved else "suffix match")

    return {"resolved": resolved, "matches": matches, "note": note}


__all__ = ["build_fullname2adcode", "fullname2adcode", "get_adcode"]


if __name__ == "__main__":
    mapping = fullname2adcode
    print(f"Loaded {len(mapping)} fullname entries from {DATA_FILE.name}")
