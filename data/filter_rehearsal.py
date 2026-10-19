#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rehearsal label filter (read-only on its input; writes a filtered copy).

Removes rehearsal rows whose label (= the base model's own output) contains
  (a) a script foreign to the language (e.g. Cyrillic in a Latin-script language), or
  (b) a loop: the same letter 8+ times in a row (a looping failure mode of the base model).
In v0.1 this removed 31 of 12,680 rows (12,649 kept). Prints only counts and file names.

TR: Prova etiketlerinde dile yabancı yazı ya da 8+ aynı harf tekrarı (döngü) olan satırları çıkarır.
"""
import argparse
import collections
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ayaz_asr import read_jsonl, work_dir  # noqa: E402

SCRIPTS = {
    "thai": r"[฀-๿]", "han": r"[一-鿿㐀-䶿]", "kana": r"[぀-ヿ]",
    "hangul": r"[가-힯ᄀ-ᇿ㄰-㆏]", "cyrillic": r"[Ѐ-ӿ]", "greek": r"[Ͱ-Ͽ]",
    "arabic": r"[؀-ۿ]", "hebrew": r"[֐-׿]", "devanagari": r"[ऀ-ॿ]",
}
OWN = {"ja-JP": {"han", "kana"}, "ko-KR": {"hangul", "han"}, "zh-CN": {"han"}, "bg-BG": {"cyrillic"}, "ru-RU": {"cyrillic"},
       "uk-UA": {"cyrillic"}, "el-GR": {"greek"}, "ar-AR": {"arabic"}, "he-IL": {"hebrew"}, "hi-IN": {"devanagari"}}
LOOP = re.compile(r"([^\W\d_])\1{7,}")
REPEATED_WORD = re.compile(r"\b(\w{3,})\b(?:\W+\1\b){4,}")


def defect(x):
    t, lang = x.get("text", ""), x.get("target_lang", "")
    if LOOP.search(t):
        return "loop"
    for name, pattern in SCRIPTS.items():
        if name not in OWN.get(lang, set()) and re.search(pattern, t):
            return "foreign script"
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", help="working directory / çalışma dizini (default: $AYAZ_ASR_WORK or ./work)")
    ap.add_argument("--input", help="rehearsal manifest (default: <work>/rehearsal/manifest_rehearsal.jsonl)")
    ap.add_argument("--output", help="filtered manifest (default: <work>/manifests/rehearsal_filtered.jsonl)")
    a = ap.parse_args()
    work = work_dir(a.work_dir)
    src = a.input or os.path.join(work, "rehearsal", "manifest_rehearsal.jsonl")
    dst = a.output or os.path.join(work, "manifests", "rehearsal_filtered.jsonl")
    rows = read_jsonl(src)
    removed, reason = [], collections.Counter()
    for x in rows:
        n = defect(x)
        if n:
            removed.append(x)
            reason[(x.get("target_lang"), n)] += 1
    print("removed (language, reason): " + (" · ".join(f"{k[0]} {k[1]} {v}" for k, v in sorted(reason.items())) or "none"))
    print("info — rehearsal rows with a word repeated 5+ times (not filtered): "
          + str(sum(1 for x in rows if REPEATED_WORD.search(x.get("text", "")))))
    key = lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True)
    gone = {key(x) for x in removed}
    kept = [x for x in rows if key(x) not in gone]
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        for x in kept:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(open(dst, "rb").read()).hexdigest()
    print(f"rehearsal {len(rows)} -> {len(kept)} rows (removed {len(rows) - len(kept)}) · written {os.path.basename(dst)} · SHA-256 {sha}")


if __name__ == "__main__":
    main()
