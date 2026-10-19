"""FLEURS (google/fleurs on Hugging Face, CC-BY-4.0): language table and curl-based download helpers.

Network work is done with ``curl``; Python only extracts files from the stream.
"""
import hashlib
import re
import subprocess

HF_FLEURS = "https://huggingface.co/datasets/google/fleurs/resolve/main"
OK_STATUS = (200, 301, 302, 303, 307, 308)

# (FLEURS config, Nemotron prompt, metric, main language). These are the model's language tags that have a FLEURS
# counterpart; en-GB, es-ES, pt-PT, fr-CA and nn-NO have no separate FLEURS set. "main" marks the 30 languages
# (29 without Turkish) whose median regression is a multilingual acceptance criterion.
LANGUAGES = [
    ("bg_bg", "bg-BG", "wer", True), ("cs_cz", "cs-CZ", "wer", True), ("da_dk", "da-DK", "wer", True),
    ("de_de", "de-DE", "wer", True), ("el_gr", "el-GR", "wer", True), ("et_ee", "et-EE", "wer", True),
    ("fi_fi", "fi-FI", "wer", True), ("fr_fr", "fr-FR", "wer", True), ("hu_hu", "hu-HU", "wer", True),
    ("hr_hr", "hr-HR", "wer", True), ("it_it", "it-IT", "wer", True), ("lt_lt", "lt-LT", "wer", True),
    ("lv_lv", "lv-LV", "wer", True), ("nl_nl", "nl-NL", "wer", True), ("pl_pl", "pl-PL", "wer", True),
    ("pt_br", "pt-BR", "wer", True), ("ro_ro", "ro-RO", "wer", True), ("ru_ru", "ru-RU", "wer", True),
    ("sk_sk", "sk-SK", "wer", True), ("sl_si", "sl-SI", "wer", True), ("sv_se", "sv-SE", "wer", True),
    ("uk_ua", "uk-UA", "wer", True), ("ar_eg", "ar-AR", "wer", True), ("en_us", "en-US", "wer", True),
    ("es_419", "es-US", "wer", True), ("he_il", "he-IL", "wer", False), ("hi_in", "hi-IN", "wer", True),
    ("ja_jp", "ja-JP", "cer", True), ("ko_kr", "ko-KR", "cer", True), ("nb_no", "nb-NO", "wer", False),
    ("th_th", "th-TH", "cer", False), ("tr_tr", "tr-TR", "wer", True), ("vi_vn", "vi-VN", "wer", True),
    ("cmn_hans_cn", "zh-CN", "cer", False),
]
BY_CONFIG = {d[0]: d for d in LANGUAGES}
# German and English: larger rehearsal quota, doubled in the training mixture, strictest regression limits.
PRIORITY = ("de_de", "en_us")


def curl_head(url):
    """HEAD without following redirects -> (status, lower-cased headers, error text)."""
    p = subprocess.run(["curl", "-sSI", "--connect-timeout", "30", "--max-time", "60", url], capture_output=True, text=True)
    code, h = None, {}
    for line in p.stdout.replace("\r", "").split("\n"):
        if line.upper().startswith("HTTP"):
            part = line.split()
            code = int(part[1]) if len(part) > 1 and part[1].isdigit() else None
            h = {}
        elif ":" in line:
            k, v = line.split(":", 1)
            h[k.strip().lower()] = v.strip()
    return code, h, p.stderr.strip()[:200]


def curl_get(url):
    p = subprocess.run(["curl", "-sSL", "--fail", "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "60",
                        "--retry", "2", url], capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"curl exit {p.returncode}: {p.stderr.decode(errors='replace').strip()[:200]}")
    return p.stdout


def curl_stream(url):
    return subprocess.Popen(["curl", "-sSL", "--fail", "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "60",
                             url], stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def hf_fingerprint(data, etag):
    """Hugging Face fingerprint: SHA-256 (64 hex) for LFS files, git blob SHA-1 (40 hex) for files kept in git."""
    e = (etag or "").strip()
    if e.startswith("W/"):
        e = e[2:]
    e = e.strip('"')
    if re.fullmatch(r"[0-9a-f]{64}", e):
        return "sha256", hashlib.sha256(data).hexdigest() == e
    if re.fullmatch(r"[0-9a-f]{40}", e):
        return "git-sha1", hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == e
    return "not checkable", None


class CountingReader:
    """Counts the bytes read from a stream and hashes them (a fully read archive can be checked against HF's SHA-256)."""

    def __init__(self, f):
        self.f, self.n, self.h = f, 0, hashlib.sha256()

    def read(self, k=-1):
        data = self.f.read(k)
        self.n += len(data)
        self.h.update(data)
        return data


def fmt_points(x):
    """0.0712 -> '7.1' (percentage points)."""
    return f"{x * 100:.1f}"


def fmt_signed(x):
    return ("+" if x >= 0 else "-") + fmt_points(abs(x))
