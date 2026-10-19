"""Text normalisation and error metrics.

Turkish: ``normalize`` / ``wer`` (optionally with digits spelled out as Turkish words, as in the v0.1 labels).
Other languages (FLEURS multilingual check): ``units`` / ``error_rate`` — language tag dropped, NFKC + casefold,
punctuation and symbols become spaces, combining marks are KEPT; for CER, spaces are not counted.
"""
import re
import unicodedata

# When True, digits are spelled out as Turkish words in labels and before scoring (v0.1: on).
NUMBERS_TO_WORDS = False

_ONES = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
_TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]
LANG_TAG = re.compile(r"<([a-z]{2,3})-[A-Za-z]{2,4}>")


def _below_thousand(n):
    part = []
    h, rest = divmod(n, 100)
    if h:
        part.append("yüz" if h == 1 else _ONES[h] + " yüz")
    t, o = divmod(rest, 10)
    if t:
        part.append(_TENS[t])
    if o:
        part.append(_ONES[o])
    return " ".join(part)


def number_words(n):
    """Integer -> Turkish words ("1982" -> "bin dokuz yüz seksen iki"; "1000" -> "bin"; "0" -> "sıfır")."""
    if n == 0:
        return "sıfır"
    part = []
    for div, name in ((10 ** 9, "milyar"), (10 ** 6, "milyon"), (1000, "bin")):
        q, n = divmod(n, div)
        if q:
            part.append(name if (q == 1 and name == "bin") else _below_thousand(q) + " " + name)
    if n:
        part.append(_below_thousand(n))
    return " ".join(part)


def digits_to_words(text):
    """Spell out digit sequences; an apostrophe suffix sticks to the last word ("3'ün" -> "üçün", "1962'de" ->
    "bin dokuz yüz altmış ikide"); thousands dots are joined ("1.000" -> "bin"); a decimal comma is read ("1,5" ->
    "bir virgül beş"). Tokens mixed with letters ("2x", "c1", "20si") and numbers longer than 12 digits are kept."""
    text = re.sub(r"(?<!\d)(\d{1,3})((?:\.\d{3})+)(?!\d)", lambda m: m.group(1) + m.group(2).replace(".", ""), text or "")

    def decimal(m):
        whole, frac = m.group(1), m.group(2)
        if len(whole) > 12 or len(frac) > 6:
            return m.group(0)
        k = " ".join(number_words(int(c)) for c in frac) if frac.startswith("0") else number_words(int(frac))
        return f"{number_words(int(whole))} virgül {k}"
    text = re.sub(r"(?<![\w,.])(\d+),(\d+)(?![\w,.])", decimal, text)

    def replace(m):
        digits, suffix = m.group(1), m.group(2)
        if len(digits) > 12:
            return m.group(0)
        words = number_words(int(digits)).split()
        if suffix:
            words[-1] += suffix
        return " ".join(words)
    return re.sub(r"(?<![\w'’])(\d+)(?:['’]([^\W\d_]+))?(?![\w'’])", replace, text)


def tag_sentences(text, tag="<tr-TR>"):
    """The model writes a language tag after every finished sentence; training labels must do the same."""
    text = re.sub(r"\s+", " ", text or "").strip()
    return re.sub(r"([.?!…]+)(\s+|$)", lambda m: m.group(1) + " " + tag + (" " if m.group(2) else ""), text).strip()


def normalize(s):
    """Turkish scoring normalisation -> list of words."""
    s = re.sub(r"<[a-z]{2}-[A-Z]{2}>", " ", s or "")
    s = unicodedata.normalize("NFC", s)
    if NUMBERS_TO_WORDS:
        s = digits_to_words(s)
    s = s.replace("I", "ı").replace("İ", "i").lower()
    s = s.replace("'", "").replace("’", "")
    s = re.sub(r"[^\w\s]", " ", s)
    return s.split()


def wer(ref, hyp):
    """Utterance WER on ``normalize``d words."""
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / len(r)


def units(text, metric):
    """Non-Turkish languages: words (``wer``) or characters without spaces (``cer``)."""
    s = LANG_TAG.sub(" ", text or "")
    s = unicodedata.normalize("NFKC", s).casefold()
    s = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s)
    if metric == "cer":
        return [c for c in s if not c.isspace()]
    return s.split()


def edits(r, h):
    """Levenshtein distance: substitutions + insertions + deletions."""
    if not r:
        return len(h)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        ri = r[i - 1]
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (ri != h[j - 1]))
            prev = cur
    return d[len(h)]


def error_rate(refs, hyps, metric):
    """Total edits / total reference units (per-language FLEURS metric)."""
    e = n = 0
    for r, h in zip(refs, hyps):
        rb = units(r, metric)
        if rb:
            e += edits(rb, units(h, metric))
            n += len(rb)
    return e / max(1, n)


def tr_error(rows, hyps, numbers_to_words):
    """Utterance WER weighted by reference length, summed (rows carry ``raw_text``); rounded to 4 places."""
    global NUMBERS_TO_WORDS
    old = NUMBERS_TO_WORDS
    NUMBERS_TO_WORDS = numbers_to_words
    try:
        e = n = 0
        for x, h in zip(rows, hyps):
            r = normalize(x["raw_text"])
            if r:
                e += wer(x["raw_text"], h) * len(r)
                n += len(r)
        return round(e / max(1, n), 4)
    finally:
        NUMBERS_TO_WORDS = old


def median(xs):
    xs = sorted(xs)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2
