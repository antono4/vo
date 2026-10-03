"""Prompt / parameter helpers implementing the skill's parameter policies.

- duration clamped to the TRUE model range [4, 30] (legacy 15s cap rejected)
- aspect-ratio auto-inference from user keywords
- Indonesian -> English generation prompt translation
- BBU CHANNEL branded filename slugging
"""
from __future__ import annotations

import re
import unicodedata

from . import config

# Keyword -> ratio inference table (skill prompt_guide section 3)
_RATIO_KEYWORDS: list[tuple[str, str]] = [
    ("21:9", "21:9"),
    ("16:9", "16:9"),
    ("4:3", "4:3"),
    ("1:1", "1:1"),
    ("3:4", "3:4"),
    ("9:16", "9:16"),
    ("tiktok", "9:16"),
    ("reels", "9:16"),
    ("shorts", "9:16"),
    ("story", "9:16"),
    ("vertical", "9:16"),
    ("portrait", "9:16"),
    ("mobile", "9:16"),
    ("hp", "9:16"),
    ("ponsel", "9:16"),
    ("youtube", "16:9"),
    ("landscape", "16:9"),
    ("cinema", "16:9"),
    ("sinematik", "16:9"),
    ("wide", "16:9"),
    ("tv", "16:9"),
    ("presentasi", "16:9"),
    ("instagram", "1:1"),
    ("square", "1:1"),
    ("persegi", "1:1"),
]

# Small Indonesian -> English lexicon for the procedural renderer's motion
# analysis. The full prompt translation is done by the LLM layer; this keeps
# the local engine usable without any external service.
_MOTION_LEXICON: list[tuple[str, str]] = [
    ("berlari", "running"),
    ("lari", "running"),
    ("berjalan", "walking"),
    ("terbang", "flying"),
    ("melompat", "jumping"),
    ("menari", "dancing"),
    ("berputar", "rotating"),
    ("berenang", "swimming"),
    ("menyelam", "diving"),
    ("melayang", "floating"),
    ("jatuh", "falling"),
    ("bermain", "playing"),
    ("duduk", "sitting"),
    ("tidur", "sleeping"),
    ("makan", "eating"),
    ("minum", "drinking"),
    ("senyum", "smiling"),
    ("tertawa", "laughing"),
    ("berpelukan", "hugging"),
    ("orbit", "orbiting"),
    ("zoom", "zooming"),
    ("drone", "aerial drone shot"),
    ("pantai", "beach"),
    ("gunung", "mountain"),
    ("hutan", "forest"),
    ("taman", "park"),
    ("sungai", "river"),
    ("danau", "lake"),
    ("desa", "village"),
    ("kota", "city"),
    ("pasar", "market"),
    ("sawah", "rice field"),
    ("malam", "night"),
    ("pagi", "morning"),
    ("senja", "sunset"),
    ("matahari", "sun"),
    ("bulan", "moon"),
    ("bintang", "stars"),
    ("awan", "clouds"),
    ("kabut", "fog"),
    ("hujan", "rain"),
    ("salju", "snow"),
    ("badai", "storm"),
    ("kucing", "cat"),
    ("anjing", "dog"),
    ("burung", "bird"),
    ("mobil", "car"),
    ("motor", "motorcycle"),
    ("produk", "product"),
    ("makanan", "food"),
    ("kopi", "coffee"),
    ("bunga", "flower"),
    ("laut", "ocean"),
    ("langit", "sky"),
    ("api", "fire"),
    ("air", "water"),
    ("emas", "gold"),
]

_STOPWORDS = {
    "buatkan", "buat", "video", "tolong", "dengan", "dan", "yang", "untuk",
    "dari", "ke", "di", "sebuah", "durasi", "detik", "waktu", "saat",
    "a", "an", "the", "of", "with", "for", "and", "please", "make", "create",
    # platform words carry no visual meaning for the description slug
    "tiktok", "youtube", "reels", "shorts", "instagram", "ig", "story",
}


def clamp_duration(value: int) -> tuple[int, str | None]:
    """Clamp to the true model range [4, 30]. Returns (value, note)."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return config.DURATION_DEFAULT, None
    if v < config.DURATION_MIN:
        return config.DURATION_MIN, (
            f"Durasi {v}s dinaikkan ke minimum model {config.DURATION_MIN}s."
        )
    if v > config.DURATION_MAX:
        return config.DURATION_MAX, (
            f"Durasi {v}s dipotong ke maksimum model {config.DURATION_MAX}s."
        )
    return v, None


def infer_ratio(text: str, explicit: str | None = None) -> str:
    """Explicit user ratio always wins, otherwise infer from keywords."""
    if explicit and explicit in config.SUPPORTED_RATIOS:
        return explicit
    low = (text or "").lower()
    for needle, ratio in _RATIO_KEYWORDS:
        if needle in low:
            return ratio
    return config.DEFAULT_RATIO


def translate_prompt(text: str) -> str:
    """Best-effort Indonesian -> English generation prompt.

    Proper nouns are preserved; only recognised motion/scene words are mapped.
    """
    text = (text or "").strip()
    if not text:
        return "A cinematic scene with smooth continuous motion."
    low = text.lower()
    mapped = [en for id_word, en in _MOTION_LEXICON if id_word in low]
    if mapped:
        return "Cinematic scene: " + ", ".join(dict.fromkeys(mapped)) + "."
    # Fall back to the original text (may already be English).
    return text


def motion_analysis(prompt: str) -> dict:
    """Extract simple motion/scene signals used by the procedural renderer."""
    low = (prompt or "").lower()
    return {
        "has_motion": any(w in low for w in ("run", "walk", "fly", "move", "orbit", "dance")),
        "is_night": any(w in low for w in ("night", "malam", "dark")),
        "is_sunset": any(w in low for w in ("sunset", "senja", "golden")),
        "is_ocean": any(w in low for w in ("ocean", "sea", "beach", "water", "laut", "pantai")),
        "is_forest": any(w in low for w in ("forest", "tree", "hutan", "jungle")),
    }


def slugify(text: str, max_len: int = 40) -> str:
    """Filesystem-safe slug for BBU branded filenames."""
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    text = text.lower()
    words = [w for w in re.split(r"[^a-z0-9]+", text) if w and w not in _STOPWORDS]
    slug = ""
    for word in words:
        candidate = f"{slug}_{word}" if slug else word
        if len(candidate) > max_len:
            break
        slug = candidate
    return slug or "video"


def branded_filename(prompt: str, index: int = 1) -> str:
    """bbuchannel_[deskripsi_singkat]_[nomor].mp4"""
    return f"{config.BBU['file_prefix']}{slugify(prompt)}_{index:02d}.mp4"
