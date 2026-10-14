from __future__ import annotations

import re


CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
WHITESPACE = re.compile(r"[ \t]+")
MULTI_NEWLINE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = CONTROL_CHARS.sub("", cleaned)
    cleaned = WHITESPACE.sub(" ", cleaned)
    cleaned = MULTI_NEWLINE.sub("\n\n", cleaned)
    return cleaned.strip()

