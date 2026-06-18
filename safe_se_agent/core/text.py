from __future__ import annotations

import re


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
OPEN_THINK_RE = re.compile(r"<think>.*", re.IGNORECASE | re.DOTALL)


def strip_think_blocks(text: str) -> str:
    """Remove visible model thinking blocks from text used as public reasoning or memory."""

    without_closed_blocks = THINK_BLOCK_RE.sub("", text)
    without_open_block = OPEN_THINK_RE.sub("", without_closed_blocks)
    return without_open_block.strip()
