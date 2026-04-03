"""
Constrained action vocabulary for the browser agent.

Parsed by parse_action(), executed by BrowserEnv.execute_action().
Element indices (for click/type) refer to the numbered elements in the last
observation produced by BrowserEnv.get_observation().
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Literal, Optional

Action = Literal[
    "stop",
    "back",
    "scroll_up",
    "scroll_down",
    "goto",
    "click",
    "type",
]

ACTION_VOCABULARY = """
Allowed actions (one per line, lowercase):

  stop                       -- End the episode.
  back                       -- Browser back button.
  scroll up | scroll down    -- Scroll the viewport.
  goto <url>                 -- Navigate to an absolute http(s) URL.
  click <index>              -- Click element [index] from the observation list.
  type <index> <text> [submit]
      Fill element [index] with <text>.  Quote multi-word text.
      Append "submit" to press Enter after typing.
""".strip()


class ActionParseError(ValueError):
    """Raised when a raw action string does not match the vocabulary."""


@dataclass(frozen=True)
class ParsedAction:
    action_type: Action
    index: Optional[int] = None
    url: Optional[str] = None
    text: Optional[str] = None
    submit: bool = False


_SUBMIT_TOKENS = frozenset(("submit", "true", "enter", "1"))


def parse_action(line: str) -> ParsedAction:
    """
    Parse a single raw action string into a ParsedAction.
    Raises ActionParseError on invalid input.
    """
    s = line.strip()
    if not s:
        raise ActionParseError("empty action line")

    lower = s.lower()

    # zero-arg actions
    if lower == "stop":
        return ParsedAction(action_type="stop")
    if lower == "back":
        return ParsedAction(action_type="back")
    if lower in ("scroll_up", "scroll-up", "scroll up"):
        return ParsedAction(action_type="scroll_up")
    if lower in ("scroll_down", "scroll-down", "scroll down"):
        return ParsedAction(action_type="scroll_down")

    # tokenize multi-arg actions
    parts = shlex.split(s, posix=True)
    if not parts:
        raise ActionParseError("empty action line")
    verb = parts[0].lower()

    if verb == "scroll":
        if len(parts) != 2:
            raise ActionParseError('scroll requires "up" or "down"')
        d = parts[1].lower()
        if d == "up":
            return ParsedAction(action_type="scroll_up")
        if d == "down":
            return ParsedAction(action_type="scroll_down")
        raise ActionParseError(f'scroll: expected "up" or "down", got {d!r}')

    if verb == "goto":
        m = re.match(r"^\s*goto\s+", s, re.IGNORECASE)
        if not m:
            raise ActionParseError("goto: missing URL")
        url = s[m.end():].strip()
        if not url:
            raise ActionParseError("goto: missing URL")
        if not url.startswith(("http://", "https://")):
            raise ActionParseError("goto: URL must start with http:// or https://")
        return ParsedAction(action_type="goto", url=url)

    if verb == "click":
        if len(parts) != 2:
            raise ActionParseError("click: expected exactly one integer index")
        idx = _parse_index(parts[1], "click")
        return ParsedAction(action_type="click", index=idx)

    if verb == "type":
        if len(parts) < 3:
            raise ActionParseError('type: use  type <index> <text> [submit]')
        idx = _parse_index(parts[1], "type")
        tail = parts[2:]
        submit = False
        if len(tail) >= 2 and tail[-1].casefold() in _SUBMIT_TOKENS:
            submit = True
            tail = tail[:-1]
        if not tail:
            raise ActionParseError("type: missing text")
        text = " ".join(tail)
        return ParsedAction(action_type="type", index=idx, text=text, submit=submit)

    raise ActionParseError(f"unknown action verb: {verb!r}")


def _parse_index(token: str, action_name: str) -> int:
    try:
        idx = int(token)
    except ValueError as e:
        raise ActionParseError(f"{action_name}: index must be an integer") from e
    if idx < 0:
        raise ActionParseError(f"{action_name}: index must be >= 0")
    return idx
