"""
External prompt loader for community prompts (chatgpt-prompts)

Loads a JSON catalog (data/chatgpt_prompts.json) and converts entries
to our PromptTemplate (raw template shape).

Expected JSON schema (array of objects):
[
  {
    "id": "python_interpreter",
    "name": "Python Interpreter",
    "description": "Act as a Python interpreter...",
    "template": "I want you to act as a Python interpreter..."
  },
  ...
]

This integrates content from the community repo: https://github.com/pacholoamit/chatgpt-prompts
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from ..dynamic_selector import PromptTemplate


def load_chatgpt_prompts(json_path: Path | None = None) -> List[PromptTemplate]:
    """Load external ChatGPT prompts from JSON and convert to PromptTemplate list.

    Args:
        json_path: Optional override path to the JSON catalog.

    Returns:
        List of PromptTemplate instances with raw `template` populated.
    """
    try:
        path = json_path or Path("data") / "chatgpt_prompts.json"
        if not path.exists():
            return []

        with path.open("r", encoding="utf-8") as f:
            items = json.load(f)

        templates: List[PromptTemplate] = []
        for item in items:
            # Defensive reads
            tid = str(item.get("id") or item.get("key") or item.get("name") or "").strip()
            name = str(item.get("name") or tid or "Community Prompt").strip()
            desc = str(item.get("description") or "Community prompt").strip()
            content = str(item.get("template") or item.get("prompt") or "").strip()
            if not tid or not content:
                continue

            templates.append(
                PromptTemplate(
                    id=f"community__{tid}",
                    name=name,
                    description=desc,
                    template=content,
                    variables=[],
                    category="community",
                    tags=["community", "chatgpt-prompts"],
                )
            )

        return templates

    except Exception:
        # Fail closed without breaking the application
        return []


