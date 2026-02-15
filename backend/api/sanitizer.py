"""Response sanitizer to strip internal tool call markup from agent output."""

import re


# Patterns to strip from responses, ordered from most specific to most general
_PATTERNS = [
    # <tool_call> ... </tool_call> blocks (with content)
    re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL),
    # <function_calls> ... </function_calls> blocks
    re.compile(r"<function_calls>.*?</function_calls>", re.DOTALL),
    # <invoke name="..."> ... </invoke> blocks
    re.compile(r"<invoke\b[^>]*>.*?</invoke>", re.DOTALL),
    # Self-closing variants: <tool_call ... />
    re.compile(r"<tool_call\b[^/]*/\s*>", re.DOTALL),
    re.compile(r"<function_calls\b[^/]*/\s*>", re.DOTALL),
    re.compile(r"<invoke\b[^/]*/\s*>", re.DOTALL),
    # Orphan opening/closing tags
    re.compile(r"</?tool_call[^>]*>"),
    re.compile(r"</?function_calls[^>]*>"),
    re.compile(r"</?invoke[^>]*>"),
    re.compile(r"</?parameter[^>]*>"),
    re.compile(r"</?antml:invoke[^>]*>"),
    re.compile(r"</?antml:parameter[^>]*>"),
    re.compile(r"</?antml:function_calls[^>]*>"),
    # Tool result blocks that might leak
    re.compile(r"<tool_result>.*?</tool_result>", re.DOTALL),
    re.compile(r"</?tool_result[^>]*>"),
]

# Collapse multiple blank lines into at most two
_MULTI_BLANK = re.compile(r"\n{3,}")


def sanitize_response(content: str) -> str:
    """Strip internal tool call markup from a response string.

    Removes <tool_call>, <function_calls>, <invoke>, <parameter>,
    and related XML tags that agents may include in their output.
    Collapses excessive whitespace left behind.

    Args:
        content: Raw response string from an agent.

    Returns:
        Cleaned string safe for display to users.
    """
    if not content:
        return content

    result = content
    for pattern in _PATTERNS:
        result = pattern.sub("", result)

    # Collapse excessive blank lines
    result = _MULTI_BLANK.sub("\n\n", result)

    return result.strip()
