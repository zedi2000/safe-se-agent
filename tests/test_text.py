from safe_se_agent.core.text import strip_think_blocks


def test_strip_think_blocks() -> None:
    text = "<think>private reasoning</think>\nVisible answer\n42"

    assert strip_think_blocks(text) == "Visible answer\n42"


def test_strip_unclosed_think_block() -> None:
    assert strip_think_blocks("<think>private reasoning") == ""
