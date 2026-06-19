from safe_se_agent.core.scoring import count_tool_steps, score_answer
from safe_se_agent.core.types import Task


def test_multiple_choice_answer_normalization_handles_markdown_noise() -> None:
    task = Task(
        id="medqa_eval_0005",
        question="Which option is correct?",
        answer="D",
        tags=("healthcare", "medqa"),
        metadata={
            "dataset": "medqa",
            "task_type": "medical_multiple_choice",
            "raw": {"options": {"A": "a", "B": "b", "C": "c", "D": "d"}},
        },
    )

    for answer in ["D**", "**D**", "D.", "D)", "Answer: D", "<think>C?</think>\nFinal answer: D"]:
        result = score_answer(answer, task, raw_response=answer)
        assert result.correct is True
        assert result.normalized_prediction == "D"
        assert result.normalized_gold == "D"
        assert result.method == "multiple_choice"


def test_numeric_answer_normalization_still_handles_commas_and_decimals() -> None:
    task = Task(id="gsm8k", question="x", answer="1,000", tags=("gsm8k",), metadata={"dataset": "gsm8k"})

    result = score_answer("$1000.0", task)

    assert result.correct is True
    assert result.normalized_prediction == "1000"
    assert result.normalized_gold == "1000"
    assert result.method == "numeric"


def test_tool_scoring_records_step_delta_separately_from_final_answer() -> None:
    task = Task(
        id="tool",
        question="Use the tool.",
        answer="The organization information for apple.com is Apple Inc.",
        tags=("tool_use", "toolalpaca"),
        metadata={
            "dataset": "toolalpaca",
            "task_type": "tool_use_instruction",
            "intermediate_steps": [[["getOrganization", "{}", "Action: getOrganization"], "ok"]],
            "final_thought": "The organization information for apple.com is Apple Inc.",
        },
    )
    response = (
        "<invoke>GetWeather</invoke>\n"
        "<invoke>getOrganization</invoke>\n"
        "Final Answer: The organization information for apple.com is Apple Inc."
    )

    result = score_answer("</invoke>", task, raw_response=response)

    assert result.correct is True
    assert result.method == "tool_final_answer"
    assert result.metadata["tool_step_count"] == 2
    assert result.metadata["expected_tool_step_count"] == 1
    assert result.metadata["tool_step_delta"] == 1


def test_tool_step_counter_handles_action_style_traces() -> None:
    text = "Thought...\nAction: Search\nAction Input: {\"q\":\"x\"}\nObservation...\nAction: Get\n"

    assert count_tool_steps(text) == 2
