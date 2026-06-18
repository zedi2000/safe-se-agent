from scripts.prepare_medqa import convert_rows, format_question, parse_answer


def test_parse_medqa_answer_from_idx_and_text() -> None:
    row = {
        "question": "Which vitamin deficiency causes scurvy?",
        "options": {"A": "Vitamin A", "B": "Vitamin C", "C": "Vitamin D", "D": "Vitamin K"},
        "answer_idx": "B",
        "answer": "Vitamin C",
    }

    answer, answer_text = parse_answer(row)

    assert answer == "B"
    assert answer_text == "Vitamin C"


def test_convert_medqa_rows_to_task_schema() -> None:
    rows = [
        {
            "question": "Which vitamin deficiency causes scurvy?",
            "options": {"A": "Vitamin A", "B": "Vitamin C", "C": "Vitamin D", "D": "Vitamin K"},
            "answer_idx": "B",
            "answer": "Vitamin C",
        }
    ]

    converted = convert_rows(rows, "train")

    assert converted[0]["id"] == "medqa_train_0000"
    assert converted[0]["answer"] == "B"
    assert converted[0]["tags"] == ["healthcare", "medqa"]
    assert converted[0]["metadata"]["domain"] == "healthcare"
    assert converted[0]["metadata"]["task_type"] == "medical_multiple_choice"
    assert "Options:" in converted[0]["question"]
    assert "B. Vitamin C" in format_question(rows[0])
