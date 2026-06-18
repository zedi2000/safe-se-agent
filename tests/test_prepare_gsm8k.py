from scripts.prepare_gsm8k import convert_rows, parse_final_answer


def test_parse_gsm8k_final_answer() -> None:
    assert parse_final_answer("We compute carefully.\n#### 1,234") == "1234"
    assert parse_final_answer("Some rationale\n#### 18.0") == "18"


def test_convert_rows_keeps_rationale_and_metadata() -> None:
    rows = [
        {
            "question": "Jan has 2 apples and buys 3 more. How many?",
            "answer": "2 + 3 = 5\n#### 5",
        }
    ]

    converted = convert_rows(rows, "train")

    assert converted[0]["id"] == "gsm8k_train_0000"
    assert converted[0]["answer"] == "5"
    assert converted[0]["tags"] == ["math", "gsm8k"]
    assert "total_with_fee" not in converted[0]["tags"]
    assert "discount_then_tax" not in converted[0]["tags"]
    assert "math_word_problem" not in converted[0]["tags"]
    assert converted[0]["metadata"]["dataset"] == "gsm8k"
    assert converted[0]["metadata"]["domain"] == "math"
    assert converted[0]["metadata"]["task_type"] == "grade_school_math_word_problem"
    assert converted[0]["metadata"]["kind"] == "gsm8k"
    assert converted[0]["metadata"]["sample_index"] == 0
    assert converted[0]["metadata"]["rationale"] == rows[0]["answer"]
