from scripts.prepare_toolalpaca import convert_rows, flatten_instances


def test_flatten_toolalpaca_instances() -> None:
    tools = [
        {
            "Name": "WeatherTool",
            "Category": "Weather",
            "Instances": [
                {"input": "Check the weather in Boston", "output": "The weather is clear."},
                {"input": "Check the weather in Seattle", "output": "It is raining."},
            ],
        }
    ]

    rows = flatten_instances(tools)

    assert len(rows) == 2
    assert rows[0]["tool"]["Name"] == "WeatherTool"
    assert rows[1]["instance"]["output"] == "It is raining."


def test_convert_toolalpaca_rows_to_task_schema() -> None:
    rows = [
        {
            "tool": {
                "Name": "WeatherTool",
                "Category": "Weather",
                "Description": "Weather lookup",
                "Functions": "1. Name: getWeather",
            },
            "instance": {
                "input": "Check the weather in Boston",
                "output": "The weather is clear.",
                "Final Thought": "Use getWeather.",
                "intermediate_steps": [["getWeather", "{\"city\":\"Boston\"}", "clear"]],
            },
            "tool_index": 0,
            "instance_index": 0,
        }
    ]

    converted = convert_rows(rows, "eval")

    assert converted[0]["id"] == "toolalpaca_eval_0000"
    assert converted[0]["answer"] == "The weather is clear."
    assert converted[0]["tags"] == ["tool_use", "toolalpaca"]
    assert converted[0]["metadata"]["tool_name"] == "WeatherTool"
    assert converted[0]["metadata"]["category"] == "Weather"
    assert "Available functions" in converted[0]["question"]
