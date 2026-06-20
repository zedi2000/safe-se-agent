import pytest

from safe_se_agent.core.resume import (
    ResumeConfigError,
    append_jsonl,
    completed_values,
    prepare_resumable_run,
    read_jsonl,
)


def test_append_jsonl_and_completed_values(tmp_path) -> None:
    path = tmp_path / "results.jsonl"

    append_jsonl(path, {"task_id": "a", "correct": True})
    append_jsonl(path, {"task_id": "b", "correct": False})

    assert [row["task_id"] for row in read_jsonl(path)] == ["a", "b"]
    assert completed_values(path, "task_id") == {"a", "b"}


def test_prepare_resumable_run_rejects_mismatched_config(tmp_path) -> None:
    output = tmp_path / "results.jsonl"
    prepare_resumable_run(tmp_path, {"eval": "a"}, [output], resume=False, overwrite=False)

    with pytest.raises(ResumeConfigError):
        prepare_resumable_run(tmp_path, {"eval": "b"}, [output], resume=True, overwrite=False)


def test_prepare_resumable_run_overwrite_clears_outputs(tmp_path) -> None:
    output = tmp_path / "results.jsonl"
    append_jsonl(output, {"task_id": "a"})
    prepare_resumable_run(tmp_path, {"eval": "a"}, [output], resume=False, overwrite=True)

    assert not output.exists()
    assert (tmp_path / "run_config.json").exists()


def test_prepare_resumable_run_requires_explicit_resume_or_overwrite(tmp_path) -> None:
    output = tmp_path / "results.jsonl"
    append_jsonl(output, {"task_id": "a"})

    with pytest.raises(ResumeConfigError):
        prepare_resumable_run(tmp_path, {"eval": "a"}, [output], resume=False, overwrite=False)
