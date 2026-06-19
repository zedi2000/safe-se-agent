from safe_se_agent.adapters.base import AgentAdapter
from safe_se_agent.adapters.simple import SimpleAgentAdapter
from safe_se_agent.core.experiment import ExperimentConfig, ExperimentRunner, ProgressEvent
from safe_se_agent.core.io import load_jsonl_tasks
from scripts.run_m1_demo import write_artifacts


def test_self_evolution_improves_offline_accuracy(tmp_path) -> None:
    adapter: AgentAdapter = SimpleAgentAdapter(memory_path=tmp_path / "memory.jsonl")
    runner = ExperimentRunner(adapter)
    train_tasks = load_jsonl_tasks("data/m1_train.jsonl")
    eval_tasks = load_jsonl_tasks("data/m1_eval.jsonl")

    baseline = runner.run_no_memory(eval_tasks)
    self_evo = runner.run_self_evolution(train_tasks, eval_tasks)

    assert baseline.accuracy < self_evo.accuracy
    assert baseline.correct == 0
    assert self_evo.correct == len(eval_tasks)
    assert self_evo.learned_rules
    assert self_evo.memory_update_policy == "sliding_window"
    assert self_evo.train_records
    assert self_evo.num_memory_generated >= self_evo.num_memory_added
    assert self_evo.num_memory_added == self_evo.memory_count
    assert self_evo.num_memory_skipped_duplicate >= 0
    learned_text = "\n".join(rule.text for rule in self_evo.learned_rules)
    assert "average_with_extra_item" in learned_text
    assert "discount_then_tax" in learned_text
    assert "total_with_fee" in learned_text
    assert "unit_conversion" in learned_text
    assert self_evo.retrieval_hit_rate == 1.0

    write_artifacts(tmp_path, baseline, self_evo)
    assert (tmp_path / "memory.jsonl").exists()
    assert (tmp_path / "baseline_results.jsonl").exists()
    assert (tmp_path / "train_results.jsonl").exists()
    assert (tmp_path / "self_evolution_results.jsonl").exists()
    assert (tmp_path / "summary.json").exists()
    first_result = (tmp_path / "self_evolution_results.jsonl").read_text(encoding="utf-8").splitlines()[0]
    assert '"reasoning"' in first_result
    assert '"response"' in first_result
    summary = (tmp_path / "summary.json").read_text(encoding="utf-8")
    assert '"memory_update_policy": "sliding_window"' in summary
    assert '"num_memory_generated"' in summary
    assert '"num_memory_added"' in summary


def test_experiment_runner_emits_progress_events(tmp_path) -> None:
    events: list[ProgressEvent] = []
    adapter: AgentAdapter = SimpleAgentAdapter(memory_path=tmp_path / "memory.jsonl")
    runner = ExperimentRunner(
        adapter,
        ExperimentConfig(progress_callback=events.append),
    )
    train_tasks = load_jsonl_tasks("data/m1_train.jsonl")[:2]
    eval_tasks = load_jsonl_tasks("data/m1_eval.jsonl")[:2]

    runner.run_no_memory(eval_tasks)
    runner.run_self_evolution(train_tasks, eval_tasks)

    stages = [event.stage for event in events]
    assert "baseline_solve" in stages
    assert "train_solve" in stages
    assert "reflect" in stages
    assert "memory_write" in stages
    assert "retrieve" in stages
    assert "self_evo_solve" in stages
    assert stages.count("reflect") >= 1

    baseline_progress = [
        event for event in events if event.stage == "baseline_solve" and event.status == "progress"
    ]
    assert [event.current for event in baseline_progress] == [1, 2]
    assert all(event.total == 2 for event in baseline_progress)


def test_per_interaction_training_appends_interaction_log(tmp_path) -> None:
    log_path = tmp_path / "interaction_log.jsonl"
    adapter: AgentAdapter = SimpleAgentAdapter(memory_path=tmp_path / "memory.jsonl")
    runner = ExperimentRunner(
        adapter,
        ExperimentConfig(memory_update_policy="per_interaction", interaction_log_path=log_path),
    )
    train_tasks = load_jsonl_tasks("data/m1_train.jsonl")[:2]
    eval_tasks = load_jsonl_tasks("data/m1_eval.jsonl")[:1]

    summary = runner.run_self_evolution(train_tasks, eval_tasks)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert '"generated_memory"' in lines[0]
    assert '"added_memory"' in lines[0]
    assert '"added_memory_ids"' in lines[0]
    assert '"skipped_memory"' in lines[0]
    assert '"skipped_memory_ids"' in lines[0]
    assert summary.learned_rules
    assert len(summary.learned_rules) == summary.memory_count


def test_sliding_window_training_logs_trigger_metadata(tmp_path) -> None:
    log_path = tmp_path / "interaction_log.jsonl"
    adapter: AgentAdapter = SimpleAgentAdapter(memory_path=tmp_path / "memory.jsonl")
    runner = ExperimentRunner(
        adapter,
        ExperimentConfig(
            interaction_log_path=log_path,
            reflection_window_size=3,
            reflection_window_stride=1,
        ),
    )
    train_tasks = load_jsonl_tasks("data/m1_train.jsonl")[:3]
    eval_tasks = load_jsonl_tasks("data/m1_eval.jsonl")[:1]

    summary = runner.run_self_evolution(train_tasks, eval_tasks)

    rows = log_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3
    assert summary.memory_update_policy == "sliding_window"
    assert any('"reflection_triggered": true' in row for row in rows)
    assert any('"reflection_window_task_ids"' in row for row in rows)
    assert any(record.reflection_triggered for record in summary.train_records or [])


def test_experiment_runner_without_progress_callback_still_runs(tmp_path) -> None:
    adapter: AgentAdapter = SimpleAgentAdapter(memory_path=tmp_path / "memory.jsonl")
    runner = ExperimentRunner(adapter)
    eval_tasks = load_jsonl_tasks("data/m1_eval.jsonl")[:1]

    baseline = runner.run_no_memory(eval_tasks)

    assert baseline.total == 1


def test_batch_memory_update_policy_still_runs(tmp_path) -> None:
    adapter: AgentAdapter = SimpleAgentAdapter(memory_path=tmp_path / "memory.jsonl")
    runner = ExperimentRunner(
        adapter,
        ExperimentConfig(memory_update_policy="batch"),
    )
    train_tasks = load_jsonl_tasks("data/m1_train.jsonl")[:2]
    eval_tasks = load_jsonl_tasks("data/m1_eval.jsonl")[:2]

    summary = runner.run_self_evolution(train_tasks, eval_tasks)

    assert summary.memory_update_policy == "batch"
    assert summary.train_records
