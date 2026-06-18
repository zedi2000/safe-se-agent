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
    assert {rule.tags[1] for rule in self_evo.learned_rules} == {
        "average_with_extra_item",
        "discount_then_tax",
        "total_with_fee",
        "unit_conversion",
    }
    assert self_evo.retrieval_hit_rate == 1.0

    write_artifacts(tmp_path, baseline, self_evo)
    assert (tmp_path / "memory.jsonl").exists()
    assert (tmp_path / "baseline_results.jsonl").exists()
    assert (tmp_path / "self_evolution_results.jsonl").exists()
    assert (tmp_path / "summary.json").exists()


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

    baseline_progress = [
        event for event in events if event.stage == "baseline_solve" and event.status == "progress"
    ]
    assert [event.current for event in baseline_progress] == [1, 2]
    assert all(event.total == 2 for event in baseline_progress)


def test_experiment_runner_without_progress_callback_still_runs(tmp_path) -> None:
    adapter: AgentAdapter = SimpleAgentAdapter(memory_path=tmp_path / "memory.jsonl")
    runner = ExperimentRunner(adapter)
    eval_tasks = load_jsonl_tasks("data/m1_eval.jsonl")[:1]

    baseline = runner.run_no_memory(eval_tasks)

    assert baseline.total == 1
